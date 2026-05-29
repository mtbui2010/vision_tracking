"""MOT evaluation metrics — implemented from scratch.

MOTA / IDF1 / HOTA computed directly from per-frame annotations. Cross-checked
against py-motmetrics in tests (required to agree within 0.5% on MOT17-val).

Reference definitions:
  - MOTA: Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
    Performance: The CLEAR MOT Metrics" (2008).
  - IDF1: Ristani et al., "Performance Measures and a Data Set for Multi-Target,
    Multi-Camera Tracking" (2016). Identity-aware F1 over the *entire sequence*
    via a single global GT-track <-> pred-track assignment.
  - HOTA: Luiten et al., "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking" (IJCV 2020). Geometric mean of DetA and AssA,
    averaged over IoU thresholds 0.05..0.95.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trackers.hungarian import linear_sum_assignment
from .trackers.iou import iou_batch


@dataclass
class FrameAnnotations:
    """Per-frame track IDs and bboxes ([x1, y1, x2, y2] pixel coords).

    `scores` is optional and only carries meaning for detector-format files
    (one score per box). For GT and tracker output it defaults to all-ones.
    """

    ids: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray | None = None


@dataclass
class MOTMetrics:
    mota: float
    motp: float
    idf1: float
    hota: float
    deta: float
    assa: float
    fp: int
    fn: int
    idsw: int
    num_gt: int


def compute_metrics(
    gt: list[FrameAnnotations],
    pred: list[FrameAnnotations],
    iou_threshold: float = 0.5,
) -> MOTMetrics:
    if len(gt) != len(pred):
        raise ValueError(f"frame count mismatch: gt={len(gt)} pred={len(pred)}")

    sticky = _sticky_match_sequence(gt, pred, iou_threshold)
    clear = _compute_clear(gt, pred, sticky)
    idf1 = _compute_idf1(gt, pred, sticky)
    hota, deta, assa = _compute_hota(gt, pred)

    return MOTMetrics(
        mota=clear["mota"],
        motp=clear["motp"],
        idf1=idf1,
        hota=hota,
        deta=deta,
        assa=assa,
        fp=clear["fp"],
        fn=clear["fn"],
        idsw=clear["idsw"],
        num_gt=clear["num_gt"],
    )


def _frame_matches(
    gt_frame: FrameAnnotations,
    pred_frame: FrameAnnotations,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """One-frame matching of GT to predictions by max IoU above threshold.

    Returns list of (gt_idx, pred_idx, iou). Used only by HOTA (which sweeps
    thresholds) — the CLEAR / IDF1 path uses sticky matching via
    `_sticky_match_sequence` instead.
    """
    if gt_frame.ids.size == 0 or pred_frame.ids.size == 0:
        return []
    iou = iou_batch(gt_frame.bboxes, pred_frame.bboxes)
    cost = 1.0 - iou
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(rows, cols):
        if iou[r, c] >= iou_threshold:
            matches.append((int(r), int(c), float(iou[r, c])))
    return matches


def _sticky_match_sequence(
    gt: list[FrameAnnotations],
    pred: list[FrameAnnotations],
    iou_threshold: float,
) -> list[list[tuple[int, int, int, int, float]]]:
    """CLEAR-MOT-style per-frame matching with continuity preference.

    For each frame: locks in (gt_id, pred_id) pairs that were matched on a
    prior frame and still overlap above the threshold, then Hungarian-matches
    the remainder. This is what py-motmetrics does — without it our IDF1 can
    drift on sequences with ambiguous spatial overlap.

    Returns per-frame list of (gt_idx, pred_idx, gt_id, pred_id, iou).
    """
    last_pred_for_gt: dict[int, int] = {}
    per_frame: list[list[tuple[int, int, int, int, float]]] = []

    for gt_f, pr_f in zip(gt, pred):
        frame_matches: list[tuple[int, int, int, int, float]] = []
        if gt_f.ids.size == 0 or pr_f.ids.size == 0:
            per_frame.append(frame_matches)
            continue

        iou = iou_batch(gt_f.bboxes, pr_f.bboxes)
        used_gt: set[int] = set()
        used_pred: set[int] = set()

        pred_id_to_idx: dict[int, int] = {int(pid): k for k, pid in enumerate(pr_f.ids)}
        for g_idx, gid in enumerate(gt_f.ids):
            prev_pid = last_pred_for_gt.get(int(gid))
            if prev_pid is None or prev_pid not in pred_id_to_idx:
                continue
            p_idx = pred_id_to_idx[prev_pid]
            if iou[g_idx, p_idx] >= iou_threshold:
                frame_matches.append((g_idx, p_idx, int(gid), int(prev_pid), float(iou[g_idx, p_idx])))
                used_gt.add(g_idx)
                used_pred.add(p_idx)

        free_g = [i for i in range(gt_f.ids.size) if i not in used_gt]
        free_p = [j for j in range(pr_f.ids.size) if j not in used_pred]
        if free_g and free_p:
            sub = iou[np.ix_(free_g, free_p)]
            rows, cols = linear_sum_assignment(1.0 - sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= iou_threshold:
                    g_idx = free_g[int(r)]
                    p_idx = free_p[int(c)]
                    frame_matches.append(
                        (g_idx, p_idx, int(gt_f.ids[g_idx]), int(pr_f.ids[p_idx]), float(sub[r, c]))
                    )

        for _, _, gid_int, pid_int, _ in frame_matches:
            last_pred_for_gt[gid_int] = pid_int
        per_frame.append(frame_matches)

    return per_frame


def _compute_clear(
    gt: list[FrameAnnotations],
    pred: list[FrameAnnotations],
    sticky: list[list[tuple[int, int, int, int, float]]],
) -> dict:
    """CLEAR MOT: MOTA + MOTP + raw FP/FN/IDSW counts."""
    fp = 0
    fn = 0
    idsw = 0
    num_gt = 0
    iou_accum = 0.0
    tp = 0

    last_pred_for_gt: dict[int, int] = {}

    for gt_f, pr_f, matches in zip(gt, pred, sticky):
        num_gt += int(gt_f.ids.size)
        matched_gt = {m[0] for m in matches}
        matched_pred = {m[1] for m in matches}

        fp += int(pr_f.ids.size) - len(matched_pred)
        fn += int(gt_f.ids.size) - len(matched_gt)

        for _, _, gt_id, pred_id, iou_val in matches:
            tp += 1
            iou_accum += iou_val
            prev = last_pred_for_gt.get(gt_id)
            if prev is not None and prev != pred_id:
                idsw += 1
            last_pred_for_gt[gt_id] = pred_id

    mota = 1.0 - (fp + fn + idsw) / max(num_gt, 1)
    motp = iou_accum / max(tp, 1)
    return {"mota": mota, "motp": motp, "fp": fp, "fn": fn, "idsw": idsw, "num_gt": num_gt}


def _compute_idf1(
    gt: list[FrameAnnotations],
    pred: list[FrameAnnotations],
    sticky: list[list[tuple[int, int, int, int, float]]],
) -> float:
    """IDF1 via global GT-id <-> pred-id assignment (Ristani 2016).

    The cost matrix is padded with "skip" rows/cols so the Hungarian can
    choose to leave a track unmatched (treating it as fully FN or FP) rather
    than be forced into a poor pairing. Without padding, IDTP is overcounted
    when one side has more tracks than the other.
    """
    gt_ids = sorted({int(i) for f in gt for i in f.ids})
    pred_ids = sorted({int(i) for f in pred for i in f.ids})
    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0

    gt_idx = {tid: i for i, tid in enumerate(gt_ids)}
    pred_idx = {tid: i for i, tid in enumerate(pred_ids)}

    n_g = len(gt_ids)
    n_p = len(pred_ids)

    gt_count = np.zeros(n_g, dtype=np.int64)
    pred_count = np.zeros(n_p, dtype=np.int64)
    pair_count = np.zeros((n_g, n_p), dtype=np.int64)

    for gt_f, pr_f, matches in zip(gt, pred, sticky):
        for gid in gt_f.ids:
            gt_count[gt_idx[int(gid)]] += 1
        for pid in pr_f.ids:
            pred_count[pred_idx[int(pid)]] += 1
        for _, _, gid_int, pid_int, _ in matches:
            pair_count[gt_idx[gid_int], pred_idx[pid_int]] += 1

    # Block matrix:
    #     [ pair_cost    fn_only  ]
    #     [ fp_only      zeros    ]
    # pair_cost[i,j]  = gt[i] + pred[j] - 2*m(i,j)
    # fn_only[i,i]    = gt[i] (skip cost), off-diagonal = INF (forbidden)
    # fp_only[j,j]    = pred[j], off-diagonal = INF
    n = n_g + n_p
    INF = float(gt_count.sum() + pred_count.sum() + 1)
    cost = np.full((n, n), INF, dtype=np.float64)

    pair_cost = (gt_count[:, None] + pred_count[None, :]) - 2 * pair_count
    cost[:n_g, :n_p] = pair_cost
    for i in range(n_g):
        cost[i, n_p + i] = gt_count[i]
    for j in range(n_p):
        cost[n_g + j, j] = pred_count[j]
    cost[n_g:, n_p:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    idtp = 0
    for r, c in zip(rows, cols):
        if r < n_g and c < n_p:
            idtp += int(pair_count[r, c])

    idfn = int(gt_count.sum()) - idtp
    idfp = int(pred_count.sum()) - idtp
    denom = 2 * idtp + idfp + idfn
    return 0.0 if denom == 0 else 2 * idtp / denom


def _compute_hota(
    gt: list[FrameAnnotations],
    pred: list[FrameAnnotations],
) -> tuple[float, float, float]:
    """HOTA = mean over alpha in [0.05..0.95] of sqrt(DetA(a) * AssA(a)).

    Per Luiten et al. (IJCV 2020). The IoU matrix per frame is precomputed
    once and reused across the 19 alpha thresholds — naive impls recompute
    it 19x, which dominates wall clock on large sequences (MOT17-04 etc).
    """
    alphas = np.arange(0.05, 1.0, 0.05)

    gt_ids_all = sorted({int(i) for f in gt for i in f.ids})
    pred_ids_all = sorted({int(i) for f in pred for i in f.ids})
    gt_idx = {tid: i for i, tid in enumerate(gt_ids_all)}
    pred_idx = {tid: i for i, tid in enumerate(pred_ids_all)}
    n_g = len(gt_ids_all)
    n_p = len(pred_ids_all)

    total_gt = sum(int(f.ids.size) for f in gt)
    total_pred = sum(int(f.ids.size) for f in pred)

    if n_g == 0 or n_p == 0:
        det_a_only = []
        for a in alphas:
            det_a_only.append(0.0 if total_gt + total_pred > 0 else 1.0)
        det_mean = float(np.mean(det_a_only))
        return float(np.sqrt(det_mean * 0.0)), det_mean, 0.0

    gt_count_all = np.zeros(n_g, dtype=np.int64)
    pred_count_all = np.zeros(n_p, dtype=np.int64)
    for gt_f in gt:
        for gid in gt_f.ids:
            gt_count_all[gt_idx[int(gid)]] += 1
    for pr_f in pred:
        for pid in pr_f.ids:
            pred_count_all[pred_idx[int(pid)]] += 1

    # Precompute IoU + Hungarian once per frame — independent of alpha.
    # Each entry: (rows, cols, ious, gt_ids_per_match, pred_ids_per_match).
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for gt_f, pr_f in zip(gt, pred):
        if gt_f.ids.size == 0 or pr_f.ids.size == 0:
            cached.append((np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0),
                           np.zeros(0, dtype=int), np.zeros(0, dtype=int)))
            continue
        iou = iou_batch(gt_f.bboxes, pr_f.bboxes)
        rows, cols = linear_sum_assignment(1.0 - iou)
        ious = iou[rows, cols]
        cached.append((
            rows.astype(int),
            cols.astype(int),
            ious,
            gt_f.ids[rows].astype(int),
            pr_f.ids[cols].astype(int),
        ))

    hota_a: list[float] = []
    deta_a: list[float] = []
    assa_a: list[float] = []
    for a in alphas:
        pair_count = np.zeros((n_g, n_p), dtype=np.int64)
        tp = 0
        for _, _, ious, gids, pids in cached:
            mask = ious >= a
            if not mask.any():
                continue
            tp += int(mask.sum())
            for g_id, p_id in zip(gids[mask], pids[mask]):
                pair_count[gt_idx[int(g_id)], pred_idx[int(p_id)]] += 1

        det_a = tp / max(tp + (total_pred - tp) + (total_gt - tp), 1)
        if tp == 0:
            hota_a.append(0.0)
            deta_a.append(det_a)
            assa_a.append(0.0)
            continue

        tpa_num = 0.0
        tpa_den = 0
        for _, _, ious, gids, pids in cached:
            mask = ious >= a
            for g_id, p_id in zip(gids[mask], pids[mask]):
                g = gt_idx[int(g_id)]
                p = pred_idx[int(p_id)]
                tpa = pair_count[g, p]
                fpa = pred_count_all[p] - tpa
                fna = gt_count_all[g] - tpa
                tpa_num += tpa / max(tpa + fpa + fna, 1)
                tpa_den += 1
        ass_a = tpa_num / max(tpa_den, 1)

        hota_a.append(float(np.sqrt(det_a * ass_a)))
        deta_a.append(det_a)
        assa_a.append(ass_a)

    return float(np.mean(hota_a)), float(np.mean(deta_a)), float(np.mean(assa_a))

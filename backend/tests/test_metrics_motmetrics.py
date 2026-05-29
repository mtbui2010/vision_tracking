"""Cross-check our MOTA / IDF1 against py-motmetrics on a real MOT17 sequence.

Acceptance gate (per CLAUDE.md): must agree within 0.5%.

Run with `pytest -m slow` once `datasets/MOT17/train/MOT17-09-FRCNN/` is present
(see datasets/README.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.eval_dataset import read_mot
from services.metrics import compute_metrics
from services.trackers.base import Detection
from services.trackers.registry import build


SEQ = Path(__file__).resolve().parents[2] / "datasets" / "MOT17" / "train" / "MOT17-09-FRCNN"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sort_predictions():
    if not (SEQ / "gt" / "gt.txt").exists():
        pytest.skip(f"MOT17-09-FRCNN not present at {SEQ}")

    det_seq = read_mot(SEQ / "det" / "det.txt")
    tracker = build("sort")
    pred_ids = []
    pred_boxes = []
    for f in det_seq.frames:
        scores = f.scores if f.scores is not None else np.ones(len(f.bboxes), dtype=np.float32)
        dets = [
            Detection(bbox=bbox.astype(np.float32), score=float(s))
            for bbox, s in zip(f.bboxes, scores)
            if s >= 0.3
        ]
        tracks = tracker.update(dets)
        if tracks:
            pred_ids.append(np.array([t.track_id for t in tracks], dtype=np.int64))
            pred_boxes.append(np.stack([t.bbox for t in tracks]))
        else:
            pred_ids.append(np.zeros(0, dtype=np.int64))
            pred_boxes.append(np.zeros((0, 4)))
    return pred_ids, pred_boxes


def test_mota_idf1_match_motmetrics(sort_predictions):
    """Our metrics must agree with motmetrics on MOTA and IDF1 within 0.5%."""
    import motmetrics as mm

    pred_ids, pred_boxes = sort_predictions
    gt_seq = read_mot(SEQ / "gt" / "gt.txt")

    # Build motmetrics accumulator (uses (x, y, w, h) for distance functions)
    acc = mm.MOTAccumulator(auto_id=True)
    for gt_f, p_ids, p_boxes in zip(gt_seq.frames, pred_ids, pred_boxes):
        gt_ids = gt_f.ids.tolist()
        gt_xywh = np.column_stack([
            gt_f.bboxes[:, 0],
            gt_f.bboxes[:, 1],
            gt_f.bboxes[:, 2] - gt_f.bboxes[:, 0],
            gt_f.bboxes[:, 3] - gt_f.bboxes[:, 1],
        ]) if gt_f.bboxes.size else np.zeros((0, 4))
        p_xywh = np.column_stack([
            p_boxes[:, 0],
            p_boxes[:, 1],
            p_boxes[:, 2] - p_boxes[:, 0],
            p_boxes[:, 3] - p_boxes[:, 1],
        ]) if p_boxes.size else np.zeros((0, 4))
        dist = mm.distances.iou_matrix(gt_xywh, p_xywh, max_iou=0.5)
        acc.update(gt_ids, p_ids.tolist(), dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_false_positives", "num_misses", "num_switches"])
    mm_mota = float(summary["mota"].iloc[0])
    mm_idf1 = float(summary["idf1"].iloc[0])
    mm_fp = int(summary["num_false_positives"].iloc[0])
    mm_fn = int(summary["num_misses"].iloc[0])
    mm_idsw = int(summary["num_switches"].iloc[0])

    pred_frames = [
        type(gt_seq.frames[0])(ids=ids, bboxes=boxes, scores=None)
        for ids, boxes in zip(pred_ids, pred_boxes)
    ]
    ours = compute_metrics(gt_seq.frames, pred_frames, iou_threshold=0.5)

    print(
        f"\nours  : MOTA={ours.mota:.4f} IDF1={ours.idf1:.4f} FP={ours.fp} FN={ours.fn} IDSW={ours.idsw}"
        f"\nmotmet: MOTA={mm_mota:.4f} IDF1={mm_idf1:.4f} FP={mm_fp} FN={mm_fn} IDSW={mm_idsw}"
    )

    assert abs(ours.mota - mm_mota) <= 0.005, f"MOTA off: ours={ours.mota:.4f} mm={mm_mota:.4f}"
    assert abs(ours.idf1 - mm_idf1) <= 0.005, f"IDF1 off: ours={ours.idf1:.4f} mm={mm_idf1:.4f}"

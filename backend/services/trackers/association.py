"""Association helpers shared by SORT, DeepSORT, ByteTrack."""

from __future__ import annotations

import numpy as np

from .hungarian import linear_sum_assignment
from .iou import iou_batch


def associate_iou(
    det_boxes: np.ndarray,
    trk_boxes: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to tracks by IoU using Hungarian.

    Returns:
        matched: list of (det_idx, trk_idx) pairs above the IoU threshold.
        unmatched_dets: indices of detections that did not get matched.
        unmatched_trks: indices of tracks that did not get matched.
    """
    n_det = len(det_boxes)
    n_trk = len(trk_boxes)
    if n_det == 0:
        return [], [], list(range(n_trk))
    if n_trk == 0:
        return [], list(range(n_det)), []

    iou = iou_batch(np.asarray(det_boxes), np.asarray(trk_boxes))
    cost = 1.0 - iou
    rows, cols = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    for r, c in zip(rows, cols):
        if iou[r, c] >= iou_threshold:
            matched.append((int(r), int(c)))

    matched_dets = {m[0] for m in matched}
    matched_trks = {m[1] for m in matched}
    unmatched_dets = [i for i in range(n_det) if i not in matched_dets]
    unmatched_trks = [i for i in range(n_trk) if i not in matched_trks]
    return matched, unmatched_dets, unmatched_trks

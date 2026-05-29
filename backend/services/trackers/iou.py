"""Vectorized bbox utilities."""

from __future__ import annotations

import numpy as np


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. Boxes are [x1, y1, x2, y2] in pixel coords.

    Returns matrix of shape (len(a), len(b)).
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    a_ = a[:, None, :]
    b_ = b[None, :, :]
    xx1 = np.maximum(a_[..., 0], b_[..., 0])
    yy1 = np.maximum(a_[..., 1], b_[..., 1])
    xx2 = np.minimum(a_[..., 2], b_[..., 2])
    yy2 = np.minimum(a_[..., 3], b_[..., 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    out = np.zeros_like(union)
    np.divide(inter, union, out=out, where=union > 0)
    return out

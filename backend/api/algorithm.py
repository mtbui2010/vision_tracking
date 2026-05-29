"""Algorithm inspector: expose Kalman state, Hungarian cost matrix, association
decisions for a single frame so the frontend can visualize them.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from services.trackers.hungarian import linear_sum_assignment
from services.trackers.iou import iou_batch
from services.trackers.kalman import KalmanBoxTracker


router = APIRouter(prefix="/api/algorithm", tags=["algorithm"])


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)


class KalmanRequest(BaseModel):
    bbox: BBox
    steps: int = 1


class KalmanResponse(BaseModel):
    predicted_bbox: list[float]
    state: list[float]
    covariance_diag: list[float]


@router.post("/kalman", response_model=KalmanResponse)
def kalman_step(req: KalmanRequest) -> KalmanResponse:
    kf = KalmanBoxTracker(req.bbox.to_array())
    for _ in range(req.steps):
        bbox = kf.predict()
    else:
        bbox = kf.bbox
    return KalmanResponse(
        predicted_bbox=bbox.tolist(),
        state=kf.x.tolist(),
        covariance_diag=np.diag(kf.P).tolist(),
    )


class AssociationRequest(BaseModel):
    detections: list[BBox]
    tracks: list[BBox]
    iou_threshold: float = 0.3


class AssociationResponse(BaseModel):
    iou_matrix: list[list[float]]
    cost_matrix: list[list[float]]
    matches: list[tuple[int, int]]
    unmatched_detections: list[int]
    unmatched_tracks: list[int]


@router.post("/associate", response_model=AssociationResponse)
def associate(req: AssociationRequest) -> AssociationResponse:
    det = np.stack([d.to_array() for d in req.detections]) if req.detections else np.zeros((0, 4))
    trk = np.stack([t.to_array() for t in req.tracks]) if req.tracks else np.zeros((0, 4))

    iou = iou_batch(det, trk)
    cost = 1.0 - iou
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
    else:
        rows = cols = np.zeros(0, dtype=int)

    matches: list[tuple[int, int]] = []
    for r, c in zip(rows, cols):
        if iou[r, c] >= req.iou_threshold:
            matches.append((int(r), int(c)))

    matched_dets = {m[0] for m in matches}
    matched_trks = {m[1] for m in matches}
    return AssociationResponse(
        iou_matrix=iou.tolist(),
        cost_matrix=cost.tolist(),
        matches=matches,
        unmatched_detections=[i for i in range(len(req.detections)) if i not in matched_dets],
        unmatched_tracks=[i for i in range(len(req.tracks)) if i not in matched_trks],
    )

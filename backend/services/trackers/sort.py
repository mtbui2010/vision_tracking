"""SORT — Simple Online and Realtime Tracking.

Bewley et al., ICIP 2016. Kalman filter + IoU + Hungarian. No appearance.

This is the simplest tracker in the lab. Reference number on MOT17-val: MOTA ~0.59.
"""

from __future__ import annotations

import numpy as np

from .association import associate_iou
from .base import BaseTracker, Detection, Track
from .kalman import KalmanBoxTracker


class _TrackState:
    __slots__ = ("track_id", "kf", "hits", "time_since_update", "age", "score", "class_id")

    def __init__(self, track_id: int, det: Detection) -> None:
        self.track_id = track_id
        self.kf = KalmanBoxTracker(det.bbox)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.score = det.score
        self.class_id = det.class_id


class SORT(BaseTracker):
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 1,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_TrackState] = []
        self._next_id = 1
        self._frame = 0

    @property
    def name(self) -> str:
        return "sort"

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        self._frame += 1

        for trk in self._tracks:
            trk.kf.predict()
            trk.age += 1

        det_boxes = np.array([d.bbox for d in detections]) if detections else np.zeros((0, 4))
        trk_boxes = np.array([t.kf.bbox for t in self._tracks]) if self._tracks else np.zeros((0, 4))

        matched, unmatched_dets, unmatched_trks = associate_iou(
            det_boxes, trk_boxes, self.iou_threshold
        )

        for d_idx, t_idx in matched:
            trk = self._tracks[t_idx]
            det = detections[d_idx]
            trk.kf.update(det.bbox)
            trk.hits += 1
            trk.time_since_update = 0
            trk.score = det.score
            trk.class_id = det.class_id

        for t_idx in unmatched_trks:
            self._tracks[t_idx].time_since_update += 1

        for d_idx in unmatched_dets:
            self._tracks.append(_TrackState(self._next_id, detections[d_idx]))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        return [
            Track(
                track_id=t.track_id,
                bbox=t.kf.bbox,
                score=t.score,
                class_id=t.class_id,
                age=t.age,
                time_since_update=t.time_since_update,
                hits=t.hits,
            )
            for t in self._tracks
            if t.time_since_update == 0
            and (t.hits >= self.min_hits or self._frame <= self.min_hits)
        ]

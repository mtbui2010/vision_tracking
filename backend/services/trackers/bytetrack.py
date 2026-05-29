"""ByteTrack — 2-stage IoU association distinguishing high- and low-score dets.

Zhang et al., ECCV 2022. Key idea: don't throw away low-confidence detections.
Most non-pedestrian, non-occluded objects get high scores; the *useful* low
scores tend to be partially occluded versions of tracked objects.

Stage 1: high-conf dets <-> all tracks via IoU.
Stage 2: low-conf dets <-> unmatched tracks via IoU (lower threshold).
Stage 3: unmatched high-conf dets become new tentative tracks.

Reference number on MOT17-val: MOTA ~0.78, IDF1 ~0.78 with YOLOX-X.
"""

from __future__ import annotations

import numpy as np

from .association import associate_iou
from .base import BaseTracker, Detection, Track
from .kalman import KalmanBoxTracker


class _BTrack:
    __slots__ = (
        "track_id",
        "kf",
        "hits",
        "time_since_update",
        "age",
        "score",
        "class_id",
        "state",
    )

    STATE_TENTATIVE = 0
    STATE_TRACKED = 1
    STATE_LOST = 2

    def __init__(self, track_id: int, det: Detection) -> None:
        self.track_id = track_id
        self.kf = KalmanBoxTracker(det.bbox)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.score = det.score
        self.class_id = det.class_id
        self.state = _BTrack.STATE_TENTATIVE


class ByteTrack(BaseTracker):
    def __init__(
        self,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        new_track_threshold: float = 0.7,
        match_threshold_stage1: float = 0.2,
        match_threshold_stage2: float = 0.5,
        max_age: int = 30,
        n_init: int = 3,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = new_track_threshold
        self.match_threshold_stage1 = match_threshold_stage1
        self.match_threshold_stage2 = match_threshold_stage2
        self.max_age = max_age
        self.n_init = n_init
        self._tracks: list[_BTrack] = []
        self._next_id = 1
        self._frame = 0

    @property
    def name(self) -> str:
        return "bytetrack"

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        self._frame += 1

        for trk in self._tracks:
            trk.kf.predict()
            trk.age += 1

        high_det_idxs = [i for i, d in enumerate(detections) if d.score >= self.high_threshold]
        low_det_idxs = [
            i for i, d in enumerate(detections)
            if self.low_threshold <= d.score < self.high_threshold
        ]

        active_trk_idxs = [
            i for i, t in enumerate(self._tracks)
            if t.state in (_BTrack.STATE_TRACKED, _BTrack.STATE_TENTATIVE, _BTrack.STATE_LOST)
        ]

        # Stage 1: high-score dets <-> all active tracks
        stage1_matched, unmatched_dets_s1, unmatched_trks_s1 = self._iou_match(
            detections,
            high_det_idxs,
            active_trk_idxs,
            1.0 - self.match_threshold_stage1,
        )

        # Stage 2: low-score dets <-> tracks that survived stage 1 and were *previously tracked*
        # (don't try to match lost or tentative tracks to weak detections — that's noise)
        stage2_candidates = [
            i for i in unmatched_trks_s1 if self._tracks[i].state == _BTrack.STATE_TRACKED
        ]
        stage2_matched, _, unmatched_trks_s2 = self._iou_match(
            detections,
            low_det_idxs,
            stage2_candidates,
            1.0 - self.match_threshold_stage2,
        )
        unmatched_high_trks = [
            i for i in unmatched_trks_s1
            if self._tracks[i].state != _BTrack.STATE_TRACKED or i in unmatched_trks_s2
        ]

        matched = stage1_matched + stage2_matched
        unmatched_trks = list({*unmatched_high_trks, *unmatched_trks_s2})

        for d_idx, t_idx in matched:
            trk = self._tracks[t_idx]
            det = detections[d_idx]
            trk.kf.update(det.bbox)
            trk.hits += 1
            trk.time_since_update = 0
            trk.score = det.score
            trk.class_id = det.class_id
            if trk.state == _BTrack.STATE_TENTATIVE and trk.hits >= self.n_init:
                trk.state = _BTrack.STATE_TRACKED
            elif trk.state == _BTrack.STATE_LOST:
                trk.state = _BTrack.STATE_TRACKED

        for t_idx in unmatched_trks:
            trk = self._tracks[t_idx]
            trk.time_since_update += 1
            if trk.state == _BTrack.STATE_TRACKED:
                trk.state = _BTrack.STATE_LOST

        for d_idx in unmatched_dets_s1:
            if detections[d_idx].score >= self.new_track_threshold:
                self._tracks.append(_BTrack(self._next_id, detections[d_idx]))
                self._next_id += 1

        self._tracks = [
            t for t in self._tracks
            if not (t.time_since_update > self.max_age
                    or (t.state == _BTrack.STATE_TENTATIVE and t.time_since_update > 0))
        ]

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
            if t.state == _BTrack.STATE_TRACKED and t.time_since_update == 0
        ]

    def _iou_match(
        self,
        detections: list[Detection],
        det_idxs: list[int],
        trk_idxs: list[int],
        iou_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not det_idxs or not trk_idxs:
            return [], det_idxs, trk_idxs

        det_boxes = np.stack([detections[i].bbox for i in det_idxs])
        trk_boxes = np.stack([self._tracks[i].kf.bbox for i in trk_idxs])
        matched_local, unm_d_local, unm_t_local = associate_iou(det_boxes, trk_boxes, iou_threshold)

        matched = [(det_idxs[d], trk_idxs[t]) for d, t in matched_local]
        unmatched_dets = [det_idxs[i] for i in unm_d_local]
        unmatched_trks = [trk_idxs[i] for i in unm_t_local]
        return matched, unmatched_dets, unmatched_trks

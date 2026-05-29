"""Custom tracker — appearance-gated ByteTrack with confidence-aware Kalman.

Motivation (see docs/BLOG_DRAFT.md, Lesson 3): vanilla ByteTrack's stage-2
"low-confidence rescue" rests on the assumption that low-score detections are
mostly occluded versions of tracked objects. When the detector is sparse and
noisy (which is the case for off-the-shelf FRCNN dets on MOT17-val), most
low-score boxes are false positives. The IoU-only stage-2 then drives spurious
matches and inflates ID switches — we measured this: ByteTrack got 506 IDSW vs
SORT's 281 on the same data.

The fix:
  1) Embed every detection with a ReID head (any BaseEmbedder).
  2) Stage 1 unchanged — high-conf dets matched to all tracks by IoU.
  3) Stage 2 ALSO requires appearance match: a low-conf det can only revive a
     track if its embedding is close (cosine) to the track's gallery.
  4) Confidence-aware Kalman update — measurement noise R is scaled by
     1 / max(score, eps). Low-conf updates don't pull the state as hard, so a
     spurious match in stage 2 (if one slips through the appearance gate)
     causes less damage to the Kalman track.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..reid import BaseEmbedder, HashEmbedder, crop_for_reid
from .association import associate_iou
from .base import BaseTracker, Detection, Track
from .iou import iou_batch
from .hungarian import linear_sum_assignment
from .kalman import KalmanBoxTracker


class _CTrack:
    __slots__ = (
        "track_id",
        "kf",
        "hits",
        "time_since_update",
        "age",
        "score",
        "class_id",
        "gallery",
        "state",
    )

    STATE_TENTATIVE = 0
    STATE_TRACKED = 1
    STATE_LOST = 2

    def __init__(self, track_id: int, det: Detection, gallery_size: int = 30) -> None:
        self.track_id = track_id
        self.kf = KalmanBoxTracker(det.bbox)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.score = det.score
        self.class_id = det.class_id
        self.gallery: deque[np.ndarray] = deque(maxlen=gallery_size)
        if det.embedding is not None:
            self.gallery.append(det.embedding)
        self.state = _CTrack.STATE_TENTATIVE

    def appearance_distance(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.gallery or embeddings.size == 0:
            return np.full(embeddings.shape[0], 1.0)
        gal = np.stack(list(self.gallery))
        sim = embeddings @ gal.T
        return 1.0 - sim.max(axis=1)


def _conf_aware_update(kf: KalmanBoxTracker, bbox: np.ndarray, score: float) -> None:
    """Kalman update where measurement noise R is inflated for low-score dets.

    Standard SORT update uses fixed R. We multiply R by 1 / max(score, eps),
    so a det with score=0.2 contributes ~5x less than score=1.0. This is the
    "confidence-aware Kalman" piece of the custom tracker.
    """
    eps = 0.05
    weight = 1.0 / max(float(score), eps)
    R_inflated = kf._R * weight
    from .kalman import bbox_to_z

    z = bbox_to_z(np.asarray(bbox, dtype=np.float64))
    y = z - kf._H @ kf.x
    S = kf._H @ kf.P @ kf._H.T + R_inflated
    K = kf.P @ kf._H.T @ np.linalg.inv(S)
    kf.x = kf.x + K @ y
    kf.P = (np.eye(7) - K @ kf._H) @ kf.P


class CustomTracker(BaseTracker):
    """Appearance-gated ByteTrack."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        new_track_threshold: float = 0.7,
        iou_threshold_stage1: float = 0.2,
        iou_threshold_stage2: float = 0.5,
        appearance_threshold: float = 0.5,
        max_age: int = 30,
        n_init: int = 3,
        gallery_size: int = 30,
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = new_track_threshold
        self.iou_threshold_stage1 = iou_threshold_stage1
        self.iou_threshold_stage2 = iou_threshold_stage2
        self.appearance_threshold = appearance_threshold
        self.max_age = max_age
        self.n_init = n_init
        self.gallery_size = gallery_size
        self._tracks: list[_CTrack] = []
        self._next_id = 1
        self._frame = 0

    @property
    def name(self) -> str:
        return "custom"

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        self._frame += 1

        if frame is not None:
            crops = [crop_for_reid(frame, d.bbox) for d in detections]
            embeds = self.embedder.embed(crops)
            for d, e in zip(detections, embeds):
                d.embedding = e

        for trk in self._tracks:
            trk.kf.predict()
            trk.age += 1

        high = [i for i, d in enumerate(detections) if d.score >= self.high_threshold]
        low = [
            i for i, d in enumerate(detections)
            if self.low_threshold <= d.score < self.high_threshold
        ]
        active = [
            i for i, t in enumerate(self._tracks)
            if t.state in (_CTrack.STATE_TRACKED, _CTrack.STATE_TENTATIVE, _CTrack.STATE_LOST)
        ]

        # Stage 1: high-conf dets vs all active tracks, IoU only
        s1_matched, unm_d1, unm_t1 = self._iou_match(
            detections, high, active, self.iou_threshold_stage1
        )

        # Stage 2: low-conf dets vs unmatched TRACKED tracks — IoU AND appearance gate
        s2_candidates = [i for i in unm_t1 if self._tracks[i].state == _CTrack.STATE_TRACKED]
        s2_matched, _, unm_t2 = self._appearance_gated_match(
            detections, low, s2_candidates,
            iou_threshold=self.iou_threshold_stage2,
            appearance_threshold=self.appearance_threshold,
        )
        non_tracked_unm = [i for i in unm_t1 if self._tracks[i].state != _CTrack.STATE_TRACKED]
        unmatched_trks = list({*non_tracked_unm, *unm_t2})

        matched = s1_matched + s2_matched

        for d_idx, t_idx in matched:
            trk = self._tracks[t_idx]
            det = detections[d_idx]
            _conf_aware_update(trk.kf, det.bbox, det.score)
            trk.hits += 1
            trk.time_since_update = 0
            trk.score = det.score
            trk.class_id = det.class_id
            if det.embedding is not None:
                trk.gallery.append(det.embedding)
            if trk.state == _CTrack.STATE_TENTATIVE and trk.hits >= self.n_init:
                trk.state = _CTrack.STATE_TRACKED
            elif trk.state == _CTrack.STATE_LOST:
                trk.state = _CTrack.STATE_TRACKED

        for t_idx in unmatched_trks:
            trk = self._tracks[t_idx]
            trk.time_since_update += 1
            if trk.state == _CTrack.STATE_TRACKED:
                trk.state = _CTrack.STATE_LOST

        for d_idx in unm_d1:
            if detections[d_idx].score >= self.new_track_threshold:
                self._tracks.append(
                    _CTrack(self._next_id, detections[d_idx], gallery_size=self.gallery_size)
                )
                self._next_id += 1

        self._tracks = [
            t for t in self._tracks
            if not (t.time_since_update > self.max_age
                    or (t.state == _CTrack.STATE_TENTATIVE and t.time_since_update > 0))
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
            if t.state == _CTrack.STATE_TRACKED and t.time_since_update == 0
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

    def _appearance_gated_match(
        self,
        detections: list[Detection],
        det_idxs: list[int],
        trk_idxs: list[int],
        iou_threshold: float,
        appearance_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match dets to tracks where BOTH IoU >= iou_thr AND appearance_dist <= app_thr."""
        if not det_idxs or not trk_idxs:
            return [], det_idxs, trk_idxs

        det_boxes = np.stack([detections[i].bbox for i in det_idxs])
        trk_boxes = np.stack([self._tracks[i].kf.bbox for i in trk_idxs])
        iou = iou_batch(det_boxes, trk_boxes)

        app_cost = np.full((len(trk_idxs), len(det_idxs)), 1.0, dtype=np.float64)
        for ti, t_idx in enumerate(trk_idxs):
            embeds = np.stack([
                detections[d_idx].embedding
                if detections[d_idx].embedding is not None
                else np.zeros(0, dtype=np.float32)
                for d_idx in det_idxs
            ])
            if embeds.ndim != 2 or embeds.shape[1] == 0:
                continue
            app_cost[ti] = self._tracks[t_idx].appearance_distance(embeds)

        # Combined cost — IoU-driven, appearance-gated
        cost = (1.0 - iou.T) + 0.5 * app_cost  # transpose so rows=tracks, cols=dets
        rows, cols = linear_sum_assignment(cost)

        matched: list[tuple[int, int]] = []
        matched_d, matched_t = set(), set()
        for r, c in zip(rows, cols):
            d_idx = det_idxs[c]
            t_idx = trk_idxs[r]
            if iou[c, r] >= iou_threshold and app_cost[r, c] <= appearance_threshold:
                matched.append((d_idx, t_idx))
                matched_d.add(d_idx)
                matched_t.add(t_idx)

        unmatched_dets = [i for i in det_idxs if i not in matched_d]
        unmatched_trks = [i for i in trk_idxs if i not in matched_t]
        return matched, unmatched_dets, unmatched_trks

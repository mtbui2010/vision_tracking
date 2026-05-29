"""DeepSORT — SORT + appearance ReID + matching cascade.

Wojke, Bewley, Paulus, "Simple Online and Realtime Tracking with a Deep
Association Metric" (ICIP 2017).

Differences from SORT:
  - each track keeps a gallery of the last K appearance embeddings;
  - cost = lambda * (1 - cos_sim(appearance)) gated by Mahalanobis motion gate;
  - matching cascade: tracks with smaller `time_since_update` are matched
    first, so recently-seen tracks get priority over long-lost ones.

Reference IDF1 on MOT17-val: ~64-67 with OSNet embeddings.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..reid import BaseEmbedder, HashEmbedder, crop_for_reid
from .association import associate_iou
from .base import BaseTracker, Detection, Track
from .hungarian import linear_sum_assignment
from .kalman import KalmanBoxTracker


# 95th percentile of the chi-square distribution with 4 dof — DeepSORT default.
CHI2_INV95 = 9.4877


class _DSTrack:
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
    STATE_CONFIRMED = 1

    def __init__(self, track_id: int, det: Detection, gallery_size: int = 100) -> None:
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
        self.state = _DSTrack.STATE_TENTATIVE

    def appearance_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Minimum cosine distance from each embedding to anything in the gallery."""
        if not self.gallery or embeddings.size == 0:
            return np.full(embeddings.shape[0], 1.0)
        gal = np.stack(list(self.gallery))
        sim = embeddings @ gal.T
        return 1.0 - sim.max(axis=1)


class DeepSORT(BaseTracker):
    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        max_age: int = 30,
        n_init: int = 3,
        appearance_threshold: float = 0.4,
        iou_threshold: float = 0.3,
        gallery_size: int = 100,
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self.max_age = max_age
        self.n_init = n_init
        self.appearance_threshold = appearance_threshold
        self.iou_threshold = iou_threshold
        self.gallery_size = gallery_size
        self._tracks: list[_DSTrack] = []
        self._next_id = 1
        self._frame = 0

    @property
    def name(self) -> str:
        return "deepsort"

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

        matched, unmatched_trks, unmatched_dets = self._matching_cascade(detections)
        iou_matched, unmatched_dets, unmatched_trks = self._iou_fallback(
            detections, unmatched_dets, unmatched_trks
        )
        matched.extend(iou_matched)

        for d_idx, t_idx in matched:
            trk = self._tracks[t_idx]
            det = detections[d_idx]
            trk.kf.update(det.bbox)
            trk.hits += 1
            trk.time_since_update = 0
            trk.score = det.score
            trk.class_id = det.class_id
            if det.embedding is not None:
                trk.gallery.append(det.embedding)
            if trk.state == _DSTrack.STATE_TENTATIVE and trk.hits >= self.n_init:
                trk.state = _DSTrack.STATE_CONFIRMED

        for t_idx in unmatched_trks:
            self._tracks[t_idx].time_since_update += 1

        for d_idx in unmatched_dets:
            self._tracks.append(
                _DSTrack(self._next_id, detections[d_idx], gallery_size=self.gallery_size)
            )
            self._next_id += 1

        self._tracks = [
            t for t in self._tracks
            if not (t.time_since_update > self.max_age
                    or (t.state == _DSTrack.STATE_TENTATIVE and t.time_since_update > 0))
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
            if t.state == _DSTrack.STATE_CONFIRMED and t.time_since_update == 0
        ]

    def _matching_cascade(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match confirmed tracks to detections, prioritizing recently-seen tracks."""
        matched: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        track_idxs = [i for i, t in enumerate(self._tracks) if t.state == _DSTrack.STATE_CONFIRMED]
        unmatched_trks = [i for i, t in enumerate(self._tracks) if t.state != _DSTrack.STATE_CONFIRMED]

        for level in range(self.max_age + 1):
            if not unmatched_dets:
                break
            level_trks = [
                i for i in track_idxs if self._tracks[i].time_since_update == level
            ]
            if not level_trks:
                continue
            level_matched, unmatched_dets, level_unmatched = self._appearance_match(
                detections, level_trks, unmatched_dets
            )
            matched.extend(level_matched)
            unmatched_trks.extend(level_unmatched)

        return matched, unmatched_trks, unmatched_dets

    def _appearance_match(
        self,
        detections: list[Detection],
        track_idxs: list[int],
        det_idxs: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_idxs or not det_idxs:
            return [], det_idxs, track_idxs

        embeds = np.stack(
            [
                d.embedding if d.embedding is not None else np.zeros(0, dtype=np.float32)
                for d in (detections[i] for i in det_idxs)
            ]
        )
        if embeds.ndim != 2 or embeds.shape[1] == 0:
            return [], det_idxs, track_idxs

        cost = np.zeros((len(track_idxs), len(det_idxs)), dtype=np.float64)
        for ti, t_idx in enumerate(track_idxs):
            trk = self._tracks[t_idx]
            cost[ti] = trk.appearance_distance(embeds)
            for di, d_idx in enumerate(det_idxs):
                if trk.kf.mahalanobis_sq(detections[d_idx].bbox) > CHI2_INV95:
                    cost[ti, di] = 1.0

        rows, cols = linear_sum_assignment(cost)
        matched: list[tuple[int, int]] = []
        matched_trks: set[int] = set()
        matched_dets: set[int] = set()
        for r, c in zip(rows, cols):
            if cost[r, c] < self.appearance_threshold:
                matched.append((det_idxs[c], track_idxs[r]))
                matched_trks.add(track_idxs[r])
                matched_dets.add(det_idxs[c])

        unmatched_dets = [i for i in det_idxs if i not in matched_dets]
        unmatched_trks = [i for i in track_idxs if i not in matched_trks]
        return matched, unmatched_dets, unmatched_trks

    def _iou_fallback(
        self,
        detections: list[Detection],
        det_idxs: list[int],
        trk_idxs: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not det_idxs or not trk_idxs:
            return [], det_idxs, trk_idxs

        det_boxes = np.stack([detections[i].bbox for i in det_idxs])
        trk_boxes = np.stack([self._tracks[i].kf.bbox for i in trk_idxs])
        matched_local, unm_d_local, unm_t_local = associate_iou(
            det_boxes, trk_boxes, self.iou_threshold
        )

        matched = [(det_idxs[d], trk_idxs[t]) for d, t in matched_local]
        unmatched_dets = [det_idxs[i] for i in unm_d_local]
        unmatched_trks = [trk_idxs[i] for i in unm_t_local]
        return matched, unmatched_dets, unmatched_trks

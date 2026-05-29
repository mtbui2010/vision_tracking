"""Tracker interface — every tracker (SORT, DeepSORT, ByteTrack, custom) implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detection in one frame.

    Bbox is in pixel coords (x1, y1, x2, y2). Embedding is optional and only set
    when the tracker needs appearance features (DeepSORT, custom).
    """

    bbox: np.ndarray  # shape (4,), float32, [x1, y1, x2, y2]
    score: float
    class_id: int = 0
    embedding: np.ndarray | None = None  # shape (D,), float32, L2-normalized


@dataclass
class Track:
    """A single track's state at one frame."""

    track_id: int
    bbox: np.ndarray  # shape (4,), predicted-and-updated bbox this frame
    score: float
    class_id: int = 0
    age: int = 0  # frames since first detected
    time_since_update: int = 0  # frames since last successful association
    hits: int = 0  # total successful associations


class BaseTracker(ABC):
    """All trackers in this repo subclass this.

    The interface is intentionally minimal: one method, one return value.
    `update` is called once per frame with that frame's detections; it returns
    the currently confirmed tracks.

    Implementations differ in:
      - how they predict (Kalman, no prediction, ...);
      - how they score association costs (IoU, appearance, hybrid);
      - how they resolve assignment (Hungarian, greedy, cascade, byte-association);
      - when they confirm / delete tracks.
    """

    @abstractmethod
    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        """Process one frame.

        Args:
            detections: per-frame detections from the detector.
            frame: optional BGR frame; required only for trackers that compute
                appearance embeddings on the fly (e.g. DeepSORT).

        Returns:
            confirmed tracks for this frame. Order is unspecified.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name, e.g. "sort", "deepsort", "bytetrack"."""
        raise NotImplementedError

"""YOLO detector wrapper. Class-filter and score-filter applied here so the
tracker only sees the detections it cares about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .trackers.base import Detection


class YOLODetector:
    """Thin wrapper around an Ultralytics YOLO checkpoint.

    Loaded lazily so the backend can be imported on a machine without torch.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        class_ids: tuple[int, ...] = (0,),  # COCO person
        score_threshold: float = 0.1,
        device: str | None = None,
    ) -> None:
        self.weights = str(weights)
        self.class_ids = set(class_ids)
        self.score_threshold = score_threshold
        self.device = device
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # type: ignore[import-not-found]

        self._model = YOLO(self.weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._lazy_load()
        assert self._model is not None
        result = self._model(frame, verbose=False, device=self.device)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)

        dets: list[Detection] = []
        for box, score, c in zip(xyxy, conf, cls):
            if c not in self.class_ids:
                continue
            if score < self.score_threshold:
                continue
            dets.append(
                Detection(
                    bbox=box.astype(np.float32),
                    score=float(score),
                    class_id=int(c),
                )
            )
        return dets

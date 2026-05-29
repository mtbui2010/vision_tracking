"""ReID embedder interface + two backends.

`OSNetEmbedder` wraps a pretrained OSNet (torchreid). Used in production.
`HashEmbedder` is a deterministic image-hash placeholder used in tests so the
algorithmic flow can be unit-tested without GPU / torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return L2-normalized embeddings of shape (N, D)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    """Deterministic, low-dim hash of a crop. For tests, not for real tracking."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.zeros((len(crops), self._dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop.size == 0:
                continue
            small = _resize_avg(crop, 8, 8).reshape(-1)
            v = np.tile(small, self._dim // small.size + 1)[: self._dim]
            v = v.astype(np.float32) / 255.0
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            out[i] = v
        return out


class OSNetEmbedder(BaseEmbedder):
    """OSNet-x0.25 ReID backbone.

    Loaded lazily so the rest of the codebase has no hard torch dependency
    until ReID is actually used.
    """

    def __init__(self, weights: str = "osnet_x0_25") -> None:
        self._weights_name = weights
        self._model = None
        self._device = None
        self._dim_value = 512

    @property
    def dim(self) -> int:
        return self._dim_value

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore[import-not-found]
        import torchreid  # type: ignore[import-not-found]

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = torchreid.models.build_model(
            name=self._weights_name, num_classes=1000, pretrained=True
        )
        self._model.eval().to(self._device)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self._dim_value), dtype=np.float32)
        self._lazy_init()
        import torch  # type: ignore[import-not-found]

        x = np.stack([_resize(c, 128, 256) for c in crops]).astype(np.float32)
        x = x.transpose(0, 3, 1, 2) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        x = (x - mean) / std
        t = torch.from_numpy(x).to(self._device)
        with torch.no_grad():
            feats = self._model(t).cpu().numpy()
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        return feats.astype(np.float32)


def crop_for_reid(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def _resize_avg(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Cheap downsample by bilinear-ish averaging (no cv2 dependency)."""
    h, w = arr.shape[:2]
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    ys = (np.linspace(0, h - 1, target_h)).astype(int)
    xs = (np.linspace(0, w - 1, target_w)).astype(int)
    return arr[np.ix_(ys, xs)]


def _resize(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Nearest-neighbour resize used only as fallback. OSNetEmbedder swaps in cv2 when available."""
    try:
        import cv2  # type: ignore[import-not-found]

        return cv2.resize(arr, (target_w, target_h))
    except ImportError:
        h, w = arr.shape[:2]
        ys = (np.linspace(0, h - 1, target_h)).astype(int)
        xs = (np.linspace(0, w - 1, target_w)).astype(int)
        return arr[np.ix_(ys, xs)]

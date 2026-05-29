"""Constant-velocity Kalman filter for bbox tracking (SORT formulation).

State (7-d): [u, v, s, r, du, dv, ds]
  u, v = bbox center; s = area (w*h); r = aspect ratio (w/h, held constant)
Measurement (4-d): [u, v, s, r]

No `filterpy` — F, H, Q, R are explicit numpy arrays. See
docs/notebooks/01_kalman_derivation.ipynb for the derivation.
"""

from __future__ import annotations

import numpy as np


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [u, v, s, r]."""
    x1, y1, x2, y2 = bbox
    w = float(x2 - x1)
    h = float(y2 - y1)
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive bbox dims: w={w}, h={h}")
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """[u, v, s, r] -> [x1, y1, x2, y2]."""
    u, v, s, r = z[0], z[1], z[2], z[3]
    s = max(float(s), 1e-6)
    r = max(float(r), 1e-6)
    w = float(np.sqrt(s * r))
    h = s / w
    return np.array([u - w / 2, v - h / 2, u + w / 2, v + h / 2], dtype=np.float64)


class KalmanBoxTracker:
    """Single-object constant-velocity Kalman filter.

    One instance per track. `predict` is called every frame; `update` is called
    only on frames where the track is associated with a detection.
    """

    # State transition: position += velocity, velocity unchanged
    _F = np.eye(7, dtype=np.float64)
    _F[0, 4] = 1.0
    _F[1, 5] = 1.0
    _F[2, 6] = 1.0

    # Measurement projection: observe [u, v, s, r]
    _H = np.zeros((4, 7), dtype=np.float64)
    _H[0, 0] = 1.0
    _H[1, 1] = 1.0
    _H[2, 2] = 1.0
    _H[3, 3] = 1.0

    # Process / measurement noise (SORT defaults)
    _Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
    _R = np.diag([1.0, 1.0, 10.0, 10.0])

    # Initial covariance: high uncertainty on velocity components
    _P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def __init__(self, bbox: np.ndarray) -> None:
        z = bbox_to_z(np.asarray(bbox, dtype=np.float64))
        self.x = np.concatenate([z, np.zeros(3, dtype=np.float64)])
        self.P = self._P0.copy()

    def predict(self) -> np.ndarray:
        # If scale velocity would push area non-positive, zero it out before propagating
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return z_to_bbox(self.x[:4])

    def update(self, bbox: np.ndarray) -> None:
        z = bbox_to_z(np.asarray(bbox, dtype=np.float64))
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self._H) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        return z_to_bbox(self.x[:4])

    def mahalanobis_sq(self, bbox: np.ndarray) -> float:
        """Squared Mahalanobis distance from this track's prediction to bbox.

        Used by DeepSORT for motion gating.
        """
        z = bbox_to_z(np.asarray(bbox, dtype=np.float64))
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        return float(y @ np.linalg.solve(S, y))

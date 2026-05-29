import numpy as np
import pytest

from services.trackers.kalman import KalmanBoxTracker, bbox_to_z, z_to_bbox


def test_bbox_z_roundtrip() -> None:
    bbox = np.array([10.0, 20.0, 50.0, 100.0])
    out = z_to_bbox(bbox_to_z(bbox))
    assert np.allclose(out, bbox)


def test_predict_with_constant_velocity() -> None:
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.x[4] = 5.0
    kf.x[5] = -3.0
    pred = kf.predict()
    expected_center_u = 150.0 + 5.0
    expected_center_v = 150.0 - 3.0
    w = pred[2] - pred[0]
    h = pred[3] - pred[1]
    assert ((pred[0] + pred[2]) / 2) == pytest.approx(expected_center_u, abs=1e-6)
    assert ((pred[1] + pred[3]) / 2) == pytest.approx(expected_center_v, abs=1e-6)
    assert w * h == pytest.approx(100 * 100, abs=1e-6)


def test_update_pulls_state_to_measurement() -> None:
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.predict()
    kf.update(np.array([110.0, 110.0, 210.0, 210.0]))
    bbox = kf.bbox
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    assert 100.0 < cx < 160.0
    assert 100.0 < cy < 160.0


def test_track_linear_trajectory() -> None:
    """Object moves at constant velocity for 10 frames; Kalman should track it tightly."""
    bbox0 = np.array([100.0, 100.0, 150.0, 200.0])
    kf = KalmanBoxTracker(bbox0)
    rng = np.random.default_rng(0)
    last_err = None
    for t in range(1, 11):
        kf.predict()
        truth = bbox0 + np.array([5.0 * t, 2.0 * t, 5.0 * t, 2.0 * t])
        noisy = truth + rng.normal(0, 0.5, size=4)
        kf.update(noisy)
        last_err = np.linalg.norm(kf.bbox - truth)
    assert last_err is not None and last_err < 3.0

import numpy as np

from services.trackers.base import Detection
from services.trackers.bytetrack import ByteTrack


def _det(x1, y1, x2, y2, s=0.9) -> Detection:
    return Detection(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), score=s)


def test_high_conf_creates_track() -> None:
    tracker = ByteTrack(n_init=2, new_track_threshold=0.7)
    tracker.update([_det(100, 100, 150, 200, s=0.9)])
    out = tracker.update([_det(105, 100, 155, 200, s=0.9)])
    assert len(out) == 1


def test_low_conf_alone_does_not_create_track() -> None:
    tracker = ByteTrack(n_init=1, new_track_threshold=0.7)
    for _ in range(5):
        out = tracker.update([_det(100, 100, 150, 200, s=0.3)])
    assert out == []


def test_low_conf_rescues_tracked_object() -> None:
    """A confirmed track should survive a frame with only a low-conf detection on it."""
    tracker = ByteTrack(n_init=1, new_track_threshold=0.7, max_age=2)
    tracker.update([_det(100, 100, 150, 200, s=0.9)])
    out1 = tracker.update([_det(105, 100, 155, 200, s=0.9)])
    original_id = out1[0].track_id
    out2 = tracker.update([_det(110, 100, 160, 200, s=0.3)])
    assert out2 and out2[0].track_id == original_id


def test_two_objects_independent_ids() -> None:
    tracker = ByteTrack(n_init=1, new_track_threshold=0.7)
    tracker.update([_det(100, 100, 150, 200), _det(400, 400, 450, 500)])
    out = tracker.update([_det(105, 100, 155, 200), _det(405, 400, 455, 500)])
    assert len({t.track_id for t in out}) == 2

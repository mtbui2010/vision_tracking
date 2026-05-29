"""Smoke tests that should always pass — catch import-time regressions."""

from services.trackers import BaseTracker, Detection, Track


def test_detection_dataclass_constructs() -> None:
    import numpy as np

    d = Detection(bbox=np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32), score=0.9)
    assert d.score == 0.9
    assert d.embedding is None


def test_track_dataclass_constructs() -> None:
    import numpy as np

    t = Track(track_id=1, bbox=np.zeros(4, dtype=np.float32), score=0.8)
    assert t.track_id == 1


def test_base_tracker_is_abstract() -> None:
    import pytest

    with pytest.raises(TypeError):
        BaseTracker()  # type: ignore[abstract]

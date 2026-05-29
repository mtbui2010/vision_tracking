import numpy as np

from services.reid import HashEmbedder
from services.trackers.base import Detection
from services.trackers.deepsort import DeepSORT


def _det(x1, y1, x2, y2, s=0.9, emb=None) -> Detection:
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        score=s,
        embedding=emb,
    )


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)


def test_single_object_confirmed_after_n_init() -> None:
    tracker = DeepSORT(embedder=HashEmbedder(), n_init=3, max_age=10)
    f = _frame()
    for t in range(2):
        out = tracker.update([_det(100 + t * 5, 100, 150 + t * 5, 200)], frame=f)
        assert out == []
    out = tracker.update([_det(110, 100, 160, 200)], frame=f)
    assert len(out) == 1


def test_two_objects_get_distinct_ids() -> None:
    tracker = DeepSORT(embedder=HashEmbedder(), n_init=1, max_age=10)
    f = _frame()
    for _ in range(2):
        tracker.update([_det(100, 100, 150, 200), _det(400, 400, 450, 500)], frame=f)
    out = tracker.update([_det(110, 100, 160, 200), _det(410, 400, 460, 500)], frame=f)
    assert len({t.track_id for t in out}) == 2


def test_handles_no_detection_frame() -> None:
    tracker = DeepSORT(embedder=HashEmbedder(), n_init=1, max_age=10)
    f = _frame()
    tracker.update([_det(100, 100, 150, 200)], frame=f)
    assert tracker.update([], frame=f) == []


def test_id_persists_through_gap() -> None:
    tracker = DeepSORT(embedder=HashEmbedder(), n_init=1, max_age=10)
    f = _frame()
    for _ in range(2):
        tracker.update([_det(100, 100, 150, 200)], frame=f)
    out1 = tracker.update([_det(105, 100, 155, 200)], frame=f)
    original_id = out1[0].track_id
    for _ in range(3):
        tracker.update([], frame=f)
    out2 = tracker.update([_det(120, 100, 170, 200)], frame=f)
    assert out2 and out2[0].track_id == original_id

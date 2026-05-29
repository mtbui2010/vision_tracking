import numpy as np

from services.reid import HashEmbedder
from services.trackers.base import Detection
from services.trackers.custom import CustomTracker


def _det(x1, y1, x2, y2, s=0.9, emb=None) -> Detection:
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        score=s,
        embedding=emb,
    )


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)


def test_high_conf_path_works() -> None:
    tracker = CustomTracker(embedder=HashEmbedder(), n_init=1, new_track_threshold=0.7)
    f = _frame()
    tracker.update([_det(100, 100, 150, 200, s=0.9)], frame=f)
    out = tracker.update([_det(105, 100, 155, 200, s=0.9)], frame=f)
    assert len(out) == 1


def test_low_conf_alone_does_not_create_track() -> None:
    tracker = CustomTracker(embedder=HashEmbedder(), n_init=1, new_track_threshold=0.7)
    f = _frame()
    for _ in range(5):
        out = tracker.update([_det(100, 100, 150, 200, s=0.3)], frame=f)
    assert out == []


def test_low_conf_rescue_requires_appearance_match() -> None:
    """A low-confidence detection at the same location should still rescue a
    confirmed track if and only if the appearance also matches.

    With HashEmbedder, deterministic crops at the same position produce
    similar embeddings — so the rescue SHOULD succeed in the easy case.
    """
    tracker = CustomTracker(
        embedder=HashEmbedder(), n_init=1, new_track_threshold=0.7, max_age=5
    )
    f = _frame()
    tracker.update([_det(100, 100, 150, 200, s=0.9)], frame=f)
    out1 = tracker.update([_det(105, 100, 155, 200, s=0.9)], frame=f)
    original_id = out1[0].track_id
    # Now a low-conf det at roughly the same position — same appearance
    out2 = tracker.update([_det(108, 100, 158, 200, s=0.3)], frame=f)
    assert out2 and out2[0].track_id == original_id


def test_two_objects_get_distinct_ids() -> None:
    tracker = CustomTracker(embedder=HashEmbedder(), n_init=1, new_track_threshold=0.7)
    f = _frame()
    tracker.update([_det(100, 100, 150, 200), _det(400, 400, 450, 500)], frame=f)
    out = tracker.update([_det(105, 100, 155, 200), _det(405, 400, 455, 500)], frame=f)
    assert len({t.track_id for t in out}) == 2


def test_appears_in_registry() -> None:
    from services.trackers.registry import available, build

    assert "custom" in available()
    t = build("custom")
    assert t.name == "custom"

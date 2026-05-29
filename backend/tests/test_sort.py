import numpy as np

from services.trackers.base import Detection
from services.trackers.sort import SORT


def _det(x1: float, y1: float, x2: float, y2: float, s: float = 0.9) -> Detection:
    return Detection(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), score=s)


def test_single_object_constant_id() -> None:
    tracker = SORT(iou_threshold=0.3, max_age=1, min_hits=1)
    ids_seen = set()
    for t in range(10):
        x = 100 + t * 5
        tracks = tracker.update([_det(x, 100, x + 50, 200)])
        assert len(tracks) == 1
        ids_seen.add(tracks[0].track_id)
    assert len(ids_seen) == 1


def test_two_objects_get_distinct_ids() -> None:
    tracker = SORT(iou_threshold=0.3, max_age=1, min_hits=1)
    tracks = tracker.update([_det(100, 100, 150, 200), _det(400, 400, 450, 500)])
    assert {t.track_id for t in tracks} == {1, 2}


def test_lost_object_after_max_age() -> None:
    tracker = SORT(iou_threshold=0.3, max_age=2, min_hits=1)
    tracker.update([_det(100, 100, 150, 200)])
    for _ in range(5):
        tracker.update([])
    after = tracker.update([_det(100, 100, 150, 200)])
    assert after[0].track_id != 1


def test_handles_empty_frame() -> None:
    tracker = SORT(min_hits=1)
    assert tracker.update([]) == []


def test_id_persists_across_short_gap() -> None:
    tracker = SORT(iou_threshold=0.3, max_age=5, min_hits=1)
    first = tracker.update([_det(100, 100, 150, 200)])
    original_id = first[0].track_id
    for _ in range(2):
        tracker.update([])
    later = tracker.update([_det(100, 100, 150, 200)])
    assert later and later[0].track_id == original_id

import numpy as np

from services.metrics import FrameAnnotations, compute_metrics


def _ann(ids, boxes) -> FrameAnnotations:
    return FrameAnnotations(ids=np.array(ids, dtype=np.int64), bboxes=np.array(boxes, dtype=np.float64))


def test_perfect_tracker_scores_one() -> None:
    gt = [
        _ann([1], [[0, 0, 10, 10]]),
        _ann([1], [[5, 0, 15, 10]]),
        _ann([1], [[10, 0, 20, 10]]),
    ]
    m = compute_metrics(gt, gt)
    assert m.mota == 1.0
    assert m.idf1 == 1.0
    assert m.hota >= 0.99
    assert m.fp == 0 and m.fn == 0 and m.idsw == 0


def test_all_missed_detections() -> None:
    gt = [_ann([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]])]
    pred = [_ann([], np.zeros((0, 4)))]
    m = compute_metrics(gt, pred)
    assert m.fn == 2
    assert m.fp == 0
    assert m.mota == 0.0  # 1 - 2/2
    assert m.idf1 == 0.0


def test_all_false_positives() -> None:
    gt = [_ann([], np.zeros((0, 4)))]
    pred = [_ann([1], [[0, 0, 10, 10]])]
    m = compute_metrics(gt, pred)
    assert m.fp == 1
    assert m.fn == 0
    assert m.idf1 == 0.0


def test_id_switch_detected() -> None:
    gt = [
        _ann([1], [[0, 0, 10, 10]]),
        _ann([1], [[5, 0, 15, 10]]),
    ]
    pred = [
        _ann([1], [[0, 0, 10, 10]]),
        _ann([2], [[5, 0, 15, 10]]),  # same GT, different pred id
    ]
    m = compute_metrics(gt, pred)
    assert m.idsw == 1


def test_mota_formula() -> None:
    gt = [
        _ann([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
        _ann([1, 2], [[5, 0, 15, 10], [20, 20, 30, 30]]),
    ]
    pred = [
        _ann([1], [[0, 0, 10, 10]]),
        _ann([1, 9], [[5, 0, 15, 10], [200, 200, 210, 210]]),
    ]
    m = compute_metrics(gt, pred)
    assert m.num_gt == 4
    assert m.fn == 2
    assert m.fp == 1
    assert m.mota == 1.0 - (1 + 2 + 0) / 4

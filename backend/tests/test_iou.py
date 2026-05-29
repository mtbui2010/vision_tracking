import numpy as np
import pytest

from services.trackers.iou import iou_batch


def test_identical_boxes_iou_one() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_batch(a, a)[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_iou_zero() -> None:
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    b = np.array([[5.0, 5.0, 6.0, 6.0]])
    assert iou_batch(a, b)[0, 0] == 0.0


def test_half_overlap() -> None:
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[1.0, 0.0, 3.0, 2.0]])
    # intersection = 2, union = 6
    assert iou_batch(a, b)[0, 0] == pytest.approx(2 / 6)


def test_shape() -> None:
    a = np.zeros((3, 4))
    b = np.zeros((5, 4))
    assert iou_batch(a, b).shape == (3, 5)


def test_empty() -> None:
    a = np.zeros((0, 4))
    b = np.zeros((3, 4))
    assert iou_batch(a, b).shape == (0, 3)
    assert iou_batch(b, a).shape == (3, 0)

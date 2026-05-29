import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment as scipy_lsa

from services.trackers.hungarian import linear_sum_assignment


def _total_cost(C: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return float(C[rows, cols].sum())


def test_2x2_trivial() -> None:
    C = np.array([[1.0, 2.0], [2.0, 1.0]])
    rows, cols = linear_sum_assignment(C)
    assert sorted(zip(rows, cols)) == [(0, 0), (1, 1)]


def test_2x2_swapped() -> None:
    C = np.array([[3.0, 1.0], [1.0, 3.0]])
    rows, cols = linear_sum_assignment(C)
    assert sorted(zip(rows, cols)) == [(0, 1), (1, 0)]


def test_rectangular_more_cols() -> None:
    C = np.array([[1.0, 5.0, 9.0], [2.0, 4.0, 8.0]])
    rows, cols = linear_sum_assignment(C)
    assert len(rows) == 2
    assert _total_cost(C, rows, cols) == 5.0


def test_rectangular_more_rows() -> None:
    C = np.array([[1.0, 5.0], [2.0, 4.0], [9.0, 8.0]])
    rows, cols = linear_sum_assignment(C)
    assert len(rows) == 2


def test_empty() -> None:
    rows, cols = linear_sum_assignment(np.zeros((0, 3)))
    assert rows.size == 0 and cols.size == 0


def test_matches_scipy_random() -> None:
    rng = np.random.default_rng(seed=42)
    for _ in range(20):
        n = rng.integers(2, 12)
        m = rng.integers(2, 12)
        C = rng.uniform(0, 10, size=(n, m))
        ours = linear_sum_assignment(C)
        theirs = scipy_lsa(C)
        ours_cost = _total_cost(C, ours[0], ours[1])
        theirs_cost = _total_cost(C, theirs[0], theirs[1])
        assert ours_cost == pytest.approx(theirs_cost, abs=1e-9), (
            f"shape={C.shape}, ours={ours_cost}, scipy={theirs_cost}"
        )


def test_matches_scipy_with_zeros() -> None:
    C = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    ours = linear_sum_assignment(C)
    theirs = scipy_lsa(C)
    assert _total_cost(C, *ours) == pytest.approx(_total_cost(C, *theirs))

"""Hungarian algorithm (Munkres) for linear sum assignment.

O(n^3). Handles rectangular cost matrices by zero-padding to square. No
`scipy.optimize.linear_sum_assignment`.

Reference: Munkres, "Algorithms for the Assignment and Transportation
Problems" (1957), with the matrix-marking refinement.
"""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a min-cost assignment of rows to columns.

    Args:
        cost_matrix: shape (n_rows, n_cols), real-valued.

    Returns:
        (row_ind, col_ind) of equal length k = min(n_rows, n_cols),
        such that ``cost_matrix[row_ind, col_ind].sum()`` is minimized
        subject to each row and column being used at most once.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    pad_value = float(cost.max()) + 1.0
    square = np.full((n, n), pad_value, dtype=np.float64)
    square[:n_rows, :n_cols] = cost

    rows, cols = _munkres(square)

    valid = (rows < n_rows) & (cols < n_cols)
    return rows[valid], cols[valid]


_STAR = 1
_PRIME = 2


def _munkres(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = cost.shape[0]
    C = cost - cost.min(axis=1, keepdims=True)
    C = C - C.min(axis=0, keepdims=True)

    mark = np.zeros((n, n), dtype=np.int8)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if C[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                mark[i, j] = _STAR
                row_covered[i] = True
                col_covered[j] = True
    row_covered[:] = False
    col_covered[:] = False

    col_covered[:] = (mark == _STAR).any(axis=0)

    while col_covered.sum() < n:
        while True:
            loc = _find_uncovered_zero(C, row_covered, col_covered)
            if loc is None:
                _adjust_matrix(C, row_covered, col_covered)
                continue

            i, j = loc
            mark[i, j] = _PRIME
            star_col = _find_in_row(mark, i, _STAR)
            if star_col is None:
                _augment(mark, i, j)
                row_covered[:] = False
                col_covered[:] = (mark == _STAR).any(axis=0)
                mark[mark == _PRIME] = 0
                break
            row_covered[i] = True
            col_covered[star_col] = False

    rows, cols = np.where(mark == _STAR)
    order = np.argsort(rows)
    return rows[order].astype(np.intp), cols[order].astype(np.intp)


def _find_uncovered_zero(
    C: np.ndarray, row_covered: np.ndarray, col_covered: np.ndarray
) -> tuple[int, int] | None:
    uncov = (C == 0) & (~row_covered)[:, None] & (~col_covered)[None, :]
    idx = np.argwhere(uncov)
    if idx.size == 0:
        return None
    return int(idx[0, 0]), int(idx[0, 1])


def _find_in_row(mark: np.ndarray, row: int, value: int) -> int | None:
    cols = np.where(mark[row] == value)[0]
    return int(cols[0]) if cols.size else None


def _find_in_col(mark: np.ndarray, col: int, value: int) -> int | None:
    rows = np.where(mark[:, col] == value)[0]
    return int(rows[0]) if rows.size else None


def _adjust_matrix(C: np.ndarray, row_covered: np.ndarray, col_covered: np.ndarray) -> None:
    mask = (~row_covered)[:, None] & (~col_covered)[None, :]
    min_val = C[mask].min()
    C[row_covered, :] += min_val
    C[:, ~col_covered] -= min_val


def _augment(mark: np.ndarray, start_row: int, start_col: int) -> None:
    path = [(start_row, start_col)]
    while True:
        col = path[-1][1]
        star_row = _find_in_col(mark, col, _STAR)
        if star_row is None:
            break
        path.append((star_row, col))
        prime_col = _find_in_row(mark, star_row, _PRIME)
        assert prime_col is not None
        path.append((star_row, prime_col))

    for r, c in path:
        if mark[r, c] == _STAR:
            mark[r, c] = 0
        elif mark[r, c] == _PRIME:
            mark[r, c] = _STAR

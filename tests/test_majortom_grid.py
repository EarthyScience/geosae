"""MajorTOM cell naming: the ids we publish must be the ids MajorTOM uses.

A cell id is how a reader pulls our exact cells from the upstream release, so
what is pinned here is the naming convention itself: ``0U`` is the equator row,
``0R`` the prime-meridian column of *that* row, and the id of a cell's centre
names that cell.
"""

import numpy as np
import pytest

from geosae.data.download.grid import MajorTomGrid


@pytest.fixture(scope="module")
def grid():
    return MajorTomGrid(5)


def _row_start(grid, row):
    return int(grid.flat_index(row, 0))


def test_row_zero_is_the_equator_and_column_zero_the_prime_meridian(grid):
    eq = grid.equator_row
    assert grid.row_lats[eq] == 0.0
    assert grid.cell_id(eq, 0).startswith("0U_") and grid.cell_id(eq - 1, 0).startswith("1D_")
    assert grid.cell_id(eq + 3, 0).startswith("3U_") and grid.cell_id(eq - 3, 0).startswith("3D_")

    west = grid.bounds[_row_start(grid, eq) : _row_start(grid, eq + 1), 0]
    zero = int(np.flatnonzero(west == 0.0)[0])
    assert grid.cell_id(eq, zero) == "0U_0R"
    assert grid.cell_id(eq, zero + 2) == "0U_2R" and grid.cell_id(eq, zero - 2) == "0U_2L"


def test_names_are_not_truncated_at_high_row_or_column_numbers(grid):
    row, col = grid.cell_id(len(grid.row_lats) - 1, 0).split("_")
    assert len(row) > 4 and row.endswith("U")  # '1892U', not '189'
    assert col.endswith("L")


def test_the_id_of_a_cell_centre_names_that_cell(grid):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(grid), 25, replace=False)
    lon, lat = grid.centers[idx].T
    rows, cols = grid.rowcol(lon, lat)
    assert np.array_equal(grid.flat_index(rows, cols), idx)


def test_column_naming_is_per_row_not_per_grid(grid):
    """Rows shrink towards the poles, so the same column index is a different cell."""
    eq = grid.equator_row
    polar = len(grid.row_lats) - 10
    assert grid.n_cols[polar] < grid.n_cols[eq]
    assert grid.cell_id(polar, 100).split("_")[1] != grid.cell_id(eq, 100).split("_")[1]


def test_a_point_west_of_a_rows_first_edge_lies_in_its_last_cell(grid):
    """Polar rows do not divide evenly: the last cell wraps across the antimeridian."""
    assert grid.bounds[0, 0] > -180.0
    row, col = grid.rowcol(-180.0, grid.centers[0, 1])
    assert (int(row[0]), int(col[0])) == (0, grid.n_cols[0] - 1)


def test_a_latitude_outside_the_rows_is_rejected(grid):
    with pytest.raises(ValueError, match="latitude"):
        grid.rowcol(0.0, 89.0)

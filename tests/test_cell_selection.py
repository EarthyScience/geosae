"""Choosing the cells of a store: ids, points, boxes and seeded samples name the right MajorTOM cells."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.data.download import grid as G
from geosae.data.download.grid import MajorTomGrid, cells_at, cells_from_ids, cells_in_bbox, select_cells


@pytest.fixture(scope="module")
def grid():
    return MajorTomGrid(5)


def test_a_cell_id_parses_back_to_its_row_and_column(grid):
    rng = np.random.default_rng(0)
    for idx in rng.choice(len(grid), 50, replace=False):
        row, col = grid.rowcol(*grid.centers[idx])
        assert grid.parse_cell_id(grid.cell_id(int(row[0]), int(col[0]))) == (int(row[0]), int(col[0]))


@pytest.mark.parametrize("bad", ["", "645U", "645X_1077R", "645U_1077Q", "U_R", "99999U_0R", "0U_99999R"])
def test_a_malformed_or_foreign_cell_id_is_rejected(grid, bad):
    with pytest.raises(ValueError):
        grid.parse_cell_id(bad)


def test_ids_and_points_name_the_same_cells():
    lon, lat = [55.311564237843996, 11.5, -73.9], [28.98952095808383, 48.1, 40.7]
    assert np.array_equal(cells_from_ids(["645U_1077R", "1071U_170R", "906U_1248L"]), cells_at(lon, lat))


def test_a_point_lies_inside_the_bounds_of_its_cell(grid):
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-180, 180, 200), rng.uniform(-80, 80, 200)
    west, south, east, north = grid.bounds[cells_at(lon, lat)].T
    # The last cell of a row wraps the antimeridian, so compare longitudes modulo 360.
    assert np.all((lat >= south) & (lat < north))
    assert np.all(np.mod(lon - west, 360) < np.mod(east - west, 360) + 1e-9)


def test_a_box_holds_exactly_the_cells_whose_centre_it_contains(grid):
    bbox = (11.3, 50.7, 11.8, 51.0)
    cells = cells_in_bbox(bbox, land_fraction=None)
    lon, lat = grid.centers[:, 0], grid.centers[:, 1]
    expected = np.flatnonzero((lon >= 11.3) & (lon < 11.8) & (lat >= 50.7) & (lat < 51.0))
    assert cells.size > 0 and np.array_equal(cells, expected)


@pytest.mark.parametrize("bbox", [(12, 50, 11, 51), (11, 51, 12, 50)])
def test_an_empty_or_antimeridian_box_is_rejected(bbox):
    with pytest.raises(ValueError):
        cells_in_bbox(bbox, land_fraction=None)


def test_a_box_keeps_only_land_cells_when_asked(grid, monkeypatch):
    shapely = pytest.importorskip("shapely")
    # "Land" is the western half of the box: only cells mostly west of 11.55 E survive.
    monkeypatch.setattr(G, "land_mask", lambda: shapely.box(0, 0, 11.55, 60))
    land = cells_in_bbox((11.3, 50.7, 11.8, 51.0), land_fraction=0.5)
    every = cells_in_bbox((11.3, 50.7, 11.8, 51.0), land_fraction=None)
    assert 0 < land.size < every.size and np.isin(land, every).all()
    assert np.all(grid.centers[land, 0] < 11.55 + 0.05)


def test_a_seeded_sample_is_reproducible_and_drawn_from_the_box(monkeypatch):
    shapely = pytest.importorskip("shapely")
    monkeypatch.setattr(G, "land_mask", lambda: shapely.box(-180, -90, 180, 90))
    box = (11.0, 50.0, 12.0, 51.0)
    a = select_cells(n=20, bbox=box, seed=3)
    assert a.size == 20 and np.unique(a).size == 20
    assert np.array_equal(a, select_cells(n=20, bbox=box, seed=3))
    assert not np.array_equal(a, select_cells(n=20, bbox=box, seed=4))
    assert np.isin(a, cells_in_bbox(box, land_fraction=None)).all()


def test_points_in_one_cell_select_it_once(grid):
    (lon_a, lat_a), (lon_b, lat_b) = grid.centers[[17_798_802, 16_852_988]]
    cells = select_cells(points=[[lon_a, lat_a], [lon_a + 0.001, lat_a + 0.001], [lon_b, lat_b]])
    assert cells.tolist() == [17_798_802, 16_852_988]


@pytest.mark.parametrize("kwargs", [{}, {"n": 5, "ids": ["0U_0R"]}, {"points": [[0, 0]], "ids": ["0U_0R"]}])
def test_exactly_one_way_of_choosing_is_required(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        select_cells(**kwargs)

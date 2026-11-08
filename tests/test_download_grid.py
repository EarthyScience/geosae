"""The MajorTOM grid and everything persisted from it, pinned value for value.

The store's ``cell_indices``/``lon``/``lat`` and the released cell ids are
published, so the grid arithmetic behind them must never drift: every value here
is compared exactly, not to a tolerance. The expected numbers were recorded from
the implementation that built the published store.
"""

import hashlib

import numpy as np
import pytest

from geosae.data.download.presets import MAJORTOM210K_BASE_NCELLS, MAJORTOM210K_DRAW_SIZE
from geosae.data.download.selection import sample_subset_indices

# --------------------------------------------------------------------------- #
# API under test
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def grid():
    from geosae.data.download.grid import MajorTomGrid

    return MajorTomGrid(5)


def _n_rows(g):
    return len(g.row_lats)


def _n_cols(g):
    return g.n_cols


def _row_lats(g):
    return g.row_lats


def _equator_row(g):
    return g.equator_row


def _centers(g):
    return g.centers


def _bounds(g, idx):
    return g.bounds[np.asarray(idx)]


def _overlapping(g, geometry, min_fraction):
    return g.cells_overlapping(geometry, min_fraction)


def _cell_ids(lon, lat):
    from geosae.data.download.release import majortom_cell_ids

    return majortom_cell_ids(np.asarray(lon), np.asarray(lat))


def _utm_geobox(lon, lat):
    pytest.importorskip("rasterio")
    from geosae.data.download.store import utm_geobox

    return utm_geobox(lon, lat)


# --------------------------------------------------------------------------- #
# Expected values
# --------------------------------------------------------------------------- #

#: The first, prime-meridian and last cell of rows at both grid edges, +-45 deg
#: and the equator: ``(flat_index, centre_lon, centre_lat)``...
CENTERS = [
    (0, -179.48790896159318, -84.94760479041916),
    (351, 0.25604551920341395, -84.94760479041916),
    (702, 179.99999999999997, -84.94760479041916),
    (703, -179.49224259520452, -84.90269461077845),
    (1057, 0.2538787023977433, -84.90269461077845),
    (1411, 180.0, -84.90269461077845),
    (2953555, -179.9682427664079, -44.977544910179645),
    (2956389, 0.03175723359209598, -44.977544910179645),
    (2959222, 179.9682427664079, -44.977544910179645),
    (10175324, -179.97754491017963, -0.0224550898203546),
    (10179332, 0.02245508982035928, -0.0224550898203546),
    (10183339, 179.97754491017963, -0.0224550898203546),
    (10183340, -179.97754491017963, 0.0224550898203546),
    (10187348, 0.02245508982035928, 0.0224550898203546),
    (10191355, 179.97754491017963, 0.0224550898203546),
    (10191356, -179.97754491017963, 0.06736526946107801),
    (10195364, 0.02245508982035928, 0.06736526946107801),
    (10199371, 179.97754491017963, 0.06736526946107801),
    (17415473, -179.9682427664079, 45.022455089820355),
    (17418307, 0.03175723359209598, 45.022455089820355),
    (17421140, 179.9682427664079, 45.022455089820355),
    (20373284, -179.49224259520452, 84.94760479041916),
    (20373638, 0.2538787023977433, 84.94760479041916),
    (20373992, 180.0, 84.94760479041916),
    (20373993, -179.48790896159318, 84.99251497005987),
    (20374344, 0.25604551920341395, 84.99251497005987),
    (20374695, 179.99999999999997, 84.99251497005987),
]

#: ...and their ``(flat_index, west, south, east, north)`` bounds.
BOUNDS = [
    (0, -179.7439544807966, -84.97005988023952, -179.23186344238977, -84.92514970059881),
    (351, 0.0, -84.97005988023952, 0.5120910384068279, -84.92514970059881),
    (702, 179.74395448079656, -84.97005988023952, 180.25604551920338, -84.92514970059881),
    (703, -179.74612129760226, -84.92514970059881, -179.23836389280677, -84.8802395209581),
    (1057, 0.0, -84.92514970059881, 0.5077574047954866, -84.8802395209581),
    (1411, 179.74612129760226, -84.92514970059881, 180.25387870239774, -84.8802395209581),
    (2953555, -180.0, -45.0, -179.9364855328158, -44.95508982035929),
    (2956389, 0.0, -45.0, 0.06351446718419196, -44.95508982035929),
    (2959222, 179.93648553281582, -45.0, 180.00000000000003, -44.95508982035929),
    (10175324, -180.0, -0.0449101796407092, -179.9550898203593, 0.0),
    (10179332, 0.0, -0.0449101796407092, 0.04491017964071856, 0.0),
    (10183339, 179.95508982035926, -0.0449101796407092, 179.99999999999997, 0.0),
    (10183340, -180.0, 0.0, -179.9550898203593, 0.0449101796407092),
    (10187348, 0.0, 0.0, 0.04491017964071856, 0.0449101796407092),
    (10191355, 179.95508982035926, 0.0, 179.99999999999997, 0.0449101796407092),
    (10191356, -180.0, 0.04491017964072341, -179.9550898203593, 0.08982035928143262),
    (10195364, 0.0, 0.04491017964072341, 0.04491017964071856, 0.08982035928143262),
    (10199371, 179.95508982035926, 0.04491017964072341, 179.99999999999997, 0.08982035928143262),
    (17415473, -180.0, 45.0, -179.9364855328158, 45.04491017964071),
    (17418307, 0.0, 45.0, 0.06351446718419196, 45.04491017964071),
    (17421140, 179.93648553281582, 45.0, 180.00000000000003, 45.04491017964071),
    (20373284, -179.74612129760226, 84.92514970059881, -179.23836389280677, 84.97005988023952),
    (20373638, 0.0, 84.92514970059881, 0.5077574047954866, 84.97005988023952),
    (20373992, 179.74612129760226, 84.92514970059881, 180.25387870239774, 84.97005988023952),
    (20373993, -179.7439544807966, 84.97005988023952, -179.23186344238977, 85.01497005988023),
    (20374344, 0.0, 84.97005988023952, 0.5120910384068279, 85.01497005988023),
    (20374695, 179.74395448079656, 84.97005988023952, 180.25604551920338, 85.01497005988023),
]

#: ``(lon, lat, cell_id, row, col)``: the equator, the antimeridian on both sides,
#: a few cities, and the corner cells of the southernmost and northernmost rows.
CELL_IDS = [
    (0.0, 0.0, "0U_0R", 1892, 4008),
    (1e-09, -1e-09, "1D_0R", 1891, 4008),
    (179.999, 0.01, "0U_4007R", 1892, 8015),
    (-179.999, 0.01, "0U_4008L", 1892, 0),
    (11.5, 48.1, "1071U_170R", 2963, 2846),
    (-73.9, 40.7, "906U_1248L", 2798, 1791),
    (139.7, 35.7, "794U_2527R", 2686, 5783),
    (-68.3, -54.8, "1221D_876L", 671, 1432),
    (-179.48790896159318, -84.94760479041916, "1892D_351L", 0, 0),
    (179.99999999999997, -84.94760479041916, "1892D_351R", 0, 702),
    (-179.48790896159318, 84.99251497005987, "1892U_351L", 3784, 0),
    (179.99999999999997, 84.99251497005987, "1892U_351R", 3784, 702),
]


def test_rows_and_columns(grid):
    assert _n_rows(grid) == 3785
    assert int(_n_cols(grid).sum()) == 20_374_696
    eq = _equator_row(grid)
    assert eq == 1892
    lats = _row_lats(grid)
    assert lats[eq] == 0.0
    assert float(lats[1] - lats[0]) == 0.0449101796407092
    assert lats[0] == -84.97005988023952 and lats[-1] == 84.97005988023952
    rows = [0, 1, 890, 1891, 1892, 1893, 2894, 3783, 3784]
    assert _n_cols(grid)[rows].tolist() == [703, 709, 5668, 8016, 8016, 8016, 5668, 709, 703]


def test_every_cell_centre_is_unchanged(grid):
    """A fingerprint of all 20,374,696 centres, the array store ``lon``/``lat`` are read from."""
    centers = np.ascontiguousarray(_centers(grid), dtype=np.float64)
    assert centers.shape == (20_374_696, 2)
    digest = hashlib.sha256(centers.tobytes()).hexdigest()
    assert digest == "5ba7a6e717c7d19241764b5b64e2bb9758554e75fc7086c48a6861ba645a5146"


@pytest.mark.parametrize(("idx", "lon", "lat"), CENTERS)
def test_cell_centre(grid, idx, lon, lat):
    assert _centers(grid)[idx].tolist() == [lon, lat]


@pytest.mark.parametrize(("idx", "west", "south", "east", "north"), BOUNDS)
def test_cell_bounds(grid, idx, west, south, east, north):
    assert _bounds(grid, [idx])[0].tolist() == [west, south, east, north]


@pytest.mark.parametrize(("lon", "lat", "cell_id", "row", "col"), CELL_IDS)
def test_cell_id(grid, lon, lat, cell_id, row, col):
    ids, rows, cols = _cell_ids([lon], [lat])
    assert (str(ids[0]), int(rows[0]), int(cols[0])) == (cell_id, row, col)


def test_cell_ids_of_the_cell_centres_round_trip_to_their_rows(grid):
    """The id of a cell's own centre names that cell, across all latitudes."""
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(20_374_696, 200, replace=False))
    centers = _centers(grid)[idx]
    _, rows, cols = _cell_ids(centers[:, 0], centers[:, 1])
    offsets = np.concatenate([[0], np.cumsum(_n_cols(grid))[:-1]])
    assert np.array_equal(offsets[rows] + cols, idx)


@pytest.mark.parametrize(
    ("lon", "lat", "crs", "params"),
    [
        (-180.0, 0.0, "EPSG:32601", [10.0, 0.0, 165701.44308054057, 0.0, -10.0, 320.0]),
        (179.99, 0.0, "EPSG:32660", [10.0, 0.0, 832544.2750236953, 0.0, -10.0, 320.0]),
        (3.0, 60.0, "EPSG:32631", [10.0, 0.0, 499680.00000000047, 0.0, -10.0, 6651731.190362714]),
        (11.5, 84.0, "EPSG:32632", [10.0, 0.0, 528846.1961343538, 0.0, -10.0, 9329046.75510543]),
        (-70.0, -84.0, "EPSG:32719", [10.0, 0.0, 488010.4792398149, 0.0, -10.0, 672124.888899656]),
        (0.0, -0.001, "EPSG:32731", [10.0, 0.0, 165701.44313116156, 0.0, -10.0, 10000209.317244012]),
    ],
)
def test_patch_crs_and_transform(lon, lat, crs, params):
    """A patch is 64 x 10 m pixels in the plain UTM zone of its centre (no Norway exceptions)."""
    geobox, crs_str, got = _utm_geobox(lon, lat)
    assert crs_str == crs
    assert got.tolist() == params
    assert tuple(geobox.shape) == (64, 64)


def _alps():
    import shapely

    return shapely.union(
        shapely.box(10.03, 47.02, 10.31, 47.23),
        shapely.Polygon([(10.31, 47.02), (10.6, 47.02), (10.31, 47.3)]),
    )


def _fiji():
    """Two polygons meeting at the antimeridian."""
    import shapely

    return shapely.MultiPolygon(
        [shapely.box(179.93, -16.2, 180.0, -16.02), shapely.box(-180.0, -16.2, -179.91, -16.02)]
    )


#: Expected cells as runs ``(first, stop)`` of consecutive flat indices.
@pytest.mark.parametrize(
    ("shape", "min_fraction", "runs"),
    [
        (
            _alps,
            0.1,
            [
                (17668965, 17668974),
                (17674428, 17674436),
                (17679885, 17679892),
                (17685337, 17685344),
                (17690786, 17690793),
                (17696234, 17696236),
            ],
        ),
        (
            _alps,
            0.5,
            [
                (17668965, 17668974),
                (17674428, 17674436),
                (17679885, 17679892),
                (17685338, 17685344),
                (17690787, 17690792),
            ],
        ),
        (
            _fiji,
            0.1,
            [
                (7328366, 7328368),
                (7336061, 7336065),
                (7343760, 7343764),
                (7351460, 7351464),
                (7359162, 7359166),
                (7366866, 7366868),
            ],
        ),
        (
            _fiji,
            0.5,
            [
                (7328366, 7328367),
                (7336061, 7336062),
                (7336063, 7336064),
                (7343760, 7343761),
                (7343762, 7343764),
                (7351461, 7351464),
                (7359163, 7359164),
            ],
        ),
    ],
)
def test_cells_overlapping_a_geometry(grid, shape, min_fraction, runs):
    """The land-cell rule: a cell counts once at least ``min_fraction`` of its area is covered."""
    pytest.importorskip("shapely")
    expected = [i for first, stop in runs for i in range(first, stop)]
    assert np.asarray(_overlapping(grid, shape(), min_fraction)).tolist() == expected


def test_store_rows_map_to_the_published_grid_cells():
    """``cell_indices`` of the published 1M store: MajorTOM land cells drawn with seed 42."""
    idx = sample_subset_indices(MAJORTOM210K_BASE_NCELLS, MAJORTOM210K_DRAW_SIZE, seed=42)
    assert idx[:5].tolist() == [2801816, 3078819, 5169808, 886522, 2202819]
    assert idx[-5:].tolist() == [4324988, 380998, 3323831, 3416182, 4720376]
    digest = hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()
    assert digest == "5227e859d3a2fe1584c55110fd350f154c30a069367800a4f88dfbdf752086af"

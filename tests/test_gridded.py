"""Contract tests for the regular lon/lat grid and patch-store coarsening."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from geosae.data.dataset import PatchData
from geosae.data.download.gridded import (
    RegularGrid,
    aggregate_categorical,
    aggregate_continuous,
    coarsen_patch_store,
)
from geosae.data.download.selection import SplitSpec
from geosae.data.download.store import build_schemas, create_store


# --------------------------------------------------------------------- grid
def test_global_grid_node_count_and_spacing():
    g = RegularGrid.global_grid(0.25)
    assert (g.n_lat, g.n_lon) == (721, 1440)
    assert g.n_cells == 721 * 1440
    lats, lons = g.lat_nodes(), g.lon_nodes()
    assert lats[0] == 90.0 and lats[-1] == -90.0
    assert lons[0] == -180.0 and lons[-1] == 179.75


def test_nearest_node_round_trips_node_coordinates():
    """A node's own coordinates must map back to that node."""
    g = RegularGrid.global_grid(0.25)
    rows = np.array([0, 98, 360, 720])
    cols = np.array([0, 1, 719, 1439])
    lon, lat = g.centers(g.flat_index_of(rows, cols))
    r2, c2 = g.nearest(lon, lat)
    assert np.array_equal(r2, rows)
    assert np.array_equal(c2, cols)


def test_nearest_node_snaps_within_half_a_cell():
    g = RegularGrid.global_grid(0.25)
    # 0.1 deg off a node is still that node; 0.2 deg off crosses to the next.
    r, c = g.nearest(np.array([-180.0 + 0.1]), np.array([90.0 - 0.1]))
    assert (int(r[0]), int(c[0])) == (0, 0)
    r, c = g.nearest(np.array([-180.0 + 0.2]), np.array([90.0 - 0.2]))
    assert (int(r[0]), int(c[0])) == (1, 1)


def test_longitude_wraps_at_the_antimeridian():
    g = RegularGrid.global_grid(0.25)
    _, c_west = g.nearest(np.array([-180.0]), np.array([0.0]))
    _, c_east = g.nearest(np.array([180.0]), np.array([0.0]))
    assert int(c_west[0]) == int(c_east[0]) == 0


def test_centers_returns_longitudes_in_minus180_180():
    g = RegularGrid.global_grid(0.25)
    lon, lat = g.centers(np.arange(g.n_cells, dtype=np.int64)[::9973])
    assert lon.min() >= -180.0 and lon.max() < 180.0
    assert lat.min() >= -90.0 and lat.max() <= 90.0


# -------------------------------------------------------------- aggregation
def test_categorical_aggregation_returns_the_majority_class():
    codes = np.array([[3, 3, 7], [3, 7, 7], [5, 5, 5]], dtype=np.uint8)
    group = np.array([0, 0, 1], dtype=np.int64)
    mode, frac, n = aggregate_categorical(codes, group, n_groups=2, fill_value=255)
    assert mode.tolist() == [3, 5]
    assert n.tolist() == [6, 3]
    # group 0 sees 3 three times out of six pixels.
    assert frac[0] == pytest.approx(0.5)
    assert frac[1] == pytest.approx(1.0)


def test_categorical_aggregation_ignores_the_fill_value():
    """Nodata pixels must not win the vote, nor count towards the total."""
    codes = np.array([[255, 255, 255, 4]], dtype=np.uint8)
    group = np.array([0], dtype=np.int64)
    mode, frac, n = aggregate_categorical(codes, group, n_groups=1, fill_value=255)
    assert int(mode[0]) == 4
    assert int(n[0]) == 1
    assert frac[0] == pytest.approx(1.0)


def test_categorical_aggregation_marks_all_nodata_groups_as_fill():
    codes = np.array([[255, 255]], dtype=np.uint8)
    group = np.array([0], dtype=np.int64)
    mode, frac, n = aggregate_categorical(codes, group, n_groups=2, fill_value=255)
    assert int(mode[0]) == 255 and int(mode[1]) == 255
    assert n.tolist() == [0, 0]
    assert np.isnan(frac).all()


def test_categorical_aggregation_handles_signed_and_wide_dtypes():
    """int8 / int16 codes (negative fills, >256 range) aggregate like uint8."""
    for dtype, fill in ((np.int8, -1), (np.int16, -200), (np.uint16, 0)):
        codes = np.array([[900 % np.iinfo(dtype).max, 7, 7]], dtype=dtype)
        codes[0, 0] = fill
        group = np.array([0], dtype=np.int64)
        mode, _, n = aggregate_categorical(codes, group, n_groups=1, fill_value=fill)
        assert int(mode[0]) == 7, dtype
        assert int(n[0]) == 2, dtype


def test_continuous_aggregation_is_the_nanmean_over_pixels():
    vals = np.array([[1.0, 3.0], [np.nan, 5.0], [np.nan, np.nan]], dtype=np.float32)
    group = np.array([0, 0, 1], dtype=np.int64)
    mean, n = aggregate_continuous(vals, group, n_groups=2)
    assert mean[0] == pytest.approx(3.0)  # (1+3+5)/3
    assert int(n[0]) == 3
    assert np.isnan(mean[1]) and int(n[1]) == 0


def test_continuous_aggregation_keeps_bands_separate():
    vals = np.array([[[1.0, 10.0], [3.0, 30.0]]], dtype=np.float32)  # (1 cell, 2 px, 2 bands)
    group = np.array([0], dtype=np.int64)
    mean, n = aggregate_continuous(vals, group, n_groups=1)
    assert mean.shape == (1, 2)
    assert mean[0].tolist() == pytest.approx([2.0, 20.0])
    assert n[0].tolist() == [2, 2]


# ---------------------------------------------------------------- coarsening
@pytest.fixture(scope="module")
def source_store(tmp_path_factory):
    """A tiny 4x4-pixel patch store whose cells fall into three 1-degree nodes."""
    rng = np.random.default_rng(0)
    n_src, px = 12, 4
    path = tmp_path_factory.mktemp("src") / "patches.zarr"
    schemas = build_schemas(["lc_esa_wc", "dem_copernicus"], n_src, embeddings=("alphaearth",), patch_size=px)
    g = create_store(path, schemas, n_cells=n_src, patch_size=px)
    # Four source cells per node, at nodes (lon, lat) = (0,0), (1,0), (10,-5).
    node_lon = np.repeat([0.0, 1.0, 10.0], 4)
    node_lat = np.repeat([0.0, 0.0, -5.0], 4)
    g["lon"][:] = node_lon + rng.uniform(-0.2, 0.2, n_src)
    g["lat"][:] = node_lat + rng.uniform(-0.2, 0.2, n_src)
    g["cell_indices"][:] = np.arange(n_src)
    g["mask_alphaearth"][:] = True
    # Land cover: node 0 -> class 10 dominant, node 1 -> class 40, node 2 -> nodata.
    lc = np.full((n_src, px, px), 255, dtype=np.uint8)
    lc[0:4] = 10
    lc[0, 0, 0] = 40  # a minority pixel that must not win
    lc[4:8] = 40
    g["lc_esa_wc"][:] = lc
    g["mask_lc_esa_wc"][:] = True
    dem = np.zeros((n_src, px, px), dtype=np.float32)
    dem[0:4] = 100.0
    dem[4:8] = 200.0
    dem[8:12] = np.nan
    g["dem_cop30"][:] = dem
    g["mask_dem_cop30"][:] = True
    return str(path)


def test_coarsen_produces_one_row_per_occupied_node(source_store, tmp_path):
    dst = tmp_path / "coarse.zarr"
    grid = RegularGrid.global_grid(1.0)
    summary = coarsen_patch_store(
        source_store, dst, grid=grid, products=["lc_esa_wc", "dem_cop30"], embeddings=("alphaearth",)
    )
    assert summary["n_cells"] == 3
    z = zarr.open_group(str(dst), mode="r")
    assert z.attrs["patch_size"] == 1
    assert z.attrs["n_cells"] == 3
    assert z["lc_esa_wc"].shape == (3, 1, 1)
    assert z["dem_cop30"].shape == (3, 1, 1)
    assert z["alphaearth"].shape == (3, 1, 1, 64)
    assert sorted(np.asarray(z["n_source_patches"][:]).tolist()) == [4, 4, 4]


def test_coarsen_carries_the_majority_class_and_the_mean(source_store, tmp_path):
    dst = tmp_path / "coarse.zarr"
    grid = RegularGrid.global_grid(1.0)
    coarsen_patch_store(source_store, dst, grid=grid, products=["lc_esa_wc", "dem_cop30"])
    z = zarr.open_group(str(dst), mode="r")
    lon = np.asarray(z["lon"][:])
    lc = np.asarray(z["lc_esa_wc"][:, 0, 0])
    dem = np.asarray(z["dem_cop30"][:, 0, 0])
    order = np.argsort(lon)  # nodes at lon 0, 1, 10
    assert lc[order].tolist() == [10, 40, 255]
    assert dem[order][0] == pytest.approx(100.0)
    assert dem[order][1] == pytest.approx(200.0)
    assert np.isnan(dem[order][2])
    # The all-nodata node has no land cover, so its mask is False.
    assert np.asarray(z["mask_lc_esa_wc"][:])[order].tolist() == [True, True, False]


def test_coarsen_places_nodes_at_their_grid_coordinates(source_store, tmp_path):
    """Output lon/lat are the *node* coordinates, not the source-cell means."""
    dst = tmp_path / "coarse.zarr"
    coarsen_patch_store(source_store, dst, grid=RegularGrid.global_grid(1.0), products=["lc_esa_wc"])
    z = zarr.open_group(str(dst), mode="r")
    got = sorted(zip(np.asarray(z["lon"][:]).tolist(), np.asarray(z["lat"][:]).tolist(), strict=True))
    assert got == [(0.0, 0.0), (1.0, 0.0), (10.0, -5.0)]


def test_coarsen_drops_nodes_below_the_sample_floor(source_store, tmp_path):
    """Only nodes with enough source patches survive; none surviving is an error."""
    summary = coarsen_patch_store(
        source_store,
        tmp_path / "keep.zarr",
        grid=RegularGrid.global_grid(1.0),
        products=["lc_esa_wc"],
        min_source_patches=4,
    )
    assert summary["n_cells"] == 3
    with pytest.raises(ValueError, match="min_source_patches"):
        coarsen_patch_store(
            source_store,
            tmp_path / "none.zarr",
            grid=RegularGrid.global_grid(1.0),
            products=["lc_esa_wc"],
            min_source_patches=5,
        )


def test_coarsen_writes_a_split_readable_by_patchdata(source_store, tmp_path):
    dst = tmp_path / "coarse.zarr"
    spec = SplitSpec(name="tiny", test=(0, 1), train=(1, 3), val=None, chunk_size=1)
    coarsen_patch_store(
        source_store,
        dst,
        grid=RegularGrid.global_grid(1.0),
        products=["lc_esa_wc"],
        embeddings=("alphaearth",),
        splits={"tiny": spec},
    )
    z = zarr.open_group(str(dst), mode="r+")
    z["alphaearth"][:] = 1.0
    z["mask_alphaearth"][:] = True
    # patch_size=1 (one vector per grid cell) has zero spatial std everywhere, and
    # this fixture's uniform fill value is a flat pixel by construction -- both
    # degenerate-cell filters (on by default) must stay off for this synthetic store.
    data = PatchData(
        str(dst),
        "alphaearth",
        "split_tiny",
        "train",
        compute=True,
        n_load_workers=1,
        min_spatial_std=None,
        drop_flat_pixel_cells=False,
    )
    assert len(data) == 2
    batch = next(data.iter_batches(2))
    assert batch["x"].shape == (2, 1, 1, 64)


def test_coarsen_is_deterministic_in_row_order(source_store, tmp_path):
    """The same seed must place the same node at the same store row."""
    lons = []
    for i in range(2):
        dst = tmp_path / f"coarse{i}.zarr"
        coarsen_patch_store(source_store, dst, grid=RegularGrid.global_grid(1.0), products=["lc_esa_wc"], seed=7)
        lons.append(np.asarray(zarr.open_group(str(dst), mode="r")["lon"][:]))
    assert np.array_equal(lons[0], lons[1])


def test_coarsen_votes_per_band_for_multi_band_categoricals(tmp_path):
    """Overture's five layers are independent class systems, not one vote."""
    n_src, px, bands = 6, 4, 5
    src = tmp_path / "multi.zarr"
    schemas = build_schemas(["overture"], n_src, embeddings=(), patch_size=px)
    g = create_store(src, schemas, n_cells=n_src, patch_size=px)
    g["lon"][:] = np.zeros(n_src)
    g["lat"][:] = np.zeros(n_src)
    codes = np.zeros((n_src, px, px, bands), dtype=np.uint8)
    for b in range(bands):
        codes[..., b] = b + 1
    g["overture"][:] = codes
    g["mask_overture"][:] = True

    coarsen_patch_store(src, tmp_path / "coarse.zarr", grid=RegularGrid.global_grid(1.0), products=["overture"])
    z = zarr.open_group(str(tmp_path / "coarse.zarr"), mode="r")
    assert z["overture"].shape == (1, 1, 1, bands)
    assert np.asarray(z["overture"][0, 0, 0]).tolist() == [1, 2, 3, 4, 5]

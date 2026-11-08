"""Contract tests for the Copernicus-FM grid-embedding lookup (no network)."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from geosae.data.dataset import PatchData
from geosae.data.download.fetchers.copernicus_fm import (
    COPERNICUSFM_DIM,
    CopernicusFMGridEmbeddings,
    add_copernicusfm_embeddings,
)
from geosae.data.download.gridded import RegularGrid
from geosae.data.download.sources import get_product
from geosae.data.download.store import build_schemas, create_store

RES = 4.0  # a 46 x 90 stand-in for the 0.25 deg map, so the raster stays tiny


@pytest.fixture(scope="module")
def raster_file(tmp_path_factory):
    """A small dense embedding map with a known-valid and a known-empty node."""
    grid = RegularGrid.global_grid(RES)
    rng = np.random.default_rng(0)
    raster = np.zeros((*grid.shape, COPERNICUSFM_DIM), dtype=np.float32)
    # Fill the equatorial band only; everything else stays zero (= no embedding).
    equator = grid.n_lat // 2
    raster[equator] = rng.normal(size=(grid.n_lon, COPERNICUSFM_DIM)).astype(np.float32)
    path = tmp_path_factory.mktemp("cfm") / "embed_map.npz"
    np.savez(path, arr_0=raster)
    return path


@pytest.fixture
def fetcher(raster_file):
    return CopernicusFMGridEmbeddings(raster_file, resolution_deg=RES)


def test_registry_entry_matches_the_fetcher_constants():
    """The registered product must describe what the fetcher actually returns."""
    spec = get_product("copernicusfm")
    assert spec.array == "copernicusfm"
    assert spec.shape == (1, 1, COPERNICUSFM_DIM)
    assert spec.fetcher.endswith("copernicus_fm:CopernicusFMGridEmbeddings")


def test_lookup_returns_the_node_vector_and_flags_empty_nodes(fetcher):
    grid = RegularGrid.global_grid(RES)
    lat_equator = float(grid.lat_nodes()[grid.n_lat // 2])
    lon = np.array([0.0, 0.0])
    lat = np.array([lat_equator, lat_equator + 4 * RES])
    vectors, valid = fetcher.lookup(lon, lat)
    assert vectors.shape == (2, COPERNICUSFM_DIM)
    assert valid.tolist() == [True, False]
    assert np.array_equal(vectors[0], fetcher.raster[grid.n_lat // 2, grid.n_lon // 2])


def test_lookup_snaps_off_node_points_to_the_nearest_node(fetcher):
    grid = RegularGrid.global_grid(RES)
    lat_equator = float(grid.lat_nodes()[grid.n_lat // 2])
    on_node, _ = fetcher.lookup(np.array([0.0]), np.array([lat_equator]))
    nudged, _ = fetcher.lookup(np.array([0.4]), np.array([lat_equator - 0.4]))
    assert np.array_equal(on_node, nudged)


def test_lookup_rejects_a_raster_of_the_wrong_shape(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, arr_0=np.zeros((10, 10, COPERNICUSFM_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="embedding map"):
        CopernicusFMGridEmbeddings(path, resolution_deg=RES).lookup([0.0], [0.0])


def _coarse_store(path, lon, lat):
    n = len(lon)
    schemas = build_schemas([], n, embeddings=("copernicusfm",), splits=("split_t",), patch_size=1)
    g = create_store(path, schemas, n_cells=n, patch_size=1)
    g["lon"][:] = lon
    g["lat"][:] = lat
    g["split_t"][:] = 0
    return g


def test_add_embeddings_fills_valid_cells_and_masks_the_rest(raster_file, tmp_path):
    grid = RegularGrid.global_grid(RES)
    lat_equator = float(grid.lat_nodes()[grid.n_lat // 2])
    lon = [0.0, 40.0, 0.0]
    lat = [lat_equator, lat_equator, lat_equator + 4 * RES]
    _coarse_store(tmp_path / "s.zarr", lon, lat)

    n_filled = add_copernicusfm_embeddings(tmp_path / "s.zarr", cache_dir=raster_file, resolution_deg=RES)
    assert n_filled == 2

    z = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
    assert z["copernicusfm"].shape == (3, 1, 1, COPERNICUSFM_DIM)
    assert np.asarray(z["mask_copernicusfm"][:]).tolist() == [True, True, False]
    assert np.isfinite(np.asarray(z["copernicusfm"][0])).all()
    assert np.isnan(np.asarray(z["copernicusfm"][2])).all()
    assert np.asarray(z["feature_copernicusfm"][:]).tolist() == list(range(COPERNICUSFM_DIM))


def test_filled_store_is_readable_as_training_data(raster_file, tmp_path):
    """The end product must load through the normal training data path."""
    grid = RegularGrid.global_grid(RES)
    lat_equator = float(grid.lat_nodes()[grid.n_lat // 2])
    _coarse_store(tmp_path / "s.zarr", [0.0, 40.0, 80.0], [lat_equator] * 3)
    add_copernicusfm_embeddings(tmp_path / "s.zarr", cache_dir=raster_file, resolution_deg=RES)

    # patch_size=1 (one vector per grid cell) has zero spatial std everywhere,
    # so the degenerate-cell filter (on by default) must stay off here.
    data = PatchData(
        str(tmp_path / "s.zarr"),
        "copernicusfm",
        "split_t",
        "train",
        compute=True,
        n_load_workers=1,
        min_spatial_std=None,
    )
    assert len(data) == 3
    batch = next(data.iter_batches(3))
    assert batch["x"].shape == (3, 1, 1, COPERNICUSFM_DIM)
    assert np.isfinite(batch["x"].numpy()).all()


def test_add_embeddings_refuses_a_patch_store(tmp_path):
    schemas = build_schemas([], 2, embeddings=("alphaearth",), patch_size=64)
    create_store(tmp_path / "p.zarr", schemas, n_cells=2, patch_size=64)
    with pytest.raises(ValueError, match="patch_size=1"):
        add_copernicusfm_embeddings(tmp_path / "p.zarr", cache_dir=tmp_path)

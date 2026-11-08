"""The release tables: cell ids, splits, the product registry and its licences.

What these pin is what a reader rebuilds the dataset from, so the contract is
the *content* of the tables (a cell id that names the same MajorTOM cell, a
split that matches the store, a redistribution verdict for every product), not
their formatting.
"""

import csv
import json

import numpy as np
import pytest
import zarr

from geosae.data.download.release import (
    cell_records,
    concept_records,
    product_records,
    write_release_tables,
)
from geosae.data.download.sources import Redistribution, all_products


@pytest.fixture(scope="module")
def majortom_grid():
    from geosae.data.download.grid import MajorTomGrid

    return MajorTomGrid(5)


def _store(tmp_path):
    """4 cells: two train, one val, one test, all with AlphaEarth, one without Tessera."""
    g = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    g.attrs.update({"n_cells": 4, "patch_size": 64})
    lon = np.array([11.0, -73.0, 139.0, 25.0])
    lat = np.array([48.0, 40.0, 35.0, -30.0])
    g.create_array("lon", shape=(4,), chunks=(4,), dtype="float64")[:] = lon
    g.create_array("lat", shape=(4,), chunks=(4,), dtype="float64")[:] = lat
    crs = g.create_array("crs", shape=(4,), chunks=(4,), dtype=str)
    crs[:] = np.array(["EPSG:32632", "EPSG:32618", "EPSG:32654", "EPSG:32735"])
    g.create_array("split_210k", shape=(4,), chunks=(4,), dtype="int8")[:] = np.array([0, 0, 1, 2], dtype="int8")
    g.create_array("mask_alphaearth", shape=(4,), chunks=(4,), dtype="bool")[:] = True
    g.create_array("mask_tessera", shape=(4,), chunks=(4,), dtype="bool")[:] = np.array([True, True, True, False])
    return str(tmp_path / "store.zarr")


def test_every_product_carries_a_licence_and_a_redistribution_verdict():
    for key, spec in all_products().items():
        assert spec.licence, key
        assert spec.licence_url.startswith("http"), key
        assert spec.redistribution in set(Redistribution), key


def test_product_table_marks_the_layers_we_may_not_pass_on():
    by_key = {r["key"]: r for r in product_records()}
    # OpenStreetMap-derived themes are ODbL: redistributing them relicenses the release.
    assert by_key["overture"]["redistribution"] == "share_alike"
    # The CCI licence grants research use, not redistribution.
    assert by_key["lc_esa_cci"]["redistribution"] == "restricted"
    assert by_key["lc_esa_wc"]["redistribution"] == "attribution"
    assert by_key["tessera"]["redistribution"] == "open"


def test_product_table_records_the_preprocessing_a_rebuild_needs():
    rec = {r["key"]: r for r in product_records()}["lc_esa_wc"]
    spec = all_products()["lc_esa_wc"]
    assert rec["resampling"] == spec.resampling == "nearest"  # categorical must not be interpolated
    assert rec["native_resolution_m"] == spec.resolution_m
    assert rec["fetcher"] == spec.fetcher
    assert json.loads(rec["fetcher_kwargs"]) is not None


def test_product_subset_is_the_subset_asked_for():
    assert [r["key"] for r in product_records(["lc_esa_wc"])] == ["lc_esa_wc"]


def test_concept_table_names_a_product_class_per_row():
    recs = concept_records(["lc_esa_wc"])
    assert len(recs) > 5
    assert {r["product"] for r in recs} == {"lc_esa_wc"}
    assert all(isinstance(r["code"], int) and r["name"] for r in recs)


def test_cell_records_carry_the_store_split_and_embedding_coverage(tmp_path, majortom_grid):
    recs = cell_records(_store(tmp_path))
    assert [r["split"] for r in recs] == ["train", "train", "val", "test"]
    assert [r["has_tessera"] for r in recs] == [True, True, True, False]
    assert all(r["has_alphaearth"] for r in recs)
    assert [r["crs"] for r in recs][0] == "EPSG:32632"


def test_cell_id_names_the_cell_the_coordinates_fall_in(tmp_path, majortom_grid):
    """The exported id must round-trip through the MajorTOM grid it came from."""
    grid = majortom_grid
    for rec in cell_records(_store(tmp_path)):
        row, col = rec["majortom_row"], rec["majortom_col"]
        assert grid.cell_id(row, col) == rec["majortom_cell"]
        west, south, east, north = grid.bounds[grid.flat_index(row, col)]
        assert west <= rec["lon"] < east and south <= rec["lat"] < north


def test_write_release_tables_writes_four_readable_tables(tmp_path, majortom_grid):
    out = tmp_path / "release"
    paths = write_release_tables(_store(tmp_path), out)
    assert set(paths) == {"cells", "products", "concepts", "manifest"}

    with open(paths["cells"]) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["majortom_cell"].count("_") == 1

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert manifest["selection"]["seed"] == 42
    assert manifest["redistribution"]["overture"] == "share_alike"


def _store_with_products(tmp_path):
    """A store carrying one packable product, one share-alike one, one unregistered."""
    g = zarr.open_group(str(tmp_path / "full.zarr"), mode="w")
    g.attrs.update({"n_cells": 2, "patch_size": 2})
    g.create_array("lon", shape=(2,), chunks=(2,), dtype="float64")[:] = [11.0, 12.0]
    g.create_array("lat", shape=(2,), chunks=(2,), dtype="float64")[:] = [48.0, 49.0]
    g.create_array("split_210k", shape=(2,), chunks=(2,), dtype="int8")[:] = np.array([0, 2], dtype="int8")
    for name in ("lc_esa_wc", "overture", "mystery_layer", "chmv2_check"):
        g.create_array(name, shape=(2, 2, 2), chunks=(2, 2, 2), dtype="uint8")[:] = 1
        g.create_array("mask_" + name, shape=(2,), chunks=(2,), dtype="bool")[:] = True
    return str(tmp_path / "full.zarr")


def test_only_licence_compatible_arrays_are_packable(tmp_path):
    from geosae.data.download.release import redistributable_arrays

    packable, excluded = redistributable_arrays(_store_with_products(tmp_path))
    assert "lc_esa_wc" in packable and "mask_lc_esa_wc" in packable
    assert "lon" in packable and "split_210k" in packable
    assert "overture" in excluded and "share_alike" in excluded["overture"]
    assert "registry" in excluded["mystery_layer"]
    # a re-fetched array inherits the licence of the product it was fetched from
    assert "chmv2_check" in excluded
    packable, _ = redistributable_arrays(_store_with_products(tmp_path), array_products={"chmv2_check": "chmv2"})
    assert "chmv2_check" in packable


def test_packing_copies_the_packable_arrays_and_nothing_else(tmp_path):
    from geosae.data.download.release import pack_release_store

    src = _store_with_products(tmp_path)
    out = pack_release_store(src, tmp_path / "public.zarr")
    packed = zarr.open_group(str(tmp_path / "public.zarr"), mode="r")
    assert "overture" not in packed and "mystery_layer" not in packed
    assert np.array_equal(packed["lc_esa_wc"][:], zarr.open_group(src, mode="r")["lc_esa_wc"][:])
    assert np.array_equal(packed["split_210k"][:], np.array([0, 2], dtype="int8"))
    assert packed.attrs["patch_size"] == 2
    assert out["excluded"]["overture"]


def test_packing_twice_is_idempotent(tmp_path):
    from geosae.data.download.release import pack_release_store

    src = _store_with_products(tmp_path)
    pack_release_store(src, tmp_path / "public.zarr")
    second = pack_release_store(src, tmp_path / "public.zarr")
    packed = zarr.open_group(str(tmp_path / "public.zarr"), mode="r")
    assert sorted(packed.array_keys()) == sorted(second["arrays"])


def test_packing_honours_an_explicit_exclusion(tmp_path):
    """A licence-clean layer can still be left out, e.g. for its size."""
    from geosae.data.download.release import pack_release_store

    out = pack_release_store(_store_with_products(tmp_path), tmp_path / "small.zarr", exclude=("lc_esa_wc",))
    packed = zarr.open_group(str(tmp_path / "small.zarr"), mode="r")
    assert "lc_esa_wc" not in packed and "mask_lc_esa_wc" not in packed
    assert "lon" in packed
    assert out["excluded"]["lc_esa_wc"] == "excluded by the caller"

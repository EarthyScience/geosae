"""Contract tests for the cell selection of a Sentinel-2 subset (core install, synthetic store)."""

from __future__ import annotations

import numpy as np
import zarr

from geosae.data.embed.subset import select_subset_rows


def _parent(path, labels):
    group = zarr.open_group(str(path), mode="w")
    n = len(labels)
    group.create_array("split_toy", shape=(n,), dtype="int8")[:] = np.asarray(labels, dtype=np.int8)
    group.create_array("lon", shape=(n,), dtype="float64")[:] = np.arange(n) * 1.0
    group.create_array("lat", shape=(n,), dtype="float64")[:] = -np.arange(n) * 1.0
    return str(path)


def test_the_subset_keeps_one_split_and_its_labels(tmp_path):
    labels = [2, 2, -1, 0, 0, 0, 1, 1, -1, 1]
    rows, lon, lat, got = select_subset_rows(_parent(tmp_path / "p.zarr", labels), "split_toy")
    assert rows.tolist() == [0, 1, 3, 4, 5, 6, 7, 9]
    assert got.tolist() == [labels[r] for r in rows]
    assert lon.tolist() == rows.tolist() and lat.tolist() == (-rows).tolist()


def test_a_cap_draws_the_same_cells_for_the_same_seed(tmp_path):
    labels = [0] * 20 + [1] * 30
    path = _parent(tmp_path / "p.zarr", labels)
    first, *_ = select_subset_rows(path, "split_toy", {"val": 5}, seed=42)
    again, *_ = select_subset_rows(path, "split_toy", {"val": 5}, seed=42)
    other, *_ = select_subset_rows(path, "split_toy", {"val": 5}, seed=7)
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    assert (first < 20).sum() == 20 and (first >= 20).sum() == 5


def test_excluded_rows_are_never_selected(tmp_path):
    rows, *_ = select_subset_rows(_parent(tmp_path / "p.zarr", [0] * 6), "split_toy", exclude=np.array([1, 4]))
    assert rows.tolist() == [0, 2, 3, 5]

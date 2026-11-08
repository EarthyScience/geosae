"""Contract tests for building a store from sources: resume, row coverage, splits, statistics.

The fetchers are replaced by a synthetic one, so nothing touches the network; the
geometry needs the ``download`` extra (rasterio, odc-geo).
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

pytest.importorskip("rasterio")
pytest.importorskip("odc.geo")

from geosae.data.download.selection import EXCLUDED, TEST, TRAIN, SplitSpec  # noqa: E402
from geosae.data.download.sources import Auth  # noqa: E402
from geosae.data.download.store import build_store, geobox_params, has_embedding, utm_geobox  # noqa: E402

N = 12
PATCH = 4
SPLIT = SplitSpec(name="toy", test=(0, 6), train=(6, 12), val=None, chunk_size=2)


class _FakeFetcher:
    """Answers every cell with a value derived from its row, except the rows it has no data for."""

    calls: list[tuple[str, int]] = []

    def __init__(self, key: str, empty: frozenset[int]):
        self.key, self.empty = key, empty

    def tile_key(self, lon, lat):
        return int(lon)

    def fetch_patches_for_tile(self, tile_key, cells, resampling, nodata):
        out = []
        for row, _lon, _lat, geobox in cells:
            _FakeFetcher.calls.append((self.key, row))
            if self.key == "alphaearth":
                value = np.nan if row in self.empty else float(row)
                out.append((row, np.full((*geobox.shape, 64), value, dtype=np.float32)))
            elif row not in self.empty:
                out.append((row, np.full(geobox.shape, row, dtype=np.uint8)))
        return out


@pytest.fixture
def fake_sources(monkeypatch):
    _FakeFetcher.calls = []
    empty = {"alphaearth": frozenset({3}), "lc_esa_wc": frozenset({1})}
    monkeypatch.setattr(
        "geosae.data.download.sources.build_fetcher", lambda key, **kw: _FakeFetcher(key, empty.get(key, frozenset()))
    )
    monkeypatch.setattr("geosae.data.download.plan.check_credentials", lambda: {a.value: "ok" for a in Auth})
    return _FakeFetcher


def _build(path, **kwargs):
    lon = np.linspace(10.05, 12.95, N)
    lat = np.linspace(45.0, 46.0, N)
    defaults = dict(
        products=["alphaearth", "lc_esa_wc"],
        cell_indices=np.arange(100, 100 + N),
        lon=lon,
        lat=lat,
        splits={"toy": SPLIT},
        patch_size=PATCH,
        block_rows=4,
        max_workers=1,
        log=lambda line: None,
    )
    return build_store(str(path), **{**defaults, **kwargs})


def test_a_resumed_build_fetches_only_what_is_missing(tmp_path, fake_sources):
    path = tmp_path / "store.zarr"
    _build(path, rows=np.arange(3))
    assert sorted(fake_sources.calls) == sorted((k, r) for k in ("alphaearth", "lc_esa_wc") for r in range(3))

    fake_sources.calls.clear()
    _build(path)
    fetched = {k: sorted(r for kk, r in fake_sources.calls if kk == k) for k in ("alphaearth", "lc_esa_wc")}
    # rows 0-2 are already there, except row 1, which the label source had no data for
    assert fetched["alphaearth"] == list(range(3, N))
    assert fetched["lc_esa_wc"] == [1, *range(3, N)]

    fake_sources.calls.clear()
    _build(path)
    assert fake_sources.calls == [], "every block was complete: nothing is asked for again"

    group = zarr.open_group(str(path), mode="r")
    assert np.asarray(group["cell_indices"][:]).tolist() == list(range(100, 100 + N))
    assert float(group["alphaearth"][5, 0, 0, 0]) == 5.0
    assert int(group["lc_esa_wc"][7, 0, 0]) == 7


def test_reference_products_cover_only_the_reference_rows(tmp_path, fake_sources):
    path = tmp_path / "store.zarr"
    summary = _build(path, reference_rows=(0, 6))
    assert summary["products"]["lc_esa_wc"]["requested"] == 6
    assert summary["products"]["alphaearth"]["requested"] == N
    mask = np.asarray(zarr.open_group(str(path), mode="r")["mask_lc_esa_wc"][:])
    assert mask.tolist() == [True, False, True, True, True, True] + [False] * 6


def test_a_cell_without_an_embedding_is_unmasked_and_left_out_of_every_split(tmp_path, fake_sources):
    path = tmp_path / "store.zarr"
    summary = _build(path)
    group = zarr.open_group(str(path), mode="r")
    assert not bool(group["mask_alphaearth"][3])
    split = np.asarray(group["split_toy"][:])
    assert split[3] == EXCLUDED
    assert (split[[0, 1, 2, 4, 5]] == TEST).all() and (split[6:] == TRAIN).all()
    assert summary["splits"]["split_toy"]["excluded"] == 1


def test_a_store_of_other_cells_is_not_resumed(tmp_path, fake_sources):
    path = tmp_path / "store.zarr"
    _build(path, rows=np.arange(2))
    with pytest.raises(ValueError, match="different cells"):
        _build(path, cell_indices=np.arange(N))


def test_a_later_run_can_add_a_product(tmp_path, fake_sources):
    path = tmp_path / "store.zarr"
    _build(path, products=["alphaearth"])
    _build(path, products=["alphaearth", "lc_esa_wc"])
    group = zarr.open_group(str(path), mode="r")
    assert group["lc_esa_wc"].attrs["product"] == "lc_esa_wc"
    assert int(np.asarray(group["mask_lc_esa_wc"][:]).sum()) == N - 1


def test_normalisation_statistics_are_readable_by_the_transforms(tmp_path, fake_sources):
    from geosae.data.dataset import Transforms

    stats = {"mean": np.zeros(64), "std": np.full(64, 2.0), "clip_lo": np.full(64, -1.0), "clip_hi": np.ones(64)}
    path = tmp_path / "store.zarr"
    _build(path, norm_stats={"alphaearth_norm": stats})
    t = Transforms.from_zarr_stats(str(path), "alphaearth_norm")
    assert np.allclose(t.std, 2.0) and np.allclose(t.clip_hi, 1.0)


def test_vectorised_geometry_matches_the_per_cell_geobox():
    lon = np.array([-179.9, -52.37, 0.01, 6.0, 115.7, 179.9])
    lat = np.array([-55.0, 70.35, 0.0, -0.01, 44.93, 60.0])
    crs, transform = geobox_params(lon, lat)
    for i in range(lon.size):
        _, want_crs, want = utm_geobox(lon[i], lat[i])
        assert crs[i] == want_crs
        assert np.allclose(transform[i], want, atol=1e-6)


def test_an_all_fill_patch_is_no_embedding():
    assert not has_embedding(np.full((2, 2, 3), np.nan))
    assert has_embedding(np.zeros((2, 2, 3)))
    assert has_embedding(np.array([np.nan, 0.5]))


def test_a_tessera_window_on_the_cells_grid_is_written_unresampled():
    """Resampling a window that is within a pixel of the geobox would shift it and leave an edge empty."""
    from rasterio.transform import Affine

    from geosae.data.download.fetchers.tessera_embeddings import TesseraTiledFetcher

    geobox, crs, params = utm_geobox(10.0, 45.0, patch_size=8)
    native = np.arange(8 * 8 * 128, dtype=np.float32).reshape(8, 8, 128)
    offset = Affine(*params) * Affine.translation(0.4, 0.3)  # the tile's own grid, a fraction of a pixel off

    class _Inner:
        def fetch_patches_for_tile(self, tile_lon, tile_lat, cells, patch_size):
            return [(cid, native, crs, offset) for cid, _, _ in cells]

    fetcher = TesseraTiledFetcher.__new__(TesseraTiledFetcher)
    fetcher._inner = _Inner()
    [(row, patch)] = fetcher.fetch_patches_for_tile(fetcher.tile_key(10.0, 45.0), [(0, 10.0, 45.0, geobox)])
    assert row == 0 and np.array_equal(patch, native)


def test_overture_queries_from_concurrent_tile_fetches_use_their_own_cursor(monkeypatch):
    """One DuckDB connection shared by the fetch pool's threads crashes the process."""
    import threading

    from geosae.data.download.fetchers import overture

    class _Database:
        created = 0

        def __init__(self):
            _Database.created += 1

        def cursor(self):
            return object()

    monkeypatch.setattr(overture, "_create_connection", _Database)
    fetcher = overture.OvertureFetcher(release="2026-01-01.0")
    seen = {}

    def grab(name):
        seen[name] = (fetcher._get_connection(), fetcher._get_connection())

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(a is b for a, b in seen.values()), "a thread keeps its connection"
    assert len({id(a) for a, _ in seen.values()}) == 3, "threads do not share one"
    assert _Database.created == 1, "one database serves every thread"


def test_a_store_of_chosen_cells_holds_them_at_their_centres_with_fractional_splits(tmp_path, fake_sources):
    from geosae.data.download.grid import cells_at, cells_in_bbox
    from geosae.data.download.store import build_majortom_store

    cells = cells_in_bbox((11.3, 50.7, 11.8, 51.0), land_fraction=None)[:40]
    path = tmp_path / "cells.zarr"
    kwargs = dict(products=["alphaearth", "lc_esa_wc"], patch_size=PATCH, max_workers=1, log=lambda line: None)
    summary = build_majortom_store(str(path), cells, val_frac=0.1, test_frac=0.2, **kwargs)
    group = zarr.open_group(str(path), mode="r")
    rows = np.asarray(group["cell_indices"][:])
    assert sorted(rows.tolist()) == sorted(cells.tolist()) and not np.array_equal(rows, cells)  # shuffled
    assert np.array_equal(cells_at(group["lon"][:], group["lat"][:]), rows)  # every row sits in its own cell
    counts = summary["splits"]["split_default"]
    # rows 0-7 test, 8-35 train, 36-39 val; row 3 has no embedding and is excluded
    assert counts == {"train": 28, "val": 4, "test": 7, "excluded": 1}
    # the same call resumes into the store (same cells, same order) and has nothing left to fetch
    fake_sources.calls.clear()
    build_majortom_store(str(path), cells, **kwargs)
    assert fake_sources.calls == []


def test_a_store_of_chosen_cells_refuses_a_duplicate(tmp_path, fake_sources):
    from geosae.data.download.store import build_majortom_store

    with pytest.raises(ValueError, match="duplicates"):
        build_majortom_store(str(tmp_path / "s.zarr"), [5, 6, 5], products=["lc_esa_wc"])


def test_a_build_without_the_credentials_it_needs_says_how_to_set_them_up(tmp_path, monkeypatch):
    monkeypatch.setattr("geosae.data.download.plan.check_credentials", lambda: {"public": "ok", "gee": "missing x"})
    with pytest.raises(RuntimeError, match="earthengine authenticate"):
        _build(tmp_path / "store.zarr")
    assert not (tmp_path / "store.zarr").exists()

"""Degenerate (constant cell / flat pixel) embedding-patch detection + PatchData filtering."""

from pathlib import Path

import numpy as np
import zarr

from geosae.data.dataset import PatchData
from geosae.data.quality import (
    cached_constant_mask,
    cached_extreme_value_mask,
    cached_flat_pixel_mask,
    has_extreme_value,
    has_flat_pixel,
    patch_spatial_std,
    quality_cache_path,
    scan_constant_cells,
    scan_extreme_value_cells,
    scan_flat_pixel_cells,
)


def _store(path, n=40, ps=8, d=16, const_cells=(3, 17), flat_pixel_cells=(), extreme_value_cells=()):
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update({"n_cells": n, "patch_size": ps})
    rng = np.random.default_rng(0)
    ae = rng.standard_normal((n, ps, ps, d)).astype("float32")
    for c in const_cells:
        ae[c] = np.full((ps, ps, d), 3.0, dtype="float32")  # constant fill patch
    for c in flat_pixel_cells:
        ae[c, 0, 0] = 5.0  # one flat pixel, rest of the patch untouched
    for c in extreme_value_cells:
        ae[c, 0, 0, :] = 0.0
        ae[c, 0, 0, 1] = 2e27  # a Tessera-like magnitude sentinel, one dim
    g.create_array("alphaearth", shape=ae.shape, chunks=(8, ps, ps, d), dtype="float32")[:] = ae
    g.create_array("mask_alphaearth", shape=(n,), chunks=(n,), dtype="bool")[:] = np.ones(n, bool)
    g.create_array("split_210k", shape=(n,), chunks=(n,), dtype="int8")[:] = np.zeros(n, "int8")
    g.create_array("lon", shape=(n,), chunks=(n,), dtype="float64")[:] = np.zeros(n)
    g.create_array("lat", shape=(n,), chunks=(n,), dtype="float64")[:] = np.zeros(n)


def test_patch_spatial_std_flags_constant():
    emb = np.stack([np.full((8, 8, 4), 2.0), np.random.default_rng(0).standard_normal((8, 8, 4))])
    s = patch_spatial_std(emb)
    assert s[0] < 1e-6 and s[1] > 0.1


def test_has_flat_pixel_flags_a_single_pixel():
    patch = np.random.default_rng(0).standard_normal((8, 8, 4)).astype("float32")
    clean = np.stack([patch, patch])
    flat = clean.copy()
    flat[1, 3, 5] = 7.0  # one flat pixel is enough
    s = has_flat_pixel(np.stack([clean[0], flat[1]]))
    assert not s[0] and s[1]


def test_has_extreme_value_flags_a_single_sentinel_dim():
    patch = np.random.default_rng(0).standard_normal((8, 8, 4)).astype("float32")
    clean = np.stack([patch, patch])
    sentinel = clean.copy()
    sentinel[1, 2, 6, 1] = 2.3e27  # a Tessera-like magnitude in one dim of one pixel
    s = has_extreme_value(np.stack([clean[0], sentinel[1]]), max_abs_value=1e6)
    assert not s[0] and s[1]


def test_has_extreme_value_does_not_overflow_on_squaring():
    """The check itself never squares, so it must not warn/overflow on a 1e27-scale value."""
    emb = np.zeros((1, 4, 4, 4), dtype="float32")
    emb[0, 0, 0, 0] = 2.9e27
    with np.errstate(over="raise"):
        s = has_extreme_value(emb, max_abs_value=1e6)
    assert s[0]


def test_scan_constant_cells(tmp_path):
    _store(tmp_path / "d.zarr", const_cells=(3, 17))
    cells = np.arange(40)
    bad = scan_constant_cells(str(tmp_path / "d.zarr"), "alphaearth", cells, thresh=1e-3)
    assert set(np.where(bad)[0]) == {3, 17}


def test_scan_flat_pixel_cells(tmp_path):
    _store(tmp_path / "d.zarr", const_cells=(), flat_pixel_cells=(5, 21))
    cells = np.arange(40)
    bad = scan_flat_pixel_cells(str(tmp_path / "d.zarr"), "alphaearth", cells)
    assert set(np.where(bad)[0]) == {5, 21}


def test_scan_extreme_value_cells(tmp_path):
    _store(tmp_path / "d.zarr", const_cells=(), extreme_value_cells=(6, 30))
    cells = np.arange(40)
    bad = scan_extreme_value_cells(str(tmp_path / "d.zarr"), "alphaearth", cells, max_abs_value=1e6)
    assert set(np.where(bad)[0]) == {6, 30}


def test_patchdata_drops_extreme_value_cells(tmp_path):
    _store(tmp_path / "d.zarr", const_cells=(), extreme_value_cells=(6, 30))
    keep = PatchData(
        str(tmp_path / "d.zarr"),
        split="train",
        compute=False,
        min_spatial_std=None,
        drop_flat_pixel_cells=False,
        max_abs_value=None,
    )
    drop = PatchData(
        str(tmp_path / "d.zarr"),
        split="train",
        compute=False,
        min_spatial_std=None,
        drop_flat_pixel_cells=False,
    )
    assert len(keep) == 40
    assert len(drop) == 38
    assert 6 not in drop.cells and 30 not in drop.cells


def test_cached_extreme_value_mask_matches_the_uncached_scan(tmp_path):
    path = tmp_path / "d.zarr"
    _store(path, const_cells=(), extreme_value_cells=(6, 30))
    cache_root = tmp_path / "cache"
    direct = scan_extreme_value_cells(str(path), "alphaearth", np.arange(40), max_abs_value=1e6)
    cached = cached_extreme_value_mask(str(path), "alphaearth", max_abs_value=1e6, cache_root=cache_root)
    assert np.array_equal(cached, direct)
    path_on_disk = quality_cache_path(
        str(path), "alphaearth", "extreme_value", split_name="split_210k", thresh=1e6, cache_root=cache_root
    )
    assert path_on_disk.exists()


def test_patchdata_drops_degenerate(tmp_path):
    _store(tmp_path / "d.zarr", const_cells=(3, 17))
    keep = PatchData(
        str(tmp_path / "d.zarr"), split="train", compute=False, min_spatial_std=None, drop_flat_pixel_cells=False
    )
    drop = PatchData(str(tmp_path / "d.zarr"), split="train", compute=False, min_spatial_std=1e-3)
    assert len(keep) == 40
    assert len(drop) == 38
    assert 3 not in drop.cells and 17 not in drop.cells


def test_patchdata_drops_flat_pixel_cells(tmp_path):
    _store(tmp_path / "d.zarr", const_cells=(), flat_pixel_cells=(5, 21))
    keep = PatchData(
        str(tmp_path / "d.zarr"), split="train", compute=False, min_spatial_std=None, drop_flat_pixel_cells=False
    )
    drop = PatchData(str(tmp_path / "d.zarr"), split="train", compute=False, min_spatial_std=None)
    assert len(keep) == 40
    assert len(drop) == 38
    assert 5 not in drop.cells and 21 not in drop.cells


def test_patchdata_filters_degenerate_and_flat_pixel_cells_by_default(tmp_path):
    """min_spatial_std=1e-3 and drop_flat_pixel_cells=True are the defaults."""
    _store(tmp_path / "d.zarr", const_cells=(3,), flat_pixel_cells=(21,))
    default = PatchData(str(tmp_path / "d.zarr"), split="train", compute=False)
    assert len(default) == 38
    assert 3 not in default.cells and 21 not in default.cells


# --------------------------------------------------------------- cached whole-store scans


def test_quality_cache_path_is_content_addressed(tmp_path):
    a_path, b_path = tmp_path / "a.zarr", tmp_path / "b.zarr"
    _store(a_path)
    _store(b_path)
    kw = dict(split_name="split_210k", cache_root=tmp_path)
    a = quality_cache_path(str(a_path), "alphaearth", "constant", thresh=1e-3, **kw)
    same = quality_cache_path(str(a_path), "alphaearth", "constant", thresh=1e-3, **kw)
    other_store = quality_cache_path(str(b_path), "alphaearth", "constant", thresh=1e-3, **kw)
    other_thresh = quality_cache_path(str(a_path), "alphaearth", "constant", thresh=1e-2, **kw)
    other_kind = quality_cache_path(str(a_path), "alphaearth", "flat_pixel", **kw)
    assert a == same
    assert len({a, other_store, other_thresh, other_kind}) == 4


def test_cached_constant_mask_matches_the_uncached_scan_and_is_written_once(tmp_path):
    path = tmp_path / "d.zarr"
    _store(path, const_cells=(3, 17))
    cache_root = tmp_path / "cache"
    direct = scan_constant_cells(str(path), "alphaearth", np.arange(40), thresh=1e-3)

    cached = cached_constant_mask(str(path), "alphaearth", thresh=1e-3, cache_root=cache_root)
    assert np.array_equal(cached, direct)
    cache_file = quality_cache_path(
        str(path), "alphaearth", "constant", split_name="split_210k", thresh=1e-3, cache_root=cache_root
    )
    assert cache_file.exists()

    # A store change that leaves store_fingerprint's inputs untouched (here:
    # an embedding value, not the split labels / mask / cell_indices the
    # fingerprint hashes) is invisible to a second call: the scan was
    # memoised, not repeated -- exactly the cost this exists to remove.
    g = zarr.open_group(str(path), mode="a")
    ae = np.asarray(g["alphaearth"][:])
    ae[5] = np.full_like(ae[5], 9.0)  # cell 5 now also constant
    g["alphaearth"][:] = ae
    still_cached = cached_constant_mask(str(path), "alphaearth", thresh=1e-3, cache_root=cache_root)
    assert np.array_equal(still_cached, direct)
    assert not still_cached[5]


def test_cached_flat_pixel_mask_matches_the_uncached_scan(tmp_path):
    path = tmp_path / "d.zarr"
    _store(path, const_cells=(), flat_pixel_cells=(5, 21))
    cache_root = tmp_path / "cache"
    direct = scan_flat_pixel_cells(str(path), "alphaearth", np.arange(40))
    cached = cached_flat_pixel_mask(str(path), "alphaearth", cache_root=cache_root)
    assert np.array_equal(cached, direct)
    path_on_disk = quality_cache_path(
        str(path), "alphaearth", "flat_pixel", split_name="split_210k", cache_root=cache_root
    )
    assert path_on_disk.exists()


def test_quality_cache_invalidates_when_the_store_fingerprint_changes(tmp_path):
    """A store change store_fingerprint *does* track (here: the split labels) lands on a new
    cache file rather than silently returning the old file's now-stale masks."""
    path = tmp_path / "d.zarr"
    _store(path, const_cells=(3,))
    cache_root = tmp_path / "cache"
    kw = dict(thresh=1e-3, cache_root=cache_root)

    before = cached_constant_mask(str(path), "alphaearth", **kw)
    before_path = quality_cache_path(str(path), "alphaearth", "constant", split_name="split_210k", **kw)
    assert before_path.exists()

    g = zarr.open_group(str(path), mode="a")
    split = np.asarray(g["split_210k"][:])
    split[0] = 1  # one cell moves out of train -- store_fingerprint's split histogram changes
    g["split_210k"][:] = split

    after = cached_constant_mask(str(path), "alphaearth", **kw)
    after_path = quality_cache_path(str(path), "alphaearth", "constant", split_name="split_210k", **kw)
    assert after_path != before_path
    assert after_path.exists()
    assert np.array_equal(after, before)  # the constant-cell mask itself is unaffected here


def test_patchdata_caches_quality_scans_by_default(tmp_path):
    """`quality_cache_root` unset -> PatchData still caches, at DEFAULT_QUALITY_CACHE_ROOT.

    The suite's autouse fixture (tests/conftest.py) redirects that default under this
    test's own tmp_path, so this exercises the real default-on behaviour without
    touching the shared, real ``data/eval_cache/quality``.
    """
    import geosae.data.quality as quality

    path = tmp_path / "d.zarr"
    _store(path, const_cells=(3,), flat_pixel_cells=(21,))
    default = PatchData(str(path), split="train", compute=False)
    assert len(default) == 38

    cache_dir = Path(quality.DEFAULT_QUALITY_CACHE_ROOT)
    assert cache_dir.exists()
    assert len(list(cache_dir.glob("*.npy"))) == 3  # constant + flat_pixel + extreme_value


def test_patchdata_quality_cache_root_none_disables_caching(tmp_path):
    import geosae.data.quality as quality

    path = tmp_path / "d.zarr"
    _store(path, const_cells=(3,))
    PatchData(str(path), split="train", compute=False, quality_cache_root=None)
    cache_dir = Path(quality.DEFAULT_QUALITY_CACHE_ROOT)
    assert not cache_dir.exists() or not list(cache_dir.glob("*.npy"))


def test_patchdata_quality_cache_root_reproduces_the_uncached_result(tmp_path):
    path = tmp_path / "d.zarr"
    _store(path, const_cells=(3,), flat_pixel_cells=(21,))
    cache_root = tmp_path / "cache"
    uncached = PatchData(str(path), split="train", compute=False, quality_cache_root=None)
    first = PatchData(str(path), split="train", compute=False, quality_cache_root=cache_root)
    second = PatchData(str(path), split="train", compute=False, quality_cache_root=cache_root)
    assert np.array_equal(first.cells, uncached.cells)
    assert np.array_equal(second.cells, uncached.cells)


def test_fill_pixels_flags_each_kind_of_fill_and_no_real_pixel():
    from geosae.data.quality import fill_pixels

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(5, 8))
    emb[1] = -1.0078585  # dequantised nodata code: every dim the same
    emb[2] = 0.0  # all-zero pixel
    emb[3, 2] = 2e27  # magnitude sentinel
    emb[4, 0] = np.nan
    assert fill_pixels(emb).tolist() == [False, True, True, True, True]

"""Black-box tests for geosae.eval.pool — the frozen evaluation pool.

Contracts asserted (not implementation):
  * a spec's digest separates any two recipes and does not depend on the process.
  * a pool is a deterministic function of its spec, and reproduces the draw
    ``sample_registry_pixels`` makes from the same recipe.
  * a pool round-trips through one ``.npz`` exactly, manifest included.
  * ``verify()`` refuses a pool whose store changed underneath it.
  * the frozen balanced draws are what a metric scores on: ``balanced_indices``
    never re-draws, and feeding them to ``concept_auroc_matrix`` reproduces the
    matrix the ad-hoc path computes.
  * ``audit_product_coverage`` separates covered from empty reference products.
  * the scorable-concept ladder is monotone in pool size, agrees with
    ``valid_concepts`` at full size, and reports a draw-to-draw spread.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from geosae.eval.coverage import balanced_binary_subsample, concept_auroc_matrix
from geosae.eval.pool import (
    EvalPool,
    PoolSpec,
    audit_product_coverage,
    scorable_concept_ladder,
    store_fingerprint,
)
from geosae.eval.registry import sample_registry_pixels

PRODUCTS = ["lc_esa_wc"]


@pytest.fixture
def pool_store(patch_store):
    """A tiny labelled store: 4 land-cover classes per patch, plus product masks."""
    path, _ = patch_store(n=24, patch_size=4, dim=8)
    g = zarr.open_group(path, mode="a")
    rng = np.random.default_rng(1)
    codes = rng.choice(np.array([10, 20, 30, 40], dtype="int16"), size=(24, 4, 4))
    g.create_array("lc_esa_wc", shape=(24, 4, 4), chunks=(6, 4, 4), dtype="int16")[:] = codes
    g.create_array("cell_indices", shape=(24,), chunks=(24,), dtype="int64")[:] = np.arange(24)
    g.create_array("mask_lc_esa_wc", shape=(24,), chunks=(24,), dtype="bool")[:] = np.ones(24, bool)
    g.create_array("mask_globeland30", shape=(24,), chunks=(24,), dtype="bool")[:] = np.zeros(24, bool)
    partial = np.zeros(24, bool)
    partial[18:] = True  # 6 cells, all inside the 6-cell test split (rows 18..23)
    g.create_array("mask_ghs_built_c", shape=(24,), chunks=(24,), dtype="bool")[:] = partial
    return path


def _spec(path, **kw) -> PoolSpec:
    fields = dict(
        name="unit",
        version=1,
        data_zarr=path,
        n_cells=4,
        pixels_per_cell=8,
        products=tuple(PRODUCTS),
        max_per_side=6,
        min_positive=2,
    )
    fields.update(kw)
    return PoolSpec(**fields)


# --------------------------------------------------------------------- PoolSpec


def test_digest_separates_every_field_and_pool_id_names_the_version(pool_store):
    base = _spec(pool_store)
    assert base.pool_id == "unit.v1"
    variants = [
        base.__class__(**{**base.to_dict(), "products": base.products, **{k: v}})
        for k, v in [
            ("name", "other"),
            ("version", 2),
            ("split", "train"),
            ("split_name", "split_10k"),
            ("embedding_name", "tessera"),
            ("n_cells", 5),
            ("pixels_per_cell", 9),
            ("seed", 1),
            ("products", ("lc_glc_fcs30d",)),
            ("max_per_side", 7),
            ("min_positive", 3),
        ]
    ]
    digests = {base.digest} | {v.digest for v in variants}
    assert len(digests) == len(variants) + 1
    assert len(base.digest) == 12


def test_digest_is_stable_across_processes_and_releases():
    # A golden value: the digest is a pure function of the recipe, so it may not
    # move with the interpreter's hash seed, with dict order, or with a release.
    spec = PoolSpec(name="reference", version=1, data_zarr="/store/patches.zarr")
    assert spec.digest == "5528fab092d1"


def test_spec_roundtrips_through_its_dict(pool_store):
    spec = _spec(pool_store)
    assert PoolSpec.from_dict(spec.to_dict()) == spec
    assert PoolSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert PoolSpec.from_dict({**spec.to_dict(), "products": None}).products is None


# --------------------------------------------------------------------- building


def test_build_is_deterministic_in_the_spec(pool_store):
    spec = _spec(pool_store)
    a, b = EvalPool.build(spec), EvalPool.build(spec)
    assert np.array_equal(a.cells, b.cells)
    assert np.array_equal(a.pixel_rows, b.pixel_rows) and np.array_equal(a.pixel_cols, b.pixel_cols)
    assert a.store_fingerprint == b.store_fingerprint
    assert set(a.balanced) == set(b.balanced) and set(a.balanced)
    for c in a.balanced:
        assert np.array_equal(a.balanced_indices(c)[0], b.balanced_indices(c)[0])
        assert np.array_equal(a.balanced_indices(c)[1], b.balanced_indices(c)[1])


def test_the_pool_reproduces_the_ad_hoc_registry_draw(pool_store):
    spec = _spec(pool_store)
    pool = EvalPool.build(spec)
    ad_hoc = sample_registry_pixels(
        pool_store,
        split=spec.split,
        split_name=spec.split_name,
        embedding_name=spec.embedding_name,
        n_patches=spec.n_cells,
        pixels_per_patch=spec.pixels_per_cell,
        products=list(PRODUCTS),
        seed=spec.seed,
    )
    frozen = pool.registry_sample()
    assert np.array_equal(frozen.raw, ad_hoc.raw)
    assert np.array_equal(frozen.cell, ad_hoc.cell)
    assert np.array_equal(frozen.labels["lc_esa_wc"], ad_hoc.labels["lc_esa_wc"])
    assert pool.n_pixels == ad_hoc.n_pixels == len(pool.cells) * len(pool.pixel_rows)
    assert np.array_equal(pool.n_positive, ad_hoc.n_positive)


def test_registry_sample_cache_returns_the_same_pixels(tmp_path, pool_store):
    pool = EvalPool.build(_spec(pool_store))
    cache = tmp_path / "reference.v1.sample.npz"

    fresh = pool.registry_sample(cache=cache)
    assert cache.exists()
    cached = pool.registry_sample(cache=cache)

    assert np.array_equal(cached.raw, fresh.raw)
    assert np.array_equal(cached.cell, fresh.cell)
    assert np.array_equal(cached.n_positive, fresh.n_positive)
    assert np.array_equal(cached.labels["lc_esa_wc"], fresh.labels["lc_esa_wc"])
    assert cached.product_order == fresh.product_order
    assert np.array_equal(cached.class_table, fresh.class_table)


def test_building_can_cache_the_pixels_it_already_read(tmp_path, pool_store):
    """Building reads the draw once; the cache is that read, not a second one."""
    cache = tmp_path / "unit.v1.sample.npz"
    pool = EvalPool.build(_spec(pool_store), sample_cache=cache)
    assert cache.exists()

    from_cache = pool.registry_sample(cache=cache)
    from_store = pool.registry_sample()
    assert np.array_equal(from_cache.raw, from_store.raw)
    assert np.array_equal(from_cache.cell, from_store.cell)
    assert np.array_equal(from_cache.labels["lc_esa_wc"], from_store.labels["lc_esa_wc"])


def test_registry_sample_refuses_a_cache_from_another_pool(tmp_path, pool_store):
    small = EvalPool.build(_spec(pool_store, n_cells=2))
    other = EvalPool.build(_spec(pool_store, n_cells=4))
    cache = tmp_path / "sample.npz"
    small.registry_sample(cache=cache)

    with pytest.raises(ValueError, match="cache"):
        other.registry_sample(cache=cache)


def test_every_cell_contributes_the_same_pixel_positions(pool_store):
    pool = EvalPool.build(_spec(pool_store))
    cells = pool.registry_sample().cell
    _, counts = np.unique(cells, return_counts=True)
    assert (counts == len(pool.pixel_rows)).all()


def test_prediction_indices_find_the_pool_cells_and_flag_a_gap(tmp_path, pool_store):
    pool = EvalPool.build(_spec(pool_store))
    pred = tmp_path / "test.zarr"
    g = zarr.open_group(pred, mode="w")
    # A predictions zarr holds one row per test-split cell, in cell order.
    test_cells = np.arange(18, 24, dtype=np.int64)
    g.create_array("cell", shape=(6,), chunks=(6,), dtype="int64")[:] = test_cells

    idx = pool.prediction_indices(pred)
    assert np.array_equal(test_cells[idx], pool.cells)
    assert np.array_equal(idx, np.sort(idx))

    g["cell"][int(idx[0])] = 999  # a cell the pool needs is no longer predicted
    with pytest.raises(ValueError, match="missing"):
        pool.prediction_indices(pred)


def test_dense_codes_line_up_with_registry_sample_pixel_for_pixel(tmp_path, pool_store):
    spec = _spec(pool_store, n_cells=4, pixels_per_cell=6)
    pool = EvalPool.build(spec)
    n_feat, max_k = 5, 3

    # Build a predictions zarr covering the whole 24-cell store, one row per
    # cell in cell order, with a distinctive dense-code pattern per pixel so a
    # wrong (cell, row, col) join is caught rather than merely a wrong shape.
    rng = np.random.default_rng(3)
    side = 4  # patch_store's default patch_size
    dense = np.zeros((24, side, side, n_feat), dtype=np.float32)
    for cell in range(24):
        for r in range(side):
            for c in range(side):
                k = rng.choice(n_feat, size=2, replace=False)
                dense[cell, r, c, k] = cell * 100 + r * 10 + c + 1

    pred = tmp_path / "test.zarr"
    g = zarr.open_group(pred, mode="w")
    g.create_array("cell", shape=(24,), chunks=(24,), dtype="int64")[:] = np.arange(24)
    values = g.create_array("z_values", shape=(24, side, side, max_k), dtype="float32")
    indices = g.create_array("z_indices", shape=(24, side, side, max_k), dtype="int16")
    for cell in range(24):
        for r in range(side):
            for c in range(side):
                nz = np.nonzero(dense[cell, r, c])[0]
                v, idx = np.zeros(max_k, "float32"), np.zeros(max_k, "int16")
                v[: len(nz)], idx[: len(nz)] = dense[cell, r, c, nz], nz
                values[cell, r, c], indices[cell, r, c] = v, idx

    codes = pool.dense_codes(pred, n_feat)
    assert codes.shape == (pool.n_pixels, n_feat)

    expect = dense[pool.cells[:, None], pool.pixel_rows[None, :], pool.pixel_cols[None, :], :]
    assert np.array_equal(codes, expect.reshape(-1, n_feat))

    # And it lines up with registry_sample's own pixel order (same cell axis).
    sample = pool.registry_sample()
    assert np.array_equal(sample.cell, np.repeat(pool.cells, len(pool.pixel_rows)))


def test_iter_dense_codes_chunks_concatenate_to_dense_codes(tmp_path, pool_store):
    spec = _spec(pool_store, n_cells=4, pixels_per_cell=6)
    pool = EvalPool.build(spec)
    n_feat, max_k = 5, 3

    rng = np.random.default_rng(7)
    side = 4
    dense = np.zeros((24, side, side, n_feat), dtype=np.float32)
    for cell in range(24):
        for r in range(side):
            for c in range(side):
                k = rng.choice(n_feat, size=2, replace=False)
                dense[cell, r, c, k] = cell * 100 + r * 10 + c + 1

    pred = tmp_path / "test.zarr"
    g = zarr.open_group(pred, mode="w")
    g.create_array("cell", shape=(24,), chunks=(24,), dtype="int64")[:] = np.arange(24)
    values = g.create_array("z_values", shape=(24, side, side, max_k), dtype="float32")
    indices = g.create_array("z_indices", shape=(24, side, side, max_k), dtype="int16")
    for cell in range(24):
        for r in range(side):
            for c in range(side):
                nz = np.nonzero(dense[cell, r, c])[0]
                v, idx = np.zeros(max_k, "float32"), np.zeros(max_k, "int16")
                v[: len(nz)], idx[: len(nz)] = dense[cell, r, c, nz], nz
                values[cell, r, c], indices[cell, r, c] = v, idx

    whole = pool.dense_codes(pred, n_feat)

    # chunk_cells=1 -- the smallest possible chunk, and not a multiple of the
    # pool's 4 cells, so this also exercises an uneven final chunk.
    chunks = list(pool.iter_dense_codes(pred, n_feat, chunk_cells=1))
    assert len(chunks) == pool.cells.size  # one chunk per cell, at chunk_cells=1
    n_pix = pool.pixel_rows.size
    for i, (start, end, chunk) in enumerate(chunks):
        assert (start, end) == (i * n_pix, (i + 1) * n_pix)
        assert chunk.shape == (n_pix, n_feat)
    rebuilt = np.concatenate([c for _, _, c in chunks], axis=0)
    assert np.array_equal(rebuilt, whole)

    # A chunk_cells that does not evenly divide the pool still covers every
    # pixel exactly once, in order.
    uneven = list(pool.iter_dense_codes(pred, n_feat, chunk_cells=3))
    assert np.array_equal(np.concatenate([c for _, _, c in uneven], axis=0), whole)
    assert uneven[0][1] - uneven[0][0] == 3 * n_pix
    assert uneven[-1][1] == pool.n_pixels


def test_dense_codes_flags_a_prediction_gap_like_prediction_indices(tmp_path, pool_store):
    pool = EvalPool.build(_spec(pool_store))
    pred = tmp_path / "test.zarr"
    g = zarr.open_group(pred, mode="w")
    test_cells = np.arange(18, 24, dtype=np.int64)
    g.create_array("cell", shape=(6,), chunks=(6,), dtype="int64")[:] = test_cells
    g.create_array("z_values", shape=(6, 4, 4, 2), dtype="float32")
    g.create_array("z_indices", shape=(6, 4, 4, 2), dtype="int16")

    idx = pool.prediction_indices(pred)  # a row this pool actually needs
    g["cell"][int(idx[0])] = 999
    with pytest.raises(ValueError, match="missing"):
        pool.dense_codes(pred, n_features=3)


def test_pixel_lonlat_gives_one_coordinate_per_pool_pixel(pool_store):
    pool = EvalPool.build(_spec(pool_store))
    lon, lat = pool.pixel_lonlat()
    assert lon.shape == lat.shape == (pool.n_pixels,)

    store = zarr.open_group(pool_store, mode="r")
    expect = np.asarray(store["lon"][:])[pool.cells]
    # Every pixel of a cell carries that cell's centroid, in the pool's order.
    assert np.array_equal(lon.reshape(pool.cells.size, -1)[:, 0], expect)
    assert (lon.reshape(pool.cells.size, -1) == expect[:, None]).all()


# ------------------------------------------------------------- frozen balancing


def test_balanced_indices_are_frozen_and_match_the_build_time_draw(pool_store):
    spec = _spec(pool_store)
    pool = EvalPool.build(spec)
    labels = pool.registry_sample().labels["lc_esa_wc"]
    scorable = np.where(pool.scorable_concepts())[0]
    assert scorable.size

    for c in scorable:
        idx, y = pool.balanced_indices(int(c))
        again, y_again = pool.balanced_indices(int(c))
        assert np.array_equal(idx, again) and np.array_equal(y, y_again)
        expect_idx, expect_y = balanced_binary_subsample(
            labels,
            int(pool.class_table[c, 1]),
            max_per_side=spec.max_per_side,
            seed=spec.seed + int(c),
            min_positive=spec.min_positive,
        )
        assert np.array_equal(idx, expect_idx) and np.array_equal(y, expect_y)
        assert (labels[idx[y == 1]] == int(pool.class_table[c, 1])).all()

    unscorable = np.where(~pool.scorable_concepts())[0]
    assert pool.balanced_indices(int(unscorable[0]))[0].size == 0


def test_auroc_matrix_with_frozen_draws_equals_the_ad_hoc_matrix(pool_store):
    spec = _spec(pool_store, seed=0)
    pool = EvalPool.build(spec)
    sample = pool.registry_sample()
    rng = np.random.default_rng(3)
    codes = rng.standard_normal((sample.n_pixels, 5)).astype(np.float32)
    common = dict(max_per_side=spec.max_per_side, min_positive=spec.min_positive)

    ad_hoc = concept_auroc_matrix(
        codes, sample.labels, sample.class_table, sample.product_order, seed=spec.seed, **common
    )
    frozen = concept_auroc_matrix(
        codes,
        sample.labels,
        sample.class_table,
        sample.product_order,
        balanced={c: pool.balanced_indices(c) for c in pool.balanced},
        **common,
    )
    assert np.array_equal(frozen, ad_hoc)


# ------------------------------------------------------------------- io + audit


def test_save_load_roundtrips_and_writes_a_manifest(tmp_path, pool_store):
    pool = EvalPool.build(_spec(pool_store))
    path = tmp_path / "pools" / "unit.v1.npz"
    pool.save(path)
    back = EvalPool.load(path)

    assert back.spec == pool.spec
    assert back.store_fingerprint == pool.store_fingerprint
    assert back.concept_names == pool.concept_names and back.concept_groups == pool.concept_groups
    assert back.product_order == pool.product_order
    for name in ("cells", "pixel_rows", "pixel_cols", "class_table", "n_positive"):
        assert np.array_equal(getattr(back, name), getattr(pool, name))
    assert set(back.balanced) == set(pool.balanced)
    for c in pool.balanced:
        assert np.array_equal(back.balanced_indices(c)[0], pool.balanced_indices(c)[0])
        assert np.array_equal(back.balanced_indices(c)[1], pool.balanced_indices(c)[1])

    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest == pool.manifest()
    assert manifest["pool_id"] == "unit.v1" and manifest["spec_digest"] == pool.spec.digest
    assert manifest["n_pixels"] == pool.n_pixels
    assert manifest["scorable_per_product"]["lc_esa_wc"] == int(pool.scorable_concepts().sum())


def test_verify_rejects_a_pool_whose_store_changed(pool_store):
    pool = EvalPool.build(_spec(pool_store))
    assert pool.verify() is True

    g = zarr.open_group(pool_store, mode="a")
    split = np.asarray(g["split_210k"][:])
    split[0] = 2  # move one cell into the test split
    g["split_210k"][:] = split

    assert pool.verify(strict=False) is False
    with pytest.raises(ValueError, match=pool.store_fingerprint):
        pool.verify()


def test_store_fingerprint_is_stable_and_content_addressed(pool_store):
    kw = dict(split_name="split_210k", embedding_name="alphaearth")
    first = store_fingerprint(pool_store, **kw)
    assert len(first) == 16
    assert store_fingerprint(pool_store, **kw) == first

    g = zarr.open_group(pool_store, mode="a")
    mask = np.asarray(g["mask_alphaearth"][:])
    mask[3] = False
    g["mask_alphaearth"][:] = mask
    assert store_fingerprint(pool_store, **kw) != first


def test_audit_reports_empty_and_partial_products(pool_store):
    audit = audit_product_coverage(pool_store, split_name="split_210k", split="test")
    assert audit["globeland30"] == {"n_store": 0, "n_split": 0, "frac_split": 0.0}
    assert audit["lc_esa_wc"]["n_store"] == 24 and audit["lc_esa_wc"]["frac_split"] == 1.0
    # 6 partially covered cells, all of them in the 6-cell test split.
    assert audit["ghs_built_c"] == {"n_store": 6, "n_split": 6, "frac_split": 1.0}
    assert audit["alphaearth"]["n_split"] == 6


# ----------------------------------------------------------- pool-size ladder


@pytest.fixture
def ladder_sample(pool_store):
    """The whole test split at full pixel density — the ladder's input."""
    return sample_registry_pixels(
        pool_store, split="test", n_patches=6, pixels_per_patch=16, products=list(PRODUCTS), seed=0
    )


def test_ladder_is_monotone_and_ends_at_the_full_sample(ladder_sample):
    rungs = scorable_concept_ladder(ladder_sample, [1, 2, 4, 6], min_positive=4)
    assert [r["n_cells"] for r in rungs] == [1, 2, 4, 6]
    assert [r["n_pixels"] for r in rungs] == [16, 32, 64, 96]
    # Nested subsets, so a bigger pool can only add positives, never remove them.
    counts = [r["n_scorable"] for r in rungs]
    assert counts == sorted(counts)
    # The last rung is the whole sample, where "scorable" is exactly the rule
    # the balanced sampler applies.
    assert rungs[-1]["n_scorable"] == float(ladder_sample.valid_concepts(4).sum())
    assert rungs[-1]["n_concepts"] == ladder_sample.class_table.shape[0]


def test_ladder_breaks_the_count_down_by_product(ladder_sample):
    rung = scorable_concept_ladder(ladder_sample, [6], min_positive=4)[0]
    per_product = rung["scorable_per_product"]
    assert set(per_product) == set(ladder_sample.product_order)
    assert sum(per_product.values()) == rung["n_scorable"]
    assert per_product["lc_esa_wc"] == rung["n_scorable"]


def test_ladder_counts_fall_as_the_positive_floor_rises(ladder_sample):
    sizes = [2, 6]
    low = scorable_concept_ladder(ladder_sample, sizes, min_positive=1)
    high = scorable_concept_ladder(ladder_sample, sizes, min_positive=40)
    assert all(h["n_scorable"] <= v["n_scorable"] for h, v in zip(high, low, strict=True))
    assert low[-1]["n_scorable"] == len(PRODUCTS) * 4  # all four classes clear a floor of 1


def test_ladder_reports_draw_to_draw_spread_and_is_seed_reproducible(ladder_sample):
    kw = dict(min_positive=4, n_draws=5, seed=7)
    rungs = scorable_concept_ladder(ladder_sample, [3], **kw)
    assert rungs == scorable_concept_ladder(ladder_sample, [3], **kw)
    r = rungs[0]
    assert r["n_draws"] == 5
    assert r["n_scorable_min"] <= r["n_scorable"] <= r["n_scorable_max"]
    assert r["n_scorable_std"] >= 0.0
    # A single draw has no spread to report, and is the nested prefix.
    single = scorable_concept_ladder(ladder_sample, [3], min_positive=4)[0]
    assert single["n_draws"] == 1 and single["n_scorable_std"] == 0.0


def test_ladder_clamps_a_size_beyond_the_sample(ladder_sample):
    beyond, full = scorable_concept_ladder(ladder_sample, [999, 6], min_positive=4)
    assert beyond["n_cells"] == 6
    assert beyond == full

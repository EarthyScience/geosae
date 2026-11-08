"""Black-box tests for the majortom210k preset (pure data/config, core install only)."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.data.download.presets import MAJORTOM210K, PRESETS, DatasetPreset
from geosae.data.download.selection import EXCLUDED, TEST, TRAIN, VAL, SplitSpec, compute_split, sample_subset_indices
from geosae.data.download.sources import Kind, get_product


def test_majortom210k_is_the_leading_210k_rows_of_a_1m_draw():
    assert PRESETS["majortom210k"] is MAJORTOM210K
    assert (MAJORTOM210K.base_ncells, MAJORTOM210K.seed) == (5_418_682, 42)
    assert (MAJORTOM210K.draw_size, MAJORTOM210K.n_cells) == (1_000_000, 210_000)
    assert set(MAJORTOM210K.splits) == {"210k", "10k"}
    assert all(isinstance(s, SplitSpec) for s in MAJORTOM210K.splits.values())


@pytest.mark.parametrize("total", [5_000, 77_777])
def test_cell_indices_are_the_leading_rows_of_the_full_draw(total):
    """Drawing fewer cells reorders them; the store must keep the full draw's rows."""
    preset = DatasetPreset("toy", total, seed=42, draw_size=3_000, n_cells=630, products=(), splits={})
    full = sample_subset_indices(total, 3_000, seed=42)
    assert np.array_equal(preset.cell_indices(), full[:630])
    assert not np.array_equal(preset.cell_indices(), sample_subset_indices(total, 630, seed=42))


def test_cell_indices_take_the_actual_grid_size():
    preset = DatasetPreset("toy", 1_000, seed=0, draw_size=500, n_cells=100, products=(), splits={})
    assert np.array_equal(preset.cell_indices(2_000), sample_subset_indices(2_000, 500, seed=0)[:100])


def test_majortom210k_split_ranges():
    n = MAJORTOM210K.n_cells
    split = compute_split(MAJORTOM210K.splits["210k"], n, np.ones(n, dtype=bool))
    assert (split[:100_000] == TEST).all()
    assert (split[100_000:200_000] == TRAIN).all()
    assert (split[200_000:] == VAL).all()
    assert MAJORTOM210K.reference_rows == (0, 100_000)


def test_the_10k_split_is_a_subset_of_the_210k_split():
    n = MAJORTOM210K.n_cells
    both = np.random.default_rng(0).random(n) > 0.001
    big = compute_split(MAJORTOM210K.splits["210k"], n, both)
    small = compute_split(MAJORTOM210K.splits["10k"], n, both)
    member = small != EXCLUDED
    assert (small[member] == big[member]).all()
    assert [int((small == v).sum()) for v in (TRAIN, VAL, TEST)] == [
        int(both[100_000:110_000].sum()),
        int(both[200_000:202_000].sum()),
        int(both[:10_000].sum()),
    ]


def test_majortom210k_products_fill_each_array_once():
    specs = [get_product(k) for k in MAJORTOM210K.products]
    arrays = [s.array for s in specs]
    assert len(arrays) == len(set(arrays))
    assert {s.array for s in specs if s.kind is Kind.EMBEDDING} == set(MAJORTOM210K.embeddings)


def test_majortom210k_holds_every_array_the_analyses_read():
    """The concept registry, the covariates and the sub-concept reference partitions."""
    from geosae.eval.covariates import CONTINUOUS_COVARIATES
    from geosae.eval.subconcepts import REFERENCE_PRODUCTS

    arrays = {get_product(k).array for k in MAJORTOM210K.products}
    concept_arrays = {
        "overture",
        "lc_esa_wc",
        "lc_esa_cci",
        "lc_esri",
        "glc_fcs30d",
        "glance30",
        "lc_modis_igbp",
        "worldcereal",
        "geomorphons",
        "soilgrids_class",
        "lcz",
        "chmv2",
        "dem_cop30",
        "gpp",
    }
    assert concept_arrays <= arrays
    assert set(REFERENCE_PRODUCTS) <= arrays
    assert {c.key for c in CONTINUOUS_COVARIATES.values()} - {"soc"} <= arrays


def test_shipped_tessera_statistics():
    stats = MAJORTOM210K.load_norm_stats()["tessera_norm"]
    assert set(stats) == {"mean", "std", "clip_lo", "clip_hi"}
    for value in stats.values():
        assert value.shape == (128,) and value.dtype == np.float32 and np.isfinite(value).all()
    assert (stats["std"] > 0).all() and (stats["clip_lo"] < stats["clip_hi"]).all()

"""Black-box tests for the general cell selection + split logic.

Pure-numpy, CPU, synthetic. These assert the contract that makes a dataset
build reproducible: determinism of the seeded draw, the two-generator
scatter, chunk-aligned index-range splits, and mask exclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.data.download.presets import MAJORTOM210K_N_CELLS, SPLIT_210K
from geosae.data.download.selection import (
    EXCLUDED,
    TEST,
    TRAIN,
    VAL,
    SplitSpec,
    compute_split,
    sample_subset_indices,
    split_counts,
)


def test_sample_subset_indices_deterministic():
    a = sample_subset_indices(10_000, 1_000, seed=7)
    b = sample_subset_indices(10_000, 1_000, seed=7)
    assert np.array_equal(a, b)
    assert a.dtype == np.int64
    assert a.size == 1_000
    # values are valid base-grid indices, drawn without replacement
    assert a.min() >= 0 and a.max() < 10_000
    assert np.unique(a).size == a.size


def test_sample_subset_indices_seed_sensitive():
    a = sample_subset_indices(10_000, 1_000, seed=7)
    b = sample_subset_indices(10_000, 1_000, seed=8)
    assert not np.array_equal(a, b)


def test_scatter_uses_second_generator():
    """The scatter must use default_rng(seed+1), not the draw generator, so the
    order differs from the raw choice order."""
    total, n, seed = 10_000, 2_000, 3
    raw = np.random.default_rng(seed).choice(total, size=n, replace=False).astype(np.int64)
    scattered = sample_subset_indices(total, n, seed)
    # same set of cells, different order
    assert np.array_equal(np.sort(raw), np.sort(scattered))
    assert not np.array_equal(raw, scattered)
    # reproduces the explicit two-generator recipe
    expected = raw.copy()
    np.random.default_rng(seed + 1).shuffle(expected)
    assert np.array_equal(scattered, expected)


def test_full_grid_uses_arange():
    idx = sample_subset_indices(500, 500, seed=1)
    # n_cells >= total -> every cell, but still scattered
    assert np.array_equal(np.sort(idx), np.arange(500))


def test_compute_split_210k_ranges():
    n = MAJORTOM210K_N_CELLS
    both = np.ones(n, dtype=bool)
    split = compute_split(SPLIT_210K, n, both)
    # exact ranges when nothing is excluded
    assert np.all(split[:100_000] == TEST)
    assert np.all(split[100_000:200_000] == TRAIN)
    assert np.all(split[200_000:210_000] == VAL)
    assert np.all(split[210_000:] == EXCLUDED)
    c = split_counts(split)
    assert c == {"train": 100_000, "val": 10_000, "test": 100_000, "excluded": n - 210_000}


def test_compute_split_respects_both_mask():
    n = MAJORTOM210K_N_CELLS
    both = np.ones(n, dtype=bool)
    both[:65] = False  # 65 test-range cells missing an embedding
    split = compute_split(SPLIT_210K, n, both)
    assert np.all(split[:65] == EXCLUDED)
    assert split_counts(split)["test"] == 100_000 - 65


def test_compute_split_disjoint_by_row():
    n = MAJORTOM210K_N_CELLS
    split = compute_split(SPLIT_210K, n, np.ones(n, dtype=bool))
    # a row is in at most one split
    for a, b in [(TRAIN, VAL), (TRAIN, TEST), (VAL, TEST)]:
        assert not np.any((split == a) & (split == b))


def test_compute_split_rejects_misaligned_boundary():
    spec = SplitSpec(name="bad", test=(0, 111), train=None, val=None, chunk_size=50)
    with pytest.raises(ValueError, match="aligned"):
        compute_split(spec, 1_000, np.ones(1_000, dtype=bool))


def test_compute_split_bad_mask_shape():
    with pytest.raises(ValueError, match="both_mask"):
        compute_split(SPLIT_210K, 1_000, np.ones(5, dtype=bool))

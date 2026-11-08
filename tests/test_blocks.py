"""The panel blocks: their register, their fill loop and the concept block's ceiling."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval.auroc_moran import patch_mean_moran_from_arrays
from geosae.eval.blocks import (
    BLOCK_OF,
    BLOCK_SCALARS,
    HIGHER_IS_BETTER,
    concept_block,
    fidelity_block,
    fill_panel,
    geometry_block,
)
from geosae.eval.statistic import PanelAccumulator, calibrate_activation_bins


def _world(n: int, *, seed: int = 0, dim: int = 8, n_features: int = 6, rate: float = 0.3):
    """Pixels, a unit dictionary and codes whose active sets are pure chance."""
    rng = np.random.default_rng(seed)
    decoder = rng.normal(size=(n_features, dim))
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    e = rng.normal(size=(n, dim)) @ np.linalg.qr(rng.normal(size=(dim, dim)))[0]
    codes = np.abs(rng.normal(size=(n, n_features))) * (rng.uniform(size=(n, n_features)) < rate)
    return e, decoder, codes


def _fill(e, decoder, codes, *, concept_mask=None, n_bins=32):
    """A filled accumulator over one batch, the long way round."""
    n_concepts = 0 if concept_mask is None else concept_mask.shape[1]
    acc = PanelAccumulator.create(
        n_features=decoder.shape[0],
        embedding_dim=decoder.shape[1],
        concept_names=[f"c{i}" for i in range(n_concepts)],
        product_names=["p"] if n_concepts else [],
        product_of_concept=np.zeros(n_concepts, dtype=np.int64),
        bin_edges=calibrate_activation_bins(codes, n_bins=n_bins),
    )
    kw = {}
    if concept_mask is not None:
        kw = {"concept_mask": concept_mask, "product_valid": np.ones((len(e), 1), dtype=bool)}
    acc.update(e, codes @ decoder, codes, **kw)
    return acc


def _pixel_source(e: np.ndarray, chunk: int = 256):
    """A re-iterable pixel source over ``e``, in the block protocol."""

    def source():
        for start in range(0, len(e), chunk):
            yield e[start : start + chunk], None

    return source


def test_register_is_complete_and_consistent():
    """Every block scalar has a block and a direction of merit, and no other does."""
    names = [s for block in BLOCK_SCALARS.values() for s in block]
    assert len(names) == len(set(names)), "a scalar is listed in two blocks"
    assert set(BLOCK_OF) == set(names)
    assert set(HIGHER_IS_BETTER) == set(names)


def test_blocks_1_and_2_read_only_the_accumulator():
    """The cheap blocks return their declared scalars and nothing else."""
    e, decoder, codes = _world(4000)
    acc = _fill(e, decoder, codes)
    assert set(fidelity_block(acc)) == set(BLOCK_SCALARS[1])
    assert set(geometry_block(decoder, acc)) == set(BLOCK_SCALARS[2])


def test_fill_panel_matches_a_single_batch_fill():
    """Streaming a source in chunks gives the accumulator one batch would."""
    e, decoder, codes = _world(3000, seed=3)
    mask = codes[:, :3] > 0
    valid = np.ones((len(e), 1), dtype=bool)

    def source():
        for s in range(0, len(e), 512):
            yield e[s : s + 512], (mask[s : s + 512], valid[s : s + 512])

    acc, sample = fill_panel(
        source,
        lambda raw: _lookup(e, codes, raw),
        decoder,
        concept_names=["a", "b", "c"],
        product_names=["p"],
        product_of_concept=np.zeros(3, dtype=np.int64),
        n_bins=32,
        codes_sample_pixels=100,
    )
    direct = _fill(e, decoder, codes, concept_mask=mask, n_bins=32)
    assert acc.n_pixels == direct.n_pixels == len(e)
    assert np.allclose(acc.sum_a, direct.sum_a)
    assert np.array_equal(acc.n_fc, direct.n_fc)
    assert sample.shape == (100, decoder.shape[0])


def _lookup(e: np.ndarray, codes: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """The codes of the rows of ``e`` that ``raw`` is, in order — a stand-in encoder."""
    start = int(np.flatnonzero((e == raw[0]).all(axis=1))[0])
    return codes[start : start + len(raw)]


def test_fill_panel_rejects_an_empty_source():
    """An empty source is an error, not a zeroed accumulator nobody notices."""
    _, decoder, _ = _world(10)
    with pytest.raises(ValueError, match="no pixels"):
        fill_panel(lambda: iter(()), lambda raw: raw, decoder)


def test_concept_block_scores_a_perfect_dictionary_at_its_ceiling():
    """A dictionary whose features *are* the concepts covers every one of them."""
    rng = np.random.default_rng(0)
    n, dim, n_features = 6000, 8, 4
    decoder = np.eye(n_features, dim)
    codes = np.abs(rng.normal(size=(n, n_features))) * (rng.uniform(size=(n, n_features)) < 0.3)
    mask = codes > 0
    acc = _fill(rng.normal(scale=0.01, size=(n, dim)) + codes @ decoder, decoder, codes, concept_mask=mask)
    class_table = np.stack([np.zeros(n_features, dtype=np.int64), np.arange(n_features)], axis=1)
    out = concept_block(
        acc,
        class_table=class_table,
        product_order=["p"],
        class_to_group={("p", f): f"g{f}" for f in range(n_features)},
        min_positive=50,
    )
    assert set(out) == set(BLOCK_SCALARS[3])
    assert out["ucov"] == n_features
    assert out["argmax_coverage"] == n_features
    assert out["ucov_frac_ge_thresh"] == pytest.approx(1.0)


def test_patch_mean_moran_separates_a_smooth_field_from_noise():
    """Moran's I is high for a field that varies smoothly in space, ~0 for white noise."""
    rng = np.random.default_rng(0)
    lon = np.linspace(-40, 40, 120)
    lat = np.zeros_like(lon)
    means = np.stack([np.sin(np.radians(lon) * 2), rng.normal(size=lon.size)])
    out = patch_mean_moran_from_arrays(means, lon, lat, k_neighbors=4)
    per_feature = [patch_mean_moran_from_arrays(means[i : i + 1], lon, lat, k_neighbors=4)["moran_i"] for i in range(2)]
    assert per_feature[0] > 0.8
    assert abs(per_feature[1]) < 0.4
    assert out["n_patches"] == lon.size


def test_patch_mean_moran_reports_every_feature_in_order():
    rng = np.random.default_rng(0)
    lon = np.linspace(-40, 40, 120)
    lat = np.zeros_like(lon)
    means = np.stack([rng.normal(size=lon.size), np.ones_like(lon), np.sin(np.radians(lon) * 2)])
    out = patch_mean_moran_from_arrays(means, lon, lat, k_neighbors=4, per_feature=True)
    assert "moran_per_feature" not in patch_mean_moran_from_arrays(means, lon, lat, k_neighbors=4)
    single = [patch_mean_moran_from_arrays(means[i : i + 1], lon, lat, k_neighbors=4)["moran_i"] for i in (0, 2)]
    assert out["moran_per_feature"].shape == (3,)
    assert np.isnan(out["moran_per_feature"][1])  # a constant field has no Moran's I
    assert np.allclose(out["moran_per_feature"][[0, 2]], single)

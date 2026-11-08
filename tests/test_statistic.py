"""The sufficient-statistic contract: every derivation against a direct computation.

:class:`~geosae.eval.statistic.PanelAccumulator` is the single streaming pass
behind the panel blocks. These tests build tiny synthetic pixel populations,
fill the accumulator batch by batch, and check every derivation against a
brute-force computation on the same data.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval.coverage import auc_all_features, concept_perplexity
from geosae.eval.statistic import (
    PanelAccumulator,
    auc_from_histograms,
    auroc_matrix_from_accumulator,
    calibrate_activation_bins,
    contingency_tables,
    dictionary_geometry_metrics,
    fidelity_metrics,
    merge_accumulators,
)

D = 4
F = 5
C = 3
P = 2


def _synthetic(seed=0, n=400):
    """A small pixel population with a known decoder, codes, labels and reconstruction."""
    rng = np.random.default_rng(seed)
    decoder = rng.normal(size=(F, D))
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    bias = rng.normal(size=D) * 0.1

    codes = np.clip(rng.normal(0.5, 1.0, size=(n, F)), 0, None)
    codes *= rng.uniform(size=(n, F)) < 0.4  # ~40% active per feature
    codes[:, -1] = 0.0  # a genuinely dead feature

    ehat = codes @ decoder + bias
    e = ehat + rng.normal(scale=0.2, size=(n, D))  # imperfect reconstruction

    concept_names = ["water", "forest", "urban"]
    product_names = ["lc", "clim"]
    product_of_concept = np.array([0, 0, 1])  # water/forest from "lc", urban from "clim"
    concept_mask = np.zeros((n, C), dtype=bool)
    concept_mask[:, 0] = rng.uniform(size=n) < 0.3
    concept_mask[:, 1] = (~concept_mask[:, 0]) & (rng.uniform(size=n) < 0.4)
    concept_mask[:, 2] = rng.uniform(size=n) < 0.5
    product_valid = np.zeros((n, P), dtype=bool)
    product_valid[:, 0] = concept_mask[:, 0] | concept_mask[:, 1] | (rng.uniform(size=n) < 0.9)
    product_valid[:, 1] = concept_mask[:, 2] | (rng.uniform(size=n) < 0.9)

    return {
        "decoder": decoder,
        "bias": bias,
        "codes": codes,
        "e": e,
        "ehat": ehat,
        "concept_names": concept_names,
        "product_names": product_names,
        "product_of_concept": product_of_concept,
        "concept_mask": concept_mask,
        "product_valid": product_valid,
    }


def _accumulator(data, n_bins=32, batch_size=37):
    edges = calibrate_activation_bins(data["codes"], n_bins)
    acc = PanelAccumulator.create(
        n_features=F,
        embedding_dim=D,
        concept_names=data["concept_names"],
        product_names=data["product_names"],
        product_of_concept=data["product_of_concept"],
        bin_edges=edges,
    )
    n = data["codes"].shape[0]
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        acc.update(
            data["e"][sl],
            data["ehat"][sl],
            data["codes"][sl],
            concept_mask=data["concept_mask"][sl],
            product_valid=data["product_valid"][sl],
        )
    return acc


# ---------------------------------------------------------------------------
# Accumulator lifecycle
# ---------------------------------------------------------------------------


def test_bin_index_matches_a_naive_per_feature_searchsorted_loop():
    """`_bin_index`'s batched `torch.searchsorted` against one `np.searchsorted` call per feature.

    `test_concept_and_product_hist_match_a_naive_python_loop` below reuses
    `_bin_index`'s own output as its reference's input, so it cannot catch a
    bug in the binning itself — only in the histogram scatter built on top of
    it. This test is independent of `_bin_index`'s implementation.
    """
    from geosae.eval.statistic import _bin_index

    rng = np.random.default_rng(11)
    n_bins, f = 16, 7
    codes = rng.normal(0.3, 1.0, size=(500, f))
    edges = np.sort(rng.normal(0.0, 1.0, size=(f, n_bins - 1)), axis=1)

    batched = _bin_index(codes, edges)
    ref = np.stack([np.searchsorted(edges[j], codes[:, j], side="right") for j in range(f)], axis=1)
    assert np.array_equal(batched, ref)


def test_concept_and_product_hist_match_a_naive_python_loop():
    """The sparse-matmul histogram fill against the reference per-column bincount loop it replaced."""
    from geosae.eval.statistic import _bin_index

    data = _synthetic(seed=10, n=700)
    acc = _accumulator(data, n_bins=16)

    codes, cm, pv = data["codes"], data["concept_mask"], data["product_valid"]
    bin_idx = _bin_index(codes, acc.bin_edges)
    flat = bin_idx + np.arange(F)[None, :] * acc.n_bins
    concept_ref = np.zeros((F, C, acc.n_bins), dtype=np.int32)
    for c in range(C):
        rows = cm[:, c]
        if rows.any():
            counts = np.bincount(flat[rows].ravel(), minlength=F * acc.n_bins)
            concept_ref[:, c, :] += counts.reshape(F, acc.n_bins)
    product_ref = np.zeros((F, P, acc.n_bins), dtype=np.int32)
    for p in range(P):
        rows = pv[:, p]
        if rows.any():
            counts = np.bincount(flat[rows].ravel(), minlength=F * acc.n_bins)
            product_ref[:, p, :] += counts.reshape(F, acc.n_bins)
    assert np.array_equal(acc.concept_hist, concept_ref)
    assert np.array_equal(acc.product_hist, product_ref)


def test_unique_argmax_coverage_counts_concepts_with_a_dedicated_winner():
    from geosae.eval.statistic import unique_argmax_coverage

    rng = np.random.default_rng(0)
    n = 2000
    codes = np.zeros((n, 4))
    concept_mask = np.zeros((n, 2), dtype=bool)
    # feature 0 always wins on concept-0 pixels, feature 1 always wins on concept-1 pixels;
    # features 2/3 never win anything
    half = n // 2
    codes[:half, 0] = 1.0
    concept_mask[:half, 0] = True
    codes[half:, 1] = 1.0
    concept_mask[half:, 1] = True

    acc = PanelAccumulator.create(
        n_features=4,
        embedding_dim=D,
        concept_names=["a", "b"],
        product_names=["p"],
        product_of_concept=np.array([0, 0]),
        bin_edges=calibrate_activation_bins(codes, 4),
    )
    acc.update(
        rng.normal(size=(n, D)),
        rng.normal(size=(n, D)),
        codes,
        concept_mask=concept_mask,
        product_valid=np.ones((n, 1), dtype=bool),
    )
    out = unique_argmax_coverage(acc)
    assert out["coverage"] == 2
    assert out["best_concept"][0] == 0
    assert out["best_concept"][1] == 1
    assert out["best_concept"][2] == -1 and out["best_concept"][3] == -1


def test_update_is_invariant_to_batch_size():
    data = _synthetic()
    a1 = _accumulator(data, batch_size=1)
    a400 = _accumulator(data, batch_size=400)
    assert a1.sum_e == pytest.approx(a400.sum_e, rel=1e-8)
    assert a1.sum_e_outer == pytest.approx(a400.sum_e_outer, rel=1e-8)
    assert a1.n_fc == pytest.approx(a400.n_fc)
    assert np.array_equal(a1.concept_hist, a400.concept_hist)
    assert np.array_equal(a1.product_hist, a400.product_hist)


def test_merge_equals_one_big_update():
    data = _synthetic()
    whole = _accumulator(data, batch_size=400)
    half = data["codes"].shape[0] // 2
    edges = whole.bin_edges
    a = PanelAccumulator.create(
        n_features=F,
        embedding_dim=D,
        concept_names=data["concept_names"],
        product_names=data["product_names"],
        product_of_concept=data["product_of_concept"],
        bin_edges=edges,
    )
    b = PanelAccumulator.create(
        n_features=F,
        embedding_dim=D,
        concept_names=data["concept_names"],
        product_names=data["product_names"],
        product_of_concept=data["product_of_concept"],
        bin_edges=edges,
    )
    a.update(
        data["e"][:half],
        data["ehat"][:half],
        data["codes"][:half],
        concept_mask=data["concept_mask"][:half],
        product_valid=data["product_valid"][:half],
    )
    b.update(
        data["e"][half:],
        data["ehat"][half:],
        data["codes"][half:],
        concept_mask=data["concept_mask"][half:],
        product_valid=data["product_valid"][half:],
    )
    merged = merge_accumulators([a, b])
    assert merged.n_pixels == whole.n_pixels
    assert merged.sum_e == pytest.approx(whole.sum_e)
    assert merged.coact == pytest.approx(whole.coact)
    assert merged.sum_pixel_cos == pytest.approx(whole.sum_pixel_cos)
    assert np.array_equal(merged.concept_hist, whole.concept_hist)
    assert np.array_equal(merged.n_fc, whole.n_fc)


def test_accumulator_roundtrips_through_npz(tmp_path):
    data = _synthetic()
    acc = _accumulator(data)
    p = acc.save(tmp_path / "panel.npz")
    loaded = PanelAccumulator.load(p)
    assert loaded.n_features == acc.n_features
    assert loaded.concept_names == acc.concept_names
    assert loaded.sum_e_outer == pytest.approx(acc.sum_e_outer)
    assert loaded.sum_pixel_cos == pytest.approx(acc.sum_pixel_cos)
    assert np.array_equal(loaded.concept_hist, acc.concept_hist)


def test_update_rejects_mismatched_shapes():
    data = _synthetic()
    acc = _accumulator(data)
    with pytest.raises(ValueError):
        acc.update(data["e"][:10, :1], data["ehat"][:10], data["codes"][:10])
    with pytest.raises(ValueError):
        acc.update(data["e"][:10], data["ehat"][:10], data["codes"][:10], concept_mask=data["concept_mask"][:10])


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def test_fve_and_cosine_match_direct_computation():
    data = _synthetic()
    acc = _accumulator(data)
    out = fidelity_metrics(acc)

    e, ehat = data["e"], data["ehat"]
    mean_e = e.mean(0)
    dev2 = float(((e - mean_e) ** 2).sum())
    err2 = float(((e - ehat) ** 2).sum())
    assert out["fve"] == pytest.approx(1.0 - err2 / dev2, rel=1e-8)

    ec, ehc = e - mean_e, ehat - ehat.mean(0)
    cos = float((ec * ehc).sum()) / (np.linalg.norm(ec) * np.linalg.norm(ehc))
    assert out["recon_cosine"] == pytest.approx(cos, rel=1e-8)

    assert out["recon_mse"] == pytest.approx(err2 / e.size, rel=1e-8)
    pixel_cos = (e * ehat).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(ehat, axis=1))
    assert out["pixel_cosine"] == pytest.approx(float(pixel_cos.mean()), rel=1e-8)

    assert out["l0"] == pytest.approx((data["codes"] > 0).sum() / data["codes"].shape[0])
    assert out["dead_frac"] == pytest.approx(1.0 / F)  # the last feature is planted dead


# ---------------------------------------------------------------------------
# Dictionary geometry
# ---------------------------------------------------------------------------


def test_dictionary_geometry_flags_planted_duplicate():
    decoder = np.eye(D)[: F - 1]
    decoder = np.vstack([decoder, decoder[0] * 1.0])  # atom F-1 duplicates atom 0
    data = _synthetic()
    acc = _accumulator(data)
    out = dictionary_geometry_metrics(decoder, acc, dup_thresh=0.9)
    assert out["max_offdiag_coherence"] == pytest.approx(1.0, abs=1e-8)
    assert out["near_duplicate_count"] == 1.0
    assert out["dead_frac"] == pytest.approx(1.0 / F)


def test_orthogonal_dictionary_has_zero_coherence():
    decoder = np.eye(D)[:F] if F <= D else np.eye(D)
    data = _synthetic()
    acc = _accumulator(data)
    out = dictionary_geometry_metrics(decoder[: acc.n_features] if decoder.shape[0] >= acc.n_features else decoder, acc)
    assert out["dead_frac"] == pytest.approx(1.0 / F)


# ---------------------------------------------------------------------------
# Known concepts — contingency + AUROC-from-histogram
# ---------------------------------------------------------------------------


def test_contingency_tables_match_dense_precision():
    data = _synthetic()
    acc = _accumulator(data)
    precision, support, prevalence = contingency_tables(acc)

    active = data["codes"] > 0
    dense_support = active.sum(0).astype(np.float64)
    dense_precision = (active.T.astype(np.float64) @ data["concept_mask"]) / np.maximum(dense_support, 1)[:, None]
    assert support == pytest.approx(dense_support)
    assert precision == pytest.approx(dense_precision)
    assert prevalence == pytest.approx(data["concept_mask"].sum(0) / data["product_valid"].sum(0).max())


def test_concept_perplexity_is_one_for_a_single_concept_feature():
    n_fc = np.array([[10.0, 0.0, 0.0], [4.0, 4.0, 0.0]])
    n_f = np.array([10.0, 8.0])
    out = concept_perplexity(n_fc, n_f)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(2.0)


def test_auroc_from_histogram_matches_exact_ranking_and_reports_bin_error():
    """Binned AUROC should track the exact rank statistic; bigger B, smaller error."""
    rng = np.random.default_rng(3)
    n = 3000
    pos = rng.normal(1.2, 1.0, size=n // 2)
    neg = rng.normal(0.0, 1.0, size=n // 2)
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    exact = float(auc_all_features(values[:, None], labels)[0])

    errors = {}
    for b in (8, 64, 256):
        edges = np.quantile(values[values > -np.inf], np.linspace(0, 1, b + 1)[1:-1])
        bin_idx = np.searchsorted(edges, values, side="right")
        pos_hist = np.bincount(bin_idx[labels == 1], minlength=b).astype(np.float64)
        neg_hist = np.bincount(bin_idx[labels == 0], minlength=b).astype(np.float64)
        binned = float(auc_from_histograms(pos_hist, neg_hist))
        errors[b] = abs(binned - exact)

    # more bins should not make the estimate meaningfully worse
    assert errors[256] <= errors[8] + 1e-6
    assert errors[256] < 0.01


def test_auroc_matrix_from_accumulator_matches_exact_per_concept():
    data = _synthetic(seed=7, n=1200)
    acc = _accumulator(data, n_bins=128)
    binned = auroc_matrix_from_accumulator(acc)

    for c in range(C):
        prod = data["product_of_concept"][c]
        labelled = data["product_valid"][:, prod]
        pos = data["concept_mask"][:, c]
        rows = labelled & (pos | ~pos)  # every product-labelled row is either pos or neg
        y = pos[rows].astype(np.int64)
        if y.sum() == 0 or (len(y) - y.sum()) == 0:
            continue
        exact = auc_all_features(data["codes"][rows], y)
        # bin resolution introduces a small, bounded deviation
        assert np.max(np.abs(binned[:, c] - exact)) < 0.03


# ---------------------------------------------------------------------------
# Bin calibration
# ---------------------------------------------------------------------------


def test_calibrate_activation_bins_handles_a_dead_feature():
    codes = np.zeros((100, 3))
    codes[:, 0] = np.abs(np.random.default_rng(0).normal(size=100))
    edges = calibrate_activation_bins(codes, n_bins=8)
    assert edges.shape == (3, 7)
    assert np.all(np.diff(edges, axis=1) >= 0)
    assert np.all(edges > 0)  # a value of exactly 0 must always land in bin 0

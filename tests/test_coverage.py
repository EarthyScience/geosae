"""Black-box tests for geosae.eval.coverage — the AUROC-matrix + UCov/EMF panel.

Contracts asserted (not implementation):
  * folded AUROC is direction-invariant and lands in [0.5, 1]; a perfectly
    concept-separating feature scores 1.0; pure noise scores ~0.5.
  * balanced subsampling is exactly 50/50 and honours the caps / min-positive.
  * UCov counts concepts, EMF counts features; both are monotone in the
    threshold in the expected direction; a one-feature-per-concept oracle gives
    UCov == n_concepts and EMF == n_features.
  * the width-fair reductions: the best-of-M null grows with M, calibrated
    coverage does not, the block null is wider than the i.i.d. one on clumped
    labels, and the matched-budget subset is the deterministic top-usage set.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import coverage as CV


def test_auc_folding_and_range():
    rng = np.random.default_rng(0)
    n = 400
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    # f0: perfectly separates (positives higher). f1: perfectly separates but
    # anti-correlated (positives lower) -> folds to 1.0. f2: pure noise ~0.5.
    f0 = np.where(y == 1, 1.0, 0.0) + 1e-3 * rng.standard_normal(n)
    f1 = np.where(y == 1, 0.0, 1.0) + 1e-3 * rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    feats = np.stack([f0, f1, f2], axis=1)
    auc = CV.auc_all_features(feats, y)
    assert auc.shape == (3,)
    assert auc.min() >= 0.5 - 1e-9 and auc.max() <= 1.0 + 1e-9
    assert auc[0] > 0.99 and auc[1] > 0.99  # folding makes anti-correlation identical
    assert abs(auc[2] - 0.5) < 0.1


def test_auc_degenerate_returns_half():
    feats = np.random.default_rng(1).standard_normal((50, 4))
    assert np.allclose(CV.auc_all_features(feats, np.zeros(50, dtype=int)), 0.5)
    assert np.allclose(CV.auc_all_features(feats, np.ones(50, dtype=int)), 0.5)


def test_balanced_subsample_is_5050_and_capped():
    labels = np.array([1] * 1000 + [2] * 1000 + [0] * 500)  # 0 = nodata
    idx, y = CV.balanced_binary_subsample(labels, class_code=1, max_per_side=300, seed=42)
    assert idx.size == 600 and int(y.sum()) == 300  # exactly balanced, capped
    # positives are class 1; negatives are labelled non-nodata (class 2), never 0.
    assert set(labels[idx[y == 1]].tolist()) == {1}
    assert 0 not in labels[idx[y == 0]].tolist()


def test_balanced_subsample_min_positive_guard():
    labels = np.array([1] * 10 + [2] * 1000)
    idx, y = CV.balanced_binary_subsample(labels, class_code=1, min_positive=50)
    assert idx.size == 0 and y.size == 0


def test_ucov_and_emf_oracle():
    # F == C block-diagonal oracle: feature j separates only concept j.
    n, F = 600, 6
    C = F
    rng = np.random.default_rng(3)
    class_table = np.stack([np.zeros(C, dtype=int), np.arange(1, C + 1)], axis=1)  # one product
    product_order = ["prod"]
    # labels: pixel i belongs to class (i % C) + 1
    lab = (np.arange(n) % C) + 1
    labels = {"prod": lab}
    codes = np.zeros((n, F), dtype=np.float32)
    for j in range(F):
        codes[lab == (j + 1), j] = 1.0  # feature j fires exactly on concept j
    codes += 1e-3 * rng.standard_normal(codes.shape)
    auc = CV.concept_auroc_matrix(codes, labels, class_table, product_order, max_per_side=200)
    # Every concept perfectly covered (each has its dedicated separating feature).
    assert CV.ucov(auc, tau=0.65) == C
    # Every feature aligns with >= 1 group -> all alive; EMF is a positive count of
    # features bounded above by the number of alive features.
    c2g: dict = {}  # empty map -> all singleton groups
    ng = CV.n_groups_per_feature(auc, class_table, product_order, c2g, tau=0.65)
    assert (ng >= 1).all()
    e = CV.emf(ng)
    assert e["n_alive"] == F
    assert 0.0 < e["emf"] <= F
    assert 0.0 < CV.emf_simple(ng) <= F


def test_ucov_monotone_in_threshold():
    rng = np.random.default_rng(4)
    F, C = 20, 15
    auc = 0.5 + 0.45 * rng.random((F, C))
    covs = [CV.ucov(auc, tau=t) for t in (0.55, 0.65, 0.75, 0.85)]
    assert covs == sorted(covs, reverse=True)  # fewer concepts covered as tau rises


def test_threshold_sweep_keys():
    rng = np.random.default_rng(5)
    F, C = 12, 9
    auc = 0.5 + 0.45 * rng.random((F, C))
    class_table = np.stack([np.zeros(C, dtype=int), np.arange(1, C + 1)], axis=1)
    sweep = CV.threshold_sweep(auc, class_table, ["p"], {})
    assert set(sweep) == set(CV.THRESHOLD_SWEEP)
    for v in sweep.values():
        assert {"ucov", "emf", "emf_simple", "n_alive", "mean_n_groups"} <= set(v)


# ------------------------------------------------------- width-fair reductions
def _noise_panel(n_pixels: int, width: int, n_concepts: int, seed: int):
    """Pure-noise codes plus one balanced binary product per concept."""
    rng = np.random.default_rng(seed)
    codes = rng.standard_normal((n_pixels, width)).astype(np.float32)
    labels = {}
    for c in range(n_concepts):
        labels[f"p{c}"] = rng.integers(1, 3, size=n_pixels)
    class_table = np.stack([np.arange(n_concepts), np.ones(n_concepts, dtype=int)], axis=1)
    return codes, labels, class_table, [f"p{c}" for c in range(n_concepts)]


def test_iid_null_best_auroc_grows_with_dictionary_width():
    """The reviewer's critique as a unit test: best-of-M is inflated by M alone."""
    rng = np.random.default_rng(0)
    codes = rng.standard_normal((1000, 1024)).astype(np.float32)
    narrow = CV.permuted_best_auroc(codes[:, :64], n_per_side=150, n_perm=40, seed=1)
    wide = CV.permuted_best_auroc(codes, n_per_side=150, n_perm=40, seed=1)
    assert narrow.shape == (40,) and wide.shape == (40,)
    assert np.median(wide) > np.median(narrow) + 0.01
    assert narrow.min() >= 0.5


def _reference_permuted_best_auroc(codes, *, n_per_side, n_perm, seed):
    """The pre-vectorisation design: a fresh subsample *and* a fresh label
    permutation on every draw, ranked from scratch each time via
    :func:`CV.auc_all_features`. What ``permuted_best_auroc`` replaced, kept
    here only to check the replacement against it.
    """
    rng = np.random.default_rng(seed)
    n_rows = codes.shape[0]
    n_draw = 2 * n_per_side
    y = np.concatenate([np.ones(n_per_side, dtype=np.int64), np.zeros(n_per_side, dtype=np.int64)])
    out = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        idx = rng.choice(n_rows, size=n_draw, replace=False)
        out[i] = CV.auc_all_features(codes[idx], rng.permutation(y)).max()
    return out


def test_permuted_best_auroc_matches_the_per_permutation_resample_it_replaced():
    """The fixed-subsample, ranked-once design is a faster route to the same null.

    ``permuted_best_auroc`` used to redraw the subsample and re-rank it on
    every permutation; it now ranks one fixed subsample once and only permutes
    labels over it (see its docstring). Both are valid permutation-null
    designs for the same null hypothesis (best-of-M AUROC under labels
    independent of the codes), so their draws should be statistically
    indistinguishable -- checked here at a dictionary width large enough to be
    the real cost driver (:math:`M = 512`, a stand-in for Copernicus-FM's own
    :math:`M = 1024`) by comparing summary quantiles at a tolerance well
    inside run-to-run seed spread, not by matching individual draws. Kept
    small enough that the slow reference this compares against still runs in
    the suite's budget.
    """
    rng = np.random.default_rng(11)
    codes = rng.standard_normal((400, 512)).astype(np.float32)
    fast = CV.permuted_best_auroc(codes, n_per_side=60, n_perm=150, seed=2)
    slow = _reference_permuted_best_auroc(codes, n_per_side=60, n_perm=150, seed=2)
    assert fast.shape == slow.shape == (150,)
    for q in (0.5, 0.9, 0.99):
        assert np.quantile(fast, q) == pytest.approx(np.quantile(slow, q), abs=0.02)
    assert np.mean(fast) == pytest.approx(np.mean(slow), abs=0.01)


def test_concept_best_auroc_masks_invalid_concepts():
    auc = np.array([[0.9, 0.6, 0.7], [0.5, 0.8, 0.5]])
    best = CV.concept_best_auroc(auc)
    assert np.allclose(best, [0.9, 0.8, 0.7])
    masked = CV.concept_best_auroc(auc, valid=np.array([True, False, True]))
    assert np.isnan(masked[1]) and np.allclose(masked[[0, 2]], [0.9, 0.7])


def test_calibrated_coverage_is_bounded_and_near_alpha_under_noise():
    codes, labels, class_table, order = _noise_panel(600, 128, 60, seed=2)
    auc = CV.concept_auroc_matrix(codes, labels, class_table, order, max_per_side=60, min_positive=50)
    best = CV.concept_best_auroc(auc)
    null = CV.permuted_best_auroc(codes, n_per_side=60, n_perm=200, seed=3)
    out = CV.calibrated_coverage(best, null, alpha=0.05)
    assert 0.0 <= out["ccov"] <= 1.0
    assert out["n_valid"] == 60 and out["ccov_count"] == out["ccov"] * out["n_valid"]
    assert out["ccov"] < 0.25  # loose: expected 0.05, binomial noise at C = 60
    assert abs(out["mean_excess"]) < 0.05


def test_calibrated_coverage_is_one_for_planted_concepts():
    rng = np.random.default_rng(4)
    n, F, C = 600, 16, 8
    labels = {}
    codes = rng.standard_normal((n, F)).astype(np.float32) * 0.05
    class_table = np.stack([np.arange(C), np.ones(C, dtype=int)], axis=1)
    for c in range(C):
        member = rng.random(n) < 0.5
        labels[f"p{c}"] = np.where(member, 1, 2)
        codes[member, c] += 5.0  # feature c is a near-perfect detector of concept c
    order = [f"p{c}" for c in range(C)]
    auc = CV.concept_auroc_matrix(codes, labels, class_table, order, max_per_side=100, min_positive=50)
    best = CV.concept_best_auroc(auc)
    null = CV.permuted_best_auroc(codes, n_per_side=100, n_perm=100, seed=5)
    out = CV.calibrated_coverage(best, null, alpha=0.05)
    assert out["ccov"] == 1.0
    assert out["mean_excess"] > 0.3


def test_calibrated_coverage_is_flat_in_width_while_ucov_is_not():
    """Raw UCov rises with M on pure noise; the calibrated version does not."""
    got = {}
    for width in (64, 512):
        codes, labels, class_table, order = _noise_panel(600, width, 40, seed=6)
        auc = CV.concept_auroc_matrix(codes, labels, class_table, order, max_per_side=60, min_positive=50)
        null = CV.permuted_best_auroc(codes, n_per_side=60, n_perm=200, seed=7)
        cal = CV.calibrated_coverage(CV.concept_best_auroc(auc), null, alpha=0.05)
        got[width] = (CV.ucov(auc, tau=0.65), cal["ccov"])
    assert got[512][0] > got[64][0] + 5  # UCov inflated by width alone
    assert abs(got[512][1] - got[64][1]) < 0.2  # calibrated coverage is flat
    assert got[512][1] < 0.3 and got[64][1] < 0.3


def _clumped_world(seed: int = 8, n_cells: int = 40, per_cell: int = 30, width: int = 24):
    rng = np.random.default_rng(seed)
    cell_effect = rng.standard_normal((n_cells, width))
    codes = np.repeat(cell_effect, per_cell, axis=0) + 0.1 * rng.standard_normal((n_cells * per_cell, width))
    cell = np.repeat(np.arange(n_cells), per_cell)
    binary = (cell < n_cells // 2).astype(np.int64)  # whole cells are positive
    return codes.astype(np.float32), cell, binary


def test_block_null_rejects_ragged_cell_layout():
    codes, cell, binary = _clumped_world()
    ragged = cell.copy()
    ragged[:5] = 999  # one cell now contributes 5 pixels, another 25
    with pytest.raises(ValueError, match="pixels per cell"):
        CV.block_permuted_best_auroc(codes, ragged, binary, n_perm=2)


def test_block_null_exceeds_iid_null_on_spatially_clumped_labels():
    """The i.i.d. null is anti-conservative when concepts are spatially clumped."""
    codes, cell, binary = _clumped_world()
    block = CV.block_permuted_best_auroc(codes, cell, binary, n_perm=30, seed=9, max_per_side=200)
    iid = CV.permuted_best_auroc(codes, n_per_side=200, n_perm=30, seed=9)
    assert block.shape == (30,) and iid.shape == (30,)
    assert np.median(block) > np.median(iid) + 0.05


def test_matched_budget_subset_picks_the_most_used_features():
    usage = np.array([0.1, 0.9, 0.4, 0.9, 0.0])
    idx = CV.matched_budget_subset(usage, 3)
    assert idx.tolist() == [1, 2, 3]  # ties (features 1 and 3) broken by index
    assert CV.matched_budget_subset(usage, 3).tolist() == idx.tolist()  # deterministic


def test_matched_budget_subset_returns_all_indices_when_budget_exceeds_width():
    usage = np.array([0.3, 0.2, 0.5])
    assert CV.matched_budget_subset(usage, 3).tolist() == [0, 1, 2]
    assert CV.matched_budget_subset(usage, 99).tolist() == [0, 1, 2]


def test_code_activation_rate_counts_nonzero_rows():
    codes = np.array([[0.0, 1.0, -2.0], [0.0, 0.0, 3.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    assert CV.code_activation_rate(codes).tolist() == [0.0, 0.5, 0.5]


def test_coverage_report_bundles_the_panel_and_its_null():
    rng = np.random.default_rng(10)
    n, F, C = 500, 12, 6
    labels, codes = {}, rng.standard_normal((n, F)).astype(np.float32) * 0.05
    class_table = np.stack([np.arange(C), np.ones(C, dtype=int)], axis=1)
    for c in range(C):
        member = rng.random(n) < 0.5
        labels[f"p{c}"] = np.where(member, 1, 2)
        codes[member, c] += 5.0
    scalars, arrays = CV.coverage_report(
        codes, labels, class_table, [f"p{c}" for c in range(C)], class_to_group={}, max_per_side=100, n_perm=50
    )
    assert arrays["auc_matrix"].shape == (F, C)
    assert arrays["best_auroc"].shape == (C,) and arrays["null_best"].shape == (C, 50)
    assert arrays["activation_rate"].shape == (F,) and arrays["valid_concepts"].sum() == C
    assert scalars["ucov"] == C and scalars["ccov"] == 1.0
    assert scalars["null_n_per_side"] == 100
    assert all(isinstance(v, (int, float)) for v in scalars.values())
    assert "ucov_tau0.85" in scalars


def test_size_matched_null_widens_for_smaller_concepts():
    """A concept scored on few pixels needs a wider null than one scored on many.

    One null at the median sample size would under-state the best-of-M spread for
    the small concepts and so declare them recovered too readily.
    """
    rng = np.random.default_rng(3)
    codes = rng.standard_normal((4000, 128)).astype(np.float32)
    sizes = np.array([40, 40, 1500, 1500, 0])
    null, level = CV.size_matched_null(codes, sizes, n_perm=40, n_levels=2, seed=0)

    assert null.shape == (5, 40)
    assert np.isnan(null[4]).all() and level[4] == 0  # unscored concept has no null
    assert level[0] == level[1] == 40 and level[2] == level[3] == 1500
    assert np.median(null[0]) > np.median(null[2])  # small sample => wider null
    assert (null[0] == null[1]).all()  # concepts at one level share the draws


def test_effect_size_coverage_discriminates_where_significance_saturates():
    """The significance form saturates on a large sample; the margin form does not.

    Every concept here is *weakly* but genuinely aligned, so at thousands of
    pixels each clears its null and ``alpha``-thresholded coverage reads 1.0 —
    the failure mode that makes an effect-size bar the reduction to prefer.
    """
    rng = np.random.default_rng(11)
    C, n_perm = 40, 200
    null = rng.normal(0.52, 0.01, size=(C, n_perm))
    weak = np.full(C, 0.58)  # real, but far under UCov's 0.65 bar

    significance = CV.calibrated_coverage(weak, null, alpha=0.05)
    effect_size = CV.calibrated_coverage(weak, null, alpha=0.5, margin=0.15)
    assert significance["ccov"] == 1.0
    assert effect_size["ccov"] == 0.0
    assert 0.05 < significance["mean_excess"] < 0.07  # ~0.58 - 0.52

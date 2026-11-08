"""Black-box tests for geosae.eval.aoa (unweighted + ridge variants; no sklearn).

Contracts:
  * closed-form Ridge recovers a known linear map (and matches sklearn if present).
  * the kNN DI is ~1 in-distribution (median of train distances) and larger for
    out-of-distribution test points.
  * on data whose error grows with distance from the training region, the AoA
    Spearman rho is strongly positive; degenerate folds return {}.
  * continent / PFT split assignment is correct and the fold generators are well formed.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import aoa


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    n, d = 500, 5
    x = rng.standard_normal((n, d))
    true_w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = x @ true_w + 0.7 + 0.01 * rng.standard_normal(n)
    coef, intercept = aoa.ridge_fit(x, y, alpha=1e-6)
    assert np.allclose(coef, true_w, atol=0.05)
    assert abs(intercept - 0.7) < 0.05


def test_ridge_matches_sklearn_if_available():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 8))
    y = rng.standard_normal(300)
    coef, intercept = aoa.ridge_fit(x, y, alpha=1.0)
    m = sk.Ridge(alpha=1.0).fit(x, y)
    assert np.allclose(coef, m.coef_, atol=1e-8)
    assert abs(intercept - m.intercept_) < 1e-8


def test_knn_distance_in_vs_out_of_distribution():
    rng = np.random.default_rng(2)
    train = rng.standard_normal((400, 3))  # cluster at origin
    test_in = rng.standard_normal((50, 3))  # same distribution
    test_out = rng.standard_normal((50, 3)) + 20.0  # far away
    _, d_in, _ = aoa.knn_distance(train, test_in, k=5)
    d_tr, d_out, norm = aoa.knn_distance(train, test_out, k=5)
    assert np.isfinite(norm) and norm > 0
    assert abs(np.median(d_tr) - 1.0) < 1e-9  # median-normalized by construction
    assert d_out.mean() > 10 * d_in.mean()  # far test points are flagged as OOD


def test_score_fold_positive_rho_and_keys():
    # 1-D geographic gradient: target and error both grow with x; training is the
    # low-x region, so held-out high-x pixels are both far (high DI) and wrong.
    rng = np.random.default_rng(3)
    n = 3000
    pos = rng.uniform(0.0, 10.0, n)
    feats = np.stack([pos, 0.01 * rng.standard_normal(n)], axis=1)
    # convex target -> a linear Ridge fit on the low-x region underfits more and
    # more as x grows, so |error| rises monotonically with distance-from-train.
    y = pos**2 + 0.05 * rng.standard_normal(n)
    train_mask = pos < 3.0
    test_mask = pos >= 3.0
    rec = aoa.score_fold(feats, y, train_mask, test_mask, k=5, variants=("unweighted", "ridge"))
    assert rec, "fold should be scorable"
    assert {"rho_unweighted", "rho_ridge", "d_test_unweighted", "err", "n_train", "n_test"} <= set(rec)
    assert rec["rho_unweighted"] > 0.3  # distance-to-train tracks held-out error


def test_score_fold_decision_statistics():
    # Same gradient, with half the held-out side inside the training range: the
    # interval calibrated on the training side covers that half and misses the far
    # pixels, and a distance that tracks the error retains mostly the covered half.
    rng = np.random.default_rng(3)
    n = 4000
    pos = rng.uniform(0.0, 10.0, n)
    feats = np.stack([pos, 0.01 * rng.standard_normal(n)], axis=1)
    y = pos**2 + 0.05 * rng.standard_normal(n)
    train_mask = (pos < 5.0) & (rng.random(n) < 0.5)
    test_mask = ~train_mask
    base = aoa.score_fold(feats, y, train_mask, test_mask, variants=("unweighted",))
    rec = aoa.score_fold(feats, y, train_mask, test_mask, variants=("unweighted",), n_calibration=300)
    assert rec["n_train"] == base["n_train"] - 300 and rec["n_test"] == base["n_test"]
    assert rec["aurc_gain_unweighted"] > 0.5
    assert rec["coverage_all"] < 0.6 and rec["coverage_unweighted"] > 0.85
    assert rec["area_unweighted"] == pytest.approx(0.3, abs=0.06)
    kept = rec["d_test_unweighted"] <= rec["threshold_unweighted"]
    assert kept.mean() == rec["area_unweighted"]
    assert (rec["err"][kept] <= rec["halfwidth"]).mean() == rec["coverage_unweighted"]
    assert "aurc_gain_unweighted" not in base  # the statistics need a calibration side


def test_score_fold_scores_one_space_against_another_spaces_head():
    # The index lives in one space, the deployed head in another: the error must be
    # the head's own, the distances the index space's own.
    rng = np.random.default_rng(7)
    head_x = rng.standard_normal((3000, 4))
    index_x = np.column_stack([head_x[:, :2] ** 2, rng.standard_normal(3000)])
    y = head_x @ np.array([1.0, -1.0, 0.5, 0.0]) + 0.1 * rng.standard_normal(3000)
    train = head_x[:, 0] < 0.5
    kw = dict(variants=("unweighted", "ridge"), n_calibration=300)
    own_head = aoa.score_fold(head_x, y, train, ~train, **kw)
    own_index = aoa.score_fold(index_x, y, train, ~train, **kw)
    mixed = aoa.score_fold(index_x, y, train, ~train, probe_features=head_x, **kw)
    assert np.allclose(mixed["err"], own_head["err"]) and mixed["r2_test"] == own_head["r2_test"]
    for v in ("unweighted", "ridge"):
        assert np.allclose(mixed[f"d_test_{v}"], own_index[f"d_test_{v}"])


def test_score_fold_conformal_covers_in_domain():
    rng = np.random.default_rng(6)
    feats = rng.standard_normal((6000, 4))
    y = feats @ np.array([1.0, -1.0, 0.5, 0.0]) + rng.standard_normal(6000)
    train_mask = rng.random(6000) < 0.5
    rec = aoa.score_fold(feats, y, train_mask, ~train_mask, variants=("unweighted",), n_calibration=1000)
    assert rec["coverage_all"] == pytest.approx(0.9, abs=0.02)
    assert rec["area_unweighted"] == pytest.approx(0.9, abs=0.03)


def _novel_atom_fold(n: int = 4000, seed: int = 8):
    # A one-hot-plus-background sparse code: far pixels fire atoms the training side
    # never used, and a head fitted in domain errs on exactly those pixels.
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 10.0, n)
    code = np.zeros((n, 12))
    code[np.arange(n), np.minimum(pos.astype(int), 9)] = 1.0 + 0.1 * rng.random(n)
    code[:, 10] = rng.random(n)  # a dense background atom, always on
    y = pos**2 + 0.05 * rng.standard_normal(n)
    train = (pos < 5.0) & (rng.random(n) < 0.6)
    return code, y, train, ~train


def test_score_fold_scores_support_novelty_as_a_calibrated_variant():
    code, y, train, test = _novel_atom_fold()
    rec = aoa.score_fold(code, y, train, test, variants=("support", "support_probe"), n_calibration=300)
    for v in ("support", "support_probe"):
        assert {f"aurc_gain_{v}", f"coverage_{v}", f"area_{v}", f"rho_{v}"} <= set(rec)
        kept = rec[f"d_test_{v}"] <= rec[f"threshold_{v}"]
        assert kept.mean() == rec[f"area_{v}"]
    # far pixels use unseen atoms and carry the error: novelty ranks them last
    assert rec["aurc_gain_support"] > 0.5
    assert rec["coverage_support"] > 0.85 and rec["coverage_all"] < 0.7


def test_score_fold_support_novelty_matches_the_standalone_score():
    code, y, train, test = _novel_atom_fold()
    rec = aoa.score_fold(code, y, train, test, variants=("support",))
    expect = aoa.support_novelty(code[rec["test_idx"]], aoa.support_stats(code[train]))
    assert np.allclose(rec["d_test_support"], expect)


def test_support_novelty_is_uninformative_on_a_dense_space():
    # Every coordinate of a dense embedding is always active, so novelty is constant
    # and its ranking gain is exactly that of a random order.
    rng = np.random.default_rng(9)
    feats = rng.standard_normal((3000, 4))
    y = feats[:, 0] ** 2
    train = feats[:, 0] < 0.5
    rec = aoa.score_fold(feats, y, train, ~train, variants=("support",), n_calibration=300)
    assert np.ptp(rec["d_test_support"]) == 0
    assert rec["aurc_gain_support"] == pytest.approx(0.0, abs=1e-12)
    assert rec["area_support"] == 1.0


def test_score_fold_rejects_tiny_folds():
    feats = np.random.default_rng(4).standard_normal((100, 3))
    y = np.random.default_rng(5).standard_normal(100)
    mask = np.zeros(100, dtype=bool)
    mask[:10] = True
    assert aoa.score_fold(feats, y, mask, ~mask, min_side=200) == {}


def test_assign_continent_known_points():
    # (lon, lat): Berlin ~ Europe, a Sahara point ~ Africa, mid-Pacific ~ none.
    lonlat = np.array([[13.4, 52.5], [10.0, 20.0], [-140.0, 0.0]])
    cont = aoa.assign_continent(lonlat)
    names = list(aoa.CONTINENT_BBOXES)
    assert names[cont[0]] == "Europe"
    assert names[cont[1]] == "Africa"
    assert cont[2] == -1


def test_fold_generators_shapes():
    rng = np.random.default_rng(6)
    lonlat = np.column_stack([rng.uniform(-25, 45, 5000), rng.uniform(35, 72, 5000)])  # all Europe
    folds = list(aoa.continent_folds(lonlat, leave_one_out=False, min_train=100))
    # Only Europe is populated -> train-on-Europe has empty test -> no fold yielded.
    assert all(tr.sum() >= 100 for _, tr, _ in folds)
    pft = np.full(3000, 12)  # all cropland
    pf = list(aoa.pft_folds(pft, min_train=100))
    assert all(name == "Cropland" for name, _, _ in pf)


def test_score_fold_scores_the_mahalanobis_distance_as_a_calibrated_variant():
    from geosae.eval.extrapolation import mahalanobis

    rng = np.random.default_rng(14)
    feats = rng.standard_normal((3000, 5))
    y = feats[:, 0] + (feats[:, 1] > 1.5) * 5.0 * feats[:, 1]  # error grows off the training range
    train = feats[:, 1] < 0.5
    rec = aoa.score_fold(feats, y, train, ~train, variants=("mahalanobis",))
    assert np.allclose(rec["d_test_mahalanobis"], mahalanobis(feats[train], feats[rec["test_idx"]]))
    cal = aoa.score_fold(feats, y, train, ~train, variants=("mahalanobis",), n_calibration=300)
    assert {"aurc_gain_mahalanobis", "coverage_mahalanobis", "area_mahalanobis"} <= set(cal)
    assert cal["aurc_gain_mahalanobis"] > 0.3


# --- canonical area of applicability (Meyer & Pebesma 2021; CAST trainDI / aoa) ---------------


def _cast_reference(train: np.ndarray, test: np.ndarray, folds: np.ndarray | None):
    """CAST's trainDI + aoa, written as slowly and literally as its R source."""
    n = len(train)
    d = np.sqrt(((train[:, None, :] - train[None, :, :]) ** 2).sum(-1))
    mean_dist = np.mean([d[i, np.arange(n) != i].mean() for i in range(n)])
    nn = []
    for i in range(n):
        allowed = (np.arange(n) != i) if folds is None else (folds != folds[i])
        nn.append(d[i, allowed].min())
    di_train = np.array(nn) / mean_dist
    q25, q75 = np.quantile(di_train, [0.25, 0.75])
    thr = min(q75 + 1.5 * (q75 - q25), di_train.max())
    di_test = np.sqrt(((test[:, None, :] - train[None, :, :]) ** 2).sum(-1)).min(1) / mean_dist
    return di_train, di_test, thr


@pytest.mark.parametrize("with_folds", [False, True])
def test_dissimilarity_index_matches_a_literal_cast_reference(with_folds):
    rng = np.random.default_rng(20)
    train, test = rng.standard_normal((120, 3)), rng.standard_normal((40, 3)) * 1.5
    folds = rng.integers(0, 5, 120) if with_folds else None
    got = aoa.dissimilarity_index(train, test, cv_folds=folds)
    want = _cast_reference(train, test, folds)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-9)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-9)
    assert got[2] == pytest.approx(want[2], rel=1e-9)


def test_dissimilarity_index_threshold_flags_few_of_an_exchangeable_test_set():
    # Test drawn from the training distribution: only the training DI's upper outliers'
    # share lies beyond the whisker, a few per cent. The median rule flags about half.
    rng = np.random.default_rng(21)
    train, test = rng.standard_normal((2000, 4)), rng.standard_normal((1000, 4))
    _, di_test, thr = aoa.dissimilarity_index(train, test)
    assert (di_test > thr).mean() < 0.05
    _, d_old, _ = aoa.knn_distance(train, test)
    assert (d_old > 1.0).mean() > 0.3


def test_dissimilarity_index_needs_folds_at_the_scale_of_clustering():
    # Training pixels come in tight clusters; the test set is new clusters from the same
    # process. Leave-one-out finds a same-cluster neighbour for every training pixel, so the
    # threshold measures within-cluster spacing and flags nearly all of the test set; folds
    # that hold out whole clusters measure the spacing between clusters and flag few.
    rng = np.random.default_rng(22)

    def clusters(n_clusters: int, size: int = 30) -> tuple[np.ndarray, np.ndarray]:
        centres = rng.standard_normal((n_clusters, 4))
        x = np.repeat(centres, size, axis=0) + 0.03 * rng.standard_normal((n_clusters * size, 4))
        return x, np.repeat(np.arange(n_clusters), size)

    train, cluster = clusters(150)
    test, _ = clusters(40)
    _, di_loo, thr_loo = aoa.dissimilarity_index(train, test)
    _, di_cv, thr_cv = aoa.dissimilarity_index(train, test, cv_folds=cluster % 10)
    assert (di_loo > thr_loo).mean() > 0.8
    assert (di_cv > thr_cv).mean() < 0.1


def test_dissimilarity_index_flags_a_far_test_set_and_caps_at_the_training_maximum():
    rng = np.random.default_rng(23)
    train = rng.standard_normal((300, 3))
    di_train, di_far, thr = aoa.dissimilarity_index(train, rng.standard_normal((50, 3)) + 20.0)
    assert (di_far > thr).all()
    assert thr <= di_train.max()
    # Two well-separated pairs: the whisker would exceed the largest training DI.
    tiny = np.array([[0.0], [0.1], [5.0], [5.1]])
    di_tiny, _, thr_tiny = aoa.dissimilarity_index(tiny, tiny)
    assert thr_tiny == pytest.approx(di_tiny.max())


def test_dissimilarity_index_rejects_folds_of_the_wrong_length_and_a_single_fold():
    x = np.zeros((10, 2)) + np.arange(10)[:, None]
    with pytest.raises(ValueError):
        aoa.dissimilarity_index(x, x, cv_folds=np.zeros(9, dtype=int))
    with pytest.raises(ValueError):
        aoa.dissimilarity_index(x, x, cv_folds=np.zeros(10, dtype=int))


def test_score_fold_canonical_threshold_changes_only_the_out_of_domain_share():
    rng = np.random.default_rng(24)
    feats = rng.standard_normal((4000, 4))
    y = feats @ np.array([1.0, -1.0, 0.5, 0.0]) + rng.standard_normal(4000)
    train = rng.random(4000) < 0.5
    kw = dict(variants=("unweighted", "ridge"), n_calibration=300)
    old = aoa.score_fold(feats, y, train, ~train, **kw)
    folds = rng.integers(0, 10, 4000)
    new = aoa.score_fold(feats, y, train, ~train, threshold_rule="cast", cv_folds=folds, **kw)
    for v in ("unweighted", "ridge"):
        for key in ("rho", "aurc_gain", "coverage", "area", "d_test"):
            np.testing.assert_array_equal(new[f"{key}_{v}"], old[f"{key}_{v}"])
        assert old[f"frac_out_{v}"] > 0.3 and new[f"frac_out_{v}"] < 0.05
        out = new[f"aoa_di_test_{v}"] > new[f"aoa_threshold_{v}"]
        assert out.mean() == new[f"frac_out_{v}"]


def test_score_fold_rejects_an_unknown_threshold_rule():
    rng = np.random.default_rng(25)
    feats, y = rng.standard_normal((600, 2)), rng.standard_normal(600)
    train = np.arange(600) < 400
    with pytest.raises(ValueError):
        aoa.score_fold(feats, y, train, ~train, threshold_rule="mean", min_side=100)


def test_aoa_threshold_boxplot_rule_matches_r_boxplot_stats():
    # R: fivenum(1:10) = 1 3 5.5 8 10; fence 8 + 1.5 * 5 = 15.5; boxplot.stats(1:10)$stats[5] = 10.
    assert aoa.aoa_threshold(np.arange(1.0, 11.0), rule="boxplot") == 10.0
    # R: x <- c(1:10, 100); fivenum hinges 3.5 and 8.5 (n4 = 3.5, averaged ranks); fence 16;
    # stats[5] is the largest observation inside it, 10, not the fence. quantile(type 7): 3.5, 8.5,
    # so current CAST returns the fence itself, 16.
    x = np.r_[np.arange(1.0, 11.0), 100.0]
    assert aoa.aoa_threshold(x, rule="boxplot") == 10.0
    assert aoa.aoa_threshold(x, rule="quantile") == pytest.approx(16.0)
    # odd length: x <- c(1:8, 30); fivenum hinges 3 and 7, fence 13, stats[5] = 8.
    odd = np.r_[np.arange(1.0, 9.0), 30.0]
    assert aoa.aoa_threshold(odd, rule="boxplot") == 8.0
    # NaNs are ignored by both rules, and the order of the input does not matter.
    shuffled = np.r_[np.nan, odd[::-1], np.nan]
    assert aoa.aoa_threshold(shuffled, rule="boxplot") == 8.0
    assert aoa.aoa_threshold(shuffled, rule="quantile") == aoa.aoa_threshold(odd, rule="quantile")
    assert np.isnan(aoa.aoa_threshold(np.array([np.nan])))
    with pytest.raises(ValueError):
        aoa.aoa_threshold(odd, rule="fence")

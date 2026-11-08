"""Black-box tests for geosae.eval.selective (risk-coverage and conformal selection).

Contracts:
  * the risk-coverage AUC equals a slow reference that averages over every
    tie-breaking order, and its gain is 1 for a perfect ranking, 0 for an
    uninformative one and negative for a reversed one;
  * the statistic depends on the score only through its ranking;
  * a weighted AUROC equals the AUROC of the rows repeated by their integer weights;
  * split-conformal selection covers at the target level when calibration and
    test are exchangeable, and a score that tracks the shift restores coverage
    on the retained pixels where an uninformative score does not.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from geosae.eval import selective


def _aurc_reference(score: np.ndarray, err: np.ndarray) -> float:
    """AURC averaged over every order of the tied scores (brute force)."""
    n = len(score)
    values = []
    for perm in itertools.permutations(range(n)):
        order = sorted(perm, key=lambda i: score[i])  # stable: ties keep ``perm`` order
        cum = np.cumsum(err[order])
        values.append(np.mean(cum / np.arange(1, n + 1)))
    return float(np.mean(values))


def test_aurc_matches_brute_force_with_ties():
    score = np.array([0.1, 0.5, 0.5, 0.2, 0.5, 0.9])
    err = np.array([1.0, 4.0, 0.0, 2.0, 1.0, 3.0])
    rc = selective.risk_coverage(score, err)
    assert rc.aurc == pytest.approx(_aurc_reference(score, err), rel=1e-12)


def test_gain_at_the_extremes():
    rng = np.random.default_rng(0)
    err = rng.exponential(size=5000)
    assert selective.risk_coverage(err, err).gain == pytest.approx(1.0)
    assert selective.risk_coverage(np.zeros_like(err), err).gain == pytest.approx(0.0, abs=1e-12)
    assert abs(selective.risk_coverage(rng.random(err.size), err).gain) < 0.05
    assert selective.risk_coverage(-err, err).gain < -0.9


def test_random_and_oracle_bounds():
    rng = np.random.default_rng(1)
    err = rng.exponential(size=400)
    rc = selective.risk_coverage(rng.random(400), err)
    assert rc.random == pytest.approx(err.mean())
    assert rc.oracle == pytest.approx(selective.risk_coverage(err, err).aurc)
    assert rc.oracle <= rc.aurc


def test_aurc_is_rank_invariant():
    rng = np.random.default_rng(2)
    score, err = rng.random(300), rng.exponential(size=300)
    a = selective.risk_coverage(score, err).aurc
    b = selective.risk_coverage(np.exp(5 * score) - 3, err).aurc
    assert a == pytest.approx(b, rel=1e-12)


def test_risk_coverage_curve_ends_at_the_mean():
    rng = np.random.default_rng(3)
    err = rng.exponential(size=50)
    cov, risk = selective.risk_coverage_curve(rng.random(50), err)
    assert cov[-1] == pytest.approx(1.0) and cov[0] == pytest.approx(1 / 50)
    assert risk[-1] == pytest.approx(err.mean())


def test_risk_coverage_rejects_bad_input():
    with pytest.raises(ValueError):
        selective.risk_coverage(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        selective.risk_coverage(np.array([0.0, np.nan]), np.ones(2))


def test_conformal_quantile_order_statistic():
    vals = np.arange(1.0, 20.0)  # n = 19, alpha = 0.1 -> ceil(20 * 0.9) = 18th smallest
    assert selective.conformal_quantile(vals, 0.1) == 18.0
    assert selective.conformal_quantile(np.arange(5.0), 0.1) == np.inf  # too few to certify


def test_conformal_selection_covers_when_exchangeable():
    rng = np.random.default_rng(4)
    covs, areas = [], []
    for _ in range(40):
        cal_err, test_err = rng.exponential(size=500), rng.exponential(size=2000)
        cal_s, test_s = rng.random(500), rng.random(2000)
        sel = selective.conformal_selection(cal_err, cal_s, test_err, test_s, alpha=0.1)
        covs.append(sel.coverage_all)
        areas.append(sel.area)
    assert np.mean(covs) == pytest.approx(0.9, abs=0.01)
    assert np.mean(areas) == pytest.approx(0.9, abs=0.01)


def test_conformal_selection_informative_score_restores_coverage():
    rng = np.random.default_rng(5)
    cal_s = rng.random(2000)
    cal_err = rng.exponential(size=2000)
    test_s = rng.random(4000) * 3.0  # two thirds of the test side lies beyond the calibration range
    test_err = rng.exponential(size=4000) * (1.0 + 3.0 * np.maximum(test_s - 1.0, 0.0))
    good = selective.conformal_selection(cal_err, cal_s, test_err, test_s, alpha=0.1)
    blind = selective.conformal_selection(cal_err, cal_s, test_err, rng.random(4000), alpha=0.1)
    assert good.coverage_all < 0.8
    assert good.coverage == pytest.approx(0.9, abs=0.03)
    assert good.area == pytest.approx(0.3, abs=0.03)
    assert blind.coverage == pytest.approx(blind.coverage_all, abs=0.03)


def test_conformal_selection_rejects_bad_alpha():
    with pytest.raises(ValueError):
        selective.conformal_selection(np.ones(10), np.ones(10), np.ones(5), np.ones(5), alpha=1.5)


def _auroc_reference(score, positive):
    pos, neg = score[positive], score[~positive]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def test_stratified_auroc_with_one_bin_is_the_plain_auroc():
    rng = np.random.default_rng(11)
    score = rng.integers(0, 5, 300).astype(float)
    positive = rng.random(300) < 0.3
    ref = rng.normal(size=300)
    assert selective.stratified_auroc(score, positive, ref, n_bins=1) == pytest.approx(
        _auroc_reference(score, positive)
    )


def test_stratified_auroc_removes_what_the_reference_already_explains():
    rng = np.random.default_rng(12)
    ref = rng.normal(size=20000)
    positive = rng.random(20000) < 1 / (1 + np.exp(-2 * ref))  # positives depend on ref only
    copy_of_ref = ref + 0.01 * rng.normal(size=20000)
    extra = positive + rng.normal(size=20000)  # carries information beyond ref
    assert selective.stratified_auroc(ref, positive, ref, n_bins=1) > 0.75
    assert selective.stratified_auroc(copy_of_ref, positive, ref, n_bins=50) == pytest.approx(0.5, abs=0.03)
    assert selective.stratified_auroc(extra, positive, ref, n_bins=50) > 0.7


def test_stratified_auroc_depends_on_the_score_only_through_its_ranking():
    rng = np.random.default_rng(13)
    score, ref = rng.normal(size=500), rng.normal(size=500)
    positive = rng.random(500) < 0.2
    a = selective.stratified_auroc(score, positive, ref)
    b = selective.stratified_auroc(np.exp(3 * score), positive, ref)
    assert a == pytest.approx(b)


def test_silent_share_is_one_minus_the_false_alarm_rate_for_an_uninformative_score():
    rng = np.random.default_rng(14)
    score = rng.normal(size=40000)
    positive = rng.random(40000) < 0.1
    assert selective.silent_share(score, positive, false_alarm=0.05) == pytest.approx(0.95, abs=0.01)
    assert selective.silent_share(score, positive, false_alarm=0.2) == pytest.approx(0.8, abs=0.015)


def test_silent_share_is_zero_when_every_positive_scores_above_every_negative():
    score = np.r_[np.arange(100.0), 1000 + np.arange(10.0)]
    positive = np.r_[np.zeros(100, bool), np.ones(10, bool)]
    assert selective.silent_share(score, positive) == 0.0
    assert selective.silent_share(-score, positive) == 1.0


def test_silent_share_rejects_a_missing_class():
    with pytest.raises(ValueError):
        selective.silent_share(np.arange(5.0), np.zeros(5, bool))
    with pytest.raises(ValueError):
        selective.silent_share(np.arange(5.0), np.ones(5, bool))


def test_weighted_auroc_with_integer_weights_equals_the_auroc_of_repeated_rows():
    rng = np.random.default_rng(15)
    score = rng.integers(0, 6, 200).astype(float)  # many ties
    positive = rng.random(200) < 0.4
    weight = rng.integers(0, 4, 200).astype(float)
    rep = np.repeat(np.arange(200), weight.astype(int))
    assert selective.auroc(score, positive, weight) == pytest.approx(_auroc_reference(score[rep], positive[rep]))


def test_auroc_with_a_matrix_of_weights_scores_each_row():
    rng = np.random.default_rng(16)
    score = rng.normal(size=300)
    positive = rng.random(300) < 0.3
    weights = rng.integers(0, 3, (4, 300)).astype(float)
    rows = selective.auroc(score, positive, weights)
    assert rows.shape == (4,)
    assert rows == pytest.approx([selective.auroc(score, positive, w) for w in weights])
    assert selective.auroc(score, positive, np.ones(300)) == pytest.approx(selective.auroc(score, positive))

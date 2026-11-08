"""Black-box tests for geosae.eval.extrapolation (per-unit range checks).

Contracts:
  * ``mess`` reproduces Elith et al.'s per-variable similarity: 100 at the training
    median, 0 at the training extremes, negative outside the range in units of the
    range, and the most dissimilar unit is the one pushed out of range;
  * the one-sided variant ignores values at or below the training maximum, as a
    sparse code's zeros are never novel;
  * ``coverage_gap`` is the share of the unlabelled rows beyond the labelled maximum:
    the firing rate for a unit the labels never activate, zero for one they cover;
  * both are equivariant under a permutation of the units;
  * ``best_unit`` picks the unit whose activation alone best separates a concept, reads
    activation one-sided, skips units firing on too few of its rows, and agrees with a
    pairwise-count AUROC.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import extrapolation


@pytest.fixture
def train():
    return np.random.default_rng(0).uniform(0.0, 1.0, size=(2001, 3))


def test_mess_closed_form(train):
    med = np.median(train, axis=0)
    lo, hi = train.min(0), train.max(0)
    test = np.stack([med, lo, hi - 0.0, lo - 0.5 * (hi - lo)])
    test[3, 1:] = med[1:]
    res = extrapolation.mess(train, test)
    np.testing.assert_allclose(res.per_unit[0], 100.0, atol=0.2)
    assert res.per_unit[1].min() == pytest.approx(0.0, abs=0.2)
    assert res.per_unit[2].min() == pytest.approx(0.0, abs=0.2)
    assert res.similarity[3] == pytest.approx(-50.0)
    assert res.most_dissimilar[3] == 0


def test_mess_one_sided_ignores_the_lower_side(train):
    test = np.array([[-5.0, 0.5, 0.5], [0.5, 0.5, 3.0]])
    res = extrapolation.mess(train, test, one_sided=True)
    assert res.similarity[0] == 100.0
    assert res.similarity[1] < 0
    assert res.most_dissimilar[1] == 2


def test_unit_never_active_in_the_labels_is_finite_and_most_dissimilar(train):
    labelled = np.c_[train, np.zeros(len(train))]
    test = np.array([[0.5, 0.5, 0.5, 0.01], [0.5, 0.5, 0.5, 0.0]])
    res = extrapolation.mess(labelled, test, one_sided=True)
    assert np.isfinite(res.similarity).all()
    assert res.most_dissimilar[0] == 3
    assert res.similarity[1] == 100.0


def test_coverage_gap_is_the_rate_beyond_the_labelled_maximum():
    rng = np.random.default_rng(1)
    labelled = np.zeros((500, 2))
    labelled[:, 1] = rng.uniform(0, 1, 500)
    world = np.zeros((1000, 2))
    world[:300, 0] = rng.uniform(0.1, 1, 300)  # unit 0: never active in the labels
    world[:, 1] = rng.uniform(0, 1, 1000) * labelled[:, 1].max()
    gap = extrapolation.coverage_gap(labelled, world)
    assert gap[0] == pytest.approx(0.3)
    assert gap[1] == 0.0


def test_permutation_equivariance(train):
    rng = np.random.default_rng(2)
    test = rng.uniform(-0.5, 1.5, size=(50, 3))
    perm = np.array([2, 0, 1])
    a = extrapolation.mess(train, test)
    b = extrapolation.mess(train[:, perm], test[:, perm])
    np.testing.assert_allclose(a.similarity, b.similarity)
    np.testing.assert_array_equal(perm[b.most_dissimilar], a.most_dissimilar)
    np.testing.assert_allclose(
        extrapolation.coverage_gap(train, test)[perm], extrapolation.coverage_gap(train[:, perm], test[:, perm])
    )


def test_shape_mismatch_raises(train):
    with pytest.raises(ValueError):
        extrapolation.mess(train, np.zeros((4, 2)))


# --- tail_probability: the rank form of the one-sided range check -------------------------


def test_tail_probability_counts_the_labelled_values_at_or_above():
    labelled = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 0.0]])
    test = np.array([[0.0, 2.5], [0.0, 3.5], [0.2, 0.5], [0.0, 0.0]])
    res = extrapolation.tail_probability(labelled, test)
    n = len(labelled)
    # 2.5 is exceeded by one labelled value; 3.5 by none; unit 0 never fired in the labels
    np.testing.assert_allclose(res.probability, [2 / (n + 1), 1 / (n + 1), 1 / (n + 1), 1.0])
    assert res.unit[0] == 1 and res.unit[1] == 1 and res.unit[2] == 0
    # an inactive row is never novel
    assert res.probability[3] == 1.0


def test_tail_probability_is_robust_to_one_leaky_labelled_value():
    rng = np.random.default_rng(3)
    labelled = np.zeros((1000, 2))
    labelled[:, 0] = rng.uniform(0, 1, 1000)
    labelled[0, 1] = 5.0  # one labelled pixel fires the novel unit strongly
    test = np.array([[0.5, 2.0]])
    assert extrapolation.mess(labelled, test, one_sided=True).similarity[0] == 100.0  # the max hides it
    res = extrapolation.tail_probability(labelled, test)
    assert res.probability[0] == pytest.approx(2 / 1001)
    assert res.unit[0] == 1


def test_tail_probability_is_valid_for_one_unit_under_exchangeability():
    rng = np.random.default_rng(4)
    v = rng.exponential(size=(6000, 1))
    p = extrapolation.tail_probability(v[:3000], v[3000:]).probability
    for level in (0.01, 0.05, 0.2):
        assert (p <= level).mean() <= level + 0.01


def test_tail_probability_permutation_equivariance(train):
    rng = np.random.default_rng(5)
    test = rng.uniform(0.0, 0.95, size=(40, 3))  # inside the range: no ties between units
    perm = np.array([1, 2, 0])
    a = extrapolation.tail_probability(train, test)
    b = extrapolation.tail_probability(train[:, perm], test[:, perm])
    np.testing.assert_allclose(a.probability, b.probability)
    np.testing.assert_array_equal(perm[b.unit], a.unit)


# --- mahalanobis: the combination check --------------------------------------------------


def test_mahalanobis_without_shrinkage_is_the_classical_distance():
    rng = np.random.default_rng(6)
    labelled = rng.normal(size=(4000, 4)) @ rng.normal(size=(4, 4))
    test = rng.normal(size=(20, 4))
    mu = labelled.mean(0)
    prec = np.linalg.inv(np.cov(labelled, rowvar=False))
    ref = np.einsum("ni,ij,nj->n", test - mu, prec, test - mu)
    np.testing.assert_allclose(extrapolation.mahalanobis(labelled, test, shrinkage=0.0), ref, rtol=1e-8)


def test_mahalanobis_is_rotation_and_scale_invariant():
    rng = np.random.default_rng(7)
    labelled, test = rng.normal(size=(500, 5)), rng.normal(size=(30, 5)) * 2
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    a = extrapolation.mahalanobis(labelled, test)
    b = extrapolation.mahalanobis(3.0 * labelled @ q, 3.0 * test @ q)
    np.testing.assert_allclose(a, b, rtol=1e-8)


def test_mahalanobis_sees_a_direction_the_labels_never_vary_in():
    rng = np.random.default_rng(8)
    labelled = np.c_[rng.normal(size=(2000, 3)), np.zeros(2000)]
    inside = np.array([[0.5, 0.0, 0.0, 0.0]])
    novel = np.array([[0.0, 0.0, 0.0, 0.5]])  # the same offset, along the direction the labels never vary in
    d = extrapolation.mahalanobis(labelled, np.r_[inside, novel])
    assert np.isfinite(d).all()
    assert d[1] > d[0]
    assert extrapolation.mahalanobis(labelled, labelled.mean(0, keepdims=True))[0] == pytest.approx(0.0, abs=1e-12)


def test_mahalanobis_rejects_negative_shrinkage(train):
    with pytest.raises(ValueError):
        extrapolation.mahalanobis(train, train[:3], shrinkage=-1.0)


def _slow_auroc(score, positive):
    p, n = score[positive], score[~positive]
    return ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (p.size * n.size)


def test_best_unit_finds_the_unit_that_marks_the_concept():
    rng = np.random.default_rng(1)
    positive = rng.random(400) < 0.2
    code = rng.exponential(1.0, size=(400, 6)) * (rng.random((400, 6)) < 0.3)
    code[:, 4] = np.where(positive, 1.0 + rng.random(400), 0.0)  # fires exactly on the concept
    unit, auroc = extrapolation.best_unit(code, positive)
    assert unit == 4
    assert auroc == pytest.approx(1.0)


def test_best_unit_auroc_agrees_with_a_pairwise_count():
    rng = np.random.default_rng(2)
    positive = rng.random(300) < 0.3
    code = np.maximum(rng.normal(size=(300, 5)) + 0.8 * positive[:, None] * rng.random(5), 0.0)
    unit, auroc = extrapolation.best_unit(code, positive, min_firing=0.0)
    slow = [_slow_auroc(code[:, j], positive) for j in range(5)]
    assert unit == int(np.argmax(slow))
    assert auroc == pytest.approx(max(slow))


def test_best_unit_reads_activation_one_sided_and_skips_rare_units():
    positive = np.r_[np.ones(50, bool), np.zeros(150, bool)]
    code = np.zeros((200, 3))
    code[50:, 0] = 1.0  # fires only off the concept: an AUROC below one half, never chosen
    code[:1, 1] = 5.0  # fires on one concept pixel only: below min_firing
    code[:40, 2] = 1.0  # fires on 80 % of the concept
    unit, auroc = extrapolation.best_unit(code, positive, min_firing=0.05)
    assert unit == 2
    assert auroc == pytest.approx(0.9)


def test_best_unit_without_a_candidate_returns_no_unit():
    positive = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    unit, auroc = extrapolation.best_unit(np.zeros((20, 4)), positive)
    assert unit == -1
    assert np.isnan(auroc)


def test_best_unit_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        extrapolation.best_unit(np.zeros((20, 4)), np.zeros(19, bool))

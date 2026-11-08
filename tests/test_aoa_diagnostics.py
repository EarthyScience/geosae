"""Black-box tests for the AoA diagnostic layer of :mod:`geosae.eval.aoa`.

Contracts:
  * feature scaling and the distance metric are honoured by the kNN DI, and the
    cosine metric is invariant to a per-row rescaling that Euclidean is not.
  * after a per-feature rescaling, the ridge-weighted DI does not depend on the
    units the features came in.
  * local point density falls as a query moves away from the reference cloud, and
    equals the reference count when the threshold swallows the whole cloud.
  * per-feature support statistics recover known in-domain frequencies, and the
    novelty scores rank a code using unsupported atoms above one that does not.
  * the DI-agreement summary reports an exact 2x2 threshold contingency and an
    overlay code in ``{0,1,2,3}``; identical inputs give rho 1 and no disagreement.
  * k-occurrence skewness is larger for a cloud with a planted hub than for a
    symmetric one.
  * the scenario runner emits one record per (scenario, target, space, variant)
    and its records carry the fold bookkeeping needed to aggregate.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import aoa

# --------------------------------------------------------------------------- scaling / metric


def test_cosine_metric_ignores_row_magnitude_but_euclidean_does_not():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((300, 8))
    test = rng.standard_normal((60, 8))
    scale = rng.uniform(0.5, 4.0, size=(60, 1))  # per-row magnitude only

    _, d_cos, _ = aoa.knn_distance(train, test, k=5, metric="cosine")
    _, d_cos_scaled, _ = aoa.knn_distance(train, test * scale, k=5, metric="cosine")
    assert np.allclose(d_cos, d_cos_scaled, atol=1e-8)

    _, d_euc, _ = aoa.knn_distance(train, test, k=5)
    _, d_euc_scaled, _ = aoa.knn_distance(train, test * scale, k=5)
    assert not np.allclose(d_euc, d_euc_scaled, atol=1e-3)


def test_feature_scaling_equalises_an_inflated_dimension():
    # Dimension 0 has 100x the spread of the rest, so unscaled Euclidean kNN is
    # essentially a 1-D distance along it; z-scaling must remove that dominance.
    rng = np.random.default_rng(1)
    train = rng.standard_normal((400, 6))
    train[:, 0] *= 100.0
    test = rng.standard_normal((80, 6))
    test[:, 0] *= 100.0

    plain = aoa.scale_features(train, test, mode="none")
    zscored = aoa.scale_features(train, test, mode="std")
    assert np.allclose(plain[0], train)
    assert np.std(zscored[0], axis=0).max() / np.std(zscored[0], axis=0).min() < 1.5


def test_scale_features_uses_train_statistics_only():
    rng = np.random.default_rng(2)
    train = rng.standard_normal((200, 4)) * np.array([1.0, 5.0, 0.2, 3.0])
    test = train[:50] * 7.0  # a wildly different test spread must not change the scaler
    tr_a, te_a = aoa.scale_features(train, test, mode="std")
    tr_b, te_b = aoa.scale_features(train, test * 100.0, mode="std")
    assert np.allclose(tr_a, tr_b)
    assert np.allclose(te_b, te_a * 100.0)


def test_active_scaling_uses_mean_active_magnitude_of_sparse_codes():
    # Two atoms: atom 0 fires rarely but hugely, atom 1 often and small. "active"
    # scaling divides by the mean magnitude *when active*, so both end up ~1.
    train = np.zeros((100, 2))
    train[:5, 0] = 50.0
    train[:, 1] = 0.5
    tr, _ = aoa.scale_features(train, train[:10], mode="active")
    active_means = np.array([tr[tr[:, j] > 0, j].mean() for j in range(2)])
    assert np.allclose(active_means, 1.0, atol=1e-8)


@pytest.mark.parametrize("scaling", ["std", "active"])
def test_ridge_weighted_distance_after_scaling_does_not_depend_on_feature_units(scaling):
    # Rescaled features are unit-free, so the ridge weights applied to them must be
    # the head's coefficients in the same units. Changing the unit of one input
    # column (x_j -> a x_j) then leaves the ridge-weighted distance unchanged.
    rng = np.random.default_rng(3)
    feats = rng.standard_normal((500, 5))
    y = feats @ np.array([2.0, -1.0, 0.5, 0.0, 3.0]) + 0.1 * rng.standard_normal(500)
    tr = np.zeros(500, dtype=bool)
    tr[:350] = True
    unit = np.array([1.0, 1000.0, 1.0, 0.001, 1.0])
    kw = dict(variants=("ridge",), scaling=scaling, alpha=1e-9, min_side=100)
    rec = aoa.score_fold(feats, y, tr, ~tr, **kw)
    rec_units = aoa.score_fold(feats * unit, y, tr, ~tr, **kw)
    assert np.allclose(rec["d_test_ridge"], rec_units["d_test_ridge"], rtol=1e-6)


def test_unknown_scaling_and_metric_raise():
    x = np.zeros((10, 3))
    with pytest.raises(ValueError):
        aoa.scale_features(x, x, mode="nope")
    with pytest.raises(ValueError):
        aoa.knn_distance(x, x, metric="nope")


# --------------------------------------------------------------------------- local point density


def test_local_point_density_decays_with_distance_from_the_cloud():
    rng = np.random.default_rng(3)
    train = rng.standard_normal((500, 3))
    near = rng.standard_normal((40, 3))
    far = rng.standard_normal((40, 3)) + 12.0
    _, lpd_near, _ = aoa.local_point_density(train, near, k=5, threshold=1.0)
    _, lpd_far, _ = aoa.local_point_density(train, far, k=5, threshold=1.0)
    assert lpd_near.mean() > lpd_far.mean()
    assert lpd_far.max() == 0


def test_local_point_density_saturates_at_the_reference_count():
    rng = np.random.default_rng(4)
    train = rng.standard_normal((200, 3))
    test = rng.standard_normal((20, 3))
    _, lpd, _ = aoa.local_point_density(train, test, k=5, threshold=1e6)
    assert np.all(lpd == len(train))


# --------------------------------------------------------------------------- support novelty


def test_support_stats_recover_known_activation_frequencies():
    code = np.zeros((100, 4))
    code[:50, 0] = 1.0  # fires on half the training set
    code[:10, 1] = 2.0  # fires on a tenth
    code[:, 2] = 0.3  # always on
    # atom 3 never fires
    st = aoa.support_stats(code)
    assert np.allclose(st.frequency, [0.5, 0.1, 1.0, 0.0])
    assert st.frequency.shape == (4,)


def test_support_novelty_flags_codes_using_unsupported_atoms():
    train = np.zeros((200, 5))
    train[:, :3] = 1.0  # atoms 0-2 are the in-domain support
    st = aoa.support_stats(train)
    in_domain = np.zeros((10, 5))
    in_domain[:, :3] = 1.0
    novel = np.zeros((10, 5))
    novel[:, 0] = 1.0
    novel[:, 3:] = 1.0  # atoms 3-4 never seen in training
    for mode in ("surprisal", "count"):
        s_in = aoa.support_novelty(in_domain, st, mode=mode)
        s_out = aoa.support_novelty(novel, st, mode=mode)
        assert s_out.mean() > s_in.mean(), mode


def test_support_novelty_weighting_follows_the_probe_coefficients():
    # Two unseen atoms; only atom 3 carries probe weight, so the weighted score
    # must separate a pixel that lights atom 3 from one that lights atom 4.
    train = np.zeros((100, 5))
    train[:, :3] = 1.0
    st = aoa.support_stats(train)
    coef = np.array([0.1, 0.1, 0.1, 10.0, 0.0])
    lights_weighted = np.zeros((5, 5))
    lights_weighted[:, 3] = 1.0
    lights_ignored = np.zeros((5, 5))
    lights_ignored[:, 4] = 1.0
    s_w = aoa.support_novelty(lights_weighted, st, mode="surprisal", weights=np.abs(coef))
    s_i = aoa.support_novelty(lights_ignored, st, mode="surprisal", weights=np.abs(coef))
    assert s_w.mean() > s_i.mean()
    # without weighting the two are indistinguishable
    s_w0 = aoa.support_novelty(lights_weighted, st, mode="surprisal")
    s_i0 = aoa.support_novelty(lights_ignored, st, mode="surprisal")
    assert np.allclose(s_w0, s_i0)


def test_support_novelty_rejects_unknown_mode():
    st = aoa.support_stats(np.ones((10, 3)))
    with pytest.raises(ValueError):
        aoa.support_novelty(np.ones((2, 3)), st, mode="nope")


# --------------------------------------------------------------------------- DI agreement


def test_di_agreement_of_a_field_with_itself_is_perfect():
    d = np.abs(np.random.default_rng(5).standard_normal(500)) + 0.5
    a = aoa.di_agreement(d, d)
    assert a["spearman"] == pytest.approx(1.0)
    assert a["n_only_a"] == 0 and a["n_only_b"] == 0
    assert a["mask_jaccard"] == pytest.approx(1.0)


def test_di_agreement_contingency_and_overlay_codes_are_exact():
    # threshold 1.0: (below, below), (below, above), (above, below), (above, above)
    a_d = np.array([0.5, 0.5, 2.0, 2.0])
    b_d = np.array([0.5, 2.0, 0.5, 2.0])
    ag = aoa.di_agreement(a_d, b_d, threshold=1.0)
    assert (ag["n_both_in"], ag["n_only_b"], ag["n_only_a"], ag["n_both_out"]) == (1, 1, 1, 1)
    assert set(np.unique(ag["overlay"])) <= {0, 1, 2, 3}
    assert len(ag["overlay"]) == 4
    assert ag["mask_jaccard"] == pytest.approx(1.0 / 3.0)


def test_di_agreement_reports_a_low_rank_correlation_for_unrelated_fields():
    rng = np.random.default_rng(6)
    a_d = rng.uniform(0.1, 3.0, 2000)
    b_d = rng.uniform(0.1, 3.0, 2000)
    assert abs(aoa.di_agreement(a_d, b_d)["spearman"]) < 0.1


# --------------------------------------------------------------------------- rank profile / atoms


def test_di_rank_profile_saturates_at_the_intrinsic_rank():
    # A cloud that lives on a 3-D subspace of a 12-D space: the DI must be
    # reproduced by rank 3 and gain nothing after it.
    rng = np.random.default_rng(9)
    basis = rng.standard_normal((3, 12))
    train = rng.standard_normal((400, 3)) @ basis
    test = rng.standard_normal((80, 3)) @ basis + 0.5
    prof = aoa.di_rank_profile(train, test, ranks=(1, 2, 3, 6, 12), k=5)
    by_rank = dict(zip(prof["ranks"], prof["spearman"], strict=True))
    assert by_rank[3] > 0.99
    assert by_rank[1] < by_rank[3]
    assert prof["rank_95"] is not None and prof["rank_95"] <= 3
    assert prof["explained_variance"][-1] == pytest.approx(1.0, abs=1e-6)


def test_di_rank_profile_skips_ranks_above_the_feature_count():
    rng = np.random.default_rng(10)
    prof = aoa.di_rank_profile(rng.standard_normal((80, 4)), rng.standard_normal((20, 4)), ranks=(1, 4, 99))
    assert prof["ranks"] == [1, 4]


def test_extrapolating_atoms_finds_the_probe_weighted_frequency_jump():
    # atom 1 fires far more often off-domain and carries a large probe weight;
    # atom 2 also jumps but the probe ignores it; atom 0 is heavy but stable.
    train = np.zeros((200, 4))
    train[:, 0] = 1.0
    train[:5, 1] = 1.0
    train[:5, 2] = 1.0
    test = np.zeros((100, 4))
    test[:, 0] = 1.0
    test[:, 1] = 1.0
    test[:, 2] = 1.0
    coef = np.array([9.0, 5.0, 0.0, 1.0])
    res = aoa.extrapolating_atoms(train, test, coef, top=3)
    assert res["index"][0] == 1
    assert res["freq_test"][0] > res["freq_train"][0]
    assert res["score"][0] > 0


# --------------------------------------------------------------------------- hubness


def test_hubness_skewness_is_larger_with_a_planted_hub():
    rng = np.random.default_rng(7)
    d = 3
    symmetric = rng.standard_normal((400, d))
    hubbed = rng.standard_normal((400, d)) * 8.0
    hubbed[0] = 0.0  # one point sits at the centre of a diffuse cloud
    query = rng.standard_normal((300, d))
    h_sym = aoa.hubness(symmetric, query, k=5)
    h_hub = aoa.hubness(hubbed, query, k=5)
    assert h_hub["k_occurrence_skewness"] > h_sym["k_occurrence_skewness"]
    assert h_sym["k_occurrence_max"] >= 1


# --------------------------------------------------------------------------- scenario runner


def _toy_sites(n=2400, seed=8):
    """A synthetic AoASites: two feature spaces over a global lon/lat cloud."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-170.0, 175.0, n)
    lat = rng.uniform(-50.0, 70.0, n)
    pft = rng.choice([1, 4, 10, 12], size=n)
    raw = np.column_stack([lat / 50.0, lon / 100.0, rng.standard_normal(n) * 0.1])
    code = np.column_stack([raw, rng.standard_normal((n, 2)) * 0.1])
    y = 2.0 * (lat / 50.0) ** 2 + 0.05 * rng.standard_normal(n)
    return aoa.AoASites(
        lonlat=np.column_stack([lon, lat]),
        pft=pft,
        spaces={"raw": raw, "sae": code},
        targets={"toy": y},
        cell=np.arange(n),
    )


def test_run_scenarios_emits_one_record_per_cell_of_the_grid():
    sites = _toy_sites()
    recs = aoa.run_scenarios(
        sites,
        scenarios=("single_continent",),
        variants=("unweighted",),
        k=5,
        max_train=600,
        max_test=600,
        min_train=100,
        min_test=100,
        min_side=50,
        verbose=False,
    )
    assert recs, "the toy sites should yield at least one fold"
    keys = {"scenario", "fold", "target", "space", "variant", "rho", "n_train", "n_test"}
    assert all(keys <= set(r) for r in recs)

    # every (fold, target) is scored in every space, so the spaces pair up exactly
    def cells(space):
        return {(r["scenario"], r["fold"], r["target"], r["variant"]) for r in recs if r["space"] == space}

    assert cells("raw") == cells("sae")


def test_run_folds_scores_graded_folds_with_decision_statistics():
    from geosae.eval import holdouts

    sites = _toy_sites()
    centres = holdouts.spread_centres(sites.lonlat, 2, min_separation=3000.0, seed=0)
    folds = list(holdouts.graded_folds(sites.lonlat, centres, gaps=(0.0, 2000.0), n_test=200, n_train=800))
    meta = {f.name: {"gap": f.gap} for f in folds}
    recs = aoa.run_folds(
        sites,
        [(f.name, f.train_mask, f.test_mask) for f in folds],
        scenario="graded_distance",
        fold_meta=meta,
        variants=("unweighted",),
        min_side=50,
        n_calibration=200,
        verbose=False,
    )
    assert len(recs) == len(folds) * 2  # one per fold and space
    for r in recs:
        assert r["gap"] == meta[r["fold"]]["gap"] and r["n_train"] == 600 and r["n_test"] == 200
        assert {"aurc", "aurc_gain", "coverage", "area", "coverage_all"} <= set(r)
        assert 0.0 <= r["area"] <= 1.0 and r["probe"] == r["space"]
    judged = aoa.run_folds(
        sites,
        [(f.name, f.train_mask, f.test_mask) for f in folds[:1]],
        scenario="graded_distance",
        variants=("unweighted",),
        min_side=50,
        n_calibration=200,
        probe_space="raw",
        verbose=False,
    )
    assert {r["probe"] for r in judged} == {"raw"}
    assert len({r["r2_test"] for r in judged}) == 1  # one head, judged by every space's index


def test_run_scenarios_rejects_an_unknown_scenario():
    with pytest.raises(ValueError):
        aoa.run_scenarios(_toy_sites(), scenarios=("nope",))


def test_scenario_folds_cover_the_four_declared_scenarios():
    sites = _toy_sites()
    for name in aoa.SPLIT_SCENARIOS:
        folds = list(aoa.scenario_folds(name, sites.lonlat, sites.pft, min_train=50, min_test=50))
        assert folds, f"{name} yielded no folds"
        for _, tr, te in folds:
            assert not (tr & te).any(), f"{name}: train and test overlap"


def test_support_novelty_is_undefined_rather_than_zero_on_a_dense_space():
    # A dense factor matrix has every coordinate always active, so the support
    # surprisal is identically 0 and its correlation with error is undefined.
    rng = np.random.default_rng(11)
    dense = rng.standard_normal((600, 5)) + 3.0  # no exact zeros
    y = dense @ np.array([1.0, -1.0, 0.5, 0.0, 2.0])
    train = np.zeros(600, dtype=bool)
    train[:300] = True
    rec = aoa.score_fold(dense, y, train, ~train, variants=("unweighted",), extra_signals=True, min_side=100)
    assert np.isnan(rec["rho_support"]), "a dense space has no support signal to measure"
    assert np.all(rec["d_test_support"] == 0.0)
    assert np.isfinite(rec["rho_unweighted"]), "the distance signal is still well defined"


# --------------------------------------------------------------------------- fold saturation


def test_score_fold_reports_the_out_of_domain_saturation_of_each_variant():
    # Training on a tight cluster and testing far away puts every held-out point
    # beyond the threshold, so the fold is saturated and carries no dynamic range.
    rng = np.random.default_rng(12)
    train = rng.standard_normal((400, 4))
    far = rng.standard_normal((200, 4)) + 25.0
    feats = np.vstack([train, far])
    y = feats[:, 0] * 2.0
    tr = np.zeros(600, dtype=bool)
    tr[:400] = True
    rec = aoa.score_fold(feats, y, tr, ~tr, variants=("unweighted",), min_side=100)
    assert rec["frac_out_unweighted"] == 1.0

    # A held-out set drawn from the same distribution is mostly inside it.
    same = rng.standard_normal((200, 4))
    feats2 = np.vstack([train, same])
    y2 = feats2[:, 0] * 2.0
    rec2 = aoa.score_fold(feats2, y2, tr, ~tr, variants=("unweighted",), min_side=100)
    assert 0.0 <= rec2["frac_out_unweighted"] < 0.9
    assert rec2["frac_out_unweighted"] < rec["frac_out_unweighted"]


def test_score_fold_reports_atoms_that_are_unseen_in_fold_but_fire_out_of_fold():
    # atoms 3 and 4 never fire on the training fold and do fire on the held-out one
    code = np.zeros((600, 5))
    code[:400, :3] = 1.0
    code[400:, 0] = 1.0
    code[400:, 3:] = 1.0
    y = np.arange(600, dtype=float)
    tr = np.zeros(600, dtype=bool)
    tr[:400] = True
    rec = aoa.score_fold(code, y, tr, ~tr, variants=("unweighted",), extra_signals=True, min_side=100)
    assert rec["n_atoms_unseen_in_fold"] == 2
    assert rec["frac_pixels_using_unseen_atom"] == 1.0


def test_run_scenarios_records_carry_the_saturation_fraction():
    recs = aoa.run_scenarios(
        _toy_sites(),
        scenarios=("single_continent",),
        variants=("unweighted",),
        max_train=600,
        max_test=600,
        min_train=100,
        min_test=100,
        min_side=50,
        verbose=False,
    )
    assert recs
    assert all(0.0 <= r["frac_out"] <= 1.0 for r in recs)

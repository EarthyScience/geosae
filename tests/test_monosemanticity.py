"""Black-box tests for geosae.eval.monosemanticity (pairwise-similarity monosemanticity scores).

Contracts:
  * both scores equal the slow double loop over sample pairs of the published definition;
  * a feature active on identical images / one place scores 1, and a feature spread
    uniformly over the samples scores the mean pairwise similarity of the sample;
  * the scores do not change under a positive affine rescaling of a feature;
  * features that are constant or active on too few samples are NaN and left out of
    the summary;
  * the geographic kernel is ``exp(-d / length_scale)`` of the great-circle distance.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import monosemanticity as ms
from geosae.eval.spatial import EARTH_RADIUS_KM


def _globe(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-180, 180, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))


def _slow_score(acts: np.ndarray, sim: np.ndarray) -> np.ndarray:
    lo, hi = acts.min(0), acts.max(0)
    a = (acts - lo) / (hi - lo)
    num, den = np.zeros(acts.shape[1]), np.zeros(acts.shape[1])
    for i in range(acts.shape[0]):
        for j in range(i + 1, acts.shape[0]):
            num += a[i] * a[j] * sim[i, j]
            den += a[i] * a[j]
    return num / den


def test_visual_score_matches_pairwise_reference():
    rng = np.random.default_rng(0)
    acts = np.maximum(rng.standard_normal((40, 6)), 0.0)
    emb = rng.standard_normal((40, 5))
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    got = ms.monosemanticity_score(acts, emb, min_active=0)
    np.testing.assert_allclose(got, _slow_score(acts, unit @ unit.T), rtol=1e-9)


def test_geo_score_matches_pairwise_reference():
    rng = np.random.default_rng(1)
    lon, lat = _globe(40, seed=1)
    acts = np.maximum(rng.standard_normal((40, 6)), 0.0)
    dist = ms.pairwise_great_circle_km(lon, lat)
    for chunk in (7, 4096):
        got = ms.geo_monosemanticity_score(acts, lon, lat, length_scale_km=500.0, min_active=0, chunk=chunk)
        np.testing.assert_allclose(got, _slow_score(acts, np.exp(-dist / 500.0)), rtol=1e-9)


def test_great_circle_known_distances():
    d = ms.pairwise_great_circle_km(np.array([0.0, 0.0, 180.0, 10.0]), np.array([0.0, 90.0, 0.0, 0.0]))
    quarter = np.pi * EARTH_RADIUS_KM / 2
    np.testing.assert_allclose(d[0], [0.0, quarter, 2 * quarter, quarter / 9], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(d, d.T)


def test_feature_on_identical_images_scores_one():
    rng = np.random.default_rng(2)
    emb = rng.standard_normal((60, 8))
    emb[:25] = emb[0]
    acts = np.zeros((60, 1))
    acts[:25, 0] = rng.uniform(0.5, 2.0, 25)
    assert ms.monosemanticity_score(acts, emb)[0] == pytest.approx(1.0)


def test_localised_feature_scores_high_and_scattered_feature_scores_baseline():
    rng = np.random.default_rng(3)
    lon, lat = _globe(600, seed=3)
    lon[:40], lat[:40] = 11.0 + rng.uniform(-0.05, 0.05, 40), 50.0 + rng.uniform(-0.05, 0.05, 40)
    acts = np.zeros((600, 2))
    acts[:40, 0] = 1.0  # one place, within ~10 km
    acts[40:, 1] = 1.0  # the whole globe
    score = ms.geo_monosemanticity_score(acts, lon, lat, length_scale_km=500.0)
    baseline = ms.mean_pairwise_similarity(lon=lon[40:], lat=lat[40:], length_scale_km=500.0)
    assert score[0] > 0.97
    assert score[1] == pytest.approx(baseline, rel=1e-9)
    assert baseline < 0.02


def test_random_features_score_the_mean_pairwise_similarity():
    rng = np.random.default_rng(4)
    emb = rng.standard_normal((400, 3)) + np.array([2.0, 0.0, 0.0])  # a non-zero mean similarity
    acts = rng.uniform(0.0, 1.0, (400, 300))
    score = ms.monosemanticity_score(acts, emb)
    baseline = ms.mean_pairwise_similarity(embeddings=emb)
    assert 0.3 < baseline < 0.9
    assert np.nanmean(score) == pytest.approx(baseline, abs=5e-3)


def test_invariant_to_positive_affine_rescaling():
    rng = np.random.default_rng(5)
    lon, lat = _globe(80, seed=5)
    acts = rng.standard_normal((80, 4))
    emb = rng.standard_normal((80, 6))
    scaled = acts * np.array([0.1, 3.0, 20.0, 1.0]) + np.array([5.0, -2.0, 0.0, 1.0])
    np.testing.assert_allclose(ms.monosemanticity_score(acts, emb), ms.monosemanticity_score(scaled, emb), rtol=1e-8)
    np.testing.assert_allclose(
        ms.geo_monosemanticity_score(acts, lon, lat),
        ms.geo_monosemanticity_score(scaled, lon, lat),
        rtol=1e-8,
    )


def test_rare_and_constant_features_are_nan_and_left_out_of_the_summary():
    rng = np.random.default_rng(6)
    lon, lat = _globe(100, seed=6)
    acts = rng.uniform(0.1, 1.0, (100, 4))
    acts[:, 1] = 0.7  # constant
    acts[:, 2] = 0.0
    acts[:5, 2] = 1.0  # active on 5 samples
    score = ms.geo_monosemanticity_score(acts, lon, lat, min_active=20)
    assert np.isnan(score[[1, 2]]).all() and np.isfinite(score[[0, 3]]).all()
    assert np.isfinite(ms.geo_monosemanticity_score(acts, lon, lat, min_active=0)[2])
    summary = ms.summarise_monosemanticity(score)
    assert summary["n_scored"] == 2 and summary["n_features"] == 4
    assert summary["mean"] == pytest.approx(score[[0, 3]].mean())
    assert summary["max"] == pytest.approx(score[[0, 3]].max())


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ms.monosemanticity_score(np.ones((5, 2)), np.ones((4, 3)))
    with pytest.raises(ValueError):
        ms.geo_monosemanticity_score(np.ones((5, 2)), np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        ms.mean_pairwise_similarity()

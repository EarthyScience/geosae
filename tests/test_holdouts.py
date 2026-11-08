"""Black-box tests for geosae.eval.holdouts (distance- and environment-graded folds).

Contracts:
  * great-circle distances are right on known cases;
  * every graded fold has the requested sizes, disjoint sides, and a training
    side that lies beyond the test region by at least the gap;
  * a larger gap moves the training side further away, so the ladder is graded;
  * centres are reproducible and respect the minimum separation;
  * climate coordinates are standardised and finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import holdouts


def _globe(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))  # uniform on the sphere
    return np.column_stack([lon, lat])


def test_haversine_known_distances():
    pts = np.array([[0.0, 0.0], [0.0, 90.0], [180.0, 0.0], [10.0, 0.0]])
    d = holdouts.site_distance(pts, pts[0], metric="haversine")
    quarter = np.pi * holdouts.EARTH_RADIUS_KM / 2
    assert d == pytest.approx([0.0, quarter, 2 * quarter, quarter / 9], rel=1e-9)


@pytest.mark.parametrize("metric", ["haversine", "euclidean"])
def test_graded_folds_are_well_formed(metric):
    coords = _globe(6000) if metric == "haversine" else np.random.default_rng(1).standard_normal((6000, 3))
    gaps = (0.0, 1000.0, 3000.0) if metric == "haversine" else (0.0, 0.5, 1.5)
    centres = holdouts.spread_centres(coords, 3, metric=metric, min_separation=gaps[-1], seed=0)
    folds = list(holdouts.graded_folds(coords, centres, gaps=gaps, n_test=200, n_train=800, metric=metric))
    assert len(folds) == 3 * len(gaps)
    for f in folds:
        assert f.test_mask.sum() == 200 and f.train_mask.sum() == 800
        assert not (f.train_mask & f.test_mask).any()
        d = holdouts.site_distance(coords, coords[f.centre], metric=metric)
        assert d[f.train_mask].min() >= d[f.test_mask].max() + f.gap - 1e-9


def test_larger_gap_moves_training_away():
    coords = _globe(8000, seed=2)
    centres = holdouts.spread_centres(coords, 2, metric="haversine", min_separation=5000.0, seed=1)
    gaps = (0.0, 500.0, 2000.0, 6000.0)
    by_centre: dict[int, list[float]] = {}
    for f in holdouts.graded_folds(coords, centres, gaps=gaps, n_test=200, n_train=500, metric="haversine"):
        d = holdouts.site_distance(coords, coords[f.centre], metric="haversine")
        by_centre.setdefault(f.centre, []).append(float(np.median(d[f.train_mask])))
    for med in by_centre.values():
        assert np.all(np.diff(med) > 0)


def test_fold_is_skipped_when_too_few_sites_remain():
    coords = _globe(1000, seed=3)
    centres = np.array([0])
    folds = list(holdouts.graded_folds(coords, centres, gaps=(0.0, 30000.0), n_test=100, n_train=500))
    assert [f.gap for f in folds] == [0.0]


def test_spread_centres_reproducible_and_separated():
    coords = _globe(3000, seed=4)
    a = holdouts.spread_centres(coords, 6, metric="haversine", min_separation=2000.0, seed=7)
    b = holdouts.spread_centres(coords, 6, metric="haversine", min_separation=2000.0, seed=7)
    assert np.array_equal(a, b) and len(a) == 6
    for i in a:
        d = holdouts.site_distance(coords[a], coords[i], metric="haversine")
        assert np.sort(d)[1] >= 2000.0


def test_climate_coordinates_standardised():
    rng = np.random.default_rng(5)
    bio = rng.uniform(1, 100, size=(500, 19))
    z = holdouts.climate_coordinates(bio)
    assert z.shape == (500, len(holdouts.CLIMATE_BANDS))
    assert np.allclose(z.mean(0), 0.0, atol=1e-9) and np.allclose(z.std(0), 1.0, atol=1e-9)


def test_sites_without_climate_never_enter_a_fold():
    rng = np.random.default_rng(6)
    bio = rng.uniform(1, 100, size=(3000, 19))
    bio[:100, 11] = np.nan
    z = holdouts.climate_coordinates(bio)
    assert np.isnan(z[:100]).all() and np.isfinite(z[100:]).all()
    centres = holdouts.spread_centres(z, 2, metric="euclidean", min_separation=1.0, seed=0)
    for f in holdouts.graded_folds(z, centres, gaps=(0.0, 0.5), n_test=200, n_train=1000, metric="euclidean"):
        assert not (f.train_mask[:100] | f.test_mask[:100]).any()

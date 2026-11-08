"""Black-box tests for geosae.eval.spatial (Moran top-quintile + variogram half-life)."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval import spatial as SP


def test_top_quintile_mean_picks_largest():
    v = np.arange(100.0)  # 0..99
    # top 20% == the 20 largest == 80..99, mean 89.5
    assert abs(SP.top_quintile_mean(v) - 89.5) < 1e-9


def test_top_quintile_ignores_nonfinite():
    v = np.array([np.nan, 1.0, 2.0, np.inf, 3.0, 4.0, 5.0])
    out = SP.top_quintile_mean(v)  # 5 finite values, top-20% -> {5.0}
    assert abs(out - 5.0) < 1e-9


def test_summarise_per_feature_fields():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = SP.summarise_per_feature(v)
    assert set(s) == {"top_quintile", "median", "mean", "n"}
    assert s["median"] == 3.0 and s["n"] == 5.0


def test_variogram_halflife_short_vs_long_range():
    # Two synthetic fields over pixels on a small lon/lat grid:
    #  f0 = smooth long-range gradient in longitude (large half-life)
    #  f1 = high-frequency checkerboard (short half-life)
    rng = np.random.default_rng(0)
    n = 4000
    lon = rng.uniform(0.0, 5.0, n)  # ~ up to a few hundred km
    lat = rng.uniform(0.0, 5.0, n)
    f0 = lon.copy()  # slowly varying
    f1 = np.sin(lon * 200.0) + np.sin(lat * 200.0)  # rapidly varying
    codes = np.stack([f0, f1], axis=1)
    hl = SP.variogram_halflife(codes, lon, lat, n_pairs=200_000, d_max_km=800.0, seed=1)
    assert hl.shape == (2,)
    assert np.isfinite(hl).all()
    # the smooth gradient decorrelates over a longer distance than the checkerboard
    assert hl[0] > hl[1]


def test_halflife_from_curve_interpolation():
    centers = np.array([1.0, 10.0, 100.0, 1000.0])
    # a monotone rising semivariogram; sill = 4, half = 2 reached between bins 1 and 2
    curve = np.array([[1.0, 1.5, 3.0, 4.0]])
    hl = SP.halflife_from_curve(centers, curve)
    assert 10.0 <= hl[0] <= 100.0


def _scattered_points(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-60.0, 60.0, n), rng.uniform(-50.0, 50.0, n)


def test_moran_is_high_for_a_smooth_field_and_near_its_expectation_for_noise():
    lon, lat = _scattered_points()
    nbr = SP.knn_neighbours(lon, lat, 8)
    noise = np.random.default_rng(3).normal(size=lon.size)
    moran = SP.moran_i(np.stack([lat, noise]), nbr)
    assert nbr.shape == (lon.size, 8)
    assert moran[0] > 0.9
    assert abs(moran[1] + 1.0 / (lon.size - 1)) < 0.05


def test_moran_matches_the_patch_mean_reduction():
    from geosae.eval.auroc_moran import patch_mean_moran_from_arrays

    lon, lat = _scattered_points(400)
    vals = np.random.default_rng(1).normal(size=(3, lon.size)) + lat
    per_feature = SP.moran_i(vals, SP.knn_neighbours(lon, lat, 8))
    assert patch_mean_moran_from_arrays(vals, lon, lat)["moran_i"] == np.median(per_feature)


def test_moran_rejects_non_finite_values():
    lon, lat = _scattered_points(50)
    vals = lat.copy()
    vals[3] = np.nan
    with pytest.raises(ValueError):
        SP.moran_i(vals, SP.knn_neighbours(lon, lat, 4))


def test_permutation_test_rejects_a_smooth_field_and_not_noise():
    lon, lat = _scattered_points(600)
    nbr = SP.knn_neighbours(lon, lat, 8)
    noise = np.random.default_rng(5).normal(size=lon.size)
    out = SP.moran_permutation_test(np.stack([lat, noise]), nbr, n_perm=99, seed=0)
    assert out["p_value"][0] == pytest.approx(1 / 100)
    assert out["p_value"][1] > 0.05
    assert abs(out["null_mean"][1] + 1.0 / (lon.size - 1)) < 0.01
    assert out["z"][0] > 10


def test_crossing_fraction_reads_a_known_curve():
    centers = np.array([1.0, 10.0, 100.0, 1000.0])
    curve = np.array([[0.1, 0.5, 0.95, 1.0]])
    assert SP.halflife_from_curve(centers, curve)[0] == pytest.approx(10.0)
    assert SP.halflife_from_curve(centers, curve, fraction=0.95)[0] == pytest.approx(100.0)


def test_neighbour_pairs_fill_short_bins_that_random_pairs_miss():
    lon, lat = _scattered_points(800)
    vals = lat[:, None]
    kw = dict(n_pairs=2000, n_bins=12, d_min_km=1.0, d_max_km=20_000.0, min_pairs=20, seed=0)
    random_only, centers = SP.variogram_curve(vals, lon, lat, **kw)
    nbr = SP.knn_neighbours(lon, lat, 8)
    pairs = (np.repeat(np.arange(lon.size), 8), nbr.ravel())
    both, _ = SP.variogram_curve(vals, lon, lat, extra_pairs=pairs, **kw)
    short = centers < 150.0
    assert np.isnan(random_only[0, short]).all() and np.isfinite(both[0, short]).any()


def test_distant_neighbours_are_the_nearest_beyond_the_radius():
    """Against a brute-force great-circle search: the k nearest points farther than ``min_km``."""
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(10, 12, 300), rng.uniform(45, 47, 300)
    nbr = SP.distant_neighbours(lon, lat, 5, 20.0)
    la, lo = np.radians(lat), np.radians(lon)
    hav = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2
    km = 2 * SP.EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))
    expected = np.argsort(np.where(km > 20.0, km, np.inf), axis=1, kind="stable")[:, :5]
    np.testing.assert_array_equal(np.sort(nbr, 1), np.sort(expected, 1))
    assert (np.take_along_axis(km, nbr, 1) > 20.0).all()
    with pytest.raises(ValueError, match="fewer than"):
        SP.distant_neighbours(lon[:3], lat[:3], 5, 20.0)

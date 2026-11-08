import json

import numpy as np
import pytest

from geosae.eval.variogram import (
    BELOW_FIRST_LAG,
    IN_GAP,
    NEVER_REACHED,
    RESOLVED,
    correlation_lengths,
    haversine_km,
    merge_moments,
    nested_sample,
    patch_moments,
    semivariance_curves,
)


def _brute_between(z, dist, lo, hi):
    """Half the mean squared difference over every pixel pair of every patch pair in [lo, hi)."""
    p = len(z)
    flat = z.reshape(p, -1, z.shape[-1])
    tot, n = 0.0, 0
    for a in range(p):
        for b in range(p):
            if a != b and lo <= dist[a, b] < hi:
                d = flat[a][:, None, :] - flat[b][None, :, :]
                tot = tot + 0.5 * (d**2).mean((0, 1))
                n += 1
    return tot / n


def _brute_within(z, lag):
    h = (z[:, :, lag:] - z[:, :, :-lag]).reshape(-1, z.shape[-1])
    v = (z[:, lag:, :] - z[:, :-lag, :]).reshape(-1, z.shape[-1])
    return 0.5 * (np.concatenate([h, v]) ** 2).mean(0)


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(9, 4, 4, 3)) + rng.normal(size=(9, 1, 1, 3)) * 2.0
    lon, lat = rng.uniform(-20, 20, 9), rng.uniform(-20, 20, 9)
    return z, haversine_km(lon, lat)


def test_curves_match_brute_force_semivariance(field):
    z, dist = field
    edges = [0.0, 1500.0, 3000.0, 6000.0]
    out = semivariance_curves(z, dist, pixel_m=10.0, lags=[1, 2], edges_km=edges, min_pairs=1, chunk_patches=2)
    var = z.reshape(-1, 3).var(0)
    for i, lag in enumerate([1, 2]):
        np.testing.assert_allclose(out["curve"][:, i], _brute_within(z, lag) / var, rtol=1e-10)
    for b in range(3):
        if out["n_pairs"][b]:
            ref = _brute_between(z, dist, edges[b], edges[b + 1]) / var
            np.testing.assert_allclose(out["curve"][:, 2 + b], ref, rtol=1e-10)
    np.testing.assert_allclose(out["dist_km"][:2], [0.01, 0.02])
    np.testing.assert_allclose(out["within"], z.reshape(9, -1, 3).var(1).mean(0) / var)


def test_patch_constant_field_has_no_within_patch_variance(field):
    z, dist = field
    const = np.broadcast_to(z.mean((1, 2), keepdims=True), z.shape)
    out = semivariance_curves(const, dist, pixel_m=10.0, lags=[1], edges_km=[0, 1e5], min_pairs=1)
    np.testing.assert_allclose(out["curve"][:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["within"], 0.0, atol=1e-12)
    assert np.all(out["curve"][:, 1] > 0.5)


def test_sparse_bins_and_dead_features_are_nan(field):
    z, dist = field
    z = z.copy()
    z[..., 1] = 0.0
    out = semivariance_curves(z, dist, pixel_m=10.0, lags=[1], edges_km=[0, 1e5], min_pairs=10**6)
    assert np.isnan(out["curve"][:, 1]).all()
    assert np.isnan(out["curve"][1]).all() and np.isnan(out["within"][1])


def test_correlation_length_interpolates_first_crossing_in_log_distance():
    dist = np.array([0.01, 0.1, 10.0, 100.0])
    curve = np.array(
        [
            [0.2, 0.5, 0.9, 0.4],  # crosses between the within lags 0.1 and the first bin 10: gap
            [0.1, 0.2, 0.3, 0.8],  # crosses between 10 and 100: resolved
            [0.7, 0.9, 1.0, 1.0],  # already above at the first lag
            [0.1, 0.1, 0.2, np.nan],  # never reaches the level
        ]
    )
    out = correlation_lengths(curve, dist, n_within=2, level=0.6)
    np.testing.assert_array_equal(out["status"], [IN_GAP, RESOLVED, BELOW_FIRST_LAG, NEVER_REACHED])
    t = (0.6 - 0.5) / (0.9 - 0.5)
    assert out["length_km"][0] == pytest.approx(np.exp(np.log(0.1) + t * (np.log(10) - np.log(0.1))))
    t = (0.6 - 0.3) / (0.8 - 0.3)
    assert out["length_km"][1] == pytest.approx(np.exp(np.log(10) + t * np.log(10)))
    assert out["length_km"][2] == 0.01 and np.isnan(out["length_km"][3])
    np.testing.assert_array_equal(out["nugget"], curve[:, 0])


def test_rejects_lags_outside_the_patch(field):
    z, dist = field
    with pytest.raises(ValueError):
        semivariance_curves(z, dist, pixel_m=10.0, lags=[4], edges_km=[0, 1e5])


def test_stage_scores_runs_and_bases_on_whole_patches(patch_store, tmp_path):
    """``geosae variogram`` writes one curve set per dictionary and refuses partial patches."""
    from geosae import run as R
    from geosae.cli import cmd_variogram

    path, _ = patch_store(n=24, patch_size=4, dim=8)
    run_dir = tmp_path / "vario"
    cfg = {
        "name": "vario",
        "seed": 0,
        "device": "cpu",
        "run": {"dir": str(run_dir)},
        "data": {"zarr_path": path},
        "variogram": {
            "split": "test",
            "n_cells": 8,
            "pixels_per_cell": 16,
            "products": [],
            "bases": ["raw", "pca"],
            "baselines": {"n_fit_pixels": 64},
            "edges_km": [0, 20100],
            "min_pairs": 1,
        },
    }
    rows = cmd_variogram(cfg, run_dir)
    assert [r["name"] for r in rows] == ["raw", "pca"]
    with np.load(R.paths(run_dir)["eval"] / "variogram.npz") as z:
        assert z["raw/curve"].shape == (8, 3 + 1)  # lags 1..3 of a 4x4 grid, one distance bin
        assert int(z["n_within"]) == 3 and z["raw/status"].shape == (8,)
    assert (run_dir / "config.yaml").exists()

    cfg["variogram"]["pixels_per_cell"] = 8
    with pytest.raises(ValueError, match="whole patches"):
        cmd_variogram(cfg, tmp_path / "partial")


def test_stage_fits_bases_on_its_own_sample_when_asked(patch_store, tmp_path):
    """``variogram.fit_on: sample`` fits the bases on the scored pixels, so the training split is irrelevant."""
    import zarr

    from geosae import run as R
    from geosae.cli import cmd_variogram

    path, emb = patch_store(n=24, patch_size=4, dim=8)
    v = {
        "split": "test",
        "n_cells": 6,
        "pixels_per_cell": 16,
        "products": [],
        "bases": ["pca"],
        "baselines": {"n_fit_pixels": 64},
        "edges_km": [0, 20100],
        "min_pairs": 1,
    }

    def curves(tag: str, fit_on: str) -> np.ndarray:
        run_dir = tmp_path / tag
        cfg = {"name": "v", "seed": 0, "device": "cpu", "run": {"dir": str(run_dir)}, "data": {"zarr_path": path}}
        cmd_variogram({**cfg, "variogram": {**v, "fit_on": fit_on}}, run_dir)
        with np.load(R.paths(run_dir)["eval"] / "variogram.npz") as z:
            return z["pca/curve"]

    before = {fit_on: curves(f"{fit_on}0", fit_on) for fit_on in ("sample", "train")}
    stretched = emb.copy()
    stretched[:12, ..., 0] *= 50.0  # the training cells now have one dominant axis
    zarr.open_group(path, mode="r+")["alphaearth"][:] = stretched
    np.testing.assert_allclose(curves("sample1", "sample"), before["sample"], rtol=1e-6)
    assert not np.allclose(curves("train1", "train"), before["train"], rtol=1e-3)


def test_stage_draws_a_nested_sample_and_scores_it_in_chunks(patch_store, tmp_path):
    """``variogram.n_anchors`` reads each anchor with its neighbours; chunking does not change a curve."""
    from geosae import run as R
    from geosae.cli import cmd_variogram

    path, _ = patch_store(n=24, patch_size=4, dim=8)
    v = {
        "split": "test",
        "n_anchors": 2,
        "neighbour_ranks": [1, 2],
        "pixels_per_cell": 16,
        "products": [],
        "bases": ["raw"],
        "baselines": {"n_fit_pixels": 64},
        "edges_km": [0, 20100],
        "min_pairs": 1,
    }
    out = {}
    for chunk in (1, 64):
        run_dir = tmp_path / f"chunk{chunk}"
        cfg = {"name": "v", "seed": 0, "device": "cpu", "run": {"dir": str(run_dir)}, "data": {"zarr_path": path}}
        cmd_variogram({**cfg, "variogram": {**v, "chunk_cells": chunk}}, run_dir)
        meta = json.loads((R.paths(run_dir)["eval"] / "variogram.json").read_text())
        assert 3 <= meta["n_cells"] <= 6
        with np.load(R.paths(run_dir)["eval"] / "variogram.npz") as z:
            out[chunk] = z["raw/curve"]
    np.testing.assert_allclose(out[1], out[64], rtol=1e-6)


def test_moments_of_patch_chunks_merge_to_those_of_all_patches(field):
    z, _ = field
    whole = patch_moments(z, [1, 2])
    merged = merge_moments([patch_moments(z[:4], [1, 2]), patch_moments(z[4:], [1, 2])])
    for key in whole:
        np.testing.assert_allclose(merged[key], whole[key], rtol=1e-12)


def test_nested_sample_holds_an_anchor_with_its_ranked_neighbours():
    lon = np.cumsum(np.arange(1, 21, dtype=float)) / 10.0  # irregular spacing on the equator, no ties
    lat = np.zeros(20)
    idx = nested_sample(lon, lat, n_anchors=1, ranks=[1, 2], seed=3)
    assert len(idx) == 3
    near = {a: set(np.argsort(np.abs(lon - lon[a]))[:3]) for a in idx}
    assert any(near[a] == set(idx) for a in idx)
    np.testing.assert_array_equal(nested_sample(lon, lat, n_anchors=20, ranks=[1], seed=0), np.arange(20))

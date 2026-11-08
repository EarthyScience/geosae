"""The ``geosae coverage`` stage: width-fair concept coverage for one run.

The stage is exercised on a tiny synthetic store, so these tests pin the stage
contract — the cached pixel sample, the run-dir artefacts, the variants scored —
rather than re-testing the reductions, which own their tests.
"""

import csv
import json

import numpy as np
import pytest
import torch
import zarr

from geosae import run as R
from geosae.cli import cmd_coverage
from geosae.eval.registry import calibrate_sparsity
from geosae.models import build_model

MODEL = {"type": "batchtopk", "input_dim": 8, "latent_dim": 16, "k": 4}


@pytest.fixture
def coverage_run(patch_store, tmp_path):
    """A run dir with a checkpoint, next to a labelled tiny store."""
    path, _ = patch_store(n=24, patch_size=4, dim=8)
    g = zarr.open_group(path, mode="a")
    codes = np.tile(np.array([10, 20, 30, 40], dtype="int16"), (24, 4, 1))  # 4 classes per patch
    g.create_array("lc_esa_wc", shape=(24, 4, 4), chunks=(6, 4, 4), dtype="int16")[:] = codes

    run_dir = tmp_path / "runs" / "exp" / "20240101-000000"
    run_dir.mkdir(parents=True)
    # Calibrate the BatchTopK eval threshold before saving, as training would:
    # an uncalibrated checkpoint cannot produce sparse codes at all.
    model = calibrate_sparsity(build_model(MODEL), np.random.default_rng(0).standard_normal((256, 8)), device="cpu")
    torch.save({"model": model.state_dict(), "epoch": 0}, R.paths(run_dir)["best"])
    return run_dir, path


def _cfg(run_dir, zarr_path, **coverage_kw):
    cov = {
        "split": "test",
        "n_patches": 6,
        "pixels_per_patch": 16,
        "products": ["lc_esa_wc"],
        "min_positive": 5,
        "max_per_side": 20,
        "n_perm": 10,
        "matched_budgets": [4],
        "dense_bases": ["raw", "pca"],
    }
    cov.update(coverage_kw)
    return {
        "name": "exp",
        "seed": 0,
        "device": "cpu",
        "run": {"dir": str(run_dir)},
        "data": {"zarr_path": zarr_path},
        "model": MODEL,
        "coverage": cov,
    }


def test_stage_writes_json_csv_and_arrays(coverage_run):
    run_dir, zarr_path = coverage_run
    out = cmd_coverage(_cfg(run_dir, zarr_path), run_dir)
    P = R.paths(run_dir)

    saved = json.loads((P["eval"] / "test_coverage.json").read_text())
    assert saved == out
    assert out["n_pixels"] == 6 * 16
    for variant in ("sae", "sae_top4", "raw", "pca"):
        assert 0.0 <= out[variant]["ccov"] <= 1.0
        assert out[variant]["n_valid_concepts"] >= 1
    assert out["sae"]["n_features"] == 16 and out["sae_top4"]["n_features"] == 4
    assert out["raw"]["n_features"] == 8

    rows = list(csv.DictReader(open(P["eval"] / "test_coverage.csv")))
    assert [r["variant"] for r in rows] == ["sae", "sae_matched", "raw", "pca"]

    with np.load(P["eval"] / "test_coverage_arrays.npz") as z:
        assert z["sae__auc_matrix"].shape[0] == 16
        assert z["sae__null_best"].shape[1] == 10  # (C, n_perm), one null per concept size
        assert z["sae_top4__feature_index"].shape == (4,)
        assert z["class_table"].shape == (z["sae__auc_matrix"].shape[1], 2)
        assert set(z["product_order"][z["class_table"][:, 0]]) <= set(z["product_order"])


def test_the_pixel_sample_is_cached_and_reused(coverage_run):
    run_dir, zarr_path = coverage_run
    cmd_coverage(_cfg(run_dir, zarr_path), run_dir)
    sample_path = R.paths(run_dir)["eval"] / "test_registry_sample.npz"
    assert sample_path.exists()

    # A second run must score the identical pixels — that is what the cache is for.
    from geosae.eval.registry import RegistrySample

    first = RegistrySample.load(sample_path)
    cmd_coverage(_cfg(run_dir, zarr_path), run_dir)
    again = RegistrySample.load(sample_path)
    assert np.array_equal(first.raw, again.raw)
    assert np.array_equal(first.cell, again.cell)


def test_a_frozen_pool_replaces_the_ad_hoc_sample(coverage_run, tmp_path, capsys):
    """With ``coverage.pool`` set the stage scores the pool's pixels and says so."""
    from geosae.cli import cmd_pool
    from geosae.eval.pool import EvalPool

    run_dir, zarr_path = coverage_run
    pool_npz = tmp_path / "pools" / "unit.v1.npz"
    cache_root = tmp_path / "sample_cache"  # keep the content-addressed cache inside tmp_path
    cmd_pool(
        {
            "data": {"zarr_path": zarr_path},
            "pool": {
                "name": "unit",
                "version": 1,
                "out": str(pool_npz),
                "cache_root": str(cache_root),
                "n_cells": 6,
                "pixels_per_cell": 16,
                "products": ["lc_esa_wc"],
                "max_per_side": 20,
                "min_positive": 5,
            },
        }
    )
    pool = EvalPool.load(pool_npz)

    out = cmd_coverage(_cfg(run_dir, zarr_path, pool=str(pool_npz), cache_root=str(cache_root)), run_dir)
    printed = capsys.readouterr().out
    assert "unit.v1" in printed and "WARNING" not in printed
    assert out["pool"]["pool_id"] == "unit.v1"
    assert out["pool"]["spec_digest"] == pool.spec.digest
    assert out["n_pixels"] == pool.n_pixels
    assert out["sae"]["n_valid_concepts"] == int(pool.scorable_concepts().sum()) >= 1
    # The ad-hoc cache is never written when a pool is in charge.
    assert not (R.paths(run_dir)["eval"] / "test_registry_sample.npz").exists()


def test_the_ad_hoc_path_warns_that_it_is_not_a_pool(coverage_run, capsys):
    run_dir, zarr_path = coverage_run
    cmd_coverage(_cfg(run_dir, zarr_path), run_dir)
    assert "WARNING" in capsys.readouterr().out


def test_the_model_sees_the_configs_transformed_pixels(coverage_run):
    """``data.transforms`` reach the encoder: shrinking the input silences the SAE."""
    run_dir, zarr_path = coverage_run
    cmd_coverage(_cfg(run_dir, zarr_path, dense_bases=[], matched_budgets=[]), run_dir)
    with np.load(R.paths(run_dir)["eval"] / "test_coverage_arrays.npz") as z:
        plain = z["sae__activation_rate"].sum()
    cfg = _cfg(run_dir, zarr_path, dense_bases=[], matched_budgets=[])
    cfg["data"]["transforms"] = {"input_scale": 1e-6}
    cmd_coverage(cfg, run_dir)
    with np.load(R.paths(run_dir)["eval"] / "test_coverage_arrays.npz") as z:
        shrunk = z["sae__activation_rate"].sum()
    assert plain > 0 and shrunk < 0.1 * plain

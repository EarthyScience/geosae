"""End-to-end train -> predict on tiny synthetic data: the run-dir contract.

A run directory is the library's reproduction record. These tests pin what a
completed run must contain, that the recorded config is the one that ran, that
the checkpoint reloads into a model rebuilt from that config, and that the
stages share one directory.
"""

import json

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from geosae import run as R
from geosae.cli import main
from geosae.models import build_model

DIM = 8


def _config(zarr_path, run_root, **over):
    cfg = {
        "name": "tiny",
        "seed": 7,
        "device": "cpu",
        "run": {"root": str(run_root)},
        "data": {"zarr_path": str(zarr_path), "split_name": "split_210k", "inram_dtype": "float32"},
        "model": {"type": "batchtopk", "input_dim": DIM, "latent_dim": 16, "k": 4},
        "train": {
            "epochs": 2,
            "steps_per_epoch": 3,
            "batch_size": 4,
            "val_batch_size": 4,
            "rampup": 2,
            "amp": False,
            "num_workers": 0,
            "pin_memory": False,
        },
        "predict": {"split": "test", "compute_inram": True, "batch_size": 4, "num_workers": 0, "pin_memory": False},
    }
    for key, value in over.items():
        cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) else value
    return cfg


def _write(tmp_path, cfg, name="cfg.yaml"):
    path = tmp_path / name
    OmegaConf.save(OmegaConf.create(cfg), str(path))
    return str(path)


@pytest.fixture
def trained_run(patch_store, tmp_path):
    """Run ``geosae train`` once on a tiny store; return ``(run_dir, cfg, cfg_path)``."""
    store, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    cfg = _config(store, tmp_path / "runs")
    cfg_path = _write(tmp_path, cfg)
    main(["train", "-c", cfg_path])
    return R.latest_run_dir(str(tmp_path / "runs"), "tiny"), cfg, cfg_path


def test_train_writes_the_full_run_dir_contract(trained_run):
    run_dir, cfg, _ = trained_run
    P = R.paths(run_dir)
    assert P["config"].exists(), "the resolved config is the reproduction record"
    assert P["best"].exists() and P["last"].exists()
    assert P["metrics_jsonl"].exists()
    assert (P["eval"] / "val_metrics.csv").exists()

    # The recorded config is exactly what ran.
    assert OmegaConf.to_container(OmegaConf.load(str(P["config"]))) == cfg

    rows = [json.loads(line) for line in P["metrics_jsonl"].read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1]  # one record per epoch
    for r in rows:
        assert r["step"] > 0 and r["time_s"] >= 0
        assert {"recon_mse", "aux", "l0"} <= set(r["train"])
        assert {"fve", "recon_cosine", "l0", "threshold"} <= set(r["val"])


def test_checkpoint_reloads_into_a_model_rebuilt_from_the_config(trained_run):
    run_dir, cfg, _ = trained_run
    ckpt = torch.load(R.paths(run_dir)["best"], map_location="cpu", weights_only=False)
    assert {"model", "model_cfg", "epoch", "step", "val"} <= set(ckpt)
    assert ckpt["model_cfg"] == cfg["model"]

    model = build_model({**ckpt["model_cfg"], "seed": cfg["seed"]})
    model.load_state_dict(ckpt["model"])  # strict: no missing/unexpected keys
    assert model.latent_dim == cfg["model"]["latent_dim"] and model.k == cfg["model"]["k"]


def test_same_seed_gives_the_same_checkpoint(patch_store, tmp_path):
    store, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    weights = []
    for i in range(2):
        cfg_path = _write(tmp_path, _config(store, tmp_path / f"runs{i}"), name=f"cfg{i}.yaml")
        main(["train", "-c", cfg_path])
        run_dir = R.latest_run_dir(str(tmp_path / f"runs{i}"), "tiny")
        weights.append(torch.load(R.paths(run_dir)["last"], map_location="cpu", weights_only=False)["model"])
    assert weights[0].keys() == weights[1].keys()
    for key in weights[0]:
        assert torch.equal(weights[0][key], weights[1][key]), f"{key} is not reproducible at a fixed seed"


def test_predict_reuses_the_train_run_dir_and_writes_the_store(trained_run):
    run_dir, cfg, cfg_path = trained_run
    main(["predict", "-c", cfg_path])

    P = R.paths(run_dir)
    assert R.latest_run_dir(cfg["run"]["root"], "tiny") == run_dir, "predict must not create a new run dir"
    store = P["predictions"] / "test.zarr"
    assert store.exists()
    g = zarr.open_group(str(store), mode="r")
    assert g.attrs["latent_dim"] == cfg["model"]["latent_dim"]
    assert g.attrs["k"] == cfg["model"]["k"]
    assert g.attrs["split"] == "test"
    assert g["z_values"].shape[:3] == (g.attrs["n_patches"], 4, 4)
    assert set(np.unique(g["fold"][:]).tolist()) == {2}

    metrics = (P["eval"] / "test_metrics.csv").read_text().splitlines()
    assert metrics[0].split(",")[0] == "split"
    assert {"fve", "recon_cosine", "l0"} <= set(metrics[0].split(","))
    assert metrics[1].split(",")[0] == "test"


def test_optional_train_keys_have_working_defaults(patch_store, tmp_path):
    """Only data/model/the loop budget are required; every other key has a default."""
    store, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    cfg = {
        "name": "bare",
        "device": "cpu",
        "run": {"root": str(tmp_path / "runs")},
        "data": {"zarr_path": store},
        "model": {"type": "batchtopk", "input_dim": DIM, "latent_dim": 16, "k": 4},
        "train": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 4, "amp": False, "num_workers": 0},
    }
    main(["train", "-c", _write(tmp_path, cfg, name="bare.yaml")])
    run_dir = R.latest_run_dir(str(tmp_path / "runs"), "bare")
    assert R.paths(run_dir)["best"].exists()


def test_layered_configs_and_overrides_reach_the_run(patch_store, tmp_path):
    store, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    base = _write(tmp_path, _config(store, tmp_path / "runs"), name="base.yaml")
    layer = _write(tmp_path, {"model": {"latent_dim": 24}}, name="layer.yaml")
    main(["train", "-c", base, "-c", layer, "-o", "train.epochs=1", "-o", "seed=11"])

    run_dir = R.latest_run_dir(str(tmp_path / "runs"), "tiny")
    recorded = OmegaConf.to_container(OmegaConf.load(str(R.paths(run_dir)["config"])))
    assert recorded["model"]["latent_dim"] == 24  # later layer wins
    assert recorded["model"]["k"] == 4  # deep merge kept the base's model keys
    assert recorded["train"]["epochs"] == 1 and recorded["seed"] == 11
    rows = R.paths(run_dir)["metrics_jsonl"].read_text().splitlines()
    assert len(rows) == 1  # the override really shortened training
    ckpt = torch.load(R.paths(run_dir)["best"], map_location="cpu", weights_only=False)
    assert ckpt["model"]["decoder.weight"].shape == (DIM, 24)

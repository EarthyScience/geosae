"""Inference-threshold calibration drives eval L0 to ~k."""

import numpy as np
import torch
import zarr

from geosae.data.dataset import PatchData
from geosae.eval.metrics import calibrate_threshold
from geosae.models import TopKSAE


def _store(path, n=64, ps=8, d=16):
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update({"n_cells": n, "patch_size": ps})
    rng = np.random.default_rng(0)
    g.create_array("alphaearth", shape=(n, ps, ps, d), chunks=(16, ps, ps, d), dtype="float32")[:] = (
        rng.standard_normal((n, ps, ps, d)).astype("float32")
    )
    g.create_array("mask_alphaearth", shape=(n,), chunks=(n,), dtype="bool")[:] = np.ones(n, bool)
    g.create_array("split_210k", shape=(n,), chunks=(n,), dtype="int8")[:] = np.zeros(n, "int8")
    g.create_array("lon", shape=(n,), chunks=(n,), dtype="float64")[:] = np.zeros(n)
    g.create_array("lat", shape=(n,), chunks=(n,), dtype="float64")[:] = np.zeros(n)


def test_calibration_hits_target_l0(tmp_path):
    _store(tmp_path / "s.zarr")
    data = PatchData(str(tmp_path / "s.zarr"), split="train", compute=True, inram_dtype="float32")
    torch.manual_seed(0)
    model = TopKSAE(input_dim=16, latent_dim=64, k=8, pre_bias=False)
    calibrate_threshold(model, data, target_k=8, n_patches=32, device="cpu", amp_dtype=None)

    model.eval()
    batch = next(data.iter_batches(32, shuffle=False))
    x = batch["x"].reshape(-1, 16)
    _, z, _ = model(x)
    mean_l0 = float((z > 0).float().sum(-1).mean())
    # Quantile calibration targets an average of k active latents per pixel.
    assert abs(mean_l0 - 8) < 1.5

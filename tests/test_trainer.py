"""Trainer-level wiring of the target-L0 sparsity controller (see
geosae.train.sparsity_controller): the ReLU+l1 penalty (`Trainer.l1_coeff`)
moves in the direction the observed L0 implies, is logged in metrics.jsonl, and the whole
mechanism is a no-op when `target_l0` is unset. Convergence-to-target of the
controller itself is covered directly, under a controlled feedback function,
by tests/test_sparsity_controller.py; here we only pin the Trainer's wiring.

Also covers `best.pth` selection: with a controller wired, an epoch whose
validation L0 has not settled near the target only competes for `best.pth`
within `best_checkpoint_l0_tolerance` of it (an off-target epoch can otherwise
win on FVE alone forever -- FVE is only comparable at the same sparsity); with
no controller (the Top-K family), every epoch is comparable and the gate is a
no-op.
"""

from __future__ import annotations

import json

import pytest
import torch

from geosae.data.dataset import PatchData
from geosae.models.relu_sae import ReLUSAE
from geosae.train.trainer import Trainer

DIM = 8
LATENT = 32


def _data(patch_store, tmp_path, n=64):
    path, _ = patch_store(n=n, patch_size=2, dim=DIM, chunk=16)
    train = PatchData(path, split="train", compute=True, inram_dtype="float32")
    val = PatchData(path, split="val", compute=True, inram_dtype="float32")
    return train, val


def _fit(model, train, val, tmp_path, *, target_l0=None, l1_coeff=0.0, epochs=1, steps_per_epoch=1):
    trainer = Trainer(
        model,
        train,
        val,
        outdir=str(tmp_path / "run"),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=8,
        val_batch_size=8,
        rampup=2,
        amp=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        l1_coeff=l1_coeff,
        target_l0=target_l0,
    )
    trainer.fit()
    return trainer


def test_relu_l1_l0_converges_near_the_target(patch_store, tmp_path):
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    trainer = Trainer(
        model,
        train,
        val,
        outdir=str(tmp_path / "run"),
        epochs=80,
        steps_per_epoch=4,
        batch_size=8,
        val_batch_size=8,
        rampup=5,
        lr=8e-3,
        amp=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        l1_coeff=1e-3,
        target_l0={"value": 6.0, "gain": 1.0, "ema_beta": 0.3},
    )
    trainer.fit()
    log = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    initial_l0 = log[0]["train"]["l0"]
    final_l0 = log[-1]["train"]["l0"]
    assert abs(final_l0 - 6.0) < abs(initial_l0 - 6.0)  # moved closer
    assert abs(final_l0 - 6.0) < 5.0
    assert trainer.l1_coeff > 1e-3  # the controller actually tightened it


def test_l1_penalty_grows_when_observed_l0_is_above_target(patch_store, tmp_path):
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    # well below the ~half-latent start
    trainer = _fit(model, train, val, tmp_path, l1_coeff=1e-3, target_l0={"value": 4.0, "warmup_steps": 0})
    assert trainer.l1_coeff > 1e-3
    assert trainer.target_l0_controller.ema_l0 is not None


def test_target_l0_defaults_its_warmup_to_the_trainer_rampup(patch_store, tmp_path):
    """No explicit `warmup_steps`: the controller should hold the penalty
    through the LR rampup (matching real sweep configs, which never set
    `warmup_steps` and rely on this default) and only start adjusting once
    it ends."""
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    trainer = _fit(
        model, train, val, tmp_path, l1_coeff=1e-3, target_l0={"value": 4.0}, epochs=2, steps_per_epoch=1
    )  # rampup=2 in _fit: exactly as many steps as this run takes
    assert trainer.target_l0_controller.warmup_steps == 2
    assert trainer.l1_coeff == 1e-3  # held throughout: both steps fell inside the rampup


def test_l1_penalty_shrinks_when_observed_l0_is_below_target(patch_store, tmp_path):
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    trainer = _fit(model, train, val, tmp_path, l1_coeff=1e-3, target_l0={"value": 30.0, "warmup_steps": 0})
    assert trainer.l1_coeff < 1e-3


def test_sparsity_penalty_is_logged_in_metrics_jsonl(patch_store, tmp_path):
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    _fit(model, train, val, tmp_path, l1_coeff=1e-3, target_l0={"value": 4.0, "warmup_steps": 0})
    log = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert "sparsity_penalty" in log[-1]["train"]
    assert "l0" in log[-1]["train"]


def test_target_l0_none_leaves_the_penalty_untouched(patch_store, tmp_path):
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    trainer = _fit(model, train, val, tmp_path, l1_coeff=1e-3, target_l0=None, epochs=5, steps_per_epoch=3)
    assert trainer.target_l0_controller is None
    assert trainer.l1_coeff == 1e-3

    log = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert "sparsity_penalty" not in log[-1]["train"]


def test_relu_l1_target_l0_requires_a_nonzero_starting_penalty(patch_store, tmp_path):
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    with pytest.raises(ValueError, match="l1_coeff"):
        _fit(model, train, val, tmp_path, l1_coeff=0.0, target_l0={"value": 5.0})


def test_best_checkpoint_skips_epochs_whose_l0_missed_the_target(monkeypatch, patch_store, tmp_path):
    """Regression: with a `target_l0` controller, an epoch's FVE is only
    comparable to another's at the same sparsity. An epoch whose validation L0
    sits far above the target (e.g. during the controller's own warm-up, before
    it has tightened the penalty) reconstructs an easier, denser problem and
    trivially beats every properly-sparse epoch's FVE and would otherwise win
    `best.pth` forever regardless of how much later epochs converged toward
    the intended operating point. `Trainer.fit` must skip such an epoch for
    `best.pth` selection even though its raw FVE is higher.
    """
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)

    # Epoch 0: far denser than the target (L0=24 vs target 6) but a
    # deceptively higher FVE -- the degenerate checkpoint that must not win.
    # Epoch 1: at the target (L0=6), lower FVE -- the epoch that should win.
    fake_vals = iter(
        [
            {"fve": 0.95, "recon_cosine": 0.9, "recon_mse": 0.01, "l0": 24.0, "dead_frac": 0.0, "n_pixels": 100},
            {"fve": 0.60, "recon_cosine": 0.8, "recon_mse": 0.10, "l0": 6.0, "dead_frac": 0.0, "n_pixels": 100},
        ]
    )
    monkeypatch.setattr("geosae.train.trainer.reconstruction_metrics", lambda *a, **k: next(fake_vals))
    monkeypatch.setattr("geosae.train.trainer.calibrate_threshold", lambda *a, **k: 0.0)

    trainer = Trainer(
        model,
        train,
        val,
        outdir=str(tmp_path / "run"),
        epochs=2,
        steps_per_epoch=1,
        batch_size=8,
        val_batch_size=8,
        rampup=1,
        amp=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        l1_coeff=1e-3,
        target_l0={"value": 6.0, "warmup_steps": 0},
    )
    best = trainer.fit()
    assert best["l0"] == 6.0
    ckpt = torch.load(tmp_path / "run" / "best.pth", map_location="cpu", weights_only=False)
    assert ckpt["val"]["l0"] == 6.0


def test_best_checkpoint_ignores_the_l0_gate_without_a_controller(monkeypatch, patch_store, tmp_path):
    """The Top-K family reaches its sparsity by construction (L0 == k every
    epoch), so with no `target_l0` controller every epoch is comparable and
    the highest-FVE epoch wins `best.pth`, however far its L0 sits from any
    other epoch's."""
    train, val = _data(patch_store, tmp_path)
    model = ReLUSAE(input_dim=DIM, latent_dim=LATENT, seed=0)
    fake_vals = iter(
        [
            {"fve": 0.60, "recon_cosine": 0.8, "recon_mse": 0.10, "l0": 6.0, "dead_frac": 0.0, "n_pixels": 100},
            {"fve": 0.95, "recon_cosine": 0.9, "recon_mse": 0.01, "l0": 24.0, "dead_frac": 0.0, "n_pixels": 100},
        ]
    )
    monkeypatch.setattr("geosae.train.trainer.reconstruction_metrics", lambda *a, **k: next(fake_vals))
    monkeypatch.setattr("geosae.train.trainer.calibrate_threshold", lambda *a, **k: 0.0)

    trainer = Trainer(
        model,
        train,
        val,
        outdir=str(tmp_path / "run"),
        epochs=2,
        steps_per_epoch=1,
        batch_size=8,
        val_batch_size=8,
        rampup=2,
        amp=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
    )
    best = trainer.fit()
    assert best["l0"] == 24.0  # the higher-fve epoch wins: no controller, no gate

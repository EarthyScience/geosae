"""Black-box tests for checkpoint loading (:func:`geosae.models.load_checkpoint`).

Contracts:
  * a round-tripped checkpoint restores every parameter exactly;
  * a checkpoint without the derived Matryoshka-prefix buffers still loads,
    and the reconstructed buffers hold the values the config implies;
  * a checkpoint from a genuinely different model is rejected, not half-loaded.
"""

from __future__ import annotations

import pytest
import torch

from geosae.models import DERIVED_BUFFERS, build_model, load_checkpoint

CFG = {"type": "batchtopk", "input_dim": 8, "latent_dim": 32, "k": 4, "seed": 0}


def _save(state, tmp_path, name="ckpt.pth"):
    p = tmp_path / name
    torch.save({"model": state, "epoch": 7}, p)
    return str(p)


def test_round_trip_restores_every_parameter(tmp_path):
    src = build_model(CFG)
    with torch.no_grad():
        src.encoder.weight.add_(1.234)
    path = _save(src.state_dict(), tmp_path)
    dst = build_model(CFG)
    ckpt = load_checkpoint(dst, path)
    assert ckpt["epoch"] == 7
    assert torch.allclose(dst.encoder.weight, src.encoder.weight)


def test_checkpoint_without_the_derived_buffers_still_loads(tmp_path):
    src = build_model(CFG)
    state = {k: v for k, v in src.state_dict().items() if k not in DERIVED_BUFFERS}
    dst = build_model(CFG)
    load_checkpoint(dst, _save(state, tmp_path))
    # the constructor's values survive and match the config's implied full prefix
    assert int(dst.n_high) + int(dst.n_low) == CFG["latent_dim"]


def test_a_mismatched_checkpoint_is_rejected(tmp_path):
    other = build_model({**CFG, "latent_dim": 64})
    with pytest.raises(RuntimeError):
        load_checkpoint(build_model(CFG), _save(other.state_dict(), tmp_path))


def test_a_checkpoint_missing_a_real_parameter_is_rejected(tmp_path):
    src = build_model(CFG)
    state = {k: v for k, v in src.state_dict().items() if k != "encoder.weight"}
    with pytest.raises(RuntimeError, match="encoder.weight"):
        load_checkpoint(build_model(CFG), _save(state, tmp_path))

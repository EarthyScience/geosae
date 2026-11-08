"""Contract tests for the ReLU + ℓ1 Sparse Autoencoder."""

import math

import pytest
import torch

from geosae.models import build_model
from geosae.models.relu_sae import ReLUSAE
from geosae.train.loss import sae_loss


def test_forward_shapes_and_positivity():
    m = ReLUSAE(input_dim=8, latent_dim=32, normalize_inputs=False)
    m.train()
    x = torch.randn(16, 8)
    recon, z, z_dense = m(x)
    assert recon.shape == x.shape
    assert z.shape == (16, 32) and z_dense.shape == (16, 32)
    # ReLU: the sparse code is non-negative and equals relu(z_dense).
    assert (z >= 0).all()
    assert torch.allclose(z, torch.relu(z_dense))


def test_decoder_columns_unit_norm_after_train_forward():
    m = ReLUSAE(input_dim=8, latent_dim=32, normalize_decoder=True)
    m.train()
    m(torch.randn(10, 8))
    norms = m.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_pre_bias_removes_decoder_bias_and_centres():
    m = ReLUSAE(input_dim=8, latent_dim=16, pre_bias=True)
    assert m.decoder.bias is None
    assert hasattr(m, "b_dec") and m.b_dec.shape == (8,)


def test_tied_decoder_uses_encoder_transpose():
    m = ReLUSAE(input_dim=8, latent_dim=16, tied_decoder=True)
    assert m.decoder is None
    z = torch.zeros(4, 16)
    z[:, 3] = 2.0
    # Decode of a one-hot latent must equal the matching encoder-row direction (+b_dec).
    out = m.decode(z)
    expected = 2.0 * m.encoder.weight[3] + m.b_dec
    assert torch.allclose(out, expected, atol=1e-6)


def test_normalize_inputs_scales_to_sqrt_d():
    d = 64
    m = ReLUSAE(input_dim=d, latent_dim=16, normalize_inputs=True)
    x = torch.randn(32, d) * 0.01  # tiny norm -> must be scaled up to ~sqrt(d)
    xn = m._normalize(x)
    assert abs(float(xn.norm(dim=-1).mean()) - math.sqrt(d)) < 1e-3
    # Disabled -> identity.
    m2 = ReLUSAE(input_dim=d, latent_dim=16, normalize_inputs=False)
    assert torch.allclose(m2._normalize(x), x)


def test_l1_penalty_enters_loss_and_rewards_sparsity():
    torch.manual_seed(0)
    m = ReLUSAE(input_dim=8, latent_dim=32, normalize_inputs=False)
    m.train()
    x = torch.randn(64, 8)
    recon, z, z_dense = m(x)
    loss0, met0 = sae_loss(m, x, recon, z, z_dense, aux_penalty=0.0, l1_coeff=0.0)
    loss1, met1 = sae_loss(m, x, recon, z, z_dense, aux_penalty=0.0, l1_coeff=0.1)
    # A positive ℓ1 coefficient strictly increases the objective by l1_coeff*||z||_1.
    assert float(loss1) > float(loss0)
    assert "l1" in met1
    assert abs(float(loss1) - float(loss0) - 0.1 * met1["l1"]) < 1e-5


def test_build_model_relu_type():
    m = build_model({"type": "relu", "input_dim": 8, "latent_dim": 16})
    assert isinstance(m, ReLUSAE)
    assert not m.batch_topk  # eval calibrate_threshold is a no-op for ReLU
    m2 = build_model({"type": "relu_l1", "input_dim": 8, "latent_dim": 16})
    assert isinstance(m2, ReLUSAE)


def test_build_model_relu_ignores_a_shared_blocks_topk_only_keys():
    """A shared model block written for the Top-K variants must still build ReLU."""
    shared = {
        "input_dim": 8,
        "latent_dim": 16,
        "k": 4,
        "batch_topk": True,
        "freeze_decoder": False,
        "seed": 1,
        "n_batches_to_dead": 25,
    }
    m = build_model({"type": "relu", **shared})
    assert isinstance(m, ReLUSAE)


def test_build_model_relu_refuses_a_frozen_decoder():
    """ReLU has no frozen-decoder mode, so the null must not be silently dropped."""
    with pytest.raises(ValueError, match="freeze_decoder"):
        build_model({"type": "relu", "input_dim": 8, "latent_dim": 16, "freeze_decoder": True})


def test_dead_feature_tracking():
    m = ReLUSAE(input_dim=4, latent_dim=16, n_batches_to_dead=3, normalize_inputs=False)
    m.train()
    with torch.no_grad():
        m.encoder.weight[8:] = 0.0
        m.encoder.bias[8:] = -1e4  # rows 8: can never exceed 0 -> ReLU kills them
    for _ in range(4):
        m(torch.randn(32, 4))
    dead = m.dead_features_mask()
    assert dead[8:].all() and not dead[:8].any()


def test_forward_on_patch_shape():
    m = ReLUSAE(input_dim=64, latent_dim=128, normalize_inputs=True)
    m.train()
    x = torch.randn(2, 16, 16, 64)
    recon, z, z_dense = m(x)
    assert recon.shape == x.shape
    assert z.shape[:-1] == x.shape[:-1] and z.shape[-1] == 128

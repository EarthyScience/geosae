"""Contract tests for the Matryoshka BatchTopK SAE (prefix / slow-group loss).

The Matryoshka variant has the *same* forward pass as plain BatchTopK; the
nesting is enforced entirely by the extra slow-group prefix-reconstruction term
in the loss. These tests pin (a) the partition bookkeeping, (b) forward-pass
equality with BatchTopK, and (c) the slow-group loss contract.
"""

import torch

from geosae.models import TopKSAE, build_model
from geosae.train.loss import sae_loss


def test_build_model_matryoshka_sets_partition():
    m = build_model({"type": "matryoshka", "input_dim": 64, "latent_dim": 1024, "k": 16, "h_frac": 0.2})
    assert isinstance(m, TopKSAE)
    assert m.is_matryoshka is True
    assert m.batch_topk is True
    # round(0.2 * 1024) = 205 high features; 819 in the free (low) partition.
    assert m.n_prefix == 205
    assert int(m.n_low) == 1024 - 205


def test_h_frac_one_puts_all_features_high():
    m = build_model({"type": "batchtopk", "input_dim": 8, "latent_dim": 64, "k": 4})
    assert m.n_prefix == 64 and int(m.n_low) == 0
    assert getattr(m, "is_matryoshka", False) is False


def test_forward_matches_batchtopk_exactly():
    # Same weights + same input -> identical forward (matryoshka is loss-only).
    torch.manual_seed(0)
    base = TopKSAE(input_dim=16, latent_dim=64, k=4, seed=0)
    torch.manual_seed(0)
    matr = TopKSAE(input_dim=16, latent_dim=64, k=4, h_frac=0.2, seed=0)
    matr.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    matr.eval()
    x = torch.randn(8, 16)
    rb, zb, _ = base(x)
    rm, zm, _ = matr(x)
    assert torch.allclose(rb, rm, atol=1e-6)
    assert torch.allclose(zb, zm, atol=1e-6)


def test_slow_group_term_added_and_nonnegative():
    torch.manual_seed(0)
    m = build_model({"type": "matryoshka", "input_dim": 16, "latent_dim": 64, "k": 8, "h_frac": 0.25})
    m.train()
    x = torch.randn(32, 16)
    recon, z, z_dense = m(x)
    l0, _ = sae_loss(m, x, recon, z, z_dense, aux_penalty=0.0, slow_group_weight=0.0)
    l1, met = sae_loss(m, x, recon, z, z_dense, aux_penalty=0.0, slow_group_weight=1.0)
    assert "slow_group" in met and met["slow_group"] >= 0.0
    # The slow-group prefix uses fewer features than the full code, so its MSE is
    # >= the full reconstruction MSE; adding it strictly increases the objective.
    assert float(l1) > float(l0)
    assert abs(float(l1) - float(l0) - met["slow_group"]) < 1e-5


def test_slow_group_uses_only_prefix_features():
    torch.manual_seed(0)
    m = build_model({"type": "matryoshka", "input_dim": 8, "latent_dim": 32, "k": 8, "h_frac": 0.5})
    m.eval()
    n_s = m.n_prefix
    z = torch.rand(4, 32) + 0.1  # all features active
    z_slow = z.clone()
    z_slow[..., n_s:] = 0
    # Reconstruction from the prefix must ignore the suffix atoms entirely.
    recon_full_prefix = m.decode(z_slow)
    z_perturbed = z.clone()
    z_perturbed[..., n_s:] = 999.0  # perturb only suffix
    z_perturbed[..., :n_s] = z[..., :n_s]
    z_perturbed_slow = z_perturbed.clone()
    z_perturbed_slow[..., n_s:] = 0
    assert torch.allclose(m.decode(z_perturbed_slow), recon_full_prefix, atol=1e-6)


def test_nested_prefixes_are_sorted_proper_and_unique():
    m = build_model({"type": "matryoshka", "input_dim": 8, "latent_dim": 64, "k": 4,
                     "prefix_fracs": [0.5, 0.0625, 0.25, 0.125, 0.25, 1.0]})  # fmt: skip
    assert m.prefix_sizes == (4, 8, 16, 32)


def _prefix_mse(model, x, z, n):
    return float(torch.nn.functional.mse_loss(model.decode(torch.where(torch.arange(z.shape[-1]) < n, z, 0.0)), x))


def test_each_nested_prefix_adds_its_own_reconstruction_term():
    torch.manual_seed(0)
    cfg = {"type": "matryoshka", "input_dim": 16, "latent_dim": 64, "k": 8}
    one = build_model({**cfg, "prefix_fracs": [0.25]})
    many = build_model({**cfg, "prefix_fracs": [0.125, 0.25, 0.5]})
    many.load_state_dict(one.state_dict(), strict=False)
    x = torch.randn(32, 16)
    recon, z, z_dense = one(x)
    base, _ = sae_loss(one, x, recon, z, z_dense, aux_penalty=0.0, slow_group_weight=0.0)
    l1, _ = sae_loss(one, x, recon, z, z_dense, aux_penalty=0.0, slow_group_weight=1.0)
    l3, met = sae_loss(many, x, recon, z, z_dense, aux_penalty=0.0, slow_group_weight=1.0)
    expected = sum(_prefix_mse(one, x, z, n) for n in (8, 16, 32))
    assert abs(float(l1) - float(base) - _prefix_mse(one, x, z, 16)) < 1e-5
    assert abs(float(l3) - float(base) - expected) < 1e-5
    assert abs(met["slow_group"] - expected) < 1e-5

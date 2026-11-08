"""Algorithmic correctness of the Top-K / BatchTopK SAE."""

import torch

from geosae.models import TopKSAE, build_model


def test_per_sample_topk_keeps_exactly_k():
    m = TopKSAE(input_dim=8, latent_dim=32, k=4, batch_topk=False, pre_bias=False)
    m.train()
    x = torch.randn(16, 8)
    _, z, _ = m(x)
    assert torch.equal((z > 0).sum(-1), torch.full((16,), 4)) or ((z != 0).sum(-1) == 4).all()


def test_batch_topk_average_sparsity():
    m = TopKSAE(input_dim=8, latent_dim=64, k=4, batch_topk=True, pre_bias=False)
    m.train()
    x = torch.randn(50, 8)
    _, z, _ = m(x)
    total_active = int((z != 0).sum())
    assert total_active == 4 * 50  # exactly k * n_rows kept across the batch


def test_eval_uses_fixed_threshold_batch_independent():
    torch.manual_seed(0)
    m = TopKSAE(input_dim=8, latent_dim=64, k=4, batch_topk=True, pre_bias=False)
    m.train()
    for _ in range(5):
        m(torch.randn(64, 8))  # warm up the EMA threshold
    m.eval()
    x = torch.randn(20, 8)
    _, z_full, _ = m(x)
    _, z_sub, _ = m(x[:5])
    # Eval sparsity is a per-row threshold → the first 5 rows are identical
    # whether decoded alone or as part of the larger batch.
    assert torch.allclose(z_full[:5], z_sub)


def test_reconstruction_shape_and_decoder_unit_norm():
    m = TopKSAE(input_dim=8, latent_dim=32, k=4, normalize_decoder=True)
    m.train()
    x = torch.randn(10, 8)
    recon, _, _ = m(x)
    assert recon.shape == x.shape
    norms = m.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_init_encoder_from_decoder_ties_weights():
    m = TopKSAE(input_dim=8, latent_dim=32, k=4, init_encoder_from_decoder=True, normalize_decoder=True)
    assert torch.allclose(m.encoder.weight, m.decoder.weight.t(), atol=1e-6)


def test_pre_bias_removes_decoder_bias():
    m = TopKSAE(input_dim=8, latent_dim=32, k=4, pre_bias=True)
    assert m.decoder.bias is None
    assert hasattr(m, "b_dec") and m.b_dec.shape == (8,)


def test_dead_feature_tracking():
    m = TopKSAE(input_dim=4, latent_dim=16, k=2, batch_topk=False, pre_bias=False, n_batches_to_dead=3)
    m.train()
    # Force only a subset of features to ever activate by zeroing others' encoder rows.
    with torch.no_grad():
        m.encoder.weight[8:] = 0.0
        m.encoder.bias[8:] = -1e4
    for _ in range(4):
        m(torch.randn(32, 4))
    dead = m.dead_features_mask()
    assert dead[8:].all() and not dead[:8].any()


def test_forward_on_patch_shape():
    m = TopKSAE(input_dim=64, latent_dim=128, k=8)
    m.train()
    x = torch.randn(2, 16, 16, 64)  # (B, H, W, D)
    recon, z, z_dense = m(x)
    assert recon.shape == x.shape and z.shape[:-1] == x.shape[:-1] and z.shape[-1] == 128
    assert int((z != 0).sum()) == 8 * (2 * 16 * 16)  # batch_topk over all pixels


def test_build_model_infers_batch_topk():
    m = build_model({"type": "batchtopk", "input_dim": 8, "latent_dim": 16, "k": 2})
    assert m.batch_topk
    m2 = build_model({"type": "topk", "input_dim": 8, "latent_dim": 16, "k": 2})
    assert not m2.batch_topk


def test_load_state_accepts_a_bundled_state_dict_but_not_a_missing_threshold():
    """The state-dict entry point, for checkpoints that do not live in a run dir.

    Tolerant of the derived Matryoshka buffers, and deliberately **not** tolerant
    of ``min_activation``: without it a BatchTopK model keeps its uninitialised
    threshold of -1, every latent clears it, and the "sparse" codes come back
    dense — a silent corruption of every concept metric downstream.
    """
    import pytest

    from geosae.models import DERIVED_BUFFERS, load_state

    state = dict(TopKSAE(input_dim=8, latent_dim=16, k=2, seed=0).state_dict())
    without_derived = {k: v for k, v in state.items() if k not in DERIVED_BUFFERS}
    loaded = load_state(TopKSAE(input_dim=8, latent_dim=16, k=2, seed=1), without_derived)
    assert int(loaded.n_high) == 16 and int(loaded.n_low) == 0  # rebuilt from the config

    with pytest.raises(RuntimeError, match="min_activation"):
        load_state(TopKSAE(input_dim=8, latent_dim=16, k=2), {k: v for k, v in state.items() if k != "min_activation"})
    with pytest.raises(RuntimeError, match="bundle"):
        load_state(TopKSAE(input_dim=8, latent_dim=16, k=2), {**state, "not_a_key": torch.zeros(1)}, source="bundle")

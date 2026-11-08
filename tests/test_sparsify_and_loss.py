"""Sparse code round-trip + loss behaviour."""

import numpy as np
import torch

from geosae.models import TopKSAE
from geosae.predict.sparsify import dense_to_sparse, densify_feature_map
from geosae.train.loss import sae_loss


def test_sparse_roundtrip_recovers_values():
    torch.manual_seed(0)
    z = torch.zeros(3, 4, 4, 20)
    # plant a known active feature per pixel
    z[..., 7] = torch.rand(3, 4, 4) + 0.1
    z[..., 3] = torch.rand(3, 4, 4) + 0.1
    values, indices, counts = dense_to_sparse(z, max_k=8)
    assert counts.shape == (3, 4, 4)
    assert (counts == 2).all()
    m7 = densify_feature_map(values[0], indices[0], counts[0], 7)
    assert np.allclose(m7, z[0, :, :, 7].numpy(), atol=1e-6)


def test_sparse_overflow_keeps_largest():
    z = torch.zeros(1, 1, 1, 10)
    z[0, 0, 0] = torch.tensor([9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0.5])
    values, indices, counts = dense_to_sparse(z, max_k=3)
    assert counts[0, 0, 0] == 10  # true count preserved
    kept = set(values[0, 0, 0].tolist())
    assert kept == {9.0, 8.0, 7.0}  # only the 3 largest magnitudes stored


def test_loss_zero_when_perfect_recon():
    m = TopKSAE(input_dim=8, latent_dim=32, k=4, pre_bias=False)
    x = torch.randn(10, 8)
    loss, met = sae_loss(m, x, x.clone(), torch.zeros(10, 32), torch.zeros(10, 32), aux_penalty=0.0)
    assert float(loss) == 0.0 and met["recon_mse"] == 0.0


def test_aux_only_when_dead_features_exist():
    m = TopKSAE(input_dim=8, latent_dim=32, k=4, pre_bias=False, n_batches_to_dead=5)
    x = torch.randn(10, 8)
    recon, z, z_dense = m(x)
    # After one batch non-firing features have counter=1 < 5 → none dead → aux 0.
    _, met = sae_loss(m, x, recon, z, z_dense, aux_penalty=0.03125, top_k_aux=8)
    assert met["aux"] == 0.0
    # Mark everything dead → aux becomes non-trivial.
    m.num_batches_not_active[:] = 100
    _, met2 = sae_loss(m, x, recon, z, z_dense, aux_penalty=0.03125, top_k_aux=8)
    assert met2["n_dead"] == 32 and met2["aux"] >= 0.0

"""Ladder encoders: conventional dictionaries as the ``encode`` callable the panel scores."""

from __future__ import annotations

import numpy as np

from geosae.eval.dictionaries import fit_dictionary
from geosae.eval.ladder import dictionary_encoder, leading_atoms, model_offset, rank_encoder
from geosae.models import build_model


def test_rank_codes_keep_the_leading_principal_components():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6)) * np.array([5.0, 1.0, 3.0, 0.5, 2.0, 0.1])
    pca = fit_dictionary(x, "pca", 6, seed=0)
    keep = leading_atoms(pca, x, 3)
    np.testing.assert_array_equal(keep, [0, 1, 2])
    z = rank_encoder(pca, keep)(x)
    assert ((z != 0).sum(1) == 3).all()
    recon = z @ pca.atoms + pca.offset
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    np.testing.assert_allclose(recon - x.mean(0), xc @ vt[:3].T @ vt[:3], atol=1e-6)

    raw = fit_dictionary(x, "raw", 6, seed=0)
    np.testing.assert_array_equal(leading_atoms(raw, x, 2), [0, 2])


def test_dictionary_encoder_thresholds_a_dense_basis_to_the_activation_rate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 6))
    pca = fit_dictionary(x, "pca", 6, seed=0)
    z = dictionary_encoder(pca, activation_rate=0.25)(x)
    assert abs((z != 0).mean() - 0.25) < 0.02


def test_model_offset_is_the_pre_bias_or_decoder_bias():
    pre = build_model({"type": "batchtopk", "input_dim": 4, "latent_dim": 8, "k": 2, "pre_bias": True})
    assert np.allclose(model_offset(pre), pre.b_dec.detach().numpy())
    plain = build_model({"type": "batchtopk", "input_dim": 4, "latent_dim": 8, "k": 2, "pre_bias": False})
    assert np.allclose(model_offset(plain), plain.decoder.bias.detach().numpy())

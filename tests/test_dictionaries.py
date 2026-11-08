"""Conventional dictionary estimators: unit-norm atoms, a sign, and their own codes."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.eval.dictionaries import (
    DICTIONARY_METHODS,
    fit_dictionary,
    pursuit_codes,
    rate_matched_support,
    topk_projection_code,
)

CORE_METHODS = ("pca", "varimax", "kmeans", "random")


def sparse_world(n_pixels: int = 4000, n_atoms: int = 24, dim: int = 12, seed: int = 0):
    """A known dictionary and pixels generated from it, non-negative codes."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_atoms, dim))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a = rng.random((n_pixels, n_atoms))
    codes = np.where(a < 0.15, a, 0.0)
    x = codes @ d + 0.01 * rng.normal(size=(n_pixels, dim))
    return d, codes, x


# ----------------------------------------------------------------- the contract


@pytest.mark.parametrize("method", CORE_METHODS)
def test_atoms_are_unit_norm_and_of_the_requested_width(method):
    _, _, x = sparse_world()
    fit = fit_dictionary(x, method, 8, seed=0)
    assert fit.atoms.shape == (8, x.shape[1])
    assert np.allclose(np.linalg.norm(fit.atoms, axis=1), 1.0, atol=1e-6)
    assert fit.n_atoms == 8


@pytest.mark.parametrize("method", ("pca", "varimax"))
def test_a_dense_linear_basis_saturates_at_the_embedding_dimension(method):
    _, _, x = sparse_world()
    fit = fit_dictionary(x, method, 64, seed=0)
    assert fit.n_atoms == x.shape[1]  # asked for 64, D is 12
    assert fit.n_requested == 64
    assert np.allclose(np.linalg.norm(fit.atoms, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("method", ("kmeans", "random"))
def test_an_overcomplete_method_delivers_more_atoms_than_dimensions(method):
    _, _, x = sparse_world()
    fit = fit_dictionary(x, method, 30, seed=0)
    assert fit.n_atoms == 30 > x.shape[1]


@pytest.mark.parametrize("method", CORE_METHODS)
def test_encode_returns_one_coefficient_per_atom_per_pixel(method):
    _, _, x = sparse_world()
    fit = fit_dictionary(x, method, 10, seed=0)
    z = fit.encode(x[:200])
    assert z.shape == (200, fit.n_atoms)
    assert np.isfinite(z).all()


@pytest.mark.parametrize("method", CORE_METHODS)
def test_a_fit_is_deterministic_given_its_seed(method):
    _, _, x = sparse_world()
    a = fit_dictionary(x, method, 10, seed=3)
    b = fit_dictionary(x, method, 10, seed=3)
    assert np.array_equal(a.atoms, b.atoms)


def test_an_unknown_method_is_refused():
    _, _, x = sparse_world()
    with pytest.raises(ValueError, match="unknown"):
        fit_dictionary(x, "not-a-method", 4)
    assert "pca" in DICTIONARY_METHODS


def test_a_width_below_one_is_refused():
    _, _, x = sparse_world()
    with pytest.raises(ValueError, match="n_atoms"):
        fit_dictionary(x, "pca", 0)


def test_raw_is_the_embeddings_own_axes_up_to_sign():
    _, _, x = sparse_world()
    fit = fit_dictionary(x, "raw", 64, seed=0)
    assert fit.n_atoms == x.shape[1]  # capped at D, like any dense basis
    assert np.allclose(np.abs(fit.atoms), np.eye(x.shape[1]))
    # its code is the centred pixel itself, read on those axes
    z = fit.encode(x[:50])
    assert np.allclose(z @ fit.atoms + fit.offset, x[:50])


# ------------------------------------------------------------------ correctness


def test_pca_atoms_span_the_leading_principal_subspace():
    _, _, x = sparse_world()
    fit = fit_dictionary(x, "pca", 4, seed=0)
    xc = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    # same subspace up to sign and ordering: projecting onto either leaves the
    # same residual energy
    p_fit = xc @ fit.atoms.T @ fit.atoms
    p_ref = xc @ vt[:4].T @ vt[:4]
    assert np.allclose(p_fit, p_ref, atol=1e-8)


def test_varimax_preserves_the_pca_subspace_but_rotates_within_it():
    _, _, x = sparse_world()
    pca = fit_dictionary(x, "pca", 6, seed=0)
    vmx = fit_dictionary(x, "varimax", 6, seed=0)
    # every rotated atom lies in the PCA span (the atoms themselves are no longer
    # orthogonal, so the span has to be tested by residual, not by a projector)
    residual = vmx.atoms - vmx.atoms @ pca.atoms.T @ pca.atoms
    assert np.abs(residual).max() < 1e-8
    # ... and it is a genuine rotation, not the identity
    assert not np.allclose(np.abs(pca.atoms), np.abs(vmx.atoms), atol=1e-3)


def test_varimax_sparsifies_the_loadings_it_rotates():
    """The property varimax exists for: mass concentrated in fewer entries."""
    _, _, x = sparse_world(dim=16, n_atoms=32)
    pca = fit_dictionary(x, "pca", 8, seed=0)
    vmx = fit_dictionary(x, "varimax", 8, seed=0)

    def kurtosis(a):
        s = a**2 / (a**2).sum(axis=1, keepdims=True)
        return float(np.mean((s**2).sum(axis=1)))

    assert kurtosis(vmx.atoms) > kurtosis(pca.atoms)


def test_kmeans_centroids_sit_near_the_clusters_they_are_drawn_from():
    rng = np.random.default_rng(0)
    centres = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    x = np.repeat(centres, 300, axis=0) + 0.1 * rng.normal(size=(900, 2))
    fit = fit_dictionary(x, "kmeans", 3, seed=0)
    # the fit centres on the data's own mean first, so the directions to recover
    # are the cluster centres seen from there
    shifted = centres - x.mean(axis=0)
    unit = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    # every true cluster direction is matched by some centroid
    assert (unit @ fit.atoms.T).max(axis=1).min() > 0.99


def test_kmeans_fires_exactly_one_atom_per_pixel():
    _, _, x = sparse_world()
    fit = fit_dictionary(x, "kmeans", 10, seed=0)
    z = fit.encode(x[:300])
    assert fit.sparse
    assert ((z != 0).sum(axis=1) <= 1).all()
    assert (z >= 0).all()


def test_signed_atoms_are_oriented_so_their_coefficients_are_right_skewed():
    """A free-sign basis commits to the sign a sparse non-negative code implies."""
    _, _, x = sparse_world()
    fit = fit_dictionary(x, "pca", 8, seed=0)
    z = fit.encode(x)
    zc = z - z.mean(axis=0)
    skew = (zc**3).mean(axis=0) / np.maximum(zc.std(axis=0) ** 3, 1e-12)
    assert (skew >= -1e-9).all()


def test_a_random_dictionary_ignores_the_data_it_is_handed():
    _, _, x = sparse_world()
    _, _, y = sparse_world(seed=7)
    assert np.array_equal(
        fit_dictionary(x, "random", 20, seed=1).atoms,
        fit_dictionary(y, "random", 20, seed=1).atoms,
    )


# ------------------------------------------------------------- rate matching


def test_rate_matched_support_hits_the_requested_activation_rate():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 8))
    out = rate_matched_support(z, 0.1)
    rate = (out != 0).mean(axis=0)
    assert np.allclose(rate, 0.1, atol=0.01)
    # the survivors are the largest coefficients, unchanged in value
    kept = out != 0
    assert np.array_equal(out[kept], z[kept])
    # within each column the survivors are exactly the largest coefficients
    for j in range(z.shape[1]):
        assert z[kept[:, j], j].min() > z[~kept[:, j], j].max()


def test_rate_matched_support_accepts_a_rate_per_atom():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 3))
    out = rate_matched_support(z, np.array([0.05, 0.2, 0.5]))
    assert np.allclose((out != 0).mean(axis=0), [0.05, 0.2, 0.5], atol=0.01)


def test_rate_matched_support_refuses_a_rate_outside_the_unit_interval():
    with pytest.raises(ValueError, match="rate"):
        rate_matched_support(np.zeros((10, 2)), 1.5)


# ------------------------------------------------------------ pursuit coding


def test_pursuit_recovers_a_sparse_non_negative_code_on_an_orthonormal_dictionary():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    codes = np.zeros((200, 12))
    for row in codes:
        row[rng.choice(12, size=3, replace=False)] = rng.random(3) + 0.1
    z = pursuit_codes(codes @ q, q, 3)
    assert np.allclose(z, codes, atol=1e-8)


def test_a_one_atom_pursuit_keeps_the_best_aligned_atom_at_its_projection():
    d, _, x = sparse_world()
    z = pursuit_codes(x, d, 1)
    proj = x @ d.T
    rows = np.arange(len(x))
    assert ((z != 0).sum(axis=1) <= 1).all()
    assert np.array_equal(z.argmax(axis=1)[z.max(axis=1) > 0], proj.argmax(axis=1)[z.max(axis=1) > 0])
    hit = z.max(axis=1) > 0
    assert np.allclose(z[rows[hit], proj.argmax(axis=1)[hit]], proj.max(axis=1)[hit])


def test_pursuit_codes_are_non_negative_and_at_most_k_sparse():
    d, _, x = sparse_world()
    z = pursuit_codes(x, d, 5, offset=x.mean(axis=0))
    assert (z >= 0).all()
    assert ((z != 0).sum(axis=1) <= 5).all()


def test_a_longer_pursuit_never_reconstructs_worse():
    d, _, x = sparse_world()
    err = [((x - pursuit_codes(x, d, k) @ d) ** 2).sum() for k in (1, 2, 4, 8)]
    assert all(b <= a + 1e-9 for a, b in zip(err, err[1:], strict=False))


def test_pursuit_refuses_a_non_positive_budget():
    d, _, x = sparse_world()
    with pytest.raises(ValueError, match="k"):
        pursuit_codes(x, d, 0)


def test_topk_projection_code_keeps_the_k_largest_positive_projections():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6))
    fit = fit_dictionary(x, "random", 32, seed=1)
    z = topk_projection_code(fit, x, 4)
    proj = (x - fit.offset) @ fit.atoms.T
    assert ((z > 0).sum(1) <= 4).all() and (z >= 0).all()
    for row, p in zip(z, proj, strict=True):
        kept = np.flatnonzero(row)
        np.testing.assert_allclose(row[kept], p[kept])
        assert p[kept].min() >= np.sort(p)[-4] - 1e-12


def test_raw_basis_below_full_width_keeps_the_axes_of_largest_variance():
    x = np.random.default_rng(0).normal(size=(200, 5)) * np.array([1.0, 5.0, 0.1, 3.0, 0.5])
    fit = fit_dictionary(x, "raw", 2)
    np.testing.assert_array_equal(np.flatnonzero(np.abs(fit.atoms).sum(0)), [1, 3])
    np.testing.assert_array_equal(np.abs(fit_dictionary(x, "raw", 5).atoms), np.eye(5))

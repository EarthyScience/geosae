"""The predictions zarr store: layout, dtypes, coords and stats.

The sparse code store is the interface between inference and every evaluation
metric, so its shapes, dtypes, patch order and per-patch statistics are a
contract that downstream readers depend on.
"""

import numpy as np
import pytest
import torch
import zarr

from geosae.data.dataset import PatchData
from geosae.models import TopKSAE
from geosae.models.relu_sae import ReLUSAE
from geosae.predict.sparsify import densify_feature_map
from geosae.predict.writer import estimate_max_k, run_prediction

DIM = 8
LATENT = 32
K = 4


def _model(seed: int = 0) -> TopKSAE:
    # Per-row top-k: exactly K active latents per pixel at eval, no threshold
    # calibration needed, so the store's contents are exactly predictable.
    return TopKSAE(input_dim=DIM, latent_dim=LATENT, k=K, batch_topk=False, seed=seed)


def _predict(patch_store, tmp_path, *, split="test", max_k=None, batch_size=3, allow_truncation=False):
    path, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    data = PatchData(path, split=split, compute=True, inram_dtype="float32")
    model = _model()
    out = run_prediction(
        model,
        data,
        outdir=str(tmp_path / "predictions"),
        name=split,
        split=split,
        batch_size=batch_size,
        max_k=max_k,
        allow_truncation=allow_truncation,
        device="cpu",
        amp_dtype=None,
        num_workers=0,
        pin_memory=False,
    )
    return zarr.open_group(out, mode="r"), data, model, out


def test_store_layout_dtypes_and_attrs(patch_store, tmp_path):
    g, data, model, out = _predict(patch_store, tmp_path)
    n, hw = len(data), data.patch_size
    max_k = max(2 * K, 32)

    assert out.endswith("test.zarr")
    assert g["z_values"].shape == (n, hw, hw, max_k) and g["z_values"].dtype == np.dtype("float32")
    assert g["z_indices"].shape == (n, hw, hw, max_k) and g["z_indices"].dtype == np.dtype("int16")
    assert g["z_counts"].shape == (n, hw, hw) and g["z_counts"].dtype == np.dtype("int16")
    assert g["feat_patch_stats"].shape == (LATENT, n, 2)
    attrs = dict(g.attrs)
    for key, want in (
        ("latent_dim", LATENT),
        ("k", K),
        ("max_k", max_k),
        ("patch_size", hw),
        ("n_patches", n),
        ("split", "test"),
        ("embedding_name", "alphaearth"),
    ):
        assert attrs[key] == want


def test_max_k_defaults_to_twice_k_and_is_overridable(patch_store, tmp_path):
    g, _, _, _ = _predict(patch_store, tmp_path)
    assert g.attrs["max_k"] == max(2 * K, 32)
    g2, _, _, _ = _predict(patch_store, tmp_path / "b", max_k=8)
    assert g2.attrs["max_k"] == 8 and g2["z_values"].shape[-1] == 8


def test_coords_follow_dataset_order_across_batches(patch_store, tmp_path):
    # batch_size=3 with 6 test cells: coords must not be scrambled at batch seams.
    g, data, _, _ = _predict(patch_store, tmp_path, batch_size=3)
    assert g["cell"][:].tolist() == data.cells.tolist()
    assert np.allclose(g["lon"][:], data.lon, atol=1e-5)
    assert np.allclose(g["lat"][:], data.lat, atol=1e-5)
    assert set(np.unique(g["fold"][:]).tolist()) == {2}  # test == 2


def test_fold_label_matches_split(patch_store, tmp_path):
    for split, fold in (("train", 0), ("val", 1), ("test", 2)):
        g, _, _, _ = _predict(patch_store, tmp_path / split, split=split)
        assert set(np.unique(g["fold"][:]).tolist()) == {fold}
        assert g.attrs["split"] == split


def test_sparse_codes_reconstruct_the_dense_activations(patch_store, tmp_path):
    """The stored triple must densify back to the model's dense code exactly."""
    g, data, model, _ = _predict(patch_store, tmp_path)
    model.eval()
    with torch.no_grad():
        _, z, _ = model(next(data.iter_batches(len(data), shuffle=False))["x"])
    z = z.reshape(len(data), data.patch_size, data.patch_size, LATENT).numpy()

    values, indices, counts = g["z_values"][:], g["z_indices"][:], g["z_counts"][:]
    # k=4 of 32 latents per pixel, well under max_k=32: nothing is dropped.
    assert counts.max() <= g.attrs["max_k"]
    assert np.array_equal(counts, np.count_nonzero(z, axis=-1).astype("int16"))
    for latent in range(LATENT):
        dense = np.stack([densify_feature_map(values[i], indices[i], counts[i], latent) for i in range(len(data))])
        assert np.allclose(dense, z[..., latent], atol=1e-6)


def test_feat_patch_stats_are_mean_and_active_count(patch_store, tmp_path):
    g, data, model, _ = _predict(patch_store, tmp_path)
    model.eval()
    with torch.no_grad():
        _, z, _ = model(next(data.iter_batches(len(data), shuffle=False))["x"])
    z = z.reshape(len(data), data.patch_size**2, LATENT).numpy()

    stats = g["feat_patch_stats"][:]  # (M, N, 2)
    assert np.allclose(stats[..., 0], z.mean(1).T, atol=1e-5)
    assert np.array_equal(stats[..., 1], (z > 0).sum(1).T.astype("float32"))


def test_overflowing_max_k_keeps_the_largest_by_magnitude(patch_store, tmp_path):
    """With max_k below the true L0, counts stay truthful and the top-max_k survive.

    Truncation is opt-in (see below); this pins what it does when asked for.
    """
    g, data, model, _ = _predict(patch_store, tmp_path, max_k=2, allow_truncation=True)
    model.eval()
    with torch.no_grad():
        _, z, _ = model(next(data.iter_batches(len(data), shuffle=False))["x"])
    z = z.reshape(len(data), data.patch_size, data.patch_size, LATENT).numpy()

    counts = g["z_counts"][:]
    assert np.array_equal(counts, np.count_nonzero(z, axis=-1).astype("int16"))
    assert counts.max() > 2  # the store really is overflowing
    kept = np.sort(np.abs(g["z_values"][:]), axis=-1)[..., ::-1]
    expected = np.sort(np.abs(z), axis=-1)[..., ::-1][..., :2]
    assert np.allclose(kept, expected, atol=1e-6)


def test_writer_records_no_fixed_budget_as_minus_one(tmp_path):
    """A ReLU+l1 SAE has no k, and the store must say so rather than crash."""
    import zarr

    from geosae.predict.writer import SparseWriter

    w = SparseWriter(
        str(tmp_path),
        "test",
        n_patches=4,
        patch_size=1,
        latent_dim=8,
        max_k=8,
        k=None,
        embedding_name="emb",
        split="test",
        chunk_cells=2,
    )
    assert zarr.open_group(str(w.path), mode="r").attrs["k"] == -1


# ------------------------------------------------- truncation is never silent
def test_prediction_refuses_to_truncate_by_default(patch_store, tmp_path):
    """A max_k below a token's active count must fail, not quietly drop features.

    BatchTopK picks its top-k over the whole batch, so per-token L0 has a right
    tail well above k; a store padded to a derived `max(2k, 32)` can therefore
    silently hold a subset of the code, and every concept metric reads that store.
    """
    with pytest.raises(ValueError, match="max_k"):
        _predict(patch_store, tmp_path, max_k=2)  # K=4 active per pixel


def test_truncation_is_opt_in_and_recorded_in_the_store(patch_store, tmp_path):
    g, _, _, _ = _predict(patch_store, tmp_path, max_k=2, allow_truncation=True)
    assert g.attrs["truncated"] is True
    assert g.attrs["n_truncated_tokens"] > 0
    assert g.attrs["max_active_observed"] == K


def test_an_untruncated_store_records_the_observed_maximum(patch_store, tmp_path):
    g, _, _, _ = _predict(patch_store, tmp_path, max_k=16)
    assert g.attrs["truncated"] is False
    assert g.attrs["n_truncated_tokens"] == 0
    assert g.attrs["max_active_observed"] == K


# --------------------------------------------------- data-driven max_k ("auto")
def _relu_model_with_fixed_active_count(n_active: int, latent_dim: int = 200, input_dim: int = 4) -> ReLUSAE:
    """A ReLU SAE whose pre-activations fire the same ``n_active`` latents on every input."""
    model = ReLUSAE(input_dim=input_dim, latent_dim=latent_dim, normalize_inputs=False, seed=0)
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.encoder.bias[:n_active] = 1.0
    return model


def test_estimate_max_k_scales_the_observed_active_count_by_headroom(patch_store, tmp_path):
    path, _ = patch_store(n=24, patch_size=2, dim=4, chunk=6)
    data = PatchData(path, split="test", compute=True, inram_dtype="float32")
    model = _relu_model_with_fixed_active_count(60, latent_dim=200)
    est = estimate_max_k(model, data, batch_size=4, headroom=1.5, device="cpu", amp_dtype=None)
    assert est == 90  # ceil(60 * 1.5), well under the 200-wide dictionary


def test_estimate_max_k_floors_at_32_and_caps_at_latent_dim(patch_store, tmp_path):
    path, _ = patch_store(n=24, patch_size=2, dim=4, chunk=6)
    data = PatchData(path, split="test", compute=True, inram_dtype="float32")
    quiet = _relu_model_with_fixed_active_count(0, latent_dim=200)
    assert estimate_max_k(quiet, data, batch_size=4, device="cpu", amp_dtype=None) == 32
    saturated = _relu_model_with_fixed_active_count(200, latent_dim=200)
    assert estimate_max_k(saturated, data, batch_size=4, headroom=2.0, device="cpu", amp_dtype=None) == 200


def test_predict_max_k_auto_avoids_the_silent_truncation_a_fixed_32_caused(patch_store, tmp_path):
    """A model without a fixed k must not be padded to a blind constant that can
    undershoot its real L0."""
    path, _ = patch_store(n=24, patch_size=2, dim=4, chunk=6)
    data = PatchData(path, split="test", compute=True, inram_dtype="float32")
    model = _relu_model_with_fixed_active_count(60, latent_dim=200)
    out = run_prediction(
        model,
        data,
        outdir=str(tmp_path / "predictions"),
        name="test",
        split="test",
        batch_size=4,
        max_k="auto",
        device="cpu",
        amp_dtype=None,
        num_workers=0,
        pin_memory=False,
    )
    g = zarr.open_group(out, mode="r")
    assert g.attrs["truncated"] is False
    assert g.attrs["max_k"] >= g.attrs["max_active_observed"] == 60


def test_predict_max_k_auto_encodes_again_when_the_probe_underestimates(patch_store, tmp_path, monkeypatch):
    """A probe that misses the split's heaviest token must not leave a truncated store."""
    import geosae.predict.writer as W

    monkeypatch.setattr(W, "estimate_max_k", lambda *args, **kwargs: 32)
    path, _ = patch_store(n=24, patch_size=2, dim=4, chunk=6)
    data = PatchData(path, split="test", compute=True, inram_dtype="float32")
    model = _relu_model_with_fixed_active_count(60, latent_dim=200)
    out = run_prediction(
        model,
        data,
        outdir=str(tmp_path / "predictions"),
        name="test",
        split="test",
        batch_size=4,
        max_k="auto",
        device="cpu",
        amp_dtype=None,
        num_workers=0,
        pin_memory=False,
    )
    g = zarr.open_group(out, mode="r")
    assert g.attrs["truncated"] is False
    assert g.attrs["max_k"] == g.attrs["max_active_observed"] == 60


def test_predict_max_k_rejects_an_unknown_string(patch_store, tmp_path):
    with pytest.raises(ValueError, match="auto"):
        _predict(patch_store, tmp_path, max_k="bogus")

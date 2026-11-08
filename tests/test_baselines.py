"""The dense reference baselines the SAE panel is read against.

PCA at matched rank is the honest linear comparison for a k-sparse SAE, so its
fit and its streaming metrics have to be right in the same sense the SAE's are:
the leading subspace, an FVE that is monotone in rank and exactly 1 at full rank.
"""

import numpy as np
import pytest

from geosae.data.dataset import PatchData
from geosae.eval.baselines import fit_pca, pca_reconstruction_metrics, project_to_basis

DIM = 8


@pytest.fixture
def pca_data(patch_store):
    path, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    return PatchData(path, split="train", compute=True, inram_dtype="float32")


def test_components_are_an_orthonormal_basis_by_descending_variance(pca_data):
    mean, comps = fit_pca(pca_data, seed=0)
    assert mean.shape == (DIM,) and comps.shape == (DIM, DIM)
    assert np.allclose(comps @ comps.T, np.eye(DIM), atol=1e-4)

    x = np.concatenate([b["x"].reshape(-1, DIM).numpy() for b in pca_data.iter_batches(8)])
    assert np.allclose(mean, x.mean(0), atol=1e-4)
    variances = ((x - mean) @ comps.T).var(0)
    assert np.all(np.diff(variances) <= 1e-5), "components must be ordered by explained variance"


def test_leading_subspace_matches_a_planted_low_rank_signal(patch_store, tmp_path):
    """A rank-2 signal plus tiny noise: the top-2 components must span the signal."""
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0][:, :2]  # (D, 2)
    path, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    import zarr

    g = zarr.open_group(path, mode="a")
    coeffs = rng.standard_normal((24, 4, 4, 2)) * np.array([10.0, 5.0])
    g["alphaearth"][:] = (coeffs @ basis.T + 1e-3 * rng.standard_normal((24, 4, 4, DIM))).astype("float32")

    data = PatchData(path, split="train", compute=True, inram_dtype="float32")
    _, comps = fit_pca(data, seed=0)
    # Principal angles: the top-2 PCs and the planted basis span the same plane.
    singular = np.linalg.svd(comps[:2] @ basis, compute_uv=False)
    assert np.allclose(singular, 1.0, atol=1e-3)


def test_fve_is_monotone_in_rank_and_exact_at_full_rank(pca_data):
    mean, comps = fit_pca(pca_data, seed=0)
    fve = [
        pca_reconstruction_metrics(pca_data, mean, comps, k, batch_size=8, device="cpu")["fve"] for k in (1, 2, 4, DIM)
    ]
    assert all(a <= b + 1e-6 for a, b in zip(fve[:-1], fve[1:], strict=True))
    assert fve[-1] == pytest.approx(1.0, abs=1e-6), "full-rank PCA reconstructs a D-dim embedding exactly"

    full = pca_reconstruction_metrics(pca_data, mean, comps, DIM, batch_size=8, device="cpu")
    assert full["recon_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert full["recon_mse"] == pytest.approx(0.0, abs=1e-8)
    assert full["l0"] == DIM and full["n_components"] == DIM
    assert full["n_pixels"] == len(pca_data) * pca_data.patch_size**2


def test_metrics_match_a_direct_recomputation(pca_data):
    mean, comps = fit_pca(pca_data, seed=0)
    got = pca_reconstruction_metrics(pca_data, mean, comps, 3, batch_size=8, device="cpu")

    x = np.concatenate([b["x"].reshape(-1, DIM).numpy() for b in pca_data.iter_batches(8)]).astype(np.float64)
    vk = comps[:3].astype(np.float64)
    recon = (x - mean) @ vk.T @ vk + mean
    expected_fve = (1.0 - ((x - recon) ** 2).mean(0) / x.var(0)).mean()
    assert got["fve"] == pytest.approx(expected_fve, abs=1e-6)
    cos = (x * recon).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(recon, axis=-1))
    assert got["recon_cosine"] == pytest.approx(cos.mean(), abs=1e-6)


def test_raw_basis_is_the_identity():
    raw = np.random.default_rng(0).standard_normal((200, DIM))
    assert np.array_equal(project_to_basis(raw, "raw"), raw)


def test_pca_basis_is_a_full_rank_decorrelating_rotation():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((DIM, DIM))
    raw = rng.standard_normal((400, DIM)) @ a + 3.0  # correlated, off-centre

    out = project_to_basis(raw, "pca")
    assert out.shape == raw.shape
    # A rotation of the centred data: pairwise distances survive, columns decorrelate.
    centred = raw - raw.mean(0)
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(centred, axis=1), atol=1e-8)
    corr = np.cov(out, rowvar=False)
    assert np.allclose(corr - np.diag(np.diag(corr)), 0.0, atol=1e-8)
    assert np.all(np.diff(np.diag(corr)) <= 1e-8)  # descending variance


def test_unknown_basis_is_rejected():
    with pytest.raises(ValueError, match="unknown basis"):
        project_to_basis(np.zeros((4, DIM)), "umap")


# ---------------------------------------------------------------------------
# Fitting tables: one pixel per patch
# ---------------------------------------------------------------------------


@pytest.fixture
def fit_data(patch_store):
    path, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    return PatchData(path, split="train", compute=True, inram_dtype="float32")


def test_fit_table_draws_one_pixel_from_every_patch(fit_data):
    from geosae.eval.baselines import build_fit_table

    table = build_fit_table(fit_data, seed=0, batch_size=5)
    assert len(table) == len(fit_data)
    assert table.pixels.shape == (len(fit_data), DIM)
    assert np.array_equal(table.cell, fit_data.cells)
    assert table.meta["pixels_per_patch"] == 1 and table.meta["seed"] == 0


def test_fit_table_rows_are_store_values_of_the_patch_they_name(fit_data, patch_store):
    """Every row must be a pixel that actually sits in the patch it is attributed to."""
    from geosae.eval.baselines import build_fit_table

    table = build_fit_table(fit_data, seed=3, batch_size=7)
    raw = fit_data._read_cells(np.arange(len(fit_data)))  # (n, H, W, D) store values
    for i in range(len(table)):
        patch = raw[i].reshape(-1, DIM)
        assert np.isclose(patch, table.pixels[i]).all(axis=1).any()


def test_fit_table_is_seeded_per_patch_not_per_walk(fit_data):
    """A cell yields the same pixel however many other cells were walked."""
    from geosae.eval.baselines import build_fit_table

    full = build_fit_table(fit_data, seed=1, batch_size=4)
    keep = fit_data.cells[::3]
    part = build_fit_table(fit_data, seed=1, batch_size=4, cells=keep)
    assert np.array_equal(part.cell, keep)
    assert np.array_equal(part.pixels, full.pixels[::3])


def test_fit_table_takes_several_pixels_per_patch_without_replacement(fit_data):
    from geosae.eval.baselines import build_fit_table

    table = build_fit_table(fit_data, pixels_per_patch=3, seed=0, batch_size=6)
    assert len(table) == 3 * len(fit_data)
    assert np.array_equal(table.cell, np.repeat(fit_data.cells, 3))
    for i in range(len(fit_data)):
        block = table.pixels[3 * i : 3 * i + 3]
        assert len({tuple(row) for row in block}) == 3


def test_fit_table_round_trips_through_npz(fit_data, tmp_path):
    from geosae.eval.baselines import build_fit_table, load_fit_table

    table = build_fit_table(fit_data, seed=2, batch_size=8)
    back = load_fit_table(table.to_npz(tmp_path / "fit.npz"))
    assert np.array_equal(back.pixels, table.pixels)
    assert np.array_equal(back.cell, table.cell)
    assert back.meta == table.meta


def test_fit_table_pixels_carry_the_dataset_transforms(patch_store):
    """A table holds store values; a caller gets them as its own dataset delivers them."""
    from geosae.data.dataset import Transforms
    from geosae.eval.baselines import build_fit_table

    path, _ = patch_store(n=24, patch_size=4, dim=DIM, chunk=6)
    plain = PatchData(path, split="train", compute=True, inram_dtype="float32")
    shift = np.arange(DIM, dtype="float32")
    scaled = PatchData(
        path,
        split="train",
        compute=True,
        inram_dtype="float32",
        transforms=Transforms(mean=shift, std=np.full(DIM, 2.0, dtype="float32")),
    )
    table = build_fit_table(plain, seed=0, batch_size=8)
    assert np.allclose(table.pixels_for(scaled), (table.pixels - shift) / 2.0, atol=1e-6)
    assert np.allclose(table.pixels_for(plain), table.pixels, atol=1e-6)


def test_per_patch_fitting_draw_spends_its_budget_on_distinct_patches(fit_data):
    """The point of the decorrelated draw: as many source patches as pixels."""
    from geosae.eval.baselines import baseline_fit_pixels

    n = 8
    per_patch = baseline_fit_pixels(fit_data, n_pixels=n, sampling="per_patch", seed=0)
    whole = baseline_fit_pixels(fit_data, n_pixels=n, sampling="whole_patch", seed=0)
    assert per_patch.shape == (n, DIM) and whole.shape == (n, DIM)
    # Whole patches: 4x4 = 16 pixels, so 8 pixels come from a single patch.
    assert _n_source_patches(fit_data, per_patch) == n
    assert _n_source_patches(fit_data, whole) == 1


def _n_source_patches(data: PatchData, pixels: np.ndarray) -> int:
    raw = data._read_cells(np.arange(len(data))).reshape(len(data), -1, pixels.shape[1])
    hits = {i for i in range(len(data)) for row in pixels if np.isclose(raw[i], row, atol=1e-6).all(axis=1).any()}
    return len(hits)


def test_per_patch_draw_caps_at_the_split_and_then_widens_the_draw(fit_data):
    """More pixels than patches: the table takes several per patch rather than repeating one."""
    from geosae.eval.baselines import baseline_fit_pixels

    x = baseline_fit_pixels(fit_data, n_pixels=2 * len(fit_data), sampling="per_patch", seed=0)
    assert x.shape == (2 * len(fit_data), DIM)
    assert len({tuple(r) for r in x}) == 2 * len(fit_data)


def test_unknown_fit_sampling_is_rejected(fit_data):
    from geosae.eval.baselines import baseline_fit_pixels

    with pytest.raises(ValueError, match="unknown fit sampling"):
        baseline_fit_pixels(fit_data, n_pixels=4, sampling="every_other")


def test_a_cached_table_is_reused_and_a_mismatched_one_is_rejected(fit_data, tmp_path):
    from geosae.eval.baselines import baseline_fit_pixels, load_fit_table

    path = tmp_path / "cache.npz"
    first = baseline_fit_pixels(fit_data, n_pixels=6, sampling="per_patch", seed=0, table_path=path)
    assert path.exists()
    stamp = path.stat().st_mtime_ns
    again = baseline_fit_pixels(fit_data, n_pixels=6, sampling="per_patch", seed=0, table_path=path)
    assert np.array_equal(first, again)
    assert path.stat().st_mtime_ns == stamp, "a matching cache must not be rebuilt"

    load_fit_table(path).to_npz(path)  # unchanged file, but now ask for a different recipe
    with pytest.raises(ValueError, match="does not match"):
        baseline_fit_pixels(fit_data, n_pixels=6, sampling="per_patch", seed=99, table_path=path)

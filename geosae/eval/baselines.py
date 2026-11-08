"""Conventional baselines: the pixels they are fitted on, PCA reconstruction and dense-basis projections.

The honest "equal dimensionality" comparison for a ``k``-sparse SAE is PCA at a
matched rank: a ``k``-sparse code over ``M`` atoms stores ``k`` coefficients per
pixel, so ``PCA(k)`` (a dense ``k``-dim linear code) is the matched-budget linear
baseline. PCA at full rank (``= D``) reconstructs a ``D``-dim embedding
*perfectly* (FVE 1.0), so it is reported only as a sanity check — the meaningful
row is ``PCA(k)``.

Fit is a plain SVD on a sample of training pixels (no sklearn); evaluation streams
the test split so it scales. Reconstruction ``\\hat x = mean + (x-mean) V_k V_k^T``.

**Which pixels each number rests on** — a baseline is only a baseline if it is
scored on the population the thing it baselines was:

* :func:`pca_reconstruction_metrics` streams the **whole** split, a census rather
  than a sample, and :func:`fit_pca` fits on a seeded draw of *training* pixels,
  so no test pixel enters the fit.
* :func:`dense_basis_auroc_matrix` scores whatever ``raw`` and whatever
  ``balanced`` draws it is handed. Given
  :meth:`~geosae.eval.pool.EvalPool.registry_sample` and
  :meth:`~geosae.eval.pool.EvalPool.balanced_indices`, a dense basis rests on
  exactly the pixels the SAE cells beside it do, which is what makes the two
  comparable.
* :func:`project_to_basis` fits ICA and NMF on a seeded ``fit_subsample`` of the
  rows it is given and then applies the result to **all** of them: that draw is
  part of the estimator, not of the evaluation population.

**How many patches a fit rests on.** :func:`baseline_fit_pixels` is the entry point
for every fitting draw, and its ``sampling`` argument is the choice. ``"per_patch"``
spends the pixel budget on as many distinct patches as there are pixels, one drawn
from each, so the fit sees the whole geographic spread of the split; a store chunked
by patch makes that a walk over every chunk, so it is built once into a
:class:`FitTable` and cached. ``"whole_patch"`` instead reads
``ceil(n / patch_size**2)`` complete patches and thins them, which costs a fraction
of the I/O and gives a fitting sample whose pixels are strongly spatially correlated:
at ``n = 100,000`` and ``64 x 64`` patches it is 25 sites.

:func:`patch_pixel_matrix` is the patch-major reader the whole-patch draw and the
code-side diagnostics share; its layout matches
:func:`~geosae.predict.sparsify.read_dense_codes` so embeddings and codes line up row
for row. :mod:`geosae.eval.dictionaries` fits the conventional dictionaries the
panel scores beside the SAEs.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from geosae.data.dataset import PatchData

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: The fitting draws :func:`baseline_fit_pixels` knows.
FIT_SAMPLINGS = ("per_patch", "whole_patch")


def patch_pixel_matrix(data: PatchData, patches: np.ndarray, *, pixel_stride: int = 1) -> np.ndarray:
    """Pixels of the given patches as ``(n_patches * n_kept, D)``, patch-major.

    The layout :func:`~geosae.predict.sparsify.read_dense_codes` writes, pixel for
    pixel: same patch order, same within-patch flattening, same stride. That is
    what lets an embedding matrix and a code matrix be handed to a code-side
    diagnostic as aligned rows.

    Args:
        data: Open patch dataset.
        patches: Local patch indices into ``data`` (positions in ``data.cells``).
        pixel_stride: Keep every ``pixel_stride``-th pixel of each patch.

    Returns:
        ``(n_patches * n_kept, D)`` float64 pixels.
    """
    d = data.embedding_dim
    stride = int(pixel_stride)
    out = [data._make_batch(np.array([int(p)]))["x"].reshape(-1, d).numpy()[::stride] for p in np.asarray(patches)]
    return np.concatenate(out, axis=0).astype(np.float64)


def sample_pixels(data: PatchData, *, n_pixels: int = 500_000, seed: int = 0) -> np.ndarray:
    """A seeded draw of ``(n, D)`` pixels from whole patches of ``data``.

    Whole patches are drawn and then thinned to the requested count, which costs the
    least I/O a store chunked by patch allows. The pixels of one patch are strongly
    correlated, so the draw covers ``ceil(n / patch_size**2)`` sites and no more --
    see :func:`baseline_fit_pixels` for the draw that spends the budget on distinct
    patches instead.

    Args:
        data: Open patch dataset.
        n_pixels: Pixels wanted. Fewer come back only if the split is smaller.
        seed: Seeds the patch draw and the thinning.

    Returns:
        ``(n, D)`` float64 pixels, ``n <= n_pixels``.
    """
    rng = np.random.default_rng(seed)
    per_patch = data.patch_size**2
    n_patches = min(len(data), max(1, int(np.ceil(n_pixels / per_patch))))
    sel = np.sort(rng.choice(len(data), size=n_patches, replace=False))
    x = patch_pixel_matrix(data, sel)
    if x.shape[0] > n_pixels:
        x = x[np.sort(rng.choice(x.shape[0], size=n_pixels, replace=False))]
    return x


@dataclass(frozen=True)
class FitTable:
    """Pixels drawn from distinct patches of one split, with the recipe that drew them.

    The rows are **store values**, before any :class:`~geosae.data.dataset.Transforms`,
    so one table serves every config built on the same store; a caller takes them
    through its own dataset with :meth:`pixels_for`.

    Attributes:
        pixels: ``(N, D)`` float32 store values, patch-major.
        cell: ``(N,)`` source cell id in the store, one entry per row.
        meta: The draw's provenance: store, embedding, split, seed,
            ``pixels_per_patch`` and the number of patches walked.
    """

    pixels: np.ndarray
    cell: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.pixels.shape[0])

    def pixels_for(self, data: PatchData) -> np.ndarray:
        """The rows as ``data`` would deliver them: its input transforms applied.

        Args:
            data: The dataset whose transforms the fit is to be carried out under.

        Returns:
            ``(N, D)`` float64 pixels.
        """
        x = data.transforms.apply(torch.from_numpy(self.pixels).float())
        return x.numpy().astype(np.float64)

    def to_npz(self, path: str | Path) -> str:
        """Save to a single compressed ``.npz`` (round-trips through :func:`load_fit_table`)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pixels=self.pixels, cell=self.cell, meta=np.asarray(json.dumps(self.meta)))
        return str(path)


def load_fit_table(path: str | Path) -> FitTable:
    """Load a :class:`FitTable` written by :meth:`FitTable.to_npz`."""
    z = np.load(path, allow_pickle=False)
    return FitTable(pixels=z["pixels"], cell=z["cell"], meta=json.loads(str(z["meta"])))


def build_fit_table(
    data: PatchData,
    *,
    pixels_per_patch: int = 1,
    seed: int = 0,
    batch_size: int = 64,
    cells: np.ndarray | None = None,
    progress_every: int = 0,
) -> FitTable:
    """Draw ``pixels_per_patch`` seeded pixels from every patch of a split.

    Walks the split in cell order and reads each patch once, which is the cheapest
    way to touch every patch of a store chunked by patch. The walk is I/O bound and
    reads run on ``data.n_load_workers`` threads; the draw is seeded **per cell**, so
    a cell yields the same pixels whatever order the reads complete in, whether or not
    the rest of the split was walked, and a restricted walk selects rows of the full
    table instead of redrawing it.

    Args:
        data: Open patch dataset for the split to walk.
        pixels_per_patch: Pixels taken from each patch, without replacement.
        seed: Seeds the per-cell draw.
        batch_size: Patches per read. A multiple of the store's chunk length keeps
            each chunk to a single read.
        cells: Store cell ids to restrict the walk to; ``None`` walks the split.
        progress_every: Print every N batches (0 to silence).

    Returns:
        A :class:`FitTable` with ``len(cells) * pixels_per_patch`` rows in cell order.

    Raises:
        ValueError: If ``pixels_per_patch`` exceeds the pixels a patch holds.
    """
    per_patch = int(pixels_per_patch)
    n_in_patch = data.patch_size**2
    if not 1 <= per_patch <= n_in_patch:
        raise ValueError(f"pixels_per_patch must lie in 1..{n_in_patch}, got {per_patch}")

    local = np.arange(len(data))
    if cells is not None:
        wanted = np.asarray(cells, dtype=np.int64)
        local = np.flatnonzero(np.isin(data.cells, wanted))
    n_batches = max(1, int(np.ceil(local.size / batch_size)))

    def _draw(idx: np.ndarray) -> np.ndarray:
        raw = data._read_cells(idx).reshape(idx.size, n_in_patch, data.embedding_dim)
        take = [
            np.random.default_rng([seed, int(c)]).choice(n_in_patch, size=per_patch, replace=False)
            for c in data.cells[idx]
        ]
        return np.concatenate([raw[i, t] for i, t in enumerate(take)], axis=0)

    batches = [local[s : s + batch_size] for s in range(0, local.size, batch_size)]
    rows: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=max(1, data.n_load_workers)) as pool:
        for bi, block in enumerate(pool.map(_draw, batches)):  # `map` yields in submission order
            rows.append(block)
            if progress_every and (bi % progress_every == 0):
                print(f"[geosae] fit table: batch {bi}/{n_batches}, {sum(len(r) for r in rows)} pixels", flush=True)

    pixels = (
        np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, data.embedding_dim), dtype=np.float32)
    )
    return FitTable(
        pixels=pixels,
        cell=np.repeat(data.cells[local], per_patch),
        meta={
            "sampling": "per_patch",
            "zarr_path": data.zarr_path,
            "embedding_name": data.embedding_name,
            "split_name": data.split_name,
            "split": str(data.split),
            "seed": int(seed),
            "pixels_per_patch": per_patch,
            "n_patches": int(local.size),
            "embedding_dim": int(data.embedding_dim),
            "patch_size": int(data.patch_size),
        },
    )


def baseline_fit_pixels(
    data: PatchData,
    *,
    n_pixels: int,
    seed: int = 0,
    sampling: str = "per_patch",
    table_path: str | Path | None = None,
    batch_size: int = 64,
    progress_every: int = 0,
) -> np.ndarray:
    """The pixels a conventional decomposition is fitted on, under a stated draw.

    Args:
        data: Open patch dataset for the *training* split.
        n_pixels: Pixels wanted. Fewer come back only if the split is smaller.
        seed: Seeds the draw.
        sampling: One of :data:`FIT_SAMPLINGS`. ``"per_patch"`` spends the budget on
            as many distinct patches as it can, ``"whole_patch"`` reads whole patches
            and thins them.
        table_path: Where a ``"per_patch"`` table is cached. An existing file whose
            recipe matches is read instead of walking the split; a missing one is
            built and written. ``None`` builds in memory and keeps nothing.
        batch_size: Patches per read while building.
        progress_every: Print every N batches while building (0 to silence).

    Returns:
        ``(n, D)`` float64 pixels under ``data``'s own input transforms.

    Raises:
        ValueError: If ``sampling`` is unknown, or a cached table's recipe differs
            from the one asked for.
    """
    if sampling == "whole_patch":
        return sample_pixels(data, n_pixels=int(n_pixels), seed=int(seed))
    if sampling != "per_patch":
        raise ValueError(f"unknown fit sampling {sampling!r}, expected one of {FIT_SAMPLINGS}")

    per_patch = max(1, int(np.ceil(int(n_pixels) / max(len(data), 1))))
    wanted = {
        "sampling": "per_patch",
        "zarr_path": data.zarr_path,
        "embedding_name": data.embedding_name,
        "split_name": data.split_name,
        "split": str(data.split),
        "seed": int(seed),
        "pixels_per_patch": per_patch,
        "n_patches": len(data),
    }
    table = None
    if table_path is not None and Path(table_path).exists():
        table = load_fit_table(table_path)
        differ = {k: (v, table.meta.get(k)) for k, v in wanted.items() if table.meta.get(k) != v}
        if differ:
            raise ValueError(f"cached fit table {table_path} does not match the recipe asked for: {differ}")
    if table is None:
        table = build_fit_table(
            data, pixels_per_patch=per_patch, seed=seed, batch_size=batch_size, progress_every=progress_every
        )
        if table_path is not None:
            table.to_npz(table_path)

    x = table.pixels_for(data)
    if x.shape[0] > int(n_pixels):
        keep = np.sort(np.random.default_rng(seed).choice(x.shape[0], size=int(n_pixels), replace=False))
        x = x[keep]
    return x


def fit_pca(data: PatchData, *, n_sample_pixels: int = 500_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on a random sample of pixels. Returns ``(mean (D,), components (D, D))``.

    ``components`` rows are ordered by descending explained variance; take the
    first ``k`` rows for a rank-``k`` projector.
    """
    X = sample_pixels(data, n_pixels=n_sample_pixels, seed=seed)
    mean = X.mean(axis=0)
    Xc = X - mean
    # Right singular vectors are the principal directions.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mean.astype(np.float32), Vt.astype(np.float32)  # (D,), (D, D)


@torch.no_grad()
def pca_reconstruction_metrics(
    data: PatchData,
    mean: np.ndarray,
    components: np.ndarray,
    n_components: int,
    *,
    batch_size: int = 64,
    device: str | torch.device = DEFAULT_DEVICE,
) -> dict[str, float]:
    """Streaming FVE + reconstruction-cosine for a rank-``n_components`` PCA."""
    dev = torch.device(device)
    D = data.embedding_dim
    mu = torch.as_tensor(mean, dtype=torch.float64, device=dev)
    Vk = torch.as_tensor(components[:n_components], dtype=torch.float64, device=dev)  # (k, D)

    sum_x = torch.zeros(D, dtype=torch.float64, device=dev)
    sum_x2 = torch.zeros(D, dtype=torch.float64, device=dev)
    sum_res2 = torch.zeros(D, dtype=torch.float64, device=dev)
    cos_sum = torch.zeros((), dtype=torch.float64, device=dev)
    n_pixels = 0
    for batch in data.iter_batches(batch_size, shuffle=False):
        x = batch["x"].to(dev).reshape(-1, D).double()
        xc = x - mu
        recon = (xc @ Vk.t()) @ Vk + mu  # project onto top-k subspace, lift back
        sum_x += x.sum(0)
        sum_x2 += (x * x).sum(0)
        sum_res2 += ((x - recon) ** 2).sum(0)
        xn = x.norm(dim=-1)
        rn = recon.norm(dim=-1)
        cos_sum += ((x * recon).sum(-1) / (xn * rn).clamp_min(1e-8)).sum()
        n_pixels += x.shape[0]

    n = float(n_pixels)
    mean_x = sum_x / n
    var_x = (sum_x2 / n) - mean_x**2
    fve = 1.0 - (sum_res2 / n / var_x.clamp_min(1e-12))
    return {
        "fve": float(fve.mean()),
        "recon_cosine": float(cos_sum / n),
        "recon_mse": float((sum_res2 / n).mean()),
        "n_components": int(n_components),
        "l0": int(n_components),
        "n_pixels": int(n_pixels),
    }


def project_to_basis(raw: np.ndarray, basis: str, *, seed: int = 0, fit_subsample: int = 30_000) -> np.ndarray:
    """Project a raw pixel matrix ``(N, D)`` to a dense factor basis at matched rank ``D``.

    These dense reference bases are scored identically to the SAEs, so their
    metrics are directly comparable. ``"raw"`` is the identity (score the
    embedding itself); ``"pca"`` is a full-rank numpy SVD projection (no
    sklearn); ``"ica"`` and ``"nmf"`` use scikit-learn
    and are fit on a random ``fit_subsample`` of rows for tractability, then
    applied to all rows.

    Args:
        raw: ``(N, D)`` per-pixel embedding.
        basis: one of ``{"raw", "pca", "ica", "nmf"}``.
        seed: RNG seed for the fit subsample and the estimators.
        fit_subsample: rows used to fit ICA/NMF (PCA/raw use all rows).
    """
    raw = np.asarray(raw)
    if basis == "raw":
        return raw
    if basis == "pca":
        mean = raw.mean(axis=0)
        _, _, vt = np.linalg.svd(raw - mean, full_matrices=False)
        return (raw - mean) @ vt.T  # full-rank rotation into principal axes
    rng = np.random.default_rng(seed)
    fit_x = raw
    if raw.shape[0] > fit_subsample:
        fit_x = raw[rng.choice(raw.shape[0], size=fit_subsample, replace=False)]
    if basis == "ica":
        from sklearn.decomposition import FastICA

        ica = FastICA(n_components=raw.shape[1], random_state=seed, max_iter=500, tol=1e-4, whiten="unit-variance").fit(
            fit_x.astype(np.float32)
        )
        return ica.transform(raw.astype(np.float32))
    if basis == "nmf":
        from sklearn.decomposition import NMF

        shift = fit_x.min(axis=0).astype(np.float32) - 1e-6
        nmf = NMF(n_components=raw.shape[1], init="nndsvd", random_state=seed, max_iter=400, tol=1e-4).fit(
            (fit_x - shift).astype(np.float32)
        )
        pinv = np.linalg.pinv(nmf.components_.T).astype(np.float32)  # linear-pseudoinverse encoder
        return (raw - shift).astype(np.float32) @ pinv.T
    raise ValueError(f"unknown basis {basis!r}")


def dense_basis_auroc_matrix(raw, labels, class_table, product_order, basis, *, seed: int = 0, **kw):
    """AUROC matrix for a dense reference basis, scored exactly like an SAE.

    Projects ``raw`` to ``basis`` (:func:`project_to_basis`) and runs the same
    :func:`geosae.eval.coverage.concept_auroc_matrix`, so the resulting UCov/EMF are
    directly comparable to the SAE cells.
    """
    from geosae.eval.coverage import concept_auroc_matrix

    feats = project_to_basis(np.asarray(raw), basis, seed=seed)
    return concept_auroc_matrix(feats, labels, class_table, product_order, **kw)


__all__ = [
    "FIT_SAMPLINGS",
    "FitTable",
    "baseline_fit_pixels",
    "build_fit_table",
    "load_fit_table",
    "fit_pca",
    "pca_reconstruction_metrics",
    "project_to_basis",
    "dense_basis_auroc_matrix",
]

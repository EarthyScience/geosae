"""Dense -> variable-sparse conversion for SAE codes.

BatchTopK produces a *variable* number of active latents per pixel, so codes are
stored as padded ``(values, indices, counts)`` triples: for each pixel the
``max_k`` largest-magnitude non-zero activations, their latent ids, and the true
non-zero count (which may exceed ``max_k`` — the overflow is dropped and warned).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import scipy.sparse as sp
    import zarr

logger = logging.getLogger("geosae.sparsify")


@torch.no_grad()
def dense_to_sparse(z: torch.Tensor, max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract up to ``max_k`` non-zero activations per row of ``z`` ``(..., M)``.

    Returns numpy ``(values[..., max_k] f32, indices[..., max_k] int16,
    counts[...] int16)``; indices widen to int32 when ``M`` exceeds the int16 range. Runs the
    top-k on ``z``'s device (GPU-friendly) and moves only the small result to host.
    """
    leading = z.shape[:-1]
    M = z.shape[-1]
    flat = z.reshape(-1, M)
    eff_k = min(max_k, M)

    counts = torch.count_nonzero(flat, dim=-1).to(torch.int32)
    vals, idx = torch.topk(flat.abs(), eff_k, dim=-1, sorted=False)
    vals = torch.gather(flat, 1, idx)  # signed values at those positions
    nz = vals != 0
    vals = torch.where(nz, vals, torch.zeros((), dtype=vals.dtype, device=z.device))
    idx = torch.where(nz, idx, torch.zeros((), dtype=idx.dtype, device=z.device))

    if eff_k < max_k:
        pad = max_k - eff_k
        vals = torch.nn.functional.pad(vals, (0, pad))
        idx = torch.nn.functional.pad(idx, (0, pad))

    values = vals.reshape(*leading, max_k).float().cpu().numpy()
    index_dtype = torch.int16 if M <= np.iinfo(np.int16).max + 1 else torch.int32
    indices = idx.reshape(*leading, max_k).to(index_dtype).cpu().numpy()
    counts = counts.reshape(leading).to(torch.int16).cpu().numpy()
    return values, indices, counts


def sparse_to_dense(values: np.ndarray, indices: np.ndarray, n_features: int) -> np.ndarray:
    """Rebuild the dense ``(..., n_features)`` code from a padded sparse triple.

    A slot is real iff its value is non-zero, which is what makes this the exact
    inverse of :func:`dense_to_sparse`: that top-k is unsorted, so a pixel's
    occupied slots are *not* its leading ones and a position-versus-count rule
    would clear genuine activations.

    Args:
        values: ``(..., max_k)`` stored activations.
        indices: ``(..., max_k)`` latent id per slot.
        n_features: ``M``, the dictionary width.

    Returns:
        ``(..., n_features)`` dense codes.
    """
    values = np.asarray(values)
    indices = np.asarray(indices, dtype=np.int64)
    flat_v = values.reshape(-1, values.shape[-1])
    flat_i = indices.reshape(-1, indices.shape[-1])
    out = np.zeros((flat_v.shape[0], int(n_features)), dtype=np.float32)
    valid = flat_v != 0
    row = np.broadcast_to(np.arange(flat_v.shape[0])[:, None], flat_i.shape)
    out[row[valid], flat_i[valid]] = flat_v[valid].astype(np.float32)
    return out.reshape(*values.shape[:-1], int(n_features))


def sparse_code_matrix(values: np.ndarray, indices: np.ndarray, n_features: int) -> sp.csr_matrix:
    """A padded sparse triple's codes as a ``scipy.sparse`` CSR matrix.

    The sparse counterpart of :func:`sparse_to_dense`: same slot rule (a slot is
    real iff its value is non-zero), but the result stays ``O(N k)`` in memory.

    Args:
        values: ``(..., max_k)`` stored activations.
        indices: ``(..., max_k)`` latent id per slot.
        n_features: ``M``, the dictionary width.

    Returns:
        ``(N, M)`` float32 CSR matrix, one row per pixel of the leading axes.
    """
    import scipy.sparse as sp

    flat_v = np.asarray(values).reshape(-1, np.shape(values)[-1])
    flat_i = np.asarray(indices, dtype=np.int64).reshape(flat_v.shape)
    valid = flat_v != 0
    rows = np.broadcast_to(np.arange(flat_v.shape[0])[:, None], flat_v.shape)[valid]
    return sp.csr_matrix(
        (flat_v[valid].astype(np.float32), (rows, flat_i[valid])), shape=(flat_v.shape[0], int(n_features))
    )


def read_dense_codes(store: zarr.Group, rows: np.ndarray, n_features: int, *, pixel_stride: int = 1) -> np.ndarray:
    """Densify selected rows of a sparse code store.

    The sparse layout is the interface between stages, but a code-side
    diagnostic needs aligned dense columns. Widths reach ``M = 1024``, so the
    pixel axis is decimated rather than materialised whole.

    Args:
        store: Open zarr group holding ``z_values`` / ``z_indices``.
        rows: Rows (patches) to read.
        n_features: ``M``, the dictionary width.
        pixel_stride: Keep every ``pixel_stride``-th pixel of each patch.

    Returns:
        ``(n_rows * n_kept_pixels, M)`` dense codes, patch-major.
    """
    max_k = store["z_values"].shape[-1]
    out = []
    for row in np.asarray(rows):
        v = np.asarray(store["z_values"][int(row)]).reshape(-1, max_k)[::pixel_stride]
        i = np.asarray(store["z_indices"][int(row)]).reshape(-1, max_k)[::pixel_stride]
        out.append(sparse_to_dense(v, i, n_features))
    return np.concatenate(out, axis=0)


def densify_feature_map(values: np.ndarray, indices: np.ndarray, counts: np.ndarray, latent: int) -> np.ndarray:
    """One latent's ``(H, W)`` activation map for a single patch (sparse -> dense).

    Padding slots carry ``value == 0``, so a slot is real iff its value is
    non-zero — robust to the unsorted order produced by the top-k. ``counts`` is
    kept in the store for provenance/overflow accounting but not needed here.
    """
    values = np.asarray(values)
    indices = np.asarray(indices)
    match = (indices == int(latent)) & (values != 0)
    return np.where(match, values, 0.0).sum(-1).astype(np.float32)


__all__ = ["dense_to_sparse", "densify_feature_map", "read_dense_codes", "sparse_code_matrix", "sparse_to_dense"]

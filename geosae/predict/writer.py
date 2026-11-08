"""Sparse SAE-code writer + prediction runner.

Writes the padded ``(values, indices, counts)`` sparse code representation to a
zarr store, one contiguous cell-range per batch, plus per-patch metadata
(``cell``/``lon``/``lat``/``fold``). Alongside it writes a compact
``feat_patch_stats`` array of shape ``(n_feat, n_patch, 2)`` holding, for every
feature and patch, ``[mean activation, number of active pixels]`` — the summary
patch-level analyses read, so they never have to scan the full code arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from geosae.data.dataset import PatchData, make_dataloader
from geosae.models.base import SparseAutoencoder, model_inputs
from geosae.predict.sparsify import dense_to_sparse


class SparseWriter:
    """Create and fill the prediction zarr store.

    The store's ``k`` attr records the model's fixed sparsity budget, or ``-1``
    for a variant that has none (ReLU + l1 sparsifies through the penalty).
    :meth:`close` additionally records whether any token's code was truncated.
    """

    def __init__(
        self,
        outdir: str,
        name: str,
        *,
        n_patches: int,
        patch_size: int,
        latent_dim: int,
        max_k: int,
        k: int | None,
        embedding_name: str,
        split: str,
        chunk_cells: int = 50,
    ) -> None:
        self.path = Path(outdir) / f"{name}.zarr"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_patches = int(n_patches)
        self.patch_size = int(patch_size)
        self.latent_dim = int(latent_dim)
        self.max_k = int(max_k)

        g = zarr.open_group(str(self.path), mode="w")
        H = W = self.patch_size
        cc = int(chunk_cells)
        g.create_array(
            "z_values", shape=(n_patches, H, W, max_k), chunks=(cc, H, W, max_k), dtype="float32", fill_value=0.0
        )
        g.create_array(
            "z_indices",
            shape=(n_patches, H, W, max_k),
            chunks=(cc, H, W, max_k),
            dtype="int16" if self.latent_dim <= 32768 else "int32",
            fill_value=0,
        )
        g.create_array("z_counts", shape=(n_patches, H, W), chunks=(cc, H, W), dtype="int16", fill_value=0)
        g.create_array("cell", shape=(n_patches,), chunks=(n_patches,), dtype="int64", fill_value=0)
        g.create_array("lon", shape=(n_patches,), chunks=(n_patches,), dtype="float64", fill_value=0.0)
        g.create_array("lat", shape=(n_patches,), chunks=(n_patches,), dtype="float64", fill_value=0.0)
        g.create_array("fold", shape=(n_patches,), chunks=(n_patches,), dtype="int8", fill_value=-1)
        pc = min(n_patches, 8192)
        g.create_array(
            "feat_patch_stats",
            shape=(latent_dim, n_patches, 2),
            chunks=(latent_dim, pc, 2),
            dtype="float32",
            fill_value=0.0,
        )
        g.attrs.update(
            {
                "latent_dim": latent_dim,
                "k": int(k) if k is not None else -1,
                "max_k": max_k,
                "patch_size": self.patch_size,
                "n_patches": n_patches,
                "embedding_name": embedding_name,
                "split": split,
                "stats_layout": "[mean_activation, n_active_pixels] per (feature, patch)",
            }
        )
        self._g = g
        # feat_patch_stats accumulated in RAM (n_feat * n_patch * 2 * 4B), flushed on close.
        self._stats = np.zeros((latent_dim, n_patches, 2), dtype=np.float32)
        self._fold_value = {"train": 0, "val": 1, "test": 2}.get(split, -1)

    @property
    def group(self) -> zarr.Group:
        """The open store, for a writer that keeps further arrays alongside the codes."""
        return self._g

    def write_batch(
        self,
        start: int,
        values: np.ndarray,
        indices: np.ndarray,
        counts: np.ndarray,
        cell: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        stats: np.ndarray,  # (b, M, 2)
        fold: np.ndarray | None = None,
    ) -> None:
        """Write one contiguous range of rows.

        Args:
            fold: Per-row split label. Defaults to the writer's single ``split``,
                which is what a prediction pass writes; a store that holds every
                split at once passes the labels instead.
        """
        b = values.shape[0]
        end = start + b
        self._g["z_values"][start:end] = values
        self._g["z_indices"][start:end] = indices
        self._g["z_counts"][start:end] = counts
        self._g["cell"][start:end] = cell
        self._g["lon"][start:end] = lon
        self._g["lat"][start:end] = lat
        self._g["fold"][start:end] = self._fold_value if fold is None else fold
        self._stats[:, start:end, :] = np.transpose(stats, (1, 0, 2))

    def close(self, *, n_truncated_tokens: int = 0, max_active_observed: int = 0) -> None:
        """Flush the per-patch stats and record the truncation audit.

        ``truncated`` / ``n_truncated_tokens`` / ``max_active_observed`` make the
        store self-describing: a reader can tell whether the codes it is about to
        score are complete without consulting a log.
        """
        self._g["feat_patch_stats"][:] = self._stats
        del self._stats
        self._g.attrs.update(
            {
                "truncated": bool(n_truncated_tokens),
                "n_truncated_tokens": int(n_truncated_tokens),
                "max_active_observed": int(max_active_observed),
            }
        )


@torch.no_grad()
def estimate_max_k(
    model: SparseAutoencoder,
    data: PatchData,
    *,
    n_probe_batches: int = 4,
    batch_size: int = 64,
    headroom: float = 1.5,
    device: str = "cuda",
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> int:
    """Measure a data-driven padding width for :func:`run_prediction`'s ``max_k``.

    Models with a fixed selection budget (Top-K/BatchTopK/Matryoshka) bound the
    per-token active count near ``k`` by construction, so a fixed ``max(2k, 32)``
    heuristic is a safe guess. Models without one (ReLU+l1) have no
    such bound at all: the right tail of per-token L0 is purely data dependent,
    and a fixed 32-wide store silently truncates any token that overshoots it.

    This runs the model over a handful of batches, measures the largest
    per-token active-latent count observed, and pads it by ``headroom``,
    capped at the dictionary size.

    Args:
        model: The SAE to probe.
        data: The split to probe (probed with a fixed shuffle seed, so the
            estimate is reproducible run to run).
        n_probe_batches: Number of batches to probe.
        batch_size: Probe batch size.
        headroom: Multiplicative safety margin on the observed max.
        device: Torch device to run the probe on.
        amp_dtype: Autocast dtype (``None`` disables autocast).

    Returns:
        The estimated ``max_k``, at least ``32`` and at most ``model.latent_dim``.
    """
    model.eval().to(device)
    autocast_enabled = amp_dtype is not None and str(device).startswith("cuda")
    max_count = 0
    for bi, batch in enumerate(data.iter_batches(batch_size, shuffle=True, seed=0)):
        if bi >= n_probe_batches:
            break
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
            _, z, _ = model(*model_inputs(model, batch, device))
        max_count = max(max_count, int((z > 0).sum(dim=-1).max().item()))
    estimated = int(np.ceil(max_count * headroom))
    return min(max(estimated, 32), model.latent_dim)


@torch.no_grad()
def run_prediction(
    model: SparseAutoencoder,
    data: PatchData,
    *,
    outdir: str,
    name: str,
    split: str,
    batch_size: int = 64,
    max_k: int | str | None = None,
    allow_truncation: bool = False,
    device: str = "cuda",
    amp_dtype: torch.dtype | None = torch.bfloat16,
    chunk_cells: int = 50,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
) -> str:
    """Run the SAE over ``data`` and write sparse codes + patch stats.

    Args:
        max_k: Padding width of the stored codes. Defaults to ``max(2k, 32)``,
            which is a *starting point only*: a batch-level top-k does not bound
            a single token's active count, so the real right tail is data
            dependent and can sit far above it. Any token exceeding ``max_k``
            raises unless ``allow_truncation``. Pass the literal string
            ``"auto"`` to measure it instead (see :func:`estimate_max_k`) --
            recommended for any model without a fixed selection budget
            (ReLU+l1), where the fixed default is not a bound at all. If a
            token of the split then exceeds the estimate, the split is encoded
            once more with ``max_k`` set to the largest count observed.
        allow_truncation: Keep a bounded top-``max_k`` store instead of raising.
            Only for consumers that genuinely want the strongest few features per
            token; every concept metric reads the full code and is biased by
            truncation.

    Returns:
        Path of the prediction store.

    Raises:
        ValueError: A token had more than ``max_k`` active latents and
            ``allow_truncation`` is False.
    """
    model.eval().to(device)
    M = model.latent_dim
    auto = max_k == "auto"
    if isinstance(max_k, str):
        if max_k != "auto":
            raise ValueError(f"max_k must be an int, None, or the literal string 'auto', got {max_k!r}")
        max_k = estimate_max_k(model, data, batch_size=batch_size, device=device, amp_dtype=amp_dtype)
        print(f"[geosae] predict.max_k=auto -> estimated {max_k} (probed batches, seed=0)", flush=True)
    else:
        # A ReLU+l1 SAE has no fixed budget, so there is no k to double.
        max_k = int(max_k) if max_k else (max(2 * model.k, 32) if model.k else 32)
    H = W = data.patch_size

    writer = SparseWriter(
        outdir,
        name,
        n_patches=len(data),
        patch_size=H,
        latent_dim=M,
        max_k=max_k,
        k=model.k,
        embedding_name=data.embedding_name,
        split=split,
        chunk_cells=chunk_cells,
    )
    start = 0
    n_batches = data.num_batches(batch_size)
    loader = make_dataloader(
        data,
        batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory and str(device).startswith("cuda"),
    )
    overflow_px = 0
    max_count = 0
    n_tokens = 0
    autocast = amp_dtype is not None and str(device).startswith("cuda")
    for bi, batch in enumerate(loader):
        b = batch["x"].shape[0]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast):
            _, z, _ = model(*model_inputs(model, batch, device))  # x: (b, H, W, D)
        zv = z.float().reshape(b, H, W, M)
        # Per-patch feature stats: mean activation and #active pixels over the patch.
        zf = zv.reshape(b, H * W, M)
        stats = torch.stack([zf.mean(1), (zf > 0).float().sum(1)], dim=-1).cpu().numpy()  # (b, M, 2)
        values, indices, counts = dense_to_sparse(zv, max_k)
        overflow_px += int((counts > max_k).sum())
        max_count = max(max_count, int(counts.max()))
        n_tokens += int(counts.size)
        writer.write_batch(
            start,
            values,
            indices,
            counts,
            batch["cell"].numpy(),
            batch["lon"].numpy(),
            batch["lat"].numpy(),
            stats,
        )
        start += b
        if bi % 50 == 0:
            print(f"  predict batch {bi}/{n_batches} ({start}/{len(data)} patches)", flush=True)
    writer.close(n_truncated_tokens=overflow_px, max_active_observed=max_count)
    print(f"[geosae] wrote predictions -> {writer.path}", flush=True)
    print(f"[geosae] per-token active latents: max {max_count}, max_k {max_k}", flush=True)
    if overflow_px:
        detail = (
            f"{overflow_px}/{n_tokens} tokens ({overflow_px / max(n_tokens, 1):.3%}) had more than "
            f"max_k={max_k} active latents (max {max_count}), so their codes were truncated to the "
            f"top-{max_k} by magnitude"
        )
        if auto and not allow_truncation:
            print(f"[geosae] {detail}; encoding again with max_k={max_count}", flush=True)
            return run_prediction(
                model,
                data,
                outdir=outdir,
                name=name,
                split=split,
                batch_size=batch_size,
                max_k=max_count,
                device=device,
                amp_dtype=amp_dtype,
                chunk_cells=chunk_cells,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
        if not allow_truncation:
            raise ValueError(
                f"{detail}. A batch-level top-k does not bound a single token's active count, so "
                f"max_k cannot be derived from k: re-run with predict.max_k >= {max_count}. "
                f"Pass allow_truncation=True only if a bounded top-{max_k} store is what you want "
                f"-- every concept metric reads these codes, so truncation biases them, and it "
                f"biases a frozen-decoder null far more than a trained model because the null's "
                f"per-token L0 distribution has a much heavier tail."
            )
        banner = "!" * 78
        print(
            f"\n{banner}\n[geosae] WARNING: TRUNCATED PREDICTION STORE (allow_truncation=True)\n"
            f"[geosae] {detail}\n[geosae] every downstream concept metric reads a biased code from "
            f"this store -- re-run with predict.max_k='auto' or predict.max_k >= {max_count} unless "
            f"truncation is genuinely what this store is for.\n{banner}\n",
            flush=True,
        )
    return str(writer.path)


__all__ = ["SparseWriter", "run_prediction", "estimate_max_k"]

"""Embedding-patch data access for the pixel-wise SAE (zarr + torch DataLoader).

The dataset is a single Zarr patch store holding, for a grid
of ~64x64 embedding patches, per-cell arrays such as ``alphaearth`` ``(N, H, W,
D)`` plus precomputed ``split_*`` label arrays. Each *cell* is one patch; the SAE
encodes every pixel independently.

The module exposes three pieces:

* :class:`PatchData` — selects the cells of one split and reads their embeddings,
  either **in-RAM** (``compute=True``: load every selected cell once into a
  single ``float16`` array via chunk-grouped threaded reads, then serve from RAM)
  or **lazy** (``compute=False``: read straight from zarr, grouping cells by their
  storage chunk so each chunk is touched at most once per read). It also carries
  the optional transform pipeline and a simple :meth:`~PatchData.iter_batches`
  generator for one-off/eval iteration.
* :class:`PatchDataset` — a ``torch.utils.data.Dataset`` whose ``__getitem__``
  returns a *micro-batch* already: one chunk-aligned group of cells as a dict of
  tensors. Consumed through a ``DataLoader`` with :func:`collate_patches`, which
  stitches micro-batches into full batches. This overlaps reads/transforms with
  GPU compute via worker processes and prefetch.
* :func:`make_dataloader` — builds the ``DataLoader`` for a :class:`PatchData`,
  yielding full batches identical to ``iter_batches`` (same cells, same order,
  same shuffle-by-seed determinism) but pipelined across workers.

Transforms (all optional) mirror the training recipe: ``nan_to_num -> hard_clip
-> per-dim quantile clip -> per-dim z-score -> input_scale``. Unit-norm
embeddings need none of them; other recipes use ``hard_clip`` + external z-score
stats.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

# Distinguishes "caller passed nothing" from "caller explicitly asked for no
# caching" (`None`): the default is resolved against `geosae.data.quality`'s
# current `DEFAULT_QUALITY_CACHE_ROOT` at construction time, not baked into
# this signature at import time, so a test fixture can monkeypatch that
# module attribute and every un-overridden `PatchData` picks it up.
_QUALITY_CACHE_ROOT_DEFAULT = object()

# Same rationale as `_QUALITY_CACHE_ROOT_DEFAULT`, for `max_abs_value`: resolved
# against `geosae.data.quality.DEFAULT_MAX_ABS_VALUE` at construction time.
_MAX_ABS_VALUE_DEFAULT = object()

# Split-label convention written by the split scripts (train, val, test).
SPLIT_LABELS = {"train": 0, "val": 1, "test": 2}


@dataclass
class Transforms:
    """Per-dimension embedding transforms (see module docstring)."""

    hard_clip: float | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    clip_lo: np.ndarray | None = None
    clip_hi: np.ndarray | None = None
    input_scale: float = 1.0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.hard_clip is not None:
            x = torch.clamp(x, -self.hard_clip, self.hard_clip)
        if self.clip_lo is not None and self.clip_hi is not None:
            lo = torch.as_tensor(self.clip_lo, dtype=x.dtype, device=x.device)
            hi = torch.as_tensor(self.clip_hi, dtype=x.dtype, device=x.device)
            x = torch.minimum(torch.maximum(x, lo), hi)
        if self.mean is not None and self.std is not None:
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            x = (x - mean) / std
        if self.input_scale != 1.0:
            x = x * self.input_scale
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def from_stats_npz(cls, path: str, hard_clip: float | None = None, input_scale: float = 1.0) -> Transforms:
        s = np.load(path)
        return cls(
            hard_clip=hard_clip,
            mean=s["mean"].astype(np.float32),
            std=s["std"].astype(np.float32),
            clip_lo=s["clip_lo"].astype(np.float32) if "clip_lo" in s else None,
            clip_hi=s["clip_hi"].astype(np.float32) if "clip_hi" in s else None,
            input_scale=input_scale,
        )

    @classmethod
    def from_zarr_stats(
        cls,
        zarr_path: str,
        prefix: str,
        hard_clip: float | None = None,
        input_scale: float = 1.0,
    ) -> Transforms:
        """Load per-dim ``mean``/``std``[/``clip_lo``/``clip_hi``] from arrays already in the store.

        Some products carry their own precomputed standardisation stats as
        sibling arrays in the store (e.g. Tessera's
        ``tessera_norm_mean``/``_std``/``_clip_lo``/``_clip_hi``), so training
        needs no separate stats-export step -- unlike :meth:`from_stats_npz`,
        which reads a file produced by :func:`compute_embedding_stats`.

        Args:
            zarr_path: Path to the patch store (or any group holding
                ``<prefix>_mean`` / ``<prefix>_std``).
            prefix: Array-name prefix, e.g. ``"tessera_norm"``.
            hard_clip: Forwarded to the constructor.
            input_scale: Forwarded to the constructor.

        Returns:
            A :class:`Transforms` with ``mean``/``std`` set, and ``clip_lo``/
            ``clip_hi`` set if the store carries them.
        """
        g = zarr.open_group(str(zarr_path), mode="r")
        return cls(
            hard_clip=hard_clip,
            mean=np.asarray(g[f"{prefix}_mean"][:], dtype=np.float32),
            std=np.asarray(g[f"{prefix}_std"][:], dtype=np.float32),
            clip_lo=np.asarray(g[f"{prefix}_clip_lo"][:], dtype=np.float32) if f"{prefix}_clip_lo" in g else None,
            clip_hi=np.asarray(g[f"{prefix}_clip_hi"][:], dtype=np.float32) if f"{prefix}_clip_hi" in g else None,
            input_scale=input_scale,
        )


class PatchData:
    """Selected embedding patches for one split, with in-RAM or lazy zarr reads.

    Args:
        zarr_path: Path to the patch store.
        embedding_name: Array name in the store (e.g. ``alphaearth`` / ``tessera``).
        split_name: Name of the split-label array (e.g. ``split_default``); ``None``
            takes the store's only ``split_*`` array (see :func:`resolve_split_name`).
        split: One of ``train`` / ``val`` / ``test`` (or an explicit int label).
        compute: If True, load all selected cells into RAM as ``inram_dtype``.
        transforms: Optional :class:`Transforms`.
        n_load_workers: Threads for the in-RAM chunk loader.
        inram_dtype: ``float16`` (default, halves RAM) or ``float32``.
        max_cells: Optional cap on number of cells (debug/smoke).
        min_spatial_std: Drop *degenerate* cells whose embedding spatial std is
            below this (constant fill patches, see :mod:`geosae.data.quality`).
            Defaults to ``1e-3``; pass ``None`` to keep every cell (needed at ``patch_size=1``,
            where spatial std is trivially zero everywhere).
        drop_flat_pixel_cells: Drop cells holding at least one *flat* pixel --
            every embedding dimension the same number, a fill value no real
            embedding produces (see :mod:`geosae.data.quality`). On by default.
        max_abs_value: Drop cells holding a pixel whose largest per-dim
            magnitude exceeds this (a product-specific sentinel, e.g.
            Tessera's ~1e27-2.9e27 fill values, that neither of the two
            filters above catches -- see :mod:`geosae.data.quality`).
            Defaults to :data:`geosae.data.quality.DEFAULT_MAX_ABS_VALUE`,
            chosen to leave near-unit-norm products (AlphaEarth) untouched. Pass ``None`` to disable.
        quality_cache_root: Memoise the ``min_spatial_std`` / ``drop_flat_pixel_cells``
            / ``max_abs_value`` scans to disk under this directory (see
            :func:`geosae.data.quality.cached_constant_mask` /
            :func:`~geosae.data.quality.cached_flat_pixel_mask` /
            :func:`~geosae.data.quality.cached_extreme_value_mask`), keyed on the
            store's own content so a changed store invalidates it, instead of
            re-scanning the whole store on every construction. Defaults to
            :data:`geosae.data.quality.DEFAULT_QUALITY_CACHE_ROOT` -- **on**,
            because the scan cost is repeated by any caller that builds many
            ``PatchData`` against the same store (an evaluation-sample sweep,
            a training/ladder loop), and a cache a caller must remember to
            request is a cache that regresses again the next time. Pass
            ``None`` to force the direct, uncached scan.
    """

    def __init__(
        self,
        zarr_path: str,
        embedding_name: str = "alphaearth",
        split_name: str | None = None,
        split: str | int = "train",
        *,
        compute: bool = True,
        transforms: Transforms | None = None,
        n_load_workers: int = 16,
        inram_dtype: str = "float16",
        max_cells: int | None = None,
        min_spatial_std: float | None = 1e-3,
        drop_flat_pixel_cells: bool = True,
        max_abs_value: float | None = _MAX_ABS_VALUE_DEFAULT,
        quality_cache_root: str | Path | None = _QUALITY_CACHE_ROOT_DEFAULT,
    ) -> None:
        if quality_cache_root is _QUALITY_CACHE_ROOT_DEFAULT:
            from geosae.data.quality import DEFAULT_QUALITY_CACHE_ROOT

            quality_cache_root = DEFAULT_QUALITY_CACHE_ROOT
        if max_abs_value is _MAX_ABS_VALUE_DEFAULT:
            from geosae.data.quality import DEFAULT_MAX_ABS_VALUE

            max_abs_value = DEFAULT_MAX_ABS_VALUE
        self.zarr_path = str(zarr_path)
        self.embedding_name = embedding_name
        self.split_name = resolve_split_name(self.zarr_path, split_name)
        split_name = self.split_name
        self.split = split
        self.compute = bool(compute)
        self.transforms = transforms or Transforms()
        self.n_load_workers = int(n_load_workers)
        if inram_dtype not in ("float16", "float32"):
            raise ValueError(f"inram_dtype must be float16/float32, got {inram_dtype!r}")
        self._inram_dtype = np.float16 if inram_dtype == "float16" else np.float32

        group = zarr.open_group(self.zarr_path, mode="r")
        emb = group[embedding_name]
        self.n_total_cells = int(group.attrs["n_cells"])
        # Patch geometry comes from the array itself, not the store's pixel grid:
        # a patch-token embedding lives on a grid coarser than the store's pixels.
        if emb.shape[1] != emb.shape[2]:
            raise ValueError(f"{embedding_name} patches must be square, got {emb.shape[1]}x{emb.shape[2]}")
        self.patch_size = int(emb.shape[1])
        self.embedding_dim = int(emb.shape[-1])
        self.chunk_size = int(emb.chunks[0])

        # Per-process zarr handle cache (fork-safe: workers reopen on first read).
        self._emb_cache = emb
        self._emb_pid = os.getpid()

        label = SPLIT_LABELS[split] if isinstance(split, str) else int(split)
        cell_mask = np.asarray(group[f"mask_{embedding_name}"][:], dtype=bool)
        split_arr = np.asarray(group[split_name][:])
        self.cells = np.where(cell_mask & (split_arr == label))[0].astype(np.int64)
        if max_cells is not None:
            self.cells = self.cells[:max_cells]
        if max_abs_value is not None:
            # Runs before `min_spatial_std`: an extreme-value sentinel would
            # overflow float32 on squaring inside the spatial-std scan.
            if quality_cache_root is not None:
                from geosae.data.quality import cached_extreme_value_mask

                full_bad = cached_extreme_value_mask(
                    self.zarr_path,
                    embedding_name,
                    split_name=split_name,
                    max_abs_value=float(max_abs_value),
                    n_workers=self.n_load_workers,
                    cache_root=quality_cache_root,
                )
                bad = full_bad[self.cells]
            else:
                from geosae.data.quality import scan_extreme_value_cells

                bad = scan_extreme_value_cells(
                    self.zarr_path,
                    embedding_name,
                    self.cells,
                    max_abs_value=float(max_abs_value),
                    n_workers=self.n_load_workers,
                )
            if bad.any():
                print(
                    f"[geosae] dropping {int(bad.sum())}/{self.cells.size} cells holding a pixel "
                    f"with |value| > {max_abs_value}",
                    flush=True,
                )
            self.cells = self.cells[~bad]
        if min_spatial_std is not None:
            if quality_cache_root is not None:
                from geosae.data.quality import cached_constant_mask

                full_bad = cached_constant_mask(
                    self.zarr_path,
                    embedding_name,
                    split_name=split_name,
                    thresh=float(min_spatial_std),
                    n_workers=self.n_load_workers,
                    cache_root=quality_cache_root,
                )
                bad = full_bad[self.cells]
            else:
                from geosae.data.quality import scan_constant_cells

                bad = scan_constant_cells(
                    self.zarr_path,
                    embedding_name,
                    self.cells,
                    thresh=float(min_spatial_std),
                    n_workers=self.n_load_workers,
                )
            if bad.any():
                print(
                    f"[geosae] dropping {int(bad.sum())}/{self.cells.size} degenerate "
                    f"(constant) cells (spatial std < {min_spatial_std})",
                    flush=True,
                )
            self.cells = self.cells[~bad]
        if drop_flat_pixel_cells:
            if quality_cache_root is not None:
                from geosae.data.quality import cached_flat_pixel_mask

                full_bad = cached_flat_pixel_mask(
                    self.zarr_path,
                    embedding_name,
                    split_name=split_name,
                    n_workers=self.n_load_workers,
                    cache_root=quality_cache_root,
                )
                bad = full_bad[self.cells]
            else:
                from geosae.data.quality import scan_flat_pixel_cells

                bad = scan_flat_pixel_cells(
                    self.zarr_path,
                    embedding_name,
                    self.cells,
                    n_workers=self.n_load_workers,
                )
            if bad.any():
                print(
                    f"[geosae] dropping {int(bad.sum())}/{self.cells.size} cells holding a flat pixel",
                    flush=True,
                )
            self.cells = self.cells[~bad]
        if self.cells.size == 0:
            raise ValueError(f"no cells for split={split!r} in {split_name!r}")

        lon = np.asarray(group["lon"][:])
        lat = np.asarray(group["lat"][:])
        self.lon = lon[self.cells].astype(np.float64)
        self.lat = lat[self.cells].astype(np.float64)

        self._arr: np.ndarray | None = None
        if self.compute:
            self._arr = self._load_all()

    def __len__(self) -> int:
        return int(self.cells.size)

    # ------------------------------------------------------------ zarr handle
    @property
    def _emb(self):
        """Embedding array, reopened per process so forked workers are safe."""
        if self._emb_cache is None or self._emb_pid != os.getpid():
            self._emb_cache = zarr.open_group(self.zarr_path, mode="r")[self.embedding_name]
            self._emb_pid = os.getpid()
        return self._emb_cache

    # ------------------------------------------------------------------ in-RAM
    def _load_all(self) -> np.ndarray:
        """Load all selected cells into ``(N, H, W, D)`` via chunk-grouped reads."""
        n = self.cells.size
        out = np.empty((n, self.patch_size, self.patch_size, self.embedding_dim), dtype=self._inram_dtype)
        by_chunk: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for local, cell in enumerate(self.cells):
            by_chunk[int(cell) // self.chunk_size].append((local, int(cell)))
        emb = self._emb

        def _load(item):
            chunk_id, entries = item
            start = chunk_id * self.chunk_size
            end = min(start + self.chunk_size, emb.shape[0])
            block = emb[start:end]
            # Cast only the kept rows: a chunk can also hold cells the quality
            # filters already dropped (e.g. Tessera's ~1e27-2.9e27 sentinels),
            # which overflow inram_dtype on cast -- casting the whole block
            # first would spend that overflow on values never used.
            for local, cell in entries:
                out[local] = block[cell - start].astype(out.dtype, copy=False)

        with ThreadPoolExecutor(max_workers=self.n_load_workers) as ex:
            list(ex.map(_load, by_chunk.items()))
        return out

    # ------------------------------------------------------------------ reads
    def _read_cells(self, local_idx: np.ndarray) -> np.ndarray:
        """Raw ``(b, H, W, D)`` block for the given *local* indices (into ``self.cells``)."""
        if self._arr is not None:
            return self._arr[local_idx]
        # Lazy: group by zarr chunk so each chunk is read once.
        cells = self.cells[local_idx]
        emb = self._emb
        by_chunk: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for pos, cell in enumerate(cells):
            by_chunk[int(cell) // self.chunk_size].append((pos, int(cell)))
        out = np.empty((cells.size, self.patch_size, self.patch_size, self.embedding_dim), dtype=np.float32)
        for chunk_id, entries in by_chunk.items():
            start = chunk_id * self.chunk_size
            end = min(start + self.chunk_size, emb.shape[0])
            block = emb[start:end]
            for pos, cell in entries:
                out[pos] = block[cell - start]
        return out

    def _make_batch(self, local_idx: np.ndarray) -> dict:
        local_idx = np.asarray(local_idx, dtype=np.int64)
        raw = self._read_cells(local_idx)
        x = self.transforms.apply(torch.from_numpy(np.ascontiguousarray(raw)).float())
        return {
            "x": x,  # (b, H, W, D) float32
            "cell": torch.from_numpy(self.cells[local_idx].astype(np.int64)),
            "lon": torch.from_numpy(self.lon[local_idx].astype(np.float32)),
            "lat": torch.from_numpy(self.lat[local_idx].astype(np.float32)),
            "local_idx": torch.from_numpy(local_idx),
        }

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Yield collated batches (single-threaded). Order is cell order unless ``shuffle``.

        Kept for one-off sampling and evaluation; the training/inference loops use
        :func:`make_dataloader`, which produces identical batches with worker
        pipelining.
        """
        for window in _iter_windows(len(self), batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last):
            yield self._make_batch(window)

    def num_batches(self, batch_size: int, drop_last: bool = False) -> int:
        n = len(self)
        return n // batch_size if drop_last else (n + batch_size - 1) // batch_size


# --------------------------------------------------------------------- ordering
def _iter_windows(n: int, batch_size: int, *, shuffle: bool, seed: int, drop_last: bool):
    """Yield the per-batch local-index windows of an epoch (shared ordering rule)."""
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    last = n - (n % batch_size) if drop_last else n
    for start in range(0, last, batch_size):
        yield order[start : start + batch_size]


def _plan_micro_batches(
    data: PatchData,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    micro_size: int | None,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Partition an epoch into chunk-aligned micro-batches + the full-batch plan.

    Each full-batch window (as produced by :func:`_iter_windows`) is split into
    contiguous micro-batches whose concatenation, in order, reproduces the window
    exactly. For lazy (on-disk) reads the split follows storage-chunk boundaries
    so every ``__getitem__`` reads a single zarr chunk; for in-RAM reads it is a
    fixed ``micro_size`` slice for worker parallelism.

    Returns ``(groups, batch_plan)`` where ``groups[g]`` is a micro-batch's local
    indices and ``batch_plan[b]`` lists the micro-batch ids forming full batch b.
    """
    chunk_aligned = data._arr is None
    if micro_size is None:
        micro_size = min(data.chunk_size, batch_size)
    micro_size = max(1, int(micro_size))
    cells = data.cells

    groups: list[np.ndarray] = []
    batch_plan: list[list[int]] = []
    for window in _iter_windows(len(data), batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last):
        gids: list[int] = []
        if chunk_aligned:
            # Split at storage-chunk boundaries (contiguous runs of equal chunk id).
            chunk_of = cells[window] // data.chunk_size
            cuts = np.flatnonzero(np.diff(chunk_of)) + 1
            bounds = [0, *cuts.tolist(), len(window)]
        else:
            bounds = list(range(0, len(window), micro_size)) + [len(window)]
        for s, e in zip(bounds[:-1], bounds[1:], strict=True):
            gids.append(len(groups))
            groups.append(window[s:e])
        batch_plan.append(gids)
    return groups, batch_plan


class PatchDataset(Dataset):
    """Torch ``Dataset`` view of a :class:`PatchData` as chunk-aligned micro-batches.

    ``__getitem__(i)`` returns micro-batch ``i`` (a dict of tensors for one
    chunk-aligned group of cells). A ``DataLoader`` with a ``batch_sampler`` over
    :attr:`batch_plan` and :func:`collate_patches` stitches the micro-batches of
    each entry into a full batch.
    """

    def __init__(self, data: PatchData, groups: list[np.ndarray]):
        self.data = data
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        return self.data._make_batch(self.groups[i])


def collate_patches(micro_batches: list[dict]) -> dict:
    """Concatenate micro-batch dicts along the batch dimension into one batch."""
    if len(micro_batches) == 1:
        return micro_batches[0]
    return {k: torch.cat([mb[k] for mb in micro_batches], dim=0) for k in micro_batches[0]}


def make_dataloader(
    data: PatchData,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    num_workers: int = 0,
    prefetch_factor: int | None = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    micro_size: int | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` over ``data`` yielding full batches.

    The batches are identical (same cells, same order for a given ``seed``) to
    ``data.iter_batches(batch_size, shuffle=..., seed=..., drop_last=...)``, but
    reads and transforms are pipelined across ``num_workers`` processes with
    prefetch, so IO/CPU prep overlaps GPU compute. Each ``__getitem__`` reads a
    single storage chunk on the lazy track.
    """
    groups, batch_plan = _plan_micro_batches(
        data, batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last, micro_size=micro_size
    )
    dataset = PatchDataset(data, groups)
    kwargs: dict = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_sampler=batch_plan,
        collate_fn=collate_patches,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def build_transforms(cfg: dict | None, zarr_path: str | None = None) -> Transforms:
    """Build :class:`Transforms` from a config mapping (all keys optional).

    Args:
        cfg: The ``data.transforms`` mapping.
        zarr_path: The store :class:`PatchData` is reading, forwarded to
            :meth:`Transforms.from_zarr_stats` when ``cfg`` sets
            ``stats_zarr_prefix`` -- unused otherwise.
    """
    if cfg is None:
        return Transforms()
    cfg = dict(cfg)
    stats_npz = cfg.pop("stats_npz", None)
    if stats_npz is not None:
        return Transforms.from_stats_npz(
            stats_npz, hard_clip=cfg.get("hard_clip"), input_scale=float(cfg.get("input_scale", 1.0))
        )
    stats_zarr_prefix = cfg.pop("stats_zarr_prefix", None)
    if stats_zarr_prefix is not None:
        if zarr_path is None:
            raise ValueError("transforms.stats_zarr_prefix requires a data.zarr_path")
        return Transforms.from_zarr_stats(
            zarr_path,
            stats_zarr_prefix,
            hard_clip=cfg.get("hard_clip"),
            input_scale=float(cfg.get("input_scale", 1.0)),
        )
    return Transforms(
        hard_clip=cfg.get("hard_clip"),
        input_scale=float(cfg.get("input_scale", 1.0)),
    )


def compute_embedding_stats(
    zarr_path: str,
    embedding_name: str,
    split_name: str,
    split: str | int = "train",
    *,
    isotropic: bool = False,
    batch_size: int | None = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Mean and standard deviation of an embedding split, for :class:`Transforms`.

    Computed on one split only -- normally ``train`` -- so applying the result to
    val/test leaks nothing.

    Args:
        zarr_path: Store path.
        embedding_name: Embedding array to summarise.
        split_name: Split-label array.
        split: Which split to summarise.
        isotropic: Replace the per-dimension std with a single number, the root
            mean square deviation from the mean across all dimensions. Centering
            is then a translation and scaling a dilation, which leaves the
            geometry the dictionary has to explain intact -- unlike per-dimension
            whitening, which stretches the space along the coordinate axes.
        batch_size: Cells read per batch. Defaults to the embedding's storage
            chunk, so each chunk is decompressed once instead of once per batch.
        **kwargs: Forwarded to :class:`PatchData`.

    Returns:
        ``{"mean": (D,), "std": (D,)}`` as float32.
    """
    data = PatchData(zarr_path, embedding_name, split_name, split, compute=False, **kwargs)
    d = data.embedding_dim
    batch_size = int(batch_size or data.chunk_size)
    total = np.zeros(d, dtype=np.float64)
    total_sq = np.zeros(d, dtype=np.float64)
    n = 0
    for batch in data.iter_batches(batch_size):
        x = batch["x"].numpy().reshape(-1, d).astype(np.float64)
        total += x.sum(0)
        total_sq += (x * x).sum(0)
        n += x.shape[0]
    mean = total / n
    var = np.maximum(total_sq / n - mean**2, 0.0)
    std = np.full(d, np.sqrt(var.mean())) if isotropic else np.sqrt(var)
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-8).astype(np.float32)}


def write_embedding_stats(path, stats: dict[str, np.ndarray]) -> Path:
    """Write stats to an ``.npz`` readable by :meth:`Transforms.from_stats_npz`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    return path


def resolve_split_name(zarr_path: str, split_name: str | None = None) -> str:
    """The split-label array to read: ``split_name``, or the store's only ``split_*`` array.

    Raises:
        KeyError: ``split_name`` is not in the store, or it is ``None`` and the
            store holds no split array or several.
    """
    group = zarr.open_group(str(zarr_path), mode="r")
    names = sorted(k for k in group.array_keys() if k.startswith("split_"))
    if split_name is not None:
        if split_name not in names:
            raise KeyError(f"{zarr_path} has no split array {split_name!r}; it has {names}")
        return str(split_name)
    if len(names) != 1:
        raise KeyError(f"{zarr_path} holds split arrays {names}; choose one with split_name (data.split_name)")
    return names[0]


def open_store(zarr_path: str) -> zarr.Group:
    """Open the patch store read-only (used by the plotting tools for RGB)."""
    return zarr.open_group(str(Path(zarr_path)), mode="r")


__all__ = [
    "PatchData",
    "PatchDataset",
    "Transforms",
    "collate_patches",
    "make_dataloader",
    "build_transforms",
    "open_store",
    "resolve_split_name",
    "SPLIT_LABELS",
]

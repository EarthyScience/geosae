"""A Sentinel-2 subset of a patch store, encoded by Earth foundation models.

Embeddings such as OlmoEarth and MMEarth are not published as maps: they have
to be computed from Sentinel-2 imagery. This module cuts a subset of a parent
MajorTOM store's cells, fetches Sentinel-2 L2A reflectance for it into a compact
store of its own, runs each requested backbone over that reflectance, and writes
the per-dimension statistics the SAE standardises the tokens with. The subset
inherits the parent's split labels and records the parent row of every cell in
``cell_indices``, so its concept labels can be read from the parent.

Three stages, each a separate job: :func:`fetch_subset` (CPUs and network),
:func:`encode_subset` (a GPU) and :func:`subset_stats`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger(__name__)

_SPLIT_LABELS = (("train", 0), ("val", 1), ("test", 2))


def select_subset_rows(
    parent_path: str,
    split_name: str,
    max_per_split: dict[str, int] | None = None,
    seed: int = 42,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The parent cells a subset copies: one split, optionally capped per label.

    Args:
        parent_path: Parent store path.
        split_name: Split array to read (e.g. ``split_10k``).
        max_per_split: Optional ``{'train'|'val'|'test': n}`` caps, applied as a
            seeded random subsample within each label (in train, val, test order).
        seed: Subsampling seed.
        exclude: Parent rows never selected.

    Returns:
        ``(rows, lon, lat, labels)``: sorted parent rows, their centres and their
        int8 split labels.
    """
    parent = zarr.open_group(parent_path, mode="r")
    labels = np.asarray(parent[split_name][:])
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for name, value in _SPLIT_LABELS:
        rows = np.where(labels == value)[0]
        if exclude is not None:
            rows = np.setdiff1d(rows, exclude)
        cap = (max_per_split or {}).get(name)
        if cap is not None and rows.size > cap:
            rows = np.sort(rng.choice(rows, size=cap, replace=False))
        keep.append(rows)
    rows = np.sort(np.concatenate(keep))
    return rows, np.asarray(parent["lon"][:])[rows], np.asarray(parent["lat"][:])[rows], labels[rows]


def fetch_subset(
    parent_path: str,
    out_path: str,
    *,
    split_name: str = "split_10k",
    max_per_split: dict[str, int] | None = None,
    seed: int = 42,
    exclude_stores: list[str] | None = None,
    max_cells: int | None = None,
    patch_size: int = 128,
    year: int = 2024,
    max_cloud_cover: float = 10.0,
    max_workers: int = 1,
) -> dict[str, int]:
    """Fetch Sentinel-2 L2A for a subset of a parent store's cells into a new store.

    Needs the ``download`` extra.

    Args:
        parent_path: Parent patch store.
        out_path: Subset store to write.
        split_name: Parent split the cells are drawn from; the subset keeps it
            under the same name.
        max_per_split: Per-label caps, see :func:`select_subset_rows`.
        seed: Subsampling seed.
        exclude_stores: Subset stores whose cells are not selected again.
        max_cells: Keep only the first ``max_cells`` selected cells (a smoke test).
        patch_size: Sentinel-2 patch side in 10 m pixels.
        year: Acquisition year.
        max_cloud_cover: Scene cloud-cover ceiling, percent.
        max_workers: Concurrent tile fetches.

    Returns:
        ``{"selected", "fetched"}`` cell counts.
    """
    from geosae.data.download.store import build_cell_store

    exclude = [np.asarray(zarr.open_group(p, mode="r")["cell_indices"][:]) for p in exclude_stores or []]
    rows, lon, lat, labels = select_subset_rows(
        parent_path, split_name, max_per_split, seed, np.concatenate(exclude) if exclude else None
    )
    if max_cells is not None:
        rows, lon, lat, labels = rows[:max_cells], lon[:max_cells], lat[:max_cells], labels[:max_cells]
    logger.info("selected %d cells from %s (%s)", rows.size, parent_path, split_name)
    build_cell_store(
        out_path,
        products=["sentinel2_l2a"],
        lon=lon,
        lat=lat,
        cell_indices=rows,
        splits={split_name.removeprefix("split_"): labels},
        patch_size=patch_size,
        year=year,
        fetch_kwargs={"sentinel2_l2a": {"year": year, "max_cloud_cover": max_cloud_cover}},
        max_workers=max_workers,
    )
    fetched = int(np.asarray(zarr.open_group(out_path, mode="r")["mask_s2"][:]).sum())
    return {"selected": int(rows.size), "fetched": fetched}


def encode_subset(
    out_path: str,
    backbones: list[str],
    *,
    variants: list[str | None] | None = None,
    device: str = "cuda",
    batch_cells: int = 32,
    patch_size_tokens: int | None = None,
    crop_px: int | None = None,
) -> dict[str, int]:
    """Run each backbone over the subset's reflectance and write its token grids.

    Needs the ``embed`` extra and, in practice, a GPU.

    Args:
        out_path: Subset store (written by :func:`fetch_subset`).
        backbones: Backbone names (see :data:`geosae.data.embed.BACKBONES`).
        variants: Model variant per backbone, in the same order (``None``: default).
        device: Torch device.
        batch_cells: Cells per forward pass.
        patch_size_tokens: Token stride override for FlexiViT backbones (OlmoEarth).
        crop_px: Centre-crop each patch to this many pixels before encoding.

    Returns:
        ``{backbone: latent width}``.
    """
    from geosae.data.embed.base import build_backbone
    from geosae.data.embed.store import embed_store

    group = zarr.open_group(out_path, mode="a")
    variants = list(variants or []) + [None] * (len(backbones) - len(variants or []))
    widths = {}
    for name, variant in zip(backbones, variants, strict=True):
        kwargs: dict = {"device": device}
        if variant:
            kwargs["variant"] = variant
        if patch_size_tokens is not None and name == "olmoearth":
            kwargs["patch_size"] = patch_size_tokens
        backbone = build_backbone(name, **kwargs)
        embed_store(group, backbone, name, batch_cells=batch_cells, crop_px=crop_px)
        widths[name] = int(backbone.latent_dim)
        logger.info("wrote %s: dim %d, stride %d", name, backbone.latent_dim, backbone.stride)
    return widths


def subset_stats(
    out_path: str, backbones: list[str], stats_dir: str | Path, *, split_name: str = "split_10k", seed: int = 42
) -> dict[str, str]:
    """Write each backbone's per-dimension statistics from the subset's training cells.

    Args:
        out_path: Subset store holding the encoded arrays.
        backbones: Encoded array names.
        stats_dir: Directory for ``<backbone>_stats.npz``.
        split_name: Split array of the subset store.
        seed: Token-subsampling seed.

    Returns:
        ``{backbone: path}`` of the files written.
    """
    from geosae.data.embed.store import embedding_stats

    group = zarr.open_group(out_path, mode="r")
    outdir = Path(stats_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_rows = np.where(np.asarray(group[split_name][:]) == 0)[0]
    paths = {}
    for name in backbones:
        rows = train_rows[np.asarray(group[f"mask_{name}"][:])[train_rows]]
        stats = embedding_stats(group, name, rows=rows, seed=seed)
        path = outdir / f"{name}_stats.npz"
        np.savez(path, **stats)
        paths[name] = str(path)
        logger.info("wrote %s from %d cells", path, rows.size)
    return paths


__all__ = ["select_subset_rows", "fetch_subset", "encode_subset", "subset_stats"]

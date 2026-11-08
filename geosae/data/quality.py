"""Detect degenerate embedding patches (constant cells / flat pixels).

Two, distinct store defects, both upstream nodata artifacts that pass
``mask_<embedding>`` (which only records download success):

* A small fraction (~0.3-1%) of MajorTOM AlphaEarth cells carry a *constant*
  fill embedding: the identical 64-d vector broadcast across all 64x64 pixels
  (a nodata sentinel of L2 norm ~8.06 vs. the ~1 unit-norm of real pixels).
  Such a cell produces **flat** SAE activation maps and inflates per-pixel L0,
  because an identical input yields an identical code at every pixel.
  :func:`scan_constant_cells` flags them by spatial standard deviation.
* A rarer, pixel-level defect: a single pixel whose every embedding dimension
  carries the same number. No real embedding is flat, so such a pixel is a
  fill value too, and it dominates any moment it enters.
  :func:`scan_flat_pixel_cells`
  flags a cell that holds even one such pixel.
* A third, product-specific defect seen in Tessera: a pixel where one or two
  of the 128 dims carry a magnitude sentinel (~1e27-2.9e27) with the rest
  zero. Neither of the above catches it -- the pixel is not spatially flat
  (only a couple of dims are extreme) and it is not a flat pixel (the dims
  disagree) -- and squaring such a value overflows float32.
  :func:`scan_extreme_value_cells` flags a cell holding any pixel whose
  largest per-dim magnitude exceeds a threshold, compared in float64 and
  never squared, so the scan itself cannot overflow.

:class:`~geosae.data.dataset.PatchData` drops all three from a split via
``min_spatial_std`` / ``drop_flat_pixel_cells`` / ``max_abs_value``, on by
default at the patch geometry (``None`` / ``False`` keeps every cell --
needed at ``patch_size=1``, where a single pixel has no spatial std to
measure).

Both scans read every chunk of the store's embedding array, so a caller that
builds many :class:`~geosae.data.dataset.PatchData` against the same store --
repeated evaluation samples, several training runs -- pays this full-store
cost again on every construction.
:func:`cached_constant_mask` / :func:`cached_flat_pixel_mask` /
:func:`cached_extreme_value_mask` are the same scans, memoised to disk: every
mask is a pure function of ``(zarr_path, embedding_name[, threshold])`` alone,
over *every* cell in the store (not just one split), so the first caller
against a store pays the scan once and every later one -- any split, any
subset of cells -- reads the cached array. This mirrors the content-addressed
cache
:func:`geosae.eval.pool.default_sample_cache_path` already establishes for a
materialised evaluation sample: a digest of the recipe, not of wherever a
caller happens to be working, names the cache file.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

#: Default root for :func:`cached_constant_mask` / :func:`cached_flat_pixel_mask`,
#: shared by every caller that opts into caching (`geosae.eval.pool`,
#: `geosae.eval.registry`) so they land on the same cache files.
DEFAULT_QUALITY_CACHE_ROOT = "data/eval_cache/quality"


def patch_spatial_std(emb: np.ndarray) -> np.ndarray:
    """Mean over embedding dims of the per-pixel spatial std, per patch.

    ``emb`` is ``(..., H, W, D)``; returns the leading shape. A constant patch
    (all pixels equal) scores ~0.
    """
    emb = np.asarray(emb, dtype=np.float32)
    *lead, h, w, d = emb.shape
    flat = emb.reshape(*lead, h * w, d)
    return flat.std(axis=-2).mean(axis=-1)


def has_flat_pixel(emb: np.ndarray) -> np.ndarray:
    """Whether a patch holds a *flat* pixel (every dim the same number).

    ``emb`` is ``(..., H, W, D)``; returns the leading shape. Tested on the raw
    values, so a transform applied before this check would not "unflatten" a
    fill pixel.
    """
    emb = np.asarray(emb)
    return np.isclose(emb, emb[..., :1], atol=1e-6, equal_nan=True).all(axis=-1).any(axis=(-2, -1))


#: Default per-pixel magnitude threshold for :func:`has_extreme_value` /
#: :func:`scan_extreme_value_cells`, well above any real (near unit-norm)
#: embedding value and well below the ~1e27-2.9e27 Tessera sentinels, so
#: AlphaEarth is untouched by the default.
DEFAULT_MAX_ABS_VALUE = 1e6


def has_extreme_value(emb: np.ndarray, max_abs_value: float = DEFAULT_MAX_ABS_VALUE) -> np.ndarray:
    """Whether a patch holds a pixel with an out-of-range magnitude in any dim.

    ``emb`` is ``(..., H, W, D)``; returns the leading shape. Compared in
    float64 and never squared, so a sentinel value that would overflow
    float32 on squaring (e.g. Tessera's ~1e27-2.9e27 fill values) cannot
    overflow the check itself.

    Args:
        emb: Raw embedding block, any leading batch shape.
        max_abs_value: A pixel is flagged if any dim's magnitude exceeds this.

    Returns:
        Boolean array over the leading shape.
    """
    emb = np.asarray(emb, dtype=np.float64)
    return (np.abs(emb) > max_abs_value).any(axis=-1).any(axis=(-2, -1))


def fill_pixels(emb: np.ndarray, max_abs_value: float = DEFAULT_MAX_ABS_VALUE) -> np.ndarray:
    """Which pixels are fill values rather than embeddings: the pixel-level form of the cell checks.

    A pixel is fill when any dim is non-finite, when every dim carries the same number (a flat
    pixel: AlphaEarth's dequantised nodata code, Tessera's all-zero pixel), or when any dim's
    magnitude exceeds ``max_abs_value`` (Tessera's sentinel).

    Args:
        emb: ``(..., D)`` raw embeddings, before any transform.
        max_abs_value: Magnitude threshold, compared in float64.

    Returns:
        Boolean array over the leading shape.
    """
    e = np.asarray(emb, dtype=np.float64)
    flat = np.isclose(e, e[..., :1], atol=1e-6).all(axis=-1)
    return ~np.isfinite(e).all(axis=-1) | flat | (np.abs(np.nan_to_num(e)) > float(max_abs_value)).any(axis=-1)


def _scan_cells(
    zarr_path: str,
    embedding_name: str,
    cells: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_workers: int = 16,
) -> np.ndarray:
    """Boolean mask over ``cells``, ``score_fn`` applied chunk-by-chunk.

    Reads each embedding chunk once (rather than once per cell), so it stays
    cheap even for a scattered cell list.
    """
    g = zarr.open_group(str(zarr_path), mode="r")
    emb = g[embedding_name]
    chunk0 = int(emb.chunks[0])
    cells = np.asarray(cells, dtype=np.int64)
    out = np.zeros(cells.size, dtype=bool)

    by_chunk: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pos, c in enumerate(cells):
        by_chunk[int(c) // chunk0].append((pos, int(c)))

    def _scan(item):
        cid, entries = item
        start = cid * chunk0
        block = np.asarray(emb[start : min(start + chunk0, emb.shape[0])])
        scored = score_fn(block)  # (chunk_len,)
        for pos, c in entries:
            out[pos] = scored[c - start]

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_scan, by_chunk.items()))
    return out


def scan_constant_cells(
    zarr_path: str,
    embedding_name: str,
    cells: np.ndarray,
    *,
    thresh: float = 1e-3,
    n_workers: int = 16,
) -> np.ndarray:
    """Boolean mask over ``cells`` marking degenerate (spatial std < ``thresh``)."""
    return _scan_cells(
        zarr_path,
        embedding_name,
        cells,
        lambda block: patch_spatial_std(block) < thresh,
        n_workers=n_workers,
    )


def scan_flat_pixel_cells(
    zarr_path: str,
    embedding_name: str,
    cells: np.ndarray,
    *,
    n_workers: int = 16,
) -> np.ndarray:
    """Boolean mask over ``cells`` marking those holding at least one flat pixel."""
    return _scan_cells(zarr_path, embedding_name, cells, has_flat_pixel, n_workers=n_workers)


def scan_extreme_value_cells(
    zarr_path: str,
    embedding_name: str,
    cells: np.ndarray,
    *,
    max_abs_value: float = DEFAULT_MAX_ABS_VALUE,
    n_workers: int = 16,
) -> np.ndarray:
    """Boolean mask over ``cells`` marking those holding an out-of-range pixel."""
    return _scan_cells(
        zarr_path,
        embedding_name,
        cells,
        lambda block: has_extreme_value(block, max_abs_value),
        n_workers=n_workers,
    )


def _digest(zarr_path: str, embedding_name: str, kind: str, thresh: float | None) -> str:
    h = hashlib.sha256()
    h.update(str(zarr_path).encode())
    h.update(str(embedding_name).encode())
    h.update(str(kind).encode())
    if thresh is not None:
        h.update(repr(float(thresh)).encode())
    return h.hexdigest()[:16]


def store_fingerprint(data_zarr: str, *, split_name: str, embedding_name: str) -> str:
    """Short digest of the store contents a pool -- or a quality-mask cache -- rests on.

    Hashes the store's cell count, the ``(value, count)`` histogram of the split
    labels, the embedding availability mask and the ``cell_indices`` provenance
    array (when present) — the parts that decide *which* cells a pool could have
    drawn. Recomputing it later detects a store that changed under an existing
    pool or cache. Cheap: metadata arrays only, never the embedding values
    themselves (that full read is exactly what :func:`cached_constant_mask` /
    :func:`cached_flat_pixel_mask` exist to avoid paying twice).

    Args:
        data_zarr: Path to the patch store.
        split_name: Split-label array in the store.
        embedding_name: Embedding whose ``mask_<name>`` availability is hashed.

    Returns:
        16 hex characters of a sha256 digest.
    """
    group = zarr.open_group(str(data_zarr), mode="r")
    h = hashlib.sha256()
    h.update(str(int(group.attrs["n_cells"])).encode())
    values, counts = np.unique(np.asarray(group[split_name][:]), return_counts=True)
    h.update(values.astype(np.int64).tobytes())
    h.update(counts.astype(np.int64).tobytes())
    h.update(np.packbits(np.asarray(group[f"mask_{embedding_name}"][:], dtype=bool)).tobytes())
    if "cell_indices" in group:
        h.update(np.asarray(group["cell_indices"][:], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def quality_cache_path(
    zarr_path: str,
    embedding_name: str,
    kind: str,
    *,
    split_name: str,
    thresh: float | None = None,
    cache_root: str | Path = DEFAULT_QUALITY_CACHE_ROOT,
) -> Path:
    """Canonical, content-addressed path for a whole-store quality mask.

    Mirrors :func:`geosae.eval.pool.default_sample_cache_path`'s pattern: keyed
    by the scan's own recipe (store, embedding, kind, threshold) *and*
    :func:`store_fingerprint` rather than by wherever a caller happens to be
    working, so every caller against the same store lands on the same cache
    file by construction -- and a store that changed underneath an existing
    cache (a re-fetch, a re-blocked split) lands on a *different* file, so a
    stale mask is never read back as if it were fresh.

    Args:
        zarr_path: Store path.
        embedding_name: Embedding array the scan reads.
        kind: ``"constant"`` or ``"flat_pixel"``.
        split_name: Split-label array, forwarded to :func:`store_fingerprint`
            only -- the scan itself covers every cell in the store, not one
            split, but the fingerprint needs some split's histogram to hash.
        thresh: The degenerate-cell threshold (``constant`` only).
        cache_root: Project-level directory the cache lives under.

    Returns:
        ``<cache_root>/<kind>-<digest>.npy``.
    """
    fp = store_fingerprint(zarr_path, split_name=split_name, embedding_name=embedding_name)
    digest = _digest(zarr_path, embedding_name, kind, thresh)
    return Path(cache_root) / f"{kind}-{digest}-{fp}.npy"


def _whole_store_cells(zarr_path: str) -> np.ndarray:
    n_total = int(zarr.open_group(str(zarr_path), mode="r").attrs["n_cells"])
    return np.arange(n_total, dtype=np.int64)


def cached_constant_mask(
    zarr_path: str,
    embedding_name: str,
    *,
    split_name: str = "split_210k",
    thresh: float = 1e-3,
    n_workers: int = 16,
    cache_root: str | Path = DEFAULT_QUALITY_CACHE_ROOT,
) -> np.ndarray:
    """:func:`scan_constant_cells` over every cell of the store, memoised to disk.

    The scan is a pure function of ``(zarr_path, embedding_name, thresh)`` —
    paid once per store, ever, rather than once per
    :class:`~geosae.data.dataset.PatchData` built against it (see the module
    docstring for the cost this removes) — until the store's own content
    changes, at which point :func:`quality_cache_path`'s fingerprint moves the
    cache to a new file and this scans again rather than returning stale masks.

    Args:
        zarr_path: Store path.
        embedding_name: Embedding array to scan.
        split_name: Forwarded to :func:`store_fingerprint` for cache invalidation.
        thresh: Degenerate-cell spatial-std threshold.
        n_workers: Threads for the scan, forwarded to :func:`scan_constant_cells`.
        cache_root: Project-level directory the cache lives under.

    Returns:
        ``(n_total_cells,)`` bool, indexable by any cell id in the store.
    """
    path = quality_cache_path(
        zarr_path, embedding_name, "constant", split_name=split_name, thresh=thresh, cache_root=cache_root
    )
    if path.exists():
        print(f"[geosae] quality cache hit (constant): {path.name}", flush=True)
        return np.load(path)
    print(f"[geosae] quality cache miss (constant), scanning the whole store -> {path.name}", flush=True)
    bad = scan_constant_cells(
        zarr_path, embedding_name, _whole_store_cells(zarr_path), thresh=thresh, n_workers=n_workers
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bad)
    return bad


def cached_flat_pixel_mask(
    zarr_path: str,
    embedding_name: str,
    *,
    split_name: str = "split_210k",
    n_workers: int = 16,
    cache_root: str | Path = DEFAULT_QUALITY_CACHE_ROOT,
) -> np.ndarray:
    """:func:`scan_flat_pixel_cells` over every cell of the store, memoised to disk.

    See :func:`cached_constant_mask` — the same pure-function argument and
    fingerprint-based invalidation apply, minus the threshold.

    Args:
        zarr_path: Store path.
        embedding_name: Embedding array to scan.
        split_name: Forwarded to :func:`store_fingerprint` for cache invalidation.
        n_workers: Threads for the scan, forwarded to :func:`scan_flat_pixel_cells`.
        cache_root: Project-level directory the cache lives under.

    Returns:
        ``(n_total_cells,)`` bool, indexable by any cell id in the store.
    """
    path = quality_cache_path(zarr_path, embedding_name, "flat_pixel", split_name=split_name, cache_root=cache_root)
    if path.exists():
        print(f"[geosae] quality cache hit (flat_pixel): {path.name}", flush=True)
        return np.load(path)
    print(f"[geosae] quality cache miss (flat_pixel), scanning the whole store -> {path.name}", flush=True)
    bad = scan_flat_pixel_cells(zarr_path, embedding_name, _whole_store_cells(zarr_path), n_workers=n_workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bad)
    return bad


def cached_extreme_value_mask(
    zarr_path: str,
    embedding_name: str,
    *,
    split_name: str = "split_210k",
    max_abs_value: float = DEFAULT_MAX_ABS_VALUE,
    n_workers: int = 16,
    cache_root: str | Path = DEFAULT_QUALITY_CACHE_ROOT,
) -> np.ndarray:
    """:func:`scan_extreme_value_cells` over every cell of the store, memoised to disk.

    See :func:`cached_constant_mask` — the same pure-function argument and
    fingerprint-based invalidation apply.

    Args:
        zarr_path: Store path.
        embedding_name: Embedding array to scan.
        split_name: Forwarded to :func:`store_fingerprint` for cache invalidation.
        max_abs_value: Per-dim magnitude threshold, forwarded to
            :func:`scan_extreme_value_cells`.
        n_workers: Threads for the scan, forwarded to :func:`scan_extreme_value_cells`.
        cache_root: Project-level directory the cache lives under.

    Returns:
        ``(n_total_cells,)`` bool, indexable by any cell id in the store.
    """
    path = quality_cache_path(
        zarr_path,
        embedding_name,
        "extreme_value",
        split_name=split_name,
        thresh=max_abs_value,
        cache_root=cache_root,
    )
    if path.exists():
        print(f"[geosae] quality cache hit (extreme_value): {path.name}", flush=True)
        return np.load(path)
    print(f"[geosae] quality cache miss (extreme_value), scanning the whole store -> {path.name}", flush=True)
    bad = scan_extreme_value_cells(
        zarr_path, embedding_name, _whole_store_cells(zarr_path), max_abs_value=max_abs_value, n_workers=n_workers
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bad)
    return bad


__all__ = [
    "patch_spatial_std",
    "has_flat_pixel",
    "has_extreme_value",
    "fill_pixels",
    "DEFAULT_MAX_ABS_VALUE",
    "scan_constant_cells",
    "store_fingerprint",
    "scan_flat_pixel_cells",
    "scan_extreme_value_cells",
    "quality_cache_path",
    "cached_constant_mask",
    "cached_flat_pixel_mask",
    "cached_extreme_value_mask",
]

"""Regular lon/lat grids, and coarsening a patch store onto one.

Embedding products come at very different footprints: some are per-pixel rasters
that fit the 64x64 patch store, others ship a single vector per cell of a coarse
global grid. This module provides the second geometry:

* :class:`RegularGrid` -- a node-registered lon/lat grid (cell centres sit *on*
  the coordinate lines, ERA5 convention: latitudes run ``+90`` down to ``-90``,
  longitudes ``-180`` up to ``180 - res``), with vectorised nearest-node lookup.
* :func:`aggregate_categorical` / :func:`aggregate_continuous` -- pure-numpy
  reduction of many samples onto grid nodes: majority class (with the fraction
  the winner actually holds) and nanmean.
* :func:`coarsen_patch_store` -- the two combined: read a 64x64 patch store,
  assign every patch to its nearest node, and write a new store with
  ``patch_size = 1`` in which each row is one node carrying the aggregated
  reference products. The result is a normal geosae store, so the training,
  inference and evaluation stages read it unchanged.

Aggregating the reference products from the *same* patch store the fine-grained
experiments use keeps the label definitions identical across resolutions: only
the support changes, from a 640 m patch to a grid cell.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from geosae.data.download.selection import SplitSpec, compute_split, sample_subset_indices, split_counts
from geosae.data.download.sources import Kind, all_products
from geosae.data.download.store import (
    LABEL_CHUNK_BYTES,
    ArraySchema,
    _dims_attr,
    _mask_schema,
    build_schemas,
    cell_chunk,
    create_store,
)

#: Cells read from the source store per block during coarsening.
DEFAULT_BLOCK_CELLS = 4096


@dataclass(frozen=True)
class RegularGrid:
    """A node-registered regular lon/lat grid.

    Node ``(row, col)`` sits at ``lat = lat_origin - row * resolution_deg`` and
    ``lon = lon_origin + col * resolution_deg``; longitudes wrap, latitudes do
    not. Flat node indices are row-major (``row * n_lon + col``).

    Attributes:
        resolution_deg: Node spacing in degrees.
        n_lat: Number of latitude rows (poles inclusive for a global grid).
        n_lon: Number of longitude columns.
        lat_origin: Latitude of row 0 (degrees north).
        lon_origin: Longitude of column 0 (degrees east).
    """

    resolution_deg: float
    n_lat: int
    n_lon: int
    lat_origin: float = 90.0
    lon_origin: float = -180.0

    @classmethod
    def global_grid(cls, resolution_deg: float) -> RegularGrid:
        """Global grid at ``resolution_deg``, poles inclusive.

        At 0.25 deg this is the 721 x 1440 ERA5 grid, which is also the grid the
        Copernicus-FM embedding map is published on.
        """
        if resolution_deg <= 0 or 180.0 % resolution_deg != 0:
            raise ValueError(f"resolution_deg must divide 180, got {resolution_deg}")
        return cls(
            resolution_deg=float(resolution_deg),
            n_lat=int(round(180.0 / resolution_deg)) + 1,
            n_lon=int(round(360.0 / resolution_deg)),
        )

    @property
    def n_cells(self) -> int:
        """Total number of nodes."""
        return int(self.n_lat * self.n_lon)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_lat, n_lon)`` -- the shape of a field on this grid."""
        return (self.n_lat, self.n_lon)

    def lat_nodes(self) -> np.ndarray:
        """Latitudes of each row, descending, shape ``(n_lat,)``."""
        return self.lat_origin - np.arange(self.n_lat, dtype=np.float64) * self.resolution_deg

    def lon_nodes(self) -> np.ndarray:
        """Longitudes of each column, ascending, shape ``(n_lon,)``."""
        return self.lon_origin + np.arange(self.n_lon, dtype=np.float64) * self.resolution_deg

    def nearest(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Nearest node to each point.

        Args:
            lon: Longitudes in degrees east (any branch; wrapped internally).
            lat: Latitudes in degrees north.

        Returns:
            ``(row, col)`` int64 arrays of the same shape as the inputs.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        row = np.rint((self.lat_origin - lat) / self.resolution_deg).astype(np.int64)
        np.clip(row, 0, self.n_lat - 1, out=row)
        col = np.rint(((lon - self.lon_origin) % 360.0) / self.resolution_deg).astype(np.int64) % self.n_lon
        return row, col

    def flat_index_of(self, row, col) -> np.ndarray:
        """Row-major flat node index from ``(row, col)``."""
        return np.asarray(row, dtype=np.int64) * self.n_lon + np.asarray(col, dtype=np.int64)

    def flat_index(self, lon, lat) -> np.ndarray:
        """Flat index of the node nearest each point."""
        return self.flat_index_of(*self.nearest(lon, lat))

    def centers(self, flat) -> tuple[np.ndarray, np.ndarray]:
        """Node coordinates for flat indices, as ``(lon, lat)`` in degrees.

        Longitudes are returned on the ``[-180, 180)`` branch.
        """
        flat = np.asarray(flat, dtype=np.int64)
        row, col = np.divmod(flat, self.n_lon)
        lat = self.lat_origin - row * self.resolution_deg
        lon = (self.lon_origin + col * self.resolution_deg + 180.0) % 360.0 - 180.0
        return lon, lat


# --------------------------------------------------------------------------- #
# Reductions onto grid nodes
# --------------------------------------------------------------------------- #
def _class_offset(codes: np.ndarray) -> tuple[int, int]:
    """Return ``(offset, n_slots)`` mapping integer codes onto ``[0, n_slots)``."""
    info = np.iinfo(codes.dtype)
    if codes.dtype.itemsize == 1:
        return -info.min, 256
    lo, hi = int(codes.min()), int(codes.max())
    return -lo, hi - lo + 1


def aggregate_categorical(
    codes: np.ndarray, group: np.ndarray, n_groups: int, fill_value: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Majority class per group, ignoring ``fill_value`` pixels.

    Args:
        codes: Integer class codes, shape ``(n_samples, n_pixels)``.
        group: Group index per sample, shape ``(n_samples,)``, in ``[0, n_groups)``.
        n_groups: Number of output groups.
        fill_value: Nodata code, excluded from the vote and from the count.

    Returns:
        ``(mode, modal_fraction, n_valid)``: the winning code per group (or
        ``fill_value`` where a group has no valid pixel), the share of valid
        pixels it holds (``nan`` for empty groups), and the number of valid
        pixels. ``mode`` has the dtype of ``codes``.
    """
    hist, n_valid = _class_histogram(codes, group, n_groups, fill_value)
    offset, _ = _class_offset(codes)
    mode = np.full(n_groups, fill_value, dtype=codes.dtype)
    frac = np.full(n_groups, np.nan, dtype=np.float64)
    has = n_valid > 0
    if has.any():
        best = hist[has].argmax(axis=1)
        mode[has] = (best - offset).astype(codes.dtype)
        frac[has] = hist[has].max(axis=1) / n_valid[has]
    return mode, frac, n_valid


def _class_histogram(
    codes: np.ndarray, group: np.ndarray, n_groups: int, fill_value: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group class counts, shape ``(n_groups, n_slots)``, plus valid totals."""
    codes = np.asarray(codes)
    group = np.asarray(group, dtype=np.int64)
    offset, n_slots = _class_offset(codes)
    hist = np.zeros((n_groups, n_slots), dtype=np.int64)
    n_valid = np.zeros(n_groups, dtype=np.int64)
    accumulate_class_histogram(hist, n_valid, codes, group, fill_value, offset)
    return hist, n_valid


def accumulate_class_histogram(
    hist: np.ndarray,
    n_valid: np.ndarray,
    codes: np.ndarray,
    group: np.ndarray,
    fill_value: int,
    offset: int,
) -> None:
    """Add one block of samples into a running per-group class histogram.

    The block's groups are compacted before counting, so the temporary is sized
    by the *distinct* groups in the block rather than by ``len(hist)`` -- which
    is what keeps a streaming pass over a million-cell store affordable.

    Args:
        hist: ``(n_groups, n_slots)`` int64 counts, updated in place.
        n_valid: ``(n_groups,)`` int64 valid-pixel totals, updated in place.
        codes: Integer class codes, shape ``(n_samples, n_pixels)``.
        group: Group index per sample, shape ``(n_samples,)``.
        fill_value: Nodata code, not counted.
        offset: Added to a code to get its slot (see :func:`_class_offset`).
    """
    n_slots = hist.shape[1]
    flat_group, inverse = np.unique(np.asarray(group, dtype=np.int64), return_inverse=True)
    slots = codes.astype(np.int64, copy=False) + offset
    keep = codes != fill_value
    keys = (np.broadcast_to(inverse[:, None], codes.shape) * n_slots + slots)[keep]
    counts = np.bincount(keys.ravel(), minlength=flat_group.size * n_slots)
    hist[flat_group] += counts.reshape(flat_group.size, n_slots)
    n_valid[flat_group] += np.bincount(
        np.broadcast_to(inverse[:, None], codes.shape)[keep].ravel(), minlength=flat_group.size
    )


def aggregate_continuous(values: np.ndarray, group: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean per group over non-NaN pixels.

    Args:
        values: Float values, shape ``(n_samples, n_pixels)`` or
            ``(n_samples, n_pixels, n_bands)``.
        group: Group index per sample, shape ``(n_samples,)``.
        n_groups: Number of output groups.

    Returns:
        ``(mean, n_valid)`` of shape ``(n_groups,)`` / ``(n_groups, n_bands)``.
        Groups with no valid pixel are ``nan`` with a count of zero.
    """
    band_shape = values.shape[2:]
    total = np.zeros((n_groups, *band_shape), dtype=np.float64)
    n_valid = np.zeros((n_groups, *band_shape), dtype=np.int64)
    accumulate_sums(total, n_valid, values, group)
    with np.errstate(invalid="ignore"):
        mean = np.where(n_valid > 0, total / np.maximum(n_valid, 1), np.nan)
    return mean, n_valid


def accumulate_sums(total: np.ndarray, n_valid: np.ndarray, values: np.ndarray, group: np.ndarray) -> None:
    """Add one block of samples into running per-group sums and counts."""
    group = np.asarray(group, dtype=np.int64)
    finite = np.isfinite(values)
    per_sample_sum = np.where(finite, values, 0.0).sum(axis=1)
    per_sample_n = finite.sum(axis=1)
    np.add.at(total, group, per_sample_sum)
    np.add.at(n_valid, group, per_sample_n)


# --------------------------------------------------------------------------- #
# Coarsening a patch store onto a grid
# --------------------------------------------------------------------------- #
def _count_schema(n_cells: int) -> ArraySchema:
    return ArraySchema(
        dtype="int32",
        shape=(n_cells,),
        chunks=(min(10_000, max(n_cells, 1)),),
        fill_value=0,
        dim_names=("cell",),
        attrs=_dims_attr(("cell",)),
    )


def _fill_value_of(src, array: str):
    """Nodata sentinel for a source array: store metadata, then registry, then dtype.

    Real stores carry layers that predate the product registry, so the store's
    own ``modalities`` attr is the authority and the registry only fills gaps.
    """
    modality = dict(src.attrs.get("modalities", {})).get(array)
    if modality is not None and "fill_value" in modality:
        return modality["fill_value"]
    for spec in all_products().values():
        if spec.array == array:
            return spec.fill_value
    dtype = np.dtype(src[array].dtype)
    if dtype.kind == "f":
        return float("nan")
    return 255 if dtype == np.uint8 else 0


def _source_label_schema(src, array: str, n_cells: int, fill) -> ArraySchema:
    """Single-pixel schema mirroring a source array's dtype and band count."""
    bands = tuple(src[array].shape[3:])
    dims = ("cell", "y", "x", f"{array}_band") if bands else ("cell", "y", "x")
    dtype = str(np.dtype(src[array].dtype))
    tail = (1, 1, *bands)
    return ArraySchema(
        dtype=dtype,
        shape=(n_cells, *tail),
        chunks=(cell_chunk(dtype, tail, LABEL_CHUNK_BYTES, n_cells), *tail),
        fill_value=float("nan") if dtype.startswith("float") else fill,
        dim_names=dims,
        attrs=_dims_attr(dims),
    )


def coarsen_patch_store(
    src_path,
    dst_path,
    *,
    grid: RegularGrid,
    products: list[str] | None = None,
    embeddings: tuple[str, ...] = (),
    splits: dict[str, SplitSpec] | None = None,
    seed: int = 42,
    min_source_patches: int = 1,
    block_cells: int = DEFAULT_BLOCK_CELLS,
    n_workers: int = 1,
    progress: bool = False,
) -> dict:
    """Aggregate a patch store onto grid nodes and write a ``patch_size=1`` store.

    Every source patch is assigned to its nearest node. Nodes holding at least
    ``min_source_patches`` patches become rows of the output store, in a seeded
    scattered order so that a contiguous row range is a geographically mixed
    split (the same convention as
    :func:`~geosae.data.download.selection.sample_subset_indices`). Categorical
    products are reduced by majority vote over all pixels of all patches in the
    node, continuous products by nanmean.

    The embedding arrays are created empty; fill them with a product defined on
    the same grid (see
    :func:`~geosae.data.download.fetchers.copernicus_fm.add_copernicusfm_embeddings`).

    Args:
        src_path: Source patch store (patch-store layout).
        dst_path: Output store path.
        grid: Target grid.
        products: Source array names to aggregate. ``None`` takes every
            per-cell reference array of the source store.
        embeddings: Embedding arrays to create (empty) in the output.
        splits: ``{name: SplitSpec}`` split arrays to compute.
        seed: Seed for the row scatter.
        min_source_patches: Minimum patches a node needs to be kept.
        block_cells: Source cells read per block.
        n_workers: Products aggregated concurrently. The pass is IO-bound, so
            threads help, but each product holds its own accumulator -- a wide
            categorical product can need a gigabyte.
        progress: Print per-product progress.

    Returns:
        Summary dict with ``n_cells``, ``n_source_patches``, ``grid`` and, per
        product, the mean modal fraction (categorical) or valid-cell count.
    """
    import zarr

    src = zarr.open_group(str(src_path), mode="r")
    lon, lat = np.asarray(src["lon"][:]), np.asarray(src["lat"][:])
    n_src = lon.size

    flat = grid.flat_index(lon, lat)
    nodes, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    keep_node = counts >= int(min_source_patches)
    node_row = np.where(keep_node, np.cumsum(keep_node) - 1, -1)
    group = node_row[inverse]  # source cell -> output group, -1 if dropped
    n_cells = int(keep_node.sum())
    kept_nodes = nodes[keep_node]
    if n_cells == 0:
        raise ValueError(
            f"no grid node holds at least min_source_patches={min_source_patches} "
            f"of the {n_src} source patches at {grid.resolution_deg} deg"
        )

    products = _resolve_products(src, products)
    fills = {a: _fill_value_of(src, a) for a in products}

    schemas = build_schemas(
        [], n_cells, embeddings=embeddings, splits=tuple(f"split_{n}" for n in (splits or {})), patch_size=1
    )
    for array in products:
        schemas[array] = _source_label_schema(src, array, n_cells, fills[array])
        schemas[f"mask_{array}"] = _mask_schema(array, n_cells)
    schemas["n_source_patches"] = _count_schema(n_cells)
    out = create_store(dst_path, schemas, n_cells=n_cells, patch_size=1)

    # Scattered row order: store row r holds kept node `order[r]`, so any
    # per-group array is written to disk as `value[order]`.
    order = sample_subset_indices(max(n_cells, 1), max(n_cells, 1), seed)[:n_cells]

    node_lon, node_lat = grid.centers(kept_nodes)
    out["cell_indices"][:] = kept_nodes[order]
    out["lon"][:] = node_lon[order]
    out["lat"][:] = node_lat[order]
    out["crs"][:] = np.full(n_cells, "EPSG:4326")
    out["transform"][:] = np.tile([grid.resolution_deg, 0.0, 0.0, 0.0, -grid.resolution_deg, 0.0], (n_cells, 1))
    out["x"][:] = [0]
    out["y"][:] = [0]
    out["n_source_patches"][:] = counts[keep_node][order].astype(np.int32)

    summary: dict = {
        "n_cells": n_cells,
        "n_source_patches": int(n_src),
        "grid": {"resolution_deg": grid.resolution_deg, "n_lat": grid.n_lat, "n_lon": grid.n_lon},
        "products": {},
    }

    def run_one(array: str) -> tuple[str, dict]:
        kind = Kind.CONTINUOUS if src[array].dtype.kind == "f" else Kind.CATEGORICAL
        return array, _coarsen_product(
            src, out, array, kind, fills[array], group, order, n_cells, block_cells, progress
        )

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as pool:
            for array, stat in pool.map(run_one, products):
                summary["products"][array] = stat
    else:
        for array in products:
            summary["products"][array] = run_one(array)[1]

    # Split ranges cover every row: rows whose embedding is missing are dropped
    # at read time by PatchData's `mask_<embedding>`, which the embedding is
    # free to be attached after this build.
    for name, spec in (splits or {}).items():
        labels = compute_split(spec, n_cells, np.ones(n_cells, dtype=bool))
        out[f"split_{name}"][:] = labels
        summary.setdefault("splits", {})[name] = split_counts(labels)

    out.attrs["coarsen"] = summary
    return summary


def _resolve_products(src, products: list[str] | None) -> list[str]:
    """Reference arrays to aggregate: registered products plus the store's own labels.

    A store may carry layers the registry does not know, so the store's
    ``modalities`` attr contributes too. Embeddings and
    anything that is not a per-cell raster are excluded.
    """
    if products is not None:
        missing = [p for p in products if p not in src]
        if missing:
            raise KeyError(f"products not in source store: {missing}")
        return list(products)
    n = int(src.attrs["n_cells"])
    known = {s.array for s in all_products().values() if s.kind is not Kind.EMBEDDING}
    known |= {a for a, m in dict(src.attrs.get("modalities", {})).items() if m.get("category") == "label"}
    embeddings = {s.array for s in all_products().values() if s.kind is Kind.EMBEDDING}
    return sorted(
        a for a, v in src.arrays() if a in known and a not in embeddings and v.shape[:1] == (n,) and v.ndim in (3, 4)
    )


def _code_range(arr, block_cells: int) -> tuple[int, int]:
    """Global ``(min, max)`` of an integer store array, streamed in blocks."""
    lo, hi = None, None
    for start in range(0, arr.shape[0], block_cells):
        block = np.asarray(arr[start : start + block_cells])
        b_lo, b_hi = int(block.min()), int(block.max())
        lo = b_lo if lo is None else min(lo, b_lo)
        hi = b_hi if hi is None else max(hi, b_hi)
    return (0, 0) if lo is None else (lo, hi)


#: Refuse a class histogram larger than this many int32 entries (~4 GB).
MAX_HISTOGRAM_ENTRIES = 1_000_000_000


def _coarsen_product(src, out, array, kind, fill, group, order, n_cells, block_cells, progress) -> dict:
    """Stream one source array in blocks and write its aggregate to ``out``."""
    arr = src[array]
    n_src = arr.shape[0]
    band_shape = arr.shape[3:]
    src_mask = np.asarray(src[f"mask_{array}"][:], dtype=bool) if f"mask_{array}" in src else None

    if kind is Kind.CONTINUOUS:
        total = np.zeros((n_cells, *band_shape), dtype=np.float64)
        n_valid = np.zeros((n_cells, *band_shape), dtype=np.int64)
    else:
        if arr.dtype.itemsize == 1:
            offset, n_slots = -int(np.iinfo(arr.dtype).min), 256
        else:
            # Wider codes are sparse in their dtype range, so bound the
            # histogram by the codes the array actually contains.
            lo, hi = _code_range(arr, block_cells)
            offset, n_slots = -lo, hi - lo + 1
        if n_cells * n_slots > MAX_HISTOGRAM_ENTRIES:
            raise MemoryError(
                f"{array!r} spans {n_slots} class codes over {n_cells} cells; "
                f"the majority-vote histogram would exceed {MAX_HISTOGRAM_ENTRIES} entries"
            )
        # Bands of a categorical product are independent class systems (Overture
        # carries five), so each gets its own vote.
        n_bands = int(np.prod(band_shape)) if band_shape else 1
        hist = np.zeros((n_bands, n_cells, n_slots), dtype=np.int32)
        n_valid = np.zeros((n_bands, n_cells), dtype=np.int64)

    for lo in range(0, n_src, block_cells):
        hi = min(lo + block_cells, n_src)
        g = group[lo:hi]
        sel = g >= 0
        if src_mask is not None:
            sel &= src_mask[lo:hi]
        if not sel.any():
            continue
        block = np.asarray(arr[lo:hi])[sel]
        if kind is Kind.CONTINUOUS:
            accumulate_sums(
                total, n_valid, block.reshape(block.shape[0], -1, *band_shape).astype(np.float64, copy=False), g[sel]
            )
        else:
            flat = block.reshape(block.shape[0], -1, n_bands)
            for b in range(n_bands):
                accumulate_class_histogram(hist[b], n_valid[b], flat[..., b], g[sel], fill, offset)
        if progress:
            print(f"[geosae] {array}: {hi}/{n_src} source cells", flush=True)

    if kind is Kind.CONTINUOUS:
        with np.errstate(invalid="ignore"):
            value = np.where(n_valid > 0, total / np.maximum(n_valid, 1), np.nan)
        value = value.astype(out[array].dtype)
        ok = n_valid.reshape(n_cells, -1).sum(axis=1) > 0
        stat = {"kind": "continuous", "valid_cells": int(ok.sum())}
    else:
        value = np.full((n_bands, n_cells), fill, dtype=out[array].dtype)
        frac = np.full((n_bands, n_cells), np.nan)
        for b in range(n_bands):
            has = n_valid[b] > 0
            if has.any():
                value[b, has] = (hist[b][has].argmax(axis=1) - offset).astype(out[array].dtype)
                frac[b, has] = hist[b][has].max(axis=1) / n_valid[b][has]
        ok = (n_valid > 0).any(axis=0)
        value = np.moveaxis(value, 0, -1).reshape(n_cells, *band_shape)
        stat = {
            "kind": "categorical",
            "valid_cells": int(ok.sum()),
            "mean_modal_fraction": float(np.nanmean(frac)) if ok.any() else float("nan"),
        }

    out[array][:] = value.reshape(n_cells, 1, 1, *band_shape)[order]
    if f"mask_{array}" in out:
        out[f"mask_{array}"][:] = ok[order]
    return stat


__all__ = [
    "DEFAULT_BLOCK_CELLS",
    "RegularGrid",
    "accumulate_class_histogram",
    "accumulate_sums",
    "aggregate_categorical",
    "aggregate_continuous",
    "coarsen_patch_store",
]

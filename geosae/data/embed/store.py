"""Stream a Sentinel-2 patch store through a backbone and write the token grids back."""

from __future__ import annotations

import logging

import numpy as np

from geosae.data.download.store import COORD_CHUNK, EMBED_CHUNK_BYTES, ArraySchema, cell_chunk
from geosae.data.embed.base import S2_REFLECTANCE_SCALE, EmbeddingBackbone

logger = logging.getLogger(__name__)


def _dims(*names: str) -> dict:
    """xarray-compatible dimension-name attrs."""
    return {"_ARRAY_DIMENSIONS": list(names)}


def embedding_schema(name: str, n_cells: int, grid: int, dim: int, dtype: str = "float16") -> ArraySchema:
    """Layout of one embedding array holding a ``grid x grid`` token map per cell.

    Args:
        name: Store array name; its feature axis is called ``feature_<name>``.
        n_cells: Number of store rows.
        grid: Token-grid side length.
        dim: Embedding width.
        dtype: Store dtype.

    Returns:
        The array schema, chunked to :data:`~geosae.data.download.store.EMBED_CHUNK_BYTES`.
    """
    dims = ("cell", "y", "x", f"feature_{name}")
    chunk = cell_chunk(dtype, (grid, grid, dim), EMBED_CHUNK_BYTES, n_cells)
    return ArraySchema(
        dtype=dtype,
        shape=(n_cells, grid, grid, dim),
        chunks=(chunk, grid, grid, dim),
        fill_value=float("nan"),
        dim_names=dims,
        attrs={**_dims(*dims), "embedding_dim": dim, "coordinates": "cell_indices lat lon"},
    )


def _to_reflectance(dn: np.ndarray, fill: int, scale: float) -> np.ndarray:
    """Digital numbers to float32 reflectance, fill pixels set to zero."""
    out = dn.astype(np.float32)
    out[dn == fill] = 0.0
    return out / scale


def center_crop(patches: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop ``(B, H, W, C)`` patches to ``size`` pixels square.

    Args:
        patches: Batch of square patches.
        size: Target side length in pixels.

    Returns:
        The centred ``(B, size, size, C)`` view.

    Raises:
        ValueError: If ``size`` exceeds the patch, or the crop is not centred
            on an integer pixel offset.
    """
    height = patches.shape[1]
    if size > height:
        raise ValueError(f"crop {size} exceeds the {height} px patch")
    if (height - size) % 2:
        raise ValueError(f"cropping {height} px to {size} px is not pixel-centred")
    off = (height - size) // 2
    return patches[:, off : off + size, off : off + size, :]


def embed_store(
    group,
    backbone: EmbeddingBackbone,
    target_array: str,
    *,
    source_array: str = "s2",
    source_fill: int = -32768,
    scale: float = S2_REFLECTANCE_SCALE,
    batch_cells: int = 32,
    dtype: str = "float16",
    crop_px: int | None = None,
) -> str:
    """Encode every valid cell of ``source_array`` and write ``target_array``.

    The target array is created if absent, sized from the backbone's stride and
    latent width, together with its ``feature_*`` coordinate and a
    ``mask_<target_array>`` flag marking the rows that were written. Rows whose
    source mask is ``False`` are skipped and left at the fill value.

    Args:
        group: Open zarr group holding ``source_array`` and ``mask_<source>``.
        backbone: The frozen encoder to run.
        target_array: Store array name to write.
        source_array: Reflectance array name.
        source_fill: Digital-number fill marking a pixel with no observation.
        scale: Digital numbers per unit reflectance.
        batch_cells: Cells encoded per forward pass.
        dtype: Store dtype of the written embeddings.
        crop_px: Centre-crop each patch to this many pixels before encoding.
            Use it when a backbone's usable footprint is smaller than the
            store's patch; the value is recorded on the written array as
            ``crop_px`` so the ground area an embedding covers stays explicit.

    Returns:
        The name of the written array.

    Raises:
        KeyError: If ``source_array`` is not in the store.
    """
    if source_array not in group:
        raise KeyError(f"store has no array {source_array!r}")
    src = group[source_array]
    n_cells, height, width = src.shape[0], src.shape[1], src.shape[2]
    if width != height:
        raise ValueError(f"{source_array} patches must be square, got {height}x{width}")
    encoded_px = height if crop_px is None else int(crop_px)
    grid = backbone.output_grid(encoded_px)

    schema = embedding_schema(target_array, n_cells, grid, backbone.latent_dim, dtype)
    schema.attrs["crop_px"] = encoded_px
    if target_array not in group:
        arr = group.create_array(
            target_array,
            shape=schema.shape,
            chunks=schema.chunks,
            dtype=schema.dtype,
            fill_value=schema.fill_value,
        )
        arr.attrs.update(schema.attrs)
        feature = f"feature_{target_array}"
        coord = group.create_array(feature, shape=(backbone.latent_dim,), chunks=(backbone.latent_dim,), dtype="int64")
        coord[:] = np.arange(backbone.latent_dim, dtype=np.int64)
        coord.attrs.update(_dims(feature))
        flag = group.create_array(
            f"mask_{target_array}",
            shape=(n_cells,),
            chunks=(min(COORD_CHUNK, n_cells),),
            dtype="bool",
            fill_value=False,
        )
        flag.attrs.update(_dims("cell"))
    dst = group[target_array]
    dst_mask = group[f"mask_{target_array}"]

    src_mask = np.asarray(group[f"mask_{source_array}"][:]) if f"mask_{source_array}" in group else None
    rows = np.arange(n_cells) if src_mask is None else np.where(src_mask)[0]
    logger.info("Encoding %d/%d cells with %s -> %s", rows.size, n_cells, type(backbone).__name__, target_array)

    for start in range(0, rows.size, batch_cells):
        block = rows[start : start + batch_cells]
        reflectance = _to_reflectance(np.asarray(src.oindex[block]), source_fill, scale)
        if crop_px is not None:
            reflectance = center_crop(reflectance, encoded_px)
        tokens = backbone.embed(reflectance)
        dst.oindex[block] = tokens.astype(dtype)
        dst_mask.oindex[block] = True
    return target_array


def embedding_stats(
    group,
    array: str,
    *,
    rows: np.ndarray | None = None,
    quantile: float = 0.001,
    max_tokens: int = 2_000_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-dimension standardisation statistics for an embedding array.

    Foundation-model embeddings are not unit-norm, so the SAE needs the
    per-dimension mean/std and clipping bounds that
    :meth:`~geosae.data.dataset.Transforms.from_stats_npz` consumes. Statistics
    come from a seeded random subsample of the tokens, so the result is
    reproducible and its cost independent of store size.

    Args:
        group: Open zarr group.
        array: Embedding array name.
        rows: Store rows to draw from (typically the train split). ``None`` uses
            every row flagged in ``mask_<array>``.
        quantile: Lower/upper tail clipped by ``clip_lo`` / ``clip_hi``.
        max_tokens: Tokens drawn for the estimate.
        seed: Subsampling seed.

    Returns:
        ``{"mean", "std", "clip_lo", "clip_hi"}``, each ``(D,)`` float32.

    Raises:
        ValueError: If the sampled tokens hold no finite values.
    """
    arr = group[array]
    if rows is None:
        mask = group[f"mask_{array}"] if f"mask_{array}" in group else None
        rows = np.arange(arr.shape[0]) if mask is None else np.where(np.asarray(mask[:]))[0]
    rows = np.asarray(rows, dtype=np.int64)
    tokens_per_cell = int(arr.shape[1]) * int(arr.shape[2])
    n_cells = max(1, min(rows.size, max_tokens // tokens_per_cell))
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(rows, size=n_cells, replace=False)) if n_cells < rows.size else rows

    sample = np.asarray(arr.oindex[picked]).reshape(-1, arr.shape[-1]).astype(np.float32)
    sample = sample[np.isfinite(sample).all(axis=1)]
    if sample.size == 0:
        raise ValueError(f"{array} has no finite tokens to estimate statistics from")
    lo, hi = np.quantile(sample, [quantile, 1.0 - quantile], axis=0)
    std = sample.std(axis=0)
    return {
        "mean": sample.mean(axis=0).astype(np.float32),
        # A dead dimension would otherwise divide by zero.
        "std": np.where(std > 0, std, 1.0).astype(np.float32),
        "clip_lo": lo.astype(np.float32),
        "clip_hi": hi.astype(np.float32),
    }


__all__ = ["embedding_schema", "embed_store", "embedding_stats", "center_crop"]

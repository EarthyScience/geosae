"""Resampling source rasters onto a patch's pixel grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import rasterio.warp
from odc.geo.geobox import GeoBox
from rasterio.crs import CRS
from rasterio.warp import Resampling


def geobox_from_profile(profile: Any) -> GeoBox:
    """The :class:`~odc.geo.geobox.GeoBox` of a rasterio dataset profile (``src.profile``)."""
    return GeoBox((profile["height"], profile["width"]), profile["transform"], profile["crs"])


def reproject_to_geobox(
    data: np.ndarray,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: Resampling = Resampling.nearest,
    nodata: float | None = None,
) -> np.ndarray:
    """Resample ``data`` from ``src_geobox`` onto the pixel grid of ``dst_geobox``.

    Destination pixels no source pixel covers are set to ``nodata``, or, when it
    is ``None``, to NaN for float data and 0 otherwise, so a patch straddling a
    source-tile edge is recognisably incomplete rather than silently zero.

    Args:
        data: ``(height, width)`` or ``(bands, height, width)`` array on ``src_geobox``.
        src_geobox: Grid of ``data``.
        dst_geobox: Target grid.
        resampling: Rasterio resampling method.
        nodata: Source fill value, also written where the source has no data.

    Returns:
        Array of ``data``'s dtype and band layout on ``dst_geobox``.

    Raises:
        ValueError: ``data`` is not 2D/3D or does not match ``src_geobox``.
    """
    if data.ndim not in (2, 3):
        raise ValueError("data must be a 2D or 3D array")
    bands = data[None] if data.ndim == 2 else data
    if bands.shape[1:] != (src_geobox.height, src_geobox.width):
        raise ValueError(f"data shape {data.shape} does not match the source geobox {src_geobox.shape}")
    fill = nodata if nodata is not None else (np.nan if np.issubdtype(data.dtype, np.floating) else 0)
    out = np.full((len(bands), dst_geobox.height, dst_geobox.width), fill, dtype=data.dtype)
    # rasterio's own CRS type: odc.geo CRS objects do not carry every projection through.
    src_crs, dst_crs = CRS.from_user_input(src_geobox.crs), CRS.from_user_input(dst_geobox.crs)
    for band, dst in zip(bands, out, strict=True):
        rasterio.warp.reproject(
            source=band,
            destination=dst,
            src_transform=src_geobox.transform,
            src_crs=src_crs,
            dst_transform=dst_geobox.transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=nodata,
            dst_nodata=fill,
        )
    return out[0] if data.ndim == 2 else out


__all__ = ["geobox_from_profile", "reproject_to_geobox"]

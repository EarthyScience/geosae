"""Tile-grouped raster fetchers with cross-tile stitching.

Provides a base class :class:`TiledRasterFetcher` for downloading tiled raster
data (COGs, GeoTIFFs) with automatic stitching at tile boundaries. The same
compositing approach is used by
:class:`~geosae.data.download.fetchers.tessera_embeddings.TesseraFetcher`.

Concrete subclasses implement :meth:`tile_key`, :meth:`fetch_tile`, and
:meth:`candidate_tiles` for each data product.

Example::

    class ESAWorldCoverFetcher(TiledRasterFetcher):
        def tile_key(self, lon, lat):
            return (int(lon // 3) * 3, int(lat // 3) * 3)
        ...

    fetcher = ESAWorldCoverFetcher()
    patch = fetcher.fetch_patch(11.5, 48.1, dst_geobox, Resampling.nearest)
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import rasterio.warp
import rasterio.windows
from odc.geo.geobox import GeoBox
from rasterio.transform import Affine
from rasterio.warp import Resampling

from geosae.data.download.fetchers.reproject import geobox_from_profile, reproject_to_geobox

logger = logging.getLogger(__name__)

# Type alias for tile identifiers (subclasses define their own hashable keys)
TileKey = Any


class _ExpiringHrefCache:
    """Cache for signed URLs that auto-expires entries after ``ttl_seconds``.

    Planetary Computer SAS tokens typically last ~24h, so a 30-minute TTL
    ensures we re-sign well before expiry.
    """

    def __init__(self, ttl_seconds: float = 1800):
        import time

        self._ttl = ttl_seconds
        self._data: dict[str, tuple[str, float]] = {}  # key → (href, timestamp)
        self._time = time

    def get(self, key: str) -> str | None:
        if key in self._data:
            href, ts = self._data[key]
            if self._time.monotonic() - ts < self._ttl:
                return href
            del self._data[key]
        return None

    def set(self, key: str, href: str) -> None:
        self._data[key] = (href, self._time.monotonic())


# ---------------------------------------------------------------------------
# ESA WorldCover 10m
# ---------------------------------------------------------------------------

ESA_WC_S3 = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"


class TiledRasterFetcher(ABC):
    """Base class for tile-grouped raster fetching with cross-tile stitching.

    Concrete subclasses must implement:
    - :meth:`tile_key` — map ``(lon, lat)`` to a tile identifier
    - :meth:`candidate_tiles` — return tiles that may contribute to a patch
    - :meth:`fetch_tile` — download/open a tile and return ``(data, geobox)``

    Optionally override:
    - :meth:`cleanup_tile` — free cached tile data after extraction
    """

    @abstractmethod
    def tile_key(self, lon: float, lat: float) -> TileKey:
        """Map a geographic coordinate to the tile that contains it."""

    @abstractmethod
    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[TileKey]:
        """Return tile keys that may contribute to a patch centered at (lon, lat).

        The primary tile should be first.  Adjacent tiles are included when the
        patch may cross a tile boundary.

        Args:
            lon: Center longitude.
            lat: Center latitude.
            patch_extent_m: Approximate patch extent in meters (default 640 = 64×10m).
        """

    @abstractmethod
    def fetch_tile(self, tile_key: TileKey) -> tuple[np.ndarray, GeoBox] | None:
        """Fetch a tile and return ``(data, src_geobox)`` or ``None`` on failure.

        ``data`` can be 2D ``(H, W)`` or 3D ``(bands, H, W)``.
        """

    def cleanup_tile(self, tile_key: TileKey) -> None:  # noqa: B027 - optional hook, no-op by default
        """Free cached data for a tile.  Override if tiles are cached to disk."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_patch(
        self,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> np.ndarray | None:
        """Fetch, stitch, and reproject a patch to the target GeoBox.

        Tries the primary tile first.  If the patch extends beyond the tile
        boundary, composites from adjacent tiles (following the Tessera
        stitching pattern).

        Args:
            lon: Center longitude.
            lat: Center latitude.
            dst_geobox: Target GeoBox (patch CRS + transform + shape).
            resampling: Resampling method.
            nodata: Fill value for missing data.

        Returns:
            Array matching ``dst_geobox`` shape, or ``None`` on total failure.
        """
        candidates = self.candidate_tiles(lon, lat)
        if not candidates:
            return None

        if len(candidates) == 1:
            # Fast path: single tile, no stitching needed
            tile_result = self.fetch_tile(candidates[0])
            if tile_result is None:
                return None
            data, src_geobox = tile_result
            return reproject_to_geobox(data, src_geobox, dst_geobox, resampling, nodata)

        # Multi-tile: stitch in source CRS, then reproject
        # Use the primary tile's CRS as the stitching reference
        primary_result = self.fetch_tile(candidates[0])
        if primary_result is None:
            # Try remaining candidates
            for cand in candidates[1:]:
                primary_result = self.fetch_tile(cand)
                if primary_result is not None:
                    break
            if primary_result is None:
                return None

        primary_data, primary_geobox = primary_result

        # Determine the bounding box of the destination in the source CRS
        # by transforming dst_geobox corners
        stitch_geobox = self._compute_stitch_geobox(dst_geobox, primary_geobox)
        stitch_h, stitch_w = stitch_geobox.height, stitch_geobox.width
        ndim = primary_data.ndim

        if ndim == 2:
            canvas = np.full(
                (stitch_h, stitch_w),
                np.nan if nodata is None else nodata,
                dtype=np.float64,
            )
        else:
            n_bands = primary_data.shape[0]
            canvas = np.full(
                (n_bands, stitch_h, stitch_w),
                np.nan if nodata is None else nodata,
                dtype=np.float64,
            )

        # Fill from each candidate tile
        for tile_key in candidates:
            if tile_key == candidates[0]:
                tile_data, tile_geobox = primary_data, primary_geobox
            else:
                result = self.fetch_tile(tile_key)
                if result is None:
                    continue
                tile_data, tile_geobox = result

            self._fill_canvas(canvas, stitch_geobox, tile_data, tile_geobox, nodata)
            # Don't cleanup primary tile yet
            if tile_key != candidates[0]:
                self.cleanup_tile(tile_key)

        del primary_data
        self.cleanup_tile(candidates[0])

        # Convert canvas back to original dtype and reproject
        canvas_arr = canvas.astype(primary_result[0].dtype) if ndim == 2 else canvas.astype(primary_result[0].dtype)

        return reproject_to_geobox(canvas_arr, stitch_geobox, dst_geobox, resampling, nodata)

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Fetch one tile and extract patches for all cells in it.

        Cells near the tile edge automatically composite from adjacent tiles.

        Args:
            tile_key: Tile identifier.
            cells: List of ``(cell_idx, lon, lat, dst_geobox)`` tuples.
            resampling: Resampling method.
            nodata: Fill value.

        Returns:
            List of ``(cell_idx, patch_array)`` for successful extractions.
        """
        tile_result = self.fetch_tile(tile_key)
        if tile_result is None:
            return []

        tile_data, tile_geobox = tile_result
        results = []

        for cell_idx, lon, lat, dst_geobox in cells:
            candidates = self.candidate_tiles(lon, lat)
            if len(candidates) <= 1:
                # Fully inside this tile — direct reproject
                patch = reproject_to_geobox(tile_data, tile_geobox, dst_geobox, resampling, nodata)
                results.append((cell_idx, patch))
            else:
                # Near edge — use full fetch_patch (loads adjacent tiles as needed)
                # But reuse the already-loaded primary tile via _fill_canvas
                patch = self.fetch_patch(lon, lat, dst_geobox, resampling, nodata)
                if patch is not None:
                    results.append((cell_idx, patch))

        self.cleanup_tile(tile_key)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stitch_geobox(self, dst_geobox: GeoBox, src_ref_geobox: GeoBox) -> GeoBox:
        """Compute a GeoBox in the source CRS that covers the destination extent.

        Adds a small buffer to ensure full coverage after reprojection.
        """
        import pyproj

        dst_crs = str(dst_geobox.crs)
        src_crs = str(src_ref_geobox.crs)

        # Get destination corners in destination CRS
        dt = dst_geobox.transform
        corners_dst = [
            (dt.c, dt.f),  # top-left
            (dt.c + dt.a * dst_geobox.width, dt.f),  # top-right
            (dt.c, dt.f + dt.e * dst_geobox.height),  # bottom-left
            (
                dt.c + dt.a * dst_geobox.width,
                dt.f + dt.e * dst_geobox.height,
            ),  # bottom-right
        ]

        # Transform to source CRS
        transformer = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        corners_src = [transformer.transform(x, y) for x, y in corners_dst]

        xs = [c[0] for c in corners_src]
        ys = [c[1] for c in corners_src]

        # Add 10% buffer
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        buffer_x = x_range * 0.1
        buffer_y = y_range * 0.1

        x_min = min(xs) - buffer_x
        y_max = max(ys) + buffer_y

        # Use source resolution
        res_x = abs(src_ref_geobox.transform.a)
        res_y = abs(src_ref_geobox.transform.e)

        width = int(np.ceil((max(xs) + buffer_x - x_min) / res_x))
        height = int(np.ceil((y_max - (min(ys) - buffer_y)) / res_y))

        # Limit to reasonable size (avoid huge canvases for coarse sources)
        width = min(width, 2000)
        height = min(height, 2000)

        affine = Affine(res_x, 0, x_min, 0, -res_y, y_max)
        return GeoBox((height, width), affine, src_crs)

    def _fill_canvas(
        self,
        canvas: np.ndarray,
        canvas_geobox: GeoBox,
        tile_data: np.ndarray,
        tile_geobox: GeoBox,
        nodata: float | int | None,
    ) -> None:
        """Fill canvas pixels from a tile where canvas has nodata/NaN.

        Reprojects tile data into the canvas grid, only filling unfilled pixels.
        """
        fill_val = nodata if nodata is not None else np.nan
        reprojected = reproject_to_geobox(tile_data, tile_geobox, canvas_geobox, Resampling.nearest, fill_val)

        if canvas.ndim == 2:
            # Fill where canvas is nodata/NaN
            if nodata is not None:
                mask = canvas == nodata
            else:
                mask = np.isnan(canvas)
            valid_src = reprojected != fill_val if nodata is not None else ~np.isnan(reprojected)
            fill_mask = mask & valid_src
            canvas[fill_mask] = reprojected[fill_mask]
        else:
            # 3D: check first band for nodata
            if nodata is not None:
                mask = canvas[0] == nodata
            else:
                mask = np.isnan(canvas[0])
            valid_src = reprojected[0] != fill_val if nodata is not None else ~np.isnan(reprojected[0])
            fill_mask = mask & valid_src
            for b in range(canvas.shape[0]):
                canvas[b][fill_mask] = reprojected[b][fill_mask]


# ---------------------------------------------------------------------------
# Windowed COG read helpers
# ---------------------------------------------------------------------------


def _bbox_for_patch(lon: float, lat: float, buffer_deg: float = 0.01) -> tuple[float, float, float, float]:
    """WGS84 bbox around a patch center with buffer (for windowed COG reads)."""
    from math import cos, radians

    deg_lat = buffer_deg
    deg_lon = buffer_deg / max(cos(radians(lat)), 0.01)
    return (lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat)


def _windowed_cog_read(
    src,
    lon: float,
    lat: float,
    buffer_deg: float = 0.01,
    bands: int | list[int] = 1,
) -> tuple[np.ndarray, GeoBox] | None:
    """Read a small window from an open rasterio dataset around (lon, lat).

    Returns ``(data, window_geobox)`` where data is 2D ``(H, W)`` or 3D ``(bands, H, W)``.
    Returns ``None`` if the window falls outside the dataset.
    """
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    bbox = _bbox_for_patch(lon, lat, buffer_deg)

    # Transform bbox to source CRS if needed
    src_crs = str(src.crs)
    if src_crs != "EPSG:4326":
        src_bbox = transform_bounds("EPSG:4326", src_crs, *bbox)
    else:
        src_bbox = bbox

    window = from_bounds(*src_bbox, transform=src.transform)

    # Clamp window to dataset bounds
    window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    if window.width < 1 or window.height < 1:
        return None

    if isinstance(bands, int):
        data = src.read(bands, window=window)
    else:
        data = src.read(bands, window=window)

    # Build GeoBox from actual read dimensions (avoids off-by-one from window rounding)
    win_transform = rasterio.windows.transform(window, src.transform)
    if data.ndim == 2:
        actual_h, actual_w = data.shape
    else:
        actual_h, actual_w = data.shape[1], data.shape[2]
    win_geobox = GeoBox((actual_h, actual_w), win_transform, src_crs)
    return data, win_geobox


def _union_cell_bboxes_wgs84(
    cells: list[tuple[int, float, float, GeoBox]], buffer_deg: float
) -> tuple[float, float, float, float] | None:
    """Union of every cell's ``dst_geobox`` corners in WGS84, padded by ``buffer_deg``.

    Returns ``None`` if corner reprojection fails for any reason (caller
    falls back to center-point buffering).
    """
    import pyproj

    lons: list[float] = []
    lats: list[float] = []
    transformers: dict[str, pyproj.Transformer] = {}
    try:
        for _, _, _, dst_geobox in cells:
            gb_crs = str(dst_geobox.crs)
            t = dst_geobox.transform
            h, w = dst_geobox.shape
            corners = [
                (t.c, t.f),
                (t.c + w * t.a, t.f),
                (t.c, t.f + h * t.e),
                (t.c + w * t.a, t.f + h * t.e),
            ]
            if gb_crs == "EPSG:4326":
                for x, y in corners:
                    lons.append(x)
                    lats.append(y)
            else:
                tr = transformers.get(gb_crs)
                if tr is None:
                    tr = pyproj.Transformer.from_crs(gb_crs, "EPSG:4326", always_xy=True)
                    transformers[gb_crs] = tr
                for x, y in corners:
                    lon, lat = tr.transform(x, y)
                    lons.append(lon)
                    lats.append(lat)
    except Exception:
        return None

    if not lons:
        return None
    return (min(lons) - buffer_deg, min(lats) - buffer_deg, max(lons) + buffer_deg, max(lats) + buffer_deg)


def _window_from_bounds_any_orientation(src_bbox, transform):
    """Pixel window covering ``src_bbox`` for *any* affine orientation.

    ``src_bbox`` is ``(minx, miny, maxx, maxy)`` in the raster CRS.
    :func:`rasterio.windows.from_bounds` assumes a north-up transform (negative
    ``e``) and raises ``WindowError('Bounds and transform are inconsistent')``
    on y-flipped rasters (positive ``e``, e.g. the AlphaEarth AEF COGs). Mapping
    the four bbox corners through the inverse transform and taking the min/max
    row and column is orientation-agnostic and reduces to the same window for
    north-up rasters.
    """
    from rasterio.windows import Window

    minx, miny, maxx, maxy = src_bbox
    inv = ~transform
    cols, rows = [], []
    for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
        c, r = inv * (x, y)
        cols.append(c)
        rows.append(r)
    col_off = int(np.floor(min(cols)))
    row_off = int(np.floor(min(rows)))
    width = int(np.ceil(max(cols))) - col_off
    height = int(np.ceil(max(rows))) - row_off
    return Window(col_off, row_off, max(width, 0), max(height, 0))


#: Smallest side, in source pixels, of the regions cells are grouped into for one
#: read each; a tiled raster groups by its internal blocks when they are larger.
_MIN_READ_REGION_PX = 1024


def _group_cells_by_block(src, cells: list[tuple[int, float, float, GeoBox]], buffer_deg: float) -> list[list]:
    """Cells grouped by the source region (internal block, at least ``_MIN_READ_REGION_PX``) their centre is in."""
    bh, bw = src.block_shapes[0] if src.block_shapes else (1, 1)
    rh, rw = max(int(bh), _MIN_READ_REGION_PX), max(int(bw), _MIN_READ_REGION_PX)
    groups: dict[tuple[int, int], list] = {}
    for cell in cells:
        w = _cells_window(src, [cell], buffer_deg)
        if w is None:
            return [list(cells)]
        key = (int((w.row_off + w.height / 2) // rh), int((w.col_off + w.width / 2) // rw))
        groups.setdefault(key, []).append(cell)
    return list(groups.values())


def _cells_window(src, cells: list[tuple[int, float, float, GeoBox]], buffer_deg: float):
    """Source-pixel window (unclipped) covering every cell's ``dst_geobox``, or ``None`` if the CRS fails."""
    from math import cos, radians

    from rasterio.warp import transform_bounds

    # Union of every dst_geobox extent, projected to WGS84. Falls back to
    # center-point buffering only if geobox corner reprojection fails.
    src_crs = str(src.crs)
    bbox_wgs84 = _union_cell_bboxes_wgs84(cells, buffer_deg)
    if bbox_wgs84 is None:
        cell_lons = [c[1] for c in cells]
        cell_lats = [c[2] for c in cells]
        lat_center = (min(cell_lats) + max(cell_lats)) / 2
        buf_lon = buffer_deg / max(cos(radians(lat_center)), 0.01)
        bbox_wgs84 = (
            min(cell_lons) - buf_lon,
            min(cell_lats) - buffer_deg,
            max(cell_lons) + buf_lon,
            max(cell_lats) + buffer_deg,
        )
    if src_crs != "EPSG:4326":
        try:
            src_bbox = transform_bounds("EPSG:4326", src_crs, *bbox_wgs84)
        except Exception:
            return None
    else:
        src_bbox = bbox_wgs84
    # An orientation-agnostic window (bbox corners through the inverse transform)
    # instead of rasterio.windows.from_bounds, which assumes a north-up transform
    # and raises WindowError on y-flipped (positive-e) rasters such as the
    # AlphaEarth AEF COGs.
    return _window_from_bounds_any_orientation(src_bbox, src.transform)


def _block_read_and_extract(
    src,
    cells: list[tuple[int, float, float, GeoBox]],
    resampling: rasterio.warp.Resampling,
    nodata: float | int | None = None,
    buffer_deg: float = 0.015,
    bands: int | list[int] = 1,
    scale: float | None = None,
    src_nodata: int | float | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Read one bounding window covering all cells, then extract patches from memory.

    This replaces per-cell windowed reads (each with HTTP latency) with a
    single block read followed by in-memory reprojection per cell.

    The read window is sized from the union of every cell's
    ``dst_geobox.extent`` (each extent transformed to the source CRS / WGS84),
    so the block is guaranteed to cover every destination patch regardless
    of how large the destination geobox is. ``buffer_deg`` is applied as a
    small extra margin so reprojection near the block edges has enough
    context.

    Args:
        src: Open rasterio dataset.
        cells: List of ``(cell_idx, lon, lat, dst_geobox)`` tuples.
        resampling: Resampling method for reprojection.
        nodata: Fill value for missing data in the output.
        buffer_deg: Extra degrees added around the union of ``dst_geobox``
            extents (in WGS84), to pad reprojection near block edges.
        bands: Band(s) to read (int for single band, list for multi-band).
        scale: If set, convert to float32 and divide by this value.
        src_nodata: If set (or read from ``src.nodata``), replace with NaN before scaling.

    Returns:
        List of ``(cell_idx, patch_array)`` for successful extractions.
    """
    results = []
    if not cells:
        return results

    window = _cells_window(src, cells, buffer_deg)
    if window is None:
        # Some CRS (e.g. Homolosine) may fail — fall back to per-cell reads
        return _per_cell_fallback(src, cells, resampling, nodata, bands, scale, src_nodata)
    if len(cells) > 1:
        groups = _group_cells_by_block(src, cells, buffer_deg)
        if len(groups) > 1:
            # One read per source region: far-apart cells never pull in the tile between
            # them, and a dense set never holds more than a few blocks of every band at once.
            for group in groups:
                results.extend(
                    _block_read_and_extract(src, group, resampling, nodata, buffer_deg, bands, scale, src_nodata)
                )
            return results
    full = rasterio.windows.Window(0, 0, src.width, src.height)
    if not rasterio.windows.intersect([window, full]):
        return results
    window = window.intersection(full)
    if window.width < 1 or window.height < 1:
        return results

    if isinstance(bands, int):
        block_data = src.read(bands, window=window)
    else:
        block_data = src.read(bands, window=window)

    block_transform = rasterio.windows.transform(window, src.transform)
    src_crs = str(src.crs)
    if block_data.ndim == 2:
        block_h, block_w = block_data.shape
    else:
        block_h, block_w = block_data.shape[1], block_data.shape[2]
    block_geobox = GeoBox((block_h, block_w), block_transform, src_crs)

    # Optional scaling (e.g. GEDTM30 int32 → float32 meters)
    if scale is not None:
        fdata = block_data.astype(np.float32)
        nd = src_nodata if src_nodata is not None else src.nodata
        if nd is not None:
            if block_data.ndim == 2:
                fdata[block_data == int(nd)] = np.nan
            else:
                for b in range(fdata.shape[0]):
                    fdata[b][block_data[b] == int(nd)] = np.nan
        fdata /= scale
        block_data = fdata

    # Extract each cell's patch via in-memory reprojection
    for cell_idx, lon, lat, dst_geobox in cells:
        try:
            patch = reproject_to_geobox(block_data, block_geobox, dst_geobox, resampling, nodata)
            results.append((cell_idx, patch))
        except Exception:
            logger.debug("Failed to extract patch at (%.4f, %.4f)", lon, lat)

    return results


def _per_cell_fallback(
    src,
    cells,
    resampling,
    nodata,
    bands,
    scale,
    src_nodata,
) -> list[tuple[int, np.ndarray]]:
    """Fallback: per-cell windowed reads when block read fails (e.g. exotic CRS)."""
    results = []
    for cell_idx, lon, lat, dst_geobox in cells:
        try:
            read_result = _windowed_cog_read(src, lon, lat, buffer_deg=0.02, bands=bands)
            if read_result is None:
                continue
            data, src_geobox = read_result
            if scale is not None:
                fdata = data.astype(np.float32)
                nd = src_nodata if src_nodata is not None else src.nodata
                if nd is not None:
                    fdata[data == int(nd)] = np.nan
                fdata /= scale
                data = fdata
            patch = reproject_to_geobox(data, src_geobox, dst_geobox, resampling, nodata)
            results.append((cell_idx, patch))
        except Exception:
            logger.debug("Fallback read failed at (%.4f, %.4f)", lon, lat)
    return results


def _open_cog_and_extract(
    url: str,
    cells: list[tuple[int, float, float, GeoBox]],
    resampling: rasterio.warp.Resampling,
    nodata: float | int | None = None,
    bands: int | list[int] = 1,
    scale: float | None = None,
    src_nodata: int | float | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Open a COG URL and extract all cells via block read.

    Convenience wrapper that handles rasterio.Env and error logging.
    """
    import rasterio

    try:
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_USE_HEAD="NO",
            GDAL_HTTP_MAX_RETRY="3",
        ):
            with rasterio.open(f"/vsicurl/{url}") as src:
                return _block_read_and_extract(
                    src,
                    cells,
                    resampling,
                    nodata,
                    bands=bands,
                    scale=scale,
                    src_nodata=src_nodata,
                )
    except Exception:
        logger.warning("Failed to open COG %s", url[:80], exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Pre-downloaded raster fetcher (for coarse-resolution global datasets)
# ---------------------------------------------------------------------------


class PredownloadedRasterFetcher(TiledRasterFetcher):
    """Base for fetchers that download global raster files to a local cache first.

    For coarse-resolution products (≥ 100m: SoilGrids 250m, MODIS 500m,
    ESA CCI 300m), downloading the full file and then reading locally is
    faster than making per-cell HTTP requests.

    Subclasses implement :meth:`_download_urls` to return the files to download.
    """

    def __init__(self, cache_dir: str = "/tmp/raster_cache"):
        from pathlib import Path

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_paths: dict[str, str] = {}  # url → local path

    def _ensure_downloaded(self, url: str, filename: str | None = None) -> str:
        """Download a URL to local cache if not already present."""
        import urllib.request

        if filename is None:
            filename = url.split("/")[-1].split("?")[0]
        local = self._cache_dir / filename

        if local.exists():
            logger.info("Using cached %s", local)
            return str(local)

        logger.info("Downloading %s ...", url[:100])
        # Download beside the target and rename, so an interrupted download never looks cached.
        partial = local.with_name(f"{local.name}.part{os.getpid()}")
        urllib.request.urlretrieve(url, str(partial))
        partial.replace(local)
        logger.info("Downloaded to %s (%.1f MB)", local, local.stat().st_size / 1e6)
        return str(local)

    #: Cells are read in bins of this many degrees. The file is local, so a bin
    #: costs no request; it bounds the block one read materialises, which for
    #: cells spread over the globe would otherwise be the whole raster.
    TILE_DEG = 5

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        step = self.TILE_DEG
        return (int(np.floor(lon / step)) * step, int(np.floor(lat / step)) * step)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None


class PredownloadedCOGFetcher(PredownloadedRasterFetcher):
    """Downloads a single global COG/VRT to local cache, then reads locally.

    Used for coarse-resolution global products (MODIS 500m, etc.) where
    downloading the full file (~1-20GB) is faster than per-cell HTTP requests.

    Args:
        url: URL of the global raster file.
        cache_dir: Local cache directory.
        filename: Override cached filename (default: derived from URL).
        scale: If set, convert to float32 and divide by this value.
        src_nodata: If set, replace this raw value with NaN before scaling.
    """

    def __init__(
        self,
        url: str,
        cache_dir: str = "/tmp/raster_cache",
        filename: str | None = None,
        scale: float | None = None,
        src_nodata: int | float | None = None,
    ):
        super().__init__(cache_dir)
        self._url = url
        self._filename = filename
        self._scale = scale
        self._src_nodata = src_nodata
        self._local_path: str | None = None

    def _get_local_path(self) -> str:
        if self._local_path is None:
            self._local_path = self._ensure_downloaded(self._url, self._filename)
        return self._local_path

    def fetch_patches_for_tile(
        self,
        tile_key,
        cells,
        resampling=Resampling.nearest,
        nodata=None,
    ) -> list[tuple[int, np.ndarray]]:
        import rasterio

        local_path = self._get_local_path()
        try:
            with rasterio.open(local_path) as src:
                return _block_read_and_extract(
                    src,
                    cells,
                    resampling,
                    nodata,
                    scale=self._scale,
                    src_nodata=self._src_nodata,
                )
        except Exception:
            logger.warning("Failed to read local file %s", local_path, exc_info=True)
            return []


class PredownloadedStackFetcher(PredownloadedRasterFetcher):
    """Several single-band global rasters, downloaded once, stacked into one ``(H, W, C)`` patch.

    Each file is cached locally on first use, then every cell is cut from it by a windowed read
    and reprojected onto its geobox. Raw values become float32 through each file's own scale and
    offset metadata (``value = raw * scale + offset``), with ``src_nodata`` as NaN.

    Args:
        urls: One URL per band, in band order.
        cache_dir: Local cache directory.
        src_nodata: Raw value treated as missing in every file.
    """

    def __init__(self, urls: list[str], cache_dir: str = "/tmp/raster_cache", src_nodata: int | float | None = None):
        super().__init__(cache_dir)
        if not urls:
            raise ValueError("PredownloadedStackFetcher needs at least one URL")
        self._urls = list(urls)
        self._src_nodata = src_nodata
        self._lock = threading.Lock()

    def _paths(self) -> list[str]:
        with self._lock:
            return [self._ensure_downloaded(u) for u in self._urls]

    def fetch_patches_for_tile(
        self,
        tile_key,
        cells,
        resampling=Resampling.bilinear,
        nodata=None,
    ) -> list[tuple[int, np.ndarray]]:
        import rasterio

        bands: dict[int, list[np.ndarray]] = {c[0]: [] for c in cells}
        for path in self._paths():
            with rasterio.open(path) as src:
                scale, offset = float(src.scales[0]), float(src.offsets[0])
                got = dict(
                    _block_read_and_extract(src, cells, resampling, np.nan, scale=1.0, src_nodata=self._src_nodata)
                )
            for cell_idx in bands:
                if cell_idx in got:
                    bands[cell_idx].append(got[cell_idx].astype(np.float32) * scale + offset)
        n = len(self._urls)
        return [(i, np.stack(b, axis=-1)) for i, b in bands.items() if len(b) == n]


class _BinnedCOGTileMixin:
    """Override the inherited 'global' tile_key with a 5° lon/lat bin.

    Single-COG fetchers default to one work unit covering every cell, which
    makes :func:`_block_read_and_extract` materialise the union bbox of all
    cells. That works for coarse global COGs (≥250 m) but blows up for
    10 m global rasters (WorldCereal, GWA): the union window of even a
    handful of dispersed cells spans billions of pixels. Binning into
    5° tiles caps each work unit's read window at ~555 km × 555 km.
    """

    TILE_DEG = 5

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        step = self.TILE_DEG
        return (int(np.floor(lon / step)) * step, int(np.floor(lat / step)) * step)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        return [self.tile_key(lon, lat)]


# ---------------------------------------------------------------------------
# Concrete fetchers
# ---------------------------------------------------------------------------


class ESAWorldCoverFetcher(TiledRasterFetcher):
    """Fetches ESA WorldCover 10m land cover from AWS S3 COGs.

    Uses windowed COG reads — only downloads the small region around each
    patch (~100x100 pixels), NOT the full 3° tile (~32000x32000 pixels).
    Tile grouping is used for URL selection only.

    Tiles are 3° × 3° in WGS84.  Coverage: 60°S to 84°N.
    Data is uint8 with 11 land cover classes.

    Args:
        s3_base_url: Base URL of the tile COGs.
        read: ``"tile"`` reads each 3° tile whole and cuts every cell from memory (cheapest when many
            cells share a tile); ``"window"`` reads only a window around the cells of each work unit
            and composites cells that straddle a tile edge from the neighbouring tile (cheapest for
            a sparse sample, where a whole tile is ~10⁹ pixels for one or two cells).
    """

    TILE_DEG = 3
    LAT_MIN = -60
    LAT_MAX = 84

    def __init__(self, s3_base_url: str = ESA_WC_S3, read: str = "tile"):
        if read not in ("tile", "window"):
            raise ValueError(f"read must be 'tile' or 'window', got {read!r}")
        self._s3_base = s3_base_url
        self._read = read

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        return (
            int(np.floor(lon / self.TILE_DEG) * self.TILE_DEG),
            int(np.floor(lat / self.TILE_DEG) * self.TILE_DEG),
        )

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        primary = self.tile_key(lon, lat)
        if lat > self.LAT_MAX or lat < self.LAT_MIN:
            return []

        candidates = [primary]
        threshold = 0.02  # ~2km
        dx = lon - primary[0]
        dy = lat - primary[1]

        if dx > self.TILE_DEG - threshold:
            candidates.append((primary[0] + self.TILE_DEG, primary[1]))
        elif dx < threshold:
            candidates.append((primary[0] - self.TILE_DEG, primary[1]))
        if dy > self.TILE_DEG - threshold:
            candidates.append((primary[0], primary[1] + self.TILE_DEG))
        elif dy < threshold:
            candidates.append((primary[0], primary[1] - self.TILE_DEG))

        if len(candidates) == 3:
            candidates.append(
                (
                    primary[0] + (self.TILE_DEG if dx > self.TILE_DEG / 2 else -self.TILE_DEG),
                    primary[1] + (self.TILE_DEG if dy > self.TILE_DEG / 2 else -self.TILE_DEG),
                )
            )

        candidates = [(lo, la) for lo, la in candidates if self.LAT_MIN <= la < self.LAT_MAX and -180 <= lo < 180]
        return candidates

    def _tile_url(self, tile_key: tuple[int, int]) -> str:
        lon0, lat0 = tile_key
        lat_str = f"N{lat0:02d}" if lat0 >= 0 else f"S{abs(lat0):02d}"
        lon_str = f"E{lon0:03d}" if lon0 >= 0 else f"W{abs(lon0):03d}"
        filename = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
        return f"{self._s3_base}{filename}"

    def fetch_tile(self, tile_key: tuple[int, int]) -> tuple[np.ndarray, GeoBox] | None:
        # Not used directly — see fetch_patches_for_tile for windowed reads.
        # Kept as fallback for the base class stitching path.
        import rasterio

        url = self._tile_url(tile_key)
        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_USE_HEAD="NO",
            ):
                with rasterio.open(f"/vsicurl/{url}") as src:
                    data = src.read(1)
                    return data, geobox_from_profile(src.profile)
        except Exception:
            logger.warning("Failed to fetch ESA WC tile %s", tile_key, exc_info=True)
            return None

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Read the full tile once, then extract all patches from memory.

        For tiles with many cells (~400 in the full 1M run), reading the
        entire 3° tile (~1GB uint8) and extracting patches is much faster
        than reading individual windows (each with HTTP latency). With
        ``read="window"`` only the cells' window is read, composited across
        tile edges.
        """
        import rasterio

        if self._read == "window":
            fill = 0 if nodata is None else nodata

            def read(key, key_cells):
                if not (self.LAT_MIN <= key[1] < self.LAT_MAX):
                    return []
                return _open_cog_and_extract(self._tile_url(key), key_cells, resampling, fill)

            return _read_composited(read, tile_key, cells, self.candidate_tiles, fill)

        url = self._tile_url(tile_key)
        results = []
        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_USE_HEAD="NO",
            ):
                with rasterio.open(f"/vsicurl/{url}") as src:
                    tile_data = src.read(1)
                    tile_geobox = geobox_from_profile(src.profile)

            # Extract patches from the in-memory tile
            for cell_idx, lon, lat, dst_geobox in cells:
                try:
                    patch = reproject_to_geobox(tile_data, tile_geobox, dst_geobox, resampling, nodata)
                    results.append((cell_idx, patch))
                except Exception:
                    logger.debug("Failed to extract patch at (%.4f, %.4f)", lon, lat)

            del tile_data  # Free ~1GB
        except Exception:
            logger.warning("Failed to open ESA WC tile %s", tile_key, exc_info=True)
        return results


def _start_year(item) -> int | None:
    """Calendar year a STAC item's period starts in (``start_datetime``, else ``datetime``)."""
    stamp = item.properties.get("start_datetime") or item.properties.get("datetime")
    return int(str(stamp)[:4]) if stamp else None


class ESRILulcFetcher(TiledRasterFetcher):
    """Fetches ESRI 10m LULC from Planetary Computer STAC COGs.

    Uses STAC search to find the correct tile, then windowed COG reads.
    9 land cover classes, year 2021.
    """

    def __init__(self, year: int = 2021):
        self.year = year
        self._catalog = None
        self._item_cache = _ExpiringHrefCache()

    def _get_catalog(self):
        if self._catalog is None:
            from pystac_client import Client

            self._catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        return self._catalog

    def tile_key(self, lon: float, lat: float) -> tuple[float, float]:
        # Group by 1° grid (STAC search is per-point anyway)
        return (round(lon, 1), round(lat, 1))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list:
        return [(lon, lat)]  # Each cell searches STAC independently

    def _get_signed_href(self, lon: float, lat: float) -> str | None:
        """Search STAC for the tile covering (lon, lat) and return a signed COG URL."""
        import planetary_computer

        key = f"{round(lon, 2)}_{round(lat, 2)}"
        cached = self._item_cache.get(key)
        if cached is not None:
            return cached

        catalog = self._get_catalog()
        bbox = (max(lon - 0.01, -180), max(lat - 0.01, -90), min(lon + 0.01, 180), min(lat + 0.01, 90))
        try:
            search = catalog.search(
                collections=["io-lulc-annual-v02"],
                bbox=bbox,
                datetime=f"{self.year}-01-01/{self.year}-12-31",
            )
            # The search matches on overlap, and an annual item ends at 1 January of the next
            # year, so the previous year's item matches too: keep only the requested year's.
            items = [i for i in search.items() if _start_year(i) == self.year]
            if not items:
                return None

            # Pick the item with the smallest bbox (most specific tile).
            # Antimeridian tiles have global bbox [-180, ..., 180] and should be avoided.
            def _bbox_area(item):
                b = item.bbox
                w = b[2] - b[0]
                h = b[3] - b[1]
                return w * h

            items.sort(key=_bbox_area)
            signed = planetary_computer.sign(items[0])
            href = signed.assets["data"].href
            self._item_cache.set(key, href)
            return href
        except Exception:
            logger.warning(
                "STAC search failed for ESRI LULC at (%.4f, %.4f)",
                lon,
                lat,
                exc_info=True,
            )
            return None

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        # Not used — see fetch_patches_for_tile
        return None

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        import rasterio

        results = []
        # Group cells by STAC item (nearby cells likely share the same item)
        href_groups: dict[str, list] = {}
        for cell_idx, lon, lat, dst_geobox in cells:
            href = self._get_signed_href(lon, lat)
            if href is None:
                continue
            href_groups.setdefault(href, []).append((cell_idx, lon, lat, dst_geobox))

        for href, group_cells in href_groups.items():
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                    with rasterio.open(href) as src:
                        results.extend(_block_read_and_extract(src, group_cells, resampling, nodata))
            except Exception:
                logger.warning("Failed to open ESRI COG %s", href[:80], exc_info=True)

        return results


class OpenLandMapCOGFetcher(TiledRasterFetcher):
    """Fetches data from global COGs via windowed /vsicurl/ reads.

    Used for any single-file global raster accessible via HTTP (OpenLandMap,
    OpenGeoHub, etc.).  Supports optional integer-to-float scaling and nodata
    replacement for datasets like GEDTM30.

    Args:
        cog_url: HTTP(S) URL of the global COG.
        scale: If set, convert to float32 and divide by this value.
        src_nodata: If set, replace this raw value with NaN before scaling.
    """

    #: Bin size in degrees for tile_key. The COG is a single global file, so
    #: tile_key is only used by the orchestrator to group nearby cells into
    #: one block read. Coarser bins = larger block reads (cheaper per-cell
    #: HTTP overhead) but riskier large windows; finer bins = more HTTP calls.
    #: 5 degrees keeps each block read <= ~555 km x 555 km, safely small even
    #: at 10 m source resolution.
    TILE_DEG = 5

    def __init__(
        self,
        cog_url: str,
        scale: float | None = None,
        src_nodata: int | float | None = None,
    ):
        self._url = cog_url
        self._scale = scale
        self._src_nodata = src_nodata

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        step = self.TILE_DEG
        return (int(np.floor(lon / step)) * step, int(np.floor(lat / step)) * step)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None  # Not used — windowed reads only

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        return _open_cog_and_extract(
            self._url,
            cells,
            resampling,
            nodata,
            scale=self._scale,
            src_nodata=self._src_nodata,
        )


class SoilGridsFetcher(TiledRasterFetcher):
    """Fetches SoilGrids 250m data via windowed /vsicurl/ reads from ISRIC COGs.

    Supports both continuous properties (sand, clay, SOC, pH) and WRB soil class.
    Integer-encoded with scale factors (÷10 or ÷100).

    Args:
        mode: ``"properties"`` for the four continuous properties of
            :attr:`PROPERTY_LAYERS`, ``"class"`` for the WRB class of :attr:`CLASS_URL`.
    """

    # Top 0-5cm layer URLs
    PROPERTY_LAYERS = {
        "sand": (
            "https://files.isric.org/soilgrids/latest/data/sand/sand_0-5cm_mean.vrt",
            10.0,
        ),
        "clay": (
            "https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt",
            10.0,
        ),
        "soc": (
            "https://files.isric.org/soilgrids/latest/data/soc/soc_0-5cm_mean.vrt",
            10.0,
        ),
        "phh2o": (
            "https://files.isric.org/soilgrids/latest/data/phh2o/phh2o_0-5cm_mean.vrt",
            10.0,
        ),
    }
    CLASS_URL = "https://files.isric.org/soilgrids/latest/data/wrb/MostProbable.vrt"

    def __init__(self, mode: str = "properties"):
        self.mode = mode

    #: Cells are read in bins of this many degrees, so one windowed read of the
    #: remote VRT covers a bin rather than the union of cells spread worldwide.
    TILE_DEG = 1

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        step = self.TILE_DEG
        return (int(np.floor(lon / step)) * step, int(np.floor(lat / step)) * step)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        import rasterio

        results = []

        if self.mode == "class":
            return self._fetch_class(cells, resampling, nodata)

        # Open all property layers
        layer_names = list(self.PROPERTY_LAYERS.keys())
        layer_urls = [self.PROPERTY_LAYERS[n][0] for n in layer_names]
        layer_scales = [self.PROPERTY_LAYERS[n][1] for n in layer_names]

        try:
            srcs = []
            env = rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_USE_HEAD="NO",
                GDAL_HTTP_MAX_RETRY="3",
            )
            env.__enter__()
            for url in layer_urls:
                srcs.append(rasterio.open(f"/vsicurl/{url}"))

            for cell_idx, lon, lat, dst_geobox in cells:
                try:
                    bands = []
                    ref_geobox = None
                    for i, src in enumerate(srcs):
                        read_result = _windowed_cog_read(src, lon, lat, buffer_deg=0.01)
                        if read_result is None:
                            break
                        data, src_geobox = read_result
                        # Replace nodata with NaN, then scale
                        src_nodata = src.nodata
                        scaled = data.astype(np.float32)
                        if src_nodata is not None:
                            scaled[data == int(src_nodata)] = np.nan
                        scaled = scaled / layer_scales[i]
                        if ref_geobox is None:
                            ref_geobox = src_geobox
                        bands.append(scaled)

                    if len(bands) == len(layer_names) and ref_geobox is not None:
                        stacked = np.stack(bands, axis=0)  # (4, H, W)
                        patch = reproject_to_geobox(stacked, ref_geobox, dst_geobox, Resampling.bilinear, nodata)
                        # (4, 64, 64) → (64, 64, 4)
                        patch = np.moveaxis(patch, 0, -1)
                        results.append((cell_idx, patch))
                except Exception:
                    logger.debug("Failed SoilGrids read at (%.4f, %.4f)", lon, lat, exc_info=True)

            for src in srcs:
                src.close()
            env.__exit__(None, None, None)
        except Exception:
            logger.warning("Failed to open SoilGrids layers", exc_info=True)

        return results

    def _fetch_class(self, cells, resampling, nodata):
        return _open_cog_and_extract(self.CLASS_URL, cells, resampling, nodata)


def _has_missing(patch: np.ndarray, fill: float | int) -> bool:
    """Whether a patch still holds pixels no source tile has covered."""
    return bool(np.isnan(patch).any() if np.isnan(fill) else (patch == fill).any())


def _composite(base: np.ndarray, extra: np.ndarray, fill: float | int) -> np.ndarray:
    """``base`` with its missing pixels taken from ``extra`` where that has them."""
    gap = np.isnan(base) if np.isnan(fill) else base == fill
    have = ~np.isnan(extra) if np.isnan(fill) else extra != fill
    return np.where(gap & have, extra, base)


def _read_composited(read, tile_key, cells, candidate_tiles, fill) -> list[tuple[int, np.ndarray]]:
    """Read ``cells`` from ``tile_key`` and fill what it leaves missing from the neighbouring tiles.

    A cell near a tile edge straddles two source tiles, so the primary tile covers only part of its
    patch. Each neighbour that ``candidate_tiles`` names is read for the cells still missing pixels
    (or missing entirely) and composited in.

    Args:
        read: ``(tile_key, cells) -> [(cell_idx, patch)]`` for one source tile.
        tile_key: The primary tile.
        cells: ``(cell_idx, lon, lat, dst_geobox)`` tuples.
        candidate_tiles: ``(lon, lat) -> [tile_key, ...]``, the primary first.
        fill: The patch's nodata value.
    """
    patches = dict(read(tile_key, cells))
    straddling = [c for c in cells if len(candidate_tiles(c[1], c[2])) > 1]
    neighbours = sorted({t for c in straddling for t in candidate_tiles(c[1], c[2])} - {tile_key}, key=str)
    for neighbour in neighbours:
        missing = [c for c in straddling if c[0] not in patches or _has_missing(patches[c[0]], fill)]
        if not missing:
            break
        for cell_idx, patch in read(neighbour, missing):
            patches[cell_idx] = patch if cell_idx not in patches else _composite(patches[cell_idx], patch, fill)
    return list(patches.items())


class CopernicusDEMFetcher(TiledRasterFetcher):
    """Fetches Copernicus GLO-30 DEM from AWS public COGs.

    Tiles are 1° × 1°.  Resolution 30m.  Stored as int16.
    """

    # AWS public bucket for Copernicus DEM
    BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com/"

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        return (int(np.floor(lon)), int(np.floor(lat)))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        primary = self.tile_key(lon, lat)
        candidates = [primary]
        threshold = 0.01
        dx = lon - primary[0]
        dy = lat - primary[1]

        if dx > 1.0 - threshold:
            candidates.append((primary[0] + 1, primary[1]))
        elif dx < threshold:
            candidates.append((primary[0] - 1, primary[1]))
        if dy > 1.0 - threshold:
            candidates.append((primary[0], primary[1] + 1))
        elif dy < threshold:
            candidates.append((primary[0], primary[1] - 1))
        return candidates

    def _tile_url(self, tile_key: tuple[int, int]) -> str:
        lon0, lat0 = tile_key
        lat_str = f"N{abs(lat0):02d}" if lat0 >= 0 else f"S{abs(lat0):02d}"
        lon_str = f"E{abs(lon0):03d}" if lon0 >= 0 else f"W{abs(lon0):03d}"
        # Copernicus DEM naming: tile covers [lat0, lat0+1) → name uses lat0+1 for N
        name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
        return f"{self.BASE_URL}{name}/{name}.tif"

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.bilinear,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Block-read all cells from one 1° DEM tile, compositing across tile edges.

        The DEM is distributed as 1° tiles and a 640 m cell near a whole degree
        of longitude or latitude straddles two of them, so the primary tile
        covers only part of that cell's patch. Those cells are re-read from each
        neighbouring tile and the results composited, which is the difference
        between a patch with a seam down it and a patch without one.
        """
        fill = np.nan if nodata is None else nodata
        patches = dict(_open_cog_and_extract(self._tile_url(tile_key), cells, resampling, fill))
        straddling = [c for c in cells if len(self.candidate_tiles(c[1], c[2])) > 1]
        for neighbour in {t for c in straddling for t in self.candidate_tiles(c[1], c[2])} - {tile_key}:
            missing = [c for c in straddling if c[0] in patches and _has_missing(patches[c[0]], fill)]
            if not missing:
                break
            for cell_idx, patch in _open_cog_and_extract(self._tile_url(neighbour), missing, resampling, fill):
                patches[cell_idx] = _composite(patches[cell_idx], patch, fill)
        return list(patches.items())


class CCILandCoverFetcher(TiledRasterFetcher):
    """Fetches ESA CCI Land Cover 300m from Planetary Computer COG tiles.

    Uses STAC to find the correct 45° tile, then windowed COG reads.
    Data is LCCS class map (uint8), year 2020, v2.1.1.
    """

    COLLECTION = "esa-cci-lc"

    def __init__(self, year: int = 2020):
        self.year = year
        self._catalog = None
        self._href_cache = _ExpiringHrefCache()

    def _get_catalog(self):
        if self._catalog is None:
            from pystac_client import Client

            self._catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        return self._catalog

    def _get_signed_href(self, lon: float, lat: float) -> str | None:
        import planetary_computer

        # Cache key by 45° tile
        tile_lon = int(np.floor(lon / 45) * 45)
        tile_lat = int(np.floor(lat / 45) * 45)
        cache_key = f"{tile_lon}_{tile_lat}"
        cached = self._href_cache.get(cache_key)
        if cached is not None:
            return cached

        catalog = self._get_catalog()
        bbox = (max(lon - 0.01, -180), max(lat - 0.01, -90), min(lon + 0.01, 180), min(lat + 0.01, 90))
        try:
            search = catalog.search(
                collections=[self.COLLECTION],
                bbox=bbox,
                datetime=f"{self.year}-01-01/{self.year}-12-31",
            )
            items = list(search.items())
            if not items:
                return None
            signed = planetary_computer.sign(items[0])
            href = signed.assets["lccs_class"].href
            self._href_cache.set(cache_key, href)
            return href
        except Exception:
            logger.warning("STAC search failed for CCI LC at (%.4f, %.4f)", lon, lat, exc_info=True)
            return None

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        return (int(np.floor(lon / 45) * 45), int(np.floor(lat / 45) * 45))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None

    def fetch_patches_for_tile(
        self,
        tile_key,
        cells,
        resampling=Resampling.nearest,
        nodata=None,
    ) -> list[tuple[int, np.ndarray]]:
        import rasterio

        results = []
        # Group cells by signed href (cells in same 45° tile share one COG)
        href_groups: dict[str, list] = {}
        for cell_idx, lon, lat, dst_geobox in cells:
            href = self._get_signed_href(lon, lat)
            if href is None:
                continue
            href_groups.setdefault(href, []).append((cell_idx, lon, lat, dst_geobox))

        for href, group_cells in href_groups.items():
            try:
                with rasterio.open(href) as src:
                    results.extend(_block_read_and_extract(src, group_cells, resampling, nodata))
            except Exception:
                logger.warning("Failed to open CCI LC COG", exc_info=True)
        return results


# ---------------------------------------------------------------------------
# CHELSA V2.1 Bioclimatic variables (19 bands, ~1km, 1981-2010)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Meta CHMv2 (DINOv3) 1m canopy height
# ---------------------------------------------------------------------------


class CHMv2Fetcher(TiledRasterFetcher):
    """Fetches Meta ``dinov3_global_chm_v2_ml3`` canopy-height tiles from S3.

    Meta's CHMv2 is stored as **Web-Mercator quadkey tiles at zoom z=10** in
    ``s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/``.
    At z=10 every tile covers exactly 360/1024 ≈ 0.3516° of longitude (constant)
    and a latitude slice that narrows with Mercator distortion; a 640 m patch
    therefore almost always fits in one tile. The 10-digit quadkey encodes the
    ``(tile_x, tile_y)`` Web Mercator coordinates — one base-4 digit per zoom
    level, MSB first — and we reconstruct it from ``(lon, lat)`` directly so
    no 58 MB ``tiles.geojson`` index download is needed. Access is anonymous
    via ``/vsis3/`` + ``AWS_NO_SIGN_REQUEST=YES``.

    Data is uint8 at ~1 m, canopy height in meters (sentinel ``255`` → NaN).
    """

    ZOOM = 10
    DEFAULT_BASE_S3 = "s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/"
    # Web Mercator latitude limits (valid tile range).
    _MERCATOR_LAT_ABS = 85.05112878

    def __init__(self, base_url: str | None = None):
        """
        Args:
            base_url: Override the S3 / HTTP(S) base URL for the ``.tif`` tiles.
                Use ``s3://`` for anonymous S3 via ``/vsis3/``; ``https://`` goes
                through ``/vsicurl/``.
        """
        self._base_url = base_url if base_url is not None else self.DEFAULT_BASE_S3

    @staticmethod
    def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
        """Web Mercator slippy-tile ``(x, y)`` at ``zoom`` for ``(lon, lat)``.

        Uses the numerically stable ``asinh(tan(lat))`` form so both hemispheres
        are handled without ``log`` domain errors.
        """
        from math import asinh, floor, pi, tan

        n = 1 << zoom
        x = int(floor((lon + 180.0) / 360.0 * n)) % n
        lat_c = max(min(lat, CHMv2Fetcher._MERCATOR_LAT_ABS), -CHMv2Fetcher._MERCATOR_LAT_ABS)
        lat_rad = lat_c * pi / 180.0
        y_frac = (1.0 - asinh(tan(lat_rad)) / pi) / 2.0
        y = int(floor(y_frac * n))
        y = max(0, min(y, n - 1))
        return x, y

    @staticmethod
    def _tile_to_quadkey(x: int, y: int, zoom: int) -> str:
        """Standard Microsoft quadkey (base-4 interleaved x,y, MSB first)."""
        digits = []
        for i in range(zoom, 0, -1):
            bit = 1 << (i - 1)
            d = 0
            if x & bit:
                d |= 1
            if y & bit:
                d |= 2
            digits.append(str(d))
        return "".join(digits)

    def tile_key(self, lon: float, lat: float) -> str:
        x, y = self._lonlat_to_tile(lon, lat, self.ZOOM)
        return self._tile_to_quadkey(x, y, self.ZOOM)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[str]:
        """Primary quadkey + any neighbours within a ~1 km buffer (patch crosses a tile edge)."""
        primary = self.tile_key(lon, lat)
        # Rough degree buffer matching ``patch_extent_m`` at the equator; scales
        # with latitude via ``_lonlat_to_tile`` which already accounts for it.
        buf_deg = 0.015
        seen = {primary}
        candidates = [primary]
        for dlon, dlat in ((buf_deg, 0), (-buf_deg, 0), (0, buf_deg), (0, -buf_deg)):
            q = self.tile_key(lon + dlon, lat + dlat)
            if q not in seen:
                candidates.append(q)
                seen.add(q)
        return candidates

    def _tile_url(self, tile_key: str) -> str:
        return f"{self._base_url}{tile_key}.tif"

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None

    def _read_tile(
        self,
        tile_key: str,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling,
        nodata: float | int | None,
    ) -> list[tuple[int, np.ndarray]]:
        """Block-read ``cells`` from one quadkey tile. ``None`` on open failure."""
        import rasterio

        url = self._tile_url(tile_key)
        if url.startswith("s3://"):
            gdal_path = "/vsis3/" + url[len("s3://") :]
        elif url.startswith(("http://", "https://")):
            gdal_path = f"/vsicurl/{url}"
        else:
            gdal_path = url

        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_USE_HEAD="NO",
                GDAL_HTTP_MAX_RETRY="3",
                AWS_NO_SIGN_REQUEST="YES",
            ):
                with rasterio.open(gdal_path) as src:
                    # CHMv2 is uint8 with 255 as nodata → convert to float32 in meters.
                    return _block_read_and_extract(
                        src,
                        cells,
                        resampling,
                        nodata,
                        scale=1.0,
                        src_nodata=255,
                    )
        except Exception:
            logger.warning("Failed to open CHMv2 tile %s", url[:80], exc_info=True)
            return []

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.bilinear,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Block-read all cells from one quadkey tile, compositing across tile edges.

        CHMv2 is distributed as Web-Mercator z10 quadkey tiles (~0.35 degree of
        longitude each), so a 640 m cell near a tile edge straddles two of them
        and the primary tile covers only part of its patch. Those cells are
        re-read from each neighbouring quadkey and the results composited --
        the same fix :class:`CopernicusDEMFetcher` applies, and for the same
        reason: an uncomposited seam fills with the product's own nodata value,
        which for a canopy height of 0 m is a legitimate reading and
        indistinguishable from real ground.
        """
        fill = np.nan if nodata is None else nodata
        patches = dict(self._read_tile(tile_key, cells, resampling, fill))
        straddling = [c for c in cells if len(self.candidate_tiles(c[1], c[2])) > 1]
        for neighbour in {t for c in straddling for t in self.candidate_tiles(c[1], c[2])} - {tile_key}:
            missing = [c for c in straddling if c[0] in patches and _has_missing(patches[c[0]], fill)]
            if not missing:
                break
            for cell_idx, patch in self._read_tile(neighbour, missing, resampling, fill):
                patches[cell_idx] = _composite(patches[cell_idx], patch, fill)
        return list(patches.items())


# ---------------------------------------------------------------------------
# Local Climate Zones (LCZ) v3 — single global 100 m COG
# ---------------------------------------------------------------------------


class LCZFetcher(OpenLandMapCOGFetcher):
    """Fetches the Local Climate Zones v3 global 100 m COG.

    Source: ``https://lcz-generator.rub.de/cogs/lcz_filter_v3_cog.tif``.
    Single-band uint8 with 17 LCZ classes (1-10 built types, 11-17 natural
    land covers). Uses :class:`OpenLandMapCOGFetcher`'s windowed ``/vsicurl/``
    path unchanged; the class exists mainly to pin the URL.
    """

    DEFAULT_URL = "https://lcz-generator.rub.de/cogs/lcz_filter_v3_cog.tif"

    def __init__(self, cog_url: str | None = None):
        super().__init__(cog_url or self.DEFAULT_URL)


# ---------------------------------------------------------------------------
# GlobeLand30 — 30 m global land cover, 10°×10° tiles
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GLCLUC2020 — GLAD UMD global 30 m land cover and use change (Potapov et al. 2022)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-tile-ZIP fetcher — download one ZIP of N tiles, build a VRT, query it
# ---------------------------------------------------------------------------


class MultiTileZipVRTFetcher(_BinnedCOGTileMixin, PredownloadedCOGFetcher):
    """Downloads a ZIP containing many GeoTIFF tiles, builds a VRT mosaic, queries it.

    Used for global products distributed as a single archive of per-AEZ /
    per-tile GeoTIFFs (WorldCereal, multi-band GLC_FCS30D lon-bands, …).
    The full archive is fetched once to ``cache_dir``; on first read the
    fetcher enumerates inner ``.tif`` members matching ``inner_pattern`` and
    builds a virtual mosaic via :func:`osgeo.gdal.BuildVRT` referencing each
    member through GDAL's ``/vsizip/`` virtual FS. The resulting ``.vrt``
    behaves like a single global COG -- per-cell reads delegate to the same
    ``_block_read_and_extract`` path as :class:`OpenLandMapCOGFetcher`.

    Args:
        url: HTTP(S) URL of the ZIP archive.
        inner_pattern: ``fnmatch`` glob applied to the ZIP namelist to pick
            the tiles to mosaic (e.g. ``'*classification.tif'``). Default
            picks every ``*.tif`` member.
        vrt_basename: Filename (stem) of the cached VRT inside ``cache_dir``.
            Recomputed on first use; reused thereafter.
        cache_dir: Local cache directory.
        filename: Override cached ZIP filename (default: URL basename).
        scale: See :class:`PredownloadedCOGFetcher`.
        src_nodata: See :class:`PredownloadedCOGFetcher`.
    """

    def __init__(
        self,
        url: str,
        inner_pattern: str = "*.tif",
        vrt_basename: str | None = None,
        cache_dir: str = "/tmp/raster_cache",
        filename: str | None = None,
        scale: float | None = None,
        src_nodata: int | float | None = None,
    ):
        super().__init__(url, cache_dir=cache_dir, filename=filename, scale=scale, src_nodata=src_nodata)
        self._inner_pattern = inner_pattern
        self._vrt_basename = vrt_basename
        self._vrt_path: str | None = None

    def _build_vrt(self, local_zip: str) -> str:
        import zipfile
        from fnmatch import fnmatch
        from pathlib import Path as _Path

        from osgeo import gdal

        zip_path = str(_Path(local_zip).resolve())
        with zipfile.ZipFile(zip_path) as zf:
            inner_tifs = sorted(
                name
                for name in zf.namelist()
                if fnmatch(name, self._inner_pattern) and name.lower().endswith((".tif", ".tiff"))
            )
        if not inner_tifs:
            raise FileNotFoundError(f"No tiles matching {self._inner_pattern!r} inside {zip_path}")

        sources = [f"/vsizip/{zip_path}/{name}" for name in inner_tifs]
        stem = self._vrt_basename or _Path(zip_path).stem
        vrt_path = str(_Path(self._cache_dir) / f"{stem}.vrt")
        logger.info("Building VRT mosaic over %d tiles -> %s", len(sources), vrt_path)
        ds = gdal.BuildVRT(vrt_path, sources, options=gdal.BuildVRTOptions(resolution="highest", resampleAlg="nearest"))
        if ds is None:
            raise RuntimeError(f"gdal.BuildVRT failed for {vrt_path}")
        ds.FlushCache()
        ds = None
        return vrt_path

    def _get_local_path(self) -> str:
        if self._vrt_path is None:
            local_zip = super()._get_local_path()
            from pathlib import Path as _Path

            stem = self._vrt_basename or _Path(local_zip).stem
            cached_vrt = _Path(self._cache_dir) / f"{stem}.vrt"
            if cached_vrt.exists():
                self._vrt_path = str(cached_vrt)
            else:
                self._vrt_path = self._build_vrt(local_zip)
        return self._vrt_path


# ---------------------------------------------------------------------------
# GLC_FCS30D — 30 m fine land cover, per-10°-lon ZIP × per-5°-tile multi-band
# ---------------------------------------------------------------------------


class GLCFCS30DFetcher(TiledRasterFetcher):
    """Fetches GLC_FCS30D yearly land-cover tiles from Zenodo 8239305.

    The product ships as 36 ZIPs covering 10° of longitude each
    (named e.g. ``GLC_FCS30D_19852022maps_E40-E45.zip`` -- the label
    encodes the western 5° tile-corner but the ZIP holds two 5° tile
    columns). Inside each ZIP, the per-5°×5° annual stack is named
    ``GLC_FCS30D_20002022_{LON}{LAT}_Annual.tif`` (LON = west corner,
    LAT = north corner) and carries 23 yearly bands (2000..2022).
    Earlier 5-yearly maps (1985..2000) live in ``GLC_FCS30_19852000_5yearsMap_V1_*``
    siblings -- not used here.

    On first use per (ZIP-zone, year) combination the ZIP is fetched
    once to ``cache_dir`` and a VRT mosaic over its inner annual TIFs is
    built via :func:`osgeo.gdal.BuildVRT`, restricted to the band for
    ``year``. The VRT then behaves like a single regional COG and
    per-cell extraction reuses ``_block_read_and_extract``.

    Args:
        year: Calendar year in 2000..2022 (default 2022). Maps to band
            ``year - 2000 + 1`` of each annual TIF.
        cache_dir: Local cache directory.
    """

    LON_DEG = 10  # ZIP lon-band width
    TILE_DEG = 5  # inner-tile width
    DEFAULT_BASE_URL = "https://zenodo.org/api/records/8239305/files/"
    YEAR_MIN = 2000
    YEAR_MAX = 2022

    def __init__(self, year: int = 2022, cache_dir: str = "/tmp/raster_cache"):
        from pathlib import Path

        if not (self.YEAR_MIN <= year <= self.YEAR_MAX):
            raise ValueError(f"year={year!r} must be in [{self.YEAR_MIN}, {self.YEAR_MAX}]")
        self._year = year
        self._band = year - self.YEAR_MIN + 1  # bands are 1-indexed in rasterio
        self._cache_dir = Path(cache_dir) / "glc_fcs30d"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._missing: set[str] = set()
        self._vrt_paths: dict[str, str] = {}

    @staticmethod
    def _zip_zone(lon: float) -> tuple[str, int]:
        """Map ``lon`` to the zone label and west-edge of the 10°-wide ZIP."""
        zip_left = int(np.floor(lon / 10) * 10)  # multiple of 10, e.g. 40 or -180
        if zip_left >= 0:
            label = f"E{zip_left}-E{zip_left + 5}"
        else:
            label = f"W{abs(zip_left + 5)}-W{abs(zip_left)}"
        return label, zip_left

    @staticmethod
    def _inner_tif_name(lon: float, lat: float) -> str:
        """Inner annual TIF name for the 5° tile containing (lon, lat).

        Tiles are 5°×5°; the file name uses the **west** edge for longitude
        and the **north** edge for latitude. Tiles are south-inclusive
        (cell at lat 65 lives in the [65, 70) tile labelled ``N70``).
        """
        tile_left = int(np.floor(lon / 5) * 5)
        if tile_left >= 0:
            lon_str = f"E{tile_left:03d}"
        else:
            lon_str = f"W{abs(tile_left):03d}"
        tile_top = int(np.floor(lat / 5) * 5) + 5
        if tile_top > 0:
            lat_str = f"N{tile_top}"
        else:
            lat_str = f"S{abs(tile_top)}"
        return f"GLC_FCS30D_20002022_{lon_str}{lat_str}_Annual.tif"

    def tile_key(self, lon: float, lat: float) -> str:
        """One work unit per ZIP zone -- inner tiles are mosaicked into one VRT."""
        label, _ = self._zip_zone(lon)
        return label

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[str]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None

    def _zip_url(self, label: str) -> str:
        return f"{self.DEFAULT_BASE_URL}GLC_FCS30D_19852022maps_{label}.zip/content"

    def _ensure_zip(self, label: str) -> str | None:
        import urllib.error
        import urllib.request

        if label in self._missing:
            return None
        local = self._cache_dir / f"GLC_FCS30D_19852022maps_{label}.zip"
        if local.exists() and local.stat().st_size > 0:
            return str(local)
        url = self._zip_url(label)
        logger.info("Downloading GLC_FCS30D zone %s ...", label)
        try:
            urllib.request.urlretrieve(url, str(local))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self._missing.add(label)
                logger.debug("GLC_FCS30D zone %s missing (404)", label)
            else:
                logger.warning("GLC_FCS30D zone %s HTTP %s", label, e.code)
            return None
        except Exception:
            logger.warning("GLC_FCS30D zone %s download failed", label, exc_info=True)
            return None
        return str(local)

    def _ensure_vrt(self, label: str, local_zip: str) -> str:
        """Build one VRT mosaic (single year band) over the ZIP's annual TIFs."""
        if label in self._vrt_paths:
            return self._vrt_paths[label]

        import zipfile
        from pathlib import Path as _Path

        from osgeo import gdal

        vrt_path = str(_Path(self._cache_dir) / f"GLC_FCS30D_{self._year}_{label}.vrt")
        if _Path(vrt_path).exists():
            self._vrt_paths[label] = vrt_path
            return vrt_path

        zip_resolved = str(_Path(local_zip).resolve())
        with zipfile.ZipFile(zip_resolved) as zf:
            inner = sorted(
                name
                for name in zf.namelist()
                if name.startswith("GLC_FCS30D_20002022_") and name.endswith("_Annual.tif")
            )
        if not inner:
            raise FileNotFoundError(f"No annual TIFs inside {zip_resolved}")

        sources = [f"/vsizip/{zip_resolved}/{name}" for name in inner]
        ds = gdal.BuildVRT(
            vrt_path,
            sources,
            options=gdal.BuildVRTOptions(
                resolution="highest",
                resampleAlg="nearest",
                bandList=[self._band],
            ),
        )
        if ds is None:
            raise RuntimeError(f"gdal.BuildVRT failed for {vrt_path}")
        ds.FlushCache()
        ds = None
        self._vrt_paths[label] = vrt_path
        return vrt_path

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        import rasterio

        local_zip = self._ensure_zip(tile_key)
        if local_zip is None:
            return []
        try:
            vrt_path = self._ensure_vrt(tile_key, local_zip)
        except Exception:
            logger.warning("GLC_FCS30D VRT build failed for %s", tile_key, exc_info=True)
            return []
        try:
            with rasterio.open(vrt_path) as src:
                return _block_read_and_extract(src, cells, resampling, nodata, bands=1)
        except Exception:
            logger.warning("GLC_FCS30D VRT unreadable: %s", vrt_path, exc_info=True)
            return []


# ---------------------------------------------------------------------------
# NASA GLanCE30 — 30 m annual land cover via Earthdata-authenticated downloads
# ---------------------------------------------------------------------------


class NASAGLanCE30Fetcher(TiledRasterFetcher):
    """Fetches NASA GLanCE30 v001 yearly land-cover tiles via Earthdata Login.

    GLanCE30 lives on the LP DAAC protected bucket
    (``s3://lp-prod-protected/GLanCE30.001/...``); HTTPS access requires an
    Earthdata bearer token, supplied here by the :mod:`earthaccess` library
    reading ``~/.netrc``. Each granule is a 150 km × 150 km LAEA tile (one
    of seven per-continent grids) with seven separate per-layer GeoTIFFs;
    we only fetch the ``_LC.tif`` (categorical land-cover, ~20-30 MB).

    For each cell, the fetcher uses
    :func:`earthaccess.search_data(point=(lon, lat))` to discover the
    matching granule for ``year``, downloads its ``_LC.tif`` to ``cache_dir``
    on first use, and runs the standard ``_block_read_and_extract`` path
    so the per-continent LAEA → caller-UTM reprojection is automatic.

    Output: single-band int8. Class codes:
    1=Water, 2=Ice/Snow, 3=Developed, 4=Barren, 5=Trees, 6=Shrublands,
    7=Herbaceous; 0/255 = nodata. Coverage: NA, SA, EU, OC, AF
    (Asia/Antarctica not yet released, mid-2024). Temporal range:
    2001-2019 globally (2021 for AF).

    Args:
        year: Annual snapshot year (default 2019, the last full global year).
        cache_dir: Local cache directory for downloaded ``_LC.tif`` files.
    """

    TILE_DEG = 5  # tile_key bin -- groups multiple cells inside one granule
    YEAR_MIN = 2001
    YEAR_MAX = 2021

    _logged_in = False

    def __init__(self, year: int = 2019, cache_dir: str = "/tmp/raster_cache"):
        from pathlib import Path

        if not (self.YEAR_MIN <= year <= self.YEAR_MAX):
            raise ValueError(f"year={year!r} must be in [{self.YEAR_MIN}, {self.YEAR_MAX}]")
        self._year = year
        self._cache_dir = Path(cache_dir) / "glance30"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._granule_cache: dict[tuple[float, float], str | None] = {}

    @classmethod
    def _ensure_login(cls) -> None:
        if cls._logged_in:
            return
        import earthaccess

        earthaccess.login(strategy="netrc")
        cls._logged_in = True

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        step = self.TILE_DEG
        return (int(np.floor(lon / step)) * step, int(np.floor(lat / step)) * step)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key) -> tuple[np.ndarray, GeoBox] | None:
        return None

    def _granule_lc_path(self, lon: float, lat: float) -> str | None:
        """Discover, download (if missing), and return the local ``_LC.tif`` path."""
        from pathlib import Path

        import earthaccess

        # Quantize coordinates to avoid floating-point cache misses.
        key = (round(lon, 4), round(lat, 4))
        if key in self._granule_cache:
            return self._granule_cache[key]

        self._ensure_login()
        try:
            granules = earthaccess.search_data(
                short_name="GLanCE30",
                version="001",
                point=(lon, lat),
                temporal=(f"{self._year}-07-01", f"{self._year}-07-02"),
                count=1,
            )
        except Exception:
            logger.warning("GLanCE30 CMR search failed at (%.4f, %.4f)", lon, lat, exc_info=True)
            self._granule_cache[key] = None
            return None
        if not granules:
            self._granule_cache[key] = None
            return None

        lc_url = next((u for u in granules[0].data_links() if u.endswith("_LC.tif")), None)
        if lc_url is None:
            self._granule_cache[key] = None
            return None

        local_path = self._cache_dir / Path(lc_url).name
        if not local_path.exists() or local_path.stat().st_size == 0:
            try:
                # earthaccess.download honours netrc + handles redirects + auth.
                earthaccess.download([granules[0]], local_path=str(self._cache_dir))
            except Exception:
                logger.warning("GLanCE30 download failed: %s", lc_url, exc_info=True)
                self._granule_cache[key] = None
                return None
            # earthaccess writes every layer of the granule; only LC.tif is needed.
            for sibling in self._cache_dir.glob(local_path.stem.replace("_LC", "_*") + ".tif"):
                if sibling != local_path:
                    sibling.unlink(missing_ok=True)
        if not local_path.exists():
            self._granule_cache[key] = None
            return None
        self._granule_cache[key] = str(local_path)
        return str(local_path)

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        from collections import defaultdict

        import rasterio

        # Group cells by underlying granule (multiple cells in the same 5° bin
        # very often share one granule on continental LAEA grids).
        per_granule: dict[str, list] = defaultdict(list)
        for cell_idx, lon, lat, dst_geobox in cells:
            local = self._granule_lc_path(lon, lat)
            if local is None:
                continue
            per_granule[local].append((cell_idx, lon, lat, dst_geobox))

        results: list[tuple[int, np.ndarray]] = []
        for local_path, granule_cells in per_granule.items():
            try:
                with rasterio.open(local_path) as src:
                    results.extend(_block_read_and_extract(src, granule_cells, resampling, nodata))
            except Exception:
                logger.warning("GLanCE30 read failed: %s", local_path, exc_info=True)
        return results


# ---------------------------------------------------------------------------
# Vector → raster fetcher (one-time shapefile download + per-cell burn-in)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GHS-BUILT-C 10 m — per-tile Mollweide ZIPs from the JRC FTP
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JRC Global Surface Water (Pekel et al. 2016) — 30 m, 10°×10° GCS tiles
# ---------------------------------------------------------------------------

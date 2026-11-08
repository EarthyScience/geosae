"""Tessera embedding fetcher.

Provides two classes:

- :class:`TesseraFetcher`: Lightweight data fetcher wrapping the ``geotessera``
  library. Returns numpy arrays, no zarr coupling. Use this when you want to
  control the output format yourself.

- :class:`TesseraTiledFetcher`: Adapts :class:`TesseraFetcher` to the
  :class:`~geosae.data.download.fetchers.raster_fetchers.TiledRasterFetcher`
  API, reprojecting each native-CRS tile onto the caller's per-cell UTM geobox.

Tessera tiles are 0.1x0.1 degree squares at 10m resolution with 128 embedding
channels.

Example (fetcher only)::

    fetcher = TesseraFetcher(cache_dir="/tmp/tessera_cache", year=2024)
    result = fetcher.fetch_patch(lon=11.5, lat=48.1, patch_size=64)
    if result is not None:
        patch, crs, transform = result
        # patch.shape == (64, 64, 128)

Example (EOZarr downloader)::

    downloader = TesseraDownloader(store="tessera.zarr", cache_dir="/tmp/tessera_cache")
    with downloader:
        indices = downloader.download_single_cell(spatial_idx=42, time_start_idx=0, time_end_idx=0)
        downloader.register_indices(indices)
"""

import logging
from pathlib import Path

import numpy as np
import pyproj
from odc.geo.geobox import GeoBox
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher, TileKey

logger = logging.getLogger(__name__)


def _wrap_lon(lon: float) -> float:
    """Wrap longitude to [-180, 180). Avoids 180.0 which maps to bogus tile."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    # Modulo can return exactly 180.0 for edge cases; map to -180.0
    if lon >= 180.0:
        lon = -180.0
    return lon


# Quantization parameters matching geotessera's internal dequantization
# Tessera stores int8 internally but returns float32 from fetch_embedding().
# For EOZarr storage we re-quantize to int8 for compact storage.
TESSERA_CHANNELS = 128
TESSERA_NO_DATA = -128


class TesseraFetcher:
    """Fetches Tessera embedding patches via the geotessera library.

    Args:
        cache_dir: Directory for caching downloaded tiles.
        year: Year of embeddings to fetch (default 2024).
        dataset_version: Tessera release to read (``None``: geotessera's
            default). Releases are different models, so a store is built from
            one of them.
    """

    def __init__(self, cache_dir: str | Path, year: int = 2024, dataset_version: str | None = None):
        from geotessera import GeoTessera

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.year = year
        kwargs = {} if dataset_version is None else {"dataset_version": dataset_version}
        self._gt = GeoTessera(embeddings_dir=str(self.cache_dir), **kwargs)

    def _cleanup_tile_cache(self, lon: float, lat: float) -> None:
        """Delete cached files for a tile to save disk space."""
        import shutil

        from geotessera.registry import tile_from_world

        lon = _wrap_lon(lon)
        lat = max(-90.0, min(90.0, lat))
        tile_lon, tile_lat = tile_from_world(lon, lat)
        name = f"grid_{tile_lon:.2f}_{tile_lat:.2f}"

        # Embedding files: global_0.1_degree_representation/YEAR/grid_LON_LAT/
        emb_dir = self.cache_dir / "global_0.1_degree_representation" / str(self.year) / name
        if emb_dir.exists():
            shutil.rmtree(emb_dir)

        # Scale files live in the same dir structure
        # Landmask tiff: global_0.1_degree_tiff_all/grid_LON_LAT.tiff
        lm_file = self.cache_dir / "global_0.1_degree_tiff_all" / f"{name}.tiff"
        if lm_file.exists():
            lm_file.unlink()

    def fetch_tile(self, lon: float, lat: float) -> tuple[np.ndarray, CRS, Affine] | None:
        """Fetch a full Tessera tile containing the given coordinates.

        Coordinates are snapped to the nearest tile center (0.1-degree grid).

        Args:
            lon: Longitude in decimal degrees.
            lat: Latitude in decimal degrees.

        Returns:
            ``(embedding, crs, transform)`` where embedding has shape
            ``(H, W, 128)`` as float32, or ``None`` if the tile is unavailable.
        """
        try:
            embedding, crs, transform = self._gt.fetch_embedding(lon, lat, self.year)
            return embedding.astype(np.float32), CRS(crs), Affine(*transform[:6])
        except ValueError as e:
            if "Failed to read all data" in str(e) or "file seems not fully written" in str(e):
                # Corrupted file — clean cache and retry once with fresh download
                logger.warning("Corrupted tile at (%.4f, %.4f), cleaning cache and retrying...", lon, lat)
                self._cleanup_tile_cache(lon, lat)
                try:
                    embedding, crs, transform = self._gt.fetch_embedding(lon, lat, self.year)
                    return embedding.astype(np.float32), CRS(crs), Affine(*transform[:6])
                except Exception:
                    logger.warning("Retry failed for tile at (%.4f, %.4f), skipping", lon, lat)
                    self._cleanup_tile_cache(lon, lat)
                    return None
            logger.warning("Failed to fetch Tessera tile at (%.4f, %.4f)", lon, lat, exc_info=True)
            return None
        except Exception:
            logger.warning("Failed to fetch Tessera tile at (%.4f, %.4f)", lon, lat, exc_info=True)
            return None

    def fetch_patch(
        self,
        lon: float,
        lat: float,
        patch_size: int = 64,
    ) -> tuple[np.ndarray, CRS, Affine] | None:
        """Fetch a centered patch cutout with cross-tile compositing.

        First tries each candidate tile for a full extraction. If no single
        tile fully contains the patch, composites from the primary tile
        (most overlap) and fills remaining pixels from adjacent tiles,
        following the same pattern as the STAC downloader's gap-filling.

        Args:
            lon: Center longitude in decimal degrees.
            lat: Center latitude in decimal degrees.
            patch_size: Spatial size of the output patch (default 64).

        Returns:
            ``(patch, crs, patch_transform)`` where patch has shape
            ``(patch_size, patch_size, D)``, or ``None`` if no tiles
            are available for this location.
        """
        candidates = _candidate_tiles(lon, lat)
        half = patch_size // 2

        # Try each candidate for a full extraction first
        for tile_lon, tile_lat in candidates:
            tile_result = self.fetch_tile(tile_lon, tile_lat)
            if tile_result is None:
                continue

            embedding, crs, transform = tile_result
            patch = extract_center_patch(embedding, crs, transform, lon, lat, patch_size)
            if patch is not None:
                row, col = _lonlat_to_pixel(lon, lat, crs, transform)
                r0 = row - half
                c0 = col - half
                patch_transform = transform * Affine.translation(c0, r0)
                return patch, crs, patch_transform

        # No single tile contains the full patch — composite from multiple tiles.
        # Use the primary tile (first candidate) as the reference CRS and fill
        # gaps from adjacent tiles.
        primary_result = self.fetch_tile(candidates[0][0], candidates[0][1])
        if primary_result is None:
            return None

        primary_emb, primary_crs, primary_transform = primary_result
        H, W, D = primary_emb.shape

        # Compute patch window in primary tile coordinates
        transformer = pyproj.Transformer.from_crs("EPSG:4326", primary_crs, always_xy=True)
        cx, cy = transformer.transform(lon, lat)
        row, col = rowcol(primary_transform, cx, cy)
        row, col = int(row), int(col)
        r0, c0 = row - half, col - half

        # Initialize patch with NaN (marks unfilled pixels)
        patch = np.full((patch_size, patch_size, D), np.nan, dtype=np.float32)
        patch_transform = primary_transform * Affine.translation(c0, r0)

        # Fill from primary tile (clipped to valid region)
        pr0 = max(0, r0)
        pc0 = max(0, c0)
        pr1 = min(H, r0 + patch_size)
        pc1 = min(W, c0 + patch_size)
        if pr1 > pr0 and pc1 > pc0:
            patch[pr0 - r0 : pr1 - r0, pc0 - c0 : pc1 - c0, :] = primary_emb[pr0:pr1, pc0:pc1, :]

        # Free primary tile before loading adjacent tiles
        del primary_emb

        # Fill remaining NaN pixels from adjacent tiles (one at a time to limit memory)
        if np.any(np.isnan(patch)):
            for tile_lon, tile_lat in candidates[1:]:
                adj_result = self.fetch_tile(tile_lon, tile_lat)
                if adj_result is None:
                    continue

                adj_emb, adj_crs, adj_transform = adj_result
                adj_H, adj_W = adj_emb.shape[:2]

                nan_mask = np.isnan(patch[:, :, 0])
                if not np.any(nan_mask):
                    del adj_emb
                    break

                nan_rows, nan_cols = np.where(nan_mask)
                for pr, pc in zip(nan_rows, nan_cols, strict=True):
                    px = primary_transform.c + (c0 + pc + 0.5) * primary_transform.a
                    py = primary_transform.f + (r0 + pr + 0.5) * primary_transform.e

                    if adj_crs != primary_crs:
                        reproj = pyproj.Transformer.from_crs(primary_crs, adj_crs, always_xy=True)
                        ax, ay = reproj.transform(px, py)
                    else:
                        ax, ay = px, py

                    ar, ac = rowcol(adj_transform, ax, ay)
                    ar, ac = int(ar), int(ac)

                    if 0 <= ar < adj_H and 0 <= ac < adj_W:
                        patch[pr, pc, :] = adj_emb[ar, ac, :]

                # Free adjacent tile immediately after use
                del adj_emb
                self._cleanup_tile_cache(tile_lon, tile_lat)

                if not np.any(np.isnan(patch[:, :, 0])):
                    break

        # If still has NaN, fill with 0 (no data available)
        if np.any(np.isnan(patch)):
            n_nan = np.isnan(patch[:, :, 0]).sum()
            logger.debug(
                "Composite patch at (%.4f, %.4f) has %d/%d unfilled pixels", lon, lat, n_nan, patch_size * patch_size
            )
            patch = np.nan_to_num(patch, nan=0.0)

        return patch, primary_crs, patch_transform

    def fetch_patches_for_tile(
        self,
        tile_lon: float,
        tile_lat: float,
        cells: list[tuple[int, float, float]],
        patch_size: int = 64,
    ) -> list[tuple[int, np.ndarray, CRS, Affine]]:
        """Fetch one tile and extract patches for all cells in it.

        For cells near the tile edge, tries adjacent tiles automatically.
        Uses a single ``pyproj.Transformer`` per tile for efficiency.

        Args:
            tile_lon: Tile center longitude.
            tile_lat: Tile center latitude.
            cells: List of ``(cell_idx, lon, lat)`` tuples.
            patch_size: Patch size in pixels.

        Returns:
            List of ``(cell_idx, patch, crs, patch_transform)`` for successful
            extractions.
        """
        tile_result = self.fetch_tile(tile_lon, tile_lat)
        if tile_result is None:
            return []

        embedding, crs, transform = tile_result
        H, W = embedding.shape[:2]
        half = patch_size // 2

        # Single transformer for this tile's CRS
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        # First pass: extract patches that fit fully (use .copy() so we can free tile)
        results = []
        edge_cells = []
        for cell_idx, lon, lat in cells:
            x, y = transformer.transform(lon, lat)
            row, col = rowcol(transform, x, y)
            row, col = int(row), int(col)

            r0, c0 = row - half, col - half
            r1, c1 = r0 + patch_size, c0 + patch_size

            if r0 >= 0 and c0 >= 0 and r1 <= H and c1 <= W:
                patch = embedding[r0:r1, c0:c1, :].copy()
                patch_transform = transform * Affine.translation(c0, r0)
                results.append((cell_idx, patch, crs, patch_transform))
            else:
                edge_cells.append((cell_idx, lon, lat))

        # Free primary tile before processing edge cells (which load more tiles)
        del embedding

        # Second pass: edge cells need compositing from adjacent tiles
        for cell_idx, lon, lat in edge_cells:
            result = self.fetch_patch(lon, lat, patch_size)
            if result is not None:
                patch, alt_crs, patch_transform = result
                results.append((cell_idx, patch, alt_crs, patch_transform))

        # Clean up all cached tile files
        tiles_to_clean = {(tile_lon, tile_lat)}
        for _cell_idx, lon, lat in cells:
            lon_c = _wrap_lon(lon)
            lat_c = max(-90.0, min(90.0, lat))
            for t in _candidate_tiles(lon_c, lat_c):
                tiles_to_clean.add(t)
        for tlon, tlat in tiles_to_clean:
            self._cleanup_tile_cache(tlon, tlat)

        return results


def _lonlat_to_pixel(lon: float, lat: float, crs: CRS, transform: Affine) -> tuple[int, int]:
    """Project (lon, lat) to pixel coordinates in the tile's CRS.

    Returns:
        (row, col) integer pixel coordinates.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = rowcol(transform, x, y)
    return int(row), int(col)


def extract_center_patch(
    embedding: np.ndarray,
    crs: CRS,
    transform: Affine,
    center_lon: float,
    center_lat: float,
    patch_size: int = 64,
) -> np.ndarray | None:
    """Extract a centered patch from a tile embedding.

    Args:
        embedding: Tile array of shape ``(H, W, D)``.
        crs: Tile CRS.
        transform: Tile affine transform.
        center_lon: Center longitude in decimal degrees.
        center_lat: Center latitude in decimal degrees.
        patch_size: Size of the square cutout.

    Returns:
        Array of shape ``(patch_size, patch_size, D)`` or ``None`` if
        the center is too close to the tile edge.
    """
    H, W = embedding.shape[:2]
    half = patch_size // 2

    row, col = _lonlat_to_pixel(center_lon, center_lat, crs, transform)

    r0 = row - half
    c0 = col - half
    r1 = r0 + patch_size
    c1 = c0 + patch_size

    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        logger.debug(
            "Center (%.4f, %.4f) too close to tile edge: pixel (%d, %d), tile size (%d, %d), patch_size %d",
            center_lon,
            center_lat,
            row,
            col,
            H,
            W,
            patch_size,
        )
        return None

    return embedding[r0:r1, c0:c1, :]


def _candidate_tiles(lon: float, lat: float) -> list[tuple[float, float]]:
    """Return up to 4 candidate tile centers that could contain a patch at (lon, lat).

    The primary tile is always first. Adjacent tiles are included when the
    point is near a tile boundary (within 640m = 64 pixels at 10m).
    """
    from geotessera.registry import tile_from_world

    def _safe_tile(lo, la):
        """tile_from_world with bounds handling.

        Wraps longitude across the antimeridian so that adjacent tiles on
        the other side (e.g. 179.95 → -179.95) are found correctly. The
        compositing code handles CRS reprojection between UTM zones.
        Latitudes clamp at ±90°.
        Returns None if the coordinates are out of Tessera's coverage.
        """
        lo = _wrap_lon(lo)
        la = max(-90.0, min(90.0, la))
        try:
            return tile_from_world(lo, la)
        except ValueError:
            return None

    primary = _safe_tile(lon, lat)
    if primary is None:
        return []
    candidates = [primary]

    # Offset within the tile (0 = center, ±0.05 = edge)
    dx = lon - primary[0]  # [-0.05, 0.05]
    dy = lat - primary[1]

    # 640m ≈ 0.006 degrees at equator. Be generous: check if within 0.02 of edge.
    threshold = 0.02
    if dx > 0.05 - threshold:
        t = _safe_tile(lon + 0.1, lat)
        if t is not None:
            candidates.append(t)
    elif dx < -0.05 + threshold:
        t = _safe_tile(lon - 0.1, lat)
        if t is not None:
            candidates.append(t)
    if dy > 0.05 - threshold:
        t = _safe_tile(lon, lat + 0.1)
        if t is not None:
            candidates.append(t)
    elif dy < -0.05 + threshold:
        t = _safe_tile(lon, lat - 0.1)
        if t is not None:
            candidates.append(t)

    # Corner case: near both x and y edges
    if len(candidates) == 3:
        t = _safe_tile(
            lon + (0.1 if dx > 0 else -0.1),
            lat + (0.1 if dy > 0 else -0.1),
        )
        if t is not None:
            candidates.append(t)

    # Deduplicate (clamping may produce same tile at boundaries)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _same_grid(src: GeoBox, dst: GeoBox) -> bool:
    """Whether two geoboxes share CRS, shape and pixel size (their origins may differ)."""
    same_crs = pyproj.CRS.from_user_input(str(src.crs)) == pyproj.CRS.from_user_input(str(dst.crs))
    a, b = src.affine, dst.affine
    return same_crs and src.shape == dst.shape and np.isclose(a.a, b.a) and np.isclose(a.e, b.e)


class TesseraTiledFetcher(TiledRasterFetcher):
    """Adapt :class:`TesseraFetcher` to the :class:`TiledRasterFetcher` API.

    ``TesseraFetcher`` works in native tile
    coordinates returning ``(patch, crs, transform)``. This adapter bins to the
    0.1 degree Tessera tile grid, delegates the read to ``TesseraFetcher``, and
    reprojects each native-CRS patch onto the caller's per-cell UTM
    ``dst_geobox`` so Tessera behaves like every other fetcher in the store.
    """

    TILE_DEG = 0.1

    def __init__(self, cache_dir: str | Path, year: int = 2024, dataset_version: str | None = None) -> None:
        self._inner = TesseraFetcher(cache_dir=str(cache_dir), year=year, dataset_version=dataset_version)

    @staticmethod
    def _tile_bin(lon: float, lat: float) -> TileKey:
        from math import floor

        return (int(floor(lon / TesseraTiledFetcher.TILE_DEG)), int(floor(lat / TesseraTiledFetcher.TILE_DEG)))

    def tile_key(self, lon: float, lat: float) -> TileKey:
        return self._tile_bin(lon, lat)

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[TileKey]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key: TileKey):  # not used; per-tile handled below
        return None

    def fetch_patches_for_tile(self, tile_key, cells, resampling=None, nodata=None):
        """Cut each cell's patch from its Tessera tile.

        A patch is the tile's own ``patch_size`` window centred on the pixel the
        cell centre falls in. When the tile is in the cell's UTM zone at the
        cell's pixel size, that window is written as is: its grid is within a
        pixel of the cell's geobox, and resampling one onto the other would
        shift the patch by a pixel and leave its last row or column uncovered.
        A tile in another zone is reprojected onto the geobox.
        """
        from geosae.data.download.fetchers.reproject import reproject_to_geobox

        tile_lon = tile_key[0] * self.TILE_DEG
        tile_lat = tile_key[1] * self.TILE_DEG
        raw = self._inner.fetch_patches_for_tile(
            tile_lon,
            tile_lat,
            [(cid, lon, lat) for cid, lon, lat, _ in cells],
            patch_size=cells[0][3].shape[0] if cells else 64,
        )
        geobox_by_cid = {cid: gb for cid, _, _, gb in cells}
        results = []
        for cid, patch, src_crs, src_transform in raw:
            dst_gb = geobox_by_cid.get(cid)
            if dst_gb is None:
                continue
            src_gb = GeoBox(patch.shape[:2], src_transform, str(src_crs))
            if _same_grid(src_gb, dst_gb):
                results.append((cid, patch))
                continue
            chw = np.transpose(patch, (2, 0, 1))  # (H,W,D) -> (D,H,W) for reproject
            out = reproject_to_geobox(chw, src_gb, dst_gb, resampling or 0, nodata)
            if out is None:
                continue
            results.append((cid, np.transpose(out, (1, 2, 0))))
        return results

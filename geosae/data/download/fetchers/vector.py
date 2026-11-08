"""Categorical layers published as polygons, rasterised onto each cell.

Some reference partitions -- the RESOLVE biomes and ecoregions among them -- are
published as a zipped vector file rather than as a raster. :class:`VectorZipFetcher`
downloads the archive once, and for every cell burns one attribute of the polygons
that cover it onto the cell's own geobox: a pixel takes the value of the polygon
its centre falls in, which is what nearest-neighbour resampling of a raster of the
same layer would give.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from odc.geo.geobox import GeoBox
from rasterio.warp import Resampling

from geosae.data.download.fetchers.raster_fetchers import PredownloadedRasterFetcher, TileKey

logger = logging.getLogger(__name__)


class VectorZipFetcher(PredownloadedRasterFetcher):
    """Rasterise one attribute of a zipped polygon layer onto each cell.

    Args:
        url: URL of the zipped vector file (any format GDAL reads from a zip).
        attribute: Column whose value is burnt into the pixels.
        valid: ``(low, high)`` inclusive range of values kept; anything else
            (e.g. a "not assigned" code) and every pixel no polygon covers
            become the patch's fill value.
        cache_dir: Local cache directory for the archive.
        filename: Cached file name (default: derived from ``url``).
        tile_deg: Cells are grouped into bins of this many degrees; the
            polygons of a bin are selected and clipped once.
    """

    def __init__(
        self,
        url: str,
        attribute: str,
        valid: tuple[int, int] | None = None,
        cache_dir: str = "/tmp/raster_cache",
        filename: str | None = None,
        tile_deg: float = 1.0,
    ):
        super().__init__(cache_dir)
        self._url = url
        self._attribute = attribute
        self._valid = tuple(valid) if valid is not None else None
        self._filename = filename
        self._tile_deg = float(tile_deg)
        self._frame = None
        self._lock = threading.Lock()

    def tile_key(self, lon: float, lat: float) -> TileKey:
        step = self._tile_deg
        return (float(np.floor(lon / step) * step), float(np.floor(lat / step) * step))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[TileKey]:
        return [self.tile_key(lon, lat)]

    def _polygons(self):
        """The layer as a GeoDataFrame in WGS84, loaded once per fetcher."""
        import geopandas as gpd

        with self._lock:
            if self._frame is None:
                path = Path(self._ensure_downloaded(self._url, self._filename))
                frame = gpd.read_file(f"zip://{path}", columns=[self._attribute])
                self._frame = frame.to_crs("EPSG:4326") if frame.crs and frame.crs.to_epsg() != 4326 else frame
        return self._frame

    def _burn_value(self, values: np.ndarray, fill: int) -> np.ndarray:
        values = np.asarray(values)
        if self._valid is None:
            return values
        low, high = self._valid
        return np.where((values >= low) & (values <= high), values, fill)

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Rasterise the polygons under each cell of one bin.

        Args:
            tile_key: The bin, as returned by :meth:`tile_key`.
            cells: ``(cell_idx, lon, lat, dst_geobox)`` tuples.
            resampling: Ignored: a polygon layer has one value per point.
            nodata: Fill value of pixels no (valid) polygon covers.

        Returns:
            ``(cell_idx, patch)`` for every cell, ``(H, W)`` of the attribute's dtype.
        """
        import geopandas as gpd
        import shapely
        from rasterio.features import rasterize

        fill = 0 if nodata is None else int(nodata)
        frame = self._polygons()
        bounds = [_wgs84_bounds(gb) for _, _, _, gb in cells]
        lo = np.min(np.asarray(bounds)[:, :2], axis=0)
        hi = np.max(np.asarray(bounds)[:, 2:], axis=0)
        hits = frame.iloc[frame.sindex.query(shapely.box(*lo, *hi))]
        if hits.empty:
            return [(c[0], np.full(c[3].shape, fill, dtype=np.int64)) for c in cells]
        geoms = shapely.make_valid(hits.geometry.values)
        values = self._burn_value(hits[self._attribute].to_numpy(), fill)

        results = []
        for (cell_idx, _lon, _lat, geobox), cell_bounds in zip(cells, bounds, strict=True):
            box = shapely.box(*cell_bounds)
            keep = shapely.intersects(geoms, box)
            out = np.full(geobox.shape, fill, dtype=np.int64)
            if keep.any():
                clipped = gpd.GeoSeries(shapely.intersection(geoms[keep], box), crs="EPSG:4326")
                local = clipped.to_crs(str(geobox.crs))
                shapes = [(g, int(v)) for g, v in zip(local.values, values[keep], strict=True) if not g.is_empty]
                if shapes:
                    out = rasterize(shapes, out_shape=geobox.shape, transform=geobox.affine, fill=fill, dtype="int64")
            results.append((cell_idx, out))
        return results


def _wgs84_bounds(geobox: GeoBox, pad_deg: float = 0.005) -> tuple[float, float, float, float]:
    """A geobox's extent in lon/lat, padded so reprojection never clips a corner."""
    left, bottom, right, top = geobox.extent.to_crs("EPSG:4326").boundingbox
    return (left - pad_deg, bottom - pad_deg, right + pad_deg, top + pad_deg)


__all__ = ["VectorZipFetcher"]

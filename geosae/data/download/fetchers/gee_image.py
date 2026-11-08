"""Any Earth Engine image or image collection, sampled onto each cell's geobox.

One ``ee.data.computePixels`` request per cell, with the cell's CRS, affine transform and
dimensions as the request grid, so Earth Engine samples each layer at exactly the destination pixel
centres (nearest neighbour, or bilinear when asked) and the patch needs no client-side reprojection.
A layer is one band of an image, or one band of a collection mosaic filtered by property values;
several layers are stacked into one ``(H, W, C)`` patch.

Example::

    fetcher = GeeImageFetcher(layers=[
        {"asset": "JRC/GSW1_4/GlobalSurfaceWater", "band": "occurrence", "unmask": 0},
        {"asset": "JRC/GSW1_4/GlobalSurfaceWater", "band": "seasonality", "unmask": 0},
    ])
    patch = fetcher.fetch_patch(lon, lat, dst_geobox)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from math import floor

import numpy as np
from odc.geo.geobox import GeoBox
from rasterio.warp import Resampling

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher, TileKey

logger = logging.getLogger(__name__)

_LAYER_KEYS = {"asset", "band", "collection", "where", "unmask", "year", "scale"}


def grid_request(dst_geobox: GeoBox) -> dict:
    """The ``computePixels`` ``grid`` that pins the output to ``dst_geobox`` exactly."""
    t = dst_geobox.transform
    return {
        "dimensions": {"width": int(dst_geobox.width), "height": int(dst_geobox.height)},
        "affineTransform": {
            "scaleX": t.a,
            "shearX": t.b,
            "translateX": t.c,
            "shearY": t.d,
            "scaleY": t.e,
            "translateY": t.f,
        },
        "crsCode": str(dst_geobox.crs),
    }


class GeeImageFetcher(TiledRasterFetcher):
    """Samples Earth Engine layers on each cell's destination grid with ``computePixels``.

    Each layer is a mapping with keys:

    - ``asset``: image id, or collection id when ``collection`` is true;
    - ``band``: band to take;
    - ``collection``: mosaic the collection (after ``where``) instead of opening one image;
    - ``where``: ``{property: value}`` equality filters applied to the collection;
    - ``year``: keep the collection's images that start in this calendar year (overrides the
      fetcher's ``year``);
    - ``unmask``: value written where the layer is masked; masked pixels are the patch's
      ``nodata`` otherwise;
    - ``scale``: divide the layer's values by this (e.g. SoilGrids' mapped units to physical units).

    Bilinear resampling is applied to every layer's source images before the request; nearest
    neighbour is Earth Engine's default. Bytes received are counted in :attr:`bytes_read`.

    Args:
        layers: The layers, stacked in order (one layer gives an ``(H, W)`` patch).
        gee_project: Earth Engine project; resolved like the AlphaEarth fetcher's when ``None``.
        retries: Attempts per cell before the cell is skipped.
        year: Calendar year applied to every collection layer without its own ``year``.

    Raises:
        ValueError: A layer has unknown keys or lacks ``asset`` / ``band``.
    """

    #: Bin lon/lat for ``tile_key`` work units; every request is independent.
    TILE_STEP_DEG = 0.1

    def __init__(
        self, layers: Sequence[Mapping], gee_project: str | None = None, retries: int = 4, year: int | None = None
    ):
        for layer in layers:
            unknown = set(layer) - _LAYER_KEYS
            if unknown or "asset" not in layer or "band" not in layer:
                raise ValueError(f"layer {dict(layer)} needs asset and band, and only keys {sorted(_LAYER_KEYS)}")
        if not layers:
            raise ValueError("GeeImageFetcher needs at least one layer")
        import ee

        from geosae.data.download.fetchers.alpha_earth_gee import _build_ee_credentials, _default_gee_project

        ee.Initialize(
            credentials=_build_ee_credentials(),
            project=gee_project or _default_gee_project(),
            opt_url=ee.data.HIGH_VOLUME_API_BASE_URL,
        )
        self._ee = ee
        self._layers = [{"year": year} | dict(layer) if year is not None else dict(layer) for layer in layers]
        self._retries = int(retries)
        self.bytes_read = 0

    # ------------------------------------------------------------------
    # TiledRasterFetcher interface
    # ------------------------------------------------------------------
    def tile_key(self, lon: float, lat: float) -> TileKey:
        step = self.TILE_STEP_DEG
        return (int(floor(lon / step)), int(floor(lat / step)))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[TileKey]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key):
        return None

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """One request per cell; a cell that fails every retry is logged and left out."""
        out = []
        for cell_idx, lon, lat, dst_geobox in cells:
            patch = self.fetch_patch(lon, lat, dst_geobox, resampling, nodata)
            if patch is not None:
                out.append((cell_idx, patch))
        return out

    def fetch_patch(
        self,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> np.ndarray | None:
        fill = np.nan if nodata is None else nodata
        request = {
            "expression": self._expression(resampling),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": grid_request(dst_geobox),
        }
        for attempt in range(self._retries):
            try:
                data = self._ee.data.computePixels(request)
                break
            except Exception:
                if attempt == self._retries - 1:
                    logger.warning("[gee_image] computePixels failed at (%.4f, %.4f)", lon, lat, exc_info=True)
                    return None
                time.sleep(2.0 * 2**attempt)
        self.bytes_read += int(data.nbytes)
        n = len(self._layers)
        scales = np.array([float(layer.get("scale") or 1.0) for layer in self._layers])
        values = np.stack([data[f"v{i}"].astype(np.float64) for i in range(n)], axis=-1) / scales
        valid = np.stack([data[f"m{i}"] > 0 for i in range(n)], axis=-1)
        values = np.where(valid, values, fill)
        return values[..., 0] if n == 1 else values

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _layer_image(self, layer: dict, resampling: Resampling):
        ee = self._ee
        bilinear = resampling == Resampling.bilinear
        if layer.get("collection"):
            col = ee.ImageCollection(layer["asset"])
            for prop, value in (layer.get("where") or {}).items():
                col = col.filter(ee.Filter.eq(prop, value))
            if layer.get("year") is not None:
                y = int(layer["year"])
                col = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
            col = col.select([layer["band"]]).map(lambda im: im.toDouble())
            if bilinear:
                col = col.map(lambda im: im.resample("bilinear"))
            # A fully masked bottom layer keeps the mosaic one band wide where no image matches.
            empty = ee.Image.constant(0).toDouble().rename(layer["band"]).updateMask(0)
            image = ee.ImageCollection([empty]).merge(col).mosaic()
        else:
            image = ee.Image(layer["asset"]).select([layer["band"]])
            if bilinear:
                image = image.resample("bilinear")
        if layer.get("unmask") is not None:
            image = image.unmask(layer["unmask"], False)
        return image

    def _expression(self, resampling: Resampling):
        """Values as float64 bands ``v{i}`` and their masks as uint8 bands ``m{i}``."""
        ee = self._ee
        bands = []
        for i, layer in enumerate(self._layers):
            image = self._layer_image(layer, resampling)
            bands.append(image.toDouble().unmask(0, False).rename(f"v{i}"))
            bands.append(image.mask().gt(0).uint8().unmask(0, False).rename(f"m{i}"))
        return ee.Image.cat(bands)


__all__ = ["GeeImageFetcher", "grid_request"]

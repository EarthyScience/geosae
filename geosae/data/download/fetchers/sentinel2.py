"""Sentinel-2 L2A surface reflectance from the Planetary Computer STAC catalog."""

from __future__ import annotations

import datetime
import logging

import numpy as np
from odc.geo.geobox import GeoBox
from rasterio.warp import Resampling

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher, TileKey, _open_cog_and_extract

logger = logging.getLogger(__name__)

#: Default STAC endpoint and collection for Sentinel-2 Level-2A.
SENTINEL2_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"

#: Reflectance bands written to the store, in ascending wavelength-name order.
SENTINEL2_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

#: Digital numbers are stored as ``reflectance * 10000``.
SENTINEL2_SCALE = 10_000
#: Products from processing baseline 04.00 onwards carry this radiometric offset.
SENTINEL2_BOA_OFFSET = 1000
#: Digital number marking a pixel with no observation.
SENTINEL2_NODATA_DN = 0
#: Store fill value for a pixel with no observation.
SENTINEL2_FILL = np.int16(-32768)

_PEAK_SUMMER_DOY = {"north": 196, "south": 15}  # mid-July / mid-January


def _seasonal_distance(acquired: datetime.datetime, lat: float) -> int:
    """Days between an acquisition and the local peak-of-growing-season date."""
    target = _PEAK_SUMMER_DOY["north" if lat >= 0 else "south"]
    delta = abs(acquired.timetuple().tm_yday - target)
    return min(delta, 365 - delta)


class Sentinel2L2AFetcher(TiledRasterFetcher):
    """One low-cloud Sentinel-2 L2A scene per cell, warped onto the cell geobox.

    For each cell the STAC catalog is searched over ``year`` for items below
    ``max_cloud_cover``; among those, the acquisition closest to the local
    peak-of-growing-season date is chosen, so that scene choice is a
    deterministic function of position and year rather than of cloudiness
    alone. Every band is warped onto the cell's UTM geobox and stacked into an
    ``(H, W, len(bands))`` int16 array holding ``reflectance * 10000``, with
    the baseline-04.00 radiometric offset removed and no-observation pixels set
    to :data:`SENTINEL2_FILL`.

    Args:
        year: Calendar year searched for imagery.
        max_cloud_cover: Scene-level cloud-cover ceiling in percent.
        bands: Asset keys to stack, in output order.
        min_valid_frac: Reject a scene unless at least this fraction of the
            cell's pixels carry an observation in every band.
        max_candidates: Ranked scenes tried before the cell is given up on.
        stac_url: STAC endpoint.
        collection: STAC collection id.
    """

    def __init__(
        self,
        year: int = 2024,
        max_cloud_cover: float = 10.0,
        bands: tuple[str, ...] = SENTINEL2_BANDS,
        min_valid_frac: float = 0.98,
        max_candidates: int = 5,
        stac_url: str = SENTINEL2_STAC_URL,
        collection: str = SENTINEL2_COLLECTION,
    ):
        self.year = int(year)
        self.max_cloud_cover = float(max_cloud_cover)
        self.bands = tuple(bands)
        self.min_valid_frac = float(min_valid_frac)
        self.max_candidates = int(max_candidates)
        self.stac_url = stac_url
        self.collection = collection
        self._catalog = None

    def _get_catalog(self):
        if self._catalog is None:
            import planetary_computer
            from pystac_client import Client

            self._catalog = Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)
        return self._catalog

    def tile_key(self, lon: float, lat: float) -> tuple[float, float]:
        """Each cell searches STAC independently, so the key is the cell itself."""
        return (round(lon, 4), round(lat, 4))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[float, float]]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key: TileKey) -> tuple[np.ndarray, GeoBox] | None:
        """Unused: reflectance is read band-by-band in :meth:`fetch_patches_for_tile`."""
        return None

    def rank_items(self, lon: float, lat: float) -> list:
        """Candidate scenes for a cell, best first.

        Items are ranked by distance to the local peak-of-growing-season date
        and then by scene cloud cover, so the choice is a deterministic
        function of position and year rather than of cloudiness alone.

        Args:
            lon: Cell centre longitude in degrees.
            lat: Cell centre latitude in degrees.

        Returns:
            Up to ``max_candidates`` signed :class:`pystac.Item` objects.
        """
        bbox = (
            max(lon - 0.01, -180.0),
            max(lat - 0.01, -90.0),
            min(lon + 0.01, 180.0),
            min(lat + 0.01, 90.0),
        )
        try:
            search = self._get_catalog().search(
                collections=[self.collection],
                bbox=bbox,
                datetime=f"{self.year}-01-01/{self.year}-12-31",
                query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
            )
            items = list(search.items())
        except Exception:
            logger.warning("Sentinel-2 STAC search failed at (%.4f, %.4f)", lon, lat, exc_info=True)
            return []
        items.sort(key=lambda it: (_seasonal_distance(it.datetime, lat), it.properties["eo:cloud_cover"]))
        return items[: self.max_candidates]

    @staticmethod
    def _offset(item) -> int:
        """Radiometric offset to subtract from the item's digital numbers."""
        baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
        try:
            return SENTINEL2_BOA_OFFSET if float(baseline) >= 4.0 else 0
        except ValueError:
            return 0

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.bilinear,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        results: list[tuple[int, np.ndarray]] = []
        for cell_idx, lon, lat, dst_geobox in cells:
            # Candidates are tried in rank order: the best-ranked scene may still
            # miss the cell, because the cell can sit at a granule edge.
            for item in self.rank_items(lon, lat):
                patch = self._read_bands(item, cell_idx, lon, lat, dst_geobox, resampling)
                if patch is not None:
                    results.append((cell_idx, patch))
                    break
            else:
                logger.info(
                    "No complete Sentinel-2 %d patch below %.0f%% cloud at (%.4f, %.4f)",
                    self.year,
                    self.max_cloud_cover,
                    lon,
                    lat,
                )
        return results

    def _read_bands(self, item, cell_idx, lon, lat, dst_geobox, resampling) -> np.ndarray | None:
        """Stack every band of one item onto the cell geobox, or ``None`` on failure."""
        offset = self._offset(item)
        height, width = dst_geobox.shape
        out = np.full((height, width, len(self.bands)), SENTINEL2_FILL, dtype=np.int16)
        for b, band in enumerate(self.bands):
            asset = item.assets.get(band)
            if asset is None:
                logger.warning("Sentinel-2 item %s has no asset %s", item.id, band)
                return None
            read = _open_cog_and_extract(
                asset.href,
                [(cell_idx, lon, lat, dst_geobox)],
                resampling,
                nodata=SENTINEL2_NODATA_DN,
                bands=1,
            )
            if not read:
                return None
            dn = np.asarray(read[0][1]).reshape(height, width)
            valid = dn != SENTINEL2_NODATA_DN
            if valid.mean() < self.min_valid_frac:
                return None
            corrected = dn.astype(np.int32) - offset
            clipped = np.clip(corrected, SENTINEL2_FILL + 1, np.iinfo(np.int16).max)
            out[..., b] = np.where(valid, clipped, SENTINEL2_FILL)
        return out


__all__ = [
    "SENTINEL2_STAC_URL",
    "SENTINEL2_COLLECTION",
    "SENTINEL2_BANDS",
    "SENTINEL2_SCALE",
    "SENTINEL2_BOA_OFFSET",
    "SENTINEL2_NODATA_DN",
    "SENTINEL2_FILL",
    "Sentinel2L2AFetcher",
]

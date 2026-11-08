"""AlphaEarth Embedding Field (AEF) fetcher from Source Cooperative S3 COGs.

Streams 64-dimension pixel embeddings from the public AEF Cloud-Optimized
GeoTIFF archive. Complements the Earth Engine-based fetcher
(:class:`~geosae.data.download.fetchers.alpha_earth_gee.AlphaEarthGEEFetcher`,
which requires a Google Earth Engine project) with a provider-free path.

The COG index is a small GeoParquet file listing per-tile bounding boxes,
CRS, and S3 paths; it gets cached locally on first use. Per-tile reads go
through the shared :func:`_open_cog_and_extract` helper so all cells
sharing a COG hit one block read (one HTTP range per ~1024x1024 block)
instead of one HTTP round-trip per cell.

Dequantization follows the AEF spec: ``sign(x) * |x/127.5|^2`` over int8
inputs producing float32 embeddings.

Example::

    fetcher = AlphaEarthCOGFetcher(year=2024)
    patch = fetcher.fetch_patch(lon=11.5, lat=48.1, dst_geobox=geobox)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.warp import Resampling

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher, TileKey, _block_read_and_extract

logger = logging.getLogger(__name__)


# Public AEF archive on Source Cooperative (AWS us-west-2).
AEF_INDEX_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"

# AlphaEarth quantization constants (per the AEF spec).
AEE_POWER = 2.0
AEE_SCALE = 127.5
N_BANDS = 64
#: Fill value of the int8 COGs where a file does not record its own.
AEF_NODATA = -128
#: Margin read around a patch, degrees (~200 m): nearest-neighbour resampling of a
#: 10 m grid onto a 10 m grid needs no more, and 64 bands make every pixel count.
READ_BUFFER_DEG = 0.002
#: GDAL threads per read: the 64 bands are separate tiles, fetched in parallel.
READ_THREADS = 8
#: GDAL block cache per read, MB; its default (5% of RAM) holds every band of every tile read.
READ_CACHE_MB = 256


def _dequantize(data: np.ndarray) -> np.ndarray:
    """AEF int8 -> float32 dequantization: ``sign(x) * |x/127.5|^2``."""
    rescaled = data.astype(np.float64) / AEE_SCALE
    return (np.abs(rescaled) ** AEE_POWER * np.sign(rescaled)).astype(np.float32)


class AlphaEarthCOGFetcher(TiledRasterFetcher):
    """Streams AlphaEarth patches from Source Cooperative COGs on AWS S3.

    The parquet index at ``index_cache`` is populated from ``AEF_INDEX_URL``
    on first use. Each COG covers ~1.5°-2° and is typically 8192x8192 pixels.
    Tile keys are the S3 COG paths; ``fetch_patches_for_tile`` performs one
    block-grouped read per COG to share HTTP round-trips across cells.

    Args:
        year: AEF year to pull (default 2024).
        index_cache: Local path for the cached index parquet. Directory is
            created on demand. Set to ``None`` to force re-download every
            time the fetcher is constructed.
    """

    def __init__(self, year: int = 2024, index_cache: str | Path | None = None):
        import geopandas as gpd

        self.year = year
        cache = Path(index_cache) if index_cache is not None else Path(tempfile.gettempdir()) / "aef_index.parquet"

        if not cache.exists():
            import s3fs

            logger.info("Downloading AEF index from %s", AEF_INDEX_URL)
            fs = s3fs.S3FileSystem(anon=True)
            df = gpd.read_parquet(AEF_INDEX_URL, filesystem=fs)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(str(cache))
            logger.info("Cached AEF index to %s (%d rows)", cache, len(df))
        else:
            df = gpd.read_parquet(str(cache))

        self._index = df[df["year"] == year].copy()
        if len(self._index) == 0:
            raise ValueError(f"AEF index has no tiles for year {year}")

    # ------------------------------------------------------------------
    # TiledRasterFetcher interface
    # ------------------------------------------------------------------

    def tile_key(self, lon: float, lat: float) -> TileKey | None:
        """Return the S3 COG path that contains ``(lon, lat)``, or ``None``."""
        hits = self._index[
            (self._index["wgs84_west"] <= lon)
            & (self._index["wgs84_east"] >= lon)
            & (self._index["wgs84_south"] <= lat)
            & (self._index["wgs84_north"] >= lat)
        ]
        if len(hits) == 0:
            return None
        return str(hits.iloc[0]["path"])

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[TileKey]:
        """Primary COG plus any neighbor whose bbox is within a patch-sized buffer."""
        primary = self.tile_key(lon, lat)
        if primary is None:
            return []
        buffer_deg = 0.01  # ~1 km at the equator; always > the 640 m patch extent
        hits = self._index[
            (self._index["wgs84_west"] <= lon + buffer_deg)
            & (self._index["wgs84_east"] >= lon - buffer_deg)
            & (self._index["wgs84_south"] <= lat + buffer_deg)
            & (self._index["wgs84_north"] >= lat - buffer_deg)
        ]
        candidates = [primary]
        for path in hits["path"]:
            p = str(path)
            if p != primary:
                candidates.append(p)
        return candidates

    def fetch_tile(self, tile_key: TileKey) -> tuple[np.ndarray, GeoBox] | None:
        """Not used -- the streaming path goes through :meth:`fetch_patches_for_tile`."""
        return None

    def fetch_patches_for_tile(
        self,
        tile_key: TileKey,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Block-read the COG once and extract all cells via in-memory reprojection.

        Returns patches in ``(H, W, 64)`` float32 with AEF dequantization applied.
        """
        url = str(tile_key)
        # GDAL can't open s3:// directly via /vsicurl/; it expects /vsis3/.
        # The AEF parquet index stores s3:// paths for the Source Cooperative
        # bucket (public, requester-pays-off). Translate to /vsis3/ and set
        # AWS_NO_SIGN_REQUEST for anonymous access. HTTPS URLs go through
        # /vsicurl/ unchanged.
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
                AWS_REGION="us-west-2",
                GDAL_NUM_THREADS=str(READ_THREADS),
                GDAL_CACHEMAX=READ_CACHE_MB,
            ):
                with rasterio.open(gdal_path) as src:
                    bands = list(range(1, N_BANDS + 1))
                    # Resample the int8 codes with their own fill; ``nodata`` (NaN for the
                    # float32 store array) is applied after dequantisation.
                    src_nodata = AEF_NODATA if src.nodata is None else int(src.nodata)
                    raw = _block_read_and_extract(
                        src, cells, resampling, src_nodata, buffer_deg=READ_BUFFER_DEG, bands=bands
                    )
        except Exception:
            logger.warning("Failed to open AEF COG %s", url[:80], exc_info=True)
            return []

        fill = np.nan if nodata is None else nodata
        results: list[tuple[int, np.ndarray]] = []
        for cell_idx, patch in raw:
            if patch is None:
                continue
            # patch arrives as (bands, H, W); move bands last + dequantize
            if patch.ndim == 3 and patch.shape[0] == N_BANDS:
                patch = np.transpose(patch, (1, 2, 0))
            values = _dequantize(patch)
            values[patch == src_nodata] = fill
            results.append((cell_idx, values))
        return results

    # ------------------------------------------------------------------
    # Convenience: single-cell fetch_patch
    # ------------------------------------------------------------------

    def fetch_patch(
        self,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> np.ndarray | None:
        """Single-patch convenience wrapper.

        Goes through :meth:`fetch_patches_for_tile` with one cell so the
        dequantization path is shared with the batched path.
        """
        primary = self.tile_key(lon, lat)
        if primary is None:
            return None
        results = self.fetch_patches_for_tile(
            primary, [(0, lon, lat, dst_geobox)], resampling=resampling, nodata=nodata
        )
        if not results:
            return None
        return results[0][1]

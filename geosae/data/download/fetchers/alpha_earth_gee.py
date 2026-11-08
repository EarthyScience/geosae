"""AlphaEarth Embedding Field (AEF) fetcher via Google Earth Engine.

Streams 64-channel AEF embeddings directly from the
``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` image collection on GEE. Server-side
quantization (``int8``) minimizes download bandwidth; client-side
dequantization recovers ``float32`` embeddings.

Subclasses :class:`~geosae.data.download.fetchers.raster_fetchers.TiledRasterFetcher`.
Every call is independent (no block-grouped batching), so ``tile_key``
bucketization is cosmetic -- it just keeps the fetch loop's work-unit
structure consistent with the other fetchers.

Requires a Google Earth Engine project (``ee.Initialize(project=...)``).

Example::

    fetcher = AlphaEarthGEEFetcher(gee_project='my-gee-project')
    patch = fetcher.fetch_patch(lon, lat, dst_geobox)
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from math import floor

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.warp import Resampling

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher, TileKey
from geosae.data.download.fetchers.reproject import reproject_to_geobox

logger = logging.getLogger(__name__)


AEE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEE_POWER = 2.0
AEE_SCALE = 127.5
N_BANDS = 64

# EE-required scopes in order of preference. ``ee.Initialize()``'s default
# code path asks for a broader list (``drive``, ``devstorage.full_control``,
# ``userinfo.email``) that Google's token endpoint rejects with
# ``invalid_scope`` for most user accounts. ``getDownloadURL`` only needs
# ``earthengine[.readonly]`` + ``cloud-platform[.read-only]``.
#
# The refresh_token is bound to whatever scopes the original OAuth grant
# issued. Asking for a superset on refresh (e.g. requesting ``earthengine``
# when the grant only included ``earthengine.readonly``) is treated as a
# scope escalation and rejected. Solution: filter the stored scope list to
# the EE-relevant ones and pass *exactly those* through, preserving the
# original ``.readonly`` suffixes when present.
_EE_RELEVANT_SCOPE_PREFIXES = (
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
)
_EE_FALLBACK_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]


def _select_ee_scopes(stored_scopes: list[str] | None) -> list[str]:
    """Pick the EE scopes present in ``stored_scopes`` verbatim.

    Preserves ``.readonly`` / ``.read-only`` suffixes so the refresh grant
    matches the token's original scope set exactly. Falls back to the
    canonical full-access scopes if nothing relevant is stored.
    """
    if not stored_scopes:
        return list(_EE_FALLBACK_SCOPES)
    selected = [s for s in stored_scopes if s.startswith(_EE_RELEVANT_SCOPE_PREFIXES)]
    return selected if selected else list(_EE_FALLBACK_SCOPES)


def _build_ee_credentials():
    """Construct a :class:`google.oauth2.credentials.Credentials` from the
    standard ``~/.config/earthengine/credentials`` file using the EE-relevant
    subset of the stored OAuth scopes.
    """
    import json
    from pathlib import Path

    from google.oauth2.credentials import Credentials

    path = Path.home() / ".config" / "earthengine" / "credentials"
    if not path.exists():
        raise FileNotFoundError(f"GEE credentials not found at {path}. Run `earthengine authenticate` first.")
    data = json.loads(path.read_text())
    return Credentials(
        token=None,
        refresh_token=data["refresh_token"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=None,  # let Google use the original grant scopes; the
        # endpoint rejects an explicit scope list with
        # ``invalid_scope`` for our refresh tokens.
    )


def _dequantize(data: np.ndarray) -> np.ndarray:
    """AEF int8 -> float32 dequantization: ``sign(x) * |x/127.5|^2``."""
    rescaled = data.astype(np.float64) / AEE_SCALE
    return (np.abs(rescaled) ** AEE_POWER * np.sign(rescaled)).astype(np.float32)


def _default_gee_project() -> str | None:
    """Resolve a GEE project without an explicit argument.

    Order: ``GEOSAE_GEE_PROJECT`` env var, then the ``project`` field of the
    standard ``~/.config/earthengine/credentials`` file. Returns ``None`` if
    neither is set (the caller then raises a clear error).
    """
    env = os.environ.get("GEOSAE_GEE_PROJECT")
    if env:
        return env
    try:
        import json
        from pathlib import Path

        data = json.loads((Path.home() / ".config" / "earthengine" / "credentials").read_text())
        proj = data.get("project")
        return str(proj) if proj else None
    except Exception:
        return None


class AlphaEarthGEEFetcher(TiledRasterFetcher):
    """Fetches AlphaEarth patches from Google Earth Engine (high-volume API).

    This is the default AlphaEarth source (``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``
    via GEE);
    :class:`~geosae.data.download.fetchers.alpha_earth_cog.AlphaEarthCOGFetcher`
    is a documented credential-free fallback that streams the Source Cooperative
    S3 COG mirror instead.

    ``ee`` is initialised against the **high-volume** endpoint
    (:data:`ee.data.HIGH_VOLUME_API_BASE_URL`). Each cell is fetched with
    :func:`ee.data.computePixels`, requesting pixels **directly on the cell's
    destination geobox grid** (CRS + affine + dimensions), so GEE resamples
    server-side onto the exact UTM grid and no client-side reprojection is
    needed -- the patch is aligned to ``dst_geobox`` by construction. A
    ``getDownloadURL('NPY')`` + GeoTIFF-probe + reproject path is retained as a
    fallback (:meth:`_fetch_via_download_url`).

    Args:
        gee_project: GEE project for ``ee.Initialize``. If ``None``, resolved
            from ``$GEOSAE_GEE_PROJECT`` or the ``project`` field of
            ``~/.config/earthengine/credentials``.
        year: AEF annual vintage to pull (default 2024).
        request_timeout_s: HTTP timeout for the download-URL fallback.
    """

    #: Bin lon/lat onto a ~10 km grid for ``tile_key`` bucketization. Purely
    #: for orchestration; all GEE calls are independent.
    TILE_STEP_DEG = 0.1

    def __init__(self, gee_project: str | None = None, year: int = 2024, request_timeout_s: float = 180.0):
        import ee  # noqa: F401  -- import lazily at construction time

        project = gee_project or _default_gee_project()
        if not project:
            raise ValueError(
                "AlphaEarth via GEE needs a project. Pass gee_project=, set "
                '$GEOSAE_GEE_PROJECT, or add a "project" to '
                "~/.config/earthengine/credentials (run `earthengine authenticate`)."
            )

        # Initialise against the Earth Engine HIGH-VOLUME endpoint
        # (``ee.data.HIGH_VOLUME_API_BASE_URL``). All compute calls issued below
        # -- including ``ee.data.computePixels`` and the ``getDownloadURL``
        # fallback's ``getDownloadId`` -- are then routed through the
        # high-volume host, which is the correct endpoint for many small
        # per-cell requests (vs. the standard interactive endpoint).
        ee.Initialize(
            credentials=_build_ee_credentials(),
            project=project,
            opt_url=ee.data.HIGH_VOLUME_API_BASE_URL,
        )
        self._ee = ee
        self._collection = ee.ImageCollection(AEE_COLLECTION)
        self.year = int(year)
        self.request_timeout_s = float(request_timeout_s)
        self._bands = [f"A{str(i).zfill(2)}" for i in range(N_BANDS)]

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
        """Fetch every cell in this tile via one GEE request per cell.

        Each cell issues one high-volume ``computePixels`` request (the
        ``getDownloadURL`` fallback round-trips twice). Failures on a single
        cell are logged and skipped; the rest of the batch still proceeds.
        """
        results: list[tuple[int, np.ndarray]] = []
        for cell_idx, lon, lat, dst_geobox in cells:
            try:
                patch = self._fetch_one(lon, lat, dst_geobox, resampling, nodata)
            except Exception:
                logger.warning("[alphaearth_gee] fetch failed at (%.4f, %.4f)", lon, lat, exc_info=True)
                continue
            if patch is not None:
                results.append((cell_idx, patch))
        return results

    def fetch_patch(
        self,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> np.ndarray | None:
        try:
            return self._fetch_one(lon, lat, dst_geobox, resampling, nodata)
        except Exception:
            logger.warning("[alphaearth_gee] fetch failed at (%.4f, %.4f)", lon, lat, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_quantized_image(self, lon: float, lat: float):
        """Return ``(quant_img, image)`` -- the AEF image for this cell/year with
        the int8 requantization applied -- or ``None`` if no image covers it."""
        ee = self._ee
        point = ee.Geometry.Point([lon, lat])
        images = self._collection.filterDate(f"{self.year}-01-01", f"{self.year + 1}-01-01").filterBounds(point)
        if images.size().getInfo() == 0:
            logger.debug("[alphaearth_gee] no images at (%.4f, %.4f) for year %d", lon, lat, self.year)
            return None
        image = ee.Image(images.first())
        quantized = image.select(self._bands)
        sat = quantized.abs().pow(1 / AEE_POWER).multiply(quantized.signum())
        quant_img = sat.multiply(AEE_SCALE).clamp(-127.0, 127.0).int8()
        return quant_img, image

    def _fetch_one(
        self,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling,
        nodata: float | int | None,
    ) -> np.ndarray | None:
        built = self._build_quantized_image(lon, lat)
        if built is None:
            return None
        quant_img, image = built
        # Primary: the high-volume pixel path. ``ee.data.computePixels`` returns
        # pixels *directly on the destination geobox grid* (crs + affine +
        # dimensions), so GEE does the resampling server-side onto exactly the
        # cell's UTM grid -- no client-side reproject, no CRS probe, and the
        # result is aligned to ``dst_geobox`` by construction (offset-free).
        try:
            return self._compute_pixels_on_geobox(quant_img, dst_geobox)
        except Exception:
            logger.warning(
                "[alphaearth_gee] computePixels failed at (%.4f, %.4f); falling back to getDownloadURL",
                lon,
                lat,
                exc_info=True,
            )
            return self._fetch_via_download_url(quant_img, image, lon, lat, dst_geobox, resampling, nodata)

    def _compute_pixels_on_geobox(self, quant_img, dst_geobox: GeoBox) -> np.ndarray:
        """High-volume ``ee.data.computePixels`` request on the exact dst grid.

        The ``grid`` pins the output CRS + affine transform + dimensions to
        ``dst_geobox``, so GEE resamples onto the cell's native UTM grid and the
        returned array needs no client-side reprojection.
        """
        ee = self._ee
        t = dst_geobox.transform
        request = {
            "expression": quant_img,
            "fileFormat": "NUMPY_NDARRAY",
            "bandIds": self._bands,
            "grid": {
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
            },
        }
        data = ee.data.computePixels(request)  # structured (H, W) array, band fields
        arr = np.stack([data[name] for name in data.dtype.names], axis=-1)
        return _dequantize(arr)

    def _fetch_via_download_url(
        self,
        quant_img,
        image,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling,
        nodata: float | int | None,
    ) -> np.ndarray | None:
        """Fallback pixel path: ``getDownloadURL('NPY')`` + a GeoTIFF CRS probe,
        then reproject onto ``dst_geobox``. Used only if ``computePixels`` fails.

        Note this path reprojects GEE's own region grid onto the cell geobox and
        so can differ from the ``computePixels`` result by up to a nearest-
        neighbour pixel where the two grids do not align.
        """
        ee = self._ee
        # Size the GEE request around the destination extent so the reprojected
        # patch has full coverage. Use the dst_geobox ground footprint in meters.
        dx = abs(dst_geobox.transform.a) * dst_geobox.width
        dy = abs(dst_geobox.transform.e) * dst_geobox.height
        half_m = max(dx, dy) / 2.0 + abs(dst_geobox.transform.a)  # 1-pixel pad
        region = ee.Geometry.Point([lon, lat]).buffer(half_m).bounds()

        npy_url = quant_img.getDownloadURL({"scale": 10, "region": region, "format": "NPY", "bands": self._bands})
        with urllib.request.urlopen(npy_url, timeout=self.request_timeout_s) as resp:
            raw_npy = resp.read()

        data = np.load(io.BytesIO(raw_npy), allow_pickle=False)
        arr = _dequantize(np.stack([data[name] for name in data.dtype.names], axis=-1))

        probe_url = image.select([self._bands[0]]).getDownloadURL({"scale": 10, "region": region, "format": "GEO_TIFF"})
        with urllib.request.urlopen(probe_url, timeout=self.request_timeout_s) as resp:
            probe_raw = resp.read()
        with rasterio.MemoryFile(probe_raw) as memfile:
            with memfile.open() as ds:
                src_crs = f"EPSG:{ds.crs.to_epsg()}" if ds.crs.to_epsg() else str(ds.crs)
                src_transform = ds.transform
                src_h, src_w = ds.height, ds.width

        if arr.shape[0] != src_h or arr.shape[1] != src_w:
            h, w = arr.shape[:2]
            if h >= src_h and w >= src_w:
                r0 = (h - src_h) // 2
                c0 = (w - src_w) // 2
                arr = arr[r0 : r0 + src_h, c0 : c0 + src_w, :]
            else:
                logger.debug(
                    "[alphaearth_gee] probe/data shape mismatch: data=%s probe=(%d,%d)", arr.shape, src_h, src_w
                )
                return None

        src_geobox = GeoBox((src_h, src_w), src_transform, src_crs)
        out_chw = reproject_to_geobox(np.transpose(arr, (2, 0, 1)), src_geobox, dst_geobox, resampling, nodata)
        if out_chw is None:
            return None
        return np.transpose(out_chw, (1, 2, 0))

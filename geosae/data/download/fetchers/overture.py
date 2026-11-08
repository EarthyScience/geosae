"""Overture Maps fetcher for built-environment labels.

Queries Overture Maps GeoParquet on S3 via DuckDB and rasterizes to a 5-band
uint8 patch aligned to the cell's native UTM grid. Subclasses
:class:`~geosae.data.download.fetchers.raster_fetchers.TiledRasterFetcher`.

Bands:
    0 -- Building Use (buildings/building subtype)
    1 -- Land Use Intent (base/land_use class)
    2 -- Road Class (transportation/segment, buffered linestrings)
    3 -- Infrastructure (base/infrastructure class)
    4 -- Water / Natural (base/water + base/land subtype)
"""

from __future__ import annotations

import logging
import threading
from math import floor
from typing import TYPE_CHECKING

import numpy as np
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)


N_BANDS = 5
OVERTURE_STAC_URL = "https://stac.overturemaps.org/catalog.json"
FALLBACK_RELEASE = "2026-03-18.0"


# ---------------------------------------------------------------------------
# Band class tables. Every positive code is a label; 0 is background/no-data.
# Kept inline so this module is fully self-contained.
# ---------------------------------------------------------------------------

OVERTURE_BUILDING_USE = {
    0: "background",
    1: "residential",
    2: "commercial",
    3: "industrial",
    4: "education",
    5: "medical",
    6: "religious",
    7: "civic",
    8: "entertainment",
    9: "agricultural",
    10: "military",
    11: "transportation",
    12: "service",
    13: "outbuilding",
}

OVERTURE_LAND_USE = {
    0: "background",
    1: "farmland",
    2: "residential",
    3: "industrial",
    4: "commercial",
    5: "cemetery",
    6: "park",
    7: "quarry",
    8: "orchard",
    9: "vineyard",
    10: "military",
    11: "stadium",
    12: "golf_course",
    13: "zoo",
    14: "construction",
    15: "landfill",
    16: "forest",
    17: "meadow",
    18: "recreation_ground",
    19: "allotments",
    20: "retail",
    21: "education",
    22: "religious",
    23: "railway",
    24: "port",
    25: "aquaculture",
    26: "plant_nursery",
    27: "winter_sports",
    28: "other",
}

OVERTURE_ROAD_CLASS = {
    0: "background",
    1: "motorway",
    2: "trunk",
    3: "primary",
    4: "secondary",
    5: "tertiary",
    6: "residential",
    7: "service",
    8: "railway",
}

# Buffer half-widths in meters for road rasterization at 10 m.
OVERTURE_ROAD_BUFFER_M = {1: 10.0, 2: 7.5, 3: 6.0, 4: 5.0, 5: 3.0, 6: 3.0, 7: 2.0, 8: 3.0}

OVERTURE_INFRASTRUCTURE = {
    0: "background",
    1: "runway",
    2: "taxiway",
    3: "apron",
    4: "bridge",
    5: "dam",
    6: "pier",
    7: "substation",
    8: "tower",
    9: "solar_panel_area",
    10: "water_tower",
    11: "power_line",
    12: "pipeline",
    13: "communication_tower",
    14: "waste_disposal",
    15: "other",
}

OVERTURE_WATER_NATURAL = {
    0: "background",
    1: "ocean",
    2: "lake",
    3: "river",
    4: "canal",
    5: "reservoir",
    6: "wetland",
    7: "forest",
    8: "glacier",
    9: "sand",
    10: "bare_rock",
    11: "shrub",
    12: "grass",
}

OVERTURE_BUILDING_USE_REV = {v: k for k, v in OVERTURE_BUILDING_USE.items() if k > 0}
OVERTURE_LAND_USE_REV = {v: k for k, v in OVERTURE_LAND_USE.items() if k > 0}
OVERTURE_ROAD_CLASS_REV = {v: k for k, v in OVERTURE_ROAD_CLASS.items() if k > 0}
OVERTURE_INFRASTRUCTURE_REV = {v: k for k, v in OVERTURE_INFRASTRUCTURE.items() if k > 0}
OVERTURE_WATER_NATURAL_REV = {v: k for k, v in OVERTURE_WATER_NATURAL.items() if k > 0}

OVERTURE_BANDS = [
    (OVERTURE_BUILDING_USE, OVERTURE_BUILDING_USE_REV),
    (OVERTURE_LAND_USE, OVERTURE_LAND_USE_REV),
    (OVERTURE_ROAD_CLASS, OVERTURE_ROAD_CLASS_REV),
    (OVERTURE_INFRASTRUCTURE, OVERTURE_INFRASTRUCTURE_REV),
    (OVERTURE_WATER_NATURAL, OVERTURE_WATER_NATURAL_REV),
]

OVERTURE_BAND_NAMES = ["Building Use", "Land Use", "Road Class", "Infrastructure", "Water/Natural"]


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------


def _create_connection() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection pre-configured for Overture S3 reads."""
    import duckdb

    con = duckdb.connect()
    con.install_extension("httpfs")
    con.load_extension("httpfs")
    con.install_extension("spatial")
    con.load_extension("spatial")
    con.execute("SET s3_region = 'us-west-2';")
    return con


def _resolve_s3_base(con, release: str | None) -> str:
    """Resolve the Overture S3 base path, auto-detecting latest release if needed."""
    if release is None:
        try:
            row = con.execute(f"SELECT latest FROM '{OVERTURE_STAC_URL}'").fetchone()
            release = row[0]
            logger.info("Overture: auto-detected latest release %s from STAC", release)
        except Exception:
            release = FALLBACK_RELEASE
            logger.warning("Overture: STAC lookup failed, falling back to %s", release)
    return f"s3://overturemaps-us-west-2/release/{release}"


def _bbox_filter(xmin: float, ymin: float, xmax: float, ymax: float) -> str:
    """SQL WHERE clause for Overture bbox pushdown."""
    return f"bbox.xmin <= {xmax} AND bbox.xmax >= {xmin} AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}"


def _query_theme(
    con,
    s3_base: str,
    theme: str,
    type_: str,
    columns: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    extra_where: str = "",
) -> list[tuple]:
    """Run a bbox-pushdown query against one Overture theme/type."""
    path = f"{s3_base}/theme={theme}/type={type_}/*"
    where = _bbox_filter(xmin, ymin, xmax, ymax)
    if extra_where:
        where += f" AND ({extra_where})"
    select_cols = columns.replace("geometry", "ST_AsWKB(geometry) AS geometry")
    sql = f"SELECT {select_cols} FROM read_parquet('{path}') WHERE {where}"
    try:
        return con.execute(sql).fetchall()
    except Exception:
        logger.debug("DuckDB query failed: %s", sql, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Geometry + rasterization helpers
# ---------------------------------------------------------------------------


def _transform_geometries(geoms, src_crs: str, dst_crs: str):
    import pyproj
    import shapely.ops

    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return [shapely.ops.transform(transformer.transform, g) for g in geoms]


def _rasterize_shapes(shapes, out_shape, transform: Affine, all_touched: bool = False) -> np.ndarray:
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        dtype=np.uint8,
        fill=0,
        all_touched=all_touched,
    )


def _rasterize_polygon_band(
    rows,
    cls_rev: dict,
    cell_transform: Affine,
    cell_crs: str,
    patch_size: int,
    fallback_code: int = 0,
) -> np.ndarray:
    """Shared path for non-road polygon bands."""
    from shapely import from_wkb

    shapes = []
    for row in rows:
        geom_wkb, class_val = row[0], row[1]
        code = cls_rev.get(class_val, fallback_code)
        if code == 0:
            continue
        try:
            geom = from_wkb(geom_wkb)
        except Exception:
            continue
        shapes.append((geom, code))
    if not shapes:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    geoms, codes = zip(*shapes, strict=True)
    geoms = _transform_geometries(list(geoms), "EPSG:4326", cell_crs)
    return _rasterize_shapes(list(zip(geoms, codes, strict=True)), (patch_size, patch_size), cell_transform)


def _rasterize_roads(rows, cell_transform, cell_crs, patch_size) -> np.ndarray:
    """Band 2: road/rail segments buffered by class-specific half-width in metres."""
    from shapely import from_wkb

    shapes = []
    for geom_wkb, class_val, subtype in rows:
        if subtype == "rail":
            code = OVERTURE_ROAD_CLASS_REV.get("railway", 0)
        else:
            code = OVERTURE_ROAD_CLASS_REV.get(class_val, 0)
        if code == 0:
            continue
        try:
            geom = from_wkb(geom_wkb)
        except Exception:
            continue
        shapes.append((geom, code))
    if not shapes:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    geoms, codes = zip(*shapes, strict=True)
    geoms = _transform_geometries(list(geoms), "EPSG:4326", cell_crs)
    buffered = [(g.buffer(OVERTURE_ROAD_BUFFER_M.get(c, 3.0)), c) for g, c in zip(geoms, codes, strict=True)]
    return _rasterize_shapes(buffered, (patch_size, patch_size), cell_transform, all_touched=True)


def _rasterize_water_natural(water_rows, land_rows, cell_transform, cell_crs, patch_size) -> np.ndarray:
    """Band 4: water + natural land combined."""
    from shapely import from_wkb

    shapes = []
    for geom_wkb, class_val, subtype in water_rows:
        label = subtype if subtype else class_val
        code = OVERTURE_WATER_NATURAL_REV.get(label, 0)
        if code == 0:
            code = OVERTURE_WATER_NATURAL_REV.get(class_val, 0)
        if code == 0:
            continue
        try:
            geom = from_wkb(geom_wkb)
        except Exception:
            continue
        shapes.append((geom, code))
    for geom_wkb, subtype in land_rows:
        code = OVERTURE_WATER_NATURAL_REV.get(subtype, 0)
        if code == 0:
            continue
        try:
            geom = from_wkb(geom_wkb)
        except Exception:
            continue
        shapes.append((geom, code))
    if not shapes:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    geoms, codes = zip(*shapes, strict=True)
    geoms = _transform_geometries(list(geoms), "EPSG:4326", cell_crs)
    return _rasterize_shapes(list(zip(geoms, codes, strict=True)), (patch_size, patch_size), cell_transform)


# ---------------------------------------------------------------------------
# Cell-level orchestration
# ---------------------------------------------------------------------------


def _geobox_to_wgs84_bbox(geobox: GeoBox) -> tuple[float, float, float, float]:
    """WGS84 bbox covering a (typically UTM) GeoBox, with a ~100 m buffer."""
    import pyproj

    t = geobox.transform
    h, w = geobox.shape
    corners = [
        (t.c, t.f),
        (t.c + w * t.a, t.f),
        (t.c, t.f + h * t.e),
        (t.c + w * t.a, t.f + h * t.e),
    ]
    transformer = pyproj.Transformer.from_crs(str(geobox.crs), "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for x, y in corners:
        lon, lat = transformer.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    buf = 0.001
    return min(lons) - buf, min(lats) - buf, max(lons) + buf, max(lats) + buf


def _rasterize_cell(
    con,
    s3_base: str,
    geobox: GeoBox,
    patch_size: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> np.ndarray:
    """Fetch + rasterize all 5 Overture bands for one cell."""
    bands = np.zeros((patch_size, patch_size, N_BANDS), dtype=np.uint8)
    cell_crs = str(geobox.crs)
    t = geobox.transform

    # Band 0 — buildings
    building_rows = _query_theme(con, s3_base, "buildings", "building", "geometry, subtype", xmin, ymin, xmax, ymax)
    bands[:, :, 0] = _rasterize_polygon_band(building_rows, OVERTURE_BUILDING_USE_REV, t, cell_crs, patch_size)

    # Band 1 — land use
    land_use_rows = _query_theme(con, s3_base, "base", "land_use", "geometry, class", xmin, ymin, xmax, ymax)
    bands[:, :, 1] = _rasterize_polygon_band(
        land_use_rows,
        OVERTURE_LAND_USE_REV,
        t,
        cell_crs,
        patch_size,
        fallback_code=OVERTURE_LAND_USE_REV.get("other", 0),
    )

    # Band 2 — roads (buffered)
    road_rows = _query_theme(
        con,
        s3_base,
        "transportation",
        "segment",
        "geometry, class, subtype",
        xmin,
        ymin,
        xmax,
        ymax,
        extra_where="subtype IN ('road', 'rail')",
    )
    bands[:, :, 2] = _rasterize_roads(road_rows, t, cell_crs, patch_size)

    # Band 3 — infrastructure
    infra_rows = _query_theme(con, s3_base, "base", "infrastructure", "geometry, class", xmin, ymin, xmax, ymax)
    bands[:, :, 3] = _rasterize_polygon_band(
        infra_rows,
        OVERTURE_INFRASTRUCTURE_REV,
        t,
        cell_crs,
        patch_size,
        fallback_code=OVERTURE_INFRASTRUCTURE_REV.get("other", 0),
    )

    # Band 4 — water + natural land
    water_rows = _query_theme(con, s3_base, "base", "water", "geometry, class, subtype", xmin, ymin, xmax, ymax)
    land_natural_subtypes = "'forest','glacier','sand','bare_rock','shrub','grass'"
    land_rows = _query_theme(
        con,
        s3_base,
        "base",
        "land",
        "geometry, subtype",
        xmin,
        ymin,
        xmax,
        ymax,
        extra_where=f"subtype IN ({land_natural_subtypes})",
    )
    bands[:, :, 4] = _rasterize_water_natural(water_rows, land_rows, t, cell_crs, patch_size)
    return bands


# ---------------------------------------------------------------------------
# Fetcher class
# ---------------------------------------------------------------------------


class OvertureFetcher(TiledRasterFetcher):
    """Fetches Overture Maps built-environment data via DuckDB GeoParquet queries.

    Produces a 5-band uint8 raster per cell (see module docstring for the
    band layout). DuckDB connection + S3 base path are cached on the
    fetcher instance; ``download_to_dense`` reuses a single instance per
    worker process so the caches survive across cells.
    """

    def __init__(self, release: str | None = None):
        self._release = release
        self._s3_base: str | None = None
        # A DuckDB connection must not run two queries at once, and the store
        # fetches tiles from a thread pool: one database, one cursor per thread.
        self._base: duckdb.DuckDBPyConnection | None = None
        self._local = threading.local()
        self._lock = threading.Lock()

    # --- TiledRasterFetcher interface ---

    def tile_key(self, lon: float, lat: float) -> tuple[int, int]:
        return (floor(lon), floor(lat))

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[int, int]]:
        return [self.tile_key(lon, lat)]

    def fetch_tile(self, tile_key):
        return None

    def _get_connection(self):
        con = getattr(self._local, "con", None)
        if con is None:
            with self._lock:  # the database, its extensions and the release are set up once
                if self._base is None:
                    self._base = _create_connection()
                    self._s3_base = _resolve_s3_base(self._base, self._release)
                con = self._local.con = self._base.cursor()
        return con

    def fetch_patches_for_tile(
        self,
        tile_key,
        cells: list[tuple[int, float, float, GeoBox]],
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Run one Overture query per cell, rasterize 5 bands."""
        if not cells:
            return []
        con = self._get_connection()
        patch_size = cells[0][3].shape[0]

        results: list[tuple[int, np.ndarray]] = []
        for cell_idx, lon, lat, geobox in cells:
            try:
                xmin, ymin, xmax, ymax = _geobox_to_wgs84_bbox(geobox)
                patch = _rasterize_cell(
                    con,
                    self._s3_base,
                    geobox,
                    patch_size,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                )
                results.append((cell_idx, patch))
            except Exception:
                logger.warning(
                    "Overture rasterization failed for cell %d (%.4f, %.4f)",
                    cell_idx,
                    lon,
                    lat,
                    exc_info=True,
                )
        return results

    def fetch_patch(
        self,
        lon: float,
        lat: float,
        dst_geobox: GeoBox,
        resampling: Resampling = Resampling.nearest,
        nodata: float | int | None = None,
    ) -> np.ndarray | None:
        """Single-cell convenience shim; goes through :meth:`fetch_patches_for_tile`."""
        results = self.fetch_patches_for_tile(
            self.tile_key(lon, lat),
            [(0, lon, lat, dst_geobox)],
            resampling=resampling,
            nodata=nodata,
        )
        if not results:
            return None
        return results[0][1]

    def __del__(self):
        # Best-effort close of the DuckDB database on garbage collection.
        con = getattr(self, "_base", None)
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

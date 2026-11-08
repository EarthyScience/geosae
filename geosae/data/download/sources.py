"""Config-driven registry of every data product in the MajorTOM dataset.

One :class:`ProductSpec` per store array (the two Earth-embedding products plus
the reference-label registry). Each spec records the store-array name, the
patch schema (shape/dtype/fill), the on-disk resampling, and the data source,
its auth requirement, and the fully-qualified fetcher class
(``geosae.data.download.fetchers.*``) that produces it.

This module is pure data + light Python: importing it needs only numpy, so the
registry can be inspected, filtered and tested with the core install. Actually
*fetching* a product needs the heavy geospatial stack (rasterio, pyproj,
shapely, odc.geo, duckdb/ee/earthaccess/geotessera depending on the product);
:func:`build_fetcher` imports that lazily and raises a clear error listing the
``geosae[download]`` extra if a dependency is missing.

Several specs may write the same store array from different sources -- e.g.
``alphaearth`` (Earth Engine) and ``alphaearth_cog`` (a public COG mirror). A
dataset preset (:mod:`geosae.data.download.presets`) names which spec fills each
of its arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


class Auth(StrEnum):
    """Credential class required to fetch a product."""

    #: Public HTTP / anonymous S3 / public STAC. Reproducible with no secrets.
    PUBLIC = "public"
    #: Google Earth Engine: a GEE project + ``~/.config/earthengine/credentials``.
    GEE = "gee"
    #: NASA Earthdata Login via ``~/.netrc`` (earthaccess).
    EARTHDATA = "earthdata"
    #: CEDA archive registration (used only by the optional CCI-PFT layer).
    CEDA = "ceda"


class Kind(StrEnum):
    EMBEDDING = "embedding"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class Redistribution(StrEnum):
    """What a product's licence allows us to *pass on*, not what it allows us to use.

    Fetching a product needs only the right to use it; publishing a derived
    dataset needs the right to redistribute it. The two differ per product, so
    the flag is what decides whether a layer may be packed into a public release
    (:mod:`geosae.data.download.release`) or has to be re-fetched by the reader.
    """

    #: Public domain or equivalent (CC0): redistribute with no condition.
    OPEN = "open"
    #: Redistribution allowed if the source is credited (CC-BY and equivalents).
    ATTRIBUTION = "attribution"
    #: Redistribution allowed but the derived work inherits the licence
    #: (ODbL, CC-BY-SA, GPL). Packing one such layer would relicense the release.
    SHARE_ALIKE = "share_alike"
    #: Use granted, redistribution not granted or conditioned on permission.
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class ProductSpec:
    """One data product written as a per-cell array in the patch store.

    Attributes:
        key: Registry / CLI key for this product.
        array: Store array name (what the reader opens).
        kind: embedding / categorical / continuous.
        resolution_m: Native resolution in metres (for reference; the patch is
            resampled onto the 10 m, ``patch_size`` cell geobox).
        shape: Per-cell array shape *excluding* the leading cell axis
            (``(H, W)`` single-band or ``(H, W, C)`` multi-band).
        dtype: Store dtype.
        fill_value: Nodata / fill sentinel written where a cell has no data.
        resampling: rasterio resampling name used when the source is warped onto
            the cell geobox (``"nearest"`` for categorical, ``"bilinear"`` for
            continuous).
        source: Human-readable data source + URL/asset id.
        licence: Licence the *data* is published under (SPDX id where one
            exists, else the licence's own name).
        licence_url: Page the licence text was read from.
        redistribution: What that licence allows us to pass on
            (see :class:`Redistribution`).
        auth: Credential class required (see :class:`Auth`).
        fetcher: Fully-qualified fetcher path
            (``module:ClassName``) that produces this product.
        predownloaded: True if the fetcher reads a pre-downloaded local raster
            from a cache dir rather than streaming from the network.
        cache_gb: Size of that one-time download in GB (``None``: streamed, or
            fetched on demand in pieces).
        classes: Number of classes for categorical products, else None.
        bands: Names of the band axis of a multi-band product.
    """

    key: str
    array: str
    kind: Kind
    resolution_m: float
    shape: tuple[int, ...]
    dtype: str
    fill_value: object
    resampling: str
    source: str
    licence: str
    licence_url: str
    redistribution: Redistribution
    auth: Auth
    fetcher: str
    predownloaded: bool = False
    cache_gb: float | None = None
    classes: int | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    bands: tuple[str, ...] = field(default_factory=tuple)
    """Names of the trailing band axis of a multi-band product, in order (empty: single band)."""
    fetcher_kwargs: dict = field(default_factory=dict)
    """Constructor arguments for the fetcher (URLs, year, mode, cache paths).
    ``build_fetcher`` merges these with any per-call overrides. Cache-backed
    fetchers reference :data:`DEFAULT_RASTER_CACHE`."""


#: Local cache root for predownloaded / zip-VRT / Tessera rasters. Override with
#: ``GEOSAE_RASTER_CACHE`` or the ``cache_dir=`` kwarg to :func:`build_fetcher`.
DEFAULT_RASTER_CACHE = Path(os.environ.get("GEOSAE_RASTER_CACHE", "data/raster_cache"))
_CACHE = str(DEFAULT_RASTER_CACHE)


_HANSEN_EE = "UMD/hansen/global_forest_change_2025_v1_13"
_WORLDCEREAL_EE = "ESA/WorldCereal/2021/MODELS/v100"
_RESOLVE = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
_PNV = "https://zenodo.org/records/3631254/files/pnv_potential.landcover_probav.lc100_c_250m_s0..0cm_2017_v0.1.tif"
_CHELSA = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_{}_1981-2010_V.2.1.tif"


def _hw(patch_size: int = 64) -> tuple[int, int]:
    return (patch_size, patch_size)


def _hwc(c: int, patch_size: int = 64) -> tuple[int, int, int]:
    return (patch_size, patch_size, c)


# The registry. ``shape`` uses the default patch_size=64; a different
# ``--patch-size`` rescales H and W identically.
_REGISTRY: dict[str, ProductSpec] = {
    # -- Earth embeddings ---------------------------------------------------
    # DEFAULT AlphaEarth source: Google Earth Engine (high-volume API).
    # ``alphaearth_cog`` below is a documented credential-free FALLBACK (S3 COG
    # mirror), not the default for the ``alphaearth`` array.
    "alphaearth": ProductSpec(
        key="alphaearth",
        array="alphaearth",
        kind=Kind.EMBEDDING,
        resolution_m=10,
        shape=_hwc(64),
        dtype="float32",
        fill_value=np.nan,
        resampling="nearest",
        source="AlphaEarth Foundations, GEE collection GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL "
        "via the Earth Engine high-volume API (ee.data.computePixels).",
        licence="CC-BY-4.0",
        licence_url="https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.alpha_earth_gee:AlphaEarthGEEFetcher",
        fetcher_kwargs={"year": 2024},
    ),
    "alphaearth_cog": ProductSpec(
        key="alphaearth_cog",
        array="alphaearth",
        kind=Kind.EMBEDDING,
        resolution_m=10,
        shape=_hwc(64),
        dtype="float32",
        fill_value=np.nan,
        resampling="nearest",
        source="AlphaEarth Foundations -- documented credential-free FALLBACK to the GEE default: "
        "Source Cooperative S3 COG mirror "
        "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual (index aef_index.parquet). "
        "The COG mirror's native tiling can differ from GEE's grid by up to a nearest-neighbour pixel, "
        "so it is not guaranteed bit-identical to the GEE-built store array.",
        licence="CC-BY-4.0",
        licence_url="https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.alpha_earth_cog:AlphaEarthCOGFetcher",
        fetcher_kwargs={"year": 2024},
    ),
    "tessera": ProductSpec(
        key="tessera",
        array="tessera",
        kind=Kind.EMBEDDING,
        resolution_m=10,
        shape=_hwc(128),
        dtype="float32",
        fill_value=np.nan,
        resampling="nearest",
        source="Tessera v1.0 embeddings (0.1 degree tiles) via the geotessera library, "
        "s3://tessera/tessera/npy/v1 on data.source.coop",
        licence="CC0-1.0",
        licence_url="https://github.com/ucam-eo/tessera",
        redistribution=Redistribution.OPEN,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.tessera_embeddings:TesseraTiledFetcher",
        fetcher_kwargs={"cache_dir": _CACHE + "/tessera", "year": 2024, "dataset_version": "v1"},
    ),
    # -- Raw imagery ---------------------------------------------------------
    "sentinel2_l2a": ProductSpec(
        key="sentinel2_l2a",
        array="s2",
        kind=Kind.CONTINUOUS,
        resolution_m=10,
        shape=_hwc(12),
        dtype="int16",
        fill_value=-32768,
        resampling="bilinear",
        source="Sentinel-2 L2A surface reflectance, Planetary Computer STAC collection "
        "sentinel-2-l2a (one low-cloud scene per cell, 12 bands, reflectance x 10000)",
        licence="Copernicus Sentinel Data licence",
        licence_url="https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.sentinel2:Sentinel2L2AFetcher",
        fetcher_kwargs={"year": 2024},
    ),
    # Published as one vector per 0.25 deg grid node rather than as a raster, so
    # its cell is a single "pixel" (patch_size=1) covering roughly 28 km.
    "copernicusfm": ProductSpec(
        key="copernicusfm",
        array="copernicusfm",
        kind=Kind.EMBEDDING,
        resolution_m=27_750,
        shape=_hwc(768, patch_size=1),
        dtype="float32",
        fill_value=np.nan,
        resampling="nearest",
        source="Copernicus-FM 0.25 deg global grid embeddings (ViT-B, eight Sentinel/Copernicus "
        "modalities averaged), HuggingFace dataset wangyi111/Copernicus-Embed-025deg, file "
        "embeddings_025_v1/embed_map_310k.npz, CC-BY-4.0.",
        licence="CC-BY-4.0",
        licence_url="https://github.com/zhu-xlab/Copernicus-FM",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.copernicus_fm:CopernicusFMGridEmbeddings",
    ),
    # -- Categorical reference products --------------------------------------
    "overture": ProductSpec(
        key="overture",
        array="overture",
        kind=Kind.CATEGORICAL,
        resolution_m=1,
        shape=_hwc(5),
        dtype="uint8",
        fill_value=0,
        resampling="nearest",
        source="Overture Maps (water/roads/land-use/infrastructure/land/buildings) via "
        "DuckDB over s3://overturemaps-us-west-2/release",
        licence="ODbL-1.0 (OpenStreetMap-derived themes)",
        licence_url="https://docs.overturemaps.org/attribution/",
        redistribution=Redistribution.SHARE_ALIKE,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.overture:OvertureFetcher",
    ),
    "lc_esa_wc": ProductSpec(
        key="lc_esa_wc",
        array="lc_esa_wc",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=11,
        source="ESA WorldCover v200 2021, https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/",
        licence="CC-BY-4.0",
        licence_url="https://esa-worldcover.org/en/data-access",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:ESAWorldCoverFetcher",
    ),
    "lc_esri": ProductSpec(
        key="lc_esri",
        array="lc_esri",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=9,
        source="ESRI/Impact Observatory Land Cover 2021, Planetary Computer STAC io-lulc-annual-v02",
        licence="CC-BY-4.0",
        licence_url="https://planetarycomputer.microsoft.com/dataset/io-lulc-annual-v02",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:ESRILulcFetcher",
        fetcher_kwargs={"year": 2021},
    ),
    "worldcereal": ProductSpec(
        key="worldcereal",
        array="worldcereal",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=2,
        source="ESA WorldCereal 2021 temporary crops, Zenodo record 7875105 (zip->VRT)",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/7875105",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:MultiTileZipVRTFetcher",
        fetcher_kwargs={
            "url": "https://zenodo.org/records/7875105/files/WorldCereal_2021_tc-annual_temporarycrops_classification.zip",
            "inner_pattern": "*classification.tif",
            "vrt_basename": "worldcereal_2021_tc_annual_temporarycrops",
            "cache_dir": _CACHE,
            "filename": "WorldCereal_2021_tc-annual_temporarycrops_classification.zip",
        },
        predownloaded=True,
        cache_gb=15.5,
    ),
    "glc_fcs30d": ProductSpec(
        key="glc_fcs30d",
        array="glc_fcs30d",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hw(),
        dtype="uint8",
        fill_value=0,
        resampling="nearest",
        classes=35,
        source="GLC_FCS30D 2022, Zenodo record 8239305 (per-5deg zip->VRT)",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/8239305",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:GLCFCS30DFetcher",
        fetcher_kwargs={"year": 2022, "cache_dir": _CACHE},
        predownloaded=True,
        cache_gb=194.4,
    ),
    "geomorphons": ProductSpec(
        key="geomorphons",
        array="geomorphons",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=10,
        source="GEDTM30 geomorphons, "
        "https://s3.opengeohub.org/global/dtm/v3/geomorphon_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20241230.tif",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/14920357",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:OpenLandMapCOGFetcher",
        fetcher_kwargs={
            "cog_url": "https://s3.opengeohub.org/global/dtm/v3/geomorphon_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20241230.tif"
        },
    ),
    "glance30": ProductSpec(
        key="glance30",
        array="glance30",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hw(),
        dtype="int8",
        fill_value=0,
        resampling="nearest",
        classes=7,
        source="NASA GLanCE30 v001 2019, LP DAAC s3://lp-prod-protected/GLanCE30.001 (earthaccess)",
        licence="CC-BY-4.0 (NASA EOSDIS open data)",
        licence_url="https://www.earthdata.nasa.gov/data/catalog/lpcloud-glance30-001",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.EARTHDATA,
        fetcher="geosae.data.download.fetchers.raster_fetchers:NASAGLanCE30Fetcher",
        fetcher_kwargs={"year": 2019, "cache_dir": _CACHE},
        predownloaded=True,
    ),
    "lcz": ProductSpec(
        key="lcz",
        array="lcz",
        kind=Kind.CATEGORICAL,
        resolution_m=100,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=17,
        source="Local Climate Zones 2018, https://lcz-generator.rub.de/cogs/lcz_filter_v3_cog.tif",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/6364594",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:LCZFetcher",
    ),
    "soilgrids_class": ProductSpec(
        key="soilgrids_class",
        array="soilgrids_class",
        kind=Kind.CATEGORICAL,
        resolution_m=250,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=30,
        source="SoilGrids WRB MostProbable, https://files.isric.org/soilgrids/latest/data/wrb/MostProbable.vrt",
        licence="CC-BY-4.0",
        licence_url="https://docs.isric.org/globaldata/soilgrids/",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:SoilGridsFetcher",
        fetcher_kwargs={"mode": "class"},
    ),
    "lc_esa_cci": ProductSpec(
        key="lc_esa_cci",
        array="lc_esa_cci",
        kind=Kind.CATEGORICAL,
        resolution_m=300,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=37,
        source="ESA CCI Land Cover 2020, Planetary Computer STAC esa-cci-lc",
        licence="ESA CCI licence (rev. 1)",
        licence_url="https://cds.climate.copernicus.eu/licences/satellite-land-cover",
        redistribution=Redistribution.RESTRICTED,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:CCILandCoverFetcher",
        fetcher_kwargs={"year": 2020},
    ),
    # -- Continuous reference products ----------------------------------------
    "chmv2": ProductSpec(
        key="chmv2",
        array="chmv2",
        kind=Kind.CONTINUOUS,
        resolution_m=10,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="CHMv2 canopy height (Meta), s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm",
        licence="CC-BY-4.0",
        licence_url="https://registry.opendata.aws/dataforgood-fb-forestsv2/",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:CHMv2Fetcher",
    ),
    "dem_copernicus": ProductSpec(
        key="dem_copernicus",
        array="dem_cop30",
        kind=Kind.CONTINUOUS,
        resolution_m=30,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="Copernicus DEM GLO-30, https://copernicus-dem-30m.s3.amazonaws.com/",
        licence="Copernicus WorldDEM-30 free licence",
        licence_url="https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/license/License-COPDEM-30.pdf",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:CopernicusDEMFetcher",
    ),
    "gpp": ProductSpec(
        key="gpp",
        array="gpp",
        kind=Kind.CONTINUOUS,
        resolution_m=30,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="Global Pasture Watch grassland GPP 2022, "
        "https://s3.openlandmap.org/arco/gpw_gpp.grass_lue.model_m_30m_s_20220101_20221231_go_epsg.4326_v1.vrt",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/15675358",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:OpenLandMapCOGFetcher",
        fetcher_kwargs={
            "cog_url": "https://s3.openlandmap.org/arco/gpw_gpp.grass_lue.model_m_30m_s_20220101_20221231_go_epsg.4326_v1.vrt",
            "scale": 1.0 / 0.86,
            "src_nodata": 65000,
        },
    ),
    # -- Extra layers available in the registry (not in the majortom210k default) --
    "lc_modis_igbp": ProductSpec(
        key="lc_modis_igbp",
        array="lc_modis_igbp",
        kind=Kind.CATEGORICAL,
        resolution_m=500,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=17,
        source="MODIS MCD12Q1 v061 IGBP 2021 (used for the AOA per-PFT split & concept groups), "
        "https://s3.openlandmap.org/arco/lc_mcd12q1v061.t1_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif",
        licence="Public domain (NASA LP DAAC)",
        licence_url="https://www.earthdata.nasa.gov/engage/open-data-services-software-policies",
        redistribution=Redistribution.OPEN,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:PredownloadedCOGFetcher",
        fetcher_kwargs={
            "url": "https://s3.openlandmap.org/arco/lc_mcd12q1v061.t1_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif",
            "cache_dir": _CACHE,
            "filename": "modis_igbp_2021.tif",
        },
        predownloaded=True,
        cache_gb=0.13,
    ),
    "soilgrids": ProductSpec(
        key="soilgrids",
        array="soilgrids",
        kind=Kind.CONTINUOUS,
        resolution_m=250,
        shape=_hwc(4),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="SoilGrids properties (sand/clay/soc/phh2o 0-5cm), https://files.isric.org/soilgrids/latest/data",
        licence="CC-BY-4.0",
        licence_url="https://docs.isric.org/globaldata/soilgrids/",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:SoilGridsFetcher",
        fetcher_kwargs={"mode": "properties"},
    ),
    "dem_gedtm30": ProductSpec(
        key="dem_gedtm30",
        array="dem_gedtm30",
        kind=Kind.CONTINUOUS,
        resolution_m=30,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="GEDTM30 DEM, https://s3.opengeohub.org/global/edtm/legendtm_rf_30m_m_s_20000101_20231231_go_epsg.4326_v20250130.tif",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/14900181",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:OpenLandMapCOGFetcher",
        fetcher_kwargs={
            "cog_url": "https://s3.opengeohub.org/global/edtm/legendtm_rf_30m_m_s_20000101_20231231_go_epsg.4326_v20250130.tif",
            "scale": 10.0,
            "src_nodata": -2147483648,  # INT32_MIN, as the COG header declares it
        },
    ),
    "gpw_grassland_class": ProductSpec(
        key="gpw_grassland_class",
        array="gpw_grassland_class",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=3,
        source="Global Pasture Watch dominant grassland class 2022 (0 other, 1 cultivated, 2 natural/semi-natural), "
        "https://s3.opengeohub.org/gpw/arco/gpw_grassland_rf.savgol.bthr_c_30m_20220101_20221231_go_epsg.4326_v1.tif",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/13890401",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:OpenLandMapCOGFetcher",
        fetcher_kwargs={
            "cog_url": "https://s3.opengeohub.org/gpw/arco/"
            "gpw_grassland_rf.savgol.bthr_c_30m_20220101_20221231_go_epsg.4326_v1.tif",
        },
    ),
    "natural_forest": ProductSpec(
        key="natural_forest",
        array="natural_forest",
        kind=Kind.CONTINUOUS,
        resolution_m=10,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="Natural forests of the world 2020, probability of natural forest (Neumann et al. 2025), "
        "Figshare+ https://doi.org/10.25452/figshare.plus.28881731.v2 (per-UTM-zone zips, read by range)",
        licence="CC-BY-4.0",
        licence_url="https://doi.org/10.25452/figshare.plus.28881731.v2",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.natural_forest:NaturalForestFetcher",
    ),
    # -- Reference layers sampled from Earth Engine (GeeImageFetcher) and CHELSA --
    "lc_esri_gee": ProductSpec(
        key="lc_esri_gee",
        array="lc_esri_gee",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=9,
        source="Esri / Impact Observatory 10 m annual LULC 2017-2025 (one year per fetch, default 2024), "
        "Earth Engine community catalog projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
        licence="CC-BY-4.0",
        licence_url="https://www.arcgis.com/home/item.html?id=cfcb7609de5f478eb7666240902d4d3d",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={
            "layers": [
                {
                    "asset": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
                    "band": "b1",
                    "collection": True,
                }
            ],
            "year": 2024,
        },
    ),
    "soilgrids_gee": ProductSpec(
        key="soilgrids_gee",
        array="soilgrids_gee",
        kind=Kind.CONTINUOUS,
        resolution_m=250,
        shape=_hwc(4),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        bands=("sand", "clay", "soc", "phh2o"),
        source="SoilGrids 2.0 mean 0-5 cm: sand and clay (%), SOC (g/kg), pH(H2O); the mapped units divided by 10, "
        "as `soilgrids`. Earth Engine projects/soilgrids-isric/{sand,clay,soc,phh2o}_mean",
        licence="CC-BY-4.0",
        licence_url="https://docs.isric.org/globaldata/soilgrids/",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={
            "layers": [
                {"asset": f"projects/soilgrids-isric/{p}_mean", "band": f"{p}_0-5cm_mean", "scale": 10.0}
                for p in ("sand", "clay", "soc", "phh2o")
            ]
        },
    ),
    "glc_fcs30d_gee": ProductSpec(
        key="glc_fcs30d_gee",
        array="glc_fcs30d",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hw(),
        dtype="uint8",
        fill_value=0,
        resampling="nearest",
        classes=35,
        source="GLC_FCS30D 2022 (band b23 of the 2000-2022 annual maps), Earth Engine community catalog "
        "projects/sat-io/open-datasets/GLC-FCS30D/annual (copy of Zenodo record 8239305)",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/8239305",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={
            "layers": [{"asset": "projects/sat-io/open-datasets/GLC-FCS30D/annual", "band": "b23", "collection": True}]
        },
    ),
    "worldcereal_gee": ProductSpec(
        key="worldcereal_gee",
        array="worldcereal",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=2,
        source="ESA WorldCereal 2021 v100 temporary crops (0 / 100), the same maps as Zenodo record 7875105, "
        f"Earth Engine {_WORLDCEREAL_EE}",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/7875105",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={
            "layers": [
                {
                    "asset": _WORLDCEREAL_EE,
                    "band": "classification",
                    "collection": True,
                    "where": {"product": "temporarycrops"},
                }
            ]
        },
    ),
    "worldcereal_types": ProductSpec(
        key="worldcereal_types",
        array="worldcereal_types",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hwc(5),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=2,
        bands=("temporarycrops", "maize", "wintercereals", "springcereals", "irrigation"),
        source=f"ESA WorldCereal 2021 v100 classification (0 / 100 per product), Earth Engine {_WORLDCEREAL_EE}",
        licence="CC-BY-4.0",
        licence_url="https://zenodo.org/records/7875105",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={
            "layers": [
                {
                    "asset": _WORLDCEREAL_EE,
                    "band": "classification",
                    "collection": True,
                    "where": {"product": p},
                }
                for p in ("temporarycrops", "maize", "wintercereals", "springcereals", "irrigation")
            ]
        },
    ),
    "hansen_lossyear": ProductSpec(
        key="hansen_lossyear",
        array="hansen_lossyear",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hwc(2),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=26,
        bands=("lossyear", "gain"),
        source="Hansen Global Forest Change v1.13 (2000-2025): lossyear (0 none, k = 2000 + k) and gain "
        "(2000-2012), Earth Engine " + _HANSEN_EE,
        licence="CC-BY-4.0",
        licence_url="https://glad.earthengine.app/view/global-forest-change",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={"layers": [{"asset": _HANSEN_EE, "band": b} for b in ("lossyear", "gain")]},
    ),
    "hansen_treecover2000": ProductSpec(
        key="hansen_treecover2000",
        array="hansen_treecover2000",
        kind=Kind.CONTINUOUS,
        resolution_m=30,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="Hansen Global Forest Change v1.13: tree canopy cover in 2000 (%), Earth Engine " + _HANSEN_EE,
        licence="CC-BY-4.0",
        licence_url="https://glad.earthengine.app/view/global-forest-change",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={"layers": [{"asset": _HANSEN_EE, "band": "treecover2000"}]},
    ),
    "jrc_gsw": ProductSpec(
        key="jrc_gsw",
        array="jrc_gsw",
        kind=Kind.CONTINUOUS,
        resolution_m=30,
        shape=_hwc(3),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        bands=("occurrence", "seasonality", "recurrence"),
        source="JRC Global Surface Water v1.4 (1984-2021): occurrence (%), seasonality (months of water), "
        "recurrence (%); 0 where no water was ever mapped. Earth Engine JRC/GSW1_4/GlobalSurfaceWater",
        licence="Copernicus programme, free and without restriction of use (cite Pekel et al. 2016)",
        licence_url="https://global-surface-water.appspot.com/download",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={
            "layers": [
                {"asset": "JRC/GSW1_4/GlobalSurfaceWater", "band": b, "unmask": 0}
                for b in ("occurrence", "seasonality", "recurrence")
            ]
        },
    ),
    "ghsl_built_s": ProductSpec(
        key="ghsl_built_s",
        array="ghsl_built_s",
        kind=Kind.CONTINUOUS,
        resolution_m=10,
        shape=_hw(),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        source="GHS-BUILT-S R2023A, built-up surface 2018 (m² per 10 m pixel), "
        "Earth Engine JRC/GHSL/P2023A/GHS_BUILT_S_10m/2018",
        licence="CC-BY-4.0 (European Commission reuse notice, Decision 2011/833/EU)",
        licence_url="https://human-settlement.emergency.copernicus.eu/legal.php",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={"layers": [{"asset": "JRC/GHSL/P2023A/GHS_BUILT_S_10m/2018", "band": "built_surface"}]},
    ),
    "usda_cdl": ProductSpec(
        key="usda_cdl",
        array="usda_cdl",
        kind=Kind.CATEGORICAL,
        resolution_m=30,
        shape=_hw(),
        dtype="uint8",
        fill_value=0,
        resampling="nearest",
        classes=256,
        source="USDA NASS Cropland Data Layer 2021 (CONUS; 0 outside), Earth Engine USDA/NASS/CDL/2021",
        licence="Public domain (USDA NASS)",
        licence_url="https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php",
        redistribution=Redistribution.OPEN,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={"layers": [{"asset": "USDA/NASS/CDL/2021", "band": "cropland"}]},
    ),
    "eucropmap": ProductSpec(
        key="eucropmap",
        array="eucropmap",
        kind=Kind.CATEGORICAL,
        resolution_m=10,
        shape=_hw(),
        dtype="uint16",
        fill_value=0,
        resampling="nearest",
        classes=20,
        source="JRC EUCROPMAP 2022, EU27 (codes 100-900; 0 outside), Earth Engine JRC/D5/EUCROPMAP/V1 EU27_2022",
        licence="CC-BY-4.0",
        licence_url="https://data.jrc.ec.europa.eu/dataset/555e7c2a-6a88-4e5a-9e45-7fd7a6c5ec5b",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.GEE,
        fetcher="geosae.data.download.fetchers.gee_image:GeeImageFetcher",
        fetcher_kwargs={"layers": [{"asset": "JRC/D5/EUCROPMAP/V1/EU27_2022", "band": "classification"}]},
    ),
    "chelsa_kg": ProductSpec(
        key="chelsa_kg",
        array="chelsa_kg",
        kind=Kind.CATEGORICAL,
        resolution_m=1000,
        shape=_hw(),
        dtype="uint8",
        fill_value=0,
        resampling="nearest",
        classes=31,
        source="CHELSA V2.1 Köppen-Geiger (kg2, after Peel et al. 2007), 1981-2010, " + _CHELSA.format("kg2"),
        licence="CC0-1.0",
        licence_url="https://www.envidat.ch/dataset/chelsa-climatologies",
        redistribution=Redistribution.OPEN,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:PredownloadedCOGFetcher",
        fetcher_kwargs={"url": _CHELSA.format("kg2"), "cache_dir": _CACHE},
        predownloaded=True,
        cache_gb=0.013,
    ),
    "chelsa_bioclim": ProductSpec(
        key="chelsa_bioclim",
        array="chelsa_bioclim",
        kind=Kind.CONTINUOUS,
        resolution_m=1000,
        shape=_hwc(19),
        dtype="float32",
        fill_value=np.nan,
        resampling="bilinear",
        bands=tuple(f"bio{v}" for v in range(1, 20)),
        source="CHELSA V2.1 bioclim bio1-bio19, 1981-2010 (scale / offset of each file applied), "
        "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/",
        licence="CC0-1.0",
        licence_url="https://www.envidat.ch/dataset/chelsa-climatologies",
        redistribution=Redistribution.OPEN,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:PredownloadedStackFetcher",
        fetcher_kwargs={
            "urls": [
                "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_{}_1981-2010_V.2.1.tif".format(
                    f"bio{v}"
                )
                for v in range(1, 20)
            ],
            "cache_dir": _CACHE,
            "src_nodata": 65535,
        },
        predownloaded=True,
        cache_gb=4.83,
    ),
    # -- Reference partitions of the sub-concept analysis --------------------
    "resolve_biomes": ProductSpec(
        key="resolve_biomes",
        array="resolve_biomes",
        kind=Kind.CATEGORICAL,
        resolution_m=1000,
        shape=_hw(),
        dtype="uint8",
        fill_value=0,
        resampling="nearest",
        classes=14,
        source="RESOLVE Ecoregions 2017 (Dinerstein et al. 2017), biome number BIOME_NUM 1-14, polygons rasterised "
        "onto each cell, " + _RESOLVE,
        licence="CC-BY-4.0",
        licence_url="https://ecoregions.appspot.com/",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.vector:VectorZipFetcher",
        fetcher_kwargs={"url": _RESOLVE, "attribute": "BIOME_NUM", "valid": (1, 14), "cache_dir": _CACHE},
        predownloaded=True,
        cache_gb=0.15,
    ),
    "resolve_ecoregions": ProductSpec(
        key="resolve_ecoregions",
        array="resolve_ecoregions",
        kind=Kind.CATEGORICAL,
        resolution_m=1000,
        shape=_hw(),
        dtype="uint16",
        fill_value=0,
        resampling="nearest",
        classes=846,
        source="RESOLVE Ecoregions 2017 (Dinerstein et al. 2017), ecoregion id ECO_ID, polygons rasterised onto "
        "each cell, " + _RESOLVE,
        licence="CC-BY-4.0",
        licence_url="https://ecoregions.appspot.com/",
        redistribution=Redistribution.ATTRIBUTION,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.vector:VectorZipFetcher",
        fetcher_kwargs={"url": _RESOLVE, "attribute": "ECO_ID", "valid": (1, 65534), "cache_dir": _CACHE},
        predownloaded=True,
        cache_gb=0.15,
    ),
    "pnv": ProductSpec(
        key="pnv",
        array="pnv",
        kind=Kind.CATEGORICAL,
        resolution_m=250,
        shape=_hw(),
        dtype="uint8",
        fill_value=255,
        resampling="nearest",
        classes=25,
        source="Potential natural vegetation, dominant land-cover class on the Copernicus LC100 legend "
        "(Hengl et al. 2018), Zenodo record 3631254, " + _PNV,
        licence="CC-BY-SA-4.0",
        licence_url="https://zenodo.org/records/3631254",
        redistribution=Redistribution.SHARE_ALIKE,
        auth=Auth.PUBLIC,
        fetcher="geosae.data.download.fetchers.raster_fetchers:OpenLandMapCOGFetcher",
        fetcher_kwargs={"cog_url": _PNV},
    ),
}


def all_products() -> dict[str, ProductSpec]:
    """Return a copy of the full product registry keyed by product key."""
    return dict(_REGISTRY)


def get_product(key: str) -> ProductSpec:
    """Look up a :class:`ProductSpec` by key (accepts aliases)."""
    if key in _REGISTRY:
        return _REGISTRY[key]
    for spec in _REGISTRY.values():
        if key in spec.aliases:
            return spec
    raise KeyError(f"unknown product {key!r}; known: {sorted(_REGISTRY)}")


def build_fetcher(key: str, **kwargs):
    """Instantiate the fetcher for ``key`` (needs the ``download`` extra).

    The product's :attr:`ProductSpec.fetcher_kwargs` (source URLs / year / mode /
    cache paths) are applied first; ``kwargs`` override them per call (e.g.
    ``cache_dir=`` or, for ``alphaearth``, ``gee_project=``).

    Args:
        key: Product key.
        **kwargs: Per-call overrides merged over ``spec.fetcher_kwargs``.

    Raises:
        ImportError: If the geospatial download stack is not installed. Install
            with ``pip install 'geosae[download]'``.
    """
    import importlib

    spec = get_product(key)
    module_path, cls_name = spec.fetcher.split(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"fetching {key!r} needs the geospatial download stack "
            f"({module_path}). Install it with: pip install 'geosae[download]'. "
            f"Original error: {exc}"
        ) from exc
    merged = {**spec.fetcher_kwargs, **kwargs}
    return getattr(module, cls_name)(**merged)


__all__ = [
    "Auth",
    "Kind",
    "ProductSpec",
    "all_products",
    "get_product",
    "build_fetcher",
]

"""Natural-forest fetcher: UTM-band tile naming, the per-band tile grid, and patch reads from a zone zip.

A local zip stands in for a Figshare+ zone archive, so the tests need no network.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the download extra

import rasterio  # noqa: E402
from odc.geo.geobox import GeoBox  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from geosae.data.download.fetchers.natural_forest import NaturalForestFetcher, utm_band  # noqa: E402

CRS = "EPSG:32632"  # zone 32, northern hemisphere; easting 500 km is the 9 deg E meridian
X0, YTOP, STEP = 500000.0, 6000000.0, 640.0  # a 2 x 2 grid of 640 m tiles near 54.1 N (band U)
VALUES = {(0, 0): 50, (0, 1): 100, (1, 0): 150, (1, 1): 255}  # scaled by 250; 255 is nodata


def _tile_bytes(tmp_path, i: int, j: int, value: int) -> bytes:
    path = tmp_path / f"t{i}{j}.tif"
    profile = dict(driver="GTiff", height=64, width=64, count=1, dtype="uint8", crs=CRS, nodata=255,
                   transform=from_origin(X0 + j * STEP, YTOP - i * STEP, 10.0, 10.0))  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, 64, 64), value, dtype="uint8"))
    return path.read_bytes()


@pytest.fixture(scope="module")
def fetcher(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zone")
    zpath = tmp / "zone_32.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for (i, j), v in VALUES.items():
            z.writestr(f"tmp/zip_inference/raster_32U_10_{i}_{j}.tif", _tile_bytes(tmp, i, j, v))
    return NaturalForestFetcher(zones={32: lambda: open(zpath, "rb")})


def _geobox(left: float, top: float, n: int = 8) -> GeoBox:
    return GeoBox((n, n), from_origin(left, top, 10.0, 10.0), CRS)


def _lonlat(x: float, y: float) -> tuple[float, float]:
    from pyproj import Transformer

    return Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)


def test_utm_band_matches_known_places():
    assert utm_band(11.58, 48.14) == "32U"  # Munich
    assert utm_band(151.21, -33.87) == "56H"  # Sydney
    assert utm_band(-0.13, 51.51) == "30U"  # London
    assert utm_band(20.0, 80.0) == "34X"  # the 12-degree band X
    assert utm_band(180.0, 0.0) == "60N"  # the antimeridian closes zone 60


def test_a_patch_inside_one_tile_reads_that_tiles_probability(fetcher):
    left, top = X0 + STEP + 100.0, YTOP - 100.0  # inside tile (0, 1)
    lon, lat = _lonlat(left + 40.0, top - 40.0)
    out = fetcher.fetch_patch(lon, lat, _geobox(left, top))
    assert out is not None
    np.testing.assert_allclose(out, 100 / 250)


def test_a_patch_straddling_two_tiles_is_composited_from_both(fetcher):
    left, top = X0 + STEP - 40.0, YTOP - 200.0  # 4 px in tile (0, 0), 4 px in tile (0, 1)
    lon, lat = _lonlat(left + 40.0, top - 40.0)
    out = fetcher.fetch_patch(lon, lat, _geobox(left, top))
    np.testing.assert_allclose(out[:, :4], 50 / 250)
    np.testing.assert_allclose(out[:, 4:], 100 / 250)


def test_nodata_reads_as_nan_not_as_zero_probability(fetcher):
    left, top = X0 + STEP + 100.0, YTOP - STEP - 100.0  # inside tile (1, 1), all nodata
    lon, lat = _lonlat(left + 40.0, top - 40.0)
    out = fetcher.fetch_patch(lon, lat, _geobox(left, top))
    assert out is None or np.isnan(out).all()


def test_a_place_the_archive_has_no_tile_for_yields_nothing(fetcher):
    lon, lat = _lonlat(X0 + 20 * STEP, YTOP - 20 * STEP)  # same band, far outside the 2 x 2 grid
    assert fetcher.candidate_tiles(lon, lat) == []
    assert fetcher.fetch_patch(lon, lat, _geobox(X0 + 20 * STEP, YTOP - 20 * STEP)) is None

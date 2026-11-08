"""The AlphaEarth COG fetcher: int8 codes with their own fill become float embeddings with NaN gaps."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from geosae.data.download.fetchers.alpha_earth_cog import AlphaEarthCOGFetcher  # noqa: E402
from geosae.data.download.store import utm_geobox  # noqa: E402

LON0, LAT0 = 11.0, 50.0  # the raster's south-west corner, in UTM 32N
SIZE = 1200  # 12 km of 10 m pixels


@pytest.fixture(scope="module")
def fetcher(tmp_path_factory):
    """A one-tile AEF archive: band b holds code b - 32, with a nodata column in the middle.

    The rows run south to north (positive y step), as in the published COGs.
    """
    from rasterio.warp import transform

    tmp = tmp_path_factory.mktemp("aef")
    (x0,), (y0,) = transform("EPSG:4326", "EPSG:32632", [LON0], [LAT0])
    codes = np.broadcast_to((np.arange(64, dtype=np.int8) - 32)[:, None, None], (64, SIZE, SIZE)).copy()
    codes[:, :, SIZE // 2] = -128
    tif = tmp / "tile.tif"
    profile = dict(driver="GTiff", width=SIZE, height=SIZE, count=64, dtype="int8", crs="EPSG:32632", nodata=-128)
    with rasterio.open(tif, "w", transform=Affine(10, 0, x0, 0, 10, y0), **profile) as dst:
        dst.write(codes)
    index = gpd.GeoDataFrame(
        # the index box reaches past the raster, as the lon/lat box of a UTM tile does
        {"year": [2024], "path": [str(tif)], "wgs84_west": [LON0], "wgs84_east": [LON0 + 0.3]}
        | {"wgs84_south": [LAT0], "wgs84_north": [LAT0 + 0.1]},
        geometry=[shapely.box(LON0, LAT0, LON0 + 0.3, LAT0 + 0.1)],
        crs="EPSG:4326",
    )
    index.to_parquet(tmp / "index.parquet")
    return AlphaEarthCOGFetcher(year=2024, index_cache=tmp / "index.parquet")


def _fetch(fetcher, points):
    cells = [(i, lon, lat, utm_geobox(lon, lat)[0]) for i, (lon, lat) in enumerate(points)]
    return dict(fetcher.fetch_patches_for_tile(fetcher.tile_key(*points[0]), cells, nodata=np.nan))


def test_codes_are_dequantised_and_their_fill_becomes_nan(fetcher):
    centre_lon = LON0 + SIZE * 10 / 2 / (111_320 * np.cos(np.radians(LAT0 + 0.05)))
    patch = _fetch(fetcher, [(centre_lon, LAT0 + 0.05)])[0]
    assert patch.shape == (64, 64, 64) and patch.dtype == np.float32
    gap = np.isnan(patch).all(-1)
    assert 0 < gap.sum() < 64 * 64 and not np.isnan(patch[~gap]).any()
    expected = np.sign(np.arange(64) - 32) * ((np.arange(64) - 32) / 127.5) ** 2
    assert np.allclose(patch[~gap][0], expected)


def test_far_apart_cells_of_one_tile_read_as_if_alone(fetcher):
    points = [(LON0 + 0.02, LAT0 + 0.02), (LON0 + 0.13, LAT0 + 0.08)]
    together = _fetch(fetcher, points)
    alone = [_fetch(fetcher, [p])[0] for p in points]
    assert set(together) == {0, 1}
    for i in range(2):
        assert np.array_equal(together[i], alone[i], equal_nan=True)


def test_a_cell_off_the_raster_does_not_cost_its_tile_mates(fetcher):
    points = [(LON0 + 0.02, LAT0 + 0.02), (LON0 + 0.13, LAT0 + 0.08), (LON0 + 0.27, LAT0 + 0.05)]
    got = _fetch(fetcher, points)
    assert {0, 1} <= set(got)
    assert 2 not in got or np.isnan(got[2]).all()

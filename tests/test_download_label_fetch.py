"""Label fetching for a sparse cell sample: per-cell work units, tile compositing, Earth Engine and stacked rasters."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from geosae.data.download.sources import all_products

pytest.importorskip("rasterio")  # the download extra
pytest.importorskip("odc.geo")

from affine import Affine  # noqa: E402
from odc.geo.geobox import GeoBox  # noqa: E402


def _utm_geobox(x0=500000.0, y0=5600000.0, size=4, res=10.0):
    return GeoBox((size, size), Affine(res, 0, x0, 0, -res, y0), "EPSG:32632")


def test_every_band_named_product_names_each_band():
    for spec in all_products().values():
        if spec.bands:
            assert len(spec.shape) == 3 and spec.shape[2] == len(spec.bands), spec.key


def _label_group(tmp_path, n_cells=3, patch=4):
    import zarr

    group = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w")
    group.attrs.update({"n_cells": n_cells, "patch_size": patch})
    return group


class _UnitSizeFetcher:
    """Writes into every cell the number of cells its work unit carried."""

    def tile_key(self, lon, lat):
        return "one-tile"

    def fetch_patches_for_tile(self, tile_key, cells, resampling, nodata):
        return [(c[0], np.full((4, 4), float(len(cells)), np.float32)) for c in cells]


@pytest.mark.parametrize("group_by_tile, expected", [(True, 3.0), (False, 1.0)])
def test_cells_of_one_tile_are_fetched_together_or_alone(tmp_path, monkeypatch, group_by_tile, expected):
    from geosae.data.download.store import add_product_array, populate_products

    monkeypatch.setattr("geosae.data.download.sources.build_fetcher", lambda key, **kw: _UnitSizeFetcher())
    group = _label_group(tmp_path)
    add_product_array(group, "dem_copernicus")
    gbs = {r: _utm_geobox() for r in range(3)}
    populate_products(group, ["dem_copernicus"], [0, 1, 2], [9.0] * 3, [50.0] * 3, gbs, group_by_tile=group_by_tile)
    assert (np.asarray(group["dem_cop30"][:]) == expected).all()
    assert np.asarray(group["mask_dem_cop30"][:]).all()


def test_a_windowed_worldcover_read_composites_a_cell_across_the_tile_edge(monkeypatch):
    """A cell straddling two 3° tiles gets each half from its own tile, even if the primary has none of it."""
    from geosae.data.download.fetchers import raster_fetchers as R

    fetcher = R.ESAWorldCoverFetcher(read="window")
    lon, lat = 2.995, 10.0  # 500 m west of the 3° edge
    primary = fetcher.tile_key(lon, lat)
    assert len(fetcher.candidate_tiles(lon, lat)) > 1
    left = np.full((4, 4), 10, np.uint8)
    left[:, 2:] = 255
    right = np.full((4, 4), 80, np.uint8)
    right[:, :2] = 255
    urls = {fetcher._tile_url(primary): left, fetcher._tile_url((3, 9)): right}

    def fake_extract(url, cells, resampling, nodata, **kw):
        patch = urls.get(url)
        return [] if patch is None else [(c[0], patch.copy()) for c in cells]

    monkeypatch.setattr(R, "_open_cog_and_extract", fake_extract)
    cells = [(0, lon, lat, _utm_geobox()), (1, 3.005, 10.0, _utm_geobox())]
    out = dict(fetcher.fetch_patches_for_tile(primary, cells, nodata=255))
    assert (out[0][:, :2] == 10).all() and (out[0][:, 2:] == 80).all()

    urls[fetcher._tile_url(primary)] = None  # primary has nothing: the neighbour alone fills the cell
    out = dict(fetcher.fetch_patches_for_tile(primary, cells[:1], nodata=255))
    assert (out[0][:, 2:] == 80).all()


class _FakeImage:
    def __init__(self, name="img"):
        self.name = name

    def __getattr__(self, attr):
        return lambda *a, **k: self


def _fake_ee(requests, values, masks):
    ee = types.ModuleType("ee")
    ee.Initialize = lambda **kw: None
    ee.Image = lambda *a, **k: _FakeImage()
    ee.Image.cat = lambda bands: _FakeImage("cat")
    ee.Image.constant = lambda v: _FakeImage()
    ee.ImageCollection = lambda *a, **k: _FakeImage()
    ee.Filter = types.SimpleNamespace(eq=lambda p, v: None)

    def compute_pixels(request):
        requests.append(request)
        h, w = request["grid"]["dimensions"]["height"], request["grid"]["dimensions"]["width"]
        n = len(values)
        dtype = [(f"{k}{i}", "f8" if k == "v" else "u1") for i in range(n) for k in "vm"]
        out = np.zeros((h, w), dtype=dtype)
        for i in range(n):
            out[f"v{i}"] = values[i]
            out[f"m{i}"] = masks[i]
        return out

    ee.data = types.SimpleNamespace(computePixels=compute_pixels, HIGH_VOLUME_API_BASE_URL="hv")
    return ee


def test_earth_engine_layers_are_requested_on_the_cell_grid_and_stacked_with_masks_as_fill(monkeypatch):
    from geosae.data.download.fetchers import alpha_earth_gee
    from geosae.data.download.fetchers.gee_image import GeeImageFetcher

    requests = []
    mask0 = np.ones((4, 4), np.uint8)
    mask0[0, 0] = 0
    ee = _fake_ee(requests, values=[np.full((4, 4), 7.0), np.full((4, 4), 3.0)], masks=[mask0, np.ones((4, 4))])
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setattr(alpha_earth_gee, "_build_ee_credentials", lambda: None)
    monkeypatch.setattr(alpha_earth_gee, "_default_gee_project", lambda: "p")

    fetcher = GeeImageFetcher([{"asset": "a", "band": "x"}, {"asset": "b", "band": "y", "collection": True}])
    gb = _utm_geobox(x0=600010.0, y0=5500000.0)
    patch = fetcher.fetch_patch(9.0, 50.0, gb, nodata=255)

    grid = requests[0]["grid"]
    assert grid["crsCode"] == "EPSG:32632"
    assert (grid["affineTransform"]["translateX"], grid["affineTransform"]["translateY"]) == (600010.0, 5500000.0)
    assert (grid["affineTransform"]["scaleX"], grid["affineTransform"]["scaleY"]) == (10.0, -10.0)
    assert patch.shape == (4, 4, 2)
    assert patch[0, 0, 0] == 255 and (patch[..., 0].ravel()[1:] == 7.0).all()
    assert (patch[..., 1] == 3.0).all()
    assert fetcher.bytes_read > 0


def test_a_layer_without_asset_or_band_is_refused(monkeypatch):
    from geosae.data.download.fetchers.gee_image import GeeImageFetcher

    with pytest.raises(ValueError):
        GeeImageFetcher([{"asset": "a"}])
    with pytest.raises(ValueError):
        GeeImageFetcher([{"asset": "a", "band": "x", "colection": True}])


def test_stacked_rasters_apply_each_files_scale_and_offset(tmp_path):
    import rasterio

    from geosae.data.download.fetchers.raster_fetchers import PredownloadedStackFetcher

    gb = _utm_geobox(size=8)
    paths = []
    for i, (scale, offset) in enumerate(((0.1, -273.15), (1.0, 0.0))):
        raw = np.arange(64, dtype=np.uint16).reshape(8, 8) + 2800
        raw[0, 0] = 65535
        p = tmp_path / "src" / f"b{i}.tif"
        p.parent.mkdir(exist_ok=True)
        with rasterio.open(
            p, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint16", crs="EPSG:32632",
            transform=gb.transform,
        ) as dst:  # fmt: skip
            dst.write(raw, 1)
            dst.scales, dst.offsets = (scale,), (offset,)
        paths.append(p)

    fetcher = PredownloadedStackFetcher(
        [p.as_uri() for p in paths], cache_dir=str(tmp_path / "cache"), src_nodata=65535
    )
    out = dict(fetcher.fetch_patches_for_tile("global", [(5, 9.0, 50.0, gb)], nodata=np.nan))
    patch = out[5]
    assert patch.shape == (8, 8, 2)
    assert np.isnan(patch[0, 0]).all()
    np.testing.assert_allclose(patch[1, 1, 0], (2800 + 9) * 0.1 - 273.15, rtol=1e-6)
    np.testing.assert_allclose(patch[1, 1, 1], 2800 + 9)
    assert sorted(x.name for x in (tmp_path / "cache").iterdir()) == ["b0.tif", "b1.tif"]


def test_a_layer_scale_divides_its_values(monkeypatch):
    from geosae.data.download.fetchers import alpha_earth_gee
    from geosae.data.download.fetchers.gee_image import GeeImageFetcher

    ee = _fake_ee([], values=[np.full((4, 4), 250.0), np.full((4, 4), 250.0)], masks=[np.ones((4, 4))] * 2)
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setattr(alpha_earth_gee, "_build_ee_credentials", lambda: None)
    monkeypatch.setattr(alpha_earth_gee, "_default_gee_project", lambda: "p")
    fetcher = GeeImageFetcher([{"asset": "a", "band": "x", "scale": 10.0}, {"asset": "a", "band": "y"}])
    patch = fetcher.fetch_patch(9.0, 50.0, _utm_geobox(), nodata=np.nan)
    assert (patch[..., 0] == 25.0).all() and (patch[..., 1] == 250.0).all()

"""The Esri annual LULC fetcher reads the requested year's map, not the year before."""

from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the download extra
pytest.importorskip("planetary_computer")

from affine import Affine  # noqa: E402
from odc.geo.geobox import GeoBox  # noqa: E402


def test_an_annual_esri_search_keeps_only_the_requested_years_item(monkeypatch):
    """Annual items end on 1 January of the next year, so a year's search also matches the year before."""
    pytest.importorskip("planetary_computer")
    from geosae.data.download.fetchers.raster_fetchers import ESRILulcFetcher

    def item(year):
        return types.SimpleNamespace(
            id=f"32U-{year}",
            bbox=[6.0, 48.0, 12.0, 56.0],
            properties={"start_datetime": f"{year}-01-01T00:00:00Z", "end_datetime": f"{year + 1}-01-01T00:00:00Z"},
            assets={"data": types.SimpleNamespace(href=f"https://x/{year}.tif")},
        )

    catalog = types.SimpleNamespace(search=lambda **kw: types.SimpleNamespace(items=lambda: [item(2022), item(2023)]))
    monkeypatch.setattr("planetary_computer.sign", lambda i: i)
    opened = []

    class _Src:
        def __init__(self, href):
            opened.append(href)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from geosae.data.download.fetchers import raster_fetchers as R

    monkeypatch.setattr("rasterio.open", _Src)
    monkeypatch.setattr(
        R, "_block_read_and_extract", lambda src, cells, *a, **k: [(c[0], np.zeros((4, 4))) for c in cells]
    )
    cells = [(0, 11.5, 50.9, GeoBox((4, 4), Affine(10.0, 0, 500000.0, 0, -10.0, 5600000.0), "EPSG:32632"))]
    for year, expected in ((2023, ["https://x/2023.tif"]), (2024, [])):
        opened.clear()
        fetcher = ESRILulcFetcher(year=year)
        monkeypatch.setattr(fetcher, "_get_catalog", lambda: catalog)
        out = fetcher.fetch_patches_for_tile(fetcher.tile_key(11.5, 50.9), cells)
        assert opened == expected
        assert len(out) == len(expected)

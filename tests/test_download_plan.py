"""Contract tests for build plans, the download CLI's plan mode and the polygon rasteriser."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from geosae.data.download.plan import build_plan, check_credentials, format_plan
from geosae.data.download.presets import MAJORTOM210K


def test_the_plan_counts_rows_and_bytes_per_product():
    records = build_plan(["alphaearth", "lc_esa_wc", "chelsa_bioclim"], 1000, reference_rows=(0, 400))
    by_key = {r["key"]: r for r in records}
    assert by_key["alphaearth"]["rows"] == 1000
    assert by_key["lc_esa_wc"]["rows"] == 400
    assert by_key["alphaearth"]["store_gb"] == pytest.approx(1000 * 64 * 64 * 64 * 4 / 1e9)
    assert by_key["chelsa_bioclim"]["store_gb"] == pytest.approx(400 * 64 * 64 * 19 * 4 / 1e9)
    assert by_key["alphaearth"]["auth"] == "gee" and by_key["lc_esa_wc"]["auth"] == "public"
    assert by_key["chelsa_bioclim"]["cache_gb"] > 1


def test_the_plan_of_a_partial_run_counts_only_its_rows():
    records = build_plan(["alphaearth", "lc_esa_wc"], 1000, reference_rows=(0, 400), rows=np.arange(395, 405))
    assert [r["rows"] for r in records] == [10, 5]


def test_credentials_are_read_off_the_files_the_fetchers_use(tmp_path, monkeypatch):
    monkeypatch.delenv("NETRC", raising=False)
    status = check_credentials(tmp_path)
    assert status["public"] == "ok" and status["gee"] != "ok" and status["earthdata"] != "ok"
    (tmp_path / ".config" / "earthengine").mkdir(parents=True)
    (tmp_path / ".config" / "earthengine" / "credentials").write_text("{}")
    (tmp_path / ".netrc").write_text("machine urs.earthdata.nasa.gov login a password b\n")
    status = check_credentials(tmp_path)
    assert status["gee"] == "ok" and status["earthdata"] == "ok"


def test_plan_mode_fetches_nothing_and_lists_every_product(capsys):
    from geosae.cli import cmd_download

    out = cmd_download({"download": {"plan": True, "preset": "majortom210k", "out": "never-written.zarr"}})
    assert [r["key"] for r in out["plan"]] == list(MAJORTOM210K.products)
    text = capsys.readouterr().out
    assert "resolve_ecoregions" in text and "credentials gee" in text
    assert format_plan(out["plan"]).count("\n") == len(out["plan"]) + 3


def test_polygons_are_burnt_onto_the_cell_by_pixel_centre(tmp_path):
    """West half one class, east half another, and an out-of-range code that becomes fill."""
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    shapely = pytest.importorskip("shapely")
    from geosae.data.download.fetchers.vector import VectorZipFetcher
    from geosae.data.download.store import utm_geobox

    lon, lat = 9.0, 0.0
    geobox, _, _ = utm_geobox(lon, lat, patch_size=8)
    frame = gpd.GeoDataFrame(
        {"BIOME": [4, 7, 98]},
        geometry=[
            shapely.box(lon - 1, lat - 1, lon, lat + 1),
            shapely.box(lon, lat, lon + 1, lat + 1),
            shapely.box(lon, lat - 1, lon + 1, lat),
        ],
        crs="EPSG:4326",
    )
    shp = tmp_path / "layer"
    shp.mkdir()
    frame.to_file(shp / "layer.shp")
    with zipfile.ZipFile(tmp_path / "layer.zip", "w") as zf:
        for f in shp.iterdir():
            zf.write(f, f.name)

    fetcher = VectorZipFetcher("https://example.invalid/layer.zip", "BIOME", valid=(1, 14), cache_dir=str(tmp_path))
    [(row, patch)] = fetcher.fetch_patches_for_tile(fetcher.tile_key(lon, lat), [(5, lon, lat, geobox)], nodata=0)
    assert row == 5 and patch.shape == (8, 8)
    assert (patch[:, :4] == 4).all()
    assert (patch[:4, 4:] == 7).all()
    assert (patch[4:, 4:] == 0).all(), "a code outside the valid range is fill"


def test_missing_credentials_name_the_products_and_the_setup():
    from geosae.data.download.plan import missing_credentials

    status = {"public": "ok", "gee": "missing ~/.config/earthengine/credentials", "earthdata": "ok"}
    missing = missing_credentials(["alphaearth", "alphaearth_cog", "lc_esa_wc", "glance30"], status)
    assert list(missing) == ["alphaearth"]
    assert "earthengine authenticate" in missing["alphaearth"]
    assert missing_credentials(["alphaearth_cog", "lc_esa_wc"], status) == {}


def test_the_download_stage_plans_a_store_of_chosen_cells(capsys):
    from geosae.cli import main

    ids = ["645U_1077R", "1071U_170R", "906U_1248L"]
    main(
        [
            "download",
            "-c",
            "download",
            "-o",
            f"download.cells.ids=[{','.join(ids)}]",
            "-o",
            "download.products=[alphaearth_cog,lc_esa_wc]",
            "-o",
            "download.plan=true",
        ]
    )
    out = capsys.readouterr().out
    assert "3 cells selected" in out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith(("alphaearth_cog", "lc_esa"))}
    assert rows["alphaearth_cog"][3] == "3" and rows["lc_esa_wc"][3] == "3"


def test_the_download_stage_wants_products_for_chosen_cells():
    from geosae.cli import main

    with pytest.raises(ValueError, match="download.products"):
        main(["download", "-c", "download", "-o", "download.cells.ids=[645U_1077R]", "-o", "download.plan=true"])

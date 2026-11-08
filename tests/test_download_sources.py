"""Black-box tests for the product registry (pure data; core install only)."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.data.download.presets import MAJORTOM210K
from geosae.data.download.sources import (
    Auth,
    Kind,
    all_products,
    build_fetcher,
    get_product,
)


def test_alternative_sources_of_one_array_share_its_schema():
    """Specs that fill the same store array must agree on what they write there."""
    by_array: dict = {}
    for spec in all_products().values():
        by_array.setdefault(spec.array, []).append(spec)
    shared = {a: specs for a, specs in by_array.items() if len(specs) > 1}
    assert {"alphaearth", "glc_fcs30d", "worldcereal"} <= set(shared)
    for specs in shared.values():
        assert len({(s.kind, s.shape, s.dtype, s.fill_value) for s in specs}) == 1


def test_registry_keys_consistent():
    for key, spec in all_products().items():
        assert spec.key == key
        assert spec.shape[0] == spec.shape[1]  # square patch
        assert ":" in spec.fetcher  # module:Class form
    assert set(MAJORTOM210K.products) <= set(all_products())


def test_categorical_nearest_continuous_bilinear():
    for spec in all_products().values():
        if spec.kind is Kind.CATEGORICAL:
            assert spec.resampling == "nearest"
        elif spec.kind is Kind.CONTINUOUS:
            assert spec.resampling == "bilinear"


def test_categorical_class_counts():
    # A few known class-count anchors.
    assert get_product("lc_esa_wc").classes == 11
    assert get_product("lc_esri").classes == 9
    assert get_product("worldcereal").classes == 2
    assert get_product("glc_fcs30d").classes == 35
    assert get_product("lc_esa_cci").classes == 37
    assert get_product("soilgrids_class").classes == 30
    assert get_product("gpw_grassland_class").classes == 3


def test_auth_classes():
    assert get_product("alphaearth").auth is Auth.GEE
    assert get_product("alphaearth_cog").auth is Auth.PUBLIC
    assert get_product("glance30").auth is Auth.EARTHDATA
    assert get_product("lc_esa_wc").auth is Auth.PUBLIC


def test_alias_lookup():
    assert get_product("alphaearth_cog").array == "alphaearth"
    with pytest.raises(KeyError):
        get_product("does_not_exist")


def test_fill_values_dtype_sane():
    for spec in all_products().values():
        if spec.dtype.startswith("float"):
            assert np.isnan(spec.fill_value)
        else:
            assert isinstance(spec.fill_value, int)


def test_build_fetcher_without_extra_raises_clearly():
    # In the CPU/core test env the geospatial stack is absent; the error must
    # name the download extra. (If the extra IS installed this returns an object.)
    try:
        obj = build_fetcher("lc_esa_wc")
    except ImportError as exc:
        assert "geosae[download]" in str(exc)
    else:
        assert obj is not None


def test_a_patch_the_source_does_not_cover_comes_back_as_fill():
    """An uncovered destination pixel must be recognisably missing, not zero.

    `reproject` writes only the pixels a source covers and fills the rest with
    zero, which for a DEM is sea level — indistinguishable from data. That is what
    put a hard seam through 1.3% of the Copernicus DEM patches: exactly the cells
    straddling two 1-degree source tiles, where the primary tile covers only part
    of the patch.
    """
    import numpy as np

    pytest.importorskip("rasterio")  # the download extra
    from odc.geo.geobox import GeoBox

    from geosae.data.download.fetchers.reproject import reproject_to_geobox

    src = GeoBox.from_bbox((0.0, 0.0, 1.0, 1.0), "epsg:4326", shape=(64, 64))
    # a destination window half outside the source, so half of it has no coverage
    dst = GeoBox.from_bbox((0.5, 0.0, 1.5, 1.0), "epsg:4326", shape=(64, 64))
    data = np.ones((64, 64), dtype="float32")

    for _ in range(3):
        out = reproject_to_geobox(data, src, dst, nodata=None)
        uncovered = out[:, 32:]
        assert np.isnan(uncovered).all(), "uncovered pixels must not be real numbers"
        assert (out[:, :32] == 1.0).all()

    out = reproject_to_geobox(data, src, dst, nodata=-9999.0)
    assert (out[:, 32:] == -9999.0).all()


def test_refetch_into_a_consolidated_store_leaves_it_readable(tmp_path, monkeypatch):
    """A new array must survive the store's own consolidated listing.

    A store that carries consolidated metadata answers `group[name]` from that
    listing alone, so an array created afterwards is invisible to every ordinary
    reader until the listing is rebuilt. This is the contract that lets a wrong
    product be corrected in place on a store other work depends on.
    """
    import numpy as np
    import zarr

    pytest.importorskip("rasterio")  # the download extra
    from geosae.cli import cmd_refetch

    path = tmp_path / "store.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.attrs["n_cells"] = 4
    group.attrs["patch_size"] = 2
    for name, value in (("lon", [0.5, 1.5, 2.5, 3.5]), ("lat", [0.5, 1.5, 2.5, 3.5])):
        group.create_array(name, shape=(4,), chunks=(4,), dtype="float64")[:] = np.asarray(value)
    group.create_array("mask_dem_cop30", shape=(4,), chunks=(4,), dtype="bool")[:] = True
    zarr.consolidate_metadata(group.store)

    def fake_populate(dest, products, rows, lon, lat, geoboxes, **kw):
        array = kw["into"][products[0]]
        dest[array][rows] = 42.0
        dest[f"mask_{array}"][rows] = True

    monkeypatch.setattr("geosae.data.download.store.populate_products", fake_populate)
    out = cmd_refetch(
        {"refetch": {"zarr_path": str(path), "product": "dem_copernicus", "array": "dem_clean", "rows": "masked"}}
    )
    assert out["covered"] == 4

    reopened = zarr.open_group(str(path), mode="r")  # through the consolidated listing
    assert "dem_clean" in reopened, "the new array must be visible to an ordinary reader"
    assert np.asarray(reopened["dem_clean"][:]).shape == (4, 2, 2)
    assert (np.asarray(reopened["dem_clean"][:]) == 42.0).all()
    assert "dem_copernicus" not in reopened, "the array it replaces is left untouched"
    assert np.asarray(reopened["lon"][:]).tolist() == [0.5, 1.5, 2.5, 3.5]


def test_chmv2_composites_across_a_straddling_quadkey_boundary(monkeypatch):
    """A cell near a z10 tile edge must not carry a canopy-height seam.

    Reading only the primary quadkey would leave the rest of the patch at the
    product's nodata fill -- 0 m, a legitimate canopy-height reading and
    indistinguishable from real ground -- so the neighbouring tile is composited
    in before returning.
    """
    pytest.importorskip("rasterio")
    from odc.geo.geobox import GeoBox

    from geosae.data.download.fetchers.raster_fetchers import CHMv2Fetcher

    fetcher = CHMv2Fetcher()
    # A longitude a few metres east of a z10 quadkey edge (tiles are 360/1024
    # degrees wide), so the cell straddles the primary tile and its neighbour.
    tile_width = 360.0 / (1 << CHMv2Fetcher.ZOOM)
    lon = tile_width - 0.001
    lat = 10.0
    primary = fetcher.tile_key(lon, lat)
    candidates = fetcher.candidate_tiles(lon, lat)
    assert len(candidates) > 1, "the test coordinate must actually straddle a tile"

    gb = GeoBox.from_bbox((lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01), "epsg:4326", shape=(4, 4))
    cells = [(0, lon, lat, gb)]

    # Primary tile covers the left half of the patch, its neighbour the right
    # half -- each NaN where the other has data, mirroring an on-disk read.
    left = np.full((4, 4), 5.0, dtype="float32")
    left[:, 2:] = np.nan
    right = np.full((4, 4), 7.0, dtype="float32")
    right[:, :2] = np.nan

    def fake_read_tile(tile_key, tile_cells, resampling, nodata):
        patch = left if tile_key == primary else right
        return [(c[0], patch.copy()) for c in tile_cells]

    monkeypatch.setattr(fetcher, "_read_tile", fake_read_tile)
    out = dict(fetcher.fetch_patches_for_tile(primary, cells, nodata=None))
    patch = out[0]
    assert not np.isnan(patch).any(), "the neighbour tile must fill what the primary tile leaves NaN"
    assert (patch[:, :2] == 5.0).all()
    assert (patch[:, 2:] == 7.0).all()

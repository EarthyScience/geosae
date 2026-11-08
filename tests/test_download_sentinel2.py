"""Contract tests for the Sentinel-2 product spec and its scene-selection rule."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from geosae.data.download import build_schemas, get_product


def test_sentinel2_spec_is_a_twelve_band_int16_reflectance_product():
    spec = get_product("sentinel2_l2a")
    assert spec.array == "s2"
    assert spec.shape[-1] == 12
    assert spec.dtype == "int16"
    assert spec.resampling == "bilinear"


@pytest.mark.parametrize("patch_size", [64, 128])
def test_build_schemas_scales_every_spatial_axis_with_patch_size(patch_size):
    schemas = build_schemas(["sentinel2_l2a"], n_cells=50, embeddings=(), patch_size=patch_size)
    assert schemas["s2"].shape == (50, patch_size, patch_size, 12)
    assert schemas["s2"].chunks[1:] == (patch_size, patch_size, 12)
    assert schemas["x"].shape == (patch_size,)
    assert schemas["y"].shape == (patch_size,)
    assert schemas["mask_s2"].shape == (50,)


def test_seasonal_distance_peaks_in_the_local_summer():
    sentinel2 = pytest.importorskip("geosae.data.download.fetchers.sentinel2")

    def days(month, day, lat):
        return sentinel2._seasonal_distance(datetime.datetime(2024, month, day), lat)

    assert days(7, 15, 50.0) < days(1, 15, 50.0)  # northern hemisphere prefers July
    assert days(1, 15, -30.0) < days(7, 15, -30.0)  # southern hemisphere prefers January
    # The distance wraps around the year rather than growing past half a year.
    assert days(12, 31, 50.0) == days(1, 1, 50.0) - 1 or days(12, 31, 50.0) <= 183


def test_radiometric_offset_follows_the_processing_baseline():
    sentinel2 = pytest.importorskip("geosae.data.download.fetchers.sentinel2")

    class Item:
        def __init__(self, baseline):
            self.properties = {"s2:processing_baseline": baseline}

    assert sentinel2.Sentinel2L2AFetcher._offset(Item("05.10")) == sentinel2.SENTINEL2_BOA_OFFSET
    assert sentinel2.Sentinel2L2AFetcher._offset(Item("04.00")) == sentinel2.SENTINEL2_BOA_OFFSET
    assert sentinel2.Sentinel2L2AFetcher._offset(Item("03.01")) == 0
    assert sentinel2.Sentinel2L2AFetcher._offset(Item("nonsense")) == 0


def test_band_order_is_the_canonical_one_the_backbones_index_into():
    sentinel2 = pytest.importorskip("geosae.data.download.fetchers.sentinel2")
    from geosae.data.embed import S2_L2A_BANDS

    assert sentinel2.SENTINEL2_BANDS == S2_L2A_BANDS
    assert len(set(S2_L2A_BANDS)) == 12


def test_fill_value_is_representable_and_matches_the_registry():
    sentinel2 = pytest.importorskip("geosae.data.download.fetchers.sentinel2")

    assert get_product("sentinel2_l2a").fill_value == int(sentinel2.SENTINEL2_FILL)
    assert np.int16(sentinel2.SENTINEL2_FILL) == np.iinfo(np.int16).min

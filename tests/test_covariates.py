"""Continuous-covariate registry and reader: band selection, nodata, mask handling."""

import numpy as np
import pytest
import zarr

from geosae.eval.covariates import (
    CONTINUOUS_COVARIATES,
    DEFAULT_COVARIATES,
    Covariate,
    available_covariates,
    read_covariate_stack,
    resolve_covariates,
)


def test_registry_covers_all_19_bioclim_bands():
    bio = [c for c in CONTINUOUS_COVARIATES.values() if c.key == "chelsa_bioclim"]
    assert len(bio) == 19
    assert sorted(c.band for c in bio) == list(range(19))


def test_registry_entries_are_self_consistent():
    for name, cov in CONTINUOUS_COVARIATES.items():
        assert cov.name == name
        assert cov.unit and cov.long_name
        # single-band arrays must not carry a band index, multi-band ones must
        assert (cov.band is None) == (cov.key not in ("chelsa_bioclim", "soilgrids"))


def test_default_set_is_a_subset_and_spans_distinct_sources():
    assert set(DEFAULT_COVARIATES) <= set(CONTINUOUS_COVARIATES)
    keys = {CONTINUOUS_COVARIATES[n].key for n in DEFAULT_COVARIATES}
    # the curated default must not be 8 flavours of the same raster
    assert len(keys) >= 4


def test_resolve_covariates_accepts_names_objects_and_default():
    covs = resolve_covariates(["mat", "elevation"])
    assert [c.name for c in covs] == ["mat", "elevation"]
    assert resolve_covariates(None) == resolve_covariates(DEFAULT_COVARIATES)
    passthrough = [CONTINUOUS_COVARIATES["mat"]]
    assert resolve_covariates(passthrough) == passthrough
    with pytest.raises(KeyError):
        resolve_covariates(["not_a_covariate"])


def _data_store(tmp_path):
    """2 cells, 2x2 px. bioclim band 0 = temperature, dem, chmv2 with a NaN.

    cell 0 is masked-in for every product; cell 1 is masked *out* for ``chmv2``.
    """
    ps = 2
    data = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")

    bio = np.zeros((2, ps, ps, 19), dtype="float32")
    bio[0, :, :, 0] = np.array([[10.0, 20.0], [30.0, 40.0]])
    bio[1, :, :, 0] = 5.0
    bio[..., 11] = 100.0  # annual precip band, constant
    data.create_array("chelsa_bioclim", shape=bio.shape, chunks=bio.shape, dtype="float32")[:] = bio

    dem = np.stack([np.full((ps, ps), 1.0, "float32"), np.full((ps, ps), 2.0, "float32")])
    data.create_array("dem_cop30", shape=dem.shape, chunks=dem.shape, dtype="float32")[:] = dem

    chm = np.zeros((2, ps, ps), dtype="float32")
    chm[0] = np.array([[1.0, np.nan], [3.0, 0.0]])
    chm[1] = 7.0
    data.create_array("chmv2", shape=chm.shape, chunks=chm.shape, dtype="float32")[:] = chm

    for k, v in (("chelsa_bioclim", [1, 1]), ("dem_cop30", [1, 1]), ("chmv2", [1, 0])):
        arr = data.create_array("mask_" + k, shape=(2,), chunks=(2,), dtype="bool")
        arr[:] = np.array(v, dtype=bool)
    return data


def test_read_stack_selects_bands_and_flattens_pixels(tmp_path):
    data = _data_store(tmp_path)
    covs = resolve_covariates(["mat", "elevation"])
    out = read_covariate_stack(data, covs, 0, 2, np.array([0, 1]))
    assert out.shape == (2, 4, 2)  # (cells, pixels, covariates)
    assert np.allclose(out[0, :, 0], [10.0, 20.0, 30.0, 40.0])
    assert np.allclose(out[1, :, 0], 5.0)
    assert np.allclose(out[:, :, 1], [[1.0] * 4, [2.0] * 4])


def test_read_stack_propagates_nan_and_per_cell_mask(tmp_path):
    data = _data_store(tmp_path)
    out = read_covariate_stack(data, resolve_covariates(["canopy_height"]), 0, 2, np.array([0, 1]))
    # stored NaN survives; cell 1 is masked out entirely
    assert np.isnan(out[0, 1, 0])
    assert np.allclose(out[0, [0, 2, 3], 0], [1.0, 3.0, 0.0])
    assert np.isnan(out[1, :, 0]).all()


def test_read_stack_honours_zero_is_nodata(tmp_path):
    data = _data_store(tmp_path)
    cov = Covariate("canopy_nz", "chmv2", None, "m", "canopy height (0 = nodata)", zero_is_nodata=True)
    out = read_covariate_stack(data, [cov], 0, 2, np.array([0, 1]))
    assert np.isnan(out[0, 3, 0])  # the stored 0.0 becomes nodata
    assert out[0, 0, 0] == 1.0


def test_read_stack_reindexes_a_sparse_cell_selection(tmp_path):
    data = _data_store(tmp_path)
    out = read_covariate_stack(data, resolve_covariates(["elevation"]), 0, 2, np.array([1]))
    assert out.shape == (1, 4, 1)
    assert np.allclose(out[0, :, 0], 2.0)


def test_available_covariates_keeps_order_and_drops_absent_arrays(tmp_path):
    data = _data_store(tmp_path)  # holds chelsa_bioclim, dem_cop30, chmv2
    covs = resolve_covariates(["canopy_height", "gpp", "mat", "elevation"])
    kept = available_covariates(data, covs)
    assert [c.name for c in kept] == ["canopy_height", "mat", "elevation"]
    assert read_covariate_stack(data, kept, 0, 2, np.array([0, 1])).shape == (2, 4, 3)


def test_available_covariates_is_what_a_store_without_the_array_needs(tmp_path):
    """The coarsened Copernicus-FM store carries no ``dem_cop30``."""
    data = _data_store(tmp_path)
    del data["dem_cop30"]
    kept = available_covariates(data, resolve_covariates(DEFAULT_COVARIATES))
    assert "elevation" not in [c.name for c in kept]
    read_covariate_stack(data, kept, 0, 2, np.array([0, 1]))

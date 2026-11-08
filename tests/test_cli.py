"""``geosae`` argument parsing, and the config-to-object helpers the stages share."""

import numpy as np
import pytest
import zarr

from geosae.cli import main
from geosae.pipeline import baseline_fit_pixels_from_config, load_run_transforms, make_data

# Every stage the CLI documents. New stages may be added; none of these may
# silently disappear, since configs and job scripts name them.
DOCUMENTED_STAGES = (
    "train",
    "predict",
    "run",
    "panel",
    "coverage",
    "pool",
    "variogram",
    "subconcepts",
    "aoa",
    "stats",
    "fit-table",
    "download",
    "subset",
    "gridded",
    "refetch",
    "release",
)


def _parse_error(argv, capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("stage", DOCUMENTED_STAGES)
def test_documented_stage_is_accepted(stage, capsys):
    # Omitting -c stops the parser before any stage runs; the complaint must be
    # about the missing config, never about the stage name.
    err = _parse_error([stage], capsys)
    assert "invalid choice" not in err
    assert "--config" in err


def test_unknown_stage_is_rejected(capsys):
    assert "invalid choice" in _parse_error(["evaluate-everything"], capsys)


def test_config_is_required(capsys):
    assert "--config" in _parse_error(["train"], capsys)


def test_no_arguments_is_rejected(capsys):
    _parse_error([], capsys)


# --------------------------------------------------------- make_data wiring


def _store(path, n=20, ps=4, d=4):
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update({"n_cells": n, "patch_size": ps})
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, ps, ps, d)).astype("float32")
    emb[5, 0, 0, :] = 0.0
    emb[5, 0, 0, 1] = 2e27  # one sentinel pixel
    g.create_array("alphaearth", shape=emb.shape, chunks=(n, ps, ps, d), dtype="float32")[:] = emb
    g.create_array("mask_alphaearth", shape=(n,), chunks=(n,), dtype="bool")[:] = np.ones(n, bool)
    g.create_array("split_210k", shape=(n,), chunks=(n,), dtype="int8")[:] = np.zeros(n, "int8")
    g.create_array("lon", shape=(n,), chunks=(n,), dtype="float64")[:] = np.zeros(n)
    g.create_array("lat", shape=(n,), chunks=(n,), dtype="float64")[:] = np.zeros(n)


def test_make_data_forwards_max_abs_value_from_the_config(tmp_path):
    path = tmp_path / "d.zarr"
    _store(path)
    cfg = {"data": {"zarr_path": str(path), "max_abs_value": 1e6}}
    data = make_data(cfg, "train", compute=False)
    assert 5 not in data.cells

    cfg_disabled = {"data": {"zarr_path": str(path), "max_abs_value": None}}
    data_disabled = make_data(cfg_disabled, "train", compute=False)
    assert 5 in data_disabled.cells


def test_make_data_forwards_zarr_path_to_stats_zarr_prefix_transform(tmp_path):
    path = tmp_path / "d.zarr"
    _store(path)
    g = zarr.open_group(str(path), mode="a")
    d = 4
    g.create_array("norm_mean", shape=(d,), chunks=(d,), dtype="float32")[:] = np.zeros(d, "float32")
    g.create_array("norm_std", shape=(d,), chunks=(d,), dtype="float32")[:] = np.full(d, 2.0, "float32")

    cfg = {
        "data": {
            "zarr_path": str(path),
            "max_abs_value": None,
            "transforms": {"stats_zarr_prefix": "norm"},
        }
    }
    data = make_data(cfg, "train", compute=False)
    assert np.allclose(data.transforms.std, 2.0)


def test_load_run_transforms_resolves_store_backed_stats(tmp_path):
    from omegaconf import OmegaConf

    path = tmp_path / "d.zarr"
    _store(path)
    g = zarr.open_group(str(path), mode="a")
    d = 4
    g.create_array("norm_mean", shape=(d,), chunks=(d,), dtype="float32")[:] = np.ones(d, "float32")
    g.create_array("norm_std", shape=(d,), chunks=(d,), dtype="float32")[:] = np.full(d, 3.0, "float32")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create({"data": {"zarr_path": str(path), "transforms": {"stats_zarr_prefix": "norm"}}}),
        run_dir / "config.yaml",
    )
    transforms = load_run_transforms(run_dir)
    assert np.allclose(transforms.mean, 1.0) and np.allclose(transforms.std, 3.0)


def test_fit_table_stage_writes_a_table_the_fitting_draw_then_reuses(patch_store, tmp_path):
    """The cached table is the contract between ``geosae fit-table`` and every fit."""
    from geosae.cli import cmd_fit_table
    from geosae.eval.baselines import load_fit_table

    path, _ = patch_store(n=24, patch_size=4, dim=8, chunk=6)
    out = tmp_path / "fit.npz"
    cfg = {
        "data": {"zarr_path": path, "min_spatial_std": None},
        "baselines": {"n_fit_pixels": 6, "sample_seed": 0, "fit_table_path": str(out)},
    }
    assert cmd_fit_table(cfg) == str(out)
    table = load_fit_table(out)
    assert table.meta["n_patches"] == 12 and table.meta["pixels_per_patch"] == 1

    stamp = out.stat().st_mtime_ns
    x = baseline_fit_pixels_from_config(cfg, cfg["baselines"], path, seed=0)
    assert x.shape == (6, 8)
    assert out.stat().st_mtime_ns == stamp, "the stage's table must be read, not rebuilt"


def test_whole_patch_fitting_draw_stays_available_but_must_be_asked_for(patch_store):
    path, _ = patch_store(n=24, patch_size=4, dim=8, chunk=6)
    cfg = {"data": {"zarr_path": path, "min_spatial_std": None}}
    block = {"n_fit_pixels": 6, "fit_sampling": "whole_patch"}
    assert baseline_fit_pixels_from_config(cfg, block, path, seed=0).shape == (6, 8)
    with pytest.raises(ValueError, match="unknown fit sampling"):
        baseline_fit_pixels_from_config(cfg, {**block, "fit_sampling": "nowhere"}, path, seed=0)

"""Black-box tests for AoA site sampling.

Contracts:
  * sites come from the whole split by default — one draw per usable patch, not
    a subsample of the patches — and only from pixels whose targets and land
    cover are all valid.
  * a cell restriction (an evaluation pool's cells) is honoured exactly, and the
    sites record it, so an applicability number can be tied to the same cells as
    the coverage numbers beside it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from geosae.data.dataset import PatchData
from geosae.eval.aoa import sample_sites
from geosae.models import build_model

PATCH, DIM = 4, 8


@pytest.fixture
def aoa_store(patch_store):
    """A store carrying the two AoA targets and a land-cover map."""
    path, _ = patch_store(n=24, patch_size=PATCH, dim=DIM)
    g = zarr.open_group(path, mode="a")
    rng = np.random.default_rng(0)
    shape, chunks = (24, PATCH, PATCH), (6, PATCH, PATCH)
    g.create_array("dem_cop30", shape=shape, chunks=chunks, dtype="float32")[:] = rng.normal(500, 50, shape).astype(
        "float32"
    )
    g.create_array("chmv2", shape=shape, chunks=chunks, dtype="float32")[:] = rng.uniform(0, 30, shape).astype(
        "float32"
    )
    g.create_array("lc_modis_igbp", shape=shape, chunks=chunks, dtype="int16")[:] = rng.integers(
        1, 6, shape, dtype="int16"
    )
    g.create_array("mask_lc_modis_igbp", shape=(24,), chunks=(24,), dtype="bool")[:] = np.ones(24, bool)
    return path


def _sites(path, **kw):
    data = PatchData(path, embedding_name="alphaearth", split_name="split_210k", split="test", compute=False)
    model = build_model({"input_dim": DIM, "latent_dim": 6, "k": 2, "seed": 0})
    with torch.no_grad():
        return sample_sites(model, data, batch_size=4, device="cpu", progress_every=0, **kw), data


def test_sites_are_a_census_of_the_split_not_a_subsample(aoa_store):
    sites, data = _sites(aoa_store)
    # Every patch of the split is usable here, and each contributes one pixel.
    assert len(sites) == len(data.cells)
    assert set(np.unique(sites.cell)) == set(data.cells.tolist())
    assert sites.spaces["raw"].shape == (len(sites), DIM)
    assert sites.spaces["sae"].shape[0] == len(sites)
    assert sites.meta["n_cells_requested"] is None


def test_a_cell_restriction_is_honoured_and_recorded(aoa_store):
    _, data = _sites(aoa_store)
    keep = np.asarray(data.cells[:3], dtype=np.int64)

    sites, _ = _sites(aoa_store, cells=keep)
    assert set(np.unique(sites.cell)) == set(keep.tolist())
    assert len(sites) == len(keep)
    assert sites.meta["n_cells_requested"] == 3


def test_restricting_to_pool_cells_reproduces_those_rows_of_the_full_draw(aoa_store):
    """The restriction narrows which cells are walked; it does not re-draw.

    The subset is deliberately not a prefix — a per-walk RNG would give this
    cell a different pixel once the cells before it stopped being sampled.
    """
    full, data = _sites(aoa_store)
    keep = np.asarray(data.cells[[1, 4, 5]], dtype=np.int64)
    part, _ = _sites(aoa_store, cells=keep)

    take = np.isin(full.cell, keep)
    assert np.array_equal(part.cell, full.cell[take])
    assert np.allclose(part.spaces["raw"], full.spaces["raw"][take])


def test_stage_sites_follow_the_aoa_seed_not_the_run_seed(aoa_store, tmp_path):
    """Runs of different seeds scored with one ``aoa.seed`` share their held-out pixels."""
    from geosae.cli import cmd_aoa
    from geosae.eval.aoa import load_sites

    model_cfg = {"type": "batchtopk", "input_dim": DIM, "latent_dim": 6, "k": 2}
    raw = []
    for run_seed in (0, 1):
        run_dir = tmp_path / f"s{run_seed}"
        run_dir.mkdir()
        torch.save({"model": build_model({**model_cfg, "seed": run_seed}).state_dict()}, run_dir / "best.pth")
        cfg = {
            "seed": run_seed,
            "device": "cpu",
            "data": {"zarr_path": aoa_store},
            "model": model_cfg,
            "aoa": {"seed": 5, "pixels_per_patch": 2, "batch_size": 4, "scenarios": []},
        }
        cmd_aoa(cfg, run_dir)
        raw.append(load_sites(str(run_dir / "eval" / "aoa" / "sites.npz")).spaces["raw"])
    np.testing.assert_array_equal(raw[0], raw[1])

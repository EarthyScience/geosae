"""Black-box tests for ``geosae panel``.

Contracts asserted (not implementation):
  * ``geosae panel`` scores trained runs and conventional baselines against a
    store's own concept catalog, and records ``concepts_available`` in the
    provenance rather than reducing an unlabelled store to ``nan``.
  * The panel's encode path (:func:`~geosae.eval.ladder.real_ladder_sample_from_registry`
    + :func:`~geosae.eval.ladder.model_encoder`) applies a run's transform and
    forward pass exactly once each, so its L0 matches the same pixels run
    through the ordinary training/validation path
    (:class:`~geosae.data.dataset.PatchData` + a bare model call) bit for bit.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import yaml

from geosae.cli import cmd_panel
from geosae.config import save_config
from geosae.data.dataset import Transforms
from geosae.eval.ladder import model_encoder, real_ladder_sample_from_registry
from geosae.eval.registry import sample_registry_pixels
from geosae.models import build_model


def _tiny_run_dir(tmp_path, *, dim=6, width=8, k=2, seed=0):
    """A trained-looking run: a resolved config.yaml plus a real best.pth."""
    model_cfg = {
        "type": "batchtopk",
        "k": k,
        "batch_topk": True,
        "normalize_inputs": False,
        "input_dim": dim,
        "latent_dim": width,
        "pre_bias": False,
        "normalize_decoder": True,
        "init_encoder_from_decoder": True,
        "n_batches_to_dead": 25,
        "h_frac": 1.0,
        "seed": seed,
    }
    model = build_model(dict(model_cfg))
    run_dir = tmp_path / "runs" / f"cell_s{seed}"
    run_dir.mkdir(parents=True)
    save_config({"model": model_cfg, "seed": seed}, str(run_dir / "config.yaml"))
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg, "epoch": 0}, run_dir / "best.pth")
    return run_dir


def test_real_panel_scores_trained_runs_against_the_catalog(tmp_path, patch_store):
    import zarr

    path, _ = patch_store(n=16, patch_size=4, dim=6, name="store.zarr")
    g = zarr.open_group(path, mode="a")
    codes = np.repeat(np.arange(16) % 3 * 10, 4 * 4).reshape(16, 4, 4).astype("int16")
    g.create_array("lc_esa_wc", shape=(16, 4, 4), chunks=(16, 4, 4), dtype="int16")[:] = codes

    run_dir = _tiny_run_dir(tmp_path)
    out_dir = tmp_path / "out"
    cfg = {
        "data": {"zarr_path": path, "embedding_name": "alphaearth", "split_name": "split_210k"},
        "device": "cpu",
        "run": {"dir": str(out_dir)},
        "panel": {
            "n_cells": 4,
            "pixels_per_cell": 8,
            "products": ["lc_esa_wc"],
            "runs": [str(run_dir)],
            "blocks": [1, 2, 3],
        },
    }
    rows = cmd_panel(cfg, out_dir)
    assert len(rows) == 1
    row = rows[0]
    assert np.isfinite(row["fve"]) and np.isfinite(row["ucov"])

    written = json.loads((out_dir / "eval" / "panel.json").read_text())
    assert written["_provenance"]["concepts_available"] is True
    assert (out_dir / "eval" / "panel.csv").exists()


def test_real_panel_flags_concepts_unavailable_without_a_catalog(tmp_path, patch_store):
    path, _ = patch_store(n=8, patch_size=4, dim=6, name="store.zarr")
    run_dir = _tiny_run_dir(tmp_path)
    out_dir = tmp_path / "out"
    cfg = {
        "data": {"zarr_path": path, "embedding_name": "alphaearth", "split_name": "split_210k"},
        "device": "cpu",
        "panel": {
            "n_cells": 2,
            "pixels_per_cell": 8,
            "products": [],
            "runs": [str(run_dir)],
            "blocks": [1, 2, 3],
        },
    }
    rows = cmd_panel(cfg, out_dir)
    assert "ucov" not in rows[0] and np.isfinite(rows[0]["fve"])
    written = json.loads((out_dir / "eval" / "panel.json").read_text())
    assert written["_provenance"]["concepts_available"] is False


def test_real_panel_applies_each_runs_own_transform(tmp_path, patch_store):
    """Two rows on the same store may carry different ``data.transforms``
    (each z-scores against its own stats). ``geosae panel`` must apply each run's own
    transform, read from *its own* ``config.yaml`` -- not the panel-level
    config's, and not a shared one baked into the sample -- so two runs that
    differ only in ``data.transforms`` must score differently.
    """
    path = _store_with_catalog(patch_store)
    run_a = _tiny_run_dir(tmp_path, seed=0)
    run_b = _tiny_run_dir(tmp_path, seed=1)
    for run_dir, scale in [(run_a, 10.0), (run_b, 0.1)]:
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        run_cfg["data"] = {"transforms": {"input_scale": scale}}
        save_config(run_cfg, str(run_dir / "config.yaml"))

    out_dir = tmp_path / "out"
    cfg = {
        "data": {"zarr_path": path, "embedding_name": "alphaearth", "split_name": "split_210k"},
        "device": "cpu",
        "panel": {
            "n_cells": 4,
            "pixels_per_cell": 8,
            "products": ["lc_esa_wc"],
            "runs": [str(run_a), str(run_b)],
            "blocks": [1],
        },
    }
    rows = cmd_panel(cfg, out_dir)
    fve = {r["name"]: r["fve"] for r in rows}
    assert set(fve) == {"cell_s0", "cell_s1"}
    assert all(np.isfinite(v) for v in fve.values())
    assert fve["cell_s0"] != fve["cell_s1"]  # only their transform differs


def test_real_encode_path_matches_the_training_path_under_a_zscore_transform(tmp_path, patch_store):
    """The panel's real-store encode path -- ``real_ladder_sample_from_registry``
    reading raw pixels then applying a run's transform, followed by
    ``model_encoder`` -- must give the identical L0 (and codes) that the
    ordinary training/validation path gives the same pixels: ``PatchData``
    applying the same transform inside ``_make_batch``, followed by a bare
    ``model(x)`` call. A regression here (the transform applied twice, or not
    reaching the encode call, or a different pixel composition changing a
    batch-dependent input rescale) would silently change every emergent-
    sparsity model's (ReLU+l1) reported L0 without touching FVE
    parity checks alone.
    """
    dim = 6
    path = _store_with_catalog(patch_store, n=16)
    import zarr

    g = zarr.open_group(path, mode="a")
    rng = np.random.default_rng(0)
    g.create_array("norm_mean", shape=(dim,), chunks=(dim,), dtype="float32")[:] = rng.normal(size=dim).astype(
        "float32"
    )
    g.create_array("norm_std", shape=(dim,), chunks=(dim,), dtype="float32")[:] = np.abs(
        rng.normal(loc=2.0, size=dim)
    ).astype("float32")

    model_cfg = {
        "type": "relu_l1",
        "input_dim": dim,
        "latent_dim": 12,
        "pre_bias": True,
        "normalize_decoder": True,
        "normalize_inputs": True,  # batch-dependent rescale: exercises the batch-composition suspect
        "init_encoder_from_decoder": True,
        "seed": 0,
    }
    model = build_model(dict(model_cfg))
    model.eval()

    transforms = Transforms.from_zarr_stats(path, "norm")
    reg = sample_registry_pixels(
        path, split="test", split_name="split_210k", embedding_name="alphaearth", n_patches=4, pixels_per_patch=8
    )

    # Training/validation path: PatchData's `_make_batch` applies the same
    # transform, then the model is called directly on its output.
    with torch.no_grad():
        x_train = transforms.apply(torch.as_tensor(reg.raw, dtype=torch.float32))
        _, z_train, _ = model(x_train)
    l0_train = (z_train > 0).sum(-1).double().mean().item()

    # Panel path: the same raw pixels, wrapped and transformed by the ladder.
    sample = real_ladder_sample_from_registry(reg, transforms=transforms)
    raw_panel = np.concatenate([chunk for chunk, _ in sample.pixels()], axis=0)
    z_panel = model_encoder(model, device="cpu")(raw_panel)
    l0_panel = (z_panel > 0).sum(-1).mean()

    np.testing.assert_allclose(raw_panel, x_train.numpy(), atol=1e-5)
    assert l0_panel == l0_train


def _store_with_catalog(patch_store, *, n=16):
    import zarr

    path, _ = patch_store(n=n, patch_size=4, dim=6, name="store.zarr")
    g = zarr.open_group(path, mode="a")
    codes = np.repeat(np.arange(n) % 3 * 10, 4 * 4).reshape(n, 4, 4).astype("int16")
    g.create_array("lc_esa_wc", shape=(n, 4, 4), chunks=(n, 4, 4), dtype="int16")[:] = codes
    return path


def test_panel_merge_shards_joins_two_shards_into_one_table(tmp_path, patch_store):
    path = _store_with_catalog(patch_store)
    s0 = _tiny_run_dir(tmp_path, seed=0)
    s1 = _tiny_run_dir(tmp_path, seed=1)

    def _score(shard_name, run_dir):
        out_dir = tmp_path / shard_name
        cfg = {
            "data": {"zarr_path": path, "embedding_name": "alphaearth", "split_name": "split_210k"},
            "device": "cpu",
            "panel": {
                "n_cells": 4,
                "pixels_per_cell": 8,
                "products": ["lc_esa_wc"],
                "runs": [str(run_dir)],
                "blocks": [1],
            },
        }
        cmd_panel(cfg, out_dir)
        return out_dir

    shard_a = _score("shard_a", s0)
    shard_b = _score("shard_b", s1)

    merged_dir = tmp_path / "merged"
    rows = cmd_panel({"panel": {"merge_shards": [str(shard_a), str(shard_b)]}}, merged_dir)
    assert {r["name"] for r in rows} == {"cell_s0", "cell_s1"}

    written = json.loads((merged_dir / "eval" / "panel.json").read_text())
    assert written["_provenance"]["n_shards"] == 2
    assert (merged_dir / "eval" / "panel.csv").exists()


def test_panel_merge_shards_lets_the_last_listed_shard_win_a_duplicated_row(tmp_path):
    def _shard(name, fve):
        d = tmp_path / name / "eval"
        d.mkdir(parents=True)
        (d / "panel.json").write_text(json.dumps({"rows": [{"name": "cell_s0", "fve": fve}], "_provenance": {}}))
        return tmp_path / name

    first = _shard("z_original", 0.1)  # sorts *after* the rescore by name
    rescore = _shard("a_rescore", 0.9)
    rows = cmd_panel({"panel": {"merge_shards": [str(first), str(rescore)]}}, tmp_path / "merged")
    assert [r["fve"] for r in rows] == [0.9]


def test_panel_scores_conventional_baselines_and_their_rank_truncations(tmp_path, patch_store):
    path = _store_with_catalog(patch_store, n=24)
    out_dir = tmp_path / "out"
    cfg = {
        "data": {
            "zarr_path": path,
            "embedding_name": "alphaearth",
            "split_name": "split_210k",
            "min_spatial_std": None,
        },
        "device": "cpu",
        "model": {"k": 2},
        "panel": {
            "n_cells": 4,
            "pixels_per_cell": 8,
            "products": ["lc_esa_wc"],
            "blocks": [1],
            "baselines": {"methods": ["pca"], "widths": [6], "ranks": [2], "n_fit_pixels": 8},
        },
    }
    rows = {r["name"]: r for r in cmd_panel(cfg, out_dir)}
    assert set(rows) == {"pca_M6_s0", "pca_M6_rank2_s0"}
    assert rows["pca_M6_rank2_s0"]["l0"] <= 2.0 + 1e-9
    assert all(np.isfinite(r["fve"]) for r in rows.values())

"""Shared fixtures: tiny synthetic patch stores and core-only import bookkeeping."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
import zarr

# The suite runs on tiny tensors, where torch's default "one thread per core"
# costs far more in contention than it saves.
torch.set_num_threads(1)

# Plotting modules outside the core install; the library and its test suite never import them.
VIZ_ONLY_MODULES = ("nicefigs", "cartopy", "matplotlib")

_MODULES_AFTER_COLLECTION: set[str] = set()


def pytest_collection_finish(session):
    """Snapshot ``sys.modules`` once every test module has been imported."""
    _MODULES_AFTER_COLLECTION.clear()
    _MODULES_AFTER_COLLECTION.update(sys.modules)


@pytest.fixture(scope="session")
def modules_imported_at_collection() -> set[str]:
    """Top-level module names present after the suite finished collecting."""
    return {name.split(".")[0] for name in _MODULES_AFTER_COLLECTION}


@pytest.fixture(autouse=True)
def _quality_cache_stays_in_tmp_path(tmp_path, monkeypatch):
    """Redirect PatchData's (on-by-default) quality-mask cache away from ``data/``.

    ``PatchData`` resolves an unset ``quality_cache_root`` from
    ``geosae.data.quality.DEFAULT_QUALITY_CACHE_ROOT`` at construction time
    (see that module and ``geosae/data/dataset.py``), so patching the
    attribute here — before every test, session-wide — is enough to keep
    every test's throwaway store from ever writing a cache file into the
    real, shared ``data/eval_cache/quality`` a worktree's ``data/`` symlinks
    to. No test constructing a `PatchData` needs to know this fixture exists.
    """
    import geosae.data.quality as quality

    monkeypatch.setattr(quality, "DEFAULT_QUALITY_CACHE_ROOT", str(tmp_path / "quality_cache"))


@pytest.fixture
def patch_store(tmp_path):
    """Factory writing a tiny patch store and returning ``(path, embeddings)``.

    The store mirrors the real one: an ``(N, H, W, D)`` embedding array, a
    validity mask, a split-label array, and per-cell ``lon``/``lat``. Splits are
    laid out so every split is non-empty for the default ``n``.
    """

    def _make(
        name: str = "patches.zarr",
        *,
        n: int = 24,
        patch_size: int = 4,
        dim: int = 8,
        chunk: int = 6,
        embedding_name: str = "alphaearth",
        split_name: str = "split_210k",
        seed: int = 0,
    ):
        path = tmp_path / name
        g = zarr.open_group(str(path), mode="w")
        g.attrs.update({"n_cells": n, "patch_size": patch_size})
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((n, patch_size, patch_size, dim)).astype("float32")
        g.create_array(
            embedding_name,
            shape=(n, patch_size, patch_size, dim),
            chunks=(chunk, patch_size, patch_size, dim),
            dtype="float32",
        )[:] = emb
        g.create_array(f"mask_{embedding_name}", shape=(n,), chunks=(n,), dtype="bool")[:] = np.ones(n, bool)
        # Half train, a quarter val, a quarter test.
        split = np.full(n, 0, dtype="int8")
        split[n // 2 : 3 * n // 4] = 1
        split[3 * n // 4 :] = 2
        g.create_array(split_name, shape=(n,), chunks=(n,), dtype="int8")[:] = split
        g.create_array("lon", shape=(n,), chunks=(n,), dtype="float64")[:] = np.linspace(-170, 170, n)
        g.create_array("lat", shape=(n,), chunks=(n,), dtype="float64")[:] = np.linspace(-80, 80, n)
        return str(path), emb

    return _make

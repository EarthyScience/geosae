"""PatchData selection, in-RAM vs lazy reads, transforms, and the DataLoader.

Covers the torch pipeline: PatchDataset micro-batches, collate_patches, and
make_dataloader producing full batches identical to PatchData.iter_batches.
"""

import warnings

import numpy as np
import pytest
import torch
import zarr

from geosae.data.dataset import (
    PatchData,
    PatchDataset,
    Transforms,
    collate_patches,
    make_dataloader,
)


def _make_store(path, n=20, ps=4, d=8, chunk=5):
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update({"n_cells": n, "patch_size": ps, "year": 2024})
    emb = g.create_array("alphaearth", shape=(n, ps, ps, d), chunks=(chunk, ps, ps, d), dtype="float32")
    rng = np.random.default_rng(0)
    emb[:] = rng.standard_normal((n, ps, ps, d)).astype("float32")
    mask = np.ones(n, dtype=bool)
    mask[3] = False  # one invalid cell
    g.create_array("mask_alphaearth", shape=(n,), chunks=(n,), dtype="bool")[:] = mask
    # split: first 10 = test(2), next 6 = train(0), next 4 = val(1)
    split = np.full(n, -1, dtype="int8")
    split[:10] = 2
    split[10:16] = 0
    split[16:] = 1
    g.create_array("split_210k", shape=(n,), chunks=(n,), dtype="int8")[:] = split
    g.create_array("lon", shape=(n,), chunks=(n,), dtype="float64")[:] = np.arange(n, dtype="float64")
    g.create_array("lat", shape=(n,), chunks=(n,), dtype="float64")[:] = np.arange(n, dtype="float64") + 100
    return emb[:]


def test_split_filtering_excludes_invalid(tmp_path):
    _make_store(tmp_path / "s.zarr")
    train = PatchData(str(tmp_path / "s.zarr"), split="train", compute=True)
    # cells 10..15 are train, all valid → 6 cells
    assert len(train) == 6
    test = PatchData(str(tmp_path / "s.zarr"), split="test", compute=True)
    # cells 0..9 are test but cell 3 is masked invalid → 9 cells
    assert len(test) == 9
    assert 3 not in test.cells.tolist()


def test_inram_matches_lazy(tmp_path):
    raw = _make_store(tmp_path / "s.zarr")
    ram = PatchData(str(tmp_path / "s.zarr"), split="test", compute=True, inram_dtype="float32")
    lazy = PatchData(str(tmp_path / "s.zarr"), split="test", compute=False)
    b_ram = next(ram.iter_batches(4, shuffle=False))
    b_lazy = next(lazy.iter_batches(4, shuffle=False))
    assert torch.allclose(b_ram["x"], b_lazy["x"], atol=1e-6)
    # values match the raw store for the selected cells (cells 0,1,2,4)
    expected = torch.from_numpy(raw[[0, 1, 2, 4]])
    assert torch.allclose(b_ram["x"], expected, atol=1e-6)
    assert b_ram["cell"].tolist() == [0, 1, 2, 4]


def test_inram_load_does_not_overflow_on_a_dropped_chunk_sibling(tmp_path):
    """A masked-out cell's fill sentinel must not spend float16 range on kept cells.

    ``_make_store``'s cell 3 is masked invalid and shares its zarr chunk
    (chunk=5) with kept test cells 0,1,2,4. Cast the whole chunk to float16
    at once and cell 3's sentinel would trigger (harmlessly, since it's
    discarded) a RuntimeWarning; loading is expected to raise none and to
    return exact, finite values for the kept cells regardless.
    """
    path = tmp_path / "s.zarr"
    raw = _make_store(path)
    emb = zarr.open_group(str(path), mode="a")["alphaearth"]
    emb[3] = 1e20  # masked-out cell 3's fill sentinel, shares chunk 0 with 0,1,2,4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = PatchData(str(path), split="test", compute=True, inram_dtype="float16")
    x = next(data.iter_batches(4, shuffle=False))["x"]
    assert torch.isfinite(x).all()
    expected = torch.from_numpy(raw[[0, 1, 2, 4]]).to(torch.float16).to(torch.float32)
    assert torch.allclose(x, expected, atol=1e-2)


def test_micro_batch_is_chunk_aligned(tmp_path):
    _make_store(tmp_path / "s.zarr", n=20, chunk=5)
    lazy = PatchData(str(tmp_path / "s.zarr"), split="test", compute=False)
    from geosae.data.dataset import _plan_micro_batches

    groups, plan = _plan_micro_batches(lazy, 4, shuffle=False, seed=0, drop_last=False, micro_size=None)
    ds = PatchDataset(lazy, groups)
    # Every micro-batch (chunk-aligned group) has all its cells in one storage chunk.
    for g in groups:
        chunk_ids = lazy.cells[g] // lazy.chunk_size
        assert len(np.unique(chunk_ids)) == 1
    # __getitem__ returns a micro-batch dict of tensors.
    mb = ds[0]
    assert set(mb) == {"x", "cell", "lon", "lat", "local_idx"}
    assert mb["x"].ndim == 4  # (b, H, W, D)


def _batches_equal(a, b):
    return (
        torch.allclose(a["x"], b["x"], atol=1e-6)
        and a["cell"].tolist() == b["cell"].tolist()
        and a["lon"].tolist() == b["lon"].tolist()
    )


def test_dataloader_matches_iter_batches_no_shuffle(tmp_path):
    _make_store(tmp_path / "s.zarr")
    for compute in (True, False):
        d = PatchData(str(tmp_path / "s.zarr"), split="test", compute=compute, inram_dtype="float32")
        ref = list(d.iter_batches(4, shuffle=False))
        got = list(make_dataloader(d, 4, shuffle=False, num_workers=0, pin_memory=False))
        assert len(got) == len(ref)
        for a, b in zip(got, ref, strict=True):
            assert _batches_equal(a, b)


def test_dataloader_matches_iter_batches_shuffle(tmp_path):
    _make_store(tmp_path / "s.zarr")
    d = PatchData(str(tmp_path / "s.zarr"), split="test", compute=True, inram_dtype="float32")
    ref = list(d.iter_batches(4, shuffle=True, seed=7, drop_last=True))
    got = list(make_dataloader(d, 4, shuffle=True, seed=7, drop_last=True, num_workers=0, pin_memory=False))
    assert len(got) == len(ref)
    for a, b in zip(got, ref, strict=True):
        assert _batches_equal(a, b)


def test_dataloader_with_workers_covers_all_cells(tmp_path):
    _make_store(tmp_path / "s.zarr")
    d = PatchData(str(tmp_path / "s.zarr"), split="test", compute=False)
    seen = []
    for b in make_dataloader(d, 4, shuffle=False, num_workers=2, pin_memory=False):
        seen.extend(b["cell"].tolist())
    assert sorted(seen) == sorted(d.cells.tolist())


def test_collate_patches_concatenates():
    a = {"x": torch.zeros(2, 3), "cell": torch.tensor([0, 1])}
    b = {"x": torch.ones(3, 3), "cell": torch.tensor([2, 3, 4])}
    out = collate_patches([a, b])
    assert out["x"].shape == (5, 3)
    assert out["cell"].tolist() == [0, 1, 2, 3, 4]


def test_iter_batches_covers_all_cells(tmp_path):
    _make_store(tmp_path / "s.zarr")
    d = PatchData(str(tmp_path / "s.zarr"), split="test", compute=True)
    seen = []
    for b in d.iter_batches(4, shuffle=True, seed=1):
        seen.extend(b["cell"].tolist())
    assert sorted(seen) == sorted(d.cells.tolist())


def test_transforms_zscore_and_clip():
    t = Transforms(hard_clip=2.0, mean=np.zeros(3, "float32"), std=np.full(3, 2.0, "float32"))
    x = torch.tensor([[10.0, -10.0, 1.0]])
    out = t.apply(x)
    # hard_clip to [-2,2] then /std=2 → [1,-1,0.5]
    assert torch.allclose(out, torch.tensor([[1.0, -1.0, 0.5]]), atol=1e-6)


def test_transforms_nan_to_num():
    t = Transforms()
    x = torch.tensor([[float("nan"), float("inf"), 1.0]])
    out = t.apply(x)
    assert torch.isfinite(out).all()


def test_token_grid_coarser_than_the_store_pixel_grid_is_read_correctly(tmp_path):
    """A patch-token product has its own spatial size; the store's patch_size must not win."""
    path = tmp_path / "tokens.zarr"
    raw = _make_store(path, n=20, ps=4, d=8)
    g = zarr.open_group(str(path), mode="a")
    # A 16 px store whose token grid is 4x4: patch_size disagrees with the array.
    g.attrs["patch_size"] = 16

    data = PatchData(str(path), split="test", compute=True, inram_dtype="float32")
    assert data.patch_size == 4
    batch = next(data.iter_batches(4, shuffle=False))
    assert batch["x"].shape == (4, 4, 4, 8)
    assert torch.allclose(batch["x"], torch.from_numpy(raw[[0, 1, 2, 4]]), atol=1e-6)


def test_non_square_embedding_patches_are_rejected(tmp_path):
    import pytest

    path = tmp_path / "ragged.zarr"
    _make_store(path, n=20, ps=4, d=8)
    g = zarr.open_group(str(path), mode="a")
    del g["alphaearth"]
    g.create_array("alphaearth", shape=(20, 4, 6, 8), chunks=(5, 4, 6, 8), dtype="float32")
    with pytest.raises(ValueError, match="must be square"):
        PatchData(str(path), split="test", compute=True)


# ------------------------------------------------------- normalisation stats
def _stats_store(path, values):
    """A patch_size=1 store holding ``values`` of shape (n, D) as its embedding."""
    import zarr

    from geosae.data.download.store import ArraySchema, build_schemas, create_store

    n, d = values.shape
    schemas = build_schemas([], n, embeddings=(), splits=("split_s",), patch_size=1)
    dims = ("cell", "y", "x", "feature_emb")
    schemas["emb"] = ArraySchema("float32", (n, 1, 1, d), (n, 1, 1, d), np.nan, dims, {"_ARRAY_DIMENSIONS": list(dims)})
    schemas["mask_emb"] = ArraySchema("bool", (n,), (n,), False, ("cell",), {"_ARRAY_DIMENSIONS": ["cell"]})
    g = create_store(path, schemas, n_cells=n, patch_size=1)
    g["emb"][:] = values.reshape(n, 1, 1, d)
    g["mask_emb"][:] = True
    g["split_s"][:] = 0
    g["lon"][:] = np.linspace(-10, 10, n)
    g["lat"][:] = np.linspace(-10, 10, n)
    return zarr.open_group(str(path), mode="r")


def test_embedding_stats_recovers_the_mean_and_per_dim_std(tmp_path):
    from geosae.data.dataset import compute_embedding_stats

    rng = np.random.default_rng(0)
    x = rng.normal(loc=[5.0, -2.0, 0.0], scale=[1.0, 4.0, 0.5], size=(4000, 3)).astype(np.float32)
    _stats_store(tmp_path / "s.zarr", x)
    # patch_size=1 cells have zero spatial std, so the degenerate-cell filter
    # (on by default) would drop every row at this geometry.
    stats = compute_embedding_stats(str(tmp_path / "s.zarr"), "emb", "split_s", "train", min_spatial_std=None)
    assert np.allclose(stats["mean"], x.mean(0), atol=0.1)
    assert np.allclose(stats["std"], x.std(0), rtol=0.05)


def test_isotropic_stats_use_one_scale_for_every_dimension(tmp_path):
    """Isotropic scaling is a dilation, so it must not flatten the dims' spread."""
    from geosae.data.dataset import compute_embedding_stats

    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, scale=[1.0, 4.0], size=(4000, 2)).astype(np.float32)
    _stats_store(tmp_path / "s.zarr", x)
    stats = compute_embedding_stats(
        str(tmp_path / "s.zarr"), "emb", "split_s", "train", isotropic=True, min_spatial_std=None
    )
    assert len(set(stats["std"].tolist())) == 1
    # The single scale is the RMS deviation over all dimensions.
    assert stats["std"][0] == pytest.approx(float(np.sqrt((x - x.mean(0)).var(0).mean())), rel=0.05)


def test_stats_round_trip_through_transforms_centres_the_data(tmp_path):
    from geosae.data.dataset import PatchData, Transforms, compute_embedding_stats, write_embedding_stats

    rng = np.random.default_rng(2)
    x = (rng.normal(size=(2000, 8)) * 3.0 + 20.0).astype(np.float32)
    _stats_store(tmp_path / "s.zarr", x)
    stats = compute_embedding_stats(
        str(tmp_path / "s.zarr"), "emb", "split_s", "train", isotropic=True, min_spatial_std=None
    )
    npz = write_embedding_stats(tmp_path / "stats.npz", stats)

    data = PatchData(
        str(tmp_path / "s.zarr"),
        "emb",
        "split_s",
        "train",
        compute=True,
        n_load_workers=1,
        min_spatial_std=None,
        transforms=Transforms.from_stats_npz(str(npz)),
    )
    z = np.concatenate([b["x"].numpy().reshape(-1, 8) for b in data.iter_batches(512)])
    assert np.abs(z.mean(0)).max() < 0.05
    assert np.linalg.norm(z, axis=1).mean() == pytest.approx(np.sqrt(8), rel=0.1)


# ------------------------------------------- stats read directly from the store


def test_transforms_from_zarr_stats_reads_store_arrays(tmp_path):
    path = tmp_path / "s.zarr"
    g = zarr.open_group(str(path), mode="w")
    d = 4
    g.create_array("foo_mean", shape=(d,), chunks=(d,), dtype="float32")[:] = np.array([1, 2, 3, 4], "float32")
    g.create_array("foo_std", shape=(d,), chunks=(d,), dtype="float32")[:] = np.full(d, 2.0, "float32")
    g.create_array("foo_clip_lo", shape=(d,), chunks=(d,), dtype="float32")[:] = np.full(d, -5.0, "float32")
    g.create_array("foo_clip_hi", shape=(d,), chunks=(d,), dtype="float32")[:] = np.full(d, 5.0, "float32")

    t = Transforms.from_zarr_stats(str(path), "foo")
    out = t.apply(torch.tensor([[10.0, 10.0, 10.0, 10.0]]))
    # clipped to 5 first, then (5 - mean) / std
    expected = (torch.tensor([5.0, 5.0, 5.0, 5.0]) - torch.tensor([1.0, 2.0, 3.0, 4.0])) / 2.0
    assert torch.allclose(out[0], expected, atol=1e-5)


def test_transforms_from_zarr_stats_without_clip_arrays(tmp_path):
    path = tmp_path / "s.zarr"
    g = zarr.open_group(str(path), mode="w")
    d = 3
    g.create_array("bar_mean", shape=(d,), chunks=(d,), dtype="float32")[:] = np.zeros(d, "float32")
    g.create_array("bar_std", shape=(d,), chunks=(d,), dtype="float32")[:] = np.ones(d, "float32")

    t = Transforms.from_zarr_stats(str(path), "bar")
    assert t.clip_lo is None and t.clip_hi is None


def test_build_transforms_stats_zarr_prefix_requires_a_zarr_path():
    from geosae.data.dataset import build_transforms

    with pytest.raises(ValueError, match="zarr_path"):
        build_transforms({"stats_zarr_prefix": "foo"})


def test_build_transforms_stats_zarr_prefix_wires_through(tmp_path):
    from geosae.data.dataset import build_transforms

    path = tmp_path / "s.zarr"
    g = zarr.open_group(str(path), mode="w")
    d = 3
    g.create_array("bar_mean", shape=(d,), chunks=(d,), dtype="float32")[:] = np.zeros(d, "float32")
    g.create_array("bar_std", shape=(d,), chunks=(d,), dtype="float32")[:] = np.ones(d, "float32")

    t = build_transforms({"stats_zarr_prefix": "bar"}, zarr_path=str(path))
    assert t.mean is not None and t.std is not None
    assert t.clip_lo is None and t.clip_hi is None


def test_the_split_array_defaults_to_the_stores_only_one(patch_store):
    from geosae.data.dataset import resolve_split_name

    path, _ = patch_store(split_name="split_default")
    assert resolve_split_name(path) == "split_default"
    assert len(PatchData(path, split="test")) == len(PatchData(path, split_name="split_default", split="test")) > 0
    with pytest.raises(KeyError, match="split_210k"):
        PatchData(path, split_name="split_210k", split="test")


def test_a_store_with_several_splits_must_name_one(patch_store):
    from geosae.data.dataset import resolve_split_name

    path, _ = patch_store()
    zarr.open_group(path, mode="r+").create_array("split_other", shape=(24,), dtype="int8")
    with pytest.raises(KeyError, match="split_name"):
        resolve_split_name(path)
    assert resolve_split_name(path, "split_other") == "split_other"

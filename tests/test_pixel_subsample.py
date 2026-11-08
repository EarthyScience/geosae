"""Black-box CPU tests for the per-step pixel subsampler.

Covers: correct per-step pixel count + bounds, output shape (feeds BatchTopK as
Q flat tokens), determinism under seed, and no leakage (every returned pixel is a
genuine array pixel, never out-of-range / interpolated garbage). Core deps only
(torch + numpy), so it runs under the ``--no-deps`` CI install.
"""

import numpy as np
import torch

from geosae.data.dataset import Transforms
from geosae.train.pixel_subsample import PixelSubsampler, sample_pixels


def _fingerprint_arr(n_patches: int, H: int, W: int) -> np.ndarray:
    """(n_patches, H, W, 1) where each pixel value == its global flat id.

    Lets a leakage test decode any returned pixel back to (patch, y, x).
    """
    ids = np.arange(n_patches * H * W, dtype=np.float64).reshape(n_patches, H, W, 1)
    return ids


def test_shape_and_count_bounds():
    N, M, K = 32, 8, 16
    n_patches, H, W, D = 40, 8, 8, 5
    arr = torch.randn(n_patches, H, W, D)
    g = torch.Generator().manual_seed(0)
    x = sample_pixels(
        arr, n_anchor_patches=N, m_anchor_pixels=M, recon_pixels_per_patch=K, pair_mode="local", generator=g
    )
    Q = x.shape[0]
    assert x.ndim == 4 and x.shape[1:] == (1, 1, D)  # flat (Q,1,1,D) tokens for BatchTopK
    assert x.dtype == torch.float32
    # P_loaded = |unique(anchor draws)| <= N.
    assert Q >= 1 * K  # at least the random draws of one loaded patch survive
    assert Q <= 3 * N * M + N * K  # upper bound with P_loaded <= N


def test_count_matches_the_default_scale():
    # 64x64 patches, N=512, K=128 -> Q in [P_loaded*128, 3*512*8 + P_loaded*128].
    # With a large pool P_loaded ~= 512 -> Q ~ 78k (12,288 paired + 65,536 random).
    N, M, K = 512, 8, 128
    n_patches, H, W, D = 4000, 64, 64, 4
    arr = torch.zeros(n_patches, H, W, D)  # values irrelevant here
    g = torch.Generator().manual_seed(1)
    x = sample_pixels(arr, n_anchor_patches=N, m_anchor_pixels=M, recon_pixels_per_patch=K, generator=g)
    Q = x.shape[0]
    # Collisions among 512 draws from 4000 are rare -> P_loaded close to 512.
    assert 500 <= Q - (3 * N * M) <= N * K  # random block is P_loaded*K, P_loaded in ~[500,512]
    assert 70_000 <= Q <= 78_000  # ~78k pixels/step at the defaults


def test_determinism_under_seed():
    N, M, K = 16, 8, 12
    arr = torch.randn(50, 8, 8, 6)
    x1 = sample_pixels(
        arr,
        n_anchor_patches=N,
        m_anchor_pixels=M,
        recon_pixels_per_patch=K,
        generator=torch.Generator().manual_seed(123),
    )
    x2 = sample_pixels(
        arr,
        n_anchor_patches=N,
        m_anchor_pixels=M,
        recon_pixels_per_patch=K,
        generator=torch.Generator().manual_seed(123),
    )
    x3 = sample_pixels(
        arr,
        n_anchor_patches=N,
        m_anchor_pixels=M,
        recon_pixels_per_patch=K,
        generator=torch.Generator().manual_seed(999),
    )
    assert x1.shape == x2.shape and torch.equal(x1, x2)  # same seed -> identical
    assert not (x1.shape == x3.shape and torch.equal(x1, x3))  # different seed -> different


def test_no_leakage_every_pixel_is_genuine():
    N, M, K = 24, 8, 10
    n_patches, H, W = 30, 8, 8
    arr = torch.from_numpy(_fingerprint_arr(n_patches, H, W))  # (n_patches,H,W,1), value==flat id
    g = torch.Generator().manual_seed(7)
    x = sample_pixels(arr, n_anchor_patches=N, m_anchor_pixels=M, recon_pixels_per_patch=K, generator=g)
    vals = x.reshape(-1).to(torch.int64)
    # Every returned value must be a valid global flat id in [0, n_patches*H*W).
    assert int(vals.min()) >= 0 and int(vals.max()) < n_patches * H * W
    # And every value must be integer-valued (a genuine gathered pixel, no blending).
    assert torch.equal(x.reshape(-1), vals.to(torch.float32))
    # Pixels come only from at most N distinct patches (the anchor draws).
    patch_of = vals // (H * W)
    assert int(torch.unique(patch_of).numel()) <= N


def test_patch_mode_and_invalid_mode():
    arr = torch.randn(20, 8, 8, 3)
    g = torch.Generator().manual_seed(0)
    x = sample_pixels(
        arr, n_anchor_patches=8, m_anchor_pixels=4, recon_pixels_per_patch=6, pair_mode="patch", generator=g
    )
    assert x.ndim == 4 and x.shape[-1] == 3
    try:
        sample_pixels(arr, pair_mode="regional", generator=g)
    except ValueError as e:
        assert "regional" in str(e)
    else:
        raise AssertionError("regional pair_mode must raise")


def test_subsampler_batches_stream():
    arr = np.random.randn(40, 8, 8, 5).astype(np.float32)
    sub = PixelSubsampler(arr, recon_pixels_per_patch=8, n_anchor_patches=12, m_anchor_pixels=4, seed=42)
    batches = list(sub.batches(3))
    assert len(batches) == 3
    for b in batches:
        assert set(b) == {"x"} and b["x"].ndim == 4 and b["x"].shape[1:3] == (1, 1)
    # Persistent generator -> the three steps advance (are not identical).
    assert not torch.equal(batches[0]["x"], batches[1]["x"]) or batches[0]["x"].shape != batches[1]["x"].shape


def test_subsampler_requires_inram():
    try:
        PixelSubsampler(None)
    except ValueError as e:
        assert "in-RAM" in str(e)
    else:
        raise AssertionError("None array must raise")


def test_sample_pixels_applies_transforms():
    # arr is raw (untransformed) data; sample_pixels must apply `transforms`
    # to the gathered pixels itself, since it reads straight from `arr` and
    # bypasses PatchData.iter_batches (the only other place Transforms are
    # normally applied); otherwise training and validation would see
    # differently transformed inputs.
    D = 4
    arr = torch.rand(20, 8, 8, D) * 100.0  # far outside a z-scored range
    mean = np.full(D, 50.0, dtype=np.float32)
    std = np.full(D, 10.0, dtype=np.float32)
    t = Transforms(mean=mean, std=std)
    g = torch.Generator().manual_seed(0)
    x_raw = sample_pixels(arr, n_anchor_patches=8, m_anchor_pixels=4, recon_pixels_per_patch=6, generator=g)
    g2 = torch.Generator().manual_seed(0)
    x_transformed = sample_pixels(
        arr, n_anchor_patches=8, m_anchor_pixels=4, recon_pixels_per_patch=6, generator=g2, transforms=t
    )
    # Same RNG seed -> same pixels drawn, so this is exactly `t.apply(x_raw)`.
    assert torch.allclose(x_transformed, t.apply(x_raw))
    assert not torch.allclose(x_transformed, x_raw)


def test_subsampler_wires_transforms_through():
    # A PixelSubsampler built with a non-trivial Transforms must apply it to
    # every batch it yields, not just return the raw in-RAM values.
    D = 3
    arr = np.full((10, 4, 4, D), 5.0, dtype=np.float32)
    t = Transforms(mean=np.zeros(D, dtype=np.float32), std=np.full(D, 2.0, dtype=np.float32))
    sub = PixelSubsampler(arr, recon_pixels_per_patch=4, n_anchor_patches=4, m_anchor_pixels=2, seed=0, transforms=t)
    batch = sub.sample()
    assert torch.allclose(batch, torch.full_like(batch, 2.5))  # (5 - 0) / 2


def test_subsampler_defaults_to_no_op_transform():
    # No `transforms` passed -> behaves exactly as before (raw values through).
    arr = np.random.randn(10, 4, 4, 3).astype(np.float32)
    sub = PixelSubsampler(arr, recon_pixels_per_patch=4, n_anchor_patches=4, m_anchor_pixels=2, seed=0)
    g = torch.Generator().manual_seed(0)
    expected = sample_pixels(
        torch.from_numpy(arr), n_anchor_patches=4, m_anchor_pixels=2, recon_pixels_per_patch=4, generator=g
    )
    torch.manual_seed(0)
    sub.generator.manual_seed(0)
    assert torch.allclose(sub.sample(), expected)

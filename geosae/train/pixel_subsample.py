"""Per-step reconstruction-pixel subsampler for BatchTopK training.

geosae's default trainer feeds **full 64x64 patches** (``batch_size`` patches ->
``batch_size * H * W`` pixels/step, BatchTopK over all of them). This module
instead draws a **paired pixel-subsample** per step: a few hundred anchor
patches contribute a mix of paired pixels + uniformly random pixels, giving
~78k reconstruction pixels/step, and BatchTopK selects the top ``k * Q`` latents
across exactly those ``Q`` tokens.

Per step (defaults ``pair_mode="local"``, N=512, M=8, K=128, on 64x64
patches):

1. Draw ``N`` anchor patches uniformly **with replacement** from the in-RAM
   split (``torch.randint``).
2. ``loaded_idx = unique(anchor_idx)`` -> ``P_loaded`` distinct patches.
3. Per anchor: ``M`` anchor pixels (uniform in the patch) + ``M`` p1 + ``M`` p2
   positives. ``local``: p1/p2 are truncated-Gaussian offsets (sigma, radius) of
   the anchor pixel; ``patch``: p1/p2 uniform in the patch.
4. Pixel set = ``unique(anchor u p1 u p2)`` u ``K`` uniform random pixels per
   loaded patch -> ``Q = |unique_paired| + P_loaded * K`` tokens (<= 3*N*M +
   P_loaded*K, >= P_loaded*K).
5. Return the gathered embeddings as ``(Q, 1, 1, D)`` so the pixel-independent
   SAE + per-pixel MSE chain is unchanged (H = W = 1).

Determinism: all draws come from a single persistent :class:`torch.Generator`,
seeded once and advanced across all steps (never reseeded per step).
"""

from __future__ import annotations

import numpy as np
import torch

from geosae.data.dataset import Transforms

_VALID_MODES = ("local", "patch")


def _sample_local_offsets(
    n: int,
    sigma_px: float,
    radius_px: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Truncated discrete Gaussian ``(n, 2)`` int64 ``(dy, dx)`` offsets, non-zero.

    Gaussian offsets rounded + clamped to ``[-radius, radius]``; any zero offset
    is nudged to a random unit step so a positive never coincides with its
    anchor pixel.
    """
    offs = (torch.randn(n, 2, generator=generator) * sigma_px).round().long().clamp(-int(radius_px), int(radius_px))
    zero_mask = offs.abs().sum(dim=-1) == 0
    if zero_mask.any():
        n_zero = int(zero_mask.sum().item())
        directions = torch.randint(0, 4, (n_zero,), generator=generator)
        unit = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.long)
        offs[zero_mask] = unit[directions]
    return offs


def _sample_uniform_yx(n: int, H: int, W: int, generator: torch.Generator | None) -> torch.Tensor:
    """``(n, 2)`` uniform pixel coords in ``[0, H) x [0, W)``."""
    return torch.stack(
        [
            torch.randint(0, H, (n,), generator=generator),
            torch.randint(0, W, (n,), generator=generator),
        ],
        dim=-1,
    )


def sample_pixels(
    arr: torch.Tensor,
    *,
    n_anchor_patches: int = 512,
    m_anchor_pixels: int = 8,
    recon_pixels_per_patch: int = 128,
    pair_mode: str = "local",
    local_sigma_px: float = 2.0,
    local_radius_px: int = 8,
    generator: torch.Generator | None = None,
    transforms: Transforms | None = None,
    return_patch_index: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One reconstruction-pixel subsample as ``(Q, 1, 1, D)`` float32.

    Args:
        arr: in-RAM patch tensor ``(N_patches, H, W, D)`` (any dtype; returned as
            float32). Typically ``PatchData._arr`` wrapped with ``torch.from_numpy``.
        n_anchor_patches: ``N`` anchor patches drawn per step (uniform, with
            replacement).
        m_anchor_pixels: ``M`` anchor pixels per anchor patch (+ M p1 + M p2).
        recon_pixels_per_patch: ``K`` uniform random pixels per *loaded* patch.
        pair_mode: ``"local"`` (p1/p2 are sigma-Gaussian offsets of the anchor
            pixel) or ``"patch"`` (p1/p2 uniform in the patch).
        local_sigma_px: Gaussian sigma of the local pixel offsets.
        local_radius_px: Hard radius of the local pixel offsets.
        generator: torch RNG for every draw (determinism under seed).
        transforms: Applied to the gathered pixels before they are returned, so
            the subsampler sees exactly what :meth:`PatchData.iter_batches` (and
            therefore validation) sees. ``arr`` is the *raw* in-RAM array --
            :class:`~geosae.data.dataset.PatchData` transforms it only in its
            ``__getitem__``/``iter_batches`` path, never in place -- so a caller
            that reads straight from ``arr`` (this subsampler) must apply the
            same :class:`~geosae.data.dataset.Transforms` itself or train on a
            different distribution than it validates on. Defaults to a no-op
            :class:`~geosae.data.dataset.Transforms`.
        return_patch_index: Also return the ``(Q,)`` index into ``arr`` of every
            pixel's patch. The draw is the same either way.

    Returns:
        ``(Q, 1, 1, D)`` float32 tensor, ``Q = |unique(anchor u p1 u p2)| +
        P_loaded * K`` (``P_loaded = |unique(anchor_idx)| <= N``); with
        ``return_patch_index``, the pair ``(pixels, patch_index)``.
    """
    if pair_mode not in _VALID_MODES:
        raise ValueError(f"pair_mode must be one of {_VALID_MODES}, got {pair_mode!r}")
    if arr.ndim != 4:
        raise ValueError(f"arr must be (N, H, W, D), got shape {tuple(arr.shape)}")
    N = int(n_anchor_patches)
    M = int(m_anchor_pixels)
    K = int(recon_pixels_per_patch)
    if N < 1 or M < 1 or K < 1:
        raise ValueError(f"n_anchor_patches/m_anchor_pixels/recon_pixels_per_patch must be >= 1, got {N}/{M}/{K}")
    n_patches, H, W, D = (int(s) for s in arr.shape)
    N = min(N, n_patches)
    HW = H * W

    # 1. Anchor patches (uniform, with replacement) and the deduplicated load set.
    anchor_idx = torch.randint(0, n_patches, (N,), generator=generator)
    loaded_idx, inverse = torch.unique(anchor_idx, return_inverse=True)
    local_anchor = inverse.long()  # (N,) position of each anchor within loaded_idx
    P_loaded = int(loaded_idx.numel())

    # 2. Anchor pixels (uniform in the patch), then p1 / p2 positives.
    anchor_yx = _sample_uniform_yx(N * M, H, W, generator).view(N, M, 2)
    if pair_mode == "local":
        off1 = _sample_local_offsets(N * M, local_sigma_px, local_radius_px, generator).view(N, M, 2)
        off2 = _sample_local_offsets(N * M, local_sigma_px, local_radius_px, generator).view(N, M, 2)
        p1_yx = torch.stack(
            [(anchor_yx[..., 0] + off1[..., 0]).clamp(0, H - 1), (anchor_yx[..., 1] + off1[..., 1]).clamp(0, W - 1)],
            dim=-1,
        )
        p2_yx = torch.stack(
            [(anchor_yx[..., 0] + off2[..., 0]).clamp(0, H - 1), (anchor_yx[..., 1] + off2[..., 1]).clamp(0, W - 1)],
            dim=-1,
        )
    else:  # "patch": both positives uniform within the anchor patch
        p1_yx = _sample_uniform_yx(N * M, H, W, generator).view(N, M, 2)
        p2_yx = _sample_uniform_yx(N * M, H, W, generator).view(N, M, 2)

    # 3. Flat global ids over the (P_loaded, HW) grid. Anchor/p1/p2 all address
    #    the anchor's loaded patch (local mode: cross == anchor).
    la = local_anchor.view(N, 1).expand(N, M)  # (N, M)

    def _flat(yx: torch.Tensor) -> torch.Tensor:
        return (la * HW + yx[..., 0] * W + yx[..., 1]).reshape(-1)

    unique_paired = torch.unique(torch.cat([_flat(anchor_yx), _flat(p1_yx), _flat(p2_yx)]))

    # 4. K uniform random reconstruction pixels per loaded patch.
    rand_g = torch.randint(0, P_loaded * HW, (P_loaded * K,), generator=generator, dtype=torch.long)
    all_g = torch.cat([unique_paired, rand_g])

    # 5. Gather (Q, D) directly from arr (decompose flat id -> patch, y, x) and
    #    reshape to (Q, 1, 1, D). Avoids materialising the full (P_loaded, H, W, D)
    #    flat grid; the gather touches only the Q selected pixels.
    local_patch = all_g // HW
    rem = all_g % HW
    yy = rem // W
    xx = rem % W
    global_patch = loaded_idx[local_patch]
    recon = arr[global_patch, yy, xx].to(torch.float32)  # (Q, D)
    recon = recon.view(recon.shape[0], 1, 1, D)
    if transforms is not None:
        recon = transforms.apply(recon)
    if return_patch_index:
        return recon, global_patch
    return recon


class PixelSubsampler:
    """Bind the pixel subsampler to an in-RAM patch array and one RNG stream.

    Holds a single persistent :class:`torch.Generator` (seeded once), so the
    per-step pixel sets are deterministic under the run seed and advance across
    steps.
    """

    def __init__(
        self,
        arr,
        *,
        recon_pixels_per_patch: int = 128,
        n_anchor_patches: int = 512,
        m_anchor_pixels: int = 8,
        pair_mode: str = "local",
        local_sigma_px: float = 2.0,
        local_radius_px: int = 8,
        seed: int = 42,
        transforms: Transforms | None = None,
        coordinates: np.ndarray | None = None,
    ) -> None:
        if arr is None:
            raise ValueError("PixelSubsampler needs an in-RAM patch array (compute_inram must be True)")
        # torch.from_numpy shares memory (no copy); a plain tensor is used as-is.
        self.arr = arr if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
        self.params = dict(
            recon_pixels_per_patch=int(recon_pixels_per_patch),
            n_anchor_patches=int(n_anchor_patches),
            m_anchor_pixels=int(m_anchor_pixels),
            pair_mode=str(pair_mode),
            local_sigma_px=float(local_sigma_px),
            local_radius_px=int(local_radius_px),
        )
        # ``arr`` is raw in-RAM data (see :func:`sample_pixels`'s ``transforms``
        # arg): apply the same pipeline PatchData.iter_batches applies, so
        # training and validation see the same input distribution. Defaults to
        # a no-op Transforms, matching a caller that never had one to pass.
        self.transforms = transforms if transforms is not None else Transforms()
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        # Optional (N_patches, 2) lon/lat per patch, attached to every drawn pixel.
        self.coordinates = (
            None if coordinates is None else torch.as_tensor(np.asarray(coordinates), dtype=torch.float32)
        )

    def sample(self) -> torch.Tensor:
        """Draw one ``(Q, 1, 1, D)`` reconstruction-pixel subsample."""
        return sample_pixels(self.arr, generator=self.generator, transforms=self.transforms, **self.params)

    def batches(self, steps: int):
        """Yield ``steps`` batch dicts ``{"x": (Q, 1, 1, D)}`` for the trainer.

        With ``coordinates`` set, each dict also holds every pixel's ``"lon"``/``"lat"`` ``(Q,)``.
        """
        for _ in range(int(steps)):
            if self.coordinates is None:
                yield {"x": self.sample()}
                continue
            x, patch = sample_pixels(
                self.arr, generator=self.generator, transforms=self.transforms, return_patch_index=True, **self.params
            )
            c = self.coordinates[patch]
            yield {"x": x, "lon": c[:, 0], "lat": c[:, 1]}


__all__ = ["sample_pixels", "PixelSubsampler"]

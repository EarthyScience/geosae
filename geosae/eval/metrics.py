"""Bare-minimum SAE evaluation: reconstruction fidelity + sparsity.

* **FVE** — fraction of embedding variance explained, averaged over dims.
* **recon_cosine** — mean per-pixel cosine similarity of input vs reconstruction.
* **recon_mse** — mean squared reconstruction error.
* **l0** — mean number of active latents per pixel.
* **dead_frac** — fraction of latents that never fire on the eval split.

All are computed in a single streaming pass so they scale to the full test split.
"""

from __future__ import annotations

import torch

from geosae.data.dataset import PatchData
from geosae.models.base import SparseAutoencoder, model_inputs

# Default compute device: use CUDA when present, else fall back to CPU (so the defaults are safe on
# CPU-only machines and in CI). Callers can always pass an explicit ``device``.
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def reconstruction_metrics(
    model: SparseAutoencoder,
    data: PatchData,
    *,
    batch_size: int = 64,
    device: str | torch.device = DEFAULT_DEVICE,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, float]:
    """Streaming reconstruction + sparsity metrics over ``data`` (eval mode)."""
    model.eval()
    D = model.input_dim
    M = model.latent_dim

    sum_x = torch.zeros(D, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(D, dtype=torch.float64, device=device)
    sum_res2 = torch.zeros(D, dtype=torch.float64, device=device)
    cos_sum = torch.zeros((), dtype=torch.float64, device=device)
    l0_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_pixels = 0
    ever_active = torch.zeros(M, dtype=torch.bool, device=device)

    autocast = amp_dtype is not None and str(device).startswith("cuda")
    for batch in data.iter_batches(batch_size, shuffle=False):
        inputs = model_inputs(model, batch, device)
        xp = inputs[0]  # (b, H, W, D)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast):
            recon, z, _ = model(*inputs)
        x = xp.reshape(-1, D).double()
        recon = recon.reshape(-1, D).double()
        z = z.reshape(-1, M)
        sum_x += x.sum(0)
        sum_x2 += (x * x).sum(0)
        sum_res2 += ((x - recon) ** 2).sum(0)
        xn = x.norm(dim=-1)
        rn = recon.norm(dim=-1)
        cos = (x * recon).sum(-1) / (xn * rn).clamp_min(1e-8)
        cos_sum += cos.sum()
        l0_sum += (z > 0).sum()
        ever_active |= (z > 0).any(0)
        n_pixels += x.shape[0]

    n = float(n_pixels)
    mean_x = sum_x / n
    var_x = (sum_x2 / n) - mean_x**2
    mse_dim = sum_res2 / n
    fve = 1.0 - (mse_dim / var_x.clamp_min(1e-12))
    return {
        "fve": float(fve.mean()),
        "recon_cosine": float(cos_sum / n),
        "recon_mse": float(mse_dim.mean()),
        "l0": float(l0_sum / n),
        "dead_frac": float((~ever_active).float().mean()),
        "n_pixels": int(n_pixels),
    }


@torch.no_grad()
def calibrate_threshold(
    model: SparseAutoencoder,
    data: PatchData,
    target_k: int | None = None,
    *,
    n_patches: int = 32,
    max_sample: int = 4_000_000,
    device: str | torch.device = DEFAULT_DEVICE,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> float:
    """Set the BatchTopK inference threshold so that the average L0 equals ``k``.

    At eval time BatchTopK keeps latents whose dense activation exceeds a scalar
    threshold. The EMA maintained during training can drift (peaky activations,
    warm-up), so here we set it directly: over a calibration sample of dense
    activations, the threshold is the ``1 - k/M`` quantile — i.e. on average
    ``k`` of the ``M`` latents exceed it per pixel. No-op for non-BatchTopK models.
    Dual-bank variants provide their own ``calibrate`` (per-bank thresholds).
    """
    if hasattr(model, "calibrate"):
        return model.calibrate(data, target_k, n_patches=n_patches, device=device, amp_dtype=amp_dtype)
    if not model.batch_topk:
        return float(model.min_activation)
    model.eval()
    k = int(target_k or model.k)
    acts = []
    autocast = amp_dtype is not None and str(device).startswith("cuda")
    for batch in data.iter_batches(n_patches, shuffle=False):
        inputs = model_inputs(model, batch, device)  # x: (b, H, W, D)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast):
            # The threshold acts on the activations the model gates it on.
            pre = model.preacts_for_threshold(*inputs)
        acts.append(pre.float().reshape(-1))
        break  # one calibration batch of n_patches is enough for a stable quantile
    return set_batchtopk_threshold(model, torch.cat(acts), target_k=k, max_sample=max_sample)


@torch.no_grad()
def set_batchtopk_threshold(
    model: SparseAutoencoder, acts: torch.Tensor, *, target_k: int | None = None, max_sample: int = 4_000_000
) -> float:
    """Set a BatchTopK model's eval threshold to the ``1 - k/M`` quantile of ``acts``.

    This is the definition of the threshold: keeping the activations above the
    ``1 - k/M`` quantile leaves ``k`` of the ``M`` latents active per pixel on
    average, which is the sparsity the dictionary was trained at. It is a
    **far-tail** quantile — at ``M = 4096, k = 16`` the 99.6th percentile — so
    ``acts`` must be a representative sample of the activations the model will
    be evaluated on, and a large one: estimated off a handful of correlated
    pixels the threshold lands far enough out to silence most of the dictionary.

    Args:
        model: A BatchTopK-family SAE, modified in place.
        acts: Dense pre-sparsification activations, any shape; flattened here.
        target_k: Sparsity to calibrate to; ``model.k`` when None.
        max_sample: Activations the quantile is taken over; larger samples are
            randomly subsampled to this, to bound memory.

    Returns:
        The threshold that was set.
    """
    k = int(target_k or model.k)
    flat = acts.float().reshape(-1)
    if flat.numel() > max_sample:  # subsample for a cheap, memory-safe quantile
        idx = torch.randint(0, flat.numel(), (max_sample,), device=flat.device)
        flat = flat[idx]
    thr = torch.quantile(flat, 1.0 - k / model.latent_dim)
    model.min_activation.copy_(thr.to(model.min_activation.dtype).to(model.min_activation.device))
    return float(thr)


__all__ = ["reconstruction_metrics", "calibrate_threshold", "set_batchtopk_threshold"]

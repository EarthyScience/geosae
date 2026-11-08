"""SAE training loss: MSE reconstruction + AuxK, plus the ℓ1 and Matryoshka terms a model asks for.

* **AuxK** (every model with a dead-feature mask and a linear decoder): the ``top_k_aux`` largest
  pre-activations of the dead latents reconstruct the detached reconstruction residual, which is the
  only gradient that can revive a latent that has stopped firing.
* **ℓ1** (``l1_coeff > 0``): the ReLU+ℓ1 model's only sparsity mechanism.
* **Matryoshka** (``slow_group_weight > 0`` on a model with ``is_matryoshka``): each nested prefix of
  the dictionary reconstructs the input on its own.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from geosae.models.base import SparseAutoencoder


def _auxk(model: SparseAutoencoder, target, recon, z_dense, aux_penalty: float, top_k_aux: int):
    """AuxK term: dead latents explain the (detached) reconstruction residual."""
    if aux_penalty <= 0 or not hasattr(model, "decoder"):
        return torch.zeros((), device=z_dense.device), 0.0, 0
    dead = model.dead_features_mask()
    n_dead = int(dead.sum())
    k_aux = min(int(top_k_aux), n_dead)
    if k_aux <= 0:
        return torch.zeros((), device=z_dense.device), 0.0, n_dead
    z_dead = z_dense[..., dead]
    vals, idx = torch.topk(z_dead, k_aux, dim=-1)
    z_aux = torch.zeros_like(z_dense)
    z_aux[..., dead] = torch.zeros_like(z_dead).scatter_(-1, idx, vals)
    e_hat = model.decoder(z_aux)
    if getattr(model, "rescale_reconstruction", False):
        e_hat = e_hat / model.input_scale(target)  # codes live in the normalised space
    aux = F.mse_loss(e_hat, (target - recon).detach())
    return aux, float(aux.detach()), n_dead


def sae_loss(
    model: SparseAutoencoder,
    target: torch.Tensor,
    recon: torch.Tensor,
    z: torch.Tensor,
    z_dense: torch.Tensor,
    *,
    aux_penalty: float = 0.03125,
    top_k_aux: int = 32,
    l1_coeff: float = 0.0,
    slow_group_weight: float = 0.0,
    penalty_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(loss, metrics)`` for one batch.

    Args:
        model: the SAE (for the decoder and the dead-feature mask).
        target: ground-truth embedding ``(..., D)``.
        recon: reconstruction, output of ``model.forward``.
        z: sparse latent codes, output of ``model.forward``.
        z_dense: dense (pre-top-k) latent codes, output of ``model.forward``.
        aux_penalty: weight of the AuxK term (0 disables it).
        top_k_aux: number of dead latents used for the auxiliary reconstruction.
        l1_coeff: ℓ1 sparsity penalty on ``z`` (``||z||_1`` summed over features,
            averaged over pixels). The ReLU+ℓ1 variant's only sparsity mechanism;
            0 for the top-k variants.
        slow_group_weight: weight of the Matryoshka slow-group prefix
            reconstruction ``MSE(Dec([z[..., :n_prefix]; 0]), target)``, added when
            the model is a Matryoshka SAE (``is_matryoshka`` and a proper prefix);
            a model with nested ``prefix_sizes`` adds one such term per prefix.
        penalty_scale: multiplier on the ℓ1 penalty; the trainer ramps it from 0 to 1
            to warm the penalty up. The logged ``l1`` stays unscaled.
    """
    metrics = {"l0": float((z > 0).float().sum(-1).mean().detach())}
    recon_mse = F.mse_loss(recon, target)
    metrics["recon_mse"] = float(recon_mse.detach())
    loss = recon_mse

    aux, aux_val, n_dead = _auxk(model, target, recon, z_dense, aux_penalty, top_k_aux)
    loss = loss + aux_penalty * aux
    metrics["aux"] = aux_val
    metrics["n_dead"] = n_dead

    # ReLU+ℓ1: sparsity comes entirely from the ℓ1 penalty on z.
    if l1_coeff > 0:
        l1 = z.abs().sum(-1).mean()
        loss = loss + penalty_scale * l1_coeff * l1
        metrics["l1"] = float(l1.detach())

    # Matryoshka: extra prefix-reconstruction term on the high ("slow") group.
    if slow_group_weight > 0 and getattr(model, "is_matryoshka", False):
        sizes = getattr(model, "prefix_sizes", None) or (int(getattr(model, "n_prefix", model.latent_dim)),)
        slow = sum(
            F.mse_loss(model.decode(z * (_latent_index(z) < n)), target) for n in sizes if 0 < n < model.latent_dim
        )
        if isinstance(slow, torch.Tensor):
            loss = loss + slow_group_weight * slow
            metrics["slow_group"] = float(slow.detach())
    return loss, metrics


def _latent_index(z: torch.Tensor) -> torch.Tensor:
    """Index of every latent along the last axis of ``z``, broadcastable against it."""
    return torch.arange(z.shape[-1], device=z.device)


__all__ = ["sae_loss"]

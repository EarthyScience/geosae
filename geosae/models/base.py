"""The contract every sparse autoencoder of this package implements.

A model maps embeddings ``(..., D)`` to a sparse code ``(..., M)`` over a dictionary of ``M`` atoms and
back. The trainer, the loss, the predictor and the evaluation suite only rely on what
:class:`SparseAutoencoder` declares, so a new architecture plugs in by subclassing it and registering a
builder with :func:`geosae.models.register_model`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class SparseAutoencoder(nn.Module, ABC):
    """Base class of a single-layer sparse autoencoder ``x -> z -> x_hat``.

    Subclasses set ``input_dim``, ``latent_dim``, ``n_batches_to_dead`` and the buffers
    ``num_batches_not_active`` ``(M,)``, ``min_activation`` ``()`` (the eval-time activation threshold;
    ``0`` when the model has none) and ``n_high`` (the size of the Matryoshka prefix, ``M`` when there
    is none), and implement :meth:`encode`, :meth:`decode` and :meth:`forward`.

    Attributes:
        batch_topk: Whether inference uses a calibrated activation threshold
            (:func:`geosae.eval.metrics.calibrate_threshold`) instead of a per-row selection.
        k: Target number of active latents per row, or ``None`` when sparsity is emergent.
        uses_coordinates: Whether :meth:`forward` takes the pixel coordinates as a second argument
            (see :func:`model_inputs`).
    """

    batch_topk: bool = False
    k: int | None = None
    uses_coordinates: bool = False

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dense pre-activation latents ``(..., M)`` of ``x`` ``(..., D)``."""

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction ``(..., D)`` of latents ``z`` ``(..., M)``."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(recon, z, z_dense)``: reconstruction, sparse code and dense pre-activations."""

    def preacts_for_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """The dense activations an eval-time threshold acts on (default: :meth:`encode`)."""
        return self.encode(x)

    @property
    def n_prefix(self) -> int:
        """Size of the Matryoshka prefix (``== latent_dim`` when there is none)."""
        return int(self.n_high)

    def dictionary(self) -> np.ndarray:
        """The decoder atoms as ``(M, D)`` float64 rows, one row per feature."""
        decoder = getattr(self, "decoder", None)
        weight = decoder.weight if decoder is not None else self.encoder.weight.t()
        return weight.detach().cpu().numpy().T.astype(np.float64).copy()

    @torch.no_grad()
    def _update_dead_features(self, z: torch.Tensor) -> None:
        """Increment the inactive counter of latents that did not fire in ``z``; reset the others."""
        fired = (z > 0).any(dim=tuple(range(z.ndim - 1)))
        self.num_batches_not_active += (~fired).float()
        self.num_batches_not_active[fired] = 0

    def dead_features_mask(self) -> torch.Tensor:
        """Boolean ``(M,)`` mask of latents that have not fired for ``n_batches_to_dead`` batches."""
        return self.num_batches_not_active >= self.n_batches_to_dead


def model_inputs(model: nn.Module, batch: dict, device: str | torch.device) -> tuple[torch.Tensor, ...]:
    """The positional arguments ``model`` takes for a data batch, on ``device``.

    A model with ``uses_coordinates`` gets ``(x, coords)`` with ``coords = stack([lon, lat], -1)`` from the
    batch; every other model gets ``(x,)``.

    Args:
        model: Any model of this package.
        batch: A batch dict with ``"x"`` and, for a location-aware model, ``"lon"`` and ``"lat"``.
        device: Target device.

    Returns:
        The tuple to unpack into ``model(...)`` or ``model.preacts_for_threshold(...)``.
    """
    x = batch["x"].to(device, non_blocking=True)
    if not getattr(model, "uses_coordinates", False):
        return (x,)
    coords = torch.stack([batch["lon"], batch["lat"]], dim=-1).to(device, non_blocking=True)
    return x, coords


def sparse_codes(model: Any, x: torch.Tensor) -> torch.Tensor:
    """The sparse code ``(n, M)`` a model assigns to embeddings ``x`` ``(n, D)``.

    Every model returns ``(recon, z, z_dense)`` from ``forward``; a top-k model sparsifies after
    :meth:`~SparseAutoencoder.encode` while a ReLU model's ``encode`` returns pre-activations, so the
    sparse code is read from ``forward`` rather than from either half.
    """
    return model(x)[1]


__all__ = ["SparseAutoencoder", "model_inputs", "sparse_codes"]

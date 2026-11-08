"""ReLU + ℓ1 Sparse Autoencoder (Anthropic "Towards Monosemanticity", 2023).

A single linear encoder followed by a ReLU produces the sparse latent; sparsity
comes from an ℓ1 penalty on ``z`` applied in the loss (:mod:`geosae.train.loss`) —
there is no top-k selection and no learned threshold. Tune the ℓ1 penalty per
``(M, k)`` to land the average L0 in the desired band.

It exposes the same forward contract as :class:`~geosae.models.topk_sae.TopKSAE`
(``forward(x) -> (recon, z, z_dense)``) plus the attributes the eval/training
code reads (``batch_topk``, ``k``, ``min_activation``, ``dead_features_mask``,
``preacts_for_threshold``) so it drops into the existing trainer, metrics, and
predictor unchanged.

Differences from the top-k backbone:

* ``z = relu(z_dense)`` — the activation, not a selection, drives sparsity.
* ``normalize_inputs`` defaults to **True**: inputs are scaled by
  :meth:`input_scale` so the batch's mean L2 norm is ``sqrt(D)`` before
  encoding (Conerly et al. 2024). With ``rescale_reconstruction`` the
  reconstruction is scaled back to input units, so ``forward`` is equivariant
  to the input's scale and the codes (hence the ℓ1 penalty) are in normalised
  units. Without it (the default) the reconstruction is not scaled back: the
  encoder sees the scaled input, the loss the raw one, and the pre-encoder bias
  is subtracted in one space and added back in the other.
* No BatchTopK EMA threshold; ``min_activation`` is a fixed 0 kept only so the
  shared :func:`~geosae.eval.metrics.calibrate_threshold` is a no-op.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from geosae.models.base import SparseAutoencoder


class ReLUSAE(SparseAutoencoder):
    """Single-layer ReLU + ℓ1 sparse autoencoder.

    Args:
        input_dim: Embedding dimensionality ``D``.
        latent_dim: Dictionary size ``M``.
        output_dim: Decoder output dim (defaults to ``input_dim``).
        pre_bias: Subtract a learned ``b_dec`` before encoding and add it back
            after decoding (Anthropic convention). The decoder then carries no
            bias of its own; ``b_dec`` is the sole reconstruction bias.
        normalize_decoder: Renormalise decoder columns to unit L2 norm before
            each training forward (disabled automatically when ``tied_decoder``).
        normalize_inputs: Scale inputs so the batch's mean L2 norm equals
            ``sqrt(D)`` before encoding.
        rescale_reconstruction: With ``normalize_inputs``, divide the
            reconstruction by the same scale, so the model encodes and decodes
            in the normalised space and :meth:`forward` returns input units.
            ``False`` keeps the non-equivariant form.
        tied_decoder: Use ``W_enc.T`` as the decoder instead of a separate matrix
            (a learned ``b_dec`` is always present in this mode).
        init_encoder_from_decoder: Initialise ``W_enc = W_dec.T`` (untied only).
        n_batches_to_dead: A latent is "dead" after this many consecutive batches
            without firing (used by AuxK diagnostics; ReLU runs use no aux loss).
        h_frac: Fraction of latents in the high ("slow") Matryoshka prefix.
            ``1.0`` (the default) puts every latent in the high partition.
            Stored as ``n_high``/``n_low`` buffers.
        seed: Optional weight-init seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        output_dim: int | None = None,
        pre_bias: bool = False,
        normalize_decoder: bool = True,
        normalize_inputs: bool = True,
        rescale_reconstruction: bool = False,
        tied_decoder: bool = False,
        init_encoder_from_decoder: bool = True,
        n_batches_to_dead: int = 25,
        h_frac: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim) if output_dim is not None else self.input_dim
        self.pre_bias = bool(pre_bias)
        self.tied_decoder = bool(tied_decoder)
        # Tying + per-step column renorm would double-mutate the encoder weight;
        # disable renorm in the tied case.
        self.normalize_decoder = bool(normalize_decoder) and not self.tied_decoder
        self.normalize_inputs = bool(normalize_inputs)
        self.rescale_reconstruction = bool(rescale_reconstruction) and self.normalize_inputs
        self.n_batches_to_dead = int(n_batches_to_dead)
        self.h_frac = float(h_frac)
        self.seed = None if seed is None else int(seed)
        # API-compat with the top-k eval/predict path: ReLU has no top-k budget
        # and no BatchTopK threshold. ``batch_topk=False`` makes
        # ``calibrate_threshold`` a no-op; ``k`` is emergent (set by the ℓ1 tune).
        self.batch_topk = False
        self.k: int | None = None

        if self.pre_bias and self.input_dim != self.output_dim:
            raise ValueError(f"pre_bias requires input_dim == output_dim (got {self.input_dim} != {self.output_dim})")

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        if self.tied_decoder:
            if self.input_dim != self.output_dim:
                raise ValueError("tied_decoder requires input_dim == output_dim")
            self.decoder = None
            self.b_dec = nn.Parameter(torch.zeros(self.output_dim))
        else:
            # Under pre_bias, ``b_dec`` plays both roles, so the decoder Linear
            # carries no bias of its own.
            self.decoder = nn.Linear(self.latent_dim, self.output_dim, bias=not self.pre_bias)
            if self.pre_bias:
                self.b_dec = nn.Parameter(torch.zeros(self.output_dim))
            if init_encoder_from_decoder:
                with torch.no_grad():
                    self.encoder.weight.copy_(self.decoder.weight.t().contiguous())

        if self.normalize_decoder:
            self._normalize_decoder_columns()
            if init_encoder_from_decoder and not self.tied_decoder:
                with torch.no_grad():
                    self.encoder.weight.copy_(self.decoder.weight.t().contiguous())

        # Fixed 0 threshold (ReLU has no BatchTopK EMA); kept for eval-API parity.
        self.register_buffer("min_activation", torch.zeros((), dtype=torch.float32))
        self.register_buffer("num_batches_not_active", torch.zeros(self.latent_dim))
        if self.h_frac >= 1.0:
            n_high = self.latent_dim
        else:
            n_high = max(1, min(self.latent_dim - 1, round(self.h_frac * self.latent_dim)))
        self.register_buffer("n_high", torch.tensor(n_high, dtype=torch.long))
        self.register_buffer("n_low", torch.tensor(self.latent_dim - n_high, dtype=torch.long))

    # ------------------------------------------------------------------ helpers
    def input_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar that maps ``x`` into the normalised space (``1`` when disabled).

        Args:
            x: A batch ``(..., D)`` in input units.

        Returns:
            A 0-d tensor ``s`` with mean row norm of ``s * x`` equal to ``sqrt(D)``.
        """
        if not self.normalize_inputs:
            return torch.ones((), dtype=x.dtype, device=x.device)
        return math.sqrt(self.input_dim) / (x.norm(dim=-1).mean() + 1e-8)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.input_scale(x) if self.normalize_inputs else x

    @torch.no_grad()
    def _normalize_decoder_columns(self) -> None:
        if self.decoder is None:
            return
        w = self.decoder.weight  # (D, M): column j is feature j's direction
        w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))

    # -------------------------------------------------------------- encode/decode
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dense pre-activation latents ``(..., M)`` (before ReLU), in normalised units."""
        x = self._normalize(x)
        if self.pre_bias:
            x = x - self.b_dec
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(..., D)`` from latents ``z``.

        With ``rescale_reconstruction`` this is the normalised space: divide by
        :meth:`input_scale` of the encoded batch to return to input units, as
        :meth:`forward` does.
        """
        if self.tied_decoder:
            return F.linear(z, self.encoder.weight.t(), self.b_dec)
        out = self.decoder(z)
        if self.pre_bias:
            out = out + self.b_dec
        return out

    def preacts_for_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """Activations the eval threshold would act on (ReLU-gated dense latents)."""
        return F.relu(self.encode(x))

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(recon, z, z_dense)`` for input ``x`` of shape ``(..., D)``."""
        if self.training and self.normalize_decoder:
            self._normalize_decoder_columns()
        z_dense = self.encode(x)
        z = F.relu(z_dense)
        recon = self.decode(z)
        if self.rescale_reconstruction:
            recon = recon / self.input_scale(x)
        if self.training:
            self._update_dead_features(z)
        return recon, z, z_dense

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"pre_bias={self.pre_bias}, tied_decoder={self.tied_decoder}, "
            f"normalize_decoder={self.normalize_decoder}, normalize_inputs={self.normalize_inputs}, "
            f"rescale_reconstruction={self.rescale_reconstruction}"
        )


__all__ = ["ReLUSAE"]

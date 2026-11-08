"""Top-K / BatchTopK Sparse Autoencoder.

A single-layer linear SAE ``x -> z -> x_hat`` with top-k activation sparsity.
Works on inputs of arbitrary leading shape ``(..., D)`` — for MajorTOM patches
the input is ``(B, H, W, D)`` and every pixel is encoded independently.

Two sparsity modes:

* ``batch_topk=False`` — per-sample: keep the ``k`` largest latents of every row.
* ``batch_topk=True``  — keep the ``k * n_rows`` largest latents across the whole
  batch (so sparsity is ``k`` *on average*). At eval time the per-batch selection
  is replaced by a fixed threshold (an EMA of the smallest kept activation),
  making inference batch-independent.

The reconstruction/aux losses live in :mod:`geosae.train.loss`; this module only
owns the architecture and the dead-feature bookkeeping used for AuxK resampling.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from geosae.models.base import SparseAutoencoder


class TopKSAE(SparseAutoencoder):
    """Single-layer Top-K sparse autoencoder.

    Args:
        input_dim: Embedding dimensionality ``D`` (e.g. 64 for AlphaEarth).
        latent_dim: Dictionary size ``M`` (number of features).
        k: Active latents per row (per batch-average when ``batch_topk``).
        batch_topk: Select top-k across the batch rather than per row.
        pre_bias: Subtract a learned ``b_dec`` before encoding and add it back
            after decoding (Anthropic convention). The decoder then carries no
            bias of its own.
        normalize_decoder: Renormalise decoder columns to unit L2 norm before
            each training step, so every feature is a unit dictionary atom.
        init_encoder_from_decoder: Initialise ``W_enc = W_dec.T``.
        n_batches_to_dead: A latent is "dead" after this many consecutive
            batches without firing (used by AuxK).
        min_activation_beta: EMA decay for the eval-time BatchTopK threshold.
        freeze_decoder: Replace the decoder with a **fixed random Gaussian
            dictionary** (unit-norm columns) and freeze it together with the
            reconstruction biases, so only the encoder trains. This is the
            frozen-random-decoder control: any interpretability metric that does
            not cleanly rank a *trained* decoder above this random one is not
            measuring interpretability.
            The unit-sphere columns match the isotropic covariance of
            L2-normalised AlphaEarth embeddings; per-step decoder renormalisation
            is disabled so the frozen weights never move.
        seed: If not ``None``, seed the weight initialisation (encoder/decoder,
            and the frozen random dictionary) so multi-seed runs are reproducible
            and the frozen control's random dictionary is deterministic.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int,
        *,
        batch_topk: bool = True,
        pre_bias: bool = True,
        normalize_decoder: bool = True,
        init_encoder_from_decoder: bool = True,
        n_batches_to_dead: int = 25,
        min_activation_beta: float = 0.999,
        normalize_inputs: bool = False,
        h_frac: float = 1.0,
        freeze_decoder: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.k = int(k)
        self.batch_topk = bool(batch_topk)
        self.pre_bias = bool(pre_bias)
        self.normalize_decoder = bool(normalize_decoder)
        self.normalize_inputs = bool(normalize_inputs)
        self.min_activation_beta = float(min_activation_beta)
        self.n_batches_to_dead = int(n_batches_to_dead)
        self.freeze_decoder = bool(freeze_decoder)
        self.seed = None if seed is None else int(seed)
        # Contiguous high/low feature partition (Matryoshka prefix). ``h_frac``
        # is the fraction of latents in the high ("slow") prefix; the Matryoshka
        # slow-group loss reconstructs from ``z[..., :n_high]``. ``h_frac == 1.0``
        # (the plain-BatchTopK default) puts every feature in the high partition
        # (``n_low == 0``). Stored as buffers so checkpoints round-trip the split.
        self.h_frac = float(h_frac)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=not self.pre_bias)
        if self.pre_bias:
            self.b_dec = nn.Parameter(torch.zeros(self.input_dim))

        # Eval-time BatchTopK threshold (EMA of the smallest kept activation).
        # ``-1`` marks "uninitialised" so the first training batch seeds it
        # directly (rather than crawling up from 0 under the slow EMA).
        self.register_buffer("min_activation", torch.full((), -1.0, dtype=torch.float32))
        # Consecutive-batches-without-firing counter, one per latent.
        self.register_buffer("num_batches_not_active", torch.zeros(self.latent_dim))
        # High/low feature partition (Matryoshka prefix). ``n_high`` = size of the
        # high ("slow") prefix; ``round(h_frac*M)`` clamped to ``[1, M-1]`` unless
        # ``h_frac >= 1`` (then ``n_high == M``).
        if self.h_frac >= 1.0:
            n_high = self.latent_dim
        else:
            n_high = max(1, min(self.latent_dim - 1, round(self.h_frac * self.latent_dim)))
        self.register_buffer("n_high", torch.tensor(n_high, dtype=torch.long))
        self.register_buffer("n_low", torch.tensor(self.latent_dim - n_high, dtype=torch.long))

        self._init_weights(init_encoder_from_decoder)
        if self.freeze_decoder:
            self._init_frozen_decoder(init_encoder_from_decoder)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def _init_weights(self, init_encoder_from_decoder: bool) -> None:
        if self.normalize_decoder:
            self._normalize_decoder_columns()
        if init_encoder_from_decoder:
            self.encoder.weight.copy_(self.decoder.weight.t().contiguous())
            self.encoder.bias.zero_()

    @torch.no_grad()
    def _normalize_decoder_columns(self) -> None:
        """Renormalise each decoder column (feature direction) to unit L2 norm."""
        w = self.decoder.weight  # (D, M): column j is feature j's direction
        w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def _init_frozen_decoder(self, init_encoder_from_decoder: bool) -> None:
        """Replace the decoder with a fixed random unit-norm dictionary and freeze it.

        The frozen-random-decoder control. Decoder columns (feature directions)
        become i.i.d. standard-Gaussian vectors normalised to unit L2 norm — a
        random dictionary on the unit sphere — then frozen. ``b_dec`` / the
        decoder bias stay at their zero init and are frozen too, so the decode
        mapping is fully fixed and only the encoder learns. The encoder is
        re-tied to the new frozen dictionary (``W_enc = W_dec.T``) for a warm
        start when ``init_encoder_from_decoder`` is set.
        """
        w = self.decoder.weight  # (D, M)
        rand = torch.randn_like(w)
        rand.div_(rand.norm(dim=0, keepdim=True).clamp_min(1e-8))
        w.copy_(rand)
        w.requires_grad_(False)
        if self.decoder.bias is not None:
            self.decoder.bias.zero_()
            self.decoder.bias.requires_grad_(False)
        if self.pre_bias:
            self.b_dec.zero_()
            self.b_dec.requires_grad_(False)
        if init_encoder_from_decoder:
            self.encoder.weight.copy_(w.t().contiguous())
            self.encoder.bias.zero_()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` so its mean L2 norm equals ``sqrt(input_dim)``.

        No-op when ``normalize_inputs`` is False (the default for embeddings
        that are already unit-norm). The scale is a single scalar derived from
        the batch's mean row norm.
        """
        if not self.normalize_inputs:
            return x
        target = math.sqrt(self.input_dim)
        mean_norm = x.norm(dim=-1).mean()
        return x * (target / (mean_norm + 1e-8))

    # -------------------------------------------------------------- encode/decode
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dense pre-activation latents ``(..., M)`` (before top-k).

        Applies input normalization, then subtracts the pre-encoder bias
        ``b_dec`` (if enabled), then the linear encoder.
        """
        x = self._normalize(x)
        if self.pre_bias:
            x = x - self.b_dec
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(..., D)`` from (possibly sparse) latents ``z``."""
        out = self.decoder(z)
        if self.pre_bias:
            out = out + self.b_dec
        return out

    def apply_topk(self, z_dense: torch.Tensor) -> torch.Tensor:
        """Sparsify dense latents. Supports any leading shape ``(..., M)``."""
        if not self.batch_topk:
            vals, idx = torch.topk(z_dense, self.k, dim=-1)
            return torch.zeros_like(z_dense).scatter_(-1, idx, vals)

        if self.training:
            n_rows = z_dense[..., 0].numel()
            flat = z_dense.reshape(-1)
            vals, idx = torch.topk(flat, self.k * n_rows)
            z = torch.zeros_like(flat).scatter_(0, idx, vals).view_as(z_dense)
            self._update_min_activation(z)
            return z
        # Eval: fixed threshold instead of a batch-dependent selection.
        return z_dense * (z_dense > self.min_activation)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(recon, z, z_dense)`` for input ``x`` of shape ``(..., D)``."""
        if self.training and self.normalize_decoder and not self.freeze_decoder:
            self._normalize_decoder_columns()
        z_dense = self.encode(x)
        z = self.apply_topk(z_dense)
        recon = self.decode(z)
        if self.training:
            self._update_dead_features(z)
        return recon, z, z_dense

    # -------------------------------------------------------------- bookkeeping
    @torch.no_grad()
    def _update_min_activation(self, z: torch.Tensor) -> None:
        """EMA of the smallest positive kept activation (BatchTopK eval threshold).

        Uses ``masked_fill(+inf).min()`` rather than boolean indexing so it never
        materialises a ``nonzero`` index tensor (which overflows past 2**31 elems
        at large batch × latent sizes).
        """
        masked = z.masked_fill(z <= 0, float("inf"))
        m = masked.min()
        if torch.isinf(m):
            return  # nothing positive this batch; leave the threshold unchanged
        m = m.detach().float()
        if self.min_activation < 0:  # uninitialised: seed directly
            self.min_activation.copy_(m)
        else:
            self.min_activation.mul_(self.min_activation_beta).add_((1.0 - self.min_activation_beta) * m)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, latent_dim={self.latent_dim}, k={self.k}, "
            f"batch_topk={self.batch_topk}, pre_bias={self.pre_bias}, "
            f"normalize_decoder={self.normalize_decoder}, freeze_decoder={self.freeze_decoder}"
        )


__all__ = ["TopKSAE"]

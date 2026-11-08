"""OlmoEarth (AI2) backbone turning Sentinel-2 L2A reflectance into a spatial token grid."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from geosae.data.embed.base import S2_REFLECTANCE_SCALE, EmbeddingBackbone

#: Encoder hidden width per released checkpoint size (verified against the installed
#: ``olmoearth_pretrain_minimal.olmoearth_pretrain_v1.olmoearth_pretrain_v1.MODEL_SIZE_CONFIGS``).
_LATENT_DIM: dict[str, int] = {"nano": 128, "tiny": 192, "base": 768, "large": 1024}

#: OlmoEarth v1 pretraining took "a random square crop from the input with side
#: length in tokens in the range {1...12}" (arXiv:2511.13655, Sec. 3.1). The
#: encoder's absolute 2D sin-cos position encoding is recomputed on the fly for
#: whatever token-grid size it is asked to cover, so it does not hard-error past
#: 12 tokens/side, but tokens at row and column indices >= 12 sit outside the
#: range the position encoding was ever trained against; when a token is beyond
#: that range on *both* axes at once the two out-of-distribution position
#: embeddings compound, and the token's mean feature diverges sharply from its
#: in-range neighbours (measured relative deviation from the grand-mean token
#: jumps from ~0.4 in-range to ~0.6-0.9 in that corner, on real Sentinel-2
#: patches). :class:`OlmoEarthBackbone` therefore refuses a configuration whose
#: grid would exceed this, rather than silently returning degraded tokens. The
#: remedy is to reduce the *input extent*, because ``patch_size`` cannot be
#: coarsened past the checkpoint's 8 px ceiling: at that ceiling the largest
#: single pass covers ``8 * MAX_TRAINED_GRID_TOKENS`` pixels.
MAX_TRAINED_GRID_TOKENS = 12

#: Largest input extent, in pixels, one forward pass can cover: the checkpoint's
#: FlexiViT patch-size ceiling times the pretrained token budget.
MAX_PATCH_SIZE_PX = 8
MAX_SINGLE_PASS_PX = MAX_PATCH_SIZE_PX * MAX_TRAINED_GRID_TOKENS


class OlmoEarthBackbone(EmbeddingBackbone):
    """AI2 OlmoEarth v1 encoder run on Sentinel-2 L2A reflectance patches.

    Loads a released OlmoEarth v1 checkpoint (package ``olmoearth-pretrain-minimal``,
    the ``embed`` install extra) from Hugging Face lazily, on first :meth:`embed` call,
    so constructing the backbone needs neither torch nor a network connection. The
    encoder's FlexiViT tokenizer emits one token per ``patch_size`` x ``patch_size``
    input pixels, per timestep and per Sentinel-2 band-set (10 m/20 m/60 m bands are
    tokenized separately); :meth:`embed` mean-reduces over the time and band-set axes
    and keeps the spatial token grid, since the SAE trains on per-token features.

    The package's shipped ``Normalizer`` uses per-band mean/std statistics computed on
    raw Sentinel-2 L2A digital numbers (its config means are in the ~1000-3000 range),
    not on reflectance scaled to ``[0, 1]``. This class therefore rescales the
    ``[0, ~1]`` reflectance input back to DN units (multiplying by
    :data:`S2_REFLECTANCE_SCALE`) before normalizing.

    The token grid may not exceed :data:`MAX_TRAINED_GRID_TOKENS` on either axis
    (see that constant): a larger grid pushes the position encoding outside its
    pretrained range, so both :meth:`output_grid` and :meth:`embed` raise rather
    than return tokens that look valid and are not. Because ``patch_size`` is
    itself capped at :data:`MAX_PATCH_SIZE_PX`, the remedy is always to reduce
    the input extent to at most :data:`MAX_SINGLE_PASS_PX` pixels — for the 10 m
    Sentinel-2 grid, a 960 m footprint. Pass ``crop_px`` to
    :func:`~geosae.data.embed.store.embed_store` to do that explicitly, so the
    ground area an embedding covers is recorded rather than assumed.

    Attributes:
        variant: Released checkpoint size (``nano``, ``tiny``, ``base``, ``large``).
        device: torch device string the model runs on.
        batch_size: Patches per internal forward pass; ``None`` runs everything in
            one batch.
    """

    _VARIANT_MODEL_IDS: ClassVar[dict[str, str]] = {
        "nano": "OLMOEARTH_V1_NANO",
        "tiny": "OLMOEARTH_V1_TINY",
        "base": "OLMOEARTH_V1_BASE",
        "large": "OLMOEARTH_V1_LARGE",
    }

    def __init__(
        self,
        variant: str = "base",
        patch_size: int = 4,
        device: str = "cuda",
        batch_size: int | None = None,
    ) -> None:
        """Configure the backbone; no weights are loaded until :meth:`embed` runs.

        Args:
            variant: Released checkpoint size, one of ``nano``, ``tiny``, ``base``,
                ``large``.
            patch_size: FlexiViT tokenizer patch size in input pixels (1-8).
            device: torch device string the model is moved to on first use.
            batch_size: Patches per internal forward pass; ``None`` runs one batch.

        Raises:
            ValueError: If ``variant`` is not a released checkpoint size.
        """
        if variant not in _LATENT_DIM:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(_LATENT_DIM)}")
        self.variant = variant
        self.patch_size = patch_size
        self.device = device
        self.batch_size = batch_size
        self.latent_dim = _LATENT_DIM[variant]
        self.stride = patch_size
        # Verified via Modality.SENTINEL2_L2A.band_order in the installed package.
        self.band_order: tuple[str, ...] = (
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        )
        self._model = None

    def _load_model(self):
        """Download (if needed) and cache the checkpoint on first call."""
        if self._model is not None:
            return self._model
        from olmoearth_pretrain_minimal import ModelID, load_model_from_id

        model_id = getattr(ModelID, self._VARIANT_MODEL_IDS[self.variant])
        model = load_model_from_id(model_id, load_weights=True)
        model.eval()
        model.to(self.device)
        self._model = model
        return model

    def output_grid(self, patch_size: int) -> int:
        """Token-grid side length for a square input patch of ``patch_size`` pixels.

        Args:
            patch_size: Side length of the square input patch, in pixels.

        Returns:
            Token-grid side length.

        Raises:
            ValueError: If ``patch_size`` is not a multiple of the stride, or if
                the grid would exceed :data:`MAX_TRAINED_GRID_TOKENS`.
        """
        grid = super().output_grid(patch_size)
        if grid > MAX_TRAINED_GRID_TOKENS:
            usable = min(self.stride, MAX_PATCH_SIZE_PX) * MAX_TRAINED_GRID_TOKENS
            raise ValueError(
                f"a {patch_size} px patch at stride {self.stride} needs a {grid}x{grid} token grid, "
                f"beyond the {MAX_TRAINED_GRID_TOKENS} tokens per side OlmoEarth's position encoding "
                f"was pretrained on. Reduce the input extent to at most {usable} px "
                f"(e.g. embed_store(crop_px={usable})); patch_size cannot be coarsened past "
                f"{MAX_PATCH_SIZE_PX} px, so a larger extent has no single-pass configuration."
            )
        return grid

    def _checked(self, array: np.ndarray) -> np.ndarray:
        """Return ``array`` if its token grid is within the pretrained range.

        Args:
            array: ``(B, H, W, C)`` array with ``H == W``.

        Raises:
            ValueError: If the patch is not square, or its grid is too large.
        """
        height, width = array.shape[1], array.shape[2]
        if height != width:
            raise ValueError(f"expected a square patch, got {height}x{width}")
        self.output_grid(height)
        return array

    def embed(self, reflectance: np.ndarray) -> np.ndarray:
        """Encode a batch of reflectance patches with the OlmoEarth v1 encoder.

        Args:
            reflectance: ``(B, H, W, 12)`` float32 surface reflectance in
                ``[0, ~1]``, bands ordered as
                :data:`geosae.data.embed.base.S2_L2A_BANDS`.

        Returns:
            ``(B, G, G, latent_dim)`` float32 token grid, ``G = H // stride``.

        Raises:
            ValueError: If the resulting grid exceeds
                :data:`MAX_TRAINED_GRID_TOKENS` per side.
        """
        # Validate the geometry before importing torch or fetching weights: a
        # misconfigured grid should cost nothing to discover.
        self._checked(reflectance)

        import torch
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
            MaskValue,
        )

        model = self._load_model()
        normalizer = Normalizer(std_multiplier=2.0)
        selected = self.select_bands(reflectance)
        digital_numbers = selected.astype(np.float32) * S2_REFLECTANCE_SCALE

        n = digital_numbers.shape[0]
        chunk = self.batch_size or n
        outputs = []
        for start in range(0, n, chunk):
            batch = digital_numbers[start : start + chunk]
            # (b, H, W, C) -> (b, H, W, T=1, C): OlmoEarth always expects a time axis.
            normalized = normalizer.normalize(Modality.SENTINEL2_L2A, batch[:, :, :, None, :])
            sentinel2 = torch.from_numpy(normalized.astype(np.float32)).to(self.device)
            mask = torch.full(sentinel2.shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.float32, device=self.device)
            # No per-patch acquisition date is available at this call boundary; a
            # fixed placeholder timestamp only shifts the temporal embedding, which
            # is a no-op at T=1 since there is nothing to distinguish it from.
            timestamps = torch.zeros((sentinel2.shape[0], 1, 3), dtype=torch.long, device=self.device)
            sample = MaskedOlmoEarthSample(
                timestamps=timestamps,
                sentinel2_l2a=sentinel2,
                sentinel2_l2a_mask=mask,
            )
            with torch.inference_mode():
                out = model.encoder(sample, fast_pass=True, patch_size=self.patch_size)
            # tokens: (b, H', W', T, S, D) -> mean over time (T) and band-set (S).
            tokens = out["tokens_and_masks"].sentinel2_l2a
            pooled = tokens.mean(dim=(3, 4))
            outputs.append(pooled.to(torch.float32).cpu().numpy())
        return np.concatenate(outputs, axis=0)


__all__ = ["OlmoEarthBackbone"]

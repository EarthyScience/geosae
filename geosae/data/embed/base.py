"""Backbone interface turning Sentinel-2 reflectance patches into embedding token grids."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: Canonical Sentinel-2 L2A band order used by the ``s2`` store array.
S2_L2A_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

#: Scale factor converting Sentinel-2 L2A digital numbers to surface reflectance.
S2_REFLECTANCE_SCALE = 10_000.0


class EmbeddingBackbone(ABC):
    """A frozen Earth foundation-model encoder producing a token grid per patch.

    Subclasses declare which Sentinel-2 bands they consume and in which order,
    and map a batch of reflectance patches to a batch of spatial embedding
    grids. The grid is generally coarser than the input: a model of stride
    ``s`` turns an ``(H, W)`` patch into an ``(H // s, W // s)`` token grid.

    Attributes:
        band_order: Band names, in the order the model expects them.
        latent_dim: Width of one embedding token.
        stride: Input pixels per output token along each spatial axis.
    """

    band_order: tuple[str, ...]
    latent_dim: int
    stride: int

    @abstractmethod
    def embed(self, reflectance: np.ndarray) -> np.ndarray:
        """Encode a batch of reflectance patches.

        Args:
            reflectance: ``(B, H, W, C)`` float32 surface reflectance in
                ``[0, ~1]``, bands ordered as :data:`S2_L2A_BANDS`.

        Returns:
            ``(B, H // stride, W // stride, latent_dim)`` float32 token grid.
        """

    def select_bands(self, reflectance: np.ndarray) -> np.ndarray:
        """Reorder the canonical band axis into this backbone's ``band_order``.

        Args:
            reflectance: ``(B, H, W, len(S2_L2A_BANDS))`` array.

        Returns:
            ``(B, H, W, len(band_order))`` array.

        Raises:
            ValueError: If a requested band is not in :data:`S2_L2A_BANDS` or
                the input band axis has the wrong length.
        """
        if reflectance.shape[-1] != len(S2_L2A_BANDS):
            raise ValueError(f"expected {len(S2_L2A_BANDS)} bands on the last axis, got {reflectance.shape[-1]}")
        unknown = set(self.band_order) - set(S2_L2A_BANDS)
        if unknown:
            raise ValueError(f"unknown Sentinel-2 bands {sorted(unknown)}")
        index = [S2_L2A_BANDS.index(b) for b in self.band_order]
        return reflectance[..., index]

    def output_grid(self, patch_size: int) -> int:
        """Token-grid side length for a square input patch of ``patch_size`` pixels."""
        if patch_size % self.stride:
            raise ValueError(f"patch_size {patch_size} is not a multiple of stride {self.stride}")
        return patch_size // self.stride


#: Registry of backbone name -> fully-qualified ``module:ClassName``.
BACKBONES: dict[str, str] = {
    "olmoearth": "geosae.data.embed.olmoearth:OlmoEarthBackbone",
    "mmearth": "geosae.data.embed.mmearth:MMEarthBackbone",
}


def build_backbone(name: str, **kwargs) -> EmbeddingBackbone:
    """Instantiate a registered backbone by name.

    Args:
        name: Key in :data:`BACKBONES`.
        **kwargs: Forwarded to the backbone constructor.

    Returns:
        The instantiated backbone.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    import importlib

    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(BACKBONES)}")
    module_path, class_name = BACKBONES[name].split(":")
    return getattr(importlib.import_module(module_path), class_name)(**kwargs)


__all__ = ["S2_L2A_BANDS", "S2_REFLECTANCE_SCALE", "EmbeddingBackbone", "BACKBONES", "build_backbone"]

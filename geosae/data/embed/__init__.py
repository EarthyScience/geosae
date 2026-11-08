"""Run frozen Earth foundation-model encoders over a Sentinel-2 patch store.

:mod:`~geosae.data.embed.base` defines the backbone contract (band order, token
stride, latent width) that every encoder implements, plus a name registry;
:mod:`~geosae.data.embed.store` streams a patch store through a backbone and
writes the resulting token grids back as a new store array, ready for the SAE
dataset layer. The backbones themselves live in one module each and pull their
model dependency lazily, so importing this package needs only numpy.
"""

from __future__ import annotations

from geosae.data.embed.base import (
    BACKBONES,
    S2_L2A_BANDS,
    S2_REFLECTANCE_SCALE,
    EmbeddingBackbone,
    build_backbone,
)
from geosae.data.embed.store import center_crop, embed_store, embedding_schema, embedding_stats

__all__ = [
    "BACKBONES",
    "S2_L2A_BANDS",
    "S2_REFLECTANCE_SCALE",
    "EmbeddingBackbone",
    "build_backbone",
    "center_crop",
    "embed_store",
    "embedding_schema",
    "embedding_stats",
]

"""Sparse-autoencoder architectures, their shared contract and the config-driven model factory."""

from geosae.models.base import SparseAutoencoder, model_inputs, sparse_codes
from geosae.models.registry import (
    DERIVED_BUFFERS,
    MODEL_REGISTRY,
    build_model,
    load_checkpoint,
    load_state,
    register_model,
)
from geosae.models.relu_sae import ReLUSAE
from geosae.models.topk_sae import TopKSAE

__all__ = [
    "DERIVED_BUFFERS",
    "MODEL_REGISTRY",
    "ReLUSAE",
    "SparseAutoencoder",
    "TopKSAE",
    "build_model",
    "load_checkpoint",
    "load_state",
    "model_inputs",
    "register_model",
    "sparse_codes",
]

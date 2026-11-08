"""Config-driven model factory and checkpoint loading.

A model config is a mapping with a ``type`` key naming a registered builder; every other key is passed
to it. The built-in types are:

* ``batchtopk`` / ``topk`` -- :class:`~geosae.models.topk_sae.TopKSAE` with batch-level or per-row
  top-k selection;
* ``matryoshka`` -- a BatchTopK :class:`~geosae.models.topk_sae.TopKSAE` with one nested prefix
  (``h_frac``) or several (``prefix_fracs``), each reconstructing on its own in the loss;
* ``relu`` / ``relu_l1`` -- :class:`~geosae.models.relu_sae.ReLUSAE`, sparse through an ℓ1 penalty.

A new architecture registers its builder with :func:`register_model`::

    @register_model("mysae")
    def _build_mysae(cfg: dict) -> SparseAutoencoder:
        return MySAE(**cfg)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from geosae.models.base import SparseAutoencoder
from geosae.models.relu_sae import ReLUSAE
from geosae.models.topk_sae import TopKSAE

#: ``type`` name -> builder taking the config without its ``type`` key.
MODEL_REGISTRY: dict[str, Callable[[dict], SparseAutoencoder]] = {}

#: Buffers whose value is fully determined by the model config, so a checkpoint without them is
#: still complete: the constructor rebuilds them.
DERIVED_BUFFERS: tuple[str, ...] = ("n_high", "n_low")


def register_model(*names: str) -> Callable[[Callable[[dict], SparseAutoencoder]], Callable[[dict], SparseAutoencoder]]:
    """Register a model builder under one or more ``type`` names.

    Args:
        *names: Config ``type`` values (case-insensitive) that select this builder.

    Returns:
        A decorator that registers the builder and returns it unchanged.

    Raises:
        ValueError: A name is already registered.
    """

    def decorator(builder: Callable[[dict], SparseAutoencoder]) -> Callable[[dict], SparseAutoencoder]:
        for name in names:
            key = name.lower()
            if key in MODEL_REGISTRY:
                raise ValueError(f"model type {key!r} is already registered")
            MODEL_REGISTRY[key] = builder
        return builder

    return decorator


def build_model(cfg: Mapping[str, Any]) -> SparseAutoencoder:
    """Instantiate a model from a config mapping with a ``type`` key (default ``batchtopk``).

    Args:
        cfg: Model config; ``type`` selects the builder, every other key is a constructor argument.

    Returns:
        The model, freshly initialised.

    Raises:
        ValueError: ``type`` is not registered.
    """
    cfg = dict(cfg)
    mtype = str(cfg.pop("type", "batchtopk")).lower()
    if mtype not in MODEL_REGISTRY:
        raise ValueError(f"unknown model type {mtype!r} (expected one of {tuple(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[mtype](cfg)


@register_model("batchtopk")
def _build_batchtopk(cfg: dict) -> TopKSAE:
    cfg.setdefault("batch_topk", True)
    model = TopKSAE(**cfg)
    model.is_matryoshka = False
    return model


@register_model("topk")
def _build_topk(cfg: dict) -> TopKSAE:
    cfg.setdefault("batch_topk", False)
    model = TopKSAE(**cfg)
    model.is_matryoshka = False
    return model


@register_model("matryoshka")
def _build_matryoshka(cfg: dict) -> TopKSAE:
    # Same architecture as BatchTopK; the nesting is enforced by the prefix loss of geosae.train.loss.
    cfg.setdefault("batch_topk", True)
    cfg.setdefault("h_frac", 0.2)
    fracs = cfg.pop("prefix_fracs", None)
    model = TopKSAE(**cfg)
    model.is_matryoshka = True
    if fracs is not None:
        sizes = {round(float(f) * model.latent_dim) for f in fracs}
        model.prefix_sizes = tuple(sorted(n for n in sizes if 0 < n < model.latent_dim))
    return model


@register_model("relu", "relu_l1")
def _build_relu(cfg: dict) -> ReLUSAE:
    # Sparsity comes from the ℓ1 penalty, so the selection keys of a shared top-k config are dropped.
    cfg.pop("batch_topk", None)
    cfg.pop("k", None)
    # A truthy flag is rejected rather than dropped: it asks for the frozen-decoder null.
    if cfg.pop("freeze_decoder", False):
        raise ValueError("freeze_decoder is not supported by the relu model type")
    return ReLUSAE(**cfg)


def load_state(model: torch.nn.Module, state: dict, *, source: str = "state dict") -> torch.nn.Module:
    """Load a ``state_dict`` into ``model``, tolerating only missing :data:`DERIVED_BUFFERS`.

    Strict on every parameter and on every buffer that carries trained state: a missing
    ``min_activation`` would leave a BatchTopK model with an uncalibrated threshold, which silently
    returns dense codes.

    Args:
        model: A model built from the run's config.
        state: The ``state_dict`` to load.
        source: Where the state came from, for the error message.

    Returns:
        ``model``, loaded in place.

    Raises:
        RuntimeError: Anything other than a derived buffer is missing, or the state carries a key the
            model does not have.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    unresolved = [k for k in missing if k not in DERIVED_BUFFERS]
    if unresolved or unexpected:
        raise RuntimeError(
            f"{source} does not match the model built from the config: "
            f"missing={unresolved} unexpected={list(unexpected)}"
        )
    return model


def load_checkpoint(model: torch.nn.Module, path: str, *, map_location: str = "cpu") -> dict:
    """Load a run checkpoint (``{"model": state_dict, ...}``) into ``model``.

    Args:
        model: A model built from the run's config.
        path: Path to the ``.pth`` checkpoint.
        map_location: Torch map location for the load.

    Returns:
        The full checkpoint dict (model state plus its bookkeeping).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    load_state(model, ckpt["model"], source=f"checkpoint {path}")
    return ckpt


__all__ = ["DERIVED_BUFFERS", "MODEL_REGISTRY", "build_model", "load_checkpoint", "load_state", "register_model"]

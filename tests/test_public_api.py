"""The package-level public surface of ``geosae.models`` and ``geosae.eval``.

Two things must not drift: every model type ``build_model`` accepts resolves to
a class the package exports, and every name a subpackage advertises in
``__all__`` is actually importable from it (with the core install only).
"""

import importlib
import sys

import pytest
import torch

import geosae.eval as E
import geosae.models as M

# ``build_model`` type string -> the exported class it must return.
MODEL_TYPES = {
    "batchtopk": "TopKSAE",
    "topk": "TopKSAE",
    "matryoshka": "TopKSAE",
    "relu": "ReLUSAE",
    "relu_l1": "ReLUSAE",
}

D, LATENT, K = 8, 16, 4


@pytest.mark.parametrize("module", [M, E])
def test_advertised_names_are_importable(module):
    for name in module.__all__:
        assert hasattr(module, name), f"{module.__name__}.{name} missing"


@pytest.mark.parametrize(("mtype", "cls_name"), sorted(MODEL_TYPES.items()))
def test_build_model_reaches_every_type(mtype, cls_name):
    model = M.build_model({"type": mtype, "input_dim": D, "latent_dim": LATENT, "k": K})
    assert isinstance(model, getattr(M, cls_name))
    assert isinstance(model, M.SparseAutoencoder)


@pytest.mark.parametrize("mtype", sorted(MODEL_TYPES))
def test_every_built_model_reconstructs_its_input_shape(mtype):
    torch.manual_seed(0)
    model = M.build_model({"type": mtype, "input_dim": D, "latent_dim": LATENT, "k": K})
    x = torch.randn(3, 6, 6, D)
    batch = {"x": x, "lon": torch.linspace(-170, 170, 3), "lat": torch.linspace(-60, 60, 3)}
    recon, z, _ = model(*M.model_inputs(model, batch, "cpu"))
    assert recon.shape == x.shape
    assert z.shape == (3, 6, 6, LATENT)


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="unknown model type"):
        M.build_model({"type": "does-not-exist", "input_dim": D, "latent_dim": LATENT, "k": K})


def test_a_registered_builder_is_reachable_and_names_cannot_collide():
    @M.register_model("test_registry_tiny")
    def _build(cfg):
        return M.TopKSAE(**cfg)

    try:
        model = M.build_model({"type": "test_registry_tiny", "input_dim": D, "latent_dim": LATENT, "k": K})
        assert isinstance(model, M.TopKSAE)
        with pytest.raises(ValueError, match="already registered"):
            M.register_model("test_registry_tiny")(_build)
    finally:
        M.MODEL_REGISTRY.pop("test_registry_tiny")


def test_dictionary_is_the_decoder_atoms_as_rows():
    model = M.build_model({"type": "batchtopk", "input_dim": D, "latent_dim": LATENT, "k": K})
    atoms = model.dictionary()
    assert atoms.shape == (LATENT, D)
    assert torch.allclose(torch.as_tensor(atoms.T, dtype=torch.float32), model.decoder.weight)


def test_eval_surface_needs_no_plotting_stack(monkeypatch):
    for blocked in ("nicefigs", "cartopy", "matplotlib"):
        monkeypatch.setitem(sys.modules, blocked, None)
    importlib.reload(E)

"""The committed configs: every one loads, and every cell of the paper's sweep builds its model."""

from pathlib import Path

import pytest

from geosae.config import load_config
from geosae.models import SparseAutoencoder
from geosae.pipeline import build_model_from_cfg

ROOT = Path(__file__).resolve().parents[1] / "configs"
SWEEP = ROOT / "neurips2026"
VARIANTS = sorted(p.stem for p in (SWEEP / "variants").glob("*.yaml"))


@pytest.mark.parametrize("embedding", ["alphaearth", "tessera"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(("m", "k"), [(1024, 8), (512, 16), (2048, 16)])
def test_every_sweep_cell_builds_its_model(embedding, variant, m, k):
    cfg = load_config(
        [str(SWEEP / f"{embedding}.yaml"), str(SWEEP / "variants" / f"{variant}.yaml")],
        [f"model.latent_dim={m}", f"model.k={k}", "seed=1"],
    )
    model = build_model_from_cfg(cfg)
    assert isinstance(model, SparseAutoencoder)
    assert model.latent_dim == m and model.input_dim == cfg["model"]["input_dim"]
    if variant == "relu_l1":
        assert cfg["train"]["target_l0"]["value"] == k
    if variant == "frozen":
        assert not model.decoder.weight.requires_grad


def test_variants_cover_the_paper_recipes():
    assert VARIANTS == ["batchtopk", "frozen", "matryoshka", "relu_l1"]


def test_store_path_is_the_overridable_default(monkeypatch):
    monkeypatch.delenv("GEOSAE_STORE", raising=False)
    assert load_config([str(SWEEP / "alphaearth.yaml")])["data"]["zarr_path"] == "data/majortom210k.zarr"
    monkeypatch.setenv("GEOSAE_STORE", "/elsewhere/store.zarr")
    assert load_config([str(SWEEP / "tessera.yaml")])["data"]["zarr_path"] == "/elsewhere/store.zarr"

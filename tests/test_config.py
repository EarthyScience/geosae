"""Config layering, dotted overrides and the resolved-config round trip.

The resolved config is the reproduction record written into every run dir, so
what these tests pin is exactly what a run can be re-created from.
"""

import pytest
from omegaconf import OmegaConf

from geosae.config import load_config, save_config


def _yaml(path, text):
    path.write_text(text)
    return str(path)


def test_layers_merge_deeply_and_later_wins(tmp_path):
    base = _yaml(tmp_path / "base.yaml", "name: exp\nseed: 1\nmodel:\n  type: batchtopk\n  k: 16\n  latent_dim: 1024\n")
    over = _yaml(tmp_path / "over.yaml", "seed: 7\nmodel:\n  k: 32\n")
    cfg = load_config([base, over])
    assert cfg["seed"] == 7
    assert cfg["model"]["k"] == 32
    # Deep merge: keys the later layer does not mention survive.
    assert cfg["model"]["latent_dim"] == 1024
    assert cfg["model"]["type"] == "batchtopk"
    assert cfg["name"] == "exp"


def test_overrides_beat_every_layer_and_are_typed(tmp_path):
    a = _yaml(tmp_path / "a.yaml", "train:\n  epochs: 50\n  lr: 1.0e-3\n")
    cfg = load_config([a], ["train.epochs=2", "train.lr=0.5", "train.amp=false", "run.name=tiny"])
    assert cfg["train"]["epochs"] == 2 and isinstance(cfg["train"]["epochs"], int)
    assert cfg["train"]["lr"] == 0.5
    assert cfg["train"]["amp"] is False
    # A dotted override may introduce a section that no layer declared.
    assert cfg["run"]["name"] == "tiny"


def test_interpolations_are_resolved(tmp_path):
    a = _yaml(tmp_path / "a.yaml", "name: exp\nrun:\n  name: ${name}\n")
    cfg = load_config([a])
    assert cfg["run"]["name"] == "exp"
    # A plain dict, not a DictConfig: downstream code indexes and copies it.
    assert type(cfg) is dict


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config([str(tmp_path / "absent.yaml")])


def test_valueless_override_is_rejected_not_nulled(tmp_path):
    """A mistyped ``-o train.epochs`` must fail loudly, never null out the key."""
    a = _yaml(tmp_path / "a.yaml", "seed: 1\ntrain:\n  epochs: 50\n")
    with pytest.raises(ValueError, match="key=value"):
        load_config([a], ["train.epochs"])


def test_save_config_round_trips(tmp_path):
    cfg = {"name": "exp", "seed": 3, "model": {"type": "batchtopk", "k": 16}, "train": {"epochs": 2, "amp": False}}
    out = tmp_path / "nested" / "config.yaml"
    save_config(cfg, str(out))
    assert out.exists()  # parent dirs are created
    assert load_config([str(out)]) == cfg
    assert OmegaConf.to_container(OmegaConf.load(str(out))) == cfg


def test_a_bundled_config_is_found_by_name():
    from geosae.config import bundled_configs, resolve_config

    assert "batchtopk" in bundled_configs() and "download" in bundled_configs()
    cfg = load_config(["batchtopk"], ["model.k=8"])
    assert cfg["model"]["type"] == "batchtopk" and cfg["model"]["k"] == 8
    assert resolve_config("batchtopk.yaml") == resolve_config("batchtopk")


def test_a_path_wins_over_a_bundled_config_of_the_same_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _yaml(tmp_path / "batchtopk", "name: mine\n")
    assert load_config(["batchtopk"])["name"] == "mine"


def test_an_unknown_config_lists_the_bundled_ones():
    with pytest.raises(FileNotFoundError, match="batchtopk"):
        load_config(["no_such_config"])

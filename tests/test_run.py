"""Run-directory resolution + artefact paths."""

import csv

from geosae import run as R


def test_new_run_dir_is_timestamped_and_unique(tmp_path):
    a = R.new_run_dir(str(tmp_path), "exp")
    b = R.new_run_dir(str(tmp_path), "exp")
    assert a.parent == tmp_path / "exp" and a.exists()
    assert a != b  # same-second collisions get a suffix
    assert R.latest_run_dir(str(tmp_path), "exp") == max(a, b)


def test_resolve_explicit_dir_wins(tmp_path):
    cfg = {"name": "exp", "run": {"root": str(tmp_path), "dir": str(tmp_path / "explicit")}}
    d = R.resolve_run_dir(cfg, create=True)
    assert d == tmp_path / "explicit" and d.exists()


def test_resolve_create_vs_latest(tmp_path):
    cfg = {"name": "exp", "run": {"root": str(tmp_path)}}
    created = R.resolve_run_dir(cfg, create=True)
    reused = R.resolve_run_dir(cfg, create=False)
    assert created == reused  # later stages reuse the latest train dir


def test_paths_layout(tmp_path):
    p = R.paths(tmp_path / "run")
    assert p["best"].name == "best.pth"
    assert p["predictions"].name == "predictions"
    assert p["config"].name == "config.yaml"


def test_write_metrics_csv(tmp_path):
    out = tmp_path / "eval" / "m.csv"
    R.write_metrics_csv(out, [{"split": "val", "fve": 0.9}, {"split": "test", "fve": 0.8, "l0": 16}])
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["split"] == "val" and rows[1]["l0"] == "16"
    assert set(rows[0].keys()) == {"split", "fve", "l0"}  # union of keys

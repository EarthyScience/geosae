"""Run directories: one self-contained folder per experiment invocation.

Every run lives at ``<root>/<name>/<timestamp>/`` and collects *all* of its
artefacts so it can be reproduced and inspected in isolation:

    runs/<name>/<YYYYmmdd-HHMMSS>/
        config.yaml          fully-resolved config (the reproduction record)
        best.pth  last.pth   checkpoints
        metrics.jsonl        per-epoch train/val metrics
        eval/                metric tables of every evaluation stage
        predictions/*.zarr   sparse SAE codes + patch stats

``<root>`` defaults to ``runs`` (``run.root``).

Every stage resolves to the *same* run dir: ``geosae run`` creates it once and
threads it through; a standalone ``geosae train`` creates a fresh timestamped
dir; every other standalone stage reuses the latest (or an explicit ``run.dir``).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def new_run_dir(root: str, name: str) -> Path:
    """Create and return a fresh ``<root>/<name>/<timestamp>`` directory."""
    base = Path(root) / name
    ts = _timestamp()
    d = base / ts
    i = 1
    while d.exists():  # collision within the same second
        d = base / f"{ts}_{i}"
        i += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_run_dir(root: str, name: str) -> Path:
    """Return the most recent ``<root>/<name>/<timestamp>`` directory."""
    base = Path(root) / name
    subs = sorted(p for p in base.glob("*") if p.is_dir())
    if not subs:
        raise FileNotFoundError(f"no runs found under {base} — train first, or set run.dir")
    return subs[-1]


def resolve_run_dir(cfg: dict, *, create: bool) -> Path:
    """Resolve the run dir for this invocation.

    ``run.dir`` (explicit) wins. Otherwise ``create`` picks a new timestamped
    dir (train) while ``not create`` reuses the latest (every later stage).
    """
    run = dict(cfg.get("run") or {})
    if run.get("dir"):
        d = Path(run["dir"])
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d
    root = run.get("root", "runs")
    name = run.get("name") or cfg.get("name", "run")
    return new_run_dir(root, name) if create else latest_run_dir(root, name)


def paths(run_dir: str | Path) -> dict[str, Path]:
    """Canonical artefact paths within a run dir."""
    d = Path(run_dir)
    return {
        "dir": d,
        "config": d / "config.yaml",
        "best": d / "best.pth",
        "last": d / "last.pth",
        "metrics_jsonl": d / "metrics.jsonl",
        "eval": d / "eval",
        "predictions": d / "predictions",
    }


def write_metrics_csv(path: str | Path, rows: list[dict]) -> None:
    """Write a list of flat metric dicts to a CSV (union of keys as header)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


__all__ = [
    "new_run_dir",
    "latest_run_dir",
    "resolve_run_dir",
    "paths",
    "write_metrics_csv",
]

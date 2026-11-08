"""Fig. 6: gain of a sparse code over the raw embedding in ranking out-of-distribution errors.

On the graded hold-outs of ``geosae aoa`` (``configs/neurips2026/aoa_graded.yaml``), every distance
ranks the absolute error of one ridge head fitted on the raw embedding. Per embedding, target and
code, the gain of a fold is the Spearman rho of the ridge-weighted distance in the code minus that
of the ridge-weighted raw distance, each averaged over the seeds; the figure shows the median over
folds and its 95 % fold-bootstrap interval, for the BatchTopK SAE and the untrained random sparse
code of the same size and sparsity. Reads ``eval/aoa/records_graded.csv`` of each run, or with
``--paper-data`` the camera-ready gains (``applicability.json``).

    python scripts/neurips2026/fig6_aoa_gain.py --alphaearth runs/alphaearth/batchtopk_M1024_k16_s*/* \
        --tessera runs/tessera/batchtopk_M1024_k16_s*/* --out figures
    python scripts/neurips2026/fig6_aoa_gain.py --paper-data paper_data --out figures
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from _common import (
    EMBEDDINGS,
    METHOD_COLORS,
    add_paper_data_argument,
    fs,
    paper_file,
    parser,
    save,
    set_style,
    write_json,
)

TARGETS = {"canopy_height": "canopy height", "elevation": "elevation"}
#: Space name in the records -> (label, filled marker).
CODES = {"sae": ("SAE", True), "random": ("random code", False)}
#: Code name in the camera-ready ``applicability.json`` -> space name in the records.
PAPER_CODES = {"sae": "sae", "random_sparse": "random"}
MARKERS = {"alphaearth": "o", "tessera": "s"}
NULL_GREY = "#555555"
N_BOOT = 10_000
#: Printed width and height of the wrap figure (in).
WIDTH, HEIGHT = 2.45, 1.75


def graded_rho(run_dirs: list[Path], *, variant: str = "ridge") -> dict:
    """``(target, space) -> {fold: rho averaged over runs}`` of the raw-head graded records.

    Only the Euclidean, unscaled distance is read; a fold is ``(design, fold name)``.
    """
    acc: dict = defaultdict(lambda: defaultdict(list))
    for run in run_dirs:
        path = Path(run) / "eval" / "aoa" / "records_graded.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `geosae aoa` with configs/neurips2026/aoa_graded.yaml")
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if (r["variant"], r["metric"], r["scaling"], r["probe"]) != (variant, "euclidean", "none", "raw"):
                    continue
                acc[(r["target"], r["space"])][(r["scenario"], r["fold"])].append(float(r["rho"]))
    return {key: {f: float(np.mean(v)) for f, v in folds.items()} for key, folds in acc.items()}


def paired_gain(code: dict, raw: dict, *, seed: int = 0) -> dict:
    """Median over shared folds of ``code - raw``, its 95 % fold-bootstrap interval and the folds above zero."""
    folds = sorted(set(code) & set(raw))
    d = np.array([code[f] - raw[f] for f in folds])
    rng = np.random.default_rng(seed)
    boot = np.median(d[rng.integers(0, len(d), size=(N_BOOT, len(d)))], axis=1)
    return {
        "n": len(d),
        "wins": int((d > 0).sum()),
        "median": float(np.median(d)),
        "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
    }


def graded_gains(runs: dict[str, list[Path]]) -> dict:
    """``embedding -> target -> code -> paired gain`` over the raw ridge-weighted distance."""
    out: dict = {}
    for emb, run_dirs in runs.items():
        rho = graded_rho(run_dirs)
        for t in TARGETS:
            out.setdefault(emb, {})[t] = {c: paired_gain(rho[(t, c)], rho[(t, "raw")]) for c in CODES}
    return out


def paper_gains(paper_data: Path) -> dict:
    """The :func:`graded_gains` of the paper, read from the camera-ready ``applicability.json``."""
    app = json.loads(paper_file(paper_data, "applicability.json").read_text())
    return {
        emb: {t: {code: app[emb][t][name] for name, code in PAPER_CODES.items()} for t in TARGETS}
        for emb in EMBEDDINGS
        if emb in app
    }


def draw_gains(ax, gains: dict) -> None:
    """All gains in one axis: x the target, marker the embedding, fill the code (median and 95 % interval)."""
    import matplotlib.pyplot as plt

    offsets = np.linspace(-0.27, 0.27, len(gains) * len(CODES))
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
    handles = []
    for j, t in enumerate(TARGETS):
        i = 0
        for emb, marker in MARKERS.items():
            if emb not in gains:
                continue
            for c, (label, filled) in CODES.items():
                st = gains[emb][t][c]
                col = METHOD_COLORS["batchtopk"] if filled else NULL_GREY
                style = dict(marker=marker, ms=4, mfc=col if filled else "white", mec=col, mew=0.8)
                yerr = [[st["median"] - st["ci"][0]], [st["ci"][1] - st["median"]]]
                ax.errorbar(j + offsets[i], st["median"], yerr=yerr, fmt="none", ecolor="#222222", elinewidth=0.7)
                ax.plot(j + offsets[i], st["median"], ls="none", **style)
                if j == 0:
                    handles.append(plt.Line2D([], [], ls="none", label=f"{EMBEDDINGS[emb][0]}, {label}", **style))
                i += 1
    ax.set_xticks(range(len(TARGETS)), list(TARGETS.values()))
    ax.set_xlim(-0.5, len(TARGETS) - 0.5)
    ax.set_ylim(-0.05, 0.5)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylabel("gain in Spearman $\\rho$\nover raw")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.0,
              labelspacing=0.25, fontsize=fs("legend"))  # fmt: skip


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    for emb in EMBEDDINGS:
        ap.add_argument(f"--{emb}", nargs="+", type=Path, default=[], help=f"{emb} BatchTopK run directories")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    if not args.paper_data and not any(getattr(args, emb) for emb in EMBEDDINGS):
        ap.error("needs run directories of at least one embedding (or --paper-data)")
    if args.paper_data:
        gains = paper_gains(args.paper_data)
    else:
        gains = graded_gains({emb: getattr(args, emb) for emb in EMBEDDINGS if getattr(args, emb)})
    for emb, by_target in gains.items():
        for t, by_code in by_target.items():
            for c, st in by_code.items():
                print(f"{emb:10s} {t:13s} {c:6s} {st['wins']:2d}/{st['n']} folds above 0, median "
                      f"{st['median']:+.3f} [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]")  # fmt: skip

    import matplotlib.pyplot as plt

    set_style(WIDTH, WIDTH)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    draw_gains(fig.add_axes([0.22, 0.13, 0.77, 0.85]), gains)
    save(fig, args.out, "fig6_aoa_gain")
    write_json(gains, args.out / "fig6_aoa_gain.json")


if __name__ == "__main__":
    main()

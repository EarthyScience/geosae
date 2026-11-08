"""Fig. S5: per-feature patch-mean Moran's I on (a) AlphaEarth and (b) Tessera.

Moran's I of a feature over the first ``--n-patches`` test patches of the store (its rows are a
random draw of MajorTOM cells, so these are a random spatial sample), with each patch represented
by the feature's mean over all its pixels and a row-standardised 8-nearest-neighbour weight. The
three SAE variants (M = 1024, k = 16, seed 0) and the frozen-decoder baseline (seed 0; other seeds
reported) read the patch means of their ``geosae predict`` stores; the raw embedding and PCA, ICA and
NMF are fitted to and applied to the patch-mean embeddings, in the SAEs' input units. ``--paper-data``
reads the camera-ready per-feature values of ``spatial_coherence.npz`` instead.

    python scripts/neurips2026/figS_moran.py --runs-alphaearth runs/alphaearth --runs-tessera runs/tessera \
        --out figures
    python scripts/neurips2026/figS_moran.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from _common import (
    DENSE,
    DENSE_LABEL,
    EMBEDDINGS,
    SAE_VARIANT,
    VARIANTS,
    add_paper_data_argument,
    find_runs,
    fs,
    get_run,
    panel_title,
    paper_file,
    parser,
    predictions,
    require,
    save,
    set_style,
    store_path,
    write_json,
)

from geosae.config import load_config
from geosae.data.dataset import open_store
from geosae.eval.auroc_moran import patch_mean_moran_from_arrays
from geosae.eval.baselines import project_to_basis
from geosae.pipeline import apply_transforms, load_run_transforms

K_NEIGHBOURS = 8
#: Violin colours: dense bases on a grey ramp, the SAEs coloured, the frozen decoder light grey.
COLORS = {
    "raw": "#222222",
    "pca": "#555555",
    "ica": "#888888",
    "nmf": "#aaaaaa",
    "relu_l1": "#d62728",
    "batchtopk": "#2ca02c",
    "matryoshka": "#1f77b4",
    "frozen": "#c8c8c8",
}
LABELS = {**DENSE_LABEL, "raw": "Raw", **{v: SAE_VARIANT[v][0] for v in VARIANTS}, "frozen": "Frozen decoder"}
ORDER = (*DENSE, *VARIANTS, "frozen")


def moran(means: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Moran's I of every feature, ``(F,)``, from ``(F, P)`` patch means; ``nan`` where constant."""
    out = patch_mean_moran_from_arrays(means, lon, lat, k_neighbors=K_NEIGHBOURS, per_feature=True)
    return out["moran_per_feature"]


def patch_mean_embeddings(run_dir: Path, cells: np.ndarray, chunk: int = 50) -> np.ndarray:
    """``(P, D)`` mean of every patch's pixels, in the input units of the run's SAE."""
    cfg = load_config([str(run_dir / "config.yaml")], [])
    emb = open_store(store_path(run_dir))[cfg["data"].get("embedding_name", "alphaearth")]
    transforms = load_run_transforms(run_dir)
    out = []
    for i in range(0, len(cells), chunk):
        block = np.asarray(emb.get_orthogonal_selection((cells[i : i + chunk],)), dtype=np.float32)
        px = apply_transforms(transforms, block.reshape(-1, block.shape[-1]))
        out.append(px.reshape(len(block), -1, px.shape[-1]).mean(1))
    return np.concatenate(out)


def collect(runs_root: Path, n_patches: int) -> tuple[dict, dict]:
    """Per arm the Moran's I of its varying features, and the medians per frozen-decoder seed."""
    runs = find_runs(runs_root)
    ref = zarr.open_group(str(predictions(get_run(runs, "matryoshka"))), mode="r")
    cell = np.asarray(ref["cell"], dtype=np.int64)
    order = np.argsort(cell)[:n_patches]  # the first test patches of the store
    lon, lat, cells = np.asarray(ref["lon"])[order], np.asarray(ref["lat"])[order], cell[order]

    def sae_moran(run_dir: Path) -> np.ndarray:
        pred = zarr.open_group(str(predictions(run_dir)), mode="r")
        if not np.array_equal(np.asarray(pred["cell"], dtype=np.int64)[order], cells):
            raise ValueError(f"{run_dir}: predictions cover different patches than the reference run")
        return moran(np.asarray(pred["feat_patch_stats"][:, order, 0], dtype=np.float64), lon, lat)

    arms = {v: sae_moran(get_run(runs, v)) for v in VARIANTS}
    # every frozen-decoder seed whose `geosae predict` has finished
    frozen = {
        k.seed: sae_moran(p)
        for k, p in sorted(runs.items())
        if k[:3] == ("frozen", 1024, 16) and (p / "eval" / "test_metrics.csv").exists()
    }
    arms["frozen"] = frozen[0]
    means = patch_mean_embeddings(get_run(runs, "matryoshka"), cells)
    for basis in DENSE:
        arms[basis] = moran(project_to_basis(means, basis).T, lon, lat)
    arms = {k: v[np.isfinite(v)] for k, v in arms.items()}
    seeds = {s: float(np.nanmedian(v)) for s, v in frozen.items()}
    return arms, seeds


def from_paper_data(paper_data: Path, embedding: str) -> tuple[dict, dict]:
    """The output of :func:`collect` for one embedding from the camera-ready ``spatial_coherence.npz``."""
    short = {"alphaearth": "ae", "tessera": "te"}[embedding]
    with np.load(paper_file(paper_data, "spatial_coherence.npz")) as z:
        arms = {m: z[f"{short}/{m}"] for m in ORDER if f"{short}/{m}" in z.files}
        seeds = {0: float(np.nanmedian(arms["frozen"]))} if "frozen" in arms else {}
        seeds |= {
            s: float(np.nanmedian(z[f"{short}/frozen_s{s}"])) for s in (1, 2) if f"{short}/frozen_s{s}" in z.files
        }
    return {k: v[np.isfinite(v)] for k, v in arms.items()}, seeds


def draw(ax, data: dict[str, np.ndarray]) -> None:
    """Violins with the median printed above and the feature count below."""
    order = [m for m in ORDER if m in data]
    vp = ax.violinplot([data[m] for m in order], positions=range(len(order)), widths=0.75, showextrema=False)
    for body, m in zip(vp["bodies"], order, strict=True):
        body.set_facecolor(COLORS[m])
        body.set_edgecolor("#666666" if m == "frozen" else COLORS[m])
        body.set_alpha(0.7 if m == "frozen" else 0.55)
        if m == "frozen":
            body.set_linestyle("--")
            body.set_linewidth(0.8)
    for i, m in enumerate(order):
        med = float(np.median(data[m]))
        ax.hlines(med, i - 0.18, i + 0.18, colors="black", linewidth=1.2, zorder=4)
        ax.text(i, med + 0.025, f"{med:.2f}", ha="center", va="bottom", fontsize=fs("annot"), fontweight="bold")
        ax.text(
            i,
            -0.18,
            f"n={len(data[m])}" if i == 0 else f"{len(data[m])}",
            ha="center",
            va="top",
            fontsize=fs("annot") - 1,
            color="#777777" if m == "frozen" else COLORS[m],
        )
    for x in (len(DENSE) - 0.5, len(DENSE) + len(VARIANTS) - 0.5):
        ax.axvline(x, color="#cccccc", lw=0.6, ls=":", zorder=1)
    ax.axhline(0.0, color="#bbbbbb", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[m] for m in order], rotation=55, ha="right")
    ax.set_ylim(-0.25, 1.05)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--runs-alphaearth", type=Path, help="root of the AlphaEarth sweep runs")
    ap.add_argument("--runs-tessera", type=Path, help="root of the Tessera sweep runs")
    ap.add_argument("--n-patches", type=int, default=1500, help="test patches the statistic is computed on")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "runs-alphaearth", "runs-tessera")
    set_style(5.5)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)
    numbers = {}
    for ax, letter, (emb, root) in zip(
        axes, "ab", (("alphaearth", args.runs_alphaearth), ("tessera", args.runs_tessera)), strict=True
    ):
        arms, seeds = (
            collect(root, args.n_patches) if args.paper_data is None else from_paper_data(args.paper_data, emb)
        )
        numbers[emb] = {
            "median": {m: float(np.median(v)) for m, v in arms.items()},
            "n_features": {m: int(v.size) for m, v in arms.items()},
            "frozen_median_by_seed": seeds,
        }
        print(EMBEDDINGS[emb][0], numbers[emb])
        draw(ax, arms)
        panel_title(ax, letter, EMBEDDINGS[emb][0])
    axes[0].set_ylabel(r"patch-mean Moran's $I$")
    fig.tight_layout(w_pad=1.0)
    save(fig, args.out, "figS_moran")
    write_json(numbers, args.out / "figS_moran.json")


if __name__ == "__main__":
    main()

"""Fig. S6: the leading principal components of AlphaEarth beside the largest SAE features.

On the baseline fitting table (``geosae fit-table``: seeded pixels from every training patch) the
embedding's principal components are fitted, the Matryoshka SAE (M = 1024, k = 16, seed 0) is run, and
its features are ranked by energy share, the mean of ``z_j^2`` over the mean of ``||z||^2``.
(a) Scores of PC 1-3 on a 2-degree grid (top, each oriented to correlate positively with the feature
below) and the activation of the feature whose atom is closest to that component (bottom).
(b) |cosine| between PC 1-6 and the atoms of the six features of largest energy share; the closest
atom of each component among all atoms is outlined. With ``--paper-data``, reads the camera-ready
``pcs_vs_atoms.npz`` instead of encoding the fitting table.

    python scripts/neurips2026/figS_pcs_vs_atoms.py --runs runs/alphaearth --out figures
    python scripts/neurips2026/figS_pcs_vs_atoms.py --paper-data paper_data --out figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nicefigs as nf
import numpy as np
from _common import (
    METHOD_COLORS,
    add_paper_data_argument,
    find_runs,
    fs,
    get_run,
    paper_file,
    parser,
    require,
    save,
    set_style,
    store_path,
    write_json,
)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

from geosae.config import load_config
from geosae.data.dataset import open_store
from geosae.eval.baselines import load_fit_table
from geosae.eval.ladder import model_encoder
from geosae.pipeline import apply_transforms, load_run_model, load_run_transforms

N_TOP = 6
N_MAPS = 3
PRINTED_WIDTH = 5.5


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(A, B)`` Pearson correlations between the columns of ``a`` and of ``b``."""
    a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-12)
    b = (b - b.mean(0)) / np.maximum(b.std(0), 1e-12)
    return (a.T @ b) / a.shape[0]


def collect(run_dir: Path, fit_table: Path, device: str) -> dict:
    """Per-pixel PC scores and activations of the leading features, and the caption numbers."""
    table = load_fit_table(fit_table)
    x = apply_transforms(load_run_transforms(run_dir), table.pixels).astype(np.float64)
    g = open_store(store_path(run_dir))
    lon, lat = np.asarray(g["lon"][:])[table.cell], np.asarray(g["lat"][:])[table.cell]

    mean = x.mean(0)
    _, sing, vt = np.linalg.svd(x - mean, full_matrices=False)
    var_share = sing**2 / np.sum(sing**2)
    scores = (x - mean) @ vt[:N_TOP].T

    model = load_run_model(run_dir)
    atoms = model.dictionary()
    codes = model_encoder(model, device=device)(x.astype(np.float32))
    energy = (codes**2).mean(0) * (atoms**2).sum(1)
    share = energy / energy.sum()
    rank = np.argsort(-share)
    top = rank[:N_TOP]
    unit = atoms / np.maximum(np.linalg.norm(atoms, axis=1, keepdims=True), 1e-12)
    cos_all = np.abs(vt[:N_TOP] @ unit.T)
    alive = share > 0
    corr_all = np.zeros_like(cos_all)
    corr_all[:, alive] = pearson(scores, codes[:, alive])
    best = np.argmax(cos_all, axis=1)
    # The closest of M random unit vectors to a component, over 200 draws: the chance level of |cos|.
    rand = np.random.default_rng(0).normal(size=(200 * len(atoms), x.shape[1]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    random_best = np.abs(vt[:N_TOP] @ rand.T).reshape(N_TOP, 200, len(atoms)).max(2)
    meta = {
        "n_pixels": int(len(x)),
        "pc_var_share": var_share[:N_TOP].tolist(),
        "top_features": top.tolist(),
        "energy_share": share[top].tolist(),
        "energy_share_cum": float(share[top].sum()),
        "cos_top": np.abs(vt[:N_TOP] @ unit[top].T).tolist(),
        "pc_best_feature": best.tolist(),
        "pc_best_cos": [float(cos_all[i, j]) for i, j in enumerate(best)],
        "pc_best_in_top": [bool(j in top) for j in best],
        "pc_best_corr": [float(corr_all[i, j]) for i, j in enumerate(best)],
        "random_best_cos_mean": float(random_best.mean()),
        "random_best_cos_p95": float(np.percentile(random_best, 95)),
    }
    return {"lon": lon, "lat": lat, "pc": scores, "act": codes[:, best[:N_MAPS]], "meta": meta, "top": top}


def from_paper_data(paper_data: Path) -> dict:
    """The :func:`collect` output of the paper, read from the camera-ready ``pcs_vs_atoms.npz``.

    The file holds the activations of the six features of largest energy share, so the mapped
    feature of each component is looked up among them; its correlation with the component comes from
    the stored correlations with those features (``None`` where the closest atom is not among them).
    """
    z = np.load(paper_file(paper_data, "pcs_vs_atoms.npz"))
    m = json.loads(str(z["meta"]))
    top, best = list(m["top_atoms"]), list(m["pc_best_atom"])
    corr = np.asarray(m["corr_top"])
    best_corr = [float(corr[i, top.index(j)]) if j in top else None for i, j in enumerate(best)]
    meta = {
        "n_pixels": m["n_pixels"],
        "pc_var_share": m["pc_var_share"],
        "top_features": top,
        "energy_share": m["atom_energy_share"],
        "energy_share_cum": m["atom_energy_share_cum"],
        "cos_top": m["cos_top"],
        "pc_best_feature": best,
        "pc_best_cos": m["pc_best_cos"],
        "pc_best_in_top": [j in top for j in best],
        "pc_best_corr": best_corr,
    }
    act = np.asarray(z["atom"], dtype=np.float32)[:, [top.index(j) for j in best[:N_MAPS]]]
    pc = np.asarray(z["pc"], dtype=np.float32)
    return {
        "lon": z["lon"].astype(float),
        "lat": z["lat"].astype(float),
        "pc": pc,
        "act": act,
        "meta": meta,
        "top": top,
    }


def _map(fig, spec, lon, lat, values, *, cmap, vmin, vmax, title):
    ax, ccrs = nf.make_map_axes(fig, spec, projection="robinson")
    nf.draw_background(ax, ccrs, land=True, coastline=True)
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    _, mesh = nf.binned_map(lon, lat, values, ax=ax, ccrs=ccrs, cmap=cmap, vmin=vmin, vmax=vmax, bin_deg=2.0, zorder=2)
    ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=fs("annot"))
    return ax, mesh


def render(d: dict, out: Path) -> Path:
    m = d["meta"]
    set_style(5.5, PRINTED_WIDTH)
    fig = plt.figure(figsize=(5.5, 1.85))
    outer = GridSpec(
        1, 2, figure=fig, width_ratios=[3.9, 1.35], wspace=0.24, left=0.005, right=0.995, top=0.94, bottom=0.03
    )
    maps = GridSpecFromSubplotSpec(2, N_MAPS, subplot_spec=outer[0], hspace=0.30, wspace=0.10)
    rows: dict = {"top": [], "bottom": []}
    meshes = {}
    for i in range(N_MAPS):
        r = m["pc_best_corr"][i]
        score = d["pc"][:, i] * np.sign(r)
        lim = float(np.percentile(np.abs(score), 98))
        ax, meshes["top"] = _map(
            fig,
            maps[0, i],
            d["lon"],
            d["lat"],
            score / lim,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            title=f"PC {i + 1} ({100 * m['pc_var_share'][i]:.0f}%)",
        )
        rows["top"].append(ax)
        act = d["act"][:, i]
        title = f"f{m['pc_best_feature'][i]}\n|cos| {m['pc_best_cos'][i]:.2f}, $r$ {abs(r):.2f}"
        ax, meshes["bottom"] = _map(
            fig,
            maps[1, i],
            d["lon"],
            d["lat"],
            act / max(np.percentile(act, 99.5), 1e-12),
            cmap="inferno",
            vmin=0,
            vmax=1,
            title=title,
        )
        rows["bottom"].append(ax)
    fig.canvas.draw()
    for row, ticks in (("top", ("−", "+")), ("bottom", ("0", "1"))):
        pos = rows[row][-1].get_position()
        cb = fig.colorbar(meshes[row], cax=fig.add_axes([pos.x1 + 0.004, pos.y0 + 0.04, 0.005, pos.height - 0.08]))
        cb.set_ticks(list(meshes[row].get_clim()))
        cb.set_ticklabels(ticks)
        cb.ax.tick_params(labelsize=fs("annot"), length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)

    ax = fig.add_subplot(outer[1])
    ax.set_anchor("N")
    cos, top = np.array(m["cos_top"]), list(d["top"])
    ax.imshow(cos, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    for i in range(N_TOP):
        for k in range(N_TOP):
            ax.text(
                k,
                i,
                f"{cos[i, k]:.2f}"[1:],
                ha="center",
                va="center",
                fontsize=fs("annot") - 0.5,
                color="white" if cos[i, k] > 0.55 else "#222222",
            )
        if m["pc_best_feature"][i] in top:
            ax.add_patch(
                Rectangle(
                    (top.index(m["pc_best_feature"][i]) - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=METHOD_COLORS["batchtopk"],
                    linewidth=1.1,
                    zorder=3,
                )
            )
    ax.set_xticks(range(N_TOP))
    ax.set_xticklabels([f"f{j}" for j in top], rotation=90)
    ax.set_yticks(range(N_TOP))
    ax.set_yticklabels([f"PC {i + 1}" for i in range(N_TOP)])
    ax.tick_params(length=0, labelsize=fs("annot"), pad=2)
    ax.set_xlabel("atoms", labelpad=2)
    ax.set_title(r"$|\cos|$ of directions", fontsize=fs("annot"), pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.canvas.draw()
    for letter, x in (("a", 0.0), ("b", ax.get_position().x0 - 0.075)):
        fig.text(x, 0.96, letter, fontsize=fs("panel"), fontweight="bold", ha="left", va="bottom")
    return save(fig, out, "figS_pcs_vs_atoms")


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, help="root of the AlphaEarth sweep runs")
    ap.add_argument(
        "--fit-table", type=Path, help="baseline fitting table (default: the run's baselines.fit_table_path)"
    )
    ap.add_argument("--device", default="cuda", help="torch device")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "runs")
    if args.paper_data:
        d = from_paper_data(args.paper_data)
    else:
        run_dir = get_run(find_runs(args.runs), "matryoshka")
        cfg = load_config([str(run_dir / "config.yaml")], [])
        d = collect(run_dir, args.fit_table or Path(cfg["baselines"]["fit_table_path"]), args.device)
    for key, val in d["meta"].items():
        print(f"{key}: {np.round(np.asarray(val, dtype=float), 3).tolist()}")
    render(d, args.out)
    write_json(d["meta"], args.out / "figS_pcs_vs_atoms.json")


if __name__ == "__main__":
    main()

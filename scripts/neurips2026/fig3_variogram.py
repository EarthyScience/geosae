"""Fig. 3 and Figs. S3, S4: the spatial scales of the features.

Fig. 3 (``fig3_correlation_length``) is the share of features by correlation length on AlphaEarth.
Figs. S3 and S4 (``figS_variogram_<embedding>``, AlphaEarth and Tessera) add (a) the median normalised semivariance
against separation (bands: interquartile range of the raw dimensions and of the frozen decoder) and
(c) the share of a feature's variance within a patch. Reads the ``eval/variogram.{npz,json}`` of one
``geosae variogram`` run per embedding (``configs/neurips2026/variogram.yaml``); the frozen-decoder
baseline drawn is seed 0, the other seeds are reported. ``--paper-data`` reads the camera-ready
``variogram_{ae,te}.npz`` instead.

    python scripts/neurips2026/fig3_variogram.py --variogram-alphaearth eval/alphaearth/variogram \
        --variogram-tessera eval/tessera/variogram --out figures
    python scripts/neurips2026/fig3_variogram.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import (
    EMBEDDINGS,
    METHOD_COLORS,
    add_paper_data_argument,
    fs,
    panel_title,
    paper_file,
    parser,
    read_json,
    require,
    save,
    set_style,
    write_json,
)
from matplotlib.lines import Line2D

from geosae.eval.variogram import BELOW_FIRST_LAG, CORRELATION_LEVEL, IN_GAP, NEVER_REACHED, RESOLVED

#: Arm key -> label, in drawing order.
ARMS = (
    ("raw", "raw dimensions"),
    ("pca", "PCA"),
    ("ica", "ICA"),
    ("nmf", "NMF"),
    ("relu_l1", r"ReLU+$\ell_1$"),
    ("batchtopk", "BatchTopK"),
    ("matryoshka", "Matryoshka"),
    ("frozen", "frozen decoder"),
)
#: Correlation-length classes: below a patch, in the unsampled band, and two ranges between patches.
CLASSES = (
    ("< 640 m", "#F2F0E6"),
    ("0.64 to 13 km (not sampled)", "#BDBDBD"),
    ("13 to 250 km", "#7E9CC4"),
    ("> 250 km", "#2F4B7C"),
)
FAR_KM = 250.0
#: Arrays of one arm, as stored per arm in ``eval/variogram.npz`` and in the camera-ready files.
FIELDS = ("curve", "within", "nugget", "length_km", "status")
#: Largest within-patch separation (km); the camera-ready files store every separation in ``<arm>/d_km``.
LAST_WITHIN_KM = 0.63


def load(variogram_dir: Path) -> dict:
    """Per arm the arrays of its live features (seed 0 for every SAE arm), plus the separations.

    Returns:
        ``{arm: {curve, within, nugget, length_km, status}}``, ``dist_km``, ``n_within`` and
        ``frozen_seeds`` (the arrays of every frozen-decoder seed).
    """
    meta = read_json(variogram_dir / "eval" / "variogram.json")
    with np.load(variogram_dir / "eval" / "variogram.npz") as z:
        arrays = {k: z[k] for k in z.files}
    out: dict = {"dist_km": arrays["dist_km"], "n_within": int(arrays["n_within"]), "frozen_seeds": {}}
    for row in meta["rows"]:
        name = row["name"]
        live = np.isfinite(arrays[f"{name}/within"])
        feats = {key: arrays[f"{name}/{key}"][live] for key in FIELDS}
        if row["kind"] != "sae":
            out[row["kind"]] = feats
            continue
        arm = "frozen" if row.get("frozen") else row.get("arch")
        if arm == "frozen":
            out["frozen_seeds"][int(row["seed"])] = feats
        if int(row.get("seed", 0)) == 0 and int(row.get("k", 16)) == 16 and int(row.get("latent_dim", 1024)) == 1024:
            out[arm] = feats
    return out


def load_paper(npz: Path) -> dict:
    """The structure of :func:`load` from a camera-ready ``variogram_<ae|te>.npz``.

    The frozen-decoder arm pools its seeds, ``<arm>/n_by_run`` features per seed; seed 0 is drawn.
    """
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    dist_km = arrays["raw/d_km"]
    out: dict = {"dist_km": dist_km, "n_within": int((dist_km <= LAST_WITHIN_KM).sum()), "frozen_seeds": {}}
    for arm, _ in ARMS:
        if f"{arm}/status" not in arrays:
            continue
        edges = np.cumsum([0, *arrays[f"{arm}/n_by_run"]])
        runs = [{key: arrays[f"{arm}/{key}"][a:b] for key in FIELDS} for a, b in zip(edges, edges[1:], strict=False)]
        out[arm] = runs[0]
        if arm == "frozen":
            out["frozen_seeds"] = dict(enumerate(runs))
    return out


def length_classes(feats: dict, last_within_km: float) -> np.ndarray:
    """Share of features in each of :data:`CLASSES`."""
    st, ln = feats["status"], feats["length_km"]
    below = (st == BELOW_FIRST_LAG) | ((st == RESOLVED) & (ln < last_within_km))
    gap = st == IN_GAP
    mid = (st == RESOLVED) & (ln >= last_within_km) & (ln < FAR_KM)
    far = ((st == RESOLVED) & (ln >= FAR_KM)) | (st == NEVER_REACHED)
    return np.array([below.mean(), gap.mean(), mid.mean(), far.mean()])


def summary(d: dict) -> dict:
    """The numbers the paper quotes, per arm."""
    last = float(d["dist_km"][d["n_within"] - 1])
    out = {}
    for arm, _ in ARMS:
        if arm not in d:
            continue
        f = d[arm]
        resolved = f["status"] != NEVER_REACHED
        out[arm] = {
            "n_features": int(len(f["status"])),
            "classes": dict(zip([c for c, _ in CLASSES], length_classes(f, last).tolist(), strict=True)),
            "median_length_km": float(np.median(np.where(resolved, f["length_km"], np.inf))),
            "median_nugget": float(np.median(f["nugget"])),
            "median_within": float(np.median(f["within"])),
        }
    out["frozen_seeds"] = {
        s: {"median_within": float(np.median(f["within"])), "classes": length_classes(f, last).tolist()}
        for s, f in sorted(d["frozen_seeds"].items())
    }
    return out


def draw_curves(ax, d: dict, arms: list) -> None:
    dk, j = d["dist_km"], d["n_within"]
    for key, _ in arms:
        c = d[key]["curve"]
        cols = np.isfinite(c).any(0)
        x = dk[cols]
        q25, q50, q75 = (np.nanpercentile(c[:, cols], q, axis=0) for q in (25, 50, 75))
        jj = int(cols[:j].sum())
        col, lw = METHOD_COLORS[key], 1.2 if key in ("raw", "frozen") else 0.9
        ax.plot(x[:jj], q50[:jj], color=col, lw=lw, zorder=3)
        ax.plot(x[jj:], q50[jj:], color=col, lw=lw, zorder=3)
        ax.plot(x[jj - 1 : jj + 1], q50[jj - 1 : jj + 1], color=col, lw=0.6, ls=(0, (2, 2)), zorder=2)
        if key in ("raw", "frozen"):
            for sl in (slice(0, jj), slice(jj, None)):
                ax.fill_between(x[sl], q25[sl], q75[sl], color=col, alpha=0.15, lw=0)
    ax.axhline(CORRELATION_LEVEL, color="#999999", lw=0.5, ls=":")
    ax.axhline(1.0, color="#999999", lw=0.3, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 2.2e4)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([0.01, 1, 100, 1e4])
    ax.set_xticklabels(["10 m", "1 km", "100 km", "10,000 km"])
    ax.minorticks_off()
    ax.set_ylabel("normalised semivariance")
    ax.grid(True, alpha=0.25, lw=0.4)


def draw_classes(ax, d: dict, arms: list) -> None:
    last = float(d["dist_km"][d["n_within"] - 1])
    for i, (key, _) in enumerate(arms):
        left = 0.0
        for share, (_, col) in zip(length_classes(d[key], last), CLASSES, strict=True):
            ax.barh(i, share, left=left, color=col, height=0.72, lw=0)
            left += share
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(len(arms) - 0.4, -0.6)
    ax.set_yticks(range(len(arms)))
    ax.tick_params(axis="y", length=0)


def draw_within(ax, d: dict, arms: list) -> None:
    for i, (key, _) in enumerate(arms):
        q25, q50, q75 = np.percentile(d[key]["within"], [25, 50, 75])
        ax.plot([q25, q75], [i, i], color=METHOD_COLORS[key], lw=1.4, solid_capstyle="butt")
        ax.plot([q50], [i], "o", ms=3.6, color=METHOD_COLORS[key], mec="white", mew=0.4, zorder=3)
    q75 = max(np.percentile(d[key]["within"], 75) for key, _ in arms)
    ax.set_xlim(0, max(0.65, 1.05 * q75))
    ax.set_xlabel("share of variance")
    ax.set_xticks([0.25, 0.5])
    ax.set_xticklabels(["0.25", "0.5"])
    ax.set_ylim(len(arms) - 0.4, -0.6)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", alpha=0.25, lw=0.4)


def class_handles(labels) -> list:
    return [
        plt.Rectangle((0, 0), 1, 1, fc=c, ec="#999999", lw=0.3, label=lab)
        for lab, (_, c) in zip(labels, CLASSES, strict=True)
    ]


def render_main(d: dict, out: Path, width: float = 2.55) -> Path:
    """Figure 3: the share of features by correlation length, set beside the text."""
    arms = [a for a in ARMS if a[0] in d]
    set_style(width, width)
    fig = plt.figure(figsize=(width, 1.85))
    ax = fig.add_axes([0.34, 0.41, 0.64, 0.58])
    draw_classes(ax, d, arms)
    ax.set_yticklabels([lab for _, lab in arms])
    ax.set_xlabel("share of features,\nby correlation length")
    fig.legend(
        handles=class_handles(("< 640 m", "0.64 to 13 km", "13 to 250 km", "> 250 km")),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=fs("legend"),
        handlelength=1.0,
        columnspacing=0.9,
        labelspacing=0.2,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.0,
        borderpad=0.0,
    )
    return save(fig, out, "fig3_correlation_length")


def render_appendix(d: dict, out: Path, name: str) -> Path:
    """The three panels: semivariance curves, correlation-length classes, within-patch share."""
    arms = [a for a in ARMS if a[0] in d]
    set_style(5.5)
    fig = plt.figure(figsize=(5.5, 2.35))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.40, left=0.08, right=0.985, top=0.89, bottom=0.33)
    inner = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.16)
    axes = [fig.add_subplot(outer[0]), fig.add_subplot(inner[0]), fig.add_subplot(inner[1])]
    draw_curves(axes[0], d, arms)
    draw_classes(axes[1], d, arms)
    axes[1].set_yticklabels([lab for _, lab in arms])
    axes[1].set_xlabel("share of features")
    draw_within(axes[2], d, arms)
    for ax, letter, title in zip(
        axes, "abc", ("semivariance", "correlation length", "within-patch share"), strict=True
    ):
        panel_title(ax, letter, title)
    handles = [Line2D([], [], color=METHOD_COLORS[k], lw=1.4, label=lab) for k, lab in arms]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=fs("legend"),
        handlelength=1.3,
        columnspacing=1.0,
        labelspacing=0.3,
        bbox_to_anchor=(0.27, 0.0),
    )
    fig.legend(
        handles=class_handles([c for c, _ in CLASSES]),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=fs("legend"),
        handlelength=1.0,
        columnspacing=0.8,
        labelspacing=0.3,
        bbox_to_anchor=(0.79, 0.0),
    )
    return save(fig, out, name)


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--variogram-alphaearth", type=Path, help="geosae variogram run dir (AlphaEarth)")
    ap.add_argument("--variogram-tessera", type=Path, help="geosae variogram run dir (Tessera)")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "variogram-alphaearth")
    if args.paper_data:
        sources = {
            emb: paper_file(args.paper_data, f"variogram_{s}.npz")
            for emb, s in (("alphaearth", "ae"), ("tessera", "te"))
        }
    else:
        sources = {"alphaearth": args.variogram_alphaearth, "tessera": args.variogram_tessera}
    numbers = {}
    for emb, path in sources.items():
        if path is None:
            continue
        d = load_paper(path) if args.paper_data else load(path)
        numbers[emb] = summary(d)
        print(f"== {EMBEDDINGS[emb][0]}")
        for arm, s in numbers[emb].items():
            print(f"  {arm:10s} {s}")
        if emb == "alphaearth":
            render_main(d, args.out)
        render_appendix(d, args.out, f"figS_variogram_{emb}")
    write_json(numbers, args.out / "fig3_variogram.json")


if __name__ == "__main__":
    main()

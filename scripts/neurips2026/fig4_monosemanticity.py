"""Fig. 4 and Figs. S7, S8: concept-group alignment of SAE features, raw dimensions and PCA.

a     Per live feature, the number of concept groups it is aligned with (folded AUROC >= 0.65 on a
      member class) against its largest folded AUROC, for the SAE, the frozen-decoder baseline
      (grey, beside the SAE column), the raw dimensions and PCA, with a density of the counts.
b, c  Folded AUROC of one feature per concept group (columns) on every shown group (rows), for the
      SAE and the raw dimensions; the features are assigned jointly by the Hungarian algorithm.

Reads the ``geosae coverage`` outputs (``eval/test_coverage_arrays.npz``) of the SAE run, whose
file also holds the raw, PCA, ICA and NMF bases scored on the same pixels, and of the
frozen-decoder runs; ``--paper-data`` reads the camera-ready ``monosemanticity_<variant>_<ae|te>.json``
instead. Writes ``<name>.{pdf,png}`` and ``<name>.json`` with the numbers the captions quote.

    python scripts/neurips2026/fig4_monosemanticity.py --runs runs/alphaearth --variant matryoshka --out figures
    # Fig. S8: --variant batchtopk --name figS_mono_batchtopk;
    # Fig. S7: --runs runs/tessera --name figS_mono_tessera
    python scripts/neurips2026/fig4_monosemanticity.py --paper-data paper_data --out figures
    # Fig. S7: --paper-data paper_data --embedding tessera --name figS_mono_tessera
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from _common import (
    METHOD_COLORS,
    SAE_VARIANT,
    add_paper_data_argument,
    find_runs,
    fs,
    get_run,
    paper_file,
    parser,
    read_json,
    require,
    save,
    set_style,
    write_json,
)
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

from geosae.eval.concepts import CLASS_TO_GROUP, CONCEPT_GROUPS

ALIGN = 0.65  # folded AUROC at which a feature is aligned with a class
LIVE = 0.51  # a live feature exceeds this folded AUROC on at least one class
PEAK = 0.7  # a group is a heatmap candidate only if some feature or raw dimension reaches this
N_SHOW = 13  # groups shown in panels b and c
GROUPS = list(CONCEPT_GROUPS)
STYLE = {
    "sae": {"marker": "o", "size": 12, "alpha": 0.45, "zorder": 2, "bw": 0.3},
    "frozen": {"marker": "o", "size": 6, "alpha": 0.35, "zorder": 1, "bw": 0.3},
    "raw": {"marker": "s", "size": 42, "alpha": 0.85, "zorder": 4, "bw": 0.5},
    "pca": {"marker": "^", "size": 32, "alpha": 0.75, "zorder": 3, "bw": 0.5},
}
FROZEN_DODGE = 0.3


def coverage_arrays(run_dir: Path) -> dict:
    """The arrays of a run's ``geosae coverage`` output."""
    path = Path(run_dir) / "eval" / "test_coverage_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `geosae coverage --run {run_dir} -c configs/neurips2026/coverage.yaml`"
        )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def group_auc(auc: np.ndarray, class_table: np.ndarray, product_order, valid: np.ndarray) -> np.ndarray:
    """``(F, G)`` largest folded AUROC of each feature over the scorable member classes of each concept group."""
    out = np.full((auc.shape[0], len(GROUPS)), 0.5)
    for c, (pid, code) in enumerate(class_table):
        g = CLASS_TO_GROUP.get((str(product_order[int(pid)]), int(code)))
        if g is not None and valid[c]:
            j = GROUPS.index(g)
            out[:, j] = np.maximum(out[:, j], np.nan_to_num(auc[:, c], nan=0.5))
    return out


def live_groups(z: dict, key: str) -> np.ndarray:
    """Group-level AUROC of the live features of one decomposition in a coverage file."""
    valid = z["sae__valid_concepts"].astype(bool)
    auc = z[f"{key}__auc_matrix"]
    live = np.nanmax(np.where(valid, auc, 0.5), axis=1) > LIVE
    return group_auc(auc[live], z["class_table"], z["product_order"], valid)


def counts(g: np.ndarray) -> dict:
    """Aligned groups and largest AUROC of every feature of a group-level AUROC matrix, and their summary."""
    return summarise((g >= ALIGN).sum(1), g.max(1))


def summarise(n: np.ndarray, max_auc: np.ndarray) -> dict:
    """Aligned groups ``n`` and largest AUROC ``max_auc`` of every feature, and their summary."""
    n = np.asarray(n)
    return {
        "n_groups": n,
        "max_auc": np.asarray(max_auc, dtype=float),
        "n_live": int(n.size),
        "median_groups": float(np.median(n)),
        "share_le1": float(np.mean(n <= 1)),
        "share_eq1": float(np.mean(n == 1)),
        "share_0": float(np.mean(n == 0)),
        "n_aligned": int((n >= 1).sum()),
    }


def from_stages(runs_root: Path, variant: str) -> dict:
    """The alignment of one embedding from the ``geosae coverage`` outputs of its sweep.

    Returns:
        ``stats`` (:func:`counts` of ``sae``, ``raw``, ``pca`` and ``frozen_s<seed>`` for every frozen-decoder
        seed with coverage), the group-level AUROC matrices ``sae`` and ``raw``, and the ``groups`` they span.
    """
    runs = find_runs(runs_root)
    z = coverage_arrays(get_run(runs, variant))
    sae, raw, pca = live_groups(z, "sae"), live_groups(z, "raw"), live_groups(z, "pca")
    stats = {"sae": counts(sae), "raw": counts(raw), "pca": counts(pca)}
    for s in range(3):
        key = ("frozen", 1024, 16, s)
        if key in runs and (runs[key] / "eval" / "test_coverage_arrays.npz").exists():
            stats[f"frozen_s{s}"] = counts(live_groups(coverage_arrays(runs[key]), "sae"))
    return {"stats": stats, "sae": sae, "raw": raw, "groups": GROUPS}


def from_paper_data(paper_data: Path, variant: str, embedding: str) -> dict:
    """The output of :func:`from_stages` from the camera-ready ``monosemanticity_<variant>_<ae|te>.json``.

    The file holds the group-level AUROC of the live SAE features and raw dimensions on the groups that
    have a scorable class, and the alignment counts of PCA and of the frozen-decoder seeds.
    """
    short = {"alphaearth": "ae", "tessera": "te"}[embedding]
    d = read_json(paper_file(paper_data, f"monosemanticity_{variant}_{short}.json"))
    m = d["methods"]
    seeds = {f"frozen_s{m['frozen'].get('seed', 0)}": m["frozen"]}
    seeds |= {f"frozen_{s}": v for s, v in m.get("frozen_other_seeds", {}).items()}
    stats = {k: summarise(v["n_groups"], v["max_auc"]) for k, v in {**m, **seeds}.items() if "n_groups" in v}
    stats.pop("frozen", None)
    full = d["full"]
    return {
        "stats": stats,
        "sae": np.asarray(full["sae"]["auc"], dtype=float),
        "raw": np.asarray(full["raw"]["auc"], dtype=float),
        "groups": list(full["groups"]),
    }


def assign_diagonal(auc: np.ndarray, rows: list[int]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """One feature per shown group: aligned and with the fewest aligned groups, chosen jointly (Hungarian).

    Ties break towards the higher AUROC on the row's group; a row without an aligned feature
    takes its feature of highest AUROC. Returns the features, their AUROC on their row and their
    number of aligned groups.
    """
    n_aligned = (auc >= ALIGN).sum(1)
    sub = auc[:, rows].T
    cost = np.where(sub >= ALIGN, n_aligned[None, :] + 0.5 * (1.0 - sub), 1000.0 + (1.0 - sub))
    g, f = linear_sum_assignment(cost)
    feats = f[np.argsort(g)].tolist()
    return feats, auc[feats, rows], n_aligned[feats]


def auc_cmap() -> LinearSegmentedColormap:
    """Grey ramp on [0.5, 1] that steps from light to dark at the alignment threshold."""
    t = (ALIGN - 0.5) / 0.5
    return LinearSegmentedColormap.from_list(
        "auc", [(0, "#FFFFFF"), (t - 1e-4, "#D9D9D9"), (t, "#8C8C8C"), (1, "#1A1A1A")]
    )


def kde(vals: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian density of ``vals`` on ``grid``, bandwidth ``bw`` times the sample std."""
    h = bw * max(vals.std(), 1e-2)
    z = (grid[:, None] - vals[None, :]) / h
    return np.exp(-0.5 * z**2).sum(1) / (vals.size * h * np.sqrt(2 * np.pi))


def draw_scatter(ax, series: list[tuple[str, str, dict, float]], xmax: int) -> None:
    """Panel a: aligned groups against largest AUROC per feature, with a density on a twin axis."""
    axk = ax.twinx()
    grid = np.linspace(-0.5, xmax + 0.5, 300)
    rng = np.random.default_rng(0)
    peak = 0.0
    for key, label, c, dodge in series:
        st, color = STYLE[key], c["color"]
        n = np.asarray(c["n_groups"], float)
        jitter = rng.uniform(-0.07, 0.07, n.size) if key in ("sae", "frozen") else 0.0
        ax.scatter(
            n + dodge + jitter,
            c["max_auc"],
            s=st["size"] * 0.22,
            marker=st["marker"],
            c=color,
            alpha=st["alpha"],
            edgecolors="none",
            zorder=st["zorder"],
            rasterized=True,
            label=f"{label} ({n.size} features)" if key == "sae" else f"{label} ({n.size})",
        )
        dens = kde(n, grid, st["bw"])
        peak = max(peak, dens.max())
        axk.fill_between(grid, dens, color=color, alpha=0.12, zorder=0, lw=0)
        axk.plot(grid, dens, color=color, lw=0.8, alpha=0.7, zorder=0)
    ax.set_zorder(axk.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlim(-0.6, xmax + 0.5)
    ax.set_xticks(range(0, xmax + 1, 2))
    ax.set_ylim(0.47, 1.02)
    ax.set_xlabel("Aligned concept groups per feature")
    ax.set_ylabel("Max folded AUROC")
    ax.grid(True, color="#E5E5E5", lw=0.5)
    axk.set_ylim(0, 2.2 * peak)
    axk.set_yticks([])
    for a in (ax, axk):
        a.spines[["top", "right"]].set_visible(False)
    axk.spines["left"].set_visible(False)
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.29, -0.30),
        ncol=4,
        frameon=False,
        labelspacing=0.1,
        handletextpad=0.1,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    for h in leg.legend_handles:
        h.set_alpha(0.85)
        h.set_sizes([10])


def draw_heatmap(
    ax, auc: np.ndarray, rows: list[int], cols: list[int], xlabel: str, names: list[str] | None = None
) -> None:
    """Panels b, c: shown groups down, the feature assigned to each row across."""
    m = auc[np.ix_(cols, rows)].T
    ax.imshow(m, cmap=auc_cmap(), norm=mpl.colors.Normalize(0.5, 1.0), aspect="auto", interpolation="nearest")
    for j in range(m.shape[1] + 1):
        ax.axvline(j - 0.5, color="#E0E0E0", lw=0.3, zorder=3)
    for i in range(m.shape[0] + 1):
        ax.axhline(i - 0.5, color="#9A9A9A", lw=0.3, alpha=0.7, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names[g].replace("_", " ") for g in rows] if names else [], fontsize=fs("annot"))
    ax.set_xticks([])
    ax.set_xlabel(xlabel, labelpad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines[:].set_visible(False)


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, help="sweep root of one embedding, e.g. runs/alphaearth")
    ap.add_argument("--variant", choices=("matryoshka", "batchtopk"), default="matryoshka")
    ap.add_argument("--frozen-seeds", type=int, nargs="+", default=[0], help="frozen-decoder seeds drawn (pooled)")
    ap.add_argument("--name", default="fig4_monosemanticity")
    add_paper_data_argument(ap)
    ap.add_argument("--embedding", choices=("alphaearth", "tessera"), default="alphaearth", help="with --paper-data")
    args = ap.parse_args()
    require(ap, args, "runs")

    if args.paper_data is None:
        d = from_stages(args.runs, args.variant)
    else:
        d = from_paper_data(args.paper_data, args.variant, args.embedding)
    stats, sae, raw, groups = d["stats"], d["sae"], d["raw"], d["groups"]
    missing = [s for s in args.frozen_seeds if f"frozen_s{s}" not in stats]
    if missing:
        raise FileNotFoundError(f"no coverage of the frozen-decoder seeds {missing}")
    frozen = [stats[f"frozen_s{s}"] for s in args.frozen_seeds]
    stats["frozen"] = summarise(
        np.concatenate([f["n_groups"] for f in frozen]), np.concatenate([f["max_auc"] for f in frozen])
    )

    label = SAE_VARIANT[args.variant][0]
    series = [
        ("sae", label, {**stats["sae"], "color": METHOD_COLORS[args.variant]}, 0.0),
        ("frozen", "frozen-decoder baseline", {**stats["frozen"], "color": METHOD_COLORS["frozen"]}, FROZEN_DODGE),
        ("raw", "raw dim.", {**stats["raw"], "color": METHOD_COLORS["raw"]}, 0.0),
        ("pca", "PCA", {**stats["pca"], "color": METHOD_COLORS["pca"]}, 0.0),
    ]

    # Rows of b, c: the N_SHOW groups whose best SAE feature reaches the highest AUROC, among the
    # groups on which some feature or raw dimension reaches PEAK, in the order of CONCEPT_GROUPS.
    best = np.where(np.maximum(sae.max(0), raw.max(0)) >= PEAK, sae.max(0), -1.0)
    rows = sorted(np.argsort(-best, kind="stable")[:N_SHOW].tolist())
    rows = [g for g in rows if best[g] >= 0]
    cols_b, diag_b, n_b = assign_diagonal(sae, rows)
    cols_c, diag_c, n_c = assign_diagonal(raw, rows)

    set_style()
    width, height, pitch = 5.5, 2.15, 0.095
    fig = plt.figure(figsize=(width, height))

    def box(x0, x1, y0, y1):
        return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]

    y1 = height - 0.22
    y0 = y1 - N_SHOW * pitch
    xb0 = 2.42
    xb1 = xb0 + len(rows) * pitch
    xc0 = xb1 + 0.07
    xc1 = xc0 + len(rows) * pitch
    ax_a = fig.add_axes(box(0.40, 1.78, y0, y1))
    ax_b = fig.add_axes(box(xb0, xb1, y0, y1))
    ax_c = fig.add_axes(box(xc0, xc1, y0, y1))
    cax = fig.add_axes(box(xc1 + 0.08, xc1 + 0.17, y0, y1))
    xmax = max(14, int(max(c["n_groups"].max() for _, _, c, _ in series)))
    draw_scatter(ax_a, series, xmax + xmax % 2)
    draw_heatmap(ax_b, sae, rows, cols_b, "SAE features", names=groups)
    draw_heatmap(ax_c, raw, rows, cols_c, "raw dim.")
    cb = mpl.colorbar.ColorbarBase(cax, cmap=auc_cmap(), norm=mpl.colors.Normalize(0.5, 1.0))
    cb.set_ticks([0.5, ALIGN, 0.8, 0.9, 1.0])
    cb.set_ticklabels(["0.5", f"{ALIGN:.2f}", "0.8", "0.9", "1.0"])
    cb.set_label("folded AUROC", labelpad=1)
    cax.tick_params(length=2, pad=1)
    for ax, letter, dx in ((ax_a, "a", -0.28), (ax_b, "b", -0.55), (ax_c, "c", -0.04)):  # paper layout
        ax.annotate(
            letter, (dx, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
            fontsize=fs("panel"), fontweight="bold", ha="left", va="bottom",
        )  # fmt: skip
    save(fig, args.out, args.name)

    summary = {
        "variant": args.variant,
        "frozen_seeds": args.frozen_seeds,
        "align": ALIGN,
        **{k: {kk: v for kk, v in s.items() if kk not in ("n_groups", "max_auc")} for k, s in stats.items()},
        "shown_groups": [groups[g] for g in rows],
        "selected": {
            "sae": {
                "mean_auc": float(diag_b.mean()),
                "min_auc": float(diag_b.min()),
                "median_groups": float(np.median(n_b)),
            },
            "raw": {
                "mean_auc": float(diag_c.mean()),
                "min_auc": float(diag_c.min()),
                "median_groups": float(np.median(n_c)),
            },
        },
    }
    for k, s in stats.items():
        print(
            f"{k:10s} live {s['n_live']:5d}  median groups {s['median_groups']:.0f}  <=1 {s['share_le1']:.2f}  "
            f"=1 {s['share_eq1']:.2f}  0 {s['share_0']:.2f}  aligned {s['n_aligned']}"
        )
    print(f"shown groups: {summary['shown_groups']}")
    print(f"selected SAE features: mean AUROC {diag_b.mean():.3f}, median groups {np.median(n_b):.0f}")
    print(f"selected raw dims:     mean AUROC {diag_c.mean():.3f}, median groups {np.median(n_c):.0f}")
    write_json(summary, Path(args.out) / f"{args.name}.json")


if __name__ == "__main__":
    main()

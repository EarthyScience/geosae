"""Fig. 2: reconstruction against sparsity on AlphaEarth (a) and Tessera (b).

FVE against the mean number of active features per pixel, L0, for every SAE of the sweep (seed
means; lines join k = 8, 16, 32 at M = 1024, marker size is M), the frozen-decoder baseline (seed 0)
and the dense baselines with k components (raw, PCA / ICA, NMF). Reads the ``eval/panel.json`` of one
``geosae panel`` run per embedding (``configs/neurips2026/panel.yaml``), or with ``--paper-data`` the
camera-ready ``real_products.json`` (SAEs) and ``dense_topk.json`` (dense baselines).

    python scripts/neurips2026/fig2_pareto.py --panel-alphaearth eval/alphaearth/panel \
        --panel-tessera eval/tessera/panel --out figures
    python scripts/neurips2026/fig2_pareto.py --paper-data paper_data --out figures
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import (
    DENSE,
    EMBEDDINGS,
    METHOD_COLORS,
    SAE_VARIANT,
    VARIANTS,
    add_paper_data_argument,
    fs,
    panel_rows,
    panel_title,
    paper_file,
    parser,
    require,
    save,
    set_style,
    variant_style,
    write_json,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

WIDTHS = (512, 1024, 2048)
KS = (8, 16, 32)
REF_WIDTH = 1024
#: Marker size per dictionary size, and the horizontal offset of the three sizes at k = 16.
MARKER = {512: 2.8, 1024: 4.2, 2048: 5.8}
DODGE = {512: 0.93, 1024: 1.0, 2048: 1.075}
#: Dense baselines drawn: ICA with k components spans the PCA subspace, so it shares the PCA line.
DENSE_DRAWN = (("raw", "raw"), ("pca", "PCA / ICA"), ("nmf", "NMF"))
FVE_TICKS = (0.1, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99)


def nines(fve) -> np.ndarray:
    """``-log10(1 - FVE)``: the FVE axis is logarithmic in the unexplained variance."""
    return -np.log10(1.0 - np.asarray(fve, dtype=float))


def paper_rows(paper_data: Path, embedding: str) -> list[dict]:
    """The camera-ready reconstruction numbers of one embedding as rows of ``geosae panel``.

    SAE rows come from ``real_products.json``, the dense baselines with ``k`` components (``rank_<k>``)
    from ``dense_topk.json``. Neither file holds the MSE, so ``recon_mse`` is NaN; ``pixel_cosine`` is
    the stored ``recon_cosine`` (NaN for the dense baselines).
    """
    products = json.loads(paper_file(paper_data, "real_products.json").read_text())["products"][embedding]
    rows = [
        {**r, "pixel_cosine": r["recon_cosine"], "recon_mse": np.nan} for r in products["rows"] if r["kind"] == "sae"
    ]
    dense = json.loads(paper_file(paper_data, "dense_topk.json").read_text())["products"][embedding]
    for method in DENSE:
        for k in KS:
            for seed, r in enumerate(dense.get(method, {}).get(f"rank_{k}", [])):
                rows.append(
                    {"kind": method, "latent_dim": k, "seed": seed, "fve": r["fve"], "l0": float(k)}
                    | {"recon_mse": np.nan, "pixel_cosine": np.nan}
                )
    return rows


def summarise(rows: list[dict]) -> dict:
    """Seed means per SAE cell, the frozen decoder (seed 0) per k, and the dense baselines per k.

    Returns:
        ``sae`` ``{(variant, M, k): stats}``, ``frozen`` ``{k: stats}``, ``dense`` ``{method: {k: stats}}``,
        each ``stats`` holding ``n``, ``l0``, ``fve``, ``fve_sd``, ``recon_mse`` and ``pixel_cosine``.
    """
    sae, frozen, dense = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] == "sae":
            if r["frozen"]:
                frozen[int(r["k"])].append(r)
            elif r["arch"] in VARIANTS:
                sae[(r["arch"], int(r["latent_dim"]), int(r["k"]))].append(r)
        else:
            dense[r["kind"]][int(r["latent_dim"])].append(r)

    def stats(rs: list[dict]) -> dict:
        out = {"n": len(rs), "fve_sd": float(np.std([r["fve"] for r in rs]))}
        for key in ("l0", "fve", "recon_mse", "pixel_cosine"):
            out[key] = float(np.mean([r[key] for r in rs]))
        return out

    return {
        "sae": {key: stats(v) for key, v in sae.items()},
        "frozen": {k: stats([r for r in v if int(r["seed"]) == 0] or v) for k, v in sorted(frozen.items())},
        "frozen_seeds": {k: [float(r["fve"]) for r in sorted(v, key=lambda r: r["seed"])] for k, v in frozen.items()},
        # A dense baseline uses all k of its components; its signed coefficients are not counted by the panel's l0.
        "dense": {m: {k: {**stats(v), "l0": float(k)} for k, v in sorted(d.items())} for m, d in dense.items()},
    }


def draw(ax, cells: dict) -> None:
    """FVE against log2 L0 for one embedding."""
    for variant in VARIANTS:
        sweep = [cells["sae"][(variant, REF_WIDTH, k)] for k in KS if (variant, REF_WIDTH, k) in cells["sae"]]
        col = METHOD_COLORS[variant]
        ax.plot(np.log2([c["l0"] for c in sweep]), nines([c["fve"] for c in sweep]), color=col, lw=0.9, zorder=2)
        for (v, width, k), c in cells["sae"].items():
            if v == variant:
                x = c["l0"] * (DODGE[width] if k == 16 else 1.0)
                ax.plot(
                    [np.log2(x)],
                    nines([c["fve"]]),
                    zorder=4 if width == REF_WIDTH else 3,
                    **variant_style(variant, markersize=MARKER[width]),
                )
    fr = list(cells["frozen"].values())
    fx, fy = np.log2([c["l0"] for c in fr]), nines([c["fve"] for c in fr])
    col = METHOD_COLORS["frozen"]
    ax.plot(fx, fy, color=col, lw=0.9, ls=(0, (3, 1.5)), zorder=2)
    ax.plot(fx, fy, "o", color=col, mfc="white", mew=0.9, ms=MARKER[REF_WIDTH], zorder=3)
    for method, _ in DENSE_DRAWN:
        d = cells["dense"].get(method, {})
        ks = [k for k in KS if k in d]
        xs, ys = np.log2(ks), nines([d[k]["fve"] for k in ks])
        ax.plot(xs, ys, color=METHOD_COLORS[method], lw=0.9, zorder=2)
        ax.plot(xs, ys, "D", color=METHOD_COLORS[method], ms=3.2, mec="white", mew=0.3, zorder=3)
    ax.xaxis.set_major_locator(FixedLocator(np.log2(KS)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{2.0**v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(np.log2(KS[0]) - 0.45, np.log2(KS[-1]) + 0.45)
    ax.yaxis.set_major_locator(FixedLocator(nines(FVE_TICKS)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{1.0 - 10.0 ** (-v):.2f}".rstrip("0").rstrip(".")))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.07, nines(0.9945))
    ax.set_xlabel(r"Active features per pixel $L_0$")
    ax.grid(True, alpha=0.25, lw=0.4)


def legend(fig, ax) -> None:
    grey = "#555555"
    handles = [
        Line2D([], [], label=SAE_VARIANT[v][0], **variant_style(v, markersize=MARKER[REF_WIDTH])) for v in VARIANTS
    ]
    handles.append(Line2D([], [], label="dictionary size", ls="none"))
    handles += [
        Line2D([], [], label=f"$M$={w}", **variant_style("batchtopk", color=grey, markersize=MARKER[w])) for w in WIDTHS
    ]
    handles.append(
        Line2D(
            [],
            [],
            label="frozen-decoder\nbaseline",
            color=METHOD_COLORS["frozen"],
            marker="o",
            ls="none",
            mfc="white",
            mew=0.9,
            markersize=MARKER[REF_WIDTH],
        )
    )
    handles += [
        Line2D([], [], label=name, color=METHOD_COLORS[m], marker="D", ls="none", markersize=3.4, mec="white", mew=0.3)
        for m, name in DENSE_DRAWN
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(ax.get_position().x1 + 0.012, 0.5),
        frameon=False,
        fontsize=fs("legend"),
        handletextpad=0.4,
        labelspacing=0.5,
        borderaxespad=0.0,
    )


def report(label: str, cells: dict) -> dict:
    """Print and return the numbers the paper quotes for one embedding."""
    print(f"== {label}")
    for (v, m, k), c in sorted(cells["sae"].items()):
        print(f"  {v:10s} M={m:4d} k={k:2d} n={c['n']} L0={c['l0']:5.2f} FVE={c['fve']:.4f} (sd {c['fve_sd']:.4f})")
    for k, c in cells["frozen"].items():
        print(
            f"  frozen     M={REF_WIDTH} k={k:2d} L0={c['l0']:5.2f} FVE={c['fve']:.4f} seeds {cells['frozen_seeds'][k]}"
        )
    for m, d in cells["dense"].items():
        print(f"  {m:10s} FVE " + ", ".join(f"k={k}: {c['fve']:.4f}" for k, c in d.items()))
    at16 = [c["fve"] for (v, w, k), c in cells["sae"].items() if k == 16]
    out = {
        "max_seed_sd": max(c["fve_sd"] for c in cells["sae"].values()),
        "fve_range_k16_over_M": [min(at16), max(at16)] if at16 else None,
        "frozen_fve_k16": cells["frozen"].get(16, {}).get("fve"),
        "frozen_seed_fve_range": {k: max(v) - min(v) for k, v in cells["frozen_seeds"].items()},
        "pca_fve_k16": cells["dense"].get("pca", {}).get(16, {}).get("fve"),
        "unexplained_k16": {
            v: 1.0 - cells["sae"][(v, REF_WIDTH, 16)]["fve"] for v in VARIANTS if (v, REF_WIDTH, 16) in cells["sae"]
        },
        "every_sae_above_every_baseline": all(
            c["fve"] > max(d[k]["fve"] for d in cells["dense"].values() if k in d)
            for (v, w, k), c in cells["sae"].items()
            if any(k in d for d in cells["dense"].values())
        ),
    }
    if out["frozen_fve_k16"] is not None:
        out["unexplained_k16"]["frozen"] = 1.0 - out["frozen_fve_k16"]
    print("  " + str(out))
    return out


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--panel-alphaearth", type=Path, help="geosae panel run dir (AlphaEarth)")
    ap.add_argument("--panel-tessera", type=Path, help="geosae panel run dir (Tessera)")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "panel-alphaearth", "panel-tessera")
    set_style(5.5)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True, gridspec_kw={"wspace": 0.08, "right": 0.80})
    numbers = {}
    for ax, letter, emb, panel in zip(axes, "ab", EMBEDDINGS, (args.panel_alphaearth, args.panel_tessera), strict=True):
        cells = summarise(paper_rows(args.paper_data, emb) if args.paper_data else panel_rows(panel))
        draw(ax, cells)
        panel_title(ax, letter, EMBEDDINGS[emb][0])
        numbers[emb] = report(EMBEDDINGS[emb][0], cells)
    axes[0].set_ylabel("FVE")
    fig.canvas.draw()
    legend(fig, axes[1])
    save(fig, args.out, "fig2_pareto")
    write_json(numbers, args.out / "fig2_pareto.json")


if __name__ == "__main__":
    main()

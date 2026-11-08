"""Fig. S12: area of applicability for canopy height with Europe held out, and the graded gains.

(a, b) Meyer & Pebesma's dissimilarity index of every held-out European pixel in units of its
threshold, in the raw embedding and in the SAE code (``aoa_europe.npz`` of ``table_aoa_europe.py``);
(c) the gain in Spearman rho of a sparse code over the raw distance on the graded hold-outs, per
embedding and target (the panel of Fig. 6, from the runs' ``eval/aoa/records_graded.csv``). With
``--paper-data``, reads the camera-ready ``aoa_europe.npz`` and ``applicability.json`` instead.

    python scripts/neurips2026/figS_aoa_gain.py figures/aoa_europe.npz \
        --alphaearth runs/alphaearth/batchtopk_M1024_k16_s*/* --tessera runs/tessera/batchtopk_M1024_k16_s*/* \
        --out figures
    python scripts/neurips2026/figS_aoa_gain.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import (
    EMBEDDINGS,
    TEXT_WIDTH,
    add_paper_data_argument,
    fs,
    panel_title,
    paper_file,
    parser,
    require,
    save,
    set_style,
    write_json,
)
from fig6_aoa_gain import draw_gains, graded_gains, paper_gains
from figS_aoa import europe_axes
from scipy.stats import spearmanr

#: Inside the area of applicability blue, outside warm, white at the threshold.
AOA_COLORS = [
    (0.00, "#003f5c"),
    (0.20, "#007a8a"),
    (0.38, "#4db6ac"),
    (0.48, "#b2dfdb"),
    (0.50, "#ffffff"),
    (0.55, "#fde68a"),
    (0.70, "#f59e0b"),
    (0.85, "#dc2626"),
    (1.00, "#581c87"),
]


def draw_index(fig, spec, lonlat: np.ndarray, index: np.ndarray, *, label: str, vmax: float):
    """One map of the dissimilarity index in threshold units, with a histogram colour bar."""
    import nicefigs as nf
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("inside_outside", AOA_COLORS)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    ax, ccrs = europe_axes(fig, spec)
    nf.binned_map(lonlat[:, 0], lonlat[:, 1], index, ax=ax, ccrs=ccrs, cmap=cmap, norm=norm, bin_deg=1.0,
                  fill_bins=1, clip_ocean=True, land_resolution="50m")  # fmt: skip
    inset = nf.inset_histogram_colorbar(ax, index, cmap=cmap, vmin=0.0, vmax=vmax, norm=norm,
                                        bounds=(0.03, 0.44, 0.2, 0.16), label=label)  # fmt: skip
    if inset is not None:
        inset.set_xticks([0.0, 1.0])
        inset.xaxis.label.set_size(fs("annot") - 0.5)
    return ax


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("europe", nargs="?", type=Path, help="aoa_europe.npz of table_aoa_europe.py")
    for emb in EMBEDDINGS:
        ap.add_argument(f"--{emb}", nargs="+", type=Path, default=[], help=f"{emb} BatchTopK run directories")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "europe")
    if not args.paper_data and not any(getattr(args, emb) for emb in EMBEDDINGS):
        ap.error("needs run directories of at least one embedding (or --paper-data)")
    if args.paper_data:
        eu = dict(np.load(paper_file(args.paper_data, "aoa_europe.npz")))
        gains = paper_gains(args.paper_data)
    else:
        eu = dict(np.load(args.europe))
        gains = graded_gains({emb: getattr(args, emb) for emb in EMBEDDINGS if getattr(args, emb)})

    import matplotlib.pyplot as plt

    set_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.15, left=0.0, right=0.99, top=0.84,
                          bottom=0.17)  # fmt: skip
    vmax = float(np.ceil(10 * max(eu["aoa_raw"].max(), eu["aoa_sae"].max())) / 10)
    ax_a = draw_index(fig, gs[0, 0], eu["lonlat"], eu["aoa_raw"], label="raw DI", vmax=vmax)
    ax_b = draw_index(fig, gs[0, 1], eu["lonlat"], eu["aoa_sae"], label="SAE DI", vmax=vmax)
    ax_c = fig.add_subplot(gs[0, 2])
    draw_gains(ax_c, gains)
    ax_c.set_ylabel("")
    panel_title(ax_a, "a", "raw, test = Europe")
    panel_title(ax_b, "b", "SAE, test = Europe")
    panel_title(ax_c, "c", "gain in Spearman $\\rho$ over raw")
    save(fig, args.out, "figS_aoa_gain")

    # Both ranking distances against the error of the head fitted on the raw embedding.
    summary = {
        "europe": {
            "n_test": int(eu["err_raw"].size),
            "rho_raw_head": {s: float(spearmanr(eu[f"di_{s}"], eu["err_raw"])[0]) for s in ("raw", "sae")},
            "inside": {s: float((eu[f"aoa_{s}"] <= 1.0).mean()) for s in ("raw", "sae")},
        },
        "graded": gains,
    }
    print(summary["europe"])
    write_json(summary, args.out / "figS_aoa_gain.json")


if __name__ == "__main__":
    main()

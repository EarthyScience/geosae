"""Fig. S13: SAE against raw distance on the leave-Europe-out canopy-height fold.

(a) the log2 ratio of the ridge-weighted SAE and raw distances of every held-out pixel; (b) the two
out-of-domain masks (dissimilarity index above its threshold) overlaid. Reads ``aoa_europe.npz``,
written by ``table_aoa_europe.py`` from the cached sites of ``geosae aoa``, or with ``--paper-data``
the camera-ready one.

    python scripts/neurips2026/figS_aoa.py figures/aoa_europe.npz --out figures
    python scripts/neurips2026/figS_aoa.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import (
    METHOD_COLORS,
    TEXT_WIDTH,
    add_paper_data_argument,
    fs,
    paper_file,
    parser,
    require,
    save,
    set_style,
    write_json,
)

EXTENT = (-25.0, 45.0, 34.0, 72.0)
BIN_DEG = 1.0
DROPPED, KEPT = "#C8C8C8", "#7A7A7A"
#: ``geosae.eval.aoa.di_agreement`` overlay code (a = raw, b = SAE) -> label, colour, count key.
OVERLAY = {
    3: ("in domain in both", KEPT, "n_both_in"),
    0: ("out of domain in both", DROPPED, "n_both_out"),
    2: ("out in raw only", METHOD_COLORS["raw"], "n_only_a"),
    1: ("out in SAE only", METHOD_COLORS["batchtopk"], "n_only_b"),
}


def europe_axes(fig, spec):
    """A Robinson map axes over Europe with land and coastlines."""
    import nicefigs as nf

    ax, ccrs = nf.make_map_axes(fig, spec, projection="robinson", central_longitude=10.0)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    nf.draw_background(ax, ccrs, land=True, coastline=True, resolution="50m")
    return ax, ccrs


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("europe", nargs="?", type=Path, help="aoa_europe.npz of table_aoa_europe.py")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "europe")
    d = dict(np.load(paper_file(args.paper_data, "aoa_europe.npz") if args.paper_data else args.europe))
    lon, lat = d["lonlat"][:, 0], d["lonlat"][:, 1]

    import matplotlib.pyplot as plt
    import nicefigs as nf

    set_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.05))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.05], hspace=0.12, wspace=0.08, left=0.02, right=0.98,
                            top=0.9, bottom=0.2)  # fmt: skip

    ax_a, ccrs = europe_axes(fig, grid[0, 0])
    both = (d["di_sae"] > 0) & (d["di_raw"] > 0)  # a held-out pixel identical to a training pixel has distance 0
    ratio = np.full(lon.shape, np.nan)
    ratio[both] = np.log2(d["di_sae"][both] / d["di_raw"][both])
    lim = float(np.nanpercentile(np.abs(ratio), 98))
    _, mesh = nf.binned_map(lon, lat, ratio, ax=ax_a, ccrs=ccrs, cmap="RdBu_r", vmin=-lim, vmax=lim, bin_deg=BIN_DEG)
    cb = fig.colorbar(mesh, cax=fig.add_subplot(grid[1, 0]), orientation="horizontal", extend="both")
    cb.set_label(r"$\log_2$(SAE / raw); red: SAE farther", fontsize=fs("annot"), labelpad=1.5)
    cb.ax.tick_params(labelsize=fs("annot"), length=1.5, width=0.3, pad=1.5)

    ax_b, ccrs = europe_axes(fig, grid[0, 1])
    order = tuple(OVERLAY)
    nf.categorical_map(lon, lat, d["overlay"], ax=ax_b, categories={c: OVERLAY[c][0] for c in order},
                       colors=[OVERLAY[c][1] for c in order], ccrs=ccrs, bin_deg=BIN_DEG, legend=False)  # fmt: skip
    counts = {OVERLAY[c][0]: int(d["agree_" + OVERLAY[c][2]]) for c in order}
    handles = [plt.Rectangle((0, 0), 1, 1, fc=OVERLAY[c][1], ec="none") for c in order]
    ax_b.legend(handles, [f"{name} ({n:,})" for name, n in counts.items()], loc="upper center",
                bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=fs("legend"), frameon=False, handlelength=0.9,
                handletextpad=0.4, labelspacing=0.3, columnspacing=0.8, borderaxespad=0.0)  # fmt: skip

    summary = {
        "n_test": int(lon.size),
        "rank_spearman": float(d["agree_spearman"]),
        "mask_kappa": float(d["agree_mask_kappa"]),
        "counts": counts,
    }
    nf.panel_label(ax_a, "a", f"distance ratio; maps agree at $\\rho$ = {summary['rank_spearman']:.2f}")
    nf.panel_label(ax_b, "b", f"threshold masks, $\\kappa$ = {summary['mask_kappa']:.2f}")
    save(fig, args.out, "figS_aoa")
    print(summary)
    write_json(summary, args.out / "figS_aoa.json")


if __name__ == "__main__":
    main()

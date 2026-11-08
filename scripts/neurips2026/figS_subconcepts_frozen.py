"""Fig. S9: the woody-vegetation features of the frozen-decoder baseline, laid out as Fig. 5b-f.

(a) Two top patches per feature, imagery above activation; (b) the 2,000 most active test patches of
each feature; (c-e) the matched strata of Köppen-Geiger climate zones, RESOLVE biomes and potential
natural vegetation. Per Köppen-Geiger pair, the feature is the woody feature whose top patches lie
most in it, as for the trained SAE of ``fig5_aligned_features.py``.

Reads the frozen-decoder run's ``predictions/test.zarr`` (``geosae predict``) and
``eval/test_coverage_arrays.npz`` with its evaluation pool (``geosae coverage -c
configs/neurips2026/coverage.yaml``), and the data store. Writes ``figS_subconcepts_frozen.{pdf,png,json}``.
With ``--paper-data <dir>`` it draws the camera-ready inputs instead (``subconcepts_frozen.{npz,json}``,
Esri World Imagery at the recorded patch centres).

    python scripts/neurips2026/figS_subconcepts_frozen.py runs/alphaearth/frozen_M1024_k16_s0/<ts> --out figures
    python scripts/neurips2026/figS_subconcepts_frozen.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import _common as C
import fig5_aligned_features as F5
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def render(d: dict):
    """Gallery on top, the four maps below, the key at the bottom."""
    C.set_style(5.5)
    fig = plt.figure(figsize=(5.5, 4.9))
    outer = GridSpec(
        3, 1, figure=fig, height_ratios=[1.0, 1.75, 0.13], hspace=0.2, left=0.02, right=0.98, top=0.975, bottom=0.01
    )
    F5.draw_gallery(fig, outer[0], d)
    F5.draw_maps(fig, outer[1], d, letters="bcde", support_title="frozen-decoder features")
    leg = fig.add_subplot(outer[2])
    leg.axis("off")
    leg.legend(
        handles=F5.legend_handles(d),
        loc="center",
        ncol=2,
        fontsize=C.fs("legend"),
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.text(0.0, 0.955, "a", fontsize=C.fs("panel"), fontweight="bold", ha="left", va="bottom")
    return fig


def main() -> None:
    ap = C.parser(__doc__.splitlines()[0])
    ap.add_argument(
        "run", type=Path, nargs="?", help="frozen-decoder run (M = 1024, k = 16) with predictions and coverage"
    )
    ap.add_argument("--cache-root", default="data/eval_cache/samples", help="pool sample cache of the coverage stage")
    ap.add_argument("--device", default="cpu", help="torch device that encodes the pool pixels")
    C.add_imagery_argument(ap)
    C.add_paper_data_argument(ap)
    args = ap.parse_args()
    C.require(ap, args, "run")
    if args.paper_data is not None:
        d = F5.paper_subconcepts(args.paper_data, "frozen")
    else:
        d = F5.subconcepts(
            args.run, F5.Alignment(args.run, cache_root=args.cache_root, device=args.device), esri=args.esri
        )
    picked = [next(r for r in d["woody"] if r["feat"] == f) for f in d["features"]]
    for r in picked:
        print(f"f{r['feat']}: {r['class']}, KG pair shares {[round(x, 2) for x in r['pair_share']]}")
    print("woody:", F5.woody_summary(d["woody"]))
    C.write_json(
        {"features": d["features"], "picked": picked, "table": d["table"], "woody": F5.woody_summary(d["woody"])},
        args.out / "figS_subconcepts_frozen.json",
    )
    C.save(render(d), args.out, "figS_subconcepts_frozen")


if __name__ == "__main__":
    main()

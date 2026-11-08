"""Table S8: agreement of the three woody-vegetation features with climate and biome maps.

For a trained SAE (Fig. 5b-f) and the frozen-decoder baseline of the same size and sparsity,
each feature's 2,000 most active test patches are labelled with their majority class in
Köppen-Geiger climate zones, RESOLVE biomes and potential natural vegetation; each table cell lists
the dominant classes with their share and lift. Also reports the woody features of both runs and
how concentrated they are on the Köppen-Geiger pairs (Appendix, sub-concepts of the frozen-decoder
baseline). Features are chosen as in ``fig5_aligned_features.py``.

Reads, per run, ``predictions/test.zarr`` (``geosae predict``) and ``eval/test_coverage_arrays.npz``
with its evaluation pool (``geosae coverage -c configs/neurips2026/coverage.yaml``). Writes
``table_subconcepts.{tex,json}``. With ``--paper-data <dir>`` it reads the camera-ready inputs instead
(``subconcepts.json``, ``subconcepts_frozen.json``); they hold no agreement table of the frozen run.

    python scripts/neurips2026/table_subconcepts.py runs/alphaearth/matryoshka_M1024_k16_s0/<ts> \\
        runs/alphaearth/frozen_M1024_k16_s0/<ts> --out figures
    python scripts/neurips2026/table_subconcepts.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import _common as C
import fig5_aligned_features as F5

PRODUCTS = ("chelsa_kg", "resolve_biomes", "pnv")
MIN_SHARE = 0.15  # a class enters a cell when it holds at least this share (the top class always does)


def cell(classes: list[dict]) -> str:
    """The dominant classes of one feature and product, ``name share (lift)``."""
    keep = [c for i, c in enumerate(classes[:2]) if i == 0 or c["share"] >= MIN_SHARE]
    return "; ".join(f"{c['class']} {c['share']:.2f} ({c['lift']:.1f})" for c in keep)


def latex(table: list[dict]) -> str:
    """Table S8 as a LaTeX tabularx body."""
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}l r X X X@{}}",
        r"\toprule",
        r"Feature & $\overline{|\text{lat}|}$ & Köppen-Geiger & RESOLVE biome & PNV \\",
        r"\midrule",
    ]
    for row in table:
        cells = " & ".join(cell(row[p]).replace("&", r"\&") for p in PRODUCTS)
        lines.append(f"f{row['feat']} & ${row['mean_abs_lat']:.1f}^\\circ$ & {cells} \\\\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabularx}", ""])


def paper_arm(paper_data: Path, arm: str) -> dict:
    """The ``table``, ``features`` and ``woody`` of one arm (``trained`` or ``frozen``) from the camera-ready inputs."""
    sub = C.read_json(C.paper_file(paper_data, "subconcepts_frozen.json"))
    if arm == "trained":
        table = C.read_json(C.paper_file(paper_data, "subconcepts.json"))["table"]
        features = [row["feat"] for row in table]
    else:
        table, features = [], sub["frozen_picks"]
    return {"table": table, "features": features, "woody": sub["features"][arm]}


def main() -> None:
    ap = C.parser(__doc__.splitlines()[0])
    ap.add_argument("trained", type=Path, nargs="?", help="trained SAE run (Matryoshka, M = 1024, k = 16)")
    ap.add_argument("frozen", type=Path, nargs="?", help="frozen-decoder run of the same M and k")
    ap.add_argument("--cache-root", default="data/eval_cache/samples", help="pool sample cache of the coverage stage")
    ap.add_argument("--device", default="cpu", help="torch device that encodes the pool pixels")
    C.add_paper_data_argument(ap)
    args = ap.parse_args()
    C.require(ap, args, "trained", "frozen")
    out = {}
    for arm, run in (("trained", args.trained), ("frozen", args.frozen)):
        if args.paper_data is not None:
            d = paper_arm(args.paper_data, arm)
        else:
            d = F5.subconcepts(run, F5.Alignment(run, cache_root=args.cache_root, device=args.device), gallery=False)
        picked = [next(r for r in d["woody"] if r["feat"] == f) for f in d["features"]]
        out[arm] = {"table": d["table"], "picked": picked, "woody": F5.woody_summary(d["woody"])}
        print(f"{arm}: {out[arm]['woody']}")
        rows = {row["feat"]: row for row in d["table"]}
        for r in picked:
            shares = ", ".join(f"{s:.2f}" for s in r["pair_share"])
            print(f"  f{r['feat']} ({r['class']}): KG pair shares {shares}")
            if r["feat"] in rows:
                print(f"    mean |lat| {rows[r['feat']]['mean_abs_lat']:.1f}")
                for p in PRODUCTS:
                    print(f"    {p:15s} {cell(rows[r['feat']][p])}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_subconcepts.tex").write_text(latex(out["trained"]["table"]))
    print("wrote", args.out / "table_subconcepts.tex")
    C.write_json(out, args.out / "table_subconcepts.json")


if __name__ == "__main__":
    main()

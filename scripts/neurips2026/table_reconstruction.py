"""Table S3: reconstruction at L0 = 16 on AlphaEarth and Tessera.

MSE per embedding dimension, FVE, mean per-pixel cosine and L0 of the dense baselines with 16
components, the frozen-decoder baseline (seed 0) and the three SAE variants at M = 1024, k = 16 (seed
means). Reads the same ``geosae panel`` outputs as ``fig2_pareto.py``; writes a LaTeX tabular and a JSON.
The panel scores Tessera in its standardised input space; the table reports its MSE in the paper's
units, divided by the square of ``TESSERA_INPUT_FACTOR``. ``--paper-data`` reads the camera-ready
``recon_table.json`` instead (on the clean test pixels, in the paper's units).

    python scripts/neurips2026/table_reconstruction.py --panel-alphaearth eval/alphaearth/panel \
        --panel-tessera eval/tessera/panel --out figures
    python scripts/neurips2026/table_reconstruction.py --paper-data paper_data --out figures
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import (
    EMBEDDINGS,
    VARIANTS,
    add_paper_data_argument,
    panel_rows,
    paper_file,
    parser,
    require,
    write_json,
)
from fig2_pareto import summarise

K = 16
ROWS = (
    ("raw", f"Raw (${K}$ dims)"),
    ("pca", f"PCA-${K}$"),
    ("ica", f"ICA-${K}$"),
    ("nmf", f"NMF-${K}$"),
    ("frozen", "Frozen decoder"),
    ("relu_l1", r"ReLU+$\ell_1$"),
    ("batchtopk", "BatchTopK"),
    ("matryoshka", "Matryoshka"),
)
METRICS = (("recon_mse", False), ("fve", True), ("pixel_cosine", True))
#: The isotropic factor between the paper's Tessera units and the standardised Tessera input: the
#: square root of D over the root-mean-square pixel norm of the paper's evaluation pixels.
TESSERA_INPUT_FACTOR = 11.054
MSE_SCALE = {"alphaearth": 1.0, "tessera": TESSERA_INPUT_FACTOR**-2}
#: Key of each embedding and row in the camera-ready ``recon_table.json``.
PAPER_EMBEDDING = {"alphaearth": "ae", "tessera": "te"}
PAPER_ROW = {"frozen": "frozen_decoder", "relu_l1": "relu"}


def table_rows(rows: list[dict], width: int = 1024, mse_scale: float = 1.0) -> dict[str, dict]:
    """``{row key: {recon_mse, fve, pixel_cosine, l0}}`` at k = 16 for one embedding, MSE times ``mse_scale``."""
    cells = summarise(rows)
    out = {m: cells["dense"][m][K] for m in ("raw", "pca", "ica", "nmf") if K in cells["dense"].get(m, {})}
    if K in cells["frozen"]:
        out["frozen"] = cells["frozen"][K]
    out.update({v: cells["sae"][(v, width, K)] for v in VARIANTS if (v, width, K) in cells["sae"]})
    return {key: {**r, "recon_mse": r["recon_mse"] * mse_scale} for key, r in out.items()}


def paper_table_rows(paper_data: Path, embedding: str) -> dict[str, dict]:
    """The rows of :func:`table_rows` from the camera-ready ``recon_table.json`` (its clean test pixels)."""
    table = json.loads(paper_file(paper_data, "recon_table.json").read_text())[PAPER_EMBEDDING[embedding]]
    out = {}
    for key, _ in ROWS:
        r = table[PAPER_ROW.get(key, key)]["clean"]
        out[key] = {"recon_mse": r["mse"], "fve": r["fve"], "pixel_cosine": r["cos"], "l0": r["l0"]}
    return out


def _sci(x: float) -> str:
    mant, exp = f"{x:.2e}".split("e")
    return rf"{mant}{{\cdot}}10^{{{int(exp)}}}"


def latex(results: dict[str, dict]) -> str:
    """The tabular body, best value per embedding in bold."""
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & MSE ($\downarrow$) & FVE ($\uparrow$) & cos ($\uparrow$) & $L_0$ \\",
    ]
    for emb, rows in results.items():
        label, dim = EMBEDDINGS[emb]
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{label} ($D{{=}}{dim}$)}}}}\\"]
        best = {key: (min if not up else max)(r[key] for r in rows.values()) for key, up in METRICS}
        for key, name in ROWS:
            if key not in rows:
                continue
            r = rows[key]
            cells = [_sci(r["recon_mse"]), f"{r['fve']:.3f}", f"{r['pixel_cosine']:.3f}"]
            cells = [
                rf"$\mathbf{{{c}}}$" if r[m] == best[m] else f"${c}$" for c, (m, _) in zip(cells, METRICS, strict=True)
            ]
            l0 = f"{r['l0']:.0f}" if key in ("raw", "pca", "ica", "nmf") else f"{r['l0']:.1f}"
            if key == "relu_l1":
                lines.append(r"\addlinespace[2pt]")
            lines.append(f"{name:24s} & " + " & ".join(cells) + f" & ${l0}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--panel-alphaearth", type=Path, help="geosae panel run dir (AlphaEarth)")
    ap.add_argument("--panel-tessera", type=Path, help="geosae panel run dir (Tessera)")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "panel-alphaearth", "panel-tessera")
    if args.paper_data:
        results = {emb: paper_table_rows(args.paper_data, emb) for emb in EMBEDDINGS}
    else:
        panels = {"alphaearth": args.panel_alphaearth, "tessera": args.panel_tessera}
        results = {emb: table_rows(panel_rows(d), mse_scale=MSE_SCALE[emb]) for emb, d in panels.items()}
    for emb, rows in results.items():
        print(f"== {EMBEDDINGS[emb][0]}")
        for key, r in rows.items():
            print(
                f"  {key:10s} MSE={r['recon_mse']:.3e} FVE={r['fve']:.3f} cos={r['pixel_cosine']:.3f} L0={r['l0']:.1f}"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_reconstruction.tex").write_text(latex(results))
    print("wrote", args.out / "table_reconstruction.tex")
    write_json(results, args.out / "table_reconstruction.json")


if __name__ == "__main__":
    main()

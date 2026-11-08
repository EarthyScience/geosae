"""Table S6: UCov of every representation at two alignment thresholds, plus the EMF numbers of the text.

UCov counts the scorable product classes on which at least one feature reaches a folded AUROC of
tau. It is recomputed at tau = 0.60 and 0.65 from the AUROC matrices of the ``geosae coverage``
outputs (``eval/test_coverage_arrays.npz``) of the M = 1024, k = 16 runs of each embedding: the
dense bases (raw, PCA, ICA, NMF with D components), the three SAEs, the frozen-decoder baseline
(seed 0) and, where the BatchTopK run was scored with ``coverage.random_widths=[1024]``, the
untrained random dictionary. ``--paper-data`` reads the camera-ready ``tables.json`` (UCov) and
``null_coverage.json`` (EMF of the M = 1024 BatchTopK, frozen-decoder and random dictionaries)
instead. Writes ``table_ucov.tex`` and ``table_ucov.json``.

    python scripts/neurips2026/table_ucov.py --runs runs/alphaearth runs/tessera --out figures
    python scripts/neurips2026/table_ucov.py --paper-data paper_data --out figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _common import (
    DENSE,
    DENSE_LABEL,
    EMBEDDINGS,
    SAE_VARIANT,
    VARIANTS,
    add_paper_data_argument,
    find_runs,
    paper_file,
    parser,
    read_json,
    require,
    write_json,
)

from geosae.config import load_config
from geosae.eval.coverage import ucov

TAUS = (0.60, 0.65)
#: Row label of every representation, by its key in the camera-ready ``tables.json``.
PAPER_ROWS = {
    "raw": "Raw",
    **{b: f"{DENSE_LABEL[b]}-D" for b in DENSE[1:]},
    "relu": SAE_VARIANT["relu_l1"][0],
    "batchtopk": SAE_VARIANT["batchtopk"][0],
    "matryoshka": SAE_VARIANT["matryoshka"][0],
    "frozen_decoder": "Frozen decoder",
    "untrained_random": "Untrained random dictionary",
}
#: The arm of ``null_coverage.json`` whose M = 1024 EMF belongs to a row.
PAPER_EMF = {
    "BatchTopK": "trained",
    "Frozen decoder": "frozen decoder",
    "Untrained random dictionary": "untrained random",
}


def rows_of(runs: dict) -> dict[str, dict]:
    """``label -> {ucov_0.60, ucov_0.65, n_valid, emf}`` for every representation scored under one sweep root."""

    def load(variant: str) -> tuple[dict, dict] | None:
        run = runs.get(("frozen" if variant == "frozen" else variant, 1024, 16, 0))
        if run is None or not (run / "eval" / "test_coverage_arrays.npz").exists():
            return None
        with np.load(run / "eval" / "test_coverage_arrays.npz") as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, json.loads((run / "eval" / "test_coverage.json").read_text())

    def score(arrays: dict, report: dict, key: str) -> dict:
        valid = arrays["sae__valid_concepts"].astype(bool)
        out = {f"ucov_{t:.2f}": ucov(arrays[f"{key}__auc_matrix"], tau=t, valid=valid) for t in TAUS}
        return {**out, "n_valid": int(valid.sum()), "emf": float(report[key]["emf_simple"])}

    out: dict[str, dict] = {}
    base = load("batchtopk")
    if base is not None:
        for basis in DENSE:
            if f"{basis}__auc_matrix" in base[0]:
                out["Raw" if basis == "raw" else f"{DENSE_LABEL[basis]}-D"] = score(*base, basis)
    for variant in VARIANTS:
        loaded = load(variant)
        if loaded is not None:
            out[SAE_VARIANT[variant][0]] = score(*loaded, "sae")
    frozen = load("frozen")
    if frozen is not None:
        out["Frozen decoder"] = score(*frozen, "sae")
    if base is not None and "random_m1024__auc_matrix" in base[0]:
        out["Untrained random dictionary"] = score(*base, "random_m1024")
    return out


def from_paper_data(paper_data: Path) -> dict[str, dict[str, dict]]:
    """The table of :func:`rows_of` per embedding from the camera-ready ``tables.json`` and ``null_coverage.json``.

    ``n_valid`` is known for AlphaEarth only, and ``emf`` for the rows of :data:`PAPER_EMF`; both are
    ``None`` elsewhere.
    """
    tables = read_json(paper_file(paper_data, "tables.json"))
    emf = {
        r["arm"]: float(r["emf_sum"])
        for r in json.loads(paper_file(paper_data, "null_coverage.json").read_text())
        if int(r["n_features"]) == 1024
    }
    n_valid = {"alphaearth": int(tables["n_classes_ae"][0])}
    table = {}
    for emb, short in (("alphaearth", "ae"), ("tessera", "te")):
        rows = {}
        for key, label in PAPER_ROWS.items():
            if key in tables["ucov"][short]:
                r = tables["ucov"][short][key]
                rows[label] = {
                    **{f"ucov_{t:.2f}": int(r[f"ucov_{t:.2f}".replace(".", "")]) for t in TAUS},
                    "n_valid": n_valid.get(emb),
                    "emf": emf.get(PAPER_EMF.get(label)) if emb == "alphaearth" else None,
                }
        table[emb] = rows
    return table


def latex(table: dict[str, dict[str, dict]]) -> str:
    """The tabular of Table S6: one row per representation, two thresholds per embedding."""
    embs = list(table)
    labels = list(dict.fromkeys(label for rows in table.values() for label in rows))
    head = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{EMBEDDINGS[e][0]}}}" for e in embs) + r" \\"
    taus = " & " + " & ".join(rf"$\tau{{=}}{t:.2f}$" for _ in embs for t in TAUS) + r" \\"
    lines = [rf"\begin{{tabular}}{{l{'cc' * len(embs)}}}", r"\toprule", head, taus, r"\midrule"]
    for label in labels:
        cells = [
            f"${table[e][label][f'ucov_{t:.2f}']}$" if label in table[e] else "--" for e in embs for t in TAUS
        ]  # fmt: skip
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, nargs="+", help="sweep roots, e.g. runs/alphaearth runs/tessera")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "runs")
    table = {} if args.paper_data is None else from_paper_data(args.paper_data)
    for root in args.runs if args.paper_data is None else []:
        runs = find_runs(root)
        if not runs:
            raise FileNotFoundError(f"no runs under {root}")
        cfg = load_config([str(next(iter(runs.values())) / "config.yaml")], [])
        table[cfg["data"]["embedding_name"]] = rows_of(runs)
    for emb, rows in table.items():
        n_valid = {r["n_valid"] for r in rows.values() if r["n_valid"] is not None}
        print(f"== {emb}: scorable classes {sorted(n_valid)}")
        for label, r in rows.items():
            emf = "--" if r["emf"] is None else f"{r['emf']:.1f}"
            print(f"  {label:30s} UCov@0.60 {r['ucov_0.60']:4d}  UCov@0.65 {r['ucov_0.65']:4d}  EMF {emf}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_ucov.tex").write_text(latex(table))
    print("wrote", args.out / "table_ucov.tex")
    write_json(table, args.out / "table_ucov.json")


if __name__ == "__main__":
    main()

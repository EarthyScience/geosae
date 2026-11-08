"""Tables S4 and S5: reconstruction and concept alignment on further Earth embeddings.

Table S4: FVE on the test split of Copernicus-FM, OlmoEarth and MMEarth for BatchTopK SAEs and the
frozen-decoder baselines of the same M and k (seed means), and for the first L0 principal components
and the L0 raw dimensions of largest variance, L0 being the SAE's (the baseline width closest to it).
Reads one ``geosae panel`` run per embedding. Table S5: on OlmoEarth and MMEarth at the larger k, the
number of live features (best folded AUROC above 0.51), the shares aligned with at most one and with
exactly one of the 20 concept groups (folded AUROC >= 0.65), the number aligned with at least one and
the median count, from the ``geosae coverage`` arrays of every run (raw and PCA from the first SAE run).
``--paper-data`` reads the camera-ready ``products/<embedding>.json`` files instead.

    python scripts/neurips2026/table_further_embeddings.py --panel-copernicusfm eval/copernicusfm/panel \
        --panel-olmoearth eval/olmoearth/panel --panel-mmearth eval/mmearth/panel \
        --runs-olmoearth runs/olmoearth --runs-mmearth runs/mmearth --out figures
    python scripts/neurips2026/table_further_embeddings.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from _common import (
    add_paper_data_argument,
    find_runs,
    panel_rows,
    paper_file,
    parser,
    read_json,
    require,
    write_json,
)

from geosae.eval.coverage import default_class_to_group, n_groups_per_feature

LABELS = {"copernicusfm": "Copernicus-FM", "olmoearth": "OlmoEarth", "mmearth": "MMEarth"}
DIMS = {"copernicusfm": 768, "olmoearth": 768, "mmearth": 320}
#: Dictionary size of the SAEs of each embedding.
WIDTHS = {"copernicusfm": 1024, "olmoearth": 4096, "mmearth": 4096}
#: The embeddings of Table S5.
ALIGNED_EMBEDDINGS = ("olmoearth", "mmearth")
LIVE, TAU = 0.51, 0.65


def reconstruction(rows: list[dict]) -> list[dict]:
    """One row per (M, k) setting: SAE and frozen seed means, and PCA / raw at the SAE's L0."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] == "sae":
            cells[(int(r["latent_dim"]), int(r["k"]))]["frozen" if r["frozen"] else "sae"].append(r)
    dense = {r["kind"]: {} for r in rows if r["kind"] != "sae"}
    for r in rows:
        if r["kind"] != "sae":
            dense[r["kind"]][int(r["latent_dim"])] = r
    out = []
    for (m, k), arms in sorted(cells.items(), key=lambda c: c[0][1]):
        l0 = float(np.mean([r["l0"] for r in arms["sae"]]))
        row = {"M": m, "k": k, "l0": l0, "n_seeds": len(arms["sae"]), "n_frozen": len(arms["frozen"])}
        row["sae"] = float(np.mean([r["fve"] for r in arms["sae"]]))
        row["frozen"] = float(np.mean([r["fve"] for r in arms["frozen"]])) if arms["frozen"] else None
        for method, widths in dense.items():
            width = min(widths, key=lambda w: abs(w - l0))
            row[method], row[f"{method}_width"] = float(widths[width]["fve"]), width
        out.append(row)
    return out


def alignment(auc: np.ndarray, class_table: np.ndarray, product_order) -> dict:
    """Alignment counts over the 20 concept groups of the live features of one AUROC matrix."""
    c2g = default_class_to_group()
    grouped = np.array([(str(product_order[p]), int(c)) in c2g for p, c in class_table])
    auc = np.where(grouped[None, :], auc, 0.5)
    live = auc.max(1) > LIVE
    n = n_groups_per_feature(auc[live], class_table, list(product_order), c2g, tau=TAU)
    return {
        "live": int(live.sum()),
        "le1": float((n <= 1).mean()),
        "eq1": float((n == 1).mean()),
        "ge1": int((n >= 1).sum()),
        "median": float(np.median(n)),
    }


def alignment_table(runs_root: Path) -> dict:
    """Seed-mean alignment counts of the SAE and frozen decoder at the largest k, plus raw and PCA."""
    runs = find_runs(runs_root)
    k = max(key.k for key in runs)
    arms: dict = defaultdict(list)
    for key, run in sorted(runs.items()):
        if key.k != k:
            continue
        with np.load(run / "eval" / "test_coverage_arrays.npz") as z:
            ct, po = z["class_table"], z["product_order"]
            arms["frozen" if key.variant == "frozen" else "sae"].append(alignment(z["sae__auc_matrix"], ct, po))
            if key.variant != "frozen" and "raw" not in arms:
                for basis in ("raw", "pca"):
                    arms[basis] = [alignment(z[f"{basis}__auc_matrix"], ct, po)]
    return {arm: {s: float(np.mean([r[s] for r in rs])) for s in rs[0]} for arm, rs in arms.items()}


def from_paper_data(paper_data: Path) -> tuple[dict, dict]:
    """Table S4 rows and Table S5 counts from the camera-ready ``products/<embedding>.json`` files.

    Returns:
        ``(recon, align)`` in the structure of :func:`reconstruction` (with the embedding dimension) and
        :func:`alignment_table`.
    """
    recon, align = {}, {}
    for p in LABELS:
        d = read_json(paper_file(paper_data, f"products/{p}.json"))
        rows = []
        for sae, frozen, k in (("btk16", "frozen16", 16), ("btk", "frozen", DIMS[p] // 4)):
            l0 = float(np.mean([r["l0"] for r in d[sae]]))
            width = min((int(key.split("_")[1]) for key in d if key.startswith("pca_")), key=lambda w: abs(w - l0))
            row = {"M": WIDTHS[p], "k": k, "l0": l0, "n_seeds": len(d[sae]), "n_frozen": len(d[frozen])}
            row["sae"] = float(np.mean([r["fve"] for r in d[sae]]))
            row["frozen"] = float(np.mean([r["fve"] for r in d[frozen]]))
            for method in ("pca", "raw"):
                row[method], row[f"{method}_width"] = float(d[f"{method}_{width}"]["fve"]), width
            rows.append(row)
        recon[p] = (DIMS[p], rows)
        if p in ALIGNED_EMBEDDINGS:
            arms = {"sae": [r["align"] for r in d["btk"]], "frozen": [r["align"] for r in d["frozen"]]}
            arms |= {basis: [d[f"align_{basis}"]] for basis in ("raw", "pca")}
            keys = {"live": "n_live", "le1": "le1", "eq1": "eq1", "ge1": "n_aligned", "median": "median"}
            align[p] = {
                arm: {s: float(np.mean([r[src] for r in rs])) for s, src in keys.items()} for arm, rs in arms.items()
            }
    return recon, align


def latex_reconstruction(results: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Embedding & $D$ & $M$ & $k$ & $L_0$ & SAE & frozen & PCA & raw \\",
    ]
    for product, (dim, rows) in results.items():
        lines.append(r"\midrule")
        for i, r in enumerate(rows):
            name = LABELS[product] if i == 0 else ""
            frozen = f"{r['frozen']:.3f}" if r["frozen"] is not None else "--"
            lines.append(
                f"{name} & {dim} & {r['M']} & {r['k']} & {r['l0']:.0f} & {r['sae']:.3f} & {frozen} & "
                f"{r.get('pca', float('nan')):.3f} & {r.get('raw', float('nan')):.3f} \\\\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def latex_alignment(results: dict) -> str:
    names = (("raw", "raw dimensions"), ("pca", "PCA"), ("sae", "SAE"), ("frozen", "frozen decoder"))
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Embedding & Representation & live & $\leq 1$ & $=1$ & $\geq 1$ & median \\",
    ]
    for product, arms in results.items():
        lines.append(r"\midrule")
        for i, (arm, name) in enumerate(names):
            a = arms[arm]
            lines.append(
                f"{LABELS[product] if i == 0 else ''} & {name} & {a['live']:.0f} & {a['le1']:.2f} & "
                f"{a['eq1']:.2f} & {a['ge1']:.0f} & {a['median']:.0f} \\\\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    for p in LABELS:
        ap.add_argument(f"--panel-{p}", type=Path, help=f"geosae panel run dir ({LABELS[p]})")
    for p in ALIGNED_EMBEDDINGS:
        ap.add_argument(f"--runs-{p}", type=Path, help=f"root of the {LABELS[p]} runs, with coverage")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, *(f"panel-{p}" for p in LABELS), *(f"runs-{p}" for p in ALIGNED_EMBEDDINGS))
    if args.paper_data is not None:
        recon, align = from_paper_data(args.paper_data)
    else:
        recon = {p: (DIMS[p], reconstruction(panel_rows(getattr(args, f"panel_{p}")))) for p in LABELS}
        align = {p: alignment_table(getattr(args, f"runs_{p}")) for p in ALIGNED_EMBEDDINGS}
    for p, (_, rows) in recon.items():
        for r in rows:
            print(LABELS[p], r)
    for p, arms in align.items():
        for arm, a in arms.items():
            print(LABELS[p], arm, a)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_further_embeddings_fve.tex").write_text(latex_reconstruction(recon))
    (args.out / "table_further_embeddings_alignment.tex").write_text(latex_alignment(align))
    write_json(
        {"fve": {p: rows for p, (_, rows) in recon.items()}, "alignment": align},
        args.out / "table_further_embeddings.json",
    )


if __name__ == "__main__":
    main()

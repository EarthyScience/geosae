"""Table S9 and the area-of-applicability numbers of the appendix, plus the leave-Europe-out maps' data.

Re-scores the canopy-height Europe folds of ``geosae aoa`` (``configs/neurips2026/aoa.yaml``) from the
cached held-out sites of each run (``eval/aoa/sites.npz``) with ``geosae.eval.aoa.score_fold``:

* Table S9: Spearman rho between the ridge-weighted distance and the absolute error of the head
  fitted on the same representation, raw and SAE, on the train-on-Europe and leave-Europe-out folds,
  per checkpoint, with the heads' test R^2.
* For the map checkpoint (``--map-run``, the Matryoshka SAE): per held-out pixel of the
  leave-Europe-out fold the ridge-weighted distances and Meyer & Pebesma's dissimilarity index in
  units of its threshold (training index over 10-degree block CV folds), and the agreement of the
  two out-of-domain masks; written to ``aoa_europe.npz`` for ``figS_aoa.py`` and ``figS_aoa_gain.py``.
* The continent and plant-functional-type folds: from each run's ``eval/aoa/records.csv``, the cells
  (fold, target, checkpoint) in which the SAE distance ranks the error better than the raw distance.

    python scripts/neurips2026/table_aoa_europe.py runs/alphaearth/batchtopk_M1024_k16_s*/* \
        --map-run runs/alphaearth/matryoshka_M1024_k16_s0/* --out figures
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from _common import parser, run_key, write_json

from geosae.eval import aoa as A

FOLD, TARGET = "Europe", "canopy_height"
SCENARIOS = {"train_on_europe": "single_continent", "leave_europe_out": "leave_one_continent_out"}
LABEL = {"batchtopk": "BatchTopK", "matryoshka": "Matryoshka", "relu_l1": "ReLU+$\\ell_1$", "frozen": "Frozen decoder"}


def block_cv_folds(lonlat: np.ndarray, size_deg: float = 10.0, n_folds: int = 10) -> np.ndarray:
    """CV fold per site: ``size_deg`` lon-lat tiles assigned at random (seed 0) to ``n_folds`` folds."""
    tiles = np.column_stack([np.floor(lonlat[:, 0] / size_deg), np.floor(lonlat[:, 1] / size_deg)])
    tile = np.unique(tiles, axis=0, return_inverse=True)[1].ravel()
    fold_of_tile = np.empty(int(tile.max()) + 1, dtype=np.int64)
    fold_of_tile[np.random.default_rng(0).permutation(fold_of_tile.size)] = np.arange(fold_of_tile.size) % n_folds
    return fold_of_tile[tile]


def score_europe(sites: A.AoASites, scenario: str, variant: str, settings: dict, **kw) -> dict:
    """Raw and SAE space of one canopy-height Europe fold, each against its own ridge head."""
    tr, te = {n: (a, b) for n, a, b in A.scenario_folds(scenario, sites.lonlat, sites.pft)}[FOLD]
    y = sites.targets[TARGET]
    fin = np.isfinite(y)
    rec = {
        s: A.score_fold(sites.spaces[s], np.nan_to_num(y), tr & fin, te & fin, variants=(variant,), **settings, **kw)
        for s in ("raw", "sae")
    }
    if not np.array_equal(rec["raw"]["test_idx"], rec["sae"]["test_idx"]):
        raise ValueError("raw and SAE spaces scored different held-out pixels")
    return rec


def europe_maps(sites: A.AoASites, settings: dict) -> tuple[dict, dict]:
    """Per-pixel maps of the leave-Europe-out fold and the agreement of the two spaces on both folds."""
    v = "ridge"
    cv = block_cv_folds(sites.lonlat)
    rec = score_europe(sites, SCENARIOS["leave_europe_out"], v, settings, threshold_rule="cast", cv_folds=cv)
    di = {s: r[f"aoa_di_test_{v}"] / r[f"aoa_threshold_{v}"] for s, r in rec.items()}
    mask = A.di_agreement(di["raw"], di["sae"], threshold=1.0)
    rank = A.di_agreement(rec["raw"][f"d_test_{v}"], rec["sae"][f"d_test_{v}"])
    maps = {"lonlat": sites.lonlat[rec["raw"]["test_idx"]], "overlay": np.asarray(mask["overlay"])}
    for s, r in rec.items():
        maps[f"di_{s}"], maps[f"err_{s}"], maps[f"aoa_{s}"] = r[f"d_test_{v}"], r["err"], di[s]
    maps |= {f"agree_{k}": np.asarray(val) for k, val in mask.items() if k != "overlay"}
    maps["agree_spearman"] = np.asarray(rank["spearman"])

    single = score_europe(sites, SCENARIOS["train_on_europe"], "unweighted", settings, threshold_rule="cast",
                          cv_folds=cv)  # fmt: skip
    s_di = {s: r["aoa_di_test_unweighted"] / r["aoa_threshold_unweighted"] for s, r in single.items()}
    s_mask = A.di_agreement(s_di["raw"], s_di["sae"], threshold=1.0)
    s_rank = A.di_agreement(single["raw"]["d_test_unweighted"], single["sae"]["d_test_unweighted"])
    summary = {
        "leave_europe_out": {
            "n_test": int(maps["lonlat"].shape[0]),
            "inside_raw": 1.0 - mask["frac_out_a"],
            "inside_sae": 1.0 - mask["frac_out_b"],
            "n_differ": int(mask["n_only_a"] + mask["n_only_b"]),
            "n_sae_out_only": int(mask["n_only_b"]),
            "n_raw_out_only": int(mask["n_only_a"]),
            "mask_kappa": mask["mask_kappa"],
            "rank_spearman": rank["spearman"],
        },
        "train_on_europe": {
            "n_test": int(single["raw"]["test_idx"].size),
            "out_raw_unweighted": s_mask["frac_out_a"],
            "out_sae_unweighted": s_mask["frac_out_b"],
            "mask_kappa_unweighted": s_mask["mask_kappa"],
            "rank_spearman_unweighted": s_rank["spearman"],
        },
    }
    return maps, summary


def europe_table(run_dirs: list[Path], settings: dict) -> list[dict]:
    """Table S9: per checkpoint, rho and test R^2 of raw and SAE on both Europe folds."""
    rows = []
    for run in run_dirs:
        sites = A.load_sites(str(Path(run) / "eval" / "aoa" / "sites.npz"))
        key = run_key(run)
        row = {"checkpoint": f"{LABEL.get(key.variant, key.variant)}, seed {key.seed}", "run": str(run)}
        for name, scenario in SCENARIOS.items():
            rec = score_europe(sites, scenario, "ridge", settings)
            for s, r in rec.items():
                row[f"{name}_rho_{s}"] = r["rho_ridge"]
                row[f"{name}_r2_{s}"] = r["r2_test"]
        rows.append(row)
    return rows


def fold_cells(run_dirs: list[Path]) -> dict:
    """Wins of the SAE over the raw distance on the continent and PFT folds, over every checkpoint."""
    cells: dict = defaultdict(dict)
    for run in run_dirs:
        path = Path(run) / "eval" / "aoa" / "records.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `geosae aoa` with configs/neurips2026/aoa.yaml")
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if r["space"] not in ("raw", "sae") or r["metric"] != "euclidean":
                    continue
                key = (str(run), r["scenario"], r["fold"], r["target"], r["scaling"], r["variant"])
                cells[key][r["space"]] = float(r["rho"])
    gain = {k: v["sae"] - v["raw"] for k, v in cells.items() if {"raw", "sae"} <= set(v)}

    def stat(**where) -> dict:
        d = np.array([g for k, g in gain.items() if _match(k, **where)])
        return {"n": int(d.size), "wins": int((d > 0).sum()), "median": float(np.median(d)) if d.size else np.nan}

    out = {
        f"{scaling}/{variant}": stat(scaling=scaling, variant=variant)
        for scaling in sorted({k[4] for k in gain})
        for variant in ("unweighted", "ridge")
    }
    for run in run_dirs:
        for variant in ("unweighted", "ridge"):
            out[f"{run}/none/{variant}"] = stat(run=str(run), scaling="none", variant=variant)
    out["none/ridge/leave_one_out"] = stat(scaling="none", variant="ridge", loo=True)
    out["none/ridge/single"] = stat(scaling="none", variant="ridge", loo=False)
    for t in sorted({k[3] for k in gain}):
        out[f"none/ridge/{t}"] = stat(scaling="none", variant="ridge", target=t)
    return out


def _match(key: tuple, *, run=None, scaling=None, variant=None, target=None, loo=None) -> bool:
    k_run, scenario, _, k_target, k_scaling, k_variant = key
    return (
        (run is None or k_run == run)
        and (scaling is None or k_scaling == scaling)
        and (variant is None or k_variant == variant)
        and (target is None or k_target == target)
        and (loo is None or scenario.startswith("leave") == loo)
    )


def latex(rows: list[dict]) -> str:
    """Table S9 as a LaTeX tabular."""
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{train on Europe} & \\multicolumn{2}{c}{leave Europe out} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "checkpoint & raw & SAE & raw & SAE \\\\",
        "\\midrule",
    ]
    for r in rows:
        vals = [r[f"{f}_rho_{s}"] for f in SCENARIOS for s in ("raw", "sae")]
        lines.append(r["checkpoint"] + " & " + " & ".join(f"${v:+.2f}$" for v in vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run directories scored with `geosae aoa` (Table S9 rows)")
    ap.add_argument("--map-run", type=Path, required=True, help="run directory whose leave-Europe-out fold is mapped")
    ap.add_argument("--k", type=int, default=5, help="nearest training pixels")
    ap.add_argument("--max-train", type=int, default=12_000, help="training sites of a head")
    ap.add_argument("--max-test", type=int, default=8_000, help="held-out sites scored")
    ap.add_argument("--seed", type=int, default=0, help="seeds the site subsamples")
    args = ap.parse_args()
    settings = dict(k=args.k, alpha=1.0, max_train=args.max_train, max_test=args.max_test, seed=args.seed)
    runs = list(dict.fromkeys([*args.runs, args.map_run]))

    rows = europe_table(runs, settings)
    for r in rows:
        print(f"{r['checkpoint']:24s} " + "  ".join(
            f"{f}: rho raw {r[f'{f}_rho_raw']:+.2f} SAE {r[f'{f}_rho_sae']:+.2f} "
            f"(R2 {r[f'{f}_r2_raw']:.3f} / {r[f'{f}_r2_sae']:.3f})" for f in SCENARIOS))  # fmt: skip
    maps, agreement = europe_maps(A.load_sites(str(args.map_run / "eval" / "aoa" / "sites.npz")), settings)
    for fold, st in agreement.items():
        print(fold + ": " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in st.items()))
    cells = fold_cells(runs)
    for name, st in cells.items():
        print(f"continent/PFT {name}: SAE better in {st['wins']}/{st['n']} cells, median gain {st['median']:+.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "aoa_europe.npz", **maps)
    (args.out / "table_aoa_europe.tex").write_text(latex(rows))
    write_json({"table": rows, "map_run": str(args.map_run), "agreement": agreement, "folds": cells},
               args.out / "table_aoa_europe.json")  # fmt: skip
    print("wrote", args.out / "aoa_europe.npz", args.out / "table_aoa_europe.tex")


if __name__ == "__main__":
    main()

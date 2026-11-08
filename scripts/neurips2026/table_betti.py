"""Table S7: the monosemanticity scores of Betti et al. (GeoMS, and MS when image embeddings are given).

A patch is represented by the mean activation of each feature over its pixels, located at its
centre. GeoMS weighs the pairwise similarity ``exp(-distance / 500 km)`` of the patches by the
min-max scaled activations of a feature (:mod:`geosae.eval.monosemanticity`); MS does the same
with the cosine similarity of image embeddings of the patches. Features active on fewer than 20
patches are not scored; *uniform* is the score of a feature that fires equally everywhere.

GeoMS uses 1,500 test patches drawn at random (seed 0) from the runs' predictions. SAE patch means
come from the ``feat_patch_stats`` of ``geosae predict``; the raw embedding is averaged over the
same patches from the store, in the units of the runs' transforms, and PCA and ICA are fitted to
and applied to these patch means. MS needs ``--image-embeddings``: an ``.npz`` with ``cell``
(store rows of test patches) and ``embedding`` (one image embedding per patch, e.g. the SatCLIP
image tower on the Sentinel-2 L2A crop of the patch, as in Betti et al.); without it MS is skipped.
Writes ``table_betti.tex`` and ``table_betti.json``.

``--paper-data`` reads the camera-ready scores instead: ``betti_scores_<ae|te>.json`` (every arm, with
the further BatchTopK seeds of AlphaEarth) and ``betti_relu_fixed.json`` (the ReLU+l1 rows).

    python scripts/neurips2026/table_betti.py --runs runs/alphaearth runs/tessera --out figures
    python scripts/neurips2026/table_betti.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from _common import (
    EMBEDDINGS,
    SAE_VARIANT,
    VARIANTS,
    add_paper_data_argument,
    find_runs,
    paper_file,
    parser,
    predictions,
    read_json,
    require,
    write_json,
)

from geosae.config import load_config
from geosae.eval import monosemanticity as ms
from geosae.eval.baselines import project_to_basis
from geosae.eval.registry import sample_registry_pixels
from geosae.pipeline import apply_transforms, load_run_transforms

N_PATCHES, MIN_ACTIVE, LENGTH_KM = 1500, 20, 500.0
SEEDS = (0, 1, 2)
#: Row label of every arm of the camera-ready scores, in table order.
PAPER_ROWS = {
    "raw": "Raw",
    "pca": "PCA",
    "ica": "ICA",
    "frozen": "Frozen decoder",
    **{v: SAE_VARIANT[v][0] for v in VARIANTS},
    "sweep_batchtopk": "BatchTopK, 3 seeds",
}
#: Key of each score in the camera-ready files.
PAPER_SCORES = {"geoms": "geoms_500km", "ms": "ms_image_set"}


def patch_means_of_run(run: Path, cells: np.ndarray) -> np.ndarray:
    """``(N, F)`` mean activation of every feature on the test patches ``cells``."""
    g = zarr.open_group(str(predictions(run)), mode="r")
    row = {int(c): i for i, c in enumerate(np.asarray(g["cell"][:]))}
    idx = np.array([row[int(c)] for c in cells])
    order = np.argsort(idx)
    stats = np.asarray(g["feat_patch_stats"].get_orthogonal_selection((slice(None), idx[order], 0)))
    out = np.empty((len(cells), stats.shape[0]), dtype=np.float64)
    out[order] = stats.T
    return out


def raw_patch_means(run: Path, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Patch-mean embedding of the test patches ``cells``, in the units of ``run``'s transforms.

    Returns:
        The cells the store's quality filters keep, in the order of ``cells``, and their
        ``(N, D)`` patch means.
    """
    cfg = load_config([str(run / "config.yaml")], [])
    d = cfg["data"]
    reg = sample_registry_pixels(
        d["zarr_path"],
        split="test",
        split_name=d.get("split_name", "split_210k"),
        embedding_name=d.get("embedding_name", "alphaearth"),
        pixels_per_patch=4096,
        products=[],
        cells=cells,
    )
    x = apply_transforms(load_run_transforms(run), reg.raw).astype(np.float64)
    got, inv = np.unique(reg.cell, return_inverse=True)
    sums = np.zeros((got.size, x.shape[1]))
    np.add.at(sums, inv, x)
    means = sums / np.bincount(inv)[:, None]
    pos = {int(c): i for i, c in enumerate(got)}
    kept = np.array([c for c in cells if int(c) in pos], dtype=np.int64)
    return kept, means[[pos[int(c)] for c in kept]]


def summary(acts: np.ndarray, sets: dict) -> dict:
    """GeoMS (and MS) mean and max over the scored features of one dictionary."""
    lon, lat = sets["geo"]["lonlat"].T
    out = {"geoms": ms.summarise_monosemanticity(ms.geo_monosemanticity_score(acts["geo"], lon, lat,
                                                  length_scale_km=LENGTH_KM, min_active=MIN_ACTIVE))}  # fmt: skip
    if "img" in sets:
        out["ms"] = ms.summarise_monosemanticity(
            ms.monosemanticity_score(acts["img"], sets["img"]["embedding"], min_active=MIN_ACTIVE)
        )
    return out


def score_embedding(root: Path, image_npz: Path | None) -> dict:
    """Every row of Table S7 for one sweep root."""
    runs = find_runs(root)
    cell_runs = {
        label: run
        for label, key in (
            *((SAE_VARIANT[v][0], (v, 1024, 16, 0)) for v in VARIANTS),
            *((f"frozen_s{s}", ("frozen", 1024, 16, s)) for s in SEEDS),
            *((f"batchtopk_s{s}", ("batchtopk", 1024, 16, s)) for s in SEEDS[1:]),
        )
        if (run := runs.get(key)) is not None and (run / "predictions" / "test.zarr").exists()
    }
    if not cell_runs:
        raise FileNotFoundError(f"no run under {root} has predictions; run `geosae predict --run <run>`")
    ref = next(iter(cell_runs.values()))
    g = zarr.open_group(str(predictions(ref)), mode="r")
    pred_cells = np.asarray(g["cell"][:])
    pick = np.sort(np.random.default_rng(0).choice(pred_cells.size, N_PATCHES, replace=False))  # 1,500 test patches
    sets = {"geo": {"cell": pred_cells[pick], "lonlat": np.column_stack([g["lon"][:][pick], g["lat"][:][pick]])}}
    if image_npz is not None:
        with np.load(image_npz) as z:
            keep = np.isin(z["cell"], pred_cells)
            sets["img"] = {"cell": z["cell"][keep], "embedding": z["embedding"][keep].astype(np.float64)}

    raw = {}
    for k, s in sets.items():
        kept, raw[k] = raw_patch_means(ref, s["cell"])
        keep = np.isin(s["cell"], kept)
        sets[k] = {name: v[keep] for name, v in s.items()}
    lon, lat = sets["geo"]["lonlat"].T
    uniform = {"geoms": ms.mean_pairwise_similarity(lon=lon, lat=lat, length_scale_km=LENGTH_KM)}
    if "img" in sets:
        uniform["ms"] = ms.mean_pairwise_similarity(embeddings=sets["img"]["embedding"])
    rows: dict[str, list[dict]] = {}
    rows["Raw"] = [summary(raw, sets)]
    rows["PCA"] = [summary({k: project_to_basis(v, "pca") for k, v in raw.items()}, sets)]
    rows["ICA"] = [summary({k: project_to_basis(v, "ica", seed=s) for k, v in raw.items()}, sets) for s in SEEDS]
    for label, run in cell_runs.items():
        print(f"  scoring {label}", flush=True)
        rows[label] = [summary({k: patch_means_of_run(run, s["cell"]) for k, s in sets.items()}, sets)]
    frozen = [rows.pop(f"frozen_s{s}")[0] for s in SEEDS if f"frozen_s{s}" in rows]
    if frozen:
        rows["Frozen decoder"] = frozen
    seeds = [rows[SAE_VARIANT["batchtopk"][0]][0]] if SAE_VARIANT["batchtopk"][0] in rows else []
    seeds += [rows.pop(f"batchtopk_s{s}")[0] for s in SEEDS[1:] if f"batchtopk_s{s}" in rows]
    if len(seeds) > 1:
        rows["BatchTopK, 3 seeds"] = seeds
    return {"uniform": uniform, "n_patches": {k: int(len(s["cell"])) for k, s in sets.items()}, "rows": rows}


def from_paper_data(paper_data: Path) -> dict:
    """The table of :func:`score_embedding` per embedding from the camera-ready Betti et al. scores."""
    relu = read_json(paper_file(paper_data, "betti_relu_fixed.json"))
    table = {}
    for emb, short in (("alphaearth", "ae"), ("tessera", "te")):
        d = read_json(paper_file(paper_data, f"betti_scores_{short}.json"))
        arms = {**d["arms"], "relu_l1": {"seeds": [relu[short]]}}

        def fit(scores: dict) -> dict:
            return {s: scores[key] for s, key in PAPER_SCORES.items() if key in scores}

        table[emb] = {
            "uniform": {s: float(d["mean_pairwise_similarity"][key]) for s, key in PAPER_SCORES.items()},
            "n_patches": {"geo": int(d["n_patches"]["moran_set"]), "img": int(d["n_patches"]["image_set"])},
            "rows": {label: [fit(f) for f in arms[arm]["seeds"]] for arm, label in PAPER_ROWS.items() if arm in arms},
        }
    return table


def aggregate(fits: list[dict], score: str, stat: str) -> tuple[float, float]:
    """Mean and sample standard deviation over seeds (or fits) of one statistic."""
    v = np.array([f[score][stat] for f in fits if score in f], dtype=float)
    if not v.size:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def latex(table: dict) -> str:
    """The tabular of Table S7: GeoMS and MS mean and max per embedding."""
    embs = list(table)
    scores = [s for s in ("geoms", "ms") if all(s in t["uniform"] for t in table.values())]
    labels = list(dict.fromkeys(lb for t in table.values() for lb in ["Uniform", *t["rows"]]))
    lines = [rf"\begin{{tabular}}{{l{'cc' * len(embs) * len(scores)}}}", r"\toprule"]
    lines.append(
        " & " + " & ".join(rf"\multicolumn{{{2 * len(scores)}}}{{c}}{{{EMBEDDINGS[e][0]}}}" for e in embs) + r" \\"
    )
    names = {"geoms": "GeoMS", "ms": "MS"}
    lines.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{names[s]}}}" for _ in embs for s in scores) + r" \\")
    lines.append(" & " + " & ".join("mean & max" for _ in embs for _ in scores) + r" \\")
    lines.append(r"\midrule")
    for label in labels:
        cells = []
        for e in embs:
            for s in scores:
                if label == "Uniform":
                    cells += [f"${table[e]['uniform'][s]:.3f}$", "--"]
                    continue
                fits = table[e]["rows"].get(label)
                for stat in ("mean", "max"):
                    if not fits:
                        cells.append("--")
                        continue
                    m, sd = aggregate(fits, s, stat)
                    pm = rf"{{\scriptstyle\pm{sd:.3f}}}" if len(fits) > 1 and label != "ICA" else ""
                    cells.append(f"${m:.3f}{pm}$")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, nargs="+", help="sweep roots, e.g. runs/alphaearth runs/tessera")
    ap.add_argument("--image-embeddings", type=Path, default=None, help="optional .npz (cell, embedding) for MS")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "runs")
    table = {} if args.paper_data is None else from_paper_data(args.paper_data)
    for root in args.runs if args.paper_data is None else []:
        runs = find_runs(root)
        emb = load_config([str(next(iter(runs.values())) / "config.yaml")], [])["data"]["embedding_name"]
        print(f"== {emb}", flush=True)
        table[emb] = score_embedding(Path(root), args.image_embeddings)
    for emb, t in table.items():
        print(f"== {emb}")
        for label, fits in t["rows"].items():
            parts = [
                f"{s} mean {aggregate(fits, s, 'mean')[0]:.3f} max {aggregate(fits, s, 'max')[0]:.3f}"
                f" scored {aggregate(fits, s, 'n_scored')[0]:.0f}"
                for s in t["uniform"]
            ]
            print(f"  {label:20s} n={len(fits)} " + "; ".join(parts))
        print(f"  uniform: {t['uniform']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_betti.tex").write_text(latex(table))
    print("wrote", args.out / "table_betti.tex")
    write_json(table, args.out / "table_betti.json")


if __name__ == "__main__":
    main()

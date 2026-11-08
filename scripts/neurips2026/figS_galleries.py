"""Figs. S10 and S11: galleries of aligned features, six concepts on AlphaEarth (figS_qual_only) or Tessera.

Per concept, the Matryoshka feature and the raw embedding dimension best aligned with one reference
class (largest folded AUROC in the run's ``geosae coverage`` arrays, ``eval/test_coverage_arrays.npz``;
no feature is shown twice, and a feature must be more active on the class than off it, judged by its
patch means weighted by the class share of each candidate patch) are drawn on three test patches
where the class covers part of the patch and three where it is absent, under the patch's Sentinel-2
true-colour image (the store's ``rgb``, else its ``sentinel2_l2a`` bands; ``--esri``: Esri World
Imagery).
The raw dimension (``raw d<i>``) is oriented so that the class is high (``−`` marks a sign flip).

Patch choice is illustrative, not random. Candidates are the first ``--n-candidates`` test patches of
the store (whose rows are in random order) within 65° of the equator. A present patch has the class
on more than 5% and at most a concept-specific cap of its pixels, and, if more than four times
enough qualify, the feature in the upper half of its patch mean among them. An absent patch has none
of the class, preferably a silent feature (patch mean at most 0.2 times the median over present
candidates) and, among many, the raw dimension in its top quartile. Three of each are drawn with
``--seed + concept index``.

With ``--paper-data <dir>`` the gallery of ``--embedding`` is drawn from the camera-ready inputs
instead (``qualitative_{ae,te}.npz``), with Esri World Imagery at the recorded patch centres.

    python scripts/neurips2026/figS_galleries.py runs/alphaearth/matryoshka_M1024_k16_s0 --out figures
    python scripts/neurips2026/figS_galleries.py --paper-data paper_data --embedding tessera --out figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from _common import (
    add_imagery_argument,
    add_paper_data_argument,
    fs,
    imagery,
    parser,
    require,
    save,
    set_style,
    write_json,
)
from fig5_aligned_features import esri_tile, paper_gallery
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from geosae.config import load_config
from geosae.eval.ladder import model_encoder
from geosae.pipeline import apply_transforms, load_run_model, load_run_transforms, make_data

#: The concepts of each gallery: (name, product, class code, largest class share of a present patch).
CONCEPTS = {
    "alphaearth": [
        ("natural water", "lc_esa_wc", 80, 0.35),
        ("built-up", "lc_esa_wc", 50, 0.40),
        ("forest", "lc_esa_wc", 10, 0.95),
        ("cropland", "lc_esa_wc", 40, 0.95),
        ("bare / sparse", "lc_esa_wc", 60, 0.95),
        ("snow / ice", "lc_esa_wc", 70, 0.95),
    ],
    "tessera": [
        ("natural water", "lc_esa_wc", 80, 0.35),
        ("built-up", "lc_esa_wc", 50, 0.40),
        ("forest", "lc_esa_wc", 10, 0.95),
        ("cropland", "lc_esa_wc", 40, 0.95),
        ("bare / paved", "lcz", 15, 0.95),
        ("snow / ice", "lc_esa_wc", 70, 0.95),
    ],
}
NAME = {"alphaearth": "figS_qual_only", "tessera": "figS_qual_only_tessera"}
PRESENT_MIN = 0.05
LAT_MAX = 65.0
CMAP = "inferno"


def coverage_columns(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Folded AUROC of the run's features and of the raw dimensions, and the ``(product, code)`` of each column."""
    path = run_dir / "eval" / "test_coverage_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `geosae coverage --run {run_dir} -c <coverage.yaml>` first")
    with np.load(path) as z:
        order = [str(p) for p in z["product_order"]]
        columns = [(order[int(p)], int(c)) for p, c in z["class_table"]]
        return z["sae__auc_matrix"], z["raw__auc_matrix"], columns


def read_patches(group: zarr.Group, key: str, cells: np.ndarray) -> np.ndarray:
    """``group[key]`` at sorted store rows ``cells``."""
    return np.asarray(group[key].get_orthogonal_selection((np.asarray(cells, dtype=np.int64),)))


def patch_statistics(model, transforms, group, embedding: str, cells, *, device: str, batch: int = 8):
    """Patch means of every feature and every (transformed) raw dimension, and the patches' raw embeddings."""
    encode = model_encoder(model, device=device)
    feat, raw_mean, raw_all = [], [], []
    for s in range(0, len(cells), batch):
        x = read_patches(group, embedding, cells[s : s + batch]).astype(np.float32)
        n, h, w, d = x.shape
        x = apply_transforms(transforms, x.reshape(-1, d))
        feat.append(encode(x).reshape(n, h * w, -1).mean(1).astype(np.float32))
        raw_mean.append(x.reshape(n, h * w, d).mean(1))
        raw_all.append(x.reshape(n, h, w, d).astype(np.float16))
    return np.concatenate(feat), np.concatenate(raw_mean), np.concatenate(raw_all)


def choose(frac, sae_mean, raw_mean, eligible, *, n, present_max, rng):
    """``n`` present and ``n`` absent candidate indices (see the module docstring)."""
    present = np.flatnonzero((frac > PRESENT_MIN) & (frac <= present_max) & eligible)
    if len(present) > 4 * n:
        present = present[np.argsort(-sae_mean[present])][: max(2 * n, len(present) // 2)]
    absent = np.flatnonzero((frac <= 0.0) & eligible)
    if len(present):
        silent = absent[sae_mean[absent] <= 0.2 * float(np.median(sae_mean[present]))]
        if len(silent) >= n:
            absent = silent
    if len(absent) > 4 * n:
        loud = absent[raw_mean[absent] >= np.quantile(raw_mean[absent], 0.75)]
        if len(loud) >= n:
            absent = loud
    rng.shuffle(present)
    rng.shuffle(absent)
    return present[:n].tolist(), absent[:n].tolist()


def collect(
    run_dir: Path, *, n_candidates: int, n_half: int, seed: int, device: str, esri: bool = False
) -> tuple[list[dict], str]:
    """Choose features, dimensions and patches of every concept, and their 64 x 64 maps."""
    cfg = load_config([str(run_dir / "config.yaml")], [])
    embedding = cfg["data"].get("embedding_name", "alphaearth")
    group = zarr.open_group(str(cfg["data"]["zarr_path"]), mode="r")
    cells = make_data(cfg, "test").cells[:n_candidates]
    lat = np.asarray(group["lat"][:])[cells]
    lon = np.asarray(group["lon"][:])[cells]
    eligible = np.abs(lat) <= LAT_MAX
    auc_sae, auc_raw, columns = coverage_columns(run_dir)
    model, transforms = load_run_model(run_dir), load_run_transforms(run_dir)
    feat_mean, raw_mean, raw_all = patch_statistics(model, transforms, group, embedding, cells, device=device)
    labels = {p: read_patches(group, p, cells) for p in {c[1] for c in CONCEPTS[embedding]}}
    encode = model_encoder(model, device=device)
    used: set[int] = set()
    out = []
    for ci, (name, product, code, present_max) in enumerate(CONCEPTS[embedding]):
        j = columns.index((product, code))
        lab = labels[product]
        in_class, other = lab == code, (lab != code) & (lab != 0)
        frac = in_class.mean((1, 2))
        on = (frac[:, None] * feat_mean).sum(0) / max(frac.sum(), 1e-12)
        off = ((1 - frac)[:, None] * feat_mean).sum(0) / max((1 - frac).sum(), 1e-12)
        feat = next(int(f) for f in np.argsort(-auc_sae[:, j]) if int(f) not in used and on[f] > off[f])
        used.add(feat)
        dim = int(np.argmax(auc_raw[:, j]))
        x_dim = raw_all[..., dim].astype(np.float32)
        sign = 1.0 if x_dim[in_class].mean() >= x_dim[other].mean() else -1.0
        present, absent = choose(
            frac,
            feat_mean[:, feat],
            sign * raw_mean[:, dim],
            eligible,
            n=n_half,
            present_max=present_max,
            rng=np.random.default_rng(seed + ci),
        )
        picks = present + absent
        sae = [
            encode(raw_all[p].reshape(-1, raw_all.shape[-1]).astype(np.float32))[:, feat].reshape(lab.shape[1:])
            for p in picks
        ]
        rgb = [imagery(group, int(cells[p]), esri=esri) for p in picks]
        out.append(
            {
                "name": name,
                "product": product,
                "code": code,
                "feature": feat,
                "feature_auroc": float(auc_sae[feat, j]),
                "dim": dim,
                "dim_sign": sign,
                "dim_auroc": float(auc_raw[dim, j]),
                "n_present": len(present),
                "cell": [int(cells[p]) for p in picks],
                "lon": [float(lon[p]) for p in picks],
                "lat": [float(lat[p]) for p in picks],
                "class_share": [float(frac[p]) for p in picks],
                "_sae": np.asarray(sae, np.float32),
                "_raw": np.asarray([sign * x_dim[p] for p in picks], np.float32),
                "_rgb": rgb,
            }
        )
        print(
            f"{name:<14} f{feat} AUROC {out[-1]['feature_auroc']:.3f} | raw {'−' if sign < 0 else ''}d{dim} "
            f"AUROC {out[-1]['dim_auroc']:.3f} | present {len(present)}, absent {len(absent)}"
        )
    return out, embedding


def from_paper_data(paper_data: Path, embedding: str) -> list[dict]:
    """The concepts of :func:`collect` from the camera-ready ``qualitative_<ae|te>.npz``."""
    backbone = {"alphaearth": "ae", "tessera": "te"}[embedding]
    out = []
    for c, sae, raw in paper_gallery(paper_data, backbone, range(len(CONCEPTS[embedding]))):
        out.append(
            {
                "name": c["name"],
                "product": c["product"],
                "code": int(c["code"]),
                "feature": int(c["feat"]),
                "feature_auroc": float(c["feat_auc"]),
                "dim": int(c["dim"]),
                "dim_sign": float(c["dim_sign"]),
                "dim_auroc": float(c["dim_auc"]),
                "n_present": int(c["n_present"]),
                "cell": c["cell"],
                "lon": c["lon"],
                "lat": c["lat"],
                "class_share": c["class_frac"],
                "_sae": sae,
                "_raw": raw,
                "_rgb": [esri_tile(lo, la) for lo, la in zip(c["lon"], c["lat"], strict=True)],
            }
        )
        print(f"{c['name']:<14} f{c['feat']} AUROC {c['feat_auc']:.3f} | raw d{c['dim']} AUROC {c['dim_auc']:.3f}")
    return out


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _coord(lat: float, lon: float) -> str:
    return f"{abs(lat):.0f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.0f}°{'E' if lon >= 0 else 'W'}"


def draw_tiles(fig, spec, r0: int, c0: int, concept: dict, *, coord_size: float | None) -> None:
    """Imagery, feature and raw-dimension rows of one concept at grid rows ``r0..r0+2``, columns ``c0..``.

    The feature row is scaled from zero to its maximum, the raw row from its 5th to 95th percentile,
    both over the concept's tiles.
    """
    sae, raw = concept["_sae"], concept["_raw"]
    vmax = float(sae.max()) or 1.0
    lo, hi = (float(v) for v in np.percentile(raw, [5, 95]))
    hi = hi if hi > lo else lo + 1.0
    for k in range(len(sae)):
        ax_i = fig.add_subplot(spec[r0, c0 + k])
        rgb = concept["_rgb"][k]
        if rgb is None:
            ax_i.set_facecolor("#e6e6e6")
        else:
            ax_i.imshow(rgb, interpolation="nearest")
        if coord_size:
            ax_i.text(
                0.04,
                0.96,
                _coord(concept["lat"][k], concept["lon"][k]),
                transform=ax_i.transAxes,
                fontsize=coord_size,
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.06", "fc": "white", "alpha": 0.78, "lw": 0},
            )
        ax_s = fig.add_subplot(spec[r0 + 1, c0 + k])
        ax_s.imshow(sae[k], cmap=CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        ax_r = fig.add_subplot(spec[r0 + 2, c0 + k])
        ax_r.imshow(raw[k], cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest")
        for ax in (ax_i, ax_s, ax_r):
            _clean(ax)
            if k == concept["n_present"] - 1:
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_color("#444444")
                ax.spines["right"].set_linewidth(0.6)


def raw_label(concept: dict) -> str:
    """Row label of the raw dimension, ``−`` marking a sign-flipped one."""
    return f"raw {'−' if concept['dim_sign'] < 0 else ''}d{concept['dim']}"


def draw_column(fig, subspec, concepts: list[dict]) -> None:
    """Concept blocks stacked in one column: a name row, three map rows, a gap row between blocks."""
    ratios = [0.30]
    for i in range(len(concepts)):
        ratios += [0.30, 1.0, 1.0, 1.0] + ([0.18] if i < len(concepts) - 1 else [])
    inner = GridSpecFromSubplotSpec(
        len(ratios),
        8,
        subplot_spec=subspec,
        height_ratios=ratios,
        width_ratios=[0.13, 0.04] + [1.0] * 6,
        hspace=0.0,
        wspace=0.05,
    )
    for cols, text, colour in ((slice(2, 5), "concept present", "#1b7a1b"), (slice(5, 8), "concept absent", "#a01010")):
        ax = fig.add_subplot(inner[0, cols])
        ax.axis("off")
        ax.text(0.5, 0.45, text, ha="center", va="center", fontsize=fs("label"), color=colour, transform=ax.transAxes)
    row = 1
    for c in concepts:
        ax = fig.add_subplot(inner[row, 1:8])
        ax.axis("off")
        ax.text(0.0, 0.5, c["name"], ha="left", va="center", fontsize=fs("title"), fontweight="bold")
        for r, text in enumerate(("imagery", f"SAE f{c['feature']}", raw_label(c))):
            ax = fig.add_subplot(inner[row + 1 + r, 1])
            ax.axis("off")
            ax.text(0.5, 0.5, text, ha="center", va="center", rotation=90, fontsize=0.75 * fs("annot"))
        draw_tiles(fig, inner, row + 1, 2, c, coord_size=0.55 * fs("annot"))
        row += 5


def render(concepts: list[dict], out: Path, name: str) -> Path:
    """The two-column gallery figure."""
    set_style(7.0)
    half = (len(concepts) + 1) // 2
    tile = 0.53  # in: the width of one 64 x 64 tile at this canvas width
    fig = plt.figure(figsize=(7.0, tile * (0.30 + half * 3.30 + (half - 1) * 0.18)))
    outer = GridSpec(1, 2, figure=fig, left=0.014, right=0.995, top=0.99, bottom=0.008, wspace=0.06)
    for col, block in enumerate((concepts[:half], concepts[half:])):
        if block:
            draw_column(fig, outer[0, col], block)
    return save(fig, out, name)


def main() -> None:
    ap = parser(__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, nargs="?", help="Matryoshka run directory with eval/test_coverage_arrays.npz")
    ap.add_argument("--n-candidates", type=int, default=1500, help="test patches the gallery patches come from")
    ap.add_argument("--n-half", type=int, default=3, help="present (and absent) patches per concept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None, help="output name (default by embedding)")
    add_imagery_argument(ap)
    add_paper_data_argument(ap)
    ap.add_argument(
        "--embedding", choices=tuple(CONCEPTS), default="alphaearth", help="gallery drawn with --paper-data"
    )
    args = ap.parse_args()
    require(ap, args, "run")
    if args.paper_data is not None:
        concepts, embedding = from_paper_data(args.paper_data, args.embedding), args.embedding
    else:
        concepts, embedding = collect(
            args.run,
            n_candidates=args.n_candidates,
            n_half=args.n_half,
            seed=args.seed,
            device=args.device,
            esri=args.esri,
        )
    name = args.name or NAME[embedding]
    render(concepts, args.out, name)
    write_json([{k: v for k, v in c.items() if not k.startswith("_")} for c in concepts], args.out / f"{name}.json")


if __name__ == "__main__":
    main()

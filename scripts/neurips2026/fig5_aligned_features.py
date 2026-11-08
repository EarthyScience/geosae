"""Fig. 5: aligned features on AlphaEarth, from patch galleries to sub-concepts.

(a) Imagery, the best-aligned SAE feature and the best-aligned raw dimension on three test patches
with the class and three without, for water and built-up. (b) Two top-activating patches of three
woody-vegetation features, (c) the 2,000 test patches each is most active on and (d-f) the strata
of Köppen-Geiger climate zones, RESOLVE biomes and potential natural vegetation that match them.

The three features are chosen by rule: among the run's woody features (folded AUROC >= 0.65 on a
forest, shrubland or savanna class, and more active on that class than elsewhere), the one whose
top patches lie most in each Köppen-Geiger pair Af+Am, BSh+Aw and Dfc+Dfb. The gallery features are
the best-aligned ones by folded AUROC that fire on their class.

Reads ``<run>/predictions/test.zarr`` (``geosae predict``), ``<run>/eval/test_coverage_arrays.npz``
and the evaluation pool it names (``geosae coverage -c configs/neurips2026/coverage.yaml``), and
the data store (Sentinel-2 imagery, or Esri World Imagery with ``--esri``; labels, ``chelsa_kg``,
``resolve_biomes``, ``pnv``). Writes
``fig5_aligned_features.{pdf,png,json}``; the JSON holds Table S8 (see ``table_subconcepts.py``).

With ``--paper-data <dir>`` the figure is drawn from the camera-ready inputs instead
(``qualitative_ae.npz``, ``subconcepts.npz``, ``subconcepts.json`` and ``subconcepts_frozen.json``),
with Esri World Imagery tiles at the recorded patch centres (network; grey tiles without it).

    python scripts/neurips2026/fig5_aligned_features.py runs/alphaearth/matryoshka_M1024_k16_s0/<ts> --out figures
    python scripts/neurips2026/fig5_aligned_features.py --paper-data paper_data --out figures
"""

from __future__ import annotations

import json
from pathlib import Path

import _common as C
import matplotlib.pyplot as plt
import nicefigs as nf
import numpy as np
import zarr
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

from geosae.config import load_config
from geosae.eval.concepts import CLASS_TO_GROUP
from geosae.eval.landcover_classes import PRODUCT_CLASSES
from geosae.eval.pool import default_sample_cache_path, load_pool
from geosae.eval.registry import encode_sample
from geosae.eval.subconcepts import patch_class_fractions, patch_feature_activation
from geosae.pipeline import apply_transforms, load_run_model, load_run_transforms

TOP_N = 2000  # a feature's support: its most active test patches
ALIGN = 0.65
WOODY = ("forest", "shrubland", "savanna")
LAT_MAX = 65.0  # gallery patches: imagery is poor at high latitude
#: Matched strata of each reference product, one tuple of class codes per sub-concept.
REFERENCES = {
    "chelsa_kg": [(1, 2), (8, 4), (20, 19)],
    "resolve_biomes": [(1,), (7,), (6,)],
    "pnv": [(112,), (20,), (111,)],
}
REFERENCE_TITLES = ("Köppen-Geiger", "RESOLVE biomes", "Natural vegetation")
SUBS = (
    ("tropical wet forest", "#009E73"),
    ("dry savanna and shrubland", "#E6C229"),
    ("boreal and conifer forest", "#882255"),
)
OTHER = "#DADADA"
#: Gallery concepts of panel a: name, product, class code, cap on the class share of a present patch.
CONCEPTS = (("water", "lc_esa_wc", 80, 0.35), ("built-up", "lc_esa_wc", 50, 0.40))
PRESENT_MIN = 0.05


# --------------------------------------------------------------------------- alignment


class Alignment:
    """A run's folded AUROC matrices on the evaluation pool, with the pool to read signs from."""

    def __init__(
        self, run: Path, *, pool: str | None = None, cache_root: str = "data/eval_cache/samples", device: str = "cpu"
    ):
        with np.load(Path(run) / "eval" / "test_coverage_arrays.npz") as z:
            self.auc = z["sae__auc_matrix"]
            self.raw_auc = z["raw__auc_matrix"]
            self.class_table = z["class_table"]
            self.product_order = [str(p) for p in z["product_order"]]
        self.run = Path(run)
        self.pool = load_pool(pool or C.read_json(self.run / "eval" / "test_coverage.json")["sample_path"])
        self.cache = default_sample_cache_path(self.pool.spec, cache_root=cache_root)
        self._sample = None
        self._codes: np.ndarray | None = None
        self.device = device

    def column(self, product: str, code: int) -> int:
        pid = self.product_order.index(product)
        return int(np.flatnonzero((self.class_table[:, 0] == pid) & (self.class_table[:, 1] == code))[0])

    def classes(self) -> list[tuple[str, int]]:
        return [(self.product_order[int(p)], int(c)) for p, c in self.class_table]

    def sample(self):
        """The pool's pixels and labels (raw embedding, untransformed)."""
        if self._sample is None:
            self._sample = self.pool.registry_sample(cache=self.cache)
        return self._sample

    def feature_codes(self, features) -> np.ndarray:
        """``(n_pixels, len(features))`` sparse codes of the pool pixels under the run's model."""
        if self._codes is None:
            x = apply_transforms(load_run_transforms(self.run), self.sample().raw)
            self._codes = encode_sample(load_run_model(self.run), x, device=self.device)
        return self._codes[:, np.asarray(features, dtype=np.int64)]

    def fires_on(self, values: np.ndarray, column: int) -> np.ndarray:
        """Per column of ``values`` (pool pixels): is the mean higher on the class than off it?"""
        idx, y = self.pool.balanced_indices(column)
        v = values[idx]
        return v[y == 1].mean(0) > v[y == 0].mean(0)


def woody_features(al: Alignment) -> list[dict]:
    """Features aligned with a woody class that are more active on it than elsewhere."""
    cols = [j for j, key in enumerate(al.classes()) if CLASS_TO_GROUP.get(key) in WOODY]
    auc = al.auc[:, cols]
    cand = np.flatnonzero(auc.max(1) >= ALIGN)
    best = [cols[int(auc[f].argmax())] for f in cand]
    codes = al.feature_codes(cand)
    out = []
    for i, (f, j) in enumerate(zip(cand, best, strict=True)):
        if al.pool.n_positive[j] >= 50 and al.fires_on(codes[:, [i]], j)[0]:
            prod, code = al.classes()[j]
            out.append({"feat": int(f), "auc": float(al.auc[f, j]), "class": f"{prod}:{code}"})
    return out


# --------------------------------------------------------------------------- sub-concepts


def majority(store, product: str, cells: np.ndarray) -> np.ndarray:
    """Per-patch majority class of ``product``, ``-1`` where a patch has no valid pixel."""
    frac, codes, valid = patch_class_fractions(store, product, cells)
    return np.where(valid > 0, codes[np.argmax(frac, axis=1)], -1)


def pair_shares(act: np.ndarray, feats: list[int], kg: np.ndarray) -> np.ndarray:
    """``(len(feats), 3)`` share of each feature's top patches in each Köppen-Geiger pair."""
    ok = kg >= 0
    out = np.zeros((len(feats), len(REFERENCES["chelsa_kg"])))
    for i, f in enumerate(feats):
        top = np.argsort(-act[f])[:TOP_N]
        t = kg[top[ok[top]]]
        out[i] = [np.isin(t, pair).mean() for pair in REFERENCES["chelsa_kg"]]
    return out


def pick_features(feats: list[int], shares: np.ndarray) -> list[int]:
    """Per Köppen-Geiger pair, the distinct feature with the largest share of its top patches in it."""
    picked: list[int] = []
    for j in range(shares.shape[1]):
        order = np.argsort(-shares[:, j], kind="stable")
        picked.append(next(feats[i] for i in order if feats[i] not in picked))
    return picked


def _short(product: str, code: int) -> str:
    name = PRODUCT_CLASSES[product].get(int(code), str(code))
    return name.split(" — ")[0] if product == "chelsa_kg" else name


def agreement_table(act, lat, feats, majorities: dict[str, np.ndarray]) -> list[dict]:
    """Share and lift of the three most common strata on each feature's top patches, and mean |lat|.

    Lift is the share divided by the stratum's share among all test patches with data.
    """
    rows = []
    for f in feats:
        top = np.argsort(-act[f])[:TOP_N]
        row = {"feat": int(f), "mean_abs_lat": float(np.abs(lat[top]).mean())}
        for product, maj in majorities.items():
            ok = maj >= 0
            t = top[ok[top]]
            codes, counts = np.unique(maj[t], return_counts=True)
            row[product] = [
                {
                    "class": _short(product, codes[i]),
                    "share": float(counts[i] / t.size),
                    "lift": float((counts[i] / t.size) / np.mean(maj[ok] == codes[i])),
                }
                for i in np.argsort(-counts)[:3]
            ]
        rows.append(row)
    return rows


def subconcepts(run: Path, al: Alignment, *, seed: int = 1, gallery: bool = True, esri: bool = False) -> dict:
    """Everything panels b-f and Table S8 need for one run; ``gallery=False`` skips the patch maps."""
    pf = patch_feature_activation(str(C.predictions(run)))
    act, lon, lat, cells = pf["act"], pf["lon"], pf["lat"], pf["cell"]
    store = zarr.open_group(C.store_path(run), mode="r")
    majorities = {p: majority(store, p, cells) for p in REFERENCES}
    woody = woody_features(al)
    feats = [r["feat"] for r in woody]
    shares = pair_shares(act, feats, majorities["chelsa_kg"])
    for r, s in zip(woody, shares, strict=True):
        r["pair_share"] = s.tolist()
    picks = pick_features(feats, shares)
    matched = []
    for product, groups in REFERENCES.items():
        g = np.zeros(cells.size, dtype=np.int8)
        for gi, codes in enumerate(groups, start=1):
            g[np.isin(majorities[product], codes)] = gi
        g[majorities[product] < 0] = -1
        matched.append(g)
    tops = np.stack([np.argsort(-act[f])[:TOP_N] for f in picks])
    out = {
        "features": picks,
        "lon": lon,
        "lat": lat,
        "tops": tops,
        "matched": np.stack(matched),
        "table": agreement_table(act, lat, picks, majorities),
        "woody": woody,
    }
    if gallery:
        rng = np.random.default_rng(seed)
        rows = np.array([_pick_patches(t, lon, lat, rng) for t in tops])
        pred = zarr.open_group(str(C.predictions(run)), mode="r")
        out["gallery"] = rows
        out["gallery_maps"] = np.stack(
            [[feature_map(pred, p, f) for p in row] for f, row in zip(picks, rows, strict=True)]
        )
        out["gallery_rgb"] = np.stack([[rgb(store, int(cells[p]), esri=esri) for p in row] for row in rows])
    return out


def woody_summary(woody: list[dict]) -> dict:
    """Woody feature count, best share per Köppen-Geiger pair and features with at least 80 % in it."""
    shares = np.array([r["pair_share"] for r in woody]).reshape(-1, len(REFERENCES["chelsa_kg"]))
    return {
        "n_woody": len(woody),
        "best_pair_share": shares.max(0).tolist(),
        "n_pair_share_ge_0.8": (shares >= 0.8).sum(0).tolist(),
    }


def _pick_patches(top, lon, lat, rng, n: int = 2, min_sep: float = 15.0) -> list[int]:
    """``n`` geographically separated patches from the top quarter of a feature's support."""
    pool = top[: TOP_N // 4]
    pool = rng.permutation(pool[np.abs(lat[pool]) <= LAT_MAX])
    chosen: list[int] = []
    for p in pool:
        if all(np.hypot(lon[p] - lon[q], lat[p] - lat[q]) >= min_sep for q in chosen):
            chosen.append(int(p))
        if len(chosen) == n:
            break
    return chosen


# --------------------------------------------------------------------------- patch maps


def feature_map(pred, p: int, feat: int) -> np.ndarray:
    """Per-pixel activation of ``feat`` on prediction patch ``p`` (0 where inactive)."""
    idx = np.asarray(pred["z_indices"][p])
    val = np.asarray(pred["z_values"][p], dtype=np.float32)
    cnt = np.asarray(pred["z_counts"][p])
    valid = np.arange(idx.shape[-1])[None, None, :] < cnt[..., None]
    return np.where((idx == feat) & valid, val, 0.0).max(axis=-1)


def rgb(store, cell: int, *, esri: bool = False) -> np.ndarray:
    """The image of ``cell`` (:func:`_common.imagery`); grey where there is none."""
    img = C.imagery(store, cell, esri=esri)
    return np.full((64, 64, 3), 0.9) if img is None else img


def concept_gallery(run: Path, al: Alignment, *, n_half: int = 3, seed: int = 0, esri: bool = False) -> list[dict]:
    """Panel a: per concept, the feature, the raw dimension and ``n_half`` present / absent pool patches."""
    sample = al.sample()
    transforms = load_run_transforms(run)
    pool_cells = al.pool.cells
    n_pix = al.pool.pixel_rows.size
    raw_x = apply_transforms(transforms, sample.raw)
    raw_cell_mean = raw_x.reshape(pool_cells.size, n_pix, -1).mean(1)
    pred_path = C.predictions(run)
    pred = zarr.open_group(str(pred_path), mode="r")
    rows = al.pool.prediction_indices(pred_path)
    order = np.argsort(np.asarray(pred["cell"])[rows])
    rows = rows[order]  # prediction patch of each pool cell, in pool order
    sae_mean = np.asarray(pred["feat_patch_stats"][:, :, 0])[:, rows]
    store = zarr.open_group(C.store_path(run), mode="r")
    lon, lat = np.asarray(store["lon"][:])[pool_cells], np.asarray(store["lat"][:])[pool_cells]
    embedding = _embedding_name(run)
    used: set[int] = set()
    out = []
    for ci, (name, product, code, present_max) in enumerate(CONCEPTS):
        j = al.column(product, code)
        fires = al.fires_on(al.feature_codes(np.argsort(-al.auc[:, j])[:50]), j)
        ranked = np.argsort(-al.auc[:, j])[:50][fires]
        feat = int(next(f for f in ranked if int(f) not in used))
        used.add(feat)
        dim = int(np.argmax(al.raw_auc[:, j]))
        sign = 1.0 if al.fires_on(raw_x[:, [dim]], j)[0] else -1.0
        labels = _labels(store, product, pool_cells)
        frac = (labels == code).mean((1, 2))
        present, absent = _choose(
            frac,
            sae_mean[feat],
            sign * raw_cell_mean[:, dim],
            np.abs(lat) <= LAT_MAX,
            n=n_half,
            present_max=present_max,
            rng=np.random.default_rng(seed + ci),
        )
        cols = present + absent
        raw_maps = []
        for p in cols:
            x = np.asarray(store[embedding][int(pool_cells[p])], dtype=np.float32)
            raw_maps.append(sign * apply_transforms(transforms, x.reshape(-1, x.shape[-1]))[:, dim].reshape(64, 64))
        out.append(
            {
                "name": name,
                "feat": feat,
                "feat_auc": float(al.auc[feat, j]),
                "dim": dim,
                "dim_sign": sign,
                "dim_auc": float(al.raw_auc[dim, j]),
                "cell": [int(pool_cells[p]) for p in cols],
                "lon": [float(lon[p]) for p in cols],
                "lat": [float(lat[p]) for p in cols],
                "class_frac": [float(frac[p]) for p in cols],
                "sae": np.stack([feature_map(pred, int(rows[p]), feat) for p in cols]),
                "raw": np.stack(raw_maps),
                "rgb": np.stack([rgb(store, int(pool_cells[p]), esri=esri) for p in cols]),
            }
        )
    return out


def _embedding_name(run: Path) -> str:
    return str(load_config([str(Path(run) / "config.yaml")], [])["data"]["embedding_name"])


def _labels(store, product: str, cells: np.ndarray, chunk: int = 5120) -> np.ndarray:
    """``(n, 64, 64)`` labels of ``product`` on ``cells`` (sorted), read chunk by chunk."""
    out = []
    for c0 in np.unique(cells // chunk):
        sel = cells[(cells // chunk) == c0]
        block = np.asarray(store[product][int(c0 * chunk) : int(sel.max()) + 1])
        out.append(block[sel - c0 * chunk])
    return np.concatenate(out)


def _choose(frac, sae_mean, raw_mean, eligible, *, n, present_max, rng):
    """``n`` patches with part of the class and an active feature, ``n`` without it and a silent feature."""
    present = np.flatnonzero((frac > PRESENT_MIN) & (frac <= present_max) & eligible)
    if len(present) > 4 * n:
        present = present[np.argsort(-sae_mean[present])][: max(2 * n, len(present) // 2)]
    absent = np.flatnonzero((frac <= 0.0) & eligible)
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


# --------------------------------------------------------------------------- camera-ready inputs


def esri_tile(lon: float, lat: float) -> np.ndarray | None:
    """Esri World Imagery of the patch centred on ``(lon, lat)``, or ``None`` when it cannot be fetched."""
    try:
        return C.esri_imagery(float(lon), float(lat))
    except OSError as err:
        print(f"Esri imagery unavailable at {lon:.2f}, {lat:.2f} ({err}); drawing a grey tile")
        return None


def _grey(img: np.ndarray | None) -> np.ndarray:
    return np.full((64, 64, 3), 0.9) if img is None else img


def paper_gallery(paper_data: Path, backbone: str, concepts) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """Per index in ``concepts``: the record, SAE maps and oriented raw maps of ``qualitative_<backbone>.npz``."""
    with np.load(C.paper_file(paper_data, f"qualitative_{backbone}.npz")) as z:
        meta = json.loads(str(z["meta"]))
        return [(meta["concepts"][i], z["sae"][i].astype(np.float32), z["raw"][i].astype(np.float32)) for i in concepts]


def paper_subconcepts(paper_data: Path, arm: str = "trained") -> dict:
    """Panels b-f (``arm="trained"``) or the frozen-decoder twin (``"frozen"``) from the camera-ready inputs.

    Returns the structure of :func:`subconcepts`; ``table`` is empty for the frozen arm, whose
    agreement table the camera-ready inputs do not hold.
    """
    stem = "subconcepts" if arm == "trained" else "subconcepts_frozen"
    with np.load(C.paper_file(paper_data, f"{stem}.npz")) as z:
        d = {k: z[k] for k in ("features", "lon", "lat", "tops", "matched", "gallery", "gallery_maps")}
    d["features"] = [int(f) for f in d["features"]]
    d["gallery_maps"] = d["gallery_maps"].astype(np.float32)
    d["gallery_rgb"] = np.stack([[_grey(esri_tile(d["lon"][p], d["lat"][p])) for p in row] for row in d["gallery"]])
    d["woody"] = C.read_json(C.paper_file(paper_data, "subconcepts_frozen.json"))["features"][arm]
    d["table"] = C.read_json(C.paper_file(paper_data, "subconcepts.json"))["table"] if arm == "trained" else []
    return d


def from_paper_data(paper_data: Path) -> tuple[list[dict], dict]:
    """The inputs of :func:`render` and :func:`report` from the camera-ready figure data."""
    gallery = []
    for (name, *_), (c, sae, raw) in zip(CONCEPTS, paper_gallery(paper_data, "ae", range(len(CONCEPTS))), strict=True):
        keys = ("feat", "feat_auc", "dim", "dim_sign", "dim_auc", "cell", "lon", "lat", "class_frac")
        rgb = [_grey(esri_tile(lo, la)) for lo, la in zip(c["lon"], c["lat"], strict=True)]
        gallery.append({"name": name, **{k: c[k] for k in keys}, "sae": sae, "raw": raw, "rgb": np.stack(rgb)})
    return gallery, paper_subconcepts(paper_data)


# --------------------------------------------------------------------------- drawing


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _tile(fig, spec, image, cmap=None, vmin=None, vmax=None):
    ax = fig.add_subplot(spec)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    _clean(ax)
    return ax


def draw_concept(fig, spec, c: dict) -> None:
    """One concept of panel a: imagery, SAE feature and raw dimension on 3 present and 3 absent patches."""
    g = GridSpecFromSubplotSpec(
        4, 7, subplot_spec=spec, height_ratios=[0.22, 1, 1, 1], width_ratios=[0.16] + [1] * 6, hspace=0.05, wspace=0.05
    )
    for cols, text, colour in ((slice(1, 4), "present", "#1b7a1b"), (slice(4, 7), "absent", "#a01010")):
        ax = fig.add_subplot(g[0, cols])
        ax.axis("off")
        ax.text(0.5, 0.3, f"{c['name']} {text}", ha="center", va="center", fontsize=C.fs("label"), color=colour)
    sign = "−" if c["dim_sign"] < 0 else ""
    for r, label in enumerate(("imagery", f"SAE f{c['feat']}", f"raw {sign}d{c['dim']}")):
        ax = fig.add_subplot(g[1 + r, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, label, ha="center", va="center", rotation=90, fontsize=C.fs("annot"))
    sae_max = float(c["sae"].max()) or 1.0
    lo, hi = (float(v) for v in np.percentile(c["raw"], [5, 95]))
    for k in range(c["sae"].shape[0]):
        axes = (
            _tile(fig, g[1, 1 + k], c["rgb"][k]),
            _tile(fig, g[2, 1 + k], c["sae"][k], "inferno", 0, sae_max),
            _tile(fig, g[3, 1 + k], c["raw"][k], "inferno", lo, hi if hi > lo else lo + 1),
        )
        if k == 2:  # separator after the present half
            for ax in axes:
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_color("#444444")
                ax.spines["right"].set_linewidth(0.6)


def draw_gallery(fig, spec, d: dict) -> None:
    """Two top patches per sub-concept feature: imagery above activation."""
    n = len(SUBS)
    g = GridSpecFromSubplotSpec(
        3,
        2 * n + 1,
        subplot_spec=spec,
        height_ratios=[0.34, 1, 1],
        width_ratios=[0.16] + [1] * (2 * n),
        hspace=0.05,
        wspace=0.05,
    )
    for r, label in enumerate(("image", "feature")):
        ax = fig.add_subplot(g[1 + r, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, label, rotation=90, ha="center", va="center", fontsize=C.fs("annot"))
    for i, (_, colour) in enumerate(SUBS):
        c0 = 1 + 2 * i
        head = fig.add_subplot(g[0, c0 : c0 + 2])
        head.axis("off")
        head.axhline(0.02, color=colour, lw=2.0)
        head.text(0.5, 0.20, f"f{int(d['features'][i])}", ha="center", va="bottom", fontsize=C.fs("label"))
        vmax = float(d["gallery_maps"][i].max()) or 1.0
        for k in range(d["gallery"].shape[1]):
            _tile(fig, g[1, c0 + k], d["gallery_rgb"][i, k])
            _tile(fig, g[2, c0 + k], d["gallery_maps"][i, k], "inferno", 0, vmax)


def _map_axes(fig, spec, letter: str, title: str):
    ax, ccrs = nf.make_map_axes(fig, spec, projection="robinson")
    nf.draw_background(ax, ccrs, land=True, coastline=True)
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    C.panel_title(ax, letter, title)
    return ax, ccrs


def draw_maps(fig, spec, d: dict, *, letters: str, support_title: str = "SAE features") -> None:
    """The features' top patches, then each reference's matched strata on a 2° majority grid."""
    lon, lat = d["lon"], d["lat"]
    g = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, hspace=0.06, wspace=0.10)
    ax, ccrs = _map_axes(fig, g[0, 0], letters[0], support_title)
    for i, (_, colour) in enumerate(SUBS):
        t = d["tops"][i]
        ax.scatter(lon[t], lat[t], s=1.2, c=colour, linewidths=0, transform=ccrs.PlateCarree(), rasterized=True)
    for j, title in enumerate(REFERENCE_TITLES):
        ax, ccrs = _map_axes(fig, g[(j + 1) // 2, (j + 1) % 2], letters[1 + j], title)
        keep = d["matched"][j] >= 0
        nf.categorical_map(
            lon[keep],
            lat[keep],
            d["matched"][j][keep].astype(np.int64),
            ax=ax,
            ccrs=ccrs,
            kind="binned",
            bin_deg=2.0,
            legend=False,
            categories={0: "other", 1: SUBS[0][0], 2: SUBS[1][0], 3: SUBS[2][0]},
            colors=[OTHER] + [c for _, c in SUBS],
        )


def legend_handles(d: dict) -> list:
    handles = [Patch(facecolor=c, label=f"f{int(f)}: {n}") for (n, c), f in zip(SUBS, d["features"], strict=True)]
    return [*handles, Patch(facecolor=OTHER, label="other stratum")]


def draw_legend(fig, spec, d: dict) -> None:
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.legend(
        handles=legend_handles(d),
        loc="center left",
        bbox_to_anchor=(0.06, 0.5),
        fontsize=C.fs("legend"),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
        labelspacing=0.5,
        title="feature and its matched strata",
        title_fontsize=C.fs("legend"),
        alignment="left",
    )


def render(gallery: list[dict], d: dict):
    """Figure 5: panel a on top, panels b-f below."""
    width, h_top, h_bottom = 5.5, 1.5, 2.1
    C.set_style(width)
    fig = plt.figure(figsize=(width, h_top + h_bottom + 0.1))
    rows = GridSpec(
        2, 1, figure=fig, height_ratios=[h_top, h_bottom], hspace=0.06, left=0.005, right=0.997, top=0.995, bottom=0.005
    )
    top = GridSpecFromSubplotSpec(1, len(gallery), subplot_spec=rows[0], wspace=0.035)
    for j, c in enumerate(gallery):
        draw_concept(fig, top[0, j], c)
    fig.text(0.0, 0.995, "a", fontsize=C.fs("panel"), fontweight="bold", ha="left", va="top")
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=rows[1], width_ratios=[2.45, 3.05], wspace=0.04)
    left = GridSpecFromSubplotSpec(2, 1, subplot_spec=bottom[0], hspace=0.05)
    draw_gallery(fig, left[0], d)
    draw_legend(fig, left[1], d)
    draw_maps(fig, bottom[1], d, letters="cdef")
    pos = rows[1].get_position(fig)
    fig.text(pos.x0, pos.y1 - 0.004, "b", fontsize=C.fs("panel"), fontweight="bold", ha="left", va="top")
    return fig


def report(gallery: list[dict], d: dict) -> dict:
    """Print and return the numbers the text and Table S8 quote."""
    out = {
        "gallery": [
            {k: c[k] for k in ("name", "feat", "feat_auc", "dim", "dim_sign", "dim_auc", "cell", "class_frac")}
            for c in gallery
        ],
        "features": [int(f) for f in d["features"]],
        "top_n": TOP_N,
        "table": d["table"],
        "woody": woody_summary(d["woody"]),
        "picked": [next(r for r in d["woody"] if r["feat"] == f) for f in d["features"]],
    }
    for c in out["gallery"]:
        print(f"{c['name']:10s} SAE f{c['feat']} AUROC {c['feat_auc']:.3f} | raw d{c['dim']} AUROC {c['dim_auc']:.3f}")
    for row, r in zip(d["table"], out["picked"], strict=True):
        print(f"f{row['feat']}: mean |lat| {row['mean_abs_lat']:.1f}, KG pair shares {np.round(r['pair_share'], 3)}")
        for product in REFERENCES:
            print("  " + ", ".join(f"{c['class']} {c['share']:.2f} ({c['lift']:.1f})" for c in row[product]))
    print("woody:", out["woody"])
    return out


def main() -> None:
    ap = C.parser(__doc__.splitlines()[0])
    ap.add_argument(
        "run", type=Path, nargs="?", help="Matryoshka run directory (M = 1024, k = 16) with predictions and coverage"
    )
    ap.add_argument("--pool", default=None, help="evaluation pool (default: the one the coverage stage used)")
    ap.add_argument("--cache-root", default="data/eval_cache/samples", help="pool sample cache of the coverage stage")
    ap.add_argument("--device", default="cpu", help="torch device that encodes the pool pixels")
    C.add_imagery_argument(ap)
    C.add_paper_data_argument(ap)
    args = ap.parse_args()
    C.require(ap, args, "run")
    if args.paper_data is not None:
        gallery, d = from_paper_data(args.paper_data)
    else:
        al = Alignment(args.run, pool=args.pool, cache_root=args.cache_root, device=args.device)
        gallery = concept_gallery(args.run, al, esri=args.esri)
        d = subconcepts(args.run, al, esri=args.esri)
    C.write_json(report(gallery, d), args.out / "fig5_aligned_features.json")
    C.save(render(gallery, d), args.out, "fig5_aligned_features")


if __name__ == "__main__":
    main()

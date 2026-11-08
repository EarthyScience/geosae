"""Fig. S1 (figS_dataset_overview): one test patch in every reference product of the registry.

Top row: the patch's Sentinel-2 true-colour image, the first three principal components of its
AlphaEarth and Tessera embeddings as RGB, and its location among all patches of the split. Below:
13 of the 14 categorical products (all but MODIS IGBP) and the three continuous ones (canopy height,
elevation, GPP), each with its legend or colour bar.

The patch is the test patch, among ``--n-candidates`` drawn with ``--seed``, of largest land-cover
diversity: the Shannon entropy of its ESA WorldCover classes, plus half the number of its ESRI
classes, plus 0.05 times its canopy-height standard deviation (capped at 20 m), plus 0.5 per Overture
layer with content, plus 1 per other categorical product with a classified pixel. Patches with
less than half of their pixels valid in any continuous product, or beyond 70° latitude, are
excluded.

Reads only the data store.

    python scripts/neurips2026/figS_dataset_overview.py data/majortom210k.zarr --out figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nicefigs as nf
import numpy as np
import zarr
from _common import fs, parser, save, set_style, true_colour, write_json
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

from geosae.eval.landcover_classes import PRODUCT_CLASSES

#: Categorical panels: (title, product, store array, band of the array).
CATEGORICAL = (
    ("Overture water (1 m)", "overture_water_natural", "overture", 4),
    ("Overture roads (1 m)", "overture_road_class", "overture", 2),
    ("Overture land use (1 m)", "overture_land_use", "overture", 1),
    ("Overture infrastructure (1 m)", "overture_infrastructure", "overture", 3),
    ("ESA WorldCover (10 m)", "lc_esa_wc", "lc_esa_wc", None),
    ("ESRI land cover (10 m)", "lc_esri", "lc_esri", None),
    ("WorldCereal (10 m)", "worldcereal", "worldcereal", None),
    ("GLC_FCS30D (30 m)", "glc_fcs30d", "glc_fcs30d", None),
    ("Geomorphons (30 m)", "geomorphons", "geomorphons", None),
    ("GLanCE (30 m)", "glance30", "glance30", None),
    ("Local Climate Zones (100 m)", "lcz", "lcz", None),
    ("SoilGrids WRB (250 m)", "soilgrids_class", "soilgrids_class", None),
    ("ESA CCI land cover (300 m)", "lc_esa_cci", "lc_esa_cci", None),
)
#: Continuous panels: (title, store array, colour map, units, whether zero means no data).
CONTINUOUS = (
    ("CHMv2 canopy height (1 m)", "chmv2", "YlGn", "m", True),
    ("Copernicus DEM (30 m)", "dem_copernicus", "terrain", "m", False),
    ("GPP (30 m)", "gpp", "viridis", "g C m$^{-2}$ d$^{-1}$", False),
)
#: nicefigs palettes of the products that have one.
PALETTES = {"lc_esa_wc": "esa_worldcover", "lc_esri": "esri_lulc"}
NODATA_COLOR = "#d9d9d9"


def class_colors(product: str) -> dict[int, str]:
    """A colour per class code: the product's nicefigs palette where it has one, else ``tab20``."""
    codes = sorted(PRODUCT_CLASSES[product])
    cycle = [plt.cm.tab20(i % 20) for i in range(len(codes))]
    out = {c: cycle[i] for i, c in enumerate(codes)}
    if product in PALETTES:
        out.update(nf.categorical(PALETTES[product]).colors)
    return out


def pick_patch(group: zarr.Group, *, n_candidates: int, seed: int) -> int:
    """The store row of the most diverse test patch among a seeded draw (see the module docstring)."""
    test = np.flatnonzero(np.asarray(group["split_210k"][:]) == 2)
    pool = np.sort(np.random.default_rng(seed).choice(test, size=min(n_candidates, test.size), replace=False))
    sel = (pool,)
    wc, esri = (np.asarray(group[k].get_orthogonal_selection(sel)) for k in ("lc_esa_wc", "lc_esri"))
    chm, dem, gpp = (
        np.asarray(group[k].get_orthogonal_selection(sel), np.float32) for k in ("chmv2", "dem_copernicus", "gpp")
    )
    overture = np.asarray(group["overture"].get_orthogonal_selection(sel))
    others = [p for _, p, key, _ in CATEGORICAL if key not in ("overture", "lc_esa_wc", "lc_esri")]
    classified = sum(
        np.isin(np.asarray(group[p].get_orthogonal_selection(sel)), list(PRODUCT_CLASSES[p])).any((1, 2))
        for p in others
    )
    lat = np.asarray(group["lat"][:])[pool]
    score = np.full(pool.size, -np.inf)
    for i in range(pool.size):
        a = wc[i][wc[i] != 0]
        cover = min(
            np.mean(np.isfinite(chm[i]) & (chm[i] > 0)),
            np.mean(np.isfinite(dem[i])),
            np.mean(np.isfinite(gpp[i]) & (gpp[i] > 0)),
        )
        if a.size == 0 or cover < 0.5 or abs(lat[i]) >= 70.0:
            continue
        p = np.unique(a, return_counts=True)[1] / a.size
        chm_sd = float(np.nanstd(chm[i])) if np.isfinite(chm[i]).any() else 0.0
        layers = sum(float(overture[i, ..., b].max()) > 0 for b in (1, 2, 3, 4))
        score[i] = (
            -(p * np.log(p)).sum()
            + 0.5 * np.unique(esri[i]).size
            + 0.05 * min(chm_sd, 20.0)
            + 0.5 * layers
            + classified[i]
        )
    return int(pool[np.argmax(score)])


def pca_rgb(x: np.ndarray) -> np.ndarray:
    """The first three principal components of ``(H, W, D)`` pixels, each stretched to its 2nd-98th percentile."""
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    flat = flat - flat.mean(0)
    proj = flat @ np.linalg.svd(flat, full_matrices=False)[2][:3].T
    lo, hi = np.percentile(proj, [2, 98], axis=0)
    return np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1).reshape(*x.shape[:2], 3)


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def draw_categorical(ax, ax_legend, data: np.ndarray, product: str, title: str) -> None:
    """A categorical map, with a legend of the classes present."""
    names, colors = PRODUCT_CLASSES[product], class_colors(product)
    codes = sorted(names)
    cmap = ListedColormap([colors[c] for c in codes])
    cmap.set_under(NODATA_COLOR)
    cmap.set_over(NODATA_COLOR)
    data = np.where(np.isin(data, codes), data, codes[-1] + 1)
    ax.imshow(
        data, cmap=cmap, norm=BoundaryNorm([codes[0] - 0.5] + [c + 0.5 for c in codes], cmap.N), interpolation="nearest"
    )
    ax.set_title(title, fontsize=fs("annot"), pad=1.5)
    _clean(ax)
    present = set(np.unique(data).tolist())
    handles = [
        Patch(facecolor=colors[c], edgecolor="none", label=names[c] if len(names[c]) <= 14 else names[c][:13] + "…")
        for c in codes
        if c in present
    ]
    ax_legend.axis("off")
    if handles:
        ax_legend.legend(
            handles=handles,
            loc="center left",
            frameon=False,
            fontsize=0.65 * fs("annot"),
            handlelength=0.9,
            handleheight=0.8,
            handletextpad=0.3,
            borderpad=0.0,
            labelspacing=0.18,
        )


def draw_continuous(ax, ax_legend, data, *, title: str, cmap: str, units: str, zero_nodata: bool) -> None:
    """A continuous map with its colour bar, stretched to the 2nd-98th percentile."""
    arr = np.where(np.isfinite(data) & ((data > 0) if zero_nodata else True), data, np.nan)
    finite = arr[np.isfinite(arr)]
    vmin, vmax = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-8), interpolation="nearest")
    ax.set_title(title, fontsize=fs("annot"), pad=1.5)
    _clean(ax)
    ax_legend.axis("off")
    cb = plt.colorbar(im, cax=ax_legend.inset_axes([0.05, 0.1, 0.18, 0.75]))
    cb.ax.tick_params(labelsize=0.7 * fs("annot"), length=1.5, pad=1.0, width=0.3)
    cb.outline.set_linewidth(0.3)
    cb.set_label(units, fontsize=0.7 * fs("annot"), labelpad=1.0)


def render(group: zarr.Group, row: int, out: Path, name: str) -> Path:
    """The 5 x 4 grid of panels for store row ``row``."""
    width = 6.5
    set_style(width)
    fig = plt.figure(figsize=(width, 7.2))
    outer = GridSpec(5, 4, figure=fig, left=0.012, right=0.992, bottom=0.008, top=0.965, wspace=0.06, hspace=0.04)

    def cell(r: int, c: int, **kw):
        sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[r, c], width_ratios=[1.0, 0.6], wspace=0.04)
        return fig.add_subplot(sub[0, 0], **kw), fig.add_subplot(sub[0, 1])

    lon, lat = float(group["lon"][row]), float(group["lat"][row])
    tiles = [("Sentinel-2 RGB (10 m)", true_colour(group, row))]
    tiles += [
        (f"{label} PCA → RGB", pca_rgb(np.asarray(group[key][row])))
        for key, label in (("alphaearth", "AlphaEarth"), ("tessera", "Tessera"))
        if key in group
    ]
    for c, (title, img) in enumerate(tiles):
        ax, ax_legend = cell(0, c)
        if img is None:
            ax.set_facecolor("#e6e6e6")
        else:
            ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=fs("annot"), pad=1.5)
        _clean(ax)
        ax_legend.axis("off")
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 3], width_ratios=[1.0, 0.6], wspace=0.04)
    ax_map, ccrs = nf.make_map_axes(fig, sub[0, 0], projection="robinson")
    fig.add_subplot(sub[0, 1]).axis("off")
    split = np.asarray(group["split_210k"][:])
    used = split >= 0
    lons, lats = np.asarray(group["lon"][:])[used], np.asarray(group["lat"][:])[used]
    nf.draw_background(ax_map, ccrs, land=True, coastline=True, land_color="#f2f2ee")
    kw = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
    ax_map.scatter(lons, lats, s=0.05, c="#9a9a9a", linewidths=0, rasterized=True, **kw)
    ax_map.plot(
        lon,
        lat,
        marker="*",
        markersize=8.5,
        markerfacecolor="#e63946",
        markeredgecolor="black",
        markeredgewidth=0.4,
        **kw,
    )
    if ccrs is not None:
        ax_map.set_global()
    ax_map.set_title("Location", fontsize=fs("annot"), pad=1.5)

    panels = [("cat", *p) for p in CATEGORICAL] + [("cont", *p) for p in CONTINUOUS]
    for i, panel in enumerate(panels):
        ax, ax_legend = cell(1 + i // 4, i % 4)
        if panel[0] == "cat":
            _, title, product, key, band = panel
            data = np.asarray(group[key][row])
            draw_categorical(ax, ax_legend, data[..., band] if band is not None else data, product, title)
        else:
            _, title, key, cmap, units, zero_nodata = panel
            draw_continuous(
                ax,
                ax_legend,
                np.asarray(group[key][row], np.float32),
                title=title,
                cmap=cmap,
                units=units,
                zero_nodata=zero_nodata,
            )
    return save(fig, out, name)


def main() -> None:
    ap = parser(__doc__.split("\n")[0])
    ap.add_argument("store", type=Path, help="the patch store (zarr)")
    ap.add_argument("--n-candidates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default="figS_dataset_overview")
    args = ap.parse_args()
    group = zarr.open_group(str(args.store), mode="r")
    row = pick_patch(group, n_candidates=args.n_candidates, seed=args.seed)
    lon, lat = float(group["lon"][row]), float(group["lat"][row])
    print(f"patch: store row {row} at lat {lat:.2f}, lon {lon:.2f}")
    render(group, row, args.out, args.name)
    write_json({"row": row, "lon": lon, "lat": lat}, args.out / f"{args.name}.json")


if __name__ == "__main__":
    main()

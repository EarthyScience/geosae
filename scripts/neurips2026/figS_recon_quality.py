"""Fig. S2 (figS_recon_quality): SAE reconstructions of raw AlphaEarth dimensions on three patches.

The three test patches nearest fixed places (near Frankfurt, in the US Midwest and in the Amazon)
whose ESA WorldCover map has no missing pixel. Per patch, the three raw dimensions with the widest
within-patch range (5th to 95th percentile; a dimension already shown for an earlier patch counts
half) are drawn as the original (GT) and as reconstructed by the Matryoshka, BatchTopK and ReLU+ℓ1
SAEs, each run forward on the patch. Every map shares one colour scale (3rd to 97th percentile of
all maps), so a colour mismatch with GT is a reconstruction error. The image is the patch's
Sentinel-2 true colour from the store, or with ``--esri`` Esri World Imagery.

Reads the sweep's run directories (``geosae run``; M = 1024, k = 16, seed 0) and their data store.

    python scripts/neurips2026/figS_recon_quality.py runs/alphaearth --out figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from _common import (
    SAE_VARIANT,
    add_imagery_argument,
    find_runs,
    fs,
    get_run,
    imagery,
    parser,
    save,
    set_style,
    store_path,
    write_json,
)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from geosae.config import load_config
from geosae.pipeline import apply_transforms, load_run_model, load_run_transforms, make_data

#: (lon, lat) of the places the patches are taken nearest to.
ANCHORS = ((8.68, 50.11), (-93.65, 41.59), (-62.95, -3.10))
ORDER = ("matryoshka", "batchtopk", "relu_l1")
CMAP = "inferno"
NODATA_WORLDCOVER = (0, 255)


def pick_patches(group: zarr.Group, cells: np.ndarray, anchors, *, search: int = 200) -> list[int]:
    """Store row of the test patch nearest each anchor whose ESA WorldCover map is complete."""
    lon, lat = np.asarray(group["lon"][:])[cells], np.asarray(group["lat"][:])[cells]
    picked: list[int] = []
    for a_lon, a_lat in anchors:
        for i in np.argsort((lon - a_lon) ** 2 + (lat - a_lat) ** 2)[:search]:
            cell = int(cells[i])
            if cell not in picked and not np.isin(np.asarray(group["lc_esa_wc"][cell]), NODATA_WORLDCOVER).any():
                picked.append(cell)
                break
        else:
            raise SystemExit(f"no complete test patch within {search} of {(a_lon, a_lat)}")
    return picked


def pick_dims(x: np.ndarray, n: int, taken: set[int]) -> list[int]:
    """The ``n`` dimensions of widest within-patch range of ``x`` ``(H, W, D)``; ``taken`` ones count half."""
    flat = x.reshape(-1, x.shape[-1])
    score = np.percentile(flat, 95, axis=0) - np.percentile(flat, 5, axis=0)
    score[list(taken)] *= 0.5
    return sorted(np.argsort(-score)[:n].tolist())


def reconstruct(run_dir: Path, raw: np.ndarray) -> np.ndarray:
    """The run's reconstruction of ``raw`` ``(P, H, W, D)``, in the run's input units."""
    model = load_run_model(run_dir).eval()
    x = apply_transforms(load_run_transforms(run_dir), raw.reshape(-1, raw.shape[-1]))
    with torch.no_grad():
        return model(torch.from_numpy(x))[0].numpy().reshape(raw.shape)


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render(gt, recon: dict, dims, rgb, lonlat, out: Path, name: str) -> Path:
    """One block per patch: imagery, then per dimension the GT map and each SAE's reconstruction."""
    pool = np.concatenate(
        [
            np.r_[gt[p, ..., d].ravel(), *(recon[v][p, ..., d].ravel() for v in ORDER)]
            for p in range(len(gt))
            for d in dims[p]
        ]
    )
    vmin, vmax = np.percentile(pool[np.isfinite(pool)], [3, 97])
    n_cols = 2 + len(ORDER)
    width = 5.6
    set_style(width, 0.6 * 5.5)
    fig = plt.figure(figsize=(width, 9 * width / n_cols + 0.45))
    outer = GridSpec(
        len(gt) + 1,
        1,
        figure=fig,
        height_ratios=[0.18] + [3] * len(gt),
        hspace=0.04,
        left=0.012,
        right=0.995,
        top=0.985,
        bottom=0.008,
    )
    head = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[0, 0], wspace=0.04)
    for c, title in enumerate(["", "GT", *(SAE_VARIANT[v][0] for v in ORDER)]):
        if title:
            ax = fig.add_subplot(head[0, c])
            ax.axis("off")
            ax.text(0.5, 0.2, title, ha="center", va="bottom", fontsize=fs("label"), transform=ax.transAxes)
    for p in range(len(gt)):
        block = GridSpecFromSubplotSpec(3, n_cols, subplot_spec=outer[p + 1, 0], hspace=0.04, wspace=0.04)
        ax = fig.add_subplot(block[0, 0])
        if rgb[p] is None:
            ax.set_facecolor("#e6e6e6")
        else:
            ax.imshow(rgb[p], interpolation="nearest")
        lon, lat = lonlat[p]
        ax.text(
            0.04,
            0.96,
            f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}",
            transform=ax.transAxes,
            fontsize=0.8 * fs("annot"),
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "alpha": 0.78, "lw": 0},
        )
        _clean(ax)
        for r in (1, 2):
            fig.add_subplot(block[r, 0]).axis("off")
        for r, d in enumerate(dims[p]):
            for c, img in enumerate([gt[p, ..., d], *(recon[v][p, ..., d] for v in ORDER)]):
                ax = fig.add_subplot(block[r, 1 + c])
                ax.imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
                _clean(ax)
                if c == 0:
                    ax.text(
                        0.04,
                        0.96,
                        f"d{d}",
                        transform=ax.transAxes,
                        fontsize=0.8 * fs("annot"),
                        va="top",
                        ha="left",
                        color="white",
                        bbox={"boxstyle": "round,pad=0.08", "fc": "black", "alpha": 0.55, "lw": 0},
                    )
    return save(fig, out, name)


def main() -> None:
    ap = parser(__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="root of the AlphaEarth sweep's run directories")
    ap.add_argument("--name", default="figS_recon_quality")
    add_imagery_argument(ap)
    args = ap.parse_args()
    runs = find_runs(args.runs)
    run_dirs = {v: get_run(runs, v) for v in ORDER}
    cfg = load_config([str(run_dirs["matryoshka"] / "config.yaml")], [])
    group = zarr.open_group(store_path(run_dirs["matryoshka"]), mode="r")
    cells = pick_patches(group, make_data(cfg, "test").cells, ANCHORS)
    key = cfg["data"].get("embedding_name", "alphaearth")
    raw = np.stack([np.asarray(group[key][c], dtype=np.float32) for c in cells])
    gt = apply_transforms(load_run_transforms(run_dirs["matryoshka"]), raw.reshape(-1, raw.shape[-1])).reshape(
        raw.shape
    )
    recon = {v: reconstruct(d, raw) for v, d in run_dirs.items()}
    taken: set[int] = set()
    dims = []
    for p in range(len(cells)):
        dims.append(pick_dims(gt[p], 3, taken))
        taken.update(dims[-1])
    rgb = [imagery(group, int(c), esri=args.esri) for c in cells]
    lonlat = [(float(group["lon"][c]), float(group["lat"][c])) for c in cells]
    render(gt, recon, dims, rgb, lonlat, args.out, args.name)
    fve = {
        v: float(1 - ((recon[v] - gt) ** 2).sum() / ((gt - gt.reshape(-1, gt.shape[-1]).mean(0)) ** 2).sum())
        for v in ORDER
    }
    summary = {"cells": cells, "lonlat": lonlat, "dims": dims, "fve_on_these_patches": fve}
    print(summary)
    write_json(summary, args.out / f"{args.name}.json")


if __name__ == "__main__":
    main()

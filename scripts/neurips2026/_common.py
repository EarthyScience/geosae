"""Shared helpers of the scripts that reproduce the figures and tables of the NeurIPS 2026 paper.

Run discovery (a sweep of ``geosae run`` directories, keyed by variant, dictionary size, sparsity
and seed), reading stage outputs, and the figure style: one type scale at printed size, one colour
per method, one marker per SAE variant. The plotting helpers need the ``viz`` extra
(``matplotlib``, ``nicefigs``); everything else needs ``geosae`` alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from geosae.config import load_config

#: The SAE variants of the paper, in the order every figure lists them.
VARIANTS = ("relu_l1", "batchtopk", "matryoshka")
#: Display name, marker and fill of each SAE variant.
SAE_VARIANT = {
    "batchtopk": ("BatchTopK", "o", True),
    "matryoshka": ("Matryoshka", "P", True),
    "relu_l1": (r"ReLU+$\ell_1$", "s", False),
}
#: The dense baselines, in the order every figure lists them.
DENSE = ("raw", "pca", "ica", "nmf")
DENSE_LABEL = {"raw": "raw", "pca": "PCA", "ica": "ICA", "nmf": "NMF"}
#: One colour per method (Paul Tol "bright", colour-blind safe).
METHOD_COLORS = {
    "batchtopk": "#4477AA",
    "matryoshka": "#66CCEE",
    "relu_l1": "#CCBB44",
    "frozen": "#777777",
    "raw": "#EE6677",
    "pca": "#228833",
    "ica": "#AA3377",
    "nmf": "#EE7733",
}
#: Embeddings of the paper: name -> (label, dimension).
EMBEDDINGS = {"alphaearth": ("AlphaEarth", 64), "tessera": ("Tessera", 128)}

#: Font sizes at printed size (pt); :func:`set_style` scales them to the canvas.
PRINT_PT = {"panel": 9.0, "title": 8.0, "label": 7.5, "tick": 7.0, "legend": 7.0, "annot": 6.5}
#: Width of the NeurIPS text block (in).
TEXT_WIDTH = 5.5
_scale = 1.0


# --------------------------------------------------------------------------- runs


class RunKey(NamedTuple):
    """Coordinates of one trained SAE: variant (``frozen`` for the frozen-decoder baseline), M, k, seed."""

    variant: str
    latent_dim: int
    k: int
    seed: int


def run_key(run_dir: str | Path) -> RunKey:
    """The :class:`RunKey` of a run directory, read from its ``config.yaml``."""
    cfg = load_config([str(Path(run_dir) / "config.yaml")], [])
    m = cfg["model"]
    variant = "frozen" if m.get("freeze_decoder") else str(m.get("type"))
    return RunKey(variant, int(m["latent_dim"]), int(m["k"]), int(cfg.get("seed", 0)))


def find_runs(root: str | Path) -> dict[RunKey, Path]:
    """Every run directory under ``root``, keyed by :class:`RunKey`.

    A run directory holds ``config.yaml`` and ``best.pth``; it may sit directly under ``root``
    (``<root>/<name>``) or one level deeper (``<root>/<name>/<timestamp>``, the layout of
    ``geosae run``), in which case the latest timestamp of a name wins.
    """
    root = Path(root)
    out: dict[RunKey, Path] = {}
    for cfg in sorted([*root.glob("*/config.yaml"), *root.glob("*/*/config.yaml")]):
        if (cfg.parent / "best.pth").exists():
            out[run_key(cfg.parent)] = cfg.parent
    return out


def get_run(runs: dict[RunKey, Path], variant: str, latent_dim: int = 1024, k: int = 16, seed: int = 0) -> Path:
    """The run directory of one sweep cell; raises ``KeyError`` naming the missing cell."""
    key = RunKey(variant, latent_dim, k, seed)
    if key not in runs:
        raise KeyError(f"no run {key} among {len(runs)} runs")
    return runs[key]


def predictions(run_dir: str | Path, split: str = "test") -> Path:
    """The ``geosae predict`` store of a run."""
    path = Path(run_dir) / "predictions" / f"{split}.zarr"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `geosae predict --run {run_dir}` first")
    return path


def store_path(run_dir: str | Path) -> str:
    """The data store a run was trained on."""
    return str(load_config([str(Path(run_dir) / "config.yaml")], [])["data"]["zarr_path"])


#: Positions of B04, B03, B02 among the 12 bands of the store's ``s2`` array (B01 ... B12, B8A after B08).
S2_TRUE_COLOUR = (3, 2, 1)


def true_colour(group, cell: int) -> np.ndarray | None:
    """Sentinel-2 true colour ``(H, W, 3)`` in ``[0, 1]`` of one store row, or ``None`` without an image.

    Reads the store's ``rgb`` array where it has one, else the bands B04, B03, B02 of the
    ``sentinel2_l2a`` product (array ``s2``, see ``geosae download``). Each patch is scaled to its
    99th percentile.
    """
    if "rgb" in group and ("mask_rgb" not in group or bool(group["mask_rgb"][cell])):
        x = np.asarray(group["rgb"][cell], dtype=np.float64)
    elif "s2" in group and ("mask_s2" not in group or bool(group["mask_s2"][cell])):
        x = np.asarray(group["s2"][cell], dtype=np.float64)[..., list(S2_TRUE_COLOUR)]
        x[x < 0] = np.nan
    else:
        return None
    hi = np.nanpercentile(x, 99) if np.isfinite(x).any() else np.nan
    return None if not np.isfinite(hi) or hi <= 0 else np.clip(np.nan_to_num(x / hi), 0, 1)


#: Esri World Imagery tiles (XYZ, web Mercator).
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def esri_imagery(lon: float, lat: float, *, extent_m: float = 640.0, zoom: int = 17, size: int = 256) -> np.ndarray:
    """Esri World Imagery ``(size, size, 3)`` in ``[0, 1]`` of the square ``extent_m`` wide around a point.

    Fetches the covering web-Mercator tiles over the network (Esri's terms of use apply).
    """
    import io
    import urllib.request

    from PIL import Image

    n = 2**zoom
    half_lat = extent_m / 2 / 111_320.0
    half_lon = half_lat / max(np.cos(np.radians(lat)), 1e-3)

    def pixel(lo: float, la: float) -> tuple[float, float]:
        x = (lo + 180.0) / 360.0 * n * 256
        y = (1 - np.arcsinh(np.tan(np.radians(la))) / np.pi) / 2 * n * 256
        return x, y

    x0, y0 = pixel(lon - half_lon, lat + half_lat)
    x1, y1 = pixel(lon + half_lon, lat - half_lat)
    tx, ty = range(int(x0 // 256), int(x1 // 256) + 1), range(int(y0 // 256), int(y1 // 256) + 1)
    mosaic = Image.new("RGB", (256 * len(tx), 256 * len(ty)))
    for i, x in enumerate(tx):
        for j, y in enumerate(ty):
            url = ESRI_TILES.format(z=zoom, x=x % n, y=y)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "geosae"}), timeout=30) as r:
                mosaic.paste(Image.open(io.BytesIO(r.read())).convert("RGB"), (256 * i, 256 * j))
    box = (x0 - 256 * tx[0], y0 - 256 * ty[0], x1 - 256 * tx[0], y1 - 256 * ty[0])
    return np.asarray(mosaic.crop(tuple(round(v) for v in box)).resize((size, size)), dtype=np.float64) / 255.0


def imagery(group, cell: int, *, esri: bool = False) -> np.ndarray | None:
    """The image of one store row: :func:`true_colour`, or with ``esri`` :func:`esri_imagery` at its centre.

    Falls back to the store's Sentinel-2 image when the tiles cannot be fetched.
    """
    if esri:
        try:
            return esri_imagery(float(group["lon"][cell]), float(group["lat"][cell]))
        except OSError as err:
            print(f"Esri imagery unavailable for row {cell} ({err}); using Sentinel-2")
    return true_colour(group, cell)


def add_imagery_argument(ap: argparse.ArgumentParser) -> None:
    """Add ``--esri``: draw Esri World Imagery instead of the store's Sentinel-2 true colour."""
    ap.add_argument(
        "--esri",
        action="store_true",
        help="draw Esri World Imagery tiles (network) instead of the store's Sentinel-2 true colour",
    )


def read_json(path: str | Path) -> dict:
    """A JSON file, with a hint at the stage that writes it when it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; see docs/neurips2026/reproduce-figures.md for the stage that writes it"
        )
    return json.loads(path.read_text())


def panel_rows(panel_dir: str | Path) -> list[dict]:
    """The rows of a ``geosae panel`` run directory (``eval/panel.json``)."""
    return read_json(Path(panel_dir) / "eval" / "panel.json")["rows"]


def write_json(obj: dict, path: str | Path) -> Path:
    """Write ``obj`` as indented JSON (numpy scalars and arrays allowed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    print("wrote", path)
    return path


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def parser(description: str) -> argparse.ArgumentParser:
    """An argument parser with ``--out``, the directory every output is written to."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", type=Path, default=Path("figures"), help="output directory (default: figures)")
    return ap


def add_paper_data_argument(ap: argparse.ArgumentParser) -> None:
    """Add ``--paper-data <dir>``: plot the camera-ready inputs of the Zenodo record instead of stage outputs."""
    ap.add_argument(
        "--paper-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="figure-data folder of the Zenodo record; plot its camera-ready inputs instead of stage outputs",
    )


def require(ap: argparse.ArgumentParser, args: argparse.Namespace, *names: str) -> None:
    """Exit with a usage error unless ``--paper-data`` or every stage-output argument in ``names`` is given."""
    if getattr(args, "paper_data", None) is not None:
        return
    missing = [n for n in names if not getattr(args, n.replace("-", "_"))]
    if missing:
        ap.error("needs " + ", ".join(f"--{n}" for n in missing) + " (or --paper-data)")


def paper_file(paper_data: str | Path, name: str) -> Path:
    """The camera-ready input ``name`` of the figure-data folder ``paper_data``.

    ``name`` is a path relative to the folder, e.g. ``products/olmoearth.json``. The Zenodo record
    stores the files of a subfolder flat, as ``products__olmoearth.json``; either layout is found.

    Raises:
        FileNotFoundError: Neither layout holds the file.
    """
    for path in (Path(paper_data) / name, Path(paper_data) / name.replace("/", "__")):
        if path.exists():
            return path
    raise FileNotFoundError(f"{Path(paper_data) / name} missing: --paper-data must point at the figure-data files")


# --------------------------------------------------------------------------- style


def set_style(canvas_width: float = TEXT_WIDTH, printed_width: float = TEXT_WIDTH) -> float:
    """Apply the paper style for a canvas ``canvas_width`` in wide, printed ``printed_width`` in wide.

    Returns:
        Figure points per printed point; :func:`fs` sizes text with it.
    """
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    import nicefigs as nf

    # Matplotlib's bundled DejaVu faces, so bold text renders the same on any machine.
    for ttf in sorted(Path(matplotlib.get_data_path(), "fonts", "ttf").glob("DejaVuSans*.ttf")):
        font_manager.fontManager.addfont(str(ttf))

    global _scale
    nf.set_theme("paper")
    _scale = canvas_width / printed_width
    matplotlib.rcParams.update(
        {
            "font.size": fs("label"),
            "axes.titlesize": fs("title"),
            "axes.labelsize": fs("label"),
            "xtick.labelsize": fs("tick"),
            "ytick.labelsize": fs("tick"),
            "legend.fontsize": fs("legend"),
            "figure.titlesize": fs("title"),
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "xtick.major.size": 2.0 * _scale,
            "ytick.major.size": 2.0 * _scale,
            "xtick.major.width": 0.4 * _scale,
            "ytick.major.width": 0.4 * _scale,
            "axes.linewidth": 0.5 * _scale,
        }
    )
    return _scale


def fs(role: str) -> float:
    """Font size of a text role (``panel``, ``title``, ``label``, ``tick``, ``legend``, ``annot``) on the canvas."""
    return PRINT_PT[role] * _scale


def panel_title(ax, letter: str, title: str = "") -> None:
    """Bold panel letter and an optional title above the top-left corner of ``ax``."""
    ax.annotate(
        letter,
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 2 * _scale),
        textcoords="offset points",
        fontsize=fs("panel"),
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    if title:
        ax.annotate(
            title,
            (0, 1),
            xycoords="axes fraction",
            xytext=(11 * _scale, 2 * _scale),
            textcoords="offset points",
            fontsize=fs("title"),
            ha="left",
            va="bottom",
        )


def variant_style(variant: str, *, color: str | None = None, markersize: float = 3.6, **overrides) -> dict:
    """``plot`` keywords of one SAE variant's marker; open markers get a white face."""
    _, marker, filled = SAE_VARIANT[variant]
    col = color or METHOD_COLORS[variant]
    kw = {
        "color": col,
        "marker": marker,
        "linestyle": "none",
        "markersize": markersize,
        "markeredgecolor": "white" if filled else col,
        "markerfacecolor": col if filled else "white",
        "markeredgewidth": 0.3 if filled else 0.8,
    }
    kw.update(overrides)
    return kw


def save(fig, out: str | Path, name: str, *, formats=("pdf", "png"), dpi: int = 300) -> Path:
    """Report overlapping text, write ``<out>/<name>.<format>`` for every format and close ``fig``."""
    import matplotlib.pyplot as plt
    import nicefigs as nf

    issues = nf.check_text_overlaps(fig, clipped=False)
    print(f"{name}: text overlaps: " + ("none" if not issues else "; ".join(" | ".join(i.texts) for i in issues)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out / f"{name}.{formats[0]}")
    return out / f"{name}.{formats[0]}"

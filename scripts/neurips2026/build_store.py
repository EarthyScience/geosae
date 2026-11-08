"""Build the data store of the paper: the ``majortom210k`` preset plus Sentinel-2 true colour.

The store's rows are the leading 210,000 cells of a seeded draw (seed 42) of 1,000,000 cells of
the MajorTOM 5 km land grid (5,418,682 cells); the draw's order depends on its size, so drawing
210,000 cells directly gives other cells. The embeddings (AlphaEarth, Tessera) fill every row and
the reference products the 100,000 test rows. The script then adds the ``sentinel2_l2a`` product,
which the image panels of the figures read. Both steps run the ``geosae download`` stage with
``configs/download.yaml`` and resume where an interrupted build stopped.

    python scripts/neurips2026/build_store.py --out data/majortom210k.zarr --gee-project <project>
    python scripts/neurips2026/build_store.py --out data/smoke.zarr --rows 3     # a check of every source
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geosae.cli import cmd_download
from geosae.config import load_config
from geosae.data.download.presets import MAJORTOM210K

#: The stage config the store is built with.
DOWNLOAD_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "download.yaml"
#: The product the image panels read, added after the preset's own.
IMAGERY = "sentinel2_l2a"


def download_configs(
    out: str | Path, *, gee_project: str | None = None, max_workers: int = 8, rows: int | None = None
) -> list[dict]:
    """The ``geosae download`` configs of the build: the preset's products, then the imagery.

    Args:
        out: Store path.
        gee_project: Earth Engine project of the ``gee`` products (``None``: the credentials' default).
        max_workers: Concurrent tile fetches per product.
        rows: Leading rows to fetch (``None``: every row), for a check of the sources.

    Returns:
        One resolved config per ``geosae download`` call, in build order.
    """
    common = [f"download.out={out}", "download.preset=majortom210k", f"download.max_workers={max_workers}"]
    if gee_project:
        common.append(f"download.gee_project={gee_project}")
    if rows is not None:
        common.append(f"download.rows={rows}")
    return [
        load_config([str(DOWNLOAD_CONFIG)], [*common, "download.products=null"]),
        load_config([str(DOWNLOAD_CONFIG)], [*common, f"download.products=[{IMAGERY}]"]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/majortom210k.zarr"), help="store path")
    ap.add_argument("--gee-project", default=None, help="Earth Engine project (AlphaEarth, GLC_FCS30D, WorldCereal)")
    ap.add_argument("--max-workers", type=int, default=8, help="concurrent tile fetches per product")
    ap.add_argument("--rows", type=int, default=None, help="fetch only the leading rows (a check of the sources)")
    args = ap.parse_args()

    idx = MAJORTOM210K.cell_indices()
    print(
        f"majortom210k: the leading {MAJORTOM210K.n_cells:,} of a draw of {MAJORTOM210K.draw_size:,} of "
        f"{MAJORTOM210K.base_ncells:,} cells (seed {MAJORTOM210K.seed}); first cells {idx[:3].tolist()}"
    )
    for cfg in download_configs(args.out, gee_project=args.gee_project, max_workers=args.max_workers, rows=args.rows):
        cmd_download(cfg)


if __name__ == "__main__":
    main()

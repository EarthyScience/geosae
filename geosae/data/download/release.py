"""Export what a reader needs to rebuild the reference dataset, as plain tables.

The store itself is hundreds of gigabytes and several of its reference products
may not be redistributed (:class:`~geosae.data.download.sources.Redistribution`),
so the reproducible part of the dataset is not the bytes but the *recipe*: which
MajorTOM cells were drawn, which split each one landed in, which product was
fetched how, and which concept classes the evaluation scores against. This
module writes those four tables:

``cells.csv``
    One row per populated store cell: its MajorTOM grid cell id (``645U_575R``),
    its row/column in the 5 km grid, its centre, its UTM code, its split, and
    which embeddings it carries. This is the "exact cell ids" a rebuild needs.
``products.csv``
    The product registry as fetched: store array, native resolution, resampling
    onto the cell geobox, fill value, source, licence and what that licence
    allows us to pass on.
``concepts.csv``
    The concept registry the alignment metrics score against: one row per
    product class, with its code, name and concept group.
``manifest.json``
    The selection (base grid, seed, cell count), the split ranges and their
    realised counts, and the `geosae` version that wrote the export.

The same module also packs a public copy of the store:
:func:`redistributable_arrays` says which arrays a licence lets us pass on and
why each of the others does not, and :func:`pack_release_store` copies the
former into a fresh store.

Everything but ``cells.csv`` needs only the core install; the cell table needs
the MajorTOM grid from the ``download`` extra, because a grid cell id is a
function of the grid, not of the store.
"""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

#: Value of the store's ``split_*`` array per split name (-1 = not in any split).
SPLIT_CODES: dict[int, str] = {0: "train", 1: "val", 2: "test"}

_CELL_COLUMNS = (
    "row",
    "majortom_cell",
    "majortom_row",
    "majortom_col",
    "lon",
    "lat",
    "crs",
    "split",
)


def majortom_cell_ids(lon: np.ndarray, lat: np.ndarray, grid_distance_km: int = 5) -> tuple[np.ndarray, ...]:
    """MajorTOM grid cell ids for a set of cell centres.

    A cell id is the grid's own row and column name, ``"<row><U|D>_<col><L|R>"``
    — the identifier the MajorTOM release itself uses, so a reader can pull the
    same cells from the upstream dataset without our store.

    Args:
        lon: Cell centre longitudes, degrees east.
        lat: Cell centre latitudes, degrees north.
        grid_distance_km: MajorTOM grid spacing the cells were drawn from.

    Returns:
        ``(cell_id, row_idx, col_idx)``, each of length ``len(lon)``.
    """
    from geosae.data.download.grid import MajorTomGrid

    grid = MajorTomGrid(grid_distance_km)
    rows, cols = grid.rowcol(lon, lat)
    ids = np.array([grid.cell_id(int(r), int(c)) for r, c in zip(rows, cols, strict=True)])
    return ids, rows.astype(np.int64), cols.astype(np.int64)


def cell_records(store_path: str | Path, *, split_name: str = "split_210k") -> list[dict[str, Any]]:
    """One record per cell of a store that belongs to a split.

    Args:
        store_path: Patch store (patch-store layout).
        split_name: Store array holding the split labels.

    Returns:
        Records with the keys of ``cells.csv`` plus one ``has_<embedding>`` flag
        per embedding array the store carries, ordered by store row.
    """
    import zarr

    group = zarr.open_group(str(store_path), mode="r")
    split = np.asarray(group[split_name][:])
    rows = np.nonzero(split >= 0)[0]
    lon = np.asarray(group["lon"][:])[rows]
    lat = np.asarray(group["lat"][:])[rows]
    crs = np.asarray(group["crs"][:])[rows]
    ids, grid_rows, grid_cols = majortom_cell_ids(lon, lat)

    embeddings = [k for k in ("alphaearth", "tessera", "copernicusfm") if f"mask_{k}" in group]
    present = {k: np.asarray(group[f"mask_{k}"][:], dtype=bool)[rows] for k in embeddings}

    records = []
    for i, row in enumerate(rows):
        rec: dict[str, Any] = {
            "row": int(row),
            "majortom_cell": str(ids[i]),
            "majortom_row": int(grid_rows[i]),
            "majortom_col": int(grid_cols[i]),
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "crs": str(crs[i]),
            "split": SPLIT_CODES[int(split[row])],
        }
        for k in embeddings:
            rec[f"has_{k}"] = bool(present[k][i])
        records.append(rec)
    return records


def product_records(products: list[str] | None = None) -> list[dict[str, Any]]:
    """The product registry as a table: how each array was fetched, and its licence.

    Args:
        products: Registry keys to export; ``None`` takes the whole registry.

    Returns:
        One record per product, sorted by kind then key.
    """
    from geosae.data.download.presets import MAJORTOM210K
    from geosae.data.download.sources import all_products, get_product

    specs = [get_product(k) for k in products] if products else list(all_products().values())
    records = [
        {
            "key": s.key,
            "array": s.array,
            "kind": s.kind.value,
            "classes": "" if s.classes is None else int(s.classes),
            "native_resolution_m": s.resolution_m,
            "resampling": s.resampling,
            "fill_value": "nan" if isinstance(s.fill_value, float) and np.isnan(s.fill_value) else s.fill_value,
            "dtype": s.dtype,
            "in_majortom210k": s.key in MAJORTOM210K.products,
            "auth": s.auth.value,
            "fetcher": s.fetcher,
            "fetcher_kwargs": json.dumps(s.fetcher_kwargs, default=str),
            "source": s.source,
            "licence": s.licence,
            "licence_url": s.licence_url,
            "redistribution": s.redistribution.value,
        }
        for s in specs
    ]
    return sorted(records, key=lambda r: (r["kind"], r["key"]))


def concept_records(products: list[str] | None = None) -> list[dict[str, Any]]:
    """The concept registry the alignment metrics score against, as a table.

    Args:
        products: Product subset; ``None`` takes every product the concept
            catalog covers.

    Returns:
        One record per product class: ``product``, ``code``, ``name``, ``group``.
    """
    from geosae.eval.registry import build_class_table

    class_table, product_order, names, groups = build_class_table(products)
    return [
        {
            "product": product_order[int(pid)],
            "code": int(code),
            "name": names[i],
            "group": groups[i] or "",
        }
        for i, (pid, code) in enumerate(class_table)
    ]


#: Store arrays that describe the cells rather than a product, and therefore
#: carry no licence of their own.
COORD_ARRAYS: tuple[str, ...] = ("cell_indices", "lon", "lat", "crs", "transform", "x", "y")

#: Redistribution verdicts a public release may carry without relicensing it.
PACKABLE: tuple[str, ...] = ("open", "attribution")


def redistributable_arrays(
    store_path: str | Path,
    *,
    allow: tuple[str, ...] = PACKABLE,
    array_products: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Split a store's arrays into what may be republished and what may not.

    An array may be republished when its product's licence allows us to pass it
    on (:class:`~geosae.data.download.sources.Redistribution`). Coordinate,
    split and ``feature_*`` arrays describe the cells and always travel; an
    array with no registry entry never does, because nothing records what it
    was fetched from or under what terms.

    An array fetched under a name of its own (``geosae refetch``) names its
    product in a ``product`` attribute; ``array_products`` supplies that mapping
    for stores written before the attribute existed.

    Args:
        store_path: Patch store to classify.
        allow: Redistribution verdicts that may be packed.
        array_products: ``{array: registry key}`` for arrays whose product
            cannot be read off the store.

    Returns:
        ``(packable, excluded)`` — the array names to copy, and ``{array:
        reason}`` for every array left behind.
    """
    import zarr

    from geosae.data.download.sources import all_products, get_product

    group = zarr.open_group(str(store_path), mode="r")
    by_array = {s.array: s for s in all_products().values()}
    mapping = dict(array_products or {})
    packable, excluded = [], {}
    for name in sorted(group.array_keys()):
        base = name[len("mask_") :] if name.startswith("mask_") else name
        if base in COORD_ARRAYS or base.startswith(("split_", "feature_")):
            packable.append(name)
            continue
        key = mapping.get(base) or (group[base].attrs.get("product") if base in group else None)
        spec = by_array.get(base) if key is None else get_product(key)
        if spec is None:
            excluded[name] = "no registry entry: source and licence unrecorded"
        elif spec.redistribution.value in allow:
            packable.append(name)
        else:
            excluded[name] = f"{spec.licence} ({spec.redistribution.value})"
    return packable, excluded


def pack_release_store(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    allow: tuple[str, ...] = PACKABLE,
    array_products: dict[str, str] | None = None,
    exclude: tuple[str, ...] = (),
    rows_per_block: int = 500,
    progress: bool = False,
) -> dict[str, Any]:
    """Copy the redistributable arrays of a store into a fresh store.

    The copy is array-by-array and block-by-block, so a store far larger than
    memory packs in bounded space, and re-running it overwrites the blocks it
    already wrote rather than duplicating them.

    Args:
        src_path: Store to pack.
        dst_path: Destination store (created if missing).
        allow: Redistribution verdicts that may be packed.
        array_products: ``{array: registry key}``, as in
            :func:`redistributable_arrays`.
        exclude: Arrays to leave out although their licence allows them, with
            their masks — a release may drop a layer for its size rather than
            for its terms.
        rows_per_block: Cells copied per read/write.
        progress: Print one line per array.

    Returns:
        ``{"arrays": [...], "excluded": {...}, "n_cells": int}``.
    """
    import zarr

    src = zarr.open_group(str(src_path), mode="r")
    dst = zarr.open_group(str(dst_path), mode="a")
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    packable, excluded = redistributable_arrays(src_path, allow=allow, array_products=array_products)
    for name in list(packable):
        base = name[len("mask_") :] if name.startswith("mask_") else name
        if base in exclude:
            packable.remove(name)
            excluded[name] = "excluded by the caller"
    n_cells = int(src.attrs.get("n_cells", 0))
    for name in packable:
        a = src[name]
        if name in dst:
            out = dst[name]
        else:
            out = dst.create_array(name, shape=a.shape, chunks=a.chunks, dtype=a.dtype, fill_value=a.fill_value)
            for k, v in a.attrs.items():
                out.attrs[k] = v
        if progress:
            print(f"[geosae] packing {name} {a.shape}", flush=True)
        step = a.shape[0] if a.ndim == 1 and a.shape[0] <= rows_per_block else rows_per_block
        for lo in range(0, a.shape[0], step):
            out[lo : lo + step] = a[lo : lo + step]
    zarr.consolidate_metadata(dst.store)
    return {"arrays": packable, "excluded": excluded, "n_cells": n_cells}


def _write_csv(path: Path, records: list[dict[str, Any]], columns: list[str]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return path


def write_release_tables(
    store_path: str | Path,
    out_dir: str | Path,
    *,
    split_name: str = "split_210k",
    products: list[str] | None = None,
    array_products: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write ``cells.csv``, ``products.csv``, ``concepts.csv`` and ``manifest.json``.

    Args:
        store_path: Patch store the cells are read from.
        out_dir: Directory the tables are written to (created if missing).
        split_name: Split array to export.
        products: Product subset for ``products.csv`` / ``concepts.csv``;
            ``None`` exports the whole registry.
        array_products: ``{array: registry key}`` for store arrays whose product
            cannot be read off the store (see :func:`redistributable_arrays`).

    Returns:
        ``{table: path}`` for the four files written.
    """
    from geosae.data.download.presets import MAJORTOM210K

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cells = cell_records(store_path, split_name=split_name)
    cell_columns = list(_CELL_COLUMNS) + [k for k in cells[0] if k.startswith("has_")]
    prods = product_records(products)
    concepts = concept_records(products)

    paths = {
        "cells": str(_write_csv(out / "cells.csv", cells, cell_columns)),
        "products": str(_write_csv(out / "products.csv", prods, list(prods[0]))),
        "concepts": str(_write_csv(out / "concepts.csv", concepts, list(concepts[0]))),
    }

    counts: dict[str, int] = {}
    for rec in cells:
        counts[rec["split"]] = counts.get(rec["split"], 0) + 1
    try:
        geosae_version = version("geosae")
    except PackageNotFoundError:  # pragma: no cover - only without an install
        geosae_version = "unknown"
    packable, excluded = redistributable_arrays(store_path, array_products=array_products)
    manifest = {
        "store": str(store_path),
        "split_name": split_name,
        "split_counts": counts,
        "n_cells": len(cells),
        "selection": {
            "base_grid": "MajorTOM 5 km land cells",
            "base_ncells": MAJORTOM210K.base_ncells,
            "seed": MAJORTOM210K.seed,
            "draw_size": MAJORTOM210K.draw_size,
            "subset_size": MAJORTOM210K.n_cells,
            "splits": {
                name: {"train": spec.train, "val": spec.val, "test": spec.test, "chunk_size": spec.chunk_size}
                for name, spec in MAJORTOM210K.splits.items()
            },
        },
        "redistribution": {r["key"]: r["redistribution"] for r in prods},
        "packable_arrays": packable,
        "excluded_arrays": excluded,
        "geosae_version": geosae_version,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    paths["manifest"] = str(out / "manifest.json")
    return paths


__all__ = [
    "COORD_ARRAYS",
    "PACKABLE",
    "SPLIT_CODES",
    "pack_release_store",
    "redistributable_arrays",
    "majortom_cell_ids",
    "cell_records",
    "product_records",
    "concept_records",
    "write_release_tables",
]

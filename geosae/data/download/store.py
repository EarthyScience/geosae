"""Harmonized zarr writer optimized for fast IO during training.

Given a schema (a set of products plus store geometry), writes a dataset to
disk as a single zarr group with array names, shapes, dtypes, chunking, fill
values, dimension names and attrs chosen for fast, chunk-aligned reads --
using plain ``zarr`` v3 + numpy.

Two layers:

* **Schema (numpy/zarr-only, CI-testable):** :func:`build_schemas` returns the
  full array layout for a product set as :class:`ArraySchema` objects, and
  :func:`create_store` materialises it into an empty zarr group. The cell-axis
  chunking rule targets a fixed uncompressed size per chunk: embeddings target
  :data:`EMBED_CHUNK_BYTES` per chunk, label arrays :data:`LABEL_CHUNK_BYTES`,
  1-D per-cell coordinates/masks use :data:`COORD_CHUNK`, split labels
  :data:`SPLIT_CHUNK`.

* **Assembly (needs the ``download`` extra):** :func:`build_store` closes the
  select -> fetch -> assemble pipeline over a fixed list of cells: per-cell UTM
  geobox -> per-product :func:`~geosae.data.download.sources.build_fetcher`
  fetch -> write rows + per-product success masks -> chunk-aligned splits. It
  resumes where an interrupted run stopped. :func:`build_majortom_store` builds
  one over chosen MajorTOM cells (see :mod:`geosae.data.download.grid`), and
  :func:`build_preset_store` over a preset's cells.

The output interoperates with :class:`geosae.data.dataset.PatchData` (same
coords, ``mask_<embedding>`` and ``split_<name>`` conventions). For a
preregistered default (product set, seed, split), see
:mod:`geosae.data.download.presets`.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geosae.data.download.selection import SplitSpec, compute_split
from geosae.data.download.sources import Kind, ProductSpec, all_products

# --- Layout constants (reverse-engineered from the on-disk store) ------------
#: Store-wide canonical attrs.
CANONICAL_YEAR = 2024
PATCH_SIZE = 64
RESOLUTION_M = 10.0

#: Target uncompressed bytes per chunk along the cell axis.
EMBED_CHUNK_BYTES = 50 * 2**20  # 50 MiB -> AlphaEarth 50 cells, Tessera 25
LABEL_CHUNK_BYTES = 20 * 2**20  # 20 MiB -> uint8 5120, float32 1280, ...
#: Fixed cell-axis chunk for 1-D per-cell coordinate + mask arrays.
COORD_CHUNK = 10_000
#: Fixed cell-axis chunk for the int8 split arrays.
SPLIT_CHUNK = 100_000

INT64_FILL = np.iinfo(np.int64).min  # -9223372036854775808, the store's int64 fill

#: Band (last-axis) dimension name for the multi-band label products.
_BAND_DIM = {"overture": "overture_band", "soilgrids": "soil_var"}


def cell_chunk(dtype: str, tail_shape: tuple[int, ...], target_bytes: int, n_cells: int) -> int:
    """Cells per chunk so one chunk is ``~target_bytes`` uncompressed.

    ``floor(target_bytes / bytes_per_cell)``, clamped to ``[1, n_cells]`` --
    the exact rule the on-disk store's chunk sizes follow.
    """
    bytes_per_cell = np.dtype(dtype).itemsize * int(np.prod(tail_shape)) if tail_shape else np.dtype(dtype).itemsize
    return int(min(max(1, target_bytes // bytes_per_cell), n_cells))


@dataclass
class ArraySchema:
    """Full spec of one store array (everything needed to create it identically)."""

    dtype: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    fill_value: object
    dim_names: tuple[str, ...] | None
    attrs: dict = field(default_factory=dict)


def _dims_attr(dim_names: tuple[str, ...] | None, extra: dict | None = None) -> dict:
    attrs: dict = {}
    if dim_names is not None:
        attrs["_ARRAY_DIMENSIONS"] = list(dim_names)
    if extra:
        attrs.update(extra)
    return attrs


def _embedding_schema(array: str, feature_dim: str, dim: int, n_cells: int, patch_size: int) -> ArraySchema:
    dims = ("cell", "y", "x", feature_dim)
    coords = {
        "alphaearth": "cell_indices feature_alphaearth lat lon x y",
        "tessera": "cell_indices feature_alphaearth feature_tessera lat lon x y",
    }.get(array, f"cell_indices {feature_dim} lat lon x y")
    chunk = cell_chunk("float32", (patch_size, patch_size, dim), EMBED_CHUNK_BYTES, n_cells)
    return ArraySchema(
        dtype="float32",
        shape=(n_cells, patch_size, patch_size, dim),
        chunks=(chunk, patch_size, patch_size, dim),
        fill_value=float("nan"),
        dim_names=dims,
        attrs=_dims_attr(dims, {"embedding_dim": dim, "coordinates": coords}),
    )


def _label_schema(spec: ProductSpec, n_cells: int, patch_size: int) -> ArraySchema:
    tail = (patch_size, patch_size, *tuple(spec.shape)[2:])  # (H, W) or (H, W, C)
    if len(tail) == 3:
        dims = ("cell", "y", "x", _BAND_DIM.get(spec.array, f"{spec.array}_band"))
    else:
        dims = ("cell", "y", "x")
    chunk = cell_chunk(spec.dtype, tail, LABEL_CHUNK_BYTES, n_cells)
    fill = float("nan") if spec.dtype.startswith("float") else spec.fill_value
    return ArraySchema(
        dtype=spec.dtype,
        shape=(n_cells, *tail),
        chunks=(chunk, *tail),
        fill_value=fill,
        dim_names=dims,
        attrs=_dims_attr(dims),
    )


def _mask_schema(array: str, n_cells: int) -> ArraySchema:
    coords = {
        "alphaearth": "cell_indices feature_alphaearth lat lon x y",
        "tessera": "cell_indices feature_alphaearth feature_tessera lat lon x y",
    }.get(array)
    extra = {"coordinates": coords} if coords else None
    return ArraySchema(
        dtype="bool",
        shape=(n_cells,),
        chunks=(min(COORD_CHUNK, n_cells),),
        fill_value=False,
        dim_names=("cell",),
        attrs=_dims_attr(("cell",), extra),
    )


def embedding_schemas(array: str, dim: int, n_cells: int, patch_size: int) -> dict[str, ArraySchema]:
    """Layout of one embedding: the patch array, its feature axis and its mask.

    The three arrays :class:`~geosae.data.dataset.PatchData` needs to open an
    embedding, so any writer of a store it reads — including a synthetic one —
    gets the real store's dtypes, chunking and dim
    names without restating them.

    Args:
        array: Store array name for the embedding.
        dim: ``D``, its feature width.
        n_cells: Number of cells.
        patch_size: Pixels per cell side.
    """
    fdim = f"feature_{array}"
    return {
        array: _embedding_schema(array, fdim, dim, n_cells, patch_size),
        fdim: ArraySchema("int64", (dim,), (dim,), INT64_FILL, (fdim,), _dims_attr((fdim,))),
        f"mask_{array}": _mask_schema(array, n_cells),
    }


def split_schema(n_cells: int) -> ArraySchema:
    """Layout of one ``split_<name>`` label array (int8 train/val/test)."""
    return ArraySchema("int8", (n_cells,), (min(SPLIT_CHUNK, n_cells),), np.int8(0), None, {})


def lonlat_schemas(n_cells: int) -> dict[str, ArraySchema]:
    """Layout of the per-cell ``lon``/``lat`` patch-centre coordinates."""
    return {k: v for k, v in _coord_schemas(n_cells, 1).items() if k in ("lon", "lat")}


def _coord_schemas(n_cells: int, patch_size: int) -> dict[str, ArraySchema]:
    cc = min(COORD_CHUNK, n_cells)
    ll_coords = "cell_indices lat lon x y"
    return {
        "cell_indices": ArraySchema("int64", (n_cells,), (cc,), INT64_FILL, ("cell",), _dims_attr(("cell",))),
        "lon": ArraySchema("float64", (n_cells,), (cc,), float("nan"), ("cell",), _dims_attr(("cell",))),
        "lat": ArraySchema("float64", (n_cells,), (cc,), float("nan"), ("cell",), _dims_attr(("cell",))),
        "crs": ArraySchema("str", (n_cells,), (cc,), "", ("cell",), _dims_attr(("cell",), {"coordinates": ll_coords})),
        "transform": ArraySchema(
            "float64",
            (n_cells, 6),
            (cc, 6),
            0.0,
            ("cell", "transform_param"),
            _dims_attr(("cell", "transform_param"), {"coordinates": ll_coords}),
        ),
        "x": ArraySchema("int64", (patch_size,), (patch_size,), INT64_FILL, ("x",), _dims_attr(("x",))),
        "y": ArraySchema("int64", (patch_size,), (patch_size,), INT64_FILL, ("y",), _dims_attr(("y",))),
    }


def build_schemas(
    products: list[str] | None,
    n_cells: int,
    *,
    embeddings: tuple[str, ...] = ("alphaearth", "tessera"),
    splits: tuple[str, ...] = (),
    patch_size: int = PATCH_SIZE,
) -> dict[str, ArraySchema]:
    """Return the full ``{array_name: ArraySchema}`` layout for a build.

    Args:
        products: Registry product keys to include as label arrays. ``None``
            selects every non-embedding product in the registry.
        n_cells: Number of cells (store rows).
        embeddings: Embedding arrays to include (each with its ``feature_*`` and
            ``mask_*``).
        splits: Names of split arrays to include (each written as ``split_<name>``
            elsewhere in the pipeline via :func:`~geosae.data.download.selection.compute_split`).
        patch_size: Pixels per cell side. ``1`` gives a one-vector-per-cell store
            (a coarse global grid rather than a raster patch).
    """
    reg = all_products()
    if products is None:
        # Every *reference* product. Filtering on kind, not on membership of
        # ``embeddings``, so registering a new embedding cannot turn it into a
        # label array here just because this build did not ask for it.
        products = [k for k, s in reg.items() if s.kind is not Kind.EMBEDDING]

    schemas: dict[str, ArraySchema] = {}
    schemas.update(_coord_schemas(n_cells, patch_size))

    # Embeddings + their feature coords + masks.
    by_array = {s.array: s for s in reg.values() if s.kind == "embedding"}
    for emb in embeddings:
        if emb not in by_array:
            raise KeyError(f"unknown embedding {emb!r}; registered: {sorted(by_array)}")
        schemas.update(embedding_schemas(emb, int(by_array[emb].shape[-1]), n_cells, patch_size))

    # Label products (dedup by store array) + masks.
    seen: set[str] = set()
    for key in products:
        spec = reg[key]
        if spec.array in seen or spec.array in embeddings:
            continue
        seen.add(spec.array)
        schemas[spec.array] = _label_schema(spec, n_cells, patch_size)
        schemas[f"mask_{spec.array}"] = _mask_schema(spec.array, n_cells)

    # Split arrays (int8, no _ARRAY_DIMENSIONS, provenance attrs).
    for sp in splits:
        schemas[sp] = split_schema(n_cells)
    return schemas


# --------------------------------------------------------------------------- #
# Store creation (zarr v3)
# --------------------------------------------------------------------------- #
def _zarr_fill(schema: ArraySchema):
    if schema.dtype == "str":
        return ""
    return schema.fill_value


def create_store(
    path, schemas: dict[str, ArraySchema], *, n_cells: int, patch_size: int = PATCH_SIZE, year: int = CANONICAL_YEAR
):
    """Create an empty zarr v3 store with every array from ``schemas``.

    Root attrs (``patch_size``, ``n_cells``, ``year``, ``modalities``) mirror
    the on-disk store. Returns the open :class:`zarr.Group` (mode ``w``).
    """
    import zarr

    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs["patch_size"] = int(patch_size)
    group.attrs["n_cells"] = int(n_cells)
    group.attrs["year"] = int(year)
    group.attrs["modalities"] = _modalities_attr(schemas)

    for name, s in schemas.items():
        _create_array(group, name, s)
    return group


def _create_array(group, name: str, s: ArraySchema):
    """Create one array of a schema, with its attrs."""
    if s.dtype == "str":
        dt = np.dtypes.StringDType() if hasattr(np, "dtypes") else str
        arr = group.create_array(name, shape=s.shape, chunks=s.chunks, dtype=dt, fill_value="")
    else:
        arr = group.create_array(name, shape=s.shape, chunks=s.chunks, dtype=s.dtype, fill_value=_zarr_fill(s))
    for k, v in s.attrs.items():
        arr.attrs[k] = v
    return arr


def _modalities_attr(schemas: dict[str, ArraySchema]) -> dict:
    """Reproduce the store's per-label-product ``modalities`` root attr."""
    out: dict[str, dict] = {}
    for name, s in schemas.items():
        if s.dim_names is None or s.dim_names[0] != "cell" or len(s.dim_names) < 3:
            continue
        if name.startswith(("mask_", "feature_")) or "embedding_dim" in s.attrs:
            continue
        tail = tuple(s.shape[1:])
        fv = float("nan") if s.dtype.startswith("float") else s.fill_value
        out[name] = {
            "shape": list(tail),
            "dtype": s.dtype,
            "dim_names": list(s.dim_names[1:]),
            "fill_value": fv,
            "category": "label",
            "attrs": {},
        }
    return out


# --------------------------------------------------------------------------- #
# Per-cell UTM geobox
# --------------------------------------------------------------------------- #
def utm_geobox(lon: float, lat: float, patch_size: int = PATCH_SIZE, resolution_m: float = RESOLUTION_M):
    """Return ``(geobox, crs_str, transform_params)`` for a patch centred at
    ``(lon, lat)`` in its native UTM zone (10 m pixels, ``patch_size`` square).

    Needs the ``download`` extra (rasterio / odc-geo). ``transform_params`` is
    the 6-vector ``(a, b, c, d, e, f)`` written to the store's ``transform``.
    """
    import rasterio.warp as rw
    from odc.geo.geobox import GeoBox
    from rasterio.transform import Affine

    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    crs_str = f"EPSG:{epsg}"
    xs, ys = rw.transform("EPSG:4326", crs_str, [lon], [lat])
    cx, cy = xs[0], ys[0]
    half = patch_size * resolution_m / 2.0
    transform = Affine(resolution_m, 0, cx - half, 0, -resolution_m, cy + half)
    geobox = GeoBox((patch_size, patch_size), transform, crs_str)
    params = np.array([transform.a, transform.b, transform.c, transform.d, transform.e, transform.f], dtype=np.float64)
    return geobox, crs_str, params


# --------------------------------------------------------------------------- #
# Full assembly: select -> fetch -> write (needs the download extra)
# --------------------------------------------------------------------------- #
def write_geoboxes(group, rows, lon, lat, patch_size: int = PATCH_SIZE, resolution_m: float = RESOLUTION_M) -> dict:
    """Write each row's UTM ``crs`` + ``transform`` and return its geobox.

    Args:
        group: Open store group.
        rows: Store rows to populate, aligned with ``lon`` / ``lat``.
        lon: Cell centre longitudes in degrees.
        lat: Cell centre latitudes in degrees.
        patch_size: Patch side length in pixels.
        resolution_m: Pixel size in metres.

    Returns:
        ``{row: GeoBox}`` for the populated rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    geoboxes: dict[int, object] = {}
    crs_values: list[str] = []
    transforms = np.empty((rows.size, 6), dtype=np.float64)
    for i, (r, lo, la) in enumerate(zip(rows, np.asarray(lon), np.asarray(lat), strict=True)):
        gb, crs_str, params = utm_geobox(float(lo), float(la), patch_size, resolution_m)
        geoboxes[int(r)] = gb
        crs_values.append(crs_str)
        transforms[i] = params
    # One vectorised write per array: a row-at-a-time write rewrites the whole
    # enclosing chunk, which is quadratic in the number of rows.
    group["crs"].oindex[rows] = np.asarray(crs_values, dtype=group["crs"].dtype)
    group["transform"].oindex[rows] = transforms
    return geoboxes


def add_product_array(group, key: str, array: str | None = None) -> str:
    """Create one product's array and success mask on an existing store, if absent.

    A store is built once with every array it will ever have, which leaves no way
    to add a product — or to re-fetch a wrong one *beside* the values it replaces
    rather than over them. Naming a destination array does that.

    Args:
        group: Open store group.
        key: Registry product key.
        array: Destination array name; defaults to the product's own.

    Returns:
        The destination array name.
    """
    from geosae.data.download.sources import get_product

    spec = get_product(key)
    array = array or spec.array
    n_cells, patch_size = int(group.attrs["n_cells"]), int(group.attrs["patch_size"])
    if array not in group:
        schema = _label_schema(spec, n_cells, patch_size)
        created = group.create_array(
            array, shape=schema.shape, chunks=schema.chunks, dtype=schema.dtype, fill_value=_zarr_fill(schema)
        )
        for name, value in schema.attrs.items():
            created.attrs[name] = value
        # An array fetched under a name other than the product's own has no other
        # record of where it came from, and a licence follows the product.
        created.attrs["product"] = spec.key
        if spec.bands:
            created.attrs["bands"] = list(spec.bands)
    if f"mask_{array}" not in group:
        mask = _mask_schema(array, n_cells)
        group.create_array(f"mask_{array}", shape=mask.shape, chunks=mask.chunks, dtype=mask.dtype, fill_value=False)
    return array


def populate_products(
    group,
    products,
    rows,
    lon,
    lat,
    geoboxes,
    *,
    fetch_kwargs=None,
    max_workers: int = 1,
    into=None,
    group_by_tile: bool = True,
) -> None:
    """Fetch every product for the given rows and write rows + success masks.

    Cells are grouped by the fetcher's tile key so a source tile is opened once
    per group; groups are fetched concurrently when ``max_workers > 1``.

    Args:
        group: Open store group with the product arrays already created.
        products: Registry product keys to fetch.
        rows: Store rows to populate, aligned with ``lon`` / ``lat``.
        lon: Cell centre longitudes in degrees.
        lat: Cell centre latitudes in degrees.
        geoboxes: ``{row: GeoBox}`` as returned by :func:`write_geoboxes`.
        fetch_kwargs: Optional ``{product_key: {kwargs}}`` forwarded to
            :func:`~geosae.data.download.sources.build_fetcher`.
        max_workers: Concurrent tile fetches. Network-bound fetchers benefit;
            keep at 1 for fetchers that hold large per-tile state.
        into: Optional ``{product_key: array_name}`` redirecting a product's
            values to a different array than its own — which is how a re-fetch
            lands beside the values it replaces instead of over them.
        group_by_tile: Group cells by the fetcher's tile key (one source read
            per tile). ``False`` makes every cell its own work unit, which is
            what a sparse sample needs: a block read over the union of a few
            far-apart cells in one tile can be most of the tile.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rasterio.warp import Resampling

    from geosae.data.download.sources import build_fetcher, get_product

    fetch_kwargs = fetch_kwargs or {}
    rows = np.asarray(rows, dtype=np.int64)
    coords = {int(r): (float(lo), float(la)) for r, lo, la in zip(rows, np.asarray(lon), np.asarray(lat), strict=True)}

    for key in products:
        spec = get_product(key)
        fetcher = build_fetcher(key, **fetch_kwargs.get(key, {}))
        resampling = Resampling.nearest if spec.resampling == "nearest" else Resampling.bilinear
        array = (into or {}).get(key, spec.array)
        arr = group[array]
        mask = group[f"mask_{array}"]

        by_tile: dict = {}
        for r in rows:
            lo, la = coords[int(r)]
            by_tile.setdefault(fetcher.tile_key(lo, la), []).append(int(r))
        groups = [
            (tk, [(r, *coords[r], geoboxes[r]) for r in unit])
            for tk, tcells in by_tile.items()
            for unit in ([tcells] if group_by_tile else [[r] for r in tcells])
        ]

        def _fetch(job, fetcher=fetcher, resampling=resampling, nodata=spec.fill_value, embedding=spec.kind):
            tile_key, cell_tuples = job
            attempted = [c[0] for c in cell_tuples]
            results = list(_fetch_tile(fetcher, tile_key, cell_tuples, resampling, nodata))
            if embedding is Kind.EMBEDDING:
                # a source can answer for a cell it has no embedding of, with fill
                results = [(r, p) for r, p in results if has_embedding(p)]
            return attempted, results

        writer = _ChunkWriter(arr, mask, rows)
        if max_workers <= 1:
            for job in groups:
                writer.add(*_fetch(job))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for fut in as_completed([pool.submit(_fetch, job) for job in groups]):
                    writer.add(*fut.result())
        writer.flush()


def has_embedding(patch) -> bool:
    """Whether a fetched embedding patch holds any finite value.

    Constant patches (an all-zero fill among them) are kept: the dataset's
    quality filters (:mod:`geosae.data.quality`) drop them at load time.
    """
    return bool(np.isfinite(np.asarray(patch, dtype=np.float32)).any())


class _ChunkWriter:
    """Buffer fetched patches and write each cell-axis chunk exactly once.

    Writing a single row rewrites (decompresses, merges, recompresses) its whole
    enclosing chunk, so a row-at-a-time writer costs one full chunk rewrite per
    cell. Buffering until every requested row of a chunk has arrived turns that
    into one write per chunk.
    """

    def __init__(self, arr, mask, rows: np.ndarray):
        self._arr = arr
        self._mask = mask
        self._chunk = int(arr.chunks[0])
        self._buffer: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        self._remaining: dict[int, int] = defaultdict(int)
        for r in np.asarray(rows, dtype=np.int64):
            self._remaining[int(r) // self._chunk] += 1

    def add(self, attempted, results) -> None:
        """Buffer ``(row, patch)`` results, flushing every chunk that completes.

        Args:
            attempted: Every row the fetch covered, successful or not.
            results: The ``(row, patch)`` pairs it produced.
        """
        for r, patch in results:
            self._buffer[int(r) // self._chunk][int(r)] = np.asarray(patch).reshape(self._arr.shape[1:])
        touched = set()
        for r in attempted:
            chunk_id = int(r) // self._chunk
            self._remaining[chunk_id] -= 1
            touched.add(chunk_id)
        for chunk_id in touched:
            if self._remaining[chunk_id] <= 0:
                self._flush_chunk(chunk_id)

    def flush(self) -> None:
        """Write every chunk still holding buffered rows."""
        for chunk_id in list(self._buffer):
            self._flush_chunk(chunk_id)

    def _flush_chunk(self, chunk_id: int) -> None:
        pending = self._buffer.pop(chunk_id, None)
        if not pending:
            return
        rows = np.array(sorted(pending), dtype=np.int64)
        self._arr.oindex[rows] = np.stack([pending[int(r)] for r in rows])
        self._mask.oindex[rows] = True


# --------------------------------------------------------------------------- #
# Full assembly: select -> fetch -> write (needs the download extra)
# --------------------------------------------------------------------------- #
def build_cell_store(
    path,
    *,
    products: list[str],
    lon: np.ndarray,
    lat: np.ndarray,
    cell_indices: np.ndarray | None = None,
    splits: dict[str, np.ndarray] | None = None,
    patch_size: int = PATCH_SIZE,
    resolution_m: float = RESOLUTION_M,
    year: int = CANONICAL_YEAR,
    fetch_kwargs: dict | None = None,
    max_workers: int = 1,
):
    """Build a compact store over an explicit list of cell centres.

    Unlike :func:`build_store`, which sizes the store to a seeded MajorTOM
    subset and populates a subset of its rows, this writes one row per given
    centre — the right shape when a product is fetched for a small, externally
    chosen set of cells. Split labels are supplied rather than recomputed, so a
    subset inherits the geographic blocking of the store it was drawn from.
    Needs the ``download`` extra.

    Args:
        path: Output store path.
        products: Registry product keys to fetch and write.
        lon: ``(N,)`` cell centre longitudes in degrees.
        lat: ``(N,)`` cell centre latitudes in degrees.
        cell_indices: ``(N,)`` provenance ids written to ``cell_indices``
            (e.g. the parent store's rows). Defaults to ``arange(N)``.
        splits: ``{name: (N,) int8}`` split labels written as ``split_<name>``.
        patch_size: Patch side length in pixels.
        resolution_m: Pixel size in metres.
        year: Acquisition year recorded in the store attributes.
        fetch_kwargs: Optional ``{product_key: {kwargs}}`` forwarded to
            :func:`~geosae.data.download.sources.build_fetcher`.
        max_workers: Concurrent tile fetches.

    Returns:
        The open :class:`zarr.Group`.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError(f"lon and lat must be matching 1-D arrays, got {lon.shape} and {lat.shape}")
    n_cells = int(lon.size)
    splits = splits or {}

    schemas = build_schemas(
        list(products), n_cells, embeddings=(), splits=tuple(f"split_{n}" for n in splits), patch_size=patch_size
    )
    group = create_store(path, schemas, n_cells=n_cells, patch_size=patch_size, year=year)

    rows = np.arange(n_cells, dtype=np.int64)
    group["cell_indices"][:] = rows if cell_indices is None else np.asarray(cell_indices, dtype=np.int64)
    group["lon"][:] = lon
    group["lat"][:] = lat
    group["x"][:] = np.arange(patch_size, dtype=np.int64)
    group["y"][:] = np.arange(patch_size, dtype=np.int64)
    for name, labels in splits.items():
        group[f"split_{name}"][:] = np.asarray(labels, dtype=np.int8)

    geoboxes = write_geoboxes(group, rows, lon, lat, patch_size, resolution_m)
    populate_products(group, products, rows, lon, lat, geoboxes, fetch_kwargs=fetch_kwargs, max_workers=max_workers)
    return group


def geobox_params(lon, lat, patch_size: int = PATCH_SIZE, resolution_m: float = RESOLUTION_M):
    """UTM ``crs`` and affine ``transform`` of every cell's patch, vectorised per zone.

    The same geometry :func:`utm_geobox` gives one cell at a time: a
    ``patch_size``-square patch of ``resolution_m`` pixels centred on the cell
    centre, in the cell's own UTM zone.

    Args:
        lon: ``(N,)`` cell centre longitudes, degrees.
        lat: ``(N,)`` cell centre latitudes, degrees.
        patch_size: Patch side length in pixels.
        resolution_m: Pixel size in metres.

    Returns:
        ``(crs, transform)``: ``(N,)`` ``EPSG:<code>`` strings and the ``(N, 6)``
        ``(a, b, c, d, e, f)`` affine parameters written to the store.
    """
    import rasterio.warp as rw

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    zone = ((lon + 180) / 6).astype(np.int64) + 1
    epsg = np.where(lat >= 0, 32600, 32700) + zone
    half = patch_size * resolution_m / 2.0
    transform = np.zeros((lon.size, 6), dtype=np.float64)
    transform[:, 0], transform[:, 4] = resolution_m, -resolution_m
    for code in np.unique(epsg):
        sel = np.flatnonzero(epsg == code)
        xs, ys = rw.transform("EPSG:4326", f"EPSG:{code}", lon[sel].tolist(), lat[sel].tolist())
        transform[sel, 2] = np.asarray(xs) - half
        transform[sel, 5] = np.asarray(ys) + half
    return np.array([f"EPSG:{c}" for c in epsg]), transform


def geobox_of(crs: str, transform, patch_size: int = PATCH_SIZE):
    """The :class:`~odc.geo.geobox.GeoBox` of one row from its stored ``crs`` and ``transform``."""
    from odc.geo.geobox import GeoBox
    from rasterio.transform import Affine

    return GeoBox((patch_size, patch_size), Affine(*np.asarray(transform, dtype=np.float64)), str(crs))


def open_or_create_store(
    path,
    schemas: dict[str, ArraySchema],
    *,
    cell_indices: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    patch_size: int = PATCH_SIZE,
    resolution_m: float = RESOLUTION_M,
    year: int = CANONICAL_YEAR,
):
    """Open a store to (continue to) fill, creating it and its cell geometry if new.

    A new store gets every array of ``schemas`` and the full cell axis at once:
    ``cell_indices``, ``lon``/``lat``, each row's UTM ``crs``/``transform`` and
    the pixel axes. An existing one must describe the same cells; arrays of
    ``schemas`` it lacks are added, so a later run can extend it with products.

    Args:
        path: Store path.
        schemas: ``{array: ArraySchema}`` the store must hold.
        cell_indices: ``(N,)`` base-grid index of every row.
        lon: ``(N,)`` cell centre longitudes.
        lat: ``(N,)`` cell centre latitudes.
        patch_size: Patch side length in pixels.
        resolution_m: Pixel size in metres.
        year: Acquisition year recorded in the store attributes.

    Returns:
        The open :class:`zarr.Group`.

    Raises:
        ValueError: If an existing store holds different cells.
    """
    import zarr

    n_cells = int(cell_indices.size)
    if (Path(path) / "zarr.json").exists():
        group = zarr.open_group(str(path), mode="r+", use_consolidated=False)
        same = int(group.attrs.get("n_cells", -1)) == n_cells and np.array_equal(
            np.asarray(group["cell_indices"][:]), cell_indices
        )
        if not same:
            raise ValueError(f"{path} holds different cells than this build; choose another path")
        for name, s in schemas.items():
            if name not in group:
                _create_array(group, name, s)
                if name.startswith("feature_"):
                    group[name][:] = np.arange(s.shape[0], dtype=np.int64)
        group.attrs["modalities"] = {**dict(group.attrs.get("modalities", {})), **_modalities_attr(schemas)}
        return group

    group = create_store(path, schemas, n_cells=n_cells, patch_size=patch_size, year=year)
    group["cell_indices"][:] = cell_indices
    group["lon"][:] = lon
    group["lat"][:] = lat
    group["x"][:] = np.arange(patch_size, dtype=np.int64)
    group["y"][:] = np.arange(patch_size, dtype=np.int64)
    for name in schemas:
        if name.startswith("feature_"):
            group[name][:] = np.arange(schemas[name].shape[0], dtype=np.int64)
    crs, transform = geobox_params(lon, lat, patch_size, resolution_m)
    group["crs"][:] = crs.astype(group["crs"].dtype)
    group["transform"][:] = transform
    return group


def fetch_product(
    group,
    key: str,
    rows: np.ndarray,
    *,
    array: str | None = None,
    block_rows: int = 25_600,
    max_workers: int = 8,
    fetch_kwargs: dict | None = None,
    log=print,
) -> dict:
    """Fetch one product for ``rows`` of a store, resumably, block by block.

    Rows whose ``mask_<array>`` is already set are skipped, so a re-run after an
    interruption fetches only what is missing. Each block of ``block_rows``
    store rows is written (values and mask) before the next starts, and a block
    every one of whose rows was requested is recorded in the array's
    ``fetched_blocks`` attribute: a row the source has no data for is then not
    asked for again either.

    Args:
        group: Open store group holding the product's array and mask.
        key: Registry product key.
        rows: Store rows to fetch.
        array: Destination array; defaults to the product's own.
        block_rows: Rows per block (a multiple of the arrays' chunks avoids
            rewriting a chunk twice).
        max_workers: Concurrent tile fetches.
        fetch_kwargs: Per-call fetcher overrides for this product.
        log: Progress sink, one line per block.

    Returns:
        ``{"array", "requested", "covered"}``: rows asked for and rows the
        array now holds among them.
    """
    from geosae.data.download.sources import get_product

    spec = get_product(key)
    array = array or spec.array
    patch_size = int(group.attrs["patch_size"])
    rows = np.asarray(rows, dtype=np.int64)
    mask = group[f"mask_{array}"]
    done_blocks = set(group[array].attrs.get("fetched_blocks", []))
    lon_all, lat_all = np.asarray(group["lon"][:]), np.asarray(group["lat"][:])
    crs_all, tf_all = np.asarray(group["crs"][:]), np.asarray(group["transform"][:])

    for block in np.unique(rows // block_rows):
        lo = int(block) * block_rows
        hi = min(lo + block_rows, int(group.attrs["n_cells"]))
        want = rows[(rows >= lo) & (rows < hi)]
        if int(block) in done_blocks:
            continue
        have = np.asarray(mask[lo:hi], dtype=bool)
        todo = want[~have[want - lo]]
        t0 = time.time()
        if todo.size:
            geoboxes = {int(r): geobox_of(crs_all[r], tf_all[r], patch_size) for r in todo}
            populate_products(
                group,
                [key],
                todo,
                lon_all[todo],
                lat_all[todo],
                geoboxes,
                fetch_kwargs={key: fetch_kwargs or {}},
                max_workers=max_workers,
                into={key: array},
            )
        covered = int(np.asarray(mask[lo:hi], dtype=bool)[want - lo].sum())
        if want.size == hi - lo:
            done_blocks.add(int(block))
            group[array].attrs["fetched_blocks"] = sorted(done_blocks)
        log(f"[geosae] {key} -> {array}: rows {lo}-{hi}: {covered}/{want.size} ({time.time() - t0:.0f} s)")
    covered = int(np.asarray(mask[:], dtype=bool)[rows].sum())
    return {"array": array, "requested": int(rows.size), "covered": covered}


def write_splits(group, splits: dict[str, SplitSpec], embeddings: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Write every ``split_<name>`` array from the rows that carry all ``embeddings``.

    Args:
        group: Open store group.
        splits: ``{name: SplitSpec}``.
        embeddings: Store arrays a row needs to be in a split.

    Returns:
        ``{split_name: {train, val, test, excluded}}`` counts.
    """
    from geosae.data.download.selection import split_counts

    n_cells = int(group.attrs["n_cells"])
    both = np.ones(n_cells, dtype=bool)
    for emb in embeddings:
        both &= np.asarray(group[f"mask_{emb}"][:], dtype=bool)
    counts = {}
    for name, spec in splits.items():
        _write_split(group, f"split_{name}", spec, n_cells, both)
        counts[f"split_{name}"] = split_counts(np.asarray(group[f"split_{name}"][:]))
    return counts


def write_norm_stats(group, stats: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Write per-dimension normalisation statistics as ``<prefix>_<stat>`` arrays.

    The layout :meth:`~geosae.data.dataset.Transforms.from_zarr_stats` reads.

    Args:
        group: Open store group.
        stats: ``{prefix: {"mean", "std", "clip_lo", "clip_hi": (D,) arrays}}``.

    Returns:
        The array names written.
    """
    written = []
    for prefix, values in stats.items():
        for name, value in values.items():
            value = np.asarray(value, dtype=np.float32)
            key = f"{prefix}_{name}"
            if key not in group:
                group.create_array(key, shape=value.shape, chunks=value.shape, dtype="float32", fill_value=0.0)
            group[key][:] = value
            written.append(key)
    return written


def build_store(
    path,
    *,
    products: list[str],
    cell_indices: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    splits: dict[str, SplitSpec] | None = None,
    reference_rows: tuple[int, int] | None = None,
    rows: np.ndarray | None = None,
    norm_stats: dict[str, dict[str, np.ndarray]] | None = None,
    patch_size: int = PATCH_SIZE,
    resolution_m: float = RESOLUTION_M,
    year: int = CANONICAL_YEAR,
    fetch_kwargs: dict | None = None,
    max_workers: int = 8,
    block_rows: int = 25_600,
    log=print,
):
    """Build (or resume building) a harmonized store over a fixed list of cells.

    Creates the store and its cell geometry on first call, then fetches every
    product in turn with :func:`fetch_product`, which skips what an earlier run
    already wrote. Embedding products fill every requested row; the other
    products only the requested rows inside ``reference_rows``. Splits are
    written last, from the rows that carry every embedding. Needs the
    ``download`` extra.

    Args:
        path: Output store path.
        products: Registry product keys, at most one per store array.
        cell_indices: ``(N,)`` base-grid index of every row.
        lon: ``(N,)`` cell centre longitudes.
        lat: ``(N,)`` cell centre latitudes.
        splits: ``{name: SplitSpec}`` written as ``split_<name>``.
        reference_rows: ``(start, end)`` rows the non-embedding products cover;
            ``None`` covers every row.
        rows: Rows to fetch in this run (``None``: all); a small set is a smoke
            test of every source.
        norm_stats: Normalisation statistics written by :func:`write_norm_stats`.
        patch_size: Patch side length in pixels.
        resolution_m: Pixel size in metres.
        year: Acquisition year recorded in the store attributes.
        fetch_kwargs: ``{product_key: {kwargs}}`` forwarded to the fetchers.
        max_workers: Concurrent tile fetches per product.
        block_rows: Rows written per block.
        log: Progress sink.

    Returns:
        ``{"products": {key: fetch summary}, "splits": {name: counts}}``.

    Raises:
        RuntimeError: A product needs credentials this machine lacks (see
            :func:`~geosae.data.download.plan.missing_credentials`).
        ValueError: Two products write the same array, or ``path`` holds other cells.
    """
    import zarr

    from geosae.data.download.plan import missing_credentials
    from geosae.data.download.sources import Kind, get_product

    specs = {k: get_product(k) for k in products}
    missing = missing_credentials(list(specs))
    if missing:
        lines = "\n".join(f"  {k}: {msg}" for k, msg in missing.items())
        raise RuntimeError(f"credentials missing for {len(missing)} product(s):\n{lines}")
    arrays = [s.array for s in specs.values()]
    duplicated = sorted({a for a in arrays if arrays.count(a) > 1})
    if duplicated:
        raise ValueError(f"several products write the same array: {duplicated}")
    embeddings = tuple(s.array for s in specs.values() if s.kind is Kind.EMBEDDING)
    labels = [k for k, s in specs.items() if s.kind is not Kind.EMBEDDING]
    splits = splits or {}
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    n_cells = int(cell_indices.size)

    schemas = build_schemas(
        labels, n_cells, embeddings=embeddings, splits=tuple(f"split_{n}" for n in splits), patch_size=patch_size
    )
    group = open_or_create_store(
        path,
        schemas,
        cell_indices=cell_indices,
        lon=np.asarray(lon, dtype=np.float64),
        lat=np.asarray(lat, dtype=np.float64),
        patch_size=patch_size,
        resolution_m=resolution_m,
        year=year,
    )
    for key, spec in specs.items():
        # which source filled an array, since several registry entries can fill the same one
        group[spec.array].attrs["product"] = key
        if spec.bands:
            group[spec.array].attrs["bands"] = list(spec.bands)

    requested = np.arange(n_cells, dtype=np.int64) if rows is None else np.unique(np.asarray(rows, dtype=np.int64))
    summary: dict = {"products": {}}
    for key, spec in specs.items():
        todo = requested
        if spec.kind is not Kind.EMBEDDING and reference_rows is not None:
            todo = requested[(requested >= reference_rows[0]) & (requested < reference_rows[1])]
        summary["products"][key] = fetch_product(
            group,
            key,
            todo,
            block_rows=block_rows,
            max_workers=max_workers,
            fetch_kwargs=(fetch_kwargs or {}).get(key),
            log=log,
        )
    if norm_stats:
        write_norm_stats(group, norm_stats)
    summary["splits"] = write_splits(group, splits, embeddings) if embeddings else {}
    zarr.consolidate_metadata(group.store)
    return summary


def build_preset_store(path, preset, *, products: list[str] | None = None, **kwargs):
    """Build (or resume) the store of a :class:`~geosae.data.download.presets.DatasetPreset`.

    Draws the preset's cells from the MajorTOM land grid
    (:func:`~geosae.data.download.grid.land_cell_centers`) and forwards them,
    its products, splits, reference rows and shipped statistics to
    :func:`build_store`.

    Args:
        path: Output store path.
        preset: The preset.
        products: Registry keys replacing the preset's own (e.g. a single product,
            or an alternative source for one array).
        **kwargs: Forwarded to :func:`build_store` (``rows``, ``max_workers``,
            ``fetch_kwargs``, ``block_rows``, ``log``, ...).

    Returns:
        The :func:`build_store` summary.
    """
    from geosae.data.download.grid import land_cell_centers

    centers = land_cell_centers()
    if len(centers) != preset.base_ncells:
        raise ValueError(f"the land grid has {len(centers)} cells, the preset expects {preset.base_ncells}")
    cell_indices = preset.cell_indices(len(centers))
    return build_store(
        path,
        products=list(products or preset.products),
        cell_indices=cell_indices,
        lon=centers[cell_indices, 0],
        lat=centers[cell_indices, 1],
        splits=preset.splits,
        reference_rows=preset.reference_rows,
        norm_stats=preset.load_norm_stats(),
        **kwargs,
    )


def build_majortom_store(
    path,
    cells,
    *,
    products: list[str],
    spacing_km: float = 5,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    split_name: str = "default",
    seed: int | None = 0,
    **kwargs,
):
    """Build (or resume) a store over chosen MajorTOM cells.

    Each cell becomes one row: a 64 x 64 patch of 10 m pixels around the cell
    centre, in its UTM zone. The rows are shuffled with ``seed`` and split by
    fraction into ``split_<split_name>`` (test rows first, then train, then
    validation; see :func:`~geosae.data.download.selection.fractional_split`).
    Choose the cells with :func:`~geosae.data.download.grid.sample_cells`,
    :func:`~geosae.data.download.grid.cells_in_bbox`,
    :func:`~geosae.data.download.grid.cells_at` or
    :func:`~geosae.data.download.grid.cells_from_ids`. Needs the ``download`` extra.

    Args:
        path: Output store path. Re-running with the same arguments resumes.
        cells: Flat MajorTOM grid indices, unique; written to ``cell_indices``.
        products: Registry product keys (see :func:`~geosae.data.download.sources.all_products`).
        spacing_km: Grid spacing the indices refer to.
        val_frac: Share of rows for validation.
        test_frac: Share of rows for test; the rest is train.
        split_name: Suffix of the split array.
        seed: Seed of the row shuffle; ``None`` keeps the given order, so the
            splits are contiguous runs of ``cells`` (e.g. spatial blocks of a
            bounding-box selection).
        **kwargs: Forwarded to :func:`build_store` (``rows``, ``max_workers``,
            ``fetch_kwargs``, ``block_rows``, ``log``, ...).

    Returns:
        The :func:`build_store` summary.

    Raises:
        ValueError: ``cells`` is empty or holds a cell twice.
    """
    from geosae.data.download.grid import _grid
    from geosae.data.download.selection import DEFAULT_CHUNK_SIZE, fractional_split

    cells = np.asarray(cells, dtype=np.int64).ravel()
    if cells.size == 0:
        raise ValueError("no cells to build a store from")
    if np.unique(cells).size != cells.size:
        raise ValueError("cells holds duplicates; pass np.unique(cells)")
    if seed is not None:
        cells = cells[np.random.default_rng(seed).permutation(cells.size)]
    centers = _grid(spacing_km).centers[cells]
    # Chunk-aligned ranges only once a store spans many chunks; a small one would lose whole splits.
    chunk = DEFAULT_CHUNK_SIZE if cells.size >= 20 * DEFAULT_CHUNK_SIZE else 1
    split = fractional_split(split_name, cells.size, val_frac=val_frac, test_frac=test_frac, chunk_size=chunk)
    return build_store(
        path,
        products=list(products),
        cell_indices=cells,
        lon=centers[:, 0],
        lat=centers[:, 1],
        splits={split_name: split},
        **kwargs,
    )


def _fetch_tile(fetcher, tile_key, cell_tuples, resampling, nodata=None):
    """Yield ``(row, patch)`` for a tile via the batch or single-patch API.

    ``nodata`` is the product's own fill value and has to reach the reprojection:
    a patch that straddles two source tiles has pixels this tile cannot cover, and
    without a fill they come back as whatever was in memory.
    """
    if hasattr(fetcher, "fetch_patches_for_tile"):
        res = fetcher.fetch_patches_for_tile(tile_key, cell_tuples, resampling, nodata)
        if res:
            yield from res
            return
    for r, lon, lat, gb in cell_tuples:  # single-patch fallback
        patch = fetcher.fetch_patch(lon, lat, gb, resampling, nodata)
        if patch is not None:
            yield r, patch


def _write_split(group, name, spec, n_cells, both_mask):
    split = compute_split(spec, n_cells, both_mask)
    group[name][:] = split
    has_all = spec.test is not None and spec.train is not None and spec.val is not None
    ranges = {
        "chunk_size": spec.chunk_size,
        "chunk_aligned": True,
        "strategy": "index_range_test_train_val" if has_all else "index_range",
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if spec.test is not None:
        ranges["test_end"] = spec.test[1]
    if spec.train is not None:
        ranges["train_end"] = spec.train[1]
    if spec.val is not None:
        ranges["val_end"] = spec.val[1]
    for k, v in ranges.items():
        group[name].attrs[k] = v


__all__ = [
    "ArraySchema",
    "EMBED_CHUNK_BYTES",
    "LABEL_CHUNK_BYTES",
    "COORD_CHUNK",
    "SPLIT_CHUNK",
    "CANONICAL_YEAR",
    "cell_chunk",
    "build_schemas",
    "embedding_schemas",
    "lonlat_schemas",
    "split_schema",
    "create_store",
    "utm_geobox",
    "geobox_params",
    "geobox_of",
    "open_or_create_store",
    "write_geoboxes",
    "add_product_array",
    "populate_products",
    "has_embedding",
    "fetch_product",
    "write_splits",
    "write_norm_stats",
    "build_cell_store",
    "build_store",
    "build_preset_store",
    "build_majortom_store",
]

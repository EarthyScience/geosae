"""Black-box tests for the harmonized store writer (numpy/zarr-only, CPU, synthetic).

Covers the layout contract: the chunk-byte-target rule, the per-array schema
(dtype/chunks/fill/dims/attrs), store creation, and round-trip interop with
:class:`geosae.data.dataset.PatchData`. No fetching.
"""

from __future__ import annotations

import numpy as np
import zarr

from geosae.data.download.store import (
    COORD_CHUNK,
    EMBED_CHUNK_BYTES,
    LABEL_CHUNK_BYTES,
    SPLIT_CHUNK,
    ArraySchema,
    _modalities_attr,
    build_schemas,
    cell_chunk,
    create_store,
)


def test_cell_chunk_byte_targets():
    # label 20 MiB target: uint8 64x64 -> 5120, float32 64x64 -> 1280, int16 -> 2560
    assert cell_chunk("uint8", (64, 64), LABEL_CHUNK_BYTES, 10**9) == 5120
    assert cell_chunk("float32", (64, 64), LABEL_CHUNK_BYTES, 10**9) == 1280
    assert cell_chunk("int16", (64, 64), LABEL_CHUNK_BYTES, 10**9) == 2560
    # multi-band: overture uint8x5 -> 1024, soilgrids f32x4 -> 320
    assert cell_chunk("uint8", (64, 64, 5), LABEL_CHUNK_BYTES, 10**9) == 1024
    assert cell_chunk("float32", (64, 64, 4), LABEL_CHUNK_BYTES, 10**9) == 320
    # embeddings 50 MiB: AlphaEarth 64d -> 50 cells, Tessera 128d -> 25
    assert cell_chunk("float32", (64, 64, 64), EMBED_CHUNK_BYTES, 10**9) == 50
    assert cell_chunk("float32", (64, 64, 128), EMBED_CHUNK_BYTES, 10**9) == 25
    # clamped to n_cells
    assert cell_chunk("uint8", (64, 64), LABEL_CHUNK_BYTES, 100) == 100


def test_build_schemas_embeddings_and_splits():
    sch = build_schemas(products=["lc_esa_wc", "gpp"], n_cells=1_000_000, splits=("split_210k",))
    ae = sch["alphaearth"]
    assert ae.shape == (1_000_000, 64, 64, 64)
    assert ae.chunks == (50, 64, 64, 64)
    assert ae.dtype == "float32"
    assert np.isnan(ae.fill_value)
    assert ae.dim_names == ("cell", "y", "x", "feature_alphaearth")
    assert ae.attrs["embedding_dim"] == 64
    te = sch["tessera"]
    assert te.chunks == (25, 64, 64, 128)
    # label + its mask
    assert sch["lc_esa_wc"].chunks == (5120, 64, 64)
    assert sch["lc_esa_wc"].fill_value == 255
    assert sch["gpp"].fill_value != sch["gpp"].fill_value  # NaN
    assert sch["mask_lc_esa_wc"].dtype == "bool"
    assert sch["mask_lc_esa_wc"].chunks == (COORD_CHUNK,)
    # split arrays: int8, no dims, chunk 100000
    assert sch["split_210k"].dtype == "int8"
    assert sch["split_210k"].dim_names is None
    assert sch["split_210k"].chunks == (SPLIT_CHUNK,)


def test_build_schemas_coords_present():
    sch = build_schemas(products=["lc_esa_wc"], n_cells=500)
    for name in ("cell_indices", "lon", "lat", "crs", "transform", "x", "y", "feature_alphaearth"):
        assert name in sch
    assert sch["cell_indices"].dtype == "int64"
    assert sch["cell_indices"].fill_value == np.iinfo(np.int64).min
    assert sch["transform"].shape == (500, 6)
    assert sch["crs"].dtype == "str"
    # n_cells < COORD_CHUNK -> chunk clamps to n_cells
    assert sch["lon"].chunks == (500,)


def test_create_store_roundtrip(tmp_path):
    sch = build_schemas(products=["lc_esa_wc", "overture"], n_cells=120)
    g = create_store(tmp_path / "s.zarr", sch, n_cells=120, patch_size=64, year=2024)
    assert int(g.attrs["n_cells"]) == 120
    assert int(g.attrs["patch_size"]) == 64
    assert int(g.attrs["year"]) == 2024
    assert "lc_esa_wc" in g.attrs["modalities"]  # label products listed
    assert "alphaearth" not in g.attrs["modalities"]  # embeddings excluded
    # reopen and assert schema materialised
    g2 = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
    a = g2["lc_esa_wc"]
    assert a.shape == (120, 64, 64)
    assert a.chunks == (120, 64, 64)  # clamped
    assert str(a.dtype) == "uint8"
    assert a.fill_value == 255
    assert list(a.attrs["_ARRAY_DIMENSIONS"]) == ["cell", "y", "x"]
    assert tuple(g2["overture"].shape) == (120, 64, 64, 5)
    assert list(g2["overture"].attrs["_ARRAY_DIMENSIONS"])[-1] == "overture_band"
    # embedding coordinates attr
    assert g2["alphaearth"].attrs["coordinates"] == "cell_indices feature_alphaearth lat lon x y"


def test_store_interop_with_patchdata(tmp_path):
    """A store written by create_store is readable by the training reader."""
    from geosae.data.dataset import PatchData

    n = 100
    sch = build_schemas(products=["lc_esa_wc"], n_cells=n, splits=("split_210k",))
    g = create_store(tmp_path / "s.zarr", sch, n_cells=n)
    rng = np.random.default_rng(0)
    # synthetic embedding + coords + mask + split
    g["alphaearth"][:] = rng.standard_normal((n, 64, 64, 64)).astype(np.float32)
    g["mask_alphaearth"][:] = True
    g["lon"][:] = np.linspace(-10, 10, n)
    g["lat"][:] = np.linspace(-5, 5, n)
    split = np.full(n, -1, np.int8)
    split[:40] = 0  # train
    split[40:] = 2  # test
    g["split_210k"][:] = split

    pd = PatchData(str(tmp_path / "s.zarr"), "alphaearth", "split_210k", "train", compute=True)
    assert len(pd) == 40
    batch = next(pd.iter_batches(8))
    assert tuple(batch["x"].shape) == (8, 64, 64, 64)
    assert int(pd.cells.max()) < 40  # only train rows selected


def test_arrayschema_is_plain_data():
    s = ArraySchema("uint8", (10, 64, 64), (10, 64, 64), 255, ("cell", "y", "x"))
    assert s.dtype == "uint8" and s.fill_value == 255


def test_chunk_writer_writes_each_chunk_once_and_only_successful_rows():
    """Rows arriving out of order must land correctly, with one write per chunk."""
    import numpy as np

    from geosae.data.download.store import _ChunkWriter

    class CountingArray:
        def __init__(self, shape, chunks):
            self.shape = shape
            self.chunks = chunks
            self.data = np.full(shape, -1, dtype=np.int64)
            self.writes = 0

        class _OIndex:
            def __init__(self, parent):
                self.parent = parent

            def __setitem__(self, rows, value):
                self.parent.writes += 1
                self.parent.data[rows] = value

        @property
        def oindex(self):
            return self._OIndex(self)

    arr = CountingArray((10, 2), (4, 2))
    mask = CountingArray((10,), (4,))
    rows = np.arange(10)
    writer = _ChunkWriter(arr, mask, rows)

    # Feed rows shuffled across chunks; row 5 is attempted but never returned.
    writer.add([9, 0], [(9, np.array([9, 9])), (0, np.array([0, 0]))])
    writer.add([1, 2, 3], [(r, np.array([r, r])) for r in (1, 2, 3)])
    writer.add([4, 5, 6, 7, 8], [(r, np.array([r, r])) for r in (4, 6, 7, 8)])
    writer.flush()

    for r in (0, 1, 2, 3, 4, 6, 7, 8, 9):
        np.testing.assert_array_equal(arr.data[r], [r, r])
    np.testing.assert_array_equal(arr.data[5], [-1, -1])  # the failed row stays unwritten
    assert arr.writes == 3  # one per chunk: rows 0-3, 4-7, 8-9


def test_default_products_exclude_every_registered_embedding():
    """Registering an embedding must not add it as a label array by default."""
    from geosae.data.download.sources import Kind, all_products

    embedding_arrays = {s.array for s in all_products().values() if s.kind is Kind.EMBEDDING}
    schemas = build_schemas(None, 1000, embeddings=("alphaearth",))
    # alphaearth is present as an embedding; no other embedding appears at all.
    assert schemas["alphaearth"].attrs["embedding_dim"] == 64
    for array in embedding_arrays - {"alphaearth"}:
        assert array not in schemas, f"{array} leaked into the default label products"


def test_default_products_do_not_depend_on_the_embeddings_argument():
    """The reference-product set is the same whichever embeddings are requested."""

    def labels(d):
        return {k for k, v in d.items() if v.dim_names and len(v.dim_names) >= 3 and "embedding_dim" not in v.attrs}

    a = build_schemas(None, 1000, embeddings=("alphaearth",))
    b = build_schemas(None, 1000, embeddings=("alphaearth", "tessera"))
    assert labels(a) == labels(b)


def test_build_store_geometry_is_consistent_at_a_non_default_patch_size():
    """`patch_size` must reach the array shapes, not only the root attr.

    ``build_store`` fetches each cell on a ``patch_size``-square geobox, so a
    schema still sized 64x64 would leave the arrays unwritable.
    """
    ps = 16
    schemas = build_schemas(["lc_esa_wc", "soilgrids"], 8, embeddings=("alphaearth",), patch_size=ps)
    assert schemas["alphaearth"].shape == (8, ps, ps, 64)
    assert schemas["lc_esa_wc"].shape == (8, ps, ps)
    assert schemas["soilgrids"].shape == (8, ps, ps, 4)
    assert schemas["x"].shape == (ps,) and schemas["y"].shape == (ps,)
    assert _modalities_attr(schemas)["lc_esa_wc"]["shape"] == [ps, ps]

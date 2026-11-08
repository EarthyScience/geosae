"""Preregistered dataset presets built on the general selection/sources/store APIs.

:mod:`~geosae.data.download.selection`, :mod:`~geosae.data.download.sources`
and :mod:`~geosae.data.download.store` are general-purpose building blocks for
selecting cells, registering products and writing a harmonized zarr store for
*any* dataset built on the MajorTOM grid. This module bundles concrete choices
of those building blocks into named defaults:

* ``majortom210k`` (:data:`MAJORTOM210K`) -- 210,000 MajorTOM cells with
  AlphaEarth and Tessera embeddings, the reference products the concept metrics
  score against, and the covariates and reference partitions the area-of-
  applicability and sub-concept analyses read. Built by
  :func:`~geosae.data.download.store.build_preset_store` (``geosae download``).
* ``copernicusfm025`` (:func:`build_copernicusfm025_store`) -- the reference
  catalog of a patch store coarsened onto the 0.25 degree grid of the published
  Copernicus-FM embedding map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np

from geosae.data.download.selection import DEFAULT_CHUNK_SIZE, SplitSpec, fractional_split, sample_subset_indices

#: Number of cells in the base MajorTOM (5 km) land grid the preset draws from.
MAJORTOM210K_BASE_NCELLS = 5_418_682
#: Random seed of the preset's cell draw.
MAJORTOM210K_SEED = 42
#: Size of the seeded draw the store's rows are the leading part of. The draw's
#: order depends on its size, so the same cells need the same draw size.
MAJORTOM210K_DRAW_SIZE = 1_000_000
#: Rows of the store: the leading rows of the draw.
MAJORTOM210K_N_CELLS = 210_000

#: 210k split: test=0..100k, train=100k..200k, val=200k..210k.
SPLIT_210K = SplitSpec(
    name="210k", test=(0, 100_000), train=(100_000, 200_000), val=(200_000, 210_000), chunk_size=DEFAULT_CHUNK_SIZE
)
#: The Sentinel-2 subset the patch-token embeddings (OlmoEarth, MMEarth) are
#: computed on: the first 10k test, 10k train and 2k validation rows of
#: ``split_210k``, i.e. the same rows minus those ``split_210k`` excludes.
SPLIT_10K = SplitSpec(
    name="10k", test=(0, 10_000), train=(100_000, 110_000), val=(200_000, 202_000), chunk_size=DEFAULT_CHUNK_SIZE
)

#: Registry keys of the ``majortom210k`` products, one per store array. Earth
#: Engine sources are used where the public file would be a download of
#: hundreds of gigabytes for a few cells per tile (GLC_FCS30D, WorldCereal):
#: the same maps, and the same credentials AlphaEarth already needs.
MAJORTOM210K_PRODUCTS: tuple[str, ...] = (
    # embeddings
    "alphaearth",
    "tessera",
    # categorical reference products of the concept registry
    "overture",
    "lc_esa_wc",
    "lc_esa_cci",
    "lc_esri",
    "glc_fcs30d_gee",
    "glance30",
    "lc_modis_igbp",
    "worldcereal_gee",
    "geomorphons",
    "soilgrids_class",
    "lcz",
    # continuous reference products and covariates
    "chmv2",
    "dem_copernicus",
    "gpp",
    "soilgrids",
    "chelsa_bioclim",
    # reference partitions of the sub-concept analysis
    "chelsa_kg",
    "resolve_biomes",
    "resolve_ecoregions",
    "pnv",
)


@dataclass(frozen=True)
class DatasetPreset:
    """A named, reusable bundle of selection + product + split choices.

    Attributes:
        name: Preset name.
        base_ncells: Size of the base grid the selection draws from.
        seed: Selection seed (see :func:`~geosae.data.download.selection.sample_subset_indices`).
        draw_size: Size of the seeded draw.
        n_cells: Store rows: the leading ``n_cells`` rows of the draw.
        products: Registry product keys to fetch, at most one per store array.
        splits: ``{name: SplitSpec}`` of split arrays to write.
        embeddings: Store arrays a row must carry to be in a split.
        reference_rows: ``(start, end)`` rows the non-embedding products are
            fetched for (``None``: every row).
        norm_stats: ``{prefix: file}`` per-dimension normalisation statistics
            shipped with the package and written as ``<prefix>_{mean,std,clip_lo,clip_hi}``.
    """

    name: str
    base_ncells: int
    seed: int
    draw_size: int
    n_cells: int
    products: tuple[str, ...]
    splits: dict[str, SplitSpec]
    embeddings: tuple[str, ...] = ("alphaearth", "tessera")
    reference_rows: tuple[int, int] | None = None
    norm_stats: dict[str, str] = field(default_factory=dict)

    def cell_indices(self, total: int | None = None) -> np.ndarray:
        """Base-grid index of every store row.

        Args:
            total: Size of the base grid; defaults to :attr:`base_ncells`.

        Returns:
            ``int64`` array of length :attr:`n_cells`: the leading rows of the
            seeded ``draw_size`` draw.
        """
        total = self.base_ncells if total is None else int(total)
        return sample_subset_indices(total, self.draw_size, self.seed)[: self.n_cells]

    def load_norm_stats(self) -> dict[str, dict[str, np.ndarray]]:
        """The shipped normalisation statistics, ``{prefix: {mean, std, clip_lo, clip_hi}}``."""
        out = {}
        for prefix, name in self.norm_stats.items():
            with files("geosae.data.download").joinpath(name).open("rb") as fh:
                npz = np.load(fh)
                out[prefix] = {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}
        return out


MAJORTOM210K = DatasetPreset(
    name="majortom210k",
    base_ncells=MAJORTOM210K_BASE_NCELLS,
    seed=MAJORTOM210K_SEED,
    draw_size=MAJORTOM210K_DRAW_SIZE,
    n_cells=MAJORTOM210K_N_CELLS,
    products=MAJORTOM210K_PRODUCTS,
    splits={"210k": SPLIT_210K, "10k": SPLIT_10K},
    reference_rows=SPLIT_210K.test,
    norm_stats={"tessera_norm": "majortom210k_tessera_norm.npz"},
)
"""The preregistered ``majortom210k`` dataset.

Its reference products cover the test rows, which is where every concept metric
is scored. The Tessera statistics were estimated from 10,000 training cells of
the same seeded draw.
"""

#: Presets by name, as the ``geosae download`` config names them.
PRESETS: dict[str, DatasetPreset] = {MAJORTOM210K.name: MAJORTOM210K}


#: Grid resolution of the ``copernicusfm025`` preset, in degrees.
COPERNICUSFM025_RESOLUTION_DEG = 0.25
#: Seed for the preset's row scatter.
COPERNICUSFM025_SEED = 42
#: Split shares. Training gets the larger share because the coarse grid holds
#: only ~240k vectors in total -- four orders of magnitude fewer than the
#: pixel-wise products, and the SAE has to fit a 768-wide dictionary from them.
COPERNICUSFM025_TEST_FRAC = 0.25
COPERNICUSFM025_VAL_FRAC = 0.05


def build_copernicusfm025_store(
    src_path,
    dst_path,
    *,
    cache_dir,
    resolution_deg: float = COPERNICUSFM025_RESOLUTION_DEG,
    seed: int = COPERNICUSFM025_SEED,
    products: list[str] | None = None,
    n_workers: int = 1,
    progress: bool = False,
    **fetcher_kwargs,
) -> dict:
    """Build the ``copernicusfm025`` store: coarsen a patch store, then attach.

    Aggregates the reference products of ``src_path`` onto the global
    ``resolution_deg`` grid (majority class / nanmean per node) and fills the
    ``copernicusfm`` embedding array from the published grid map, keeping only
    nodes that have both. Needs ``huggingface_hub``.

    Args:
        src_path: Source 64x64 patch store.
        dst_path: Output coarse store.
        cache_dir: Directory holding (or receiving) the downloaded map.
        resolution_deg: Grid resolution.
        seed: Row-scatter seed.
        products: Source arrays to aggregate; ``None`` takes all registered ones.
        n_workers: Products aggregated concurrently (the pass is IO-bound).
        progress: Print per-product progress.
        **fetcher_kwargs: Forwarded to the embedding fetcher.

    Returns:
        The coarsening summary, with ``n_embedded`` added.
    """
    from geosae.data.download.fetchers.copernicus_fm import add_copernicusfm_embeddings
    from geosae.data.download.gridded import RegularGrid, coarsen_patch_store

    grid = RegularGrid.global_grid(resolution_deg)
    n_nodes = _count_occupied_nodes(src_path, grid)
    split = fractional_split(
        "copernicusfm025", n_nodes, val_frac=COPERNICUSFM025_VAL_FRAC, test_frac=COPERNICUSFM025_TEST_FRAC
    )
    summary = coarsen_patch_store(
        src_path,
        dst_path,
        grid=grid,
        products=products,
        embeddings=("copernicusfm",),
        splits={"copernicusfm025": split},
        seed=seed,
        n_workers=n_workers,
        progress=progress,
    )
    summary["n_embedded"] = add_copernicusfm_embeddings(
        dst_path, cache_dir=cache_dir, resolution_deg=resolution_deg, **fetcher_kwargs
    )
    return summary


def _count_occupied_nodes(src_path, grid) -> int:
    """Number of grid nodes holding at least one patch of the source store."""
    import zarr

    src = zarr.open_group(str(src_path), mode="r")
    flat = grid.flat_index(np.asarray(src["lon"][:]), np.asarray(src["lat"][:]))
    return int(np.unique(flat).size)


__all__ = [
    "MAJORTOM210K_BASE_NCELLS",
    "MAJORTOM210K_SEED",
    "MAJORTOM210K_DRAW_SIZE",
    "MAJORTOM210K_N_CELLS",
    "MAJORTOM210K_PRODUCTS",
    "SPLIT_210K",
    "SPLIT_10K",
    "DatasetPreset",
    "MAJORTOM210K",
    "PRESETS",
    "COPERNICUSFM025_RESOLUTION_DEG",
    "COPERNICUSFM025_SEED",
    "COPERNICUSFM025_TEST_FRAC",
    "COPERNICUSFM025_VAL_FRAC",
    "build_copernicusfm025_store",
]

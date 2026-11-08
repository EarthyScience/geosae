"""MajorTOM dataset download + assembly pipeline for geosae.

General, reusable building blocks for turning a subset of the MajorTOM grid
into a training-ready zarr dataset, plus preregistered defaults built from them:

* :mod:`geosae.data.download.grid` -- the MajorTOM grid and choosing cells:
  a random sample, a bounding box, points or MajorTOM ids.
* :mod:`geosae.data.download.selection` -- pure-numpy core: seeded cell
  selection and chunk-aligned train/val/test split assignment. Dependency-light
  (only :mod:`numpy`) so it is testable on CPU without any geospatial stack.
* :mod:`geosae.data.download.sources` -- the config-driven product registry:
  every Earth-embedding and reference product, its store-array name, schema,
  data source and auth requirements. Fetcher construction lives behind the
  ``download`` extra (heavy geospatial deps: rasterio, pyproj, shapely, ...).
* :mod:`geosae.data.download.store` -- the harmonized zarr writer that closes
  the select -> fetch -> assemble pipeline, writing an IO-optimized store and
  resuming an interrupted build. The layout/schema core is numpy/zarr-only;
  :func:`~geosae.data.download.store.build_store` needs the ``download`` extra.
* :mod:`geosae.data.download.presets` -- named defaults built on the above:
  the ``majortom210k`` dataset (:data:`~geosae.data.download.presets.MAJORTOM210K`,
  built by :func:`~geosae.data.download.store.build_preset_store` or
  ``geosae download``) and the ``copernicusfm025`` coarse store.
* :mod:`geosae.data.download.plan` -- what a build will fetch: sources,
  credentials, licences and sizes, without fetching anything.
* :mod:`geosae.data.download.release` -- the dataset's recipe as plain tables:
  the exact MajorTOM cell ids and their splits, the product registry with each
  product's licence, and the concept registry the metrics score against.

The pure-numpy selection + schema API is importable with the core install;
anything that fetches rasters requires ``pip install 'geosae[download]'``.
"""

from __future__ import annotations

from geosae.data.download.grid import (
    MajorTomGrid,
    cells_at,
    cells_from_ids,
    cells_in_bbox,
    sample_cells,
    select_cells,
)
from geosae.data.download.plan import build_plan, check_credentials, missing_credentials
from geosae.data.download.presets import (
    MAJORTOM210K,
    PRESETS,
    DatasetPreset,
)
from geosae.data.download.release import (
    cell_records,
    concept_records,
    majortom_cell_ids,
    pack_release_store,
    product_records,
    redistributable_arrays,
    write_release_tables,
)
from geosae.data.download.selection import (
    DEFAULT_CHUNK_SIZE,
    SplitSpec,
    compute_split,
    sample_subset_indices,
    split_counts,
)
from geosae.data.download.sources import (
    Auth,
    Kind,
    ProductSpec,
    Redistribution,
    all_products,
    build_fetcher,
    get_product,
)
from geosae.data.download.store import (
    ArraySchema,
    build_majortom_store,
    build_preset_store,
    build_schemas,
    build_store,
    cell_chunk,
    create_store,
)

__all__ = [
    "MajorTomGrid",
    "cells_at",
    "cells_from_ids",
    "cells_in_bbox",
    "sample_cells",
    "select_cells",
    "build_plan",
    "check_credentials",
    "missing_credentials",
    "build_majortom_store",
    "ArraySchema",
    "build_schemas",
    "build_preset_store",
    "build_store",
    "cell_chunk",
    "create_store",
    "DEFAULT_CHUNK_SIZE",
    "SplitSpec",
    "compute_split",
    "sample_subset_indices",
    "split_counts",
    "Auth",
    "Kind",
    "ProductSpec",
    "Redistribution",
    "all_products",
    "build_fetcher",
    "get_product",
    "MAJORTOM210K",
    "PRESETS",
    "DatasetPreset",
    "cell_records",
    "concept_records",
    "majortom_cell_ids",
    "pack_release_store",
    "product_records",
    "redistributable_arrays",
    "write_release_tables",
]

# Building a dataset

A patch store is a Zarr group with one row per cell of the
[MajorTOM](https://github.com/ESA-PhiLab/Major-TOM) 5 km grid: a 64 x 64 patch of 10 m pixels
(640 m) around the cell centre, in the cell's own UTM zone, carrying the embeddings and reference
products you choose. [`build_majortom_store()`][geosae.data.download.store.build_majortom_store]
and `geosae download` build one from the original sources; nothing is downloaded pre-built.

## Installing the fetch stack

Choosing cells by id or point, the product registry and build plans import with the core install.
Fetching, and the land test of a box or a random sample, need the `download` extra:

```bash
pip install "geosae[download]"
```

## Choosing cells

A store holds whichever cells you pass it, as flat indices into the MajorTOM grid. Four functions
choose them. [`cells_from_ids()`][geosae.data.download.grid.cells_from_ids] takes MajorTOM cell ids,
and [`cells_at()`][geosae.data.download.grid.cells_at] the cell containing each `(lon, lat)` point:

```python
from geosae.data.download import cells_at, cells_from_ids

print(cells_from_ids(["645U_1077R", "1071U_170R"]))
print(cells_at([55.3116, 11.4665], [28.9895, 48.1213]))
```

```
[15140986 17798802]
[15140986 17798802]
```

[`cells_in_bbox()`][geosae.data.download.grid.cells_in_bbox] takes every cell whose centre lies in a
`(west, south, east, north)` box, and
[`sample_cells()`][geosae.data.download.grid.sample_cells] draws a seeded random sample, inside a
box or over the whole globe. Both keep land cells only (at least 10% on Natural Earth land, lakes
removed); `land_fraction=None` keeps sea cells in a box:

```python
from geosae.data.download import cells_in_bbox, sample_cells

box = (11.0, 50.5, 12.0, 51.2)          # west, south, east, north
print(cells_in_bbox(box).size, cells_in_bbox(box, land_fraction=None).size)
print(sample_cells(5, seed=0, bbox=box))
```

```
225 225
[18141769 18126630 18116517 18096232 18096241]
```

The box around Jena is all land, so both counts agree. The land test of a box takes about ten
seconds and downloads the Natural Earth layers once (into `$GEOSAE_CACHE_DIR`, default
`~/.cache/geosae`). A global sample (`bbox=None`) tests every cell of the globe once, which takes
about four minutes and 12 GB of memory, and caches the 5,418,682 land cells for every later call.

[`select_cells()`][geosae.data.download.grid.select_cells] dispatches to the four by keyword
(`n`, `bbox`, `points` or `ids`); it is what `geosae download` calls with its `download.cells` block.

### The MajorTOM grid

[`MajorTomGrid`][geosae.data.download.grid.MajorTomGrid] is the grid of near-equal-area cells. Cells
are numbered row by row, south to north and west to east; `centers` and `bounds` hold every cell's
`(lon, lat)` centre and `(west, south, east, north)` box by that flat index, and `cell_id` /
`parse_cell_id` convert between a row and column and the id the MajorTOM release names a cell by:

```python
from geosae.data.download.grid import MajorTomGrid

grid = MajorTomGrid(5)
row, col = grid.parse_cell_id("1071U_170R")
print(len(grid), grid.centers[grid.flat_index(row, col)], grid.bounds[grid.flat_index(row, col)])
```

```
20374696 [11.4664674  48.12125749] [11.4328414  48.0988024  11.50009341 48.14371257]
```

An id is `<row><U|D>_<col><L|R>`, counted from the equator row and from the prime-meridian column
*of that row*. A row's column count shrinks towards the poles, so the same column index names a
different cell in every row.

## Choosing products

Every array of a store is filled by one [`ProductSpec`][geosae.data.download.sources.ProductSpec]
of the registry, which records the array it writes, its kind (`embedding`, `categorical` or
`continuous`), the resampling onto the cell's patch (`nearest` for categorical products, `bilinear`
for continuous ones), its source, its credentials and its licence:

```python
from collections import Counter
from geosae.data.download import all_products

specs = all_products()
print(len(specs), Counter(s.kind.value for s in specs.values()))
for key in ["alphaearth", "alphaearth_cog", "tessera", "lc_esa_wc", "glance30"]:
    s = specs[key]
    print(f"{key:<15} {s.array:<11} {s.kind.value:<12} {s.auth.value:<10} {s.licence}")
```

```
39 Counter({'categorical': 23, 'continuous': 12, 'embedding': 4})
alphaearth      alphaearth  embedding    gee        CC-BY-4.0
alphaearth_cog  alphaearth  embedding    public     CC-BY-4.0
tessera         tessera     embedding    public     CC0-1.0
lc_esa_wc       lc_esa_wc   categorical  public     CC-BY-4.0
glance30        glance30    categorical  earthdata  CC-BY-4.0 (NASA EOSDIS open data)
```

Several keys can fill the same array from different sources: `alphaearth` reads Earth Engine and
`alphaearth_cog` a public COG mirror of the same AlphaEarth 2024 map, whose tiling can differ from
Earth Engine's grid by a pixel. Pass at most one per array. The registry, by role:

| Role | Registry keys |
|---|---|
| Embeddings (2024) | `alphaearth` / `alphaearth_cog` (AlphaEarth Foundations, `D = 64`), `tessera` (Tessera v1.0, `D = 128`) |
| Land cover | `lc_esa_wc` (ESA WorldCover 2021), `lc_esri` / `lc_esri_gee` (Esri), `lc_esa_cci` (ESA CCI 2020), `glc_fcs30d` / `glc_fcs30d_gee` (GLC_FCS30D 2022), `glance30` (GLanCE 2019), `lc_modis_igbp` (MODIS IGBP 2021), `lcz` (local climate zones) |
| Agriculture, forest, water, settlements | `worldcereal` / `worldcereal_gee`, `worldcereal_types`, `usda_cdl`, `eucropmap`, `gpw_grassland_class`, `gpp`, `natural_forest`, `hansen_lossyear`, `hansen_treecover2000`, `chmv2` (canopy height), `jrc_gsw`, `ghsl_built_s`, `overture` (water, roads, land use, infrastructure) |
| Terrain, soil, climate | `dem_copernicus`, `dem_gedtm30`, `geomorphons`, `soilgrids`, `soilgrids_gee`, `soilgrids_class`, `chelsa_bioclim` (bio1-19), `chelsa_kg` (Köppen-Geiger) |
| Ecological partitions | `resolve_biomes`, `resolve_ecoregions`, `pnv` (potential natural vegetation) |
| Imagery and coarse embeddings | `sentinel2_l2a`, `copernicusfm` (see [Further embeddings](#further-embeddings)) |

The categorical products that the concept metrics score against are listed in
[Concepts and factors](concepts.md).

!!! note
    Overture keeps only its latest two monthly releases online, and the fetcher reads the latest
    one, so the `overture` array of a rebuild differs from an earlier build by a few months of map
    edits. Pass `-o download.fetch_kwargs.overture.release=<YYYY-MM-DD.N>` to pin a release that is
    still published.

Tiled sources (Copernicus DEM, CHMv2) are read from every tile a 640 m patch straddles and
composited, so a patch on a tile edge has no seam. RESOLVE is published as polygons; each cell's
pixels take the value of the polygon their centre falls in.

### Credentials

Most products are anonymous HTTP or S3 reads. Two credential classes need a one-time setup:

| Class | Products | Setup |
|---|---|---|
| `gee` | `alphaearth`, every `*_gee` key, `worldcereal_types`, `hansen_*`, `jrc_gsw`, `ghsl_built_s`, `usda_cdl`, `eucropmap` | A Google Earth Engine project: `earthengine authenticate`, then `-o download.gee_project=<project>` (or `$GEOSAE_GEE_PROJECT`) |
| `earthdata` | `glance30` | A NASA Earthdata Login in `~/.netrc` (`machine urs.earthdata.nasa.gov login <user> password <password>`) |

A build checks them before it fetches anything and stops with the products it cannot fetch and how
to set them up; [`missing_credentials()`][geosae.data.download.plan.missing_credentials] runs the
same check. `alphaearth_cog` is the credential-free way to AlphaEarth.

## Planning a build

Plan mode lists every product of a build, its credentials, its licence, the rows it fills and the
size of its store array, and checks which credentials are configured. It fetches nothing:

```bash
geosae download -c download -o download.plan=true \
    -o "download.cells.bbox=[11.0,50.5,12.0,51.2]" -o "download.products=[alphaearth_cog,lc_esa_wc]"
```

```
[geosae] 225 cells selected
product              array                auth           rows  store GB  cache GB  licence
------------------------------------------------------------------------------------------
alphaearth_cog       alphaearth           public          225       0.2         -  CC-BY-4.0 (attribution)
lc_esa_wc            lc_esa_wc            public          225       0.0         -  CC-BY-4.0 (attribution)
------------------------------------------------------------------------------------------
total (uncompressed)                                      0.2      0.00

credentials public     ok
```

`store GB` is uncompressed: an AlphaEarth patch takes 1 MB in memory and about 0.3 MB compressed
on disk, a Tessera patch twice that, a categorical patch 4 kB. `cache GB` is a one-time source
download kept in `download.cache_dir` (default `data/raster_cache`).

## Building the store

`geosae download` takes the cells in its `download.cells` block (the keyword arguments of
`select_cells`, plus `seed`), the products in `download.products`, and the store path in
`download.out`:

```bash
geosae download -c download -o download.out=data/quickstart.zarr \
    -o "download.cells.bbox=[11.0,50.5,12.0,51.2]" -o "download.products=[alphaearth_cog,lc_esa_wc]"
```

```
[geosae] 225 cells selected
[geosae] alphaearth_cog -> alphaearth: rows 0-225: 221/225 (149 s)
[geosae] lc_esa_wc -> lc_esa_wc: rows 0-225: 225/225 (36 s)
[geosae] alphaearth_cog -> alphaearth: 221/225 rows
[geosae] lc_esa_wc -> lc_esa_wc: 225/225 rows
[geosae] split_default: {'train': 156, 'val': 23, 'test': 42, 'excluded': 4}
[geosae] data/quickstart.zarr done in 3.1 min
```

Each product prints one line per block of `download.block_rows` rows. Four cells at the edge of the
AlphaEarth tiles got no embedding and are excluded from the split. In Python,
[`build_majortom_store()`][geosae.data.download.store.build_majortom_store] is the same build:

```python
from geosae.data.download import build_majortom_store, cells_in_bbox

cells = cells_in_bbox((11.0, 50.5, 12.0, 51.2))
summary = build_majortom_store("data/quickstart.zarr", cells, products=["alphaearth_cog", "lc_esa_wc"])
print(summary["splits"])
```

```
{'split_default': {'train': 156, 'val': 23, 'test': 42, 'excluded': 4}}
```

Re-running the same command resumes: every product skips the rows it already holds, and a block of
rows the source had no data for is not asked for again. A store refuses to resume for a different
set of cells.

!!! tip
    Check every source on a few cells before a large build: `-o download.rows=3` fetches the first
    three rows only, and the full build later resumes into the same store.

### Adding products to a store

Running the build again with more keys in `download.products` adds their arrays to the existing
store and fetches only them; the cells and splits stay as they are. Pass the same cells (and seed)
as the first build.

### Build time

The fetchers are network-bound and run `download.max_workers` (default 8) tile fetches
concurrently. Cells that share a source tile share its reads, so a compact region is much faster
per cell than the same number of cells scattered over a continent. Measured one fetch at a time on
a few cells spread over the globe:

| Products | Time per cell |
|---|---|
| AlphaEarth (Earth Engine) | 2.7 s |
| Tessera (one 127 MB tile of 0.1 degree per cell, deleted after use) | 12 s |
| GLC_FCS30D, WorldCereal (Earth Engine) | 1 s |
| WorldCover, Esri, CCI, CHMv2, DEM, GPP, geomorphons, LCZ, SoilGrids, PNV, Köppen, MODIS | 0.2-15 s |
| GLanCE (Earthdata, one 25 MB granule per 150 km tile) | 7 s |
| Overture (DuckDB queries of the GeoParquet release on S3) | 30 s |
| RESOLVE biomes / ecoregions | 60 s per block to load the polygons, then milliseconds |

The AlphaEarth COG mirror stores 64 bands as separate 10 km tiles, so a scattered cell costs about
64 tile reads while a dense region shares them: the 225 cells of the box above take about
three minutes. Tessera fetches a tile of its own for nearly every scattered cell.

!!! note
    `geotessera` does not use the `https_proxy` environment variable. Behind an HTTP proxy, Tessera
    tiles fail to download with "Network is unreachable" while every other source works.

## Reading a store

A store holds, per row, `cell_indices` (the flat MajorTOM index), `lon`/`lat` (the cell centre),
`crs`/`transform` (the patch's UTM grid), one array per product with a `mask_<array>` of the rows it
covers, and one `split_<name>` array per split. Read it with
[`PatchData`](api.md#geosae.data.dataset.PatchData), the loader every stage uses:

```python
from geosae.data.dataset import PatchData

train = PatchData("data/quickstart.zarr", "alphaearth", split="train")
print(len(train), train.patch_size, train.embedding_dim, train.split_name)
```

```
[geosae] dropping 5/156 cells holding a flat pixel
151 64 64 split_default
```

`PatchData` drops cells whose embedding holds a fill pixel (see
[`geosae.data.quality`](api.md#geosae.data.quality)), so 151 of the 156 training rows remain.

### Splits

A `split_<name>` array labels every row `0` (train), `1` (validation), `2` (test) or `-1`
(excluded); a row without every embedding of the store is excluded from every split.
`build_majortom_store` shuffles its rows with `seed` and writes `split_default` from fractions
(`val_frac=0.1`, `test_frac=0.2`), test rows first, then train, then validation. The shuffle puts
neighbouring cells into different splits; `seed=None` keeps the order of `cells` instead, so the
splits are contiguous runs of it. Set the name and fractions with the `download.split` block.

A [`SplitSpec`][geosae.data.download.selection.SplitSpec] defines a split by explicit row ranges
and [`fractional_split()`][geosae.data.download.selection.fractional_split] by fractions;
[`build_store()`][geosae.data.download.store.build_store] writes any number of them over any
cells:

```python
from geosae.data.download.selection import fractional_split

split = fractional_split("default", n_cells=1000, val_frac=0.05, test_frac=0.25)
print(split.test, split.train, split.val)
```

```
(0, 250) (250, 950) (950, 1000)
```

Ranges are rounded down to whole chunks of 50 rows, so the last partial chunk is excluded rather
than assigned.

### Presets

`download.preset` builds a named, fixed store instead of chosen cells when `download.cells` is
null. The one preset, `majortom210k`, is the store of the
[NeurIPS 2026 paper](neurips2026/reproduce-data.md#data), which describes it.

## Further embeddings

OlmoEarth and MMEarth are computed from Sentinel-2 imagery rather than published as maps.
`geosae subset` copies the rows of one split of a store (`subset.split_name`, optionally capped per
label with `subset.max_per_split`), fetches Sentinel-2 L2A for `subset.year` (one scene under
`subset.max_cloud_cover`% cloud per cell, 1.28 km around the centre), encodes it with each backbone
and writes their standardisation statistics. Each stage is a separate job:

```bash
geosae subset -c download -o subset.stage=fetch   # CPU, network-bound
geosae subset -c download -o subset.stage=embed   # one GPU
geosae subset -c download -o subset.stage=stats
```

A cell keeps its token embeddings only when a cloud-free scene covers the whole patch. The subset
store's `cell_indices` are rows of the parent store, which is how its concept labels are read; see
[Evaluating further embeddings](evaluation.md#evaluating-further-embeddings).

Copernicus-FM is published as one vector per node of a 0.25 degree grid. `geosae gridded`
aggregates the store's reference products onto that grid and attaches the published map
(downloaded once, about 3 GB, from the Hugging Face dataset `wangyi111/Copernicus-Embed-025deg`).
Its cells are the grid nodes holding at least one patch of the store it is built from, so its size
follows the patch store's:

```bash
geosae gridded -c download
```

See [Coarse gridded stores](#coarse-gridded-stores) for how the coarse store is aggregated.

## Coarse gridded stores

The 64x64 patch store fits any product reprojected onto a cell's native-UTM
10 m geobox, but some embedding products are published the other way round —
as one vector per node of a coarse global lon/lat grid, coarser than a
64x64 patch. A 0.25 degree grid cell is about 28 km on a side, so there is no
64x64 px slice of it to take: the natural store for such a product has
`patch_size = 1`, one embedding vector per cell (`H = W = 1`).
`geosae.data.download.gridded` builds that geometry by **coarsening** an
existing patch store rather than fetching from scratch, so the reference
catalog stays defined identically across resolutions — only the spatial
support of each label changes, from a 640 m patch to a grid cell.

### The grid

`RegularGrid` is a node-registered lon/lat grid in the ERA5 convention: node
`(row, col)` sits at `lat = 90 - row * resolution_deg`,
`lon = -180 + col * resolution_deg`.

```python
from geosae.data.download.gridded import RegularGrid

grid = RegularGrid.global_grid(0.25)
lon, lat = grid.centers([0, 1440])
print(grid.shape, grid.n_cells)
print(lon, lat)
```

```
(721, 1440) 1038240
[-180. -180.] [90.   89.75]
```

At 0.25 degree resolution this is the 721 x 1440 ERA5 grid — the same grid the
Copernicus-FM embedding map is published on, so a fetch into it is a
nearest-node lookup rather than a reprojection.

### Aggregating the reference catalog onto nodes

`coarsen_patch_store` reads an existing patch store, assigns every source
patch to its nearest grid node, and reduces each reference product over all
pixels of all patches sharing a node: majority class for categorical
products, mean for continuous ones.

`aggregate_categorical` takes the majority vote, ignoring `fill_value`
(nodata) pixels, and reports the **modal fraction** — the share of valid
pixels that agreed with the winning class, i.e. how pure the coarse cell's
label is:

```python
import numpy as np
from geosae.data.download.gridded import aggregate_categorical

# 3 source patches of 2 pixels each; patches 0-1 fall in grid cell 0, patch 2 in cell 1.
codes = np.array([[10, 10], [10, 20], [30, 255]], dtype=np.uint8)  # 255 = nodata fill
group = np.array([0, 0, 1])
mode, modal_frac, n_valid = aggregate_categorical(codes, group, n_groups=2, fill_value=255)
print(mode, modal_frac, n_valid)
```

```
[10 30] [0.75 1.  ] [4 1]
```

Cell 0 pools 4 valid pixels (`10, 10, 10, 20`); class `10` wins with a modal
fraction of `0.75` — three quarters of the pooled pixels agreed. Cell 1 has a
single valid pixel after the nodata fill is dropped, so its modal fraction is
`1.0` but rests on very little support; `n_valid` is what lets a downstream
reader tell the two cases apart.
`aggregate_continuous` does the continuous counterpart, a plain nanmean per
group.

### The Copernicus-FM store

`build_copernicusfm025_store` bundles the two steps into one preset for the
published Copernicus-FM 0.25 degree map: it coarsens a source patch store's
reference catalog onto the 0.25 degree global grid with
`coarsen_patch_store`, then fills the `copernicusfm` embedding array by
nearest-node lookup into the map (`CopernicusFMGridEmbeddings`, from the
Hugging Face dataset `wangyi111/Copernicus-Embed-025deg`) with
`add_copernicusfm_embeddings`, keeping only nodes that end up with both a
reference label and a published embedding. The call is
`build_copernicusfm025_store(src_path, dst_path, cache_dir=...)`, and returns
the coarsening summary (`n_cells`, per-product modal-fraction / valid-cell
stats) with `n_embedded` — the number of cells that received a Copernicus-FM
vector — added. It needs the `download` extra (for `huggingface_hub`) and
downloads the ~3.2 GB embedding raster into `cache_dir` on first use;
`geosae gridded` runs it from the `gridded` block of `configs/download.yaml`.

## Publishing a dataset

`geosae release` exports the recipe a reader rebuilds the dataset from: which
MajorTOM cells were drawn, which split each landed in, how every reference
product was fetched, and which concept classes the metrics score against.

```bash
geosae release -c release -o release.out_dir=data/release
```

```
[geosae] cells: data/release/cells.csv
[geosae] products: data/release/products.csv
[geosae] concepts: data/release/concepts.csv
[geosae] manifest: data/release/manifest.json
```

`cells.csv` names every cell by its MajorTOM grid id, so the same cells can be
pulled from the upstream release without our store:

```
row,majortom_cell,majortom_row,majortom_col,lon,lat,crs,split,has_alphaearth,has_tessera
0,645U_1077R,2537,4583,55.311564237843996,28.98952095808383,EPSG:32640,test,True,True
```

An id is `<row><U|D>_<col><L|R>`, counted from the equator row and from the
prime-meridian column *of that row*. A row's column count shrinks towards the
poles, so the same column index names a different cell in every row;
`MajorTomGrid.cell_id` is the single place an id is formed.

### Licences and what may be republished

Every `ProductSpec` records the licence its data is published under and what
that licence lets us pass on:

```python
from geosae.data.download.release import product_records

rec = {r["key"]: r for r in product_records()}["lc_esa_wc"]
print(rec["licence"], rec["redistribution"], rec["resampling"])
```

```
CC-BY-4.0 attribution nearest
```

`redistribution` is one of `open`, `attribution`, `share_alike` or `restricted`.
The last two are the ones that decide the shape of a public release:
share-alike layers (the OpenStreetMap-derived Overture themes, ODbL) would
relicense whatever is published with them, and a restricted licence grants use
but not redistribution.

`redistributable_arrays` applies that to a store, and `pack_release_store`
copies what may travel into a fresh store:

```python
from geosae.data.download.release import redistributable_arrays

packable, excluded = redistributable_arrays("data/majortom210k.zarr")
print(excluded["overture"])
```

```
ODbL-1.0 (OpenStreetMap-derived themes) (share_alike)
```

!!! note
    An array with no registry entry is never packed, because nothing records
    where it came from or under what terms. Every array `geosae download` or
    `geosae refetch` writes carries a `product` attribute naming its registry
    key; a store written without it needs the mapping passed as `array_products`.

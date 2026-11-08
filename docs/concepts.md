# Concepts and factors

The concept vocabulary is what a decomposition is scored against: **categorical concepts** — the
classes of land-cover, land-use, climate, biome and terrain reference products — and **continuous
factors** — temperature, elevation, canopy height, productivity, soil. Both are stored per pixel in
the patch store, so scoring a run needs no extra data. See [Evaluation](evaluation.md) for the
metrics that consume them.

## Categorical classes

`geosae.eval.landcover_classes.PRODUCT_CLASSES` maps every categorical product in the store to
its `{code: name}` lookup. Code 0 is background/nodata and is dropped.

```python
from geosae.eval.landcover_classes import PRODUCT_CLASSES

n_classes = sum(len(v) for v in PRODUCT_CLASSES.values())
print(len(PRODUCT_CLASSES), "products,", n_classes, "classes")
print(sorted(PRODUCT_CLASSES)[:5])
print(PRODUCT_CLASSES["lc_esa_wc"][10])
```

```
22 products, 355 classes
['chelsa_kg', 'geomorphons', 'ghs_built_c', 'ghs_smod', 'glance30']
Tree cover
```

These 355 individual classes are the finest targets available;
[`build_targets()`](api.md#geosae.eval.concepts.build_targets) lists them, one target per
(product, class), and concept alignment scores every feature against each of them.

## Concept groups

Individual classes are product-specific and heavily redundant — "forest" appears under a
different code in each product. `geosae.eval.concepts.CONCEPT_GROUPS` OR-s them into
20 hand-curated cross-product groups, and `CLASS_TO_GROUP` inverts the mapping.

```python
from geosae.eval.concepts import CLASS_TO_GROUP, CONCEPT_GROUPS, HEADLINE_GROUPS

print(len(CONCEPT_GROUPS), "groups over", len(CLASS_TO_GROUP), "classes")
print(list(CONCEPT_GROUPS))
print(HEADLINE_GROUPS)
print(CONCEPT_GROUPS["water"][:4])
```

```
20 groups over 235 classes
['forest', 'shrubland', 'grassland', 'cropland', 'built_up', 'water', 'wetland', 'snow_ice', 'bare', 'savanna', 'moss_lichen', 'road', 'infrastructure', 'recreation', 'flat', 'summit', 'ridge', 'slope', 'valley', 'soil']
['forest', 'grassland', 'cropland', 'built_up', 'water', 'snow_ice', 'bare']
[('lc_esa_wc', 80), ('lc_esri', 1), ('lc_modis_igbp', 17), ('lc_esa_cci', 210)]
```

A pixel is in a group if *any* member class matches, so a feature aligned to forest across four
products counts as aligned to one concept rather than four. `HEADLINE_GROUPS` is a seven-group
subset of broad land-cover groups, the default of
[`concept_auroc()`](api.md#geosae.eval.auroc_moran.concept_auroc).

!!! note

    The grouped classes are a subset of the 355 individual ones: classes with no cross-product
    equivalent (Köppen-Geiger zones, biomes) belong to no group. Group counts therefore cover
    fewer classes than per-class scores do.

## Continuous factors

`geosae.eval.covariates.CONTINUOUS_COVARIATES` is the registry of continuous per-pixel factors:
the 19 CHELSA V2.1 bioclim bands, four SoilGrids bands, elevation, canopy height, GPP and soil
organic carbon stock.

```python
from geosae.eval.covariates import CONTINUOUS_COVARIATES, DEFAULT_COVARIATES

print(len(CONTINUOUS_COVARIATES), "registered;", len(DEFAULT_COVARIATES), "in the default panel")
print(DEFAULT_COVARIATES)
```

```
27 registered; 8 in the default panel
('mat', 'temp_seasonality', 'annual_precip', 'precip_seasonality', 'elevation', 'canopy_height', 'gpp', 'soil_soc')
```

`DEFAULT_COVARIATES` keeps one representative per physical axis. The 19 bioclim bands are
heavily colinear, so only four survive — two thermal (level, seasonality) and two hydrological —
which keeps an analysis from being dominated by nineteen restatements of temperature. The area of
applicability evaluation reads its regression targets and climate coordinates from this registry.

Each entry is a `Covariate` record carrying everything needed to read it out of the store.

```python
print(CONTINUOUS_COVARIATES["canopy_height"])
```

```
Covariate(name='canopy_height', key='chmv2', band=None, unit='m', long_name='Meta CHMv2 canopy height', zero_is_nodata=False)
```

`key` is the array name in the data zarr and `band` indexes into it for multi-band products;
`band=None` means the array is single-band.

## Selecting factors

Use [`resolve_covariates()`](api.md#geosae.eval.covariates.resolve_covariates) to turn a
selector into `Covariate` objects. It accepts `None` (the curated default set), registry names,
or `Covariate` objects, so an ad-hoc factor can be injected without touching the registry.

```python
from geosae.eval.covariates import resolve_covariates

for c in resolve_covariates(["mat", "elevation", "soil_soc"]):
    print(f"{c.name:14s} {c.key:15s} band={c.band}  [{c.unit}]  {c.long_name}")
```

```
mat            chelsa_bioclim  band=0  [degC]  CHELSA mean annual air temperature
elevation      dem_cop30       band=None  [m]  Copernicus DEM elevation
soil_soc       soilgrids       band=2  [g/kg]  SoilGrids soil organic carbon
```

Registry names are the metric-output keys, not the product names — the temperature factor is
`mat`, not `temperature`. An unknown name raises rather than being silently skipped.

```python
try:
    resolve_covariates(["temperature"])
except KeyError as err:
    print(str(err)[:76] + " ...")
```

```
"unknown covariate 'temperature'; known: ['annual_precip', 'canopy_height',  ...
```

## Reading factor values

[`read_covariate_stack()`](api.md#geosae.eval.covariates.read_covariate_stack) reads a
contiguous span of cells and returns `(b, P, T)` float32 with pixels flattened. Nodata is
normalised to `NaN` on read, from three separate sources: stored `NaN`s pass through, whole
patches whose `mask_<key>` flag is false become all-`NaN`, and exact zeros become `NaN` for
covariates declaring `zero_is_nodata`.

```python
import numpy as np
from geosae.eval.covariates import read_covariate_stack, resolve_covariates

store = {"chmv2": np.array([[[0.0, 12.0], [3.5, 0.0]], [[7.0, 8.0], [9.0, 10.0]]], dtype="float32"),
         "mask_chmv2": np.array([True, False])}
print(read_covariate_stack(store, resolve_covariates(["canopy_height"]), 0, 2, np.array([0, 1]))[..., 0])
```

```
[[ 0.  12.   3.5  0. ]
 [ nan  nan  nan  nan]]
```

The second patch is masked out entirely; the first keeps its zeros, because `canopy_height` does
not declare `zero_is_nodata` — 0 m means "no canopy", which is a legitimate value here.

!!! warning

    `global_wind_atlas` is registered as a product but is empty in the store
    (`mask_global_wind_atlas` is all-false), so it is not offered as a covariate: it would be an
    all-`NaN` factor that every metric silently skips.

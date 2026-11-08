"""Continuous geophysical covariates: the registry and reader of the store's continuous products.

Reference concepts are discrete land-cover classes; many geographic factors are
**continuous** (how warm is it, how tall is the canopy, how high are we). They serve
as downstream regression targets (:mod:`geosae.eval.aoa`) and as climate
coordinates for graded hold-outs (:mod:`geosae.eval.holdouts`). Everything this
module exposes is present per pixel in the patch store:

- **CHELSA V2.1 bioclim** (19 bands) — mean annual air temperature, seasonality,
  annual precipitation, ... A climate gradient the dictionary can silently
  stratify by if not accounted for.
- **`dem_cop30`** — elevation (Copernicus GLO-30, composited across its 1° tiles;
  see :mod:`geosae.data.download.fetchers.raster_fetchers`).
- **`chmv2`** — Meta canopy height (10 m).
- **`gpp`**, **`soc`**, **`soilgrids`** — productivity and soil properties.

Nodata is normalised to ``NaN`` on read: the float rasters store ``NaN``
directly, whole patches are gated by the store's ``mask_<key>`` flag, and a
covariate may additionally declare ``zero_is_nodata`` (canopy height 0 m means
"no canopy", which is a legitimate value for us but not for every use).

``global_wind_atlas`` is a registered product but labels no cell of the reference
store (``mask_global_wind_atlas`` is all-false), so it is not offered here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# CHELSA V2.1 bioclim band order (bio1..bio19), with units.
_BIOCLIM: tuple[tuple[str, str, str], ...] = (
    ("mat", "degC", "mean annual air temperature"),
    ("mean_diurnal_range", "degC", "mean diurnal air temperature range"),
    ("isothermality", "1", "isothermality (diurnal range / annual range) x 100"),
    ("temp_seasonality", "degC", "air temperature seasonality (sd x 100)"),
    ("tmax_warmest_month", "degC", "max temperature of the warmest month"),
    ("tmin_coldest_month", "degC", "min temperature of the coldest month"),
    ("temp_annual_range", "degC", "annual air temperature range"),
    ("temp_wettest_quarter", "degC", "mean temperature of the wettest quarter"),
    ("temp_driest_quarter", "degC", "mean temperature of the driest quarter"),
    ("temp_warmest_quarter", "degC", "mean temperature of the warmest quarter"),
    ("temp_coldest_quarter", "degC", "mean temperature of the coldest quarter"),
    ("annual_precip", "mm/yr", "annual precipitation"),
    ("precip_wettest_month", "mm", "precipitation of the wettest month"),
    ("precip_driest_month", "mm", "precipitation of the driest month"),
    ("precip_seasonality", "1", "precipitation seasonality (CV x 100)"),
    ("precip_wettest_quarter", "mm", "precipitation of the wettest quarter"),
    ("precip_driest_quarter", "mm", "precipitation of the driest quarter"),
    ("precip_warmest_quarter", "mm", "precipitation of the warmest quarter"),
    ("precip_coldest_quarter", "mm", "precipitation of the coldest quarter"),
)

# SoilGrids band order in the store.
_SOILGRIDS: tuple[tuple[str, str, str], ...] = (
    ("soil_sand", "%", "SoilGrids sand fraction"),
    ("soil_clay", "%", "SoilGrids clay fraction"),
    ("soil_soc", "g/kg", "SoilGrids soil organic carbon"),
    ("soil_ph", "1", "SoilGrids pH (H2O)"),
)


@dataclass(frozen=True)
class Covariate:
    """One continuous per-pixel factor.

    Attributes:
        name: Registry key, also the metric-output key.
        key: Array name in the data zarr.
        band: Band index for multi-band arrays, else ``None``.
        unit: Physical unit of the values.
        long_name: Human-readable description.
        zero_is_nodata: Treat exact zeros as missing (e.g. "no canopy").
    """

    name: str
    key: str
    band: int | None
    unit: str
    long_name: str
    zero_is_nodata: bool = False


def _registry() -> dict[str, Covariate]:
    reg: dict[str, Covariate] = {}
    for i, (name, unit, long_name) in enumerate(_BIOCLIM):
        reg[name] = Covariate(name, "chelsa_bioclim", i, unit, f"CHELSA {long_name}")
    for i, (name, unit, long_name) in enumerate(_SOILGRIDS):
        reg[name] = Covariate(name, "soilgrids", i, unit, long_name)
    reg["elevation"] = Covariate("elevation", "dem_cop30", None, "m", "Copernicus DEM elevation")
    reg["canopy_height"] = Covariate("canopy_height", "chmv2", None, "m", "Meta CHMv2 canopy height")
    reg["gpp"] = Covariate("gpp", "gpp", None, "gC/m2/yr", "gross primary productivity")
    reg["soc_stock"] = Covariate("soc_stock", "soc", None, "kg/m2", "soil organic carbon stock")
    return reg


CONTINUOUS_COVARIATES: dict[str, Covariate] = _registry()

# Curated default: one representative per physical axis, deliberately small.
# The 19 bioclim bands are heavily colinear, so only four are kept — two thermal
# (level, seasonality) and two hydrological (level, seasonality).
DEFAULT_COVARIATES: tuple[str, ...] = (
    "mat",
    "temp_seasonality",
    "annual_precip",
    "precip_seasonality",
    "elevation",
    "canopy_height",
    "gpp",
    "soil_soc",
)


def resolve_covariates(names: Iterable[str] | Iterable[Covariate] | None) -> list[Covariate]:
    """Resolve a covariate selector to :class:`Covariate` objects.

    Accepts ``None`` (the curated :data:`DEFAULT_COVARIATES`), an iterable of
    registry names, or an iterable of :class:`Covariate` objects (pass-through,
    so callers may inject an ad-hoc covariate).

    Raises:
        KeyError: If a name is not in :data:`CONTINUOUS_COVARIATES`.
    """
    if names is None:
        names = DEFAULT_COVARIATES
    out: list[Covariate] = []
    for n in names:
        if isinstance(n, Covariate):
            out.append(n)
        elif n in CONTINUOUS_COVARIATES:
            out.append(CONTINUOUS_COVARIATES[n])
        else:
            raise KeyError(f"unknown covariate {n!r}; known: {sorted(CONTINUOUS_COVARIATES)}")
    return out


def available_covariates(data: object, covariates: Sequence[Covariate]) -> list[Covariate]:
    """Those covariates a store actually carries, in the given order.

    A store need not hold every registered covariate: a coarsened store
    (:func:`~geosae.data.download.gridded.coarsen_patch_store`) carries only the
    arrays it was built from, so the curated default panel would otherwise fail
    on the first missing array. Callers that take covariates as an *explicit*
    target list (area of applicability) do not filter: a named target that is
    not there is an error, not a smaller panel.

    Args:
        data: Open zarr group of the data store.
        covariates: Covariates to filter.

    Returns:
        The subset whose :attr:`Covariate.key` array is present in ``data``.
    """
    return [c for c in covariates if c.key in data]


def read_covariate_stack(
    data: object,
    covariates: Sequence[Covariate],
    lo: int,
    hi: int,
    sel: np.ndarray,
) -> np.ndarray:
    """Read covariate values for a batch of cells as ``(b, P, T)`` float32.

    Follows the contiguous-span read used throughout :mod:`geosae.eval.concepts`:
    the caller slices ``data[key][lo:hi]`` once and re-indexes with ``sel``
    (positions of the wanted cells *within* the span), which keeps zarr reads
    chunk-aligned.

    Args:
        data: Open zarr group of the data store.
        covariates: Covariates to read, in output order.
        lo: Inclusive start of the contiguous cell span.
        hi: Exclusive end of the contiguous cell span.
        sel: ``(b,)`` positions within ``[lo, hi)`` of the wanted cells.

    Returns:
        ``(b, P, T)`` float32 with ``P = patch_size**2`` pixels flattened.
        Missing values are ``NaN``: stored NaNs pass through, cells whose
        ``mask_<key>`` flag is false are all-NaN, and exact zeros become NaN for
        covariates declaring ``zero_is_nodata``.
    """
    sel = np.asarray(sel)
    out: list[np.ndarray] = []
    for cov in covariates:
        arr = np.asarray(data[cov.key][lo:hi])[sel]
        if cov.band is not None:
            arr = arr[..., cov.band]
        vals = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=True)
        mask_key = "mask_" + cov.key
        if mask_key in data:
            valid = np.asarray(data[mask_key][lo:hi])[sel].astype(bool)
            vals[~valid] = np.nan
        if cov.zero_is_nodata:
            vals[vals == 0] = np.nan
        out.append(vals)
    return np.stack(out, axis=-1)


__all__ = [
    "Covariate",
    "CONTINUOUS_COVARIATES",
    "DEFAULT_COVARIATES",
    "available_covariates",
    "resolve_covariates",
    "read_covariate_stack",
]

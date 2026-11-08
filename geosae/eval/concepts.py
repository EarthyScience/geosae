"""The concept vocabulary: reference-product classes and the concept groups they form.

A *target* is one class of one reference product (:func:`build_targets`), read from the store's
categorical product arrays (:data:`~geosae.eval.landcover_classes.PRODUCT_CLASSES`). The products
overlap strongly — several have their own water, forest or built-up class — so :data:`CONCEPT_GROUPS`
groups the classes of different products that denote the same coarse geographic concept into 20
hand-curated concept groups. A feature then aligns with a concept rather than with one product's
version of it (:func:`geosae.eval.coverage.n_groups_per_feature`).
"""

from __future__ import annotations

from collections.abc import Sequence

from geosae.eval.landcover_classes import PRODUCT_CLASSES

# --------------------------------------------------------------- concept groups
# Each concept maps to a list of (product, class_code); a pixel is "in" the
# concept if any member matches. ``overture_*`` products are bands of the
# ``overture`` array (land_use=1, road_class=2, infrastructure=3, water_natural=4).
CONCEPT_GROUPS: dict[str, list[tuple[str, int]]] = {
    "forest": [
        ("lc_esa_wc", 10),
        ("lc_esri", 2),
        ("lc_modis_igbp", 1),
        ("lc_modis_igbp", 2),
        ("lc_modis_igbp", 3),
        ("lc_modis_igbp", 4),
        ("lc_modis_igbp", 5),
        ("lc_esa_cci", 50),
        ("lc_esa_cci", 60),
        ("lc_esa_cci", 61),
        ("lc_esa_cci", 62),
        ("lc_esa_cci", 70),
        ("lc_esa_cci", 71),
        ("lc_esa_cci", 72),
        ("lc_esa_cci", 80),
        ("lc_esa_cci", 81),
        ("lc_esa_cci", 82),
        ("lc_esa_cci", 90),
        ("overture_water_natural", 7),
        ("overture_land_use", 16),
        ("lcz", 11),
        ("glance30", 5),
        ("glc_fcs30d", 51),
        ("glc_fcs30d", 52),
        ("glc_fcs30d", 61),
        ("glc_fcs30d", 62),
        ("glc_fcs30d", 71),
        ("glc_fcs30d", 72),
        ("glc_fcs30d", 81),
        ("glc_fcs30d", 82),
        ("glc_fcs30d", 91),
        ("glc_fcs30d", 92),
    ],
    "shrubland": [
        ("lc_esa_wc", 20),
        ("lc_esri", 6),
        ("lc_modis_igbp", 6),
        ("lc_modis_igbp", 7),
        ("lc_esa_cci", 120),
        ("lc_esa_cci", 121),
        ("lc_esa_cci", 122),
        ("overture_water_natural", 11),
        ("lcz", 13),
        ("glance30", 6),
        ("glc_fcs30d", 120),
        ("glc_fcs30d", 121),
        ("glc_fcs30d", 122),
    ],
    "grassland": [
        ("lc_esa_wc", 30),
        ("lc_esri", 3),
        ("lc_modis_igbp", 10),
        ("lc_esa_cci", 130),
        ("overture_water_natural", 12),
        ("overture_land_use", 17),
        ("overture_land_use", 18),
        ("lcz", 14),
        ("glance30", 7),
        ("glc_fcs30d", 130),
    ],
    "cropland": [
        ("lc_esa_wc", 40),
        ("lc_esri", 5),
        ("lc_modis_igbp", 12),
        ("lc_modis_igbp", 14),
        ("lc_esa_cci", 10),
        ("lc_esa_cci", 11),
        ("lc_esa_cci", 12),
        ("lc_esa_cci", 20),
        ("lc_esa_cci", 30),
        ("overture_land_use", 1),
        ("overture_land_use", 8),
        ("overture_land_use", 9),
        ("overture_land_use", 19),
        ("overture_land_use", 26),
        ("worldcereal", 100),
        ("glc_fcs30d", 10),
        ("glc_fcs30d", 11),
        ("glc_fcs30d", 12),
        ("glc_fcs30d", 20),
    ],
    "built_up": [
        ("lc_esa_wc", 50),
        ("lc_esri", 7),
        ("lc_modis_igbp", 13),
        ("lc_esa_cci", 190),
        ("overture_land_use", 2),
        ("overture_land_use", 3),
        ("overture_land_use", 4),
        ("overture_land_use", 10),
        ("overture_land_use", 14),
        ("overture_land_use", 20),
        ("overture_land_use", 21),
        ("overture_land_use", 22),
        ("lcz", 1),
        ("lcz", 2),
        ("lcz", 3),
        ("lcz", 4),
        ("lcz", 5),
        ("lcz", 6),
        ("lcz", 7),
        ("lcz", 8),
        ("lcz", 9),
        ("lcz", 10),
        ("glance30", 3),
        ("glc_fcs30d", 190),
    ],
    "water": [
        ("lc_esa_wc", 80),
        ("lc_esri", 1),
        ("lc_modis_igbp", 17),
        ("lc_esa_cci", 210),
        ("overture_water_natural", 1),
        ("overture_water_natural", 2),
        ("overture_water_natural", 3),
        ("overture_water_natural", 4),
        ("overture_water_natural", 5),
        ("overture_land_use", 24),
        ("overture_land_use", 25),
        ("lcz", 17),
        ("glance30", 1),
        ("glc_fcs30d", 210),
    ],
    "wetland": [
        ("lc_esa_wc", 90),
        ("lc_esa_wc", 95),
        ("lc_esri", 4),
        ("lc_modis_igbp", 11),
        ("lc_esa_cci", 160),
        ("lc_esa_cci", 170),
        ("lc_esa_cci", 180),
        ("overture_water_natural", 6),
        ("glc_fcs30d", 181),
        ("glc_fcs30d", 182),
        ("glc_fcs30d", 183),
        ("glc_fcs30d", 184),
        ("glc_fcs30d", 185),
        ("glc_fcs30d", 186),
        ("glc_fcs30d", 187),
    ],
    "snow_ice": [
        ("lc_esa_wc", 70),
        ("lc_esri", 9),
        ("lc_modis_igbp", 15),
        ("lc_esa_cci", 220),
        ("overture_water_natural", 8),
        ("overture_land_use", 27),
        ("glance30", 2),
        ("glc_fcs30d", 220),
    ],
    "bare": [
        ("lc_esa_wc", 60),
        ("lc_esri", 8),
        ("lc_modis_igbp", 16),
        ("lc_esa_cci", 150),
        ("lc_esa_cci", 151),
        ("lc_esa_cci", 152),
        ("lc_esa_cci", 153),
        ("lc_esa_cci", 200),
        ("lc_esa_cci", 201),
        ("lc_esa_cci", 202),
        ("overture_water_natural", 9),
        ("overture_water_natural", 10),
        ("overture_land_use", 7),
        ("overture_land_use", 15),
        ("lcz", 15),
        ("lcz", 16),
        ("glance30", 4),
        ("glc_fcs30d", 150),
        ("glc_fcs30d", 152),
        ("glc_fcs30d", 153),
        ("glc_fcs30d", 200),
        ("glc_fcs30d", 201),
        ("glc_fcs30d", 202),
    ],
    "savanna": [
        ("lc_modis_igbp", 8),
        ("lc_modis_igbp", 9),
        ("lc_esa_cci", 100),
        ("lc_esa_cci", 110),
        ("lc_esa_cci", 40),
        ("lcz", 12),
    ],
    "moss_lichen": [
        ("lc_esa_wc", 100),
        ("lc_esa_cci", 140),
        ("glc_fcs30d", 140),
    ],
    "road": [
        ("overture_road_class", 1),
        ("overture_road_class", 2),
        ("overture_road_class", 3),
        ("overture_road_class", 4),
        ("overture_road_class", 5),
        ("overture_road_class", 6),
        ("overture_road_class", 7),
        ("overture_road_class", 8),
        ("overture_land_use", 23),
    ],
    "infrastructure": [("overture_infrastructure", i) for i in range(1, 16)],
    "recreation": [
        ("overture_land_use", 6),
        ("overture_land_use", 11),
        ("overture_land_use", 12),
        ("overture_land_use", 13),
    ],
    "flat": [("geomorphons", 1)],
    "summit": [("geomorphons", 2)],
    "ridge": [("geomorphons", 3), ("geomorphons", 5)],
    "slope": [("geomorphons", 4), ("geomorphons", 6)],
    "valley": [("geomorphons", 7), ("geomorphons", 8), ("geomorphons", 9), ("geomorphons", 10)],
    "soil": [("soilgrids_class", i) for i in range(1, 31)],
}

# The seven headline land-cover concept groups.
HEADLINE_GROUPS = ["forest", "grassland", "cropland", "built_up", "water", "snow_ice", "bare"]

# product name -> (zarr array key, band index or None), nodata = 0 for all.
_OVERTURE_BANDS = {
    "overture_land_use": 1,
    "overture_road_class": 2,
    "overture_infrastructure": 3,
    "overture_water_natural": 4,
}


def _product_source(product: str) -> tuple[str, int | None]:
    if product in _OVERTURE_BANDS:
        return "overture", _OVERTURE_BANDS[product]
    return product, None


# (product, class_code) -> concept group name.
CLASS_TO_GROUP: dict[tuple[str, int], str] = {key: g for g, members in CONCEPT_GROUPS.items() for key in members}

# Products excluded by default from the per-class targets (one class per region, far too many to score).
_DEFAULT_EXCLUDE = {"resolve_ecoregions"}

#: Products that label too little of the store to be scored, and the fraction of
#: the test split each of them covers. A coverage count is a count *out of* the
#: scorable concepts, so a product that can never contribute one only inflates
#: that denominator — and a product covering ~1% of the cells contributes
#: concepts measured on a small, geographically clustered subsample while every
#: other product's are measured on the whole pool, which is not the same
#: measurement. Both are excluded so the denominator means one thing.
UNCOVERED_PRODUCTS: dict[str, float] = {
    "global_wind_atlas": 0.0,
    "globeland30": 0.0,
    "glc_fcs30d": 0.0069,
    "ghs_built_c": 0.0159,
}


def build_targets(products: Sequence[str] | None = None) -> list[dict]:
    """One target per (product, individual class), across all products.

    Products in :data:`UNCOVERED_PRODUCTS` are skipped: they label too little of
    the store to be scored on the same population as the rest. Naming one
    explicitly in ``products`` overrides that, so the excluded set can still be
    measured on purpose.

    Args:
        products: Product subset; defaults to every registered product except
            the uncovered and the default-excluded ones.

    Returns:
        One dict per (product, class) with ``product``, ``code``, ``name``
        (human label), ``key``/``band`` (how to read it from the zarr) and
        ``group`` (its concept group, or ``None``).
    """
    explicit = products is not None
    products = list(PRODUCT_CLASSES) if not explicit else list(products)
    targets = []
    for prod in products:
        if prod in _DEFAULT_EXCLUDE or prod not in PRODUCT_CLASSES:
            continue
        if prod in UNCOVERED_PRODUCTS and not explicit:
            continue
        key, band = _product_source(prod)
        for code, name in sorted(PRODUCT_CLASSES[prod].items()):
            targets.append(
                {
                    "product": prod,
                    "code": int(code),
                    "name": name,
                    "key": key,
                    "band": band,
                    "group": CLASS_TO_GROUP.get((prod, code)),
                }
            )
    return targets


__all__ = ["CLASS_TO_GROUP", "CONCEPT_GROUPS", "HEADLINE_GROUPS", "UNCOVERED_PRODUCTS", "build_targets"]

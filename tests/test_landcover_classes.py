"""The class-name tables every concept catalogue and dossier is labelled from.

A product whose names drift from the registry's class count silently mislabels
every row that reads off it, so the two are pinned against each other here.
"""

import pytest

from geosae.data.download.presets import MAJORTOM210K
from geosae.data.download.sources import Kind, all_products
from geosae.eval.landcover_classes import PRODUCT_CLASSES

# The four bands of the packed ``overture`` array get one name table each.
OVERTURE_BANDS = ("overture_land_use", "overture_road_class", "overture_infrastructure", "overture_water_natural")


@pytest.mark.parametrize("product", sorted(PRODUCT_CLASSES))
def test_codes_are_positive_and_names_are_unique(product):
    classes = PRODUCT_CLASSES[product]
    assert classes, f"{product} has no classes"
    assert all(isinstance(code, int) and code > 0 for code in classes), "code 0 is background and is dropped"
    names = list(classes.values())
    assert all(isinstance(n, str) and n.strip() for n in names)
    assert len(set(names)) == len(names), "class names must be unambiguous within a product"


def test_every_categorical_reference_product_is_named():
    """Adding a categorical product without names would leave catalogue rows blank."""
    specs = [all_products()[k] for k in MAJORTOM210K.products]
    arrays = {s.array for s in specs if s.kind is Kind.CATEGORICAL}
    named = set(PRODUCT_CLASSES)
    # ecoregions are identified by number and excluded from the concept targets
    missing = {a for a in arrays if a not in ("overture", "resolve_ecoregions") and a not in named}
    assert not missing, f"categorical products without a class-name table: {sorted(missing)}"
    if "overture" in arrays:
        assert set(OVERTURE_BANDS) <= named


def test_name_counts_track_the_registry_class_counts():
    """Names cover every class the registry declares, minus the dropped background code."""
    registry = {s.array: s for s in all_products().values() if s.kind is Kind.CATEGORICAL}
    for array, spec in registry.items():
        if array not in PRODUCT_CLASSES:
            continue  # packed arrays (overture) are named per band, checked above
        n_named = len(PRODUCT_CLASSES[array])
        assert spec.classes - 1 <= n_named <= spec.classes, (
            f"{array}: {n_named} names vs {spec.classes} registry classes"
        )


def test_koppen_names_follow_the_chelsa_kg2_legend():
    """The stored raster is CHELSA's kg2 layer, whose 31 codes differ in order from Beck et al. (2018)."""
    kg = PRODUCT_CLASSES["chelsa_kg"]
    assert sorted(kg) == list(range(1, 32))
    short = {code: name.split(" — ")[0] for code, name in kg.items()}
    assert [short[c] for c in (1, 3, 4, 5, 6, 8, 12, 20, 30, 31)] == [
        "Af",
        "As",
        "Aw",
        "BWk",
        "BWh",
        "BSh",
        "Csa",
        "Dfc",
        "ET",
        "EF",
    ]


def test_esri_names_follow_the_io_lulc_v02_legend():
    """The stored raster is Impact Observatory's annual v02 map, which has no grass or scrub but rangeland (11)."""
    esri = PRODUCT_CLASSES["lc_esri"]
    assert sorted(esri) == [1, 2, 4, 5, 7, 8, 9, 10, 11]
    assert (esri[2], esri[10], esri[11]) == ("Trees", "Clouds", "Rangeland")


def test_hilda_names_follow_the_v1_states():
    """The stored HILDA+ raster holds v1 states, where 44 is forest, not a forest transition."""
    hilda = PRODUCT_CLASSES["hildaplus"]
    assert [hilda[c] for c in (11, 22, 33, 44, 55, 66, 77)] == [
        "Urban",
        "Cropland",
        "Pasture / rangeland",
        "Forest",
        "Other grass / shrubland",
        "Sparse / no vegetation",
        "Water",
    ]


def test_pnv_names_follow_the_published_legend():
    """The PNV raster's own legend (Zenodo 3631254): LC100 codes plus mangrove (127) and sub-polar barren (21)."""
    pnv = PRODUCT_CLASSES["pnv"]
    legend = {20, 21, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116}
    legend |= {121, 122, 123, 124, 125, 126, 127, 200}
    assert set(pnv) == legend
    assert "mangrove" in pnv[127].lower()
    assert "barren" in pnv[21].lower()

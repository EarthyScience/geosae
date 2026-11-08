"""The concept vocabulary: per-class targets and the concept groups they form."""

from geosae.eval.concepts import (
    CLASS_TO_GROUP,
    CONCEPT_GROUPS,
    HEADLINE_GROUPS,
    UNCOVERED_PRODUCTS,
    _product_source,
    build_targets,
)
from geosae.eval.landcover_classes import PRODUCT_CLASSES


def test_product_source_overture_bands():
    assert _product_source("lc_esa_wc") == ("lc_esa_wc", None)
    assert _product_source("overture_land_use") == ("overture", 1)
    assert _product_source("overture_water_natural") == ("overture", 4)


def test_taxonomy_has_twenty_groups_and_the_headline_groups_are_among_them():
    assert len(CONCEPT_GROUPS) == 20
    assert set(HEADLINE_GROUPS) <= set(CONCEPT_GROUPS)


def test_every_grouped_class_belongs_to_a_known_product():
    for group, members in CONCEPT_GROUPS.items():
        for product, _ in members:
            assert product in PRODUCT_CLASSES, (group, product)


def test_class_to_group_inverts_the_taxonomy():
    assert CLASS_TO_GROUP[("lc_esa_wc", 10)] == "forest"
    assert sum(len(m) for m in CONCEPT_GROUPS.values()) >= len(CLASS_TO_GROUP)


def test_targets_carry_their_group():
    targets = {(t["product"], t["code"]): t for t in build_targets(["lc_esa_wc"])}
    assert targets[("lc_esa_wc", 10)]["group"] == "forest"
    assert all(t["key"] == "lc_esa_wc" and t["band"] is None for t in targets.values())


def test_uncovered_products_are_out_of_the_vocabulary_unless_asked_for():
    products = {t["product"] for t in build_targets()}
    assert not (products & set(UNCOVERED_PRODUCTS)), "an unscorable product is inflating the denominator"
    named = build_targets(["globeland30"])
    assert named and {t["product"] for t in named} == {"globeland30"}

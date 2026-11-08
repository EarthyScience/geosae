"""Black-box tests for geosae.eval.registry — the labelled pixel sample.

Contracts asserted (not implementation):
  * the class table enumerates every (product, class) target exactly once, and
    its product ids index back into the product order.
  * a sample round-trips through its single ``.npz`` unchanged, including the
    ``None`` concept groups and the per-concept positive counts.
  * ``valid_concepts`` mirrors the ``min_positive`` rule the AUROC matrix uses.
"""

from __future__ import annotations

import numpy as np

from geosae.eval.concepts import build_targets
from geosae.eval.registry import RegistrySample, build_class_table


def _tiny_sample(n: int = 40, dim: int = 6) -> RegistrySample:
    rng = np.random.default_rng(0)
    labels = {"prod_a": np.repeat([1, 2, 0, 1], n // 4), "prod_b": np.repeat([7, 0, 7, 7], n // 4)}
    class_table = np.array([[0, 1], [0, 2], [1, 7]])
    return RegistrySample.from_labels(
        raw=rng.standard_normal((n, dim)).astype(np.float32),
        labels=labels,
        class_table=class_table,
        product_order=["prod_a", "prod_b"],
        concept_names=["a-one", "a-two", "b-seven"],
        concept_groups=["water", None, "forest"],
        cell=np.repeat(np.arange(4), n // 4),
    )


def test_build_class_table_enumerates_every_target_once():
    class_table, product_order, names, groups = build_class_table()
    targets = build_targets()
    assert class_table.shape == (len(targets), 2)
    assert len(names) == len(groups) == len(targets)
    pairs = {(product_order[int(p)], int(c)) for p, c in class_table}
    assert pairs == {(t["product"], t["code"]) for t in targets}
    assert set(product_order) == {t["product"] for t in targets}
    assert names[0] == targets[0]["name"] and groups[0] == targets[0]["group"]


def test_build_class_table_honours_a_product_subset():
    class_table, product_order, names, _ = build_class_table(products=["lc_esa_wc"])
    assert product_order == ["lc_esa_wc"]
    assert (class_table[:, 0] == 0).all()
    assert len(names) == class_table.shape[0] > 0


def test_positive_counts_and_valid_concepts_follow_min_positive():
    s = _tiny_sample()
    assert s.n_positive.tolist() == [20, 10, 30]
    assert s.valid_concepts(min_positive=15).tolist() == [True, False, True]
    assert s.valid_concepts(min_positive=1).all()


def test_concept_arrays_mask_membership_and_nodata():
    s = _tiny_sample()
    concept_mask, product_valid = s.concept_arrays()
    assert concept_mask.shape == (s.n_pixels, s.class_table.shape[0])
    assert product_valid.shape == (s.n_pixels, len(s.product_order))
    assert concept_mask.dtype == bool and product_valid.dtype == bool

    # Concept columns mirror n_positive exactly (they are the same membership test).
    assert concept_mask.sum(axis=0).tolist() == s.n_positive.tolist()
    # product_valid is the nodata (== 0) test, independent of which class an
    # in-concept pixel belongs to.
    assert product_valid[:, 0].sum() == (s.labels["prod_a"] != 0).sum()
    assert product_valid[:, 1].sum() == (s.labels["prod_b"] != 0).sum()
    # Every concept-positive pixel is necessarily valid for its own product.
    for c in range(s.class_table.shape[0]):
        pid = int(s.class_table[c, 0])
        assert (product_valid[concept_mask[:, c], pid]).all()


def test_registry_sample_roundtrips_through_one_npz(tmp_path):
    s = _tiny_sample()
    path = s.save(tmp_path / "sample.npz")
    loaded = RegistrySample.load(path)
    assert np.array_equal(loaded.raw, s.raw)
    assert loaded.product_order == s.product_order
    assert loaded.concept_names == s.concept_names
    assert loaded.concept_groups == s.concept_groups  # None survives the round trip
    assert np.array_equal(loaded.class_table, s.class_table)
    assert np.array_equal(loaded.cell, s.cell)
    assert np.array_equal(loaded.n_positive, s.n_positive)
    assert set(loaded.labels) == set(s.labels)
    for k, v in s.labels.items():
        assert np.array_equal(loaded.labels[k], v)


def test_sample_scores_through_the_auroc_matrix():
    """The sample is directly consumable by the coverage matrix it exists for."""
    from geosae.eval.coverage import concept_auroc_matrix

    s = _tiny_sample(n=400, dim=5)
    auc = concept_auroc_matrix(s.raw, s.labels, s.class_table, s.product_order, max_per_side=50, min_positive=50)
    assert auc.shape == (5, 3)
    assert (auc >= 0.5).all()


def test_sampled_pixels_keep_the_equal_per_cell_invariant(patch_store):
    """Every cell contributes the same pixels in the same order — what the block null needs."""
    import zarr

    from geosae.eval.coverage import block_permuted_best_auroc
    from geosae.eval.registry import sample_registry_pixels

    path, _ = patch_store(n=16, patch_size=4, dim=6)
    g = zarr.open_group(path, mode="a")
    codes = np.repeat(np.arange(1, 17), 16).reshape(16, 4, 4) % 3 * 10  # 0/10/20 per cell
    g.create_array("lc_esa_wc", shape=(16, 4, 4), chunks=(4, 4, 4), dtype="int16")[:] = codes.astype("int16")

    s = sample_registry_pixels(
        path, split="test", n_patches=3, pixels_per_patch=5, products=["lc_esa_wc"], seed=0, embedding_name="alphaearth"
    )
    assert s.raw.shape == (15, 6)
    assert s.labels["lc_esa_wc"].shape == (15,)
    counts = np.unique(s.cell, return_counts=True)[1]
    assert counts.tolist() == [5, 5, 5]
    assert set(np.unique(s.labels["lc_esa_wc"]).tolist()) <= {0, 10, 20}
    # The invariant is exactly what the block null validates.
    binary = (s.labels["lc_esa_wc"] == s.labels["lc_esa_wc"][0]).astype(int)
    assert block_permuted_best_auroc(s.raw, s.cell, binary, n_perm=3, max_per_side=5).shape == (3,)


def test_single_pixel_patches_need_min_spatial_std_disabled(patch_store):
    """A patch_size=1 store's within-patch std is trivially zero -- the default filter drops every cell."""
    import pytest

    from geosae.eval.registry import sample_registry_pixels

    path, _ = patch_store(n=8, patch_size=1, dim=4)
    with pytest.raises(ValueError, match="no cells"):
        sample_registry_pixels(path, split="test", n_patches=2, pixels_per_patch=1)

    s = sample_registry_pixels(
        path, split="test", n_patches=2, pixels_per_patch=1, min_spatial_std=None, drop_flat_pixel_cells=False
    )
    assert s.n_pixels == 2


def test_uncalibrated_batchtopk_cannot_pretend_to_be_sparse():
    """An untrained BatchTopK model must refuse to hand back "sparse" codes.

    Its eval threshold starts at -1, which every latent clears, so the codes
    would come out fully dense while claiming to be sparse — an untrained random
    dictionary would then silently be scored as a dense basis.
    """
    import pytest

    from geosae.eval.registry import calibrate_sparsity, encode_sample
    from geosae.models import build_model

    raw = np.random.default_rng(0).standard_normal((512, 8)).astype(np.float32)
    model = build_model({"type": "batchtopk", "input_dim": 8, "latent_dim": 32, "k": 4, "seed": 0})
    assert float(model.min_activation) < 0

    with pytest.raises(ValueError, match="calibrated sparsity threshold"):
        encode_sample(model, raw, device="cpu", sparse=True)

    dense = encode_sample(model, raw, device="cpu", sparse=False)
    codes = encode_sample(calibrate_sparsity(model, raw, device="cpu", batch_size=128), raw, device="cpu")
    assert float(model.min_activation) > 0
    assert (codes != 0).mean() < (dense != 0).mean()  # now genuinely sparser than the pre-top-k latents


def test_calibrated_sparsity_spends_the_k_budget_at_every_width():
    """A calibrated threshold must keep ~``k`` latents per pixel, however wide the dictionary.

    The threshold is the ``1 - k/M`` quantile of the activations, so it moves
    further into the tail as ``M`` grows. Estimating it from one batch of
    *spatially correlated* pixels puts it far enough off at large ``M`` to
    silence most of the dictionary and drive any coverage statistic to zero, so
    the budget — not merely "sparser than dense" — is what has to hold.
    """
    from geosae.eval.registry import calibrate_sparsity, encode_sample
    from geosae.models import build_model

    rng = np.random.default_rng(0)
    # Pixels arrive grouped by cell, and cells differ in embedding magnitude, so
    # the leading rows of the array are not a fair sample of the whole of it.
    # Here the first block is the brightest: a threshold read off it alone sits
    # far above the one the rest of the array needs.
    blocks = [(3.0 if i == 0 else 1.0) * rng.standard_normal((256, 16)) for i in range(8)]
    raw = np.concatenate(blocks).astype(np.float32)

    k = 4
    for latent_dim in (32, 256, 1024):
        model = build_model({"type": "batchtopk", "input_dim": 16, "latent_dim": latent_dim, "k": k, "seed": 0})
        codes = encode_sample(
            calibrate_sparsity(model, raw, device="cpu", batch_size=256), raw, device="cpu", batch_size=256
        )
        l0 = float((codes != 0).sum()) / codes.shape[0]
        assert 0.5 * k <= l0 <= 2.0 * k, f"M={latent_dim}: mean L0 {l0:.2f} is not the k={k} budget"
        # A dictionary that spends its budget cannot have gone mostly silent.
        assert (codes != 0).any(axis=0).mean() > 0.5, f"M={latent_dim}: most features never fire"


def test_majority_vote_tokens_picks_the_true_majority_and_masks_ties_and_nodata():
    """The label-store indirection's core primitive: a token's
    label is its footprint's majority class, or nodata (0) when no class holds
    an absolute majority of valid pixels -- a tie, or an all-nodata footprint.
    """
    from geosae.eval.registry import _majority_vote_tokens

    patch = np.zeros((4, 4), dtype=np.int64)
    patch[:2, :2] = 1  # token (0,0): unanimous class 1
    patch[:2, 2:] = np.array([[1, 1], [2, 2]])  # token (0,1): 2 vs 2 -- a tie, no absolute majority
    patch[2:, :2] = np.array([[1, 2], [2, 2]])  # token (1,0): 3/4 class 2 -- a true majority
    patch[2:, 2:] = 0  # token (1,1): entirely nodata

    out = _majority_vote_tokens(patch, footprint=2, fill_value=0)
    assert out.dtype == patch.dtype
    assert out.tolist() == [[1, 0], [2, 0]]


def test_sample_registry_pixels_reads_labels_from_a_parent_store_via_indirection(patch_store):
    """A patch-token subset store (OlmoEarth/MMEarth-style geometry)
    carries no catalog products of its own -- its ``cell_indices`` array names
    the parent row each cell was cropped from, and its own token grid is
    coarser than the parent's label pixels. ``label_store`` must resolve the
    parent's products through that mapping, majority-voted onto the child's
    own grid, rather than finding no concept at all.
    """
    import zarr

    from geosae.eval.registry import LabelStoreSpec, sample_registry_pixels

    n = 8
    parent_path, _ = patch_store(name="parent.zarr", n=n, patch_size=4, dim=4, chunk=n)
    gp = zarr.open_group(parent_path, mode="a")
    # Every parent patch is split into 2x2 quadrants of four distinct classes,
    # so the child's 2x2 token grid (footprint=2) has an unambiguous, fully
    # deterministic expected label per token, identical on every cell.
    token_codes = np.array([[1, 2], [3, 4]], dtype="int16")
    quadrant_codes = token_codes.repeat(2, axis=0).repeat(2, axis=1)
    codes = np.broadcast_to(quadrant_codes, (n, 4, 4)).astype("int16")
    gp.create_array("lc_esa_wc", shape=(n, 4, 4), chunks=(n, 4, 4), dtype="int16")[:] = codes

    child_path, _ = patch_store(name="child.zarr", n=n, patch_size=2, dim=5, chunk=n)
    gc = zarr.open_group(child_path, mode="a")
    gc.create_array("cell_indices", shape=(n,), chunks=(n,), dtype="int64")[:] = np.arange(n, dtype="int64")

    label_store = LabelStoreSpec(zarr_path=parent_path, cell_index_array="cell_indices", token_footprint=2)
    sample = sample_registry_pixels(
        child_path,
        split="test",
        embedding_name="alphaearth",
        n_patches=n,
        pixels_per_patch=4,
        products=["lc_esa_wc"],
        label_store=label_store,
        min_spatial_std=None,
        drop_flat_pixel_cells=False,
    )
    assert len(sample.concept_names) > 0  # block7_available, unlike the un-indirected child store
    labels = sample.labels["lc_esa_wc"].reshape(-1, 2, 2)  # (n_cells_drawn, 2, 2) token grid, cell-major
    assert (labels == token_codes).all()


def test_sample_registry_pixels_without_a_label_store_yields_no_concepts_on_an_unlabelled_store(patch_store):
    """The default (no indirection) is unchanged: a store with no catalog
    product of its own still yields an empty vocabulary, not an error."""
    from geosae.eval.registry import sample_registry_pixels

    path, _ = patch_store(n=8, patch_size=2, dim=4)
    sample = sample_registry_pixels(
        path, split="test", n_patches=8, pixels_per_patch=4, products=["lc_esa_wc"], min_spatial_std=None
    )
    assert sample.concept_names  # the product is known to the catalog...
    assert "lc_esa_wc" not in sample.labels  # ...but this store carries none of it, and reads no key

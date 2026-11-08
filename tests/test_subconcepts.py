"""Sub-concept discovery: class fractions, agreement metrics, nulls, dedup (synthetic ground truth)."""

import numpy as np
import pytest
import torch
import zarr

from geosae.cli import cmd_subconcepts, main
from geosae.eval.subconcepts import (
    SubconceptConfig,
    _block_permutation,
    _geographic_kmeans,
    _geographic_sizematched,
    _grid_block_id,
    _label_permutation,
    align_features,
    contingency,
    dedup_features,
    mutual_information,
    normalized_mutual_information,
    null_scores,
    patch_class_fractions,
    patch_feature_activation,
    score_partition,
    subconcept_labels,
    subconcept_members,
    validate_parent,
    write_subconcepts,
)

# ---------------------------------------------------------------------------
# patch_class_fractions
# ---------------------------------------------------------------------------


def _data_store(tmp_path, *, ps=4):
    """4x4-px 'lc_esa_wc' (nodata=0) + 'pnv' (nodata=255) product, 3 patches.

    Patch 0: lc_esa_wc all code 10 (pure); pnv all code 40.
    Patch 1: lc_esa_wc half code 10 / half code 20; pnv all nodata (255).
    Patch 2: lc_esa_wc all nodata (0); pnv half code 40 / half nodata.
    A per-cell mask (``mask_lc_esa_wc``) additionally excludes patch 2 entirely.
    """
    g = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    wc = np.zeros((3, ps, ps), dtype="uint8")
    wc[0, :, :] = 10
    wc[1, :, : ps // 2] = 10
    wc[1, :, ps // 2 :] = 20
    # patch 2 stays all-0 (nodata)
    g.create_array("lc_esa_wc", shape=wc.shape, chunks=wc.shape, dtype="uint8")[:] = wc
    g.create_array("mask_lc_esa_wc", shape=(3,), chunks=(3,), dtype="bool")[:] = [True, True, False]

    pnv = np.full((3, ps, ps), 255, dtype="uint8")
    pnv[0, :, :] = 40
    pnv[2, :, : ps // 2] = 40
    g.create_array("pnv", shape=pnv.shape, chunks=pnv.shape, dtype="uint8")[:] = pnv
    return str(tmp_path / "data.zarr")


def test_patch_class_fractions_exact_values(tmp_path):
    data_p = _data_store(tmp_path)
    data = zarr.open_group(data_p, mode="r")
    cells = np.array([0, 1, 2])
    fractions, codes, valid_frac = patch_class_fractions(data, "lc_esa_wc", cells)
    assert list(codes) == [10, 20]
    assert fractions[0].tolist() == pytest.approx([1.0, 0.0])
    assert fractions[1].tolist() == pytest.approx([0.5, 0.5])
    assert valid_frac[0] == pytest.approx(1.0)
    assert valid_frac[1] == pytest.approx(1.0)


def test_patch_class_fractions_255_nodata(tmp_path):
    data_p = _data_store(tmp_path)
    data = zarr.open_group(data_p, mode="r")
    cells = np.array([0, 1, 2])
    fractions, codes, valid_frac = patch_class_fractions(data, "pnv", cells)
    assert list(codes) == [40]
    assert fractions[0, 0] == pytest.approx(1.0)  # all 40
    assert fractions[1, 0] == pytest.approx(0.0)  # all 255 nodata
    assert fractions[2, 0] == pytest.approx(1.0)  # half 40 / half nodata -> valid frac of 40 is 1.0
    assert valid_frac[1] == pytest.approx(0.0)
    assert valid_frac[2] == pytest.approx(0.5)


def test_patch_class_fractions_mask_gate(tmp_path):
    data_p = _data_store(tmp_path)
    data = zarr.open_group(data_p, mode="r")
    cells = np.array([0, 1, 2])
    _, _, valid_frac = patch_class_fractions(data, "lc_esa_wc", cells)
    assert valid_frac[2] == pytest.approx(0.0)  # mask_lc_esa_wc gates patch 2 out entirely


# ---------------------------------------------------------------------------
# contingency / mutual information
# ---------------------------------------------------------------------------


def test_contingency_hand_built():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 1, 1])
    table, ca, cb = contingency(a, b)
    assert ca.tolist() == [0, 1] and cb.tolist() == [0, 1]
    assert table.tolist() == [[1, 1], [0, 2]]


def test_mutual_information_hand_computed():
    # Independent 2x2 table: joint = product of marginals -> MI = 0.
    table = np.array([[25, 25], [25, 25]])
    assert mutual_information(table) == pytest.approx(0.0, abs=1e-10)
    # Perfectly dependent (diagonal) table: MI = H(marginal) = ln(2).
    table = np.array([[50, 0], [0, 50]])
    assert mutual_information(table) == pytest.approx(np.log(2))


def test_nmi_identity_and_constant():
    identical = np.diag([10, 20, 5])
    assert normalized_mutual_information(identical) == pytest.approx(1.0)
    constant = np.array([[7], [13]])  # b has a single value -> H(b) = 0
    assert normalized_mutual_information(constant) == 0.0


# ---------------------------------------------------------------------------
# score_partition
# ---------------------------------------------------------------------------


def test_score_partition_precision_recall_lift():
    # 2 sub-concepts x 2 ref strata, 40 parent patches, no exclusions.
    labels = np.array([0] * 20 + [1] * 20)
    ref = np.array([0] * 15 + [1] * 5 + [0] * 5 + [1] * 15)
    out = score_partition(labels, ref, min_stratum_patches=5)
    sub0 = next(s for s in out["subconcepts"] if s["subconcept"] == 0)
    sub1 = next(s for s in out["subconcepts"] if s["subconcept"] == 1)
    assert sub0["stratum_code"] == 0 and sub0["precision"] == pytest.approx(0.75)
    assert sub0["recall"] == pytest.approx(15 / 20)  # 15 of the 20 ref==0 patches are sub==0
    assert sub0["lift"] == pytest.approx(0.75 / 0.5)  # base rate P(ref=0|parent) = 20/40
    assert sub1["stratum_code"] == 1 and sub1["precision"] == pytest.approx(0.75)


def test_score_partition_excludes_sentinels():
    labels = np.array([0, 0, -1, 1, 1])
    ref = np.array([0, 0, 0, -1, 1])
    out = score_partition(labels, ref, min_stratum_patches=1)
    assert out["n_valid"] == 3  # patch 2 (label -1) and patch 3 (ref -1) dropped
    # only sub-concept 1 has any valid (label, ref) pair; sub 0 has 2 valid pairs, both ref=0
    sub0 = next(s for s in out["subconcepts"] if s["subconcept"] == 0)
    assert sub0["precision"] == pytest.approx(1.0)
    assert sub0["support"] == 2  # support counts all label==0 patches, valid or not


# ---------------------------------------------------------------------------
# Null generators
# ---------------------------------------------------------------------------


def test_label_permutation_preserves_multiset():
    labels = np.array([0, 0, 1, 1, 1, -1, 2])
    out = _label_permutation(labels, np.random.default_rng(0))
    assert out.shape == labels.shape
    assert np.array_equal(np.sort(out), np.sort(labels))


def test_block_permutation_preserves_multiset_and_is_seeded():
    rng = np.random.default_rng(0)
    n = 100
    labels = rng.integers(0, 3, size=n)
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-60, 70, n)
    block_id = _grid_block_id(lon, lat, 5.0)
    out1 = _block_permutation(labels, block_id, np.random.default_rng(7))
    out2 = _block_permutation(labels, block_id, np.random.default_rng(7))
    out3 = _block_permutation(labels, block_id, np.random.default_rng(9))
    assert np.array_equal(np.sort(out1), np.sort(labels))
    assert np.array_equal(out1, out2)  # seeded reproducibility
    assert not np.array_equal(out1, out3)  # a different seed moves something


def test_geographic_kmeans_cluster_count_and_seeded():
    rng = np.random.default_rng(0)
    # Three well-separated geographic clusters so Lloyd's algorithm converges cleanly.
    centers = [(-100.0, 40.0), (30.0, -10.0), (140.0, 55.0)]
    lon, lat = [], []
    for clon, clat in centers:
        lon.append(clon + rng.normal(0, 1.0, 30))
        lat.append(clat + rng.normal(0, 1.0, 30))
    lon, lat = np.concatenate(lon), np.concatenate(lat)
    labels = np.zeros(90, dtype=np.int64)  # 3 distinct sub-concepts requested via `labels`
    labels[30:60] = 1
    labels[60:] = 2
    out1 = _geographic_kmeans(labels, lon, lat, np.random.default_rng(3))
    out2 = _geographic_kmeans(labels, lon, lat, np.random.default_rng(3))
    assert np.unique(out1).size == 3
    assert np.array_equal(out1, out2)


def test_null_scores_shapes_and_seeded_reproducibility():
    rng = np.random.default_rng(0)
    n = 80
    labels = rng.integers(0, 2, size=n)
    ref = rng.integers(0, 2, size=n)
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-60, 70, n)
    out1 = null_scores(labels, ref, lon, lat, kind="label_permutation", n_null=25, seed=5, min_stratum_patches=5)
    out2 = null_scores(labels, ref, lon, lat, kind="label_permutation", n_null=25, seed=5, min_stratum_patches=5)
    assert out1["n_null"] == 25
    assert set(out1["nmi"]) == {"mean", "sd", "p95", "p_value", "gap", "z"}
    assert out1 == out2  # seeded reproducibility


# ---------------------------------------------------------------------------
# dedup_features
# ---------------------------------------------------------------------------


def test_dedup_collapses_near_duplicates_keeps_orthogonal():
    decoder = np.zeros((4, 4))
    decoder[:, 0] = [1, 0, 0, 0]
    decoder[:, 1] = [0.99, np.sqrt(1 - 0.99**2), 0, 0]  # near-duplicate of atom 0 (cos ~0.99)
    decoder[:, 2] = [0, 1, 0, 0]  # orthogonal to 0
    decoder[:, 3] = [0, 0, 1, 0]  # orthogonal to everything
    aucs = np.array([0.9, 0.85, 0.8, 0.75])
    kept = dedup_features(np.array([0, 1, 2, 3]), aucs, decoder, cosine=0.9, max_keep=6)
    assert 1 not in kept  # dropped as a near-duplicate of 0
    assert set(kept.tolist()) == {0, 2, 3}


def test_dedup_max_keep_caps_output():
    decoder = np.eye(5)
    aucs = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    kept = dedup_features(np.arange(5), aucs, decoder, cosine=0.9, max_keep=2)
    assert kept.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# subconcept_labels / align_features
# ---------------------------------------------------------------------------


def test_subconcept_labels_argmax_and_unassigned():
    act = np.zeros((2, 4))
    act[0, [0, 1]] = [1.0, 0.5]
    act[1, [1, 2]] = [0.9, 1.0]
    labels = subconcept_labels(act, np.array([0, 1]), np.array([0, 1, 2, 3]))
    assert labels.tolist() == [0, 1, 1, -1]  # patch 3: both features 0 -> unassigned


def test_align_features_recovers_planted_feature():
    rng = np.random.default_rng(0)
    n = 400
    parent = np.zeros(n, dtype=np.int64)
    parent[:200] = 1
    act = rng.uniform(0, 0.1, size=(3, n)).astype(np.float32)
    act[0, :200] += 1.0  # feature 0 perfectly tracks the parent indicator
    aligned, aucs = align_features(act, parent, threshold=0.65)
    assert 0 in aligned
    assert aucs[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: validate_parent / write_subconcepts / CLI stage
# ---------------------------------------------------------------------------


def _pred_and_data_stores(tmp_path, *, n=240, n_nonparent=80, ps=4, seed=0):
    """A tiny prediction + data store with a planted parent/reference structure.

    ``n`` parent patches split into two geographic regions (lon < 0 vs
    lon >= 0), all ``lc_esa_wc == 10``; feature 0 fires only in the western
    region, feature 1 only in the eastern one, and ``resolve_biomes`` carries a
    matching 2-stratum split — so both aligned features recover their region's
    biome perfectly (``NMI == 1``), clearly beating the ``label_permutation``
    null. ``n_nonparent``
    additional patches carry a different ``lc_esa_wc`` code (excluded from the
    parent set) so the parent indicator is non-degenerate for
    :func:`~geosae.eval.subconcepts.align_features`.
    """
    rng = np.random.default_rng(seed)
    lon = np.concatenate(
        [rng.uniform(-170, -10, n // 2), rng.uniform(10, 170, n // 2), rng.uniform(-170, 170, n_nonparent)]
    )
    lat = rng.uniform(-50, 50, n + n_nonparent)
    cell = np.arange(n + n_nonparent, dtype=np.int64)

    act = np.zeros((2, n + n_nonparent), dtype=np.float32)
    west = lon[:n] < 0
    act[0, :n][west] = 1.0
    act[1, :n][~west] = 1.0

    pred = zarr.open_group(str(tmp_path / "pred.zarr"), mode="w")
    stats = np.zeros((2, n + n_nonparent, 2), dtype="float32")
    stats[:, :, 0] = act
    pred.create_array("feat_patch_stats", shape=stats.shape, chunks=stats.shape, dtype="float32")[:] = stats
    pred.create_array("lon", shape=lon.shape, chunks=lon.shape, dtype="float64")[:] = lon
    pred.create_array("lat", shape=lat.shape, chunks=lat.shape, dtype="float64")[:] = lat
    pred.create_array("cell", shape=cell.shape, chunks=cell.shape, dtype="int64")[:] = cell

    data = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    wc = np.full((n + n_nonparent, ps, ps), 10, dtype="uint8")
    wc[n:] = 20  # non-parent patches
    data.create_array("lc_esa_wc", shape=wc.shape, chunks=wc.shape, dtype="uint8")[:] = wc
    biomes = np.full(n + n_nonparent, 3, dtype="uint8")  # non-parent patches: unrelated stratum
    biomes[:n] = np.where(west, 1, 2)
    biome_px = np.zeros((n + n_nonparent, ps, ps), dtype="uint8")
    biome_px[:] = biomes[:, None, None]
    data.create_array("resolve_biomes", shape=biome_px.shape, chunks=biome_px.shape, dtype="uint8")[:] = biome_px
    # The other reference products exist (as any real store would) but carry
    # no signal here (all-nodata), so they rank last without crashing the sweep.
    empty_u8 = np.zeros((n + n_nonparent, ps, ps), dtype="uint8")
    data.create_array("chelsa_kg", shape=empty_u8.shape, chunks=empty_u8.shape, dtype="uint8")[:] = empty_u8
    empty_255 = np.full((n + n_nonparent, ps, ps), 255, dtype="uint8")
    data.create_array("pnv", shape=empty_255.shape, chunks=empty_255.shape, dtype="uint8")[:] = empty_255
    empty_u16 = np.zeros((n + n_nonparent, ps, ps), dtype="uint16")
    data.create_array("resolve_ecoregions", shape=empty_u16.shape, chunks=empty_u16.shape, dtype="uint16")[:] = (
        empty_u16
    )

    return str(tmp_path / "pred.zarr"), str(tmp_path / "data.zarr")


def test_validate_parent_recovers_planted_structure(tmp_path):
    pred_p, data_p = _pred_and_data_stores(tmp_path)
    pf = patch_feature_activation(pred_p)
    data = zarr.open_group(data_p, mode="r")
    parent_frac, parent_codes, _ = patch_class_fractions(data, "lc_esa_wc", pf["cell"])
    ref_frac, ref_codes, _ = patch_class_fractions(data, "resolve_biomes", pf["cell"])

    config = SubconceptConfig(n_null=200, n_null_rank=200, min_stratum_patches=5)
    result = validate_parent(
        pf["act"],
        pf["lon"],
        pf["lat"],
        parent_frac,
        parent_codes,
        10,
        ref_frac,
        ref_codes,
        "resolve_biomes",
        config=config,
    )
    assert result["n_parent"] == 240  # the `n` parent patches; n_nonparent ones excluded
    assert result["n_kept"] == 2
    assert result["nmi"] == pytest.approx(1.0, abs=1e-6)  # perfect 2-way agreement
    # label_permutation destroys the region structure entirely -> the observed
    # NMI (1.0) is essentially never matched by a random relabelling.
    lab = result["nulls"]["label_permutation"]
    assert lab["observed_nmi"] == pytest.approx(1.0, abs=1e-6)
    assert lab["nmi"]["p_value"] < 0.1
    # geographic_kmeans, by contrast, can rediscover the same west/east split
    # from coordinates alone (the planted structure IS purely geographic) —
    # exactly the failure mode this null is designed to catch, so it does not
    # have to reject here.
    geo = result["nulls"]["geographic_kmeans"]
    assert geo["observed_nmi"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= geo["nmi"]["p_value"] <= 1.0


def test_patch_feature_activation_shape(tmp_path):
    pred_p, _ = _pred_and_data_stores(tmp_path, n=40, n_nonparent=10)
    pf = patch_feature_activation(pred_p)
    assert pf["act"].shape == (2, 50)
    assert pf["lon"].shape == (50,) and pf["lat"].shape == (50,) and pf["cell"].shape == (50,)


def test_write_subconcepts_writes_expected_outputs(tmp_path):
    pred_p, data_p = _pred_and_data_stores(tmp_path)
    config = SubconceptConfig(n_null=10, n_null_rank=10, min_stratum_patches=5, max_subconcepts=2)
    outdir = tmp_path / "eval"
    result = write_subconcepts(pred_p, data_p, str(outdir), config=config)
    assert (outdir / "subconcept_ranking.csv").exists()
    assert (outdir / "subconcept_validation.csv").exists()
    assert (outdir / "subconcepts.json").exists()
    assert len(result["ranking"]) >= 1
    # The sweep covers every parent class of the product, so pin the parent too
    # (code 10 is the planted one); ranking order is by significance, not parent.
    row = next(r for r in result["ranking"] if r["reference_product"] == "resolve_biomes" and r["parent_code"] == 10)
    assert row["nmi"] == pytest.approx(1.0, abs=1e-6)


def _striped_pred_and_data_stores(tmp_path, *, n=300, n_nonparent=60, ps=4, seed=1):
    """Like :func:`_pred_and_data_stores`, but the split follows 8-degree lon
    stripes rather than a single west/east threshold — too fine-grained for a
    ``k=2`` geographic k-means split to rediscover, so it clears even the
    hardest null (unlike the coarse west/east fixture above).
    """
    rng = np.random.default_rng(seed)
    lon = np.concatenate([rng.uniform(-180, 180, n), rng.uniform(-180, 180, n_nonparent)])
    lat = np.concatenate([rng.uniform(-60, 60, n), rng.uniform(-60, 60, n_nonparent)])
    cell = np.arange(n + n_nonparent, dtype=np.int64)
    stripe_a = (lon[:n] // 8.0).astype(np.int64) % 2 == 0

    act = np.zeros((2, n + n_nonparent), dtype=np.float32)
    act[0, :n][stripe_a] = 1.0
    act[1, :n][~stripe_a] = 1.0

    pred = zarr.open_group(str(tmp_path / "pred.zarr"), mode="w")
    stats = np.zeros((2, n + n_nonparent, 2), dtype="float32")
    stats[:, :, 0] = act
    pred.create_array("feat_patch_stats", shape=stats.shape, chunks=stats.shape, dtype="float32")[:] = stats
    pred.create_array("lon", shape=lon.shape, chunks=lon.shape, dtype="float64")[:] = lon
    pred.create_array("lat", shape=lat.shape, chunks=lat.shape, dtype="float64")[:] = lat
    pred.create_array("cell", shape=cell.shape, chunks=cell.shape, dtype="int64")[:] = cell

    data = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    wc = np.full((n + n_nonparent, ps, ps), 10, dtype="uint8")
    wc[n:] = 20
    data.create_array("lc_esa_wc", shape=wc.shape, chunks=wc.shape, dtype="uint8")[:] = wc
    biomes = np.full(n + n_nonparent, 3, dtype="uint8")
    biomes[:n] = np.where(stripe_a, 1, 2)
    biome_px = np.zeros((n + n_nonparent, ps, ps), dtype="uint8")
    biome_px[:] = biomes[:, None, None]
    data.create_array("resolve_biomes", shape=biome_px.shape, chunks=biome_px.shape, dtype="uint8")[:] = biome_px
    empty_u8 = np.zeros((n + n_nonparent, ps, ps), dtype="uint8")
    data.create_array("chelsa_kg", shape=empty_u8.shape, chunks=empty_u8.shape, dtype="uint8")[:] = empty_u8
    empty_255 = np.full((n + n_nonparent, ps, ps), 255, dtype="uint8")
    data.create_array("pnv", shape=empty_255.shape, chunks=empty_255.shape, dtype="uint8")[:] = empty_255
    empty_u16 = np.zeros((n + n_nonparent, ps, ps), dtype="uint16")
    data.create_array("resolve_ecoregions", shape=empty_u16.shape, chunks=empty_u16.shape, dtype="uint16")[:] = (
        empty_u16
    )

    return str(tmp_path / "pred.zarr"), str(tmp_path / "data.zarr")


def test_subconcept_labels_normalisation_beats_a_loud_feature():
    """A high-amplitude feature must not swallow patches a quieter one owns.

    Feature 0 is ten times louder everywhere; feature 1 is relatively more
    active on the second half. Raw argmax hands everything to feature 0;
    ``mean_active`` normalisation splits the parent evenly.
    """
    act = np.zeros((2, 10), dtype=np.float32)
    act[0, :] = 10.0
    act[0, 5:] = 8.0
    act[1, :] = 0.5
    act[1, 5:] = 1.5
    feats = np.array([0, 1], dtype=np.int64)
    parent_idx = np.arange(10, dtype=np.int64)

    raw = subconcept_labels(act, feats, parent_idx, normalize="none")
    assert np.all(raw == 0)

    normed = subconcept_labels(act, feats, parent_idx, normalize="mean_active")
    assert np.all(normed[:5] == 0)
    assert np.all(normed[5:] == 1)


def test_subconcept_labels_rejects_an_unknown_normalisation():
    with pytest.raises(ValueError, match="unknown normalize rule"):
        subconcept_labels(np.ones((1, 2), np.float32), np.array([0]), np.array([0, 1]), normalize="bogus")


def test_geographic_sizematched_null_reproduces_the_observed_cluster_sizes():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 12 + [1] * 5 + [2] * 3, dtype=np.int64)
    lon = rng.uniform(-180, 180, labels.size)
    lat = rng.uniform(-60, 60, labels.size)

    drawn = _geographic_sizematched(labels, lon, lat, np.random.default_rng(7))
    assert sorted(np.bincount(drawn).tolist()) == sorted(np.bincount(labels).tolist())

    again = _geographic_sizematched(labels, lon, lat, np.random.default_rng(7))
    assert np.array_equal(drawn, again)


def test_geographic_sizematched_null_keeps_unassigned_patches_unassigned():
    labels = np.array([0, 0, -1, 1, -1, 1], dtype=np.int64)
    lon = np.array([0.0, 1.0, 2.0, 40.0, 41.0, 42.0])
    lat = np.zeros(6)
    drawn = _geographic_sizematched(labels, lon, lat, np.random.default_rng(3))
    assert np.array_equal(drawn == -1, labels == -1)


def test_align_features_refuses_a_nearly_empty_positive_set():
    """A perfect-looking AUROC on a handful of positives must not count as alignment."""
    rng = np.random.default_rng(0)
    n = 500
    act = rng.normal(size=(3, n)).astype(np.float32)
    parent = np.zeros(n, dtype=np.int64)
    parent[:5] = 1
    act[0, :5] = 50.0  # feature 0 separates the 5 positives perfectly

    aligned, aucs = align_features(act, parent, threshold=0.65, min_positive=30)
    assert aucs[0] > 0.95  # the score is high...
    assert aligned.size == 0  # ...but it is not trusted

    # With enough positives the same separation does count.
    parent2 = np.zeros(n, dtype=np.int64)
    parent2[:60] = 1
    act2 = rng.normal(size=(3, n)).astype(np.float32)
    act2[0, :60] = 50.0
    aligned2, _ = align_features(act2, parent2, threshold=0.65, min_positive=30)
    assert 0 in aligned2.tolist()


def test_family_wise_minp_corrects_for_choosing_the_best_subconcept():
    """A single sub-concept at an uncorrected p of ~1/n must not stay significant.

    Six sub-concepts, one of which sits at the very top of its own null. Its
    uncorrected p is tiny, but the family-wise p has to pay for the fact that we
    picked it out of six, so it must come out substantially larger.
    """
    from geosae.eval.subconcepts import _family_wise_minp

    rng = np.random.default_rng(0)
    n_null, n_sub = 200, 6
    null_lift = rng.gamma(shape=2.0, scale=1.0, size=(n_null, n_sub))
    obs = np.full(n_sub, 1.0)
    obs[3] = null_lift[:, 3].max() + 0.1  # beats every draw in its own column

    out = _family_wise_minp(obs, null_lift)
    assert out["argmin"] == 3
    assert out["min_p"] < 0.01
    assert out["p_value"] > out["min_p"]  # multiplicity is paid for
    assert 0.0 < out["p_value"] <= 1.0


def test_family_wise_minp_is_not_fooled_by_a_small_high_lift_cluster():
    """The statistic must compare each sub-concept only against its own column.

    Column 0 is a small cluster whose null lift is huge; column 1 is a large
    cluster whose null lift is modest. An observation that is unremarkable in
    its own column must not become significant just because another column
    reaches higher absolute values — the failure mode of a raw max-over-clusters
    statistic.
    """
    from geosae.eval.subconcepts import _family_wise_minp

    rng = np.random.default_rng(1)
    null_lift = np.column_stack([rng.uniform(5.0, 15.0, 200), rng.uniform(1.0, 2.0, 200)])
    obs = np.array([10.0, 1.5])  # both sit mid-distribution in their own column
    out = _family_wise_minp(obs, null_lift)
    assert out["p_value"] > 0.1


def test_score_partition_reports_family_wise_maxima():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    ref = np.array([7, 7, 7, 7, 8, 8], dtype=np.int64)
    scored = score_partition(labels, ref, min_stratum_patches=1)
    per_sub_max = max(s["lift"] for s in scored["subconcepts"])
    assert scored["max_lift"] == pytest.approx(per_sub_max)
    assert scored["max_precision"] == pytest.approx(max(s["precision"] for s in scored["subconcepts"]))
    assert scored["cluster_sizes"] == [3, 3]


def test_null_scores_reports_one_record_per_observed_subconcept():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 20).astype(np.int64)
    ref = np.repeat([5, 6, 7], 20).astype(np.int64)
    lon = rng.uniform(-180, 180, labels.size)
    lat = rng.uniform(-60, 60, labels.size)
    out = null_scores(labels, ref, lon, lat, kind="geographic_sizematched", n_null=20, seed=0, min_stratum_patches=1)
    assert len(out["subconcepts"]) == 3
    for rec in out["subconcepts"]:
        assert set(rec["precision"]) >= {"mean", "sd", "p95", "p_value", "gap", "z"}
    assert "max_lift" in out and "max_precision" in out


def test_validate_parent_beats_the_geographic_kmeans_null_for_a_fine_split(tmp_path):
    pred_p, data_p = _striped_pred_and_data_stores(tmp_path)
    pf = patch_feature_activation(pred_p)
    data = zarr.open_group(data_p, mode="r")
    parent_frac, parent_codes, _ = patch_class_fractions(data, "lc_esa_wc", pf["cell"])
    ref_frac, ref_codes, _ = patch_class_fractions(data, "resolve_biomes", pf["cell"])
    config = SubconceptConfig(n_null=100, min_stratum_patches=5)
    result = validate_parent(
        pf["act"],
        pf["lon"],
        pf["lat"],
        parent_frac,
        parent_codes,
        10,
        ref_frac,
        ref_codes,
        "resolve_biomes",
        config=config,
    )
    assert result["nmi"] == pytest.approx(1.0, abs=1e-6)
    for kind in ("label_permutation", "block_5deg", "block_20deg", "geographic_kmeans", "geographic_sizematched"):
        assert result["nulls"][kind]["nmi"]["p_value"] < 0.05


def test_write_subconcepts_produces_a_validated_pair(tmp_path):
    pred_p, data_p = _striped_pred_and_data_stores(tmp_path)
    config = SubconceptConfig(n_null=200, n_null_rank=200, min_stratum_patches=5, max_subconcepts=2)
    outdir = tmp_path / "eval"
    result = write_subconcepts(pred_p, data_p, str(outdir), config=config)
    assert len(result["validated"]) >= 1  # the (stripped, JSON-safe) validated results
    assert (outdir / "subconcepts.json").exists()
    assert (outdir / "subconcept_ranking.csv").exists()
    assert (outdir / "subconcept_validation.csv").exists()


def test_cli_dispatches_the_subconcepts_stage(tmp_path):
    from geosae import run as R

    pred_p, data_p = _pred_and_data_stores(tmp_path)
    run_dir = tmp_path / "runs" / "exp" / "20240101-000000"
    P = R.paths(run_dir)
    P["predictions"].mkdir(parents=True, exist_ok=True)
    P["eval"].mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(pred_p, P["predictions"] / "test.zarr")
    torch.save({"model": {"decoder.weight": torch.zeros(64, 2)}}, P["best"])

    cfg = {
        "name": "exp",
        "run": {"dir": str(run_dir)},
        "data": {"zarr_path": data_p},
        "subconcepts": {"n_null": 10, "n_null_rank": 10, "min_stratum_patches": 5, "max_subconcepts": 2},
    }
    result = cmd_subconcepts(cfg, run_dir)
    assert (P["eval"] / "subconcepts.json").exists()
    assert len(result["ranking"]) >= 1


def test_cli_parses_subconcepts_stage(tmp_path):
    from omegaconf import OmegaConf

    from geosae import run as R

    pred_p, data_p = _pred_and_data_stores(tmp_path)
    run_dir = tmp_path / "runs" / "exp" / "20240101-000000"
    P = R.paths(run_dir)
    P["predictions"].mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(pred_p, P["predictions"] / "test.zarr")
    torch.save({"model": {"decoder.weight": torch.zeros(64, 2)}}, P["best"])

    cfg_path = tmp_path / "cfg.yaml"
    cfg = {
        "name": "exp",
        "run": {"dir": str(run_dir)},
        "data": {"zarr_path": data_p},
        "subconcepts": {"n_null": 10, "n_null_rank": 10, "min_stratum_patches": 5, "max_subconcepts": 2},
    }
    OmegaConf.save(OmegaConf.create(cfg), str(cfg_path))
    main(["subconcepts", "-c", str(cfg_path)])
    assert (P["eval"] / "subconcepts.json").exists()


# ---------------------------------------------------------------------------
# subconcept_members (threshold partition)
# ---------------------------------------------------------------------------


def test_subconcept_members_thresholds_each_feature_at_its_typical_level():
    act = np.zeros((2, 5))
    act[0, :4] = [1.0, 3.0, 2.0, 0.0]  # active mean 2.0
    act[1, 1:] = [10.0, 30.0, 0.0, 20.0]  # active mean 20.0
    members = subconcept_members(act, np.array([0, 1]), np.arange(5), tau=1.0)
    assert members.tolist() == [[False, True, True, False, False], [False, False, True, False, True]]
    loose = subconcept_members(act, np.array([0, 1]), np.arange(5), tau=0.5)
    assert loose[0].tolist() == [True, True, True, False, False]


def test_subconcept_members_does_not_depend_on_the_sibling_set():
    rng = np.random.default_rng(0)
    act = np.where(rng.random((6, 200)) < 0.4, rng.gamma(2.0, 1.0, (6, 200)), 0.0)
    pidx = np.arange(0, 200, 2)
    alone = subconcept_members(act, np.array([2]), pidx, tau=1.0)
    with_siblings = subconcept_members(act, np.array([0, 2, 5]), pidx, tau=1.0)
    np.testing.assert_array_equal(with_siblings[1], alone[0])
    # and members of a feature are invariant to rescaling that feature (amplitude is per-seed noise)
    scaled = act.copy()
    scaled[2] *= 7.3
    np.testing.assert_array_equal(subconcept_members(scaled, np.array([2]), pidx, tau=1.0), alone)

"""Sub-concept discovery: do aligned features split a parent class the way an
*independent* reference product does, beating an explicit spatial null?

A **parent class** is a (product, code) pair — e.g. ``(lc_esa_wc, 10)`` "Tree
cover" — and a **parent patch** is one whose valid pixels are dominated by that
code (:func:`patch_class_fractions` + ``parent_purity``). The SAE features whose
patch-mean activation tracks parent membership (folded AUROC, reused from
:mod:`geosae.eval.coverage`) are the *aligned* features; :func:`dedup_features`
thins near-duplicate decoder atoms among them, and the remaining features'
``argmax`` partitions the parent patches into **sub-concepts**
(:func:`subconcept_labels`).

The question is whether that partition agrees with an **independent** coarse
reference product (RESOLVE biomes, Köppen-Geiger, potential natural vegetation,
RESOLVE ecoregions — never the concept registry the SAE was validated against
elsewhere) beyond what three nulls already explain: shuffling labels
(``label_permutation``), shuffling which geographic block gets which patch of
labels (``block_permutation``), or splitting the same parent geographically with
no reference information at all — by k-means (``geographic_kmeans``) or, the
null that decides the verdict, into compact blobs carrying exactly the observed
cluster sizes (``geographic_sizematched``). Every null is scored through the
*same* pipeline used for the observed partition — including the
argmax-over-strata best-match selection — so it absorbs the multiple-comparison
advantage that selection carries, and the verdict itself rests on a
multiplicity-corrected minP statistic (:func:`_family_wise_minp`) because
naming a parent's *best* sub-concept is a second selection on top of that.

:func:`rank_candidates` sweeps every parent class of one product against every
reference product; :func:`write_subconcepts` runs the sweep, re-validates the
significant pairs at full null resolution, and writes the JSON/CSV outputs for a
run directory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zarr

from geosae.eval.concepts import _product_source
from geosae.eval.coverage import auc_all_features
from geosae.eval.landcover_classes import PRODUCT_CLASSES

# Nodata code per product. 0 covers most categorical products; a handful use
# 255 (fill byte) or -200 (GHS-SMOD's negative sentinel). Undeclared products
# default to 0.
PRODUCT_NODATA: dict[str, int] = {
    "pnv": 255,
    "lcz": 255,
    "hildaplus": 255,
    "globeland30": 255,
    "worldcereal": 255,
    "jrc_gsw": 255,
    "ghs_built_c": 255,
    "elu": 255,
    "ghs_smod": -200,
}
DEFAULT_NODATA: int = 0

# Coarse products independent of the SAE's concept-alignment registry — the
# reference partitions sub-concept discovery is validated against.
REFERENCE_PRODUCTS: tuple[str, ...] = ("resolve_biomes", "chelsa_kg", "pnv", "resolve_ecoregions")

NULL_KINDS: tuple[str, ...] = (
    "label_permutation",
    "block_permutation",
    "geographic_kmeans",
    "geographic_sizematched",
)

# The binding null: the only one that is both spatially compact and matched to
# the observed cluster sizes. ``geographic_kmeans`` is upper-biased because
# k-means produces near-balanced clusters and agreement rises with
# cluster-size entropy.
BINDING_NULL: str = "geographic_sizematched"

EPS = 1e-12

# Below this many positive patches an alignment AUROC is not a measurement; see
# the note in :func:`align_features`. Deliberately lower than
# ``coverage.balanced_binary_subsample``'s ``min_positive=50``: that guard counts
# *pixels* on a balanced subsample, this one counts whole 640 m *patches*, of
# which there are three orders of magnitude fewer. The value matches
# ``SubconceptConfig.min_stratum_patches``, the other patch-level count in this
# module, so the two patch-level floors agree.
MIN_POSITIVE_FOR_ALIGNMENT: int = 30


@dataclass(frozen=True)
class SubconceptConfig:
    """Thresholds and null-test settings for sub-concept validation.

    Attributes:
        parent_product: Product the parent class is drawn from.
        parent_purity: Minimum valid-pixel fraction of the parent code for a
            patch to count as a parent patch.
        reference_purity: Minimum valid-pixel fraction of the modal reference
            stratum for a parent patch to carry a reference label.
        align_threshold: Minimum folded patch-level AUROC (vs. the parent
            indicator) for a feature to be "aligned".
        dedup_cosine: Decoder-atom cosine similarity above which a lower-AUROC
            aligned feature is dropped as a near-duplicate.
        max_subconcepts: Maximum kept (deduplicated) features.
        min_stratum_patches: Minimum parent-patch count for a reference
            stratum to be eligible as a sub-concept's best match.
        n_null: Null draws per null kind for a full validation.
        n_null_rank: Reduced null draws used by :func:`rank_candidates`.
        block_deg: Macro-cell sizes (degrees) for ``block_permutation``; one
            null is emitted per value, keyed ``block_<deg>deg``. Several sizes
            are needed because a partition clumpy at continental scale is
            barely disturbed by a small block: on forest the 5-degree null sits
            at NMI 0.008 against the 20-degree null's 0.029.
        normalize: Partition rule passed to :func:`subconcept_labels`.
        seed: Base RNG seed for null generation.
    """

    parent_product: str = "lc_esa_wc"
    parent_purity: float = 0.5
    reference_purity: float = 0.5
    align_threshold: float = 0.65
    dedup_cosine: float = 0.9
    max_subconcepts: int = 6
    min_stratum_patches: int = 30
    n_null: int = 200
    n_null_rank: int = 50
    block_deg: tuple[float, ...] = (5.0, 20.0)
    normalize: str = "mean_active"
    seed: int = 0


# ---------------------------------------------------------------------------
# Per-patch class fractions
# ---------------------------------------------------------------------------


def patch_class_fractions(
    data: zarr.Group, product: str, cells: np.ndarray, *, chunk: int = 5000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patch valid-pixel class-fraction table for one categorical product.

    Nodata (:data:`PRODUCT_NODATA`, default 0) and, when present, the store's
    ``mask_<key>`` per-cell flag are excluded from both the fraction
    denominator and the set of observed class codes.

    Args:
        data: Open zarr group of the data store.
        product: Product name (a key of :data:`PRODUCT_NODATA`/
            :data:`~geosae.eval.landcover_classes.PRODUCT_CLASSES`, or one of
            the ``overture_*`` band products).
        cells: ``(N,)`` row index into ``data`` for each patch (the
            prediction store's ``cell`` array).
        chunk: Patches per contiguous-span zarr read.

    Returns:
        ``(fractions, codes, valid_frac)``: ``fractions`` is ``(N, K)`` — the
        fraction of *valid* pixels carrying each of the ``K`` observed class
        ``codes`` (``(K,)``, ascending); ``valid_frac`` is ``(N,)``, the
        fraction of a patch's pixels that are valid.
    """
    cells = np.asarray(cells, dtype=np.int64)
    n = cells.shape[0]
    key, band = _product_source(product)
    nodata = PRODUCT_NODATA.get(product, DEFAULT_NODATA)
    mask_key = f"mask_{key}"
    has_mask = mask_key in data

    valid_frac = np.zeros(n, dtype=np.float64)
    per_patch_counts: list[dict[int, int]] = []
    codes_seen: set[int] = set()

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        cids = cells[s:e]
        lo, hi = int(cids.min()), int(cids.max()) + 1
        sel = cids - lo
        arr = np.asarray(data[key][lo:hi])[sel]
        if band is not None:
            arr = arr[..., band]
        b = arr.shape[0]
        arr = arr.reshape(b, -1)
        valid = arr != nodata
        if has_mask:
            m = np.asarray(data[mask_key][lo:hi])[sel].astype(bool)
            valid &= m[:, None]
        valid_frac[s:e] = valid.mean(axis=1)
        for i in range(b):
            vals = arr[i][valid[i]]
            if vals.size == 0:
                per_patch_counts.append({})
                continue
            u, c = np.unique(vals, return_counts=True)
            d = dict(zip(u.tolist(), c.tolist(), strict=True))
            per_patch_counts.append(d)
            codes_seen.update(d)

    codes = np.asarray(sorted(codes_seen), dtype=np.int64)
    code_to_idx = {int(c): i for i, c in enumerate(codes.tolist())}
    fractions = np.zeros((n, codes.size), dtype=np.float64)
    n_px = arr.shape[1] if n else 0  # patch pixel count, constant across chunks
    for i, d in enumerate(per_patch_counts):
        n_valid_px = valid_frac[i] * n_px
        if n_valid_px <= 0:
            continue
        for code, cnt in d.items():
            fractions[i, code_to_idx[int(code)]] = cnt / n_valid_px
    return fractions, codes, valid_frac


# ---------------------------------------------------------------------------
# Feature activation / alignment / dedup / labelling
# ---------------------------------------------------------------------------


def patch_feature_activation(pred_path: str) -> dict:
    """Per-patch mean activation for every feature, plus geolocation.

    Returns:
        ``{"act": (F, N) float32, "lon": (N,), "lat": (N,), "cell": (N,) int64}``
        — ``act`` is column 0 of ``feat_patch_stats`` (mean activation);
        ``cell`` is the row index into the data zarr for each patch.
    """
    g = zarr.open_group(str(pred_path), mode="r")
    stats = np.asarray(g["feat_patch_stats"])  # (F, N, 2)
    return {
        "act": stats[:, :, 0].astype(np.float32),
        "lon": np.asarray(g["lon"], dtype=np.float64),
        "lat": np.asarray(g["lat"], dtype=np.float64),
        "cell": np.asarray(g["cell"], dtype=np.int64),
    }


def align_features(
    act: np.ndarray,
    parent: np.ndarray,
    *,
    threshold: float = 0.65,
    min_positive: int = MIN_POSITIVE_FOR_ALIGNMENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Features whose patch-level folded AUROC against ``parent`` clears ``threshold``.

    An AUROC computed against a nearly empty positive set is not a measurement:
    with a handful of positive patches it rewards whichever features are strongly
    expressed, whatever they represent. So no feature is reported as aligned
    below ``min_positive`` positive patches.

    Args:
        act: ``(F, N)`` per-patch mean activation (as from
            :func:`patch_feature_activation`).
        parent: ``(N,)`` binary parent-patch indicator, over *all* patches.
        threshold: Minimum folded AUROC to count as aligned.
        min_positive: Minimum number of positive patches required before any
            AUROC is trusted. Below this no feature is reported as aligned.

    Returns:
        ``(aligned, aucs)``: ``aligned`` — indices of features with
        ``aucs >= threshold``, empty when the positive set is smaller than
        ``min_positive``; ``aucs`` — ``(F,)`` folded AUROC for every
        feature (see :func:`geosae.eval.coverage.auc_all_features`).
    """
    features = np.asarray(act).T  # (N, F)
    y = np.asarray(parent).astype(np.int64)
    aucs = auc_all_features(features, y)
    if int((y == 1).sum()) < min_positive:
        return np.empty(0, dtype=np.int64), aucs
    aligned = np.where(aucs >= threshold)[0]
    return aligned, aucs


def dedup_features(
    feats: np.ndarray,
    aucs: np.ndarray,
    decoder: np.ndarray | None,
    *,
    cosine: float = 0.9,
    max_keep: int = 6,
) -> np.ndarray:
    """Greedily thin ``feats`` to at most ``max_keep``, dropping near-duplicate atoms.

    Features are visited in descending ``aucs`` order; a feature is dropped
    when its unit-normalised decoder atom has cosine similarity ``> cosine``
    with an atom already kept. Without a ``decoder``, similarity cannot be
    judged, so only the AUROC ordering and ``max_keep`` cap apply.

    Args:
        feats: ``(K,)`` candidate feature indices (e.g. the aligned set).
        aucs: ``(F,)`` per-feature AUROC (indexed by the values in ``feats``).
        decoder: ``(D, F)`` decoder weight (columns are feature atoms), or
            ``None`` to skip the cosine-similarity gate.
        cosine: Similarity threshold above which a feature is a duplicate.
        max_keep: Maximum number of features returned.

    Returns:
        ``(<=max_keep,)`` kept feature indices, in descending-AUROC order.
    """
    feats = np.asarray(feats, dtype=np.int64)
    if feats.size == 0:
        return feats
    order = feats[np.argsort(-np.asarray(aucs)[feats])]
    if decoder is None:
        return order[:max_keep]
    w = np.asarray(decoder, dtype=np.float64)
    norms = np.linalg.norm(w, axis=0, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    wn = w / norms
    kept: list[int] = []
    for f in order.tolist():
        if len(kept) >= max_keep:
            break
        if all(float(wn[:, f] @ wn[:, k]) <= cosine for k in kept):
            kept.append(f)
    return np.asarray(kept, dtype=np.int64)


def subconcept_labels(
    act: np.ndarray,
    feats: np.ndarray,
    parent_idx: np.ndarray,
    *,
    normalize: str = "mean_active",
) -> np.ndarray:
    """Argmax-over-``feats`` sub-concept label of every parent patch.

    A raw argmax over mean activation is decided by each feature's activation
    *amplitude* rather than by which feature owns the patch, so a single
    large-amplitude feature can swallow the whole parent (cluster sizes such as
    ``[2283, 2, 2, 0, 0, 0]`` under the raw rule against
    ``[634, 359, 376, 189, 487, 242]`` normalised). Dividing each feature by its own
    typical level in the parent makes the comparison one of relative rather
    than absolute activation.

    Args:
        act: ``(F, N)`` per-patch mean activation.
        feats: ``(K,)`` kept feature indices (order defines the label codes
            ``0..K-1``).
        parent_idx: ``(P,)`` indices (into the ``N`` axis of ``act``) of the
            parent patches to label.
        normalize: ``"mean_active"`` divides each feature's activation over the
            parent patches by that feature's mean activation over the parent
            patches where it is active (``> 0``) before the argmax; a feature
            active nowhere in the parent keeps a divisor of ``1.0`` and so
            stays all-zero. ``"none"`` argmaxes the raw activation.

    Returns:
        ``(P,)`` int64 labels in ``{-1, 0, ..., K-1}``; ``-1`` marks a patch
        where every kept feature is exactly 0 (unassigned).

    Raises:
        ValueError: If ``normalize`` is not a recognised rule.
    """
    if normalize not in ("mean_active", "none"):
        raise ValueError(f"unknown normalize rule {normalize!r}; expected 'mean_active' or 'none'")
    parent_idx = np.asarray(parent_idx, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.int64)
    if feats.size == 0:
        return np.full(parent_idx.size, -1, dtype=np.int64)
    sub = np.asarray(act)[np.ix_(feats, parent_idx)]  # (K, P)
    all_zero = np.all(sub == 0, axis=0)
    if normalize == "mean_active":
        sub = sub / _mean_active_scale(sub)[:, None]
    labels = np.argmax(sub, axis=0).astype(np.int64)
    return np.where(all_zero, -1, labels)


def _mean_active_scale(sub: np.ndarray) -> np.ndarray:
    """Each row's mean over its positive entries; ``1.0`` for a row with none."""
    active = sub > 0
    totals = np.where(active, sub, 0.0).sum(axis=1)
    counts = active.sum(axis=1)
    scale = np.where(counts > 0, totals / np.maximum(counts, 1), 1.0)
    return np.where(scale > 0, scale, 1.0)


def subconcept_members(
    act: np.ndarray,
    feats: np.ndarray,
    parent_idx: np.ndarray,
    *,
    tau: float = 0.5,
) -> np.ndarray:
    """Threshold sub-concepts: the parent patches where each feature is strongly active.

    The argmax of :func:`subconcept_labels` hands a patch to whichever sibling is relatively
    loudest, so a sub-concept changes whenever the sibling set does. Here a patch belongs to
    feature ``f``'s sub-concept when ``f``'s activation, divided by ``f``'s mean over the parent
    patches where it is active (the ``"mean_active"`` normalisation), is at least ``tau``. Each
    feature's members depend on that feature alone; sub-concepts may overlap or leave patches out.

    Args:
        act: ``(F, N)`` per-patch mean activation.
        feats: ``(K,)`` feature indices.
        parent_idx: ``(P,)`` indices (into the ``N`` axis of ``act``) of the parent patches.
        tau: Threshold on the normalised activation; ``0.5`` keeps the patches where a feature is
            at least half as active as it typically is inside the parent.

    Returns:
        ``(K, P)`` bool membership, one row per feature in ``feats`` order.
    """
    parent_idx = np.asarray(parent_idx, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.int64)
    sub = np.asarray(act, dtype=np.float64)[np.ix_(feats, parent_idx)]
    return (sub > 0) & (sub / _mean_active_scale(sub)[:, None] >= tau)


# ---------------------------------------------------------------------------
# Agreement metrics (numpy/scipy only)
# ---------------------------------------------------------------------------


def contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-tabulation of two equal-length label arrays.

    Returns:
        ``(table, codes_a, codes_b)`` — ``table[i, j]`` counts
        ``labels_a == codes_a[i]`` and ``labels_b == codes_b[j]`` jointly;
        ``codes_a``/``codes_b`` are the sorted unique values of each array
        (any sentinel, e.g. ``-1``, is included as an ordinary code).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"labels_a shape {a.shape} != labels_b shape {b.shape}")
    codes_a, ia = np.unique(a, return_inverse=True)
    codes_b, ib = np.unique(b, return_inverse=True)
    table = np.zeros((codes_a.size, codes_b.size), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    return table, codes_a, codes_b


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_information(table: np.ndarray) -> float:
    """Natural-log mutual information of a joint contingency table."""
    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    if n <= 0:
        return 0.0
    pij = t / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    outer = pi * pj
    nz = (pij > 0) & (outer > 0)
    ratio = np.ones_like(pij)
    ratio[nz] = pij[nz] / outer[nz]
    return float((pij[nz] * np.log(ratio[nz])).sum())


def normalized_mutual_information(table: np.ndarray) -> float:
    """MI normalised by ``sqrt(H_a * H_b)``; ``0.0`` when either entropy is 0."""
    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    if n <= 0:
        return 0.0
    ha = _entropy(t.sum(axis=1) / n)
    hb = _entropy(t.sum(axis=0) / n)
    if ha <= 0 or hb <= 0:
        return 0.0
    return float(mutual_information(t) / np.sqrt(ha * hb))


def score_partition(
    labels: np.ndarray,
    ref: np.ndarray,
    *,
    min_stratum_patches: int = 30,
    reference_product: str | None = None,
) -> dict:
    """Agreement of a sub-concept partition with a reference partition.

    Both arrays cover the same parent patches; ``-1`` marks an excluded patch
    (unassigned sub-concept, or a reference stratum below ``reference_purity``)
    in either array and such patches are dropped before scoring.

    For each sub-concept ``k`` the eligible reference stratum ``b`` maximising
    ``precision(k, b) = P(ref=b | sub=k)`` among strata with
    ``>= min_stratum_patches`` parent patches is chosen; ``recall(k, b) =
    P(sub=k | ref=b)`` and ``lift(k, b) = precision / P(ref=b | parent)`` (the
    base rate over *all* parent patches, valid or not) follow.

    Args:
        labels: ``(P,)`` sub-concept labels (``-1`` = unassigned).
        ref: ``(P,)`` reference-stratum labels (``-1`` = excluded).
        min_stratum_patches: Minimum parent-patch count for a stratum to be a
            candidate best match.
        reference_product: If given, attaches a human-readable
            ``stratum_name`` from
            :data:`~geosae.eval.landcover_classes.PRODUCT_CLASSES`.

    Returns:
        Dict with ``nmi``, ``mi``, ``n_valid``, ``weighted_precision``
        (support-weighted mean best-stratum precision, ``nan`` if no
        sub-concept qualifies) and ``subconcepts`` — a list of per-sub-concept
        records (``subconcept``, ``support``, ``stratum_code``,
        ``stratum_name``, ``precision``, ``recall``, ``lift``).
    """
    labels = np.asarray(labels, dtype=np.int64)
    ref = np.asarray(ref, dtype=np.int64)
    if labels.shape != ref.shape:
        raise ValueError(f"labels shape {labels.shape} != ref shape {ref.shape}")
    n_parent = labels.size
    valid = (labels != -1) & (ref != -1)
    table, sub_codes, ref_codes = contingency(labels[valid], ref[valid])
    nmi = normalized_mutual_information(table)
    mi = mutual_information(table)

    ref_counts = {int(b): int((ref == b).sum()) for b in ref_codes if b != -1}
    names = PRODUCT_CLASSES.get(reference_product, {}) if reference_product else {}

    subconcepts = []
    total_support, weighted_sum = 0, 0.0
    for i, k in enumerate(sub_codes.tolist()):
        if k == -1:
            continue
        support = int((labels == k).sum())
        row = table[i]
        row_total = int(row.sum())
        eligible = [
            j for j, b in enumerate(ref_codes.tolist()) if b != -1 and ref_counts.get(b, 0) >= min_stratum_patches
        ]
        if row_total == 0 or not eligible:
            continue
        precisions = row[eligible] / row_total
        best = eligible[int(np.argmax(precisions))]
        b = int(ref_codes[best])
        precision = float(row[best] / row_total)
        n_b = ref_counts[b]
        recall = float(row[best] / n_b) if n_b > 0 else 0.0
        base_rate = n_b / n_parent if n_parent > 0 else 0.0
        lift = float(precision / base_rate) if base_rate > 0 else float("nan")
        subconcepts.append(
            {
                "subconcept": int(k),
                "support": support,
                "stratum_code": b,
                "stratum_name": names.get(b, str(b)),
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }
        )
        total_support += support
        weighted_sum += precision * support

    weighted_precision = weighted_sum / total_support if total_support > 0 else float("nan")
    # Family-wise statistics. Reporting "this parent has a sub-concept that
    # matches a stratum" maximises over sub-concepts as well as over strata, so
    # the max must be carried through the nulls too or its p-value is
    # anticonservative.
    lifts = [s["lift"] for s in subconcepts if np.isfinite(s["lift"])]
    precs = [s["precision"] for s in subconcepts if np.isfinite(s["precision"])]
    return {
        "nmi": nmi,
        "mi": mi,
        "n_valid": int(valid.sum()),
        "weighted_precision": weighted_precision,
        "max_lift": float(max(lifts)) if lifts else float("nan"),
        "max_precision": float(max(precs)) if precs else float("nan"),
        "cluster_sizes": [s["support"] for s in subconcepts],
        "n_degenerate": int(sum(1 for s in subconcepts if s["support"] < min_stratum_patches)),
        "subconcepts": subconcepts,
    }


# ---------------------------------------------------------------------------
# Null generators
# ---------------------------------------------------------------------------


def _grid_block_id(lon: np.ndarray, lat: np.ndarray, deg: float) -> np.ndarray:
    """Flat integer macro-cell id per point on a ``deg`` x ``deg`` lon/lat grid."""
    nbx = max(int(round(360.0 / deg)), 1)
    nby = max(int(round(180.0 / deg)), 1)
    bx = np.clip(((np.asarray(lon) + 180.0) / deg).astype(np.int64), 0, nbx - 1)
    by = np.clip(((np.asarray(lat) + 90.0) / deg).astype(np.int64), 0, nby - 1)
    return by * nbx + bx


def _label_permutation(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of ``labels`` — preserves cluster sizes, destroys geography."""
    return labels[rng.permutation(labels.size)]


def _block_permutation(labels: np.ndarray, block_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap whole blocks' label vectors among same-size blocks.

    Patches are grouped by their ``block_deg`` macro-cell id. Within each
    group of blocks sharing the same patch count, the blocks' *label content*
    is randomly re-paired to a same-size block's *positions* (a permutation of
    same-size blocks) — an exact bijection, so the label multiset is
    preserved exactly. A block whose size is unique among all blocks keeps its
    own labels (nothing else to swap it with). This preserves the label
    partition's spatial clumpiness (each swapped-in vector is still a
    spatially compact block, just moved) while destroying its geographic
    registration to the block it originally covered.
    """
    labels = np.asarray(labels)
    out = labels.copy()
    blocks: dict[int, list[int]] = {}
    for i, b in enumerate(block_id.tolist()):
        blocks.setdefault(b, []).append(i)
    by_size: dict[int, list[int]] = {}
    for b, idxs in blocks.items():
        by_size.setdefault(len(idxs), []).append(b)
    for _, block_ids in by_size.items():
        ids = np.asarray(block_ids)
        src_order = ids[rng.permutation(ids.size)]
        for dst_b, src_b in zip(block_ids, src_order.tolist(), strict=True):
            out[blocks[dst_b]] = labels[blocks[int(src_b)]]
    return out


def _lonlat_to_sphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """``(N, 3)`` unit-sphere xyz coordinates from lon/lat degrees."""
    lam = np.radians(np.asarray(lon, dtype=np.float64))
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


def _kmeans_sphere(xyz: np.ndarray, k: int, rng: np.random.Generator, *, max_iter: int = 50) -> np.ndarray:
    """Lloyd's algorithm (numpy, seeded, random-point init) on unit-sphere coordinates."""
    n = xyz.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(k, n))
    centroids = xyz[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for it in range(max_iter):
        d = ((xyz[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(d, axis=1)
        converged = it > 0 and np.array_equal(new_labels, labels)
        labels = new_labels
        if converged:
            break
        for j in range(k):
            m = labels == j
            if not m.any():
                continue
            c = xyz[m].mean(axis=0)
            norm = np.linalg.norm(c)
            if norm > 0:
                centroids[j] = c / norm
    return labels


def _geographic_kmeans(labels: np.ndarray, lon: np.ndarray, lat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random geographically-coherent partition of the same size as ``labels``.

    K-means (spherical Lloyd's) on the parent patches' lon/lat, with
    ``k = number of distinct sub-concepts`` — a spatial split carrying no
    information about the reference product at all.
    """
    k = np.unique(labels[labels != -1]).size
    return _kmeans_sphere(_lonlat_to_sphere(lon, lat), k, rng)


def _geographic_sizematched(
    labels: np.ndarray, lon: np.ndarray, lat: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A compact geographic partition with *exactly* the observed cluster sizes.

    The binding null. Clusters are grown greedily from random seeds, largest
    first: each takes the ``size_k`` nearest still-unassigned parent patches in
    unit-sphere coordinates. Matching the size distribution matters because
    agreement statistics rise with cluster-size entropy, so
    :func:`_geographic_kmeans`, whose clusters come out near-balanced, is
    upper-biased as a null.

    Args:
        labels: ``(P,)`` observed labels; supplies the cluster sizes to match.
        lon: ``(P,)`` longitudes.
        lat: ``(P,)`` latitudes.
        rng: Seeded generator.

    Returns:
        ``(P,)`` int64 labels over the same patches, ``-1`` where ``labels``
        is ``-1`` (unassigned patches stay unassigned).
    """
    assigned = labels != -1
    idx_all = np.flatnonzero(assigned)
    out = np.full(labels.shape, -1, dtype=np.int64)
    if idx_all.size == 0:
        return out
    _, inv = np.unique(labels[assigned], return_inverse=True)
    sizes = np.bincount(inv)
    xyz = _lonlat_to_sphere(lon, lat)[idx_all]
    free = np.ones(idx_all.size, dtype=bool)
    for cluster in np.argsort(-sizes):
        pool = np.flatnonzero(free)
        if pool.size == 0:
            break
        seed_pos = pool[rng.integers(pool.size)]
        d = ((xyz[pool] - xyz[seed_pos]) ** 2).sum(axis=1)
        take = pool[np.argsort(d, kind="stable")[: int(sizes[cluster])]]
        out[idx_all[take]] = int(cluster)
        free[take] = False
    leftover = np.flatnonzero(free)
    if leftover.size:
        out[idx_all[leftover]] = int(np.argsort(-sizes)[-1])
    return out


def _draw_null(
    labels: np.ndarray, lon: np.ndarray, lat: np.ndarray, *, kind: str, block_deg: float, rng: np.random.Generator
) -> np.ndarray:
    if kind == "label_permutation":
        return _label_permutation(labels, rng)
    if kind == "block_permutation":
        return _block_permutation(labels, _grid_block_id(lon, lat, block_deg), rng)
    if kind == "geographic_kmeans":
        return _geographic_kmeans(labels, lon, lat, rng)
    if kind == "geographic_sizematched":
        return _geographic_sizematched(labels, lon, lat, rng)
    raise ValueError(f"unknown null kind {kind!r}; expected one of {NULL_KINDS}")


def _summarise_null(observed: float, draws: np.ndarray) -> dict:
    draws = np.asarray(draws, dtype=np.float64)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        nan = float("nan")
        return {"mean": nan, "sd": nan, "p95": nan, "p_value": nan, "gap": nan, "z": nan}
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    p95 = float(np.percentile(finite, 95))
    p_value = float((1 + int((finite >= observed).sum())) / (finite.size + 1))
    gap = float(observed - mean) if np.isfinite(observed) else float("nan")
    z = float(gap / sd) if sd > 0 and np.isfinite(gap) else float("nan")
    return {"mean": mean, "sd": sd, "p95": p95, "p_value": p_value, "gap": gap, "z": z}


def _family_wise_minp(obs_lift: np.ndarray, null_lift: np.ndarray) -> dict:
    """Westfall-Young minP over the size-rank-matched per-sub-concept nulls.

    "Parent P has a sub-concept that matches a stratum" maximises over
    sub-concepts, so its p-value needs a multiplicity correction. The obvious
    statistic, ``max_k lift_k`` against the null's own max, is **wrong here**:
    lift falls with cluster size, so a null draw's maximum is almost always
    taken by its smallest cluster while the observed maximum may come from a
    large one, and the comparison silently penalises the observation. Instead
    each sub-concept is tested against its own size-matched null, and the
    smallest of those p-values is calibrated against the distribution of the
    smallest p-value under the null — which controls the family-wise error
    rate without ever comparing clusters of different sizes.

    Args:
        obs_lift: ``(K,)`` observed lift per sub-concept.
        null_lift: ``(n_null, K)`` null lift, column ``j`` rank-matched by
            support to observed sub-concept ``j``.

    Returns:
        Dict with ``per_subconcept_p`` (the uncorrected rank-matched
        p-values), ``min_p`` (their minimum), ``argmin`` (index of the
        sub-concept attaining it), and ``p_value`` — the family-wise-corrected
        p-value for "some sub-concept beats its null".
    """
    n_null, n_sub = null_lift.shape
    if n_sub == 0 or n_null == 0:
        return {"per_subconcept_p": [], "min_p": float("nan"), "argmin": -1, "p_value": float("nan")}
    finite = np.isfinite(null_lift)
    # Uncorrected per-sub-concept p-values, each against its own size-matched null.
    obs_p = np.array(
        [
            (1 + int(np.sum(null_lift[finite[:, j], j] >= obs_lift[j]))) / (int(finite[:, j].sum()) + 1)
            if np.isfinite(obs_lift[j]) and finite[:, j].any()
            else 1.0
            for j in range(n_sub)
        ]
    )
    # Each null draw's own per-sub-concept p-values, against the same columns.
    order = np.argsort(-np.where(finite, null_lift, -np.inf), axis=0)
    ranks = np.empty_like(order)
    rows = np.arange(n_null)[:, None]
    np.put_along_axis(ranks, order, rows, axis=0)
    null_p = (ranks + 1.0) / (n_null + 1.0)
    null_p = np.where(finite, null_p, 1.0)
    min_p_obs = float(obs_p.min())
    min_p_null = null_p.min(axis=1)
    family_p = float((1 + int(np.sum(min_p_null <= min_p_obs))) / (n_null + 1))
    return {
        "per_subconcept_p": [float(p) for p in obs_p],
        "min_p": min_p_obs,
        "argmin": int(np.argmin(obs_p)),
        "p_value": family_p,
    }


def null_scores(
    labels: np.ndarray,
    ref: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    kind: str,
    n_null: int = 200,
    block_deg: float = 5.0,
    seed: int = 0,
    min_stratum_patches: int = 30,
) -> dict:
    """Null distribution of NMI and weighted precision for one null ``kind``.

    Scores the observed ``labels`` once, then draws ``n_null`` surrogate label
    vectors (:func:`_draw_null`) and scores each through the exact same
    :func:`score_partition` pipeline — including its argmax-over-strata
    best-match selection — so the null absorbs that selection's
    multiple-comparison advantage.

    Args:
        labels: ``(P,)`` observed sub-concept labels over the parent patches.
        ref: ``(P,)`` reference-stratum labels over the same patches.
        lon: ``(P,)`` parent-patch longitudes.
        lat: ``(P,)`` parent-patch latitudes.
        kind: One of :data:`NULL_KINDS`.
        n_null: Number of surrogate draws.
        block_deg: Macro-cell size for ``block_permutation``.
        seed: RNG seed (draws are reproducible for a fixed seed).
        min_stratum_patches: Forwarded to :func:`score_partition`.

    Returns:
        Dict with ``kind``, ``n_null``, ``observed_nmi``,
        ``observed_weighted_precision``, and ``nmi``/``weighted_precision``/
        ``max_lift``/``max_precision`` sub-dicts each holding ``mean``, ``sd``,
        ``p95``, ``p_value`` (one-sided, ``P(null >= observed)``), ``gap``
        (observed − mean) and ``z`` (``gap / sd``). ``subconcepts`` holds one
        such pair of ``precision``/``lift`` summaries per observed sub-concept,
        matched to the null cluster of the same rank by size.
    """
    labels = np.asarray(labels, dtype=np.int64)
    ref = np.asarray(ref, dtype=np.int64)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    observed = score_partition(labels, ref, min_stratum_patches=min_stratum_patches)
    obs_subs = observed["subconcepts"]
    n_sub = len(obs_subs)
    # Rank the observed sub-concepts by support so a null draw's k-th largest
    # cluster is compared against the observed k-th largest.
    obs_rank = np.argsort([-s["support"] for s in obs_subs], kind="stable")
    rng = np.random.default_rng(seed)
    nmi_draws = np.empty(n_null)
    prec_draws = np.empty(n_null)
    maxlift_draws = np.empty(n_null)
    maxprec_draws = np.empty(n_null)
    sub_prec = np.full((n_null, n_sub), np.nan)
    sub_lift = np.full((n_null, n_sub), np.nan)
    for i in range(n_null):
        surrogate = _draw_null(labels, lon, lat, kind=kind, block_deg=block_deg, rng=rng)
        s = score_partition(surrogate, ref, min_stratum_patches=min_stratum_patches)
        nmi_draws[i] = s["nmi"]
        prec_draws[i] = s["weighted_precision"]
        maxlift_draws[i] = s["max_lift"]
        maxprec_draws[i] = s["max_precision"]
        null_subs = s["subconcepts"]
        null_rank = np.argsort([-x["support"] for x in null_subs], kind="stable")
        for pos in range(min(n_sub, len(null_subs))):
            sub_prec[i, obs_rank[pos]] = null_subs[null_rank[pos]]["precision"]
            sub_lift[i, obs_rank[pos]] = null_subs[null_rank[pos]]["lift"]
    per_sub = [
        {
            "subconcept": obs_subs[j]["subconcept"],
            "support": obs_subs[j]["support"],
            "precision": _summarise_null(obs_subs[j]["precision"], sub_prec[:, j]),
            "lift": _summarise_null(obs_subs[j]["lift"], sub_lift[:, j]),
        }
        for j in range(n_sub)
    ]
    obs_lift = np.array([s["lift"] for s in obs_subs], dtype=np.float64)
    return {
        "kind": kind,
        "n_null": int(n_null),
        "observed_nmi": observed["nmi"],
        "observed_weighted_precision": observed["weighted_precision"],
        "nmi": _summarise_null(observed["nmi"], nmi_draws),
        "weighted_precision": _summarise_null(observed["weighted_precision"], prec_draws),
        "max_lift": _summarise_null(observed["max_lift"], maxlift_draws),
        "max_precision": _summarise_null(observed["max_precision"], maxprec_draws),
        "family_wise": _family_wise_minp(obs_lift, sub_lift),
        "subconcepts": per_sub,
    }


# ---------------------------------------------------------------------------
# End-to-end validation
# ---------------------------------------------------------------------------


def validate_parent(
    act: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    parent_fractions: np.ndarray,
    parent_codes: np.ndarray,
    parent_code: int,
    reference_fractions: np.ndarray,
    reference_codes: np.ndarray,
    reference_product: str,
    *,
    config: SubconceptConfig | None = None,
    decoder: np.ndarray | None = None,
    n_null: int | None = None,
    n_subconcepts: int | None = None,
) -> dict:
    """Sub-concept discovery + null validation for one parent class x reference product.

    Args:
        act: ``(F, N)`` per-patch mean feature activation.
        lon: ``(N,)`` patch longitudes.
        lat: ``(N,)`` patch latitudes.
        parent_fractions: ``(N, Kp)`` valid-pixel class fractions for the
            parent product (:func:`patch_class_fractions`).
        parent_codes: ``(Kp,)`` class codes matching ``parent_fractions``.
        parent_code: The parent class code to validate.
        reference_fractions: ``(N, Kr)`` valid-pixel class fractions for the
            reference product.
        reference_codes: ``(Kr,)`` class codes matching ``reference_fractions``.
        reference_product: Reference product name (for stratum names + nodata
            lookups upstream).
        config: Thresholds; defaults to :class:`SubconceptConfig`.
        decoder: ``(D, F)`` decoder weight for :func:`dedup_features`, or
            ``None`` to skip deduplication.
        n_null: Null draws per null kind; defaults to ``config.n_null``.
        n_subconcepts: Overrides ``config.max_subconcepts``. Needed to compare a
            trained run against a frozen-decoder control at matched ``K``: the
            frozen control aligns far fewer features (2 vs 6 on forest), and
            agreement statistics depend on the number of clusters.

    Returns:
        Dict with ``parent_code``, ``reference_product``, ``n_patches``,
        ``n_parent``, ``n_aligned``, ``n_kept``, ``dedup_used``,
        ``cluster_sizes``, ``n_degenerate``, ``normalize``, the
        :func:`score_partition` output, and ``nulls`` — ``{key: null_scores
        output}``, one per kind in :data:`NULL_KINDS`, with
        ``block_permutation`` expanded to one entry per ``config.block_deg``
        value keyed ``block_<deg>deg``.
    """
    cfg = config or SubconceptConfig()
    n_null = cfg.n_null if n_null is None else n_null
    max_keep = cfg.max_subconcepts if n_subconcepts is None else int(n_subconcepts)
    n_patches = act.shape[1]

    p_idx = int(np.searchsorted(parent_codes, parent_code))
    has_parent_code = p_idx < parent_codes.size and parent_codes[p_idx] == parent_code
    parent_frac = parent_fractions[:, p_idx] if has_parent_code else np.zeros(n_patches)
    parent_mask = parent_frac >= cfg.parent_purity
    parent_idx = np.where(parent_mask)[0]

    has_ref_codes = bool(reference_fractions.shape[1])
    if has_ref_codes:
        ref_frac_max = reference_fractions.max(axis=1)
        ref_argmax = reference_fractions.argmax(axis=1)
        ref_label_all = np.where(ref_frac_max >= cfg.reference_purity, reference_codes[ref_argmax], -1)
    else:
        ref_label_all = np.full(n_patches, -1, dtype=np.int64)
    ref_parent = ref_label_all[parent_idx]

    aligned, aucs = align_features(act, parent_mask.astype(np.int64), threshold=cfg.align_threshold)
    kept = dedup_features(aligned, aucs, decoder, cosine=cfg.dedup_cosine, max_keep=max_keep)
    sub_labels = subconcept_labels(act, kept, parent_idx, normalize=cfg.normalize)

    scored = score_partition(
        sub_labels, ref_parent, min_stratum_patches=cfg.min_stratum_patches, reference_product=reference_product
    )
    for sc in scored["subconcepts"]:
        sc["auroc_vs_parent"] = float(aucs[int(kept[sc["subconcept"]])])

    block_degs = (cfg.block_deg,) if isinstance(cfg.block_deg, int | float) else tuple(cfg.block_deg)
    nulls = {}
    for kind in NULL_KINDS:
        degs = block_degs if kind == "block_permutation" else (0.0,)
        for deg in degs:
            key = f"block_{deg:g}deg" if kind == "block_permutation" else kind
            nulls[key] = null_scores(
                sub_labels,
                ref_parent,
                lon[parent_idx],
                lat[parent_idx],
                kind=kind,
                n_null=n_null,
                block_deg=deg,
                seed=cfg.seed,
                min_stratum_patches=cfg.min_stratum_patches,
            )

    return {
        "parent_product": cfg.parent_product,
        "parent_code": int(parent_code),
        "parent_name": PRODUCT_CLASSES.get(cfg.parent_product, {}).get(int(parent_code), str(parent_code)),
        "reference_product": reference_product,
        "n_patches": int(n_patches),
        "n_parent": int(parent_idx.size),
        "n_aligned": int(aligned.size),
        "n_kept": int(kept.size),
        "dedup_used": decoder is not None,
        "normalize": cfg.normalize,
        "nmi": scored["nmi"],
        "weighted_precision": scored["weighted_precision"],
        "max_lift": scored["max_lift"],
        "max_precision": scored["max_precision"],
        "cluster_sizes": scored["cluster_sizes"],
        "n_degenerate": scored["n_degenerate"],
        "n_valid": scored["n_valid"],
        "subconcepts": scored["subconcepts"],
        "nulls": nulls,
        "_kept_features": kept,
        "_parent_idx": parent_idx,
        "_labels": sub_labels,
        "_lon_parent": lon[parent_idx],
        "_lat_parent": lat[parent_idx],
    }


def rank_candidates(
    act: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    cells: np.ndarray,
    data: zarr.Group,
    *,
    parent_classes: list[int] | None = None,
    reference_products: tuple[str, ...] = REFERENCE_PRODUCTS,
    config: SubconceptConfig | None = None,
    decoder: np.ndarray | None = None,
) -> list[dict]:
    """Validate every parent class of ``config.parent_product`` against every reference product.

    Each product's per-patch class fractions are read from ``data`` **once**
    and cached, so the sweep over ``len(parent_classes) x len(reference_products)``
    pairs costs one zarr pass per product plus pure-numpy scoring at
    :attr:`SubconceptConfig.n_null_rank` (a reduced null count).

    Args:
        act: ``(F, N)`` per-patch mean feature activation.
        lon: ``(N,)`` patch longitudes.
        lat: ``(N,)`` patch latitudes.
        cells: ``(N,)`` row index into ``data`` per patch.
        data: Open zarr group of the data store.
        parent_classes: Codes to sweep; defaults to every code registered for
            ``config.parent_product`` in
            :data:`~geosae.eval.landcover_classes.PRODUCT_CLASSES`.
        reference_products: Reference products to sweep.
        config: Thresholds; defaults to :class:`SubconceptConfig`.
        decoder: Forwarded to :func:`dedup_features`.

    Returns:
        One flat summary dict per ``(parent_code, reference_product)`` pair,
        sorted by descending ``geographic_kmeans`` weighted-precision gap (the
        hardest null) — the candidates most likely to survive full validation
        come first.
    """
    cfg = config or SubconceptConfig()
    if parent_classes is None:
        parent_classes = sorted(PRODUCT_CLASSES.get(cfg.parent_product, {}))

    parent_fractions, parent_codes, _ = patch_class_fractions(data, cfg.parent_product, cells)
    ref_cache = {p: patch_class_fractions(data, p, cells) for p in reference_products}

    rows: list[dict] = []
    for parent_code in parent_classes:
        for ref_product in reference_products:
            ref_frac, ref_codes, _ = ref_cache[ref_product]
            result = validate_parent(
                act,
                lon,
                lat,
                parent_fractions,
                parent_codes,
                parent_code,
                ref_frac,
                ref_codes,
                ref_product,
                config=cfg,
                decoder=decoder,
                n_null=cfg.n_null_rank,
            )
            geo = result["nulls"][BINDING_NULL]
            rows.append(
                {
                    "parent_product": result["parent_product"],
                    "parent_code": result["parent_code"],
                    "parent_name": result["parent_name"],
                    "reference_product": ref_product,
                    "n_parent": result["n_parent"],
                    "n_aligned": result["n_aligned"],
                    "n_kept": result["n_kept"],
                    "cluster_sizes": result["cluster_sizes"],
                    "n_degenerate": result["n_degenerate"],
                    "nmi": result["nmi"],
                    "weighted_precision": result["weighted_precision"],
                    "max_lift": result["max_lift"],
                    "geo_nmi_gap": geo["nmi"]["gap"],
                    "geo_nmi_p": geo["nmi"]["p_value"],
                    "geo_precision_gap": geo["weighted_precision"]["gap"],
                    "geo_precision_p": geo["weighted_precision"]["p_value"],
                    "geo_max_lift_gap": geo["max_lift"]["gap"],
                    "geo_max_lift_p": geo["max_lift"]["p_value"],
                    "geo_family_min_p": geo["family_wise"]["min_p"],
                    "geo_family_p": geo["family_wise"]["p_value"],
                }
            )

    def _sort_key(r: dict) -> float:
        p = r["geo_family_p"]
        return -p if np.isfinite(p) else np.inf

    rows.sort(key=_sort_key, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SIGNIFICANCE_ALPHA = 0.05


def _is_validated(row: dict) -> bool:
    """A parent x reference pair counts as validated only against the binding null.

    The gate is the family-wise minP p-value (:func:`_family_wise_minp`) under
    :data:`BINDING_NULL` — the size-matched compact geographic null. Naming a
    parent's best sub-concept maximises over sub-concepts *and* over reference
    strata, so only a multiplicity-corrected p-value is honest here; the weaker
    permutation nulls are reported but never decide the verdict.
    """
    return row["n_kept"] >= 2 and np.isfinite(row["geo_family_p"]) and row["geo_family_p"] < SIGNIFICANCE_ALPHA


def write_subconcepts(
    pred_path: str,
    data_zarr: str,
    outdir: str,
    *,
    config: SubconceptConfig | None = None,
    decoder_path: str | None = None,
) -> dict:
    """Sub-concept discovery + null validation for one run, written to ``outdir``.

    Sweeps :func:`rank_candidates` over every class of ``config.parent_product``
    x every :data:`REFERENCE_PRODUCTS`, then re-validates the candidates that
    clear the significance gate (``n_kept >= 2`` and both the
    ``geographic_kmeans`` NMI and weighted-precision one-sided p-values below
    0.05) at full ``config.n_null`` resolution.

    Args:
        pred_path: Prediction zarr (needs ``feat_patch_stats``, ``lon``,
            ``lat``, ``cell``).
        data_zarr: Data store zarr (needs the parent + reference products).
        outdir: Output directory (created if missing).
        config: Thresholds; defaults to :class:`SubconceptConfig`.
        decoder_path: Path to a ``best.pth`` checkpoint (``state["model"]
            ["decoder.weight"]``, ``(D, F)``); used for
            :func:`dedup_features`. ``None`` skips deduplication.

    Returns:
        Dict with ``config``, ``ranking`` (every swept pair) and ``validated``
        (the full-resolution results for the pairs that passed the gate).
    """
    cfg = config or SubconceptConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pf = patch_feature_activation(pred_path)
    act, lon, lat, cells = pf["act"], pf["lon"], pf["lat"], pf["cell"]
    data = zarr.open_group(str(data_zarr), mode="r")

    decoder = None
    if decoder_path is not None:
        import torch

        ckpt = torch.load(str(decoder_path), map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        decoder = np.asarray(state["decoder.weight"].detach().cpu().numpy())

    ranking = rank_candidates(act, lon, lat, cells, data, config=cfg, decoder=decoder)
    _write_ranking_csv(ranking, outdir / "subconcept_ranking.csv")

    parent_fractions, parent_codes, _ = patch_class_fractions(data, cfg.parent_product, cells)
    ref_cache: dict[str, tuple] = {}
    validated: list[dict] = []
    for row in ranking:
        if not _is_validated(row):
            continue
        ref_product = row["reference_product"]
        if ref_product not in ref_cache:
            ref_cache[ref_product] = patch_class_fractions(data, ref_product, cells)
        ref_frac, ref_codes, _ = ref_cache[ref_product]
        validated.append(
            validate_parent(
                act,
                lon,
                lat,
                parent_fractions,
                parent_codes,
                row["parent_code"],
                ref_frac,
                ref_codes,
                ref_product,
                config=cfg,
                decoder=decoder,
            )
        )

    _write_validation_csv(validated, outdir / "subconcept_validation.csv")
    result = {"config": asdict(cfg), "ranking": ranking, "validated": [_strip_private(v) for v in validated]}
    (outdir / "subconcepts.json").write_text(json.dumps(result, indent=2, default=_json_default))

    print(
        f"[geosae] subconcepts: {cfg.parent_product} x {len(REFERENCE_PRODUCTS)} reference products, "
        f"{len(ranking)} candidates, {len(validated)} validated -> {outdir}",
        flush=True,
    )
    return result


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer | np.floating):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _strip_private(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _write_ranking_csv(ranking: list[dict], path: Path) -> None:
    cols = [
        "parent_product",
        "parent_code",
        "parent_name",
        "reference_product",
        "n_parent",
        "n_aligned",
        "n_kept",
        "cluster_sizes",
        "n_degenerate",
        "nmi",
        "weighted_precision",
        "max_lift",
        "geo_nmi_gap",
        "geo_nmi_p",
        "geo_precision_gap",
        "geo_precision_p",
        "geo_max_lift_gap",
        "geo_max_lift_p",
        "geo_family_min_p",
        "geo_family_p",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in ranking:
            out = {c: row[c] for c in cols}
            out["cluster_sizes"] = " ".join(str(s) for s in row["cluster_sizes"])
            w.writerow(out)


def _write_validation_csv(validated: list[dict], path: Path) -> None:
    cols = [
        "parent_product",
        "parent_code",
        "parent_name",
        "reference_product",
        "subconcept",
        "support",
        "stratum_code",
        "stratum_name",
        "precision",
        "recall",
        "lift",
        "auroc_vs_parent",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in validated:
            for sc in r["subconcepts"]:
                w.writerow(
                    {
                        "parent_product": r["parent_product"],
                        "parent_code": r["parent_code"],
                        "parent_name": r["parent_name"],
                        "reference_product": r["reference_product"],
                        **sc,
                    }
                )


__all__ = [
    "PRODUCT_NODATA",
    "DEFAULT_NODATA",
    "REFERENCE_PRODUCTS",
    "NULL_KINDS",
    "SubconceptConfig",
    "patch_class_fractions",
    "patch_feature_activation",
    "align_features",
    "dedup_features",
    "subconcept_labels",
    "contingency",
    "mutual_information",
    "normalized_mutual_information",
    "score_partition",
    "null_scores",
    "validate_parent",
    "rank_candidates",
    "write_subconcepts",
]

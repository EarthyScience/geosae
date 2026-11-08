"""The interpretability panel, assembled from its sufficient statistics.

Every panel scalar belongs to exactly one **block**, and a block answers one question about a
decomposition:

===== =========================== ===================================================
Block Name                        Reads
===== =========================== ===================================================
1     fidelity                    :class:`~geosae.eval.statistic.PanelAccumulator`
2     dictionary geometry         decoder + ``PanelAccumulator``
3     known concepts              ``PanelAccumulator`` + a concept vocabulary
===== =========================== ===================================================

Each ``*_block`` function returns the scalars of one block and nothing else, so a caller assembles
exactly the blocks it needs and the resulting flat dict stays self-describing through
:data:`BLOCK_OF` and :data:`HIGHER_IS_BETTER`. Given a filled accumulator the functions cost
milliseconds. :func:`fill_panel` is the one loop that turns any pixel source plus any encoder into a
filled accumulator, so a trained SAE and a fitted conventional dictionary are scored by the same code.
New blocks extend :data:`BLOCK_TITLES`, :data:`BLOCK_SCALARS` and :data:`HIGHER_IS_BETTER`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np

from geosae.eval.coverage import DEFAULT_TAU, concept_perplexity, emf, n_groups_per_feature, ucov
from geosae.eval.statistic import (
    PanelAccumulator,
    auroc_matrix_from_accumulator,
    calibrate_activation_bins,
    dictionary_geometry_metrics,
    fidelity_metrics,
    unique_argmax_coverage,
)

#: Pixel chunks: ``(raw (n, D), concepts)``, where ``concepts`` is ``None`` for a
#: source with no concept vocabulary, or a ``(concept_mask (n, C) bool,
#: product_valid (n, P) bool)`` pair — membership and, per product, whether that
#: pixel actually carries a label for it (``False`` where the pixel is nodata for
#: that product). A source is a zero-argument callable returning a fresh
#: iterator, because :func:`fill_panel` consumes it twice (calibration, then the fill).
PixelSource = Callable[[], Iterator[tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]]]

BLOCK_TITLES: dict[int, str] = {
    1: "fidelity",
    2: "dictionary geometry",
    3: "known concepts",
}

BLOCK_SCALARS: dict[int, tuple[str, ...]] = {
    1: ("fve", "recon_cosine", "recon_mse", "pixel_cosine", "l0"),
    2: ("max_offdiag_coherence", "near_duplicate_count", "dead_frac", "rate_tail_index"),
    3: (
        "ucov",
        "ucov_star",
        "emf",
        "concept_perplexity",
        "argmax_coverage",
        "ucov_frac_ge_thresh",
    ),
}

BLOCK_OF: dict[str, int] = {name: block for block, names in BLOCK_SCALARS.items() for name in names}

#: Direction of merit per panel scalar: ``True`` higher is better, ``False``
#: lower is better, ``None`` contextual — a description of the dictionary, not a
#: quality claim.
HIGHER_IS_BETTER: dict[str, bool | None] = {
    "fve": True,
    "recon_cosine": True,
    "recon_mse": False,
    "pixel_cosine": True,
    "l0": None,
    "max_offdiag_coherence": False,
    "near_duplicate_count": False,
    "dead_frac": False,
    "rate_tail_index": None,
    "ucov": True,
    "ucov_star": True,
    "emf": True,
    "concept_perplexity": False,
    "argmax_coverage": True,
    "ucov_frac_ge_thresh": True,
}


# -- block 1 ------------------------------------------------------------


def fidelity_block(acc: PanelAccumulator) -> dict[str, float]:
    """Block 1: does the sparse code still stand in for the embedding?

    Args:
        acc: A filled accumulator.

    Returns:
        ``fve``, ``recon_cosine``, ``recon_mse``, ``pixel_cosine`` and ``l0``. The dead-atom fraction
        :func:`~geosae.eval.statistic.fidelity_metrics` also returns belongs to
        block 2 and is reported there rather than twice.
    """
    fid = fidelity_metrics(acc)
    return {k: float(fid[k]) for k in BLOCK_SCALARS[1]}


# -- block 2 ------------------------------------------------------------


def geometry_block(decoder: np.ndarray, acc: PanelAccumulator) -> dict[str, float]:
    """Block 2: is the dictionary itself well formed?

    Args:
        decoder: ``(M, D)`` dictionary.
        acc: A filled accumulator (read only for ``n_f`` and ``n_pixels``).

    Returns:
        ``max_offdiag_coherence``, ``near_duplicate_count``, ``dead_frac`` and
        ``rate_tail_index``.
    """
    geom = dictionary_geometry_metrics(decoder, acc)
    return {
        k: float(geom[k]) for k in ("max_offdiag_coherence", "near_duplicate_count", "dead_frac", "rate_tail_index")
    }


# -- block 3 ------------------------------------------------------------


def concept_block(
    acc: PanelAccumulator,
    *,
    class_table: np.ndarray,
    product_order: Sequence[str],
    class_to_group: Mapping[tuple[str, int], str],
    codes_sample: np.ndarray | None = None,
    tau: float = DEFAULT_TAU,
    min_positive: int = 50,
    n_perm: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Block 3: the label-based metrics, off one folded-AUROC matrix.

    Every scalar but ``ucov_star`` reduces from the folded single-feature
    concept AUROC matrix and the accumulator's contingency counts, so it costs
    no pass over the pixels; ``ucov_star``'s width-fair null permutes labels on
    a dense pool of codes.

    Args:
        acc: A filled accumulator carrying a concept vocabulary.
        class_table: ``(C, 2)`` ``[product index, class id]`` per concept.
        product_order: Product names, indexed by ``class_table[:, 0]``.
        class_to_group: ``(product name, class id) -> group name``, the
            taxonomy EMF counts groups over.
        codes_sample: ``(n, M)`` dense codes for ``ucov_star``'s permutation
            null. Omit and that scalar is ``nan``.
        tau: UCov's AUROC threshold.
        min_positive: Concepts with fewer positive pixels are not scorable.
        n_perm: Permutations per level in the ``ucov_star`` null.
        seed: Seeds the permutation null.

    Returns:
        The six block-3 scalars.
    """
    from geosae.eval.coverage import size_matched_null

    auroc = auroc_matrix_from_accumulator(acc)
    valid = np.asarray(acc.n_c) >= min_positive
    best = auroc.max(axis=0)

    star = float("nan")
    if codes_sample is not None and len(codes_sample) >= 8 and valid.any():
        sizes = np.zeros(auroc.shape[1], dtype=np.int64)
        sizes[valid] = np.minimum(np.asarray(acc.n_c)[valid], len(codes_sample) // 2).astype(np.int64)
        null, _ = size_matched_null(codes_sample, sizes, n_perm=n_perm, seed=seed)
        # A concept the null could not be drawn at is all-nan; median it as nan
        # rather than letting numpy warn once per such concept.
        drawn = np.isfinite(null).any(axis=1)
        level = np.full(null.shape[0], np.nan)
        level[drawn] = np.median(null[drawn], axis=1)
        lift = best - level
        star = float(np.sum(valid & np.isfinite(lift) & (lift >= (tau - 0.5))))

    groups = n_groups_per_feature(auroc, class_table, product_order, class_to_group, tau=tau)
    perplexity = concept_perplexity(acc.n_fc, acc.n_f)
    alive = np.asarray(acc.n_f) > 0
    return {
        "ucov": float(ucov(auroc, tau=tau, valid=valid)),
        "ucov_star": star,
        "emf": float(emf(groups)["emf"]),
        "concept_perplexity": _nanmedian(np.where(alive, perplexity, np.nan)),
        "argmax_coverage": float(unique_argmax_coverage(acc)["coverage"]),
        "ucov_frac_ge_thresh": float(np.mean(auroc[:, valid].max(axis=1) >= tau)) if valid.any() else float("nan"),
    }


# -- filling the accumulator --------------------------------------------


def fill_panel(
    source: PixelSource,
    encode: Callable[[np.ndarray], np.ndarray],
    decoder: np.ndarray,
    *,
    concept_names: Sequence[str] = (),
    product_names: Sequence[str] = (),
    product_of_concept: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    n_bins: int = 64,
    n_calibration_pixels: int = 200_000,
    codes_sample_pixels: int = 0,
    seed: int = 0,
) -> tuple[PanelAccumulator, np.ndarray]:
    """Stream a pixel source through one encoder into a filled accumulator.

    The one loop every decomposition is scored by: a trained SAE and a fitted PCA
    basis differ only in ``encode`` and ``decoder``. The source is consumed **twice** — once to calibrate the
    per-feature activation bins the histograms need, once to fill — which is why
    it is a callable returning a fresh iterator rather than an iterable.

    Args:
        source: Zero-argument callable yielding ``(raw (n, D), concepts)``
            chunks — see :data:`PixelSource`.
        encode: ``(n, D) -> (n, M)`` dense activations.
        decoder: ``(M, D)`` dictionary the reconstruction is formed with.
        concept_names: Length-``C`` concept names; empty for a world with none.
        product_names: Length-``P`` label-product names.
        product_of_concept: ``(C,)`` product index of each concept.
        offset: ``(D,)`` additive term of the reconstruction — a pre-bias, or a
            dense basis's mean. Defaults to zero.
        n_bins: Activation-histogram bins per (feature, concept).
        n_calibration_pixels: Pixels drawn for the bin-edge calibration.
        codes_sample_pixels: Dense code rows retained for block 3's permutation
            null. ``0`` retains none.
        seed: Seeds the retained code subsample.

    Returns:
        ``(accumulator, codes_sample)``, the latter ``(n, M)`` or empty.

    Raises:
        ValueError: The source yielded no pixels.
    """
    decoder = np.asarray(decoder, dtype=np.float64)
    bias = np.zeros(decoder.shape[1]) if offset is None else np.asarray(offset, dtype=np.float64)
    product_of_concept = (
        np.zeros(len(concept_names), dtype=np.int64)
        if product_of_concept is None
        else np.asarray(product_of_concept, dtype=np.int64)
    )

    seen, calibration = 0, []
    for raw, _ in source():
        calibration.append(encode(raw))
        seen += len(raw)
        if seen >= n_calibration_pixels:
            break
    if not calibration:
        raise ValueError("pixel source yielded no pixels")
    edges = calibrate_activation_bins(np.concatenate(calibration, axis=0), n_bins=n_bins)
    del calibration

    acc = PanelAccumulator.create(
        n_features=decoder.shape[0],
        embedding_dim=decoder.shape[1],
        concept_names=list(concept_names),
        product_names=list(product_names),
        product_of_concept=product_of_concept,
        bin_edges=edges,
    )
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    n_kept = 0
    for raw, concepts in source():
        codes = encode(raw)
        ehat = codes.astype(np.float64) @ decoder + bias
        if concepts is None:
            acc.update(raw, ehat, codes)
        else:
            concept_mask, product_valid = concepts
            acc.update(raw, ehat, codes, concept_mask=concept_mask, product_valid=product_valid)
        if n_kept < codes_sample_pixels:
            take = min(codes_sample_pixels - n_kept, len(codes))
            kept.append(codes[rng.choice(len(codes), size=take, replace=False)])
            n_kept += take
    sample = np.concatenate(kept, axis=0) if kept else np.zeros((0, decoder.shape[0]))
    return acc, sample


def _nanmedian(v: np.ndarray) -> float:
    """Median over the finite entries of ``v``, or ``nan`` when there are none."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


__all__ = [
    "BLOCK_OF",
    "BLOCK_SCALARS",
    "BLOCK_TITLES",
    "HIGHER_IS_BETTER",
    "PixelSource",
    "concept_block",
    "fidelity_block",
    "fill_panel",
    "geometry_block",
]

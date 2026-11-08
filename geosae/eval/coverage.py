"""Concept-alignment AUROC matrix + UCov / EMF / aligned concept groups per feature.

For every SAE feature ``f`` and registry concept ``c`` we compute the
**direction-invariant single-feature AUROC** of ``z_f`` as a one-feature
classifier of ``c`` on a **class-balanced** pixel sample, giving an ``(F, C)``
matrix. Every scalar on the monosemanticity axis is a pure reduction of that
matrix:

* **UCov** — number of registry concepts covered by at least one feature at
  ``AUROC >= tau`` (``tau = 0.65`` by default): ``|{c : max_f A[f,c] >= tau}|``.
* **EMF** (effective monosemantic features) — over the concept-**group** taxonomy:
  ``n_alive / max(mean_n_groups, 1)`` where a feature is *alive* if it aligns
  (``>= tau``) with at least one group and ``n_groups(f)`` counts the distinct
  groups it aligns with. A "simple" variant ``sum_f 1/n_groups(f)`` is also
  provided.
* **n_groups_per_feature** — the per-feature aligned-concept-group count over the
  groups of :data:`geosae.eval.concepts.CONCEPT_GROUPS`.
* **threshold sweep** — UCov/EMF recomputed at ``tau in {0.55, 0.65, 0.75, 0.85}``.

The dense reference bases (Raw / PCA / ICA / NMF) are scored with the SAME
:func:`concept_auroc_matrix` — pass their per-pixel factor matrix as ``codes`` — so
their UCov/EMF are directly comparable to the SAEs (see :mod:`geosae.eval.baselines`).

UCov reduces the matrix by ``max_f``, so a dictionary of ``M`` atoms gets ``M``
chances at that maximum and its selection bias grows with ``M``: a wide SAE and a
``D``-dimensional dense basis are not on the same footing. The width-fair
reductions correct for it:

* **best-of-M null** — :func:`permuted_best_auroc` (labels i.i.d. over pixels) and
  :func:`block_permuted_best_auroc` (labels permuted whole cells at a time, so the
  spatial clumping of a real concept survives) give the null distribution of
  ``max_f A[f,c]`` for *this* dictionary at *this* width.
* **null-corrected UCov** — :func:`calibrated_coverage` counts a concept as
  recovered when its observed best beats that null rather than a fixed ``tau``,
  which removes the width term by construction. Against the null *median* plus
  ``tau - 0.5`` it is UCov's own bar with the width taken out; against a small
  upper-tail quantile it is a significance test, which saturates on a large
  pixel sample.
* **matched capacity** — :func:`matched_budget_subset` restricts a wide dictionary
  to its ``M'`` most-used atoms so UCov can be recomputed at the baseline's width.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Default alignment threshold and the sensitivity sweep.
DEFAULT_TAU: float = 0.65
THRESHOLD_SWEEP: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85)


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------


def auc_all_features(features: np.ndarray, binary_labels: np.ndarray) -> np.ndarray:
    """Direction-invariant single-feature AUROC for every column of ``features``.

    Mann-Whitney U via average ranks, folded to ``max(auc, 1 - auc)`` so a feature
    that is *anti*-correlated with the concept scores the same as a positively
    correlated one. Ties are handled by ``scipy.stats.rankdata(method="average")``,
    which is what makes the many tied zeros of a sparse code exact.

    Args:
        features: ``(N, F)`` per-pixel activations (dense; zeros allowed).
        binary_labels: ``(N,)`` in ``{0, 1}`` — concept membership.

    Returns:
        ``(F,)`` folded AUROC in ``[0.5, 1.0]``. Degenerate (all-positive or
        all-negative) samples return ``0.5`` for every feature.
    """
    from scipy.stats import rankdata

    features = np.asarray(features)
    y = np.asarray(binary_labels)
    n_features = features.shape[1]
    pos_mask = y.astype(bool)
    n_pos = int(pos_mask.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(n_features, 0.5)
    ranks = rankdata(features, method="average", axis=0)
    rank_sum_pos = ranks[pos_mask].sum(axis=0).astype(np.float64)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return np.maximum(auc, 1.0 - auc)


def balanced_binary_subsample(
    labels: np.ndarray,
    class_code: int,
    *,
    max_per_side: int = 2500,
    seed: int = 42,
    min_positive: int = 50,
    nodata: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-balanced positive/negative index set for one concept.

    Positives are pixels whose label equals ``class_code``; negatives are any
    labelled pixel of a *different* class (``!= nodata``). Both sides are
    down-sampled without replacement to ``n = min(n_pos, n_neg, max_per_side)`` and
    the concatenation is permuted.

    Returns ``(indices, binary_labels)`` — empty arrays if fewer than
    ``min_positive`` positives exist.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == class_code)[0]
    neg_idx = np.where((labels != class_code) & (labels != nodata))[0]
    empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    if len(pos_idx) < min_positive:
        return empty
    n = min(len(pos_idx), len(neg_idx), max_per_side)
    if n == 0:
        return empty
    pos_sel = rng.choice(pos_idx, size=n, replace=False)
    neg_sel = rng.choice(neg_idx, size=n, replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    perm = rng.permutation(len(idx))
    return idx[perm], y[perm]


def concept_auroc_matrix(
    codes: np.ndarray,
    labels: Mapping[str, np.ndarray],
    class_table: np.ndarray,
    product_order: Sequence[str],
    *,
    max_per_side: int = 2500,
    seed: int = 42,
    min_positive: int = 50,
    balanced: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Assemble the ``(F, C)`` folded single-feature AUROC matrix.

    One balanced sample per concept (seed ``seed + column_index``), then
    :func:`auc_all_features`. ``codes`` is any per-pixel factor matrix — SAE
    codes, or a dense basis (Raw/PCA/ICA/NMF) for the reference baselines.

    Args:
        codes: ``(N, F)`` per-pixel factor activations.
        labels: ``product_name -> (N,) int label array`` (nodata = 0).
        class_table: ``(C, 2)`` int ``[product_id, class_code]`` per concept.
        product_order: product name for each ``product_id`` (matrix column axis).
        max_per_side: Balanced-sample size cap per class, forwarded to
            :func:`balanced_binary_subsample`.
        seed: Base RNG seed (offset per concept column), forwarded to
            :func:`balanced_binary_subsample`.
        min_positive: Minimum positive-pixel count to score a concept,
            forwarded to :func:`balanced_binary_subsample`.
        balanced: Frozen ``concept -> (indices, binary labels)`` draws, e.g. from
            :meth:`geosae.eval.pool.EvalPool.balanced_indices`. When given, they
            are used verbatim and nothing is re-sampled — a concept absent from
            the mapping stays at chance.

    Returns:
        ``(F, C)`` float32; a concept with too few positives / no label array is
        left at ``0.5`` (chance) for every feature.
    """
    codes = np.asarray(codes)
    n_feat = codes.shape[1]
    n_concept = class_table.shape[0]
    out = np.full((n_feat, n_concept), 0.5, dtype=np.float32)
    for c in range(n_concept):
        pid, cls = int(class_table[c, 0]), int(class_table[c, 1])
        prod = product_order[pid]
        arr = labels.get(prod)
        if arr is None:
            continue
        if balanced is None:
            idx, y = balanced_binary_subsample(
                arr, cls, max_per_side=max_per_side, seed=seed + c, min_positive=min_positive
            )
        else:
            idx, y = balanced.get(c, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        if idx.size == 0:
            continue
        out[:, c] = auc_all_features(codes[idx], y).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Pure reductions of the AUROC matrix
# ---------------------------------------------------------------------------


def ucov(auc_matrix: np.ndarray, *, tau: float = DEFAULT_TAU, valid: np.ndarray | None = None) -> int:
    """Number of concepts covered by at least one feature at ``AUROC >= tau``.

    ``valid`` (``(C,)`` bool) restricts the count to a validated concept subset
    (e.g. concepts with sufficient positive support); defaults to all columns.
    """
    a = np.asarray(auc_matrix, dtype=np.float64)
    best = np.nanmax(np.where(np.isnan(a), -np.inf, a), axis=0)  # (C,)
    covered = best >= tau
    if valid is not None:
        covered = covered & np.asarray(valid, dtype=bool)
    return int(covered.sum())


def _class_to_group_id(
    class_table: np.ndarray,
    product_order: Sequence[str],
    class_to_group: Mapping[tuple[str, int], str],
) -> tuple[np.ndarray, list[str]]:
    """Map each concept column to a concept-group id (ungrouped classes are singletons)."""
    group_names: list[str] = []
    name_to_id: dict[str, int] = {}
    ids = np.empty(class_table.shape[0], dtype=np.int64)
    for c in range(class_table.shape[0]):
        pid, cls = int(class_table[c, 0]), int(class_table[c, 1])
        prod = product_order[pid]
        g = class_to_group.get((prod, cls), f"_ungrouped_{prod}_{cls}")
        if g not in name_to_id:
            name_to_id[g] = len(group_names)
            group_names.append(g)
        ids[c] = name_to_id[g]
    return ids, group_names


def n_groups_per_feature(
    auc_matrix: np.ndarray,
    class_table: np.ndarray,
    product_order: Sequence[str],
    class_to_group: Mapping[tuple[str, int], str],
    *,
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """Per-feature count of distinct concept groups aligned at ``AUROC >= tau``.

    Used for polysemanticity scatter/heatmap plots and is the raw material for
    :func:`emf`. Ungrouped classes count as their own singleton group.
    """
    a = np.asarray(auc_matrix)
    group_id, group_names = _class_to_group_id(class_table, product_order, class_to_group)
    n_feat = a.shape[0]
    n_groups = len(group_names)
    aligned = np.where(np.isnan(a), 0.0, a) >= tau  # (F, C)
    per_feat = np.zeros((n_feat, n_groups), dtype=bool)
    # OR each class's alignment into its group column.
    for c in range(a.shape[1]):
        per_feat[:, group_id[c]] |= aligned[:, c]
    return per_feat.sum(axis=1).astype(np.int64)


def emf(n_groups: np.ndarray) -> dict[str, float]:
    """EMF from a per-feature aligned-group count.

    ``n_alive`` = features aligning with >= 1 group; ``mean_n_groups`` averaged over
    those; ``emf = n_alive / max(mean_n_groups, 1.0)``.
    """
    n = np.asarray(n_groups, dtype=np.float64)
    alive = n > 0
    n_alive = int(alive.sum())
    if n_alive == 0:
        return {"n_alive": 0.0, "mean_n_groups": 0.0, "emf": 0.0}
    mean_n_groups = float(n[alive].mean())
    return {"n_alive": float(n_alive), "mean_n_groups": mean_n_groups, "emf": n_alive / max(mean_n_groups, 1.0)}


def emf_simple(n_groups: np.ndarray) -> float:
    """The ``sum_f 1/n_groups(f)`` EMF variant, used by the dense-baseline comparisons."""
    n = np.asarray(n_groups, dtype=np.float64)
    alive = n > 0
    return float((1.0 / n[alive]).sum()) if alive.any() else 0.0


def threshold_sweep(
    auc_matrix: np.ndarray,
    class_table: np.ndarray,
    product_order: Sequence[str],
    class_to_group: Mapping[tuple[str, int], str],
    *,
    thresholds: Sequence[float] = THRESHOLD_SWEEP,
    valid: np.ndarray | None = None,
) -> dict[float, dict[str, float]]:
    """UCov and EMF (grouped + simple) at each threshold in ``thresholds``."""
    out: dict[float, dict[str, float]] = {}
    for tau in thresholds:
        ng = n_groups_per_feature(auc_matrix, class_table, product_order, class_to_group, tau=tau)
        e = emf(ng)
        out[float(tau)] = {
            "ucov": float(ucov(auc_matrix, tau=tau, valid=valid)),
            "emf": e["emf"],
            "emf_simple": emf_simple(ng),
            "n_alive": e["n_alive"],
            "mean_n_groups": e["mean_n_groups"],
        }
    return out


# ---------------------------------------------------------------------------
# Width-fair reductions
# ---------------------------------------------------------------------------


def concept_best_auroc(auc_matrix: np.ndarray, *, valid: np.ndarray | None = None) -> np.ndarray:
    """Per-concept best-feature AUROC, ``max_f A[f, c]``.

    Args:
        auc_matrix: ``(F, C)`` folded AUROC matrix.
        valid: Optional ``(C,)`` bool mask; invalid concepts return ``NaN``.

    Returns:
        ``(C,)`` float64, ``NaN`` where the concept is not valid or has no
        finite entry.
    """
    a = np.asarray(auc_matrix, dtype=np.float64)
    finite = np.isfinite(a)
    best = np.where(finite.any(axis=0), np.nanmax(np.where(finite, a, -np.inf), axis=0), np.nan)
    if valid is not None:
        best = np.where(np.asarray(valid, dtype=bool), best, np.nan)
    return best


def code_activation_rate(codes: np.ndarray) -> np.ndarray:
    """Per-feature fraction of rows with a non-zero code.

    Args:
        codes: ``(N, F)`` per-pixel activations.

    Returns:
        ``(F,)`` float64 in ``[0, 1]``.
    """
    c = np.asarray(codes)
    return (np.abs(c) > 0).mean(axis=0).astype(np.float64)


def concept_perplexity(n_fc: np.ndarray, n_f: np.ndarray) -> np.ndarray:
    """Threshold-free "how many concepts does this feature mix?", ``2**H(C | active_f)``.

    The replacement for a fixed-threshold "primary concept" count: for each
    feature, the entropy of its concept distribution *conditional on being
    active* (``n_fc[f] / n_f[f]``), exponentiated in base 2 so a feature
    split exactly evenly between two concepts scores exactly 2.

    Args:
        n_fc: ``(F, C)`` in-concept active-pixel contingency (e.g.
            :attr:`geosae.eval.statistic.PanelAccumulator.n_fc`).
        n_f: ``(F,)`` per-feature active-pixel count.

    Returns:
        ``(F,)`` float64, ``nan`` where ``n_f == 0``. Concept mass that never
        registers (pixels active but unlabelled for every concept) is dropped
        from the distribution before normalising, so an unlabelled background
        does not inflate the count.
    """
    counts = np.asarray(n_fc, dtype=np.float64)
    support = np.asarray(n_f, dtype=np.float64)
    labelled = counts.sum(axis=1)
    safe = np.where(labelled > 0, labelled, 1.0)
    p = counts / safe[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
    return np.where(support > 0, np.exp2(entropy), np.nan)


def matched_budget_subset(usage: np.ndarray, m_prime: int) -> np.ndarray:
    """Indices of the ``m_prime`` most-used features, ties broken by index.

    Restricting a wide dictionary to its most-used atoms puts it at a baseline's
    width, which is the *capacity* control for the best-of-M selection bias.

    Note:
        Usage is not merit: the ``m_prime`` most-used atoms are not the SAE's
        ``m_prime`` best-aligned ones, so UCov recomputed on this subset is a
        **lower bound** on what the dictionary could reach at that width.

    Args:
        usage: ``(F,)`` per-feature activation rate (see
            :func:`code_activation_rate`).
        m_prime: Number of features to keep; all of them when ``>= F``.

    Returns:
        Sorted-ascending feature indices, ``min(m_prime, F)`` of them.
    """
    u = np.asarray(usage, dtype=np.float64)
    m = int(min(max(m_prime, 0), u.size))
    order = np.argsort(-u, kind="stable")[:m]  # stable sort => ties keep index order
    return np.sort(order).astype(np.int64)


def permuted_best_auroc(
    codes: np.ndarray,
    *,
    n_per_side: int,
    n_perm: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Width-matched null distribution of the best-of-M AUROC, i.i.d. over pixels.

    For a *balanced* design the null distribution of ``max_f AUROC[f, c]`` under
    labels independent of the codes depends only on ``n_per_side`` and the
    dictionary's own feature-correlation structure — not on which concept it is —
    so one null serves every concept scored at that sample size. One fixed
    subsample of ``2 * n_per_side`` rows is drawn and ranked **once**
    (``method="average"``, so the many tied zeros of a sparse code are handled
    exactly, as in :func:`auc_all_features`); every draw then labels half of
    that *same* subsample 1 at random and folds the rank-sum AUROC by ``max`` —
    the standard permutation-test design (fix the data, permute the labels),
    computed for every draw at once as a
    ``(n_perm, 2 * n_per_side) @ (2 * n_per_side, F)`` matmul against the one
    ranking. At Copernicus-FM's :math:`M = 1024` this keeps
    :func:`size_matched_null`'s cost to a fraction of a second per occupied
    size level rather than minutes.

    Warning:
        The labels are drawn i.i.d. over pixels, so this null ignores the spatial
        clumping of real concept labels. Clumping lowers the effective sample
        size and widens the true null, which makes this null
        **anti-conservative** — use :func:`block_permuted_best_auroc` when the
        pixels carry a cell layout.

    Args:
        codes: ``(N, F)`` per-pixel factor activations.
        n_per_side: Positives (and negatives) per draw; match it to the balanced
            sample size the observed statistic used.
        n_perm: Number of permutations.
        seed: RNG seed.

    Returns:
        ``(n_perm,)`` null maxima in ``[0.5, 1.0]``.

    Raises:
        ValueError: If ``2 * n_per_side`` exceeds the number of rows.
    """
    from scipy.stats import rankdata

    codes = np.asarray(codes)
    n_rows = codes.shape[0]
    n_per_side = int(n_per_side)
    n_perm = int(n_perm)
    n_draw = 2 * n_per_side
    if n_draw > n_rows:
        raise ValueError(f"need 2*n_per_side={n_draw} rows, got {n_rows}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=n_draw, replace=False)
    ranks = rankdata(np.asarray(codes[idx]), method="average", axis=0)  # (n_draw, F), ranked once

    # n_perm independent balanced 0/1 label vectors over the n_draw positions:
    # the argsort of one random score per (permutation, position) is a uniform
    # random permutation of the positions, and its first n_per_side entries are
    # a uniform random subset of them -- vectorised over every permutation at
    # once rather than one `rng.permutation` per draw.
    order = np.argsort(rng.random((n_perm, n_draw)), axis=1)
    pos_mask = np.zeros((n_perm, n_draw), dtype=np.float64)
    np.put_along_axis(pos_mask, order[:, :n_per_side], 1.0, axis=1)

    rank_sum_pos = pos_mask @ ranks  # (n_perm, F)
    auc = (rank_sum_pos - n_per_side * (n_per_side + 1) / 2) / (n_per_side * n_per_side)
    return np.maximum(auc, 1.0 - auc).max(axis=1)


def _cell_position_index(cell: np.ndarray) -> np.ndarray:
    """``(n_cells, per_cell)`` row indices, cells in first-appearance order.

    Raises:
        ValueError: If the cells do not all contribute the same pixel count.
    """
    cell = np.asarray(cell)
    order = np.unique(cell)
    counts = np.array([int((cell == c).sum()) for c in order])
    if counts.size == 0 or not (counts == counts[0]).all():
        raise ValueError(
            "block permutation needs an equal number of pixels per cell; "
            f"got counts between {counts.min() if counts.size else 0} and "
            f"{counts.max() if counts.size else 0}"
        )
    return np.stack([np.where(cell == c)[0] for c in order])


def block_permuted_best_auroc(
    codes: np.ndarray,
    cell: np.ndarray,
    binary_labels: np.ndarray,
    *,
    n_perm: int = 50,
    seed: int = 0,
    max_per_side: int = 2500,
) -> np.ndarray:
    """Spatially honest null of the best-of-M AUROC for **one** concept.

    Each permutation relabels the pixels by a random permutation ``pi`` over
    *cells*: the pixel at position ``p`` of cell ``c`` takes the label of position
    ``p`` of cell ``pi(c)``. The within-patch spatial structure of the concept is
    preserved exactly while the code-to-label association is destroyed, so the
    null keeps the clumping that :func:`permuted_best_auroc` throws away. Each
    relabelled draw is then balanced (:func:`balanced_binary_subsample`) and
    reduced by ``max`` exactly like the observed statistic.

    Args:
        codes: ``(N, F)`` per-pixel factor activations.
        cell: ``(N,)`` cell id of each pixel; every cell must contribute the same
            number of pixels, in the same within-patch order (the invariant
            :class:`geosae.eval.registry.RegistrySample` guarantees).
        binary_labels: ``(N,)`` in ``{0, 1}`` for one concept. Every non-positive
            pixel is a negative here — pre-filter nodata if that is not wanted.
        n_perm: Number of permutations.
        seed: RNG seed.
        max_per_side: Balanced-sample cap per class.

    Returns:
        ``(n_perm,)`` null maxima in ``[0.5, 1.0]``.

    Raises:
        ValueError: If the equal-pixels-per-cell invariant fails, or the concept
            has no positive pixel.
    """
    codes = np.asarray(codes)
    y = np.asarray(binary_labels).astype(np.int64)
    grid = _cell_position_index(cell)  # (n_cells, per_cell)
    if int((y == 1).sum()) == 0:
        raise ValueError("binary_labels has no positive pixel")
    label_grid = y[grid]
    rng = np.random.default_rng(seed)
    out = np.empty(int(n_perm), dtype=np.float64)
    for i in range(int(n_perm)):
        permuted = np.empty_like(y)
        permuted[grid] = label_grid[rng.permutation(grid.shape[0])]
        # nodata=-1: the zeros of a binary concept are genuine negatives.
        idx, yy = balanced_binary_subsample(
            permuted, 1, max_per_side=max_per_side, seed=seed + i, min_positive=1, nodata=-1
        )
        out[i] = auc_all_features(codes[idx], yy).max() if idx.size else np.nan
    return out


def calibrated_coverage(
    best_auroc: np.ndarray,
    null_best: np.ndarray,
    *,
    alpha: float = 0.05,
    margin: float = 0.0,
    valid: np.ndarray | None = None,
) -> dict:
    """Width-fair coverage: concepts whose best feature beats its own null.

    A concept is *recovered* when its observed best-feature AUROC exceeds the
    ``1 - alpha`` quantile of the best-of-M null for the same dictionary and the
    same balanced sample size, by at least ``margin``. Because the null carries
    the width, the result is comparable across dictionary widths where UCov is
    not. The two settings answer different questions:

    * ``alpha`` small, ``margin = 0`` — a **significance** test: is this concept
      aligned at all, beyond best-of-M luck?
    * ``alpha = 0.5``, ``margin > 0`` — an **effect-size** test against the null
      *median*: is the alignment at least ``margin`` better than a random
      dictionary of the same width would manage? With ``margin = tau - 0.5``
      this is UCov's own bar, re-expressed relative to the null instead of to
      chance, and it is the reduction to prefer at large pixel samples.

    Note:
        The significance form **saturates** on a large sample: with thousands of
        pixels per concept, a tiny but real alignment clears its null, so every
        basis scores near 1. That is this metric's main failure mode — read it
        together with ``mean_excess``, or use the effect-size form.

    Args:
        best_auroc: ``(C,)`` observed per-concept best-feature AUROC; ``NaN``
            marks a concept that was not scored.
        null_best: ``(n_perm,)`` shared null, or ``(C, n_perm)`` per-concept nulls.
        alpha: Upper-tail level of the null quantile used as the reference
            (``0.5`` = the null median).
        margin: Additional AUROC the observed best must clear the reference by.
        valid: Optional ``(C,)`` bool mask of scoreable concepts; combined with
            the finite entries of ``best_auroc``.

    Returns:
        ``ccov`` (fraction of valid concepts recovered, in ``[0, 1]``),
        ``ccov_count``, ``n_valid``, ``alpha``, ``margin``, ``mean_excess``
        (mean of observed minus null *median*, the threshold-free effect size),
        ``median_null_best`` and ``null_quantile`` (means over concepts when the
        null is per-concept).
    """
    best = np.asarray(best_auroc, dtype=np.float64)
    null = np.asarray(null_best, dtype=np.float64)
    if null.ndim == 1:
        null = np.broadcast_to(null, (best.size, null.size))
    if null.shape[0] != best.size:
        raise ValueError(f"null_best has {null.shape[0]} rows, expected {best.size}")
    ok = np.isfinite(best)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    # An unscored concept carries an all-NaN null row; reduce only the scored
    # ones so the aggregate does not warn on (and average over) empty slices.
    thresh = np.full(best.size, np.nan)
    median_null = np.full(best.size, np.nan)
    has_null = np.isfinite(null).any(axis=1)
    if has_null.any():
        thresh[has_null] = np.nanquantile(null[has_null], 1.0 - float(alpha), axis=1)
        median_null[has_null] = np.nanmedian(null[has_null], axis=1)
    ok &= has_null
    n_valid = int(ok.sum())
    if n_valid == 0:
        return {
            "ccov": 0.0,
            "ccov_count": 0,
            "n_valid": 0,
            "alpha": float(alpha),
            "margin": float(margin),
            "mean_excess": float("nan"),
            "median_null_best": float("nan"),
            "null_quantile": float("nan"),
        }
    recovered = ok & (best > thresh + float(margin))
    return {
        "ccov": float(recovered.sum()) / n_valid,
        "ccov_count": int(recovered.sum()),
        "n_valid": n_valid,
        "alpha": float(alpha),
        "margin": float(margin),
        "mean_excess": float(np.mean(best[ok] - median_null[ok])),
        "median_null_best": float(np.nanmean(median_null[ok])),
        "null_quantile": float(np.nanmean(thresh[ok])),
    }


def balanced_sample_sizes(
    labels: Mapping[str, np.ndarray],
    class_table: np.ndarray,
    product_order: Sequence[str],
    *,
    max_per_side: int = 2500,
) -> np.ndarray:
    """``(C,)`` per-side sample size ``min(n_pos, n_neg, max_per_side)`` per concept.

    The size :func:`concept_auroc_matrix` actually draws, and therefore the size a
    best-of-M null must be matched to.
    """
    out = np.zeros(class_table.shape[0], dtype=np.int64)
    for c in range(class_table.shape[0]):
        arr = labels.get(product_order[int(class_table[c, 0])])
        if arr is None:
            continue
        arr = np.asarray(arr)
        cls = int(class_table[c, 1])
        n_pos = int((arr == cls).sum())
        n_neg = int(((arr != cls) & (arr != 0)).sum())
        out[c] = min(n_pos, n_neg, int(max_per_side))
    return out


def size_matched_null(
    codes: np.ndarray,
    sizes: np.ndarray,
    *,
    n_perm: int = 200,
    n_levels: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Best-of-M nulls at a few sample sizes, mapped onto every concept.

    The null distribution of ``max_f AUROC[f, c]`` tightens as the balanced
    sample grows, so a concept scored on 200 pixels per side and one scored on
    2500 do not share a null. Computing ``C`` separate nulls is wasteful, and one
    null at the median size is anti-conservative for the small concepts. This
    quantises the per-side sizes onto ``n_levels`` log-spaced levels, draws one
    null per occupied level, and returns the per-concept assignment.

    Args:
        codes: ``(N, F)`` factor activations to draw the null from. Pass only the
            rows the observed statistic could have drawn (i.e. labelled pixels).
        sizes: ``(C,)`` per-side balanced sample size, ``0`` where not scored
            (see :func:`balanced_sample_sizes`).
        n_perm: Permutations per level.
        n_levels: Maximum number of distinct sizes the null is drawn at.
        seed: Base RNG seed (offset per level).

    Returns:
        ``(null, level_size)`` — ``null`` is ``(C, n_perm)``, ``NaN`` for
        unscored concepts; ``level_size`` is ``(C,)`` the size each concept's
        null was drawn at, ``0`` where unscored.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    n_rows = int(np.asarray(codes).shape[0])
    scored = sizes > 0
    null = np.full((sizes.size, int(n_perm)), np.nan, dtype=np.float64)
    level_size = np.zeros(sizes.size, dtype=np.int64)
    if not scored.any() or n_rows < 4:
        return null, level_size

    s = sizes[scored]
    edges = np.unique(np.round(np.exp(np.linspace(np.log(s.min()), np.log(s.max()), int(n_levels)))).astype(np.int64))
    assigned = edges[np.abs(s[:, None] - edges[None, :]).argmin(axis=1)]
    level_size[scored] = assigned
    for i, lv in enumerate(np.unique(assigned)):
        n_per_side = int(max(2, min(int(lv), n_rows // 2)))
        draws = permuted_best_auroc(codes, n_per_side=n_per_side, n_perm=n_perm, seed=seed + i)
        rows = np.zeros(sizes.size, dtype=bool)
        rows[scored] = assigned == lv
        null[rows] = draws
    return null, level_size


def coverage_report(
    codes: np.ndarray,
    labels: Mapping[str, np.ndarray],
    class_table: np.ndarray,
    product_order: Sequence[str],
    *,
    class_to_group: Mapping[tuple[str, int], str] | None = None,
    valid: np.ndarray | None = None,
    tau: float = DEFAULT_TAU,
    max_per_side: int = 2500,
    min_positive: int = 50,
    seed: int = 42,
    n_perm: int = 200,
    n_null_levels: int = 4,
    alpha: float = 0.05,
    balanced: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict, dict]:
    """Score one dictionary: the raw UCov/EMF panel plus its width-fair companions.

    The best-of-M null is matched to each concept's own balanced sample size
    (:func:`size_matched_null`) and drawn from the *labelled* pixels only — the
    same pool the observed statistic samples from. Carrying the width inside the
    null is what makes ``ccov`` comparable across dictionary widths.

    Args:
        codes: ``(N, F)`` per-pixel factor activations (SAE codes or a dense basis).
        labels: ``product name -> (N,) int`` label arrays, nodata ``0``.
        class_table: ``(C, 2)`` int ``[product_id, class_code]``.
        product_order: Product name per ``product_id``.
        class_to_group: ``(product, code) -> group``; defaults to
            :func:`default_class_to_group`.
        valid: Optional ``(C,)`` bool mask of scoreable concepts.
        tau: Alignment threshold for UCov / EMF.
        max_per_side: Balanced-sample cap per concept side.
        min_positive: Minimum positives for a concept to be scored.
        seed: Base RNG seed for the balanced samples and the null.
        n_perm: Permutations per null level.
        n_null_levels: Distinct sample sizes the null is drawn at.
        alpha: Significance level of :func:`calibrated_coverage`.
        balanced: Frozen per-concept balanced draws (see
            :func:`concept_auroc_matrix`). When given, nothing is re-sampled and
            the null is matched to the frozen sample sizes.

    Returns:
        ``(scalars, arrays)`` — a flat json-serialisable dict, and the
        ``auc_matrix`` ``(F, C)``, ``best_auroc`` ``(C,)``, ``null_best``
        ``(C, n_perm)`` and ``activation_rate`` ``(F,)`` arrays behind it.
    """
    codes = np.asarray(codes)
    c2g = default_class_to_group() if class_to_group is None else class_to_group
    auc = concept_auroc_matrix(
        codes,
        labels,
        class_table,
        product_order,
        max_per_side=max_per_side,
        seed=seed,
        min_positive=min_positive,
        balanced=balanced,
    )
    if balanced is None:
        sizes = balanced_sample_sizes(labels, class_table, product_order, max_per_side=max_per_side)
    else:
        # The null must match the size the observed statistic was scored at,
        # which for a frozen draw is the draw's own size, not the store's.
        sizes = np.zeros(class_table.shape[0], dtype=np.int64)
        for c, (idx, _) in balanced.items():
            sizes[int(c)] = idx.size // 2
    scoreable = sizes > 0
    if valid is not None:
        scoreable &= np.asarray(valid, dtype=bool)

    # Negatives are drawn from labelled pixels only, so the null must be too.
    is_labelled = np.zeros(codes.shape[0], dtype=bool)
    for arr in labels.values():
        is_labelled |= np.asarray(arr) != 0
    pool = codes[is_labelled] if is_labelled.any() else codes

    best = concept_best_auroc(auc, valid=scoreable)
    null, level_size = size_matched_null(
        pool, np.where(scoreable, sizes, 0), n_perm=n_perm, n_levels=n_null_levels, seed=seed
    )
    cal = calibrated_coverage(best, null, alpha=alpha, valid=scoreable)
    # UCov's own bar (tau over chance) re-expressed against the width-matched
    # null median instead of against 0.5 — same semantics, no width term.
    star = calibrated_coverage(best, null, alpha=0.5, margin=tau - 0.5, valid=scoreable)
    ng = n_groups_per_feature(auc, class_table, product_order, c2g, tau=tau)
    e = emf(ng)
    n_valid = int(scoreable.sum())

    scalars = {
        "n_features": int(codes.shape[1]),
        "n_pixels": int(codes.shape[0]),
        "n_concepts": int(class_table.shape[0]),
        "n_valid_concepts": n_valid,
        "tau": float(tau),
        "ucov": float(ucov(auc, tau=tau, valid=scoreable)),
        "ucov_frac": float(ucov(auc, tau=tau, valid=scoreable)) / n_valid if n_valid else float("nan"),
        "emf": e["emf"],
        "emf_simple": emf_simple(ng),
        "n_alive": e["n_alive"],
        "mean_n_groups": e["mean_n_groups"],
        "null_n_per_side": float(np.median(level_size[scoreable])) if n_valid else float("nan"),
        "null_pool_pixels": int(pool.shape[0]),
        "median_best_auroc": float(np.nanmedian(best)) if n_valid else float("nan"),
        "ucov_null_corrected": float(star["ccov_count"]),
        "ucov_null_corrected_frac": star["ccov"],
        **{k: v for k, v in cal.items()},
    }
    for t, row in threshold_sweep(auc, class_table, product_order, c2g, valid=scoreable).items():
        for k, v in row.items():
            scalars[f"{k}_tau{t:g}"] = float(v)
    arrays = {
        "auc_matrix": auc,
        "best_auroc": best,
        "null_best": null,
        "null_level_size": level_size,
        "balanced_size": sizes,
        "activation_rate": code_activation_rate(codes),
        "valid_concepts": scoreable,
    }
    return scalars, arrays


def default_class_to_group() -> dict[tuple[str, int], str]:
    """The ``(product, code) -> concept group`` map from :data:`concepts.CONCEPT_GROUPS`."""
    from geosae.eval.concepts import CONCEPT_GROUPS

    return {key: g for g, members in CONCEPT_GROUPS.items() for key in members}


__all__ = [
    "DEFAULT_TAU",
    "THRESHOLD_SWEEP",
    "auc_all_features",
    "balanced_binary_subsample",
    "concept_auroc_matrix",
    "ucov",
    "n_groups_per_feature",
    "emf",
    "emf_simple",
    "threshold_sweep",
    "concept_best_auroc",
    "concept_perplexity",
    "code_activation_rate",
    "matched_budget_subset",
    "permuted_best_auroc",
    "block_permuted_best_auroc",
    "calibrated_coverage",
    "balanced_sample_sizes",
    "size_matched_null",
    "coverage_report",
    "default_class_to_group",
]

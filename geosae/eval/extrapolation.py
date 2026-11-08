"""Range and combination checks of a held-out set against a labelled set.

A distance to the labelled set says *how far* a point is; a range check per unit says
*along which unit* it leaves the labelled range. On a sparse code whose units are
concepts, that unit is the reason a point is out of the area of applicability.

* :func:`mess` is the multivariate environmental similarity surface of Elith, Kearney
  & Phillips (2010): per unit, the similarity of a value to the labelled distribution
  (100 at the median, 0 at the extremes, negative outside the range in percent of the
  range); per point, its minimum and the unit attaining it (the most dissimilar unit).
* :func:`coverage_gap` is the dataset-level version: per unit, the share of an
  unlabelled sample that exceeds the labelled maximum.
* :func:`tail_probability` is the rank form of the one-sided range check: per unit,
  the share of labelled values at or above a point's value. It reads the labelled
  distribution rather than its maximum, so a few labelled points on which a unit
  happens to fire do not hide every later exceedance.
* :func:`mahalanobis` is the combination check (the RX detector of hyperspectral
  imaging, the NT2 statistic of Mesgaran et al. 2014): a shrunk Mahalanobis distance to
  the labelled mean, which also sees values that are each inside their range but
  combined in a way the labels never showed, and directions they never varied in.
* :func:`best_unit` asks whether a dictionary has a unit for a concept at all: the unit
  whose activation alone separates the concept's points best. A range check can name a
  concept only through such a unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class MessResult:
    """MESS of a set of points against a labelled set.

    Attributes:
        similarity: ``(n,)`` minimum per-unit similarity (negative = outside the range).
        most_dissimilar: ``(n,)`` index of the unit attaining the minimum.
        per_unit: ``(n, F)`` similarity of every unit.
    """

    similarity: np.ndarray
    most_dissimilar: np.ndarray
    per_unit: np.ndarray


def _check(labelled: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labelled, other = np.asarray(labelled, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if labelled.ndim != 2 or other.ndim != 2 or labelled.shape[1] != other.shape[1]:
        raise ValueError(f"expected (n, F) arrays with equal F, got {labelled.shape} and {other.shape}")
    return labelled, other


def mess(labelled: np.ndarray, test: np.ndarray, *, one_sided: bool = False) -> MessResult:
    """Multivariate environmental similarity of ``test`` to ``labelled``, unit by unit.

    Args:
        labelled: ``(n_labelled, F)`` reference values (the downstream training set).
        test: ``(n, F)`` values to score.
        one_sided: Score only the upper side: values at or below the labelled maximum
            get similarity 100, values above it ``-(value - max) / range * 100``. The
            reading for a non-negative sparse code, whose zeros are never novel.

    Returns:
        A :class:`MessResult`. A unit constant in ``labelled`` is scored against a range of
        ``1e-6`` times the widest unit's, so any departure from its constant dominates.

    Raises:
        ValueError: The arrays are not 2-D with the same number of units.
    """
    labelled, test = _check(labelled, test)
    lo, hi = labelled.min(axis=0), labelled.max(axis=0)
    span = hi - lo
    # A unit constant in the labels (a feature they never activate) has no range; a floor far
    # below every real range keeps its similarity finite and ranks it as the most dissimilar.
    span = np.maximum(span, 1e-6 * max(float(span.max()), np.finfo(np.float64).tiny))
    if one_sided:
        per_unit = np.where(test > hi, (hi - test) / span * 100.0, 100.0)
    else:
        ordered = np.sort(labelled, axis=0)
        below = np.stack(
            [np.searchsorted(ordered[:, j], test[:, j], side="left") for j in range(labelled.shape[1])], axis=1
        )
        f = 100.0 * below / labelled.shape[0]
        per_unit = np.where(f <= 50.0, 2.0 * f, 2.0 * (100.0 - f))
        per_unit = np.where(f == 0.0, (test - lo) / span * 100.0, per_unit)
        per_unit = np.where(test > hi, (hi - test) / span * 100.0, per_unit)
    most = per_unit.argmin(axis=1)
    return MessResult(similarity=per_unit[np.arange(len(test)), most], most_dissimilar=most, per_unit=per_unit)


def coverage_gap(labelled: np.ndarray, unlabelled: np.ndarray) -> np.ndarray:
    """Share of ``unlabelled`` rows exceeding the labelled maximum, per unit.

    Args:
        labelled: ``(n_labelled, F)`` codes of the downstream training set.
        unlabelled: ``(n, F)`` codes of the region the model is applied to (or the world).

    Returns:
        ``(F,)`` gaps in ``[0, 1]``; a unit the labels never activate scores its firing
        rate on ``unlabelled``, a unit the labels span scores 0.

    Raises:
        ValueError: The arrays are not 2-D with the same number of units.
    """
    labelled, unlabelled = _check(labelled, unlabelled)
    return (unlabelled > labelled.max(axis=0)).mean(axis=0)


@dataclass(frozen=True)
class TailResult:
    """Rank-based one-sided range check of a set of points against a labelled set.

    Attributes:
        probability: ``(n,)`` smallest upper-tail probability over the point's active
            units, ``(1 + #{labelled >= value}) / (n_labelled + 1)``; 1 for a point with
            no active unit.
        unit: ``(n,)`` index of the unit attaining it (0 when no unit is active).
    """

    probability: np.ndarray
    unit: np.ndarray


def tail_probability(labelled: np.ndarray, test: np.ndarray) -> TailResult:
    """Smallest empirical upper-tail probability of each test point over its active units.

    For unit ``j`` and a test value ``v > 0``, the tail probability is the conformal
    p-value ``(1 + #{i: labelled[i, j] >= v}) / (n + 1)``; for a fixed unit it is valid
    under exchangeability, ``P(p <= t) <= t``. A unit the labels never activate gives
    ``1 / (n + 1)`` as soon as it fires. Values at or below zero are never novel, the
    reading for a non-negative sparse code. The minimum over units is *not* corrected
    for testing many units; use it as a ranking, or compare it with ``alpha / F``.

    Args:
        labelled: ``(n_labelled, F)`` reference values (the downstream training set).
        test: ``(n, F)`` values to score.

    Returns:
        A :class:`TailResult`.

    Raises:
        ValueError: The arrays are not 2-D with the same number of units.
    """
    labelled, test = _check(labelled, test)
    n = labelled.shape[0]
    ordered = np.sort(labelled, axis=0)
    prob = np.ones(test.shape[0])
    unit = np.zeros(test.shape[0], dtype=np.int64)
    for j in range(test.shape[1]):
        rows = np.flatnonzero(test[:, j] > 0)
        if rows.size == 0:
            continue
        at_or_above = n - np.searchsorted(ordered[:, j], test[rows, j], side="left")
        p = (1.0 + at_or_above) / (n + 1.0)
        better = p < prob[rows]
        prob[rows[better]] = p[better]
        unit[rows[better]] = j
    return TailResult(probability=prob, unit=unit)


def mahalanobis(labelled: np.ndarray, test: np.ndarray, *, shrinkage: float = 1.0) -> np.ndarray:
    """Squared Mahalanobis distance of each test point to the labelled set, with ridge shrinkage.

    ``(x - mu)^T (C + lambda I)^{-1} (x - mu)``, with ``mu`` and ``C`` the labelled mean and
    covariance and ``lambda = shrinkage * trace(C) / F``. The shrinkage keeps a direction
    the labels never vary in finite: an offset ``delta`` along it scores
    ``delta**2 / lambda``, far above an offset of the same size along a direction the
    labels span. It is also the posterior predictive variance, up to scale, of a ridge
    head on these features with an isotropic prior on its coefficients. Invariant to a
    rotation and a common rescaling of the features.

    Args:
        labelled: ``(n_labelled, F)`` reference values.
        test: ``(n, F)`` values to score.
        shrinkage: Ridge added to the covariance, in units of its mean eigenvalue. 0 gives
            the classical distance (the covariance must then be invertible).

    Returns:
        ``(n,)`` non-negative squared distances.

    Raises:
        ValueError: The arrays are not 2-D with the same number of units, or
            ``shrinkage`` is negative.
    """
    if shrinkage < 0:
        raise ValueError(f"shrinkage must be non-negative, got {shrinkage}")
    labelled, test = _check(labelled, test)
    mu = labelled.mean(axis=0)
    cov = np.atleast_2d(np.cov(labelled - mu, rowvar=False))
    f = cov.shape[0]
    lam = shrinkage * float(np.trace(cov)) / f
    offset = test - mu
    solved = np.linalg.solve(cov + lam * np.eye(f), offset.T).T
    return np.maximum(np.einsum("nf,nf->n", offset, solved), 0.0)


def best_unit(code: np.ndarray, positive: np.ndarray, *, min_firing: float = 0.05) -> tuple[int, float]:
    """The unit whose activation alone best separates the positive rows, and its AUROC.

    Whether a dictionary has a unit for a concept: each unit's activation is read as a
    score for the concept (higher is more likely positive; ties count one half) and the
    unit with the largest AUROC wins. Units firing on fewer than ``min_firing`` of the
    positive rows are skipped, so a unit that happens to fire on a handful of them
    cannot win on a small sample.

    Args:
        code: ``(n, F)`` non-negative activations.
        positive: ``(n,)`` boolean concept labels, both classes present.
        min_firing: Share of the positive rows a unit must be active on to compete.

    Returns:
        ``(unit, auroc)``; ``(-1, nan)`` when no unit is active on enough positive rows.

    Raises:
        ValueError: ``code`` is not 2-D, ``positive`` does not have one entry per row, or
            one class is empty.
    """
    code = np.asarray(code, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool).ravel()
    if code.ndim != 2 or positive.size != code.shape[0]:
        raise ValueError(f"need (n, F) code and (n,) labels, got {code.shape} and {positive.shape}")
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("best_unit needs both positive and negative rows")
    cand = np.flatnonzero((code[positive] > 0).mean(axis=0) >= min_firing)
    cand = cand[(code[positive][:, cand] > 0).any(axis=0)]
    if cand.size == 0:
        return -1, float("nan")
    ranks = rankdata(code[:, cand], axis=0)
    auroc = (ranks[positive].sum(axis=0) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    i = int(np.argmax(auroc))
    return int(cand[i]), float(auroc[i])


__all__ = ["MessResult", "TailResult", "best_unit", "coverage_gap", "mahalanobis", "mess", "tail_probability"]

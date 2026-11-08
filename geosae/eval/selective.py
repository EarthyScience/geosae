"""Decision-level scores of an applicability index: risk-coverage AUC and conformal selection.

A rank correlation between an applicability score and the held-out error says the
score orders pixels better than chance; it does not say what using the score as
a mask buys. The two statistics here answer that directly.

* **Risk-coverage AUC** (AURC): keep the pixels with the lowest score first and
  track the mean error of what is kept; the area under that curve is small when
  the score removes the worst pixels first. :attr:`RiskCoverage.gain` places it
  between an uninformative score (0) and the error itself (1).
* **Stratified AUROC**: how well a score separates positives (pixels carrying a
  held-out concept, say) *within* bins of a reference score (the raw distance), so it
  measures only what the score adds to the reference.
* **Silent share**: the fraction of positives a reference score leaves unflagged at a
  set false-alarm rate on the negatives; what is left for another score to find.
* **Conformal selection**: calibrate an absolute-error interval and a score
  threshold on held-in pixels at one level ``1 - alpha``, keep the held-out
  pixels under the threshold, and report how many of them the interval covers
  and what share of the map was kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskCoverage:
    """Area under the risk-coverage curve of one score, with its two reference values.

    Attributes:
        aurc: Mean over coverage levels of the mean error of the retained pixels.
        random: The same for an uninformative score (the mean error).
        oracle: The same for the error itself as the score (the lowest attainable).
    """

    aurc: float
    random: float
    oracle: float

    @property
    def gain(self) -> float:
        """``(random - aurc) / (random - oracle)``: 1 perfect, 0 uninformative, < 0 inverted."""
        span = self.random - self.oracle
        return float((self.random - self.aurc) / span) if span > 0 else float("nan")


@dataclass(frozen=True)
class SelectiveCoverage:
    """Conformal coverage on the pixels an applicability threshold retains.

    Attributes:
        alpha: Miscoverage level; the target coverage is ``1 - alpha``.
        halfwidth: Calibrated absolute-error interval half-width.
        threshold: Calibrated score threshold; pixels at or below it are retained.
        coverage: Share of retained held-out pixels whose error is within ``halfwidth``.
        area: Share of held-out pixels retained.
        coverage_all: Share of all held-out pixels covered, with no selection.
    """

    alpha: float
    halfwidth: float
    threshold: float
    coverage: float
    area: float
    coverage_all: float


def _check_pair(score: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()
    if score.shape != err.shape or score.size == 0:
        raise ValueError(f"score and err must be non-empty and of equal length, got {score.size} and {err.size}")
    if not (np.isfinite(score).all() and np.isfinite(err).all()):
        raise ValueError("score and err must be finite")
    return score, err


def _tie_averaged(score: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Errors sorted by ascending score, each tied block replaced by its mean.

    The cumulative error is linear in the retained set, so this gives the
    risk-coverage curve averaged over every order of the tied pixels.
    """
    order = np.argsort(score, kind="stable")
    s, e = score[order], err[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    sums = np.add.reduceat(e, starts)
    counts = np.diff(np.r_[starts, s.size])
    return np.repeat(sums / counts, counts)


def risk_coverage_curve(score: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve of retaining pixels in ascending order of ``score``.

    Args:
        score: ``(N,)`` applicability score, higher meaning less applicable.
        err: ``(N,)`` non-negative error of each pixel.

    Returns:
        ``(coverage, risk)``, both ``(N,)``: after retaining the ``i + 1``
        lowest-scored pixels, ``coverage[i] = (i + 1) / N`` and ``risk[i]`` is their
        mean error, averaged over the order of tied scores.
    """
    score, err = _check_pair(score, err)
    n = np.arange(1, score.size + 1)
    return n / score.size, np.cumsum(_tie_averaged(score, err)) / n


def risk_coverage(score: np.ndarray, err: np.ndarray) -> RiskCoverage:
    """Area under the risk-coverage curve of ``score``, with its random and oracle values.

    Args:
        score: ``(N,)`` applicability score, higher meaning less applicable.
        err: ``(N,)`` non-negative error of each pixel.

    Returns:
        The :class:`RiskCoverage`; its ``gain`` is comparable across error scales.
    """
    score, err = _check_pair(score, err)
    n = np.arange(1, err.size + 1)
    oracle = float(np.mean(np.cumsum(np.sort(err)) / n))
    return RiskCoverage(
        aurc=float(np.mean(risk_coverage_curve(score, err)[1])), random=float(err.mean()), oracle=oracle
    )


def auroc(score: np.ndarray, positive: np.ndarray, weight: np.ndarray | None = None) -> float | np.ndarray:
    """Mann-Whitney AUROC with ties counted one half; ``nan`` without both classes.

    Args:
        score: ``(N,)`` score, higher meaning more likely positive.
        positive: ``(N,)`` boolean labels.
        weight: Optional ``(N,)`` or ``(B, N)`` non-negative sample weights (e.g. block-bootstrap
            multiplicities); a weight of ``k`` counts a sample ``k`` times. One sort serves all ``B`` rows.

    Returns:
        The AUROC, or ``(B,)`` AUROCs for a ``(B, N)`` weight.
    """
    if weight is not None:
        return _weighted_auroc(np.asarray(score), np.asarray(positive, bool), np.asarray(weight, np.float64))
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    counts = np.diff(np.r_[starts, s.size])
    mid = np.repeat(starts + (counts + 1) / 2.0, counts)  # average 1-based rank of each tie block
    ranks = np.empty(score.size)
    ranks[order] = mid
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _weighted_auroc(score: np.ndarray, positive: np.ndarray, weight: np.ndarray) -> float | np.ndarray:
    """:func:`auroc` with sample weights: per tie level, positive weight times the negative weight below it."""
    single = weight.ndim == 1
    w = np.atleast_2d(weight)
    order = np.argsort(score, kind="stable")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos = np.add.reduceat(w[:, order] * positive[order], starts, axis=1)
    neg = np.add.reduceat(w[:, order] * ~positive[order], starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    denom = pos.sum(1) * neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, (pos * (below + 0.5 * neg)).sum(1) / denom, np.nan)
    return float(out[0]) if single else out


def stratified_auroc(
    score: np.ndarray, positive: np.ndarray, reference: np.ndarray, *, n_bins: int = 10, min_class: int = 5
) -> float:
    """AUROC of ``score`` for ``positive`` within quantile bins of ``reference``.

    Each bin's AUROC is weighted by its number of positives. A score that is a
    function of the reference scores about 0.5; a score that carries information about
    the positives beyond the reference scores above it. With ``n_bins=1`` this is the
    plain AUROC.

    Args:
        score: ``(N,)`` score to evaluate, higher meaning more likely positive.
        positive: ``(N,)`` boolean labels.
        reference: ``(N,)`` score whose information is conditioned away.
        n_bins: Number of equal-count bins of ``reference``.
        min_class: Bins with fewer positives or negatives than this are skipped.

    Returns:
        The weighted AUROC, ``nan`` if no bin holds both classes.
    """
    score, ref = _check_pair(score, reference)
    positive = np.asarray(positive, dtype=bool).ravel()
    if positive.shape != score.shape:
        raise ValueError(f"positive must have {score.size} entries, got {positive.size}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    ranks = np.argsort(np.argsort(ref, kind="stable"), kind="stable")
    bins = ranks * n_bins // ref.size
    num = den = 0.0
    for b in range(n_bins):
        m = bins == b
        pos = positive[m]
        if pos.sum() < min_class or (~pos).sum() < min_class:
            continue
        num += auroc(score[m], pos) * pos.sum()
        den += pos.sum()
    return float(num / den) if den > 0 else float("nan")


def silent_share(score: np.ndarray, positive: np.ndarray, *, false_alarm: float = 0.05) -> float:
    """Fraction of positives that ``score`` leaves unflagged at a false-alarm rate on the negatives.

    The flag threshold is the ``1 - false_alarm`` quantile of the negatives' scores; a
    positive is silent when its score is at or below it. An uninformative score leaves
    ``1 - false_alarm`` of the positives silent, a perfect one none.

    Args:
        score: ``(N,)`` score, higher meaning flagged sooner (a distance to the labels).
        positive: ``(N,)`` boolean labels (pixels carrying a held-out concept, say).
        false_alarm: Share of negatives flagged, in ``(0, 1)``.

    Returns:
        The silent share of the positives, in ``[0, 1]``.

    Raises:
        ValueError: Shapes differ, ``false_alarm`` is outside ``(0, 1)``, or either class
            is empty.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=bool).ravel()
    if positive.shape != score.shape:
        raise ValueError(f"positive must have {score.size} entries, got {positive.size}")
    if not 0.0 < false_alarm < 1.0:
        raise ValueError(f"false_alarm must lie in (0, 1), got {false_alarm}")
    if positive.all() or not positive.any():
        raise ValueError("silent_share needs both positives and negatives")
    threshold = np.quantile(score[~positive], 1.0 - false_alarm)
    return float((score[positive] <= threshold).mean())


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ``ceil((n + 1)(1 - alpha))``-th smallest value.

    Args:
        values: ``(n,)`` calibration values.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        The quantile, or ``inf`` when ``n`` is too small to certify ``1 - alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    rank = int(np.ceil((v.size + 1) * (1.0 - alpha)))
    return float(v[rank - 1]) if rank <= v.size else float("inf")


def conformal_selection(
    cal_err: np.ndarray,
    cal_score: np.ndarray,
    test_err: np.ndarray,
    test_score: np.ndarray,
    *,
    alpha: float = 0.1,
) -> SelectiveCoverage:
    """Coverage and retained area of a conformally calibrated applicability mask.

    Both the interval half-width (on ``|error|``) and the score threshold are
    split-conformal quantiles of the calibration pixels at the same level
    ``1 - alpha``. When the held-out pixels are exchangeable with the
    calibration ones, about ``1 - alpha`` of them are retained and covered. Under
    covariate shift the coverage of all held-out pixels falls; a score that tracks
    the shift keeps the coverage of the retained ones at the target.

    Args:
        cal_err: ``(n,)`` absolute errors of held-in calibration pixels.
        cal_score: ``(n,)`` applicability scores of the same pixels.
        test_err: ``(m,)`` absolute errors of held-out pixels.
        test_score: ``(m,)`` applicability scores of the same pixels.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        The :class:`SelectiveCoverage`; ``coverage`` is ``nan`` if nothing is retained.
    """
    cal_score, cal_err = _check_pair(cal_score, cal_err)
    test_score, test_err = _check_pair(test_score, test_err)
    q = conformal_quantile(np.abs(cal_err), alpha)
    t = conformal_quantile(cal_score, alpha)
    covered = np.abs(test_err) <= q
    kept = test_score <= t
    return SelectiveCoverage(
        alpha=float(alpha),
        halfwidth=q,
        threshold=t,
        coverage=float(covered[kept].mean()) if kept.any() else float("nan"),
        area=float(kept.mean()),
        coverage_all=float(covered.mean()),
    )


__all__ = [
    "RiskCoverage",
    "SelectiveCoverage",
    "risk_coverage_curve",
    "risk_coverage",
    "auroc",
    "stratified_auroc",
    "silent_share",
    "conformal_quantile",
    "conformal_selection",
]

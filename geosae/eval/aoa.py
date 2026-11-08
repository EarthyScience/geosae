"""Area-of-Applicability (AoA) downstream generalization eval.

Does a feature-space *distance to the training set* predict where a downstream probe
will fail? We fit a Ridge head on a geophysical target (CHMv2 canopy height, or DEM
elevation) in raw-embedding space and in SAE-code space, hold out an
out-of-distribution region, and score the **Spearman correlation between a
kNN-distance dissimilarity index (DI) and the held-out absolute error**. A higher
positive rho means the representation's geometry *knows where it is extrapolating* —
the Meyer & Pebesma (2021) Area-of-Applicability idea.

Three kNN distance variants (all ``k=5``), differing in how the dimensions are
weighted before the distance:

* **unweighted** — plain kNN distance, normalized by the median train<->train
  kNN distance, so ``1`` marks the in-domain boundary.
* **ridge-weighted** — features rescaled by ``sqrt(|beta| / mean|beta|)`` (the
  Ridge-residual importance) before kNN.
* **rf-weighted** — features rescaled by ``sqrt(rf_importance)`` (Meyer & Pebesma's
  original random-forest impurity importance; supplementary, needs scikit-learn).

Which held-out pixels lie *outside* the area of applicability depends on the
threshold, not on the ranking. :data:`THRESHOLD_RULES` offers the median rule above
and Meyer & Pebesma's canonical one (:func:`dissimilarity_index`): nearest-neighbour
distance over the mean training distance, cut at the upper whisker of the training
DI computed across CV folds.

A distance is not the only applicability signal, and in a sparse code it is not
obviously the best one. Two more are offered:

* **local point density** — how many training points fall inside the in-domain
  radius, which responds to the *shape* of the neighbourhood rather than to the
  single nearest distance.
* **support novelty** — how far a code's active set departs from in-domain atom
  usage. This has no dense counterpart: only a sparse code can say "this pixel
  switched on an atom the training fold never used". Weighting it by the probe's
  ``|beta|`` isolates the atoms the downstream head actually extrapolates along.

The geometry itself is configurable, because it is not neutral: :data:`METRICS`
selects Euclidean / cosine / $L_1$ and :data:`SCALINGS` selects the per-feature
rescaling applied first. A unit-norm raw embedding and an unnormalised sparse code
are otherwise compared under two different estimators. :func:`di_agreement`
quantifies how far two such maps actually differ, and :func:`hubness` reports the
high-dimensional pathology that can make them agree for the wrong reason.

Four OOD split scenarios (:data:`SPLIT_SCENARIOS`): train-on-single-continent,
leave-one-continent-out, train-on-single-PFT, leave-one-PFT-out. Continent =
lon/lat bounding box; PFT = dominant MODIS-IGBP class per patch
(``lc_modis_igbp``). :func:`sample_sites` draws the held-out pixels and encodes
them into paired raw and SAE spaces; :func:`run_scenarios` sweeps the grid.

The scoring layer uses only numpy/scipy (closed-form Ridge, exact brute-force
BLAS kNN, ``scipy.stats.spearmanr``) so it runs in core CI; the RF variant lazily
imports ``sklearn`` and :func:`sample_sites` lazily imports torch/zarr.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from geosae.eval import selective
from geosae.eval.extrapolation import mahalanobis
from geosae.models import sparse_codes

if TYPE_CHECKING:  # heavy imports stay out of the numpy-only scoring path
    from geosae.data.dataset import PatchData
    from geosae.models.base import SparseAutoencoder

# Continent bounding boxes (lon_min, lon_max, lat_min, lat_max).
CONTINENT_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "Europe": (-25.0, 45.0, 35.0, 72.0),
    "Africa": (-20.0, 55.0, -35.0, 37.0),
    "Asia": (45.0, 180.0, 5.0, 78.0),
    "Oceania": (110.0, 180.0, -50.0, 0.0),
    "NorthAmerica": (-170.0, -50.0, 10.0, 72.0),
    "SouthAmerica": (-90.0, -34.0, -56.0, 13.0),
}

# Plant functional type -> MODIS IGBP class code (dominant per patch).
PFT_CODES: dict[str, int] = {
    "ENF": 1,
    "EBF": 2,
    "DBF": 4,
    "Savanna": 9,
    "Grassland": 10,
    "Cropland": 12,
    "Bare": 16,
}


# ---------------------------------------------------------------------------
# Closed-form Ridge (numeric-identical to sklearn Ridge(alpha, fit_intercept=True))
# ---------------------------------------------------------------------------


def ridge_fit(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """Ridge regression ``(coef, intercept)`` via centered closed form.

    Matches ``sklearn.linear_model.Ridge(alpha, fit_intercept=True)``: center X, y;
    solve ``(Xc^T Xc + alpha I) w = Xc^T yc``; ``intercept = ybar - xbar . w``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    d = x.shape[1]
    gram = xc.T @ xc
    gram.flat[:: d + 1] += alpha  # add alpha to the diagonal
    coef = np.linalg.solve(gram, xc.T @ yc)
    intercept = y_mean - x_mean @ coef
    return coef, float(intercept)


def ridge_predict(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ coef + intercept


# ---------------------------------------------------------------------------
# kNN dissimilarity index
# ---------------------------------------------------------------------------


#: Distance metrics the kNN dissimilarity index supports.
METRICS: tuple[str, ...] = ("euclidean", "cosine", "cityblock")

#: Per-feature rescalings applied before the distance is taken.
SCALINGS: tuple[str, ...] = ("none", "std", "active")

#: :func:`score_fold` variants scored by :func:`support_novelty` (surprisal), not by a
#: distance: plain, and weighted by the fold head's ``|beta|``.
SUPPORT_VARIANTS: tuple[str, ...] = ("support", "support_probe")

#: :func:`score_fold` variant scored by the shrunk Mahalanobis distance to the training
#: side (:func:`geosae.eval.extrapolation.mahalanobis`), the classical combination check.
MAHALANOBIS_VARIANT: str = "mahalanobis"

#: In-domain boundaries :func:`score_fold` can apply: ``"median"`` (the mean-kNN
#: distance over its median leave-one-out training value, cut at ``threshold``) and
#: ``"cast"`` (Meyer & Pebesma's area of applicability, :func:`dissimilarity_index`).
THRESHOLD_RULES: tuple[str, ...] = ("median", "cast")

#: How :func:`aoa_threshold` removes outliers from the training DI: current CAST
#: (``"quantile"``) or CAST 0.5's ``boxplot.stats`` (``"boxplot"``).
AOA_THRESHOLD_RULES: tuple[str, ...] = ("quantile", "boxplot")


def scale_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    mode: str = "none",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale both sides of a fold by a per-feature statistic of the *training* side.

    Euclidean kNN weights every dimension by its ambient spread, so an unscaled
    distance in a space whose dimensions have wildly different magnitudes is
    effectively a distance along the few widest ones. The two representations we
    compare differ exactly here: the raw embedding is unit-norm by construction,
    while an SAE code is an unnormalised, mostly-zero vector whose active
    magnitudes vary by orders of magnitude across atoms. Comparing them without a
    common scaling compares two different estimators.

    Args:
        train_x: ``(n_train, F)`` in-fold factor matrix; the scaler is fit on this
            side only, so the held-out side cannot leak into the geometry.
        test_x: ``(n_test, F)`` held-out factor matrix.
        mode: ``"none"`` (leave as is), ``"std"`` (divide by the training standard
            deviation), or ``"active"`` (divide by the mean magnitude *while
            active*, the sparse-code analogue of ``"std"`` — it measures how big a
            firing is, not how often it fires).
        eps: Floor on the divisor; features below it are left untouched.

    Returns:
        ``(train_scaled, test_scaled)``, both float64.
    """
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    denom = _feature_scale(train_x, mode=mode, eps=eps)
    return train_x / denom, test_x / denom


def _feature_scale(train_x: np.ndarray, *, mode: str, eps: float = 1e-12) -> np.ndarray:
    """Per-feature divisor :func:`scale_features` applies (ones for ``"none"``)."""
    if mode not in SCALINGS:
        raise ValueError(f"unknown scaling {mode!r}, expected one of {SCALINGS}")
    train_x = np.asarray(train_x, dtype=np.float64)
    if mode == "none":
        return np.ones(train_x.shape[1])
    if mode == "std":
        denom = train_x.std(axis=0)
    else:  # "active": mean |value| over the rows where the feature fires
        active = train_x != 0
        n_active = active.sum(axis=0)
        denom = np.where(n_active > 0, np.abs(train_x * active).sum(axis=0) / np.maximum(n_active, 1), 0.0)
    return np.where(np.isfinite(denom) & (denom > eps), denom, 1.0)


def _metric_embed(x: np.ndarray, metric: str, *, eps: float = 1e-12) -> np.ndarray:
    """Row transform under which plain Euclidean distance realises ``metric``.

    ``cosine`` becomes Euclidean on the unit sphere (``|a-b|^2 = 2 - 2 cos``, which
    is monotone in the cosine distance, so every rank statistic is preserved).
    ``euclidean`` and ``cityblock`` pass through — the latter has no such
    embedding and is computed directly by :func:`_mean_knn_dist_cityblock`.
    """
    if metric == "cosine":
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, eps)
    return x


def _mean_knn_dist(query: np.ndarray, ref: np.ndarray, *, k: int, exclude_self: bool, chunk: int = 4096) -> np.ndarray:
    """Mean Euclidean distance to the ``k`` nearest ``ref`` points, per ``query`` row.

    Exact brute-force kNN via a BLAS Gram expansion (``|q|^2 + |r|^2 - 2 q.r``),
    chunked over queries to bound memory. Identical to sklearn
    ``NearestNeighbors(algorithm="brute")`` but fast in the high-dim SAE space where a
    KD-tree degenerates. When ``exclude_self`` (query IS ref), the nearest point at
    distance 0 is dropped (the ``k+1`` trick).
    """
    r2 = np.einsum("ij,ij->i", ref, ref)  # (n_ref,)
    kk = k + 1 if exclude_self else k
    kk = min(kk, ref.shape[0])
    out = np.empty(query.shape[0], dtype=np.float64)
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk]
        d2 = r2[None, :] + np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ ref.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]  # kk smallest squared dists
        part = np.sqrt(np.sort(part, axis=1))
        block = part[:, 1:kk] if exclude_self else part[:, :kk]  # drop self column
        out[s : s + chunk] = block.mean(axis=1)
    return out


def _mean_knn_dist_cityblock(
    query: np.ndarray, ref: np.ndarray, *, k: int, exclude_self: bool, chunk: int = 256
) -> np.ndarray:
    """Mean $L_1$ distance to the ``k`` nearest ``ref`` points, per ``query`` row.

    $L_1$ admits no Gram trick, so this materialises ``|q - r|`` one small query
    chunk at a time — ``O(chunk * n_ref * F)`` work and memory per step. Much
    slower than the Euclidean path; offered because $L_1$ is the better-behaved
    metric in high dimension (Aggarwal et al., 2001).
    """
    kk = min(k + 1 if exclude_self else k, ref.shape[0])
    out = np.empty(query.shape[0], dtype=np.float64)
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk]
        d = np.abs(q[:, None, :] - ref[None, :, :]).sum(axis=2)
        part = np.sort(np.partition(d, kk - 1, axis=1)[:, :kk], axis=1)
        out[s : s + chunk] = (part[:, 1:kk] if exclude_self else part[:, :kk]).mean(axis=1)
    return out


def _knn_dist(query: np.ndarray, ref: np.ndarray, *, k: int, exclude_self: bool, metric: str) -> np.ndarray:
    """Dispatch the mean-kNN distance to the backend that realises ``metric``."""
    if metric == "cityblock":
        return _mean_knn_dist_cityblock(query, ref, k=k, exclude_self=exclude_self)
    return _mean_knn_dist(query, ref, k=k, exclude_self=exclude_self)


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}")


def knn_distance(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    k: int = 5,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Median-normalized mean-kNN distance for train and test sets.

    ``d_train`` uses the ``k`` nearest *other* training points (self at distance 0 is
    dropped); ``norm`` is the median of ``d_train``; ``d_test`` uses the ``k`` nearest
    training points. Exact kNN (brute force; the Euclidean path matches sklearn
    ``NearestNeighbors(algorithm="brute")``).

    Args:
        train_x: ``(n_train, F)`` in-fold factor matrix.
        test_x: ``(n_test, F)`` held-out factor matrix.
        k: Number of neighbours averaged over.
        metric: One of :data:`METRICS`. ``"cosine"`` is realised as Euclidean on
            the unit sphere, which preserves every rank statistic of the cosine
            distance.

    Returns:
        ``(d_train / norm, d_test / norm, norm)`` — distances in units of the
        median train-to-train distance, so ``1`` marks the in-domain boundary.
    """
    _check_metric(metric)
    train_x = np.ascontiguousarray(_metric_embed(np.asarray(train_x, dtype=np.float64), metric))
    test_x = np.ascontiguousarray(_metric_embed(np.asarray(test_x, dtype=np.float64), metric))
    d_tr = _knn_dist(train_x, train_x, k=k, exclude_self=True, metric=metric)
    norm = float(np.nanmedian(d_tr))
    if not np.isfinite(norm) or norm <= 0:
        norm = 1.0
    d_te = _knn_dist(test_x, train_x, k=k, exclude_self=False, metric=metric)
    return d_tr / norm, d_te / norm, norm


def _pairwise_distance(query: np.ndarray, ref: np.ndarray, metric: str) -> np.ndarray:
    """Dense ``(len(query), len(ref))`` distance matrix (rows already metric-embedded)."""
    if metric == "cityblock":
        return np.abs(query[:, None, :] - ref[None, :, :]).sum(axis=2)
    d2 = np.einsum("ij,ij->i", query, query)[:, None] + np.einsum("ij,ij->i", ref, ref)[None, :] - 2.0 * (query @ ref.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _fivenum_hinges(x: np.ndarray) -> tuple[float, float]:
    """Tukey's lower and upper hinges of sorted ``x``, as R's ``stats::fivenum``."""
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([n4, n + 1 - n4]) - 1.0  # 1-based positions -> 0-based
    lo, hi = np.floor(d).astype(int), np.ceil(d).astype(int)
    h = 0.5 * (x[lo] + x[hi])
    return float(h[0]), float(h[1])


def aoa_threshold(di_train: np.ndarray, *, rule: str = "quantile") -> float:
    """Outlier-removed maximum of the training DI, the area-of-applicability threshold.

    Args:
        di_train: Training dissimilarity indices; NaNs are ignored.
        rule: ``"quantile"`` (CAST since 0.6: ``min(Q75 + 1.5 IQR, max)`` with R's
            default type-7 quantiles) or ``"boxplot"`` (CAST 0.5, as published with
            Meyer & Pebesma 2021: ``grDevices::boxplot.stats(DI)$stats[5]``, the
            largest DI at or below the upper Tukey hinge plus 1.5 hinge spreads).

    Returns:
        The threshold; NaN if no finite value is given.

    Raises:
        ValueError: On an unknown ``rule``.
    """
    if rule not in AOA_THRESHOLD_RULES:
        raise ValueError(f"unknown rule {rule!r}, expected one of {AOA_THRESHOLD_RULES}")
    x = np.sort(np.asarray(di_train, dtype=np.float64).ravel())
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan")
    if rule == "quantile":
        q25, q75 = np.quantile(x, [0.25, 0.75])
        return float(min(q75 + 1.5 * (q75 - q25), x[-1]))
    lower, upper = _fivenum_hinges(x)
    return float(x[x <= upper + 1.5 * (upper - lower)].max())


def dissimilarity_index(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    cv_folds: np.ndarray | None = None,
    metric: str = "euclidean",
    rule: str = "quantile",
    chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Meyer & Pebesma's (2021) dissimilarity index and area-of-applicability threshold.

    Follows CAST's ``trainDI`` and ``aoa``. The DI of a point is its distance to the
    nearest training point divided by the mean pairwise distance between training
    points. A training point's own DI uses only training points outside its CV fold,
    so with spatial folds the threshold reflects the distance a held-out region
    really has from the rest. The threshold is the outlier-removed maximum of the
    training DI (:func:`aoa_threshold`); a point beyond it lies outside the area of
    applicability. Scaling and weighting are left to the caller, who passes the
    features already in the space the distance should be taken in.

    Args:
        train_x: ``(n_train, F)`` training features.
        test_x: ``(n_test, F)`` features to score.
        cv_folds: ``(n_train,)`` CV fold id of each training point. ``None`` uses the
            nearest other training point (CAST without CV folds).
        metric: One of :data:`METRICS`.
        rule: Threshold rule of :func:`aoa_threshold`.
        chunk: Rows per block of the pairwise distance computation.

    Returns:
        ``(di_train, di_test, threshold)``.

    Raises:
        ValueError: On fewer than two training points, or ``cv_folds`` of the wrong
            length or with a single fold.
    """
    _check_metric(metric)
    tr = np.ascontiguousarray(_metric_embed(np.asarray(train_x, dtype=np.float64), metric))
    te = np.ascontiguousarray(_metric_embed(np.asarray(test_x, dtype=np.float64), metric))
    n = len(tr)
    if n < 2:
        raise ValueError("need at least two training points")
    folds = None if cv_folds is None else np.asarray(cv_folds).ravel()
    if folds is not None and (len(folds) != n or len(np.unique(folds)) < 2):
        raise ValueError(f"cv_folds must give each of the {n} training points one of at least two folds")
    total, nearest = 0.0, np.empty(n)
    for s in range(0, n, chunk):
        d = _pairwise_distance(tr[s : s + chunk], tr, metric)
        rows = np.arange(len(d))
        d[rows, s + rows] = 0.0  # exact zero on the diagonal, whatever the Gram rounding
        total += float(d.sum())
        if folds is None:
            d[rows, s + rows] = np.inf
        else:
            d[folds[s : s + chunk, None] == folds[None, :]] = np.inf
        nearest[s : s + chunk] = d.min(axis=1)
    mean_dist = total / (n * (n - 1))
    if not np.isfinite(mean_dist) or mean_dist <= 0:
        mean_dist = 1.0
    di_train = nearest / mean_dist
    di_test = np.concatenate(
        [_pairwise_distance(te[s : s + chunk], tr, metric).min(axis=1) for s in range(0, len(te), chunk)]
        or [np.empty(0)]
    )
    return di_train, di_test / mean_dist, aoa_threshold(di_train, rule=rule)


def local_point_density(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    k: int = 5,
    metric: str = "euclidean",
    threshold: float = 1.0,
    chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Count training points inside the applicability radius of every point.

    The dissimilarity index looks only at the ``k`` closest training points, so a
    query sitting beside one isolated training pixel scores as in-domain even
    though its neighbourhood is nearly empty. Local point density (Schumacher et
    al., 2025) instead counts how many training points fall within the in-domain
    radius, and so responds to the *shape* of the neighbourhood rather than to a
    single nearest distance.

    Args:
        train_x: ``(n_train, F)`` in-fold factor matrix.
        test_x: ``(n_test, F)`` held-out factor matrix.
        k: Neighbours used to set the radius, matching :func:`knn_distance`.
        metric: One of :data:`METRICS`.
        threshold: Radius in units of the median train-to-train kNN distance.
        chunk: Query rows per distance block.

    Returns:
        ``(count_train, count_test, radius)`` — integer counts of training points
        within ``radius`` (the training count excludes the point itself).
    """
    _check_metric(metric)
    if metric == "cityblock":
        raise ValueError("local_point_density supports euclidean/cosine only (the L1 path is not blocked for radii)")
    train_x = np.ascontiguousarray(_metric_embed(np.asarray(train_x, dtype=np.float64), metric))
    test_x = np.ascontiguousarray(_metric_embed(np.asarray(test_x, dtype=np.float64), metric))
    d_tr = _mean_knn_dist(train_x, train_x, k=k, exclude_self=True)
    norm = float(np.nanmedian(d_tr))
    if not np.isfinite(norm) or norm <= 0:
        norm = 1.0
    radius = float(threshold) * norm

    r2 = np.einsum("ij,ij->i", train_x, train_x)
    rad2 = radius * radius

    def _counts(query: np.ndarray, self_query: bool) -> np.ndarray:
        out = np.empty(query.shape[0], dtype=np.int64)
        for s in range(0, query.shape[0], chunk):
            q = query[s : s + chunk]
            d2 = r2[None, :] + np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ train_x.T)
            out[s : s + chunk] = (d2 <= rad2).sum(axis=1)
        return out - 1 if self_query else out

    return _counts(train_x, True), _counts(test_x, False), radius


def hubness(
    train_x: np.ndarray, test_x: np.ndarray, *, k: int = 5, metric: str = "euclidean", chunk: int = 2048
) -> dict:
    """Summarise how unevenly the training points serve as nearest neighbours.

    In high dimension a few points become the nearest neighbour of very many
    others (Radovanovic et al., 2010). Such hubs flatten any distance-based
    out-of-distribution map, because most queries end up pinned to the same small
    set of references. The standard diagnostic is the skewness of the
    *k-occurrence* distribution — how often each reference point appears in
    someone's top-``k``.

    Args:
        train_x: ``(n_train, F)`` reference points.
        test_x: ``(n_test, F)`` query points.
        k: Neighbourhood size.
        metric: One of :data:`METRICS` (``"cityblock"`` is not supported here).
        chunk: Query rows per distance block.

    Returns:
        ``k_occurrence_skewness`` (0 for a perfectly even cloud, large and
        positive when hubs dominate), ``k_occurrence_max``, ``k_occurrence_mean``,
        and ``antihub_fraction`` (share of training points never retrieved).
    """
    _check_metric(metric)
    if metric == "cityblock":
        raise ValueError("hubness supports euclidean/cosine only")
    train_x = np.ascontiguousarray(_metric_embed(np.asarray(train_x, dtype=np.float64), metric))
    test_x = np.ascontiguousarray(_metric_embed(np.asarray(test_x, dtype=np.float64), metric))
    kk = min(k, train_x.shape[0])
    r2 = np.einsum("ij,ij->i", train_x, train_x)
    occ = np.zeros(train_x.shape[0], dtype=np.int64)
    for s in range(0, test_x.shape[0], chunk):
        q = test_x[s : s + chunk]
        d2 = r2[None, :] + np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ train_x.T)
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        np.add.at(occ, idx.ravel(), 1)
    mu, sd = float(occ.mean()), float(occ.std())
    skew = float(((occ - mu) ** 3).mean() / sd**3) if sd > 0 else 0.0
    return {
        "k_occurrence_skewness": skew,
        "k_occurrence_max": int(occ.max()),
        "k_occurrence_mean": mu,
        "antihub_fraction": float((occ == 0).mean()),
    }


def ridge_weights(coef: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Meyer & Pebesma feature weights ``|beta| / mean|beta|`` (renormalized to mean 1)."""
    w = np.abs(np.asarray(coef, dtype=np.float64))
    s = float(w.mean())
    return np.ones_like(w) if (not np.isfinite(s) or s <= eps) else w / s


def rf_importance_weights(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 60,
    max_depth: int = 20,
    n_subsample: int = 4_000,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Random-forest impurity feature-importance weights (mean 1). Needs scikit-learn."""
    from sklearn.ensemble import RandomForestRegressor

    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sub = rng.choice(x.shape[0], size=min(x.shape[0], n_subsample), replace=False)
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt", n_jobs=-1, random_state=seed
    ).fit(x[sub], np.asarray(y)[sub])
    w = np.asarray(rf.feature_importances_, dtype=np.float64)
    s = float(w.mean())
    return np.ones_like(w) if (not np.isfinite(s) or s <= eps) else w / s


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ``nan`` when either side is constant.

    A constant side makes the correlation undefined rather than zero, and that
    distinction matters here: support novelty is identically zero on a *dense*
    factor matrix (every coordinate is always active, so there is no support to be
    surprised by), and a ``nan`` records "this signal does not exist in this
    space" where a ``0.0`` would claim it was measured and found useless.
    """
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


# ---------------------------------------------------------------------------
# Per-feature support: the signal a dense space cannot express
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSupport:
    """In-domain usage statistics of a sparse dictionary, one entry per atom.

    Attributes:
        frequency: ``(F,)`` fraction of training points on which the atom fires.
        active_mean: ``(F,)`` mean magnitude while active (0 for atoms that never
            fire).
        active_quantile: ``(F,)`` upper quantile of the active magnitude, the
            level above which a firing counts as unusually strong.
        quantile: The quantile used for ``active_quantile``.
        n_train: Number of training points the statistics were accumulated over.
    """

    frequency: np.ndarray
    active_mean: np.ndarray
    active_quantile: np.ndarray
    quantile: float
    n_train: int


def support_stats(train_code: np.ndarray, *, quantile: float = 0.99) -> FeatureSupport:
    """Accumulate the in-domain support of every dictionary atom.

    Args:
        train_code: ``(n_train, F)`` non-negative sparse codes of the training
            fold. An atom is *active* on a row where its value is non-zero.
        quantile: Upper quantile of the active magnitudes to record.

    Returns:
        A :class:`FeatureSupport`.
    """
    code = np.asarray(train_code, dtype=np.float64)
    active = code != 0
    n = code.shape[0]
    freq = active.sum(axis=0) / max(n, 1)
    amean = np.zeros(code.shape[1])
    aq = np.zeros(code.shape[1])
    for j in np.flatnonzero(active.any(axis=0)):
        vals = np.abs(code[active[:, j], j])
        amean[j] = vals.mean()
        aq[j] = np.quantile(vals, quantile)
    return FeatureSupport(frequency=freq, active_mean=amean, active_quantile=aq, quantile=float(quantile), n_train=n)


def support_novelty(
    test_code: np.ndarray,
    support: FeatureSupport,
    *,
    mode: str = "surprisal",
    weights: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Score how far a code's *support set* departs from in-domain usage.

    A distance asks "how far is this point from the training cloud"; this asks the
    question a sparse code makes available and a dense one does not: "does this
    point switch on atoms the training set never used?" A probe trained in-domain
    has a coefficient on every atom, including ones that never fired in-domain;
    when such an atom fires off-domain the probe extrapolates along a direction it
    was never fit on. Passing the probe's ``|beta|`` as ``weights`` makes exactly
    that pixel score high.

    Args:
        test_code: ``(n_test, F)`` sparse codes of the held-out points.
        support: In-domain statistics from :func:`support_stats`.
        mode: ``"surprisal"`` — mean of ``-log(frequency)`` over the active atoms,
            the information content of the support set under the in-domain firing
            rates; ``"count"`` — number of active atoms whose magnitude exceeds
            their in-domain ``active_quantile`` (an atom unseen in-domain has
            quantile 0, so any firing counts).
        weights: Optional ``(F,)`` non-negative per-atom weights (e.g. ``|beta|``
            of the downstream probe). ``None`` weights every atom equally.
        eps: Floor on the frequency inside the logarithm, which caps the
            surprisal of a never-seen atom at ``-log(eps)``.

    Returns:
        ``(n_test,)`` novelty scores; higher means the code leans more on atoms
        the training fold did not support.
    """
    if mode not in ("surprisal", "count"):
        raise ValueError(f"unknown mode {mode!r}, expected 'surprisal' or 'count'")
    code = np.asarray(test_code, dtype=np.float64)
    active = code != 0
    w = np.ones(code.shape[1]) if weights is None else np.abs(np.asarray(weights, dtype=np.float64))
    if mode == "surprisal":
        per_atom = -np.log(np.maximum(support.frequency, eps))
    else:
        per_atom = np.ones(code.shape[1])
        active = active & (np.abs(code) > support.active_quantile[None, :])
    num = (active * (per_atom * w)[None, :]).sum(axis=1)
    denom = (active * w[None, :]).sum(axis=1)
    if mode == "count":
        return num
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


# ---------------------------------------------------------------------------
# Do two dissimilarity maps actually differ?
# ---------------------------------------------------------------------------


def di_rank_profile(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    ranks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    k: int = 5,
    metric: str = "euclidean",
) -> dict:
    """How few leading directions reproduce the full-space dissimilarity index.

    If the DI of a wide space is rank-equivalent to the DI of its first two or
    three principal directions, then the space's width is decorative: the map is
    a picture of those directions, and any other representation sharing them
    produces the same map. This profiles that directly — the DI is recomputed on
    a rank-``r`` truncation of the training fold's principal basis and ranked
    against the full-space DI.

    Args:
        train_x: ``(n_train, F)`` in-fold factor matrix; the basis is fit on it.
        test_x: ``(n_test, F)`` held-out factor matrix.
        ranks: Truncation ranks to profile; values above ``min(n_train, F)`` are
            skipped.
        k: Neighbours for the dissimilarity index.
        metric: One of :data:`METRICS`.

    Returns:
        ``ranks`` (those actually evaluated), ``spearman`` (agreement of each
        truncated DI with the full-space DI), ``explained_variance`` (cumulative
        share of the training variance at each rank), and ``rank_95`` — the
        smallest profiled rank reaching Spearman 0.95, or ``None``.
    """
    _check_metric(metric)
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    _, d_full, _ = knn_distance(train_x, test_x, k=k, metric=metric)

    mean = train_x.mean(axis=0)
    _, sv, vt = np.linalg.svd(train_x - mean, full_matrices=False)
    var = sv**2
    cum = np.cumsum(var) / max(var.sum(), 1e-300)
    usable = [int(r) for r in ranks if 0 < r <= vt.shape[0]]

    rhos, evs = [], []
    for r in usable:
        basis = vt[:r]
        _, d_r, _ = knn_distance((train_x - mean) @ basis.T, (test_x - mean) @ basis.T, k=k, metric=metric)
        rhos.append(_spearman(d_r, d_full))
        evs.append(float(cum[r - 1]))
    rank_95 = next((r for r, rho in zip(usable, rhos, strict=True) if rho >= 0.95), None)
    return {"ranks": usable, "spearman": rhos, "explained_variance": evs, "rank_95": rank_95}


def extrapolating_atoms(
    train_code: np.ndarray,
    test_code: np.ndarray,
    coef: np.ndarray,
    *,
    top: int = 20,
) -> dict:
    """Rank dictionary atoms by how far the probe extrapolates along them off-domain.

    The failure mode a monosemantic decomposition is supposed to expose: the probe
    was fit in-domain, so its coefficient on an atom that barely fired in-domain is
    essentially unconstrained; when that atom fires often off-domain, the head is
    predicting along a direction it never saw. The score is the probe's absolute
    coefficient times the increase in the atom's firing rate, which is exactly the
    atom's contribution to the off-domain shift in the linear prediction.

    Args:
        train_code: ``(n_train, F)`` in-fold sparse codes.
        test_code: ``(n_test, F)`` held-out sparse codes.
        coef: ``(F,)`` probe coefficients fit on ``train_code``.
        top: Number of atoms to return, ranked by score.

    Returns:
        ``index`` (atom ids, worst first), ``score``, ``freq_train``,
        ``freq_test``, ``coef``, and ``mean_shift`` — the change in the atom's
        mean contribution ``beta_j * z_j`` from the training to the held-out fold.
    """
    tr = np.asarray(train_code, dtype=np.float64)
    te = np.asarray(test_code, dtype=np.float64)
    beta = np.asarray(coef, dtype=np.float64)
    f_tr = (tr != 0).mean(axis=0)
    f_te = (te != 0).mean(axis=0)
    score = np.abs(beta) * np.maximum(f_te - f_tr, 0.0)
    shift = beta * (te.mean(axis=0) - tr.mean(axis=0))
    order = np.argsort(-score)[:top]
    return {
        "index": order,
        "score": score[order],
        "freq_train": f_tr[order],
        "freq_test": f_te[order],
        "coef": beta[order],
        "mean_shift": shift[order],
    }


def di_agreement(d_a: np.ndarray, d_b: np.ndarray, *, threshold: float = 1.0) -> dict:
    """Quantify how far two dissimilarity maps of the same points agree.

    Two AoA maps can look alike for two different reasons: the fields are rank-
    equivalent, or the fields differ but both sit far above the in-domain
    threshold everywhere, so the *maps* are the same picture. This separates the
    two by reporting a rank statistic and a threshold contingency.

    Args:
        d_a: ``(n,)`` dissimilarity index of the first space, in threshold units.
        d_b: ``(n,)`` dissimilarity index of the second space, same points, same
            units.
        threshold: In-domain boundary; ``1`` for a median-normalised DI.

    Returns:
        ``spearman`` and ``pearson_log`` (rank and log-scale agreement),
        ``median_log_ratio`` / ``iqr_log_ratio`` (systematic offset and spread of
        ``log(d_a / d_b)``), the four contingency counts ``n_both_in``,
        ``n_both_out``, ``n_only_a``, ``n_only_b`` (``only_a`` = out of domain in
        *a* alone), ``mask_jaccard`` and ``mask_kappa`` for the two
        out-of-domain masks, the two ``frac_out_*`` rates, and ``overlay`` — a
        per-point code in ``{0: both out, 1: out in b only, 2: out in a only,
        3: both in}`` for mapping.
    """
    a = np.asarray(d_a, dtype=np.float64)
    b = np.asarray(d_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    out_a, out_b = a > threshold, b > threshold
    n_both_out = int((out_a & out_b & finite).sum())
    n_only_a = int((out_a & ~out_b & finite).sum())
    n_only_b = int((~out_a & out_b & finite).sum())
    n_both_in = int((~out_a & ~out_b & finite).sum())
    n = max(n_both_out + n_only_a + n_only_b + n_both_in, 1)
    # Cohen's kappa on the two binary out-of-domain masks.
    p_obs = (n_both_out + n_both_in) / n
    p_exp = ((n_both_out + n_only_a) * (n_both_out + n_only_b) + (n_both_in + n_only_b) * (n_both_in + n_only_a)) / n**2
    union = n_both_out + n_only_a + n_only_b
    pos = finite & (a > 0) & (b > 0)
    log_ratio = np.log(a[pos]) - np.log(b[pos])
    overlay = np.where(out_a & out_b, 0, np.where(out_b, 1, np.where(out_a, 2, 3))).astype(np.int8)
    return {
        "spearman": _spearman(a[finite], b[finite]),
        "pearson_log": float(np.corrcoef(np.log(a[pos]), np.log(b[pos]))[0, 1]) if pos.sum() > 2 else float("nan"),
        "median_log_ratio": float(np.median(log_ratio)) if log_ratio.size else float("nan"),
        "iqr_log_ratio": float(np.subtract(*np.percentile(log_ratio, [75, 25]))) if log_ratio.size else float("nan"),
        "n_both_in": n_both_in,
        "n_both_out": n_both_out,
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "mask_jaccard": float(n_both_out / union) if union else 1.0,
        "mask_kappa": float((p_obs - p_exp) / (1.0 - p_exp)) if p_exp < 1.0 else 1.0,
        "frac_out_a": float(out_a[finite].mean()) if finite.any() else float("nan"),
        "frac_out_b": float(out_b[finite].mean()) if finite.any() else float("nan"),
        "overlay": overlay,
    }


def score_fold(
    features: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    k: int = 5,
    alpha: float = 1.0,
    max_train: int = 15_000,
    max_test: int | None = None,
    variants: Sequence[str] = ("unweighted", "ridge"),
    metric: str = "euclidean",
    scaling: str = "none",
    extra_signals: bool = False,
    threshold: float = 1.0,
    seed: int = 0,
    min_side: int = 200,
    n_calibration: int = 0,
    alpha_conformal: float = 0.1,
    probe_features: np.ndarray | None = None,
    threshold_rule: str = "median",
    cv_folds: np.ndarray | None = None,
) -> dict:
    """Fit a Ridge head on one OOD fold and score DI-vs-error Spearman for each variant.

    Args:
        features: ``(N, F)`` per-pixel factor matrix (raw embedding OR dense SAE codes).
        y: ``(N,)`` continuous target (e.g. CHMv2 canopy height).
        train_mask: ``(N,)`` boolean training-fold membership.
        test_mask: ``(N,)`` boolean held-out-fold membership.
        k: Number of kNN neighbours for the dissimilarity index.
        alpha: Ridge penalty.
        max_train: cap on training pixels (subsampled, seed ``seed + n_train``).
        max_test: optional cap on held-out pixels (subsampled, same generator).
        variants: any of ``{"unweighted", "ridge", "rf"}`` (kNN distances; ``rf``
            needs sklearn), of :data:`SUPPORT_VARIANTS` (support novelty, mean
            surprisal of the active atoms under the training side's firing rates),
            and :data:`MAHALANOBIS_VARIANT` (shrunk Mahalanobis distance to the
            training side, on the rescaled features).
        metric: Distance metric, one of :data:`METRICS`.
        scaling: Per-feature rescaling applied before the distance, one of
            :data:`SCALINGS`. The Ridge head is always fit on the *unscaled*
            features, so the scaling changes the geometry and not the prediction;
            the ``ridge`` weights are that head's coefficients re-expressed in the
            rescaled units (``|beta_j| * scale_j``).
        extra_signals: Also score local point density and both
            :data:`SUPPORT_VARIANTS`. Adds a radius pass over the fold.
        threshold: In-domain boundary in median-normalised distance units, used
            for the reported saturation fraction under ``threshold_rule="median"``.
        min_side: minimum pixels per side for a valid fold.
        n_calibration: Training-side pixels held back from the head and the kNN
            reference set to calibrate the decision statistics of
            :mod:`geosae.eval.selective`. ``0`` skips them.
        alpha_conformal: Miscoverage level of the conformal selection.
        probe_features: ``(N, F')`` features of the deployed head whose error is
            scored, when it is not ``features`` itself. The distances, and the
            ``ridge`` weights, still come from a head fitted on ``features``, so
            one space's index can be judged on another space's model.
        threshold_rule: One of :data:`THRESHOLD_RULES`; decides only ``frac_out_*``.
            ``"cast"`` applies :func:`dissimilarity_index` to the same scaled and
            weighted features the variant's distance uses.
        cv_folds: ``(N,)`` CV fold id per row, used by ``threshold_rule="cast"``
            (``None``: nearest other training point).

    Returns:
        ``{}`` if the fold is too small, else a dict with ``rho_<signal>``, the
        per-pixel normalized distances ``d_train_<variant>`` / ``d_test_<variant>``,
        ``err`` (held-out ``|error|``), the held-out row indices ``test_idx``, and
        bookkeeping (``n_train``, ``n_test``, ``r2_train``, ``r2_test``).

        Also ``frac_out_<variant>`` — the share of held-out pixels the variant puts
        beyond ``threshold``. **Read this before comparing two spaces on a fold:**
        when it is near 1 in both, the fold is saturated and the two applicability
        *maps* cannot differ much however different the underlying geometries are,
        so a high map agreement says more about the split than about the spaces.
        With ``extra_signals``, ``n_atoms_unseen_in_fold`` and
        ``frac_pixels_using_unseen_atom`` report how much of the held-out fold
        activates dictionary atoms the in-fold data never used. Under
        ``threshold_rule="cast"``, also the held-out and training DI
        ``aoa_di_test_<variant>`` / ``aoa_di_train_<variant>`` and the threshold
        ``aoa_threshold_<variant>`` (:func:`aoa_threshold`, ``rule="quantile"``).

        With ``n_calibration``, also ``aurc_<variant>`` and ``aurc_gain_<variant>``
        (:func:`~geosae.eval.selective.risk_coverage`), ``coverage_<variant>`` and
        ``area_<variant>`` (:func:`~geosae.eval.selective.conformal_selection`),
        the calibrated ``threshold_<variant>`` and error ``halfwidth`` behind them,
        and the unselected ``coverage_all``.
    """
    _check_metric(metric)
    if threshold_rule not in THRESHOLD_RULES:
        raise ValueError(f"unknown threshold rule {threshold_rule!r}, expected one of {THRESHOLD_RULES}")
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if int(train_mask.sum()) < min_side or int(test_mask.sum()) < min_side:
        return {}
    rng = np.random.default_rng(seed + int(train_mask.sum()))
    train_idx = np.where(train_mask)[0]
    if len(train_idx) > max_train:
        train_idx = np.sort(rng.choice(train_idx, size=max_train, replace=False))
    test_idx = np.where(test_mask)[0]
    if max_test is not None and len(test_idx) > max_test:
        test_idx = np.sort(rng.choice(test_idx, size=max_test, replace=False))
    cal_idx = np.empty(0, dtype=np.int64)
    if n_calibration > 0:
        if len(train_idx) - n_calibration < min_side:
            return {}
        held = rng.choice(len(train_idx), size=n_calibration, replace=False)
        cal_idx = train_idx[np.sort(held)]
        train_idx = np.delete(train_idx, held)

    x_tr, x_te, x_cal = features[train_idx], features[test_idx], features[cal_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    coef, intercept = ridge_fit(x_tr, y_tr, alpha=alpha)
    if probe_features is None:
        h_tr, h_te, h_cal, h_coef, h_int = x_tr, x_te, x_cal, coef, intercept
    else:
        probe = np.asarray(probe_features, dtype=np.float64)
        h_tr, h_te, h_cal = probe[train_idx], probe[test_idx], probe[cal_idx]
        h_coef, h_int = ridge_fit(h_tr, y_tr, alpha=alpha)
    err = np.abs(ridge_predict(h_te, h_coef, h_int) - y_te)
    err_cal = np.abs(ridge_predict(h_cal, h_coef, h_int) - y[cal_idx])

    out: dict = {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "err": err,
        "test_idx": test_idx,
        "r2_train": _r2(y_tr, ridge_predict(h_tr, h_coef, h_int)),
        "r2_test": _r2(y_te, ridge_predict(h_te, h_coef, h_int)),
    }
    if n_calibration > 0:
        out["coverage_all"] = float((err <= selective.conformal_quantile(err_cal, alpha_conformal)).mean())
    scale = _feature_scale(x_tr, mode=scaling)
    s_tr, s_te, s_cal = x_tr / scale, x_te / scale, x_cal / scale
    signals = list(variants)
    if extra_signals:
        signals += [v for v in SUPPORT_VARIANTS if v not in signals]
    sup: FeatureSupport | None = None
    for v in signals:
        if v in SUPPORT_VARIANTS:
            sup = support_stats(x_tr) if sup is None else sup
            weights = np.abs(coef) if v == "support_probe" else None
            d_te, d_cal = (support_novelty(x, sup, mode="surprisal", weights=weights) for x in (x_te, x_cal))
            out[f"frac_out_{v}"] = float("nan")
        elif v == MAHALANOBIS_VARIANT:
            d_q = mahalanobis(s_tr, np.concatenate([s_te, s_cal]))
            d_te, d_cal = d_q[: len(test_idx)], d_q[len(test_idx) :]
            out[f"frac_out_{v}"] = float("nan")
        else:
            if v == "unweighted":
                w_sqrt = None
            elif v == "ridge":
                # The head's coefficients in the units of the rescaled features.
                w_sqrt = np.sqrt(ridge_weights(coef * scale))
            elif v == "rf":
                w_sqrt = np.sqrt(rf_importance_weights(x_tr, y_tr, seed=seed))
            else:
                raise ValueError(f"unknown variant {v!r}")
            w = 1.0 if w_sqrt is None else w_sqrt[None, :]
            d_tr, d_q, norm = knn_distance(s_tr * w, np.concatenate([s_te, s_cal]) * w, k=k, metric=metric)
            d_te, d_cal = d_q[: len(test_idx)], d_q[len(test_idx) :]
            out[f"d_train_{v}"] = d_tr
            out[f"norm_{v}"] = norm
            if threshold_rule == "cast":
                folds_tr = None if cv_folds is None else np.asarray(cv_folds)[train_idx]
                di_tr, di_te, aoa_thr = dissimilarity_index(s_tr * w, s_te * w, cv_folds=folds_tr, metric=metric)
                out[f"frac_out_{v}"] = float((di_te > aoa_thr).mean())
                out[f"aoa_di_test_{v}"], out[f"aoa_threshold_{v}"] = di_te, aoa_thr
                out[f"aoa_di_train_{v}"] = di_tr
            else:
                out[f"frac_out_{v}"] = float((d_te > threshold).mean())
        out[f"d_test_{v}"] = d_te
        out[f"rho_{v}"] = _spearman(d_te, err)
        if n_calibration > 0:
            rc = selective.risk_coverage(d_te, err)
            sel = selective.conformal_selection(err_cal, d_cal, err, d_te, alpha=alpha_conformal)
            out[f"aurc_{v}"], out[f"aurc_gain_{v}"] = rc.aurc, rc.gain
            out[f"coverage_{v}"], out[f"area_{v}"] = sel.coverage, sel.area
            out[f"threshold_{v}"], out["halfwidth"] = sel.threshold, sel.halfwidth

    if extra_signals:
        _, lpd_te, _ = local_point_density(s_tr, s_te, k=k, metric=metric)
        # A dense neighbourhood means applicable, so negate to keep "higher = worse".
        out["d_test_lpd"] = -lpd_te.astype(np.float64)
        out["rho_lpd"] = _spearman(out["d_test_lpd"], err)
        out["frac_out_lpd"] = float((lpd_te == 0).mean())
        out["support"] = sup
        unseen = sup.frequency == 0
        fires_unseen = (x_te[:, unseen] != 0).any(axis=1) if unseen.any() else np.zeros(len(x_te), dtype=bool)
        out["n_atoms_unseen_in_fold"] = int((unseen & ((x_te != 0).any(axis=0))).sum())
        out["frac_pixels_using_unseen_atom"] = float(fires_unseen.mean())
    return out


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_continent(lonlat: np.ndarray) -> np.ndarray:
    """Continent id per pixel from ``(N, 2)`` lon/lat (first matching bbox; -1 = none)."""
    lonlat = np.asarray(lonlat, dtype=np.float64)
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    out = np.full(len(lonlat), -1, dtype=int)
    for i, (lo0, lo1, la0, la1) in enumerate(CONTINENT_BBOXES.values()):
        m = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1) & (out < 0)
        out[m] = i
    return out


def continent_folds(lonlat: np.ndarray, *, leave_one_out: bool = False, min_train: int = 1000, min_test: int = 200):
    """Yield ``(name, train_mask, test_mask)`` continent folds.

    ``leave_one_out=False`` trains on one continent, tests on all others;
    ``True`` trains on all-but-one, tests on the held-out continent.
    """
    fid = assign_continent(lonlat)
    names = list(CONTINENT_BBOXES)
    for f in sorted(int(v) for v in np.unique(fid) if v >= 0):
        if leave_one_out:
            test = fid == f
            train = (fid >= 0) & ~test
        else:
            train = fid == f
            test = fid != f
        if train.sum() < min_train or test.sum() < min_test:
            continue
        yield names[f], train, test


def pft_folds(pft: np.ndarray, *, leave_one_out: bool = False, min_train: int = 1000, min_test: int = 200):
    """Yield ``(name, train_mask, test_mask)`` PFT folds from per-pixel MODIS-IGBP codes."""
    pft = np.asarray(pft)
    for name, code in PFT_CODES.items():
        if leave_one_out:
            test = pft == code
            train = (pft >= 0) & ~test
        else:
            train = pft == code
            test = pft != code
        if train.sum() < min_train or test.sum() < min_test:
            continue
        yield name, train, test


#: The four out-of-distribution split scenarios, as ``(axis, leave_one_out)``.
SPLIT_SCENARIOS: dict[str, tuple[str, bool]] = {
    "single_continent": ("continent", False),
    "leave_one_continent_out": ("continent", True),
    "single_pft": ("pft", False),
    "leave_one_pft_out": ("pft", True),
}


def scenario_folds(
    scenario: str,
    lonlat: np.ndarray,
    pft: np.ndarray,
    *,
    min_train: int = 1000,
    min_test: int = 200,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(fold_name, train_mask, test_mask)`` for one of :data:`SPLIT_SCENARIOS`."""
    if scenario not in SPLIT_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {tuple(SPLIT_SCENARIOS)}")
    axis, loo = SPLIT_SCENARIOS[scenario]
    gen = continent_folds(lonlat, leave_one_out=loo, min_train=min_train, min_test=min_test)
    if axis == "pft":
        gen = pft_folds(pft, leave_one_out=loo, min_train=min_train, min_test=min_test)
    yield from gen


# ---------------------------------------------------------------------------
# Sites: paired feature spaces over the same held-out pixels
# ---------------------------------------------------------------------------


@dataclass
class AoASites:
    """Paired factor spaces, targets and geography over one set of sampled pixels.

    Every array is indexed by the same site axis, so the raw embedding and the SAE
    code describe *the same pixels* and the two dissimilarity maps are directly
    comparable point by point.

    Attributes:
        lonlat: ``(N, 2)`` longitude/latitude in degrees.
        pft: ``(N,)`` MODIS-IGBP class code of the pixel's patch.
        spaces: Factor matrices keyed by space name, each ``(N, F_space)``.
        targets: Continuous regression targets keyed by name, each ``(N,)``.
        cell: ``(N,)`` source cell id in the data store.
        meta: Free-form provenance (checkpoint, split, sampling parameters).
    """

    lonlat: np.ndarray
    pft: np.ndarray
    spaces: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    cell: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.lonlat.shape[0])

    def subsample(self, n: int, *, seed: int = 0) -> AoASites:
        """A uniformly subsampled copy with at most ``n`` sites (order preserved)."""
        if n >= len(self):
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), size=n, replace=False))
        return AoASites(
            lonlat=self.lonlat[idx],
            pft=self.pft[idx],
            spaces={k: v[idx] for k, v in self.spaces.items()},
            targets={k: v[idx] for k, v in self.targets.items()},
            cell=self.cell[idx],
            meta={**self.meta, "subsampled_to": int(n), "subsample_seed": int(seed)},
        )

    def to_npz(self, path: str) -> str:
        """Save to a single compressed ``.npz`` (round-trips through :func:`load_sites`)."""
        import json

        payload = {
            "lonlat": self.lonlat,
            "pft": self.pft,
            "cell": self.cell,
            "meta": np.asarray(json.dumps(self.meta)),
            "space_names": np.asarray(list(self.spaces)),
            "target_names": np.asarray(list(self.targets)),
        }
        payload.update({f"space__{k}": v for k, v in self.spaces.items()})
        payload.update({f"target__{k}": v for k, v in self.targets.items()})
        np.savez_compressed(path, **payload)
        return str(path)


def load_sites(path: str) -> AoASites:
    """Load an :class:`AoASites` written by :meth:`AoASites.to_npz`."""
    import json

    z = np.load(path, allow_pickle=False)
    return AoASites(
        lonlat=z["lonlat"],
        pft=z["pft"],
        spaces={k: z[f"space__{k}"] for k in z["space_names"].tolist()},
        targets={k: z[f"target__{k}"] for k in z["target_names"].tolist()},
        cell=z["cell"],
        meta=json.loads(str(z["meta"])),
    )


def sample_sites(
    model: SparseAutoencoder,
    data: PatchData,
    *,
    targets: Sequence[str] = ("canopy_height", "elevation"),
    pft_key: str = "lc_modis_igbp",
    pixels_per_patch: int = 1,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
    progress_every: int = 200,
    cells: np.ndarray | None = None,
) -> AoASites:
    """Sample pixels from a split and encode them into paired raw and SAE spaces.

    Walks the split in cell order, reads each patch's embedding and covariates,
    draws ``pixels_per_patch`` pixels that have a finite value for *every* target
    and a valid land-cover class, and encodes exactly those pixels. The raw space
    is the embedding as the SAE consumes it (after the run's input transforms), so
    the comparison is between two encodings of one input, not between two
    preprocessing pipelines.

    With no ``cells`` this is a census of the split — one draw from every patch,
    not a subsample of the patches; ``cells`` narrows the walk to a named set,
    such as :attr:`~geosae.eval.pool.EvalPool.cells`, so an applicability number
    can rest on the same cells as the coverage numbers reported beside it.

    Args:
        model: A trained SAE from this package's zoo.
        data: A :class:`~geosae.data.dataset.PatchData` for the split to sample.
        targets: Covariate names to carry as regression targets.
        pft_key: Store key of the categorical class map used for the PFT splits.
        pixels_per_patch: Pixels drawn per patch.
        batch_size: Patches per read.
        device: Torch device for the encoder.
        seed: Seed of the pixel draw.
        progress_every: Print every N batches (0 to silence).
        cells: Store cell ids to restrict the walk to; ``None`` walks the split.

    Returns:
        An :class:`AoASites` with spaces ``{"raw", "sae"}``; its ``meta`` records
        the split, the seed and the cell restriction the sites came from.
    """
    import torch
    import zarr

    from geosae.eval.covariates import read_covariate_stack, resolve_covariates

    covs = resolve_covariates(list(targets))
    store = zarr.open_group(data.zarr_path, mode="r")
    dev = torch.device(device)
    model = model.to(dev).eval()

    chunks: dict[str, list[np.ndarray]] = {"raw": [], "sae": [], "lonlat": [], "pft": [], "cell": []}
    tchunks: dict[str, list[np.ndarray]] = {c.name: [] for c in covs}
    n_batches = data.num_batches(batch_size)

    keep_cells = None if cells is None else np.asarray(cells, dtype=np.int64)
    for bi, batch in enumerate(data.iter_batches(batch_size, shuffle=False)):
        cids = batch["cell"].numpy()
        lo, hi = int(cids.min()), int(cids.max()) + 1
        sel = cids - lo
        b = len(cids)

        tvals = read_covariate_stack(store, covs, lo, hi, sel)  # (b, P, T)
        pft = np.asarray(store[pft_key][lo:hi])[sel].reshape(b, -1)
        valid_patch = np.asarray(store[f"mask_{pft_key}"][lo:hi])[sel].astype(bool)
        usable = np.isfinite(tvals).all(axis=-1) & (pft > 0) & valid_patch[:, None]
        if keep_cells is not None:
            usable &= np.isin(cids, keep_cells)[:, None]

        rows, cols = [], []
        for i in np.flatnonzero(usable.any(axis=1)):
            cand = np.flatnonzero(usable[i])
            # Seeded per cell, not per walk: a cell yields the same pixels
            # whether or not the other cells were walked, so restricting to a
            # pool's cells selects rows of the full draw instead of re-drawing.
            cell_rng = np.random.default_rng([seed, int(cids[i])])
            take = cell_rng.choice(cand, size=min(pixels_per_patch, cand.size), replace=False)
            rows.extend([i] * take.size)
            cols.extend(take.tolist())
        if not rows:
            continue
        rows, cols = np.asarray(rows), np.asarray(cols)

        x = batch["x"].reshape(b, -1, batch["x"].shape[-1])[rows, cols].to(dev)
        with torch.no_grad():
            z = sparse_codes(model, x)
        chunks["raw"].append(x.float().cpu().numpy())
        chunks["sae"].append(z.float().cpu().numpy())
        chunks["lonlat"].append(np.column_stack([batch["lon"].numpy()[rows], batch["lat"].numpy()[rows]]))
        chunks["pft"].append(pft[rows, cols].astype(np.int16))
        chunks["cell"].append(cids[rows])
        for t, c in enumerate(covs):
            tchunks[c.name].append(tvals[rows, cols, t])
        if progress_every and (bi % progress_every == 0):
            print(f"[aoa] batch {bi}/{n_batches} sites={sum(len(a) for a in chunks['cell'])}", flush=True)

    if not chunks["cell"]:
        raise ValueError("no usable pixels found: every patch was masked out by the targets or the PFT map")
    return AoASites(
        lonlat=np.concatenate(chunks["lonlat"]),
        pft=np.concatenate(chunks["pft"]),
        spaces={"raw": np.concatenate(chunks["raw"]), "sae": np.concatenate(chunks["sae"])},
        targets={n: np.concatenate(v) for n, v in tchunks.items()},
        cell=np.concatenate(chunks["cell"]),
        meta={
            "split": getattr(data, "split", None),
            "embedding_name": data.embedding_name,
            "pixels_per_patch": int(pixels_per_patch),
            "seed": int(seed),
            "targets": list(targets),
            "pft_key": pft_key,
            "n_cells_requested": None if keep_cells is None else int(keep_cells.size),
        },
    )


# ---------------------------------------------------------------------------
# Scenario sweep
# ---------------------------------------------------------------------------


def run_folds(
    sites: AoASites,
    folds: Iterable[tuple[str, np.ndarray, np.ndarray]],
    *,
    scenario: str,
    fold_meta: Mapping[str, dict] | None = None,
    spaces: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    variants: Sequence[str] = ("unweighted", "ridge"),
    metric: str = "euclidean",
    scaling: str = "none",
    extra_signals: bool = False,
    k: int = 5,
    alpha: float = 1.0,
    max_train: int = 15_000,
    max_test: int | None = None,
    min_side: int = 200,
    n_calibration: int = 0,
    alpha_conformal: float = 0.1,
    probe_space: str | None = None,
    threshold_rule: str = "median",
    cv_folds: np.ndarray | None = None,
    seed: int = 0,
    keep_fields: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Score every (fold, target, space, signal) cell of an arbitrary set of folds.

    Every space sees the identical fold, so the dissimilarity maps of a fold are
    paired point by point and can be differenced directly.

    Args:
        sites: The sampled sites.
        folds: ``(fold_name, train_mask, test_mask)`` over the site axis.
        scenario: Name recorded in the ``scenario`` column.
        fold_meta: Extra columns per fold name (e.g. a graded fold's gap).
        spaces: Space names to score (default: all of ``sites.spaces``).
        target_names: Targets to score (default: all of ``sites.targets``).
        variants: kNN distance variants, see :func:`score_fold`.
        metric: Distance metric, one of :data:`METRICS`.
        scaling: Per-feature rescaling, one of :data:`SCALINGS`.
        extra_signals: Also score local point density and support novelty.
        k: Neighbours for the dissimilarity index.
        alpha: Ridge penalty.
        max_train: Cap on in-fold pixels.
        max_test: Cap on held-out pixels.
        min_side: Minimum per-side size inside :func:`score_fold`.
        n_calibration: Calibration pixels for the decision statistics, see
            :func:`score_fold`; ``0`` skips them.
        alpha_conformal: Miscoverage level of the conformal selection.
        probe_space: Space whose head's error every space's index is scored
            against (see ``probe_features`` of :func:`score_fold`); ``None`` scores
            each space against its own head. Recorded in the ``probe`` column.
        threshold_rule: In-domain boundary behind ``frac_out``, see :func:`score_fold`.
        cv_folds: ``(N,)`` CV fold id per site for ``threshold_rule="cast"``.
        seed: Seed of the subsampling.
        keep_fields: Attach the per-pixel ``d_test``, ``err`` and ``test_idx``
            arrays to each record (needed for maps; large).
        verbose: Print one line per fold.

    Returns:
        One flat record per (fold, target, space, signal) with ``rho``,
        ``frac_out``, the decision statistics when calibrated (``aurc``,
        ``aurc_gain``, ``coverage``, ``area``, ``coverage_all``) and the fold
        bookkeeping.
    """
    space_names = list(spaces) if spaces is not None else list(sites.spaces)
    tnames = list(target_names) if target_names is not None else list(sites.targets)
    fold_meta = fold_meta or {}
    nan = float("nan")

    records: list[dict] = []
    for fold, train_mask, test_mask in folds:
        for tname in tnames:
            y = sites.targets[tname]
            finite = np.isfinite(y)
            for space in space_names:
                rec = score_fold(
                    sites.spaces[space],
                    np.nan_to_num(y),
                    train_mask & finite,
                    test_mask & finite,
                    k=k,
                    alpha=alpha,
                    max_train=max_train,
                    max_test=max_test,
                    variants=variants,
                    metric=metric,
                    scaling=scaling,
                    extra_signals=extra_signals,
                    seed=seed,
                    min_side=min_side,
                    n_calibration=n_calibration,
                    alpha_conformal=alpha_conformal,
                    probe_features=None if probe_space is None else sites.spaces[probe_space],
                    threshold_rule=threshold_rule,
                    cv_folds=cv_folds,
                )
                if not rec:
                    continue
                base = {
                    "scenario": scenario,
                    "fold": fold,
                    **fold_meta.get(fold, {}),
                    "target": tname,
                    "space": space,
                    "probe": space if probe_space is None else probe_space,
                    "metric": metric,
                    "scaling": scaling,
                    "k": k,
                    "n_train": rec["n_train"],
                    "n_test": rec["n_test"],
                    "r2_train": rec["r2_train"],
                    "r2_test": rec["r2_test"],
                    "n_atoms_unseen_in_fold": rec.get("n_atoms_unseen_in_fold"),
                    "frac_pixels_using_unseen_atom": rec.get("frac_pixels_using_unseen_atom"),
                }
                if n_calibration > 0:
                    base["coverage_all"] = rec["coverage_all"]
                signals = [key[len("rho_") :] for key in rec if key.startswith("rho_")]
                for sig in signals:
                    out = {
                        **base,
                        "variant": sig,
                        "rho": rec[f"rho_{sig}"],
                        "frac_out": rec.get(f"frac_out_{sig}", nan),
                    }
                    if n_calibration > 0:
                        for stat in ("aurc", "aurc_gain", "coverage", "area"):
                            out[stat] = rec.get(f"{stat}_{sig}", nan)
                    if keep_fields:
                        out["d_test"] = rec[f"d_test_{sig}"]
                        out["err"] = rec["err"]
                        out["test_idx"] = rec["test_idx"]
                    records.append(out)
                if verbose:
                    shown = " ".join(f"{s}={rec[f'rho_{s}']:+.3f}" for s in signals)
                    print(
                        f"[aoa] {scenario}/{fold} {tname} {space:>4} "
                        f"n={rec['n_train']}/{rec['n_test']} R2te={rec['r2_test']:+.3f} {shown}",
                        flush=True,
                    )
    return records


def run_scenarios(
    sites: AoASites,
    *,
    scenarios: Sequence[str] = tuple(SPLIT_SCENARIOS),
    min_train: int = 1000,
    min_test: int = 200,
    **kwargs: Any,
) -> list[dict]:
    """Score every (scenario, fold, target, space, signal) cell of the AoA grid.

    Args:
        sites: The sampled sites.
        scenarios: Which of :data:`SPLIT_SCENARIOS` to run.
        min_train: Minimum fold-train size for a fold to be emitted.
        min_test: Minimum fold-test size for a fold to be emitted.
        **kwargs: Passed to :func:`run_folds` (spaces, targets, geometry, head,
            calibration, sizes).

    Returns:
        The concatenated :func:`run_folds` records of every scenario.
    """
    for s in scenarios:
        if s not in SPLIT_SCENARIOS:
            raise ValueError(f"unknown scenario {s!r}, expected one of {tuple(SPLIT_SCENARIOS)}")
    records: list[dict] = []
    for scenario in scenarios:
        folds = scenario_folds(scenario, sites.lonlat, sites.pft, min_train=min_train, min_test=min_test)
        records += run_folds(sites, folds, scenario=scenario, **kwargs)
    return records


__all__ = [
    "CONTINENT_BBOXES",
    "PFT_CODES",
    "METRICS",
    "SCALINGS",
    "SUPPORT_VARIANTS",
    "MAHALANOBIS_VARIANT",
    "THRESHOLD_RULES",
    "AOA_THRESHOLD_RULES",
    "SPLIT_SCENARIOS",
    "AoASites",
    "FeatureSupport",
    "ridge_fit",
    "ridge_predict",
    "scale_features",
    "knn_distance",
    "dissimilarity_index",
    "aoa_threshold",
    "local_point_density",
    "hubness",
    "ridge_weights",
    "rf_importance_weights",
    "support_stats",
    "support_novelty",
    "di_agreement",
    "di_rank_profile",
    "extrapolating_atoms",
    "score_fold",
    "assign_continent",
    "continent_folds",
    "pft_folds",
    "scenario_folds",
    "sample_sites",
    "load_sites",
    "run_folds",
    "run_scenarios",
]

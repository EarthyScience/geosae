"""Spatial scale of features: normalised semivariance curves and correlation lengths.

A feature's semivariance at separation ``h`` is half the mean squared difference of its
activation over pixel pairs ``h`` apart. Within a patch it is read from horizontal and
vertical pairs of the pixel grid; between two patches with pixel means ``m_p, m_q`` and
pixel variances ``v_p, v_q`` it follows exactly from ``0.5 [(m_p - m_q)^2 + v_p + v_q]``,
averaged over the patch pairs whose centres fall into a distance bin. Dividing by the
feature's variance over all pixels gives a curve that rises from the *nugget* (the first
lag) towards ``1``.

The *correlation length* is the first separation at which the normalised curve reaches
``level`` (default ``1 - 1/e``, where the correlation has fallen to ``1/e``), interpolated
in log distance between the two sampled separations around that first crossing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Default level of the normalised semivariance at which the correlation length is read.
CORRELATION_LEVEL = 1.0 - float(np.exp(-1.0))

#: Status codes of :func:`correlation_lengths`.
RESOLVED, BELOW_FIRST_LAG, IN_GAP, NEVER_REACHED = 0, 1, 2, 3


def haversine_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle distances in km between all pairs of points, ``(P, P)``."""
    lo, la = np.radians(np.asarray(lon, dtype=np.float64)), np.radians(np.asarray(lat, dtype=np.float64))
    d = (
        np.sin((la[:, None] - la[None]) / 2) ** 2
        + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(d, 0.0, 1.0)))


def patch_moments(z: np.ndarray, lags: Sequence[int]) -> dict[str, np.ndarray]:
    """The per-patch statistics :func:`semivariance_from_moments` needs, for one chunk of patches.

    Moments of disjoint chunks of patches concatenate (``mean``, ``var``) and add up (``lag_ss``,
    ``lag_n``), so a sample too large to hold as one dense array is scored chunk by chunk.

    Args:
        z: ``(P, H, W, F)`` activations on the full pixel grid of ``P`` patches.
        lags: Within-patch lags in pixels, each smaller than ``min(H, W)``.

    Returns:
        ``mean`` and ``var`` ``(P, F)``, each patch's pixel mean and variance; ``lag_ss``
        ``(len(lags), F)``, the summed squared differences over the horizontal and vertical pixel
        pairs at each lag; ``lag_n`` ``(len(lags),)``, the number of those pairs.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 4:
        raise ValueError(f"z must be (P, H, W, F), got shape {z.shape}")
    n_patch, h, w, n_feat = z.shape
    lags = [int(lag) for lag in lags]
    if any(lag < 1 or lag >= min(h, w) for lag in lags):
        raise ValueError(f"lags must lie in [1, {min(h, w) - 1}], got {lags}")
    lag_ss = np.empty((len(lags), n_feat))
    lag_n = np.empty(len(lags))
    for i, lag in enumerate(lags):
        dh = (z[:, :, lag:] - z[:, :, :-lag]) ** 2
        dv = (z[:, lag:, :] - z[:, :-lag, :]) ** 2
        lag_ss[i] = dh.sum((0, 1, 2)) + dv.sum((0, 1, 2))
        lag_n[i] = dh[..., 0].size + dv[..., 0].size
    flat = z.reshape(n_patch, -1, n_feat)
    return {"mean": flat.mean(1), "var": flat.var(1), "lag_ss": lag_ss, "lag_n": lag_n}


def semivariance_from_moments(
    moments: dict[str, np.ndarray],
    dist_km: np.ndarray,
    *,
    pixel_m: float,
    lags: Sequence[int],
    edges_km: Sequence[float],
    min_pairs: int = 300,
) -> dict[str, np.ndarray]:
    """Normalised semivariance of every feature from the moments of :func:`patch_moments`.

    Args:
        moments: The moments of all ``P`` patches (chunks concatenated and summed).
        dist_km: ``(P, P)`` distances between patch centres.
        pixel_m: Pixel size in metres; within-patch lag ``L`` is ``L * pixel_m``.
        lags: The within-patch lags the moments were computed at.
        edges_km: Increasing bin edges of the between-patch separations.
        min_pairs: A bin with fewer patch pairs is returned as ``nan``.

    Returns:
        As :func:`semivariance_curves`.
    """
    mean, pvar = np.asarray(moments["mean"], np.float64), np.asarray(moments["var"], np.float64)
    n_patch, n_feat = mean.shape
    dist_km = np.asarray(dist_km, dtype=np.float64)
    edges = np.asarray(edges_km, dtype=np.float64)
    off = ~np.eye(n_patch, dtype=bool)
    masks = [(dist_km >= edges[b]) & (dist_km < edges[b + 1]) & off for b in range(len(edges) - 1)]
    n_pairs = np.array([m.sum() for m in masks], dtype=np.int64)
    centres = [
        float(dist_km[m].mean()) if n_pairs[b] else float(np.sqrt(edges[b] * edges[b + 1])) for b, m in enumerate(masks)
    ]
    # Equal-size patches: the variance over all pixels is the mean within-patch variance plus
    # the variance of the patch means.
    var = pvar.mean(0) + mean.var(0)
    g = np.full((n_feat, len(lags) + len(masks)), np.nan)
    g[:, : len(lags)] = (0.5 * np.asarray(moments["lag_ss"]) / np.asarray(moments["lag_n"])[:, None]).T
    for b, m in enumerate(masks):
        if n_pairs[b] < int(min_pairs):
            continue
        a = m.astype(np.float64)
        deg = a.sum(1)[:, None]
        g[:, len(lags) + b] = ((deg * (mean**2 + pvar)).sum(0) - (mean * (a @ mean)).sum(0)) / n_pairs[b]
    ok = var > 0
    curve = np.full_like(g, np.nan)
    within = np.full(n_feat, np.nan)
    curve[ok] = g[ok] / var[ok, None]
    within[ok] = pvar.mean(0)[ok] / var[ok]
    dist = np.array([lag * float(pixel_m) / 1000.0 for lag in lags] + centres)
    return {"curve": curve, "dist_km": dist, "within": within, "variance": var, "n_pairs": n_pairs}


def semivariance_curves(
    z: np.ndarray,
    dist_km: np.ndarray,
    *,
    pixel_m: float,
    lags: Sequence[int],
    edges_km: Sequence[float],
    min_pairs: int = 300,
    chunk_patches: int = 64,
) -> dict[str, np.ndarray]:
    """Normalised semivariance of every feature, within and between patches.

    Args:
        z: ``(P, H, W, F)`` activations on the full pixel grid of ``P`` patches.
        dist_km: ``(P, P)`` distances between patch centres.
        pixel_m: Pixel size in metres; within-patch lag ``L`` is ``L * pixel_m``.
        lags: Within-patch lags in pixels, each smaller than ``min(H, W)``.
        edges_km: Increasing bin edges of the between-patch separations.
        min_pairs: A bin with fewer patch pairs is returned as ``nan``.
        chunk_patches: Patches processed at once (bounds memory).

    Returns:
        ``curve`` ``(F, len(lags) + n_bins)`` normalised semivariance (``nan`` for a
        feature without variance), ``dist_km`` ``(len(lags) + n_bins,)`` the separation of
        each column (the mean pair distance of a bin, or its geometric centre when the
        bin is empty), ``within`` ``(F,)`` the share of a feature's variance within
        patches, ``variance`` ``(F,)`` and ``n_pairs`` ``(n_bins,)`` patch pairs per bin.
    """
    z = np.asarray(z)
    if z.ndim != 4:
        raise ValueError(f"z must be (P, H, W, F), got shape {z.shape}")
    step = int(chunk_patches)
    moments = merge_moments([patch_moments(z[p : p + step], lags) for p in range(0, len(z), step)])
    return semivariance_from_moments(
        moments, dist_km, pixel_m=pixel_m, lags=lags, edges_km=edges_km, min_pairs=min_pairs
    )


def merge_moments(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The moments of consecutive patch chunks (from :func:`patch_moments`) as those of all patches."""
    return {
        "mean": np.concatenate([c["mean"] for c in chunks]),
        "var": np.concatenate([c["var"] for c in chunks]),
        "lag_ss": np.sum([c["lag_ss"] for c in chunks], axis=0),
        "lag_n": np.sum([c["lag_n"] for c in chunks], axis=0),
    }


def nested_sample(
    lon: np.ndarray, lat: np.ndarray, *, n_anchors: int, ranks: Sequence[int], seed: int = 0
) -> np.ndarray:
    """A nested sample of places: random anchors, each with its neighbours of given distance ranks.

    Short and long separations then come from the same random places, so one population of
    patches feeds every distance bin of a variogram.

    Args:
        lon: ``(N,)`` longitudes of the candidate places, degrees.
        lat: ``(N,)`` latitudes, degrees.
        n_anchors: Anchors drawn at random (capped at ``N``).
        ranks: Great-circle distance ranks of the neighbours added per anchor (``1`` is the
            nearest other place).
        seed: Seeds the anchor draw.

    Returns:
        Sorted, unique indices into ``lon`` / ``lat``.
    """
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    ranks = [int(r) for r in ranks]
    picked: set[int] = set()
    rng = np.random.default_rng(seed)
    for a in rng.choice(lon.size, size=min(int(n_anchors), lon.size), replace=False):
        h = np.sin((lat - lat[a]) / 2) ** 2 + np.cos(lat[a]) * np.cos(lat) * np.sin((lon - lon[a]) / 2) ** 2
        h[a] = -1.0  # the anchor is its own rank 0, ties notwithstanding
        order = np.argsort(h, kind="stable")
        picked.update(int(order[r]) for r in (0, *ranks) if r < order.size)
    return np.sort(np.fromiter(picked, dtype=np.int64))


def correlation_lengths(
    curve: np.ndarray, dist_km: np.ndarray, *, n_within: int, level: float = CORRELATION_LEVEL
) -> dict[str, np.ndarray]:
    """Nugget, correlation length and its status for every normalised curve.

    Args:
        curve: ``(F, S)`` normalised semivariance (``nan`` marks an unsampled separation).
        dist_km: ``(S,)`` increasing separations of the columns.
        n_within: Number of leading within-patch columns; a crossing between the last
            within-patch column and the first between-patch one falls into the unsampled
            gap.
        level: Normalised semivariance at which the length is read.

    Returns:
        ``nugget`` ``(F,)`` (the first column), ``length_km`` ``(F,)`` and ``status``
        ``(F,)``: :data:`RESOLVED`; :data:`BELOW_FIRST_LAG` (already at ``level`` at the
        first lag, length set to that lag); :data:`IN_GAP` (interpolated across the gap);
        :data:`NEVER_REACHED` (length ``nan``).
    """
    curve = np.asarray(curve, dtype=np.float64)
    dist_km = np.asarray(dist_km, dtype=np.float64)
    length = np.full(len(curve), np.nan)
    status = np.full(len(curve), RESOLVED, dtype=np.int8)
    last_within = int(n_within) - 1
    for f, c in enumerate(curve):
        ok = np.isfinite(c)
        if ok[0] and c[0] >= level:
            length[f], status[f] = dist_km[0], BELOW_FIRST_LAG
            continue
        hits = np.flatnonzero(ok & (c >= level))
        if hits.size == 0:
            status[f] = NEVER_REACHED
            continue
        j = hits[0]
        i = np.flatnonzero(ok[:j])[-1]
        t = (level - c[i]) / (c[j] - c[i])
        length[f] = float(np.exp(np.log(dist_km[i]) + t * (np.log(dist_km[j]) - np.log(dist_km[i]))))
        if i <= last_within < j:
            status[f] = IN_GAP
    return {"nugget": curve[:, 0].copy(), "length_km": length, "status": status}

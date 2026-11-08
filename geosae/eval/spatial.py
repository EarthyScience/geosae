"""Spatial-coherence summaries: patch-mean Moran's I top-quintile + variogram half-life.

The spatial-coherence axis reports two per-feature fields reduced to scalars:

* **patch-mean Moran's I** (from :func:`geosae.eval.auroc_moran.patch_mean_moran`,
  or any per-feature array) summarised by the **top-20% (top-quintile) mean** and
  the **median** over alive features. :func:`top_quintile_mean` /
  :func:`summarise_per_feature` do that reduction.
* **variogram half-life** (km) — the distance at which a feature's semivariance
  first reaches half its sill (range proxy). :func:`variogram_curve` builds the
  per-feature log-distance semivariogram from a pixel sample and its lon/lat, and
  :func:`halflife_from_curve` extracts the half-life by log-linear interpolation.

The default ``d_max_km=100`` is a long-range/cross-patch window; pass a
different ``d_max_km`` to focus on shorter- or longer-range coherence.

The pixels and their coordinates come from the caller. To put the half-life on
the frozen evaluation pool — the same pixels every other labelled metric scores
— encode :meth:`~geosae.eval.pool.EvalPool.registry_sample` and pass
:meth:`~geosae.eval.pool.EvalPool.pixel_lonlat` alongside it::

    hl = variogram_halflife(codes, *pool.pixel_lonlat())

That geolocates a pixel by its cell centroid, which resolves the cross-patch
window and not the sub-kilometre structure inside a single cell.

:func:`moran_i` and :func:`moran_permutation_test` give the per-map Moran's I under a k-nearest-
neighbour weight on the sphere (:func:`knn_neighbours`), the latter against a permutation null.

Inside a patch, :func:`map_fragmentation` counts the connected pieces of an
atom's active map against its dense projection thresholded to the same cover, so
the fragments the sparsifier adds can be told from those already in the input.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def top_quintile_mean(values: np.ndarray, *, pct: float = 0.20) -> float:
    """Mean of the top ``pct`` fraction of finite values (default top-20%)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    k = max(1, int(round(v.size * pct)))
    return float(np.partition(v, -k)[-k:].mean())


def summarise_per_feature(values: np.ndarray, *, pct: float = 0.20) -> dict[str, float]:
    """``{top_quintile, median, mean, n}`` over the finite entries of a per-feature array."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"top_quintile": float("nan"), "median": float("nan"), "mean": float("nan"), "n": 0.0}
    return {
        "top_quintile": top_quintile_mean(v, pct=pct),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "n": float(v.size),
    }


def _lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    latr, lonr = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)], axis=1)


def knn_neighbours(lon: np.ndarray, lat: np.ndarray, k: int) -> np.ndarray:
    """Indices of each point's ``k`` nearest neighbours on the sphere, itself excluded.

    Args:
        lon: ``(n,)`` longitudes, degrees.
        lat: ``(n,)`` latitudes, degrees.
        k: Neighbours per point (fewer when ``n <= k``).

    Returns:
        ``(n, min(k, n - 1))`` int array, nearest first.
    """
    from scipy.spatial import cKDTree

    # Chord distance on the unit sphere ranks neighbours exactly as great-circle distance does.
    xyz = _lonlat_to_xyz(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    _, nbr = cKDTree(xyz).query(xyz, k=min(k + 1, xyz.shape[0]))
    return np.atleast_2d(nbr)[:, 1:]


def distant_neighbours(
    lon: np.ndarray,
    lat: np.ndarray,
    k: int,
    min_km: float,
    *,
    ref_lon: np.ndarray | None = None,
    ref_lat: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of each point's ``k`` nearest neighbours on the sphere farther than ``min_km``.

    Args:
        lon: ``(n,)`` longitudes, degrees.
        lat: ``(n,)`` latitudes, degrees.
        k: Neighbours per point.
        min_km: Great-circle distance in km within which points are skipped (the point itself
            included).
        ref_lon: ``(m,)`` longitudes of the points to search instead of the points themselves.
        ref_lat: ``(m,)`` their latitudes.

    Returns:
        ``(n, k)`` int array of indices into the searched points, nearest first.

    Raises:
        ValueError: Some point has fewer than ``k`` points farther than ``min_km``.
    """
    from scipy.spatial import cKDTree

    xyz = _lonlat_to_xyz(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    ref = xyz if ref_lon is None else _lonlat_to_xyz(np.asarray(ref_lon, np.float64), np.asarray(ref_lat, np.float64))
    tree = cKDTree(ref)
    chord = 2.0 * np.sin(min(float(min_km), np.pi * EARTH_RADIUS_KM) / (2.0 * EARTH_RADIUS_KM))
    near = tree.query_ball_point(xyz, chord, return_length=True)
    dist, nbr = tree.query(xyz, k=min(int(np.max(near)) + k, len(ref)))
    dist, nbr = dist.reshape(len(xyz), -1), nbr.reshape(len(xyz), -1)
    far = dist > chord
    if (far.sum(1) < k).any():
        raise ValueError(f"some point has fewer than {k} points farther than {min_km} km")
    # the first k far neighbours of each row, in distance order
    pick = np.argsort(~far, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(nbr, pick, axis=1)


def _centred(values: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if not np.isfinite(v).all():
        raise ValueError("Moran's I needs finite values; subset the points first")
    return v - v.mean(axis=1, keepdims=True)


def _moran(z: np.ndarray, nbr: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = (z * z[:, nbr].mean(axis=2)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def moran_i(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Moran's I of each map under the row-normalised neighbour weight.

    Args:
        values: ``(n,)`` or ``(F, n)`` finite maps over the points.
        neighbours: ``(n, k)`` neighbour indices, e.g. from :func:`knn_neighbours`.

    Returns:
        ``(F,)`` Moran's I; NaN for a constant map. Its expectation under spatial randomness is
        ``-1 / (n - 1)``.

    Raises:
        ValueError: If a value is not finite.
    """
    z = _centred(values)
    return _moran(z, neighbours, (z * z).sum(axis=1))


def moran_permutation_test(
    values: np.ndarray, neighbours: np.ndarray, *, n_perm: int = 999, seed: int = 0
) -> dict[str, np.ndarray]:
    """Moran's I of each map against the null of values randomly permuted over the points.

    Args:
        values: ``(n,)`` or ``(F, n)`` finite maps over the points.
        neighbours: ``(n, k)`` neighbour indices, e.g. from :func:`knn_neighbours`.
        n_perm: Permutations; the smallest attainable p-value is ``1 / (n_perm + 1)``.
        seed: Seed of the permutations (shared by all maps).

    Returns:
        ``(F,)`` arrays: ``moran_i``, the null's ``null_mean`` and ``null_sd``, ``z`` (standardised
        against the null) and the one-sided ``p_value`` of a Moran's I at least as large.
    """
    z = _centred(values)
    den = (z * z).sum(axis=1)
    observed = _moran(z, neighbours, den)
    rng = np.random.default_rng(seed)
    null = np.stack([_moran(z[:, rng.permutation(z.shape[1])], neighbours, den) for _ in range(n_perm)])
    mean, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (observed - mean) / sd
    return {
        "moran_i": observed,
        "null_mean": mean,
        "null_sd": sd,
        "z": zscore,
        "p_value": (1 + (null >= observed).sum(axis=0)) / (n_perm + 1),
    }


def variogram_curve(
    codes: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    n_pairs: int = 1_000_000,
    n_bins: int = 30,
    d_min_km: float = 1e-3,
    d_max_km: float = 100.0,
    seed: int = 42,
    extra_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    min_pairs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature log-distance semivariogram from a per-pixel sample.

    Draws ``n_pairs`` random pixel pairs, bins their haversine distance into
    ``n_bins`` log-spaced bins in ``[d_min_km, d_max_km]``, and accumulates the
    per-feature semivariance ``0.5 * mean((z_i - z_j)^2)`` in each bin.

    Args:
        codes: ``(N, F)`` per-pixel activations.
        lon: ``(N,)`` pixel longitude (degrees).
        lat: ``(N,)`` pixel latitude (degrees).
        n_pairs: Number of random pixel pairs to draw.
        n_bins: Number of log-spaced distance bins.
        d_min_km: Lower edge of the distance range (km).
        d_max_km: Upper edge of the distance range (km).
        seed: RNG seed for the pair sample.
        extra_pairs: ``(i, j)`` index arrays of pairs added to the random ones, e.g.
            nearest neighbours, which random pairs of a sparse sample rarely reach.
        min_pairs: Bins holding fewer pairs are NaN.

    Returns:
        ``(curve (F, n_bins), bin_centers_km (n_bins,))`` — ``curve`` is the
        semivariance per feature per bin (empty bins are NaN).
    """
    codes = np.asarray(codes, dtype=np.float64)
    n, f = codes.shape
    xyz = _lonlat_to_xyz(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2) if n > 1 else 0
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    if extra_pairs is not None:
        i = np.concatenate([i, np.asarray(extra_pairs[0], np.int64)])
        j = np.concatenate([j, np.asarray(extra_pairs[1], np.int64)])
    keep = i != j
    i, j = i[keep], j[keep]
    chord = np.clip(np.linalg.norm(xyz[i] - xyz[j], axis=1), 0.0, 2.0)
    d_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(chord / 2.0)

    bin_edges = np.logspace(np.log10(d_min_km), np.log10(d_max_km), n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, d_km, side="right") - 1, 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)

    bin_sum = np.zeros((n_bins, f), dtype=np.float64)
    diff = codes[i] - codes[j]
    np.add.at(bin_sum, bin_idx, diff * diff)
    counts_safe = np.where((bin_counts > 0) & (bin_counts >= min_pairs), bin_counts, np.nan)
    curve = (0.5 * bin_sum / counts_safe[:, None]).T  # (F, n_bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return curve.astype(np.float32), bin_centers


def halflife_from_curve(bin_centers_km: np.ndarray, curve: np.ndarray, *, fraction: float = 0.5) -> np.ndarray:
    """Per-feature variogram half-life (km) via log-linear interpolation.

    For each feature the sill is ``max_bin`` semivariance; the half-life is the
    distance where the curve first reaches ``fraction * sill`` (interpolated in log
    distance between the two straddling bins). NaN where undefined. ``fraction=0.95``
    gives the practical range instead of the half-life.
    """
    curve = np.asarray(curve, dtype=np.float64)
    n_features = curve.shape[0]
    halflife = np.full(n_features, np.nan, dtype=np.float32)
    log_centers = np.log(np.maximum(np.asarray(bin_centers_km, dtype=np.float64), 1e-6))
    for f in range(n_features):
        row = curve[f]
        if not np.isfinite(row).any():
            continue
        sill = float(np.nanmax(row))
        if not np.isfinite(sill) or sill <= 0:
            continue
        target = fraction * sill
        crossed = np.where(row >= target)[0]
        if crossed.size == 0:
            continue
        b = int(crossed[0])
        if b == 0 or not np.isfinite(row[b - 1]):
            halflife[f] = float(bin_centers_km[b])
            continue
        y0, y1 = float(row[b - 1]), float(row[b])
        if y1 == y0:
            halflife[f] = float(bin_centers_km[b])
            continue
        t = (target - y0) / (y1 - y0)
        log_d = log_centers[b - 1] + t * (log_centers[b] - log_centers[b - 1])
        halflife[f] = float(np.exp(log_d))
    return halflife


def variogram_halflife(
    codes: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    **kw,
) -> np.ndarray:
    """Convenience: per-feature variogram half-life (km) from a pixel sample."""
    curve, centers = variogram_curve(codes, lon, lat, **kw)
    return halflife_from_curve(centers, curve)


def _components(mask: np.ndarray) -> tuple[int, int]:
    """Connected components (4-connectivity) of a boolean map, and how many are single pixels."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), int((sizes == 1).sum())


def map_fragmentation(active: np.ndarray, projection: np.ndarray) -> dict[str, int]:
    """Fragments of one partial-cover activation map against its dense projection at matched cover.

    The reference keeps the ``n`` pixels of ``projection`` with the largest values, ``n`` being the
    number of active pixels, so both maps cover the same area and differ only in how it is cut up.

    Args:
        active: ``(H, W)`` bool, where the atom fires.
        projection: ``(H, W)`` the atom's dense projection over the same patch.

    Returns:
        ``cover`` (active pixels), ``components`` and ``isolated`` (single-pixel components) of the
        active map, and ``dense_components`` / ``dense_isolated`` of the thresholded projection.
    """
    a = np.asarray(active, dtype=bool)
    n = int(a.sum())
    flat = np.asarray(projection, dtype=np.float64).ravel()
    ref = np.zeros(flat.size, dtype=bool)
    if n:
        ref[np.argsort(-flat, kind="stable")[:n]] = True
    comp, iso = _components(a)
    dcomp, diso = _components(ref.reshape(a.shape))
    return {"cover": n, "components": comp, "isolated": iso, "dense_components": dcomp, "dense_isolated": diso}


__all__ = [
    "map_fragmentation",
    "top_quintile_mean",
    "summarise_per_feature",
    "variogram_curve",
    "halflife_from_curve",
    "variogram_halflife",
    "EARTH_RADIUS_KM",
    "distant_neighbours",
]

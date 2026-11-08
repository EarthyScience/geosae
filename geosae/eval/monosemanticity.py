r"""Pairwise-similarity monosemanticity scores of Pach et al. (2025) and Betti et al. (2026).

Both scores rate one feature by the similarity of the samples it fires on. With the
feature's activations min-max scaled to :math:`\tilde a_i \in [0, 1]` over the
:math:`N` samples and a pairwise similarity :math:`s_{ij}`,

.. math::

    \mathrm{MS} = \frac{\sum_{i<j} \tilde a_i \tilde a_j \, s_{ij}}{\sum_{i<j} \tilde a_i \tilde a_j} .

* :func:`monosemanticity_score` — the visual score: :math:`s_{ij}` is the cosine
  similarity of two image embeddings from a fixed image encoder.
* :func:`geo_monosemanticity_score` — the geographic score:
  :math:`s_{ij} = \exp(-d_{ij} / \ell)` with :math:`d_{ij}` the great-circle
  distance and :math:`\ell = 500` km in the released code of Betti et al.

The score is a weighted mean of similarities, not a correlation: a feature that
fires uniformly over the samples scores the mean pairwise similarity of the sample
(:func:`mean_pairwise_similarity`), not zero, so that value is the level to read a
score against. The functions take any ``(N, K)`` activation matrix — sparse codes,
patch means, or the coordinates of a dense basis — with one row per sample.
"""

from __future__ import annotations

import numpy as np

from geosae.eval.spatial import EARTH_RADIUS_KM, _lonlat_to_xyz


def minmax_scale(activations: np.ndarray, *, min_active: int = 0) -> np.ndarray:
    """Scale each feature to ``[0, 1]`` over the samples.

    Args:
        activations: ``(N, K)`` activations, one row per sample.
        min_active: Features non-zero on fewer samples than this become NaN.

    Returns:
        ``(N, K)`` float64 array; a constant or too rarely active feature is an all-NaN column.
    """
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"activations must be (N, K), got shape {a.shape}")
    lo, hi = a.min(axis=0), a.max(axis=0)
    span = hi - lo
    scorable = (span > 0) & ((a != 0).sum(axis=0) >= min_active)
    out = (a - lo) / np.where(scorable, span, 1.0)
    out[:, ~scorable] = np.nan
    return out


def pairwise_great_circle_km(
    lon: np.ndarray, lat: np.ndarray, lon_other: np.ndarray | None = None, lat_other: np.ndarray | None = None
) -> np.ndarray:
    """Great-circle distances between two sets of points, in km.

    Args:
        lon: ``(N,)`` longitudes, degrees.
        lat: ``(N,)`` latitudes, degrees.
        lon_other: ``(M,)`` longitudes of the second set; the first set when omitted.
        lat_other: ``(M,)`` latitudes of the second set.

    Returns:
        ``(N, M)`` distances.
    """
    xyz = _lonlat_to_xyz(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    other = (
        xyz
        if lon_other is None
        else _lonlat_to_xyz(np.asarray(lon_other, np.float64), np.asarray(lat_other, np.float64))
    )
    # The half-chord form is the haversine formula and keeps precision at small distances.
    half_chord = np.sqrt(np.clip((1.0 - xyz @ other.T) / 2.0, 0.0, 1.0))
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(half_chord)


def _cosine_pair_sums(weights: np.ndarray, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(embeddings, dtype=np.float64)
    unit = e / np.linalg.norm(e, axis=1, keepdims=True)
    # sum_{i<j} w_i w_j <u_i, u_j> = (|sum_i w_i u_i|^2 - sum_i w_i^2) / 2 for unit u.
    self_term = (weights**2).sum(axis=0)
    num = ((weights.T @ unit) ** 2).sum(axis=1) - self_term
    den = weights.sum(axis=0) ** 2 - self_term
    return num / 2.0, den / 2.0


def _geo_pair_sums(
    weights: np.ndarray, lon: np.ndarray, lat: np.ndarray, length_scale_km: float, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    num = np.zeros(weights.shape[1])
    n = weights.shape[0]
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        kernel = np.exp(-pairwise_great_circle_km(lon[rows], lat[rows], lon, lat) / length_scale_km)
        kernel[np.arange(rows.size), rows] = 0.0  # a sample is not paired with itself
        num += (weights[rows] * (kernel @ weights)).sum(axis=0)
    return num / 2.0, (weights.sum(axis=0) ** 2 - (weights**2).sum(axis=0)) / 2.0


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _check_rows(activations: np.ndarray, n_other: int, name: str) -> None:
    if np.ndim(activations) != 2 or np.shape(activations)[0] != n_other:
        raise ValueError(f"activations {np.shape(activations)} and {name} ({n_other} rows) do not align")


def monosemanticity_score(activations: np.ndarray, embeddings: np.ndarray, *, min_active: int = 20) -> np.ndarray:
    """Visual monosemanticity score of every feature (Pach et al. 2025).

    Args:
        activations: ``(N, K)`` feature activations, one row per image.
        embeddings: ``(N, E)`` embeddings of the same images from a fixed image encoder.
        min_active: Features non-zero on fewer images are not scored; 20 in the code
            of Betti et al. (2026), 0 in that of Pach et al.

    Returns:
        ``(K,)`` scores in ``[-1, 1]``, NaN for features that are not scored.
    """
    _check_rows(activations, np.shape(embeddings)[0], "embeddings")
    scaled = minmax_scale(activations, min_active=min_active)
    scored = np.isfinite(scaled[0]) if scaled.shape[0] else np.zeros(scaled.shape[1], bool)
    out = np.full(scaled.shape[1], np.nan)
    out[scored] = _ratio(*_cosine_pair_sums(scaled[:, scored], embeddings))
    return out


def geo_monosemanticity_score(
    activations: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    length_scale_km: float = 500.0,
    min_active: int = 20,
    chunk: int = 2048,
) -> np.ndarray:
    """Geographic monosemanticity score of every feature (Betti et al. 2026).

    Args:
        activations: ``(N, K)`` feature activations, one row per location.
        lon: ``(N,)`` longitudes, degrees.
        lat: ``(N,)`` latitudes, degrees.
        length_scale_km: Decay length of the similarity ``exp(-distance / length_scale_km)``.
        min_active: Features non-zero at fewer locations are not scored.
        chunk: Locations per block of the pairwise kernel; bounds memory at ``chunk * N`` floats.

    Returns:
        ``(K,)`` scores in ``(0, 1]``, NaN for features that are not scored.
    """
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError(f"lon {lon.shape} and lat {lat.shape} must be equal-length vectors")
    _check_rows(activations, lon.shape[0], "coordinates")
    scaled = minmax_scale(activations, min_active=min_active)
    scored = np.isfinite(scaled[0]) if scaled.shape[0] else np.zeros(scaled.shape[1], bool)
    out = np.full(scaled.shape[1], np.nan)
    out[scored] = _ratio(*_geo_pair_sums(scaled[:, scored], lon, lat, length_scale_km, chunk))
    return out


def mean_pairwise_similarity(
    *,
    embeddings: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    length_scale_km: float = 500.0,
    chunk: int = 2048,
) -> float:
    """Mean similarity over all sample pairs: the score of a feature that fires uniformly.

    Pass ``embeddings`` for the visual similarity, or ``lon`` and ``lat`` for the geographic one.

    Args:
        embeddings: ``(N, E)`` image embeddings.
        lon: ``(N,)`` longitudes, degrees.
        lat: ``(N,)`` latitudes, degrees.
        length_scale_km: Decay length of the geographic similarity.
        chunk: Locations per block of the pairwise kernel.

    Returns:
        The mean of ``s_ij`` over ``i < j``.
    """
    if embeddings is not None:
        num, den = _cosine_pair_sums(np.ones((np.shape(embeddings)[0], 1)), embeddings)
    elif lon is not None and lat is not None:
        lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        num, den = _geo_pair_sums(np.ones((lon.shape[0], 1)), lon, lat, length_scale_km, chunk)
    else:
        raise ValueError("pass embeddings, or lon and lat")
    return float(_ratio(num, den)[0])


def summarise_monosemanticity(scores: np.ndarray) -> dict[str, float]:
    """Mean and max of a per-feature score over the scored features.

    Args:
        scores: ``(K,)`` scores, NaN for features that were not scored.

    Returns:
        ``{mean, max, median, n_scored, n_features}``. The max is taken over the
        dictionary, so it grows with the number of features.
    """
    s = np.asarray(scores, dtype=np.float64)
    v = s[np.isfinite(s)]
    stats = {"mean": np.nan, "max": np.nan, "median": np.nan}
    if v.size:
        stats = {"mean": float(v.mean()), "max": float(v.max()), "median": float(np.median(v))}
    return {**stats, "n_scored": int(v.size), "n_features": int(s.size)}

"""Graded hold-outs: folds whose training side sits a controlled distance from the test region.

Whole-region folds put almost every held-out pixel out of domain, so no
applicability index can discriminate on them. A graded fold fixes the test
region, the ``n_test`` sites nearest a centre, and moves the training side away
from it: the training side is the ``n_train`` sites nearest the centre among those
at least ``gap`` beyond the test region's edge. Sweeping ``gap`` from 0 runs from
near to far out of domain at fixed sample sizes.

The distance is a choice. Great-circle distance on longitude/latitude gives
distance-graded folds; Euclidean distance in standardised climate coordinates
(:func:`climate_coordinates`) gives environment-graded ones, whose test region
is a climate, scattered over the globe.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

#: Mean Earth radius used for great-circle distances.
EARTH_RADIUS_KM = 6371.0088

#: Site-distance metrics :func:`site_distance` knows.
SITE_METRICS = ("haversine", "euclidean")

#: CHELSA bioclim bands (0-based) of :func:`climate_coordinates`, with a log flag:
#: mean annual temperature, temperature seasonality, annual precipitation,
#: precipitation seasonality.
CLIMATE_BANDS: tuple[tuple[int, bool], ...] = ((0, False), (3, False), (11, True), (14, False))


@dataclass(frozen=True)
class GradedFold:
    """One rung of a graded hold-out.

    Attributes:
        centre: Site index the test region is centred on.
        gap: Distance between the test region's edge and the training side, in
            the units of the metric (km for ``haversine``).
        train_mask: ``(N,)`` training-side membership.
        test_mask: ``(N,)`` test-side membership.
    """

    centre: int
    gap: float
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def name(self) -> str:
        """``"c<centre>_g<gap>"``, unique within one design."""
        return f"c{self.centre}_g{self.gap:g}"


def site_distance(coords: np.ndarray, point: np.ndarray, *, metric: str = "haversine") -> np.ndarray:
    """Distance of every site to one point.

    Args:
        coords: ``(N, 2)`` longitude/latitude in degrees for ``haversine``, or
            ``(N, C)`` coordinates for ``euclidean``.
        point: ``(2,)`` or ``(C,)`` coordinates of the point.
        metric: One of :data:`SITE_METRICS`.

    Returns:
        ``(N,)`` distances, in km for ``haversine``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    if metric == "euclidean":
        return np.linalg.norm(coords - point, axis=1)
    if metric != "haversine":
        raise ValueError(f"unknown metric {metric!r}, expected one of {SITE_METRICS}")
    lon, lat = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    lon0, lat0 = np.radians(point[0]), np.radians(point[1])
    h = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def spread_centres(
    coords: np.ndarray,
    n: int,
    *,
    metric: str = "haversine",
    min_separation: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n`` centre sites at random, each at least ``min_separation`` from the others.

    Sites are visited in a seeded random order and accepted greedily, so dense
    regions are proportionally more likely to host a centre.

    Args:
        coords: Site coordinates, as for :func:`site_distance`.
        n: Number of centres.
        metric: One of :data:`SITE_METRICS`.
        min_separation: Minimum distance between two centres.
        seed: Seed of the visiting order.

    Returns:
        ``(n,)`` site indices.

    Raises:
        ValueError: If fewer than ``n`` sites can be placed that far apart.
    """
    coords = np.asarray(coords, dtype=np.float64)
    order = np.random.default_rng(seed).permutation(coords.shape[0])
    chosen: list[int] = []
    for i in order:
        if not np.isfinite(coords[i]).all():
            continue
        if not chosen or site_distance(coords[chosen], coords[i], metric=metric).min() >= min_separation:
            chosen.append(int(i))
            if len(chosen) == n:
                return np.asarray(chosen)
    raise ValueError(f"only {len(chosen)} of {n} centres fit at separation {min_separation}")


def graded_folds(
    coords: np.ndarray,
    centres: Sequence[int] | np.ndarray,
    *,
    gaps: Sequence[float],
    n_test: int,
    n_train: int,
    metric: str = "haversine",
) -> Iterator[GradedFold]:
    """Yield one fold per (centre, gap): a fixed test region and a training side ``gap`` beyond it.

    The test side is the ``n_test`` sites nearest the centre; its edge is the
    largest of their distances. The training side is the ``n_train`` sites
    nearest the centre among those at least ``gap`` beyond that edge. A rung with
    fewer than ``n_train`` such sites is skipped. Sites at a non-finite distance
    (missing coordinates) enter neither side.

    Args:
        coords: Site coordinates, as for :func:`site_distance`.
        centres: Site indices of the test-region centres.
        gaps: Gap ladder, in the units of ``metric``.
        n_test: Test-side size.
        n_train: Training-side size, the same on every rung.
        metric: One of :data:`SITE_METRICS`.

    Yields:
        :class:`GradedFold`, centre by centre, in the order of ``gaps``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    for c in centres:
        d = site_distance(coords, coords[int(c)], metric=metric)
        order = np.argsort(d, kind="stable")
        order = order[np.isfinite(d[order])]
        test_mask = np.zeros(n, dtype=bool)
        test_mask[order[:n_test]] = True
        edge = float(d[order[n_test - 1]])
        rest = order[n_test:]
        for gap in gaps:
            eligible = rest[d[rest] >= edge + float(gap)]
            if eligible.size < n_train:
                continue
            train_mask = np.zeros(n, dtype=bool)
            train_mask[eligible[:n_train]] = True
            yield GradedFold(centre=int(c), gap=float(gap), train_mask=train_mask, test_mask=test_mask.copy())


def climate_coordinates(bioclim: np.ndarray) -> np.ndarray:
    """Standardised climate coordinates of each site from its CHELSA bioclim bands.

    Takes the bands in :data:`CLIMATE_BANDS` (precipitation as ``log1p``) and
    z-scores each over the sites given, so one unit is one standard deviation of
    that climate variable across the sample.

    Args:
        bioclim: ``(N, 19)`` CHELSA V2.1 bioclim values, bio1 to bio19 in order.

    Returns:
        ``(N, len(CLIMATE_BANDS))`` float64 coordinates; a site with a missing band
        is ``NaN`` throughout and never enters a :func:`graded_folds` fold.
    """
    bio = np.asarray(bioclim, dtype=np.float64)
    cols = [np.log1p(np.maximum(bio[:, i], 0.0)) if log else bio[:, i] for i, log in CLIMATE_BANDS]
    z = np.column_stack(cols)
    z[~np.isfinite(z).all(axis=1)] = np.nan
    sd = np.nanstd(z, axis=0)
    return (z - np.nanmean(z, axis=0)) / np.where(sd > 0, sd, 1.0)


def read_site_bioclim(
    zarr_path: str, cells: np.ndarray, *, key: str = "chelsa_bioclim", span: int = 50_000
) -> np.ndarray:
    """CHELSA bioclim at the centre pixel of each site's patch.

    CHELSA's 1 km grid is coarser than a 640 m patch, so the centre pixel stands
    for the whole patch.

    Args:
        zarr_path: Path of the patch store.
        cells: ``(N,)`` store cell ids.
        key: Store array holding the ``(cells, H, W, 19)`` bioclim stack.
        span: Cells per contiguous read.

    Returns:
        ``(N, 19)`` float64, ``NaN`` where the store flags the patch as missing.
    """
    import zarr

    store = zarr.open_group(zarr_path, mode="r")
    arr, mask = store[key], store[f"mask_{key}"]
    cy, cx = arr.shape[1] // 2, arr.shape[2] // 2
    cells = np.asarray(cells, dtype=np.int64)
    out = np.full((cells.size, arr.shape[3]), np.nan)
    order = np.argsort(cells)
    for s in range(0, cells.size, span):
        idx = order[s : s + span]
        lo, hi = int(cells[idx].min()), int(cells[idx].max()) + 1
        vals = np.asarray(arr[lo:hi, cy, cx, :], dtype=np.float64)[cells[idx] - lo]
        ok = np.asarray(mask[lo:hi], dtype=bool)[cells[idx] - lo]
        vals[~ok] = np.nan
        out[idx] = vals
    return out


__all__ = [
    "EARTH_RADIUS_KM",
    "SITE_METRICS",
    "CLIMATE_BANDS",
    "GradedFold",
    "site_distance",
    "spread_centres",
    "graded_folds",
    "climate_coordinates",
    "read_site_bioclim",
]

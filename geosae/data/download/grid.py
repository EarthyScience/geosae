# The MajorTOM grid arithmetic in this file (row latitudes, per-row column
# counts, column longitudes, cell centres and names) is adapted from eodata,
# (C) 2026 Sebastian Hoffmann, Markus Zehner, Max Planck Institute for
# Biogeochemistry, https://github.com/EarthyScience/eodata (eodata/grids/majortom.py),
# distributed under the BSD-3-Clause license:
#
# BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""The MajorTOM grid: cell geometry, cell ids, and choosing the cells of a store."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import ssl
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

#: Equatorial WGS 84 radius, which MajorTOM uses as the radius of a spherical Earth.
EARTH_RADIUS_KM = 6378.137

#: Rows are kept between these latitudes, in degrees.
MAX_LATITUDE = 85.0

_NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m/physical/{name}.zip?version=5.0.0"
_NATURAL_EARTH_SHA256 = {
    "ne_10m_land": "e547d749445eaa0964aba76738090ec88f5e63c4585122170f98c67a7ea922dc",
    "ne_10m_lakes": "0803a06f9c3cb4671d89b68c48b142aad9366ba40f665245e12a913fbc61722a",
}


def _circumference_km(lat_deg: Any) -> Any:
    return 2 * math.pi * (EARTH_RADIUS_KM * np.cos(np.radians(lat_deg)))


class MajorTomGrid:
    """The MajorTOM grid of near-equal-area cells, ``spacing_km`` on a side.

    Rows run south to north; each row is split into as many columns as fit its
    circumference, so the column count shrinks towards the poles. Cells are
    numbered row by row, west to east: that flat index is what a store's
    ``cell_indices`` point into (via :func:`land_cells`).

    MajorTOM treats the Earth as a sphere with the equatorial radius, so a 5 km
    cell is about 4.97 km tall in WGS 84.

    Args:
        spacing_km: Cell side length in kilometres.
    """

    def __init__(self, spacing_km: float = 5) -> None:
        self.spacing_km = spacing_km
        n_rows = int(math.ceil(_circumference_km(0.0) / 2 / spacing_km))
        lats = np.sort(np.mod(np.linspace(-90, 90, n_rows, endpoint=False), 180) - 90)
        #: Southern edge of each row, degrees north.
        self.row_lats: np.ndarray = lats[(lats >= -MAX_LATITUDE) & (lats <= MAX_LATITUDE)]
        #: Height of every row, degrees.
        self.cell_height: float = self.row_lats[1] - self.row_lats[0]
        #: Number of columns in each row.
        self.n_cols: np.ndarray = np.ceil(_circumference_km(self.row_lats) / spacing_km).astype(int)
        #: Row whose southern edge is the equator (the ``0U`` row).
        self.equator_row: int = int(np.searchsorted(self.row_lats, 0))
        self._offsets = np.concatenate([[0], np.cumsum(self.n_cols)[:-1]])

    def __len__(self) -> int:
        return int(self.n_cols.sum())

    @cached_property
    def _row_lons(self) -> list[np.ndarray]:
        """Western edge of every column, per row, ascending."""
        rows = []
        for n in self.n_cols:
            lons = np.mod(np.arange(n, dtype=float) * (360 / n) - 180, 360) - 180
            rows.append(np.sort(lons))
        return rows

    @cached_property
    def _zero_cols(self) -> np.ndarray:
        """Column of each row whose western edge is nearest the prime meridian (``0R``)."""
        return np.array([np.argmin(np.abs(lons)) for lons in self._row_lons])

    @cached_property
    def bounds(self) -> np.ndarray:
        """``(N, 4)`` cell bounds ``(west, south, east, north)`` in degrees, by flat index.

        The last cell of a row can reach past 180 degrees east, where the row's
        width does not divide evenly.
        """
        rows = np.repeat(np.arange(len(self.n_cols)), self.n_cols)
        west = np.concatenate(self._row_lons)
        south = self.row_lats[rows]
        out = np.stack([west, south, west + 360 / self.n_cols[rows], south + self.cell_height], axis=1)
        out.setflags(write=False)
        return out

    @cached_property
    def centers(self) -> np.ndarray:
        """``(N, 2)`` cell centres ``(lon, lat)`` in degrees, by flat index."""
        rows = np.repeat(np.arange(len(self.n_cols)), self.n_cols)
        west = self.bounds[:, 0]
        south = self.bounds[:, 1]
        out = np.stack([west + 180.0 / self.n_cols[rows], south + self.cell_height / 2], axis=1)
        out.setflags(write=False)
        return out

    def rowcol(self, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of the cells containing each ``(lon, lat)``.

        Args:
            lon: Longitudes, degrees east, in ``[-180, 180)``.
            lat: Latitudes, degrees north, within the grid's rows.

        Returns:
            ``(row, col)`` integer arrays shaped like ``lon``.

        Raises:
            ValueError: A latitude lies outside the grid's rows.
        """
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        top = self.row_lats[-1] + self.cell_height
        if np.any((lat < self.row_lats[0]) | (lat >= top)):
            raise ValueError(f"latitude outside the grid's rows [{self.row_lats[0]}, {top})")
        rows = np.searchsorted(self.row_lats, lat, side="right") - 1
        cols = np.empty_like(rows)
        for r in np.unique(rows):
            sel = rows == r
            # West of the row's first edge lies the row's last cell, which wraps the antimeridian.
            cols[sel] = (np.searchsorted(self._row_lons[r], lon[sel], side="right") - 1) % self.n_cols[r]
        return rows, cols

    def cell_id(self, row: int, col: int) -> str:
        """MajorTOM id of one cell, e.g. ``"645U_1077R"``.

        Rows count from the equator (``U`` north, ``D`` south), columns from the
        prime-meridian column *of that row* (``R`` east, ``L`` west): the id the
        MajorTOM release names the cell by.
        """
        eq, zero = self.equator_row, int(self._zero_cols[row])
        row_name = f"{eq - row}D" if row < eq else f"{row - eq}U"
        col_name = f"{zero - col}L" if col < zero else f"{col - zero}R"
        return f"{row_name}_{col_name}"

    def parse_cell_id(self, cell_id: str) -> tuple[int, int]:
        """Row and column of a MajorTOM id such as ``"645U_1077R"``; the inverse of :meth:`cell_id`.

        Raises:
            ValueError: ``cell_id`` is malformed or names no cell of this grid.
        """
        try:
            row_name, col_name = cell_id.strip().upper().split("_")
            n_row, row_dir, n_col, col_dir = int(row_name[:-1]), row_name[-1], int(col_name[:-1]), col_name[-1]
        except ValueError:
            raise ValueError(f"not a MajorTOM cell id: {cell_id!r} (expected e.g. '645U_1077R')") from None
        if row_dir not in "UD" or col_dir not in "LR":
            raise ValueError(f"not a MajorTOM cell id: {cell_id!r} (expected e.g. '645U_1077R')")
        row = self.equator_row + n_row if row_dir == "U" else self.equator_row - n_row
        if not 0 <= row < len(self.row_lats):
            raise ValueError(f"{cell_id!r} lies outside the grid's rows")
        zero = int(self._zero_cols[row])
        col = zero + n_col if col_dir == "R" else zero - n_col
        if not 0 <= col < self.n_cols[row]:
            raise ValueError(f"{cell_id!r} lies outside its row's {self.n_cols[row]} columns")
        return row, col

    def flat_index(self, row: Any, col: Any) -> np.ndarray:
        """Flat cell index of ``(row, col)``, the numbering of :attr:`centers`."""
        return self._offsets[np.asarray(row)] + np.asarray(col)

    def cells_overlapping(self, geometry: Any, min_fraction: float) -> np.ndarray:
        """Cells of which at least ``min_fraction`` of the area lies inside ``geometry``.

        Areas are planar in degrees. Needs ``shapely``.

        Args:
            geometry: Shapely (multi)polygon in lon/lat degrees.
            min_fraction: Minimum covered fraction of a cell's area, in ``[0, 1]``.

        Returns:
            Sorted flat cell indices, ``int64``.
        """
        import shapely

        if not 0.0 <= min_fraction <= 1.0:
            raise ValueError("min_fraction must be in [0, 1]")
        xmin, ymin, xmax, ymax = shapely.bounds(geometry)
        b = self.bounds
        near = np.flatnonzero((b[:, 0] <= xmax) & (b[:, 2] >= xmin) & (b[:, 1] <= ymax) & (b[:, 3] >= ymin))
        boxes = shapely.box(*b[near].T)
        tree = shapely.STRtree(boxes, node_capacity=20)
        parts = shapely.get_parts(geometry)
        hit = np.unique(tree.query(parts, predicate="intersects")[1])
        inside = np.unique(tree.query(parts, predicate="covers")[1])
        edge = np.setdiff1d(hit, inside)

        def fraction(idx: np.ndarray) -> np.ndarray:
            return shapely.area(shapely.intersection(boxes[idx], geometry, grid_size=0)) / shapely.area(boxes[idx])

        with ThreadPoolExecutor() as pool:
            fractions = np.concatenate(list(pool.map(fraction, np.array_split(edge, len(edge) // 1000 + 1))))
        keep = np.concatenate([inside, edge[fractions >= min_fraction]])
        return np.sort(near[keep]).astype(np.int64)


def land_cells(grid: MajorTomGrid, min_fraction: float = 0.1) -> np.ndarray:
    """Flat indices of the grid's land cells, the base set a store's cells are drawn from.

    A cell is land when at least ``min_fraction`` of it lies on Natural Earth
    10 m land, lakes removed, Antarctica (south of 60 S) excluded. The low
    threshold keeps coastal cells the coarse coastline would cut off. Downloads
    the Natural Earth files once into :func:`cache_dir`; needs ``geopandas``.
    """
    return grid.cells_overlapping(land_mask(), min_fraction)


def land_cell_centers(spacing_km: float = 5, min_fraction: float = 0.1) -> np.ndarray:
    """``(lon, lat)`` centres of the grid's land cells, in :func:`land_cells` order.

    The land test intersects every cell of the global grid with the coastline,
    which takes minutes; the result is cached in :func:`cache_dir` and read
    back on every later call.

    Args:
        spacing_km: Grid spacing.
        min_fraction: Minimum land fraction of a cell, as in :func:`land_cells`.

    Returns:
        ``(n_land, 2)`` float64 array; row ``i`` is the base-grid cell ``i`` a
        store's ``cell_indices`` refer to.
    """
    path = cache_dir() / f"majortom_{spacing_km:g}km_land{min_fraction:g}_centers.npy"
    if path.exists():
        return np.load(path)
    grid = MajorTomGrid(spacing_km)
    centers = np.ascontiguousarray(grid.centers[land_cells(grid, min_fraction)], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".partial{os.getpid()}.npy")
    np.save(tmp, centers)
    tmp.replace(path)
    return centers


@lru_cache(maxsize=2)
def _grid(spacing_km: float) -> MajorTomGrid:
    """One shared grid per spacing, so its cell arrays are computed once per process."""
    return MajorTomGrid(spacing_km)


def land_cell_indices(spacing_km: float = 5, min_fraction: float = 0.1) -> np.ndarray:
    """Flat grid indices of the land cells, in :func:`land_cell_centers` order.

    Reads the cached centres of :func:`land_cell_centers`, so only the first call
    in a fresh cache pays for the land test.
    """
    centers = land_cell_centers(spacing_km, min_fraction)
    grid = _grid(spacing_km)
    return grid.flat_index(*grid.rowcol(centers[:, 0], centers[:, 1])).astype(np.int64)


def cells_in_bbox(
    bbox: tuple[float, float, float, float], *, spacing_km: float = 5, land_fraction: float | None = 0.1
) -> np.ndarray:
    """Cells whose centre lies inside a lon/lat bounding box.

    The land test runs on the box alone, so a regional selection takes seconds,
    not the minutes of the global :func:`land_cells`. Needs ``shapely`` and
    ``geopandas`` unless ``land_fraction`` is ``None``.

    Args:
        bbox: ``(west, south, east, north)`` in degrees, ``west < east``.
        spacing_km: Grid spacing.
        land_fraction: Keep only cells at least this fraction on land, as in
            :func:`land_cells`; ``None`` keeps every cell, sea included.

    Returns:
        Sorted flat grid indices, ``int64``.

    Raises:
        ValueError: The box is empty or crosses the antimeridian.
    """
    west, south, east, north = (float(v) for v in bbox)
    if not (west < east and south < north):
        raise ValueError(f"bbox must be (west, south, east, north) with west < east and south < north, got {bbox}")
    grid = _grid(spacing_km)
    lon, lat = grid.centers[:, 0], grid.centers[:, 1]
    inside = np.flatnonzero((lon >= west) & (lon < east) & (lat >= south) & (lat < north)).astype(np.int64)
    if land_fraction is None or inside.size == 0:
        return inside
    import shapely

    land = shapely.intersection(land_mask(), shapely.box(west - 1, south - 1, east + 1, north + 1), grid_size=0)
    return np.intersect1d(inside, grid.cells_overlapping(land, land_fraction))


def cells_at(lon: Any, lat: Any, *, spacing_km: float = 5) -> np.ndarray:
    """Flat grid index of the cell containing each ``(lon, lat)`` point.

    Args:
        lon: Longitudes, degrees east, in ``[-180, 180)``.
        lat: Latitudes, degrees north.
        spacing_km: Grid spacing.

    Returns:
        ``int64`` array, one index per point; points in the same cell share it.
    """
    grid = _grid(spacing_km)
    return grid.flat_index(*grid.rowcol(lon, lat)).astype(np.int64)


def cells_from_ids(ids: list[str], *, spacing_km: float = 5) -> np.ndarray:
    """Flat grid indices of MajorTOM cell ids such as ``"645U_1077R"``.

    Raises:
        ValueError: An id is malformed or names no cell of the grid.
    """
    grid = _grid(spacing_km)
    rows, cols = zip(*(grid.parse_cell_id(i) for i in ids), strict=True) if len(ids) else ((), ())
    return grid.flat_index(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)).astype(np.int64)


def sample_cells(
    n: int,
    *,
    seed: int = 0,
    bbox: tuple[float, float, float, float] | None = None,
    spacing_km: float = 5,
    land_fraction: float = 0.1,
) -> np.ndarray:
    """A seeded random sample of land cells, globally or inside a bounding box.

    The draw is :func:`~geosae.data.download.selection.sample_subset_indices`
    over the candidate cells, so the same arguments always give the same cells
    in the same scattered order.

    Args:
        n: Number of cells; all candidates when there are fewer.
        seed: Seed of the draw.
        bbox: ``(west, south, east, north)`` to sample inside (see
            :func:`cells_in_bbox`); ``None`` samples every land cell of the
            globe, which builds the :func:`land_cell_centers` cache on first use.
        spacing_km: Grid spacing.
        land_fraction: Minimum land fraction of a candidate cell.

    Returns:
        Flat grid indices, ``int64``, in draw order.
    """
    from geosae.data.download.selection import sample_subset_indices

    if bbox is None:
        candidates = land_cell_indices(spacing_km, land_fraction)
    else:
        candidates = cells_in_bbox(bbox, spacing_km=spacing_km, land_fraction=land_fraction)
    if candidates.size == 0:
        raise ValueError(f"no land cells to sample from in {bbox}")
    return candidates[sample_subset_indices(candidates.size, n, seed)]


def select_cells(
    *,
    n: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    points: Any = None,
    ids: list[str] | None = None,
    seed: int = 0,
    spacing_km: float = 5,
    land_fraction: float | None = 0.1,
) -> np.ndarray:
    """Cells chosen one of four ways, the selection ``geosae download`` takes as ``download.cells``.

    * ``n``: a random sample of ``n`` land cells (:func:`sample_cells`), inside
      ``bbox`` if given;
    * ``bbox`` alone: every cell inside it (:func:`cells_in_bbox`);
    * ``points``: the cell containing each ``(lon, lat)`` (:func:`cells_at`);
    * ``ids``: MajorTOM cell ids (:func:`cells_from_ids`).

    Args:
        n: Sample size.
        bbox: ``(west, south, east, north)`` in degrees.
        points: ``(N, 2)`` ``(lon, lat)`` pairs.
        ids: MajorTOM cell ids such as ``"645U_1077R"``.
        seed: Seed of a random sample.
        spacing_km: Grid spacing.
        land_fraction: Minimum land fraction for ``n`` and ``bbox`` selections
            (``None`` keeps sea cells in a ``bbox``).

    Returns:
        Unique flat grid indices, ``int64``, in selection order.

    Raises:
        ValueError: Not exactly one of ``n``/``bbox``, ``points`` and ``ids`` is given.
    """
    given = [
        name
        for name, v in (("n/bbox", n if n is not None else bbox), ("points", points), ("ids", ids))
        if v is not None
    ]
    if len(given) != 1:
        raise ValueError(f"choose cells by exactly one of n/bbox, points or ids, got {given or 'none'}")
    if ids is not None:
        cells = cells_from_ids(list(ids), spacing_km=spacing_km)
    elif points is not None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cells = cells_at(pts[:, 0], pts[:, 1], spacing_km=spacing_km)
    elif n is not None:
        cells = sample_cells(int(n), seed=seed, bbox=bbox, spacing_km=spacing_km, land_fraction=land_fraction or 0.0)
    else:
        cells = cells_in_bbox(bbox, spacing_km=spacing_km, land_fraction=land_fraction)
    _, first = np.unique(cells, return_index=True)
    return cells[np.sort(first)]


def land_mask() -> Any:
    """Natural Earth 10 m land north of 60 S, minus lakes, as one shapely geometry."""
    import shapely

    land = shapely.intersection(_natural_earth("ne_10m_land"), shapely.box(-180, -60, 180, 90), grid_size=0)
    land = shapely.difference(land, _natural_earth("ne_10m_lakes"), grid_size=0)
    shapely.prepare(land)
    return land


def cache_dir() -> Path:
    """Download cache: ``$GEOSAE_CACHE_DIR``, else ``$XDG_CACHE_HOME/geosae`` (``~/.cache/geosae``)."""
    if os.getenv("GEOSAE_CACHE_DIR"):
        return Path(os.environ["GEOSAE_CACHE_DIR"]).expanduser()
    return (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "geosae").expanduser()


def _natural_earth(name: str) -> Any:
    """Union of all polygons of one Natural Earth 10 m physical layer."""
    import geopandas as gpd
    import shapely

    path = cache_dir() / f"{name}.zip"
    if not path.exists():
        _download(_NATURAL_EARTH_URL.format(name=name), path, _NATURAL_EARTH_SHA256[name])
    gdf = gpd.read_file(f"zip://{path}!{name}.shp").to_crs(epsg=4326)
    return shapely.union_all(gdf.geometry, grid_size=0)


def _download(url: str, path: Path, sha256: str) -> None:
    """Download ``url`` to ``path``, checking its SHA-256 before it lands."""
    import certifi

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "geosae"})
    context = ssl.create_default_context(cafile=certifi.where())
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as tmp:
        with urllib.request.urlopen(request, context=context) as response:
            shutil.copyfileobj(response, tmp)
    digest = hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()
    if digest != sha256:
        Path(tmp.name).unlink()
        raise RuntimeError(f"{url}: SHA-256 {digest}, expected {sha256}")
    Path(tmp.name).replace(path)


__all__ = [
    "EARTH_RADIUS_KM",
    "MAX_LATITUDE",
    "MajorTomGrid",
    "cache_dir",
    "cells_at",
    "cells_from_ids",
    "cells_in_bbox",
    "land_cell_indices",
    "sample_cells",
    "select_cells",
    "land_cell_centers",
    "land_cells",
    "land_mask",
]

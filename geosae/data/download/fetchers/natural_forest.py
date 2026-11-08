"""Natural forests of the world 2020 (Neumann et al. 2025): natural-forest probability at 10 m.

The product is published on Figshare+ as 60 zip archives, one per UTM zone. A zone
archive holds one GeoTIFF per tile, ``raster_<zone><band>_10_<i>_<j>.tif``: the tiles
of latitude band ``<band>`` sit on a 10 x 10 grid in the zone's UTM projection, row
``i`` counted from the band's northern edge and column ``j`` from its western one.
Tiles with no land are absent. Each tile's single ``uint8`` band is the probability of
natural forest scaled by 250, with 255 as nodata.

The archives are read in place: the zip directory and a tile's compressed bytes are
fetched by HTTP range requests, so a patch costs one tile, not one archive. The grid's
origin and step differ between bands and are not published; they are read from the
headers of two tiles of the band the first time the band is needed.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from odc.geo.geobox import GeoBox

from geosae.data.download.fetchers.raster_fetchers import TiledRasterFetcher
from geosae.data.download.fetchers.reproject import geobox_from_profile

#: Figshare+ article of the probability maps (CC-BY-4.0).
ARTICLE_API = "https://api.figshare.com/v2/articles/28881731"
_BANDS = "CDEFGHJKLMNPQRSTUVWX"
_MEMBER = re.compile(r"raster_(\d+)([C-X])_10_(\d+)_(\d+)\.tif$")
_SCALE, _NODATA = 250.0, 255


def utm_band(lon: float, lat: float) -> str:
    """UTM zone and latitude band of a point, e.g. ``"32U"`` (no Norway/Svalbard exceptions).

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees, within the UTM range -80 to 84.

    Returns:
        The zone number followed by the band letter.
    """
    zone = min(int(np.floor((lon + 180.0) / 6.0)) + 1, 60)
    band = min(max(int(np.floor((lat + 80.0) / 8.0)), 0), len(_BANDS) - 1)
    return f"{zone}{_BANDS[band]}"


class _RangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file through HTTP range requests.

    Figshare answers a download link with a redirect to a signed URL valid for a few
    seconds, so the redirect is resolved again whenever the last one is older than
    ``ttl`` seconds.
    """

    def __init__(self, url: str, size: int, ttl: float = 7.0):
        import requests

        self._url, self._size, self._pos, self._ttl = url, size, 0, ttl
        self._session = requests.Session()
        self._signed: tuple[str, float] | None = None

    def _location(self) -> str:
        if self._signed is None or time.monotonic() - self._signed[1] > self._ttl:
            r = self._session.get(self._url, allow_redirects=False, timeout=60)
            r.raise_for_status()
            self._signed = (r.headers.get("location", self._url), time.monotonic())
        return self._signed[0]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = {0: offset, 1: self._pos + offset, 2: self._size + offset}[whence]
        return self._pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self._size - self._pos)
        if n <= 0:
            return 0
        r = self._session.get(
            self._location(), headers={"Range": f"bytes={self._pos}-{self._pos + n - 1}"}, timeout=300
        )
        r.raise_for_status()
        data = r.content
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


def figshare_zones(article_api: str = ARTICLE_API) -> dict[int, Callable[[], BinaryIO]]:
    """Openers for the zone archives of a Figshare article, keyed by UTM zone.

    Args:
        article_api: The article's Figshare API URL.

    Returns:
        ``{zone: opener}``; each opener returns a buffered, seekable binary file.
    """
    import requests

    files = requests.get(article_api, timeout=60).json()["files"]
    out: dict[int, Callable[[], BinaryIO]] = {}
    for f in files:
        m = re.fullmatch(r"zone_(\d+)\.zip", f["name"])
        if m:
            url, size = f["download_url"], int(f["size"])
            out[int(m.group(1))] = lambda url=url, size=size: io.BufferedReader(_RangeFile(url, size), 1 << 16)
    return out


@dataclass(frozen=True)
class _BandGrid:
    """A latitude band's tile grid: tile ``(i, j)`` starts at ``(x0 + j dx, ytop - i dy)``."""

    crs: str
    x0: float
    ytop: float
    dx: float
    dy: float

    def tile_of(self, x: float, y: float) -> tuple[int, int]:
        return int(np.floor((self.ytop - y) / self.dy)), int(np.floor((x - self.x0) / self.dx))


class NaturalForestFetcher(TiledRasterFetcher):
    """Fetches natural-forest probability (0-1, NaN for nodata) from the zone archives.

    Args:
        zones: ``{utm_zone: opener}``, each opener returning a seekable binary file of
            that zone's zip archive. ``None`` lists the Figshare+ archives on first use.
        header_bytes: Uncompressed bytes read from a tile to parse its georeferencing.
        cache_tiles: Decoded tiles kept in memory, so the cells of one tile, which the
            store fetches one after the other, download it once.
    """

    def __init__(
        self,
        zones: dict[int, Callable[[], BinaryIO]] | None = None,
        header_bytes: int = 1 << 16,
        cache_tiles: int = 4,
    ):
        from collections import OrderedDict

        self._openers = zones
        self._header_bytes = header_bytes
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_tiles
        self._archives: dict[int, tuple[zipfile.ZipFile, dict[tuple[str, int, int], str]]] = {}
        self._grids: dict[str, _BandGrid | None] = {}

    # -- archive and grid -------------------------------------------------
    def _archive(self, zone: int) -> tuple[zipfile.ZipFile, dict[tuple[str, int, int], str]] | None:
        if self._openers is None:
            self._openers = figshare_zones()
        if zone not in self._archives:
            if zone not in self._openers:
                return None
            z = zipfile.ZipFile(self._openers[zone]())
            names = {}
            for info in z.infolist():
                m = _MEMBER.search(info.filename)
                if m and int(m.group(1)) == zone:
                    names[(m.group(2), int(m.group(3)), int(m.group(4)))] = info.filename
            self._archives[zone] = (z, names)
        return self._archives[zone]

    def _header(self, z: zipfile.ZipFile, name: str) -> tuple[float, float, str]:
        from rasterio.io import MemoryFile

        with z.open(name) as f, MemoryFile(f.read(self._header_bytes)) as mem, mem.open() as src:
            return src.bounds.left, src.bounds.top, str(src.crs)

    def _grid(self, key: str) -> _BandGrid | None:
        if key not in self._grids:
            zone, band = int(key[:-1]), key[-1]
            archive = self._archive(zone)
            ij = [] if archive is None else [(i, j) for (b, i, j) in archive[1] if b == band]
            self._grids[key] = None
            if len({i for i, _ in ij}) > 1 and len({j for _, j in ij}) > 1:
                ij = np.array(ij)
                a = tuple(ij[np.argmin(ij.sum(1))])
                b = tuple(ij[np.argmax(ij.sum(1))])
                if a[0] == b[0] or a[1] == b[1]:
                    a, b = tuple(ij[np.argmin(ij[:, 0] - ij[:, 1])]), tuple(ij[np.argmax(ij[:, 0] - ij[:, 1])])
                z, names = archive
                (la, ta, crs), (lb, tb, _) = (self._header(z, names[(band, *t)]) for t in (a, b))
                dx, dy = (lb - la) / (b[1] - a[1]), (ta - tb) / (b[0] - a[0])
                self._grids[key] = _BandGrid(crs, la - a[1] * dx, ta + a[0] * dy, dx, dy)
        return self._grids[key]

    # -- TiledRasterFetcher -----------------------------------------------
    def tile_key(self, lon: float, lat: float) -> tuple[str, int, int] | None:
        """The ``(zone+band, i, j)`` of the tile containing a point, or ``None`` if the archive has none."""
        from pyproj import Transformer

        key = utm_band(lon, lat)
        grid = self._grid(key)
        if grid is None:
            return None
        x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(lon, lat)
        i, j = grid.tile_of(x, y)
        archive = self._archive(int(key[:-1]))
        return (key, i, j) if (key[-1], i, j) in archive[1] else None

    def candidate_tiles(self, lon: float, lat: float, patch_extent_m: float = 640.0) -> list[tuple[str, int, int]]:
        """Tiles under the patch centre and its four corners, the centre's first."""
        half = patch_extent_m / 2.0 / 111_320.0
        dlon = half / max(np.cos(np.radians(lat)), 1e-6)
        points = [(lon, lat)] + [(lon + sx * dlon, lat + sy * half) for sx in (-1, 1) for sy in (-1, 1)]
        keys: list[tuple[str, int, int]] = []
        for p in points:
            k = self.tile_key(*p)
            if k is not None and k not in keys:
                keys.append(k)
        return keys

    def fetch_tile(self, tile_key: tuple[str, int, int]) -> tuple[np.ndarray, GeoBox] | None:
        """The tile's probability (``float32``, NaN for nodata) and its geobox."""
        from rasterio.io import MemoryFile

        if tile_key in self._cache:
            self._cache.move_to_end(tile_key)
            return self._cache[tile_key]
        key, i, j = tile_key
        archive = self._archive(int(key[:-1]))
        if archive is None or (key[-1], i, j) not in archive[1]:
            return None
        z, names = archive
        with MemoryFile(z.read(names[(key[-1], i, j)])) as mem, mem.open() as src:
            raw = src.read(1)
            geobox = geobox_from_profile(src.profile)
        prob = raw.astype(np.float32) / _SCALE
        prob[raw == _NODATA] = np.nan
        self._cache[tile_key] = (prob, geobox)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return prob, geobox

    def cleanup_tile(self, tile_key: tuple[str, int, int]) -> None:
        """Drop a tile from the in-memory cache."""
        self._cache.pop(tile_key, None)


__all__ = ["ARTICLE_API", "NaturalForestFetcher", "figshare_zones", "utm_band"]

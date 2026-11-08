"""Copernicus-FM precomputed global grid embeddings at 0.25 degrees.

Copernicus-FM publishes one 768-dimensional vector per node of the 721 x 1440
ERA5 grid, obtained by encoding every available Sentinel-1/2/3/5P and DEM
observation of a node and averaging. Roughly 293k of the 1.04M nodes -- land and
near-coastal ocean -- carry a real vector; the rest of the raster is zero-filled.

Unlike the per-pixel raster products, nothing needs to be reprojected: the map is
already on a regular lon/lat grid, so a fetch is a nearest-node lookup
(:class:`~geosae.data.download.gridded.RegularGrid`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geosae.data.download.gridded import RegularGrid

#: HuggingFace dataset holding the published embedding map.
COPERNICUSFM_REPO_ID = "wangyi111/Copernicus-Embed-025deg"
#: Dense ``(721, 1440, 768)`` float32 raster of the 310k-node release.
COPERNICUSFM_EMBEDDING_FILE = "embeddings_025_v1/embed_map_310k.npz"
#: Grid resolution of the published map, in degrees.
COPERNICUSFM_RESOLUTION_DEG = 0.25
#: Embedding width.
COPERNICUSFM_DIM = 768


class CopernicusFMGridEmbeddings:
    """Nearest-node lookup into the Copernicus-FM 0.25 degree embedding map.

    The raster is memory-resident once loaded (about 3.2 GB as float32), which
    is what makes scattered lookups cheap.

    Args:
        path: Local ``.npz`` holding the dense raster. Downloaded from
            HuggingFace on first use if missing.
        repo_id: HuggingFace dataset id.
        filename: Path of the raster inside the dataset repo.
        resolution_deg: Grid resolution of the raster.

    Example:
        >>> fetcher = CopernicusFMGridEmbeddings("data/copernicusfm")  # doctest: +SKIP
        >>> vectors, valid = fetcher.lookup([13.4, -60.0], [52.5, -3.0])  # doctest: +SKIP
    """

    def __init__(
        self,
        path,
        *,
        repo_id: str = COPERNICUSFM_REPO_ID,
        filename: str = COPERNICUSFM_EMBEDDING_FILE,
        resolution_deg: float = COPERNICUSFM_RESOLUTION_DEG,
    ) -> None:
        self.path = Path(path)
        self.repo_id = repo_id
        self.filename = filename
        self.grid = RegularGrid.global_grid(resolution_deg)
        self._raster: np.ndarray | None = None

    @property
    def local_file(self) -> Path:
        """Path of the raster on disk (inside ``path`` if that is a directory)."""
        return self.path / self.filename if self.path.is_dir() or not self.path.suffix else self.path

    def download(self) -> Path:
        """Fetch the raster from HuggingFace unless it is already on disk.

        Returns:
            Path of the local raster file.

        Raises:
            ImportError: If ``huggingface_hub`` is missing (``geosae[download]``).
        """
        target = self.local_file
        if target.exists():
            return target
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "downloading the Copernicus-FM embedding map needs huggingface_hub. "
                "Install it with: pip install 'geosae[download]'"
            ) from exc

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(self.path),
            allow_patterns=[self.filename],
        )
        return target

    @property
    def raster(self) -> np.ndarray:
        """The dense ``(n_lat, n_lon, dim)`` float32 map, loaded on first access."""
        if self._raster is None:
            with np.load(self.download()) as handle:
                self._raster = handle[next(iter(handle.files))]
            expected = (*self.grid.shape, COPERNICUSFM_DIM)
            if self._raster.shape != expected:
                raise ValueError(f"expected a {expected} embedding map, got {self._raster.shape}")
        return self._raster

    def lookup(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Embeddings of the nodes nearest the given points.

        Args:
            lon: Longitudes in degrees east.
            lat: Latitudes in degrees north.

        Returns:
            ``(vectors, valid)``: a ``(n, dim)`` float32 array, and a ``(n,)``
            bool array that is False where the node carries no embedding (an
            all-zero vector in the published raster).
        """
        row, col = self.grid.nearest(lon, lat)
        vectors = self.raster[row, col]
        valid = np.any(vectors != 0, axis=-1)
        return vectors, valid


def add_copernicusfm_embeddings(store_path, *, cache_dir, array: str = "copernicusfm", **fetcher_kwargs) -> int:
    """Fill a coarse store's Copernicus-FM embedding array by node lookup.

    The store must have ``patch_size = 1`` and its ``lon``/``lat`` must be grid
    nodes -- i.e. it must come from
    :func:`~geosae.data.download.gridded.coarsen_patch_store` on the same grid.
    Cells with no published embedding keep their fill and are masked out, so a
    later :class:`~geosae.data.dataset.PatchData` read skips them.

    Args:
        store_path: Store to write into (opened read/write).
        cache_dir: Directory holding (or receiving) the downloaded raster.
        array: Store array to fill.
        **fetcher_kwargs: Forwarded to :class:`CopernicusFMGridEmbeddings`.

    Returns:
        Number of cells that received an embedding.
    """
    import zarr

    group = zarr.open_group(str(store_path), mode="r+")
    if int(group.attrs["patch_size"]) != 1:
        raise ValueError(f"expected a patch_size=1 store, got {group.attrs['patch_size']}")

    fetcher = CopernicusFMGridEmbeddings(cache_dir, **fetcher_kwargs)
    lon, lat = np.asarray(group["lon"][:]), np.asarray(group["lat"][:])
    vectors, valid = fetcher.lookup(lon, lat)

    out = group[array]
    out[:] = np.where(valid[:, None], vectors, np.nan).astype(out.dtype).reshape(lon.size, 1, 1, -1)
    group[f"mask_{array}"][:] = valid
    group[f"feature_{array}"][:] = np.arange(vectors.shape[-1])
    return int(valid.sum())


__all__ = [
    "COPERNICUSFM_DIM",
    "COPERNICUSFM_EMBEDDING_FILE",
    "COPERNICUSFM_REPO_ID",
    "COPERNICUSFM_RESOLUTION_DEG",
    "CopernicusFMGridEmbeddings",
    "add_copernicusfm_embeddings",
]

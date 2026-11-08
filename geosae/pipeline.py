"""Config-to-object helpers shared by the CLI stages and by analysis scripts.

Every ``geosae`` stage turns a config into the same handful of objects — a dataset, a model, a run's
trained model and transforms, the labelled evaluation sample, the pixels a conventional decomposition
is fitted on. These functions are that translation, so a script reproducing a stage's numbers builds
exactly the objects the stage would.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from geosae.config import load_config
from geosae.data.dataset import PatchData, Transforms, build_transforms, resolve_split_name
from geosae.models import SparseAutoencoder, build_model, load_checkpoint

if TYPE_CHECKING:
    from geosae.eval.registry import RegistrySample

#: Model types whose weight initialisation is seeded from the run's top-level ``seed``.
SEEDED_MODEL_TYPES = ("batchtopk", "topk", "matryoshka", "relu", "relu_l1")


def make_data(cfg: dict, split: str, *, compute: bool = False, max_cells: int | None = None) -> PatchData:
    """The :class:`~geosae.data.dataset.PatchData` a config's ``data`` block describes, for one split.

    ``data.quality_cache_root`` and ``data.max_abs_value`` are forwarded only when the config sets
    them (``null`` included), so :class:`~geosae.data.dataset.PatchData`'s own defaults apply
    otherwise.

    Args:
        cfg: A run config with a ``data`` block (``zarr_path``, ``embedding_name``, ``split_name``,
            ``transforms``, loader and quality-filter options).
        split: ``train`` / ``val`` / ``test``.
        compute: Load the split into RAM.
        max_cells: Cap on the number of cells, for smoke runs.

    Returns:
        The dataset, with the config's embedding transforms attached.
    """
    d = cfg["data"]
    optional = {k: d[k] for k in ("quality_cache_root", "max_abs_value") if k in d}
    return PatchData(
        zarr_path=d["zarr_path"],
        embedding_name=d.get("embedding_name", "alphaearth"),
        split_name=d.get("split_name"),
        split=split,
        compute=compute,
        transforms=build_transforms(d.get("transforms"), zarr_path=d["zarr_path"]),
        n_load_workers=int(d.get("n_load_workers", 16)),
        inram_dtype=d.get("inram_dtype", "float16"),
        max_cells=max_cells,
        min_spatial_std=d.get("min_spatial_std", 1e-3),
        drop_flat_pixel_cells=bool(d.get("drop_flat_pixel_cells", True)),
        **optional,
    )


def store_data(cfg: dict, store: str, split: str) -> PatchData:
    """:func:`make_data` for ``split``, re-pointed at another store (lazy, not loaded into RAM)."""
    return make_data({**cfg, "data": {**cfg["data"], "zarr_path": store}}, split, compute=False)


def build_model_from_cfg(cfg: dict) -> SparseAutoencoder:
    """A freshly initialised model from a run config's ``model`` block.

    ``model.input_dim`` defaults to 64. Unless ``model.seed`` is set, the weight initialisation of the
    types in :data:`SEEDED_MODEL_TYPES` is seeded from the run's top-level ``seed``, so multi-seed
    runs differ deterministically.

    Args:
        cfg: A run config.

    Returns:
        The model.
    """
    m = dict(cfg["model"])
    m.setdefault("input_dim", 64)
    if str(m.get("type", "batchtopk")).lower() in SEEDED_MODEL_TYPES and "seed" not in m:
        m["seed"] = cfg.get("seed")
    return build_model(m)


def _run_config(run_dir: Path, fallback: dict | None) -> dict:
    """A run directory's own ``config.yaml``, or ``fallback`` when it has none."""
    path = Path(run_dir) / "config.yaml"
    if path.exists():
        return load_config([str(path)], [])
    if fallback is None:
        raise FileNotFoundError(f"{path} does not exist and no fallback config was given")
    return fallback


def load_run_model(run_dir: str | Path, cfg: dict | None = None, *, checkpoint: str = "best.pth") -> SparseAutoencoder:
    """The trained model of one run directory, built from the run's own ``config.yaml``.

    Args:
        run_dir: A run directory holding ``config.yaml`` and the checkpoint.
        cfg: Config to build the model from when the run directory has no ``config.yaml``.
        checkpoint: Checkpoint file name inside ``run_dir``.

    Returns:
        The model with the checkpoint's weights loaded, on CPU.
    """
    run_dir = Path(run_dir)
    model = build_model_from_cfg(_run_config(run_dir, cfg))
    load_checkpoint(model, str(run_dir / checkpoint))
    return model


def load_run_transforms(run_dir: str | Path, cfg: dict | None = None) -> Transforms:
    """The embedding transform pipeline one run directory was trained with.

    Runs scored on the same store can disagree in their ``data.transforms``, so each run's own
    config is read rather than the caller's.

    Args:
        run_dir: A run directory.
        cfg: Config to read when the run directory has no ``config.yaml``.

    Returns:
        The run's :class:`~geosae.data.dataset.Transforms`.
    """
    data = _run_config(Path(run_dir), cfg).get("data") or {}
    return build_transforms(data.get("transforms"), zarr_path=data.get("zarr_path"))


def apply_transforms(transforms: Transforms, raw: np.ndarray) -> np.ndarray:
    """``raw`` ``(n, D)`` through an embedding transform pipeline, as float32."""
    import torch

    return transforms.apply(torch.as_tensor(raw, dtype=torch.float32)).numpy()


def expand_run_dirs(patterns: str | list[str]) -> list[Path]:
    """Run directories named by paths or glob patterns, deduplicated and sorted by name.

    A match counts as a run directory when it holds ``config.yaml`` or ``best.pth``.
    """
    out: list[Path] = []
    for pattern in [patterns] if isinstance(patterns, str) else list(patterns):
        hits = sorted(Path().glob(str(pattern))) if any(c in str(pattern) for c in "*?[") else [Path(pattern)]
        out.extend(h for h in hits if (h / "config.yaml").exists() or (h / "best.pth").exists())
    return sorted(dict.fromkeys(out), key=lambda q: q.name)


def run_provenance(run_dir: str | Path) -> dict:
    """A run's ladder coordinates — architecture, width, sparsity, frozen flag, seed and store.

    Returns:
        ``arch``, ``latent_dim``, ``k``, ``frozen``, ``seed`` and ``store``; empty when the run
        directory has no ``config.yaml``.
    """
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        return {}
    c = load_config([str(path)], [])
    m = c.get("model") or {}
    return {
        "arch": str(m.get("type", "")),
        "latent_dim": int(m.get("latent_dim", 0)),
        "k": m.get("k"),
        "frozen": bool(m.get("freeze_decoder", False)),
        "seed": c.get("seed"),
        "store": str((c.get("data") or {}).get("zarr_path", "")),
    }


class EvalSample(NamedTuple):
    """The labelled evaluation sample a real-store stage scores on.

    Attributes:
        sample: The raw, untransformed pixels with their reference-product labels.
        key: Identifies the store and the draw.
        concepts_available: Whether the store carries at least one scorable catalog concept.
        n_cells: Cells actually read.
        pixels_per_cell: Pixels read per cell.
    """

    sample: RegistrySample
    key: str
    concepts_available: bool
    n_cells: int
    pixels_per_cell: int


def real_eval_sample(cfg: dict, block: dict) -> EvalSample:
    """Draw the labelled evaluation sample of a real store, raw and untransformed.

    The sample is read once through :func:`~geosae.eval.registry.sample_registry_pixels`, so a
    caller scoring several decompositions re-wraps it per decomposition with
    :func:`~geosae.eval.ladder.real_ladder_sample_from_registry` and that decomposition's own
    ``data.transforms``, without re-reading the store.

    Args:
        cfg: The stage's config, read for ``data.zarr_path``, ``data.embedding_name`` and
            ``data.split_name``.
        block: The stage's options: ``split`` (default ``test``), ``n_cells`` (4000),
            ``pixels_per_cell`` (256), ``sample_seed`` (0), ``products``, ``min_spatial_std``,
            ``drop_flat_pixel_cells``, ``label_store`` and ``cells`` (store rows to read in
            place of a random draw of ``n_cells``).

    Returns:
        The sample and its description.
    """
    from geosae.eval.registry import sample_registry_pixels

    zarr_path = str(cfg["data"]["zarr_path"])
    split = block.get("split", "test")
    n_cells = int(block.get("n_cells", 4000))
    pixels_per_cell = int(block.get("pixels_per_cell", 256))
    sample_seed = int(block.get("sample_seed", 0))
    t0 = time.perf_counter()
    reg = sample_registry_pixels(
        zarr_path,
        split=split,
        split_name=resolve_split_name(zarr_path, cfg["data"].get("split_name")),
        embedding_name=cfg["data"].get("embedding_name", "alphaearth"),
        n_patches=n_cells,
        pixels_per_patch=pixels_per_cell,
        products=block.get("products"),
        seed=sample_seed,
        min_spatial_std=block.get("min_spatial_std", 1e-3),
        drop_flat_pixel_cells=bool(block.get("drop_flat_pixel_cells", True)),
        label_store=block.get("label_store"),
        cells=block.get("cells"),
    )
    n_read = int(np.unique(reg.cell).size)
    per_cell = reg.n_pixels // max(n_read, 1)
    available = len(reg.concept_names) > 0
    print(
        f"[geosae] evaluation sample read in {time.perf_counter() - t0:.0f}s "
        f"({n_read} cells x {per_cell}px, concepts {'available' if available else 'unavailable'})",
        flush=True,
    )
    key = f"{zarr_path}@{split}:{n_cells}x{pixels_per_cell}s{sample_seed}"
    return EvalSample(reg, key, available, n_read, per_cell)


def baseline_fit_pixels_from_config(cfg: dict, block: dict, store: str, *, seed: int) -> np.ndarray:
    """The training pixels a stage fits its conventional decompositions on.

    ``n_fit_pixels`` (default 200,000), ``fit_sampling`` (``per_patch``), ``fit_table_path`` and
    ``fit_progress_every`` are read from the stage's own block, falling back to the top-level
    ``baselines`` block; see :func:`geosae.eval.baselines.baseline_fit_pixels`.

    Args:
        cfg: The run config, read for the dataset, its transforms and the fallbacks.
        block: The stage's own config block.
        store: Zarr store to draw the training split from.
        seed: Seeds the draw.

    Returns:
        ``(n, D)`` float64 pixels, in the units of the config's ``data.transforms``.
    """
    from geosae.eval.baselines import baseline_fit_pixels

    shared = cfg.get("baselines") or {}

    def get(key, default):
        return block[key] if key in block else shared.get(key, default)

    sampling = str(get("fit_sampling", "per_patch"))
    data = store_data(cfg, store, "train")
    x = baseline_fit_pixels(
        data,
        n_pixels=int(get("n_fit_pixels", 200_000)),
        seed=seed,
        sampling=sampling,
        table_path=get("fit_table_path", None),
        progress_every=int(get("fit_progress_every", 200)),
    )
    sites = min(len(data), len(x)) if sampling == "per_patch" else int(np.ceil(len(x) / data.patch_size**2))
    print(
        f"[geosae]   baseline fit: {len(x)} px from {sites} of {len(data)} train patches "
        f"({sampling} draw, seed {seed})",
        flush=True,
    )
    return x


__all__ = [
    "SEEDED_MODEL_TYPES",
    "EvalSample",
    "apply_transforms",
    "baseline_fit_pixels_from_config",
    "build_model_from_cfg",
    "expand_run_dirs",
    "load_run_model",
    "load_run_transforms",
    "make_data",
    "real_eval_sample",
    "run_provenance",
    "store_data",
]

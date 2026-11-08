"""A fixed sample of labelled pixels, so any dictionary is scored on the same data.

:func:`sample_registry_pixels` draws a pixel sample from a patch store once
and caches it as a single ``.npz``. Every factor matrix — an SAE's codes at any
width, or a dense basis (Raw / PCA / ICA / NMF) — is then scored by
:func:`~geosae.eval.coverage.concept_auroc_matrix` on the **identical** pixels
with the identical concept columns, which is what makes a width comparison a
comparison of dictionaries rather than of samples.

The sample is drawn cell-block-wise: ``n_patches`` cells at random from the
split, and from every cell the *same* ``pixels_per_patch`` within-patch positions,
drawn once. Every cell therefore contributes the same number of pixels in the
same order — the invariant
:func:`~geosae.eval.coverage.block_permuted_best_auroc` needs to permute labels
cell-by-cell. Unlabelled pixels are **kept**: dropping them would break that
invariant, and the balanced subsampling in
:func:`~geosae.eval.coverage.balanced_binary_subsample` already excludes nodata
from both sides of every concept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from geosae.data.dataset import PatchData


def build_class_table(
    products: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str | None]]:
    """Enumerate the concept registry as matrix columns.

    Args:
        products: Product subset; defaults to every product
            :func:`geosae.eval.concepts.build_targets` covers.

    Returns:
        ``(class_table, product_order, concept_names, concept_groups)`` —
        ``class_table`` is ``(C, 2)`` int ``[product_id, class_code]`` with
        ``product_id`` indexing ``product_order``; ``concept_groups`` holds each
        concept's group, or ``None`` for an ungrouped class.
    """
    from geosae.eval.concepts import build_targets

    targets = build_targets(products)
    product_order: list[str] = []
    product_id: dict[str, int] = {}
    rows, names, groups = [], [], []
    for t in targets:
        prod = t["product"]
        if prod not in product_id:
            product_id[prod] = len(product_order)
            product_order.append(prod)
        rows.append((product_id[prod], int(t["code"])))
        names.append(str(t["name"]))
        groups.append(t["group"])
    class_table = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    return class_table, product_order, names, groups


@dataclass(frozen=True)
class LabelStoreSpec:
    """Where a store's concept labels actually live, when not in the store itself.

    A pixel-wise store (AlphaEarth, Tessera, Copernicus-FM) carries the concept
    catalog's reference products at its own resolution, and no indirection is
    needed. A patch-*token* subset store (OlmoEarth/MMEarth token grids) carries
    none: its cells are crops of a parent patch store, one embedding "pixel" (a
    token) covering several of the parent's label pixels, and its own ``cell_indices``
    array names the parent row each cell was cropped from. Configuring that
    indirection here — rather than special-casing either store by name — lets
    :func:`sample_registry_pixels` read the parent's catalog products and
    aggregate them onto the embedding's own grid, one token at a time, by
    majority vote (:func:`~geosae.data.download.gridded.aggregate_categorical`),
    masking a token to nodata where no class holds an absolute majority of its
    footprint's valid pixels.

    Attributes:
        zarr_path: The store the catalog products actually live in (the
            parent), read only for its label arrays.
        cell_index_array: Array in the *embedding* store mapping its own row to
            a row of ``zarr_path``. Defaults to ``"cell_indices"``.
        token_footprint: Label pixels per token side — the parent's native
            patch is ``token_footprint`` times wider and taller than the
            embedding's own grid (an 8x8 OlmoEarth/MMEarth grid over the
            parent's 64x64 px patch has ``token_footprint=8``).
    """

    zarr_path: str
    cell_index_array: str = "cell_indices"
    token_footprint: int = 1


def _as_label_store_spec(label_store: LabelStoreSpec | dict | None) -> LabelStoreSpec | None:
    if label_store is None or isinstance(label_store, LabelStoreSpec):
        return label_store
    return LabelStoreSpec(**dict(label_store))


@dataclass(frozen=True)
class RegistrySample:
    """A labelled pixel sample shared by every dictionary being compared.

    Attributes:
        raw: ``(N, D)`` float32 embedding pixels.
        labels: ``product name -> (N,) int`` label array, ``0`` = nodata.
        class_table: ``(C, 2)`` int ``[product_id, class_code]``.
        product_order: Product name for each ``product_id``.
        concept_names: ``(C,)`` human names.
        concept_groups: ``(C,)`` concept group, ``None`` where ungrouped.
        cell: ``(N,)`` store cell id of each pixel; every cell contributes the
            same number of pixels, in the same within-patch order.
        n_positive: ``(C,)`` positive-pixel count per concept.
    """

    raw: np.ndarray
    labels: dict[str, np.ndarray]
    class_table: np.ndarray
    product_order: list[str]
    concept_names: list[str]
    concept_groups: list[str | None]
    cell: np.ndarray
    n_positive: np.ndarray

    @classmethod
    def from_labels(
        cls,
        *,
        raw: np.ndarray,
        labels: dict[str, np.ndarray],
        class_table: np.ndarray,
        product_order: list[str],
        concept_names: list[str],
        concept_groups: list[str | None],
        cell: np.ndarray,
    ) -> RegistrySample:
        """Build a sample, counting the positive pixels of every concept."""
        class_table = np.asarray(class_table, dtype=np.int64).reshape(-1, 2)
        n_pos = np.zeros(class_table.shape[0], dtype=np.int64)
        for c in range(class_table.shape[0]):
            arr = labels.get(product_order[int(class_table[c, 0])])
            if arr is not None:
                n_pos[c] = int((np.asarray(arr) == int(class_table[c, 1])).sum())
        return cls(
            raw=np.asarray(raw, dtype=np.float32),
            labels={k: np.asarray(v) for k, v in labels.items()},
            class_table=class_table,
            product_order=list(product_order),
            concept_names=list(concept_names),
            concept_groups=list(concept_groups),
            cell=np.asarray(cell, dtype=np.int64),
            n_positive=n_pos,
        )

    @property
    def n_pixels(self) -> int:
        """Number of sampled pixels ``N``."""
        return int(self.raw.shape[0])

    def valid_concepts(self, min_positive: int = 50) -> np.ndarray:
        """``(C,)`` bool: concepts with enough positives to be scored.

        Mirrors the ``min_positive`` rule of
        :func:`~geosae.eval.coverage.balanced_binary_subsample`, so a concept
        marked invalid here is exactly one that function refuses to sample.
        """
        return self.n_positive >= int(min_positive)

    def concept_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense per-pixel ``(concept_mask, product_valid)`` for every pixel.

        Every existing concept metric samples one balanced subset per concept
        column; :meth:`~geosae.eval.statistic.PanelAccumulator.update` instead
        wants both label arrays whole, over every pixel of the sample at once,
        which is what this builds — one boolean membership test per concept
        column and one nodata test per product, both vectorised over ``N``.

        Returns:
            ``concept_mask`` ``(N, C)`` bool (pixel belongs to concept ``c``'s
            positive class) and ``product_valid`` ``(N, P)`` bool (the pixel
            has a real, non-nodata label from that concept's product).
        """
        n = self.n_pixels
        product_valid = np.zeros((n, len(self.product_order)), dtype=bool)
        for pid, product in enumerate(self.product_order):
            arr = self.labels.get(product)
            if arr is not None:
                product_valid[:, pid] = np.asarray(arr) != 0  # nodata == 0, see `labels`

        concept_mask = np.zeros((n, self.class_table.shape[0]), dtype=bool)
        for c in range(self.class_table.shape[0]):
            pid, cls = int(self.class_table[c, 0]), int(self.class_table[c, 1])
            arr = self.labels.get(self.product_order[pid])
            if arr is not None:
                concept_mask[:, c] = np.asarray(arr) == cls
        return concept_mask, product_valid

    def save(self, path: str | Path) -> str:
        """Write the whole sample to one ``.npz`` and return its path.

        Labels are stacked into a single ``(P, N)`` int16 array alongside the
        product order — one file rather than one per product. Uncompressed: at
        ~1e6 pixels, compression saves little space and costs far more time to
        write and to read back, and the sample is a cache every run reads.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stack = np.zeros((len(self.product_order), self.n_pixels), dtype=np.int16)
        has_labels = np.zeros(len(self.product_order), dtype=bool)
        for i, prod in enumerate(self.product_order):
            arr = self.labels.get(prod)
            if arr is not None:
                stack[i] = np.asarray(arr, dtype=np.int16)
                has_labels[i] = True
        np.savez(
            path,
            raw=self.raw,
            label_stack=stack,
            has_labels=has_labels,
            class_table=self.class_table,
            product_order=np.array(self.product_order, dtype=object),
            concept_names=np.array(self.concept_names, dtype=object),
            concept_groups=np.array(["" if g is None else g for g in self.concept_groups], dtype=object),
            cell=self.cell,
            n_positive=self.n_positive,
        )
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> RegistrySample:
        """Read a sample written by :meth:`save`."""
        with np.load(path, allow_pickle=True) as z:
            product_order = [str(p) for p in z["product_order"]]
            stack, has_labels = z["label_stack"], z["has_labels"]
            labels = {p: stack[i].astype(np.int64) for i, p in enumerate(product_order) if bool(has_labels[i])}
            return cls(
                raw=z["raw"],
                labels=labels,
                class_table=z["class_table"],
                product_order=product_order,
                concept_names=[str(n) for n in z["concept_names"]],
                concept_groups=[None if str(g) == "" else str(g) for g in z["concept_groups"]],
                cell=z["cell"],
                n_positive=z["n_positive"],
            )


def sample_registry_pixels(
    data_zarr: str,
    *,
    split: str = "test",
    split_name: str | None = None,
    embedding_name: str = "alphaearth",
    n_patches: int = 400,
    pixels_per_patch: int = 256,
    products: Sequence[str] | None = None,
    seed: int = 0,
    max_cells: int | None = None,
    min_spatial_std: float | None = 1e-3,
    drop_flat_pixel_cells: bool = True,
    label_store: LabelStoreSpec | dict | None = None,
    cells: Sequence[int] | None = None,
) -> RegistrySample:
    """Draw the shared labelled pixel sample from a patch store.

    ``n_patches`` cells are drawn at random from the split (or ``cells`` are taken), and the same
    ``pixels_per_patch`` within-patch positions — drawn once — are read from every
    cell. Unlabelled pixels are kept, so every cell contributes the same pixels in
    the same order (see the module docstring).

    Args:
        data_zarr: Path to the patch store.
        split: ``train`` / ``val`` / ``test``.
        split_name: Split-label array in the store; ``None`` takes its only ``split_*`` array.
        embedding_name: Embedding array in the store.
        n_patches: Cells to draw (capped at the split size).
        pixels_per_patch: Pixels per cell (capped at the patch area).
        products: Product subset, forwarded to :func:`build_class_table`.
        seed: RNG seed for the cell and pixel-position draws.
        max_cells: Optional cap on the split, for smoke runs.
        min_spatial_std: :class:`~geosae.data.dataset.PatchData`'s degenerate-patch
            filter. Its default assumes a patch with real spatial extent; pass
            ``None`` for a single-pixel-patch store (e.g. ``patch_size=1``),
            where every patch's within-patch std is trivially zero and the
            default would drop the whole split.
        drop_flat_pixel_cells: :class:`~geosae.data.dataset.PatchData`'s
            flat-pixel filter.
        label_store: For a store with no catalog products of its own (a
            patch-token subset store) — where
            its parent's products actually live, and how to aggregate them
            onto this store's grid; see :class:`LabelStoreSpec`. ``None`` (the
            default) reads labels from ``data_zarr`` itself.
        cells: Store rows to read instead of a random draw of ``n_patches``; rows outside
            the split or dropped by its quality filters are skipped.

    Returns:
        The sample, with ``raw`` in the same units the model is trained on (the
        store's transforms are not applied — the embeddings are already unit-norm).
    """
    from geosae.data.dataset import PatchData

    data = PatchData(
        data_zarr,
        embedding_name=embedding_name,
        split_name=split_name,
        split=split,
        compute=False,
        max_cells=max_cells,
        min_spatial_std=min_spatial_std,
        drop_flat_pixel_cells=drop_flat_pixel_cells,
    )
    local, rows, cols = _draw_sample_positions(
        data, n_patches=n_patches, pixels_per_patch=pixels_per_patch, seed=seed, cells=cells
    )
    return _read_pixel_sample(data, local, rows, cols, products=products, label_store=_as_label_store_spec(label_store))


def _draw_sample_positions(
    data: PatchData, *, n_patches: int, pixels_per_patch: int, seed: int, cells: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the cells and the shared within-patch pixel positions of a sample.

    Args:
        data: A :class:`~geosae.data.dataset.PatchData` over the split to draw from.
        n_patches: Cells to draw (capped at the split size).
        pixels_per_patch: Pixels per cell (capped at the patch area).
        seed: RNG seed for both draws.
        cells: Store rows to take instead of drawing ``n_patches`` cells.

    Returns:
        ``(local, rows, cols)`` — sorted indices into ``data.cells``, and the
        ``(n_pix,)`` within-patch row/column positions read from *every* cell.
    """
    rng = np.random.default_rng(seed)
    if cells is None:
        local = np.sort(rng.choice(len(data), size=int(min(n_patches, len(data))), replace=False))
    else:
        local = np.flatnonzero(np.isin(data.cells, np.asarray(cells, dtype=np.int64)))
    side = int(data.patch_size)
    n_pix = int(min(pixels_per_patch, side * side))
    flat_pos = np.sort(rng.choice(side * side, size=n_pix, replace=False))
    rows, cols = np.unravel_index(flat_pos, (side, side))
    return local.astype(np.int64), rows.astype(np.int64), cols.astype(np.int64)


def _majority_vote_tokens(patch: np.ndarray, footprint: int, fill_value: int) -> np.ndarray:
    """Aggregate a categorical ``(H, W)`` label patch onto its ``(H/f, W/f)`` token grid.

    Every ``footprint x footprint`` block of ``patch`` becomes one token,
    labelled by the majority class of its **valid** (non-``fill_value``)
    pixels — reusing :func:`~geosae.data.download.gridded.aggregate_categorical`,
    the same majority vote :func:`~geosae.data.download.presets.build_copernicusfm025_store`
    already coarsens a whole patch store with. A token with no absolute
    majority (the winning class holds at most half its valid pixels) is masked
    to ``fill_value`` rather than reporting an arbitrary plurality pick.

    Args:
        patch: ``(H, W)`` integer class codes, ``fill_value`` = nodata.
        footprint: Label pixels per token side. ``H`` and ``W`` are trimmed
            down to a multiple of it if they are not already one.
        fill_value: Nodata code, excluded from the vote (see :class:`LabelStoreSpec`).

    Returns:
        ``(H // footprint, W // footprint)`` array, ``patch``'s dtype.
    """
    from geosae.data.download.gridded import aggregate_categorical

    h_tok, w_tok = patch.shape[0] // footprint, patch.shape[1] // footprint
    trimmed = patch[: h_tok * footprint, : w_tok * footprint]
    blocks = trimmed.reshape(h_tok, footprint, w_tok, footprint).transpose(0, 2, 1, 3).reshape(h_tok * w_tok, -1)
    group = np.arange(h_tok * w_tok, dtype=np.int64)
    mode, frac, n_valid = aggregate_categorical(blocks, group, h_tok * w_tok, fill_value)
    has_majority = (n_valid > 0) & (frac > 0.5)
    tokens = np.where(has_majority, mode, fill_value)
    return tokens.reshape(h_tok, w_tok).astype(patch.dtype)


def _read_pixel_sample(
    data: PatchData,
    local: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    products: Sequence[str] | None = None,
    label_store: LabelStoreSpec | dict | None = None,
) -> RegistrySample:
    """Read embeddings and concept labels at explicit cell and pixel positions.

    The reading half of :func:`sample_registry_pixels`, split out so a frozen
    :class:`~geosae.eval.pool.EvalPool` can re-materialise its own recorded draw
    through exactly the same code path.

    Args:
        data: A :class:`~geosae.data.dataset.PatchData` over the split.
        local: ``(n_cells,)`` indices into ``data.cells``.
        rows: ``(n_pix,)`` within-patch row positions, read from every cell.
        cols: ``(n_pix,)`` within-patch column positions, read from every cell.
        products: Product subset, forwarded to :func:`build_class_table`.
        label_store: See :class:`LabelStoreSpec` and :func:`sample_registry_pixels`.

    Returns:
        The sample, pixels ordered cell-major and, within a cell, by the order of
        ``rows`` / ``cols``.
    """
    from geosae.data.dataset import open_store
    from geosae.eval.concepts import _product_source

    local = np.asarray(local, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    n_take, n_pix = int(local.size), int(rows.size)

    raw = np.empty((n_take * n_pix, data.embedding_dim), dtype=np.float32)
    for start in range(0, n_take, 32):
        block = local[start : start + 32]
        x = data._make_batch(block)["x"].numpy()  # (b, H, W, D)
        raw[start * n_pix : (start + len(block)) * n_pix] = x[:, rows, cols, :].reshape(-1, data.embedding_dim)

    class_table, product_order, names, groups = build_class_table(products)
    own_cells = data.cells[local]  # this store's own MajorTOM cell ids, always used for `RegistrySample.cell`
    label_store = _as_label_store_spec(label_store)
    if label_store is None:
        label_group = open_store(data.zarr_path)
        cell_ids = own_cells
        footprint = 1
    else:
        # The label products live in the *parent* store; this store's own
        # `cell_index_array` maps its rows onto the parent's, and the parent's
        # native patch is `footprint` times wider/taller than this store's grid.
        label_group = open_store(label_store.zarr_path)
        index_arr = np.asarray(open_store(data.zarr_path)[label_store.cell_index_array])
        cell_ids = index_arr[own_cells]
        footprint = int(label_store.token_footprint)
    sources = {prod: _product_source(prod) for prod in product_order}
    keys = sorted({key for key, _ in sources.values() if key in label_group})
    labels = {prod: np.zeros(n_take * n_pix, dtype=np.int64) for prod in product_order if sources[prod][0] in keys}
    # Label arrays are chunked over thousands of cells, so reading one cell
    # decompresses a whole chunk. Group the sample by chunk and read each once.
    for key in keys:
        source = label_group[key]
        chunk = int(source.chunks[0])
        by_chunk: dict[int, list[int]] = {}
        for i, cell_id in enumerate(cell_ids):
            by_chunk.setdefault(int(cell_id) // chunk, []).append(i)
        bands = [(labels[prod], band) for prod, (k, band) in sources.items() if k == key and prod in labels]
        for chunk_id, positions in by_chunk.items():
            lo = chunk_id * chunk
            block = np.asarray(source[lo : min(lo + chunk, source.shape[0])])
            for i in positions:
                patch = block[int(cell_ids[i]) - lo]
                sl = slice(i * n_pix, (i + 1) * n_pix)
                for out, band in bands:
                    full = patch[..., band] if band is not None else patch
                    if footprint != 1:
                        full = _majority_vote_tokens(full, footprint, fill_value=0)
                    out[sl] = full[rows, cols]

    return RegistrySample.from_labels(
        raw=raw,
        labels=labels,
        class_table=class_table,
        product_order=product_order,
        concept_names=names,
        concept_groups=groups,
        cell=np.repeat(own_cells.astype(np.int64), n_pix),
    )


def calibrate_sparsity(
    model: torch.nn.Module,
    raw: np.ndarray,
    *,
    device: str = "cuda",
    batch_size: int = 65_536,
    n_batches: int = 8,
    max_sample: int = 4_000_000,
    seed: int = 0,
) -> torch.nn.Module:
    """Seed an untrained BatchTopK model's eval-time sparsity threshold.

    A BatchTopK dictionary sparsifies at eval against ``min_activation``, the
    ``1 - k/M`` quantile of its activations, which training maintains as an EMA.
    On a model that was never trained it is still ``-1``, and *every* latent
    clears it — the "sparse" codes come out dense, silently. Setting it here
    from the model's own activations is what makes an untrained random
    dictionary comparable to a trained one at the same ``k``.

    The calibration pixels are drawn at **random** from ``raw``. Pixels arrive
    grouped by cell and cells differ in embedding magnitude, so a contiguous
    slice estimates the quantile off a handful of places on Earth — which at
    large ``M``, where the quantile sits far into the tail, is enough to put the
    threshold high enough to silence most of the dictionary.

    Args:
        model: A BatchTopK-family SAE.
        raw: ``(N, D)`` pixel embeddings to calibrate on.
        device: Torch device.
        batch_size: Pixels per calibration batch.
        n_batches: Calibration batches; ``n_batches * batch_size`` pixels are
            drawn in total (all of ``raw``, if it holds fewer).
        max_sample: Activations the quantile is taken over, spread evenly over
            the calibration batches. Bounds memory: a batch's full activation
            block is ``batch_size * M`` floats, gigabytes at the widest ``M``.
        seed: RNG seed for the calibration pixel draw.

    Returns:
        ``model``, with its threshold set, left in eval mode.
    """
    import torch

    from geosae.eval.metrics import set_batchtopk_threshold

    if not bool(getattr(model, "batch_topk", False)):
        return model.to(device).eval()  # per-sample top-k needs no threshold
    model = model.to(device).eval()
    x_all = torch.as_tensor(np.asarray(raw, dtype=np.float32))
    n_calib = min(int(x_all.shape[0]), int(n_batches) * int(batch_size))
    idx = np.sort(np.random.default_rng(seed).choice(x_all.shape[0], size=n_calib, replace=False))
    starts = range(0, n_calib, int(batch_size))
    per_batch = max(1, int(max_sample) // len(starts))
    gen = torch.Generator().manual_seed(int(seed))

    acts = []
    with torch.no_grad():
        for start in starts:
            x = x_all[torch.as_tensor(idx[start : start + int(batch_size)])].to(device)
            flat = model.preacts_for_threshold(x).float().reshape(-1).cpu()
            if flat.numel() > per_batch:  # keep the pooled quantile sample bounded
                flat = flat[torch.randint(0, flat.numel(), (per_batch,), generator=gen)]
            acts.append(flat)
    set_batchtopk_threshold(model, torch.cat(acts), max_sample=int(max_sample))
    return model.eval()


def encode_sample(
    model: torch.nn.Module,
    raw: np.ndarray,
    *,
    device: str = "cuda",
    batch_size: int = 65_536,
    sparse: bool = True,
) -> np.ndarray:
    """Run an SAE encoder over a pixel sample.

    The two regimes answer different questions and are not interchangeable.
    ``sparse=True`` returns the ``k``-sparse codes ``geosae predict`` writes —
    what a user of the dictionary actually gets. ``sparse=False`` returns the
    **dense pre-top-k** latents, which is the only regime in which an SAE feature
    is commensurate with a raw / PCA / ICA / NMF dimension: a rank statistic over
    a column that is exactly zero on 99% of pixels is not measuring the same
    thing as one over a dense column.

    Args:
        model: Any SAE from :mod:`geosae.models`.
        raw: ``(N, D)`` pixel embeddings.
        device: Torch device for the forward pass.
        batch_size: Pixels per forward pass.
        sparse: Post-sparsification codes (eval mode, so a BatchTopK dictionary
            thresholds against its calibrated ``min_activation`` rather than
            against the batch) when True; dense pre-top-k latents when False.

    Returns:
        ``(N, M)`` float32 activations.
    """
    import torch

    model = model.to(device).eval()
    if sparse and bool(getattr(model, "batch_topk", False)) and float(getattr(model, "min_activation", 0.0)) < 0:
        raise ValueError(
            "this BatchTopK model has no calibrated sparsity threshold (min_activation < 0), so its "
            "'sparse' codes would come out dense — call calibrate_sparsity() first, or pass sparse=False"
        )
    x_all = torch.as_tensor(np.asarray(raw, dtype=np.float32))
    out = np.empty((x_all.shape[0], int(model.latent_dim)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x_all.shape[0], int(batch_size)):
            x = x_all[start : start + int(batch_size)].to(device)
            z = model(x)[1] if sparse else model.encode(x)
            out[start : start + x.shape[0]] = z.float().cpu().numpy()
    return out


__all__ = [
    "calibrate_sparsity",
    "LabelStoreSpec",
    "RegistrySample",
    "build_class_table",
    "sample_registry_pixels",
    "encode_sample",
]

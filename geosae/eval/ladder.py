"""Scoring a ladder of decompositions on one labelled evaluation sample.

A *ladder* is a set of decompositions of the same embedding — trained SAEs at several widths,
sparsities and seeds, their frozen-decoder baselines, and conventional dictionaries such as PCA, ICA
and NMF. :func:`score_decomposition` returns the flat panel of :mod:`geosae.eval.blocks` for one of
them, presented as ``(encode, decoder)``, on a :class:`LadderSample` that is read from the store once
and shared by every rung, so all rungs rest on the identical pixel population.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from geosae.eval.blocks import PixelSource, concept_block, fidelity_block, fill_panel, geometry_block

if TYPE_CHECKING:
    from geosae.data.dataset import Transforms
    from geosae.eval.dictionaries import DictionaryFit
    from geosae.eval.registry import LabelStoreSpec, RegistrySample

Encoder = Callable[[np.ndarray], np.ndarray]


@dataclass
class LadderSample:
    """One labelled evaluation sample, held in RAM for a whole ladder.

    Attributes:
        pixels: Materialised ``(raw, concepts)`` pixel source.
        vocab: The sample's concept vocabulary (:func:`real_concept_vocabulary`).
        n_cells: Cells the pixels were drawn from.
        pixels_per_cell: Pixels contributed by one cell.
        key: Identifies these pixels — the store and the draw over it.
    """

    pixels: PixelSource
    vocab: dict
    n_cells: int
    pixels_per_cell: int
    key: str = ""


def real_concept_vocabulary(sample: RegistrySample) -> dict:
    """The concept vocabulary of a labelled pixel sample, as :func:`~geosae.eval.blocks.concept_block` reads it.

    Args:
        sample: A :class:`~geosae.eval.registry.RegistrySample`, from
            :func:`~geosae.eval.registry.sample_registry_pixels`.

    Returns:
        ``concept_names``, ``product_order``, ``class_table`` ``(C, 2)``,
        ``product_of_concept`` ``(C,)`` and ``class_to_group``. A class with no
        catalog group (``None``) is its own singleton group.
    """
    class_table = np.asarray(sample.class_table, dtype=np.int64).reshape(-1, 2)
    class_to_group = {}
    for row, group in zip(class_table, sample.concept_groups, strict=True):
        product = sample.product_order[int(row[0])]
        class_to_group[(product, int(row[1]))] = group if group is not None else f"{product}:{int(row[1])}"
    return {
        "concept_names": list(sample.concept_names),
        "product_order": list(sample.product_order),
        "class_table": class_table,
        "product_of_concept": class_table[:, 0].copy(),
        "class_to_group": class_to_group,
    }


def real_pixel_source(
    sample: RegistrySample, *, chunk_pixels: int = 200_000, transforms: Transforms | None = None
) -> PixelSource:
    """A materialised :data:`~geosae.eval.blocks.PixelSource` over a labelled sample.

    ``sample`` is already held in RAM (that is what
    :func:`~geosae.eval.registry.sample_registry_pixels` returns), so this only
    slices it into chunks.

    Args:
        sample: A labelled pixel sample.
        chunk_pixels: Pixels per yielded chunk.
        transforms: The embedding transform pipeline the *decoder* being scored
            was trained against (see :class:`~geosae.data.dataset.Transforms`),
            applied to every chunk before it is yielded. ``sample.raw`` is
            always the untransformed store reading, shared across every row of a
            panel, and rows on the same store can carry different transforms, so
            the transform is applied here, per source. ``None`` (the default)
            applies none.

    Returns:
        A zero-argument callable yielding ``(raw, concepts)`` chunks — see
        :data:`~geosae.eval.blocks.PixelSource`. ``product_valid`` is the
        sample's per-pixel nodata mask
        (:meth:`~geosae.eval.registry.RegistrySample.concept_arrays`'s
        ``arr != 0`` convention). ``concepts`` is ``None`` for a
        sample with no scorable concepts —
        :func:`~geosae.eval.blocks.fill_panel`'s "no concept vocabulary" case —
        rather than ``(n, 0)`` arrays, which
        :class:`~geosae.eval.statistic.PanelAccumulator.update`'s
        always-at-least-one-product ``product_valid`` shape cannot pair with.
    """
    raw = np.asarray(sample.raw, dtype=np.float64)
    if transforms is not None:
        import torch

        raw = transforms.apply(torch.as_tensor(raw, dtype=torch.float32)).double().numpy()
    concept_mask, product_valid = sample.concept_arrays()
    has_concepts = concept_mask.shape[1] > 0

    def source():
        n = raw.shape[0]
        for start in range(0, n, chunk_pixels):
            end = min(start + chunk_pixels, n)
            if not has_concepts:
                yield raw[start:end], None
            else:
                yield raw[start:end], (concept_mask[start:end], product_valid[start:end])

    return source


def materialise(source: PixelSource) -> PixelSource:
    """Read a pixel source into RAM once and serve every later pass from there.

    A ladder scores many decompositions against **one** sample, so its pixels and
    their concept membership are read once and reused rather than once per
    decomposition per pass.

    Args:
        source: The source to consume.

    Returns:
        A source over the same chunks, backed by memory.
    """
    chunks = list(source())

    def cached() -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
        yield from chunks

    return cached


def real_ladder_sample_from_registry(
    sample: RegistrySample, *, key: str = "", transforms: Transforms | None = None
) -> LadderSample:
    """Wrap an already-drawn :class:`~geosae.eval.registry.RegistrySample` as a :class:`LadderSample`.

    The conversion half of :func:`build_real_ladder_sample`, split out so a
    sample materialised once — by an :class:`~geosae.eval.pool.EvalPool`'s
    content-addressed cache, or shared across several rows of a panel — is
    turned into panel input without re-reading the store.

    Args:
        sample: A :class:`~geosae.eval.registry.RegistrySample`, drawn by
            :func:`~geosae.eval.registry.sample_registry_pixels` or an
            :class:`~geosae.eval.pool.EvalPool`.
        key: Identifies these pixels; see :attr:`LadderSample.key`.
        transforms: The embedding transform pipeline the decoder being scored
            against this sample was trained with (see
            :class:`~geosae.data.dataset.Transforms`), applied to the shared
            ``sample.raw`` fresh for this wrapping. ``sample`` itself is always
            read raw and untransformed and may be reused by several rows with
            *different* transforms (e.g. several architectures trained on the
            same z-scored Copernicus-FM store), so the transform is applied
            here, per call, rather than once on ``sample``. ``None`` (the
            default) applies none.

    Returns:
        The sample as a :class:`LadderSample`, with ``vocab["concept_names"]``
        empty when the sample carries no concept.
    """
    n_cells = int(np.unique(sample.cell).size)
    n_pix_per_cell = sample.n_pixels // max(n_cells, 1)
    return LadderSample(
        pixels=real_pixel_source(sample, transforms=transforms),
        vocab=real_concept_vocabulary(sample),
        n_cells=n_cells,
        pixels_per_cell=int(n_pix_per_cell),
        key=key,
    )


def build_real_ladder_sample(
    zarr_path: str,
    *,
    split: str = "test",
    split_name: str | None = None,
    embedding_name: str = "alphaearth",
    n_cells: int = 4000,
    pixels_per_cell: int = 256,
    products: Sequence[str] | None = None,
    seed: int = 0,
    key: str = "",
    min_spatial_std: float | None = 1e-3,
    drop_flat_pixel_cells: bool = True,
    transforms: Transforms | None = None,
    label_store: LabelStoreSpec | dict | None = None,
) -> LadderSample:
    """Read a real store's evaluation sample once, for the whole panel.

    The pixels and the concept vocabulary come from the concept catalog, through
    :func:`~geosae.eval.registry.sample_registry_pixels` — the same draw ``geosae
    coverage`` and ``geosae pool`` take. To reuse an already-materialised sample
    (an :class:`~geosae.eval.pool.EvalPool`'s cache, or one shared across several
    panel rows) call :func:`real_ladder_sample_from_registry` instead.

    The default draw is uniform over the test split: ``n_cells = 4000`` cells and
    ``pixels_per_cell = 256`` pixels per cell at seed ``0``. A store with fewer
    pixels per cell than that (a single-pixel-patch store) yields all of them.
    Block 3 is scorable if and only if the store carries at least one catalog
    product; otherwise the vocabulary is empty and the caller reports block 3
    unavailable rather than letting it reduce to ``nan``.

    Args:
        zarr_path: Path to the patch store.
        split: ``train`` / ``val`` / ``test``.
        split_name: Split-label array in the store; ``None`` takes its only ``split_*`` array.
        embedding_name: Embedding array in the store.
        n_cells: Cells to draw, capped at the split size.
        pixels_per_cell: Pixels per cell, capped at the patch area.
        products: Concept-catalog product subset; ``None`` scores every product
            :func:`~geosae.eval.concepts.build_targets` covers that exists in
            this store.
        seed: Seeds both the cell and the within-patch pixel draw.
        key: Identifies these pixels; see :attr:`LadderSample.key`.
        min_spatial_std: :class:`~geosae.data.dataset.PatchData`'s degenerate-patch
            filter. Pass ``None`` for a single-pixel-patch store
            (``patch_size=1``), where every patch's within-patch std is
            trivially zero and the default drops the whole split.
        drop_flat_pixel_cells: :class:`~geosae.data.dataset.PatchData`'s
            flat-pixel filter.
        transforms: The embedding transform pipeline the decoder being scored
            was trained with (see :class:`~geosae.data.dataset.Transforms`),
            applied to the sample before it is returned. ``None`` (the
            default) applies none.
        label_store: For a store with no catalog products of its own (a
            patch-token subset store) — where its parent's products live and
            how to aggregate them onto this store's grid; see
            :class:`~geosae.eval.registry.LabelStoreSpec`.
            ``None`` (the default) reads labels from ``zarr_path`` itself.

    Returns:
        The materialised sample, with ``vocab["concept_names"]`` empty when the
        store carries no scorable concept.
    """
    from geosae.eval.registry import sample_registry_pixels

    reg = sample_registry_pixels(
        zarr_path,
        split=split,
        split_name=split_name,
        embedding_name=embedding_name,
        n_patches=n_cells,
        pixels_per_patch=pixels_per_cell,
        min_spatial_std=min_spatial_std,
        drop_flat_pixel_cells=drop_flat_pixel_cells,
        products=products,
        seed=seed,
        label_store=label_store,
    )
    return real_ladder_sample_from_registry(reg, key=key, transforms=transforms)


def score_decomposition(
    sample: LadderSample,
    encode: Encoder,
    decoder: np.ndarray,
    *,
    offset: np.ndarray | None = None,
    codes_sample_pixels: int = 20_000,
    n_bins: int = 64,
    min_positive: int = 50,
    seed: int = 0,
    blocks: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    """The interpretability panel of one decomposition on one evaluation sample.

    Every block comes from one streaming pass over the sample's pixels
    (:func:`~geosae.eval.blocks.fill_panel`). A trained SAE and a fitted
    conventional dictionary differ only in ``encode`` and ``decoder``.

    Args:
        sample: The evaluation sample, from :func:`build_real_ladder_sample`
            or :func:`real_ladder_sample_from_registry`.
        encode: ``(n, D) -> (n, M)`` dense activations.
        decoder: ``(M, D)`` dictionary the reconstruction is formed with.
        offset: ``(D,)`` additive reconstruction term (a pre-bias, or a dense
            basis's mean).
        codes_sample_pixels: Dense code rows retained for ``ucov_star``'s null.
        n_bins: Activation-histogram bins.
        min_positive: Concepts with fewer positive pixels are not scorable.
        seed: Seeds the retained code subsample and the permutation null.
        blocks: Which blocks of :data:`~geosae.eval.blocks.BLOCK_SCALARS` to compute.

    Returns:
        The flat panel, plus ``n_pixels`` and ``n_cells`` recording the sample it rests on.
    """
    vocab = sample.vocab
    want = {int(b) for b in blocks}
    acc, codes_sample = fill_panel(
        sample.pixels,
        encode,
        decoder,
        concept_names=vocab["concept_names"],
        product_names=vocab["product_order"],
        product_of_concept=vocab["product_of_concept"],
        offset=offset,
        n_bins=n_bins,
        codes_sample_pixels=codes_sample_pixels if 3 in want else 0,
        seed=seed,
    )
    out: dict[str, float] = {}
    if 1 in want:
        out.update(fidelity_block(acc))
    if 2 in want:
        out.update(geometry_block(decoder, acc))
    if 3 in want and vocab["concept_names"]:
        out.update(
            concept_block(
                acc,
                class_table=vocab["class_table"],
                product_order=vocab["product_order"],
                class_to_group=vocab["class_to_group"],
                codes_sample=codes_sample,
                min_positive=min_positive,
                seed=seed,
            )
        )
    out["n_pixels"] = float(acc.n_pixels)
    out["n_cells"] = float(sample.n_cells)
    return out


def model_offset(model: Any) -> np.ndarray:
    """``(D,)`` additive term of a trained SAE's reconstruction.

    The panel forms :math:`\\hat e = a W_{\\text{dec}} + b`, so ``b`` has to be
    the model's own — a ``pre_bias`` model's ``b_dec``, a plain one's decoder
    bias, zero when it has neither. Getting this wrong shows up as an FVE
    silently below the run's own training curve.

    Args:
        model: Trained SAE.

    Returns:
        ``(D,)`` float64.
    """
    if getattr(model, "pre_bias", False):
        return model.b_dec.detach().cpu().numpy().astype(np.float64)
    bias = getattr(model.decoder, "bias", None)
    if bias is None:
        return np.zeros(model.decoder.weight.shape[0], dtype=np.float64)
    return bias.detach().cpu().numpy().astype(np.float64)


def model_encoder(model: Any, *, device: str = "cuda", batch_pixels: int = 200_000) -> Encoder:
    """A trained SAE as the ``encode`` callable :func:`score_decomposition` takes.

    Args:
        model: Trained SAE, already loaded.
        device: Device to run the forward passes on.
        batch_pixels: Pixels per forward pass.

    Returns:
        ``(n, D) -> (n, M)`` dense activations, float64.
    """
    import torch

    model.to(device).eval()

    def encode(raw: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate(
                [
                    model(torch.as_tensor(raw[i : i + batch_pixels], dtype=torch.float32, device=device))[1]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    for i in range(0, len(raw), batch_pixels)
                ]
            )

    return encode


def dictionary_encoder(fit: DictionaryFit, *, activation_rate: float | None = None) -> Encoder:
    """A fitted conventional dictionary as the ``encode`` callable.

    A dense basis has no active set, so unless it brings its own sparsity its
    codes are thresholded per atom to fire on ``activation_rate`` of the pixels.

    Args:
        fit: A :class:`~geosae.eval.dictionaries.DictionaryFit`.
        activation_rate: Target per-feature firing rate. Ignored for a method
            that is sparse by construction.

    Returns:
        ``(n, D) -> (n, M)`` dense activations, float64.
    """
    from geosae.eval.dictionaries import rate_matched_support

    def encode(raw: np.ndarray) -> np.ndarray:
        dense = fit.encode(raw)
        if fit.sparse or activation_rate is None:
            return np.asarray(dense, dtype=np.float64)
        return np.asarray(rate_matched_support(dense, min(float(activation_rate), 1.0)), dtype=np.float64)

    return encode


def leading_atoms(fit: DictionaryFit, x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` atoms whose codes vary most over ``x``.

    For PCA these are the first ``k`` principal components, for the raw basis the
    ``k`` dimensions of largest variance.

    Args:
        fit: A fitted dictionary.
        x: ``(n, D)`` pixels to rank the atoms on, e.g. the fitting pixels.
        k: Number of atoms to keep.

    Returns:
        ``(k,)`` atom indices, by decreasing code variance.
    """
    var = np.asarray(fit.encode(x), dtype=np.float64).var(0)
    return np.argsort(-var, kind="stable")[: int(k)]


def rank_encoder(fit: DictionaryFit, keep: np.ndarray) -> Encoder:
    """A fitted dictionary coded by the same atoms ``keep`` on every pixel.

    The rank-``k`` truncation of a dense basis: every pixel keeps its coefficients on
    ``keep`` and drops the others, so ``L0 = len(keep)``.

    Args:
        fit: A fitted dictionary.
        keep: Atom indices to keep, e.g. from :func:`leading_atoms`.

    Returns:
        ``(n, D) -> (n, M)`` codes, float64.
    """
    keep = np.asarray(keep, dtype=np.int64)

    def encode(raw: np.ndarray) -> np.ndarray:
        z = np.asarray(fit.encode(raw), dtype=np.float64)
        out = np.zeros_like(z)
        out[:, keep] = z[:, keep]
        return out

    return encode


def write_panel_table(eval_dir: str | Path, rows: list[dict], provenance: dict) -> None:
    """Write a panel table as ``panel.json`` (rows plus provenance) and ``panel.csv`` (rows).

    Args:
        eval_dir: Directory to write into; created if missing.
        rows: One flat dict per decomposition.
        provenance: What sample and store the rows rest on.
    """
    from geosae.run import write_metrics_csv

    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "panel.json").write_text(json.dumps({"_provenance": provenance, "rows": rows}, indent=2))
    write_metrics_csv(eval_dir / "panel.csv", rows)


def merge_panel_shards(paths: Sequence[str | Path]) -> tuple[list[dict], dict]:
    """Join several panel tables (``panel.json``) that scored disjoint slices of one ladder.

    Args:
        paths: ``panel.json`` files, in the order their rows should win a
            name collision (later paths override earlier ones).

    Returns:
        ``(rows, provenance)`` — rows deduplicated by ``name`` (last shard
        wins, never silently doubled), and the first shard's ``_provenance``
        with ``shards`` (every source path) and ``n_shards`` added.

    Raises:
        ValueError: ``paths`` is empty.
    """
    if not paths:
        raise ValueError("no shards to merge")
    rows_by_name: dict[str, dict] = {}
    provenance: dict | None = None
    for p in paths:
        data = json.loads(Path(p).read_text())
        if provenance is None:
            provenance = dict(data.get("_provenance") or {})
        for i, row in enumerate(data.get("rows", [])):
            rows_by_name[row.get("name", f"row{i}")] = row
    provenance = provenance or {}
    provenance["shards"] = [str(p) for p in paths]
    provenance["n_shards"] = len(paths)
    return list(rows_by_name.values()), provenance


__all__ = [
    "Encoder",
    "LadderSample",
    "build_real_ladder_sample",
    "dictionary_encoder",
    "leading_atoms",
    "materialise",
    "merge_panel_shards",
    "model_encoder",
    "model_offset",
    "rank_encoder",
    "real_concept_vocabulary",
    "real_ladder_sample_from_registry",
    "real_pixel_source",
    "score_decomposition",
    "write_panel_table",
]

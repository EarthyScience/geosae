"""One frozen, versioned evaluation pool that every metric draws its pixels from.

A metric that draws its own subsample rests on pixels no other metric saw, so two
published numbers are not comparable and neither can be re-derived. An
:class:`EvalPool` fixes that: a :class:`PoolSpec` is the reproducible recipe
(store, split, sizes, seed), :meth:`EvalPool.build` materialises it once — the
cells, the shared within-patch pixel positions, **and** the per-concept
class-balanced draw of
:func:`~geosae.eval.coverage.balanced_binary_subsample` — and the result is saved
as one ``.npz`` next to a JSON manifest. A metric then calls
:meth:`EvalPool.balanced_indices` instead of sampling, and every reported number
rests on the identical pixels.

The pool is tied to the store it was drawn from by :func:`store_fingerprint`;
:meth:`EvalPool.verify` recomputes it and refuses a pool whose store has changed
underneath it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geosae.data.quality import store_fingerprint
from geosae.eval.registry import RegistrySample, _draw_sample_positions, _read_pixel_sample

if TYPE_CHECKING:
    from geosae.data.dataset import PatchData

# `store_fingerprint` lives in `geosae.data.quality` (the quality-mask cache needs it
# too) and is re-exported here.


@dataclass(frozen=True)
class PoolSpec:
    """The reproducible recipe of an evaluation pool.

    Attributes:
        name: Pool name, e.g. ``reference``.
        version: Version of that name; bump it whenever any other field changes.
        data_zarr: Path to the patch store.
        split: ``train`` / ``val`` / ``test``.
        split_name: Split-label array in the store.
        embedding_name: Embedding array in the store.
        n_cells: Cells drawn from the split.
        pixels_per_cell: Pixels read from every drawn cell, at positions drawn once.
        seed: RNG seed of the cell, pixel-position and balanced draws.
        products: Concept-product subset; ``None`` = every registered product.
        max_per_side: Balanced-sample cap per concept side.
        min_positive: Minimum positive pixels for a concept to be scorable.
    """

    name: str
    version: int
    data_zarr: str
    split: str = "test"
    split_name: str = "split_210k"
    embedding_name: str = "alphaearth"
    n_cells: int = 400
    pixels_per_cell: int = 256
    seed: int = 0
    products: tuple[str, ...] | None = None
    max_per_side: int = 2500
    min_positive: int = 50

    @property
    def pool_id(self) -> str:
        """``<name>.v<version>`` — the human handle a published number cites."""
        return f"{self.name}.v{self.version}"

    @property
    def digest(self) -> str:
        """12 hex characters of a sha256 over every field, so no two recipes collide."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """JSON-serialisable field mapping (``products`` as a list)."""
        d = asdict(self)
        d["products"] = None if self.products is None else list(self.products)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PoolSpec:
        """Inverse of :meth:`to_dict`."""
        kw = dict(d)
        products = kw.get("products")
        kw["products"] = None if products is None else tuple(str(p) for p in products)
        return cls(**kw)


@dataclass(frozen=True)
class EvalPool:
    """A materialised, frozen draw: the pixels *and* the per-concept balanced sets.

    Attributes:
        spec: The recipe this pool was built from.
        cells: ``(n_cells,)`` int64 store row ids, sorted ascending.
        pixel_rows: ``(n_pix,)`` int64 within-patch row positions, identical for
            every cell (the invariant
            :func:`~geosae.eval.coverage.block_permuted_best_auroc` needs).
        pixel_cols: ``(n_pix,)`` int64 within-patch column positions.
        store_fingerprint: :func:`store_fingerprint` at build time.
        concept_names: ``(C,)`` human concept names.
        concept_groups: ``(C,)`` concept group, ``None`` where ungrouped.
        class_table: ``(C, 2)`` int ``[product_id, class_code]``.
        product_order: Product name for each ``product_id``.
        n_positive: ``(C,)`` positive-pixel count of each concept in the pool.
        balanced: ``concept index -> (indices, binary labels)``, the frozen
            balanced draw; only scorable concepts appear.
    """

    spec: PoolSpec
    cells: np.ndarray
    pixel_rows: np.ndarray
    pixel_cols: np.ndarray
    store_fingerprint: str
    concept_names: list[str]
    concept_groups: list[str | None]
    class_table: np.ndarray
    product_order: list[str]
    n_positive: np.ndarray
    balanced: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls, spec: PoolSpec, *, data: PatchData | None = None, sample_cache: str | Path | None = None
    ) -> EvalPool:
        """Draw the pool. Deterministic: the same spec always gives the same pool.

        Args:
            spec: The recipe.
            data: An open :class:`~geosae.data.dataset.PatchData` over the spec's
                split, reused instead of opening the store again.
            sample_cache: Where to write the pixels this draw already read, for
                :meth:`registry_sample` to pick up later. Building reads them
                anyway, so caching here is what stops every metric — and the
                build itself — from paying for that read twice.

        Returns:
            The frozen pool, its per-concept balanced draws already made.
        """
        from geosae.data.dataset import PatchData
        from geosae.eval.coverage import balanced_binary_subsample

        if data is None:
            data = PatchData(
                spec.data_zarr,
                embedding_name=spec.embedding_name,
                split_name=spec.split_name,
                split=spec.split,
                compute=False,
            )
        local, rows, cols = _draw_sample_positions(
            data, n_patches=spec.n_cells, pixels_per_patch=spec.pixels_per_cell, seed=spec.seed
        )
        sample = _read_pixel_sample(data, local, rows, cols, products=spec.products)
        if sample_cache is not None:
            Path(sample_cache).parent.mkdir(parents=True, exist_ok=True)
            sample.save(sample_cache)

        balanced: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c in range(sample.class_table.shape[0]):
            arr = sample.labels.get(sample.product_order[int(sample.class_table[c, 0])])
            if arr is None:
                continue
            # seed + c mirrors concept_auroc_matrix's per-column offset, so a
            # frozen draw is the draw the ad-hoc path would have made.
            idx, y = balanced_binary_subsample(
                arr,
                int(sample.class_table[c, 1]),
                max_per_side=spec.max_per_side,
                seed=spec.seed + c,
                min_positive=spec.min_positive,
            )
            if idx.size:
                balanced[c] = (idx.astype(np.int64), y.astype(np.int64))

        return cls(
            spec=spec,
            cells=np.asarray(data.cells[local], dtype=np.int64),
            pixel_rows=rows,
            pixel_cols=cols,
            store_fingerprint=store_fingerprint(
                spec.data_zarr, split_name=spec.split_name, embedding_name=spec.embedding_name
            ),
            concept_names=list(sample.concept_names),
            concept_groups=list(sample.concept_groups),
            class_table=np.asarray(sample.class_table, dtype=np.int64),
            product_order=list(sample.product_order),
            n_positive=np.asarray(sample.n_positive, dtype=np.int64),
            balanced=balanced,
        )

    # ------------------------------------------------------------------ query
    @property
    def n_pixels(self) -> int:
        """Number of pixels in the pool, ``n_cells * pixels_per_cell``."""
        return int(self.cells.size) * int(self.pixel_rows.size)

    def scorable_concepts(self) -> np.ndarray:
        """``(C,)`` bool: concepts that got a non-empty frozen balanced draw."""
        out = np.zeros(self.class_table.shape[0], dtype=bool)
        for c, (idx, _) in self.balanced.items():
            out[c] = idx.size > 0
        return out

    def balanced_indices(self, concept: int) -> tuple[np.ndarray, np.ndarray]:
        """The frozen ``(indices, binary labels)`` of one concept.

        This replaces per-metric subsampling: a caller uses what the pool froze
        and never re-draws.

        Args:
            concept: Concept column index into ``class_table``.

        Returns:
            ``(indices, y)`` into the pool's pixels; both empty for a concept
            that is not scorable.
        """
        empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return self.balanced.get(int(concept), empty)

    def verify(self, *, strict: bool = True) -> bool:
        """Check the store still matches the fingerprint recorded at build time.

        Args:
            strict: Raise instead of returning ``False`` on a mismatch.

        Returns:
            True when the store's fingerprint is unchanged.

        Raises:
            ValueError: If ``strict`` and the store changed.
        """
        current = store_fingerprint(
            self.spec.data_zarr, split_name=self.spec.split_name, embedding_name=self.spec.embedding_name
        )
        if current == self.store_fingerprint:
            return True
        if strict:
            raise ValueError(
                f"pool {self.spec.pool_id} was built on store fingerprint {self.store_fingerprint}, "
                f"but {self.spec.data_zarr} now fingerprints as {current} — rebuild the pool or "
                "restore the store"
            )
        return False

    def registry_sample(self, *, data: PatchData | None = None, cache: str | Path | None = None) -> RegistrySample:
        """Materialise the embeddings and labels of the frozen draw.

        Reading the draw out of the store costs minutes and every metric needs
        the identical pixels, so ``cache`` makes that a one-off: the sample is
        written on the first call and read back on every later one. A cache that
        does not describe *this* pool's pixels is refused rather than used.

        Args:
            data: An open :class:`~geosae.data.dataset.PatchData` over the spec's
                split, reused instead of opening the store again.
            cache: Path to an ``.npz``; read when it already holds this pool's
                sample, written after materialising otherwise.

        Returns:
            The sample, pixels in the pool's own order.

        Raises:
            ValueError: If the split no longer contains every cell of the pool,
                or if ``cache`` holds a sample of different pixels.
        """
        from geosae.data.dataset import PatchData

        expect_cells = np.repeat(self.cells, self.pixel_rows.size)
        if cache is not None and Path(cache).exists():
            sample = RegistrySample.load(cache)
            if not np.array_equal(sample.cell, expect_cells):
                raise ValueError(
                    f"the sample cache {cache} holds {sample.n_pixels} pixels over "
                    f"{np.unique(sample.cell).size} cells, which are not the pixels of pool "
                    f"{self.spec.pool_id} — point at a different cache or delete this one"
                )
            return sample

        if data is None:
            data = PatchData(
                self.spec.data_zarr,
                embedding_name=self.spec.embedding_name,
                split_name=self.spec.split_name,
                split=self.spec.split,
                compute=False,
            )
        local = np.searchsorted(data.cells, self.cells)
        if local.size and (local.max() >= data.cells.size or not np.array_equal(data.cells[local], self.cells)):
            raise ValueError(
                f"pool {self.spec.pool_id} references cells that are no longer in split "
                f"{self.spec.split!r} of {self.spec.split_name!r}"
            )
        sample = _read_pixel_sample(data, local, self.pixel_rows, self.pixel_cols, products=self.spec.products)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            sample.save(cache)
        return sample

    def _prediction_rows(self, pred_path: str | Path) -> np.ndarray:
        """Patch row per pool cell, **in the pool's own cell order**.

        Shared by :meth:`prediction_indices` (which re-sorts the result
        ascending, order no longer meaningful to its callers) and
        :meth:`dense_codes` (which needs the row-to-cell correspondence
        preserved, so it lines up with :meth:`registry_sample`'s pixel order).
        """
        import zarr

        cells = np.asarray(zarr.open_group(str(pred_path), mode="r")["cell"], dtype=np.int64)
        if cells.size == 0:
            raise ValueError(f"{pred_path} predicts no cells at all, so it cannot cover pool {self.spec.pool_id}")
        order = np.argsort(cells, kind="stable")
        pos = np.clip(np.searchsorted(cells[order], self.cells), 0, cells.size - 1)
        idx = order[pos]
        missing = int((cells[idx] != self.cells).sum())
        if missing:
            raise ValueError(
                f"{pred_path} is missing {missing} of pool {self.spec.pool_id}'s {self.cells.size} cells — "
                "the predictions were written for a different split or store"
            )
        return idx.astype(np.int64)

    def prediction_indices(self, pred_path: str | Path) -> np.ndarray:
        """Where this pool's cells sit in a predictions zarr.

        A metric that streams sparse codes indexes patches, not store rows, so
        this is what puts such a metric on the pool's cells instead of on a draw
        of its own.

        Args:
            pred_path: A predictions zarr written by ``geosae predict``.

        Returns:
            ``(n_cells,)`` int64 patch indices, ascending.

        Raises:
            ValueError: If the predictions do not cover every cell of the pool.
        """
        return np.sort(self._prediction_rows(pred_path))

    def dense_codes(self, pred_path: str | Path, n_features: int) -> np.ndarray:
        """Dense SAE codes at exactly this pool's pixels, in the pool's own order.

        :meth:`registry_sample` reads the raw embeddings of the pool's ``(cell,
        row, col)`` triples straight from the store; this reads the matching
        sparse codes from a predictions zarr and densifies them, so the two
        outputs line up pixel-for-pixel — the pair a
        :class:`~geosae.eval.statistic.PanelAccumulator` pass needs as its
        ``e`` / ``codes`` arguments. Only the pool's own pixel positions are
        read from each patch, never the whole ``(H, W)`` grid.

        **Materialises the whole pool at once** — ``n_pixels * n_features``
        float32, which at a large sample size and dictionary width is tens of
        GB (8.1M pixels x M=1024 is 33 GB). A caller streaming a large sample
        through a :class:`~geosae.eval.statistic.PanelAccumulator` should use
        :meth:`iter_dense_codes` instead and never hold the full array.

        Args:
            pred_path: A predictions zarr written by ``geosae predict``.
            n_features: ``M``, the dictionary width.

        Returns:
            ``(n_pixels, n_features)`` float32 dense codes, cell-major and,
            within a cell, ordered like ``pixel_rows`` / ``pixel_cols`` — the
            same order :meth:`registry_sample` returns.

        Raises:
            ValueError: If the predictions do not cover every cell of the pool.
        """
        return np.concatenate(
            [chunk for _, _, chunk in self.iter_dense_codes(pred_path, n_features, chunk_cells=self.cells.size)],
            axis=0,
        )

    def iter_dense_codes(
        self, pred_path: str | Path, n_features: int, *, chunk_cells: int = 200
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """:meth:`dense_codes`, ``chunk_cells`` cells at a time — never the whole pool at once.

        Densifying the pool's sparse codes in one shot costs ``n_pixels *
        n_features`` float32; this is the same read :meth:`dense_codes` does,
        split by cell so a caller can fold each chunk into an accumulator and
        discard it before reading the next.

        Args:
            pred_path: A predictions zarr written by ``geosae predict``.
            n_features: ``M``, the dictionary width.
            chunk_cells: Cells densified per yielded chunk.

        Yields:
            ``(pixel_start, pixel_end, dense)`` — the ``[pixel_start:pixel_end)``
            slice this chunk covers in :meth:`registry_sample`'s pixel order,
            and ``dense`` itself, ``((pixel_end - pixel_start), n_features)``
            float32.

        Raises:
            ValueError: If the predictions do not cover every cell of the pool.
        """
        import zarr

        from geosae.predict.sparsify import sparse_to_dense

        rows_by_cell = self._prediction_rows(pred_path)
        group = zarr.open_group(str(pred_path), mode="r")
        z_values, z_indices = group["z_values"], group["z_indices"]
        r, c = self.pixel_rows, self.pixel_cols
        n_pix = r.size
        n_cells = rows_by_cell.size
        max_k = z_values.shape[-1]
        chunk_cells = max(1, int(chunk_cells))
        for start in range(0, n_cells, chunk_cells):
            end = min(start + chunk_cells, n_cells)
            values = np.empty((end - start, n_pix, max_k), dtype=z_values.dtype)
            indices = np.empty((end - start, n_pix, max_k), dtype=z_indices.dtype)
            for i, row in enumerate(rows_by_cell[start:end]):
                values[i] = np.asarray(z_values[int(row)])[r, c, :]
                indices[i] = np.asarray(z_indices[int(row)])[r, c, :]
            dense = sparse_to_dense(values, indices, n_features).reshape((end - start) * n_pix, n_features)
            yield start * n_pix, end * n_pix, dense

    def pixel_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lon, lat)`` of every pool pixel, in the pool's pixel order.

        The store geolocates a cell, not a pixel, so all pixels of a cell share
        its centroid. That resolves distances *between* cells — the cross-patch
        window a variogram is read over — and not the sub-kilometre structure
        inside one.

        Returns:
            Two ``(n_pixels,)`` float64 arrays of degrees.
        """
        import zarr

        group = zarr.open_group(str(self.spec.data_zarr), mode="r")
        n_pix = int(self.pixel_rows.size)
        lon = np.asarray(group["lon"][:], dtype=np.float64)[self.cells]
        lat = np.asarray(group["lat"][:], dtype=np.float64)[self.cells]
        return np.repeat(lon, n_pix), np.repeat(lat, n_pix)

    def manifest(self) -> dict:
        """JSON-serialisable provenance: what a published number cites."""
        scorable = self.scorable_concepts()
        per_product: dict[str, int] = {p: 0 for p in self.product_order}
        for c in np.where(scorable)[0]:
            per_product[self.product_order[int(self.class_table[c, 0])]] += 1
        return {
            "pool_id": self.spec.pool_id,
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.digest,
            "store_fingerprint": self.store_fingerprint,
            "n_cells": int(self.cells.size),
            "n_pixels": self.n_pixels,
            "n_concepts": int(self.class_table.shape[0]),
            "n_scorable_concepts": int(scorable.sum()),
            "scorable_per_product": per_product,
        }

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> str:
        """Write the pool to one ``.npz`` plus a sibling ``.json`` manifest.

        Args:
            path: Destination ``.npz``; the manifest is written alongside it with
                a ``.json`` suffix.

        Returns:
            The ``.npz`` path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        concepts = np.array(sorted(self.balanced), dtype=np.int64)
        sizes = np.array([self.balanced[int(c)][0].size for c in concepts], dtype=np.int64)
        flat_idx = np.concatenate([self.balanced[int(c)][0] for c in concepts]) if concepts.size else np.zeros(0, "i8")
        flat_y = np.concatenate([self.balanced[int(c)][1] for c in concepts]) if concepts.size else np.zeros(0, "i8")
        np.savez_compressed(
            path,
            spec=json.dumps(self.spec.to_dict(), sort_keys=True),
            cells=self.cells,
            pixel_rows=self.pixel_rows,
            pixel_cols=self.pixel_cols,
            store_fingerprint=self.store_fingerprint,
            concept_names=np.array(self.concept_names, dtype=object),
            concept_groups=np.array(["" if g is None else g for g in self.concept_groups], dtype=object),
            class_table=self.class_table,
            product_order=np.array(self.product_order, dtype=object),
            n_positive=self.n_positive,
            balanced_concepts=concepts,
            balanced_sizes=sizes,
            balanced_indices=flat_idx.astype(np.int64),
            balanced_labels=flat_y.astype(np.int64),
        )
        path.with_suffix(".json").write_text(json.dumps(self.manifest(), indent=2, sort_keys=True))
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> EvalPool:
        """Read a pool written by :meth:`save`."""
        with np.load(path, allow_pickle=True) as z:
            concepts = z["balanced_concepts"].astype(np.int64)
            bounds = np.concatenate([[0], np.cumsum(z["balanced_sizes"].astype(np.int64))])
            idx, y = z["balanced_indices"], z["balanced_labels"]
            balanced = {
                int(c): (idx[bounds[i] : bounds[i + 1]].astype(np.int64), y[bounds[i] : bounds[i + 1]].astype(np.int64))
                for i, c in enumerate(concepts)
            }
            return cls(
                spec=PoolSpec.from_dict(json.loads(str(z["spec"]))),
                cells=z["cells"].astype(np.int64),
                pixel_rows=z["pixel_rows"].astype(np.int64),
                pixel_cols=z["pixel_cols"].astype(np.int64),
                store_fingerprint=str(z["store_fingerprint"]),
                concept_names=[str(n) for n in z["concept_names"]],
                concept_groups=[None if str(g) == "" else str(g) for g in z["concept_groups"]],
                class_table=z["class_table"].astype(np.int64),
                product_order=[str(p) for p in z["product_order"]],
                n_positive=z["n_positive"].astype(np.int64),
                balanced=balanced,
            )


def audit_product_coverage(data_zarr: str, *, split_name: str = "split_210k", split: str = "test") -> dict[str, dict]:
    """Report how many cells carry each reference product, globally and in a split.

    A product can be registered in the concept vocabulary and still label no cell
    at all; this is what separates the scorable products from the empty ones
    before a metric is run on them.

    Args:
        data_zarr: Path to the patch store.
        split_name: Split-label array in the store.
        split: ``train`` / ``val`` / ``test``.

    Returns:
        ``product -> {"n_store", "n_split", "frac_split"}``, one entry per
        ``mask_*`` array, keyed by the product name the mask belongs to.
        ``frac_split`` is the fraction of the split's cells the product covers.
    """
    import zarr

    from geosae.data.dataset import SPLIT_LABELS

    group = zarr.open_group(str(data_zarr), mode="r")
    in_split = np.asarray(group[split_name][:]) == SPLIT_LABELS[split]
    n_split = int(in_split.sum())
    out: dict[str, dict] = {}
    for name in sorted(group.array_keys()):
        if not name.startswith("mask_"):
            continue
        mask = np.asarray(group[name][:], dtype=bool)
        both = int((mask & in_split).sum())
        out[name[len("mask_") :]] = {
            "n_store": int(mask.sum()),
            "n_split": both,
            "frac_split": both / n_split if n_split else 0.0,
        }
    return out


def _count_scorable(sample: RegistrySample, keep: np.ndarray, min_positive: int) -> dict[str, int]:
    """Scorable concepts per product among the pixels ``keep`` selects.

    A concept is scorable when it has at least ``min_positive`` positive pixels.
    One label histogram per product rather than one pass per concept, so the
    count is cheap enough to repeat over every rung of a ladder.

    Args:
        sample: The labelled pixel sample.
        keep: ``(N,)`` bool mask selecting the pixels that count.
        min_positive: Positive-pixel floor a concept must clear.

    Returns:
        ``product -> scorable concept count``, one entry per product of the
        sample, zero included.
    """
    table = sample.class_table
    out: dict[str, int] = {}
    for product_id, product in enumerate(sample.product_order):
        arr = sample.labels.get(product)
        codes = table[table[:, 0] == product_id, 1]
        if arr is None or not codes.size:
            out[product] = 0
            continue
        values = np.asarray(arr)[keep]
        values = values[values >= 0]  # a negative label is no registry code
        hist = np.bincount(values, minlength=int(codes.max()) + 1)
        out[product] = int((hist[codes] >= min_positive).sum())
    return out


def scorable_concept_ladder(
    sample: RegistrySample,
    sizes: Sequence[int],
    *,
    min_positive: int = 50,
    n_draws: int = 1,
    seed: int = 0,
) -> list[dict]:
    """How many concepts a pool can score, against how many cells it holds.

    This is how a pool size gets chosen. A coverage count is a count *out of* the
    scorable concepts, so ``n_cells`` sets the denominator of every coverage
    statistic; reading the curve — and the draw-to-draw spread beside it —
    separates a pool size that is large enough from one that is still buying
    concepts.

    Args:
        sample: A labelled pixel sample at least as large as the biggest rung,
            from :func:`~geosae.eval.registry.sample_registry_pixels` or
            :meth:`EvalPool.registry_sample`.
        sizes: Cell counts to evaluate, each clamped to the sample's cell count.
        min_positive: Positive-pixel floor a concept must clear to be scorable;
            mirrors :attr:`PoolSpec.min_positive`.
        n_draws: Cell subsets per rung. ``1`` counts on a nested prefix of the
            sample's cells, so the rungs share one draw and the curve is that
            one draw refined; more than one draws independent subsets per rung,
            which is what bounds how much of the curve is sampling noise.
        seed: RNG seed of the independent draws (unused when ``n_draws`` is 1).

    Returns:
        One dict per entry of ``sizes``, in order, holding the realised
        ``n_cells`` and ``n_pixels``, ``n_draws``, the mean ``n_scorable`` with
        its ``n_scorable_std`` / ``_min`` / ``_max`` over draws, the mean
        ``scorable_per_product``, and the total ``n_concepts`` the count is
        out of.
    """
    cells = np.unique(sample.cell)
    rng = np.random.default_rng(seed)
    rungs: list[dict] = []
    for size in sizes:
        n = int(min(int(size), cells.size))
        subsets = [cells[:n]] if n_draws <= 1 else [rng.choice(cells, size=n, replace=False) for _ in range(n_draws)]
        per_draw, n_pixels = [], 0
        for subset in subsets:
            keep = np.isin(sample.cell, subset)
            n_pixels = int(keep.sum())
            per_draw.append(_count_scorable(sample, keep, min_positive))
        totals = np.array([sum(d.values()) for d in per_draw], dtype=np.float64)
        rungs.append(
            {
                "n_cells": n,
                "n_pixels": n_pixels,
                "n_draws": len(subsets),
                "n_scorable": float(totals.mean()),
                "n_scorable_std": float(totals.std()),
                "n_scorable_min": int(totals.min()),
                "n_scorable_max": int(totals.max()),
                "scorable_per_product": {p: float(np.mean([d[p] for d in per_draw])) for p in sample.product_order},
                "n_concepts": int(sample.class_table.shape[0]),
            }
        )
    return rungs


def sample_cache_path(pool_path: str | Path) -> Path:
    """The materialised-sample cache that belongs beside a saved pool.

    Args:
        pool_path: The ``.npz`` a pool was saved to.

    Returns:
        The sibling ``.sample.npz`` path :meth:`EvalPool.registry_sample` caches to.
    """
    return Path(pool_path).with_suffix(".sample.npz")


def default_sample_cache_path(spec: PoolSpec, *, cache_root: str | Path = "data/eval_cache/samples") -> Path:
    """The canonical, content-addressed materialised-sample path for a spec.

    Unlike :func:`sample_cache_path`, which derives a cache location from
    wherever a caller chose to save the pool file, this is keyed by
    :attr:`PoolSpec.digest` (the draw recipe) and :func:`store_fingerprint` (the
    store's content at draw time) alone. Two runs that build or score the
    identical spec against the identical store therefore land on the identical
    path *by construction*, with no coordination between callers — two configs
    with different ``pool.out`` but the same recipe pay the draw cost once, not
    once each.

    Args:
        spec: The pool's recipe.
        cache_root: Project-level directory the cache lives under.

    Returns:
        ``<cache_root>/<spec.digest>-<store_fingerprint>.sample.npz``.
    """
    fp = store_fingerprint(spec.data_zarr, split_name=spec.split_name, embedding_name=spec.embedding_name)
    return Path(cache_root) / f"{spec.digest}-{fp}.sample.npz"


def load_pool(path: str | Path, *, verify: bool = True) -> EvalPool:
    """Load a saved pool and, by default, check its store is unchanged.

    Args:
        path: The ``.npz`` written by :meth:`EvalPool.save`.
        verify: Run :meth:`EvalPool.verify` in strict mode.

    Returns:
        The pool.

    Raises:
        ValueError: If ``verify`` and the store changed since the pool was built.
    """
    pool = EvalPool.load(path)
    if verify:
        pool.verify(strict=True)
    return pool


def spec_from_config(block: dict, *, data_zarr: str | None = None) -> PoolSpec:
    """Build a :class:`PoolSpec` from a config ``pool:`` block.

    Args:
        block: Mapping of :class:`PoolSpec` field names; unknown keys (such as
            the output path) are ignored.
        data_zarr: Store path used when the block does not carry one.

    Returns:
        The spec.

    Raises:
        ValueError: If no store path is available.
    """
    kw = {k: v for k, v in block.items() if k in set(PoolSpec.__dataclass_fields__)}
    path = kw.pop("data_zarr", None) or data_zarr
    if not path:
        raise ValueError("a pool spec needs a store: set pool.data_zarr or data.zarr_path")
    products = kw.pop("products", None)
    return PoolSpec(data_zarr=str(path), products=None if products is None else tuple(str(p) for p in products), **kw)


__all__ = [
    "EvalPool",
    "PoolSpec",
    "audit_product_coverage",
    "default_sample_cache_path",
    "load_pool",
    "sample_cache_path",
    "scorable_concept_ladder",
    "spec_from_config",
    "store_fingerprint",
]

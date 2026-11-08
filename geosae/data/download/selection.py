"""Seeded cell selection and chunk-aligned train/val/test split assignment.

General building blocks for turning a large candidate grid of cells into a
dataset: draw a reproducible subset, then label each row with a split. Pure
:mod:`numpy`, so it is testable on CPU without any geospatial dependency.

Two stages:

1. **Cell selection** (:func:`sample_subset_indices`). From a base grid of
   ``total`` candidate cells, draw ``n_cells`` *without replacement* using
   ``np.random.default_rng(seed)`` and then shuffle the drawn indices with a
   **second** generator ``np.random.default_rng(seed + 1)`` so that any
   contiguous chunk of the resulting array is geographically scattered. The
   returned array is the ``cell_indices`` coordinate of the store: for slot
   ``i`` it holds the index into the base grid whose patch lives at store
   row ``i``.

2. **Split assignment** (:func:`compute_split`, via a :class:`SplitSpec`).
   Splits are deterministic contiguous *index ranges* over the
   (already-scattered) cell ordering, chunk-aligned to a configurable chunk
   size. Cells missing required data (``both_mask``) are labelled
   :data:`EXCLUDED`. Because the ordering from stage 1 is already shuffled,
   an index-range split is spatially disjoint by cell yet trivially
   reproducible and chunk-contiguous for fast IO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Split label convention (int8), matching the on-disk ``split_*`` arrays ---
TRAIN = np.int8(0)
VAL = np.int8(1)
TEST = np.int8(2)
EXCLUDED = np.int8(-1)

#: Default zarr chunk size (cells per chunk) split ranges are aligned to.
DEFAULT_CHUNK_SIZE = 50


def sample_subset_indices(total: int, n_cells: int, seed: int) -> np.ndarray:
    """Deterministically select ``n_cells`` base-grid indices, then scatter them.

    Args:
        total: Number of cells in the base grid.
        n_cells: Number of cells to draw. If ``>= total`` all cells are used.
        seed: Base RNG seed. The draw uses ``default_rng(seed)``; the subsequent
            scatter uses ``default_rng(seed + 1)``.

    Returns:
        ``int64`` array of length ``min(n_cells, total)`` of base-grid indices in
        the scattered order used as the store's ``cell_indices`` coordinate.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")

    rng = np.random.default_rng(seed)
    if n_cells >= total:
        idx = np.arange(total, dtype=np.int64)
    else:
        idx = rng.choice(total, size=n_cells, replace=False).astype(np.int64)
    # Second, independent generator so chunk-level splits are geographically
    # scattered (this is what makes an index-range split spatially disjoint).
    np.random.default_rng(seed + 1).shuffle(idx)
    return idx


@dataclass(frozen=True)
class SplitSpec:
    """A chunk-aligned index-range train/val/test split over the cell ordering.

    Ranges are half-open ``[start, end)`` row indices into the store (i.e. into
    the scattered :func:`sample_subset_indices` ordering). Exactly one of
    train/val/test owns each contiguous range; any row not covered by a range,
    or excluded by ``both_mask``, is labelled :data:`EXCLUDED`.

    Attributes:
        name: Suffix of the on-disk array (``split_<name>``), e.g. ``"210k"``.
        test: ``(start, end)`` half-open row range for test, or ``None``.
        train: ``(start, end)`` half-open row range for train, or ``None``.
        val: ``(start, end)`` half-open row range for val, or ``None``.
        chunk_size: Zarr chunk size the boundaries must be multiples of.
    """

    name: str
    test: tuple[int, int] | None
    train: tuple[int, int] | None
    val: tuple[int, int] | None
    chunk_size: int = DEFAULT_CHUNK_SIZE


def compute_split(spec: SplitSpec, n_cells: int, both_mask: np.ndarray) -> np.ndarray:
    """Materialise a :class:`SplitSpec` into a per-cell ``int8`` label array.

    Args:
        spec: The split ranges.
        n_cells: Number of rows in the store (length of the output).
        both_mask: Boolean array, ``True`` where the cell has the data required
            for the split (e.g. every embedding it needs). Cells with ``False``
            become :data:`EXCLUDED` regardless of range membership.

    Returns:
        ``int8`` array of length ``n_cells`` with values in
        ``{TRAIN, VAL, TEST, EXCLUDED}``.
    """
    both_mask = np.asarray(both_mask, dtype=bool)
    if both_mask.shape != (n_cells,):
        raise ValueError(f"both_mask must have shape ({n_cells},), got {both_mask.shape}")

    split = np.full(n_cells, EXCLUDED, dtype=np.int8)
    for label, rng_ in ((TEST, spec.test), (TRAIN, spec.train), (VAL, spec.val)):
        if rng_ is None:
            continue
        start, end = rng_
        if start % spec.chunk_size or (end % spec.chunk_size and end < n_cells):
            raise ValueError(f"range {rng_} for label {int(label)} not aligned to chunk_size={spec.chunk_size}")
        split[start : min(end, n_cells)] = label
    split[~both_mask] = EXCLUDED
    return split


def fractional_split(
    name: str, n_cells: int, *, val_frac: float, test_frac: float, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> SplitSpec:
    """A :class:`SplitSpec` sized by fractions rather than absolute row counts.

    Useful when the store's row count is only known once the data is assembled.
    The ranges follow the store convention (test first, then train, then val)
    and are rounded down to whole chunks, so the last partial chunk is excluded.

    Args:
        name: Split name (the array becomes ``split_<name>``).
        n_cells: Number of rows in the store.
        val_frac: Share of rows for validation.
        test_frac: Share of rows for test; the remainder is train.
        chunk_size: Boundaries are aligned to this multiple.

    Returns:
        The split specification.

    Raises:
        ValueError: If the fractions are not in ``[0, 1)`` or sum to ``>= 1``.
    """
    if not (0 <= val_frac < 1 and 0 <= test_frac < 1) or val_frac + test_frac >= 1:
        raise ValueError(f"val_frac + test_frac must be in [0, 1), got {val_frac} + {test_frac}")

    def align(x: float) -> int:
        return int(x) // chunk_size * chunk_size

    test_end = align(n_cells * test_frac)
    val_start = align(n_cells * (1.0 - val_frac))
    return SplitSpec(
        name=name,
        test=(0, test_end),
        train=(test_end, val_start),
        val=(val_start, n_cells),
        chunk_size=chunk_size,
    )


def split_counts(split: np.ndarray) -> dict[str, int]:
    """Return ``{'train','val','test','excluded'}`` counts for a label array."""
    split = np.asarray(split)
    return {
        "train": int(np.sum(split == TRAIN)),
        "val": int(np.sum(split == VAL)),
        "test": int(np.sum(split == TEST)),
        "excluded": int(np.sum(split == EXCLUDED)),
    }


__all__ = [
    "TRAIN",
    "VAL",
    "TEST",
    "EXCLUDED",
    "DEFAULT_CHUNK_SIZE",
    "SplitSpec",
    "sample_subset_indices",
    "compute_split",
    "fractional_split",
    "split_counts",
]

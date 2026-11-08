r"""Streaming sufficient statistics of a decomposition, and the panel metrics derived from them.

One pass over a pixel sample fills a :class:`PanelAccumulator` with the moments every panel block reads
(:mod:`geosae.eval.blocks`); every derivation below is a pure function of the finished accumulator (plus,
where needed, the decoder, which is not data-dependent and so is never stored here).

**Accumulator set.**

* *global* (fidelity) — ``sum_e``, ``sum_e_outer`` = :math:`\sum e e^\top`, ``sum_ehat``,
  ``sum_ehat_sq``, ``sum_cross`` = :math:`\sum e \cdot \hat e`, ``sum_err2`` =
  :math:`\sum \lVert e - \hat e \rVert^2`, ``sum_pixel_cos`` = :math:`\sum \cos(e, \hat e)`,
  ``sum_l0``. FVE folds out of ``sum_e`` / ``sum_e_outer`` /
  ``sum_err2`` through :math:`\sum \lVert e - \bar e \rVert^2 = \operatorname{tr}(\sum e e^\top) - N
  \lVert \bar e \rVert^2`; the centred reconstruction cosine also needs ``sum_ehat``, ``sum_ehat_sq``
  and ``sum_cross``.
* *per feature* (dictionary geometry) — ``n_f`` (active count) and ``sum_a`` / ``sum_a2``
  (activation moments).
* *per feature pair* — ``coact`` = :math:`A^\top A`, binary co-activation counts.
* *per (feature, concept)* (known concepts) — ``n_fc`` (in-concept active-pixel contingency) and a
  quantised activation histogram ``concept_hist`` ``(F, C, B)`` restricted to each concept's positive
  pixels, plus the much smaller ``product_hist`` ``(F, P, B)`` over each label product's whole labelled
  population. The negative side of a concept's histogram is ``product_hist`` minus ``concept_hist``
  for that concept's own product, so the accumulator never needs a second ``(F, C, B)`` array; the
  folded AUROC and every coverage reduction are closed-form in these two arrays. ``argmax_n_fc``
  ``(F, C)`` is the one-vote-per-pixel counterpart of ``n_fc`` (the per-pixel activation argmax rather
  than every active feature), the statistic of :func:`unique_argmax_coverage`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

EPS = 1e-12


# ---------------------------------------------------------------------------
# The accumulator
# ---------------------------------------------------------------------------


@dataclass
class PanelAccumulator:
    """Sufficient statistics of one decomposition on one pixel sample.

    Created once with :meth:`create`, filled batch by batch with
    :meth:`update`, combined across batches or shards with :meth:`merge`, and
    persisted with :meth:`save` / :meth:`load`. Every derivation in this
    module is a pure function of the finished accumulator (plus, where the
    metric needs it, the decoder — which is not data-dependent and so is never
    stored here).

    Attributes:
        n_features: :math:`F`, the dictionary width.
        embedding_dim: :math:`D`.
        concept_names: Length-:math:`C` names of the raw (ungrouped) concept
            classes the concept metrics are scored against.
        product_names: Length-:math:`P` label-product names.
        product_of_concept: ``(C,)`` int index into ``product_names`` — the
            product each concept column belongs to.
        n_bins: Bins per (feature, concept/product) activation histogram.
        bin_edges: ``(F, n_bins - 1)`` interior quantile edges of each
            feature's *nonzero* activation, from :func:`calibrate_activation_bins`.
            A value of exactly ``0`` (inactive) always falls in bin 0.
        n_pixels: Pixels folded into the global/per-feature sums.
        sum_e, sum_e_outer: ``(D,)`` / ``(D, D)`` raw-embedding sums.
        sum_ehat, sum_ehat_sq, sum_cross, sum_err2: Reconstruction sums —
            :math:`\\sum \\hat e`, :math:`\\sum \\lVert\\hat e\\rVert^2`,
            :math:`\\sum e\\cdot\\hat e`, :math:`\\sum \\lVert e-\\hat
            e\\rVert^2`.
        sum_pixel_cos: :math:`\\sum \\cos(e, \\hat e)`, the per-pixel cosine
            between embedding and reconstruction.
        sum_l0: :math:`\\sum_x |A(x)|`, total active-latent count.
        n_f, sum_a, sum_a2: ``(F,)`` per-feature active count and activation
            moments.
        coact: ``(F, F)`` binary co-activation counts, :math:`A^\\top A`.
        n_fc: ``(F, C)`` in-concept active-pixel contingency.
        n_c: ``(C,)`` concept-positive pixel counts.
        n_p: ``(P,)`` product-labelled pixel counts (each product's own
            nodata-excluded population).
        concept_hist: ``(F, C, n_bins)`` int32 activation histogram of
            feature ``f`` restricted to concept ``c``'s positive pixels — the
            accumulator's dominant memory cost (``B=256`` at ``M=1024``,
            ``C=300`` is ~314 MB; int32 headroom is ~2.1 billion pixels per
            cell, far past any per-store pixel count here, so it costs nothing
            to halve ``concept_hist`` relative to counting in int64).
        product_hist: ``(F, P, n_bins)`` int32 activation histogram of
            feature ``f`` over product ``p``'s whole labelled population.
        argmax_n_fc: ``(F, C)`` count of pixels where feature ``f`` is the
            per-pixel activation *argmax* (a hard, one-vote-per-pixel winner,
            unlike the multi-hot ``n_fc``) and concept ``c`` is positive — the
            sufficient statistic for :func:`unique_argmax_coverage`.
    """

    n_features: int
    embedding_dim: int
    concept_names: list[str]
    product_names: list[str]
    product_of_concept: np.ndarray
    n_bins: int
    bin_edges: np.ndarray
    n_pixels: int
    sum_e: np.ndarray
    sum_e_outer: np.ndarray
    sum_ehat: np.ndarray
    sum_ehat_sq: float
    sum_cross: float
    sum_err2: float
    sum_pixel_cos: float
    sum_l0: float
    n_f: np.ndarray
    sum_a: np.ndarray
    sum_a2: np.ndarray
    coact: np.ndarray
    n_fc: np.ndarray
    n_c: np.ndarray
    n_p: np.ndarray
    concept_hist: np.ndarray
    product_hist: np.ndarray
    argmax_n_fc: np.ndarray

    # -- construction --------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        n_features: int,
        embedding_dim: int,
        concept_names: Sequence[str],
        product_names: Sequence[str],
        product_of_concept: np.ndarray,
        bin_edges: np.ndarray,
    ) -> PanelAccumulator:
        """Allocate a zeroed accumulator.

        Args:
            n_features: :math:`F`.
            embedding_dim: :math:`D`.
            concept_names: Length-:math:`C` raw concept names.
            product_names: Length-:math:`P` label-product names.
            product_of_concept: ``(C,)`` int, product index of each concept.
            bin_edges: ``(F, n_bins - 1)`` per-feature interior quantile
                edges, from :func:`calibrate_activation_bins`.
        Returns:
            A :class:`PanelAccumulator` with every sum at zero.

        Raises:
            ValueError: ``bin_edges`` does not have ``n_features`` rows, or
                ``product_of_concept`` does not have ``len(concept_names)``
                entries.
        """
        f, d = int(n_features), int(embedding_dim)
        edges = np.asarray(bin_edges, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[0] != f:
            raise ValueError(f"bin_edges must be (n_features={f}, n_bins-1), got {edges.shape}")
        poc = np.asarray(product_of_concept, dtype=np.int64)
        c, p = len(concept_names), len(product_names)
        if poc.shape != (c,):
            raise ValueError(f"product_of_concept must have shape ({c},), got {poc.shape}")
        n_bins = edges.shape[1] + 1
        return cls(
            n_features=f,
            embedding_dim=d,
            concept_names=list(concept_names),
            product_names=list(product_names),
            product_of_concept=poc,
            n_bins=n_bins,
            bin_edges=edges,
            n_pixels=0,
            sum_e=np.zeros(d, dtype=np.float64),
            sum_e_outer=np.zeros((d, d), dtype=np.float64),
            sum_ehat=np.zeros(d, dtype=np.float64),
            sum_ehat_sq=0.0,
            sum_cross=0.0,
            sum_err2=0.0,
            sum_pixel_cos=0.0,
            sum_l0=0.0,
            n_f=np.zeros(f, dtype=np.float64),
            sum_a=np.zeros(f, dtype=np.float64),
            sum_a2=np.zeros(f, dtype=np.float64),
            coact=np.zeros((f, f), dtype=np.float64),
            n_fc=np.zeros((f, c), dtype=np.float64),
            n_c=np.zeros(c, dtype=np.float64),
            n_p=np.zeros(p, dtype=np.float64),
            concept_hist=np.zeros((f, c, n_bins), dtype=np.int32),
            product_hist=np.zeros((f, p, n_bins), dtype=np.int32),
            argmax_n_fc=np.zeros((f, c), dtype=np.float64),
        )

    # -- streaming update -------------------------------------------------

    def update(
        self,
        e: np.ndarray,
        ehat: np.ndarray,
        codes: np.ndarray,
        *,
        concept_mask: np.ndarray | None = None,
        product_valid: np.ndarray | None = None,
    ) -> None:
        """Fold one batch of pixels into the accumulator, in place.

        ``concept_hist`` and ``product_hist`` are filled by one sparse-dense matrix product each
        against a shared ``(pixel, feature-bin)`` one-hot incidence matrix, rather than one
        :func:`numpy.bincount` per concept and product.

        Args:
            e: ``(n, D)`` raw embeddings, float64 upcast internally.
            ehat: ``(n, D)`` reconstructions, same pixels and order as ``e``.
            codes: ``(n, F)`` dense activations (0 = inactive).
            concept_mask: ``(n, C)`` bool concept-positive membership. Omit to
                update only the reconstruction and per-feature statistics.
            product_valid: ``(n, P)`` bool, whether each label product has a
                real (non-nodata) label at that pixel. Required together with
                ``concept_mask``.

        Raises:
            ValueError: Shapes disagree, or exactly one of ``concept_mask`` /
                ``product_valid`` is given.
        """
        e = np.asarray(e, dtype=np.float64)
        ehat = np.asarray(ehat, dtype=np.float64)
        a = np.asarray(codes, dtype=np.float64)
        n = e.shape[0]
        if e.shape != ehat.shape or e.shape[1] != self.embedding_dim:
            raise ValueError(f"e/ehat must both be (n, {self.embedding_dim}), got {e.shape} and {ehat.shape}")
        if a.shape != (n, self.n_features):
            raise ValueError(f"codes must be (n, {self.n_features}), got {a.shape}")
        if (concept_mask is None) != (product_valid is None):
            raise ValueError("concept_mask and product_valid must be given together")

        active = a > 0
        ea = active.astype(np.float64)

        self.n_pixels += n
        self.sum_e += e.sum(0)
        self.sum_e_outer += e.T @ e
        self.sum_ehat += ehat.sum(0)
        self.sum_ehat_sq += float((ehat * ehat).sum())
        self.sum_cross += float((e * ehat).sum())
        self.sum_err2 += float(((e - ehat) ** 2).sum())
        norms = np.linalg.norm(e, axis=1) * np.linalg.norm(ehat, axis=1)
        self.sum_pixel_cos += float(((e * ehat).sum(1) / np.maximum(norms, EPS)).sum())
        self.sum_l0 += float(active.sum())

        self.n_f += ea.sum(0)
        self.sum_a += a.sum(0)
        self.sum_a2 += (a * a).sum(0)
        self.coact += ea.T @ ea

        if concept_mask is None:
            return
        cm = np.asarray(concept_mask, dtype=bool)
        pv = np.asarray(product_valid, dtype=bool)
        if cm.shape != (n, len(self.concept_names)) or pv.shape != (n, len(self.product_names)):
            raise ValueError(
                f"concept_mask must be (n, {len(self.concept_names)}) and product_valid "
                f"(n, {len(self.product_names)}), got {cm.shape} and {pv.shape}"
            )
        self.n_fc += ea.T @ cm.astype(np.float64)
        self.n_c += cm.sum(0)
        self.n_p += pv.sum(0)

        # Unique-argmax coverage's sufficient statistic: the per-pixel argmax
        # feature is a hard, one-vote winner (unlike the multi-hot `active`
        # mask above), undefined where no feature fired at all.
        has_winner = active.any(1)
        if has_winner.any():
            winner = np.argmax(a, axis=1)[has_winner]
            onehot = sparse.csr_matrix(
                (np.ones(winner.size), (winner, np.arange(winner.size))),
                shape=(self.n_features, winner.size),
            )
            self.argmax_n_fc += onehot @ cm[has_winner].astype(np.float64)

        bin_idx = _bin_index(a, self.bin_edges)  # (n, F) in [0, n_bins)
        flat = bin_idx + np.arange(self.n_features)[None, :] * self.n_bins  # (n, F), in [0, F * n_bins)
        # One shared (n, F * n_bins) one-hot incidence, one entry per (pixel, feature).
        z_incidence = sparse.csr_matrix(
            (np.ones(n * self.n_features), (np.repeat(np.arange(n), self.n_features), flat.ravel())),
            shape=(n, self.n_features * self.n_bins),
        )
        n_concept, n_product = len(self.concept_names), len(self.product_names)
        concept_counts = (z_incidence.T @ cm.astype(np.float64)).T.reshape(n_concept, self.n_features, self.n_bins)
        self.concept_hist += concept_counts.transpose(1, 0, 2).astype(np.int32)
        product_counts = (z_incidence.T @ pv.astype(np.float64)).T.reshape(n_product, self.n_features, self.n_bins)
        self.product_hist += product_counts.transpose(1, 0, 2).astype(np.int32)

    # -- merge across batches/shards --------------------------------------

    def merge(self, other: PanelAccumulator) -> PanelAccumulator:
        """Combine with another accumulator sharing this one's bin edges and vocabulary.

        Merging is what makes the accumulator shardable: a large evaluation
        sample can be split across workers or GPUs, each filling its own
        accumulator, and the result summed. Every field here is a plain sum, so
        merging is associative and commutative.

        Args:
            other: Another accumulator, from the same :meth:`create` call
                (same shapes, same ``bin_edges``, same concept/product order).

        Returns:
            A new, merged accumulator; neither input is modified.

        Raises:
            ValueError: The two accumulators are not directly comparable.
        """
        if (
            self.n_features != other.n_features
            or self.embedding_dim != other.embedding_dim
            or self.concept_names != other.concept_names
            or self.product_names != other.product_names
            or not np.array_equal(self.bin_edges, other.bin_edges)
        ):
            raise ValueError("accumulators must share shapes, vocabulary and bin edges to be merged")
        out = PanelAccumulator.create(
            n_features=self.n_features,
            embedding_dim=self.embedding_dim,
            concept_names=self.concept_names,
            product_names=self.product_names,
            product_of_concept=self.product_of_concept,
            bin_edges=self.bin_edges,
        )
        for f in (
            "n_pixels",
            "sum_e",
            "sum_e_outer",
            "sum_ehat",
            "sum_ehat_sq",
            "sum_cross",
            "sum_err2",
            "sum_pixel_cos",
            "sum_l0",
            "n_f",
            "sum_a",
            "sum_a2",
            "coact",
            "n_fc",
            "n_c",
            "n_p",
            "concept_hist",
            "product_hist",
            "argmax_n_fc",
        ):
            setattr(out, f, getattr(self, f) + getattr(other, f))
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Write the accumulator to a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            n_features=self.n_features,
            embedding_dim=self.embedding_dim,
            concept_names=np.array(self.concept_names, dtype=object),
            product_names=np.array(self.product_names, dtype=object),
            product_of_concept=self.product_of_concept,
            n_bins=self.n_bins,
            bin_edges=self.bin_edges,
            n_pixels=self.n_pixels,
            sum_e=self.sum_e,
            sum_e_outer=self.sum_e_outer,
            sum_ehat=self.sum_ehat,
            sum_ehat_sq=self.sum_ehat_sq,
            sum_cross=self.sum_cross,
            sum_err2=self.sum_err2,
            sum_pixel_cos=self.sum_pixel_cos,
            sum_l0=self.sum_l0,
            n_f=self.n_f,
            sum_a=self.sum_a,
            sum_a2=self.sum_a2,
            coact=self.coact,
            n_fc=self.n_fc,
            n_c=self.n_c,
            n_p=self.n_p,
            concept_hist=self.concept_hist,
            product_hist=self.product_hist,
            argmax_n_fc=self.argmax_n_fc,
        )
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> PanelAccumulator:
        """Read an accumulator written by :meth:`save`."""
        with np.load(str(path), allow_pickle=True) as z:
            kw = {k: z[k] for k in z.files}
        kw["n_features"] = int(kw["n_features"])
        kw["embedding_dim"] = int(kw["embedding_dim"])
        kw["n_bins"] = int(kw["n_bins"])
        kw["n_pixels"] = int(kw["n_pixels"])
        kw["sum_ehat_sq"] = float(kw["sum_ehat_sq"])
        kw["sum_cross"] = float(kw["sum_cross"])
        kw["sum_err2"] = float(kw["sum_err2"])
        kw["sum_pixel_cos"] = float(kw.get("sum_pixel_cos", np.nan))
        kw["sum_l0"] = float(kw["sum_l0"])
        kw["concept_names"] = [str(s) for s in kw["concept_names"]]
        kw["product_names"] = [str(s) for s in kw["product_names"]]
        return cls(**kw)


def merge_accumulators(accumulators: Sequence[PanelAccumulator]) -> PanelAccumulator:
    """Reduce a sequence of shard accumulators to one, via repeated :meth:`PanelAccumulator.merge`.

    Raises:
        ValueError: ``accumulators`` is empty.
    """
    accs = list(accumulators)
    if not accs:
        raise ValueError("need at least one accumulator to merge")
    out = accs[0]
    for other in accs[1:]:
        out = out.merge(other)
    return out


# ---------------------------------------------------------------------------
# Bin calibration
# ---------------------------------------------------------------------------


def calibrate_activation_bins(codes_sample: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-feature interior quantile edges of nonzero activation, for the AUROC histogram.

    Args:
        codes_sample: ``(n, F)`` dense activations from a representative
            calibration sample (need not be the whole evaluation sample).
        n_bins: Total bins per feature, including the reserved inactive bin.

    Returns:
        ``(F, n_bins - 1)`` edges, strictly positive and non-decreasing per
        row. A feature with fewer than ``n_bins - 1`` distinct nonzero values
        falls back to an evenly spaced range over its observed nonzero span
        (or ``[EPS, 1]`` if it never fired in the calibration sample), so
        every feature's edges are well-defined even when nearly dead.
    """
    codes = np.asarray(codes_sample, dtype=np.float64)
    n_feat = codes.shape[1]
    n_edges = int(n_bins) - 1
    edges = np.empty((n_feat, n_edges), dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_edges + 2)[1:-1]  # interior quantiles only
    for f in range(n_feat):
        vals = codes[:, f]
        vals = vals[vals > 0]
        if vals.size >= 2:
            e = np.quantile(vals, qs)
            e = np.maximum.accumulate(e)  # ties collapse to a non-decreasing sequence
            lo = e[0] if e[0] > 0 else EPS
            e = np.where(e > 0, e, lo)
        else:
            hi = float(vals.max()) if vals.size else 1.0
            e = np.linspace(max(hi, EPS) / (n_edges + 1), max(hi, EPS), n_edges)
        edges[f] = e
    return edges


def _bin_index(codes: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """``(n, F)`` bin index of each activation, per :attr:`PanelAccumulator.bin_edges`.

    Every feature has its own sorted edge row, so :func:`numpy.searchsorted` cannot batch across
    them; :func:`torch.searchsorted` with a multi-row ``sorted_sequence`` searches each row of
    ``input`` against its own row of edges in one call.
    """
    import torch

    idx = torch.searchsorted(
        torch.from_numpy(np.ascontiguousarray(bin_edges)),
        torch.from_numpy(np.ascontiguousarray(codes.T)),
        right=True,
    )
    return idx.numpy().T.astype(np.int64)


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def fidelity_metrics(acc: PanelAccumulator) -> dict[str, float]:
    """FVE, MSE, centred and per-pixel reconstruction cosine, mean :math:`\\ell_0` and dead-atom fraction.

    Every scalar folds out of :attr:`PanelAccumulator.sum_e`,
    ``sum_e_outer``, ``sum_ehat``, ``sum_ehat_sq``, ``sum_cross``,
    ``sum_err2``, ``sum_l0`` and ``n_f`` — no second pass.

    Returns:
        ``fve``, ``recon_cosine`` (both ``nan`` if the embedding has no
        variance), ``recon_mse`` (per embedding dimension), ``pixel_cosine``
        (the mean over pixels of the cosine between embedding and
        reconstruction), ``l0``, ``dead_frac``.
    """
    n = max(int(acc.n_pixels), 1)
    mean_e = acc.sum_e / n
    mean_ehat = acc.sum_ehat / n
    dev2 = float(np.trace(acc.sum_e_outer) - n * float(mean_e @ mean_e))
    ehat_dev2 = float(acc.sum_ehat_sq - n * float(mean_ehat @ mean_ehat))
    cross_dev = float(acc.sum_cross - n * float(mean_e @ mean_ehat))
    fve = float("nan") if dev2 <= EPS else 1.0 - float(acc.sum_err2) / dev2
    denom = np.sqrt(max(dev2, 0.0) * max(ehat_dev2, 0.0))
    recon_cosine = float("nan") if denom <= EPS else cross_dev / denom
    return {
        "fve": fve,
        "recon_cosine": recon_cosine,
        "recon_mse": float(acc.sum_err2) / (n * acc.embedding_dim),
        "pixel_cosine": float(acc.sum_pixel_cos) / n,
        "l0": float(acc.sum_l0) / n,
        "dead_frac": float(np.mean(acc.n_f == 0)),
    }


# ---------------------------------------------------------------------------
# Dictionary geometry
# ---------------------------------------------------------------------------


def dictionary_geometry_metrics(
    decoder: np.ndarray, acc: PanelAccumulator, *, dup_thresh: float = 0.9
) -> dict[str, float]:
    """Coherence, near-duplicates, dead fraction and the activation-rate tail index.

    Free: the decoder is a small, non-data-dependent :math:`(F, D)` matrix,
    and ``n_f`` is already in the accumulator.

    Args:
        decoder: ``(F, D)`` decoder atoms.
        acc: The accumulator (needs ``n_f``, ``n_pixels``).
        dup_thresh: Off-diagonal cosine above which two atoms count as
            near-duplicates.

    Returns:
        ``max_offdiag_coherence``, ``near_duplicate_count``, ``dead_frac``,
        ``rate_tail_index`` (Hill estimator on the alive features' activation
        rate; ``nan`` with fewer than two alive features).
    """
    d = _unit_rows(decoder)
    gram = d @ d.T
    np.fill_diagonal(gram, 0.0)
    off = np.abs(gram)
    dead = acc.n_f == 0
    rate = acc.n_f / max(int(acc.n_pixels), 1)
    alive_rate = np.sort(rate[~dead])[::-1]
    tail = float("nan")
    if alive_rate.size >= 2:
        k = max(1, int(0.1 * alive_rate.size))
        tail_sample = alive_rate[: k + 1]
        logs = np.log(np.maximum(tail_sample, EPS))
        tail = float(1.0 / max(np.mean(logs[:-1] - logs[-1]), EPS))
    return {
        "max_offdiag_coherence": float(off.max()) if off.size else float("nan"),
        "near_duplicate_count": float(np.sum(off > dup_thresh) // 2),
        "dead_frac": float(np.mean(dead)),
        "rate_tail_index": tail,
    }


# ---------------------------------------------------------------------------
# Known concepts: contingency and histogram AUROC
# ---------------------------------------------------------------------------


def contingency_tables(acc: PanelAccumulator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(precision (F, C), support (F,), prevalence (C,))`` from the accumulator's contingency counts.

    ``precision[f, c] = n_fc[f, c] / n_f[f]`` is the share of feature ``f``'s active pixels inside
    concept ``c``; ``prevalence[c]`` is concept ``c``'s share of the labelled pixels (the largest
    per-product labelled-pixel count).
    """
    support = np.asarray(acc.n_f, dtype=np.float64)
    precision = np.asarray(acc.n_fc, dtype=np.float64) / np.where(support > 0, support, 1.0)[:, None]
    n_valid = float(np.max(acc.n_p)) if acc.n_p.size else 0.0
    prevalence = np.asarray(acc.n_c, dtype=np.float64) / max(n_valid, 1.0)
    return precision, support, prevalence


def _negative_histogram(acc: PanelAccumulator) -> np.ndarray:
    """``(F, C, n_bins)`` histogram of each concept's negative population.

    ``product_hist`` for a concept's own product minus that concept's own
    positive histogram — the entire reason ``product_hist`` is kept at
    :math:`P` columns instead of a second :math:`(F, C, B)` array.
    """
    return acc.product_hist[:, acc.product_of_concept, :] - acc.concept_hist


def auc_from_histograms(pos_hist: np.ndarray, neg_hist: np.ndarray) -> np.ndarray:
    """Folded AUROC from paired binned histograms, exact to bin resolution.

    The tie-corrected Mann-Whitney rank-sum statistic computed directly from
    per-bin counts: every value inside a bin is treated as tied (average
    rank), which is exactly what the exact estimator does for genuine ties and
    an approximation for values a coarse bin has merged.

    Args:
        pos_hist: ``(..., B)`` positive-class per-bin counts, bins ordered by
            ascending activation.
        neg_hist: ``(..., B)`` negative-class per-bin counts, same bins.

    Returns:
        ``(...,)`` folded AUROC in ``[0.5, 1.0]``; ``0.5`` wherever a side is
        empty.
    """
    pos = np.asarray(pos_hist, dtype=np.float64)
    neg = np.asarray(neg_hist, dtype=np.float64)
    total = pos + neg
    below = np.cumsum(total, axis=-1) - total  # pixels strictly in earlier bins
    avg_rank = below + (total + 1.0) / 2.0
    rank_sum_pos = np.sum(pos * avg_rank, axis=-1)
    n_pos = pos.sum(axis=-1)
    n_neg = neg.sum(axis=-1)
    denom = n_pos * n_neg
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / denom
    auc = np.where(denom > 0, auc, 0.5)
    return np.maximum(auc, 1.0 - auc)


def auroc_matrix_from_accumulator(acc: PanelAccumulator) -> np.ndarray:
    """``(F, C)`` folded AUROC matrix, closed-form in the accumulator's histograms."""
    return auc_from_histograms(acc.concept_hist, _negative_histogram(acc))


def unique_argmax_coverage(acc: PanelAccumulator) -> dict:
    """Unique-argmax coverage: a hard, one-vote-per-pixel concept count.

    Every other coverage number (:func:`auroc_matrix_from_accumulator`
    → UCov, EMF) is threshold- or rank-based and lets several features share
    credit for one pixel. Unique-argmax coverage instead gives each pixel a
    single vote — the feature with the highest activation — and asks how many
    concepts win at least one feature's vote: assign each feature to the
    concept its argmax-winning pixels agree on most (:attr:`PanelAccumulator.argmax_n_fc`
    divided by :attr:`PanelAccumulator.n_c`, its own win count as denominator
    would double-count a feature that wins many pixels of a rare concept), and
    count the distinct concepts that are some feature's top pick.

    Args:
        acc: The accumulator (needs ``argmax_n_fc``, ``n_c``).

    Returns:
        ``coverage`` (int count of concepts that are some feature's top argmax
        pick), ``best_concept`` ``(F,)`` int index into ``concept_names`` (``-1``
        if the feature never won a labelled pixel), ``win_rate`` ``(F,)`` share
        of that best concept's positive population the feature's argmax wins
        claimed.
    """
    n_c = np.maximum(acc.n_c, EPS)
    rate = acc.argmax_n_fc / n_c[None, :]  # (F, C): share of each concept's own pixels this feature won
    total_wins = acc.argmax_n_fc.sum(axis=1)
    best_concept = np.where(total_wins > 0, np.argmax(acc.argmax_n_fc, axis=1), -1)
    win_rate = np.where(total_wins > 0, rate[np.arange(acc.n_features), np.clip(best_concept, 0, None)], np.nan)
    covered = np.unique(best_concept[best_concept >= 0])
    return {
        "coverage": int(covered.size),
        "best_concept": best_concept,
        "win_rate": win_rate,
    }


__all__ = [
    "PanelAccumulator",
    "merge_accumulators",
    "calibrate_activation_bins",
    "fidelity_metrics",
    "dictionary_geometry_metrics",
    "contingency_tables",
    "auc_from_histograms",
    "auroc_matrix_from_accumulator",
    "unique_argmax_coverage",
]


def _unit_rows(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(norm, EPS)

"""Concept-alignment AUROC and patch-mean Moran's I from a predictions zarr.

:func:`concept_auroc` scores how well one feature separates the pixels of a
concept from all other labelled pixels, and reports the UCov-style fraction of
features clearing an AUROC threshold; :func:`patch_mean_moran` scores the
spatial autocorrelation of each feature's patch-mean activation field.

A random unit-norm overcomplete dictionary attains both — it still linearly
separates concepts, and its patch-mean fields are still spatially smooth — so
neither scalar alone separates a trained decoder from a frozen random one. Read
them against the frozen-decoder baseline (``model.freeze_decoder: true``).

Both are computed on a sample of test patches for tractability.
"""

from __future__ import annotations

import numpy as np
import zarr

from geosae.eval.concepts import CONCEPT_GROUPS, _product_source


def _densify_patch(zi: np.ndarray, zv: np.ndarray, n_feat: int) -> np.ndarray:
    P, K = zi.shape
    dense = np.zeros((P, n_feat), dtype=np.float32)
    valid = zv != 0
    rows = np.repeat(np.arange(P), K).reshape(P, K)
    dense[rows[valid], zi[valid].astype(np.int64)] = zv[valid]
    return dense


def _sample_dense_and_masks(
    pred_path: str,
    data_zarr: str,
    *,
    groups="headline",
    n_patches: int = 120,
    seed: int = 0,
    patches: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Sample patches → ``(codes (n_px, F), masks (n_px, C) bool, names, patch ids)``.

    Only labelled pixels (≥ one product valid) are kept. ``patches`` names the
    patches explicitly — from
    :meth:`~geosae.eval.pool.EvalPool.prediction_indices` — instead of drawing
    ``n_patches`` of them at random.
    """
    from geosae.eval.concepts import HEADLINE_GROUPS

    gmap = (
        {g: CONCEPT_GROUPS[g] for g in HEADLINE_GROUPS}
        if groups == "headline"
        else (CONCEPT_GROUPS if groups == "all" else {g: CONCEPT_GROUPS[g] for g in groups})
    )
    names = list(gmap)
    members = {c: [(*_product_source(p), code) for p, code in gmap[c]] for c in names}
    keys = sorted({src for c in names for src, _, _ in members[c]})

    pred = zarr.open_group(str(pred_path), mode="r")
    data = zarr.open_group(str(data_zarr), mode="r")
    F = int(pred.attrs["latent_dim"])
    ps = int(pred.attrs["patch_size"])
    cell = np.asarray(pred["cell"])
    if patches is None:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(cell), size=min(n_patches, len(cell)), replace=False))
    else:
        sel = np.sort(np.asarray(patches, dtype=np.int64))

    codes_list, mask_list = [], []
    for p in sel:
        c = int(cell[p])
        zi = np.asarray(pred["z_indices"][p]).reshape(ps * ps, -1)
        zv = np.asarray(pred["z_values"][p]).reshape(ps * ps, -1)
        dense = _densify_patch(zi, zv, F)
        raw = {k: np.asarray(data[k][c]) for k in keys}
        valid = np.zeros(ps * ps, dtype=bool)
        gm = np.zeros((ps * ps, len(names)), dtype=bool)
        for ci, cn in enumerate(names):
            m = np.zeros(ps * ps, dtype=bool)
            for key, band, code in members[cn]:
                arr = raw[key][..., band] if band is not None else raw[key]
                arr = arr.reshape(ps * ps)
                m |= arr == code
                valid |= arr != 0
            gm[:, ci] = m
        codes_list.append(dense[valid])
        mask_list.append(gm[valid])
    return np.concatenate(codes_list), np.concatenate(mask_list), names, sel


def concept_auroc(
    pred_path: str,
    data_zarr: str,
    *,
    groups="headline",
    n_patches: int = 120,
    seed: int = 0,
    auroc_thresh: float = 0.65,
    patches: np.ndarray | None = None,
) -> dict:
    """Best-per-feature concept-alignment AUROC.

    AUROC(f, c) = P(activation_f(pos) > activation_f(neg)) via mean-rank
    (Mann-Whitney U), computed once over a labelled-pixel sample for all
    (feature, concept) pairs. Returns the median best-per-feature AUROC and the
    "UCov"-style fraction of features whose best AUROC clears ``auroc_thresh``,
    alongside ``n_patches`` — the population the number rests on.

    Pass ``patches`` (from
    :meth:`~geosae.eval.pool.EvalPool.prediction_indices`) to score the frozen
    evaluation pool's cells rather than a draw of this function's own.
    """
    from scipy.stats import rankdata

    codes, masks, names, sel = _sample_dense_and_masks(
        pred_path, data_zarr, groups=groups, n_patches=n_patches, seed=seed, patches=patches
    )
    n, F = codes.shape
    C = masks.shape[1]
    n_pos = masks.sum(0).astype(np.float64)  # (C,)
    # Average ranks per feature (handles the many tied zeros correctly).
    ranks = np.empty_like(codes, dtype=np.float64)
    for f in range(F):
        ranks[:, f] = rankdata(codes[:, f])
    pos_rank_sum = masks.astype(np.float64).T @ ranks  # (C, F)
    n_neg = n - n_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        auroc = (pos_rank_sum - n_pos[:, None] * (n_pos[:, None] + 1) / 2) / (n_pos[:, None] * n_neg[:, None])
    auroc = np.where((n_pos[:, None] > 0) & (n_neg[:, None] > 0), auroc, np.nan)  # (C, F)
    best = np.nanmax(auroc, axis=0)  # (F,) best concept AUROC per feature
    best = best[np.isfinite(best)]
    return {
        "auroc_median": float(np.median(best)) if best.size else float("nan"),
        "auroc_mean": float(np.mean(best)) if best.size else float("nan"),
        "ucov_frac_ge_thresh": float(np.mean(best >= auroc_thresh)) if best.size else float("nan"),
        "n_features_scored": float(best.size),
        "n_concepts": float(C),
        "auroc_thresh": auroc_thresh,
        "n_patches": float(sel.size),
    }


def patch_mean_moran(
    pred_path: str,
    *,
    k_neighbors: int = 8,
    max_patches: int | None = 20000,
    seed: int = 0,
    patches: np.ndarray | None = None,
) -> dict:
    """Patch-mean Moran's I per feature (median), via a k-NN spatial weight.

    Each feature's per-patch mean activation (``feat_patch_stats[..., 0]``) is a
    field over patch centroids ``(lat, lon)``; Moran's I measures its spatial
    autocorrelation under a row-normalised k-nearest-neighbour (haversine) weight.

    Pass ``patches`` (from
    :meth:`~geosae.eval.pool.EvalPool.prediction_indices`) to compute it over the
    frozen evaluation pool's cells rather than a draw of this function's own.
    This is a patch-level statistic — a patch mean is over *all* of a patch's
    pixels, so a pool contributes its cells here but not its pixel positions.
    """
    pred = zarr.open_group(str(pred_path), mode="r")
    lon = np.asarray(pred["lon"], dtype=np.float64)
    lat = np.asarray(pred["lat"], dtype=np.float64)
    stats = np.asarray(pred["feat_patch_stats"])  # (F, N, 2)
    N = lon.shape[0]
    idx = np.arange(N)
    if patches is not None:
        idx = np.sort(np.asarray(patches, dtype=np.int64))
    elif max_patches is not None and N > max_patches:
        idx = np.sort(np.random.default_rng(seed).choice(N, size=max_patches, replace=False))
    return patch_mean_moran_from_arrays(stats[:, idx, 0], lon[idx], lat[idx], k_neighbors=k_neighbors)


def patch_mean_moran_from_arrays(
    patch_means: np.ndarray, lon: np.ndarray, lat: np.ndarray, *, k_neighbors: int = 8, per_feature: bool = False
) -> dict:
    """Patch-mean Moran's I per feature (median), from arrays already in hand.

    The reduction :func:`patch_mean_moran` performs, split out so a caller that
    computed per-patch mean activations itself — a run scored without a stored
    predictions zarr — does not have to write the codes to disk first.

    Args:
        patch_means: ``(F, n_patches)`` mean activation of each feature per patch.
        lon: ``(n_patches,)`` patch-centre longitudes, degrees.
        lat: ``(n_patches,)`` patch-centre latitudes, degrees.
        k_neighbors: Neighbours in the row-normalised k-NN spatial weight.
        per_feature: Also return every feature's value.

    Returns:
        ``moran_i`` (median over features), ``moran_i_mean``,
        ``n_features_scored``, ``k_neighbors`` and ``n_patches``; with
        ``per_feature``, also ``moran_per_feature``: the ``(F,)`` values in input
        order, ``nan`` for a feature constant over the patches.
    """
    from geosae.eval.spatial import knn_neighbours, moran_i

    means = np.atleast_2d(np.asarray(patch_means, dtype=np.float64))
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    n = lon.size

    nbr = knn_neighbours(lon, lat, k_neighbors)
    kk = nbr.shape[1]
    finite = np.isfinite(means).all(axis=1)
    values = np.full(means.shape[0], np.nan)
    if finite.any():
        values[finite] = moran_i(means[finite], nbr)
    moran = values[np.isfinite(values)]
    out = {
        "moran_i": float(np.median(moran)) if moran.size else float("nan"),
        "moran_i_mean": float(np.mean(moran)) if moran.size else float("nan"),
        "n_features_scored": float(moran.size),
        "k_neighbors": float(kk),
        "n_patches": float(n),
    }
    if per_feature:
        out["moran_per_feature"] = values
    return out


__all__ = ["concept_auroc", "patch_mean_moran", "patch_mean_moran_from_arrays"]

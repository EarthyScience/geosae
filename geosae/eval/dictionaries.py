"""Conventional decompositions as dictionaries, scored like a trained SAE.

Every panel metric is a statement about a *dictionary* — ``M`` unit-norm atoms plus a
code — so any estimator that produces one can be scored on the same pixels as an SAE.
This module fits the conventional ones: PCA, its varimax rotation, ICA, NMF,
``k``-means, and a random draw, and reads the embedding's own axes (``raw``) as a
dictionary of the same kind.

Three things every method supplies, and they are what the interface is:

* **Unit-norm atoms.** Where they come from differs per method — a rotation's atoms
  are the normalised columns of its loading matrix, ``k``-means' are its centroids
  — and :func:`fit_dictionary` states it per method.
* **A sign.** PCA, varimax, ICA and the raw axes leave each atom's sign free, so each
  is flipped to make its own coefficient distribution right-skewed, the sign a sparse
  non-negative code implies. NMF and ``k``-means fix their own sign by construction;
  the random draw is left untouched, so it stays a data-free floor.
* **Codes.** Each estimator's own code is :meth:`DictionaryFit.encode`: a dense
  projection for a basis, a one-hot assignment for ``k``-means. A dense basis has
  no active set; :func:`rate_matched_support` gives it one by keeping each
  dimension's largest coefficients at a stated activation rate — the same rate the
  SAE it is compared against achieved. :func:`pursuit_codes` codes any dictionary
  by a non-negative greedy pursuit at a fixed number of atoms per pixel, which
  separates a *coding rule* from the dictionary it is applied to.

A dense linear basis cannot carry more atoms than the embedding has dimensions, so
the raw axes, PCA, varimax and ICA saturate at ``M = D`` and report an ``n_atoms`` below the width
asked for. NMF, ``k``-means and the random draw are overcomplete at any width.

``ica`` and ``nmf`` use scikit-learn; the others are numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12

#: The estimators :func:`fit_dictionary` knows.
DICTIONARY_METHODS = ("raw", "pca", "varimax", "ica", "nmf", "kmeans", "random")

#: Methods whose atom count is capped by the embedding dimension.
DENSE_LINEAR_METHODS = ("raw", "pca", "varimax", "ica")


@dataclass(frozen=True)
class DictionaryFit:
    """A fitted dictionary and the encoder that produced it.

    Attributes:
        method: The estimator, one of :data:`DICTIONARY_METHODS`.
        atoms: ``(M, D)`` unit-norm atoms, one per row.
        offset: ``(D,)`` vector subtracted from a pixel before encoding.
        n_requested: The width asked for, which ``n_atoms`` may fall short of.
        sparse: Whether the codes are already sparse, so that rate matching does
            not apply. True for ``k``-means, whose code has ``l0 = 1`` by
            construction.
    """

    method: str
    atoms: np.ndarray
    offset: np.ndarray
    n_requested: int
    sparse: bool
    _encoder: np.ndarray | None
    _centroids: np.ndarray | None

    @property
    def n_atoms(self) -> int:
        """``M``, the number of atoms actually fitted."""
        return int(self.atoms.shape[0])

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Coefficients of ``(N, D)`` pixels on this dictionary, as ``(N, M)``.

        A linear basis returns its (signed, dense) projection; ``k``-means returns
        a one-hot code carrying the winning centroid's non-negative similarity.
        Neither is thresholded — see :func:`rate_matched_support`.
        """
        xc = np.asarray(x, dtype=np.float64) - self.offset
        if self._centroids is not None:
            d2 = (xc**2).sum(1, keepdims=True) - 2.0 * xc @ self._centroids.T + (self._centroids**2).sum(1)
            win = np.argmin(d2, axis=1)
            z = np.zeros((xc.shape[0], self.n_atoms), dtype=np.float64)
            rows = np.arange(xc.shape[0])
            z[rows, win] = np.maximum(np.einsum("nd,nd->n", xc, self.atoms[win]), 0.0)
            return z
        return xc @ self._encoder


def rate_matched_support(codes: np.ndarray, rate: float | np.ndarray) -> np.ndarray:
    """Threshold each column of ``(N, M)`` codes to a given activation rate.

    A dense basis has no active set, so a code-side diagnostic that asks "where does
    this atom fire" has no answer until one is chosen. The choice made here is the
    one that removes sparsity as a confound: keep each dimension's largest
    coefficients until it fires at exactly the rate the estimator it is being
    compared against does, so no method can win on support overlap merely by being
    denser.

    Args:
        codes: ``(N, M)`` coefficients.
        rate: Target per-column activation rate in ``[0, 1]``, one value shared by
            every column or ``(M,)`` values.

    Returns:
        ``(N, M)`` codes with the surviving coefficients at their original values
        and everything else zero.

    Raises:
        ValueError: If a rate falls outside ``[0, 1]`` or the shapes disagree.
    """
    z = np.asarray(codes, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"codes must be (n_pixels, n_atoms), got {z.shape}")
    r = np.broadcast_to(np.asarray(rate, dtype=np.float64), (z.shape[1],)).copy()
    if np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError(f"rate must lie in [0, 1], got {np.min(r)}..{np.max(r)}")
    n = z.shape[0]
    keep = np.rint(r * n).astype(np.int64)
    out = np.zeros_like(z)
    order = np.argsort(-z, axis=0, kind="stable")
    for j in range(z.shape[1]):
        idx = order[: keep[j], j]
        out[idx, j] = z[idx, j]
    return out


def topk_projection_code(fit: DictionaryFit, x: np.ndarray, k: int) -> np.ndarray:
    """Sparse code of the ``k`` largest positive projections of each pixel onto a dictionary's atoms.

    With a ``random`` fit this is an untrained random sparse code: the centred pixel is
    projected onto ``M`` random unit-norm directions and every coefficient but the
    ``k`` largest positive ones is set to zero.

    Args:
        fit: A fitted dictionary; its ``offset`` is subtracted first.
        x: ``(N, D)`` pixels.
        k: Active coefficients per pixel (fewer where fewer projections are positive).

    Returns:
        ``(N, M)`` non-negative codes.
    """
    z = np.maximum((np.asarray(x, dtype=np.float64) - fit.offset) @ fit.atoms.T, 0.0)
    k = int(k)
    if k < z.shape[1]:
        drop = np.argpartition(-z, k, axis=1)[:, k:]
        np.put_along_axis(z, drop, 0.0, axis=1)
    return z


def fit_dictionary(x: np.ndarray, method: str, n_atoms: int, *, seed: int = 0, max_iter: int = 200) -> DictionaryFit:
    """Fit ``n_atoms`` unit-norm atoms to ``(N, D)`` pixels with a conventional estimator.

    ============  ===================================  ====================
    ``method``    atoms are                            overcomplete?
    ============  ===================================  ====================
    ``raw``       the embedding's own coordinate axes  no, capped at ``D``
                  (below ``D``, those of largest
                  variance, in their own order)
    ``pca``       the leading right singular vectors   no, capped at ``D``
    ``varimax``   the varimax-rotated PCA loadings     no, capped at ``D``
    ``ica``       the columns of FastICA's mixing      no, capped at ``D``
    ``nmf``       the rows of NMF's component matrix   yes
    ``kmeans``    the ``k``-means centroids            yes
    ``random``    an isotropic Gaussian draw           yes
    ============  ===================================  ====================

    Every method centres the data on its own mean first, except ``nmf``, which shifts
    to the non-negative orthant instead (its offset is the per-dimension minimum of
    the fitting sample), and ``random``, which never looks at the data at all.

    Args:
        x: ``(N, D)`` pixels to fit on.
        method: One of :data:`DICTIONARY_METHODS`.
        n_atoms: The width to fit. For :data:`DENSE_LINEAR_METHODS` the result is
            capped at ``D`` and :attr:`DictionaryFit.n_atoms` says so.
        seed: Seed for every stochastic part of the fit — the random draw, the
            ``k``-means initialisation, ICA's and NMF's initialisations.
        max_iter: Iteration cap for the iterative estimators.

    Returns:
        The :class:`DictionaryFit`.

    Raises:
        ValueError: If ``method`` is unknown or ``n_atoms`` is below 1.
        ImportError: If ``method`` needs scikit-learn and it is not installed.
    """
    if method not in DICTIONARY_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {DICTIONARY_METHODS}")
    if int(n_atoms) < 1:
        raise ValueError(f"n_atoms must be at least 1, got {n_atoms}")
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"x must be (n_pixels, n_dims), got {data.shape}")
    requested = int(n_atoms)
    d = data.shape[1]
    width = min(requested, d) if method in DENSE_LINEAR_METHODS else requested

    if method == "raw":
        offset = data.mean(axis=0)
        atoms = np.eye(d)[np.sort(np.argsort(-data.var(axis=0), kind="stable")[:width])]
    elif method == "random":
        rng = np.random.default_rng(seed)
        atoms = _unit_rows(rng.normal(size=(width, d)))
        offset = data.mean(axis=0)
    elif method == "pca":
        offset = data.mean(axis=0)
        atoms = _principal_axes(data - offset, width)
    elif method == "varimax":
        offset = data.mean(axis=0)
        atoms = _unit_rows(_varimax(_loadings(data - offset, width), max_iter=max_iter).T)
    elif method == "ica":
        offset = data.mean(axis=0)
        atoms = _unit_rows(_ica_mixing(data - offset, width, seed=seed, max_iter=max_iter).T)
    elif method == "nmf":
        offset = data.min(axis=0)
        atoms = _unit_rows(_nmf_components(data - offset, width, seed=seed, max_iter=max_iter))
    else:  # kmeans
        offset = data.mean(axis=0)
        centroids = _kmeans(data - offset, width, seed=seed, max_iter=max_iter)
        atoms = _unit_rows(centroids)
        return DictionaryFit(
            method=method,
            atoms=atoms,
            offset=offset,
            n_requested=requested,
            sparse=True,
            _encoder=None,
            _centroids=centroids,
        )

    encoder = np.linalg.pinv(atoms)  # (D, M): least-squares coefficients on the atoms
    if method in DENSE_LINEAR_METHODS:
        atoms, encoder = _orient(atoms, encoder, data - offset)
    return DictionaryFit(
        method=method,
        atoms=atoms,
        offset=offset,
        n_requested=requested,
        sparse=False,
        _encoder=encoder,
        _centroids=None,
    )


def pursuit_codes(
    x: np.ndarray,
    atoms: np.ndarray,
    k: int,
    *,
    offset: np.ndarray | None = None,
    chunk: int = 16_384,
) -> np.ndarray:
    """Code pixels on a dictionary with at most ``k`` non-negative atoms each.

    A non-negative orthogonal matching pursuit: at each of ``k`` steps a pixel adds
    the unused atom most positively aligned with its residual, and the coefficients
    of its support are refitted by least squares and clipped at zero. A step that
    would raise the pixel's residual is not taken, so the error never grows with
    ``k``. ``k = 1`` is ``k``-means' coding rule — the single best-aligned atom
    at its projection — and a larger ``k`` is a sparse coder's.

    Args:
        x: ``(N, D)`` pixels.
        atoms: ``(M, D)`` dictionary; rows are unit-normalised first.
        k: Atoms per pixel, clipped to ``M``.
        offset: ``(D,)`` subtracted from every pixel first. Defaults to zero.
        chunk: Pixels coded at once.

    Returns:
        ``(N, M)`` non-negative codes with at most ``k`` non-zeros per row.

    Raises:
        ValueError: ``k`` is below 1, or the shapes disagree.
    """
    if int(k) < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    a = _unit_rows(atoms)
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != a.shape[1]:
        raise ValueError(f"x must be (n_pixels, {a.shape[1]}), got {data.shape}")
    if offset is not None:
        data = data - np.asarray(offset, dtype=np.float64)
    steps = min(int(k), a.shape[0])
    out = np.zeros((data.shape[0], a.shape[0]), dtype=np.float64)
    for start in range(0, data.shape[0], int(chunk)):
        xb = data[start : start + int(chunk)]
        out[start : start + len(xb)] = _pursuit_chunk(xb, a, steps)
    return out


# ------------------------------------------------------------------ internals


def _pursuit_chunk(x: np.ndarray, a: np.ndarray, steps: int) -> np.ndarray:
    """Non-negative OMP of one chunk; see :func:`pursuit_codes`."""
    n = x.shape[0]
    rows = np.arange(n)
    support = np.zeros((n, steps), dtype=np.int64)
    coef = np.zeros((n, steps))
    active = np.zeros((n, steps), dtype=bool)
    residual = x.copy()
    err = (residual**2).sum(axis=1)
    used = np.zeros((n, a.shape[0]), dtype=bool)
    for t in range(steps):
        score = residual @ a.T
        score[used] = -np.inf
        pick = score.argmax(axis=1)
        grow = score[rows, pick] > EPS
        cand_support = support.copy()
        cand_support[:, t] = pick
        cand_active = active.copy()
        cand_active[:, t] = grow
        chosen = a[cand_support[:, : t + 1]] * cand_active[:, : t + 1, None]  # (n, t+1, D)
        # an unused slot is an all-zero atom; a unit diagonal there keeps the solve regular
        idle = (~cand_active[:, : t + 1]).astype(np.float64)
        gram = chosen @ chosen.transpose(0, 2, 1) + idle[:, :, None] * np.eye(t + 1) + 1e-10 * np.eye(t + 1)
        c = np.linalg.solve(gram, (chosen @ x[:, :, None]))[..., 0]
        c = np.clip(c, 0.0, None) * cand_active[:, : t + 1]
        new_res = x - np.einsum("nt,ntd->nd", c, chosen)
        new_err = (new_res**2).sum(axis=1)
        take = grow & (new_err <= err + 1e-12)
        support[take, t] = pick[take]
        active[take, t] = True
        coef[take, : t + 1] = c[take]
        residual[take] = new_res[take]
        err[take] = new_err[take]
        used[rows[take], pick[take]] = True
    z = np.zeros((n, a.shape[0]))
    np.add.at(z, (np.repeat(rows, steps), support.ravel()), (coef * active).ravel())
    return z


def _unit_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)


def _principal_axes(xc: np.ndarray, width: int) -> np.ndarray:
    """``(width, D)`` leading right singular vectors of centred pixels."""
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return _unit_rows(vt[:width])


def _loadings(xc: np.ndarray, width: int) -> np.ndarray:
    """``(D, width)`` PCA loading matrix — the axes scaled by their singular values.

    Varimax rotates *loadings*, not unit axes: the criterion it maximises is
    scale-dependent, so rotating the unit basis would optimise a different thing.
    """
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[:width].T * s[:width]


def _varimax(loadings: np.ndarray, *, gamma: float = 1.0, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Varimax-rotate a ``(D, r)`` loading matrix, returning the rotated loadings.

    The standard iterative SVD form: each step aligns the rotation with the cubed
    loadings, which drives each column's mass towards a few large entries.
    """
    d, r = loadings.shape
    if r < 2:
        return loadings.copy()
    rot = np.eye(r)
    last = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rot
        gradient = loadings.T @ (rotated**3 - (gamma / d) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        u, s, vt = np.linalg.svd(gradient, full_matrices=False)
        rot = u @ vt
        total = float(s.sum())
        if total <= last * (1.0 + tol):
            break
        last = total
    return loadings @ rot


def _ica_mixing(xc: np.ndarray, width: int, *, seed: int, max_iter: int) -> np.ndarray:
    """``(D, width)`` FastICA mixing matrix — the directions the sources map to."""
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=width, random_state=seed, max_iter=max_iter, tol=1e-4, whiten="unit-variance")
    ica.fit(xc.astype(np.float32))
    return np.asarray(ica.mixing_, dtype=np.float64)


def _nmf_components(xs: np.ndarray, width: int, *, seed: int, max_iter: int) -> np.ndarray:
    """``(width, D)`` NMF component matrix of shifted, non-negative pixels."""
    from sklearn.decomposition import NMF

    nmf = NMF(
        n_components=width,
        init="nndsvda" if width <= xs.shape[1] else "random",
        random_state=seed,
        max_iter=max_iter,
        tol=1e-4,
    )
    nmf.fit(np.maximum(xs, 0.0).astype(np.float32))
    return np.asarray(nmf.components_, dtype=np.float64)


def _kmeans(xc: np.ndarray, width: int, *, seed: int, max_iter: int) -> np.ndarray:
    """``(width, D)`` Lloyd's-algorithm centroids from a ``k``-means++ start.

    In numpy rather than through scikit-learn, so the one conventional method that is
    genuinely overcomplete needs no further dependency.
    """
    rng = np.random.default_rng(seed)
    n = xc.shape[0]
    centres = np.empty((width, xc.shape[1]), dtype=np.float64)
    centres[0] = xc[rng.integers(n)]
    d2 = ((xc - centres[0]) ** 2).sum(1)
    for j in range(1, width):
        total = float(d2.sum())
        pick = rng.integers(n) if total <= EPS else int(rng.choice(n, p=d2 / total))
        centres[j] = xc[pick]
        d2 = np.minimum(d2, ((xc - centres[j]) ** 2).sum(1))

    sq = (xc**2).sum(1, keepdims=True)
    for _ in range(max_iter):
        assign = np.argmin(sq - 2.0 * xc @ centres.T + (centres**2).sum(1), axis=1)
        counts = np.bincount(assign, minlength=width)
        moved = np.zeros_like(centres)
        np.add.at(moved, assign, xc)
        alive = counts > 0
        moved[alive] /= counts[alive, None]
        # an emptied centroid restarts on the pixel furthest from its own centre,
        # which is what keeps a requested width from silently collapsing
        if not alive.all():
            far = np.argsort(-np.min(sq - 2.0 * xc @ centres.T + (centres**2).sum(1), axis=1))
            moved[~alive] = xc[far[: int((~alive).sum())]]
        if np.allclose(moved, centres, atol=1e-10):
            centres = moved
            break
        centres = moved
    return centres


def _orient(atoms: np.ndarray, encoder: np.ndarray, xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip each atom so its own coefficient distribution is right-skewed.

    A non-negative code needs the estimator to commit to a sign, and the sign a sparse
    non-negative generator implies is the one that puts the coefficient's heavy tail on
    the positive side.
    """
    z = xc @ encoder
    zc = z - z.mean(axis=0)
    skew = (zc**3).mean(axis=0) / np.maximum(zc.std(axis=0) ** 3, EPS)
    sign = np.where(skew < 0.0, -1.0, 1.0)
    return atoms * sign[:, None], encoder * sign


__all__ = [
    "DENSE_LINEAR_METHODS",
    "DICTIONARY_METHODS",
    "DictionaryFit",
    "fit_dictionary",
    "pursuit_codes",
    "rate_matched_support",
    "topk_projection_code",
]

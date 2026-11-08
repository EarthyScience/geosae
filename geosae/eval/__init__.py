"""Evaluation of sparse codes and dictionaries.

Every metric is a function of a decomposition — a trained SAE, its frozen-decoder baseline, or a
conventional dictionary (:mod:`geosae.eval.dictionaries`: raw dimensions, PCA, ICA, NMF) — on a
fixed sample of pixels, so all of them are scored on identical data. The modules group by question:

* **Reconstruction** — :mod:`geosae.eval.metrics` (FVE, reconstruction cosine, L0) and the
  streaming panel of :mod:`geosae.eval.ladder` / :mod:`geosae.eval.blocks` /
  :mod:`geosae.eval.statistic`, which scores a whole ladder of decompositions on one sample.
* **Concept alignment** — the labelled pixel sample of :mod:`geosae.eval.registry` and
  :mod:`geosae.eval.pool`, the class vocabulary and concept groups of :mod:`geosae.eval.concepts`
  and :mod:`geosae.eval.landcover_classes`, and the folded single-feature AUROC, UCov and
  concept-group counts of :mod:`geosae.eval.coverage`; sub-concept discovery in
  :mod:`geosae.eval.subconcepts`; the monosemanticity scores of Pach et al. and Betti et al. in
  :mod:`geosae.eval.monosemanticity`.
* **Spatial scale** — semivariance curves and correlation lengths (:mod:`geosae.eval.variogram`),
  patch-mean Moran's I (:mod:`geosae.eval.auroc_moran`, :mod:`geosae.eval.spatial`).
* **Area of applicability** — :mod:`geosae.eval.aoa`, with graded hold-outs
  (:mod:`geosae.eval.holdouts`), extrapolation checks (:mod:`geosae.eval.extrapolation`),
  decision-level scores (:mod:`geosae.eval.selective`) and continuous covariates
  (:mod:`geosae.eval.covariates`).
* **Conventional baselines** — :mod:`geosae.eval.baselines` (fitting pixels, PCA) and
  :mod:`geosae.eval.dictionaries`.
"""

from geosae.eval.metrics import calibrate_threshold, reconstruction_metrics
from geosae.eval.pool import EvalPool, PoolSpec, load_pool
from geosae.eval.spatial import variogram_halflife

__all__ = [
    "EvalPool",
    "PoolSpec",
    "calibrate_threshold",
    "load_pool",
    "reconstruction_metrics",
    "variogram_halflife",
]

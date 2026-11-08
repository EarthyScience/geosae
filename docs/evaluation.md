# Evaluation

Every metric in `geosae.eval` is a function of a **decomposition** — a trained SAE, its
frozen-decoder baseline, or a conventional dictionary (raw dimensions, PCA, ICA, NMF) — on a fixed
sample of pixels, so all decompositions are scored on identical data. This page covers
reconstruction and the panel, concept alignment, spatial scale, monosemanticity scores,
sub-concepts and the area of applicability, in that order.

| Question | Stage | Module |
|---|---|---|
| Reconstruction at matched sparsity | `geosae predict`, `geosae panel` | `geosae.eval.metrics`, `geosae.eval.ladder` |
| Alignment with reference concepts | `geosae coverage`, `geosae panel` | `geosae.eval.coverage` |
| Spatial scale of each feature | `geosae variogram` | `geosae.eval.variogram`, `geosae.eval.auroc_moran` |
| Monosemanticity scores (MS, GeoMS) | — | `geosae.eval.monosemanticity` |
| Sub-concepts of a parent class | `geosae subconcepts` | `geosae.eval.subconcepts` |
| Area of applicability | `geosae aoa` | `geosae.eval.aoa`, `geosae.eval.holdouts` |

## Comparing against the frozen-decoder baseline

The frozen-decoder baseline is an SAE whose decoder is a fixed random unit-norm dictionary; only
the encoder trains. It is an ordinary run with `model.freeze_decoder: true`:

```bash
geosae train -c batchtopk -o model.freeze_decoder=true -o name=batchtopk_frozen
```

Score it next to the trained run with every stage below. A metric that does not rank the trained
dictionary above this baseline does not measure what the dictionary learned.

## Scoring a ladder with the panel

`geosae panel` scores a whole ladder of decompositions — trained runs, frozen-decoder baselines and
conventional dictionaries — on one labelled evaluation sample, read from the store once. Each row
is scored in the units its own model was trained on (its own `data.transforms`).

```yaml
panel:
  runs: [runs/batchtopk/*, runs/batchtopk_frozen/*]   # run directories or glob patterns
  n_cells: 4000              # cells drawn from the split
  pixels_per_cell: 256       # pixels per cell, at positions shared by every cell
  sample_seed: 0
  split: test
  products: null             # concept-product subset; null = every product in the store
  blocks: [1, 2, 3]
  baselines:
    methods: [raw, pca, ica, nmf]
    widths: [64]
    ranks: [8, 16, 32]       # rank-k truncations onto the k atoms of largest code variance
```

```bash
geosae panel -c batchtopk -c panel.yaml -o name=panel
```

It writes `eval/panel.json` (rows plus provenance) and `eval/panel.csv` of the panel's run
directory. A dense basis is thresholded per atom to fire on `panel.baselines.activation_rate` of
the pixels (default `model.k / width`), so it is compared at the same sparsity. A large ladder can
be scored as several jobs over disjoint `panel.runs` slices and joined with
`-o "panel.merge_shards=[runs/panel_a/*, runs/panel_b/*]"`.

Every scalar belongs to one block:

| Block | Name | Scalars |
|---|---|---|
| 1 | fidelity | `fve`, `recon_cosine`, `l0` |
| 2 | dictionary geometry | `max_offdiag_coherence`, `near_duplicate_count`, `dead_frac`, `rate_tail_index` |
| 3 | known concepts | `ucov`, `ucov_star`, `emf`, `concept_perplexity`, `argmax_coverage`, `ucov_frac_ge_thresh` |

The blocks are pure functions of one streaming accumulator,
[`PanelAccumulator`](api.md#geosae.eval.statistic.PanelAccumulator), filled by
[`fill_panel()`](api.md#geosae.eval.blocks.fill_panel) from any pixel source and any encoder.
Here a PCA basis, thresholded to a 20% firing rate, is scored on synthetic pixels built from six
known directions:

```python
import numpy as np
from geosae.eval.blocks import BLOCK_TITLES, concept_block, fidelity_block, fill_panel, geometry_block
from geosae.eval.dictionaries import fit_dictionary
from geosae.eval.ladder import dictionary_encoder

rng = np.random.default_rng(0)
atoms = np.linalg.qr(rng.normal(size=(8, 8)))[0][:6]          # 6 directions in D = 8
codes = np.abs(rng.normal(size=(5000, 6))) * (rng.uniform(size=(5000, 6)) < 0.2)
x = codes @ atoms + 0.05 * rng.normal(size=(5000, 8))
mask = codes > 0                                               # concept c: atom c is active
valid = np.ones((5000, 1), dtype=bool)

def source():
    for s in range(0, 5000, 1000):
        yield x[s : s + 1000], (mask[s : s + 1000], valid[s : s + 1000])

pca = fit_dictionary(x, "pca", 8, seed=0)
acc, sample = fill_panel(
    source, dictionary_encoder(pca, activation_rate=0.2), pca.atoms, offset=pca.offset,
    concept_names=[f"c{i}" for i in range(6)], product_names=["p"],
    product_of_concept=np.zeros(6, dtype=np.int64), codes_sample_pixels=2000,
)
print(BLOCK_TITLES)
print({k: round(v, 3) for k, v in fidelity_block(acc).items()})
print({k: round(v, 3) for k, v in geometry_block(pca.atoms, acc).items()})
table = np.stack([np.zeros(6, dtype=np.int64), np.arange(6)], axis=1)
out = concept_block(acc, class_table=table, product_order=["p"],
                    class_to_group={("p", c): f"g{c}" for c in range(6)}, codes_sample=sample)
print({k: round(v, 3) for k, v in out.items()})
```

```
{1: 'fidelity', 2: 'dictionary geometry', 3: 'known concepts'}
{'fve': 0.67, 'recon_cosine': 0.88, 'l0': 1.6}
{'max_offdiag_coherence': 0.0, 'near_duplicate_count': 0.0, 'dead_frac': 0.0, 'rate_tail_index': 1000000000000.0}
{'ucov': 6.0, 'ucov_star': 6.0, 'emf': 4.0, 'concept_perplexity': 5.122, 'argmax_coverage': 6.0, 'ucov_frac_ge_thresh': 0.75}
```

The orthonormal PCA atoms have zero coherence, and every planted concept is covered by some
component; the thresholded code reconstructs only two thirds of the variance at `l0 = 1.6`.
In code, [`score_decomposition()`](api.md#geosae.eval.ladder.score_decomposition) runs the same
blocks on a [`LadderSample`](api.md#geosae.eval.ladder.LadderSample), and
[`real_eval_sample()`](api.md#geosae.pipeline.real_eval_sample) draws the sample `geosae panel`
uses, so a script reproduces the stage's numbers exactly.

!!! note

    Block 3 needs at least one reference product in the store. On a store with none, the panel
    drops block 3 and records `concepts_available: false` in its provenance rather than reporting
    `nan`.

## Measuring concept alignment

Concept alignment is scored on the folded single-feature AUROC of every feature against every
reference-product class.

### Scoring a feature against a concept

The measurement rests on one matrix. For every SAE feature $f$ and every registry concept $c$,
[`auc_all_features()`](api.md#geosae.eval.coverage.auc_all_features) computes the
**direction-invariant single-feature AUROC** of $z_f$ as a one-feature classifier of $c$, on a
class-balanced pixel sample.

```python
import numpy as np
from geosae.eval.coverage import auc_all_features

rng = np.random.default_rng(0)
water = rng.random(4000) < 0.3
codes = np.column_stack([water * rng.uniform(0.5, 1.5, 4000), rng.random(4000)])
print(np.round(auc_all_features(codes, water), 3))
```

```
[1.    0.508]
```

The planted water feature scores a perfect 1.0 and the noise feature sits at chance. Direction
invariance means a feature that perfectly *anti*-predicts a concept also scores 1.0, so an AUROC
of 1.0 asserts separation, not presence.

[`concept_auroc_matrix()`](api.md#geosae.eval.coverage.concept_auroc_matrix) does this for every
concept at once, given the per-pixel label arrays, a `(C, 2)` table of `(product index, class
code)` pairs, and the product order.

```python
from geosae.eval.coverage import concept_auroc_matrix

labels = {"lc_esa_wc": np.where(water, 80, 10).astype("int16")}
class_table = np.array([[0, 80], [0, 10]])
print(np.round(concept_auroc_matrix(codes, labels, class_table, ["lc_esa_wc"]), 3))
```

```
[[1.    1.   ]
 [0.502 0.512]]
```

The result is `(F, C)`. Here the two concepts are complementary — class 80 is exactly where
class 10 is not — so the water feature separates both perfectly, which is direction invariance
showing up in a matrix. Every scalar below is a pure reduction of this matrix, so it can be
recomputed at a new threshold without re-reading the codes.

### Counting covered concepts with UCov

**UCov** is the number of registry concepts covered by at least one feature at
`AUROC >= tau`, with `tau = 0.65` by default.

```python
from geosae.eval.coverage import DEFAULT_TAU, ucov

auc = np.array([[0.91, 0.52, 0.60], [0.55, 0.78, 0.51], [0.50, 0.50, 0.50]])
print("tau =", DEFAULT_TAU, "-> UCov =", ucov(auc))
for tau in (0.55, 0.65, 0.75, 0.85):
    print(f"tau={tau}  UCov={ucov(auc, tau=tau)}")
```

```
tau = 0.65 -> UCov = 2
tau=0.55  UCov=3
tau=0.65  UCov=2
tau=0.75  UCov=2
tau=0.85  UCov=1
```

The count is strongly threshold-dependent — three concepts at 0.55, one at 0.85 — which is why
[`threshold_sweep()`](api.md#geosae.eval.coverage.threshold_sweep) recomputes UCov and EMF
across `(0.55, 0.65, 0.75, 0.85)`, so a ranking can be checked against the threshold choice
rather than resting on it.

### Counting monosemantic features with EMF

**EMF** (effective monosemantic features) works over the concept-*group* taxonomy: a feature is
*alive* if it aligns with at least one group, and `n_groups(f)` counts how many distinct groups it
aligns with. EMF is `n_alive / max(mean_n_groups, 1)`, penalising features that are polysemantic
across groups.

```python
from geosae.eval.coverage import emf, emf_simple

n_groups = np.array([1, 1, 3, 0, 2])
print(emf(n_groups))
print(round(emf_simple(n_groups), 4))
```

```
{'n_alive': 4.0, 'mean_n_groups': 1.75, 'emf': 2.2857142857142856}
2.8333
```

Four of the five features are alive, but they average 1.75 groups each, so they are worth 2.29
monosemantic features. [`emf_simple()`](api.md#geosae.eval.coverage.emf_simple) is the
alternative $\sum_f 1/n_\text{groups}(f)$ form, which weights every feature by its own
polysemanticity rather than by the population mean, and so reports a different number (2.83) on
the same input.

[`n_groups_per_feature()`](api.md#geosae.eval.coverage.n_groups_per_feature) produces the input
from the AUROC matrix and a class-to-group mapping;
[`default_class_to_group()`](api.md#geosae.eval.coverage.default_class_to_group) supplies the
mapping of the 20 concept groups (see [Concepts](concepts.md)).

### Scoring concept AUROC and patch-mean Moran's I from predictions

Two drivers read a predictions zarr (see [Inference](inference.md)) directly.
`geosae.eval.auroc_moran.concept_auroc()` returns `auroc_median`, `auroc_mean` and
`ucov_frac_ge_thresh` over the headline concept groups, and
`geosae.eval.auroc_moran.patch_mean_moran()` returns `moran_i` (the median over alive features)
and `moran_i_mean` from an 8-neighbour patch-mean Moran's $I$;
`patch_mean_moran_from_arrays()` computes the same statistic from `(F, n_patches)` patch means
and patch centres, with `per_feature=True` for one value per feature.

`geosae.eval.baselines.dense_basis_auroc_matrix()` scores the dense reference bases — raw, PCA,
ICA, NMF — through the identical `concept_auroc_matrix()` machinery, which is what makes their
UCov and EMF directly comparable to an SAE's.

### Comparing dictionaries of different width

UCov reduces the AUROC matrix by $\max_f$, so a dictionary of $M$ atoms gets $M$ chances at that
maximum. The selection bias grows with $M$, and it grows for a dictionary that contains no concept
at all. [`permuted_best_auroc()`](api.md#geosae.eval.coverage.permuted_best_auroc) measures it: it
draws a balanced pixel sample, assigns the labels at random, and returns the distribution of
$\max_f \text{AUROC}[f, c]$ for that dictionary at that width.

```python
import numpy as np
from geosae.eval.coverage import permuted_best_auroc

rng = np.random.default_rng(0)
codes = rng.standard_normal((4000, 1024))  # a dictionary with no concept in it
for width in (64, 256, 1024):
    null = permuted_best_auroc(codes[:, :width], n_per_side=500, n_perm=100, seed=1)
    print(f"M={width:5d}  median best-of-M AUROC under the null = {np.median(null):.3f}")
```

```
M=   64  median best-of-M AUROC under the null = 0.547
M=  256  median best-of-M AUROC under the null = 0.554
M= 1024  median best-of-M AUROC under the null = 0.562
```

Pure noise "aligns" better at $M = 1024$ than at $M = 64$, so an SAE at $M = 1024$ starts ahead of a
dense basis at $M = D = 64$ before either has learned anything. The null depends only on the
dictionary's own feature-correlation structure and on the balanced sample size, not on which
concept is being scored, so one null serves every concept scored at that size.

### Calibrating coverage against the width

[`calibrated_coverage()`](api.md#geosae.eval.coverage.calibrated_coverage) counts a concept as
*recovered* when its observed best-feature AUROC — from
[`concept_best_auroc()`](api.md#geosae.eval.coverage.concept_best_auroc) — beats the $1 - \alpha$
quantile of that null. The width sits in the null, so it cancels.

```python
from geosae.eval.coverage import (
    calibrated_coverage, concept_auroc_matrix, concept_best_auroc, permuted_best_auroc, ucov)

rng = np.random.default_rng(6)
n_pixels, n_concepts = 600, 40
labels = {f"p{c}": rng.integers(1, 3, size=n_pixels) for c in range(n_concepts)}
class_table = np.stack([np.arange(n_concepts), np.ones(n_concepts, int)], axis=1)
order = list(labels)

for width in (64, 512):
    codes = rng.standard_normal((n_pixels, width))
    auc = concept_auroc_matrix(codes, labels, class_table, order, max_per_side=60)
    null = permuted_best_auroc(codes, n_per_side=60, n_perm=200, seed=7)
    cal = calibrated_coverage(concept_best_auroc(auc), null, alpha=0.05)
    print(f"M={width:4d}  UCov={ucov(auc):2d}/40   CCov={cal['ccov']:.3f}   mean_excess={cal['mean_excess']:+.3f}")
```

```
M=  64  UCov=11/40   CCov=0.075   mean_excess=+0.001
M= 512  UCov=35/40   CCov=0.100   mean_excess=+0.009
```

Labels and codes are independent here, so the honest answer is "nothing is recovered". UCov reports
11 of 40 concepts at $M = 64$ and 35 of 40 at $M = 512$ — width alone; CCov reports 0.075 and 0.100,
which is $\alpha = 0.05$ plus binomial noise at 40 concepts.

!!! warning

    CCov reports **significance, not effect size**, and on a real registry sample it saturates:
    with thousands of pixels per concept a tiny but genuine alignment clears its null, so every
    basis — raw, PCA, SAE — scores at or near 1.0 and the metric stops discriminating. Use the
    effect-size form below for anything but a toy.

### Counting null-corrected coverage with UCov*

Setting `alpha=0.5` puts the reference at the null *median* and `margin` restores a bar above it.
With `margin` $= \tau - 0.5$ this is UCov's own criterion, measured from the width-matched null
instead of from chance — same units, same semantics, no width term. `coverage_report()` reports it
as `ucov_null_corrected`.

```python
rng = np.random.default_rng(6)  # the same noise dictionaries as above
labels = {f"p{c}": rng.integers(1, 3, size=n_pixels) for c in range(n_concepts)}
for width in (64, 512):
    codes = rng.standard_normal((n_pixels, width))
    auc = concept_auroc_matrix(codes, labels, class_table, order, max_per_side=60)
    null = permuted_best_auroc(codes, n_per_side=60, n_perm=200, seed=7)
    star = calibrated_coverage(concept_best_auroc(auc), null, alpha=0.5, margin=0.15)
    print(f"M={width:4d}  UCov={ucov(auc):2d}/40   UCov*={star['ccov_count']:2d}/40")
```

```
M=  64  UCov=11/40   UCov*= 0/40
M= 512  UCov=35/40   UCov*= 0/40
```

Still pure noise, and UCov* correctly reports nothing recovered at either width where UCov climbs
from 11 to 35.

!!! note

    `mean_excess` — the mean margin over the median null — is the threshold-free companion, and the
    number to read when the question is *how much* better than a random dictionary of the same width
    the alignment is, rather than *how many* concepts clear a bar.

### Permuting labels by whole cells

The null above draws labels i.i.d. over pixels. Real concepts are clumped: land cover is constant
over most of a 64 × 64 patch, which lowers the effective sample size and widens the true null.
[`block_permuted_best_auroc()`](api.md#geosae.eval.coverage.block_permuted_best_auroc) permutes the
labels **whole cells at a time**, so the within-patch structure of the concept survives the
permutation while the code-to-label association does not.

```python
from geosae.eval.coverage import block_permuted_best_auroc

rng = np.random.default_rng(8)
n_cells, per_cell, width = 40, 30, 24
cell = np.repeat(np.arange(n_cells), per_cell)
codes = np.repeat(rng.standard_normal((n_cells, width)), per_cell, axis=0)  # spatially smooth
codes += 0.1 * rng.standard_normal(codes.shape)
concept = (cell < n_cells // 2).astype(int)  # a clumped concept: whole cells

iid = permuted_best_auroc(codes, n_per_side=200, n_perm=30, seed=9)
block = block_permuted_best_auroc(codes, cell, concept, n_perm=30, seed=9, max_per_side=200)
print(f"i.i.d. null : median {np.median(iid):.3f}   95th pct {np.quantile(iid, 0.95):.3f}")
print(f"block  null : median {np.median(block):.3f}   95th pct {np.quantile(block, 0.95):.3f}")
```

```
i.i.d. null : median 0.562   95th pct 0.588
block  null : median 0.695   95th pct 0.754
```

The block null's rejection threshold is 0.754 where the i.i.d. one says 0.588: on spatially clumped
labels the i.i.d. null is **anti-conservative**, and every AUROC between the two would be reported
as a recovered concept in error. The block null costs one AUROC matrix per concept, so it is the
audit run on the headline concepts rather than the default for all of them. It requires every cell
to contribute the same number of pixels in the same order, and raises `ValueError` otherwise.

### Matching the capacity

The other control is to spend the baseline's budget:
[`matched_budget_subset()`](api.md#geosae.eval.coverage.matched_budget_subset) keeps the $M'$
most-used atoms, with usage from
[`code_activation_rate()`](api.md#geosae.eval.coverage.code_activation_rate).

```python
from geosae.eval.coverage import code_activation_rate, matched_budget_subset

codes = np.zeros((1000, 8))
codes[:, [1, 3, 5]] = np.random.default_rng(0).standard_normal((1000, 3))
codes[:500, 6] = 1.0
usage = code_activation_rate(codes)
print("activation rate:", np.round(usage, 2))
print("top-3 features :", matched_budget_subset(usage, 3))
print("budget above F :", matched_budget_subset(usage, 99))
```

```
activation rate: [0.  1.  0.  1.  0.  1.  0.5 0. ]
top-3 features : [1 3 5]
budget above F : [0 1 2 3 4 5 6 7]
```

Selection is deterministic and ties break by index. Usage is not merit, though: the $M'$ most-used
atoms are not the SAE's $M'$ best-aligned ones, so UCov on this subset is a **lower bound** on what
the dictionary reaches at that width.

### Running the coverage stage

Every comparison above assumes the dictionaries are scored on the *same* pixels.
[`sample_registry_pixels()`](api.md#geosae.eval.registry.sample_registry_pixels) draws that sample
once — `n_patches` cells from the split, and from every cell the same `pixels_per_patch` positions,
drawn once — and [`RegistrySample`](api.md#geosae.eval.registry.RegistrySample) caches it as a
single `.npz` holding the raw pixels, one label array per product, the concept table and the cell
id of every pixel. Unlabelled pixels are kept, so every cell contributes the same pixels in the same
order, which is the invariant the block null needs; the balanced subsampling already excludes nodata
from both sides of every concept.

### Scoring on a frozen evaluation pool

An [`EvalPool`](api.md#geosae.eval.pool.EvalPool) fixes the labelled pixels once: a
[`PoolSpec`](api.md#geosae.eval.pool.PoolSpec) is the recipe (store, split, sizes, seed), and the
pool stores the cells, the shared within-patch positions and the per-concept class-balanced draws.
Build it with `geosae pool`, then point `coverage.pool` at it:

```bash
geosae pool -c batchtopk -c neurips2026/eval_pool.yaml
geosae coverage -c batchtopk -c neurips2026/eval_pool.yaml
```

The pool's pixels are materialised once into a content-addressed cache (`pool.cache_root`), so a
second stage scoring the same recipe reads nothing from the store again. A pool is verified against
its store's fingerprint on load, so scoring against a changed store raises instead of silently
drawing different pixels.

### Choosing which activations to score

An SAE can be scored on its $k$-sparse codes or on its dense pre-top-$k$ latents, and the two are
different measurements. A rank statistic over a column that is exactly zero on 99% of pixels is not
comparable to one over a dense raw dimension, so the dense regime
([`encode_sample(..., sparse=False)`](api.md#geosae.eval.registry.encode_sample), config key
`coverage.sparse_codes: false`) is the one in which an SAE feature and a raw dimension are
commensurate; the sparse regime is what a user of the dictionary actually receives. Report both.

!!! warning

    A BatchTopK model sparsifies at eval against `min_activation`, the `1 - k/M` quantile of its
    activations, which only training maintains. On an **untrained** dictionary it is still `-1`,
    every latent clears it, and the "sparse" codes come out fully dense — so an untrained random
    dictionary would silently be scored as a dense basis. `encode_sample` raises rather than let
    that happen; [`calibrate_sparsity()`](api.md#geosae.eval.registry.calibrate_sparsity) sets the
    threshold from the model's own activations, and the `coverage` stage calls it for every
    `coverage.random_widths` dictionary.

    It draws its calibration pixels **at random**: the quantile sits further into the tail the
    wider the dictionary, and a contiguous slice of a cell-grouped pixel array covers only a
    handful of places, which overshoots the threshold at large `M`.

[`coverage_report()`](api.md#geosae.eval.coverage.coverage_report) is the whole reduction for one
dictionary — the raw panel, the null, UCov* and CCov — matching the null to each concept's own
balanced sample size, quantised onto a few log-spaced levels.

```python
from geosae.eval.coverage import coverage_report

rng = np.random.default_rng(10)
n_pixels, width, n_concepts = 500, 12, 6
codes = 0.05 * rng.standard_normal((n_pixels, width))
labels, class_table = {}, np.stack([np.arange(n_concepts), np.ones(n_concepts, int)], axis=1)
for c in range(n_concepts):
    member = rng.random(n_pixels) < 0.5
    labels[f"p{c}"] = np.where(member, 1, 2)
    codes[member, c] += 5.0  # feature c detects concept c

scalars, arrays = coverage_report(
    codes, labels, class_table, list(labels), class_to_group={}, max_per_side=100, n_perm=50)
keys = ("ucov", "ccov", "emf", "mean_excess", "null_n_per_side", "median_null_best")
print({k: round(v, 3) for k, v in scalars.items() if k in keys})
print({k: v.shape for k, v in arrays.items()})
```

```
{'ucov': 6.0, 'emf': 6.0, 'null_n_per_side': 100.0, 'ccov': 1.0, 'mean_excess': 0.423, 'median_null_best': 0.577}
{'auc_matrix': (12, 6), 'best_auroc': (6,), 'null_best': (6, 50), 'null_level_size': (6,), 'balanced_size': (6,), 'activation_rate': (12,), 'valid_concepts': (6,)}
```

Six planted concepts, six recovered by both metrics, each by 0.42 AUROC over its null — this is what
a real result looks like, against the noise runs above.

`geosae coverage` runs all of it for a trained run: it builds or loads the sample, encodes it with
`best.pth`, and scores the SAE, the matched-capacity subsets in `coverage.matched_budgets` and the
dense bases in `coverage.dense_bases` on those pixels.

```bash
geosae coverage -c batchtopk -o coverage.sample_path=runs/registry_sample.npz
```

It writes `eval/<split>_coverage.json`, a one-row-per-variant `eval/<split>_coverage.csv`, and
`eval/<split>_coverage_arrays.npz` with the `(F, C)` AUROC matrices, the per-concept bests and the
null draws, so the scalars can be re-reduced at another threshold or another $\alpha$ without
re-encoding. Point `coverage.sample_path` — or, better, `coverage.pool` (below) — at one shared
sample for every cell of a width sweep; that is what makes the widths comparable.

## Measuring spatial scale with semivariance

[`geosae.eval.variogram`](api.md#geosae.eval.variogram) asks at which separations a feature varies.
Its semivariance at separation $h$ is half the mean squared difference of the activation over pixel
pairs $h$ apart. Within a patch, `semivariance_curves()` reads it from the horizontal and vertical
pairs of the pixel grid. Between two patches with pixel means $m_p, m_q$ and variances $v_p, v_q$,
the semivariance over all their pixel pairs is exactly $\tfrac12[(m_p - m_q)^2 + v_p + v_q]$, so
it is binned by the distance of the patch centres without touching a single pixel pair. Divided by
the feature's variance, the curve rises from the nugget (the first lag) towards 1, and
`correlation_lengths()` reads the first separation at which it reaches $1 - 1/e$, interpolated in
log distance; a crossing between the last within-patch lag and the first distance bin is flagged
as falling into the unsampled gap.

```python
import numpy as np
from geosae.eval.variogram import correlation_lengths, haversine_km, semivariance_curves

z = np.random.default_rng(0).normal(size=(50, 8, 8, 4))           # 50 patches, 8x8 grid, 4 features
dist = haversine_km(np.linspace(0, 40, 50), np.zeros(50))
out = semivariance_curves(z, dist, pixel_m=80.0, lags=range(1, 8), edges_km=[0, 1000, 5000], min_pairs=1)
cl = correlation_lengths(out["curve"], out["dist_km"], n_within=7)
print(out["curve"].shape, cl["status"])
```

```text
(4, 9) [1 1 1 1]
```

White noise is decorrelated at the first lag, so every feature reaches the level below it.

`geosae variogram` scores trained runs and conventional bases (`variogram.bases`, fitted with all
`D` components on the training pixels, or on pixels of the scored sample with
`variogram.fit_on: sample`) on one sample of **whole patches** and writes
`eval/variogram.npz` (per row `<name>/curve`, `/nugget`, `/length_km`, `/status`, `/within`) and
`eval/variogram.json` (one summary row per dictionary); `within` is the share of a feature's variance
that lies within a patch. `variogram.sample_cache` names a sample `.npz` to reuse:

```bash
geosae variogram -c neurips2026/embeddings/patch_tokens.yaml -c neurips2026/embeddings/olmoearth.yaml -o variogram.pixel_m=80 \
    -o variogram.sample_cache=data/samples/olmoearth_whole_patches.npz \
    -o "variogram.runs=[runs/olmoearth/batchtopk_s*]" -o "variogram.bases=[raw,pca]"
```

## Scoring monosemanticity by the similarity of the samples a feature fires on

[`geo_monosemanticity_score()`](api.md#geosae.eval.monosemanticity.geo_monosemanticity_score) and
[`monosemanticity_score()`](api.md#geosae.eval.monosemanticity.monosemanticity_score) are the
geographic score of Betti et al. (2026) and the visual score of Pach et al. (2025): the mean
similarity over all pairs of samples, weighted by the product of the feature's min-max scaled
activations at the two samples. The geographic similarity is `exp(-distance / length_scale_km)`.

```python
import numpy as np
from geosae.eval.monosemanticity import (
    geo_monosemanticity_score, mean_pairwise_similarity, monosemanticity_score, summarise_monosemanticity,
)

rng = np.random.default_rng(0)
n = 2000
lon, lat = rng.uniform(-180, 180, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
near = np.exp(-(((lon - 10) / 15) ** 2 + ((lat - 50) / 10) ** 2))  # fires around one place
codes = np.column_stack([near, rng.uniform(0, 1, n)])  # ... next to a feature that fires anywhere

geo = geo_monosemanticity_score(codes, lon, lat, length_scale_km=500.0)
print("GeoMS    regional=%.3f  scattered=%.3f" % tuple(geo))
print("uniform  %.3f" % mean_pairwise_similarity(lon=lon, lat=lat, length_scale_km=500.0))
```

```
GeoMS    regional=0.145  scattered=0.003
uniform  0.003
```

The score is a weighted mean, not a correlation, so a feature unrelated to place does not score
zero: it scores the mean pairwise similarity of the sample, which
[`mean_pairwise_similarity()`](api.md#geosae.eval.monosemanticity.mean_pairwise_similarity)
returns. Read every score against that level, and against a frozen random decoder, since any map
that is smooth in space scores above it.

The visual score takes one embedding per sample from a fixed image encoder in place of the
coordinates, and
[`summarise_monosemanticity()`](api.md#geosae.eval.monosemanticity.summarise_monosemanticity)
reduces either to the mean and max over features that the two papers report.

```python
images = rng.standard_normal((n, 16)) + 4.0 * np.outer(near > 0.2, np.ones(16))  # stand-in image embeddings
ms = monosemanticity_score(codes, images)
print("MS       regional=%.3f  scattered=%.3f" % tuple(ms))
print("uniform  %.3f" % mean_pairwise_similarity(embeddings=images))
print(summarise_monosemanticity(ms))
```

```
MS       regional=0.628  scattered=0.000
uniform  0.000
{'mean': 0.31418838416163763, 'max': 0.6282041307643365, 'median': 0.31418838416163763, 'n_scored': 2, 'n_features': 2}
```

The max is taken over the dictionary, so it grows with the number of features; compare it only
between dictionaries of the same width. Features non-zero on fewer than `min_active` samples are
NaN and left out of the summary.

!!! note
    Both scores take one row per sample. For a pixelwise product, pass patch-mean activations with
    patch centroids; the pairwise kernel of the geographic score is `(N, N)` and is built in blocks
    of `chunk` rows.

## Discovering sub-concepts

[`geosae.eval.subconcepts`](api.md#geosae.eval.subconcepts) asks whether a *parent* land-cover
class — a (product, code) pair, e.g. `(lc_esa_wc, 10)` "Tree cover" — splits into the sub-concepts a
decomposition claims, by checking that split against a coarse reference product the features were
not aligned with: RESOLVE biomes, Köppen-Geiger climate zones, potential natural vegetation or
RESOLVE ecoregions (`REFERENCE_PRODUCTS`).
[`patch_class_fractions()`](api.md#geosae.eval.subconcepts.patch_class_fractions) is the shared
building block: the fraction of each patch's valid pixels carrying every class of one categorical
product.

[`validate_parent()`](api.md#geosae.eval.subconcepts.validate_parent) runs the pipeline for one
parent class against one reference product: align features by folded patch-level AUROC against the
parent indicator, deduplicate near-identical decoder atoms, label every parent patch by its
best-aligned feature, and score that partition against the reference — normalised mutual
information, and per sub-concept the best-matching reference stratum with its precision, recall
and lift over the stratum's base rate.

A match alone is not evidence, because climate and biome maps vary smoothly in space. The result
therefore reports NMI against four null partitions (`NULL_KINDS`), each preserving something
different about the observed partition while destroying its agreement with the reference:

| Null | Preserves |
|---|---|
| `label_permutation` | the partition's sizes |
| `block_permutation` | the sizes, permuting whole geographic blocks |
| `geographic_kmeans` | spatial compactness (k-means on the patch coordinates) |
| `geographic_sizematched` | spatial compactness *and* the observed sub-concept sizes |

`geographic_sizematched` is the binding null (`BINDING_NULL`): a random compact subdivision of the
parent patches of the same sizes. `geographic_kmeans` is biased upward, because k-means produces
near-balanced clusters and agreement rises with cluster-size entropy. A sub-concept split counts as
validated only when it clears the binding null.

[`rank_candidates()`](api.md#geosae.eval.subconcepts.rank_candidates) sweeps every class of one parent
product against every reference product and returns a ranked table. `geosae subconcepts` runs the
sweep for a run's predictions, re-validates the candidates that clear the significance gate at full
null resolution, and writes `eval/subconcept_ranking.csv`, `eval/subconcept_validation.csv` and
`eval/subconcepts.json`:

```bash
geosae subconcepts -c batchtopk -o subconcepts.parent_product=lc_esa_wc -o subconcepts.n_null=200
```

## Estimating an area of applicability

`geosae.eval.aoa` asks whether a representation's geometry knows where it is extrapolating: fit a
Ridge head on a geophysical target, hold out an out-of-distribution region, and score the Spearman
correlation between a dissimilarity index and the held-out absolute error. A higher positive
correlation means the representation knows where a downstream model will fail.

[`AoASites`](api.md#geosae.eval.aoa.AoASites) holds several factor spaces over *the same* pixels, so
two representations are scored on identical folds and their dissimilarity maps are paired point by
point. `sample_sites()` builds one from a run; here it is built directly from arrays — a latitude
gradient, a 12-atom code that fires one atom per latitude band, and magnitudes spanning three
decades.

```python
import numpy as np
from geosae.eval import aoa

rng = np.random.default_rng(0)
n = 4000
lon, lat = rng.uniform(-160.0, 170.0, n), rng.uniform(-45.0, 65.0, n)
raw = np.column_stack([lat / 60.0, lon / 160.0, rng.normal(0, 0.1, n)])
code = np.zeros((n, 12))
magnitude = np.logspace(-1, 2, 12)
for j, band in enumerate(np.array_split(np.argsort(lat), 12)):
    code[band, j] = magnitude[j] * (np.abs(raw[band, 0]) + 0.05)

sites = aoa.AoASites(
    lonlat=np.column_stack([lon, lat]),
    pft=rng.choice([1, 4, 10, 12], n),
    spaces={"raw": raw, "sae": code},
    targets={"canopy": 3.0 * (lat / 60.0) ** 2},
    cell=np.arange(n),
)
print(len(sites), sorted(sites.spaces), sorted(sites.targets))
```

```
4000 ['raw', 'sae'] ['canopy']
```

### Sweeping folds, spaces and signals

[`run_scenarios()`](api.md#geosae.eval.aoa.run_scenarios) crosses the four split scenarios —
`single_continent`, `leave_one_continent_out`, `single_pft`, `leave_one_pft_out` — with every space,
target and signal, returning one flat record per cell. `extra_signals=True` adds the two signals that
are not distances.

```python
records = aoa.run_scenarios(
    sites, scenarios=("single_continent",), variants=("unweighted", "ridge"),
    extra_signals=True, max_train=1500, max_test=1500,
    min_train=300, min_test=300, verbose=False,
)
for r in records:
    if r["fold"] == "Africa":
        print(f"{r['space']:>4} {r['variant']:14s} rho={r['rho']:+.3f}")
```

```
 raw unweighted     rho=+0.131
 raw ridge          rho=-0.053
 raw support        rho=+nan
 raw support_probe  rho=+nan
 raw lpd            rho=+0.044
 sae unweighted     rho=+0.686
 sae ridge          rho=+0.660
 sae support        rho=+0.709
 sae support_probe  rho=-0.761
 sae lpd            rho=+0.468
```

`unweighted` and `ridge` are kNN distances (the latter rescaling dimensions by
$\sqrt{|\hat\beta_j| / \overline{|\hat\beta|}}$, the Meyer–Pebesma analogue; under a `scaling`, $\hat\beta_j$ is
the head's coefficient in the rescaled units, so the weights do not depend on the units the features
came in); `rf` adds the original
random-forest weighting (scikit-learn). `lpd` is local point density and
`support` / `support_probe` are support novelty.

!!! note

    The two `support` rows are `nan` in the dense space, not zero. Support novelty asks which atoms a
    pixel switched on; a dense factor matrix has every coordinate always active, so the score is
    identically zero and its correlation with error is undefined. A `nan` records "this signal does
    not exist here" — do not read it as a measured failure, and do not compare a sparse-space value
    against it as though it were a baseline.

### Comparing two dissimilarity maps

[`di_agreement()`](api.md#geosae.eval.aoa.di_agreement) answers whether two maps of the same pixels
actually differ. Two maps can be rank-equivalent, or they can differ in rank while both sitting far
outside the threshold everywhere — in which case the *pictures* are the same. It separates the two by
reporting a rank statistic and a threshold contingency.

```python
folds = {name: (tr, te) for name, tr, te in aoa.scenario_folds(
    "single_continent", sites.lonlat, sites.pft, min_train=300, min_test=300)}
train, test = folds["Africa"]
di = {}
for space in ("raw", "sae"):
    rec = aoa.score_fold(sites.spaces[space], sites.targets["canopy"], train, test,
                         variants=("unweighted",), max_train=1500, max_test=1500)
    di[space] = rec["d_test_unweighted"]

agree = aoa.di_agreement(di["raw"], di["sae"])
print(f"spearman        {agree['spearman']:+.3f}")
print(f"mask kappa      {agree['mask_kappa']:+.3f}")
print(f"out of domain   raw {agree['frac_out_a']:.1%}  sae {agree['frac_out_b']:.1%}")
print(f"differing mask  {agree['n_only_a'] + agree['n_only_b']} of {len(di['raw'])} pixels")
```

```
spearman        +0.148
mask kappa      -0.001
out of domain   raw 99.9%  sae 69.3%
differing mask  462 of 1500 pixels
```

`frac_out_a` is the diagnostic to read first. When both spaces flag nearly every held-out pixel as
out of domain, the masks cannot differ much whatever the representation, and a high agreement says
more about the split than about the two spaces. `agree["overlay"]` is a per-pixel class in
`{0: out in both, 1: out in b only, 2: out in a only, 3: in domain in both}`.

### Choosing the out-of-domain threshold

The default `threshold_rule="median"` marks a pixel out of domain when its distance exceeds the
median train-to-train distance. Training pixels come in spatial clusters, so that median measures
spacing inside a region, and a test region drawn from the same distribution can still come out
mostly "out of domain". For an area-of-applicability mask, use Meyer and Pebesma's rule as current
CAST implements it: the dissimilarity index is the nearest-training-pixel distance over the mean
pairwise training distance, each training pixel is scored only against pixels outside its own CV
fold, and the threshold is the upper whisker of those training values. Pass spatially blocked folds,
one id per site, here $10^\circ$ tiles:

```python
lon, lat = sites.lonlat.T
blocks = np.floor(lon / 10).astype(int) * 1000 + np.floor(lat / 10).astype(int)
rec = aoa.score_fold(sites.spaces["sae"], sites.targets["canopy"], train, test,
                     threshold_rule="cast", cv_folds=blocks)
```

Only the `frac_out_*` outputs change; the rank statistics do not depend on the threshold.
[`dissimilarity_index()`](api.md#geosae.eval.aoa.dissimilarity_index) and
[`aoa_threshold()`](api.md#geosae.eval.aoa.aoa_threshold) expose the index and the threshold on their
own, and `aoa_threshold(rule="boxplot")` reproduces the rule of older CAST releases.

### Choosing the distance geometry

The geometry is not neutral, and this is the choice most likely to invalidate a comparison between
two spaces. AlphaEarth embeddings are unit-norm by construction, so a Euclidean distance on them is
effectively a cosine distance; a sparse code is unnormalised with active magnitudes spanning orders
of magnitude, so the same formula is dominated by the highest-magnitude atoms.
[`scale_features()`](api.md#geosae.eval.aoa.scale_features) offers `none`, `std` (training standard
deviation) and `active` (mean magnitude while active, the sparse-code analogue);
`knn_distance(..., metric=...)` offers `euclidean`, `cosine` and `cityblock`.
[`hubness()`](api.md#geosae.eval.aoa.hubness) reports the pathology that motivates the choice — the
skewness of the $k$-occurrence distribution, which is how strongly a few reference points monopolise
everyone's neighbourhood.

```python
for scaling in ("none", "std", "active"):
    tr, te = aoa.scale_features(sites.spaces["sae"][train], sites.spaces["sae"][test], mode=scaling)
    h = aoa.hubness(tr, te, k=5)
    print(f"{scaling:7s} k-occurrence skewness {h['k_occurrence_skewness']:6.2f}")
```

```
none    k-occurrence skewness  10.51
std     k-occurrence skewness  10.51
active  k-occurrence skewness  10.50
```

Scaling changes nothing here because each synthetic pixel activates exactly one atom, so rescaling
atoms cannot reorder neighbours. On a real $k{=}16$ code it matters a great deal — measured for a BatchTopK SAE
($M = 1024$, $k = 16$) on AlphaEarth, an unscaled code space is more than twice as hubby as the
unit-norm embedding, and `active` scaling removes most of the excess:

| space and scaling | $k$-occurrence skewness, train Europe | train Africa |
|---|---:|---:|
| raw embedding, unscaled | 15.6 | 26.0 |
| SAE code, unscaled | 36.6 | 41.8 |
| SAE code, `active` | 17.8 | 27.5 |
| SAE code, cosine | 17.6 | 24.4 |

!!! tip

    Sweep the geometry rather than assuming it. The `geosae aoa` config takes a list of
    `{metric, scaling}` pairs and scores every fold under each, because which geometry is most
    favourable is itself a finding worth reporting.

### Scoring per-feature support

[`support_stats()`](api.md#geosae.eval.aoa.support_stats) records each atom's in-fold firing
frequency and active-magnitude quantile; [`support_novelty()`](api.md#geosae.eval.aoa.support_novelty)
scores a held-out pixel by the surprisal of its active set under those frequencies. Passing the
probe's $|\hat\beta|$ as `weights` isolates the atoms the downstream head actually extrapolates
along, and [`extrapolating_atoms()`](api.md#geosae.eval.aoa.extrapolating_atoms) names them.

```python
support = aoa.support_stats(sites.spaces["sae"][train])
print("atoms never active in fold:", int((support.frequency == 0).sum()), "of", support.frequency.size)
novelty = aoa.support_novelty(sites.spaces["sae"][test], support, mode="surprisal")
print(f"novelty: median {np.median(novelty):.3f}  max {novelty.max():.3f}")
dense = aoa.support_novelty(sites.spaces["raw"][test], aoa.support_stats(sites.spaces["raw"][train]))
print(f"same score on the dense space: all zero = {bool(np.all(dense == 0))}")
```

```
atoms never active in fold: 4 of 12
novelty: median 2.289  max 13.816
same score on the dense space: all zero = True
```

Four of the twelve atoms never fire inside the Africa fold, so a held-out pixel that switches one on
is using a direction the head was never fit on. The last line is the point of the signal and its
limit at once: it carries information no distance in the dense space can express, and for exactly
that reason it has no dense baseline to be compared against.

### Running the sweep on a trained run

`geosae aoa` samples held-out pixels once, caches them as `eval/aoa/sites.npz`, then sweeps every
geometry and writes `eval/aoa/records.csv`.

```bash
geosae aoa -c batchtopk -c neurips2026/aoa.yaml
```

Sampling streams the whole embedding store, so it dominates the runtime; the cache means a second
sweep with different geometries re-reads nothing. Set `aoa.resample: true` to force a fresh draw.

## Scoring an applicability index on graded hold-outs

An applicability index is a per-pixel score, higher meaning further from the training data, that is
meant to flag where a downstream model's error is large. `geosae.eval.selective` scores such an index
as a decision rather than as a rank correlation, and `geosae.eval.holdouts` builds the folds to score
it on.

### Risk-coverage AUC

`risk_coverage()` retains pixels in ascending order of the score and averages the mean error of the
retained set over every coverage level. Its `gain` places that area between an uninformative score
(0) and the error itself used as the score (1):

```python
import numpy as np
from geosae.eval import selective

rng = np.random.default_rng(0)
err = rng.exponential(size=2000)
for name, score in {"oracle": err, "random": rng.random(2000), "inverted": -err}.items():
    print(f"{name:9s} gain={selective.risk_coverage(score, err).gain:+.3f}")
```

```text
oracle    gain=+1.000
random    gain=+0.027
inverted  gain=-1.572
```

The gain is bounded above by 1 but not below by -1. Tied scores are averaged over every order, so a
constant score has a gain of exactly 0.

### Conformal selection at one target level

`conformal_selection()` calibrates two split-conformal quantiles on held-in pixels at the same level
`1 - alpha`: an absolute-error half-width and a score threshold. It then keeps the held-out pixels
under the threshold and reports the coverage of the kept pixels and the share of the map kept:

```python
rng = np.random.default_rng(1)
cal_score, cal_err = rng.random(2000), rng.exponential(size=2000)
test_score = rng.random(4000) * 3.0  # two thirds beyond the calibration range
test_err = rng.exponential(size=4000) * (1.0 + 3.0 * np.maximum(test_score - 1.0, 0.0))
sel = selective.conformal_selection(cal_err, cal_score, test_err, test_score, alpha=0.1)
print(f"all pixels: coverage {sel.coverage_all:.2f}")
print(f"retained:   coverage {sel.coverage:.2f} on {sel.area:.0%} of the map")
```

```text
all pixels: coverage 0.61
retained:   coverage 0.90 on 29% of the map
```

Under covariate shift the interval under-covers the whole map; a score that tracks the shift restores
the target on what it retains. Read the two numbers together: a score can reach the target by
retaining almost nothing.

### Building graded hold-outs

`graded_folds()` fixes a test region, the `n_test` sites nearest a centre, and takes the training
side as the `n_train` sites nearest the centre among those at least `gap` beyond the region's edge.
Sweeping the gap runs from near to far out of domain at constant sizes:

```python
from geosae.eval import holdouts

rng = np.random.default_rng(2)
lonlat = np.column_stack([rng.uniform(-180, 180, 20000), np.degrees(np.arcsin(rng.uniform(-1, 1, 20000)))])
centres = holdouts.spread_centres(lonlat, 2, min_separation=5000.0, seed=0)
for f in holdouts.graded_folds(lonlat, centres, gaps=(0, 1000, 4000), n_test=500, n_train=3000):
    d = holdouts.site_distance(lonlat, lonlat[f.centre])
    print(f"{f.name:12s} test within {d[f.test_mask].max():5.0f} km, train from {d[f.train_mask].min():5.0f} km")
```

```text
c11639_g0    test within  1974 km, train from  1978 km
c11639_g1000 test within  1974 km, train from  2974 km
c11639_g4000 test within  1974 km, train from  5974 km
c8499_g0     test within  2042 km, train from  2044 km
c8499_g1000  test within  2042 km, train from  3044 km
c8499_g4000  test within  2042 km, train from  6043 km
```

Great-circle distance gives distance-graded folds. For environment-graded folds pass
`climate_coordinates()` of the sites' CHELSA bioclim values (`read_site_bioclim()` reads them from
the store) with `metric="euclidean"`; the test region is then a climate, scattered over the globe.

!!! note
    A graded fold says how far the training side is, not how far out of domain the test side is.
    Report each fold's out-of-domain fraction (`frac_out` of the raw space) next to every score: a
    fold on which nearly every pixel is out of domain cannot separate two indices.

### Naming why a pixel is out of range

`geosae.eval.extrapolation.mess()` is the multivariate environmental similarity surface of Elith et
al. (2010), run unit by unit: a unit's similarity is negative when a pixel's value leaves the labelled
set's range, a pixel's similarity is its minimum over units, and `most_dissimilar` is the unit that
attains it. On an SAE code that unit names the concept along which the pixel is out of range. Pass
`one_sided=True` for a non-negative sparse code, whose zeros are never novel:

```python
from geosae.eval import extrapolation

rng = np.random.default_rng(3)
codes = rng.exponential(size=(3000, 6)) * (rng.random((3000, 6)) < 0.3)  # sparse, non-negative
labelled, test = codes[:2000].copy(), codes[2000:].copy()
labelled[:, 4] = 0.0  # unit 4 never fires in the labelled set
res = extrapolation.mess(labelled, test, one_sided=True)
out = res.similarity < 0
print(f"out of range: {out.mean():.2f}")
print("reasons:", np.bincount(res.most_dissimilar[out], minlength=6))
```

```text
out of range: 0.31
reasons: [  0   0   0   0 307   0]
```

Every out-of-range pixel is out of range because of unit 4. `coverage_gap()` gives the same answer
for a whole data set before any prediction: the share of unlabelled rows above the labelled maximum,
per unit.

```python
print("coverage gap:", np.round(extrapolation.coverage_gap(labelled, test), 3))
```

```text
coverage gap: [0.    0.    0.    0.    0.307 0.   ]
```

The maximum is fragile. If a single labelled pixel fires unit 4, every later exceedance is hidden.
`tail_probability()` reads the labelled distribution instead: for each active unit, the share of
labelled values at or above the pixel's (a conformal p-value), and per pixel the smallest one and its
unit.

```python
labelled[0, 4] = 5.0  # one labelled pixel fires unit 4 strongly
print("mess, out of range:", (extrapolation.mess(labelled, test, one_sided=True).similarity < 0).mean())
tail = extrapolation.tail_probability(labelled, test)
low = tail.probability < 0.01
print(f"tail < 0.01: {low.mean():.2f}")
print("reasons:", np.bincount(tail.unit[low], minlength=6))
```

```text
mess, out of range: 0.002
tail < 0.01: 0.34
reasons: [  8   5   7   9 306   3]
```

`mahalanobis()` is the combination check (the RX detector of hyperspectral imaging): a shrunk
Mahalanobis distance to the labelled mean, which also sees units that are each in range but combined
as the labels never were, and directions the labels never vary in. `score_fold()` scores it as the
variant `"mahalanobis"`.

To ask what a score adds to a distance, `selective.stratified_auroc()` computes its AUROC for a set of
positive pixels within equal-count bins of a reference score, so a score that only repeats the
reference gets about 0.5:

```python
from geosae.eval import selective

carries = test[:, 4] > 0
dist = np.linalg.norm(test - labelled.mean(0), axis=1)
print(f"{selective.stratified_auroc(-tail.probability, carries, dist):.2f}")
```

```text
1.00
```

The share of positives a distance does not flag at all is `selective.silent_share()`: at a 5 %
false-alarm rate on the negatives, an uninformative distance leaves 0.95 of them unflagged. It says how
much is left for a score like the range check to find.

The range check can name a concept only through a unit that fires on it. `extrapolation.best_unit()`
asks whether a code has one: the unit whose activation alone separates the concept's pixels best, and
its AUROC. Compare it with a linear probe on the embedding: a probe well above the best unit means the
embedding encodes the concept but the dictionary spreads it over several units.

```python
print(extrapolation.best_unit(test, carries))
```

```text
(4, 1.0)
```

The range check explains; it does not detect well. A pixel that combines two units each seen in the
labels, but never together, stays in range, and the minimum over many units picks up noise. Use a
distance (`knn_distance()`, or the head-weighted index of `score_fold()`) to decide which pixels to
flag, and the range check to say why.

### Running it as a stage

`score_fold()` computes both statistics when `n_calibration` training pixels are held back from the
head and the reference set, and `run_folds()` scores any set of folds. The `aoa` stage runs the graded
designs from `configs/neurips2026/aoa_graded.yaml`, optionally with a conventional dictionary's code as a further
feature space (`aoa.dictionary_space`), and writes `eval/aoa/records_graded.csv`:

```bash
geosae aoa -c batchtopk -c neurips2026/aoa.yaml -c neurips2026/aoa_graded.yaml \
  -o 'aoa.dictionary_space={method: random, width: 1024, rate: 0.015625, n_fit_pixels: 100000}'
```

The config scores support novelty (`support`, `support_probe`) next to the two distances, with the
same statistics; its calibration scores come from the held-back pixels like the distances'. With
`method: random` and `rate: k/M`, `aoa.dictionary_space` adds the random sparse code control: a random
dictionary whose code is sparsified by `rate_matched_support()` to the SAE's firing rate, as a space
named `random`. `method: kmeans` adds a $k$-means one-hot code instead.

By default each space's index is judged on a head fitted in that space, so two spaces differ in both
the index and the model whose error it ranks. Set `aoa.graded.probe_space: raw` to judge every index
on the raw embedding's head instead (`probe_features` of `score_fold()`); the difference between the
two readings is the part of an advantage that comes from the head rather than from the index.

## Fitting conventional baselines

[`fit_dictionary()`](api.md#geosae.eval.dictionaries.fit_dictionary) fits a conventional
decomposition as a dictionary with the same `(encode, atoms, offset)` interface a trained SAE
exposes: `raw` (the embedding's own dimensions), `pca`, `ica` and `nmf`, plus `varimax`, `kmeans`
and `random`. Every stage fits them on the same training pixels,
[`baseline_fit_pixels_from_config()`](api.md#geosae.pipeline.baseline_fit_pixels_from_config):
one seeded pixel per training patch by default. `geosae fit-table` draws that table once and caches
it at `baselines.fit_table_path`, so every later fit reads it back:

```bash
geosae fit-table -c neurips2026/alphaearth.yaml -o baselines.fit_table_path=data/fit_tables/alphaearth.npz
```

## Evaluating further embeddings

OlmoEarth and MMEarth embed Sentinel-2 imagery into patch tokens rather than pixels.
`geosae subset` fetches Sentinel-2 L2A for a split of the store into a subset store, runs each
backbone over it and writes per-dimension standardisation statistics, in three stages (see
[Building the dataset](dataset.md#further-embeddings)):

```bash
geosae subset -c download -o subset.stage=fetch   # CPUs and network
geosae subset -c download -o subset.stage=embed   # one GPU
geosae subset -c download -o subset.stage=stats
```

`configs/neurips2026/embeddings/patch_tokens.yaml` carries the token geometry and training recipe, and
`configs/neurips2026/embeddings/olmoearth.yaml` and `configs/neurips2026/embeddings/mmearth.yaml` layer each product on top. Token embeddings are neither unit-norm nor zero-mean, so their `data.transforms` z-score
them; every stage applies each run's own transforms. A subset store has no reference products of
its own, so `panel.label_store` / `coverage.label_store` read the labels of the parent store
(see [`LabelStoreSpec`](api.md#geosae.eval.registry.LabelStoreSpec)).

```bash
geosae train -c neurips2026/embeddings/patch_tokens.yaml -c neurips2026/embeddings/olmoearth.yaml
```

The backbones need the `embed` extra (`pip install "geosae[embed]"`).

Copernicus-FM is published as a 0.25 degree global embedding map; `build_copernicusfm025_store`
coarsens the reference products of a patch store onto that grid and fills the embedding by
nearest-node lookup (see [Coarse gridded stores](dataset.md#coarse-gridded-stores)).
`configs/neurips2026/embeddings/gridded.yaml` with `configs/neurips2026/embeddings/copernicusfm.yaml` on top trains on the resulting
single-pixel-patch store.

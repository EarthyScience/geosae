# Inference

`geosae predict` runs a trained model over a data split, sparsifies its dense
activations, and writes the sparse codes plus per-patch summary statistics to
a prediction Zarr store.

```bash
geosae predict -c batchtopk
```

```yaml
predict:
  split: test              # train | val | test
  checkpoint: null          # defaults to the run dir's best.pth
  batch_size: 64
  max_k: null                # padding width for the sparse arrays; defaults to max(2k, 32),
                              # or 32 for a model without a fixed k (ReLU+l1) --
                              # pass "auto" instead to measure it (see below)
  chunk_cells: 50             # cells written per contiguous batch
```

`max_k` bounds how many active latents a pixel can have *stored*; any token
that exceeds it is truncated to its top-`max_k` by magnitude, and `geosae
predict` refuses to run with a truncated store unless `allow_truncation: true`
is also set (every downstream concept metric reads the full code and is
biased by truncation). The default `max(2k, 32)` is only a safe guess for a
model with a fixed selection budget (Top-K/BatchTopK/Matryoshka); a model
without one (ReLU+l1) has no such bound on its per-token L0 at all,
so a fixed 32-wide store can silently undershoot the real right tail. Set
`max_k: auto` to measure it instead
(`geosae.predict.writer.estimate_max_k`): a handful of batches are probed, the
largest observed per-token active count is padded by a safety margin, and the
result is printed before the full prediction pass starts.

## What gets written

`predictions/<split>.zarr`:

- **`z_values` / `z_indices` / `z_counts`**, each `(N, H, W, max_k)` — a
  padded sparse representation of the per-pixel activations: the top
  `max_k` nonzero latent values and their indices per pixel (`z_counts`
  records how many are actually active, in case a pixel exceeds `max_k`).
- **`cell` / `lon` / `lat` / `fold`** — per-patch metadata.
- **`feat_patch_stats`**, shape `(n_feat, n_patch, 2)` — for every
  `(feature, patch)` pair, `[mean activation, number of active pixels]`. This
  compact summary is what patch-level analyses (sub-concepts, patch-mean
  Moran's I) read, so they never need to scan the full sparse code arrays.

`geosae predict` also writes `eval/<split>_metrics.csv` with the
reconstruction metrics (`geosae.eval.metrics.reconstruction_metrics`): FVE,
reconstruction cosine similarity, reconstruction MSE, and L0 (mean active
latents per pixel).

## Running the downstream stages

Every stage after `predict` that reads codes reads the prediction store, plus the original patch
store for labels:

```bash
geosae subconcepts -c cfg.yaml   # sub-concepts of a parent class, validated against nulls
```

`geosae coverage`, `geosae panel`, `geosae variogram` and `geosae aoa` instead encode their own
fixed pixel samples with the run's checkpoint, so they do not need a prediction store. See
[Evaluation](evaluation.md) for all of them.

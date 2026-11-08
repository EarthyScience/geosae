# Configs of *Decomposing Earth Embeddings with Sparse Autoencoders* (NeurIPS 2026)

```
configs/neurips2026/
├── alphaearth.yaml, tessera.yaml   base config of the SAE sweep, one per embedding
├── variants/                       batchtopk, matryoshka, relu_l1, frozen (frozen-decoder baseline)
├── panel.yaml                      reconstruction of a sweep and of the dense baselines
├── eval_pool.yaml, coverage.yaml   the frozen evaluation pool, and concept alignment on it
├── variogram.yaml                  spatial scale: semivariance curves, correlation lengths
├── subconcepts.yaml                sub-concepts of a coarse class, against random partitions
├── aoa.yaml, aoa_graded.yaml       area of applicability: continent/PFT folds, graded hold-outs
└── embeddings/                     further embeddings: patch tokens (OlmoEarth, MMEarth),
                                    gridded maps (Copernicus-FM)
```

## The sweep

A trained SAE is one base config, one variant overlay and three overrides: the dictionary size
`model.latent_dim` (M), the sparsity `model.k` and the `seed`. The paper's sweep trains, on both
AlphaEarth and Tessera,

- BatchTopK, Matryoshka BatchTopK and ReLU+ℓ1 at M = 1024 with k ∈ {8, 16, 32}, and at
  M ∈ {512, 2048} with k = 16, seeds 0, 1, 2 (90 runs);
- the frozen-decoder baseline at M = 1024 with k ∈ {8, 16, 32}, seeds 0, 1, 2 (18 runs).

| variant | `model.type` | sparsity |
|---|---|---|
| BatchTopK | `batchtopk` | the `B·k` largest pre-activations of a batch of `B` pixels |
| Matryoshka BatchTopK | `matryoshka` | BatchTopK, plus the reconstruction of the nested prefix of `round(0.2 M)` atoms |
| ReLU+ℓ1 | `relu_l1` | ReLU and an ℓ1 penalty tuned during training until the mean L0 matches `k` |
| frozen decoder | `batchtopk`, `freeze_decoder: true` | BatchTopK with the decoder fixed at its random unit-norm initialisation |

All variants share a single linear encoder and decoder with unit-norm atoms, the encoder initialised
as the decoder's transpose, a pre-encoder bias, AuxK revival of dead latents, Adam (β₁ = 0,
β₂ = 0.999) at a peak learning rate of 1e-3 with 1000 warm-up steps and cosine decay over
50 × 1000 steps, and a per-step pixel subsample (`train.pixel_subsample`). AlphaEarth is used as
released (unit norm); Tessera is clipped and standardised per dimension with the store's
`tessera_norm_*` statistics.

`geosae run` trains and then predicts the test split in one run directory,
`runs/<name>/<timestamp>/`. The commands of the whole sweep, and of every evaluation stage and
figure, are on the documentation pages
[Reproducing: data, checkpoints and training](https://earthyscience.github.io/geosae/neurips2026/reproduce-data/#training-the-sweep)
and [Reproducing the figures and tables](https://earthyscience.github.io/geosae/neurips2026/reproduce-figures/).

## Evaluation and the other experiments

The evaluation stages (`geosae panel`, `pool`, `coverage`, `variogram`, `subconcepts`, `aoa`) read
these run directories; the configs in this folder hold their settings. The scripts that draw every
table and figure of the paper are in [`scripts/neurips2026/`](../../scripts/neurips2026/README.md).

# Training

`geosae train` fits one sparse autoencoder from one YAML config, resolved with
[OmegaConf](https://omegaconf.readthedocs.io/), into a fresh run directory:

```bash
geosae train -c batchtopk
```

`-c` takes a YAML path or the name of a config that ships with the package
([`bundled_configs()`](api.md#geosae.config.bundled_configs) lists them; `batchtopk` is the default
recipe). Configs can be layered (`-c batchtopk -c my_overrides.yaml`, later files win) and any field
can be overridden from the command line with dotted keys:

```bash
geosae train -c batchtopk -o model.k=32 -o train.epochs=10
```

## Pointing at the data

The `data` section names the patch store (see [Building a dataset](dataset.md)) and configures the
reader, [`PatchData`](api.md#geosae.data.dataset.PatchData):

```yaml
data:
  zarr_path: ${oc.env:GEOSAE_STORE,data/majortom210k.zarr}
  embedding_name: alphaearth      # or "tessera"
  split_name: null                # null: the store's only split_* array
  min_spatial_std: 1e-3           # drops degenerate (constant-fill) cells; null keeps every cell
  drop_flat_pixel_cells: true     # drops cells with a fill pixel (every dimension equal)
  max_abs_value: 1.0e6            # drops cells with an out-of-range pixel magnitude; null keeps every cell
  transforms:
    hard_clip: 8.0                # optional per-dimension clip / z-score, see Transforms
```

Set the `GEOSAE_STORE` environment variable to point every config at your store, or override one
run with `-o data.zarr_path=<store>`. `data.split_name` names the split array; a store with several
(`split_default`, ...) needs it set.

All three filters are on by default (see [`geosae.data.quality`](api.md#geosae.data.quality)) and
run in this order — `max_abs_value` first, since a sentinel value can overflow the spatial-std scan.
Training loads the `train` split into RAM as `float16` (`train.compute_inram: false` switches to
lazy, chunk-grouped zarr reads) and validates on `val` after every epoch.

## Choosing a model

`model.type` selects a registered builder; every other key of `model` is a constructor argument.

```yaml
model:
  type: batchtopk          # batchtopk | topk | matryoshka | relu | relu_l1
  input_dim: 64            # D, the embedding width
  latent_dim: 1024         # M, the dictionary size
  k: 16                    # active latents per pixel (a batch average for batchtopk)
  pre_bias: false
  normalize_decoder: true
  h_frac: 0.2              # matryoshka only: fraction of latents in the nested prefix
  prefix_fracs: null       # matryoshka only: several nested prefixes, e.g. [0.0625, 0.25]
  freeze_decoder: false    # true: the frozen-decoder baseline
```

- **`batchtopk`** — top-k selection across the whole training batch (average sparsity `k`). At
  inference a calibrated activation threshold replaces the batch-dependent selection, so codes do
  not depend on the batch.
- **`topk`** — the same architecture with a per-pixel top-k.
- **`matryoshka`** — BatchTopK plus a reconstruction term for the first `h_frac * M` latents (or
  one per entry of `prefix_fracs`), which nests a coarse dictionary inside the full one. Set
  `train.slow_group_weight` to weight that term.
- **`relu`** / **`relu_l1`** — a ReLU SAE whose sparsity comes from an ℓ1 penalty
  (`train.l1_coeff`) rather than a selection.

[`build_model()`](api.md#geosae.models.registry.build_model) is the same factory in code:

```python
import torch
from geosae.models import MODEL_REGISTRY, SparseAutoencoder, build_model

print(sorted(MODEL_REGISTRY))
model = build_model({"type": "matryoshka", "input_dim": 64, "latent_dim": 1024, "k": 16, "seed": 0})
model.train()
recon, z, z_dense = model(torch.randn(8, 4, 4, 64))
print(type(model).__name__, isinstance(model, SparseAutoencoder), recon.shape, z.shape)
print("active per pixel:", (z > 0).sum(-1).float().mean().item(), "prefix:", model.n_prefix)
print(model.dictionary().shape)
```

```
['batchtopk', 'matryoshka', 'relu', 'relu_l1', 'topk']
TopKSAE True torch.Size([8, 4, 4, 64]) torch.Size([8, 4, 4, 1024])
active per pixel: 16.0 prefix: 205
(1024, 64)
```

Every model maps inputs of any leading shape `(..., D)` to `(recon, z, z_dense)` and encodes every
pixel independently; [`dictionary()`](api.md#geosae.models.base.SparseAutoencoder.dictionary)
returns the decoder atoms as `(M, D)` rows.

### Training the frozen-decoder baseline

`model.freeze_decoder: true` replaces the decoder of a top-k model with a fixed random unit-norm
dictionary and freezes it with the reconstruction biases, so only the encoder trains:

```python
from geosae.models import build_model

frozen = build_model({"type": "batchtopk", "input_dim": 64, "latent_dim": 1024, "k": 16,
                      "freeze_decoder": True, "seed": 0})
print([name for name, p in frozen.named_parameters() if p.requires_grad])
```

```
['encoder.weight', 'encoder.bias']
```

The random dictionary is seeded by `model.seed`, or by the run's top-level `seed` when unset. The
`relu` type rejects `freeze_decoder: true`.

### Adding an architecture

A new architecture subclasses [`SparseAutoencoder`](api.md#geosae.models.base.SparseAutoencoder)
— `encode`, `decode` and `forward` returning `(recon, z, z_dense)` — and registers a builder with
[`register_model()`](api.md#geosae.models.registry.register_model); configs then select it by name.

```python
from geosae.models import TopKSAE, build_model, register_model


@register_model("pertopk")
def _build_pertopk(cfg: dict) -> TopKSAE:
    return TopKSAE(**cfg, batch_topk=False)


print(build_model({"type": "pertopk", "input_dim": 64, "latent_dim": 256, "k": 8}))
```

```
TopKSAE(
  input_dim=64, latent_dim=256, k=8, batch_topk=False, pre_bias=True, normalize_decoder=True, freeze_decoder=False
  (encoder): Linear(in_features=64, out_features=256, bias=True)
  (decoder): Linear(in_features=256, out_features=64, bias=False)
)
```

The trainer, the predictor and every evaluation stage only use the `SparseAutoencoder` contract,
so a registered model runs through all of them unchanged.

## Setting the training recipe

```yaml
train:
  batch_size: 128
  epochs: 50
  steps_per_epoch: 1000
  lr: 1e-3
  beta1: 0.0
  beta2: 0.999
  clip_grad_norm: 1.0
  rampup: 1000             # learning-rate warm-up steps, then cosine decay to 0
  aux_penalty: 0.03125     # AuxK dead-feature-revival loss weight
  top_k_aux: 32
  l1_coeff: 0.0            # ReLU+l1 sparsity penalty
  slow_group_weight: 0.0   # Matryoshka prefix reconstruction weight
  amp: true                # bf16 mixed precision
```

The training loop ([`Trainer`](api.md#geosae.train.trainer.Trainer)) is single-device PyTorch:
Adam with `beta1 = 0`, gradient clipping, linear warm-up and cosine decay, MSE reconstruction plus
an AuxK term that lets dead latents reconstruct the residual. Validation runs after every epoch;
the checkpoint with the best validation FVE is kept as `best.pth`, beside `last.pth`.

`device: auto` (the default of `batchtopk`) trains on the GPU when one is visible and on the CPU
otherwise; `-o device=cpu` or `-o device=cuda:1` picks one. bf16 autocast (`train.amp`) applies on
a GPU only. The bundled `small` overlay (`-c batchtopk -c small`) shrinks the model and the
schedule to minutes on a CPU, for a store of a few hundred patches or a first look.

### Batching pixel subsamples

`train.pixel_subsample` replaces full-patch batches with a per-step pixel subsample: each step draws
`n_anchor_patches` patches and `recon_pixels_per_patch` pixels from each, and feeds them as a flat
`(Q, 1, 1, D)` batch.

```yaml
train:
  compute_inram: true      # required: the subsampler reads the in-RAM split
  pixel_subsample:
    n_anchor_patches: 512
    recon_pixels_per_patch: 128
    m_anchor_pixels: 8
    pair_mode: local       # pixels drawn around anchors, see geosae.train.pixel_subsample
    local_sigma_px: 2.0
    local_radius_px: 8
```

### Matching sparsity with a target L0

Top-k models reach their sparsity by construction (`model.k`). A ReLU+ℓ1 model's sparsity emerges
from `train.l1_coeff`, so `train.target_l0` adds a closed-loop controller
([`TargetL0Controller`](api.md#geosae.train.sparsity_controller.TargetL0Controller)) that rescales
the penalty toward a target mean L0 every step:

```yaml
train:
  l1_coeff: 1.0e-3                 # a nonzero starting point for the controller to rescale
  target_l0:
    value: 16.0                    # target mean active latents per pixel
    gain: 0.05
    ema_beta: 0.9
    best_checkpoint_tolerance: 0.5
```

The realised penalty and L0 are logged every epoch in `metrics.jsonl` (`train.sparsity_penalty`,
`train.l0`).

!!! note

    FVE is only comparable at the same sparsity. With a controller, an epoch only competes for
    `best.pth` while its validation L0 lies within `best_checkpoint_tolerance` (a fraction of the
    target) of `target_l0.value`; otherwise an under-sparsified early epoch would win on FVE alone.

`model.normalize_inputs` (ReLU+ℓ1, default `true`) scales every batch to mean norm `sqrt(D)` before
encoding; on unit-norm embeddings set it to `false`. `train.penalty_warmup_steps` ramps the ℓ1
penalty linearly from 0.

## Organising runs

Every invocation writes into one self-contained run directory:

```
runs/<name>/<YYYYmmdd-HHMMSS>/
├── config.yaml          # fully-resolved config (the reproduction record)
├── best.pth  last.pth   # checkpoints (incl. the calibrated inference threshold)
├── metrics.jsonl        # per-epoch train/val metrics
├── eval/                # metric tables of every evaluation stage
└── predictions/         # sparse codes + feat_patch_stats (see Inference)
```

`geosae train` creates a fresh timestamped run directory; every other stage reuses the most recent
one for the config's `name` unless you pass `--run runs/<name>/<timestamp>` (which also uses that
run's own `config.yaml` as the base config) or `-o run.dir=...`. `geosae run` chains train and
predict in one run directory.

## Training on a coarse gridded store

A `patch_size = 1` store (see [Coarse gridded stores](dataset.md#coarse-gridded-stores)) treats a
single grid cell as a "patch". Set `data.min_spatial_std: null`, since the spatial-std filter would
score every one-pixel cell as constant and drop the whole split, and a much larger
`train.batch_size` (e.g. 4096), since a batch of cells holds far fewer pixels than a batch of
patches.

## Sweeping hyperparameters

A sweep is the same config with per-cell overrides; each run directory stores the resolved config
it used, so the grid stays reproducible without one config file per cell:

```bash
for m in 512 1024 2048; do for k in 8 16 32; do
  geosae run -c batchtopk -o model.latent_dim=$m -o model.k=$k \
             -o run.dir=runs/sweep/m${m}_k${k}
done; done
```

The [NeurIPS 2026 paper](neurips2026/reproduce-data.md#training-the-sweep) pages list the sweep
behind the paper.

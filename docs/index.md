---
title: geosae
---

# geosae { .visually-hidden }

![geosae](assets/logo/geosae-lockup-light.svg#only-light){ .geosae-lockup }
![geosae](assets/logo/geosae-lockup-dark.svg#only-dark){ .geosae-lockup }

`geosae` is a Python library for doing science with sparse autoencoders (SAEs) on Earth
embeddings — pixel-level representations such as AlphaEarth and Tessera, and patch tokens of
foundation models such as OlmoEarth and MMEarth. It downloads patches of embeddings together with
reference products (land cover, climate, terrain, soils) for the cells you choose, trains SAEs
(BatchTopK, Matryoshka, ReLU+ℓ1, and a frozen-decoder control), encodes embeddings into sparse
features, and evaluates any decomposition on reconstruction at matched sparsity, alignment with
reference concepts, spatial scale, sub-concepts and the area of applicability, against baselines
such as raw dimensions, PCA, ICA and NMF.

Everything is plain PyTorch, NumPy and Zarr. Every stage is a Python function and a `geosae`
command driven by one YAML config; every run directory stores the resolved config it used.

## Installing

```bash
pip install geosae
```

`geosae` needs Python 3.11 or newer. `pip` installs the default PyTorch build for your platform; for
a specific CUDA version, install PyTorch first following
[pytorch.org](https://pytorch.org/get-started/locally/), then `geosae`. Optional extras add more:

| Extra | Needed for |
|---|---|
| `download` | Building a patch store from the upstream sources — see [Building a dataset](dataset.md) |
| `embed` | Encoding Sentinel-2 patches with OlmoEarth and MMEarth |
| `viz` | Plotting (`matplotlib`, `nicefigs`) |
| `docs` | Building this documentation site |
| `dev` | Tests and linting (`pytest`, `pre-commit`, `ruff`) |

```bash
pip install "geosae[download]"
```

## Getting started

The [Quickstart](quickstart.md) builds a store of a few hundred patches, trains an SAE on it,
evaluates it and maps a feature, in about ten minutes on a laptop CPU. In short:

```bash
geosae download -c download -o download.out=data/quickstart.zarr \
    -o "download.cells.bbox=[11.0,50.5,12.0,51.2]" -o "download.products=[alphaearth_cog,lc_esa_wc]"
export GEOSAE_STORE=data/quickstart.zarr
geosae train    -c batchtopk -c small
geosae predict  -c batchtopk -c small
geosae coverage -c batchtopk -c small
```

`-c batchtopk -c small` layers two configs that ship with the package; `-c path/to/your.yaml`
takes your own, and `-o key=value` overrides any field. `geosae --help` lists every stage.

## Package layout

```
geosae/
  data/            # patch dataset and loader, quality filters
  data/download/   # building a store: MajorTOM grid and cell choice, product registry, fetchers, store
  data/embed/      # foundation-model backbones for patch-token embeddings
  models/          # SparseAutoencoder contract, model registry, TopKSAE, ReLUSAE
  train/           # loss (MSE + AuxK + l1 + Matryoshka), trainer, sparsity controller, pixel subsampler
  predict/         # dense -> sparse codes, sparse zarr writer, per-patch stats
  eval/            # reconstruction, the panel, concept alignment, spatial scale, sub-concepts, AoA
  pipeline.py      # config -> dataset, model, run model, evaluation sample, baseline fitting pixels
  run.py, config.py, cli.py   # run directories, OmegaConf configs, the `geosae` CLI
```

The guide pages follow a run: [Building a dataset](dataset.md), [Training](training.md),
[Inference](inference.md), [Concepts and factors](concepts.md) and [Evaluation](evaluation.md). The
[API reference](api.md) documents every module. `geosae` is the code of *Decomposing Earth
Embeddings with Sparse Autoencoders*; the [NeurIPS 2026 paper](neurips2026/index.md) page covers
reproducing it.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/EarthyScience/geosae/main/docs/assets/logo/geosae-lockup-dark.svg">
    <img alt="geosae" src="https://raw.githubusercontent.com/EarthyScience/geosae/main/docs/assets/logo/geosae-lockup-light.svg" width="360">
  </picture>
</p>

<p align="center"><b>Sparse autoencoders for Earth embeddings.</b></p>

<p align="center">
  <a href="https://github.com/EarthyScience/geosae/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/EarthyScience/geosae/ci.yml?branch=main&label=CI"></a>
  <a href="https://github.com/EarthyScience/geosae/actions/workflows/docs.yml"><img alt="Coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fearthyscience.github.io%2Fgeosae%2Fcoverage.json"></a>
  <a href="https://pypi.org/project/geosae/"><img alt="PyPI" src="https://img.shields.io/pypi/v/geosae"></a>
  <a href="https://pypi.org/project/geosae/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/geosae?label=python"></a>
  <a href="https://earthyscience.github.io/geosae"><img alt="Documentation" src="https://img.shields.io/github/actions/workflow/status/EarthyScience/geosae/docs.yml?branch=main&label=docs"></a>
  <a href="https://github.com/EarthyScience/geosae/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://doi.org/10.5281/zenodo.24046835"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24046835-blue"></a>
  <a href="https://openreview.net/forum?id=kB29xkjNQK"><img alt="Paper: NeurIPS 2026" src="https://img.shields.io/badge/paper-NeurIPS%202026-8c1b13"></a>
</p>

`geosae` is a Python library for doing science with sparse autoencoders (SAEs) on Earth
embeddings: pixel-level representations such as AlphaEarth and Tessera, and patch tokens of
foundation models such as OlmoEarth and MMEarth. It builds patch stores of embeddings and reference
products for the cells you choose, and trains SAEs that rewrite each embedding with a few active
features from an overcomplete dictionary. It then evaluates any decomposition on the same pixels:
an SAE, a frozen-decoder control, or raw dimensions, PCA, ICA and NMF. Everything is plain PyTorch,
NumPy and Zarr, and every stage is both a Python function and a `geosae` command driven by one YAML
config.

## Features

**Data**

- Build a Zarr store of 64 x 64 patches of 10 m pixels for any cells of the
  [MajorTOM](https://github.com/ESA-PhiLab/Major-TOM) grid: a bounding box, a list of points or
  MajorTOM ids, or a seeded random sample of land cells, regional or global.
- 39 registered products: AlphaEarth and Tessera embeddings; land cover (ESA WorldCover, ESA CCI,
  Esri, GLC_FCS30D, GLanCE, MODIS), crops, forest, water, settlements, terrain, soils, climate
  (CHELSA) and ecological partitions (RESOLVE, potential natural vegetation), each reprojected onto
  the patch grid.
- Most sources are public and need no account, AlphaEarth included through a public mirror. Before
  a build starts, `geosae` checks the credentials of Google Earth Engine and NASA Earthdata
  products.
- Plan mode lists sources, licences, sizes and credentials before fetching anything. Builds resume
  where they stopped and can add products to an existing store.

**Models**

- BatchTopK, TopK, Matryoshka (nested prefixes) and ReLU+ℓ1 SAEs, and a frozen-decoder control
  whose random dictionary stays fixed while only the encoder trains.
- One `SparseAutoencoder` contract and a registry: an architecture registered under a name runs
  through training, inference and every evaluation unchanged.

**Training and inference**

- A single-device PyTorch trainer:
  - AuxK dead-feature revival;
  - warm-up and cosine decay;
  - bf16 on a GPU;
  - a controller that holds a ReLU+ℓ1 model at a target L0;
  - per-step pixel subsampling.
- Inference writes padded sparse codes and per-patch feature statistics to a Zarr store. It also
  reports FVE, cosine similarity and L0.

**Evaluation**

- Reconstruction at matched sparsity. A panel scores a whole ladder of runs and conventional
  baselines (raw, PCA, ICA, NMF) on one labelled pixel sample.
- Concept alignment:
  - folded single-feature AUROC against every reference class;
  - coverage counts (UCov, EMF) with width-matched and block-permutation nulls;
  - comparisons at matched capacity.
- Spatial scale: semivariance curves and correlation lengths per feature, and patch-mean Moran's I.
- Monosemanticity scores, sub-concept discovery validated against null partitions, and an
  area-of-applicability estimate scored on graded hold-outs.
- Further embeddings: OlmoEarth and MMEarth patch tokens computed from Sentinel-2, and the
  Copernicus-FM 0.25 degree map on a coarsened store.

**Reproducibility**

- General configs ship inside the wheel (`-c batchtopk`, `-c small`, `-c download`).
- Every run directory stores the resolved config it used, and every draw is seeded.

## Installation

```bash
pip install geosae
```

`geosae` needs Python 3.11 or newer. For a GPU, install a CUDA build of PyTorch first, following
[pytorch.org](https://pytorch.org/get-started/locally/), then `geosae`. Optional extras:

| Extra | For |
|---|---|
| `download` | building a patch store from the upstream sources |
| `embed` | encoding Sentinel-2 patches with OlmoEarth and MMEarth |
| `viz` | plotting (`matplotlib`, `nicefigs`) |
| `docs` | building the documentation |
| `dev` | tests and linting |

```bash
pip install "geosae[download]"
```

## Quickstart

On a laptop CPU, these commands take about ten minutes. They build a store of 225 patches around
Jena with AlphaEarth and ESA WorldCover (no account needed), train a small BatchTopK SAE, encode
the test split and score concept alignment:

```bash
geosae download -c download -o download.out=data/quickstart.zarr \
    -o "download.cells.bbox=[11.0,50.5,12.0,51.2]" -o "download.products=[alphaearth_cog,lc_esa_wc]"
export GEOSAE_STORE=data/quickstart.zarr
geosae train    -c batchtopk -c small   # a run directory under runs/batchtopk/
geosae predict  -c batchtopk -c small   # sparse codes of the test split
geosae coverage -c batchtopk -c small   # folded-AUROC concept alignment
```

The [Quickstart](https://earthyscience.github.io/geosae/quickstart/) page walks through the same
steps in Python and maps a feature.

## Documentation

The user guide and the API reference are at <https://earthyscience.github.io/geosae>:

- [Quickstart](https://earthyscience.github.io/geosae/quickstart/)
- [Building a dataset](https://earthyscience.github.io/geosae/dataset/)
- [Training](https://earthyscience.github.io/geosae/training/) and
  [Inference](https://earthyscience.github.io/geosae/inference/)
- [Concepts and factors](https://earthyscience.github.io/geosae/concepts/) and
  [Evaluation](https://earthyscience.github.io/geosae/evaluation/)
- [API reference](https://earthyscience.github.io/geosae/api/)

## NeurIPS 2026 paper

**Decomposing Earth Embeddings with Sparse Autoencoders**<br>
Vitus Benson, Fanny Yang, Markus Reichstein. *Advances in Neural Information Processing Systems
(NeurIPS 2026).*

Earth observation foundation models produce global embeddings, and each of their dimensions
entangles unrelated physical concepts. The paper proposes SAEs to decompose these embeddings into
sparse features over an overcomplete dictionary. It also proposes an evaluation protocol for
satellite pixels: it scores reconstruction at matched sparsity, alignment with reference concepts
and spatial scale, each against baselines that include an SAE with a frozen random decoder.

On AlphaEarth and Tessera, SAEs reconstruct the embeddings better than every baseline at matched
sparsity. Their features align with far fewer concepts than raw dimensions do, and they carry more
variance at fine spatial scales. The frozen-decoder control shares both of these properties, so
they follow from sparsity; only a learned dictionary reconstructs the embeddings accurately. Two
applications show how sparse features could be used: discovering land-cover sub-concepts, and
detecting out-of-distribution pixels by estimating an area of applicability. Neither is yet
reliable enough for practical use.

- The paper is on [OpenReview](https://openreview.net/forum?id=kB29xkjNQK).
- The [NeurIPS 2026 paper](https://earthyscience.github.io/geosae/neurips2026/) docs page covers
  the paper's configs and stages.
- [`scripts/neurips2026/`](https://github.com/EarthyScience/geosae/blob/main/scripts/neurips2026/README.md)
  holds the scripts behind every figure and table.
- The [Zenodo record](https://doi.org/10.5281/zenodo.24046835) holds the release source, every SAE
  checkpoint behind the paper and the input data of its figures and tables.

If you use `geosae`, please cite the paper:

Benson, V., Yang, F., & Reichstein, M. (2026). Decomposing Earth Embeddings with Sparse
Autoencoders. *Advances in Neural Information Processing Systems (NeurIPS 2026)*.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{benson2026decomposing,
  title     = {Decomposing Earth Embeddings with Sparse Autoencoders},
  author    = {Benson, Vitus and Yang, Fanny and Reichstein, Markus},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=kB29xkjNQK}
}
```

</details>

[`CITATION.cff`](https://github.com/EarthyScience/geosae/blob/main/CITATION.cff) holds the same
reference and the software record.

## Contributing

Issues and pull requests are welcome at
[github.com/EarthyScience/geosae](https://github.com/EarthyScience/geosae/issues). `make dev`
installs the package with its development tools and the git hooks. `make check` runs what CI
enforces: pre-commit, the test suite and a strict documentation build.

## License

MIT; see [LICENSE](https://github.com/EarthyScience/geosae/blob/main/LICENSE). The data sources
`geosae` downloads keep their own licences, which `geosae download -o download.plan=true` lists.
The MajorTOM grid arithmetic is adapted from [eodata](https://github.com/EarthyScience/eodata),
under the BSD 3-Clause License; see
[NOTICE](https://github.com/EarthyScience/geosae/blob/main/geosae/data/download/NOTICE.md).

## Acknowledgements

This work has received funding from the European Union under the Horizon Europe project
WeatherGenerator (grant agreement 101187947). The data downloads owe much to Sebastian Hoffmann and
Markus Zehner's work on eodata. `geosae` rests entirely on openly shared embeddings and reference
products; we thank the many teams who produce, curate and release them.

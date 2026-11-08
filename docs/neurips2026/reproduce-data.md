# Reproducing: data, checkpoints and training

This page sets up everything the figures of the paper need: the environment, the paper's data
store, the checkpoints and figure data on Zenodo, and the training sweep.
[Reproducing the figures and tables](reproduce-figures.md) then gives the commands for each figure
and table, from scratch or from the Zenodo figure data.

## Setting up

```bash
git clone https://github.com/EarthyScience/geosae && cd geosae
pip install -e ".[viz]" cartopy     # matplotlib, nicefigs; cartopy for the map panels
S=scripts/neurips2026
```

Run every command below from this directory. The evaluation stages cache their quality masks and
pixel samples under `data/`, relative to the working directory; a run from another directory
rebuilds them.

## Data

The paper's store is the `majortom210k` preset of `geosae download`, plus Sentinel-2 imagery for the
image panels. `build_store.py` builds both, resuming where an interrupted build stopped:

```bash
python $S/build_store.py --out data/majortom210k.zarr --gee-project <project> --max-workers 16
export GEOSAE_STORE=data/majortom210k.zarr      # every config of configs/neurips2026/ reads it
```

!!! tip
    `--rows 3` fetches the first three rows of every product in a few minutes, a check of every
    source and credential before the full build. The full build then resumes into the same store.

**The cells.** [`MAJORTOM210K`][geosae.data.download.presets.MAJORTOM210K] fixes them: a draw of
1,000,000 of the 5,418,682 cells of the MajorTOM 5 km land grid with seed 42, of which the store
keeps the leading 210,000 rows.

```python
from geosae.data.download import MAJORTOM210K, sample_subset_indices

print(MAJORTOM210K.cell_indices()[:3])
print(sample_subset_indices(5_418_682, 210_000, seed=42)[:3])
```

```
[2801816 3078819 5169808]
[4156278 2203848 4699964]
```

The draw's order depends on its size, so drawing 210,000 cells directly gives other cells; the
preset therefore draws all 1,000,000 and keeps the first rows.

**The splits.** The store carries two split arrays, index ranges over its rows:

| Split | test | train | val | Used by |
|---|---|---|---|---|
| `split_210k` | 0–100,000 | 100,000–200,000 | 200,000–210,000 | every AlphaEarth and Tessera experiment |
| `split_10k` | 0–10,000 | 100,000–110,000 | 200,000–202,000 | the Sentinel-2 subset of OlmoEarth and MMEarth |

A row without both embeddings is excluded from every split: 146 of the 210,000 rows, with the
sources as they were fetched for the paper. The rows were drawn in scattered order, so a contiguous
range is spread over the globe.

**Products.** The embeddings (AlphaEarth, Tessera) fill every row; the reference products, the
area-of-applicability covariates and the sub-concept partitions fill the 100,000 test rows, where
every concept metric is scored. See [Building a dataset](../dataset.md) for the product registry and
licences.

**Credentials.** AlphaEarth, GLC_FCS30D and WorldCereal are read from Google Earth Engine
(`earthengine authenticate`, then `--gee-project`); GLanCE needs a NASA Earthdata Login in
`~/.netrc` (`machine urs.earthdata.nasa.gov login <user> password <password>`).

**Size and time.** The store takes about 465 GB on disk, of which Tessera is 380 GB and AlphaEarth
62 GB. Tessera (one tile per cell, about 27 TB transferred) and Overture dominate the build: at 16
workers it takes several days. Run it as a CPU job and re-submit the same command if it stops.

**Preparing the evaluation.** Every evaluation stage below reads two per-embedding inputs: the table
of training pixels the dense baselines (PCA, ICA, NMF) are fitted on, and the frozen evaluation pool
that concept alignment is scored on.

```bash
for emb in alphaearth tessera; do
  geosae fit-table -c configs/neurips2026/$emb.yaml
  geosae pool      -c configs/neurips2026/$emb.yaml -c configs/neurips2026/eval_pool.yaml
done
```

## Checkpoints and figure data on Zenodo

The record [doi:10.5281/zenodo.24046835](https://doi.org/10.5281/zenodo.24046835) holds the
source of this release, every SAE checkpoint behind the paper, and the figure-data files the
camera-ready figures and tables were drawn from. Its README maps each file to a figure or table.

**Figure data.** Download the `*.json` and `*.npz` files of the record into one folder, here
`paper_data/`, and pass it as `--paper-data`. The record stores the files of the `products/`
subfolder as `products__<name>.json`; the scripts find either layout.

```bash
PD=paper_data
python $S/fig2_pareto.py --paper-data $PD --out figures
```

**Checkpoints.** The `geosae_checkpoints_<group>.zip` archives hold one `<name>.pth` and one
`<name>.yaml` per SAE; `sweep_alphaearth` and `sweep_tessera` hold the sweep, named
`<embedding>__<variant>_M<M>_k<k>_s<seed>`. The `.yaml` is the run's config. A checkpoint becomes a
run directory that every stage below accepts:

```bash
name=alphaearth__matryoshka_M1024_k16_s0
mkdir -p runs/alphaearth/matryoshka_M1024_k16_s0/zenodo
cp $name.pth  runs/alphaearth/matryoshka_M1024_k16_s0/zenodo/best.pth
cp $name.yaml runs/alphaearth/matryoshka_M1024_k16_s0/zenodo/config.yaml
geosae predict --run runs/alphaearth/matryoshka_M1024_k16_s0/zenodo -o data.zarr_path=$GEOSAE_STORE
```

!!! note
    The `data.zarr_path` of a Zenodo config names the store the model was trained on, on the
    authors' cluster. Override it with your store, as above, for `predict` and every stage that
    reads the run's config.

## Training the sweep

A trained SAE is a base config (`alphaearth.yaml` or `tessera.yaml`), a variant overlay
(`variants/{batchtopk,matryoshka,relu_l1,frozen}.yaml`), and the dictionary size `M`, sparsity `k`
and seed as overrides. `geosae run` trains and then predicts the test split into
`runs/<embedding>/<variant>_M<M>_k<k>_s<seed>/<timestamp>/` (`best.pth`, `config.yaml`,
`predictions/test.zarr`). Define one command per cell:

```bash
train() {  # train <embedding> <variant> <M> <k> <seed>
  local emb=$1 variant=$2 m=$3 k=$4 seed=$5 l1=()
  if [ "$variant" = relu_l1 ]; then  # starting l1 weight per k; the controller adapts it
    case $k in 8) l1=(-o train.l1_coeff=0.007);; 32) l1=(-o train.l1_coeff=0.003);; *) l1=(-o train.l1_coeff=0.005);; esac
  fi
  geosae run -c configs/neurips2026/$emb.yaml -c configs/neurips2026/variants/$variant.yaml "${l1[@]}" \
    -o model.latent_dim=$m -o model.k=$k -o seed=$seed -o name=$emb/${variant}_M${m}_k${k}_s${seed}
}
```

The full sweep has 108 runs: BatchTopK, Matryoshka and ReLU+ℓ1 at M = 1024 with k ∈ {8, 16, 32}
and at M ∈ {512, 2048} with k = 16, plus the frozen-decoder baseline at M = 1024 with
k ∈ {8, 16, 32}, each with seeds 0, 1, 2 on both embeddings:

```bash
for emb in alphaearth tessera; do for seed in 0 1 2; do
  for variant in batchtopk matryoshka relu_l1 frozen; do
    for mk in 1024:8 1024:16 1024:32 512:16 2048:16; do
      m=${mk%:*}; k=${mk#*:}
      [ "$variant" = frozen ] && [ "$m" != 1024 ] && continue
      train $emb $variant $m $k $seed
    done
  done
done; done
```

Each `train` call is one GPU job; on SLURM run the 108 cells as an array. Below, each figure lists
only the cells it needs, and `<run>` stands for a run directory such as
`runs/alphaearth/matryoshka_M1024_k16_s0/*`.

Retrained SAEs reproduce the paper within seed variation, not digit for digit: training is seeded,
but GPU kernels are not bit-deterministic, and a retrained dictionary orders its features
differently. The scripts therefore choose features by the rules below, never by index. The
evaluation samples are drawn from fixed seeds, so every model is scored on the same pixels.

### Choosing features

- **Regional features (Fig. 1c).** Each region box (central Australia, 122–138°E, 30–20°S; the
  Tibetan Plateau, 78–100°E, 28–37°N) gets the feature with the largest share of its summed
  prevalence inside the box, among features active on at least a fifth of the pixels of the box's
  patches on average.
- **Gallery features (Fig. 5a, Figs. S10, S11).** Per concept, the SAE feature and the raw
  dimension with the largest folded AUROC on the concept's reference class that are more active on
  the class than off it; no feature is shown twice, and a raw dimension low on the class is flipped
  in sign.
- **Woody features (Fig. 5b–f, Fig. S9, Table S8).** A woody feature reaches a folded AUROC of at
  least 0.65 on a forest, shrubland or savanna class and is more active on that class than
  elsewhere. Per Köppen-Geiger pair (Af+Am, BSh+Aw, Dfc+Dfb), the figure shows the woody feature
  whose 2,000 most active test patches lie most in that pair.

### Imagery

The image panels show the store's Sentinel-2 true colour by default, which needs no network.
`--esri` draws Esri World Imagery tiles instead, as in the paper; it needs network access, and
Esri's terms of use apply. With `--paper-data` the image panels always use Esri tiles at the
patch centres recorded in the figure data, and draw grey tiles offline.

## Compute

Times are for one NVIDIA A40.

| Step | Cost |
|---|---|
| Training, 108 runs | about 73 GPU-hours (median 22 min per AlphaEarth run, 58 min per Tessera run) |
| Encoding the test split (part of `geosae run`) | about 10 min and 13 GB of predictions per run |
| `geosae pool` | about 5–10 min per embedding (CPU) |
| `geosae panel` | about 3 min per run, so about 3 h per embedding for the 54 runs and the baselines |
| `geosae variogram` | about 10 min per dictionary |
| `geosae coverage` | 1–2 min per run |
| `geosae subconcepts` | under 15 min (CPU) |
| `geosae aoa` | the site census of all test patches, then a few hours per run for the graded and continent/PFT folds |
| Scripts | each under 5 min; the ones that read `predictions/` need up to about 16 GB of RAM; with `--paper-data`, seconds |

The first stage run against a store also scans it once for degenerate cells, which takes about
10–20 min. The masks are cached under `data/eval_cache/quality`.

# Reproducing the figures and tables

Each figure and table of the paper has one section below with the cells to train, the inference
and evaluation stages, the plotting command, and the command that draws it from the paper's figure
data instead. The commands assume the setup of
[Reproducing: data, checkpoints and training](reproduce-data.md): `S=scripts/neurips2026`, the
`train` function, `GEOSAE_STORE`, and `PD` pointing at the figure-data folder of the Zenodo record.

## Main figures

### Fig. 1: overview and two regional features

- **Train:** `train alphaearth matryoshka 1024 16 0`
- **Predict:** part of `geosae run`.
- **Evaluate:** nothing further.
- **Plot:**

    ```bash
    python $S/fig1_overview.py runs/alphaearth/matryoshka_M1024_k16_s0/* --out figures
    ```

- **From the paper data** (`regional_features.npz`):

    ```bash
    python $S/fig1_overview.py --paper-data $PD --out figures
    ```

### Fig. 2: reconstruction against sparsity

- **Train:** the full sweep, both embeddings.
- **Predict:** not needed; the panel runs every model on its own pixel sample.
- **Evaluate:** one panel per embedding, scoring every run and the dense baselines:

    ```bash
    for emb in alphaearth tessera; do
      geosae panel -c configs/neurips2026/$emb.yaml -c configs/neurips2026/panel.yaml \
        -o run.dir=eval/$emb/panel -o "panel.runs=[runs/$emb/*/*]"
    done
    ```

- **Plot:**

    ```bash
    python $S/fig2_pareto.py --panel-alphaearth eval/alphaearth/panel --panel-tessera eval/tessera/panel --out figures
    ```

- **From the paper data** (`real_products.json`, `dense_topk.json`):

    ```bash
    python $S/fig2_pareto.py --paper-data $PD --out figures
    ```

### Fig. 3: correlation lengths

One call draws Fig. 3 (`fig3_correlation_length`) and Figs. S3 and S4 (`figS_variogram_<embedding>`).

- **Train:** the M = 1024, k = 16 cells of every variant with seed 0, and the frozen decoder with
  seeds 1 and 2, on both embeddings:

    ```bash
    for emb in alphaearth tessera; do
      for v in batchtopk matryoshka relu_l1 frozen; do train $emb $v 1024 16 0; done
      train $emb frozen 1024 16 1; train $emb frozen 1024 16 2
    done
    ```

- **Predict:** not needed.
- **Evaluate:**

    ```bash
    for emb in alphaearth tessera; do
      geosae variogram -c configs/neurips2026/$emb.yaml -c configs/neurips2026/variogram.yaml \
        -o run.dir=eval/$emb/variogram \
        -o "variogram.runs=[runs/$emb/*_M1024_k16_s0/*,runs/$emb/frozen_M1024_k16_s[12]/*]"
    done
    ```

- **Plot:**

    ```bash
    python $S/fig3_variogram.py --variogram-alphaearth eval/alphaearth/variogram \
      --variogram-tessera eval/tessera/variogram --out figures
    ```

- **From the paper data** (`variogram_ae.npz`, `variogram_te.npz`):

    ```bash
    python $S/fig3_variogram.py --paper-data $PD --out figures
    ```

### Fig. 4: concept alignment

- **Train:** `train alphaearth matryoshka 1024 16 0` and `train alphaearth frozen 1024 16 0`.
- **Predict:** not needed.
- **Evaluate:** concept alignment on the evaluation pool:

    ```bash
    for run in runs/alphaearth/{matryoshka,frozen}_M1024_k16_s0/*; do
      geosae coverage --run $run -c configs/neurips2026/coverage.yaml
    done
    ```

- **Plot:**

    ```bash
    python $S/fig4_monosemanticity.py --runs runs/alphaearth --variant matryoshka --out figures
    ```

- **From the paper data** (`monosemanticity_matryoshka_ae.json`):

    ```bash
    python $S/fig4_monosemanticity.py --paper-data $PD --out figures
    ```

### Fig. 5: aligned features and sub-concepts

- **Train:** `train alphaearth matryoshka 1024 16 0`
- **Predict:** part of `geosae run`.
- **Evaluate:** `geosae coverage --run runs/alphaearth/matryoshka_M1024_k16_s0/* -c configs/neurips2026/coverage.yaml`.
  The random-partition p-values the text quotes come from
  `geosae subconcepts --run <run> -c configs/neurips2026/subconcepts.yaml`.
- **Plot** (also writes Table S8 into its JSON):

    ```bash
    python $S/fig5_aligned_features.py runs/alphaearth/matryoshka_M1024_k16_s0/* --esri --out figures
    ```

- **From the paper data** (`qualitative_ae.npz`, `subconcepts.npz`, `subconcepts.json`,
  `subconcepts_frozen.json`):

    ```bash
    python $S/fig5_aligned_features.py --paper-data $PD --out figures
    ```

### Fig. 6: area-of-applicability gain

- **Train:** BatchTopK at M = 1024, k = 16, seeds 0, 1, 2, on both embeddings:

    ```bash
    for emb in alphaearth tessera; do for seed in 0 1 2; do train $emb batchtopk 1024 16 $seed; done; done
    ```

- **Predict:** not needed.
- **Evaluate:** the graded hold-outs, with the random sparse code:

    ```bash
    for run in runs/{alphaearth,tessera}/batchtopk_M1024_k16_s*/*; do
      geosae aoa --run $run -c configs/neurips2026/aoa.yaml -c configs/neurips2026/aoa_graded.yaml
    done
    ```

- **Plot:**

    ```bash
    python $S/fig6_aoa_gain.py --alphaearth runs/alphaearth/batchtopk_M1024_k16_s*/* \
      --tessera runs/tessera/batchtopk_M1024_k16_s*/* --out figures
    ```

- **From the paper data** (`applicability.json`):

    ```bash
    python $S/fig6_aoa_gain.py --paper-data $PD --out figures
    ```

## Appendix figures

### Fig. S1: dataset overview

- **Train, predict, evaluate:** nothing; the figure reads only the store.
- **Plot:** `python $S/figS_dataset_overview.py $GEOSAE_STORE --out figures`
- **From the paper data:** not available; the figure needs the store.

### Fig. S2: per-dimension reconstruction

- **Train:** `train alphaearth $v 1024 16 0` for `v` in `matryoshka`, `batchtopk`, `relu_l1`.
- **Predict, evaluate:** nothing; the script runs each SAE forward on three patches of the store.
- **Plot:** `python $S/figS_recon_quality.py runs/alphaearth --esri --out figures`
- **From the paper data:** not available; the figure needs the store and the three checkpoints,
  which the Zenodo record has (see [Checkpoints](reproduce-data.md#checkpoints-and-figure-data-on-zenodo)).

### Figs. S3, S4: spatial scales on AlphaEarth and Tessera

Written by the Fig. 3 command; see [Fig. 3](#fig-3-correlation-lengths).

### Fig. S5: patch-mean Moran's I

- **Train:** the M = 1024, k = 16 cells of every variant with seed 0, and the frozen decoder with
  seeds 1 and 2, on both embeddings, as for [Fig. 3](#fig-3-correlation-lengths).
- **Predict:** part of `geosae run`.
- **Evaluate:** nothing further; the script computes Moran's I from the predictions.
- **Plot:**

    ```bash
    python $S/figS_moran.py --runs-alphaearth runs/alphaearth --runs-tessera runs/tessera --out figures
    ```

- **From the paper data** (`spatial_coherence.npz`):

    ```bash
    python $S/figS_moran.py --paper-data $PD --out figures
    ```

### Fig. S6: principal components against SAE atoms

- **Train:** `train alphaearth matryoshka 1024 16 0`
- **Predict, evaluate:** nothing; the script reads the fitting table of `geosae fit-table`.
- **Plot:** `python $S/figS_pcs_vs_atoms.py --runs runs/alphaearth --device cpu --out figures`
- **From the paper data** (`pcs_vs_atoms.npz`):

    ```bash
    python $S/figS_pcs_vs_atoms.py --paper-data $PD --out figures
    ```

### Fig. S7: concept alignment on Tessera

- **Train:** `train tessera matryoshka 1024 16 0` and `train tessera frozen 1024 16 0`.
- **Evaluate:** `geosae coverage --run <run> -c configs/neurips2026/coverage.yaml` on both runs.
- **Plot:**

    ```bash
    python $S/fig4_monosemanticity.py --runs runs/tessera --name figS_mono_tessera --out figures
    ```

- **From the paper data** (`monosemanticity_matryoshka_te.json`):

    ```bash
    python $S/fig4_monosemanticity.py --paper-data $PD --embedding tessera --name figS_mono_tessera --out figures
    ```

### Fig. S8: concept alignment of BatchTopK

- **Train:** `train alphaearth batchtopk 1024 16 0` and `train alphaearth frozen 1024 16 0`.
- **Evaluate:** `geosae coverage --run <run> -c configs/neurips2026/coverage.yaml` on both runs.
- **Plot:**

    ```bash
    python $S/fig4_monosemanticity.py --runs runs/alphaearth --variant batchtopk --name figS_mono_batchtopk --out figures
    ```

- **From the paper data** (`monosemanticity_batchtopk_ae.json`):

    ```bash
    python $S/fig4_monosemanticity.py --paper-data $PD --variant batchtopk --name figS_mono_batchtopk --out figures
    ```

### Fig. S9: woody features of the frozen decoder

- **Train:** `train alphaearth frozen 1024 16 0`
- **Predict:** part of `geosae run`.
- **Evaluate:** `geosae coverage --run runs/alphaearth/frozen_M1024_k16_s0/* -c configs/neurips2026/coverage.yaml`
- **Plot:**

    ```bash
    python $S/figS_subconcepts_frozen.py runs/alphaearth/frozen_M1024_k16_s0/* --esri --out figures
    ```

- **From the paper data** (`subconcepts_frozen.npz`, `subconcepts_frozen.json`):

    ```bash
    python $S/figS_subconcepts_frozen.py --paper-data $PD --out figures
    ```

### Figs. S10, S11: galleries on AlphaEarth and Tessera

- **Train:** `train $emb matryoshka 1024 16 0` for both embeddings.
- **Evaluate:** `geosae coverage --run <run> -c configs/neurips2026/coverage.yaml` on both runs.
- **Plot:**

    ```bash
    python $S/figS_galleries.py runs/alphaearth/matryoshka_M1024_k16_s0/* --esri --out figures   # figS_qual_only
    python $S/figS_galleries.py runs/tessera/matryoshka_M1024_k16_s0/* --esri --out figures      # figS_qual_only_tessera
    ```

    Add `--device cpu` on a machine without a GPU.

- **From the paper data** (`qualitative_ae.npz`, `qualitative_te.npz`):

    ```bash
    python $S/figS_galleries.py --paper-data $PD --out figures
    python $S/figS_galleries.py --paper-data $PD --embedding tessera --out figures
    ```

### Figs. S12, S13: area of applicability with Europe held out

- **Train:** as for [Fig. 6](#fig-6-area-of-applicability-gain), plus `train alphaearth matryoshka 1024 16 0`.
- **Evaluate:** the continent and plant-functional-type folds, then the leave-Europe-out maps of
  [Table S9](#table-s9-area-of-applicability-with-europe-held-out), which write `figures/aoa_europe.npz`:

    ```bash
    for run in runs/alphaearth/batchtopk_M1024_k16_s*/* runs/alphaearth/matryoshka_M1024_k16_s0/*; do
      geosae aoa --run $run -c configs/neurips2026/aoa.yaml
    done
    python $S/table_aoa_europe.py runs/alphaearth/batchtopk_M1024_k16_s*/* \
      --map-run runs/alphaearth/matryoshka_M1024_k16_s0/* --out figures
    ```

- **Plot:**

    ```bash
    python $S/figS_aoa_gain.py figures/aoa_europe.npz --alphaearth runs/alphaearth/batchtopk_M1024_k16_s*/* \
      --tessera runs/tessera/batchtopk_M1024_k16_s*/* --out figures                  # Fig. S12
    python $S/figS_aoa.py figures/aoa_europe.npz --out figures                        # Fig. S13
    ```

- **From the paper data** (`aoa_europe.npz`, `applicability.json`):

    ```bash
    python $S/figS_aoa_gain.py --paper-data $PD --out figures
    python $S/figS_aoa.py --paper-data $PD --out figures
    ```

## Appendix tables

Tables S1 (data sources) and S2 (concept groups) describe the setup and report no results.

### Table S3: reconstruction at L0 = 16

- **Train, evaluate:** as for [Fig. 2](#fig-2-reconstruction-against-sparsity).
- **Plot:**

    ```bash
    python $S/table_reconstruction.py --panel-alphaearth eval/alphaearth/panel \
      --panel-tessera eval/tessera/panel --out figures
    ```

- **From the paper data** (`recon_table.json`):

    ```bash
    python $S/table_reconstruction.py --paper-data $PD --out figures
    ```

!!! note
    The paper's table is scored on the cleaned test pixels of `recon_table.json`, Fig. 2 on the
    panel's sample, so the two differ in the third digit (PCA with 16 components: 0.696 in the
    table, 0.697 in the figure). Fresh panels reproduce Fig. 2's sample.

### Tables S4, S5: further embeddings

- **Data:** the Sentinel-2 subset with the OlmoEarth and MMEarth tokens, and the 0.25° Copernicus-FM
  store (see [Building a dataset](../dataset.md)):

    ```bash
    geosae subset -c configs/download.yaml -o subset.stage=fetch     # CPU and network
    geosae subset -c configs/download.yaml -o subset.stage=embed     # one GPU; the embed extra
    geosae subset -c configs/download.yaml -o subset.stage=stats
    geosae gridded -c configs/download.yaml
    ```

- **Train:** BatchTopK SAEs and frozen-decoder baselines, seeds 42, 43, 44: OlmoEarth (M = 4096,
  k ∈ {16, 192}) and MMEarth (M = 4096, k ∈ {16, 80}) on the tokens, Copernicus-FM (M = 1024,
  k ∈ {16, 192}) on its grid:

    ```bash
    E=configs/neurips2026/embeddings
    for seed in 42 43 44; do for v in batchtopk frozen; do
      for k in 16 192; do
        geosae run -c $E/patch_tokens.yaml -c $E/olmoearth.yaml -c configs/neurips2026/variants/$v.yaml \
          -o model.latent_dim=4096 -o model.k=$k -o seed=$seed -o name=olmoearth/${v}_M4096_k${k}_s${seed}
        geosae run -c $E/gridded.yaml -c $E/copernicusfm.yaml -c configs/neurips2026/variants/$v.yaml \
          -o model.latent_dim=1024 -o model.k=$k -o seed=$seed -o name=copernicusfm/${v}_M1024_k${k}_s${seed}
      done
      for k in 16 80; do
        geosae run -c $E/patch_tokens.yaml -c $E/mmearth.yaml -c configs/neurips2026/variants/$v.yaml \
          -o model.latent_dim=4096 -o model.k=$k -o seed=$seed -o name=mmearth/${v}_M4096_k${k}_s${seed}
      done
    done; done
    ```

- **Evaluate:** a panel per embedding with baselines at the SAEs' L0, and concept alignment of the
  token runs:

    ```bash
    for emb in olmoearth mmearth; do
      ks=$([ $emb = mmearth ] && echo 16,80 || echo 16,192)
      geosae panel -c configs/neurips2026/panel.yaml -c $E/patch_tokens.yaml -c $E/$emb.yaml \
        -o run.dir=eval/$emb/panel -o "panel.runs=[runs/$emb/*/*]" -o "panel.baselines.widths=[$ks]"
      for run in runs/$emb/*/*; do geosae coverage --run $run; done
    done
    geosae panel -c configs/neurips2026/panel.yaml -c $E/gridded.yaml -c $E/copernicusfm.yaml \
      -o run.dir=eval/copernicusfm/panel -o "panel.runs=[runs/copernicusfm/*/*]" -o "panel.baselines.widths=[16,192]"
    ```

- **Plot:**

    ```bash
    python $S/table_further_embeddings.py --panel-copernicusfm eval/copernicusfm/panel \
      --panel-olmoearth eval/olmoearth/panel --panel-mmearth eval/mmearth/panel \
      --runs-olmoearth runs/olmoearth --runs-mmearth runs/mmearth --out figures
    ```

- **From the paper data** (`products/{copernicusfm,olmoearth,mmearth}.json`):

    ```bash
    python $S/table_further_embeddings.py --paper-data $PD --out figures
    ```

### Table S6: UCov

- **Train:** the M = 1024, k = 16 cells of every variant on both embeddings.
- **Evaluate:** concept alignment of every M = 1024, k = 16 run, and the untrained random dictionary
  on one BatchTopK run:

    ```bash
    for run in runs/{alphaearth,tessera}/*_M1024_k16_s*/*; do
      geosae coverage --run $run -c configs/neurips2026/coverage.yaml
    done
    geosae coverage --run runs/alphaearth/batchtopk_M1024_k16_s0/* -c configs/neurips2026/coverage.yaml \
      -o "coverage.random_widths=[1024]"
    ```

- **Plot:** `python $S/table_ucov.py --runs runs/alphaearth runs/tessera --out figures`
- **From the paper data** (`tables.json`, `null_coverage.json`):

    ```bash
    python $S/table_ucov.py --paper-data $PD --out figures
    ```

### Table S7: monosemanticity scores of Betti et al.

- **Train:** the M = 1024, k = 16 cells of every variant, seeds 0, 1, 2, on both embeddings.
- **Predict:** part of `geosae run`.
- **Evaluate:** nothing further; the script scores the predictions.
- **Plot:** `python $S/table_betti.py --runs runs/alphaearth runs/tessera --out figures`. MS also
  needs image embeddings of the test patches (`--image-embeddings`, see `-h`); without them the
  script reports GeoMS only.
- **From the paper data** (`betti_scores_ae.json`, `betti_scores_te.json`, `betti_relu_fixed.json`):

    ```bash
    python $S/table_betti.py --paper-data $PD --out figures
    ```

### Table S8: sub-concept agreement

- **Train, evaluate:** as for [Fig. 5](#fig-5-aligned-features-and-sub-concepts) and
  [Fig. S9](#fig-s9-woody-features-of-the-frozen-decoder).
- **Plot:**

    ```bash
    python $S/table_subconcepts.py runs/alphaearth/matryoshka_M1024_k16_s0/* \
      runs/alphaearth/frozen_M1024_k16_s0/* --out figures
    ```

- **From the paper data** (`subconcepts.json`, `subconcepts_frozen.json`):

    ```bash
    python $S/table_subconcepts.py --paper-data $PD --out figures
    ```

### Table S9: area of applicability with Europe held out

- **Train, evaluate, plot:** the `table_aoa_europe.py` command of
  [Figs. S12, S13](#figs-s12-s13-area-of-applicability-with-europe-held-out); it writes
  `table_aoa_europe.tex` and `.json`.
- **From the paper data:** not available. The table re-scores the held-out sites each `geosae aoa`
  run caches (`eval/aoa/sites.npz`), which the record does not hold; `aoa_europe.npz` carries the
  leave-Europe-out numbers of the mapped Matryoshka model only.

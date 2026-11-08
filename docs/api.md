# API reference

Every public module of `geosae`, generated from its docstrings.

## geosae

The package, its configs, run directories, the CLI and the config-to-object helpers.

::: geosae
    options:
      members: false

::: geosae.config

::: geosae.run

::: geosae.pipeline

::: geosae.cli

## geosae.models

The model contract, the registry and the built-in architectures.

::: geosae.models
    options:
      members: false

::: geosae.models.base

::: geosae.models.registry

::: geosae.models.topk_sae

::: geosae.models.relu_sae

## geosae.data

The patch dataset, its transforms and quality filters.

::: geosae.data
    options:
      members: false

::: geosae.data.dataset

::: geosae.data.quality

## geosae.data.download

Building a patch store: choosing cells, the product registry, plans, the store writer and presets.

::: geosae.data.download
    options:
      members: false

::: geosae.data.download.grid

::: geosae.data.download.selection

::: geosae.data.download.sources

::: geosae.data.download.store

::: geosae.data.download.presets

::: geosae.data.download.gridded

::: geosae.data.download.plan

::: geosae.data.download.release

## geosae.data.download.fetchers

One fetcher per source; build them with
[`build_fetcher()`](#geosae.data.download.sources.build_fetcher).

::: geosae.data.download.fetchers
    options:
      members: false

::: geosae.data.download.fetchers.raster_fetchers

::: geosae.data.download.fetchers.reproject

::: geosae.data.download.fetchers.alpha_earth_gee

::: geosae.data.download.fetchers.alpha_earth_cog

::: geosae.data.download.fetchers.tessera_embeddings

::: geosae.data.download.fetchers.gee_image

::: geosae.data.download.fetchers.sentinel2

::: geosae.data.download.fetchers.copernicus_fm

::: geosae.data.download.fetchers.overture

::: geosae.data.download.fetchers.vector

::: geosae.data.download.fetchers.natural_forest

## geosae.data.embed

Foundation-model backbones for patch-token embeddings.

::: geosae.data.embed
    options:
      members: false

::: geosae.data.embed.base

::: geosae.data.embed.store

::: geosae.data.embed.subset

::: geosae.data.embed.olmoearth

::: geosae.data.embed.mmearth

## geosae.train

Loss, trainer, sparsity controller and pixel subsampler.

::: geosae.train
    options:
      members: false

::: geosae.train.loss

::: geosae.train.trainer

::: geosae.train.sparsity_controller

::: geosae.train.pixel_subsample

## geosae.predict

Sparse codes and the prediction store.

::: geosae.predict
    options:
      members: false

::: geosae.predict.sparsify

::: geosae.predict.writer

## geosae.eval — reconstruction and the panel

::: geosae.eval
    options:
      members: false

::: geosae.eval.metrics

::: geosae.eval.ladder

::: geosae.eval.blocks

::: geosae.eval.statistic

## geosae.eval — concept alignment

::: geosae.eval.registry

::: geosae.eval.pool

::: geosae.eval.coverage

::: geosae.eval.concepts

::: geosae.eval.landcover_classes

::: geosae.eval.covariates

::: geosae.eval.subconcepts

::: geosae.eval.monosemanticity

## geosae.eval — spatial scale

::: geosae.eval.variogram

::: geosae.eval.spatial

::: geosae.eval.auroc_moran

## geosae.eval — area of applicability

::: geosae.eval.aoa

::: geosae.eval.holdouts

::: geosae.eval.extrapolation

::: geosae.eval.selective

## geosae.eval — conventional baselines

::: geosae.eval.baselines

::: geosae.eval.dictionaries

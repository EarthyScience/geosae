"""``geosae`` command-line interface.

Every subcommand is config-driven (``-c cfg.yaml``, repeatable and merged left to right, plus
``-o key=value`` overrides) and writes into one run directory (see :mod:`geosae.run`):

* ``geosae train      -c cfg.yaml`` — fresh run dir; checkpoints, metrics, resolved config.
* ``geosae predict    -c cfg.yaml`` — sparse codes of one split + reconstruction metrics.
* ``geosae run        -c cfg.yaml`` — train, then predict, in one run dir.
* ``geosae panel      -c cfg.yaml`` — the interpretability panel of a ladder of runs and baselines.
* ``geosae coverage   -c cfg.yaml`` — folded-AUROC concept alignment, UCov and concept groups.
* ``geosae pool       -c cfg.yaml`` — build the frozen evaluation pool coverage draws from.
* ``geosae variogram  -c cfg.yaml`` — semivariance curves and correlation lengths of every feature.
* ``geosae subconcepts -c cfg.yaml`` — sub-concept discovery and validation against null partitions.
* ``geosae aoa        -c cfg.yaml`` — area-of-applicability sweep, incl. graded hold-outs.
* ``geosae stats      -c cfg.yaml`` — embedding normalisation statistics for the transforms.
* ``geosae fit-table  -c cfg.yaml`` — one cached pixel per training patch, for the baseline fits.
* ``geosae download   -c cfg.yaml`` — build (or resume) a patch store of chosen cells from the original sources.
* ``geosae subset     -c cfg.yaml`` — Sentinel-2 subset of a store, encoded by foundation models.
* ``geosae gridded    -c cfg.yaml`` — coarse 0.25 degree store with the Copernicus-FM embedding map.
* ``geosae refetch    -c cfg.yaml`` — re-fetch one reference product into an existing store.
* ``geosae release    -c cfg.yaml`` — export the dataset recipe: cell ids, splits, products, concepts.

``-c`` takes a path or the name of a bundled config (``-c batchtopk``; see
:func:`geosae.config.resolve_config`). ``device: auto`` resolves to ``cuda`` when a GPU is visible,
else ``cpu``. ``--run <dir>`` uses a run directory's own ``config.yaml`` as the base config and
targets that run; otherwise every stage but ``train`` reuses the latest run of ``name`` (or
``-o run.dir=...``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geosae import __version__
from geosae import run as R
from geosae.config import load_config, save_config
from geosae.data.dataset import build_transforms, resolve_split_name
from geosae.models import build_model, load_checkpoint
from geosae.pipeline import (
    apply_transforms,
    baseline_fit_pixels_from_config,
    build_model_from_cfg,
    expand_run_dirs,
    load_run_model,
    load_run_transforms,
    make_data,
    real_eval_sample,
    run_provenance,
    store_data,
)


def cmd_train(cfg: dict, run_dir: Path) -> None:
    """Train the config's model; writes ``config.yaml``, checkpoints, ``metrics.jsonl`` and ``eval/val_metrics.csv``."""
    from geosae.train.trainer import Trainer

    P = R.paths(run_dir)
    save_config(cfg, str(P["config"]))
    t = cfg["train"]
    train_data = make_data(cfg, "train", compute=t.get("compute_inram", True), max_cells=t.get("max_cells"))
    val_data = make_data(cfg, "val", compute=True, max_cells=t.get("val_max_cells"))
    print(f"[geosae] run dir: {run_dir}", flush=True)
    print(f"[geosae] train={len(train_data)} val={len(val_data)} patches", flush=True)
    model = build_model_from_cfg(cfg)
    best = Trainer(
        model,
        train_data,
        val_data,
        outdir=str(run_dir),
        model_cfg=dict(cfg["model"]),
        lr=float(t.get("lr", 1e-3)),
        beta1=float(t.get("beta1", 0.0)),
        beta2=float(t.get("beta2", 0.999)),
        eps=float(t.get("eps", 1e-8)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        clip_grad_norm=float(t.get("clip_grad_norm", 1.0)),
        batch_size=int(t.get("batch_size", 128)),
        epochs=int(t.get("epochs", 50)),
        steps_per_epoch=t.get("steps_per_epoch"),
        rampup=int(t.get("rampup", 1000)),
        aux_penalty=float(t.get("aux_penalty", 0.03125)),
        top_k_aux=int(t.get("top_k_aux", 32)),
        l1_coeff=float(t.get("l1_coeff", 0.0)),
        slow_group_weight=float(t.get("slow_group_weight", 0.0)),
        target_l0=t.get("target_l0"),
        penalty_warmup_steps=int(t.get("penalty_warmup_steps", 0)),
        seed=int(cfg.get("seed", 42)),
        device=cfg.get("device", "cuda"),
        amp=bool(t.get("amp", True)),
        val_batch_size=int(t.get("val_batch_size", 64)),
        log_every=int(t.get("log_every", 50)),
        pixel_subsample=t.get("pixel_subsample"),
        num_workers=int(t.get("num_workers", 8)),
        prefetch_factor=int(t.get("prefetch_factor", 4)),
        pin_memory=bool(t.get("pin_memory", True)),
        persistent_workers=bool(t.get("persistent_workers", False)),
    ).fit()
    R.write_metrics_csv(P["eval"] / "val_metrics.csv", [{"split": "val", **best}])


def cmd_predict(cfg: dict, run_dir: Path) -> str:
    """Encode one split with the run's checkpoint into ``predictions/<split>.zarr``, plus its reconstruction metrics."""
    from geosae.eval.metrics import reconstruction_metrics
    from geosae.predict.writer import run_prediction

    P = R.paths(run_dir)
    p = cfg.get("predict") or {}
    split = p.get("split", "test")
    model = build_model_from_cfg(cfg)
    ckpt_path = p.get("checkpoint") or str(P["best"])
    ckpt = load_checkpoint(model, ckpt_path)
    print(f"[geosae] loaded {ckpt_path} (epoch {ckpt.get('epoch')})", flush=True)

    data = make_data(cfg, split, compute=p.get("compute_inram", False), max_cells=p.get("max_cells"))
    print(f"[geosae] predict {len(data)} patches (split={split}) -> {P['predictions']}", flush=True)
    path = run_prediction(
        model,
        data,
        outdir=str(P["predictions"]),
        name=split,
        split=split,
        batch_size=int(p.get("batch_size", 64)),
        max_k=p.get("max_k"),
        allow_truncation=bool(p.get("allow_truncation", False)),
        device=cfg.get("device", "cuda"),
        chunk_cells=int(p.get("chunk_cells", 50)),
        num_workers=int(p.get("num_workers", 8)),
        prefetch_factor=int(p.get("prefetch_factor", 4)),
        pin_memory=bool(p.get("pin_memory", True)),
    )
    metrics = reconstruction_metrics(
        model, data, batch_size=int(p.get("batch_size", 64)), device=cfg.get("device", "cuda")
    )
    R.write_metrics_csv(P["eval"] / f"{split}_metrics.csv", [{"split": split, **metrics}])
    print(
        f"[geosae] {split} metrics: FVE={metrics['fve']:.4f} cos={metrics['recon_cosine']:.4f} L0={metrics['l0']:.1f}",
        flush=True,
    )
    return path


def cmd_panel(cfg: dict, run_dir: Path) -> list[dict]:
    """Score the interpretability panel for a whole ladder of decompositions on one evaluation sample.

    The labelled sample is read once (:func:`~geosae.pipeline.real_eval_sample`, options from the
    ``panel`` block) and re-wrapped per row with that row's own ``data.transforms``, so every row is
    scored in the units its model was trained on. ``panel.runs`` lists run directories or glob
    patterns over them (trained SAEs and frozen-decoder baselines alike); ``panel.baselines.methods``
    and ``panel.baselines.widths`` add conventional decompositions
    (:func:`~geosae.eval.dictionaries.fit_dictionary`) fitted on the training pixels of
    :func:`~geosae.pipeline.baseline_fit_pixels_from_config`, a dense basis thresholded per atom to
    fire on ``baselines.activation_rate`` (default ``model.k / width``) of the pixels;
    ``panel.baselines.ranks`` adds, per fitted baseline and rank ``k``, the rank-``k`` truncation onto
    its ``k`` atoms of largest code variance. ``panel.blocks`` selects the blocks of
    :data:`~geosae.eval.blocks.BLOCK_SCALARS` (default all).

    ``panel.merge_shards`` skips scoring and joins the ``eval/panel.json`` of other panel run
    directories (paths or glob patterns, later shards win a duplicated row name) — the counterpart to
    scoring disjoint slices of ``panel.runs`` as separate jobs.

    Writes ``eval/panel.json`` and ``eval/panel.csv``, one row per decomposition, rewritten after
    every row.
    """
    from geosae.eval.blocks import BLOCK_SCALARS
    from geosae.eval.dictionaries import fit_dictionary
    from geosae.eval.ladder import (
        dictionary_encoder,
        leading_atoms,
        merge_panel_shards,
        model_encoder,
        model_offset,
        rank_encoder,
        real_ladder_sample_from_registry,
        score_decomposition,
        write_panel_table,
    )

    p = cfg.get("panel") or {}
    P = R.paths(run_dir)
    if p.get("merge_shards"):
        patterns = [p["merge_shards"]] if isinstance(p["merge_shards"], str) else list(p["merge_shards"])
        shards = [
            Path(h) for pat in patterns for h in (sorted(Path().glob(pat)) if any(c in pat for c in "*?[") else [pat])
        ]
        paths = [d / "eval" / "panel.json" for d in dict.fromkeys(shards)]
        missing = [str(sp) for sp in paths if not sp.exists()]
        if missing:
            raise FileNotFoundError(f"panel.merge_shards: no eval/panel.json at {missing}")
        rows, provenance = merge_panel_shards(paths)
        write_panel_table(P["eval"], rows, provenance)
        print(f"[geosae] panel merge: {len(paths)} shards, {len(rows)} rows -> {P['eval'] / 'panel.csv'}", flush=True)
        return rows

    device = cfg.get("device", "cuda")
    zarr_path = str(cfg["data"]["zarr_path"])
    run_dirs = expand_run_dirs(p.get("runs") or [])
    blocks = tuple(int(b) for b in (p.get("blocks") or tuple(BLOCK_SCALARS)))
    ev = real_eval_sample(cfg, p)
    if not ev.concepts_available:
        blocks = tuple(b for b in blocks if b != 3)
    kw = {
        "codes_sample_pixels": int(p.get("codes_sample_pixels", 20_000)),
        "min_positive": int(p.get("min_positive", 50)),
        "seed": int(cfg.get("seed", 42)),
        "blocks": blocks,
    }
    provenance = {
        "geosae_version": __version__,
        "store": zarr_path,
        "split": p.get("split", "test"),
        "n_cells": ev.n_cells,
        "pixels_per_cell": ev.pixels_per_cell,
        "sample_seed": int(p.get("sample_seed", 0)),
        "concepts_available": ev.concepts_available,
        "blocks": list(blocks),
        "scalars": [s for b in blocks for s in BLOCK_SCALARS[b]],
    }
    print(f"[geosae] panel: {len(run_dirs)} runs on {zarr_path}, blocks {list(blocks)}", flush=True)

    rows: list[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        print(
            f"[geosae]   {row['name']:40s} FVE={row.get('fve', float('nan')):.4f} L0={row.get('l0', float('nan')):.1f} "
            f"UCov={row.get('ucov', float('nan')):.0f} ({row['seconds']:.0f}s)",
            flush=True,
        )
        write_panel_table(P["eval"], rows, provenance)

    for d in run_dirs:
        model = load_run_model(d, cfg)
        sample = real_ladder_sample_from_registry(
            ev.sample, key=f"{ev.key}#{d.name}", transforms=load_run_transforms(d, cfg)
        )
        t0 = time.perf_counter()
        panel = score_decomposition(
            sample, model_encoder(model, device=device), model.dictionary(), offset=model_offset(model), **kw
        )
        emit({"name": d.name, "kind": "sae", **run_provenance(d), **panel, "seconds": time.perf_counter() - t0})
        del model

    b = p.get("baselines") or {}
    if b.get("methods"):
        # The baselines are fitted in the units of this config's own data.transforms.
        transforms = build_transforms(cfg["data"].get("transforms"), zarr_path=zarr_path)
        sample = real_ladder_sample_from_registry(ev.sample, key=f"{ev.key}#baselines", transforms=transforms)
        x_fit = baseline_fit_pixels_from_config(cfg, b, zarr_path, seed=int(p.get("sample_seed", 0)))
        for method in b["methods"]:
            for width in [int(w) for w in (b.get("widths") or [256])]:
                for seed in [int(s) for s in (b.get("seeds") or [0])]:
                    t0 = time.perf_counter()
                    fit = fit_dictionary(x_fit, method, width, seed=seed, max_iter=int(b.get("max_iter", 200)))
                    fit_seconds = time.perf_counter() - t0
                    rate = (
                        float(b["activation_rate"])
                        if b.get("activation_rate")
                        else (float((cfg.get("model") or {}).get("k", 16)) / max(fit.n_atoms, 1))
                    )
                    meta = {
                        "kind": method,
                        "arch": method,
                        "seed": seed,
                        "latent_dim": fit.n_atoms,
                        "frozen": method == "random",
                        "store": zarr_path,
                        "fit_seconds": fit_seconds,
                    }
                    t0 = time.perf_counter()
                    panel = score_decomposition(
                        sample, dictionary_encoder(fit, activation_rate=rate), fit.atoms, offset=fit.offset, **kw
                    )
                    emit({"name": f"{method}_M{width}_s{seed}", **meta, **panel, "seconds": time.perf_counter() - t0})
                    for rank in [int(r) for r in (b.get("ranks") or [])]:
                        t0 = time.perf_counter()
                        keep = leading_atoms(fit, x_fit, rank)
                        panel = score_decomposition(sample, rank_encoder(fit, keep), fit.atoms, offset=fit.offset, **kw)
                        emit(
                            {
                                "name": f"{method}_M{width}_rank{rank}_s{seed}",
                                **meta,
                                "rank": rank,
                                **panel,
                                "seconds": time.perf_counter() - t0,
                            }
                        )
    print(f"[geosae] panel: {len(rows)} rows -> {P['eval'] / 'panel.csv'}", flush=True)
    return rows


def cmd_coverage(cfg: dict, run_dir: Path) -> dict:
    """Score concept coverage on a fixed pixel sample, width-fairly.

    ``coverage.pool`` points at a saved :class:`~geosae.eval.pool.EvalPool`: the
    pool is verified against its store, its pixels and its **frozen** per-concept
    balanced draws are used, and nothing is re-sampled. Without it the stage
    falls back to an ad-hoc draw (``n_patches``, ``pixels_per_patch``, ``seed``,
    ``products``, ``min_spatial_std``, ``drop_flat_pixel_cells`` and ``label_store``
    as for ``geosae panel``) cached at ``coverage.sample_path`` — reproducible,
    but not auditable against a pool id, so the stage warns. A pool's pixels are
    read through a content-addressed sample cache (:func:`~geosae.eval.pool.default_sample_cache_path`, rooted at
    ``coverage.cache_root``), so this stage and ``geosae pool`` reuse the same
    materialised bytes for the same spec without coordinating a path.

    Either way the sample, in the units of the config's ``data.transforms``, is
    encoded with the run's ``best.pth`` (or ``coverage.checkpoint``) and reduced to the
    resulting AUROC matrix to UCov/EMF, the calibrated coverage ``ccov`` against
    a width-matched best-of-M null, and the matched-capacity recomputation at
    every ``coverage.matched_budgets`` width. The dense reference bases in
    ``coverage.dense_bases`` are scored on the same pixels.

    Writes ``eval/<split>_coverage.json``, ``eval/<split>_coverage.csv`` and
    ``eval/<split>_coverage_arrays.npz`` (AUROC matrices, per-concept bests and
    null draws, plus the ``class_table`` and ``product_order`` naming the AUROC
    columns, so the scalars can be re-reduced without re-encoding).

    Returns:
        The dict that was written as JSON.
    """
    from geosae.eval.baselines import project_to_basis
    from geosae.eval.coverage import coverage_report, matched_budget_subset
    from geosae.eval.registry import RegistrySample, calibrate_sparsity, encode_sample, sample_registry_pixels

    P = R.paths(run_dir)
    c = cfg.get("coverage") or {}
    d = cfg["data"]
    split = c.get("split") or (cfg.get("predict") or {}).get("split", "test")
    P["eval"].mkdir(parents=True, exist_ok=True)
    pool_path = c.get("pool")
    sample_path = Path(c.get("sample_path") or P["eval"] / f"{split}_registry_sample.npz")

    pool = None
    if pool_path:
        from geosae.eval.pool import default_sample_cache_path, load_pool

        pool = load_pool(pool_path)
        split = pool.spec.split
        # Content-addressed by the pool's own spec (+ store state), not by
        # `pool_path`: any run scoring this recipe against this store lands on
        # the same cache, whatever `pool.out` it was saved under.
        cache_root = c.get("cache_root") or "data/eval_cache/samples"
        sample = pool.registry_sample(cache=default_sample_cache_path(pool.spec, cache_root=cache_root))
        print(
            f"[geosae] evaluation pool {pool.spec.pool_id} ({pool.spec.digest}): "
            f"{pool.n_pixels} pixels, {int(pool.scorable_concepts().sum())} scorable concepts "
            f"<- {pool_path}",
            flush=True,
        )
    elif sample_path.exists():
        print(f"[geosae] registry sample: {sample_path}", flush=True)
        sample = RegistrySample.load(sample_path)
    else:
        sample = sample_registry_pixels(
            d["zarr_path"],
            split=split,
            split_name=resolve_split_name(d["zarr_path"], d.get("split_name")),
            embedding_name=d.get("embedding_name", "alphaearth"),
            n_patches=int(c.get("n_patches", 400)),
            pixels_per_patch=int(c.get("pixels_per_patch", 256)),
            products=c.get("products"),
            seed=int(c.get("seed", 0)),
            max_cells=c.get("max_cells"),
            min_spatial_std=c.get("min_spatial_std", 1e-3),
            drop_flat_pixel_cells=bool(c.get("drop_flat_pixel_cells", True)),
            label_store=c.get("label_store"),
        )
        sample.save(sample_path)
        print(f"[geosae] registry sample: {sample.n_pixels} pixels -> {sample_path}", flush=True)
    if pool is None:
        print(
            "[geosae] WARNING: scoring on an ad-hoc pixel sample, not a frozen evaluation pool — "
            "these numbers cannot be audited against a pool id. Set coverage.pool to a pool built "
            "with `geosae pool` (e.g. `-c neurips2026/eval_pool` on top of the run config).",
            flush=True,
        )

    # The sample holds raw store pixels; the model and the dense bases see them in
    # the units of the config's own `data.transforms` (identity when it has none).
    x = (
        sample.raw
        if not d.get("transforms")
        else apply_transforms(build_transforms(d.get("transforms"), zarr_path=d.get("zarr_path")), sample.raw)
    )
    model = build_model_from_cfg(cfg)
    ckpt_path = c.get("checkpoint") or str(P["best"])
    load_checkpoint(model, ckpt_path)
    codes = encode_sample(
        model,
        x,
        device=cfg.get("device", "cuda"),
        batch_size=int(c.get("batch_size", 65_536)),
        sparse=bool(c.get("sparse_codes", True)),
    )

    kw = dict(
        valid=sample.valid_concepts(int(c.get("min_positive", 50))),
        max_per_side=int(c.get("max_per_side", 2500)),
        min_positive=int(c.get("min_positive", 50)),
        seed=int(cfg.get("seed", 42)),
        n_perm=int(c.get("n_perm", 200)),
        alpha=float(c.get("alpha", 0.05)),
    )
    if pool is not None:
        kw.update(
            valid=pool.scorable_concepts(),
            max_per_side=pool.spec.max_per_side,
            min_positive=pool.spec.min_positive,
            balanced={c_idx: pool.balanced_indices(c_idx) for c_idx in pool.balanced},
        )

    def _score(feats, tag: str):
        scalars, arrays = coverage_report(feats, sample.labels, sample.class_table, sample.product_order, **kw)
        print(
            f"[geosae] {tag}: UCov={scalars['ucov']:.0f}/{scalars['n_valid_concepts']} "
            f"CCov={scalars['ccov']:.3f} EMF={scalars['emf']:.1f} "
            f"(median null best {scalars['median_null_best']:.3f})",
            flush=True,
        )
        return scalars, {f"{tag}__{k}": v for k, v in arrays.items()}

    out: dict = {
        "split": split,
        "n_pixels": sample.n_pixels,
        "sample_path": str(sample_path) if pool is None else str(pool_path),
        "sparse_codes": bool(c.get("sparse_codes", True)),
        "pool": pool.manifest() if pool is not None else None,
    }
    npz: dict = {"class_table": sample.class_table, "product_order": np.asarray(sample.product_order, dtype=str)}
    rows: list[dict] = []
    scalars, arrays = _score(codes, "sae")
    out["sae"] = scalars
    npz.update(arrays)
    rows.append({"variant": "sae", "width": scalars["n_features"], **scalars})

    usage = npz["sae__activation_rate"]
    for m_prime in c.get("matched_budgets", [64, 128]):
        idx = matched_budget_subset(usage, int(m_prime))
        tag = f"sae_top{len(idx)}"
        scalars, arrays = _score(codes[:, idx], tag)
        out[tag] = scalars
        npz.update(arrays)
        npz[f"{tag}__feature_index"] = idx
        rows.append({"variant": "sae_matched", "width": scalars["n_features"], **scalars})

    for basis in c.get("dense_bases", ["raw", "pca", "ica", "nmf"]):
        try:
            feats = project_to_basis(x, basis, seed=int(c.get("seed", 0)))
        except ImportError as err:
            print(f"[geosae] skipping dense basis {basis!r}: {err}", flush=True)
            continue
        scalars, arrays = _score(feats, basis)
        out[basis] = scalars
        npz.update(arrays)
        rows.append({"variant": basis, "width": scalars["n_features"], **scalars})

    # Untrained random unit-norm dictionaries at a range of widths: the control
    # that isolates how much of a coverage score is the best-of-M selection bias
    # rather than anything the training learned. Costs no training run.
    for width in c.get("random_widths") or []:
        m = dict(cfg["model"])
        m.update(latent_dim=int(width), freeze_decoder=True, seed=int(cfg.get("seed", 42)))
        m.setdefault("input_dim", x.shape[1])
        # An untrained BatchTopK model has no eval threshold, so calibrate it
        # before asking for sparse codes (they would otherwise come out dense).
        random_model = calibrate_sparsity(
            build_model(m), x, device=cfg.get("device", "cuda"), batch_size=int(c.get("batch_size", 65_536))
        )
        feats = encode_sample(
            random_model,
            x,
            device=cfg.get("device", "cuda"),
            batch_size=int(c.get("batch_size", 65_536)),
            sparse=bool(c.get("sparse_codes", True)),
        )
        tag = f"random_m{int(width)}"
        scalars, arrays = _score(feats, tag)
        out[tag] = scalars
        npz.update(arrays)
        rows.append({"variant": "random", "width": scalars["n_features"], **scalars})

    # The two code regimes are different measurements, so they get different files.
    stem = f"{split}_coverage" if out["sparse_codes"] else f"{split}_dense_coverage"
    (P["eval"] / f"{stem}.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    R.write_metrics_csv(P["eval"] / f"{stem}.csv", rows)
    np.savez_compressed(P["eval"] / f"{stem}_arrays.npz", **npz)
    return out


def cmd_pool(cfg: dict) -> dict:
    """Build the frozen evaluation pool described by the config's ``pool:`` block.

    The pool is drawn once, saved as one ``.npz`` (plus a sibling ``.json``
    manifest) at ``pool.out``, and every metric that cites its ``pool_id``
    afterwards rests on exactly these pixels. Its pixels are materialised out of
    the store in the same pass and cached at a content-addressed sample path
    (:func:`~geosae.eval.pool.default_sample_cache_path`,
    rooted at ``pool.cache_root``) rather than beside ``pool.out``, so no later
    metric — from this config or any other with the same recipe — pays for that
    read again.

    Args:
        cfg: Config with a ``pool:`` block; ``data.zarr_path`` supplies the store
            when ``pool.data_zarr`` is absent.

    Returns:
        The pool manifest, as written next to the ``.npz``.

    Raises:
        ValueError: If the config has no ``pool:`` block or no output path.
    """
    from geosae.eval.pool import EvalPool, default_sample_cache_path, spec_from_config

    block = cfg.get("pool") or {}
    if not block:
        raise ValueError("no `pool:` block in the config (see the bundled config neurips2026/eval_pool)")
    spec = spec_from_config(block, data_zarr=(cfg.get("data") or {}).get("zarr_path"))
    out = block.get("out")
    if not out:
        raise ValueError("set pool.out to the .npz the pool is written to")
    # Content-addressed by (spec, store fingerprint), not by `out`: a second
    # config that happens to draw the identical recipe reuses this cache
    # instead of paying the draw cost again under a different pool file name.
    cache_root = block.get("cache_root") or "data/eval_cache/samples"
    cache = default_sample_cache_path(spec, cache_root=cache_root)
    pool = EvalPool.build(spec, sample_cache=cache)
    pool.save(out)
    manifest = pool.manifest()
    print(f"[geosae] pool {spec.pool_id} -> {out}", flush=True)
    print(f"[geosae] materialised sample -> {cache} ({cache.stat().st_size / 1e9:.2f} GB)", flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


#: Default between-patch distance bin edges (km) of ``geosae variogram``.
VARIOGRAM_EDGES_KM = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 20100)


def cmd_variogram(cfg: dict, run_dir: Path) -> list[dict]:
    """Spatial scale of every feature: semivariance curves and correlation lengths.

    Scores trained runs (``variogram.runs``, paths or globs, each encoded with its own
    model and ``data.transforms``) and conventional bases (``variogram.bases``, e.g.
    ``["raw", "pca"]``, fitted with all ``D`` components and coded densely) on one real-store sample of **whole
    patches**: ``variogram.pixels_per_cell`` must cover the patch, so every cell
    contributes its full pixel grid in row-major order. ``variogram.sample_cache``
    names a ``.npz`` to reuse, written on first use. Otherwise the cells are drawn like
    ``panel``'s (``n_cells``, ``sample_seed``, ``split``), or, with ``variogram.n_anchors``,
    as a nested sample (:func:`~geosae.eval.variogram.nested_sample`): ``n_anchors``
    random cells of the split, each with its neighbours of distance ranks
    ``variogram.neighbour_ranks``.

    The bases are fitted on ``n_fit_pixels`` pixels (from ``variogram.baselines``, else
    the top-level ``baselines`` block): of the training split ``geosae panel`` uses with
    ``variogram.fit_on: train`` (default), or drawn at random from this sample with
    ``fit_on: sample``. ``variogram.fit_seed`` seeds either draw.

    Within a patch the semivariance is read at ``variogram.lags`` pixels (default every
    lag of the grid), ``variogram.pixel_m`` metres each; between patches it is binned by
    the great-circle distance of the cell centres into ``variogram.edges_km`` (default
    :data:`VARIOGRAM_EDGES_KM`), a bin needing ``variogram.min_pairs`` patch pairs
    (default 300). See :mod:`geosae.eval.variogram`.

    Writes ``config.yaml``, ``eval/variogram.npz`` (per row ``<name>/curve``,
    ``/nugget``, ``/length_km``, ``/status``, ``/within``, plus ``dist_km``, ``n_pairs``
    and ``n_within``) and ``eval/variogram.json`` (one summary row per dictionary).
    """
    import zarr

    from geosae.eval.dictionaries import fit_dictionary
    from geosae.eval.ladder import model_encoder
    from geosae.eval.registry import RegistrySample
    from geosae.eval.variogram import (
        CORRELATION_LEVEL,
        correlation_lengths,
        haversine_km,
        merge_moments,
        nested_sample,
        patch_moments,
        semivariance_from_moments,
    )

    v = cfg.get("variogram") or {}
    P = R.paths(run_dir)
    P["eval"].mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(P["config"]))
    device = cfg.get("device", "cuda")
    zarr_path = str(cfg["data"]["zarr_path"])
    group = zarr.open_group(zarr_path, mode="r")
    cache = v.get("sample_cache")
    if cache and Path(cache).exists():
        reg = RegistrySample.load(cache)
        print(f"[geosae] variogram: sample from {cache}", flush=True)
    else:
        block = dict(v)
        if v.get("n_anchors"):
            pool = make_data(cfg, v.get("split", "test")).cells
            pick = nested_sample(
                np.asarray(group["lon"][:])[pool],
                np.asarray(group["lat"][:])[pool],
                n_anchors=int(v["n_anchors"]),
                ranks=v.get("neighbour_ranks", (1, 2, 3, 5, 8, 12, 18, 27)),
                seed=int(v.get("sample_seed", 0)),
            )
            block.update(cells=pool[pick].tolist(), n_cells=int(pick.size))
            print(f"[geosae] variogram: nested sample of {pick.size} cells", flush=True)
        reg = real_eval_sample(cfg, block).sample
        if cache:
            reg.save(cache)
    cells, counts = np.unique(reg.cell, return_counts=True)
    per_cell = int(counts[0])
    side = int(round(np.sqrt(per_cell)))
    order = np.asarray(reg.cell).reshape(len(cells), -1)
    if (counts != per_cell).any() or side * side != per_cell or (order != cells[:, None]).any():
        raise ValueError("variogram needs whole patches: every cell's full pixel grid, cell-major")
    dist = haversine_km(np.asarray(group["lon"][:])[cells], np.asarray(group["lat"][:])[cells])
    lags = [int(lag) for lag in (v.get("lags") or range(1, side))]
    kw = {
        "pixel_m": float(v.get("pixel_m", 10.0)),
        "lags": lags,
        "edges_km": [float(e) for e in (v.get("edges_km") or VARIOGRAM_EDGES_KM)],
        "min_pairs": int(v.get("min_pairs", 300)),
    }
    level = float(v.get("level", CORRELATION_LEVEL))
    raw = np.asarray(reg.raw, dtype=np.float32).reshape(len(cells), per_cell, -1)
    chunk = int(v.get("chunk_cells", 32))
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}

    def score(name: str, kind: str, encode, **meta) -> None:
        t0 = time.perf_counter()
        moments = merge_moments(
            [
                patch_moments(_grid(encode(raw[p : p + chunk].reshape(-1, raw.shape[-1])), side), lags)
                for p in range(0, len(cells), chunk)
            ]
        )
        out = semivariance_from_moments(moments, dist, **kw)
        cl = correlation_lengths(out["curve"], out["dist_km"], n_within=len(lags), level=level)
        live = np.isfinite(out["within"])
        for key, val in (("curve", out["curve"]), ("within", out["within"]), *cl.items()):
            arrays[f"{name}/{key}"] = np.asarray(val)
        arrays["dist_km"], arrays["n_pairs"] = out["dist_km"], out["n_pairs"]
        rows.append(
            {
                "name": name,
                "kind": kind,
                **meta,
                "n_features": int(out["curve"].shape[0]),
                "n_varying": int(live.sum()),
                "within_median": float(np.median(out["within"][live])),
                "nugget_median": float(np.median(cl["nugget"][live])),
                "seconds": time.perf_counter() - t0,
            }
        )
        print(f"[geosae]   {name:28s} within={rows[-1]['within_median']:.3f} ({rows[-1]['seconds']:.0f}s)", flush=True)

    for d in expand_run_dirs(v.get("runs") or []):
        model = load_run_model(d, cfg)
        transforms = load_run_transforms(d, cfg)
        encoder = model_encoder(model, device=device)
        score(
            d.name,
            "sae",
            lambda x, enc=encoder, tf=transforms: enc(apply_transforms(tf, x)),
            run_dir=str(d),
            **run_provenance(d),
        )
        del model
    if v.get("bases"):
        transforms = build_transforms(cfg["data"].get("transforms"), zarr_path=zarr_path)
        b = v.get("baselines") or {}
        fit_seed = int(v.get("fit_seed", 0))
        fit_on = str(v.get("fit_on", "train"))
        if fit_on == "sample":
            flat = raw.reshape(-1, raw.shape[-1])
            n_fit = int(b.get("n_fit_pixels", (cfg.get("baselines") or {}).get("n_fit_pixels", 200_000)))
            n_fit = min(n_fit, len(flat))
            pick = np.sort(np.random.default_rng(fit_seed).choice(len(flat), n_fit, replace=False))
            x_fit = np.asarray(apply_transforms(transforms, flat[pick]), dtype=np.float64)
        elif fit_on == "train":
            x_fit = baseline_fit_pixels_from_config(cfg, b, zarr_path, seed=fit_seed)
        else:
            raise ValueError(f"variogram.fit_on must be 'train' or 'sample', got {fit_on!r}")
        for method in v["bases"]:
            fit = fit_dictionary(x_fit, method, x_fit.shape[1], seed=0, max_iter=int(b.get("max_iter", 200)))
            score(method, method, lambda x, fit=fit: fit.encode(apply_transforms(transforms, x)))

    arrays["n_within"] = np.array(len(lags))
    np.savez_compressed(P["eval"] / "variogram.npz", **arrays)
    meta = {"store": zarr_path, "n_cells": int(len(cells)), "grid": side, **kw, "level": level}
    (P["eval"] / "variogram.json").write_text(json.dumps({**meta, "rows": rows}, indent=1, default=float))
    print(f"[geosae] variogram: {len(rows)} dictionaries -> {P['eval'] / 'variogram.json'}", flush=True)
    return rows


def _grid(codes: np.ndarray, side: int) -> np.ndarray:
    """``(n_cells * side**2, F)`` cell-major codes as ``(n_cells, side, side, F)`` float32."""
    codes = np.asarray(codes, dtype=np.float32)
    return codes.reshape(-1, side, side, codes.shape[-1])


_SUBCONCEPT_FIELDS = (
    "parent_product",
    "parent_purity",
    "reference_purity",
    "align_threshold",
    "dedup_cosine",
    "max_subconcepts",
    "min_stratum_patches",
    "n_null",
    "n_null_rank",
    "block_deg",
    "normalize",
    "seed",
)


def cmd_subconcepts(cfg: dict, run_dir: Path) -> dict:
    """Discover and validate sub-concepts of the run's predictions (``subconcepts`` block); writes into ``eval/``."""
    from geosae.eval.subconcepts import SubconceptConfig, write_subconcepts

    P = R.paths(run_dir)
    split = (cfg.get("predict") or {}).get("split", "test")
    pred_path = str(P["predictions"] / f"{split}.zarr")
    s = cfg.get("subconcepts") or {}
    fields = {k: s[k] for k in _SUBCONCEPT_FIELDS if k in s}
    if "block_deg" in fields:  # YAML lists arrive as a sequence; the config wants a tuple
        bd = fields["block_deg"]
        fields["block_deg"] = (float(bd),) if isinstance(bd, int | float) else tuple(float(x) for x in bd)
    config = SubconceptConfig(**fields)
    return write_subconcepts(
        pred_path,
        cfg["data"]["zarr_path"],
        str(P["eval"]),
        config=config,
        decoder_path=str(P["best"]),
    )


def _add_dictionary_space(cfg: dict, sites, q: dict) -> None:
    """Add a baseline dictionary space, fitted on the training split, encoding the raw sites.

    ``method`` (default ``kmeans``) names the space. A sparse code (``kmeans``'s
    one-hot assignment) is used as it is. A dense fit (e.g. ``random``, the random
    sparse code control) keeps, with ``k``, the ``k`` largest positive projections of
    each pixel onto the atoms (:func:`~geosae.eval.dictionaries.topk_projection_code`),
    or else is sparsified by :func:`~geosae.eval.dictionaries.rate_matched_support` to the
    per-atom ``rate``.
    """
    from geosae.eval.baselines import baseline_fit_pixels
    from geosae.eval.dictionaries import fit_dictionary, rate_matched_support, topk_projection_code

    method = str(q.get("method", "kmeans"))
    data = make_data(cfg, "train", compute=False)
    # ``table_seed`` fixes the fitting draw (it must match a cached table's recipe);
    # ``seed`` seeds the estimator only.
    x = baseline_fit_pixels(
        data, n_pixels=int(q.get("n_fit_pixels", 100_000)), seed=int(q.get("table_seed", 0)), table_path=q.get("table")
    )
    fit = fit_dictionary(x, method, int(q.get("width", 1024)), seed=int(q.get("seed", 0)))
    if fit.sparse:
        code = fit.encode(sites.spaces["raw"])
    elif q.get("k") is not None:
        code = topk_projection_code(fit, sites.spaces["raw"], int(q["k"]))
    elif q.get("rate") is not None:
        code = rate_matched_support(fit.encode(sites.spaces["raw"]), float(q["rate"]))
    else:
        raise ValueError(f"dictionary_space method {method!r} gives a dense code; set dictionary_space.k or .rate")
    sites.spaces[method] = code.astype(np.float32)
    keys = ("method", "width", "n_fit_pixels", "seed", "table_seed", "table", "k", "rate")
    sites.meta["dictionary_space"] = {k: q.get(k) for k in keys}
    print(f"[geosae] {method} dictionary M={fit.n_atoms} fitted on {len(x)} training pixels", flush=True)


def _aoa_graded(cfg: dict, sites, a: dict, geometries: list, outdir: Path) -> list[dict]:
    """Score distance- and environment-graded hold-outs and write their record table."""
    from geosae.eval import aoa as A
    from geosae.eval import holdouts as H

    g = a["graded"]
    seed = int(a.get("seed", cfg.get("seed", 0)))
    records: list[dict] = []
    for design, spec in g["designs"].items():
        if design == "distance":
            coords, metric = sites.lonlat, "haversine"
        elif design == "environment":
            coords, metric = (
                H.climate_coordinates(H.read_site_bioclim(cfg["data"]["zarr_path"], sites.cell)),
                "euclidean",
            )
        else:
            raise ValueError(f"unknown graded design {design!r}, expected 'distance' or 'environment'")
        centres = H.spread_centres(
            coords, int(g["n_centres"]), metric=metric, min_separation=float(spec["min_separation"]), seed=seed
        )
        folds = list(
            H.graded_folds(
                coords,
                centres,
                gaps=[float(v) for v in spec["gaps"]],
                n_test=int(g["n_test"]),
                n_train=int(g["n_train"]),
                metric=metric,
            )
        )
        meta = {
            f.name: {
                "design": design,
                "centre": f.centre,
                "gap": f.gap,
                "centre_lon": float(sites.lonlat[f.centre, 0]),
                "centre_lat": float(sites.lonlat[f.centre, 1]),
            }
            for f in folds
        }
        for geom in geometries:
            print(f"[geosae] graded {design}: {len(folds)} folds, geometry {geom}", flush=True)
            records += A.run_folds(
                sites,
                [(f.name, f.train_mask, f.test_mask) for f in folds],
                scenario=f"graded_{design}",
                fold_meta=meta,
                spaces=g.get("spaces"),
                target_names=list(a.get("targets", ["canopy_height", "elevation"])),
                variants=tuple(a.get("variants", ("unweighted", "ridge"))),
                metric=geom["metric"],
                scaling=geom["scaling"],
                k=int(a.get("k", 5)),
                alpha=float(a.get("alpha", 1.0)),
                max_train=int(g["n_train"]),
                n_calibration=int(g.get("n_calibration", 0)),
                alpha_conformal=float(g.get("alpha", 0.1)),
                probe_space=g.get("probe_space"),
                seed=seed,
            )
    out = outdir / g.get("out", "records_graded.csv")
    R.write_metrics_csv(out, records)
    print(f"[geosae] {len(records)} graded AoA records -> {out}", flush=True)
    return records


def cmd_aoa(cfg: dict, run_dir: Path) -> list[dict]:
    """Score area-of-applicability across split scenarios, targets and geometries.

    Samples held-out pixels once (cached as ``eval/aoa/sites.npz``), then sweeps
    every (scenario, fold, target, feature space, distance geometry, signal) cell
    and writes the flat record table ``eval/aoa/records.csv``. ``aoa.seed`` (default the
    run's ``seed``) seeds the sites, the folds and their subsamples, so runs scored with
    the same ``aoa.seed`` share their held-out pixels. ``aoa.graded``
    adds distance- and environment-graded hold-outs (``eval/aoa/records_graded.csv``),
    ``aoa.dictionary_space`` a baseline dictionary's code as a further feature space.
    """
    from geosae.eval import aoa as A

    P = R.paths(run_dir)
    a = cfg.get("aoa") or {}
    outdir = P["eval"] / "aoa"
    outdir.mkdir(parents=True, exist_ok=True)
    targets = list(a.get("targets", ["canopy_height", "elevation"]))
    seed = int(a.get("seed", cfg.get("seed", 0)))
    sites_path = outdir / "sites.npz"

    if sites_path.exists() and not a.get("resample", False):
        sites = A.load_sites(str(sites_path))
        print(f"[geosae] reusing {len(sites)} cached AoA sites from {sites_path}", flush=True)
    else:
        model = build_model_from_cfg(cfg)
        ckpt_path = a.get("checkpoint") or str(P["best"])
        load_checkpoint(model, ckpt_path)
        data = make_data(
            cfg, a.get("split", "test"), compute=a.get("compute_inram", False), max_cells=a.get("max_cells")
        )
        pool_path = a.get("pool")
        pool_cells = None
        if pool_path:
            from geosae.eval.pool import load_pool

            pool_cells = load_pool(pool_path).cells
            print(f"[geosae] AoA sites restricted to {len(pool_cells)} pool cells <- {pool_path}", flush=True)
        sites = A.sample_sites(
            model,
            data,
            targets=targets,
            pixels_per_patch=int(a.get("pixels_per_patch", 1)),
            batch_size=int(a.get("batch_size", 64)),
            device=cfg.get("device", "cuda"),
            seed=seed,
            cells=pool_cells,
        )
        sites.meta["checkpoint"] = ckpt_path
        sites.to_npz(str(sites_path))
        print(f"[geosae] sampled {len(sites)} AoA sites -> {sites_path}", flush=True)

    if a.get("max_sites"):
        sites = sites.subsample(int(a["max_sites"]), seed=seed)

    if a.get("dictionary_space"):
        _add_dictionary_space(cfg, sites, a["dictionary_space"])

    geometries = a.get("geometries") or [{"metric": "euclidean", "scaling": "none"}]
    if a.get("graded"):
        _aoa_graded(cfg, sites, a, geometries, outdir)
    scenarios = tuple(a.get("scenarios", tuple(A.SPLIT_SCENARIOS)))
    if not scenarios:
        return []
    records: list[dict] = []
    for geom in geometries:
        print(f"[geosae] AoA geometry metric={geom['metric']} scaling={geom['scaling']}", flush=True)
        records += A.run_scenarios(
            sites,
            scenarios=scenarios,
            spaces=a.get("spaces"),
            target_names=targets,
            variants=tuple(a.get("variants", ("unweighted", "ridge"))),
            metric=geom["metric"],
            scaling=geom["scaling"],
            extra_signals=bool(a.get("extra_signals", True)),
            k=int(a.get("k", 5)),
            alpha=float(a.get("alpha", 1.0)),
            max_train=int(a.get("max_train", 15_000)),
            max_test=a.get("max_test"),
            min_train=int(a.get("min_train", 1000)),
            min_test=int(a.get("min_test", 200)),
            seed=seed,
        )
    R.write_metrics_csv(outdir / "records.csv", records)
    print(f"[geosae] {len(records)} AoA records -> {outdir / 'records.csv'}", flush=True)

    return records


def cmd_stats(cfg: dict) -> str:
    """Write the embedding normalisation statistics that the transforms read back.

    Reads ``stats.split`` (default ``train``) and ``stats.isotropic``; writes to
    ``stats.out``, else to ``data.transforms.stats_npz``, else next to the store.
    """
    from geosae.data.dataset import compute_embedding_stats, write_embedding_stats

    d = cfg["data"]
    s = cfg.get("stats") or {}
    out = s.get("out") or (d.get("transforms") or {}).get("stats_npz")
    if out is None:
        out = f"{Path(d['zarr_path']).with_suffix('')}_{d.get('embedding_name', 'alphaearth')}_stats.npz"
    stats = compute_embedding_stats(
        d["zarr_path"],
        d.get("embedding_name", "alphaearth"),
        resolve_split_name(d["zarr_path"], d.get("split_name")),
        s.get("split", "train"),
        isotropic=bool(s.get("isotropic", False)),
        batch_size=s.get("batch_size"),
        n_load_workers=int(d.get("n_load_workers", 8)),
    )
    write_embedding_stats(out, stats)
    print(f"[geosae] wrote {out}: |mean|={np.linalg.norm(stats['mean']):.4f}, mean std={stats['std'].mean():.4f}")
    return str(out)


def cmd_fit_table(cfg: dict) -> str:
    """Build and persist the fitting table the conventional decompositions are fitted from.

    Walks the whole training split once and draws a seeded pixel from every patch
    (:func:`geosae.eval.baselines.build_fit_table`), so a later fit estimates its
    parameters from the spatial coverage of the split rather than from the handful of
    patches a whole-patch draw reads. Reads ``baselines.fit_table_path``,
    ``baselines.n_fit_pixels`` and ``baselines.sample_seed``; the stages that fit
    read the same file back and rebuild nothing.

    Returns:
        Path of the written ``.npz``.

    Raises:
        ValueError: If ``baselines.fit_table_path`` is unset.
    """
    from geosae.eval.baselines import build_fit_table

    b = cfg.get("baselines") or {}
    out = b.get("fit_table_path")
    if not out:
        raise ValueError("set baselines.fit_table_path to the file the table should be written to")
    data = store_data(cfg, cfg["data"]["zarr_path"], "train")
    per_patch = max(1, int(np.ceil(int(b.get("n_fit_pixels", 200_000)) / max(len(data), 1))))
    table = build_fit_table(
        data,
        pixels_per_patch=per_patch,
        seed=int(b.get("sample_seed", 0)),
        batch_size=int(b.get("fit_batch_size", 64)),
        progress_every=int(b.get("fit_progress_every", 200)),
    )
    table.to_npz(out)
    print(f"[geosae] fit table: {len(table)} px from {table.meta['n_patches']} patches -> {out}", flush=True)
    return str(out)


def _store_path(group) -> str:
    """The filesystem path behind an open group, however its store spells it."""
    store = group.store
    return str(getattr(store, "root", None) or str(store).removeprefix("file://"))


def _same_store(a, b) -> bool:
    """Whether two paths name the same store on disk."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a).rstrip("/") == str(b).rstrip("/")


def _refetch_destination(out_path, source, n_cells: int, patch_size: int):
    """The group a re-fetch writes into, created as a standalone store if new.

    Writing back into the store the cells came from needs its *consolidated*
    metadata bypassed: a group opened through it cannot see an array that is not
    already listed there, so a new destination array is invisible the moment it is
    created. The listing is rebuilt once the values are written, by
    :func:`cmd_refetch`, since until then it would advertise an empty array.

    A destination of its own instead gets the cell axis and the cell centres
    copied over, so it is readable without the store it came from.
    """
    import zarr

    if _same_store(out_path, _store_path(source)):
        return zarr.open_group(str(out_path), mode="r+", use_consolidated=False)
    fresh = not Path(out_path).exists()
    group = zarr.open_group(str(out_path), mode="a")
    if fresh:
        group.attrs["n_cells"] = n_cells
        group.attrs["patch_size"] = patch_size
        for name in ("lon", "lat"):
            arr = group.create_array(name, shape=(n_cells,), chunks=(min(65536, n_cells),), dtype="float64")
            arr[:] = np.asarray(source[name][:])
    return group


def cmd_refetch(cfg: dict) -> dict:
    """Re-fetch one reference product into a (possibly new) array of an existing store.

    A store is built once and its products are fetched into it in one pass, so a
    product found to be wrong afterwards has no way back in. This adds one: it
    creates the array if it is missing, fetches the named product for the rows
    that already carry it, and writes a fresh success mask — leaving every other
    array untouched.

    Config block ``refetch``: ``zarr_path`` (the store the cells come from),
    ``out_path`` (where to write; defaults to the same store), ``product`` (a
    registry key), ``array`` (the destination array; defaults to the product's
    own, and naming a new one keeps the old values for comparison), ``rows``
    (``"masked"`` for the rows the product already covers, ``"all"``, or a count
    for the leading rows) and ``max_workers``.

    Writing to a *separate* ``out_path`` is the safe form and the default choice
    for a shared store: the destination is created as a store in its own right,
    carrying the same cell axis and the same ``lon``/``lat``, so it can be opened
    on its own and nothing anything else depends on is touched.
    """
    import zarr

    from geosae.data.download.sources import get_product
    from geosae.data.download.store import add_product_array, populate_products, utm_geobox

    c = dict(cfg.get("refetch") or {})
    path = c["zarr_path"]
    key = c["product"]
    spec = get_product(key)
    array = c.get("array") or spec.array
    source = zarr.open_group(str(path), mode="r")
    n_cells, patch_size = int(source.attrs["n_cells"]), int(source.attrs["patch_size"])
    group = _refetch_destination(c.get("out_path") or path, source, n_cells, patch_size)

    rows_spec = c.get("rows", "masked")
    if rows_spec == "all":
        rows = np.arange(n_cells, dtype=np.int64)
    elif rows_spec == "masked":
        rows = np.where(np.asarray(source[f"mask_{spec.array}"][:], dtype=bool))[0]
    else:
        rows = np.arange(int(rows_spec), dtype=np.int64)

    array = add_product_array(group, key, array)

    lon = np.asarray(source["lon"][:])[rows]
    lat = np.asarray(source["lat"][:])[rows]
    # the geobox is recomputed rather than rewritten: the store may be shared, and
    # a re-fetch has no business touching an array it is not the subject of
    places = zip(rows, lon, lat, strict=True)
    geoboxes = {int(r): utm_geobox(float(lo), float(la), patch_size)[0] for r, lo, la in places}
    print(f"[geosae] re-fetching {key} -> {array} for {rows.size} rows of {path}", flush=True)
    populate_products(
        group,
        [key],
        rows,
        lon,
        lat,
        geoboxes,
        max_workers=int(c.get("max_workers", 8)),
        into={key: array},
    )
    covered = int(np.asarray(group[f"mask_{array}"][:], dtype=bool).sum())
    if _same_store(c.get("out_path") or path, _store_path(source)):
        # the store lists its arrays in one consolidated key; a new array is only
        # visible to an ordinary reader once that listing is rebuilt
        zarr.consolidate_metadata(group.store)
        print(f"[geosae] re-consolidated the metadata of {path}", flush=True)
    print(f"[geosae] {array}: {covered}/{rows.size} rows fetched", flush=True)
    return {"array": array, "rows": int(rows.size), "covered": covered}


def _download_rows(spec) -> np.ndarray | None:
    """Rows of a ``download.rows`` setting: ``None`` (all), a count, or ``[start, end]``."""
    if spec is None:
        return None
    if isinstance(spec, int):
        return np.arange(spec, dtype=np.int64)
    start, end = (int(v) for v in spec)
    return np.arange(start, end, dtype=np.int64)


def _download_fetch_kwargs(products: list[str], c: dict) -> dict:
    """Per-product fetcher overrides from ``download.{gee_project,cache_dir,fetch_kwargs}``."""
    from geosae.data.download.sources import _CACHE, Auth, get_product

    out: dict = {}
    for key in products:
        spec = get_product(key)
        kw: dict = {}
        if c.get("gee_project") and spec.auth is Auth.GEE:
            kw["gee_project"] = c["gee_project"]
        if c.get("cache_dir") and "cache_dir" in spec.fetcher_kwargs:
            kw["cache_dir"] = str(spec.fetcher_kwargs["cache_dir"]).replace(_CACHE, str(c["cache_dir"]), 1)
        kw.update((c.get("fetch_kwargs") or {}).get(key) or {})
        if kw:
            out[key] = kw
    return out


def cmd_download(cfg: dict) -> dict:
    """Build a patch store from the original sources, or print its plan.

    Config block ``download``: ``out`` (store path), ``cells`` (the cells to
    build: keyword arguments of :func:`geosae.data.download.grid.select_cells`,
    ``n``, ``bbox``, ``points`` or ``ids``, plus ``seed``), ``split`` (``name``,
    ``val_frac`` and ``test_frac`` of a ``cells`` store), ``preset`` (a name of
    :data:`geosae.data.download.presets.PRESETS`, used when ``cells`` is null),
    ``products`` (registry keys; required with ``cells``, else replacing the
    preset's), ``rows`` (null for all, a count of
    leading rows, or ``[start, end]``), ``plan`` (true: list sources, sizes and
    credentials, fetch nothing), ``max_workers``, ``block_rows``,
    ``gee_project``, ``cache_dir`` (where downloaded source files are cached)
    and ``fetch_kwargs`` (``{product: {kwarg: value}}``).

    Re-running the same config resumes: every product skips the rows it
    already holds.
    """
    from geosae.data.download.plan import build_plan, check_credentials, format_plan
    from geosae.data.download.presets import PRESETS

    c = dict(cfg.get("download") or {})
    rows = _download_rows(c.get("rows"))
    choice = dict(c.get("cells") or {})
    if choice:
        from geosae.data.download.grid import select_cells

        products = list(c.get("products") or [])
        if not products:
            raise ValueError(
                "download.cells needs download.products, e.g. -o 'download.products=[alphaearth_cog,lc_esa_wc]'"
            )
        seed = int(choice.pop("seed", 0))
        cells = select_cells(**choice, seed=seed)
        n_cells, reference_rows = int(cells.size), None
        print(f"[geosae] {n_cells} cells selected", flush=True)
    else:
        if c.get("preset") not in PRESETS:
            raise ValueError(f"download.preset must be one of {sorted(PRESETS)} unless download.cells is set")
        preset = PRESETS[c["preset"]]
        products = list(c.get("products") or preset.products)
        n_cells, reference_rows = preset.n_cells, preset.reference_rows
    if c.get("plan"):
        records = build_plan(products, n_cells, reference_rows=reference_rows, rows=rows)
        print(format_plan(records, check_credentials()), flush=True)
        return {"plan": records}

    from geosae.data.download.store import build_majortom_store, build_preset_store

    t0 = time.time()
    common = dict(
        products=products,
        rows=rows,
        max_workers=int(c.get("max_workers", 8)),
        block_rows=int(c.get("block_rows", 25_600)),
        fetch_kwargs=_download_fetch_kwargs(products, c),
        log=lambda line: print(line, flush=True),
    )
    if choice:
        split = dict(c.get("split") or {})
        summary = build_majortom_store(
            c["out"],
            cells,
            split_name=split.get("name", "default"),
            val_frac=float(split.get("val_frac", 0.1)),
            test_frac=float(split.get("test_frac", 0.2)),
            seed=seed,
            **common,
        )
    else:
        summary = build_preset_store(c["out"], preset, **common)
    for key, rec in summary["products"].items():
        print(f"[geosae] {key} -> {rec['array']}: {rec['covered']}/{rec['requested']} rows", flush=True)
    for name, counts in summary["splits"].items():
        print(f"[geosae] {name}: {counts}", flush=True)
    print(f"[geosae] {c['out']} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return summary


def cmd_subset(cfg: dict) -> dict:
    """Sentinel-2 subset of a patch store, its foundation-model embeddings, and their statistics.

    Config block ``subset``: ``stage`` (``fetch``, ``embed`` or ``stats``; run
    them as separate jobs), ``parent``, ``out``, ``split_name``,
    ``max_per_split``, ``seed``, ``exclude_stores``, ``max_cells``,
    ``patch_size``, ``year``, ``max_cloud_cover``, ``max_workers`` (fetch);
    ``backbones``, ``variants``, ``device``, ``batch_cells``,
    ``patch_size_tokens``, ``crop_px`` (embed); ``stats_dir`` (stats). See
    :mod:`geosae.data.embed.subset`.
    """
    from geosae.data.embed import subset as S

    c = dict(cfg.get("subset") or {})
    stage = c.get("stage", "fetch")
    split_name = c.get("split_name", "split_10k")
    seed = int(c.get("seed", 42))
    backbones = list(c.get("backbones") or [])
    if stage == "fetch":
        out = S.fetch_subset(
            c["parent"],
            c["out"],
            split_name=split_name,
            max_per_split=dict(c.get("max_per_split") or {}),
            seed=seed,
            exclude_stores=list(c.get("exclude_stores") or []),
            max_cells=c.get("max_cells"),
            patch_size=int(c.get("patch_size", 128)),
            year=int(c.get("year", 2024)),
            max_cloud_cover=float(c.get("max_cloud_cover", 10.0)),
            max_workers=int(c.get("max_workers", 1)),
        )
    elif stage == "embed":
        out = S.encode_subset(
            c["out"],
            backbones,
            variants=list(c.get("variants") or []),
            device=c.get("device", cfg.get("device", "cuda")),
            batch_cells=int(c.get("batch_cells", 32)),
            patch_size_tokens=c.get("patch_size_tokens"),
            crop_px=c.get("crop_px"),
        )
    elif stage == "stats":
        out = S.subset_stats(c["out"], backbones, c["stats_dir"], split_name=split_name, seed=seed)
    else:
        raise ValueError(f"subset.stage must be fetch, embed or stats, got {stage!r}")
    print(f"[geosae] subset {stage}: {out}", flush=True)
    return out


def cmd_gridded(cfg: dict) -> dict:
    """Coarsen a patch store onto the 0.25 degree grid and attach the Copernicus-FM map.

    Config block ``gridded``: ``src`` (the patch store), ``out``, ``cache_dir``
    (where the ~3 GB embedding map is downloaded once), ``products`` (source
    arrays to aggregate; null for all) and ``n_workers``. See
    :func:`geosae.data.download.presets.build_copernicusfm025_store`.
    """
    from geosae.data.download.presets import build_copernicusfm025_store

    c = dict(cfg.get("gridded") or {})
    summary = build_copernicusfm025_store(
        c["src"],
        c["out"],
        cache_dir=c["cache_dir"],
        products=c.get("products"),
        n_workers=int(c.get("n_workers", 4)),
        progress=True,
    )
    print(f"[geosae] {c['out']}: {summary['n_cells']} cells, {summary['n_embedded']} with an embedding", flush=True)
    return summary


def cmd_release(cfg: dict) -> dict:
    """Export the dataset's recipe as plain tables next to the store.

    Writes ``cells.csv`` (every cell's MajorTOM id, centre, UTM code and
    split), ``products.csv`` (how each reference product was fetched, its
    licence and what that licence allows us to redistribute), ``concepts.csv``
    (the concept registry the alignment metrics score against) and
    ``manifest.json`` (selection, split ranges, realised counts).

    Config block ``release``: ``zarr_path`` (the store), ``out_dir`` (where the
    tables land), ``split_name``, ``products`` (a registry subset, or null for
    all) and ``array_products`` (the product behind a re-fetched array, for
    stores written before that provenance was recorded).
    """
    from geosae.data.download.release import write_release_tables

    c = dict(cfg.get("release") or {})
    paths = write_release_tables(
        c["zarr_path"],
        c["out_dir"],
        split_name=resolve_split_name(c["zarr_path"], c.get("split_name")),
        products=c.get("products"),
        array_products=c.get("array_products"),
    )
    for name, path in paths.items():
        print(f"[geosae] {name}: {path}", flush=True)
    return paths


COMMANDS = (
    "train",
    "predict",
    "run",
    "panel",
    "coverage",
    "pool",
    "variogram",
    "subconcepts",
    "aoa",
    "stats",
    "fit-table",
    "download",
    "subset",
    "gridded",
    "refetch",
    "release",
)


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and run one ``geosae`` subcommand."""
    ap = argparse.ArgumentParser(prog="geosae", description="Sparse autoencoders for Earth embeddings")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("-c", "--config", action="append", help="YAML config (repeatable)")
    ap.add_argument("--run", help="run directory; its own config.yaml is the base config")
    ap.add_argument("-o", "--override", action="append", default=[], help="dotted override key=value")
    args = ap.parse_args(argv)
    if args.run:
        args.config = [str(Path(args.run) / "config.yaml"), *(args.config or [])]
        args.override = [*args.override, f"run.dir={args.run}"]
    if not args.config:
        ap.error("one of -c/--config or --run is required")
    cfg = load_config(args.config, args.override)
    if cfg.get("device") == "auto":
        import torch

        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if args.command == "train":
        cmd_train(cfg, R.resolve_run_dir(cfg, create=True))
    elif args.command == "predict":
        cmd_predict(cfg, R.resolve_run_dir(cfg, create=False))
    elif args.command == "run":
        run_dir = R.resolve_run_dir(cfg, create=True)
        cmd_train(cfg, run_dir)
        cmd_predict(cfg, run_dir)
    elif args.command == "panel":
        cmd_panel(cfg, R.resolve_run_dir(cfg, create=True))
    elif args.command == "coverage":
        cmd_coverage(cfg, R.resolve_run_dir(cfg, create=False))
    elif args.command == "pool":
        cmd_pool(cfg)
    elif args.command == "variogram":
        cmd_variogram(cfg, R.resolve_run_dir(cfg, create=True))
    elif args.command == "subconcepts":
        cmd_subconcepts(cfg, R.resolve_run_dir(cfg, create=False))
    elif args.command == "aoa":
        cmd_aoa(cfg, R.resolve_run_dir(cfg, create=False))
    elif args.command == "stats":
        cmd_stats(cfg)
    elif args.command == "fit-table":
        cmd_fit_table(cfg)
    elif args.command == "download":
        cmd_download(cfg)
    elif args.command == "subset":
        cmd_subset(cfg)
    elif args.command == "gridded":
        cmd_gridded(cfg)
    elif args.command == "refetch":
        cmd_refetch(cfg)
    elif args.command == "release":
        cmd_release(cfg)


if __name__ == "__main__":
    main()

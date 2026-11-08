"""Training loop for any :class:`~geosae.models.SparseAutoencoder` — plain PyTorch, single GPU, bf16 AMP.

Recipe: Adam with ``beta1=0`` (OpenAI's dead-feature trick), no weight decay, gradient clipping, and a cosine learning
rate with linear warm-up. Validation runs after every epoch; the checkpoint with
the best validation FVE is kept as ``best.pth``.

**FVE is only comparable across epochs at a fixed sparsity.** The Top-K family
reaches its sparsity by construction (L0 == k every epoch), so plain "highest
FVE wins" is sound for it. ReLU+ℓ1 reaches its sparsity only through a
penalty that a :class:`~geosae.train.sparsity_controller.TargetL0Controller`
tunes at runtime (``train.target_l0``): early on -- through the controller's
own ``warmup_steps`` and however long it takes to converge after that -- the
observed L0 can sit far from the target, and an under-sparsified epoch (many
more active latents than the model is meant to use) trivially reconstructs
better than any properly-sparse epoch ever will. Comparing its FVE against a
converged epoch's compares different operating points, not model quality, and
silently keeps that degenerate epoch as ``best.pth`` forever. So whenever a
controller is active, an epoch only competes for ``best.pth`` while its
validation L0 sits within :attr:`Trainer.best_checkpoint_l0_tolerance` of the
controller's target (see ``train.target_l0.best_checkpoint_tolerance``).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from geosae.data.dataset import PatchData, make_dataloader
from geosae.eval.metrics import calibrate_threshold, reconstruction_metrics
from geosae.models.base import SparseAutoencoder, model_inputs
from geosae.train.loss import sae_loss
from geosae.train.sparsity_controller import TargetL0Controller


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warm-up to 1.0 over ``warmup`` steps, then cosine decay to 0."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _penalty_scale(step: int, warmup: int) -> float:
    """Linear warm-up of the sparsity penalty from 0 to 1 over ``warmup`` steps."""
    if warmup <= 0:
        return 1.0
    return min(1.0, step / warmup)


class Trainer:
    """Fit one sparse autoencoder on a training split, validating and checkpointing every epoch.

    Args:
        model: The model to train (moved to ``device``).
        train_data: Training patches.
        val_data: Validation patches, or ``None`` to skip validation.
        outdir: Run directory; receives ``metrics.jsonl``, ``last.pth`` and ``best.pth``.
        model_cfg: The model config, stored in every checkpoint.
        lr: Peak learning rate of the warm-up + cosine schedule.
        beta1: Adam ``beta1``.
        beta2: Adam ``beta2``.
        eps: Adam ``eps``.
        weight_decay: Adam weight decay.
        clip_grad_norm: Gradient-norm clip (``0`` disables it).
        batch_size: Patches per step (full-patch batching).
        epochs: Training epochs.
        steps_per_epoch: Steps per epoch; defaults to one pass over ``train_data``.
        rampup: Linear learning-rate warm-up steps.
        aux_penalty: Weight of the AuxK term.
        top_k_aux: Dead latents the AuxK term reconstructs from.
        l1_coeff: ℓ1 penalty on the codes (the ReLU+ℓ1 model's sparsity mechanism).
        slow_group_weight: Weight of the Matryoshka prefix reconstruction terms.
        target_l0: Options of a :class:`~geosae.train.sparsity_controller.TargetL0Controller`
            that tunes ``l1_coeff`` toward ``value`` active latents per pixel; ``None`` disables it.
        penalty_warmup_steps: Linear warm-up of the ℓ1 penalty.
        seed: Seeds the model init, the shuffling and the pixel subsampler.
        device: Torch device.
        amp: Train under bf16 autocast.
        val_batch_size: Patches per validation batch.
        log_every: Steps between progress lines.
        pixel_subsample: Options of a :class:`~geosae.train.pixel_subsample.PixelSubsampler` that
            replaces full-patch batches with per-step pixel subsamples; ``None`` disables it.
        num_workers: DataLoader workers.
        prefetch_factor: DataLoader prefetch factor.
        pin_memory: Pin DataLoader memory (on a CUDA device only).
        persistent_workers: Keep DataLoader workers alive between epochs.
    """

    def __init__(
        self,
        model: SparseAutoencoder,
        train_data: PatchData,
        val_data: PatchData | None,
        *,
        outdir: str,
        model_cfg: dict | None = None,
        lr: float = 1e-3,
        beta1: float = 0.0,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 1.0,
        batch_size: int = 128,
        epochs: int = 50,
        steps_per_epoch: int | None = None,
        rampup: int = 1000,
        aux_penalty: float = 0.03125,
        top_k_aux: int = 32,
        l1_coeff: float = 0.0,
        slow_group_weight: float = 0.0,
        target_l0: dict | None = None,
        penalty_warmup_steps: int = 0,
        seed: int = 42,
        device: str = "cuda",
        amp: bool = True,
        val_batch_size: int = 64,
        log_every: int = 50,
        pixel_subsample: dict | None = None,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.model_cfg = model_cfg or {}
        self.clip_grad_norm = clip_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.rampup = rampup
        self.aux_penalty = aux_penalty
        self.top_k_aux = top_k_aux
        self.l1_coeff = l1_coeff
        self.slow_group_weight = slow_group_weight
        self.penalty_warmup_steps = int(penalty_warmup_steps)
        self.seed = seed
        self.device = device
        self.amp_dtype = torch.bfloat16 if amp else None
        # bf16 autocast is a GPU speed-up; on the CPU it would only change the numerics.
        self._autocast = self.amp_dtype is not None and str(device).startswith("cuda")
        self.val_batch_size = val_batch_size
        self.log_every = log_every
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory) and str(device).startswith("cuda")
        self.persistent_workers = bool(persistent_workers)

        n_batches = train_data.num_batches(batch_size, drop_last=True)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else n_batches
        self.total_steps = self.epochs * self.steps_per_epoch

        # Optional per-step pixel subsampler: each step feeds a flat (Q, 1, 1, D) pixel
        # subsample instead of full patches (see geosae.train.pixel_subsample).
        self.subsampler = None
        if pixel_subsample:
            from geosae.train.pixel_subsample import PixelSubsampler

            ps = dict(pixel_subsample)
            ps.pop("enabled", None)
            if train_data._arr is None:
                raise ValueError("pixel_subsample requires compute_inram=True (train_data has no in-RAM array)")
            # `train_data._arr` is raw in-RAM data; PatchData.iter_batches
            # (what validation/eval reads through) applies train_data's own
            # Transforms to every batch, so the subsampler must apply the same
            # pipeline or training and validation see different input
            # distributions (see geosae.train.pixel_subsample.sample_pixels).
            coords = None
            if getattr(self.model, "uses_coordinates", False):
                coords = np.stack([train_data.lon, train_data.lat], axis=1)
            self.subsampler = PixelSubsampler(
                train_data._arr, seed=seed, transforms=train_data.transforms, coordinates=coords, **ps
            )

        # Optional closed-loop controller that adapts the ℓ1 penalty
        # toward a target mean L0 every step (see
        # geosae.train.sparsity_controller.TargetL0Controller). Inert (None)
        # unless `target_l0` is set: the Top-K family reaches its sparsity by
        # construction and has nothing for this to drive.
        self.target_l0_controller: TargetL0Controller | None = None
        # How close a controlled model's validation L0 must sit to the target
        # (as a fraction of it) to compete for `best.pth` at all -- see the
        # module docstring. Unused (irrelevant) when `target_l0` is unset.
        self.best_checkpoint_l0_tolerance = 0.5
        if target_l0:
            tl = dict(target_l0)
            value = float(tl.pop("value"))
            self.best_checkpoint_l0_tolerance = float(tl.pop("best_checkpoint_tolerance", 0.5))
            tl.setdefault("warmup_steps", max(self.rampup, self.penalty_warmup_steps))
            self.target_l0_controller = TargetL0Controller(value, **tl)
            if self.l1_coeff <= 0:
                raise ValueError(
                    "train.target_l0 controls ReLU+l1 sparsity through train.l1_coeff -- start it "
                    "above 0 (the controller only ever rescales an existing penalty, it cannot "
                    "grow one from a fixed 0)"
                )

        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: _lr_lambda(s, self.rampup, self.total_steps))
        self._log_path = self.outdir / "metrics.jsonl"

    def _make_loader(self, seed: int):
        """Chunk-aligned, worker-pipelined DataLoader for one shuffled pass."""
        return make_dataloader(
            self.train_data,
            self.batch_size,
            shuffle=True,
            seed=seed,
            drop_last=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def _train_batches(self, epoch: int):
        """Yield ``steps_per_epoch`` training batches for ``epoch``.

        Subsampler mode: per-step reconstruction-pixel subsamples (single
        persistent RNG, epoch-independent). Default: shuffled full-patch batches
        from a worker-pipelined DataLoader, reshuffling within the epoch if
        ``steps_per_epoch`` exceeds one pass over the data.
        """
        if self.subsampler is not None:
            yield from self.subsampler.batches(self.steps_per_epoch)
            return
        emitted = 0
        reshuffle = 0
        while emitted < self.steps_per_epoch:
            seed = self.seed + 1000 * epoch + reshuffle
            for batch in self._make_loader(seed):
                yield batch
                emitted += 1
                if emitted >= self.steps_per_epoch:
                    return
            reshuffle += 1

    def fit(self) -> dict:
        """Train for ``epochs`` epochs and return the validation metrics of ``best.pth``."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        best_fve = -float("inf")
        best_metrics: dict = {}
        step = 0
        if self.subsampler is not None:
            geom = (
                f"pixel-subsample batch: N={self.subsampler.params['n_anchor_patches']} anchor patches, "
                f"K={self.subsampler.params['recon_pixels_per_patch']} recon px/patch, "
                f"M={self.subsampler.params['m_anchor_pixels']}, pair_mode={self.subsampler.params['pair_mode']}"
            )
        else:
            geom = f"batch={self.batch_size} patches"
        print(
            f"[geosae] training {self.total_steps} steps ({self.epochs} epochs x {self.steps_per_epoch} steps), {geom}",
            flush=True,
        )
        for epoch in range(self.epochs):
            self.model.train()
            t0 = time.time()
            run = {"recon_mse": 0.0, "aux": 0.0, "l0": 0.0, "n_dead": 0.0}
            if self.target_l0_controller is not None:
                run["sparsity_penalty"] = 0.0
            for batch in self._train_batches(epoch):
                # Patch shape (B, H, W, D) is kept: every pixel is encoded independently.
                inputs = model_inputs(self.model, batch, self.device)
                x = inputs[0]
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self._autocast):
                    recon, z, z_dense = self.model(*inputs)
                    loss, m = sae_loss(
                        self.model,
                        x,
                        recon,
                        z,
                        z_dense,
                        aux_penalty=self.aux_penalty,
                        top_k_aux=self.top_k_aux,
                        l1_coeff=self.l1_coeff,
                        slow_group_weight=self.slow_group_weight,
                        penalty_scale=_penalty_scale(step, self.penalty_warmup_steps),
                    )
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
                self.sched.step()
                if self.target_l0_controller is not None:
                    self.l1_coeff = self.target_l0_controller.update(self.l1_coeff, m["l0"])
                    m["sparsity_penalty"] = self.l1_coeff
                for kk in run:
                    run[kk] += m.get(kk, 0.0)
                if step % self.log_every == 0:
                    penalty_str = (
                        f" sparsity_penalty={m['sparsity_penalty']:.3e}"
                        if self.target_l0_controller is not None
                        else ""
                    )
                    print(
                        f"  e{epoch} s{step} lr={self.sched.get_last_lr()[0]:.2e} "
                        f"recon={m['recon_mse']:.4f} aux={m['aux']:.4f} "
                        f"l0={m['l0']:.1f} dead={m.get('n_dead', 0)}{penalty_str}",
                        flush=True,
                    )
                step += 1
            n = max(1, self.steps_per_epoch)
            train_avg = {k: v / n for k, v in run.items()}

            val = {}
            if self.val_data is not None:
                # Calibrate the BatchTopK inference threshold to L0≈k before
                # measuring, so val metrics (and best.pth) reflect the intended
                # operating point rather than the drifting training EMA.
                thr = calibrate_threshold(
                    self.model, self.val_data, self.model.k, device=self.device, amp_dtype=self.amp_dtype
                )
                val = reconstruction_metrics(
                    self.model,
                    self.val_data,
                    batch_size=self.val_batch_size,
                    device=self.device,
                    amp_dtype=self.amp_dtype,
                )
                val["threshold"] = thr
            rec = {
                "epoch": epoch,
                "step": step,
                "time_s": round(time.time() - t0, 1),
                "train": train_avg,
                "val": val,
            }
            with open(self._log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"[epoch {epoch}] train_recon={train_avg['recon_mse']:.4f} "
                + (
                    f"val_fve={val.get('fve', float('nan')):.4f} "
                    f"val_cos={val.get('recon_cosine', float('nan')):.4f} "
                    f"val_l0={val.get('l0', float('nan')):.1f} "
                    f"dead={val.get('dead_frac', float('nan')):.3f} "
                    if val
                    else ""
                )
                + f"({rec['time_s']}s)",
                flush=True,
            )
            self._save("last.pth", epoch, step, val)
            fve = val.get("fve", train_avg["recon_mse"] * -1)  # fall back if no val
            if fve > best_fve and self._competes_for_best(val):
                best_fve = fve
                best_metrics = val or train_avg
                self._save("best.pth", epoch, step, val)
        if not best_metrics and self.epochs > 0:
            # No epoch's L0 ever settled within tolerance of the controller's
            # target (a controller that failed to converge, or a run too
            # short to reach it) -- fall back to the last epoch rather than
            # leaving `best.pth` unwritten, but say so loudly: this run's
            # sparsity never reached a comparable operating point.
            print(
                "[geosae] WARNING: no epoch's validation L0 settled near the target_l0 target; "
                "best.pth falls back to the last epoch",
                flush=True,
            )
            best_metrics = val or train_avg
            self._save("best.pth", epoch, step, val)
        print(f"[geosae] done. best val: {best_metrics}", flush=True)
        return best_metrics

    def _competes_for_best(self, val: dict) -> bool:
        """Whether this epoch's FVE is comparable to the running ``best.pth`` at all.

        Always true when no :class:`~geosae.train.sparsity_controller.TargetL0Controller`
        is driving the sparsity penalty (the Top-K family: L0 == k every epoch,
        so every epoch's FVE sits at the same operating point). With a
        controller, an epoch whose validation L0 has not yet settled near the
        target reconstructs an easier, denser problem and would otherwise win
        on FVE alone forever -- see the module docstring.

        Args:
            val: This epoch's validation metrics (``reconstruction_metrics``'s
                dict, or ``{}`` when there is no validation split).

        Returns:
            ``True`` if this epoch may become ``best.pth``.
        """
        if self.target_l0_controller is None:
            return True
        l0 = val.get("l0")
        if l0 is None:
            return True
        target = self.target_l0_controller.target
        return abs(l0 - target) <= self.best_checkpoint_l0_tolerance * target

    def _save(self, name: str, epoch: int, step: int, val: dict) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_cfg": self.model_cfg,
                "epoch": epoch,
                "step": step,
                "val": val,
            },
            self.outdir / name,
        )


__all__ = ["Trainer"]

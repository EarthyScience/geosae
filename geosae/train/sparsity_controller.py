"""Adapt a sparsity *penalty* toward a target mean L0 during training.

ReLU+ℓ1 reaches a given sparsity only through its penalty
(``l1_coeff``): unlike the Top-K family, nothing in its
architecture bounds the active-latent count directly, so matching a target
L0 (to compare variants at matched ℓ0/D, as the sweep needs) otherwise means
hand-tuning the penalty per ``(M, k)`` cell. :class:`TargetL0Controller`
automates that tuning as a simple closed-loop rule, driven entirely by the
training loop's own per-step L0 -- no extra forward pass, no held-out probe.
"""

from __future__ import annotations

import math


class TargetL0Controller:
    """Multiplicative proportional controller on a running L0 EMA.

    Each :meth:`update` call folds the step's observed mean L0 into an EMA,
    then rescales the penalty by ``exp(gain * (ema_l0 / target - 1))``: the
    penalty grows when the EMA sits above the target (too dense, tighten) and
    shrinks when it sits below (too sparse, relax). The multiplicative form
    keeps the penalty positive by construction and gives it geometric, scale
    -free steps, so one ``gain`` works whether the penalty needs to move by a
    factor of 2 or of 100 to reach the target.

    Two safeguards keep a far-off start (a fresh penalty that is orders of
    magnitude away from equilibrium, or a cold-start L0 spike) from
    overshooting and killing atoms before the loop can react:

    - **Per-update step bound** (``max_step_ratio``): the raw proportional
      exponent is clamped so a single :meth:`update` can move the penalty by
      at most a factor of ``max_step_ratio``, regardless of how large the
      instantaneous error is. This turns what would otherwise be a
      one-step, many-orders-of-magnitude jump (a huge log-error scaled by
      ``gain`` and only loosely clamped) into a bounded geometric walk with
      an effective time constant of ``~1 / (gain * max_step_ratio's log)``
      updates -- a few hundred steps at the defaults below, matching the
      loop's own EMA smoothing.
    - **Range clamp relative to the start** (``range_ratio``): the penalty
      returned by :meth:`update` is also kept within a factor of
      ``range_ratio`` of the *first* penalty this controller ever saw, on
      top of the absolute ``min_penalty``/``max_penalty`` failsafe. This
      bounds how far even a long run of same-direction steps can drift,
      without needing to guess the equilibrium penalty ahead of time.

    A controller can also hold off adjusting entirely for its first
    ``warmup_steps`` calls (e.g. while the learning-rate schedule is still
    ramping up and the observed L0 is not yet representative): those calls
    still fold into the running EMA -- so a real estimate exists once
    adjustment starts -- but return the penalty unchanged.

    Args:
        target: Desired mean L0 (active latents per row).
        gain: Proportional gain -- larger moves the penalty faster per step
            (before the ``max_step_ratio`` bound applies), at the cost of
            more oscillation around the target.
        ema_beta: EMA decay on the observed L0 (closer to 1.0 smooths more
            of the per-batch noise, at the cost of reacting more slowly).
        max_step_ratio: Upper bound (>1) on the multiplicative change a
            single :meth:`update` call may apply, in either direction.
        range_ratio: Upper bound (>1) on how far the penalty may drift from
            its starting value (first value passed to :meth:`update`), in
            either direction.
        warmup_steps: Number of initial :meth:`update` calls that only feed
            the running EMA and return the penalty unchanged.
        min_penalty: Absolute lower clamp, so a very sparse start cannot
            collapse the penalty to (or through) zero and stall the
            controller there.
        max_penalty: Absolute upper clamp, symmetric safety bound.
    """

    def __init__(
        self,
        target: float,
        *,
        gain: float = 0.05,
        ema_beta: float = 0.9,
        max_step_ratio: float = 1.1,
        range_ratio: float = 1000.0,
        warmup_steps: int = 0,
        min_penalty: float = 1e-8,
        max_penalty: float = 1e4,
    ) -> None:
        if target <= 0:
            raise ValueError(f"target must be positive, got {target}")
        if max_step_ratio <= 1.0:
            raise ValueError(f"max_step_ratio must be > 1, got {max_step_ratio}")
        if range_ratio <= 1.0:
            raise ValueError(f"range_ratio must be > 1, got {range_ratio}")
        self.target = float(target)
        self.gain = float(gain)
        self.ema_beta = float(ema_beta)
        self.max_step_ratio = float(max_step_ratio)
        self.range_ratio = float(range_ratio)
        self.warmup_steps = int(warmup_steps)
        self.min_penalty = float(min_penalty)
        self.max_penalty = float(max_penalty)
        self._ema_l0: float | None = None
        self._penalty_init: float | None = None
        self._n_updates = 0

    @property
    def ema_l0(self) -> float | None:
        """The running L0 estimate, or ``None`` before the first update."""
        return self._ema_l0

    def update(self, penalty: float, observed_l0: float) -> float:
        """Fold in one step's observed L0 and return the next penalty.

        Args:
            penalty: The penalty value the step just used.
            observed_l0: That step's mean active-latent count.

        Returns:
            The penalty to use for the next step. Unchanged from ``penalty``
            during the first ``warmup_steps`` calls; afterwards, moved by at
            most a factor of ``max_step_ratio`` and clamped to
            ``[min_penalty, max_penalty]`` and to within ``range_ratio`` of
            the first penalty this controller saw.
        """
        penalty = float(penalty)
        observed_l0 = float(observed_l0)
        if self._penalty_init is None:
            self._penalty_init = penalty
        self._ema_l0 = (
            observed_l0 if self._ema_l0 is None else self.ema_beta * self._ema_l0 + (1.0 - self.ema_beta) * observed_l0
        )
        self._n_updates += 1
        if self._n_updates <= self.warmup_steps:
            return penalty

        error = self._ema_l0 / self.target - 1.0
        step_bound = math.log(self.max_step_ratio)
        exponent = max(-step_bound, min(step_bound, self.gain * error))
        next_penalty = penalty * math.exp(exponent)

        lo = max(self.min_penalty, self._penalty_init / self.range_ratio)
        hi = min(self.max_penalty, self._penalty_init * self.range_ratio)
        return min(max(next_penalty, lo), hi)


__all__ = ["TargetL0Controller"]

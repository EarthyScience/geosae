"""The standing performance gate: seconds-per-run and peak memory at a fixed small size.

Pins `PanelAccumulator.update()` — the pass that dominates the evaluation pipeline's wall-clock — at
one small, fixed shape and fails if either its wall-clock or its peak memory regresses by more than
2x. This is a *regression* gate, not a benchmark: it catches a change that reintroduces an O(N)
Python loop or a dense intermediate. The reference numbers were measured CPU-only; re-measure and
update both constants together if a deliberate change moves the true baseline, never to silence a
real regression.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from geosae.eval.statistic import PanelAccumulator, calibrate_activation_bins

# Fixed small shape: cheap enough for CI, large enough that the F-length inner
# loop the fast paths replace would dominate if it ever came back.
_N, _D, _F, _C, _P, _B = 2000, 16, 64, 10, 2, 32

# Measured on the dev GPU node, CPU-only (see module docstring); the gate is
# 2x these, not the raw numbers, so ordinary run-to-run jitter never trips it.
_BASELINE_SECONDS = 0.02
_BASELINE_PEAK_BYTES = 15_000_000
_REGRESSION_FACTOR = 2.0


def _synthetic_batch():
    rng = np.random.default_rng(0)
    decoder = rng.normal(size=(_F, _D))
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    codes = np.clip(rng.normal(0.3, 1.0, size=(_N, _F)), 0, None)
    codes *= rng.uniform(size=(_N, _F)) < 0.1
    ehat = codes @ decoder
    e = ehat + rng.normal(scale=0.15, size=(_N, _D))
    concept_mask = rng.uniform(size=(_N, _C)) < 0.2
    product_valid = np.ones((_N, _P), dtype=bool)
    edges = calibrate_activation_bins(codes, n_bins=_B)
    return e, ehat, codes, concept_mask, product_valid, edges


def _fresh_accumulator(edges):
    return PanelAccumulator.create(
        n_features=_F,
        embedding_dim=_D,
        concept_names=[f"c{i}" for i in range(_C)],
        product_names=[f"p{i}" for i in range(_P)],
        product_of_concept=np.arange(_C) % _P,
        bin_edges=edges,
    )


def test_panel_accumulator_update_cost_has_not_regressed():
    """`PanelAccumulator.update()` at a fixed small shape: seconds and peak memory, both gated at 2x."""
    e, ehat, codes, concept_mask, product_valid, edges = _synthetic_batch()

    # One untimed warm-up call: the first call in a process pays one-off costs
    # (e.g. `_bin_index`'s lazy `import torch`) that are not part of the
    # steady-state cost this gate is pinning.
    _fresh_accumulator(edges).update(e, ehat, codes, concept_mask=concept_mask, product_valid=product_valid)

    timings = []
    peak_bytes = 0
    for _ in range(3):
        acc = _fresh_accumulator(edges)
        tracemalloc.start()
        t0 = time.perf_counter()
        acc.update(e, ehat, codes, concept_mask=concept_mask, product_valid=product_valid)
        timings.append(time.perf_counter() - t0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes = max(peak_bytes, peak)

    seconds = min(timings)  # the minimum is the least noise-inflated estimate of the true cost
    assert seconds < _BASELINE_SECONDS * _REGRESSION_FACTOR, (
        f"PanelAccumulator.update() took {seconds * 1000:.1f} ms at N={_N}, F={_F} "
        f"(C={_C}, P={_P}, B={_B}) — more than {_REGRESSION_FACTOR}x the "
        f"{_BASELINE_SECONDS * 1000:.1f} ms baseline. Re-profile and update docs/performance.md "
        f"if this is a deliberate change, or fix the regression."
    )
    assert peak_bytes < _BASELINE_PEAK_BYTES * _REGRESSION_FACTOR, (
        f"PanelAccumulator.update() peak memory was {peak_bytes / 1e6:.1f} MB — more than "
        f"{_REGRESSION_FACTOR}x the {_BASELINE_PEAK_BYTES / 1e6:.1f} MB baseline."
    )

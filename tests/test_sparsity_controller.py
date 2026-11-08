"""TargetL0Controller: convergence of the closed-loop penalty update."""

import pytest

from geosae.train.sparsity_controller import TargetL0Controller


def _simulated_l0(penalty: float, scale: float = 100.0) -> float:
    """A monotone decreasing stand-in for 'sparsity penalty -> mean L0'."""
    return scale / penalty


def test_controller_converges_to_the_target_l0():
    target = 10.0
    ctrl = TargetL0Controller(target, gain=0.3, ema_beta=0.0)
    penalty = 1.0
    for _ in range(300):
        observed = _simulated_l0(penalty)
        penalty = ctrl.update(penalty, observed)
    assert _simulated_l0(penalty) == pytest.approx(target, rel=0.05)


def test_controller_converges_from_above_the_target_too():
    target = 8.0
    ctrl = TargetL0Controller(target, gain=0.3, ema_beta=0.0)
    penalty = 50.0  # starts far too tight (observed L0 << target)
    for _ in range(300):
        observed = _simulated_l0(penalty)
        penalty = ctrl.update(penalty, observed)
    assert _simulated_l0(penalty) == pytest.approx(target, rel=0.05)


def test_penalty_grows_when_l0_is_above_target():
    ctrl = TargetL0Controller(target=10.0, gain=0.3, ema_beta=0.0)
    next_penalty = ctrl.update(1.0, observed_l0=50.0)  # way too dense
    assert next_penalty > 1.0


def test_penalty_shrinks_when_l0_is_below_target():
    ctrl = TargetL0Controller(target=10.0, gain=0.3, ema_beta=0.0)
    next_penalty = ctrl.update(1.0, observed_l0=1.0)  # way too sparse
    assert next_penalty < 1.0


def test_penalty_is_unchanged_exactly_at_target():
    ctrl = TargetL0Controller(target=10.0, gain=0.3, ema_beta=0.0)
    assert ctrl.update(1.0, observed_l0=10.0) == pytest.approx(1.0)


def test_penalty_stays_positive_and_clamped():
    ctrl = TargetL0Controller(target=10.0, gain=5.0, ema_beta=0.0, min_penalty=1e-6, max_penalty=1e3)
    penalty = 1.0
    for _ in range(50):
        penalty = ctrl.update(penalty, observed_l0=0.0)  # push toward the floor every step
    assert penalty >= 1e-6

    penalty = 1.0
    for _ in range(50):
        penalty = ctrl.update(penalty, observed_l0=1e9)  # push toward the ceiling every step
    assert penalty <= 1e3


def test_ema_smooths_noisy_observations():
    ctrl = TargetL0Controller(target=10.0, gain=0.3, ema_beta=0.9)
    ctrl.update(1.0, observed_l0=10.0)
    assert ctrl.ema_l0 == pytest.approx(10.0)
    ctrl.update(1.0, observed_l0=100.0)  # one noisy spike
    assert ctrl.ema_l0 == pytest.approx(0.9 * 10.0 + 0.1 * 100.0)


def test_target_must_be_positive():
    with pytest.raises(ValueError, match="target"):
        TargetL0Controller(0.0)


def test_max_step_ratio_must_exceed_one():
    with pytest.raises(ValueError, match="max_step_ratio"):
        TargetL0Controller(10.0, max_step_ratio=1.0)


def test_range_ratio_must_exceed_one():
    with pytest.raises(ValueError, match="range_ratio"):
        TargetL0Controller(10.0, range_ratio=1.0)


@pytest.mark.parametrize("start_ratio", [20.0, 0.05])
def test_converges_from_a_far_off_start_without_overshoot(start_ratio):
    """A far-off start (penalty 20x above, or 0.05x below, equilibrium) must
    converge monotonically, never overshooting past equilibrium to the other
    side by more than a small margin.

    A penalty spiking thousands of times past its equilibrium kills atoms
    before the loop can react; it must approach the equilibrium smoothly.
    """
    target = 10.0
    ctrl = TargetL0Controller(target, gain=0.3, ema_beta=0.0)
    # Equilibrium penalty for `_simulated_l0`: scale / target.
    equilibrium = 100.0 / target
    start_penalty = equilibrium * start_ratio
    penalty = start_penalty
    seen = []
    for _ in range(300):
        observed = _simulated_l0(penalty)
        penalty = ctrl.update(penalty, observed)
        seen.append(penalty)
    # Never move further from equilibrium than the starting offset, and
    # never cross more than 10% past equilibrium on the other side.
    assert max(seen) <= max(start_penalty, equilibrium * 1.1)
    assert min(seen) >= min(start_penalty, equilibrium * 0.9)
    assert _simulated_l0(penalty) == pytest.approx(target, rel=0.1)


def test_max_step_ratio_bounds_a_single_update():
    ctrl = TargetL0Controller(10.0, gain=10.0, ema_beta=0.0, max_step_ratio=1.1)
    next_penalty = ctrl.update(1.0, observed_l0=1e6)  # huge instantaneous error
    assert next_penalty == pytest.approx(1.1)


def test_updates_during_warmup_hold_the_penalty_but_feed_the_ema():
    ctrl = TargetL0Controller(10.0, gain=0.3, ema_beta=0.0, warmup_steps=5)
    penalty = 1.0
    for _ in range(5):
        penalty = ctrl.update(penalty, observed_l0=1000.0)  # way too dense
    assert penalty == pytest.approx(1.0)  # held through warmup
    assert ctrl.ema_l0 == pytest.approx(1000.0)  # but the EMA is already warm
    penalty = ctrl.update(penalty, observed_l0=1000.0)
    assert penalty > 1.0  # adjustment starts right after warmup ends

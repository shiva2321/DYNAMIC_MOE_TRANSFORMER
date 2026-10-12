import numpy as np
import pytest

from growmoe.competence import LossChangeMonitor

ALPHA, WARMUP, HORIZON = 0.01, 50, 300


def _monitor(**kw):
    kw.setdefault("alpha", ALPHA)
    kw.setdefault("warmup", WARMUP)
    kw.setdefault("horizon", HORIZON)
    kw.setdefault("n_null", 1000)
    return LossChangeMonitor(**kw)


def _alarms(monitor, losses):
    return [monitor.update(float(x)) for x in losses]


def _stationary(rng, n, mean=1.0, sd=0.1):
    return mean + sd * rng.standard_normal(n)


def test_warmup_is_silent_and_the_monitor_arms_after_exactly_warmup_updates():
    m = _monitor()
    rng = np.random.default_rng(0)
    for x in _stationary(rng, WARMUP - 1):
        assert m.update(float(x)) is False
        assert not m.armed
    assert m.update(1.0) is False
    assert m.armed and m.lam > 0 and m.std > 0


def test_stationary_false_alarm_rate_is_within_three_times_alpha():
    runs, hits = 400, 0
    for seed in range(runs):
        m = _monitor(seed=seed)
        hits += any(_alarms(m, _stationary(np.random.default_rng(1000 + seed), WARMUP + HORIZON)))
    assert hits / runs <= 3 * ALPHA


def test_a_three_sigma_step_is_detected_within_thirty_steps_in_over_ninety_percent_of_runs():
    runs, pre, found = 200, 100, 0
    for seed in range(runs):
        rng = np.random.default_rng(5000 + seed)
        m = _monitor(seed=seed)
        _alarms(m, _stationary(rng, WARMUP + pre))
        after = _alarms(m, _stationary(rng, 30, mean=1.3))                  # +3 sigma (sd = 0.1)
        found += any(after)
    assert found / runs > 0.90


def test_a_two_sigma_step_is_also_detected_within_thirty_steps():
    runs, found = 100, 0
    for seed in range(runs):
        rng = np.random.default_rng(7000 + seed)
        m = _monitor(seed=seed)
        _alarms(m, _stationary(rng, WARMUP + 100))
        found += any(_alarms(m, _stationary(rng, 30, mean=1.2)))          # +2 sigma
    assert found / runs > 0.95


def test_threshold_is_never_below_the_gaussian_null_and_rises_for_skewed_warmups():
    # the Gaussian floor exists because a bootstrap threshold on its own is too noisy (the false-alarm
    # probability is convex in lambda); the bootstrap part exists because skewed losses need a higher one
    lam_gauss, lam_expo = [], []
    for seed in range(20):
        g, e = _monitor(seed=seed), _monitor(seed=seed)
        _alarms(g, np.random.default_rng(100 + seed).standard_normal(WARMUP))
        _alarms(e, np.random.default_rng(200 + seed).exponential(1.0, WARMUP))
        assert g.lam >= g.lam_gaussian and e.lam >= e.lam_gaussian
        lam_gauss.append(g.lam)
        lam_expo.append(e.lam)
    assert np.mean(lam_expo) > 1.5 * g.lam_gaussian          # measured 2.2x the floor
    assert np.mean(lam_gauss) < 1.25 * g.lam_gaussian        # measured 1.08x: Gaussian warm-ups stay near it


def test_a_decrease_in_loss_never_alarms():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        m = _monitor(seed=seed)
        _alarms(m, _stationary(rng, WARMUP))
        assert not any(_alarms(m, _stationary(rng, 300, mean=0.5)))


def test_decisions_are_invariant_to_the_scale_and_offset_of_the_loss():
    rng = np.random.default_rng(3)
    x = np.concatenate([_stationary(rng, WARMUP + 80), _stationary(rng, 60, mean=1.25)])
    a = _alarms(_monitor(seed=7), x)
    b = _alarms(_monitor(seed=7), 1000.0 * x + 5.0)
    assert a == b and any(a) and not any(a[:WARMUP + 80])


def test_reset_clears_state_and_the_monitor_recalibrates_to_the_new_level():
    rng = np.random.default_rng(4)
    m = _monitor()
    _alarms(m, _stationary(rng, WARMUP + 20))
    assert any(_alarms(m, _stationary(rng, 40, mean=2.0)))                  # alarmed
    assert m.statistic > m.lam
    m.reset()
    assert not m.armed and m.statistic == 0.0
    new_level = _stationary(rng, WARMUP + 200, mean=2.0)                    # the new normal
    assert not any(_alarms(m, new_level))
    assert m.armed


def test_lower_alpha_gives_a_higher_threshold():
    warm = _stationary(np.random.default_rng(6), WARMUP)
    lo, hi = _monitor(alpha=0.1, seed=1), _monitor(alpha=0.001, seed=1)
    _alarms(lo, warm)
    _alarms(hi, warm)
    assert lo.lam < hi.lam


def test_same_seed_and_stream_give_identical_decisions():
    x = np.concatenate([_stationary(np.random.default_rng(8), WARMUP + 50),
                        _stationary(np.random.default_rng(9), 50, mean=1.2)])
    assert _alarms(_monitor(seed=3), x) == _alarms(_monitor(seed=3), x)


def test_non_finite_losses_and_bad_arguments_are_rejected():
    m = _monitor()
    with pytest.raises(ValueError):
        m.update(float("nan"))
    with pytest.raises(ValueError):
        m.update(float("inf"))
    for bad in (dict(alpha=0.0), dict(alpha=1.0), dict(warmup=3), dict(horizon=0), dict(delta=-0.1)):
        with pytest.raises(ValueError):
            _monitor(**bad)


def test_constant_losses_do_not_crash_and_a_later_change_is_flagged():
    m = _monitor()
    assert not any(_alarms(m, np.zeros(WARMUP + 20)))
    assert any(_alarms(m, np.full(5, 0.5)))

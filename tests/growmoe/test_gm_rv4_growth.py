"""RV4 stage 4c, the pure pieces: the fit gap, progress, maturity, cooldown, the growth increment and the guard (spec sections 6.1-6.3)."""

import pytest
import torch

from growmoe.experts import MLPExpert
from growmoe.gapgrow import GapMonitor, guard_reverts, params_of_widen, widen_increment


def _t(v):
    return torch.tensor([float(v)])


def test_gap_is_one_minus_post_over_pre_over_the_window():
    m = GapMonitor(window=3)
    assert m.gap() is None                                                           # the window is not full
    for pre, post in ((2.0, 1.0), (2.0, 1.5), (4.0, 2.0)):
        m.add(_t(pre), _t(post))
    assert m.gap() == pytest.approx(1 - (1.0 + 1.5 + 2.0) / (2.0 + 2.0 + 4.0))
    m.add(_t(1.0), _t(1.0))                                                          # the oldest batch drops out
    assert m.gap() == pytest.approx(1 - (1.5 + 2.0 + 1.0) / (2.0 + 4.0 + 1.0))


def test_gap_sums_all_rows_of_a_batch_and_ignores_empty_batches():
    m = GapMonitor(window=2)
    m.add(torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.5]))                       # sums 4.0 and 2.0
    m.add(torch.empty(0), torch.empty(0))
    assert m.age == 1 and m.gap() is None
    m.add(torch.tensor([4.0]), torch.tensor([2.0]))
    assert m.gap() == pytest.approx(1 - (2.0 + 2.0) / (4.0 + 4.0))


def test_a_window_with_no_loss_has_no_gap():
    m = GapMonitor(window=2)
    for _ in range(2):
        m.add(_t(0.0), _t(0.0))
    assert m.gap() is None


def test_improvement_compares_the_last_window_with_the_one_before():
    m = GapMonitor(window=2)
    assert m.improvement() is None
    for v in (4.0, 4.0, 2.0, 2.0):
        m.add(_t(v), _t(v / 2))
    assert m.improvement() == pytest.approx(0.5)


def test_maturity_needs_age_and_a_plateau():
    flat = GapMonitor(window=2)
    for _ in range(4):
        flat.add(_t(2.0), _t(2.0))
    assert flat.age == 4 and flat.mature(min_age=4, plateau=0.95) and not flat.mature(min_age=5, plateau=0.95)
    falling = GapMonitor(window=2)
    for v in (4.0, 4.0, 2.0, 2.0):
        falling.add(_t(v), _t(v))
    assert not falling.mature(min_age=1, plateau=0.95)                                # still improving by half


def test_growth_resets_the_windows_and_starts_the_cooldown():
    m = GapMonitor(window=2)
    assert m.cooled(5)                                                                # never grown
    for _ in range(4):
        m.add(_t(2.0), _t(1.0))
    m.grew()
    assert m.age == 0 and m.gap() is None and m.improvement() is None and not m.cooled(2)
    m.add(_t(2.0), _t(1.0))
    assert not m.cooled(2)
    m.add(_t(2.0), _t(1.0))
    assert m.cooled(2)


def test_the_recent_mean_loss_uses_the_last_batches_only():
    m = GapMonitor(window=3)
    assert m.recent_pre_mean(2) != m.recent_pre_mean(2)                               # nan before any batch
    for v in (1.0, 2.0, 3.0, 4.0):
        m.add(_t(v), _t(v))
    assert m.recent_pre_mean(2) == pytest.approx(3.5) and m.recent_pre_mean(100) == pytest.approx(2.5)


def test_widen_increment_doubles_and_respects_the_cap():
    assert widen_increment(16, 2.0, 1024, 4) == 16
    assert widen_increment(700, 2.0, 1024, 4) == 324
    assert widen_increment(1024, 2.0, 1024, 4) == 0
    assert widen_increment(1022, 2.0, 1024, 4) == 0                                   # room 2 is below min_widen
    assert widen_increment(8, 1.5, 1024, 4) == 4


def test_params_of_widen_matches_the_real_expert():
    e = MLPExpert(d_in=16, hidden=8, depth=2, n_classes=4)
    before = e.num_params()
    e.widen(5)
    assert e.num_params() - before == params_of_widen(16, 2, 5)


def test_the_guard_reverts_only_when_the_loss_got_worse_by_more_than_eps():
    assert guard_reverts(1.0, 1.03, 0.02)
    assert not guard_reverts(1.0, 1.02, 0.02)                                         # exactly eps is not more than eps
    assert not guard_reverts(1.0, 0.9, 0.02)

def test_maturity_at_exactly_the_plateau_ratio():
    m = GapMonitor(window=2)
    for v in (4.0, 4.0, 2.0, 2.0):                                                    # the last window is exactly half of the one before
        m.add(_t(v), _t(v))
    assert m.improvement() == 0.5
    assert m.mature(min_age=1, plateau=0.5) and not m.mature(min_age=1, plateau=0.5001)

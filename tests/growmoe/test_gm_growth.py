import dataclasses
from functools import lru_cache

import numpy as np
import pytest
import torch

from growmoe import streams as S5
from growmoe.config import GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.growth import GrowthDecision, GrowthPolicy, plateau

D, C = 32, 8


def _units(n, seed):
    g = np.random.default_rng(seed)
    x = g.standard_normal((n, D))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()


def _task(n_tr, n_va, seed=0, teacher_hidden=4):
    """Teacher-student data: labels are the argmax of a small random teacher, so a student with enough width can learn
    them. (A 96-wide teacher with 2000 samples is data-limited at EVERY student width: accuracy stays at 0.62-0.65 and
    wide students overfit, so declining to grow there is correct and it cannot serve as the capacity-limited case.)"""
    teacher = MLPExpert(d_in=D, hidden=teacher_hidden, depth=1, n_classes=C, seed=1234)
    teacher.mark_seen(torch.arange(C))
    x_tr, x_va = _units(n_tr, 10 + seed), _units(n_va, 500 + seed)
    with torch.no_grad():
        y_tr, y_va = teacher(x_tr).argmax(dim=1), teacher(x_va).argmax(dim=1)
    keep = torch.isin(y_va, y_tr.unique())
    return x_tr, y_tr, x_va[keep], y_va[keep]


def _fit(model, x, y, steps, seed=0, lr=3e-3):
    g = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.from_numpy(g.integers(0, len(x), 64))
        model.train_step(x[idx], y[idx], opt)
    return model


def _val_loss(model, x, y):
    with torch.no_grad():
        return model.loss(x, y).item()


@lru_cache(maxsize=None)
def _capacity_limited_cached(seed):
    x_tr, y_tr, x_va, y_va = _task(2000, 600, seed)
    student = _fit(MLPExpert(d_in=D, hidden=4, depth=1, n_classes=C, seed=seed), x_tr, y_tr, 1500)
    return student, (x_tr, y_tr, x_va, y_va)


def _capacity_limited(seed=0):
    """A width-4 student at its plateau on a task a width-16 student fits clearly better (measured val loss 0.75 vs 0.53)."""
    student, data = _capacity_limited_cached(seed)
    return student.clone(), data


def _data_limited(seed=0):
    """60 training samples and a wide student: it memorises them and the held-out loss is far above the train loss."""
    x_tr, y_tr, x_va, y_va = _task(60, 600, seed)
    student = _fit(MLPExpert(d_in=D, hidden=64, depth=1, n_classes=C, seed=seed), x_tr, y_tr, 600)
    return student, (x_tr, y_tr, x_va, y_va)


# ------------------------------------------------------------------------- plateau
def test_plateau_detects_a_flat_history_and_ignores_noise():
    rng = np.random.default_rng(0)
    assert plateau(list(1.0 + 0.01 * rng.standard_normal(200)), window=40, tol=0.01)    # the window means differ by ~0.002


def test_plateau_is_false_while_the_loss_is_still_falling_or_history_is_short():
    assert not plateau(list(np.linspace(1.0, 0.2, 200)), window=40, tol=0.01)
    assert not plateau([1.0] * 70, window=40, tol=0.01)                      # fewer than 2 windows of evidence


def test_plateau_notices_a_fresh_improvement_after_a_long_flat_stretch():
    history = [1.0] * 100 + list(np.linspace(1.0, 0.5, 40))
    assert not plateau(history, window=40, tol=0.01)


# --------------------------------------------------------------------------- growth
def test_a_capacity_limited_student_grows_function_preservingly_and_the_growth_helps():
    student, (x_tr, y_tr, x_va, y_va) = _capacity_limited()
    reference, outputs_before, params_before = student.clone(), student(x_va).detach().clone(), student.num_params()
    d = GrowthPolicy(GrowthCfg()).step(student, x_tr, y_tr, x_va, y_va, np.random.default_rng(0))
    assert isinstance(d, GrowthDecision) and d.kind in ("widen", "deepen")
    assert d.gain >= GrowthCfg().min_gain and d.p_value < GrowthCfg().alpha
    assert d.params_added == student.num_params() - params_before > 0
    assert torch.allclose(student(x_va), outputs_before, atol=1e-5)          # the commit itself changes no output
    # and the commit helps: train both for the same steps; the grown student ends with the lower held-out loss
    grown = _fit(student, x_tr, y_tr, 300, seed=5)
    plain = _fit(reference, x_tr, y_tr, 300, seed=5)
    assert _val_loss(grown, x_va, y_va) < _val_loss(plain, x_va, y_va)


def test_a_data_limited_student_is_not_grown():
    student, (x_tr, y_tr, x_va, y_va) = _data_limited()
    assert _val_loss(student, x_va, y_va) > 3.0 * _val_loss(student, x_tr, y_tr)        # a large generalisation gap
    before = student.num_params()
    d = GrowthPolicy(GrowthCfg()).step(student, x_tr, y_tr, x_va, y_va, np.random.default_rng(0))
    assert d.kind == "none" and d.params_added == 0 and student.num_params() == before
    assert all(v["vs_status"] < 0 for v in d.tried.values())              # every candidate ends worse than the status quo


# --------------------------------------------------------------------------- back-off
def test_backoff_doubles_on_rejection_skips_calls_and_resets_on_acceptance():
    cfg = GrowthCfg()
    policy = GrowthPolicy(cfg)
    assert policy.backoff == cfg.backoff0 and policy.cooldown == 0
    lim, data_lim = _capacity_limited(), _data_limited()
    rng = np.random.default_rng(0)

    d = policy.step(data_lim[0], *data_lim[1], rng)                       # rejection 1
    assert d.kind == "none" and policy.cooldown == cfg.backoff0 and policy.backoff == 2 * cfg.backoff0

    d = policy.step(data_lim[0], *data_lim[1], rng)                       # skipped: still cooling down
    assert d.kind == "none" and d.reason == "backoff" and policy.cooldown == 0

    d = policy.step(data_lim[0], *data_lim[1], rng)                       # rejection 2: wait doubles
    assert d.kind == "none" and policy.cooldown == 2 * cfg.backoff0 and policy.backoff == 4 * cfg.backoff0

    for _ in range(2 * cfg.backoff0):
        assert policy.step(lim[0], *lim[1], rng).reason == "backoff"
    d = policy.step(lim[0], *lim[1], rng)                                 # an acceptance resets everything
    assert d.kind != "none" and policy.backoff == cfg.backoff0 and policy.cooldown == 0


def test_backoff_is_capped():
    cfg = GrowthCfg(backoff0=1, max_backoff=4, train_steps=60)               # rejection-only: a short probe is enough
    policy = GrowthPolicy(cfg)
    student, data = _data_limited()
    rng = np.random.default_rng(0)
    for _ in range(40):
        policy.step(student, *data, rng)
    assert policy.backoff == 4


# ------------------------------------------------------------------ determinism / caps / guards
def test_decision_and_resulting_weights_are_deterministic_given_the_rng_seed():
    a, data = _capacity_limited()
    b = a.clone()
    da = GrowthPolicy(GrowthCfg()).step(a, *data, np.random.default_rng(7))
    db = GrowthPolicy(GrowthCfg()).step(b, *data, np.random.default_rng(7))
    assert dataclasses.asdict(da) == dataclasses.asdict(db)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_no_candidate_is_proposed_beyond_the_capacity_caps():
    student, data = _capacity_limited()
    d = GrowthPolicy(GrowthCfg(max_hidden=student.hidden, max_depth=len(student.blocks))).step(
        student, *data, np.random.default_rng(0))
    assert d.kind == "none" and d.reason == "at capacity cap"


def test_a_cap_on_depth_alone_only_removes_the_deepen_candidate():
    student, data = _capacity_limited()
    d = GrowthPolicy(GrowthCfg(max_depth=len(student.blocks), train_steps=60)).step(
        student, *data, np.random.default_rng(0))
    assert d.tried and all(k.startswith("widen") for k in d.tried) and len(student.blocks) == 1


def test_validation_labels_the_expert_has_never_seen_are_rejected():
    student, (x_tr, y_tr, x_va, y_va) = _capacity_limited()
    c = int(y_va[0])                                        # a class that is validated on ...
    keep = y_tr != c                                        # ... but absent from the training data
    student.seen[c] = False                                 # ... and unknown to the expert
    with pytest.raises(ValueError):
        GrowthPolicy(GrowthCfg()).step(student, x_tr[keep], y_tr[keep], x_va, y_va, np.random.default_rng(0))


# ------------------------------------------------------------------ multi-scale candidates (Spike 3)
def test_widen_increments_are_multiples_of_the_current_width_unique_sorted_and_capped():
    policy = GrowthPolicy(GrowthCfg(widen_factors=(1.5, 4.0, 16.0), min_widen=4, max_hidden=1024))
    assert policy.widen_increments(4) == [4, 12, 60]
    assert policy.widen_increments(64) == [32, 192, 960]
    assert GrowthPolicy(GrowthCfg(max_hidden=100)).widen_increments(64) == [32, 36]      # larger factors are capped, then merged
    assert GrowthPolicy(GrowthCfg(max_hidden=64)).widen_increments(64) == []
    assert GrowthPolicy(GrowthCfg(widen_factors=(0.5, 1.0), min_widen=4)).widen_increments(4) == [4]


def test_selection_commits_the_smallest_sufficient_candidate():
    select = GrowthPolicy.select
    gain, level = 0.02, 0.0125
    # candidate -> (gain vs control, p, parameters added, gain vs the status quo)
    cands = {"widen+4": (0.005, 0.30, 1000, 0.01), "widen+12": (0.03, 0.001, 3000, 0.05),
             "widen+60": (0.06, 0.001, 15000, 0.08), "deepen": (0.01, 0.2, 1400, 0.0)}
    assert select(cands, gain, level) == "widen+12"                       # passes every gate and adds the fewest parameters
    assert select({"widen+60": (0.06, 0.001, 15000, 0.08)}, gain, level) == "widen+60"
    assert select({"widen+4": (0.005, 0.3, 1000, 0.01)}, gain, level) is None
    assert select({"widen+60": (0.06, 0.2, 15000, 0.08)}, gain, level) is None          # large but not significant
    assert select({"widen+60": (0.011, 0.001, 15000, 0.08)}, gain, level) is None        # significant but below min_gain
    tie = {"widen+8": (0.05, 0.001, 2000, 0.05), "deepen": (0.05, 0.001, 2000, 0.05)}
    assert select(tie, gain, level) == "deepen"                           # equal size: alphabetical, so deterministic


def test_a_candidate_must_also_beat_the_status_quo_not_just_a_degrading_control():
    # Measured: a memorised expert starts at held-out loss 2.81, its equally trained control degrades to 4.6 and the
    # widest shadow to 4.2 -- "+11 % over the control" while 49 % worse than where the expert started
    assert GrowthPolicy.select({"widen+960": (0.11, 0.001, 262000, -0.49)}, 0.02, 0.0125) is None
    assert GrowthPolicy.select({"widen+960": (0.11, 0.001, 262000, 0.0)}, 0.02, 0.0125) is None       # must be strictly better


def test_a_far_jump_is_found_when_small_increments_show_no_gain():
    # Spike 3: for a width-32 depth-2 teacher the relative held-out gain of widening a width-4 student is
    # +0.0 % (+4 units), +1.7 % (+12), +2.7 % (+28) and +5.3 % (+60); a policy that only tries +50 % never sees it
    st = S5.s5_ladder(0, width=32, depth=2, n_train=20000, n_test=500)
    x, y = st.segments[0].x, st.segments[0].y
    cut = int(0.8 * len(y))
    student = MLPExpert(d_in=128, hidden=4, depth=1, n_classes=8, seed=0)
    student.mark_seen(torch.arange(8))
    g = np.random.default_rng(31)
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    for _ in range(600):
        i = torch.from_numpy(g.integers(0, cut, 64))
        student.train_step(x[i], y[i], opt)
    d = GrowthPolicy(GrowthCfg()).step(student, x[:cut], y[:cut], x[cut:], y[cut:], np.random.default_rng(0))
    assert d.kind == "widen" and d.increment > 4 and student.hidden == 4 + d.increment
    assert d.gain >= GrowthCfg().min_gain

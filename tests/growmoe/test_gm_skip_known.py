"""The skip rule (E-063, OFF by default): a batch routed to a settled expert that already predicts it takes no optimiser steps.

``skip_known`` = k skips the steps of a batch whose mean prequential loss (taken before any step) is at most k x ln(classes the expert has seen). The batch still enters the expert's reservoir, its signature and
(in a UniPool) its closed-form head. With the flag None (the default) nothing changes."""

import math

import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.pools import phylo as phylo_module
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes
from unimoe.pools import UniCfg, UniPool


def _cfg(st, cls=PhyloCfg, **kw):
    return cls(d=st.d, n_classes=st.n_classes, seed=3, inner_steps=3, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", **kw)


def _stream():
    return function_regimes(5, 3, d=32, n_classes=4, n_train=300, n_test=50)


def _feed(pool, st, lo=0, hi=300, step=50, seg=0):
    s = st.segments[seg]
    for a in range(lo, hi, step):
        pool.observe(s.x[a:a + step], y=s.y[a:a + step])


def _weights(pool, eid=0):
    return {k: v.clone() for k, v in pool.experts[eid].module.state_dict().items()}


def test_the_flag_is_off_by_default_and_validated():
    assert PhyloCfg().skip_known is None and PhyloCfg().skip_min_assigned == 256
    assert PhyloCfg(skip_known=0.1).skip_known == 0.1 and PhyloCfg(skip_known=1.0).skip_known == 1.0
    for bad in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError):
            PhyloCfg(skip_known=bad)
    with pytest.raises(ValueError):
        PhyloCfg(skip_min_assigned=-1)


def test_with_the_flag_off_every_routed_batch_takes_optimiser_steps_and_none_is_skipped():
    st = _stream()
    pool = PhyloMoE(_cfg(st))
    _feed(pool, st)
    s = pool.skip_stats
    assert s["skipped"] == 0 and s["skipped_rows"] == 0 and s["trained"] >= 1 and s["trained_rows"] >= 50
    assert pool.summary()["skip"] == s


def test_a_threshold_that_never_fires_changes_nothing():
    st = _stream()
    off, never = PhyloMoE(_cfg(st)), PhyloMoE(_cfg(st, skip_known=1e-12, skip_min_assigned=0))
    _feed(off, st)
    _feed(never, st)
    a, b = _weights(off), _weights(never)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert off.skip_stats == never.skip_stats and never.skip_stats["skipped"] == 0


def test_a_known_batch_takes_no_optimiser_step_but_enters_the_reservoir_and_the_counters():
    st = _stream()
    pool = PhyloMoE(_cfg(st, skip_known=1.0, skip_min_assigned=0, replay_size=2000))               # a reservoir that does not cap within the test
    _feed(pool, st, 0, 250)
    rec = pool.experts[0]
    before, n_rep, n_asg = _weights(pool), rec.replay.n, rec.n_assigned
    s0 = dict(pool.skip_stats)
    _feed(pool, st, 250, 300)                                                                       # the regime the expert knows: its loss is under chance
    assert pool.skip_stats["skipped"] == s0["skipped"] + 1 and pool.skip_stats["skipped_rows"] == s0["skipped_rows"] + 50
    assert pool.skip_stats["trained"] == s0["trained"]
    after = _weights(pool)
    assert all(torch.equal(before[k], after[k]) for k in before)                                    # no step was taken
    assert rec.replay.n == n_rep + 50 and rec.n_assigned == n_asg + 50                              # but the batch is remembered


def test_a_batch_on_which_the_competence_alarm_fires_is_neither_skipped_nor_trained(monkeypatch):
    st = _stream()
    pool = PhyloMoE(_cfg(st, skip_known=1.0, skip_min_assigned=0, replay_size=2000))
    _feed(pool, st, 0, 250)
    monkeypatch.setattr(phylo_module.LossChangeMonitor, "update", lambda self, v: True)          # the monitor reports a loss change on every sample
    s0 = dict(pool.skip_stats)
    _feed(pool, st, 250, 300)
    assert pool.skip_stats == s0                                                                    # the alarm path decides: the skip rule does not override it, and nothing trains


def test_an_unknown_batch_is_trained_even_with_the_flag_on():
    st = _stream()
    pool = PhyloMoE(_cfg(st, skip_known=0.05, skip_min_assigned=0))                                 # only a near-perfect fit may skip
    _feed(pool, st, 0, 250)
    before, s0 = _weights(pool), dict(pool.skip_stats)
    _feed(pool, st, 250, 300)
    assert pool.skip_stats["trained"] == s0["trained"] + 1 and pool.skip_stats["skipped"] == s0["skipped"]
    after = _weights(pool)
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_a_young_expert_never_skips():
    st = _stream()
    pool = PhyloMoE(_cfg(st, skip_known=1.0, skip_min_assigned=10 ** 9))
    _feed(pool, st)
    assert pool.skip_stats["skipped"] == 0 and pool.skip_stats["trained"] >= 1


def test_the_decision_rule_at_its_edges():
    st = _stream()
    pool = PhyloMoE(_cfg(st, skip_known=0.5, skip_min_assigned=100))
    _feed(pool, st)
    rec = pool.experts[0]
    chance = math.log(max(int(rec.module.seen.sum()), 2))
    at = lambda v: torch.full((4,), v, dtype=torch.float64)                                         # float64: the mean of equal values is exact
    assert rec.n_assigned >= 100 and rec.state == "active"
    assert pool._known_batch(rec, at(0.5 * chance)) is True                                         # at the bar: known
    assert pool._known_batch(rec, at(0.5 * chance * 1.0001)) is False                               # just above: trained
    assert pool._known_batch(rec, at(0.0)) is True
    assert pool._known_batch(rec, torch.tensor([0.0, 0.0, 0.0, 4.0 * chance], dtype=torch.float64)) is False       # the MEAN decides: one bad row among four pulls it over the bar
    rec.state = "infant"
    assert pool._known_batch(rec, at(0.0)) is False                                                 # only a settled expert skips
    rec.state = "active"
    rec.n_assigned = 99
    assert pool._known_batch(rec, at(0.0)) is False                                                 # and only one that has trained on enough samples
    rec.n_assigned = 100
    assert pool._known_batch(rec, at(0.0)) is True
    off = PhyloMoE(_cfg(st))
    _feed(off, st)
    assert off._known_batch(off.experts[0], at(0.0)) is False                                       # flag off: never


def test_the_chance_loss_floor_is_two_classes():
    st = _stream()
    pool = PhyloMoE(_cfg(st, skip_known=1.0, skip_min_assigned=0))
    _feed(pool, st)
    rec = pool.experts[0]
    rec.module.seen.zero_()
    rec.module.seen[0] = True                                                                       # one class seen: the bar is ln 2, never 0
    at = lambda v: torch.full((4,), v, dtype=torch.float64)
    assert pool._known_batch(rec, at(0.6)) is True and pool._known_batch(rec, at(0.7)) is False


def test_a_skipped_batch_still_enters_the_closed_form_head_of_a_unipool():
    st = _stream()
    pool = UniPool(_cfg(st, UniCfg, skip_known=1.0, skip_min_assigned=0, lda_blend=0.3))
    _feed(pool, st, 0, 250)
    head = pool.experts[0].meta["lda"]
    total, before, s0, n_asg = head.total, _weights(pool), dict(pool.skip_stats), pool.experts[0].n_assigned
    _feed(pool, st, 250, 300)
    assert pool.skip_stats["skipped"] == s0["skipped"] + 1
    assert pool.experts[0].n_assigned == n_asg + 50                                                 # the shipped bookkeeping still runs
    assert head.total == total + 50                                                                 # one cheap closed-form pass
    after = _weights(pool)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_a_unipool_without_a_head_skips_without_creating_one():
    st = _stream()
    pool = UniPool(_cfg(st, UniCfg, skip_known=1.0, skip_min_assigned=0))
    _feed(pool, st)
    assert pool.skip_stats["skipped"] >= 1 and "lda" not in pool.experts[0].meta

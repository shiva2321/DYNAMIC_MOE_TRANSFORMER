"""M2-A: a newborn expert's competence monitor must not be armed from its untrained loss.

On a function-regime stream the newborn's held-out loss RISES after its first batch of training (it over-fits a tiny batch). The
one-sided CUSUM reads that as concept drift, raises an alarm on the newborn itself, and the alarm path replaces its weights with a
60-step fine-tune on 32 samples. ``growmoe/competence.py`` asks for the opposite ("arm the monitor only after the expert's loss has
plateaued"). ``PhyloCfg.arm_monitor_after_infancy`` keeps an infant's monitor silent until its infancy is over.

Evidence for the fault: research/ROUTING_RESEARCH_2026-10.md section 4 (dev seeds, exploratory).
"""

from collections import Counter

import numpy as np
import pytest
import torch

from growmoe.config import PRE_STAGE4, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes
from growmoe.vmf import sample_vmf

A_KW = dict(q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf")
_STREAMS = {}


def _stream(n_regimes=3, n_train=320, seed=100):
    key = (n_regimes, n_train, seed)
    if key not in _STREAMS:
        _STREAMS[key] = function_regimes(seed, n_regimes=n_regimes, n_train=n_train, n_test=100)
    return _STREAMS[key]


def _pool(st, steps, seed=100, **kw):
    return PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps, **A_KW, **kw))


def _run(st, pool, seed=100):
    for x, y, _ in st.algo_batches(64, seed):
        pool.observe(x, y)
    return pool


def _events(pool):
    return Counter(e.kind for e in pool.events)


# --------------------------------------------------------------------------------------------------------------- the flag
def test_flag_is_on_by_default_since_the_m2a_repair_was_confirmed_on_fresh_seeds():
    assert PhyloCfg().arm_monitor_after_infancy is True


def test_the_default_pool_behaves_exactly_like_the_explicit_flag_on():
    st = _stream()
    default = _run(st, PhyloMoE(PhyloCfg(seed=100, d=st.d, n_classes=st.n_classes, inner_steps=50, **A_KW)))     # 50 steps: the budget at which the fault shows
    explicit = _run(st, _pool(st, 50, arm_monitor_after_infancy=True))
    assert [(e.kind, e.step) for e in default.events] == [(e.kind, e.step) for e in explicit.events]
    assert _events(default)["alarm"] == 0


def test_the_phase_one_entry_point_pins_the_old_behaviour_so_recorded_results_reproduce():
    from growmoe import bench
    assert bench._phase1_cfg(seed=0).arm_monitor_after_infancy is False
    assert bench._phase1_cfg(seed=0, arm_monitor_after_infancy=True).arm_monitor_after_infancy is True       # callers can still override


# ------------------------------------------------------------------------------------- the fault, pinned as a control
def test_control_with_the_flag_off_every_newborn_alarms_on_itself_at_its_second_batch():
    """The Phase-1 / M0 behaviour. If this stops being true the scenario no longer reproduces the fault and the tests below prove nothing."""
    st = _stream()
    pool = PhyloMoE(PhyloCfg(seed=100, d=st.d, n_classes=st.n_classes, inner_steps=50, arm_monitor_after_infancy=False, **A_KW, **PRE_STAGE4))     # the fault was measured with the probe rule
    _run(st, pool)
    ev = _events(pool)
    assert ev["spawn"] == 3 and ev["alarm"] >= 2 and ev["extend"] >= 2
    newborns = {e.fields["expert"] for e in pool.events.of_kind("spawn") if e.fields.get("parent") is not None}
    assert {e.fields["expert"] for e in pool.events.of_kind("alarm")} <= newborns                      # the alarms are on newborns, not on old experts


def test_control_with_the_flag_off_the_newborns_monitor_is_armed_straight_after_its_birth_batch():
    st = _stream()
    pool = PhyloMoE(PhyloCfg(seed=100, d=st.d, n_classes=st.n_classes, inner_steps=50, arm_monitor_after_infancy=False, **A_KW))
    armed_after_birth = None
    cur = -1
    for k, seg, idx in st._walk(64, 100):
        pool.observe(seg.x[idx], y=seg.y[idx])
        if k == 1 and cur != 1:
            armed_after_birth = pool.experts[1].meta["monitor"].armed
        cur = k
    assert armed_after_birth is True


# ------------------------------------------------------------------------------------------------------------- the fix
def test_with_the_flag_newborns_raise_no_alarm_and_each_regime_gets_one_expert():
    st = _stream()
    pool = _run(st, _pool(st, 50, arm_monitor_after_infancy=True))
    ev = _events(pool)
    assert ev["spawn"] == 3 and ev["alarm"] == 0 and ev["extend"] == 0
    assert len(pool.experts) == 3


def test_with_the_flag_the_monitor_arms_only_after_infancy_on_settled_losses():
    st = _stream()
    pool = _pool(st, 50, infant_calls=2, arm_monitor_after_infancy=True)
    armed, cur = {}, -1
    for k, seg, idx in st._walk(64, 100):
        b = armed.setdefault(k, [])
        pool.observe(seg.x[idx], y=seg.y[idx])
        if k == 1:
            b.append(pool.experts[1].meta["monitor"].armed)
    assert armed[1][:2] == [False, False]                      # birth batch and the next one: still infant, nothing collected
    assert armed[1][3] is True                                 # two calls of infancy are over; a full batch of settled losses armed it
    assert pool.experts[1].state == "active"


def test_the_flag_changes_nothing_while_no_expert_is_an_infant():
    """A single bootstrap expert is never an infant, so the two configurations must be indistinguishable."""
    outs = []
    for flag in (False, True):
        cfg = PhyloCfg(d=64, n_classes=8, m_min=32, monitor_warmup=50, competence=True, seed=118, arm_monitor_after_infancy=flag)
        pool = PhyloMoE(cfg)
        rng = np.random.default_rng(118)
        mu0 = rng.standard_normal(64)
        mu0 /= np.linalg.norm(mu0)
        for _ in range(10):
            pool.observe(sample_vmf(mu0, kappa=60.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
        probe = sample_vmf(mu0, kappa=60.0, n=20, rng=rng)
        outs.append(([e.kind for e in pool.events], pool.predict(probe).probs.clone()))
    assert outs[0][0] == outs[1][0]
    assert torch.equal(outs[0][1], outs[1][1])


def test_a_real_label_remap_on_a_settled_newborn_still_raises_the_alarm_with_the_flag():
    """The alarm exists for concept drift on covered inputs. Silencing infants must not remove it once they are settled."""
    st = _stream(n_regimes=2, n_train=640)
    pool = _pool(st, 5, arm_monitor_after_infancy=True)
    last_seg = None
    for k, seg, idx in st._walk(64, 100):
        pool.observe(seg.x[idx], y=seg.y[idx])
        last_seg = seg
    assert len(pool.events.of_kind("alarm")) == 0                              # a stationary stream: no alarm
    owner = int(torch.mode(pool.predict(last_seg.x[:128]).expert).values)      # the expert the last regime's inputs are routed to
    assert pool.experts[owner].state == "active" and pool.experts[owner].meta["monitor"].armed
    x, y = last_seg.x[:192], (last_seg.y[:192] + 1) % st.n_classes             # same inputs, every label shifted by one
    for lo in range(0, 192, 64):
        pool.observe(x[lo:lo + 64], y=y[lo:lo + 64])
    alarms = pool.events.of_kind("alarm")
    assert len(alarms) >= 1 and alarms[0].fields["expert"] == owner

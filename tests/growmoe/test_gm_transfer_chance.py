"""The worse-than-chance rule of the transfer spawn test (E-061 found it, E-062 put it into the pool behind a flag, OFF by default).

The shipped verdict asks whether a new cluster's loss under the best parent is worse than the parent's OWN reference losses. A weak parent (a closed-form notebook, a network of 8 units) has high reference
losses, so it vouches for almost any cluster and new regimes are absorbed. ``transfer_chance`` = c turns a no-conflict verdict into a conflict when the cluster's loss is at least c x ln(classes the parent has
seen), i.e. no better than guessing. With the flag None (the default) nothing changes."""

import dataclasses

import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.pools import phylo as phylo_module
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes
from growmoe.transfer import TransferVerdict


def _cfg(st, **kw):
    return PhyloCfg(d=st.d, n_classes=st.n_classes, seed=3, inner_steps=3, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", **kw)


def _trained_pool(**kw):
    st = function_regimes(5, 3, d=32, n_classes=4, n_train=300, n_test=50)
    pool = PhyloMoE(_cfg(st, **kw))
    seg = st.segments[0]
    for lo in range(0, 300, 50):
        pool.observe(seg.x[lo:lo + 50], y=seg.y[lo:lo + 50])
    return st, pool


def test_the_flag_is_off_by_default_and_validated():
    assert PhyloCfg().transfer_chance is None
    assert PhyloCfg(transfer_chance=1.0).transfer_chance == 1.0
    assert dataclasses.replace(PhyloCfg(route_rule="accept_then_logf"), transfer_chance=0.5).transfer_chance == 0.5
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            PhyloCfg(transfer_chance=bad)


def test_a_verdict_event_carries_the_veto_only_when_the_rule_fired():
    base = dict(parent=0, conflict=False, p=0.5, diff=0.0, loss_cluster=1.0, loss_ref=1.0, margin=0.1, n_cluster=10, n_ref=100)
    assert "chance_veto" not in TransferVerdict(**base).fields()
    assert TransferVerdict(**base, chance_veto=True).fields()["chance_veto"] is True
    assert set(TransferVerdict(**base).fields()) == {"p_conflict", "loss_cluster", "loss_ref", "margin", "n_ref"}          # the shipped event fields are unchanged


def test_a_parent_that_does_worse_than_chance_on_a_cluster_cannot_vouch_for_it(monkeypatch):
    st, off = _trained_pool()
    _, on = _trained_pool(transfer_chance=1.0)
    monkeypatch.setattr(phylo_module, "cluster_conflict", lambda *a, **k: (False, 0.5, 0.0))       # the shipped test finds "no conflict": only the rule can change that
    seg = st.segments[0]
    x_same, y_same = seg.x[250:300], seg.y[250:300]                                                  # the regime the expert knows
    seg2 = st.segments[1]
    x_new, y_new = seg2.x[:50], seg2.y[:50]                                                          # a regime it has never seen
    xn = torch.nn.functional.normalize(x_new.float(), dim=1)
    chance = float(np.log(max(int(on.experts[0].module.seen.sum()), 2)))
    loss_new = float(torch.nn.functional.cross_entropy(on.experts[0].module(xn), y_new.long()))
    assert loss_new >= chance                                                                        # the premise: the expert does worse than chance on the new regime
    v_off = off._transfer_verdict(torch.nn.functional.normalize(x_new.float(), dim=1), y_new, None)
    v_on = on._transfer_verdict(xn, y_new, None)
    assert v_off.conflict is False and v_off.chance_veto is False                                    # flag off: the shipped verdict stands
    assert v_on.conflict is True and v_on.chance_veto is True and "chance_veto" in v_on.fields()      # flag on: a conflict, the cluster founds a child
    xs = torch.nn.functional.normalize(x_same.float(), dim=1)
    loss_same = float(torch.nn.functional.cross_entropy(on.experts[0].module(xs), y_same.long()))
    assert loss_same < chance
    v_same = on._transfer_verdict(xs, y_same, None)
    assert v_same.conflict is False and v_same.chance_veto is False                                  # a parent that predicts the cluster still vouches for it


def test_a_real_conflict_is_left_alone(monkeypatch):
    st, on = _trained_pool(transfer_chance=1.0)
    monkeypatch.setattr(phylo_module, "cluster_conflict", lambda *a, **k: (True, 0.001, 2.0))
    seg2 = st.segments[1]
    v = on._transfer_verdict(torch.nn.functional.normalize(seg2.x[:50].float(), dim=1), seg2.y[:50], None)
    assert v.conflict is True and v.chance_veto is False                                             # the rule only adds conflicts, it does not claim ones the shipped test found


def test_a_higher_multiple_lets_a_worse_parent_vouch(monkeypatch):
    st, strict = _trained_pool(transfer_chance=1.0)
    _, loose = _trained_pool(transfer_chance=1000.0)
    monkeypatch.setattr(phylo_module, "cluster_conflict", lambda *a, **k: (False, 0.5, 0.0))
    seg2 = st.segments[1]
    xn = torch.nn.functional.normalize(seg2.x[:50].float(), dim=1)
    assert strict._transfer_verdict(xn, seg2.y[:50], None).conflict is True
    assert loose._transfer_verdict(xn, seg2.y[:50], None).conflict is False


def test_a_parent_that_has_seen_one_class_is_judged_against_two_so_the_chance_loss_is_never_zero(monkeypatch):
    st, on = _trained_pool(transfer_chance=1.0)
    monkeypatch.setattr(phylo_module, "cluster_conflict", lambda *a, **k: (False, 0.5, 0.0))
    seen = on.experts[0].module.seen
    seen.zero_()
    seen[0] = True                                                                                   # one class: the loss of a label-0 cluster is exactly 0, the floor makes the chance loss ln 2
    x = torch.nn.functional.normalize(st.segments[0].x[:50].float(), dim=1)
    v = on._transfer_verdict(x, torch.zeros(50, dtype=torch.long), None)
    assert v.loss_cluster == pytest.approx(0.0, abs=1e-6) and v.conflict is False and v.chance_veto is False


def test_weak_only_lets_a_strong_parent_keep_the_shipped_verdict_and_vetoes_a_weak_one(monkeypatch):
    st, strong = _trained_pool(transfer_chance=1.0, transfer_chance_weak_only=True)
    _, plain = _trained_pool(transfer_chance=1.0)
    monkeypatch.setattr(phylo_module, "cluster_conflict", lambda *a, **k: (False, 0.5, 0.0))
    seg2 = st.segments[1]
    xn = torch.nn.functional.normalize(seg2.x[:50].float(), dim=1)
    chance = float(np.log(max(int(strong.experts[0].module.seen.sum()), 2)))
    from collections import deque
    strong.experts[0].meta["ref"] = deque([0.2 * chance] * 200, maxlen=512)                          # the parent predicts its own regime well
    plain.experts[0].meta["ref"] = deque([0.2 * chance] * 200, maxlen=512)
    assert plain._transfer_verdict(xn, seg2.y[:50], None).conflict is True                          # the unconditional rule vetoes it all the same
    v = strong._transfer_verdict(xn, seg2.y[:50], None)
    assert v.conflict is False and v.chance_veto is False                                           # weak-only: a strong parent keeps the shipped verdict
    strong.experts[0].meta["ref"] = deque([1.5 * chance] * 200, maxlen=512)                          # a parent that does worse than chance on its own regime
    v = strong._transfer_verdict(xn, seg2.y[:50], None)
    assert v.conflict is True and v.chance_veto is True
    assert PhyloCfg().transfer_chance_weak_only is False

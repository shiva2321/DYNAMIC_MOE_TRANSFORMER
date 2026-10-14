"""RV3: the decision logic of the batch-level gate (spec section 4.3), without a pool."""

import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.setgate import SetDecision, SetGate

D = 16


def _cloud(n, axis, seed, spread=0.35):
    g = np.random.default_rng(seed)
    c = np.zeros(D)
    c[axis] = 1.0
    z = c[None, :] + spread * g.standard_normal((n, D)) / np.sqrt(D)
    return torch.nn.functional.normalize(torch.from_numpy(z.astype(np.float32)), dim=1)


def _cfg(**kw):
    kw.setdefault("seed", 3)
    return PhyloCfg(set_gate=True, **kw)


def _refs(table):
    return lambda e: [(r, r.mean(0)) for r in table.get(e, [])]


def _no_loss(e, x, y):
    raise AssertionError("loss must not be asked for here")


def test_a_batch_from_the_same_source_as_one_expert_is_known_and_owned_by_it():
    gate = SetGate(_cfg())
    table = {0: [_cloud(120, 0, 1)], 1: [_cloud(120, 5, 2)], 2: [_cloud(120, 9, 3)]}
    d = gate.decide(_cloud(32, 5, 10), None, [0, 1, 2], _refs(table), _no_loss, step=7)
    assert isinstance(d, SetDecision) and d.verdict == "known" and d.owner == 1 and d.admitted == [1]
    assert d.tried[0] == 1 and gate.stats["known"] == 1 and gate.stats["tests"] == len(d.tried)


def test_a_batch_that_no_expert_admits_is_novel_as_a_set():
    gate = SetGate(_cfg())
    table = {0: [_cloud(120, 0, 1)], 1: [_cloud(120, 5, 2)]}
    d = gate.decide(_cloud(32, 12, 10), None, [0, 1], _refs(table), _no_loss, step=1)
    assert d.verdict == "novel" and d.owner == -1 and d.admitted == [] and sorted(d.tried) == [0, 1]
    assert gate.stats["novel"] == 1 and gate.stats["perms"] >= 2 * 999                  # a rejecting test is never cut short


def test_a_known_batch_costs_about_one_chunk_per_test():
    gate = SetGate(_cfg())
    table = {0: [_cloud(120, 0, 1)]}
    for s in range(10):
        gate.decide(_cloud(32, 0, 20 + s), None, [0], _refs(table), _no_loss, step=s)
    assert gate.stats["tests"] == 10 and gate.stats["early_stops"] >= 9 and gate.stats["perms"] <= 10 * 64


def test_an_expert_is_admitted_at_its_first_reference_that_does_not_reject():
    gate = SetGate(_cfg())
    table = {0: [_cloud(120, 0, 1), _cloud(120, 5, 2)]}                                  # two sources in one expert's memory
    d = gate.decide(_cloud(32, 5, 10), None, [0], _refs(table), _no_loss, step=1)
    assert d.verdict == "known" and d.owner == 0 and gate.stats["tests"] == 1             # the nearest reference (source 5) is tried first and admits


def test_only_the_configured_number_of_experts_is_tried():
    gate = SetGate(_cfg(set_gate_candidates=2))
    table = {e: [_cloud(120, e, 30 + e)] for e in range(6)}
    d = gate.decide(_cloud(32, 15, 10), None, list(range(6)), _refs(table), _no_loss, step=1)
    assert d.verdict == "novel" and len(d.tried) == 2


def test_loss_ownership_picks_the_admitted_expert_with_the_lowest_label_loss_and_needs_enough_labels():
    table = {0: [_cloud(120, 0, 1, spread=0.9)], 1: [_cloud(120, 0, 2, spread=0.9)]}     # two experts that both admit a source-0 batch
    xb = _cloud(32, 0, 10, spread=0.9)
    gate = SetGate(_cfg())
    d0 = gate.decide(xb, None, [0, 1], _refs(table), _no_loss, step=1)
    assert d0.verdict == "known" and sorted(d0.admitted) == [0, 1] and d0.owner == d0.admitted[0]      # no labels: the nearest admitted
    losses = {0: 2.0, 1: 0.5}
    yb = torch.arange(32) % 4
    d1 = SetGate(_cfg()).decide(xb, yb, [0, 1], _refs(table), lambda e, x, y: losses[e], step=1)
    assert d1.owner == 1
    few = torch.full((32,), -1)
    few[:5] = 1                                                                          # 5 labelled < 8
    d2 = SetGate(_cfg()).decide(xb, few, [0, 1], _refs(table), _no_loss, step=1)
    assert d2.owner == d2.admitted[0]
    d3 = SetGate(_cfg(set_gate_ownership="nearest")).decide(xb, yb, [0, 1], _refs(table), _no_loss, step=1)
    assert d3.owner == d3.admitted[0] and len(d3.admitted) == 1                           # nearest ownership stops at the first admitted expert


def test_an_expert_without_references_is_never_admitted_and_an_empty_pool_is_novel():
    gate = SetGate(_cfg())
    d = gate.decide(_cloud(32, 0, 10), None, [0, 1], _refs({0: []}), _no_loss, step=1)
    assert d.verdict == "novel" and d.tried == []
    assert SetGate(_cfg()).decide(_cloud(32, 0, 10), None, [], _refs({}), _no_loss, step=1).verdict == "novel"


def test_decisions_are_deterministic_and_independent_of_the_order_experts_are_listed():
    table = {0: [_cloud(60, 0, 1, 0.9)], 1: [_cloud(60, 0, 2, 0.9)], 2: [_cloud(60, 7, 3)]}
    xb = _cloud(24, 0, 10, 0.9)
    a = SetGate(_cfg()).decide(xb, None, [0, 1, 2], _refs(table), _no_loss, step=4)
    b = SetGate(_cfg()).decide(xb, None, [0, 1, 2], _refs(table), _no_loss, step=4)
    c = SetGate(_cfg()).decide(xb, None, [2, 1, 0], _refs(table), _no_loss, step=4)
    assert a == b == c
    assert a.verdict == "known"


def test_a_huge_batch_and_a_huge_reference_are_cut_to_the_row_cap(monkeypatch):
    import growmoe.setgate as sg
    seen = []
    real = sg.perm_reject
    monkeypatch.setattr(sg, "perm_reject", lambda ref, batch, *a, **k: (seen.append((ref.shape[0], batch.shape[0])), real(ref, batch, *a, **k))[1])
    gate = SetGate(_cfg(set_gate_max_rows=64, set_gate_stat="mmd"))
    d = gate.decide(_cloud(2000, 0, 10), None, [0], _refs({0: [_cloud(3000, 0, 1)]}), _no_loss, step=1)     # an uncapped MMD kernel would be 5000 x 5000 floats
    assert d.verdict == "known" and seen == [(64, 64)]


def test_the_breaker_opens_after_too_many_novel_verdicts_pauses_and_forgets():
    gate = SetGate(_cfg(set_gate_breaker_len=4, set_gate_breaker_rate=0.5, set_gate_breaker_pause=10))
    opened = [gate.note("novel", 1.0, step=s) for s in range(3)]
    assert opened == [False, False, False] and not gate.paused(3)
    assert gate.note("novel", 1.0, step=3) is True                                       # 4 of 4 > 0.5
    assert gate.paused(4) and gate.paused(12) and not gate.paused(13) and gate.stats["breaker_opens"] == 1
    assert len(gate.hist) == 0
    for s in range(20, 24):
        assert gate.note("known", 1.0, step=s) is False


def test_the_breaker_needs_strictly_more_than_the_rate():
    gate = SetGate(_cfg(set_gate_breaker_len=4, set_gate_breaker_rate=0.5, set_gate_breaker_pause=10))
    assert [gate.note(v, 1.0, step=s) for s, v in enumerate(["novel", "known", "novel", "known"])] == [False] * 4      # exactly 0.5 is not above 0.5
    assert gate.note("novel", 1.0, step=4) is False                                       # the window holds known, novel, known, novel: still 0.5
    assert gate.note("novel", 1.0, step=5) is True                                        # novel, known, novel, novel = 0.75 > 0.5
    assert gate.stats["breaker_opens"] == 1


def test_batches_the_per_sample_gate_already_doubted_do_not_feed_the_breaker():
    gate = SetGate(_cfg(set_gate_breaker_len=4, set_gate_breaker_rate=0.5, set_gate_breaker_pause=10))
    for s in range(8):
        assert gate.note("novel", 0.89, step=s) is False
    assert len(gate.hist) == 0 and not gate.paused(8)
    assert gate.note("novel", 0.9, step=9) is False and len(gate.hist) == 1

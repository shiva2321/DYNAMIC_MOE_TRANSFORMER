"""RV3: the batch-level gate (3a) and group answering (3b) inside PhyloMoE (spec sections 4.3, 4.4, 5, 9, 11)."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from growmoe.accounting import pool_report
from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.setgate import SetDecision
from growmoe.signature import VmfSignature
from growmoe.streams_m1 import function_regimes

ROOT = Path(__file__).resolve().parents[2]
D, C = 32, 4


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _centres(cos, seed=0):
    g = np.random.default_rng(seed)
    a = _unit(g.standard_normal((1, D)))
    b = g.standard_normal((1, D))
    b = _unit(b - (b @ a.T) * a)
    return a[0], (cos * a + np.sqrt(1 - cos ** 2) * b)[0]


def _draw(c, n, seed, scale=8.0):
    g = np.random.default_rng(seed)
    return torch.from_numpy(_unit(scale * c[None, :] + g.standard_normal((n, D))).astype(np.float32))


def _labels(n, seed):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, C, n))


def _pool(**kw):
    kw.setdefault("set_gate", False)                                                        # explicit: these tests must not change meaning if a default is flipped later
    kw.setdefault("group_vote", False)
    kw.setdefault("set_gate_confirm", 1)                                                  # these tests exercise the act-at-once mechanics; the persistence rule (A-15) has test_gm_rv3_confirm.py
    return PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=C, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=1, route_rule="accept_then_logf", **kw))


def _teach(pool, c, n_batches, seed0):
    recs = []
    for k in range(n_batches):
        recs.append(pool.observe(_draw(c, 64, seed0 + k), _labels(64, seed0 + 50 + k)))
    return recs


@pytest.fixture(scope="module")
def two_source_pool():
    """Expert 0 learned source A, expert 1 learned source B (far apart), set gate ON."""
    cA, cB = _centres(0.3)
    pool = _pool(set_gate=True)
    _teach(pool, cA, 6, 10)
    _teach(pool, cB, 6, 100)
    assert sorted(pool.experts) == [0, 1]
    return pool, cA, cB


# ------------------------------------------------------------------------------------------------ flags off
def test_with_both_flags_off_the_pool_is_bit_identical_to_the_recorded_golden_run():
    spec = importlib.util.spec_from_file_location("golden", ROOT / "research" / "growmoe_results" / "rv2c" / "golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert golden.differences(golden.QUICK) == []


def test_with_the_flags_off_nothing_of_stage_3_is_created_or_reported():
    st = function_regimes(100, overlap=0.6, n_regimes=4, n_train=128, n_test=64)
    pool = PhyloMoE(PhyloCfg(seed=7, d=st.d, n_classes=st.n_classes, inner_steps=1, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", set_gate=False, group_vote=False))
    for x, y, ctx in st.algo_batches(64, 7):
        rec = pool.observe(x, y, ctx)
        assert rec.meta == {}
    assert all("windows" not in r.meta for r in pool.experts.values())
    assert all(v == 0 for v in pool.set_stats.values()) and "set_stats" not in pool.summary()
    assert not any(e.kind in ("set_novel", "set_override", "breaker_open") for e in pool.events)


def test_predict_grouped_is_the_per_input_answer_unless_group_vote_is_on(two_source_pool):
    pool, cA, cB = two_source_pool
    off = copy.deepcopy(pool)
    off.cfg.group_vote = False
    x = torch.cat([_draw(cA, 10, 500), _draw(cB, 6, 501)])
    g = torch.zeros(16, dtype=torch.long)
    plain, grouped = off.predict(x), off.predict_grouped(x, g)
    assert torch.equal(plain.probs, grouped.probs) and torch.equal(plain.expert, grouped.expert) and grouped.meta == {}
    on = copy.deepcopy(pool)
    on.cfg.group_vote = True
    assert torch.equal(on.predict(x).expert, plain.expert)                               # predict itself never groups


# ------------------------------------------------------------------------------------------------ 3a: the set test
def test_a_foreign_batch_the_per_sample_gate_accepts_is_called_new_as_a_set_and_gets_its_own_expert():
    cA, cC = _centres(0.85)
    twin, pool = _pool(set_gate=False), _pool(set_gate=True)
    for p in (twin, pool):
        _teach(p, cA, 6, 10)
    foreign = _draw(cC, 64, 99)
    assert float((pool.experts[0].meta["sig"].pvalue(foreign) >= 0.02).float().mean()) > 0.3      # most of it passes the per-sample gate
    r_twin, r_set = twin.observe(foreign, _labels(64, 98)), pool.observe(foreign, _labels(64, 98))
    assert len(twin.experts) == 1 and r_twin.assigned.tolist() == [0] * 64                # today: the vote files the whole batch under expert 0
    assert r_set.meta["set_verdict"] == "novel" and len(pool.experts) == 2
    assert torch.equal(r_set.novel, r_twin.novel)                                         # StepRecord.novel is still the per-sample gate's
    assert set(r_set.assigned.tolist()) == {1} and pool.set_stats["novel"] == 1
    assert any(e.kind == "set_novel" for e in pool.events)
    again = pool.observe(_draw(cA, 64, 77), _labels(64, 76))                              # the original source is still recognised
    assert again.meta["set_verdict"] == "known" and set(again.assigned.tolist()) == {0}


def test_known_batches_train_the_owner_and_windows_appear_lazily(two_source_pool):
    pool, cA, cB = copy.deepcopy(two_source_pool[0]), two_source_pool[1], two_source_pool[2]
    assert set(pool.experts[0].meta["windows"].refs()[0].shape) == {D, 5 * 64} or pool.experts[0].meta["windows"].n_samples() > 0
    rec = pool.observe(_draw(cB, 64, 700), _labels(64, 701))
    assert rec.meta["set_verdict"] == "known" and set(rec.assigned.tolist()) == {1}
    assert pool.experts[1].meta["windows"].n_samples() >= 64


def test_fewer_than_batch_vote_min_accepted_samples_means_no_set_decision(two_source_pool):
    pool, cA, cB = copy.deepcopy(two_source_pool[0]), two_source_pool[1], two_source_pool[2]
    rec = pool.observe(_draw(cA, 6, 800), _labels(6, 801))                                # a batch of six
    assert rec.meta["set_verdict"] is None


def test_the_glue_applies_a_known_verdict_with_a_different_owner_and_reports_the_override(two_source_pool, monkeypatch):
    pool, cA, _ = copy.deepcopy(two_source_pool[0]), two_source_pool[1], two_source_pool[2]
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: SetDecision("known", 1, [1], [1]))
    x = _draw(cA, 64, 900)
    rec = pool.observe(x, _labels(64, 901))
    assert set(rec.assigned.tolist()) == {1} and pool.set_stats["override"] == 1
    ev = [e for e in pool.events if e.kind == "set_override"]
    assert len(ev) == 1 and ev[0].fields["vote"] == 0 and ev[0].fields["owner"] == 1


def test_the_glue_sends_a_novel_verdict_to_the_probation_buffer_without_touching_flagged_samples(two_source_pool, monkeypatch):
    pool, cA, cB = copy.deepcopy(two_source_pool[0]), two_source_pool[1], two_source_pool[2]
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: SetDecision("novel", -1, [], [0, 1]))
    x = torch.cat([_draw(cA, 40, 910), _draw(_centres(0.0, seed=5)[1], 24, 911)])        # 24 inputs from a source far from both experts: flagged by the per-sample gate
    pool.cfg.m_min = 10 ** 6                                                              # keep the probation buffer from spawning in this test
    before = len(pool.probation)
    rec = pool.observe(x, _labels(64, 912))
    accepted_in = int((~rec.novel).sum())
    assert rec.meta["set_verdict"] == "novel" and accepted_in >= 8
    assert (rec.assigned == -1).all()                                                     # flagged ones were never assigned, accepted ones were unassigned
    assert len(pool.probation) - before == 64                                             # flagged + set-novel samples all wait for the evidence test


def test_a_paused_gate_leaves_the_vote_alone_and_says_so(two_source_pool, monkeypatch):
    pool, cA, _ = copy.deepcopy(two_source_pool[0]), two_source_pool[1], two_source_pool[2]
    pool._set.pause_until = 10 ** 9
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: pytest.fail("a paused gate must not test"))
    rec = pool.observe(_draw(cA, 64, 920), _labels(64, 921))
    assert rec.meta["set_verdict"] == "paused" and set(rec.assigned.tolist()) == {0} and pool.set_stats["paused"] == 1


def test_the_breaker_opens_in_a_live_pool_pauses_and_resumes():
    cA, cC = _centres(0.85)
    pool = _pool(set_gate=True, set_gate_breaker_len=3, set_gate_breaker_rate=0.5, set_gate_breaker_pause=4, max_experts=1, cap_policy="drop")    # max_experts=1 with cap_policy="drop" (the pre-4.1 behaviour): novel batches cannot spawn and are not shared either
    _teach(pool, cA, 6, 10)                                                               # the last three verdicts in the breaker's memory are "known"
    verdicts = [pool.observe(_draw(cC, 64, 300 + k), _labels(64, 400 + k)).meta["set_verdict"] for k in range(6)]
    # known, known, novel -> 1/3; known, novel, novel -> 2/3 > 0.5 opens it at the second foreign batch; paused while step < opening step + 4 (three batches).
    # While paused the vote alone files the foreign batches under expert 0, which therefore learns the new source (a new window), so the gate admits it afterwards.
    assert verdicts == ["novel", "novel", "paused", "paused", "paused", "known"]
    assert pool.set_stats["breaker_opens"] == 1 and pool.set_stats["paused"] == 3
    assert [e.kind for e in pool.events].count("breaker_open") == 1
    assert len(pool.experts[0].meta["windows"]) >= 2


def test_two_identical_pools_give_identical_verdicts_and_assignments():
    cA, cC = _centres(0.85)

    def run():
        pool = _pool(set_gate=True)
        _teach(pool, cA, 6, 10)
        out = []
        for k in range(4):
            r = pool.observe(_draw(cC if k % 2 == 0 else cA, 64, 600 + k), _labels(64, 650 + k))
            out.append((r.meta["set_verdict"], r.assigned.tolist()))
        return out, dict(pool.set_stats)
    assert run() == run()


@pytest.mark.parametrize("vote,shortlist", [(True, 0), (False, 0), (True, 2), (False, 2)])
def test_every_combination_of_vote_cascade_and_set_gate_runs_and_keeps_its_invariants(vote, shortlist):
    st = function_regimes(100, overlap=0.6, n_regimes=4, n_train=128, n_test=64)
    pool = PhyloMoE(PhyloCfg(seed=7, d=st.d, n_classes=st.n_classes, inner_steps=1, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf",
                             batch_vote=vote, gate_shortlist=shortlist, gate_shortlist_min_experts=2, set_gate=True))
    for x, y, ctx in st.algo_batches(64, 7):
        rec = pool.observe(x, y, ctx)
        assert rec.meta["set_verdict"] in (None, "known", "novel", "paused", "held")
        assert int(rec.assigned.max()) < pool._next_id and int(rec.assigned.min()) >= -1
    assert pool.set_stats["tests"] > 0 and len(pool.experts) >= 2


# ------------------------------------------------------------------------------------------------ windows: merge, accounting
def test_a_merged_expert_keeps_the_windows_of_both():
    cA, _ = _centres(0.3)
    pool = _pool(set_gate=True)
    xs = [_draw(cA, 64, 1000 + k) for k in range(8)]
    for e in range(2):                                                                    # two experts that learned the same source separately: the merge rule accepts them
        s = VmfSignature(D, 256, seed=e)
        s.update(torch.cat(xs[:4]))
        s.fit()
        rec = pool.add_expert(MLPExpert(D, 16, 1, C, seed=e), parent=None, state="active", meta={"sig": s})
        rec.replay.add(torch.cat(xs[:4]), _labels(256, 5))
        for k in range(4):
            pool._windows_of(rec).update(xs[4 + k], torch.Generator().manual_seed(k), step=10 * e + k)
    assert [len(pool.experts[e].meta["windows"]) for e in (0, 1)] == [1, 1]
    pool.step = pool.cfg.merge_every - 1
    pool.cfg.delta_merge = 1.0                                                            # accept the merge: this test is about the bookkeeping, not the accuracy rule
    pool._attempt_merges()
    assert sorted(pool.experts) == [0]
    merged = pool.experts[0].meta["windows"]
    assert len(merged) == 2 and merged.n_samples() == 2 * 4 * 64                          # the union of the two stores


def test_source_windows_count_as_stored_vectors_in_the_resource_report(two_source_pool):
    pool = copy.deepcopy(two_source_pool[0])
    with_windows = pool_report(pool, D).stored_vectors
    held = sum(r.meta["windows"].n_samples() for r in pool.experts.values())
    for r in pool.experts.values():
        del r.meta["windows"]
    assert held > 0 and with_windows - pool_report(pool, D).stored_vectors == held


def test_summary_carries_the_counters_only_when_the_set_gate_is_on(two_source_pool):
    s = two_source_pool[0].summary()
    assert s["set_stats"]["tests"] > 0 and set(s["set_stats"]) == set(two_source_pool[0].set_stats)


# ------------------------------------------------------------------------------------------------ 3b: group answering
def test_a_group_is_answered_by_the_expert_most_of_its_inputs_chose_and_the_veto_protects_foreign_inputs(two_source_pool):
    base, cA, cB = two_source_pool
    x = torch.cat([_draw(cA, 10, 1100), _draw(cB, 2, 1101)])
    g = torch.zeros(12, dtype=torch.long)
    own = base.predict(x)
    assert own.expert.tolist() == [0] * 10 + [1] * 2
    vetoed = copy.deepcopy(base)
    vetoed.cfg.group_vote, vetoed.cfg.group_veto = True, True
    out = vetoed.predict_grouped(x, g)
    assert out.expert.tolist() == [0] * 10 + [1] * 2 and out.meta["group_vote"]["moved"] == 0 and out.meta["group_vote"]["vetoed"] >= 2    # both foreign inputs vetoed (an own input may fall at the conformal floor too)
    plain = copy.deepcopy(base)
    plain.cfg.group_vote, plain.cfg.group_veto = True, False
    forced = plain.predict_grouped(x, g)
    assert forced.expert.tolist() == [0] * 12 and forced.meta["group_vote"]["moved"] == 2
    assert torch.equal(forced.probs[10:], plain.experts[0].module.proba(x[10:]).detach())     # the moved inputs were re-answered by the group's expert
    assert torch.equal(forced.probs[:10], own.probs[:10]) and torch.equal(forced.unknown, own.unknown) and torch.equal(forced.p_nov, own.p_nov)


def test_small_groups_and_ungrouped_inputs_are_answered_individually(two_source_pool):
    base, cA, cB = two_source_pool
    pool = copy.deepcopy(base)
    pool.cfg.group_vote, pool.cfg.group_veto = True, False
    x = torch.cat([_draw(cA, 5, 1200), _draw(cB, 3, 1201)])
    out = pool.predict_grouped(x, torch.zeros(8, dtype=torch.int32))                       # 8 members: a group of exactly group_min
    assert out.meta["group_vote"]["groups"] == 1
    out = pool.predict_grouped(x, torch.tensor([0] * 4 + [1] * 4))                         # two groups of 4 < 8
    assert out.meta["group_vote"]["groups"] == 0 and torch.equal(out.expert, base.predict(x).expert)
    out = pool.predict_grouped(x, torch.full((8,), -1))                                    # nobody grouped
    assert out.meta["group_vote"]["groups"] == 0


def test_the_veto_reads_exact_p_values_not_the_cascades_restricted_matrix(two_source_pool, monkeypatch):
    pool = copy.deepcopy(two_source_pool[0])
    pool.cfg.group_vote, pool.cfg.group_veto = True, True
    cA, cB = two_source_pool[1], two_source_pool[2]
    x = torch.cat([_draw(cA, 10, 1300), _draw(cB, 2, 1301)])
    individual = pool.predict(x)
    monkeypatch.setattr(pool, "predict", lambda *a, **k: individual)
    monkeypatch.setattr(pool, "_gate_matrices", lambda *a, **k: pytest.fail("the veto must not use the (possibly restricted) gate matrices"))
    monkeypatch.setattr(pool, "_gate_cascade", lambda *a, **k: pytest.fail("the veto must not use the cascade"))
    out = pool.predict_grouped(x, torch.zeros(12, dtype=torch.long))
    assert out.meta["group_vote"]["vetoed"] >= 2 and out.expert.tolist() == individual.expert.tolist()


def test_group_answering_on_an_empty_pool_is_the_plain_answer():
    pool = _pool()
    pool.cfg.group_vote = True
    out = pool.predict_grouped(_draw(_centres(0.3)[0], 16, 1), torch.zeros(16, dtype=torch.long))
    assert out.expert.tolist() == [-1] * 16 and out.meta == {}


def test_a_wrong_sized_or_float_group_vector_is_refused(two_source_pool):
    pool = copy.deepcopy(two_source_pool[0])
    pool.cfg.group_vote = True
    x = _draw(two_source_pool[1], 12, 1400)
    with pytest.raises(ValueError):
        pool.predict_grouped(x, torch.zeros(11, dtype=torch.long))
    with pytest.raises(ValueError):
        pool.predict_grouped(x, torch.zeros(12))

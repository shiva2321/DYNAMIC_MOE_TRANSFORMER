"""RV3 amendments A-15 (persistence: a doubtful batch is held until the next decided batch confirms it) and A-16 (the group veto has its own level).

Pre-amendment behaviour is `set_gate_confirm=1` (act at once) and `group_veto_q` equal to `q`; the first test class below pins that this is bit for bit the first stage-3 build.
"""

import copy
import hashlib

import numpy as np
import pytest
import torch

from growmoe.accounting import pool_report
from growmoe.config import PhyloCfg
from growmoe.fairfight import rv2c_verdicts
from growmoe.pools.phylo import PhyloMoE
from growmoe.setgate import SetDecision, SetGate
from growmoe.streams_m1 import function_regimes

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
    kw.setdefault("set_gate", True)
    kw.setdefault("group_vote", False)
    kw.setdefault("set_gate_confirm", 2)
    return PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=C, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=1, route_rule="accept_then_logf", **kw))


def _teach(pool, c, n_batches, seed0):
    for k in range(n_batches):
        pool.observe(_draw(c, 64, seed0 + k), _labels(64, seed0 + 50 + k))


def _taught(confirm=2, **kw):
    """A pool whose single expert learned source A, and the centre of a source C that is close to A (cos 0.85): C's samples mostly pass the per-sample gate."""
    cA, cC = _centres(0.85)
    pool = _pool(set_gate_confirm=confirm, **kw)
    _teach(pool, cA, 6, 10)
    assert sorted(pool.experts) == [0]
    return pool, cA, cC


def _mixed(cC, seed):
    """64 inputs: 40 from the close source C (most pass the per-sample gate, so only the set test can see them) and 24 from a source far from everything (the per-sample gate
    flags them): a batch with BOTH kinds of sample, which is what a real regime change looks like."""
    far = _centres(0.0, seed=5)[1]
    return torch.cat([_draw(cC, 40, seed), _draw(far, 24, seed + 1)]), _labels(64, seed + 2)


def _held_rows(pool):
    return sum(len(h[0]) for h in pool._held)


def _spawn_sizes(pool):
    return [e.fields["n_cluster"] for e in pool.events if e.kind == "spawn"]


# ------------------------------------------------------------------------------------------------ configuration
def test_the_amended_defaults_are_two_confirming_batches_and_a_veto_level_of_a_fifth():
    cfg = PhyloCfg()
    assert cfg.set_gate_confirm == 2 and cfg.group_veto_q == 0.2


@pytest.mark.parametrize("kw", [dict(group_veto_q=0.0), dict(group_veto_q=-0.1), dict(group_veto_q=1.5), dict(set_gate_confirm=0), dict(set_gate_confirm=-1),
                                dict(set_gate=True, set_gate_confirm=2, no_probation=True)])
def test_settings_that_would_silently_break_the_mechanism_are_refused(kw):
    with pytest.raises(ValueError):
        PhyloCfg(**kw)


@pytest.mark.parametrize("kw", [dict(group_veto_q=1.0), dict(group_veto_q=0.001), dict(set_gate_confirm=5), dict(set_gate=True, set_gate_confirm=1, no_probation=True),
                                dict(set_gate=False, set_gate_confirm=3, no_probation=True)])
def test_settings_that_are_fine_are_accepted(kw):
    PhyloCfg(**kw)


# ------------------------------------------------------------------------------------------------ the counting rule
def _seq(c, raw):
    gate = SetGate(PhyloCfg(set_gate_confirm=c))
    return [gate.confirm(v) for v in raw]


def test_confirm_one_never_holds():
    assert _seq(1, ["novel", "novel", "known", "novel"]) == ["novel", "novel", "known", "novel"]


def test_confirm_two_holds_the_first_doubtful_batch_of_every_run_and_acts_on_the_rest_of_the_run():
    assert _seq(2, ["novel", "novel", "novel", "known", "novel", "novel"]) == ["held", "novel", "novel", "known", "held", "novel"]


def test_confirm_three_holds_the_first_two():
    assert _seq(3, ["novel", "novel", "novel", "novel", "known", "novel", "novel", "novel"]) == ["held", "held", "novel", "novel", "known", "held", "held", "novel"]


def test_a_known_batch_in_the_middle_of_a_run_restarts_the_count():
    assert _seq(3, ["novel", "novel", "known", "novel", "novel", "novel"]) == ["held", "held", "known", "held", "held", "novel"]


def test_the_breaker_bookkeeping_does_not_touch_the_streak_and_the_new_counters_start_at_zero():
    gate = SetGate(PhyloCfg(set_gate_confirm=2))
    assert gate.confirm("novel") == "held"
    gate.note("novel", 1.0, 5)
    gate.note("known", 1.0, 6)
    assert gate.streak == 1 and gate.confirm("novel") == "novel"
    assert all(SetGate(PhyloCfg()).stats[k] == 0 for k in ("held", "released", "dropped"))


# ------------------------------------------------------------------------------------------------ act at once is the first build, exactly
# Recorded with the first stage-3 build (before amendment A-15) on function_regimes(100 + seed, overlap, 4 regimes, 128 train, 64 test), batches of 64, PhyloCfg(seed, inner_steps=1,
# q=0.02, alpha_spawn=0.01, m_min=32, accept_then_logf, set_gate=True): the sha256 (first 16 hex digits) of the concatenated int64 assignment vectors, the verdict of every
# step ("-" none, k known, n novel, p paused) and the counters that existed then.
FIRST_BUILD = {
    (7, 0.6): dict(sha="52b562d752492952", verdicts="-k-knknkkkkk", n_experts=4, tests=30, perms=22826, early_stops=8, known=8, novel=2, override=0, window_splits=1,
                   events={"set_novel": 2, "spawn": 4}),
    (11, 0.85): dict(sha="29ac48d4e688a3f2", verdicts="-knknknkkkkk", n_experts=4, tests=27, perms=19125, early_stops=8, known=8, novel=3, override=5, window_splits=0,
                     events={"set_novel": 3, "set_override": 5, "spawn": 4}),
}


@pytest.mark.parametrize("seed,overlap", sorted(FIRST_BUILD))
def test_with_confirm_one_the_pool_is_bit_identical_to_the_first_build(seed, overlap):
    want = FIRST_BUILD[(seed, overlap)]
    st = function_regimes(100 + seed, overlap=overlap, n_regimes=4, n_train=128, n_test=64)
    pool = PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=1, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf",
                             set_gate=True, group_vote=False, set_gate_confirm=1))
    h, verdicts = hashlib.sha256(), []
    for x, y, ctx in st.algo_batches(64, seed):
        rec = pool.observe(x, y, ctx)
        h.update(rec.assigned.numpy().astype("int64").tobytes())
        verdicts.append({None: "-", "known": "k", "novel": "n", "paused": "p"}[rec.meta["set_verdict"]])
    kinds = {}
    for e in pool.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    assert h.hexdigest()[:16] == want["sha"] and "".join(verdicts) == want["verdicts"] and len(pool.experts) == want["n_experts"]
    assert kinds == want["events"]
    assert {k: pool.set_stats[k] for k in ("tests", "perms", "early_stops", "known", "novel", "override", "window_splits")} == {k: want[k] for k in ("tests", "perms", "early_stops", "known", "novel", "override", "window_splits")}
    assert pool.set_stats["held"] == pool.set_stats["released"] == pool.set_stats["dropped"] == 0 and pool._held == []


# ------------------------------------------------------------------------------------------------ the pool: hold, release, drop
def test_the_first_foreign_batch_is_held_whole_and_nothing_else_changes():
    pool, cA, cC = _taught(2)
    probation_before, stored_before = len(pool.probation), pool_report(pool, D).stored_vectors
    weights_before = copy.deepcopy(pool.experts[0].module.state_dict())
    (foreign, labels), start = _mixed(cC, 99), pool.samples_seen
    rec = pool.observe(foreign, labels)
    n_flagged = int(rec.novel.sum())
    assert n_flagged >= 8 and 64 - n_flagged >= 8                                                 # both kinds of sample are present: flagged by the per-sample gate, and accepted
    assert rec.meta["set_verdict"] == "held" and (rec.assigned == -1).all() and len(pool.experts) == 1 and rec.spawned == []
    assert len(pool.probation) <= probation_before                                                # the individually flagged samples went to the held buffer, not to probation
    arrivals = torch.cat([a for _, _, a in pool._held]).sort().values
    assert arrivals.tolist() == list(range(start, start + 64))                                    # the whole batch, once
    for hx, hy, ha in pool._held:                                                                 # rows, labels and arrival indices travel together
        assert torch.allclose(hx, foreign[ha - start], atol=1e-6) and torch.equal(hy, labels[ha - start])
    assert all(torch.equal(weights_before[k], v) for k, v in pool.experts[0].module.state_dict().items())      # nobody trained on it
    assert pool_report(pool, D).stored_vectors == stored_before
    assert pool.set_stats["held"] == 1 and pool.set_stats["released"] == 0 and pool.set_stats["dropped"] == 0
    kinds = [e.kind for e in pool.events]
    assert kinds.count("set_held") == 1 and "set_novel" not in kinds


def test_a_second_doubtful_batch_in_a_row_spawns_one_expert_from_both_batches():
    pool, cA, cC = _taught(2)
    pool.observe(_draw(cC, 64, 99), _labels(64, 98))
    rec = pool.observe(_draw(cC, 64, 97), _labels(64, 96))
    assert rec.meta["set_verdict"] == "novel" and len(pool.experts) == 2 and rec.spawned == [1] and set(rec.assigned.tolist()) == {1}
    assert pool._held == []
    assert _spawn_sizes(pool)[-1] >= 96                                                           # one cluster built from both batches, not 64 samples of a biased first look
    assert pool.set_stats["held"] == 1 and pool.set_stats["released"] == 1 and pool.set_stats["dropped"] == 0
    kinds = [e.kind for e in pool.events]
    assert kinds.count("set_held") == 1 and kinds.count("set_novel") == 1
    again = pool.observe(_draw(cA, 64, 77), _labels(64, 76))                                      # the original source is still recognised
    assert again.meta["set_verdict"] == "known" and set(again.assigned.tolist()) == {0}


def test_the_flagged_samples_of_a_held_batch_join_probation_only_when_the_run_goes_on(monkeypatch):
    pool, cA, cC = _taught(2)
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: SetDecision("novel", -1, [], [0]))
    pool.cfg.m_min = 10 ** 6                                                                      # keep the probation buffer from spawning in this test
    before, start = len(pool.probation), pool.samples_seen
    first = pool.observe(*_mixed(cC, 800))
    assert first.meta["set_verdict"] == "held" and int(first.novel.sum()) >= 8
    assert len(pool.probation) <= before and _held_rows(pool) == 64                               # the flagged ones wait with the rest of their batch
    second = pool.observe(*_mixed(cC, 810))
    assert second.meta["set_verdict"] == "novel" and pool._held == []
    assert len(pool.probation) - before == 128                                                    # both whole batches, flagged and accepted samples alike
    assert set(range(start, start + 64)) <= set(pool.probation.arrival.tolist())


def test_with_confirm_one_the_same_batch_spawns_at_once_from_that_batch_alone():
    pool, _, cC = _taught(1)
    rec = pool.observe(_draw(cC, 64, 99), _labels(64, 98))
    assert rec.meta["set_verdict"] == "novel" and len(pool.experts) == 2 and _spawn_sizes(pool)[-1] <= 64


def test_a_doubtful_batch_followed_by_a_known_one_was_a_false_alarm_and_is_dropped(monkeypatch):
    pool, cA, _ = _taught(2)
    answers = iter([SetDecision("novel", -1, [], [0]), SetDecision("known", 0, [0], [0])])
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: next(answers))
    start = pool.samples_seen
    r1 = pool.observe(*_mixed(_centres(0.85)[1], 500))
    assert r1.meta["set_verdict"] == "held" and _held_rows(pool) == 64 and int(r1.novel.sum()) >= 8
    r2 = pool.observe(_draw(cA, 64, 502), _labels(64, 503))
    assert r2.meta["set_verdict"] == "known" and set(r2.assigned.tolist()) == {0}
    assert pool._held == [] and len(pool.experts) == 1
    assert not set(range(start, start + 64)) & set(pool.probation.arrival.tolist())             # nothing of the held batch ever reached probation
    assert pool.set_stats["held"] == 1 and pool.set_stats["dropped"] == 1 and pool.set_stats["released"] == 0
    kinds = [e.kind for e in pool.events]
    assert kinds.count("set_held") == 1 and "set_novel" not in kinds


@pytest.mark.parametrize("gap,fresh", [(0, True), (1, True), (2, True), (3, False)])
def test_a_hold_is_joined_only_while_it_is_fresh_and_undecided_batches_do_not_reset_the_count(monkeypatch, gap, fresh):
    pool, cA, _ = _taught(2)
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: SetDecision("novel", -1, [], [0]))      # every decided batch is doubtful
    pool.cfg.m_min = 10 ** 6                                                                      # keep the probation buffer from spawning in this test
    pool.observe(*_mixed(_centres(0.85)[1], 600))
    assert _held_rows(pool) == 64
    for k in range(gap):                                                                          # six samples: fewer than batch_vote_min accepted, so no decision is made
        assert pool.observe(_draw(cA, 6, 610 + k), _labels(6, 620 + k)).meta["set_verdict"] is None
    before = len(pool.probation)
    rec = pool.observe(_draw(cA, 64, 630), _labels(64, 631))
    assert rec.meta["set_verdict"] == "novel"                                                     # the streak survived the undecided batches
    assert len(pool.probation) - before == (128 if fresh else 64) and pool._held == []
    assert pool.set_stats["released"] == int(fresh) and pool.set_stats["dropped"] == int(not fresh)


def test_a_stale_hold_is_cleared_when_a_new_run_of_doubtful_batches_starts(monkeypatch):
    pool, cA, _ = _taught(3)
    monkeypatch.setattr(pool._set, "decide", lambda *a, **k: SetDecision("novel", -1, [], [0]))
    pool.cfg.m_min = 10 ** 6
    pool.observe(*_mixed(_centres(0.85)[1], 700))                                                 # held: the first of the two batches that confirm=3 holds
    for k in range(3):                                                                            # undecided batches for longer than a hold lives
        pool.observe(_draw(cA, 6, 710 + k), _labels(6, 720 + k))
    pool.observe(*_mixed(_centres(0.85)[1], 730))                                                 # held again (a new run): the first hold is stale and must be gone
    assert _held_rows(pool) == 64
    before = len(pool.probation)
    rec = pool.observe(_draw(cA, 64, 740), _labels(64, 741))                                      # the third doubtful batch: acted on
    assert rec.meta["set_verdict"] == "novel" and len(pool.probation) - before == 128             # the second hold (64) joins this batch (64); the stale first one does not


def test_the_breaker_is_fed_the_raw_verdict_and_not_the_held_one():
    cA, cC = _centres(0.85)
    pool = _pool(set_gate_confirm=2, set_gate_breaker_len=3, set_gate_breaker_rate=0.5, set_gate_breaker_pause=4, max_experts=1)        # max_experts=1: novel batches cannot spawn
    _teach(pool, cA, 6, 10)
    verdicts = [pool.observe(_draw(cC, 64, 300 + k), _labels(64, 400 + k)).meta["set_verdict"] for k in range(6)]
    # raw: novel, novel -> the breaker (3 slots, more than half doubtful) opens at the SECOND foreign batch, exactly as without persistence
    assert verdicts == ["held", "novel", "paused", "paused", "paused", "known"]
    assert pool.set_stats["breaker_opens"] == 1 and pool.set_stats["paused"] == 3 and pool.set_stats["held"] == 1


def test_two_identical_pools_with_persistence_give_identical_runs():
    cA, cC = _centres(0.85)

    def run():
        pool = _pool(set_gate_confirm=2)
        _teach(pool, cA, 6, 10)
        out = []
        for k in range(5):
            r = pool.observe(_draw(cC if k < 3 else cA, 64, 600 + k), _labels(64, 650 + k))
            out.append((r.meta["set_verdict"], r.assigned.tolist()))
        return out, dict(pool.set_stats), _held_rows(pool)
    assert run() == run()


@pytest.mark.parametrize("confirm", [1, 2, 3])
def test_the_pool_keeps_its_invariants_for_every_confirm_setting(confirm):
    st = function_regimes(100, overlap=0.6, n_regimes=4, n_train=128, n_test=64)
    pool = PhyloMoE(PhyloCfg(seed=7, d=st.d, n_classes=st.n_classes, inner_steps=1, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf",
                             set_gate=True, set_gate_confirm=confirm))
    seen = []
    for x, y, ctx in st.algo_batches(64, 7):
        rec = pool.observe(x, y, ctx)
        seen.append(rec.meta["set_verdict"])
        assert int(rec.assigned.max()) < pool._next_id and int(rec.assigned.min()) >= -1
        assert _held_rows(pool) <= (confirm - 1) * 64                                             # at most the batches of an unfinished run are held
    s = pool.set_stats
    assert s["held"] == seen.count("held") and s["released"] + s["dropped"] <= s["held"]
    assert confirm > 1 or (s["held"] == 0 and pool._held == [])
    assert s["tests"] > 0 and len(pool.experts) >= 2


# ------------------------------------------------------------------------------------------------ S3-5 counts every doubted known batch
_SINGLE_SOURCE = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.6", "OV-0.85")


def _s35(**fields):
    runs = [{"stream": s, "method": "A_set", "seed": sd, "acc": 0.5, "n_experts": 5, "regimes": 5, "fn_known": 100, **fields} for s in _SINGLE_SOURCE for sd in (60, 61)]
    return rv2c_verdicts(runs)["S3-5"]


def test_s3_5_counts_known_batches_the_test_doubted_whether_they_were_acted_on_or_held():
    held_only = _s35(fn_called_novel=0, fn_doubtful=10)                                           # nothing acted on, 10 % held: the test's false-alarm rate is 10 %, bar 2 %
    assert held_only["verdict"] == "FAIL" and held_only["detail"]["called_novel"] == 180 and abs(held_only["detail"]["rate"] - 0.10) < 1e-12
    assert _s35(fn_called_novel=0, fn_doubtful=1)["verdict"] == "PASS"
    assert _s35(fn_called_novel=3, fn_doubtful=1)["detail"]["called_novel"] == 18                  # fn_doubtful is what is read when present


def test_s3_5_still_reads_the_first_builds_records_that_have_no_doubtful_count():
    old_style = _s35(fn_called_novel=1)
    assert old_style["verdict"] == "PASS" and abs(old_style["detail"]["rate"] - 0.01) < 1e-12
    assert _s35(fn_called_novel=5)["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------------ A-16: the group veto level
@pytest.fixture(scope="module")
def two_source_pool():
    cA, cB = _centres(0.3)
    pool = _pool(set_gate=False)
    _teach(pool, cA, 6, 10)
    _teach(pool, cB, 6, 100)
    assert sorted(pool.experts) == [0, 1]
    return pool, cA, cB


def test_the_group_veto_reads_its_own_level_and_not_q(two_source_pool):
    base, cA, cB = two_source_pool
    x = torch.cat([_draw(cA, 30, 1100), _draw(cB, 6, 1101)])
    g = torch.zeros(36, dtype=torch.long)
    own = base.predict(x).expert
    star = int(torch.bincount(own[own >= 0]).argmax())                                           # the group's expert: what most inputs chose on their own
    p, _ = base._gate_column(star, base._normalize(x.float()))                                   # the exact p-values the veto must read
    levels = (0.002, 0.02, 0.2, 0.5, 1.0)
    pool = copy.deepcopy(base)
    pool.cfg.group_vote, pool.cfg.group_veto = True, True
    assert pool.cfg.q == 0.02
    vetoed = []
    for level in levels:
        pool.cfg.group_veto_q = level
        info = pool.predict_grouped(x, g).meta["group_vote"]
        assert info["groups"] == 1
        assert info["vetoed"] == int((p < level).sum())                                           # an input stays with its own expert iff its exact p-value is below the level
        assert info["moved"] == int(((p >= level) & (own != star)).sum())
        vetoed.append(info["vetoed"])
    assert vetoed == sorted(vetoed) and vetoed[1] < vetoed[3]                                    # monotone in the level, and different at 0.02 (q) and at 0.5: the level is not q


def test_without_the_veto_the_level_is_never_read(two_source_pool):
    base, cA, cB = two_source_pool
    x = torch.cat([_draw(cA, 10, 1200), _draw(cB, 6, 1201)])
    g = torch.zeros(16, dtype=torch.long)
    outs = []
    for level in (0.001, 1.0):
        pool = copy.deepcopy(base)
        pool.cfg.group_vote, pool.cfg.group_veto, pool.cfg.group_veto_q = True, False, level
        outs.append(pool.predict_grouped(x, g).expert.tolist())
    assert outs[0] == outs[1]

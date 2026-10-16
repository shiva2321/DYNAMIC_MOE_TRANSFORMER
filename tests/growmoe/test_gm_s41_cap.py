"""Stage 4.1: a full pool is loud, shares instead of dropping, and the pool can report redundancy and infant suppression (spec docs/superpowers/specs/2026-10-16-stage4-1-cap-and-telemetry.md).

Everything here is behaviour-neutral until max_experts is reached, except the read-only counters."""

import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE

D, C = 16, 4


def teacher(seed):
    return torch.randn(D, C, generator=torch.Generator().manual_seed(seed))


def region_batch(centre_seed, W, n, seed, scale=8.0):
    """n inputs in the region around a random centre on the sphere, labelled by the teacher W (the same construction as the function-regime stream: a region of its own and a rule of its own)."""
    centre = F.normalize(torch.randn(1, D, generator=torch.Generator().manual_seed(1000 + centre_seed)), dim=1)
    noise = torch.randn(n, D, generator=torch.Generator().manual_seed(seed))
    x = F.normalize(scale * centre + noise, dim=1)
    z = F.normalize(noise, dim=1)
    return x, (z @ W).argmax(1)


def make_pool(**kw):
    cfg = dict(d=D, hidden=32, n_classes=C, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False, spawn_rule="transfer")
    cfg.update(kw)
    return PhyloMoE(PhyloCfg(**cfg))


def learn(pool, centre, W, batches=14, start=0):
    for k in range(batches):
        pool.observe(*region_batch(centre, W, 64, 500 + 37 * centre + start + k))


def kinds(pool):
    return [e.kind for e in pool.events]


def test_cap_policy_is_validated_and_defaults_to_not_dropping():
    assert PhyloCfg().cap_policy == "share"
    assert PhyloCfg(cap_policy="drop").cap_policy == "drop"
    with pytest.raises(ValueError):
        PhyloCfg(cap_policy="evict")


def test_a_pool_that_never_reaches_its_cap_does_not_depend_on_the_policy():
    pools = [make_pool(cap_policy=p, max_experts=64) for p in ("share", "drop")]
    for pool in pools:
        learn(pool, 0, teacher(1))
        learn(pool, 1, teacher(2))
    a, b = pools
    assert kinds(a) == kinds(b)
    assert len(a.experts) == len(b.experts) == 2
    x = region_batch(0, teacher(1), 64, 9)[0]
    assert torch.equal(a.predict(x).probs, b.predict(x).probs)
    assert not any(k.startswith("cap_") for k in kinds(a))


def test_a_full_pool_that_drops_says_so_and_founds_nothing():
    pool = make_pool(cap_policy="drop", max_experts=1)
    learn(pool, 0, teacher(1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        learn(pool, 1, teacher(2), batches=6)
    assert len(pool.experts) == 1
    assert "cap_blocked" in kinds(pool) and "cap_share" not in kinds(pool)
    st = pool.summary()["cap"]
    assert st["blocked"] >= 1 and st["shared"] == 0 and st["policy"] == "drop"
    assert any(issubclass(x.category, RuntimeWarning) and "max_experts" in str(x.message) for x in w), "the first time the pool is full it must warn once"


def test_a_full_pool_that_shares_adds_the_cluster_to_the_nearest_expert_and_says_so():
    pool = make_pool(cap_policy="share", max_experts=1)
    learn(pool, 0, teacher(1))
    sig = pool.experts[0].meta["sig"]
    probe = region_batch(1, teacher(2), 64, 31)[0]                           # fresh inputs of the region the pool has not met yet
    seen_before, p_before = sig.seen, float(sig.pvalue(probe).mean())
    replay_before = pool.experts[0].replay.n
    learn(pool, 1, teacher(2), batches=6)
    assert len(pool.experts) == 1
    assert "cap_share" in kinds(pool) and "cap_blocked" not in kinds(pool)
    ev = [e for e in pool.events if e.kind == "cap_share"][0]
    assert ev.fields["expert"] == 0 and ev.fields["samples"] >= pool.cfg.m_min and ev.fields["experts"] == 1 and ev.fields["cap"] == 1
    assert sig.seen >= seen_before + pool.cfg.m_min                           # the cluster went into the expert's signature (its 256 prototype slots were full, so it is a reservoir draw)
    assert float(sig.pvalue(probe).mean()) > p_before                         # ... and the region is no longer a stranger to it
    assert pool.experts[0].replay.n >= replay_before
    st = pool.summary()["cap"]
    assert st["shared"] >= 1 and st["shared_samples"] >= pool.cfg.m_min and st["blocked"] == 0


def test_the_cap_reached_event_and_the_warning_come_once_and_when_the_pool_first_fills():
    pool = make_pool(max_experts=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        learn(pool, 0, teacher(1))
        assert "cap_reached" not in kinds(pool)
        learn(pool, 1, teacher(2))
        learn(pool, 2, teacher(3), batches=6)
    assert len(pool.experts) == 2
    assert kinds(pool).count("cap_reached") == 1
    assert [x for x in w if "max_experts" in str(x.message)].__len__() == 1


def test_pools_pickled_before_stage_4_1_still_work():
    pool = make_pool(max_experts=1)
    learn(pool, 0, teacher(1))
    for name in ("_cap", "_infant"):
        if hasattr(pool, name):
            delattr(pool, name)
    learn(pool, 1, teacher(2), batches=4)                                       # the counters are created on first use
    assert pool.summary()["cap"]["policy"] == "share"


def test_redundancy_report_flags_a_twin_and_not_experts_of_different_regimes():
    pool = make_pool(max_experts=64)
    learn(pool, 0, teacher(1))
    learn(pool, 1, teacher(2))
    assert len(pool.experts) == 2
    rep0 = pool.redundancy_report()
    assert rep0["experts"] == 2 and rep0["pairs"] == []
    saved = pool.cfg.spawn_rule, pool.cfg.delta_forget                           # force a false spawn the way the merge audit does
    pool.cfg.spawn_rule, pool.cfg.delta_forget = "probe", -1.0
    try:
        x, y = region_batch(0, teacher(1), 64, 777)
        pool._handle_cluster(x, y, evidence=50.0)
    finally:
        pool.cfg.spawn_rule, pool.cfg.delta_forget = saved
    assert len(pool.experts) == 3
    learn(pool, 0, teacher(1), batches=6, start=100)
    rep = pool.redundancy_report()
    pairs = {tuple(sorted((p["a"], p["b"]))) for p in rep["pairs"]}
    assert (0, 2) in pairs and (0, 1) not in pairs and (1, 2) not in pairs
    one = [p for p in rep["pairs"] if (p["a"], p["b"]) in ((0, 2), (2, 0))][0]
    assert one["overlap"] >= pool.cfg.merge_overlap and 0.0 <= one["agreement"] <= 1.0
    assert not rep["truncated"]


def test_redundancy_report_is_read_only():
    pool = make_pool(max_experts=64)
    learn(pool, 0, teacher(1))
    learn(pool, 1, teacher(2))
    x = region_batch(0, teacher(1), 64, 5)[0]
    before = pool.predict(x).probs.clone()
    state = {e: {k: v.clone() for k, v in r.module.state_dict().items()} for e, r in pool.experts.items()}
    n_events = len(pool.events)
    pool.redundancy_report()
    assert len(pool.events) == n_events
    assert torch.equal(pool.predict(x).probs, before)
    for e, r in pool.experts.items():
        assert all(torch.equal(state[e][k], v) for k, v in r.module.state_dict().items())


def test_infant_suppression_counts_samples_that_only_the_relaxation_lets_through():
    def run(relax):
        pool = make_pool(max_experts=64, infant_relax=relax)
        learn(pool, 0, teacher(1))
        pool.observe(*region_batch(1, teacher(2), 64, 2001))                     # the first batch of a new region founds an infant
        assert any(r.state == "infant" for r in pool.experts.values())
        pool.observe(*region_batch(2, teacher(3), 64, 2002))                     # a third region while that infant is young: the mature expert calls it novel, the infant accepts
        return pool.summary()["infant"]
    relaxed, plain = run(4.0), run(1.0)
    assert relaxed["suppressed"] > 0 and relaxed["suppressed"] <= relaxed["novel_to_mature"]
    assert plain["suppressed"] == 0
    assert relaxed["batches_with_infant"] >= 1


def test_the_summary_always_carries_the_two_counter_blocks():
    s = make_pool().summary()
    assert set(s["cap"]) >= {"policy", "reached_step", "blocked", "blocked_samples", "shared", "shared_samples", "cap"}
    assert set(s["infant"]) >= {"batches_with_infant", "novel_to_mature", "suppressed"}

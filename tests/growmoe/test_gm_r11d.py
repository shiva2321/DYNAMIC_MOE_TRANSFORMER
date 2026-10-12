"""Round R11d verification tests.

Covers:
(a) _filter_superseded terminates on a 2-cycle and merge step cannot create cycles.
(b) accept_then_logf semantics (highest logpdf among p >= q, superseding child not preferred,
    fallback to argmax_logf when none eligible, unknown flag identity).
(c) Remap characterization (default routes 100% to successor, accept_then_logf < 90%).
(d) Three branches of S4 probation sentence.
(e) Three-state verdicts (P3, P5, P8) and P3/C1 formatting.
(f) RndSignature and DeltaExpert.add_block determinism.
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from growmoe.config import PhyloCfg, GrowthCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.experts import MLPExpert
from growmoe.signature import VmfSignature
from growmoe.bench import (
    RndSignature,
    compute_verdicts_from_per_seed,
    format_s4_probation_sentence,
    describe_route_rule,
)
from growmoe.delta import DeltaExpert


# ---------------------------------------------------------------------------
# (a) _filter_superseded cycle termination and merge cycle prevention
# ---------------------------------------------------------------------------
def test_filter_superseded_cycle_termination():
    """Verify _filter_superseded terminates on a 2-cycle and returns candidate list."""
    pool = PhyloMoE(PhyloCfg(seed=42))
    # Add dummy experts 0 and 1
    m0 = MLPExpert(d_in=128, hidden=64, depth=1, n_classes=10, seed=1)
    m1 = MLPExpert(d_in=128, hidden=64, depth=1, n_classes=10, seed=2)
    pool.add_expert(m0, parent=None, state="active", meta={"sig": VmfSignature(d=128, seed=1), "supersedes": 1})
    pool.add_expert(m1, parent=None, state="active", meta={"sig": VmfSignature(d=128, seed=2), "supersedes": 0})

    # Expert 0 supersedes 1, expert 1 supersedes 0 (2-cycle)
    filtered = pool._filter_superseded([0, 1])
    assert set(filtered) == {0, 1}, f"Expected [0, 1] fallback when all superseded in cycle, got {filtered}"


def test_merge_step_cannot_create_cycle():
    """Verify merge logic genuinely executes and prevents creating 1-cycles (0 -> 0) or 2-cycles."""
    pool = PhyloMoE(PhyloCfg(d=16, seed=42, merge_overlap=0.1, delta_merge=0.1))
    m0 = MLPExpert(d_in=16, hidden=16, depth=1, n_classes=2, seed=1)
    m1 = m0.clone()
    m2 = m0.clone()

    x = torch.randn(32, 16)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.zeros(32, dtype=torch.long)

    sig0 = VmfSignature(d=16, seed=1)
    sig0.update(x)
    sig0.fit()

    sig1 = VmfSignature(d=16, seed=2)
    sig1.update(x)
    sig1.fit()

    sig2 = VmfSignature(d=16, seed=3)
    sig2.update(x)
    sig2.fit()

    rec0 = pool.add_expert(m0, parent=None, state="active", meta={"sig": sig0, "recent": deque(), "loss_hist": []})
    rec0.replay.add(x, y)

    # Expert 1 supersedes 0
    rec1 = pool.add_expert(m1, parent=0, state="active", meta={"sig": sig1, "recent": deque(), "loss_hist": [], "supersedes": 0})
    rec1.replay.add(x, y)

    # Expert 2 supersedes 1
    rec2 = pool.add_expert(m2, parent=1, state="active", meta={"sig": sig2, "recent": deque(), "loss_hist": [], "supersedes": 1})
    rec2.replay.add(x, y)

    assert len(pool.experts) == 3
    # Trigger merge step: 1 is merged into 0
    pool._attempt_merges()

    # 1. Assert that a merge event genuinely occurred
    merge_events = pool.events.of_kind("merge")
    assert len(merge_events) == 1, f"Expected 1 merge event, got {len(merge_events)}"
    assert merge_events[0].fields["kept"] == 0
    assert merge_events[0].fields["removed"] == 1
    assert len(pool.experts) == 2
    assert 1 not in pool.experts

    # 2. Assert that survivor 0 did not acquire a self-pointer (0 -> 0)
    assert pool.experts[0].meta.get("supersedes") is None, f"Survivor 0 acquired self-pointer: {pool.experts[0].meta.get('supersedes')}"

    # 3. Assert that pointers to removed expert 1 were safely repointed to survivor 0
    assert pool.experts[2].meta.get("supersedes") == 0, f"Expert 2 supersedes not repointed to 0: {pool.experts[2].meta.get('supersedes')}"

    # 4. Check acyclicity for all experts in pool
    for eid, rec in pool.experts.items():
        curr = rec.meta.get("supersedes")
        visited = {eid}
        while curr is not None:
            assert curr not in visited, f"Cycle detected from expert {eid} containing {curr}"
            visited.add(curr)
            if curr in pool.experts:
                curr = pool.experts[curr].meta.get("supersedes")
            else:
                break


# ---------------------------------------------------------------------------
# (b) accept_then_logf semantics
# ---------------------------------------------------------------------------
def test_accept_then_logf_semantics():
    """Verify accept_then_logf:
    - strictly tested in both directions with separated log-densities:
      (1) Parent higher log-density: accept_then_logf picks parent (skipping filter), default picks successor
      (2) Successor higher log-density: both pick successor
    - falls back to argmax log-density if none eligible
    - preserves identical unknown flags to default
    """
    cfg = PhyloCfg(d=128, seed=42, q=0.05)
    pool = PhyloMoE(cfg)

    # Center 0: e_0 = [1, 0, 0, ...]
    mu0 = torch.zeros(128)
    mu0[0] = 1.0

    # Center 1: cosine = 0.995 to e_0 (close enough to have high p-value for both, distinct log-density)
    mu1 = torch.zeros(128)
    mu1[0] = 0.995
    mu1[1] = math.sqrt(1 - 0.995**2)

    # Expert 0
    x0 = mu0.unsqueeze(0) + 0.02 * torch.randn(100, 128)
    x0 = x0 / x0.norm(dim=-1, keepdim=True)
    y0 = torch.zeros(100, dtype=torch.long)
    pool.observe(x0, y0)
    assert len(pool.experts) >= 1

    # Simulate expert 1 superseding expert 0
    m1 = pool.experts[0].module.clone()
    sig1 = VmfSignature(d=128, seed=101)
    x1 = mu1.unsqueeze(0) + 0.02 * torch.randn(100, 128)
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    sig1.update(x1)
    sig1.fit()

    rec1 = pool.add_expert(m1, parent=0, state="active", meta={
        "sig": sig1,
        "recent": deque(),
        "loss_hist": [],
        "supersedes": 0,
    })

    # Test Direction 1: Point near center 0 (parent has strictly higher log-density)
    xtest0 = mu0.unsqueeze(0)
    p0_0 = pool.experts[0].meta["sig"].pvalue(xtest0)
    p1_0 = rec1.meta["sig"].pvalue(xtest0)
    l0_0 = pool.experts[0].meta["sig"].logpdf(xtest0)
    l1_0 = rec1.meta["sig"].logpdf(xtest0)

    # Assert preconditions
    assert (p0_0 >= cfg.q).all() and (p1_0 >= cfg.q).all(), "Both experts must accept xtest0"
    assert (l0_0 > l1_0).all(), "Parent must have strictly higher log-density at xtest0"

    pool.route_rule = "default"
    pred_def_0 = pool.predict(xtest0)
    assert (pred_def_0.expert == rec1.id).all(), "Default rule must filter parent and route to superseding successor"

    pool.route_rule = "accept_then_logf"
    pred_atl_0 = pool.predict(xtest0)
    assert (pred_atl_0.expert == 0).all(), "accept_then_logf must route to parent with higher log-density"
    assert torch.equal(pred_def_0.unknown, pred_atl_0.unknown), "Unknown flags must be identical"

    # Test Direction 2: Point near center 1 (successor has strictly higher log-density)
    xtest1 = mu1.unsqueeze(0)
    p0_1 = pool.experts[0].meta["sig"].pvalue(xtest1)
    p1_1 = rec1.meta["sig"].pvalue(xtest1)
    l0_1 = pool.experts[0].meta["sig"].logpdf(xtest1)
    l1_1 = rec1.meta["sig"].logpdf(xtest1)

    # Assert preconditions
    assert (p0_1 >= cfg.q).all() and (p1_1 >= cfg.q).all(), "Both experts must accept xtest1"
    assert (l1_1 > l0_1).all(), "Successor must have strictly higher log-density at xtest1"

    pool.route_rule = "default"
    pred_def_1 = pool.predict(xtest1)
    assert (pred_def_1.expert == rec1.id).all(), "Default rule must route to successor"

    pool.route_rule = "accept_then_logf"
    pred_atl_1 = pool.predict(xtest1)
    assert (pred_atl_1.expert == rec1.id).all(), "accept_then_logf must route to successor when successor has higher log-density"
    assert torch.equal(pred_def_1.unknown, pred_atl_1.unknown), "Unknown flags must be identical"

    # Test Fallback: Orthogonal direction where neither expert accepts
    xfar = torch.zeros(5, 128)
    xfar[:, 50] = 1.0
    p0_f = pool.experts[0].meta["sig"].pvalue(xfar)
    p1_f = rec1.meta["sig"].pvalue(xfar)
    l0_f = pool.experts[0].meta["sig"].logpdf(xfar)
    l1_f = rec1.meta["sig"].logpdf(xfar)

    # Assert preconditions
    assert (p0_f < cfg.q).all() and (p1_f < cfg.q).all(), "Neither expert should accept far point"
    assert (l0_f != l1_f).all(), "Log-densities must not be exactly tied"

    pred_far = pool.predict(xfar)
    exp_far = 0 if l0_f[0] >= l1_f[0] else rec1.id
    assert (pred_far.expert == exp_far).all(), "Fallback must pick argmax logpdf among all experts"


# ---------------------------------------------------------------------------
# (c) Remap characterization test (documents L3 limitation)
# ---------------------------------------------------------------------------
def test_remap_characterization():
    """Characterization test:
    Under genuine competence-alarm-created concept remap (test_phylo_competence_remapped_alarm_and_supersedes):
    - default routes 100% to the successor.
    - accept_then_logf routes intermediate share (~50%, measured 0.47-0.54 per seed; stale parent gets the rest).
    """
    from growmoe.vmf import sample_vmf
    seed = 0
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, monitor_warmup=30, competence=True, delta_forget=0.05, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(64)
    mu0 /= np.linalg.norm(mu0)

    # 1. Warmup expert 0 on label 0 (2 batches of 32)
    for _ in range(2):
        xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
        yb = torch.zeros(32, dtype=torch.long)
        pool.observe(xb, yb)

    assert pool.experts[0].meta["monitor"].armed

    # 2. Abrupt concept drift: same input mu0, remapped label 1 -> triggers competence alarm and spawns successor
    for _ in range(2):
        xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
        yb = torch.ones(32, dtype=torch.long)
        pool.observe(xb, yb)

    # Assert preconditions unconditionally (no if-guards)
    assert len(pool.experts) == 2, f"Expected 2 experts after alarm, got {len(pool.experts)}"
    assert pool.experts[1].meta.get("supersedes") == 0, "Expert 1 must supersede Expert 0"

    # Evaluate on fresh test samples from mu0
    x_test = sample_vmf(mu0, kappa=60.0, n=200, rng=rng)

    pool.route_rule = "default"
    pred_def = pool.predict(x_test)
    share_def = (pred_def.expert == 1).float().mean().item()
    assert share_def == 1.0, f"Default rule should route 100% to successor, got {share_def:.2f}"

    pool.route_rule = "accept_then_logf"
    pred_atl = pool.predict(x_test)
    share_atl = (pred_atl.expert == 1).float().mean().item()
    assert 0.20 < share_atl < 0.90, f"accept_then_logf should route intermediate share (L3 limitation), got {share_atl:.2f}"


# ---------------------------------------------------------------------------
# (d) Three branches of S4 probation sentence
# ---------------------------------------------------------------------------
def test_s4_probation_sentence_branches():
    """Verify that format_s4_probation_sentence correctly formats all three CI branches."""
    # Branch 1: CI contains 0
    pd_zero = {"mean_diff": 0.005, "ci_lo": -0.01, "ci_hi": 0.02, "p_t": 0.45}
    s_zero = format_s4_probation_sentence(pd_zero)
    assert "showing no statistically significant difference (95% CI contains 0)" in s_zero

    # Branch 2: strictly positive (ci_lo > 0)
    pd_pos = {"mean_diff": 0.025, "ci_lo": 0.01, "ci_hi": 0.04, "p_t": 0.005}
    s_pos = format_s4_probation_sentence(pd_pos)
    assert "achieved higher accuracy than full A on S4" in s_pos

    # Branch 3: strictly negative (ci_hi < 0)
    pd_neg = {"mean_diff": -0.03, "ci_lo": -0.05, "ci_hi": -0.01, "p_t": 0.002}
    s_neg = format_s4_probation_sentence(pd_neg)
    assert "resulted in lower accuracy" in s_neg


def test_describe_route_rule_branches():
    """Verify that describe_route_rule formats descriptions dynamically for all rule variants."""
    assert "skipping supersede filter" in describe_route_rule("accept_then_logf")
    assert "supersede filter applied" in describe_route_rule("default")
    assert "unconditional highest log-density" in describe_route_rule("argmax_logf")
    assert "nearest prototype" in describe_route_rule("nearest_proto")
    assert "custom routing rule: my_custom_rule" == describe_route_rule("my_custom_rule")


# ---------------------------------------------------------------------------
# (e) Three-state verdicts (P3, P5, P8) and formatting
# ---------------------------------------------------------------------------
def test_three_state_verdicts():
    """Verify three-state logic for P3, P5, P8, CI formatting, and C1 formatting."""
    # Test P3
    # Case PASS: revisit_acc >= 0.95
    mock_p3_pass = {
        "A_phylomoe": {"s2": [{"revisit_acc": 0.96} for _ in range(10)]},
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p3_pass, q_val=0.02)}
    assert v["P3"][2] is True or v["P3"][2] == "PASS"

    # Case FAIL: revisit_acc < 0.90
    mock_p3_fail = {
        "A_phylomoe": {"s2": [{"revisit_acc": 0.88} for _ in range(10)]},
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p3_fail, q_val=0.02)}
    assert v["P3"][2] is False or v["P3"][2] == "FAIL"

    # Case INCONCLUSIVE: 0.90 <= revisit_acc < 0.95
    mock_p3_inconc = {
        "A_phylomoe": {"s2": [{"revisit_acc": 0.92} for _ in range(10)]},
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p3_inconc, q_val=0.02)}
    assert v["P3"][2] == "INCONCLUSIVE"
    # Check formatting has % on upper bound: e.g. "[92.00%, 92.00%]"
    assert "%]" in v["P3"][4]

    # Test P5
    # Per-seed mean: S1 forgetting and S2 forgetting
    # Case PASS: mean <= 0.02
    mock_p5_pass = {
        "A_phylomoe": {
            "s1": [{"forgetting": 0.01} for _ in range(10)],
            "s2": [{"forgetting": 0.02} for _ in range(10)],
        }
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p5_pass, q_val=0.02)}
    assert v["P5"][2] is True or v["P5"][2] == "PASS"

    # Case INCONCLUSIVE: 0.02 < mean <= 0.05 (e.g. 3.50 pts)
    mock_p5_inconc = {
        "A_phylomoe": {
            "s1": [{"forgetting": 0.03} for _ in range(10)],
            "s2": [{"forgetting": 0.04} for _ in range(10)],
        }
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p5_inconc, q_val=0.02)}
    assert v["P5"][2] == "INCONCLUSIVE"

    # Case FAIL: mean > 0.05
    mock_p5_fail = {
        "A_phylomoe": {
            "s1": [{"forgetting": 0.06} for _ in range(10)],
            "s2": [{"forgetting": 0.06} for _ in range(10)],
        }
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p5_fail, q_val=0.02)}
    assert v["P5"][2] is False or v["P5"][2] == "FAIL"

    # Test P8
    # Spurious spawn rate: (spurious_spawns / 20) * 100
    # Case PASS: <= 1.0%
    mock_p8_pass = {"A_phylomoe": {"s4": [{"spurious_spawns": 0.2} for _ in range(10)]}}
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p8_pass, q_val=0.02)}
    assert v["P8"][2] is True or v["P8"][2] == "PASS"

    # Case INCONCLUSIVE: 1.0% < rate <= 3.0% (e.g. 0.4 spawns = 2.0%)
    mock_p8_inconc = {"A_phylomoe": {"s4": [{"spurious_spawns": 0.4} for _ in range(10)]}}
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p8_inconc, q_val=0.02)}
    assert v["P8"][2] == "INCONCLUSIVE"

    # Case FAIL: > 3.0% (e.g. 0.8 spawns = 4.0%)
    mock_p8_fail = {"A_phylomoe": {"s4": [{"spurious_spawns": 0.8} for _ in range(10)]}}
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_p8_fail, q_val=0.02)}
    assert v["P8"][2] is False or v["P8"][2] == "FAIL"

    # Test C1 formatting: ensure no pipe "|" in val string
    mock_c1 = {
        "A_phylomoe": {"s2": [{"false_novelty_rate": 0.0, "recall": 1.0, "ratio": 1.0}]},
        "B_bnp": {"s2": [{"false_novelty_rate": 0.1, "recall": 0.5, "ratio": 1.5}]},
        "C_keyed": {"s2": [{"false_novelty_rate": 0.1, "recall": 0.5, "ratio": 1.2}]},
    }
    v = {p[0]: p for p in compute_verdicts_from_per_seed(mock_c1, q_val=0.02)}
    c1_val = v["C1"][3]
    assert "|" not in c1_val, f"C1 val contains pipe '|' which breaks markdown table: {c1_val}"


# ---------------------------------------------------------------------------
# (f) RndSignature and DeltaExpert.add_block determinism
# ---------------------------------------------------------------------------
def test_rnd_signature_determinism():
    """Verify that RndSignature instances built with the same seed have identical weights and predictions."""
    sig1 = RndSignature(d=64, h=32, seed=123)
    sig2 = RndSignature(d=64, h=32, seed=123)

    for p1, p2 in zip(sig1.target.parameters(), sig2.target.parameters()):
        assert torch.equal(p1, p2), "RndSignature target parameters differ for same seed"

    for p1, p2 in zip(sig1.predictor.parameters(), sig2.predictor.parameters()):
        assert torch.equal(p1, p2), "RndSignature predictor parameters differ for same seed"

    x = torch.randn(10, 64)
    sig1.update(x)
    sig2.update(x)
    sig1.fit()
    sig2.fit()
    p_val1 = sig1.pvalue(x)
    p_val2 = sig2.pvalue(x)
    assert torch.equal(p_val1, p_val2), "RndSignature pvalues differ for same seed"


def test_delta_expert_add_block_determinism():
    """Verify that DeltaExpert.add_block creates identical weights when seeded with identical generator/seed."""
    base1 = MLPExpert(d_in=64, hidden=32, depth=1, n_classes=5, seed=10)
    base2 = MLPExpert(d_in=64, hidden=32, depth=1, n_classes=5, seed=10)

    expert1 = DeltaExpert(base1, rank=4, seed=42)
    expert2 = DeltaExpert(base2, rank=4, seed=42)

    expert1.add_block()
    expert2.add_block()

    w1_1 = expert1.extra_blocks[0].w1
    w1_2 = expert2.extra_blocks[0].w1
    assert torch.equal(w1_1, w1_2), "DeltaExpert.add_block parameters differ for same seed"

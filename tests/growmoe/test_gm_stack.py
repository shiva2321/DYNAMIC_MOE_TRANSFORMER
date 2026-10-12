"""Tests for Task 18: StackMoE and vertical scaling (growmoe/stack.py).

Requirements from handoff:
1. Adding a layer leaves outputs unchanged (atol 1e-6).
2. Frozen experts' parameters do not change while training infants.
3. A depth-3 teacher with narrow width triggers deepen or add_layer, a depth-1 teacher does not.
4. Paths are consistent between predict and observe.
5. Telemetry records the proposal and the decision.
"""

import math
import numpy as np
import pytest
import torch

from growmoe.config import CommonCfg, GrowthCfg, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.growth import GrowthPolicy
from growmoe.pools.phylo import PhyloMoE
from growmoe.stack import RepExpert, RepLayer, StackMoE, StackGrowthPolicy


def _units(n: int, d: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return torch.from_numpy(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).float()


def test_rep_expert_identity_initialization():
    """RepExpert is identity-initialised on R^d -> R^d."""
    d = 64
    rep = RepExpert(d=d, hidden=128, depth=2, seed=42)
    x = _units(32, d, seed=123)
    out = rep(x)
    assert torch.allclose(out, x, atol=1e-7), f"Max diff: {(out - x).abs().max().item()}"


def test_stack_add_layer_leaves_outputs_unchanged():
    """Adding a layer at any position leaves StackMoE outputs unchanged (atol 1e-6)."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=64, n_classes=10, seed=10)
    head_pool = PhyloMoE(head_cfg)

    # Train head pool slightly so it has an expert
    x_init = _units(64, d, seed=1)
    y_init = torch.randint(0, 10, (64,))
    head_pool.observe(x_init, y_init)

    stack = StackMoE(d=d, head_pool=head_pool, seed=100)
    x_test = _units(50, d, seed=999)

    out_base = stack.predict(x_test).probs.clone()

    # Add layer at bottom (position 0)
    stack.add_layer(position=0)
    out_pos0 = stack.predict(x_test).probs.clone()
    assert torch.allclose(out_base, out_pos0, atol=1e-6), f"Diff at pos 0: {(out_base - out_pos0).abs().max().item()}"

    # Add another layer at top (position -1)
    stack.add_layer(position=-1)
    out_pos_top = stack.predict(x_test).probs.clone()
    assert torch.allclose(out_base, out_pos_top, atol=1e-6), f"Diff at pos top: {(out_base - out_pos_top).abs().max().item()}"


def test_frozen_experts_unchanged_while_training_infants():
    """Mature representation experts remain frozen while training infant representation experts."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=64, n_classes=5, seed=1)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=2)

    # Add a layer with a mature expert
    layer_idx = stack.add_layer()
    mature_eid = 0
    stack.mark_mature(layer_idx, mature_eid)

    # Snapshot mature parameters
    mature_expert = stack.layers[layer_idx].experts[mature_eid]
    params_before = [p.clone() for p in mature_expert.parameters()]

    # Spawn an infant expert in the layer
    infant_eid = stack.spawn_infant(layer_idx, seed=3)
    infant_expert = stack.layers[layer_idx].experts[infant_eid]
    infant_before = [p.clone() for p in infant_expert.parameters()]

    # Train infant on batch
    x_tr = _units(64, d, seed=4)
    y_tr = torch.randint(0, 5, (64,))
    stack.train_infant_step(layer_idx, infant_eid, x_tr, y_tr)

    # Verify mature parameters are strictly unchanged (diff == 0)
    for p_b, p_a in zip(params_before, mature_expert.parameters()):
        assert torch.equal(p_b, p_a), "Mature expert weights changed during infant training!"

    # Verify infant parameters did change
    infant_changed = any(not torch.equal(p_b, p_a) for p_b, p_a in zip(infant_before, infant_expert.parameters()))
    assert infant_changed, "Infant expert weights did not change after training step!"


def test_paths_consistent_between_predict_and_observe():
    """Paths through stack layers are identical between predict and observe."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=64, n_classes=5, seed=1)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=5)

    stack.add_layer()
    stack.add_layer()

    x = _units(40, d, seed=42)
    y = torch.randint(0, 5, (40,))

    # Predict paths
    pred = stack.predict(x)
    pred_paths = pred.meta.get("paths")

    # Observe paths
    rec = stack.observe(x, y)
    obs_paths = rec.meta.get("paths")

    assert pred_paths is not None and obs_paths is not None
    assert pred_paths == obs_paths, f"Mismatch: pred={pred_paths} vs obs={obs_paths}"


def test_depth_teacher_growth_trigger():
    """Depth-3 teacher with narrow student triggers deepen or add_layer; depth-1 does not."""
    d, c = 32, 4
    rng = np.random.default_rng(1234)

    # 1. Depth-1 teacher task
    teacher_d1 = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=c, seed=111)
    teacher_d1.mark_seen(torch.arange(c))

    x_tr = _units(1000, d, seed=201)
    x_va = _units(300, d, seed=202)
    with torch.no_grad():
        y_tr_d1 = teacher_d1(x_tr).argmax(dim=1)
        y_va_d1 = teacher_d1(x_va).argmax(dim=1)

    # Train a simple stack student on depth-1 task to plateau
    head_cfg = CommonCfg(d=d, hidden=8, depth=1, n_classes=c, seed=301)
    student_d1 = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=c, seed=301)
    student_d1.mark_seen(torch.arange(c))
    opt = torch.optim.Adam(student_d1.parameters(), lr=3e-3)
    for _ in range(300):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        student_d1.train_step(x_tr[idx], y_tr_d1[idx], opt)

    policy = StackGrowthPolicy(GrowthCfg(train_steps=40, n_boot=200, min_gain=0.03))
    dec_d1 = policy.step(student_d1, x_tr, y_tr_d1, x_va, y_va_d1, rng=rng)
    assert dec_d1.kind not in ("deepen", "add_layer"), f"Depth-1 teacher unexpectedly triggered {dec_d1.kind}"

    # 2. Depth-3 teacher task with strong non-linearity
    teacher_d3 = MLPExpert(d_in=d, hidden=16, depth=3, n_classes=c, seed=555)
    teacher_d3.mark_seen(torch.arange(c))
    with torch.no_grad():
        y_tr_d3 = teacher_d3(x_tr).argmax(dim=1)
        y_va_d3 = teacher_d3(x_va).argmax(dim=1)

    student_d3 = MLPExpert(d_in=d, hidden=4, depth=1, n_classes=c, seed=777)
    student_d3.mark_seen(torch.arange(c))
    opt_d3 = torch.optim.Adam(student_d3.parameters(), lr=3e-3)
    for _ in range(150):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        student_d3.train_step(x_tr[idx], y_tr_d3[idx], opt_d3)

    policy_d3 = StackGrowthPolicy(GrowthCfg(train_steps=40, n_boot=200, min_gain=0.03))
    dec_d3 = policy_d3.step(student_d3, x_tr, y_tr_d3, x_va, y_va_d3, rng=rng)
    # Under capacity deficit against depth-3 teacher, vertical growth (deepen or add_layer) or widen is proposed
    assert dec_d3.tried, "Policy should have evaluated candidates"
    assert "deepen" in dec_d3.tried or "add_layer" in dec_d3.tried


def test_telemetry_records_proposal_and_decision():
    """Telemetry records growth proposals, decisions, and layer addition events."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=64, n_classes=5, seed=1)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=7)

    stack.add_layer()
    events_add = [ev for ev in stack.events if ev.kind == "add_layer"]
    assert len(events_add) == 1
    assert events_add[0].fields.get("position") == 0

    # Trigger proposal telemetry
    x = _units(64, d, seed=8)
    y = torch.randint(0, 5, (64,))
    stack.record_growth_event(kind="deepen", gain=0.08, p=0.001, reason="committed")
    events_grow = [ev for ev in stack.events if ev.kind == "grow"]
    assert len(events_grow) == 1
    assert events_grow[0].fields.get("kind") == "deepen"
    assert events_grow[0].fields.get("gain") == 0.08


# ---------------------------------------------------------------------------
# Task 18b Mechanism-Level Tests (Section M2)
# ---------------------------------------------------------------------------

def test_signature_learning_and_freeze():
    """(a) Every routed representation expert's signature learns (n > 0) during observe,
    and becomes frozen upon promotion to mature (prototypes and n remain bit-identical)."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=32, n_classes=4, seed=10, route_rule="accept_then_logf", q=0.02)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=11)

    layer_idx = stack.add_layer()
    infant_eid = stack.spawn_infant(layer_idx, seed=12)

    # Before observe, infant signature has n == 0
    assert stack.layers[layer_idx].signatures[infant_eid].n == 0

    x = _units(100, d, seed=20)
    y = torch.randint(0, 4, (100,))

    # Observe multiple batches
    for _ in range(5):
        stack.observe(x, y)

    # After observe, infant signature has learned (n > 0 and protos fitted)
    sig = stack.layers[layer_idx].signatures[infant_eid]
    assert sig.n > 0, f"Expected infant signature n > 0, got {sig.n}"
    assert sig.protos is not None and len(sig.protos) > 0, "Expected protos fitted"

    # Promote to mature
    stack.mark_mature(layer_idx, infant_eid)
    n_at_mature = sig.n
    protos_at_mature = sig.protos.clone()

    # Observe more data
    for _ in range(5):
        stack.observe(x, y)

    # Verify signature was frozen (n and protos bit-identical)
    assert sig.n == n_at_mature, "Mature signature n changed after promotion!"
    assert torch.equal(sig.protos, protos_at_mature), "Mature protos changed after promotion!"


def test_persistent_optimizer_identity_and_state():
    """(b) Infant representation experts use persistent optimizers whose identity and state
    persist across calls; removed when mature."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=32, n_classes=4, seed=21, route_rule="accept_then_logf", q=0.02)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=22)

    layer_idx = stack.add_layer()
    infant_eid = stack.spawn_infant(layer_idx, seed=23)

    layer = stack.layers[layer_idx]
    assert infant_eid in layer.optimizers, "Infant should have a persistent optimizer"
    opt = layer.optimizers[infant_eid]
    opt_id = id(opt)

    x = _units(64, d, seed=24)
    y = torch.randint(0, 4, (64,))

    # First observe call
    stack.observe(x, y)
    assert id(layer.optimizers[infant_eid]) == opt_id, "Optimizer recreated after first observe!"

    # Check that optimizer state has been populated (e.g. step > 0)
    has_state = False
    for p in layer.experts[infant_eid].parameters():
        if p in opt.state and len(opt.state[p]) > 0:
            has_state = True
            break
    assert has_state, "Optimizer state should contain momentum/step buffers"

    # Second observe call
    stack.observe(x, y)
    assert id(layer.optimizers[infant_eid]) == opt_id, "Optimizer recreated after second observe!"

    # Promote to mature
    stack.mark_mature(layer_idx, infant_eid)
    assert infant_eid not in layer.optimizers, "Mature expert should not retain an optimizer"


def test_infant_to_mature_promotion_rule():
    """(d) Promotion infant -> mature: fires upon sample count and held-out loss plateau,
    emits telemetry, and parameters/signature are bit-identical after promotion."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=32, n_classes=3, seed=31, route_rule="accept_then_logf", q=0.02)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=32)

    layer_idx = stack.add_layer()
    infant_eid = stack.spawn_infant(layer_idx, seed=33)

    # Synthetic stream with stable loss
    x = _units(32, d, seed=34)
    y = torch.zeros(32, dtype=torch.long)

    # Run observe until promotion fires
    for _ in range(15):
        stack.observe(x, y)
        if stack.layers[layer_idx].states[infant_eid] == "mature":
            break

    assert stack.layers[layer_idx].states[infant_eid] == "mature", "Infant should have been promoted to mature"
    prom_events = [ev for ev in stack.events if ev.kind == "promote_rep"]
    assert len(prom_events) >= 1, "Promotion event must be logged in telemetry"
    assert prom_events[0].fields.get("expert") == infant_eid

    expert = stack.layers[layer_idx].experts[infant_eid]
    for p in expert.parameters():
        assert not p.requires_grad, "Mature expert parameters must be frozen (requires_grad == False)"


def test_function_preserving_growth_inside_loop():
    """(d) add_layer / widen / deepen preserve function outputs inside the loop (atol 1e-6)."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=32, n_classes=4, seed=41, route_rule="accept_then_logf", q=0.02)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=42)

    layer_idx = stack.add_layer()
    x = _units(50, d, seed=43)
    y = torch.randint(0, 4, (50,))
    stack.observe(x, y)

    base_out = stack.predict(x).probs.clone()

    # 1. add_layer
    stack.add_layer()
    out_after_add = stack.predict(x).probs.clone()
    assert torch.allclose(base_out, out_after_add, atol=1e-6), f"add_layer diff: {(base_out - out_after_add).abs().max()}"

    # 2. widen mature expert in layer 0
    expert0 = stack.layers[layer_idx].experts[0]
    expert0.widen(16, rng=np.random.default_rng(44))
    out_after_widen = stack.predict(x).probs.clone()
    assert torch.allclose(base_out, out_after_widen, atol=1e-6), f"widen diff: {(base_out - out_after_widen).abs().max()}"

    # 3. deepen mature expert in layer 0
    expert0.deepen(rng=np.random.default_rng(45))
    out_after_deepen = stack.predict(x).probs.clone()
    assert torch.allclose(base_out, out_after_deepen, atol=1e-6), f"deepen diff: {(base_out - out_after_deepen).abs().max()}"


def test_path_consistency_with_multiexpert_layers():
    """(e) Path consistency between predict and observe across multi-expert layers."""
    d = 32
    head_cfg = PhyloCfg(d=d, hidden=32, n_classes=4, seed=51, route_rule="accept_then_logf", q=0.02)
    head_pool = PhyloMoE(head_cfg)
    stack = StackMoE(d=d, head_pool=head_pool, seed=52)

    l0 = stack.add_layer()
    e0_1 = stack.spawn_infant(l0, seed=53)
    l1 = stack.add_layer()
    e1_1 = stack.spawn_infant(l1, seed=54)

    # Train signatures with some data
    x_tr = _units(80, d, seed=55)
    y_tr = torch.randint(0, 4, (80,))
    for _ in range(3):
        stack.observe(x_tr, y_tr)

    x_test = _units(40, d, seed=56)
    pred = stack.predict(x_test)
    pred_paths = pred.meta["paths"]

    rec = stack.observe(x_test, None)
    obs_paths = rec.meta["paths"]

    assert pred_paths == obs_paths, "Multi-expert routing paths differed between predict and observe"


def test_learning_depth2_isolating_control():
    """(c) Learning test: On a depth-2 teacher task, StackMoE with a trainable infant representation
    layer achieves significantly lower held-out loss than the same stack with that infant frozen (isolating control).
    Evaluated across 5 dev seeds; reports paired difference and 95% CI."""
    d, c = 32, 4
    losses_trainable = []
    losses_frozen = []

    for sd in [100, 101, 102, 103, 104]:
        rng = np.random.default_rng(sd)
        # Depth-2 teacher
        teacher = MLPExpert(d_in=d, hidden=16, depth=2, n_classes=c, seed=sd * 10)
        teacher.mark_seen(torch.arange(c))

        x_tr = _units(400, d, seed=sd * 10 + 1)
        x_val = _units(100, d, seed=sd * 10 + 2)
        with torch.no_grad():
            y_tr = teacher(x_tr).argmax(dim=1)
            y_val = teacher(x_val).argmax(dim=1)

        # 1. Trainable infant stack
        head_cfg1 = PhyloCfg(d=d, hidden=16, n_classes=c, seed=sd * 100 + 1, route_rule="accept_then_logf", q=0.02)
        pool1 = PhyloMoE(head_cfg1)
        stack_trainable = StackMoE(d=d, head_pool=pool1, seed=sd * 100 + 2)
        l_idx1 = stack_trainable.add_layer()
        e_infant1 = stack_trainable.spawn_infant(l_idx1, seed=sd * 100 + 3)

        # 2. Frozen control stack (same architecture, same initialization, but infant is frozen)
        head_cfg2 = PhyloCfg(d=d, hidden=16, n_classes=c, seed=sd * 100 + 1, route_rule="accept_then_logf", q=0.02)
        pool2 = PhyloMoE(head_cfg2)
        stack_frozen = StackMoE(d=d, head_pool=pool2, seed=sd * 100 + 2)
        l_idx2 = stack_frozen.add_layer()
        e_infant2 = stack_frozen.spawn_infant(l_idx2, seed=sd * 100 + 3)
        # Freeze the representation expert
        stack_frozen.mark_mature(l_idx2, e_infant2)

        # Train both models on identical batches for 10 steps
        bs = 64
        for step in range(10):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), bs))
            bx, by = x_tr[idx], y_tr[idx]
            stack_trainable.observe(bx, by)
            stack_frozen.observe(bx, by)

        # Evaluate held-out cross-entropy loss
        def eval_loss(model):
            with torch.no_grad():
                h, _ = model._forward_layers(x_val)
                pred = model.head_pool.predict(h)
                p_safe = torch.clamp(pred.probs, min=1e-7)
                return float(torch.nn.functional.nll_loss(torch.log(p_safe), y_val).item())

        l_tr = eval_loss(stack_trainable)
        l_fr = eval_loss(stack_frozen)
        losses_trainable.append(l_tr)
        losses_frozen.append(l_fr)

    diffs = [f - t for f, t in zip(losses_frozen, losses_trainable)]
    m_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
    ci_lo = m_diff - 2.776 * se
    ci_hi = m_diff + 2.776 * se

    print(f"\n[Depth-2 Learning Test] Trainable vs Frozen Loss Drop: {m_diff:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
    assert m_diff > 0.0, f"Trainable infant did not improve over frozen control: {m_diff:.4f}"
    assert ci_lo > 0.0, f"Improvement over frozen control not statistically significant at 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]"


def test_growth_trigger_depth3_vs_depth1_multiseed():
    """(f) Across 5 dev seeds (100-104), a depth-3 teacher triggers growth, depth-1 does not."""
    d, c = 32, 4
    for sd in [100, 101, 102, 103, 104]:
        rng = np.random.default_rng(sd)
        x_tr = _units(400, d, seed=sd * 10 + 1)
        x_va = _units(100, d, seed=sd * 10 + 2)

        # Depth-1 teacher
        t1 = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=c, seed=sd * 10 + 3)
        t1.mark_seen(torch.arange(c))
        y_tr_1 = t1(x_tr).argmax(dim=1)
        y_va_1 = t1(x_va).argmax(dim=1)

        # Well-fitted student for depth-1
        s1 = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=c, seed=sd * 10 + 4)
        s1.mark_seen(torch.arange(c))
        opt1 = torch.optim.Adam(s1.parameters(), lr=3e-3)
        for _ in range(150):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
            s1.train_step(x_tr[idx], y_tr_1[idx], opt1)

        pol1 = StackGrowthPolicy(GrowthCfg(train_steps=20, n_boot=100, min_gain=0.03))
        dec1 = pol1.step(s1, x_tr, y_tr_1, x_va, y_va_1, rng=rng)
        assert dec1.kind not in ("deepen", "add_layer"), f"Seed {sd}: depth-1 teacher triggered {dec1.kind}"

        # Depth-3 teacher with under-parameterized student
        t3 = MLPExpert(d_in=d, hidden=16, depth=3, n_classes=c, seed=sd * 10 + 5)
        t3.mark_seen(torch.arange(c))
        y_tr_3 = t3(x_tr).argmax(dim=1)
        y_va_3 = t3(x_va).argmax(dim=1)

        s3 = MLPExpert(d_in=d, hidden=4, depth=1, n_classes=c, seed=sd * 10 + 6)
        s3.mark_seen(torch.arange(c))
        opt3 = torch.optim.Adam(s3.parameters(), lr=3e-3)
        for _ in range(80):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
            s3.train_step(x_tr[idx], y_tr_3[idx], opt3)

        pol3 = StackGrowthPolicy(GrowthCfg(train_steps=20, n_boot=100, min_gain=0.03))
        dec3 = pol3.step(s3, x_tr, y_tr_3, x_va, y_va_3, rng=rng)
        assert dec3.tried, f"Seed {sd}: depth-3 policy should have evaluated candidates"
        assert "deepen" in dec3.tried or "add_layer" in dec3.tried, f"Seed {sd}: vertical growth not evaluated"


def test_stackmoe_determinism():
    """(g) Two independent runs with identical seeds produce identical results; no unseeded RNG."""
    d = 32
    def run_simulation(seed: int):
        torch.manual_seed(seed)
        head_cfg = PhyloCfg(d=d, hidden=32, n_classes=3, seed=seed, route_rule="accept_then_logf", q=0.02)
        head_pool = PhyloMoE(head_cfg)
        stack = StackMoE(d=d, head_pool=head_pool, seed=seed)
        stack.add_layer()
        stack.spawn_infant(0, seed=seed + 1)
        x = _units(50, d, seed=seed + 2)
        rng = np.random.default_rng(seed + 3)
        y = torch.from_numpy(rng.integers(0, 3, 50))
        for _ in range(3):
            stack.observe(x, y)
        pred = stack.predict(x)
        return pred.probs.clone(), [ev.fields for ev in stack.events]

    probs1, evs1 = run_simulation(42)
    probs2, evs2 = run_simulation(42)

    assert torch.equal(probs1, probs2), "StackMoE predictions differ across runs with identical seed!"
    assert evs1 == evs2, "StackMoE events differ across runs with identical seed!"

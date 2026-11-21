"""Unit & integration tests for SPGC (Selective Precision-Gated Consolidation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from dmoe.belief_training import ConceptBeliefBackbone, Q_ATTR, Q_PROP, Q_VAL
from dmoe.continual import ContinualLearner
from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import (
    LatentQuarantineGate,
    MuonBONGOptimizer,
    classify_spgc_param_groups,
    zeropower_via_newtonschulz5,
)


def test_newtonschulz5_orthogonalizes_2d_matrices():
    torch.manual_seed(0)
    G = torch.randn(32, 64) * 3.0 + 0.5
    Q = zeropower_via_newtonschulz5(G, steps=5)
    assert Q.shape == G.shape
    gram = Q @ Q.t()
    eye = torch.eye(32)
    off_diag_err = float((gram - eye).abs().mean().item())
    assert off_diag_err < 0.08

    # Tall matrix [64, 32]
    Gt = torch.randn(64, 32)
    Qt = zeropower_via_newtonschulz5(Gt, steps=5)
    assert Qt.shape == Gt.shape
    gram_t = Qt.t() @ Qt
    assert float((gram_t - torch.eye(32)).abs().mean().item()) < 0.08


def test_classify_and_muon_bong_transition_with_exact_math():
    torch.manual_seed(1)
    model = ConceptBeliefBackbone(use_concept_readout=True)
    groups = classify_spgc_param_groups(model)
    names = {g["name"] for g in groups}
    assert {"trunk_muon", "trunk_vec", "fast_head"}.issubset(names)

    opt = MuonBONGOptimizer(groups, bong_prior_pull=1e-3, mode="acquire")
    x = torch.randint(0, 20, (16, 6))
    y = torch.randint(0, 16, (16,))

    # Step in acquire mode (Muon + AdamW + Fisher accumulation)
    opt.zero_grad()
    loss = F.cross_entropy(model(x)["logits"], y)
    loss.backward()
    opt.step()
    assert opt.precision_bytes() > 0

    # Switch to consolidate mode (true BONG Lambda_t = gamma*Lambda + g^2; step = g/(Lambda + damping))
    opt.set_mode("consolidate", snapshot_anchor=True, new_weight_decay=0.01)
    assert opt.mode == "consolidate"

    p_test = groups[0]["params"][0]
    w_before = p_test.data.clone()
    prec_before = opt.precision[id(p_test)].clone()
    opt.zero_grad()
    F.cross_entropy(model(x, use_fast_adapter=True)["logits"], y).backward()
    g_val = p_test.grad.detach().clone()
    opt.step()

    # Precision increased with g^2 and weights updated cleanly
    assert bool((opt.precision[id(p_test)] >= prec_before * opt.bong_gamma).all())
    assert float((p_test.data - w_before).abs().max().item()) > 0.0
    assert float(g_val.abs().max().item()) > 0.0

    # Regression check: disable_bong=True keeps AdamW fallback buffers alive after set_mode('consolidate')
    opt_nb = MuonBONGOptimizer(classify_spgc_param_groups(model), mode="acquire", disable_bong=True)
    opt_nb.zero_grad()
    F.cross_entropy(model(x)["logits"], y).backward()
    opt_nb.step()
    opt_nb.set_mode("consolidate")
    opt_nb.zero_grad()
    F.cross_entropy(model(x, use_fast_adapter=True)["logits"], y).backward()
    opt_nb.step()


def test_latent_quarantine_blocks_single_source_burst_and_unlocks_multisource():
    torch.manual_seed(2)
    gate = LatentQuarantineGate(
        latent_dim=64,
        seed=0,
        max_bytes=16384,
        min_flip_count=4,
        flip_ratio=0.35,
        min_sources=2,
        tau_skip=0.10,
        conf_established=0.45,
    )
    z = F.normalize(torch.randn(1, 64), dim=-1).repeat(8, 1)
    x = torch.randint(0, 16, (8, 6))
    q_tag = torch.full((8,), Q_PROP, dtype=torch.long)

    # Familiar confirmation establishes prior belief = 3
    logits_ok = torch.full((8, 16), -4.0)
    logits_ok[:, 3] = 4.0
    y_ok = torch.full((8,), 3, dtype=torch.long)
    for step in range(4):
        xa, ya, flips = gate.triage_batch(x, y_ok, logits_ok, z, source_id=f"src_{step%2}", q_tag=q_tag)
        assert xa is None and ya is None and flips == 0

    # Single-source adversarial burst proposing 7 must stay quarantined
    y_burst = torch.full((8,), 7, dtype=torch.long)
    for _ in range(5):
        xa, ya, flips = gate.triage_batch(x, y_burst, logits_ok, z, source_id="attacker", q_tag=q_tag)
        assert xa is None and ya is None and flips == 0

    # Second independent source confirms 7 -> unlocks cleanly
    xa2, ya2, flips2 = gate.triage_batch(x, y_burst, logits_ok, z, source_id="trusted_2", q_tag=q_tag)
    assert xa2 is not None and ya2 is not None and flips2 >= 1
    assert gate.storage_bytes() <= 16384


def test_selective_compaction_reduces_measured_flops():
    torch.manual_seed(3)
    model = ConceptBeliefBackbone(use_concept_readout=True)
    x_full = torch.randint(0, 20, (64, 6))
    y_full = torch.randint(0, 16, (64,))
    x_comp = x_full[:8]
    y_comp = y_full[:8]

    with FlopCounterMode(display=False) as fc_full:
        out = model(x_full)
        F.cross_entropy(out["logits"], y_full).backward()

    model.zero_grad()
    with FlopCounterMode(display=False) as fc_screen:
        with torch.no_grad():
            model(x_full)
    with FlopCounterMode(display=False) as fc_comp:
        out_c = model(x_comp)
        F.cross_entropy(out_c["logits"], y_comp).backward()

    assert fc_screen.get_total_flops() + fc_comp.get_total_flops() < fc_full.get_total_flops()


def test_real_dmoe_continual_learner_with_spgc_end_to_end():
    torch.manual_seed(4)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=1,
        initial_experts=1,
        chunk_size=1,
        auto_lifecycle=False,
        enable_memory=False,
        enable_thinking=False,
        enable_world_model=False,
    )
    learner = ContinualLearner(
        model,
        enable_spgc=True,
        spgc_tau_skip=0.05,
        spgc_min_sources=2,
        spgc_max_bytes=8192,
        spgc_lr=5e-3,
    )
    x = torch.randint(32, 48, (8, 6))
    targets = x.clone()
    out0 = learner.step(x, targets, output_head="byte_chunk", source_id="src_0")
    assert "skipped_backward" in out0 and out0["skipped_backward"] is False
    learner.domain_transition(new_domain_id=1, spawn_new_expert=True)
    assert isinstance(learner.optimizer, MuonBONGOptimizer)
    assert learner.optimizer.mode == "consolidate"
    out1 = learner.step(x, targets, output_head="byte_chunk", source_id="src_1")
    assert "total_loss" in out1


def test_complex_and_thin_gate_muon_guard():
    class _MixedMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Linear(16, 16, bias=False)
            self.level_gate = nn.Linear(16, 3, bias=False)
            self.stm_gate = nn.Linear(16, 1, bias=False)
            self.cplx_w = nn.Parameter(torch.randn(16, 16, dtype=torch.complex64) * 0.1)

    torch.manual_seed(11)
    mod = _MixedMod()
    groups = classify_spgc_param_groups(mod)
    by_name = {g["name"]: g for g in groups}
    muon_ids = {id(p) for p in by_name["trunk_muon"]["params"]}
    vec_ids = {id(p) for p in by_name["trunk_vec"]["params"]}
    assert id(mod.hidden.weight) in muon_ids
    assert id(mod.level_gate.weight) in vec_ids
    assert id(mod.stm_gate.weight) in vec_ids
    assert id(mod.cplx_w) in vec_ids

    # Also test defensive fallback if complex weight is forced into a use_muon=True group
    forced_groups = [
        {"name": "forced", "params": [mod.hidden.weight, mod.cplx_w, mod.stm_gate.weight], "use_muon": True, "role": "trunk", "lr": 1e-2, "lr_consolidate": 1e-2, "weight_decay": 0.01, "prec_scale": 1.0}
    ]
    opt = MuonBONGOptimizer(forced_groups, mode="acquire", track_fisher=True)
    for mode_step in range(4):
        if mode_step == 2:
            opt.set_mode("consolidate", snapshot_anchor=False)
        opt.zero_grad()
        x = torch.randn(4, 16)
        xc = torch.complex(x, torch.zeros_like(x))
        y_cplx = (xc @ mod.cplx_w).abs().pow(2).mean()
        loss = mod.hidden(x).pow(2).mean() + mod.stm_gate(x).pow(2).mean() + y_cplx
        loss.backward()
        opt.step()
    assert torch.isfinite(mod.hidden.weight).all()
    assert torch.isfinite(mod.cplx_w.real).all() and torch.isfinite(mod.cplx_w.imag).all()
    assert not opt.precision[id(mod.cplx_w)].is_complex()


def test_spgc_full_stack_dmoe_memory_hier_thinking_world_model_null_space():
    torch.manual_seed(7)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=2,
        chunk_size=1,
        auto_lifecycle=True,
        enable_memory=True,
        enable_thinking=True,
        enable_world_model=True,
        enable_three_level_attn=True,
        causal_attn=True,
        use_rope=True,
    )
    learner = ContinualLearner(
        model,
        enable_spgc=True,
        enable_null_space=True,
        spgc_tau_skip=0.05,
        spgc_min_sources=2,
        spgc_max_bytes=8192,
        spgc_lr=5e-3,
        spgc_min_flush=2,
        spgc_auto_consolidate_steps=3,
        spgc_auto_consolidate_loss=20.0,
    )
    assert learner.spgc_gate is not None
    assert learner.spgc_gate.store_samples is False
    # Verify thin gates & experts are excluded from trunk_muon
    muon_ids = {id(p) for g in learner.optimizer.param_groups if g.get("use_muon") for p in g["params"]}
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ("stm_gate", "ltm_gate", "level_gate", "salience_head", "moe")):
            assert id(p) not in muon_ids, f"{name} should not be in trunk_muon"

    x = torch.randint(32, 48, (4, 8))
    targets = torch.roll(x, shifts=-1, dims=1)
    assert learner.optimizer.mode == "acquire"
    for st in range(3):
        rep = learner.step(x, targets, output_head="byte_chunk", source_id=f"src_{st % 2}")
        assert rep["skipped_backward"] is False
        assert rep["total_loss"] > 0.0 and torch.isfinite(torch.tensor(rep["total_loss"]))
    assert learner.optimizer.mode == "consolidate"

    # Verify null-space tracking stays paused during screening & domain_transition spawns/preserves state
    prev_n_exp = len(model.layers[0].moe.experts)
    learner.domain_transition(new_domain_id=1, spawn_new_expert=True)
    assert len(model.layers[0].moe.experts) == prev_n_exp + 1
    assert isinstance(learner.optimizer, MuonBONGOptimizer)
    assert learner.optimizer.mode == "consolidate"

    rep_post = learner.step(x, targets, output_head="byte_chunk", source_id="src_new")
    assert torch.isfinite(torch.tensor(rep_post["total_loss"]))
    assert learner.spgc_gate.storage_bytes() <= 8192


def test_spgc_microqueue_mutation_and_continuous_regression():
    # 1. Micro-queue buffering & explicit flush on RealSelfStabilizingDMoE
    torch.manual_seed(13)
    m_real = RealSelfStabilizingDMoE(
        d_model=32, num_heads=4, d_ff=64, initial_layers=1, initial_experts=1, chunk_size=1,
        auto_lifecycle=False, enable_memory=False, enable_thinking=False, enable_world_model=False,
    )
    lr_q = ContinualLearner(
        m_real, enable_spgc=True, spgc_min_flush=4, spgc_max_bytes=8192, spgc_lr=5e-3,
    )
    x_small = torch.randint(32, 48, (2, 6))
    y_small = x_small.clone()
    r0 = lr_q.step(x_small, y_small, output_head="byte_chunk", source_id="s0")
    assert r0["skipped_backward"] is True
    assert len(lr_q.spgc_gate.queue_x) == 2
    assert lr_q.spgc_gate.storage_bytes() <= 8192
    r_flush = lr_q.flush_spgc_queue(output_head="byte_chunk")
    assert r_flush["skipped_backward"] is False
    assert len(lr_q.spgc_gate.queue_x) == 0

    # 2. Periodic execute_graph_mutation + continuous MSE regression head on SelfStabilizingDynamicMorphicMoE
    torch.manual_seed(17)
    m_dyn = SelfStabilizingDynamicMorphicMoE(d_model=32, num_heads=4, d_ff=64, initial_layers=1, initial_experts=2)
    lr_dyn = ContinualLearner(
        m_dyn,
        enable_spgc=True,
        mutation_check_interval=2,
        spgc_tau_skip=1e-4,
        spgc_max_bytes=8192,
        spgc_lr=5e-3,
    )
    xc = torch.randn(4, 3, 16)
    yc = torch.zeros(4, 3, 8)
    out_a = lr_dyn.step(xc, yc, output_head="continuous", source_id="c0")
    out_b = lr_dyn.step(xc, yc, output_head="continuous", source_id="c1")
    assert out_a["skipped_backward"] is False
    assert out_b["mutation_report"] is not None
    assert torch.isfinite(torch.tensor(out_b["total_loss"]))
    assert lr_dyn.spgc_gate.storage_bytes() <= 8192


def test_stale_tentative_entries_expire_and_cannot_be_piggybacked():
    gate = LatentQuarantineGate(
        latent_dim=16,
        seed=5,
        max_bytes=4096,
        min_flip_count=3,
        min_sources=2,
        tent_ttl_steps=10,
        store_samples=False,
    )
    z = torch.ones(4, 16)
    x = torch.zeros(4, 4, dtype=torch.long)
    # Confident prior predicting class 0
    logits_p = torch.zeros(4, 8)
    logits_p[:, 0] = 6.0
    # 1. Early stray contradiction from src_old
    _, _, fl0 = gate.triage_batch(x, torch.full((4,), 3, dtype=torch.long), logits_p, z, source_id="src_old")
    assert fl0 == 0
    # 2. Advance > tent_ttl_steps confirming original belief
    for _ in range(15):
        gate.triage_batch(x, torch.zeros(4, dtype=torch.long), logits_p, z, source_id="src_clean", skip_familiar=False)
    # 3. Single-source burst from attacker should NOT piggyback on expired src_old record
    _, _, fl_burst = gate.triage_batch(
        x, torch.full((4,), 3, dtype=torch.long), logits_p, z, source_id="attacker_single"
    )
    assert fl_burst == 0

    # 4. Familiar samples must still populate prior_counts even when allow_core_write=False & store_samples=False
    gate2 = LatentQuarantineGate(
        latent_dim=16, seed=9, max_bytes=4096, conf_established=0.80, min_sources=2, store_samples=False
    )
    for _ in range(8):
        gate2.triage_batch(
            x, torch.zeros(4, dtype=torch.long), logits_p, z, source_id="src_clean", allow_core_write=False, skip_familiar=True
        )
    assert len(gate2.prior_counts) > 0
    # Now even a low-confidence prior (conf=0.35 < 0.80) should quarantine a single-source burst because prior_cnt >= 3
    logits_low = torch.zeros(4, 8)
    logits_low[:, 0] = 1.2
    x_act, _, fl_low = gate2.triage_batch(
        x, torch.full((4,), 3, dtype=torch.long), logits_low, z, source_id="attacker", allow_core_write=False
    )
    assert x_act is None and fl_low == 0

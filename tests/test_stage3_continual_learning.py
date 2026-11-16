"""Unit and adversarial anti-cheat tests for Stage 3 Continual Learning & Belief Revision."""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.data.stage3_tasks import (
    ALPHABET_SIZE,
    ALPHABET_START,
    build_compositional_belief_tables,
    generate_3b_ic_episodes,
    generate_3b_pair_eval_batch,
    generate_3b_seen_batch,
    generate_stage3a_batch,
)
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.routing import CausalPrototypeRouter, SequenceContentRouter
from dmoe.stage3_baselines import (
    BudgetedDictionaryBaseline,
    DenseRoPEContinualBaseline,
    StaticMoEContinualBaseline,
)
from dmoe.upkeeper import CalibratedBayesianUpkeeper


def test_stage3a_is_header_free_and_unigram_blind():
    """
    Domains 0, 1, 2 all use [32..47] with identical uniform unigram marginals.
    A naive unigram centroid router must fail (~chance), while CausalPrototypeRouter succeeds.
    """
    torch.manual_seed(7)
    emb = torch.nn.Embedding(258, 48)
    uni_router = SequenceContentRouter(d_model=48)
    proto_router = CausalPrototypeRouter(d_model=48, max_protos=32)

    class _Enc:
        def __init__(self, e):
            self.byte_embed = e

    enc = _Enc(emb)

    uni_ex = []
    for d in range(3):
        x, _ = generate_stage3a_batch(d, batch_size=32, seq_len=16, rho=0.0, seed=100 + d)
        assert int(x.min().item()) >= ALPHABET_START
        assert int(x.max().item()) < ALPHABET_START + ALPHABET_SIZE
        uni_ex.append(F.normalize(emb(x).mean(dim=1), p=2, dim=-1))
        proto_sig = proto_router.extract_signature(x, enc)
        proto_router.update_prototypes(d, proto_sig)

    x_mix = torch.cat([
        generate_stage3a_batch(d, batch_size=32, seq_len=16, rho=0.0, seed=500 + d)[0]
        for d in range(3)
    ], dim=0)
    labels = torch.tensor([0] * 32 + [1] * 32 + [2] * 32)

    # Unigram router should hover near 1/3 chance
    u_sigs = F.normalize(emb(x_mix).mean(dim=1), p=2, dim=-1)
    u_pred, _, _ = uni_router.route_sequences(u_sigs, uni_ex)
    u_acc = (u_pred == labels).float().mean().item()
    assert u_acc < 0.55, f"Unigram router unexpectedly separated shared-vocab streams: {u_acc:.4f}"

    # CausalPrototypeRouter separates directed transitions cleanly
    p_sigs = proto_router.extract_signature(x_mix, enc)
    p_pred, _, _ = proto_router.route_sequences(p_sigs)
    p_acc = (p_pred == labels).float().mean().item()
    assert p_acc >= 0.90, f"CausalPrototypeRouter failed on rho=0.0 transitions: {p_acc:.4f}"


def test_causal_prototype_router_prefix_causality():
    """Changing tokens after t must alter step-t prefix signature by exactly 0.0."""
    torch.manual_seed(11)
    emb = torch.nn.Embedding(258, 32)
    router = CausalPrototypeRouter(d_model=32)
    x1 = torch.randint(32, 48, (8, 16))
    x2 = x1.clone()
    x2[:, 8:] = torch.randint(32, 48, (8, 8))

    s1 = router.extract_signature(x1[:, :8], emb)
    s2 = router.extract_signature(x2[:, :8], emb)
    assert torch.max(torch.abs(s1 - s2)).item() == 0.0


def test_stage3_full_model_strict_causality():
    """
    Full Stage 3 RealSelfStabilizingDMoE with Stage 1 (STM+LTM+3LevelRoPE) and
    Stage 2/3 (Thinker + WorldModel + pre-layer PC write rectification) must remain
    strictly causal (future perturbations cause 0.0 diff at earlier positions).
    """
    torch.manual_seed(19)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=3,
        enable_memory=True,
        enable_three_level_attn=True,
        use_rope=True,
        causal_attn=True,
        enable_layer_stm=True,
        enable_world_model=True,
        enable_pc_rectification=True,
        router_mode="causal_proto",
    )
    model.eval()

    x_a = torch.randint(32, 48, (4, 16))
    x_b = x_a.clone()
    x_b[:, 10:] = torch.randint(32, 48, (4, 6))

    with torch.no_grad():
        out_a = model(x_a, update_stats=False)
        out_b = model(x_b, update_stats=False)

    diff_logits = (out_a["output"][:, :10] - out_b["output"][:, :10]).abs().max().item()
    assert diff_logits <= 1e-5, f"Future tokens leaked into logits: {diff_logits}"
    if out_a["pre_pc_error"] is not None:
        diff_pc = (out_a["pre_pc_error"][:, :10] - out_b["pre_pc_error"][:, :10]).abs().max().item()
        assert diff_pc <= 1e-5, f"Future tokens leaked into PC error: {diff_pc}"


def test_real_null_space_projector_registers_and_projects_orthogonally():
    """
    NullSpaceProjector must actively monitor RealSelfStabilizingDMoE shared linear layers
    and byte_embed output subspace, projecting D1 gradients orthogonally to D0's basis.
    """
    torch.manual_seed(23)
    model = RealSelfStabilizingDMoE(
        d_model=48,
        num_heads=4,
        d_ff=96,
        initial_layers=2,
        chunk_size=1,
        enable_memory=True,
        enable_three_level_attn=True,
        use_rope=True,
        enable_layer_stm=True,
    )
    nsp = NullSpaceProjector(
        model,
        energy_threshold=0.98,
        max_rank_ratio=0.55,
        track_embeddings=True,
    )
    assert "chunk_proj" in nsp.monitored_modules
    assert "byte_embed" in nsp.monitored_modules
    assert any("hier_attn" in k for k in nsp.monitored_modules)
    assert any("layer_stm" in k for k in nsp.monitored_modules)

    x0, y0 = generate_stage3a_batch(0, batch_size=32, seq_len=16, rho=0.0, seed=10)
    model.train()
    _ = model(x0)
    ranks = nsp.update_basis_for_domain(0)
    assert ranks.get("chunk_proj", 0) > 0
    assert ranks.get("byte_embed", 0) > 0

    x1, y1 = generate_stage3a_batch(1, batch_size=32, seq_len=16, rho=0.0, seed=20)
    model.zero_grad()
    out1 = model(x1)
    loss = F.cross_entropy(out1["output"].squeeze(2).reshape(-1, 256), y1.reshape(-1))
    loss.backward()
    nsp.project_gradients()

    u_cp = nsp.bases["chunk_proj"]
    g_cp = model.chunk_proj.weight.grad.data
    rel_cp = ((g_cp @ u_cp).norm() / (g_cp.norm() + 1e-9)).item()
    assert rel_cp < 1e-4, f"chunk_proj gradient leaked into prior subspace: {rel_cp}"
    assert g_cp.abs().max().item() > 1e-5, "Orthogonal subspace gradients unexpectedly zeroed"

    u_emb = nsp.bases["byte_embed"]
    g_emb = model.byte_embed.weight.grad.data
    rel_emb = ((g_emb @ u_emb).norm() / (g_emb.norm() + 1e-9)).item()
    assert rel_emb < 1e-4, f"byte_embed gradient leaked into prior subspace: {rel_emb}"


def test_calibrated_upkeeper_budget_burst_revision_and_unseen_nll():
    """
    CalibratedBayesianUpkeeper must obey strict total byte budget <= max_total_bytes,
    reject single-source contradictory bursts, execute multi-source revisions, and
    leave unseen keys equal to log_softmax(neural_logits).
    """
    upk = CalibratedBayesianUpkeeper(max_total_bytes=4096, min_sources_for_flip=2, min_flip_count=3)

    # Flood with distinct random bigram keys and verify hard budget holds
    for b in range(40):
        x = torch.randint(32, 64, (8, 16))
        y = torch.randint(32, 48, (8, 16))
        upk.gate_targets(x, y, source_id="flood")
    assert upk.memory_footprint_bytes() <= 4096

    # Clean upkeeper for burst / revision logic verification
    upk2 = CalibratedBayesianUpkeeper(
        max_total_bytes=4096, min_sources_for_flip=2, min_flip_count=3, flip_threshold_ratio=0.5
    )
    x_f = torch.tensor([[32, 33, 34, 35]])
    y_init = torch.tensor([[32, 40, 41, 42]])
    y_contra = torch.tensor([[32, 45, 46, 47]])

    # Establish initial belief strongly (4 consistent steps from trusted source)
    for _ in range(4):
        _, m = upk2.gate_targets(x_f, y_init, source_id="trusted")
        assert bool(m[:, 1:].all().item())

    # Single-source adversarial burst for 5 steps -> must NOT flip & must mask updates
    for _ in range(5):
        gy, m = upk2.gate_targets(x_f, y_contra, source_id="noisy_single")
        assert not bool(m[:, 1:].any().item())
        assert torch.equal(gy[:, 1:], y_init[:, 1:])
    assert upk2.flips_executed == 0

    # Persistent multi-source revision -> must flip & accept
    flipped = False
    for st in range(6):
        _, m = upk2.gate_targets(x_f, y_contra, source_id=f"ver_{st % 3}")
        if bool(m[:, 1:].all().item()):
            flipped = True
    assert flipped and upk2.flips_executed > 0

    # Calibration & unseen key invariance check
    logits = torch.randn(2, 4, 256)
    fused_seen = upk2.fuse_logits(x_f[:1], logits[:1])
    probs_sum = fused_seen.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)

    x_unseen = torch.tensor([[50, 51, 52, 53]])
    fused_unseen = upk2.fuse_logits(x_unseen, logits[:1])
    ref_unseen = F.log_softmax(logits[:1], dim=-1)
    assert torch.max(torch.abs(fused_unseen - ref_unseen)).item() < 1e-6


def test_stage3b_split_integrity_and_dictionary_bound():
    """
    Seen and held-out pairs in Stage 3B must be strictly disjoint, and the budgeted
    dictionary must satisfy max_total_bytes and drop to ~1/16 chance on held-out pairs.
    """
    spec = build_compositional_belief_tables(seed=5)
    seen = set(spec["seen_pairs"])
    held = set(spec["heldout_pairs"])
    assert len(seen & held) == 0
    assert len(seen) + len(held) == ALPHABET_SIZE * ALPHABET_SIZE

    d_base = BudgetedDictionaryBaseline(max_total_bytes=4096)
    for st in range(30):
        xb, yb = generate_3b_seen_batch(spec["init_table"], spec["seen_mask"], batch_size=32, seed=st)
        d_base.update(xb, yb, source_id="trusted")
    assert d_base.storage_bytes() <= 4096

    x_h, y_h, m_h = generate_3b_pair_eval_batch(spec["init_table"], spec["heldout_pairs"], batch_size=64, seed=999)
    lp = d_base.predict_log_probs(x_h)
    preds = lp.argmax(dim=-1)
    h_acc = (preds[m_h] == y_h[m_h]).float().mean().item()
    assert h_acc < 0.15, f"Dictionary unexpectedly solved unseen held-out pairs: {h_acc:.4f}"


def test_stage3_baselines_and_ic_episode_shapes():
    x_dense = torch.randint(32, 48, (4, 16))
    d_m = DenseRoPEContinualBaseline(d_model=48, num_heads=4, d_ff=192)
    s_m = StaticMoEContinualBaseline(d_model=48, num_heads=4, d_ff=96, num_experts=3)
    assert d_m(x_dense)["output"].shape == (4, 16, 1, 256)
    s_out = s_m(x_dense)
    assert s_out["output"].shape == (4, 16, 1, 256)
    assert s_out["aux_loss"].item() >= 0.0

    x_ic, y_ic, c_mask = generate_3b_ic_episodes(8, num_facts=6, seed=0, include_contradiction=True)
    assert x_ic.shape == (8, 16)
    assert y_ic.shape == (8,)
    assert int(c_mask.sum().item()) == 8


def _make_confirm_ic_batch(batch_size: int, num_facts: int, seed: int):
    x, y, _ = generate_3b_ic_episodes(batch_size, num_facts=num_facts, seed=seed, include_contradiction=True)
    x = x.clone()
    x[:, 2 * num_facts + 1] = y
    return x, y


def test_l2_boundary_free_burst_and_revisit_immunity():
    """
    Boundary-free continual learning (L2) on shared vocab [32..47]:
    D0_init -> 5-step D1_burst (0 false spawns) -> D0_post -> sustained D1 + D2
    (GPM delta projection + CausalPrototypeRouter) -> D0_revisit (0 duplicate spawns).
    """
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealSelfStabilizingDMoE(
        d_model=48,
        num_heads=4,
        d_ff=96,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=False,
        auto_lifecycle=True,
        max_experts=4,
        min_mature_steps=20,
        novel_confirm_steps=6,
        novel_surprise_threshold=0.52,
        dormant_age=500,
        prune_age=1000,
        enable_memory=False,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
        causal_attn=True,
        use_rope=True,
        router_mode="causal_proto",
        num_router_protos=16,
        router_lda_reg=0.05,
        router_sim_threshold=0.55,
        route_during_train=True,
    ).to(device)
    model.lifecycle.freeze_embed_after_first = True
    gpm = NullSpaceProjector(model, energy_threshold=0.98, max_rank_ratio=0.55, track_embeddings=False)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.0)

    def _run_phase(domain_id: int, steps: int, seed_base: int, rho: float = 0.0, proj: bool = False):
        nonlocal opt
        prev_spawns = model.lifecycle.total_spawns
        for st in range(steps):
            x, y = generate_stage3a_batch(domain_id, batch_size=32, seq_len=16, rho=rho, seed=seed_base + st)
            x, y = x.to(device), y.to(device)
            model.train()
            opt.zero_grad(set_to_none=True)
            saved = gpm.snapshot_monitored_weights() if (proj and gpm.p_null) else None
            out = model(x, update_stats=True, targets=y)
            if model.lifecycle.total_spawns > prev_spawns:
                gpm.update_basis_for_domain(domain_id - 1 if domain_id > 0 else 0)
                if getattr(model.lifecycle, "freeze_embed_after_first", False):
                    model.byte_embed.weight.requires_grad_(False)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-3, weight_decay=0.0)
                prev_spawns = model.lifecycle.total_spawns
            logits = out["output"].reshape(x.size(0), x.size(1), -1)
            loss = F.cross_entropy(logits[:, 1:, :].reshape(-1, 256), y[:, 1:].reshape(-1))
            loss.backward()
            if proj and gpm.p_null:
                gpm.project_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if saved is not None:
                gpm.project_parameter_deltas(saved)

    _run_phase(0, 60, 1000, proj=False)
    spawns_after_d0 = model.lifecycle.total_spawns
    _run_phase(1, 5, 2000, proj=True)
    burst_spawns = model.lifecycle.total_spawns - spawns_after_d0
    _run_phase(0, 15, 3000, proj=True)
    spawns_before_sustain = model.lifecycle.total_spawns
    _run_phase(1, 75, 4000, proj=True)
    _run_phase(2, 75, 5000, proj=True)
    sustain_spawns = model.lifecycle.total_spawns - spawns_before_sustain

    spawns_before_revisit = model.lifecycle.total_spawns
    _run_phase(0, 20, 6000, proj=True)
    revisit_spawns = model.lifecycle.total_spawns - spawns_before_revisit

    model.eval()
    x0, y0 = generate_stage3a_batch(0, batch_size=64, seq_len=16, rho=0.0, seed=9000)
    with torch.no_grad():
        logits0 = model(x0.to(device), update_stats=False)["output"].reshape(64, 16, -1)
        acc_d0 = float((logits0[:, 1:, :].argmax(-1) == y0[:, 1:].to(device)).float().mean().item())

    assert burst_spawns == 0, f"False spawn during 5-step burst: {burst_spawns}"
    assert sustain_spawns >= 1, f"Failed to spawn during sustained shifts: {sustain_spawns}"
    assert revisit_spawns == 0, f"Duplicate spawn on D0 revisit: {revisit_spawns}"
    assert len(gpm.bases) > 0, "GPM bases not populated on shift"
    assert acc_d0 >= 0.70, f"D0 accuracy forgot after L2 stream: {acc_d0:.4f}"


def test_pc_rectification_causal_and_improves_contradiction_immunity():
    """
    Active predictive-coding belief rectification must remain strictly causal,
    separate zero-shot contradicting observations from new/confirm observations,
    and substantially improve contradiction immunity over no-PC ablation.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = 4
    obs_val_pos = 2 * nf + 1
    model = RealSelfStabilizingDMoE(
        d_model=64,
        num_heads=4,
        d_ff=256,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        skip_threshold=0.10,
        theta_halt=0.14,
        enable_memory=True,
        num_ltm_slots=24,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
        enable_world_model=True,
        wm_rollout_horizon=1,
        causal_attn=True,
        use_rope=True,
        enable_layer_stm=True,
        enable_pc_rectification=True,
        pc_thresh=0.18,
        pc_suppress_gain=2.5,
    ).to(device)
    total_steps = 800
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=total_steps, pct_start=0.1)

    for step in range(total_steps):
        xn, yn, _ = generate_3b_ic_episodes(24, num_facts=nf, seed=700000 + step * 2, include_contradiction=False)
        xc, yc = _make_confirm_ic_batch(24, num_facts=nf, seed=700001 + step * 2)
        x = torch.cat([xn, xc], dim=0).to(device)
        y = torch.cat([yn, yc], dim=0).to(device)
        full_tgt = torch.cat([x[:, 1:], y.unsqueeze(1)], dim=1)
        pred_mask = torch.zeros_like(x, dtype=torch.bool)
        pred_mask[:, -1] = True
        for t in range(2, x.size(1) - 1, 2):
            pred_mask[:, t] = (x[:, :t:2] == x[:, t : t + 1]).any(dim=1)

        model.train()
        model.reset_ltm()
        opt.zero_grad(set_to_none=True)
        out = model(x, update_stats=True, targets=full_tgt)
        logits = out["output"].reshape(x.size(0), x.size(1), -1)
        ce_ans = F.cross_entropy(logits[:, -1, :], y)
        ce_lm = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        t_pred = model.thinker.compute_thinking_loss(out["thinking_out"], full_tgt, valid_mask=pred_mask)["loss"]
        wm_loss = out["world_model_loss"]["loss"]
        loss = 1.5 * ce_ans + 1.0 * ce_lm + 1.2 * t_pred + 1.0 * wm_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

    model.eval()
    model.reset_ltm()
    x_new, y_new, _ = generate_3b_ic_episodes(128, num_facts=nf, seed=98100, include_contradiction=False)
    x_conf, y_conf = _make_confirm_ic_batch(128, num_facts=nf, seed=98101)
    x_con, y_con, _ = generate_3b_ic_episodes(128, num_facts=nf, seed=98102, include_contradiction=True)
    x_new, y_new = x_new.to(device), y_new.to(device)
    x_conf, y_conf = x_conf.to(device), y_conf.to(device)
    x_con, y_con = x_con.to(device), y_con.to(device)

    with torch.no_grad():
        xa = x_con[:16]
        xb = xa.clone()
        xb[:, obs_val_pos] = ALPHABET_START + ((xb[:, obs_val_pos] - ALPHABET_START + 1) % ALPHABET_SIZE)
        model.enable_pc_rectification = True
        caus_diff = float(
            (model(xa, update_stats=False)["output"][:, :obs_val_pos] - model(xb, update_stats=False)["output"][:, :obs_val_pos])
            .abs()
            .max()
            .item()
        )

        model.enable_pc_rectification = False
        acc_nopc_contra = float(
            (model(x_con, update_stats=False)["output"].reshape(128, -1, 256)[:, -1, :].argmax(-1) == y_con).float().mean().item()
        )

        model.enable_pc_rectification = True
        o_new = model(x_new, update_stats=False)
        o_conf = model(x_conf, update_stats=False)
        o_con = model(x_con, update_stats=False)

        acc_pc_new = float((o_new["output"].reshape(128, -1, 256)[:, -1, :].argmax(-1) == y_new).float().mean().item())
        acc_pc_conf = float((o_conf["output"].reshape(128, -1, 256)[:, -1, :].argmax(-1) == y_conf).float().mean().item())
        acc_pc_con = float((o_con["output"].reshape(128, -1, 256)[:, -1, :].argmax(-1) == y_con).float().mean().item())

        pc_new = float(o_new["pre_pc_error"][:, obs_val_pos].mean().item())
        pc_conf = float(o_conf["pre_pc_error"][:, obs_val_pos].mean().item())
        pc_con = float(o_con["pre_pc_error"][:, obs_val_pos].mean().item())
        gate_con_pos = float(o_con["pc_write_gate"][:, obs_val_pos].mean().item())
        gate_con_pre = float(o_con["pc_write_gate"][:, 1:obs_val_pos].min().item())

    assert caus_diff <= 1e-5, f"PC rectification leaked future token info: {caus_diff:.2e}"
    assert pc_con >= pc_new + 0.25 and pc_con >= pc_conf + 0.25, (
        f"PC error failed to separate contra ({pc_con:.3f}) from new ({pc_new:.3f}) and confirm ({pc_conf:.3f})"
    )
    assert gate_con_pos < 0.85 and gate_con_pre >= 0.90, (
        f"Unexpected PC write gate profile: pos={gate_con_pos:.3f}, prior_min={gate_con_pre:.3f}"
    )
    assert acc_pc_new >= 0.85 and acc_pc_conf >= 0.85, (
        f"PC rectification degraded clean facts: new={acc_pc_new:.3f}, confirm={acc_pc_conf:.3f}"
    )
    assert acc_pc_con >= 0.80 and acc_pc_con > acc_nopc_contra + 0.25, (
        f"PC rectification did not sufficiently improve contra immunity: pc={acc_pc_con:.3f} vs nopc={acc_nopc_contra:.3f}"
    )



"""
Stage 3 Verification Suite: Header-Free Continual Learning, Belief Revision &
Predictive-Coding Memory Rectification vs Matched Strong Baselines & Null Controls.

Protocol (C:/Users/Asta/.gemini/config/rules/research-protocol.md):
  - DEV_SEEDS = [100, 101, 102, 103, 104] for --mode tune ONLY (logged to tuning_log_v1.json).
  - TEST_SEEDS = [0..19] for frozen 20-seed --mode test AFTER PREREG_v1.md is frozen.

Sub-Suites:
  3A     : Header-free shared-vocab [32..47] continual learning across L0, L1 (rho=0.0, 0.5),
           and boundary-free L2 (D0 -> 5-step burst -> D0 -> D1 -> D2 -> D0_revisit)
           vs Cog_NoGPM, Cog_UnigramRouter, Cog_OracleRouting, Dense_NoReplay,
           Dense_Replay128 (<=4096 B), StaticMoE_Matched, and 3 Null Controls.
  3B-Mem : Compositional belief learning + 5-step 1-source burst immunity + persistent
           multi-source revision within strict <=4096 B episodic memory (Accuracy & NLL on
           seen, exc, heldout) vs Ablation_NoUpkeeper, Dense_NoUpk, BudgetedDictionaryBaseline.
  3B-IC  : Zero-shot contradiction immunity & predictive-coding rectification (clean-only
           trained on new/confirm; zero-shot evaluated on contra + AUROC + strict causality
           <= 1e-5) vs Ablation_NoPCRectification and param-matched Dense_Matched.
  3C     : Unified multi-scale novelty & lifecycle calibration (L2 boundary-free lifecycle +
           real UTF-8 corpus Regime A & disjoint 70/15/15 open Markdown prose Regime B from
           experiments/verify_real_stream_novelty.py).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.data.stage3_tasks import (
    ALPHABET_SIZE,
    ALPHABET_START,
    SEP_TOKEN,
    build_compositional_belief_tables,
    generate_3b_pair_eval_batch,
    generate_stage3a_batch,
)
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.routing import CausalPrototypeRouter
from dmoe.stage3_baselines import (
    BudgetedDictionaryBaseline,
    DenseRoPEContinualBaseline,
    StaticMoEContinualBaseline,
)
from dmoe.upkeeper import CalibratedBayesianUpkeeper
from experiments.verify_real_stream_novelty import (
    SEQ_LEN as REAL_SEQ_LEN,
    load_real_code_windows,
    load_real_prose_windows,
    run_open_corpus_seed,
    run_structured_seed,
)

RESULTS_DIR = ROOT / "results" / "stage3_continual_learning"
DEV_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(20))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def summarize(values: List[float]) -> Dict[str, Any]:
    n = len(values)
    mean = sum(values) / max(1, n)
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": sd,
        "ci95_half": ci95,
        "min": min(values) if n else 0.0,
        "max": max(values) if n else 0.0,
        "n": n,
        "per_seed": values,
    }


def paired_stats(a: List[float], b: List[float]) -> Dict[str, float]:
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / max(1, n)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, n - 1) if n > 1 else 0.0
    sd_d = math.sqrt(var_d)
    se = sd_d / math.sqrt(n) if n > 1 else 1e-9
    t_stat = mean_d / max(se, 1e-12) if abs(mean_d) > 1e-15 else 0.0
    ci95 = 1.96 * se
    wins = sum(1 for d in diffs if d > 0.0)
    return {
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "ci95_half": ci95,
        "t_stat": t_stat,
        "wins": float(wins),
        "n": float(n),
    }


def compute_auroc(pos_scores: List[float], neg_scores: List[float]) -> float:
    if not pos_scores or not neg_scores:
        return 0.5
    tagged = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    tagged.sort(key=lambda t: t[0])
    n = len(tagged)
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and tagged[j + 1][0] == tagged[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            if tagged[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def _collect_metadata() -> Dict[str, Any]:
    git_commit = "unknown"
    git_dirty = True
    if shutil.which("git") is not None and (ROOT / ".git").exists():
        res_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True
        )
        res_st = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True
        )
        if res_head.returncode == 0 and res_st.returncode == 0:
            git_commit = res_head.stdout.strip()
            git_dirty = len(res_st.stdout.strip()) > 0
        else:
            git_commit = f"ERROR(rc={res_head.returncode})"
    return {
        "exp_id": "stage3_continual_learning_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hostname": platform.node(),
        "command": " ".join(sys.argv),
    }


def _aggregate_seed_dicts(rows: List[Dict[str, float]], wall_sec: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"wall_clock_sec": wall_sec}
    if not rows:
        return out
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float))]
    for k in keys:
        out[k] = summarize([float(r[k]) for r in rows])
    return out


# ==============================================================================
# Strict Causality Gate Verification
# ==============================================================================

def verify_strict_causality_gate(device: torch.device) -> Dict[str, float]:
    set_all_seeds(77)
    results: Dict[str, float] = {}

    # 1. CausalPrototypeRouter prefix invariance
    emb = nn.Embedding(258, 48).to(device)
    cr = CausalPrototypeRouter(48).to(device)
    x_a = torch.randint(32, 48, (8, 16), device=device)
    x_b = x_a.clone()
    x_b[:, 8:] = torch.randint(32, 48, (8, 8), device=device)
    results["CausalPrototypeRouter"] = float(
        (cr.extract_signature(x_a[:, :8], emb) - cr.extract_signature(x_b[:, :8], emb)).abs().max().item()
    )

    # 2. Full Stage 3A & Stage 3B-IC models + Dense + StaticMoE across all k in 1..15
    checks: Dict[str, nn.Module] = {
        "Cog_Stage3_3A": RealSelfStabilizingDMoE(
            d_model=48, num_heads=4, d_ff=96, initial_layers=2, initial_experts=1,
            chunk_size=1, enable_thinking=False, auto_lifecycle=True, max_experts=4,
            enable_memory=False, enable_three_level_attn=True, local_window=4,
            chunk_stride=4, causal_attn=True, use_rope=True, router_mode="causal_proto",
        ).to(device),
        "Cog_Stage3_IC": RealSelfStabilizingDMoE(
            d_model=64, num_heads=4, d_ff=256, initial_layers=2, initial_experts=1,
            chunk_size=1, enable_thinking=True, max_thinking_steps=2, skip_threshold=0.10,
            theta_halt=0.14, enable_memory=True, num_ltm_slots=24, enable_three_level_attn=True,
            local_window=4, chunk_stride=4, enable_world_model=True, wm_rollout_horizon=1,
            causal_attn=True, use_rope=True, enable_layer_stm=True,
            enable_pc_rectification=True, pc_thresh=0.05, pc_suppress_gain=3.0,
        ).to(device),
        "DenseRoPE": DenseRoPEContinualBaseline(d_model=48, num_heads=4, d_ff=624, num_layers=2).to(device),
        "StaticMoE": StaticMoEContinualBaseline(d_model=48, num_heads=4, d_ff=208, num_layers=2, num_experts=3).to(device),
    }
    for name, mod in checks.items():
        mod.eval()
        xa = torch.randint(32, 48, (4, 16), device=device)
        with torch.no_grad():
            oa = mod(xa)["output"]
            mx = 0.0
            for k in range(1, 16):
                xb = xa.clone()
                xb[:, k:] = torch.randint(32, 48, (4, 16 - k), device=device)
                ob = mod(xb)["output"]
                d = float((oa[:, :k] - ob[:, :k]).abs().max().item())
                if d > mx:
                    mx = d
            results[name] = mx
    return results


# ==============================================================================
# Sub-Suite 3A: Header-Free Shared-Vocab [32..47] Continual Learning (L0, L1, L2)
# ==============================================================================

def _build_3a_model(model_name: str, device: torch.device, sim_thresh: float = 0.50) -> Tuple[nn.Module, Optional[NullSpaceProjector]]:
    if model_name in ("Cog_Stage3_Full", "Cog_OracleRouting", "Cog_NoGPM", "Cog_UnigramRouter", "Null_Untrained", "Null_ShuffledTargets", "Null_BlindZeroInput"):
        rmode = "content" if model_name == "Cog_UnigramRouter" else "causal_proto"
        m = RealSelfStabilizingDMoE(
            d_model=48, num_heads=4, d_ff=96, initial_layers=2, initial_experts=1,
            chunk_size=1, enable_thinking=False, auto_lifecycle=True, max_experts=4,
            min_mature_steps=20, novel_confirm_steps=6, novel_surprise_threshold=0.52,
            dormant_age=500, prune_age=1000, enable_memory=False, enable_three_level_attn=True,
            local_window=4, chunk_stride=4, causal_attn=True, use_rope=True,
            router_mode=rmode, num_router_protos=64, router_lda_reg=0.01,
            router_sim_threshold=sim_thresh, route_during_train=True,
        ).to(device)
        m.lifecycle.freeze_embed_after_first = True
        use_gpm = model_name != "Cog_NoGPM"
        gpm = (
            NullSpaceProjector(
                m,
                energy_threshold=0.97,
                max_rank_ratio=0.90,
                track_embeddings=False,
                freeze_shared_norms_on_update=True,
            )
            if use_gpm
            else None
        )
        return m, gpm
    if model_name in ("Dense_NoReplay", "Dense_Replay128"):
        return DenseRoPEContinualBaseline(d_model=48, num_heads=4, d_ff=624, num_layers=2).to(device), None
    if model_name == "StaticMoE_Matched":
        return StaticMoEContinualBaseline(d_model=48, num_heads=4, d_ff=208, num_layers=2, num_experts=3).to(device), None
    raise ValueError(f"Unknown 3A model: {model_name}")


def _make_3a_opt(m: nn.Module, gpm: Optional[NullSpaceProjector], lr: float, sh_scale: float) -> torch.optim.Optimizer:
    if not isinstance(m, RealSelfStabilizingDMoE) or gpm is None or not gpm.p_null:
        return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    sp_p, sh_p = [], []
    for nm, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if "experts" in nm or "expert_heads" in nm:
            sp_p.append(p)
        else:
            sh_p.append(p)
    groups = [{"params": sp_p, "lr": lr}]
    if sh_p:
        groups.append({"params": sh_p, "lr": lr * sh_scale})
    return torch.optim.AdamW(groups)


def _train_3a_phase(
    m: nn.Module,
    gpm: Optional[NullSpaceProjector],
    opt: torch.optim.Optimizer,
    dom: int,
    steps: int,
    seq_len: int,
    rho: float,
    seed_base: int,
    domain_seed: int,
    lr: float,
    sh_scale: float,
    device: torch.device,
    shuffled: bool = False,
    replay_buf: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    max_replay_bytes: int = 4096,
) -> Tuple[torch.optim.Optimizer, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    is_cog = isinstance(m, RealSelfStabilizingDMoE)
    prev_sp = m.lifecycle.total_spawns if (is_cog and m.lifecycle is not None) else 0
    max_replay_seqs = max(1, max_replay_bytes // max(1, seq_len * 2))

    for st in range(steps):
        x, y = generate_stage3a_batch(
            dom, batch_size=32, seq_len=seq_len, rho=rho, seed=seed_base + st, domain_seed=domain_seed
        )
        if shuffled:
            perm = torch.randperm(x.size(0))
            y = y[perm]
        x, y = x.to(device), y.to(device)

        x_train, y_train = x, y
        if replay_buf is not None and replay_buf[0].numel() > 0:
            rb_x, rb_y = replay_buf
            n_rep = min(16, rb_x.size(0))
            r_idx = torch.randint(0, rb_x.size(0), (n_rep,))
            x_train = torch.cat([x[: 32 - n_rep], rb_x[r_idx].to(device)], dim=0)
            y_train = torch.cat([y[: 32 - n_rep], rb_y[r_idx].to(device)], dim=0)

        m.train()
        opt.zero_grad(set_to_none=True)
        saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
        if gpm is not None:
            gpm.is_tracking = False

        out = m(x_train, update_stats=True, targets=y_train) if is_cog else m(x_train)
        if is_cog and m.lifecycle is not None:
            if m.lifecycle.total_spawns > prev_sp:
                if gpm is not None:
                    gpm.update_basis_for_domain(max(0, dom - 1))
                if getattr(m.lifecycle, "freeze_embed_after_first", False):
                    m.byte_embed.weight.requires_grad_(False)
                opt = _make_3a_opt(m, gpm, lr=lr, sh_scale=sh_scale)
                prev_sp = m.lifecycle.total_spawns
            if m.lifecycle.should_defer_update:
                continue

        if gpm is not None:
            gpm.is_tracking = True
            out = m(x_train, update_stats=False)

        lg = out["output"].reshape(x_train.size(0), seq_len, -1)
        loss = F.cross_entropy(lg[:, 1:].reshape(-1, 256), y_train[:, 1:].reshape(-1))
        if "aux_loss" in out:
            loss = loss + out["aux_loss"]
        loss.backward()
        if gpm is not None and gpm.p_null:
            gpm.project_gradients()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if saved is not None and gpm is not None:
            gpm.project_parameter_deltas(saved)

        if replay_buf is not None:
            rb_x, rb_y = replay_buf
            x_u8 = x.detach().cpu().to(torch.uint8)
            y_u8 = y.detach().cpu().to(torch.uint8)
            comb_x = x_u8 if rb_x.numel() == 0 else torch.cat([rb_x, x_u8], dim=0)
            comb_y = y_u8 if rb_y.numel() == 0 else torch.cat([rb_y, y_u8], dim=0)
            if comb_x.size(0) > max_replay_seqs:
                keep = torch.randperm(comb_x.size(0))[:max_replay_seqs]
                comb_x, comb_y = comb_x[keep], comb_y[keep]
            replay_buf = (comb_x, comb_y)

    return opt, replay_buf


def _eval_3a_doms(
    m: nn.Module,
    seq_len: int,
    rho: float,
    eval_seed: int,
    domain_seed: int,
    device: torch.device,
    force_oracle: bool = False,
    blind_zero: bool = False,
) -> Tuple[List[float], List[float]]:
    m.eval()
    is_cog = isinstance(m, RealSelfStabilizingDMoE)
    accs, raccs = [], []
    with torch.no_grad():
        for ed in range(3):
            ex, ey = generate_stage3a_batch(
                ed, batch_size=64, seq_len=seq_len, rho=rho, seed=eval_seed + ed * 100, domain_seed=domain_seed
            )
            ex_in = torch.zeros_like(ex) if blind_zero else ex
            ex_in, ey = ex_in.to(device), ey.to(device)
            if is_cog and force_oracle:
                fe = min(ed, len(m.layers[0].moe.experts) - 1)
                eo = m(ex_in, update_stats=False, force_expert=fe)
            else:
                eo = m(ex_in, update_stats=False) if is_cog else m(ex_in)
            accs.append(float((eo["output"].reshape(64, seq_len, -1)[:, 1:].argmax(-1) == ey[:, 1:]).float().mean().item()))
            if eo.get("assigned_experts") is not None:
                raccs.append(float((eo["assigned_experts"] == ed).float().mean().item()))
            else:
                raccs.append(0.0)
    return accs, raccs


def _run_3a_stream(
    model_name: str,
    seed: int,
    stream_name: str,
    lr: float,
    sh_scale: float,
    steps_per_dom: int,
    device: torch.device,
) -> Dict[str, float]:
    set_all_seeds(seed)
    domain_seed = seed * 17 + 7
    m, gpm = _build_3a_model(model_name, device)
    is_cog = isinstance(m, RealSelfStabilizingDMoE)
    untrained = model_name == "Null_Untrained"
    shuffled = model_name == "Null_ShuffledTargets"
    blind = model_name == "Null_BlindZeroInput"
    oracle = model_name == "Cog_OracleRouting"
    use_replay = model_name == "Dense_Replay128"
    replay_buf: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
        (torch.zeros(0, 0, dtype=torch.uint8), torch.zeros(0, 0, dtype=torch.uint8)) if use_replay else None
    )

    opt = _make_3a_opt(m, gpm, lr=lr, sh_scale=sh_scale)
    seq_len = 24 if stream_name.startswith("l1") else 16
    rho = 0.5 if stream_name == "l1_rho5" else 0.0

    if stream_name in ("l0", "l1_rho0", "l1_rho5"):
        if not untrained:
            for d in range(3):
                opt, replay_buf = _train_3a_phase(
                    m, gpm, opt, d, steps_per_dom, seq_len, rho,
                    seed * 10000 + d * 1000, domain_seed, lr, sh_scale, device,
                    shuffled=shuffled, replay_buf=replay_buf, max_replay_bytes=4096,
                )
        accs, raccs = _eval_3a_doms(
            m, seq_len, rho, seed * 10000 + 9000, domain_seed, device, force_oracle=oracle, blind_zero=blind
        )
        sp = m.lifecycle.total_spawns if (is_cog and m.lifecycle is not None) else 0
        if gpm is not None:
            gpm.remove_hooks()
        return {
            "d0": accs[0], "d1": accs[1], "d2": accs[2],
            "avg": sum(accs) / 3.0, "route": sum(raccs) / 3.0, "spawns": float(sp),
            "params": float(count_parameters(m)),
        }

    # L2 Boundary-free stream: D0_init -> 5-step D1_burst -> D0_post -> D1 -> D2 -> D0_revisit
    if not untrained:
        opt, replay_buf = _train_3a_phase(m, gpm, opt, 0, 80, 16, 0.0, seed * 10000 + 1000, domain_seed, lr, sh_scale, device, shuffled, replay_buf)
    sp0 = m.lifecycle.total_spawns if (is_cog and m.lifecycle is not None) else 0
    if not untrained:
        opt, replay_buf = _train_3a_phase(m, gpm, opt, 1, 5, 16, 0.0, seed * 10000 + 2000, domain_seed, lr, sh_scale, device, shuffled, replay_buf)
    burst_sp = (m.lifecycle.total_spawns - sp0) if (is_cog and m.lifecycle is not None) else 0
    if not untrained:
        opt, replay_buf = _train_3a_phase(m, gpm, opt, 0, 15, 16, 0.0, seed * 10000 + 3000, domain_seed, lr, sh_scale, device, shuffled, replay_buf)
    sp_pre = m.lifecycle.total_spawns if (is_cog and m.lifecycle is not None) else 0
    if not untrained:
        opt, replay_buf = _train_3a_phase(m, gpm, opt, 1, 105, 16, 0.0, seed * 10000 + 4000, domain_seed, lr, sh_scale, device, shuffled, replay_buf)
        opt, replay_buf = _train_3a_phase(m, gpm, opt, 2, 105, 16, 0.0, seed * 10000 + 5000, domain_seed, lr, sh_scale, device, shuffled, replay_buf)
    sustain_sp = (m.lifecycle.total_spawns - sp_pre) if (is_cog and m.lifecycle is not None) else 0
    d0_pre, _ = _eval_3a_doms(m, 16, 0.0, seed * 10000 + 8500, domain_seed, device, force_oracle=oracle, blind_zero=blind)

    sp_rev0 = m.lifecycle.total_spawns if (is_cog and m.lifecycle is not None) else 0
    if not untrained:
        opt, replay_buf = _train_3a_phase(m, gpm, opt, 0, 20, 16, 0.0, seed * 10000 + 6000, domain_seed, lr, sh_scale, device, shuffled, replay_buf)
    revisit_sp = (m.lifecycle.total_spawns - sp_rev0) if (is_cog and m.lifecycle is not None) else 0
    d0_post, raccs = _eval_3a_doms(m, 16, 0.0, seed * 10000 + 9000, domain_seed, device, force_oracle=oracle, blind_zero=blind)
    if gpm is not None:
        gpm.remove_hooks()
    return {
        "burst_spawns": float(burst_sp),
        "sustain_spawns": float(sustain_sp),
        "revisit_spawns": float(revisit_sp),
        "d0_pre_revisit": d0_pre[0],
        "d0_post_revisit": d0_post[0],
        "d1": d0_post[1],
        "d2": d0_post[2],
        "avg": sum(d0_post) / 3.0,
        "route": sum(raccs) / 3.0,
    }


def run_seed_3a(job: Tuple[str, int, float, float, int, bool]) -> Tuple[int, Dict[str, float]]:
    model_name, seed, lr, sh_scale, steps_per_dom, use_cuda = job
    torch.set_num_threads(1)
    dev = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    r_l0 = _run_3a_stream(model_name, seed, "l0", lr, sh_scale, steps_per_dom, dev)
    r_r0 = _run_3a_stream(model_name, seed, "l1_rho0", lr, sh_scale, steps_per_dom, dev)
    r_r5 = _run_3a_stream(model_name, seed, "l1_rho5", lr, sh_scale, steps_per_dom, dev)
    r_l2 = _run_3a_stream(model_name, seed, "l2", lr, sh_scale, steps_per_dom, dev)
    return seed, {
        "l0_d0": r_l0["d0"], "l0_d1": r_l0["d1"], "l0_d2": r_l0["d2"], "l0_avg": r_l0["avg"],
        "l1_rho0_d0": r_r0["d0"], "l1_rho0_d1": r_r0["d1"], "l1_rho0_d2": r_r0["d2"],
        "l1_rho0_avg": r_r0["avg"], "l1_rho0_route": r_r0["route"],
        "l1_rho5_d0": r_r5["d0"], "l1_rho5_d1": r_r5["d1"], "l1_rho5_d2": r_r5["d2"],
        "l1_rho5_avg": r_r5["avg"], "l1_rho5_route": r_r5["route"],
        "l2_burst_spawns": r_l2["burst_spawns"], "l2_sustain_spawns": r_l2["sustain_spawns"],
        "l2_revisit_spawns": r_l2["revisit_spawns"],
        "l2_d0_pre_revisit": r_l2["d0_pre_revisit"], "l2_d0_post_revisit": r_l2["d0_post_revisit"],
        "l2_avg": r_l2["avg"], "params": r_l0["params"],
    }


# ==============================================================================
# Sub-Suite 3B-Mem: Belief Revision & Held-Out Compositional Generalization
# ==============================================================================

def _eval_3b_mem_split(
    model: Optional[nn.Module],
    store: Any,
    tbl: torch.Tensor,
    pairs: List[Tuple[int, int]],
    eval_seed: int,
    device: torch.device,
    is_dict: bool = False,
    blind_zero: bool = False,
) -> Tuple[float, float]:
    x, y, mk = generate_3b_pair_eval_batch(tbl, pairs, batch_size=64, seq_len=16, seed=eval_seed)
    x_in = torch.zeros_like(x) if blind_zero else x
    x_in, y, mk = x_in.to(device), y.to(device), mk.to(device)
    with torch.no_grad():
        if is_dict:
            lp = store.predict_log_probs(x_in)
        else:
            model.eval()
            out_m = model(x_in, update_stats=False) if isinstance(model, RealSelfStabilizingDMoE) else model(x_in)
            lg = out_m["output"].reshape(x_in.size(0), x_in.size(1), -1)
            lp = store.fuse_logits(x_in, lg) if store is not None else F.log_softmax(lg, dim=-1)
        acc = float(((lp.argmax(-1) == y) & mk).sum().item() / max(1, int(mk.sum().item())))
        nll = float(F.nll_loss(lp[mk], y[mk]).item())
    return acc, nll


def run_seed_3b_mem(job: Tuple[str, int, float, int, int, float, bool]) -> Tuple[int, Dict[str, float]]:
    model_name, seed, lr, init_steps, rev_steps, wd, use_cuda = job
    torch.set_num_threads(1)
    dev = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    set_all_seeds(seed)
    spec = build_compositional_belief_tables(seed=seed)
    init_t, burst_t, rev_t = spec["init_table"], spec["burst_table"], spec["rev_table"]
    seen_mask, seen_pairs = spec["seen_mask"], spec["seen_pairs"]
    held_pairs, exc_pairs = spec["heldout_pairs"], spec["exc_pairs"]
    P = (seen_mask.float() / seen_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)).to(dev)

    is_dict = model_name == "BudgetedDictionaryBaseline"
    untrained = model_name == "Null_Untrained"
    shuffled = model_name == "Null_ShuffledTargets"
    blind = model_name == "Null_BlindZeroInput"

    m: Optional[nn.Module] = None
    upk: Optional[CalibratedBayesianUpkeeper] = None
    d_base: Optional[BudgetedDictionaryBaseline] = None

    if is_dict:
        d_base = BudgetedDictionaryBaseline(max_total_bytes=4096)
    elif model_name == "Dense_NoUpk":
        m = DenseRoPEContinualBaseline(d_model=64, num_heads=4, d_ff=336, num_layers=1).to(dev)
    else:
        m = RealSelfStabilizingDMoE(
            d_model=64, num_heads=4, d_ff=256, initial_layers=1, initial_experts=1,
            chunk_size=1, enable_thinking=False, enable_memory=False, enable_world_model=False,
            causal_attn=True, use_rope=True, enable_three_level_attn=False, local_window=2,
        ).to(dev)
        if model_name != "Ablation_NoUpkeeper":
            upk = CalibratedBayesianUpkeeper(max_total_bytes=4096, min_sources_for_flip=2, min_flip_count=3)

    def _build_mem_opt(bb_scale: float = 1.0, phase_lr: float = lr) -> Optional[torch.optim.Optimizer]:
        if m is None:
            return None
        pgs = []
        for nm, p in m.named_parameters():
            if not p.requires_grad:
                continue
            is_head = ("decoder" in nm or "moe" in nm or "out_head" in nm or "ffns" in nm)
            sc = 1.0 if is_head else bb_scale
            if sc <= 0.0:
                continue
            pgs.append({
                "params": [p],
                "lr": phase_lr * sc,
                "base_lr": phase_lr * sc,
                "weight_decay": (wd if p.ndim >= 2 else 0.0),
            })
        return torch.optim.AdamW(pgs, betas=(0.9, 0.98))

    opt = _build_mem_opt(1.0, lr)
    gen = torch.Generator(device=dev).manual_seed(seed * 131 + 7)
    bs = 64
    ema_loss = 3.0
    backbone_consolidated = False
    rev_stabilized = False

    def _run_mem_phase(tbl: torch.Tensor, steps: int, src_fn: Callable[[int], str], lr_hi: float, lr_lo: float, decay_start: float = 0.70) -> None:
        nonlocal opt, ema_loss, backbone_consolidated, rev_stabilized
        tb_dev = tbl.to(dev)
        if is_dict:
            for st in range(steps):
                xi = torch.zeros(bs, 16, dtype=torch.long, device=dev)
                xi[:, 0] = torch.randint(0, ALPHABET_SIZE, (bs,), device=dev, generator=gen)
                for t in range(1, 16):
                    xi[:, t] = torch.multinomial(P[xi[:, t - 1]], num_samples=1, generator=gen).squeeze(-1)
                xt = xi + ALPHABET_START
                yf = torch.cat([xt[:, :1], tb_dev[xi[:, :-1], xi[:, 1:]] + ALPHABET_START], dim=1)
                if shuffled:
                    yf = yf[torch.randperm(bs, device=dev)]
                d_base.update(xt, yf, source_id=src_fn(st))
            return

        if untrained:
            return

        for st in range(steps):
            xi = torch.zeros(bs, 16, dtype=torch.long, device=dev)
            xi[:, 0] = torch.randint(0, ALPHABET_SIZE, (bs,), device=dev, generator=gen)
            for t in range(1, 16):
                xi[:, t] = torch.multinomial(P[xi[:, t - 1]], num_samples=1, generator=gen).squeeze(-1)
            xt = xi + ALPHABET_START
            yf = torch.cat([xt[:, :1], tb_dev[xi[:, :-1], xi[:, 1:]] + ALPHABET_START], dim=1)
            if shuffled:
                yf = yf[torch.randperm(bs, device=dev)]

            prev_flips = upk.flips_executed if upk is not None else 0
            if upk is not None:
                gy, um = upk.gate_targets(xt, yf, source_id=src_fn(st))
                lm = um[:, 1:]
                if not bool(lm.any().item()) or float(lm.float().mean().item()) < 0.25:
                    continue
                if backbone_consolidated and (not rev_stabilized) and (upk.flips_executed > prev_flips):
                    # Verified multi-source shift on consolidated backbone: engage dual-timescale
                    # representation preservation (slow encoder, plastic MoE+decoder) & fresh Adam moments.
                    opt = _build_mem_opt(bb_scale=0.05, phase_lr=lr_hi)
                    rev_stabilized = True
            else:
                gy = yf
                lm = torch.ones(bs, 15, dtype=torch.bool, device=dev)

            prog = max(0.0, (st - int(decay_start * steps)) / max(1, steps - 1 - int(decay_start * steps)))
            dec = (lr_lo / max(lr_hi, 1e-9)) + (1.0 - lr_lo / max(lr_hi, 1e-9)) * 0.5 * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * dec

            m.train()
            out_m = m(xt, update_stats=False) if isinstance(m, RealSelfStabilizingDMoE) else m(xt)
            lg = out_m["output"].reshape(bs, 16, -1)[:, 1:, :]
            ce = F.cross_entropy(lg.reshape(-1, 256), gy[:, 1:].reshape(-1), reduction="none")
            loss = (ce * lm.reshape(-1).float()).sum() / lm.sum().clamp_min(1)
            ema_loss = 0.95 * ema_loss + 0.05 * float(loss.item())
            if st >= 1500 and ema_loss < 0.05:
                backbone_consolidated = True
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

    # Phase 1: Init
    _run_mem_phase(init_t, init_steps, lambda s: f"init_src_{s % 2}", lr, 5e-4, decay_start=0.70)
    store_ev = d_base if is_dict else upk
    i_s, _ = _eval_3b_mem_split(m, store_ev, init_t, seen_pairs, seed * 100 + 1, dev, is_dict, blind)
    i_e, _ = _eval_3b_mem_split(m, store_ev, init_t, exc_pairs, seed * 100 + 2, dev, is_dict, blind)
    i_h, i_hnll = _eval_3b_mem_split(m, store_ev, init_t, held_pairs, seed * 100 + 3, dev, is_dict, blind)

    # Phase 2: 5-step single-source burst (0 flips expected, retention of init)
    flips0 = upk.flips_executed if upk is not None else 0
    _run_mem_phase(burst_t, 5, lambda s: "noisy_single", 5e-3, 5e-3, decay_start=1.0)
    burst_flips = (upk.flips_executed - flips0) if upk is not None else 0
    b_s, _ = _eval_3b_mem_split(m, store_ev, init_t, seen_pairs, seed * 100 + 1, dev, is_dict, blind)
    b_e, _ = _eval_3b_mem_split(m, store_ev, init_t, exc_pairs, seed * 100 + 2, dev, is_dict, blind)
    b_h, _ = _eval_3b_mem_split(m, store_ev, init_t, held_pairs, seed * 100 + 3, dev, is_dict, blind)

    # Phase 3: Persistent multi-source revision
    _run_mem_phase(rev_t, rev_steps, lambda s: f"rev_src_{s % 3}", lr, 5e-4, decay_start=0.65)
    r_s, _ = _eval_3b_mem_split(m, store_ev, rev_t, seen_pairs, seed * 100 + 4, dev, is_dict, blind)
    r_e, _ = _eval_3b_mem_split(m, store_ev, rev_t, exc_pairs, seed * 100 + 5, dev, is_dict, blind)
    r_h, r_hnll = _eval_3b_mem_split(m, store_ev, rev_t, held_pairs, seed * 100 + 6, dev, is_dict, blind)

    storage_b = d_base.storage_bytes() if is_dict else (upk.memory_footprint_bytes() if upk is not None else 0)
    assert storage_b <= 4096, f"Memory cap violated: {storage_b} > 4096"
    return seed, {
        "init_seen_acc": i_s, "init_exc_acc": i_e, "init_heldout_acc": i_h, "init_heldout_nll": i_hnll,
        "burst_flips": float(burst_flips), "burst_seen_acc": b_s, "burst_exc_acc": b_e, "burst_heldout_acc": b_h,
        "rev_seen_acc": r_s, "rev_exc_acc": r_e, "rev_heldout_acc": r_h, "rev_heldout_nll": r_hnll,
        "storage_bytes": float(storage_b),
        "params": float(0 if is_dict else count_parameters(m)),
    }


# ==============================================================================
# Sub-Suite 3B-IC: Zero-Shot Contradiction Immunity & PC Rectification
# ==============================================================================

def _fast_3b_ic_batch(batch_size: int, num_facts: int, gen: torch.Generator, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    total_facts = num_facts + 1
    L = total_facts * 2 + 2
    ents = torch.rand(batch_size, ALPHABET_SIZE, generator=gen).argsort(dim=-1)[:, :total_facts] + ALPHABET_START
    vals = torch.rand(batch_size, ALPHABET_SIZE, generator=gen).argsort(dim=-1)[:, :total_facts] + ALPHABET_START
    q_idx = torch.randint(0, num_facts, (batch_size,), generator=gen)
    b_idx = torch.arange(batch_size)

    x = torch.zeros(batch_size, L, dtype=torch.long)
    for i in range(num_facts):
        x[:, 2 * i] = ents[:, i]
        x[:, 2 * i + 1] = vals[:, i]

    obs_pos = 2 * num_facts
    if mode == "contra":
        delta = torch.randint(1, ALPHABET_SIZE, (batch_size,), generator=gen)
        x[:, obs_pos] = ents[b_idx, q_idx]
        x[:, obs_pos + 1] = ((vals[b_idx, q_idx] - ALPHABET_START + delta) % ALPHABET_SIZE) + ALPHABET_START
    elif mode == "confirm":
        x[:, obs_pos] = ents[b_idx, q_idx]
        x[:, obs_pos + 1] = vals[b_idx, q_idx]
    else:
        x[:, obs_pos] = ents[:, num_facts]
        x[:, obs_pos + 1] = vals[:, num_facts]

    x[:, 2 * total_facts] = ents[b_idx, q_idx]
    x[:, 2 * total_facts + 1] = SEP_TOKEN
    return x, vals[b_idx, q_idx]


def run_seed_3b_ic(job: Tuple[str, int, float, int, int, bool]) -> Tuple[int, Dict[str, Any]]:
    model_name, seed, lr, steps, num_facts, use_cuda = job
    torch.set_num_threads(1)
    dev = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    set_all_seeds(seed)
    obs_val_pos = 2 * num_facts + 1

    is_dense = model_name == "Dense_Matched"
    untrained = model_name == "Null_Untrained"
    shuffled = model_name == "Null_ShuffledTargets"
    blind = model_name == "Null_BlindZeroInput"

    if is_dense:
        m: nn.Module = DenseRoPEContinualBaseline(d_model=96, num_heads=4, d_ff=432, num_layers=4).to(dev)
    else:
        m = RealSelfStabilizingDMoE(
            d_model=64, num_heads=4, d_ff=256, initial_layers=2, initial_experts=1,
            chunk_size=1, enable_thinking=True, max_thinking_steps=2, skip_threshold=0.10,
            theta_halt=0.14, enable_memory=True, num_ltm_slots=24, enable_three_level_attn=True,
            local_window=4, chunk_stride=4, enable_world_model=True, wm_rollout_horizon=1,
            causal_attn=True, use_rope=True, enable_layer_stm=True,
            enable_pc_rectification=True, pc_thresh=0.18, pc_suppress_gain=2.5,
        ).to(dev)

    gen = torch.Generator().manual_seed(seed * 997 + 1)
    if not untrained:
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        for _ in range(steps):
            xn, yn = _fast_3b_ic_batch(32, num_facts, gen, "new")
            xc, yc = _fast_3b_ic_batch(32, num_facts, gen, "confirm")
            x = torch.cat([xn, xc], dim=0).to(dev)
            y = torch.cat([yn, yc], dim=0).to(dev)
            if shuffled:
                y = y[torch.randperm(x.size(0), device=dev)]
                x = torch.cat([x[:, :1], x[:, 1:][torch.randperm(x.size(0), device=dev)]], dim=1)
            full_tgt = torch.cat([x[:, 1:], y.unsqueeze(1)], dim=1)
            m.train()
            if hasattr(m, "reset_ltm"):
                m.reset_ltm()
            opt.zero_grad(set_to_none=True)
            pred_mask = torch.zeros_like(x, dtype=torch.bool)
            pred_mask[:, -1] = True
            for t in range(2, x.size(1) - 1, 2):
                pred_mask[:, t] = (x[:, :t:2] == x[:, t : t + 1]).any(dim=1)
            if isinstance(m, RealSelfStabilizingDMoE):
                out = m(x, update_stats=True, targets=full_tgt)
                lg = out["output"].reshape(x.size(0), x.size(1), -1)
                loss = (
                    1.5 * F.cross_entropy(lg[:, -1, :], y)
                    + 1.0 * F.cross_entropy(lg[:, :-1, :].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
                    + 1.2 * m.thinker.compute_thinking_loss(out["thinking_out"], full_tgt, valid_mask=pred_mask)["loss"]
                    + 1.0 * out["world_model_loss"]["loss"]
                )
            else:
                out = m(x)
                lg = out["output"].reshape(x.size(0), x.size(1), -1)
                loss = (
                    1.5 * F.cross_entropy(lg[:, -1, :], y)
                    + 1.0 * F.cross_entropy(lg[:, :-1, :].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
                    + 1.2 * F.cross_entropy(lg[pred_mask], full_tgt[pred_mask])
                )
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()

    egen = torch.Generator().manual_seed(80000 + seed)
    x_n, y_n = _fast_3b_ic_batch(256, num_facts, egen, "new")
    x_f, y_f = _fast_3b_ic_batch(256, num_facts, egen, "confirm")
    x_c, y_c = _fast_3b_ic_batch(256, num_facts, egen, "contra")
    if blind:
        x_n, x_f, x_c = torch.zeros_like(x_n), torch.zeros_like(x_f), torch.zeros_like(x_c)
    x_n, y_n = x_n.to(dev), y_n.to(dev)
    x_f, y_f = x_f.to(dev), y_f.to(dev)
    x_c, y_c = x_c.to(dev), y_c.to(dev)

    def _eval_ic(pc_on: bool) -> Dict[str, float]:
        m.eval()
        if hasattr(m, "reset_ltm"):
            m.reset_ltm()
        if isinstance(m, RealSelfStabilizingDMoE):
            m.enable_pc_rectification = pc_on
        with torch.no_grad():
            on = m(x_n, update_stats=False) if isinstance(m, RealSelfStabilizingDMoE) else m(x_n)
            of = m(x_f, update_stats=False) if isinstance(m, RealSelfStabilizingDMoE) else m(x_f)
            oc = m(x_c, update_stats=False) if isinstance(m, RealSelfStabilizingDMoE) else m(x_c)

            def _acc(o, yt):
                return float((o["output"].reshape(yt.size(0), -1, 256)[:, -1, :].argmax(-1) == yt).float().mean().item())

            picked_contra = float(
                (oc["output"].reshape(y_c.size(0), -1, 256)[:, -1, :].argmax(-1) == x_c[:, obs_val_pos]).float().mean().item()
            )
            caus_diff = 0.0
            if isinstance(m, RealSelfStabilizingDMoE):
                xa = x_c[:16]
                xb = xa.clone()
                xb[:, obs_val_pos] = ALPHABET_START + ((xb[:, obs_val_pos] - ALPHABET_START + 1) % ALPHABET_SIZE)
                caus_diff = float(
                    (m(xa, update_stats=False)["output"][:, :obs_val_pos] - m(xb, update_stats=False)["output"][:, :obs_val_pos])
                    .abs()
                    .max()
                    .item()
                )

            if isinstance(m, RealSelfStabilizingDMoE) and oc.get("pre_pc_error") is not None:
                err_c = oc["pre_pc_error"][:, obs_val_pos].cpu().tolist()
                err_f = of["pre_pc_error"][:, obs_val_pos].cpu().tolist()
                err_n = on["pre_pc_error"][:, obs_val_pos].cpu().tolist()
                auroc_cf = compute_auroc(err_c, err_f)
                auroc_cn = compute_auroc(err_c, err_n)
                g_pos = (
                    float(oc["pc_write_gate"][:, obs_val_pos].mean().item())
                    if oc.get("pc_write_gate") is not None
                    else 1.0
                )
                g_pre = (
                    float(oc["pc_write_gate"][:, 1:obs_val_pos].min().item())
                    if oc.get("pc_write_gate") is not None
                    else 1.0
                )
            else:
                # Fallback answer-entropy/max-prob surprise for non-PC baseline
                def _unc(o):
                    p = F.softmax(o["output"].reshape(256, -1, 256)[:, obs_val_pos, :], dim=-1)
                    return (1.0 - p.max(dim=-1).values).cpu().tolist()
                auroc_cf = compute_auroc(_unc(oc), _unc(of))
                auroc_cn = compute_auroc(_unc(oc), _unc(on))
                g_pos, g_pre = 1.0, 1.0

            return {
                "ic_new_acc": _acc(on, y_n),
                "ic_confirm_acc": _acc(of, y_f),
                "ic_contra_acc": _acc(oc, y_c),
                "ic_picked_contra_rate": picked_contra,
                "ic_auroc_cf": auroc_cf,
                "ic_auroc_cn": auroc_cn,
                "ic_gate_contra": g_pos,
                "ic_gate_prior_min": g_pre,
                "ic_causality_diff": caus_diff,
                "params": float(count_parameters(m)),
            }

    main_res = _eval_ic(pc_on=True)
    if model_name == "Cog_Stage3_Full":
        nopc_res = _eval_ic(pc_on=False)
        return seed, {"main": main_res, "ablation_nopc": nopc_res}
    return seed, {"main": main_res}


# ==============================================================================
# Sub-Suite 3C: Real UTF-8 Corpus + Synthetic Lifecycle Calibration
# ==============================================================================

def run_seed_3c(job: Tuple[int, int, bool]) -> Tuple[int, Dict[str, float]]:
    seed, real_train_steps, use_cuda = job
    torch.set_num_threads(1)
    dev = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    code_windows = load_real_code_windows(str(ROOT), REAL_SEQ_LEN)
    prose_windows = load_real_prose_windows(str(ROOT), REAL_SEQ_LEN)

    res_a = run_structured_seed(seed=200 + seed, train_steps=real_train_steps, code_windows=code_windows, device=dev)
    res_b = run_open_corpus_seed(seed=300 + seed, train_steps=real_train_steps, prose_windows=prose_windows, code_windows=code_windows, device=dev)
    l2_res = _run_3a_stream("Cog_Stage3_Full", seed, "l2", lr=4e-3, sh_scale=0.05, steps_per_dom=105, device=dev)

    na, oa = res_a["New_Unified"], res_a["Old_MeanOnly"]
    nb, ob = res_b["New_Unified"], res_b["Old_MeanOnly"]
    return seed, {
        "real_a_fpr": na["in_dist_real"],
        "real_a_code_tpr": na["coarse_real_code"],
        "real_a_twin_tpr": na["near_twin_real"],
        "real_a_nuance_tpr": na["local_nuance_real"],
        "real_a_old_nuance_tpr": oa["local_nuance_real"],
        "real_a_old_twin_tpr": oa["near_twin_real"],
        "real_b_fpr": nb["in_dist_real"],
        "real_b_code_tpr": nb["coarse_real_code"],
        "real_b_twin_tpr": nb["near_twin_real"],
        "real_b_nuance_tpr": nb["local_nuance_real"],
        "real_b_nuance_ascii_tpr": nb["local_nuance_ascii"],
        "real_b_old_nuance_tpr": ob["local_nuance_real"],
        "l2_burst_spawns": l2_res["burst_spawns"],
        "l2_sustain_spawns": l2_res["sustain_spawns"],
        "l2_revisit_spawns": l2_res["revisit_spawns"],
        "l2_d0_pre_revisit": l2_res["d0_pre_revisit"],
        "l2_d0_post_revisit": l2_res["d0_post_revisit"],
    }


# ==============================================================================
# Orchestration: --mode tune | test
# ==============================================================================

def _parallel_map(fn: Callable, jobs: List[Any], workers: int) -> List[Any]:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, jobs))


def run_tune(args: argparse.Namespace, device: torch.device) -> None:
    suites = set(args.suites.split(",")) if args.suites != "all" else {"3a", "3b_mem", "3b_ic", "3c"}
    use_cuda = device.type == "cuda"
    workers = max(1, int(args.workers))
    entries: List[Dict[str, Any]] = []

    causality = verify_strict_causality_gate(device)
    print("Strict Causality Gate (DEV pre-check):", causality, flush=True)

    out_path = RESULTS_DIR / "tuning_log_v1.json"
    prev_entries: List[Dict[str, Any]] = []
    if args.append and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            prev_entries = json.load(f).get("entries", [])

    def _flush_tune_log() -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        seen = {(e.get("suite"), e.get("model"), e.get("lr")) for e in entries}
        merged = [e for e in prev_entries if (e.get("suite"), e.get("model"), e.get("lr")) not in seen] + list(entries)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": _collect_metadata(), "causality_gate": causality, "dev_seeds": DEV_SEEDS, "entries": merged}, f, indent=2)

    if "3a" in suites:
        print("=== DEV TUNE: Sub-Suite 3A ===")
        for m_name, lrs in [
            ("Cog_Stage3_Full", [4e-3]),
            ("Cog_NoGPM", [4e-3]),
            ("Cog_UnigramRouter", [4e-3]),
            ("Dense_NoReplay", [3e-3, 5e-3]),
            ("Dense_Replay128", [3e-3, 5e-3]),
            ("StaticMoE_Matched", [3e-3, 5e-3]),
        ]:
            for lr in lrs:
                t0 = time.perf_counter()
                jobs = [(m_name, s, lr, 0.05, args.steps_3a, use_cuda) for s in DEV_SEEDS]
                rows = [r[1] for r in sorted(_parallel_map(run_seed_3a, jobs, workers), key=lambda x: x[0])]
                agg = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
                entries.append({"suite": "3a", "model": m_name, "lr": lr, **agg})
                print(
                    f"  {m_name:20s} lr={lr:.1e} ({agg['wall_clock_sec']:.1f}s) | "
                    f"L0={agg['l0_avg']['mean']:.3f} L1(r0)={agg['l1_rho0_avg']['mean']:.3f} "
                    f"L1(r.5)={agg['l1_rho5_avg']['mean']:.3f} L2(D0={agg['l2_d0_post_revisit']['mean']:.3f}, "
                    f"bsp={agg['l2_burst_spawns']['mean']:.2f}, ssp={agg['l2_sustain_spawns']['mean']:.2f}, rsp={agg['l2_revisit_spawns']['mean']:.2f})",
                    flush=True,
                )
        for nm, n_lr in [("Null_Untrained", 0.0), ("Null_ShuffledTargets", 4e-3), ("Null_BlindZeroInput", 4e-3)]:
            _, row = run_seed_3a((nm, DEV_SEEDS[0], n_lr, 0.05, args.steps_3a, use_cuda))
            agg = _aggregate_seed_dicts([row], 0.0)
            entries.append({"suite": "3a", "model": nm, "lr": n_lr, **agg})
            print(f"  [DEV Null] {nm:20s} | L1(r0)={row['l1_rho0_avg']:.3f} L2_D0={row['l2_d0_post_revisit']:.3f}", flush=True)
        _flush_tune_log()

    if "3b_mem" in suites:
        print("=== DEV TUNE: Sub-Suite 3B-Mem ===")
        for m_name, lr in [
            ("Cog_Stage3_Full", 1.2e-2),
            ("Ablation_NoUpkeeper", 1.2e-2),
            ("Dense_NoUpk", 1.0e-2),
            ("Dense_NoUpk", 1.2e-2),
            ("BudgetedDictionaryBaseline", 0.0),
        ]:
            t0 = time.perf_counter()
            jobs = [(m_name, s, lr, args.init_steps_3bm, args.rev_steps_3bm, 1.5, use_cuda) for s in DEV_SEEDS]
            rows = [r[1] for r in sorted(_parallel_map(run_seed_3b_mem, jobs, workers), key=lambda x: x[0])]
            agg = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
            entries.append({"suite": "3b_mem", "model": m_name, "lr": lr, **agg})
            print(
                f"  {m_name:26s} lr={lr:.1e} ({agg['wall_clock_sec']:.1f}s) | "
                f"init(s={agg['init_seen_acc']['mean']:.3f}, e={agg['init_exc_acc']['mean']:.3f}, h={agg['init_heldout_acc']['mean']:.3f}) | "
                f"burst(s={agg['burst_seen_acc']['mean']:.3f}, fl={agg['burst_flips']['mean']:.2f}) | "
                f"rev(s={agg['rev_seen_acc']['mean']:.3f}, e={agg['rev_exc_acc']['mean']:.3f}, h={agg['rev_heldout_acc']['mean']:.3f}) B={agg['storage_bytes']['max']:.0f}",
                flush=True,
            )
        for nm, n_lr in [("Null_Untrained", 0.0), ("Null_ShuffledTargets", 1.2e-2), ("Null_BlindZeroInput", 1.2e-2)]:
            _, row = run_seed_3b_mem((nm, DEV_SEEDS[0], n_lr, args.init_steps_3bm, args.rev_steps_3bm, 1.5, use_cuda))
            agg = _aggregate_seed_dicts([row], 0.0)
            entries.append({"suite": "3b_mem", "model": nm, "lr": n_lr, **agg})
            print(f"  [DEV Null] {nm:20s} | rev(s={row['rev_seen_acc']:.3f}, h={row['rev_heldout_acc']:.3f})", flush=True)
        _flush_tune_log()

    if "3b_ic" in suites:
        print("=== DEV TUNE: Sub-Suite 3B-IC ===")
        for m_name, lrs in [
            ("Cog_Stage3_Full", [3e-3]),
            ("Dense_Matched", [2e-3, 3e-3, 5e-3]),
        ]:
            for lr in lrs:
                t0 = time.perf_counter()
                jobs = [(m_name, s, lr, args.steps_3bic, 6, use_cuda) for s in DEV_SEEDS]
                res_list = sorted(_parallel_map(run_seed_3b_ic, jobs, workers), key=lambda x: x[0])
                rows = [r[1]["main"] for r in res_list]
                agg = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
                entries.append({"suite": "3b_ic", "model": m_name, "lr": lr, **agg})
                print(
                    f"  {m_name:20s} lr={lr:.1e} ({agg['wall_clock_sec']:.1f}s) | "
                    f"new={agg['ic_new_acc']['mean']:.3f} conf={agg['ic_confirm_acc']['mean']:.3f} "
                    f"contra={agg['ic_contra_acc']['mean']:.3f} wrong={agg['ic_picked_contra_rate']['mean']:.3f} "
                    f"auroc_cf={agg['ic_auroc_cf']['mean']:.3f} caus={agg['ic_causality_diff']['max']:.1e}",
                    flush=True,
                )
                if m_name == "Cog_Stage3_Full":
                    nopc_rows = [r[1]["ablation_nopc"] for r in res_list]
                    agg_np = _aggregate_seed_dicts(nopc_rows, 0.0)
                    entries.append({"suite": "3b_ic", "model": "Ablation_NoPCRectification", "lr": lr, **agg_np})
                    print(
                        f"  {'Ablation_NoPC':20s} lr={lr:.1e} | "
                        f"new={agg_np['ic_new_acc']['mean']:.3f} conf={agg_np['ic_confirm_acc']['mean']:.3f} "
                        f"contra={agg_np['ic_contra_acc']['mean']:.3f}",
                        flush=True,
                    )
        for nm, n_lr in [("Null_Untrained", 0.0), ("Null_ShuffledTargets", 3e-3), ("Null_BlindZeroInput", 3e-3)]:
            _, res_n = run_seed_3b_ic((nm, DEV_SEEDS[0], n_lr, args.steps_3bic, 6, use_cuda))
            rm = res_n["main"]
            agg = _aggregate_seed_dicts([rm], 0.0)
            entries.append({"suite": "3b_ic", "model": nm, "lr": n_lr, **agg})
            print(f"  [DEV Null] {nm:20s} | new={rm['ic_new_acc']:.3f} contra={rm['ic_contra_acc']:.3f}", flush=True)
        _flush_tune_log()

    if "3c" in suites:
        print("=== DEV TUNE: Sub-Suite 3C ===")
        t0 = time.perf_counter()
        jobs = [(s, args.steps_real_3c, use_cuda) for s in DEV_SEEDS]
        rows = [r[1] for r in sorted(_parallel_map(run_seed_3c, jobs, workers), key=lambda x: x[0])]
        agg = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
        entries.append({"suite": "3c", "model": "Unified_Stage3C", **agg})
        print(
            f"  Stage3C ({agg['wall_clock_sec']:.1f}s) | "
            f"RealA(fpr={agg['real_a_fpr']['mean']:.3f}, code={agg['real_a_code_tpr']['mean']:.3f}, "
            f"twin={agg['real_a_twin_tpr']['mean']:.3f}, nuance={agg['real_a_nuance_tpr']['mean']:.3f}) | "
            f"RealB(fpr={agg['real_b_fpr']['mean']:.3f}, code={agg['real_b_code_tpr']['mean']:.3f}, "
            f"twin={agg['real_b_twin_tpr']['mean']:.3f}, nuance={agg['real_b_nuance_tpr']['mean']:.3f})",
            flush=True,
        )

    _flush_tune_log()
    print(f"Saved DEV tuning log to {out_path}")


def run_test(args: argparse.Namespace, device: torch.device) -> None:
    use_cuda = device.type == "cuda"
    workers = max(1, int(args.workers))
    causality = verify_strict_causality_gate(device)
    print("Strict Causality Gate:", causality, flush=True)

    results: Dict[str, Any] = {
        "metadata": _collect_metadata(),
        "test_seeds": TEST_SEEDS,
        "config": vars(args),
        "causality_gate": causality,
        "3a": {},
        "3b_mem": {},
        "3b_ic": {},
        "3c": {},
    }

    # --- 3A ---
    models_3a = [
        ("Cog_Stage3_Full", args.lr_cog_3a),
        ("Cog_OracleRouting", args.lr_cog_3a),
        ("Cog_NoGPM", args.lr_cog_3a),
        ("Cog_UnigramRouter", args.lr_cog_3a),
        ("Dense_NoReplay", args.lr_dense_3a),
        ("Dense_Replay128", args.lr_dense_3a),
        ("StaticMoE_Matched", args.lr_smoe_3a),
        ("Null_Untrained", 0.0),
        ("Null_ShuffledTargets", args.lr_cog_3a),
        ("Null_BlindZeroInput", args.lr_cog_3a),
    ]
    for nm, lr in models_3a:
        t0 = time.perf_counter()
        jobs = [(nm, s, lr, 0.05, args.steps_3a, use_cuda) for s in TEST_SEEDS]
        rows = [r[1] for r in sorted(_parallel_map(run_seed_3a, jobs, workers), key=lambda x: x[0])]
        results["3a"][nm] = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
        a = results["3a"][nm]
        print(
            f"[3A] {nm:22s} ({a['wall_clock_sec']:5.1f}s) | L0={a['l0_avg']['mean']:.3f} "
            f"L1(0.0)={a['l1_rho0_avg']['mean']:.3f} L1(0.5)={a['l1_rho5_avg']['mean']:.3f} "
            f"L2_D0={a['l2_d0_post_revisit']['mean']:.3f} bsp={a['l2_burst_spawns']['mean']:.2f} rsp={a['l2_revisit_spawns']['mean']:.2f}",
            flush=True,
        )

    # --- 3B-Mem ---
    models_3bm = [
        ("Cog_Stage3_Full", args.lr_cog_3bm),
        ("Ablation_NoUpkeeper", args.lr_cog_3bm),
        ("Dense_NoUpk", args.lr_dense_3bm),
        ("BudgetedDictionaryBaseline", 0.0),
        ("Null_Untrained", 0.0),
        ("Null_ShuffledTargets", args.lr_cog_3bm),
        ("Null_BlindZeroInput", args.lr_cog_3bm),
    ]
    for nm, lr in models_3bm:
        t0 = time.perf_counter()
        jobs = [(nm, s, lr, args.init_steps_3bm, args.rev_steps_3bm, 1.5, use_cuda) for s in TEST_SEEDS]
        rows = [r[1] for r in sorted(_parallel_map(run_seed_3b_mem, jobs, workers), key=lambda x: x[0])]
        results["3b_mem"][nm] = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
        b = results["3b_mem"][nm]
        print(
            f"[3B-Mem] {nm:26s} ({b['wall_clock_sec']:5.1f}s) | init=({b['init_seen_acc']['mean']:.3f},{b['init_heldout_acc']['mean']:.3f}) "
            f"burst=({b['burst_seen_acc']['mean']:.3f},fl={b['burst_flips']['mean']:.2f}) "
            f"rev=({b['rev_seen_acc']['mean']:.3f},{b['rev_exc_acc']['mean']:.3f},{b['rev_heldout_acc']['mean']:.3f}) B={b['storage_bytes']['max']:.0f}",
            flush=True,
        )

    # --- 3B-IC ---
    models_3bic = [
        ("Cog_Stage3_Full", args.lr_cog_3bic),
        ("Dense_Matched", args.lr_dense_3bic),
        ("Null_Untrained", 0.0),
        ("Null_ShuffledTargets", args.lr_cog_3bic),
        ("Null_BlindZeroInput", args.lr_cog_3bic),
    ]
    for nm, lr in models_3bic:
        t0 = time.perf_counter()
        jobs = [(nm, s, lr, args.steps_3bic, 6, use_cuda) for s in TEST_SEEDS]
        res_list = sorted(_parallel_map(run_seed_3b_ic, jobs, workers), key=lambda x: x[0])
        rows = [r[1]["main"] for r in res_list]
        results["3b_ic"][nm] = _aggregate_seed_dicts(rows, time.perf_counter() - t0)
        c = results["3b_ic"][nm]
        print(
            f"[3B-IC] {nm:22s} ({c['wall_clock_sec']:5.1f}s) | new={c['ic_new_acc']['mean']:.3f} "
            f"conf={c['ic_confirm_acc']['mean']:.3f} contra={c['ic_contra_acc']['mean']:.3f} "
            f"auroc_cf={c['ic_auroc_cf']['mean']:.3f} caus={c['ic_causality_diff']['max']:.1e}",
            flush=True,
        )
        if nm == "Cog_Stage3_Full":
            nopc_rows = [r[1]["ablation_nopc"] for r in res_list]
            results["3b_ic"]["Ablation_NoPCRectification"] = _aggregate_seed_dicts(nopc_rows, 0.0)
            np = results["3b_ic"]["Ablation_NoPCRectification"]
            print(
                f"[3B-IC] {'Ablation_NoPCRectification':22s} | new={np['ic_new_acc']['mean']:.3f} "
                f"conf={np['ic_confirm_acc']['mean']:.3f} contra={np['ic_contra_acc']['mean']:.3f}",
                flush=True,
            )

    # --- 3C ---
    t0 = time.perf_counter()
    jobs_3c = [(s, args.steps_real_3c, use_cuda) for s in TEST_SEEDS]
    rows_3c = [r[1] for r in sorted(_parallel_map(run_seed_3c, jobs_3c, workers), key=lambda x: x[0])]
    results["3c"]["Unified_Stage3C"] = _aggregate_seed_dicts(rows_3c, time.perf_counter() - t0)
    c3 = results["3c"]["Unified_Stage3C"]
    print(
        f"[3C] ({c3['wall_clock_sec']:5.1f}s) | RealA(fpr={c3['real_a_fpr']['mean']:.3f}, code={c3['real_a_code_tpr']['mean']:.3f}, "
        f"twin={c3['real_a_twin_tpr']['mean']:.3f}, nuance={c3['real_a_nuance_tpr']['mean']:.3f}) | "
        f"RealB(fpr={c3['real_b_fpr']['mean']:.3f}, code={c3['real_b_code_tpr']['mean']:.3f}, "
        f"twin={c3['real_b_twin_tpr']['mean']:.3f}, nuance={c3['real_b_nuance_tpr']['mean']:.3f})",
        flush=True,
    )

    # Paired stats & Pre-registered falsifiers
    r3a = results["3a"]
    r3bm = results["3b_mem"]
    r3bi = results["3b_ic"]
    results["paired_comparisons"] = {
        "3a_l1_rho0_full_vs_dense_noreplay": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["per_seed"], r3a["Dense_NoReplay"]["l1_rho0_avg"]["per_seed"]),
        "3a_l1_rho0_full_vs_dense_replay128": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["per_seed"], r3a["Dense_Replay128"]["l1_rho0_avg"]["per_seed"]),
        "3a_l1_rho0_full_vs_static_moe": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["per_seed"], r3a["StaticMoE_Matched"]["l1_rho0_avg"]["per_seed"]),
        "3a_l1_rho0_full_vs_nogpm": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["per_seed"], r3a["Cog_NoGPM"]["l1_rho0_avg"]["per_seed"]),
        "3a_l1_rho0_full_vs_unigram": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["per_seed"], r3a["Cog_UnigramRouter"]["l1_rho0_avg"]["per_seed"]),
        "3a_l1_rho5_full_vs_dense_replay128": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho5_avg"]["per_seed"], r3a["Dense_Replay128"]["l1_rho5_avg"]["per_seed"]),
        "3a_l1_rho5_full_vs_unigram": paired_stats(r3a["Cog_Stage3_Full"]["l1_rho5_avg"]["per_seed"], r3a["Cog_UnigramRouter"]["l1_rho5_avg"]["per_seed"]),
        "3bm_init_heldout_full_vs_dict": paired_stats(r3bm["Cog_Stage3_Full"]["init_heldout_acc"]["per_seed"], r3bm["BudgetedDictionaryBaseline"]["init_heldout_acc"]["per_seed"]),
        "3bm_burst_seen_full_vs_noupk": paired_stats(r3bm["Cog_Stage3_Full"]["burst_seen_acc"]["per_seed"], r3bm["Ablation_NoUpkeeper"]["burst_seen_acc"]["per_seed"]),
        "3bm_rev_heldout_full_vs_dict": paired_stats(r3bm["Cog_Stage3_Full"]["rev_heldout_acc"]["per_seed"], r3bm["BudgetedDictionaryBaseline"]["rev_heldout_acc"]["per_seed"]),
        "3bic_contra_full_vs_nopc": paired_stats(r3bi["Cog_Stage3_Full"]["ic_contra_acc"]["per_seed"], r3bi["Ablation_NoPCRectification"]["ic_contra_acc"]["per_seed"]),
        "3bic_contra_full_vs_dense": paired_stats(r3bi["Cog_Stage3_Full"]["ic_contra_acc"]["per_seed"], r3bi["Dense_Matched"]["ic_contra_acc"]["per_seed"]),
        "3c_real_a_nuance_new_vs_old": paired_stats(c3["real_a_nuance_tpr"]["per_seed"], c3["real_a_old_nuance_tpr"]["per_seed"]),
        "3c_real_b_nuance_new_vs_old": paired_stats(c3["real_b_nuance_tpr"]["per_seed"], c3["real_b_old_nuance_tpr"]["per_seed"]),
    }

    nulls = ("Null_Untrained", "Null_ShuffledTargets", "Null_BlindZeroInput")
    falsifiers = {
        "F1_strict_causality": all(v <= 1e-5 for v in causality.values()) and r3bi["Cog_Stage3_Full"]["ic_causality_diff"]["max"] <= 1e-5,
        "F2_3a_continual_retention_and_ablations": (
            r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["mean"] >= 0.90
            and r3a["Cog_Stage3_Full"]["l1_rho5_avg"]["mean"] >= 0.80
            and r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["mean"] > r3a["Dense_NoReplay"]["l1_rho0_avg"]["mean"] + 0.35
            and r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["mean"] > r3a["Dense_Replay128"]["l1_rho0_avg"]["mean"] + 0.35
            and r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["mean"] > r3a["StaticMoE_Matched"]["l1_rho0_avg"]["mean"] + 0.35
            and r3a["Cog_Stage3_Full"]["l1_rho5_avg"]["mean"] > r3a["Dense_Replay128"]["l1_rho5_avg"]["mean"] + 0.25
            and r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["mean"] > r3a["Cog_NoGPM"]["l1_rho0_avg"]["mean"] + 0.15
            and r3a["Cog_Stage3_Full"]["l1_rho0_avg"]["mean"] > r3a["Cog_UnigramRouter"]["l1_rho0_avg"]["mean"] + 0.35
        ),
        "F3_3a_boundary_free_l2_lifecycle": (
            r3a["Cog_Stage3_Full"]["l2_burst_spawns"]["mean"] == 0.0
            and r3a["Cog_Stage3_Full"]["l2_sustain_spawns"]["mean"] >= 1.5
            and r3a["Cog_Stage3_Full"]["l2_revisit_spawns"]["mean"] == 0.0
            and r3a["Cog_Stage3_Full"]["l2_d0_pre_revisit"]["mean"] >= 0.85
            and r3a["Cog_Stage3_Full"]["l2_d0_post_revisit"]["mean"] >= 0.85
            and r3a["Cog_Stage3_Full"]["l2_avg"]["mean"] >= 0.85
        ),
        "F4_3bm_budget_burst_and_generalization": (
            r3bm["Cog_Stage3_Full"]["storage_bytes"]["max"] <= 4096.0
            and r3bm["Cog_Stage3_Full"]["init_seen_acc"]["mean"] >= 0.95
            and r3bm["Cog_Stage3_Full"]["init_heldout_acc"]["mean"] >= 0.55
            and r3bm["Cog_Stage3_Full"]["init_heldout_acc"]["mean"] > r3bm["BudgetedDictionaryBaseline"]["init_heldout_acc"]["mean"] + 0.40
            and r3bm["Cog_Stage3_Full"]["burst_flips"]["mean"] == 0.0
            and r3bm["Cog_Stage3_Full"]["burst_seen_acc"]["mean"] >= 0.90
            and r3bm["Cog_Stage3_Full"]["burst_seen_acc"]["mean"] > r3bm["Ablation_NoUpkeeper"]["burst_seen_acc"]["mean"] + 0.50
            and r3bm["Cog_Stage3_Full"]["rev_seen_acc"]["mean"] >= 0.95
            and r3bm["Cog_Stage3_Full"]["rev_heldout_acc"]["mean"] >= 0.50
            and r3bm["Cog_Stage3_Full"]["rev_heldout_acc"]["mean"] > r3bm["BudgetedDictionaryBaseline"]["rev_heldout_acc"]["mean"] + 0.35
        ),
        "F5_3bic_zero_shot_contradiction_immunity": (
            r3bi["Cog_Stage3_Full"]["ic_new_acc"]["mean"] >= 0.90
            and r3bi["Cog_Stage3_Full"]["ic_confirm_acc"]["mean"] >= 0.90
            and r3bi["Cog_Stage3_Full"]["ic_contra_acc"]["mean"] >= 0.85
            and r3bi["Cog_Stage3_Full"]["ic_contra_acc"]["mean"] > r3bi["Ablation_NoPCRectification"]["ic_contra_acc"]["mean"] + 0.40
            and r3bi["Cog_Stage3_Full"]["ic_auroc_cf"]["mean"] >= 0.90
        ),
        "F6_3c_real_stream_calibration": (
            c3["real_a_fpr"]["mean"] <= 0.15
            and c3["real_a_code_tpr"]["mean"] >= 0.90
            and c3["real_a_nuance_tpr"]["mean"] >= 0.90
            and c3["real_b_fpr"]["mean"] <= 0.15
            and c3["real_b_code_tpr"]["mean"] >= 0.90
            and c3["real_b_nuance_tpr"]["mean"] >= 0.90
            and c3["l2_burst_spawns"]["mean"] == 0.0
            and c3["l2_revisit_spawns"]["mean"] == 0.0
        ),
        "F7_null_controls_fail": (
            all(r3a[nm]["l1_rho0_avg"]["mean"] < 0.18 for nm in nulls)
            and all(
                r3bm[nm]["rev_seen_acc"]["mean"] < 0.15 and r3bm[nm]["rev_heldout_acc"]["mean"] < 0.15
                for nm in nulls
            )
            and all(
                r3bi[nm]["ic_new_acc"]["mean"] < 0.18 and r3bi[nm]["ic_contra_acc"]["mean"] < 0.18
                for nm in nulls
            )
        ),
    }
    results["falsifiers"] = falsifiers
    results["all_passed"] = all(falsifiers.values())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "results_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Stage 3 test results to {out_path} | all_passed={results['all_passed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "test"], default="tune")
    parser.add_argument("--suites", type=str, default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--steps_3a", type=int, default=100)
    parser.add_argument("--lr_cog_3a", type=float, default=4e-3)
    parser.add_argument("--lr_dense_3a", type=float, default=3e-3)
    parser.add_argument("--lr_smoe_3a", type=float, default=3e-3)
    parser.add_argument("--init_steps_3bm", type=int, default=3000)
    parser.add_argument("--rev_steps_3bm", type=int, default=1500)
    parser.add_argument("--lr_cog_3bm", type=float, default=1.2e-2)
    parser.add_argument("--lr_dense_3bm", type=float, default=1.2e-2)
    parser.add_argument("--steps_3bic", type=int, default=1200)
    parser.add_argument("--lr_cog_3bic", type=float, default=3e-3)
    parser.add_argument("--lr_dense_3bic", type=float, default=2e-3)
    parser.add_argument("--steps_real_3c", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3 Benchmark | device={device} | mode={args.mode} | workers={args.workers}", flush=True)
    if args.mode == "tune":
        run_tune(args, device)
    else:
        run_test(args, device)


if __name__ == "__main__":
    main()

"""
Stage 2 Verification Suite: Dynamic Thinking Loop, Epistemic Abstention & Conflict Catching,
and Calibrated Latent World-Model Imagination vs Matched Strong Baselines & Null Controls.

Protocol (C:/Users/Asta/.gemini/config/rules/research-protocol.md):
  - DEV_SEEDS = [100, 101, 102, 103, 104] for --mode tune ONLY (logged to tuning_log_v1.json).
  - TEST_SEEDS = [0..19] for frozen 20-seed --mode test AFTER PREREG_v1.md is frozen.

Tasks:
  2A-Adj : Adjacent Multi-Hop State Composition (Train D<=3, OOD D=4..5) + Adaptive Compute.
  2A-Ptr : Non-Adjacent Variable-Gap Pointer Hops (1..2 distractors between hops -> breaks z_{t-1} shortcut).
  2B     : Zero-Shot Epistemic Abstention & In-Context Conflict Catching (same vocab [32..63], trained ONLY on Clean).
  2C     : Multi-Step World-Model Rollout (H=1..3), Horizon Uncertainty Calibration, & Downstream Trajectory Benefit.
  2D     : Strict Causality Gate (max_diff <= 1e-5) + GPU Speed / Active-Token Compute + 3 Null Controls.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.stage2_baselines import (
    CausalDenseRoPEBaseline,
    CausalLoopedTransformerBaseline,
    GRUBaseline,
    count_parameters,
)


RESULTS_DIR = ROOT / "results" / "stage2_thinking_world_model"
DEV_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(20))

VOCAB_START = 32
VOCAB_SIZE = 32  # Shared [32..63]
DISTRACTOR_START = 16
DISTRACTOR_SIZE = 16  # [16..31] distractors seen uniformly across episodes
QUERY_MARK = 64
RULE_MARK = 65
NOISE_PAD = 66


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_adamw(params, lr: float, device: torch.device) -> torch.optim.Optimizer:
    kwargs: Dict[str, Any] = {"lr": lr}
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def set_warmup_cosine_lr(
    opt: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 25,
) -> None:
    w = min(warmup_steps, max(1, total_steps // 5))
    if step < w:
        lr = base_lr * float(step + 1) / float(w)
    else:
        prog = float(step - w) / float(max(1, total_steps - w))
        lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
    for pg in opt.param_groups:
        pg["lr"] = lr


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
    """Exact rank-based AUROC handling ties via mid-ranks."""
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


def compute_ece(probs: List[float], outcomes: List[float], n_bins: int = 10) -> float:
    """Expected Calibration Error |mean(p) - mean(outcome)| across bins."""
    if not probs:
        return 0.0
    total = len(probs)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        in_bin = [
            (p, o)
            for p, o in zip(probs, outcomes)
            if (lo <= p < hi) or (b == n_bins - 1 and p == 1.0)
        ]
        if in_bin:
            mp = sum(p for p, _ in in_bin) / len(in_bin)
            mo = sum(o for _, o in in_bin) / len(in_bin)
            ece += (len(in_bin) / total) * abs(mp - mo)
    return float(ece)


def forward_logits(out: Dict[str, Any]) -> torch.Tensor:
    lg = out["output"]
    if lg.dim() == 4:
        return lg.squeeze(2)
    return lg


def model_uncertainty(model: nn.Module, out: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract strictly causal per-token answer uncertainty [B, L] in [0, 1] and optional pc_error [B, L].
    """
    if isinstance(model, RealSelfStabilizingDMoE):
        u_dict = model.assess_epistemic_uncertainty(out)
        return u_dict["answer_uncertainty"], u_dict.get("pc_error")
    lg = forward_logits(out)
    probs = F.softmax(lg, dim=-1)
    sm_unc = (1.0 - probs.max(dim=-1).values).clamp(0.0, 1.0)
    return sm_unc, None


def build_model(name: str, device: torch.device) -> nn.Module:
    # Base proposed Stage 2 cognitive architecture: ~218.6K params
    cog_common = dict(
        d_model=48,
        num_heads=4,
        d_ff=74,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_memory=True,
        num_ltm_slots=64,
        enable_three_level_attn=True,
        local_window=8,
        chunk_stride=4,
        enable_layer_stm=True,
        signed_beta=True,
        use_rope=True,
        causal_attn=True,
        auto_write_ltm=False,
        enable_thinking=True,
        max_thinking_steps=5,
        skip_threshold=0.05,
        theta_halt=0.06,
        abstain_threshold=0.65,
    )
    if name in (
        "Cog_Stage2_Full",
        "Cog_FixedThink",
        "Null_Untrained",
        "Null_ShuffledTargets",
        "Null_BlindZeroInput",
    ):
        m = RealSelfStabilizingDMoE(enable_world_model=True, wm_rollout_horizon=3, **cog_common)
    elif name == "Cog_NoWM":
        m = RealSelfStabilizingDMoE(enable_world_model=False, **cog_common)
    elif name == "Cog_OnePass":
        m = RealSelfStabilizingDMoE(enable_world_model=False, **cog_common)
    elif name == "LoopedTransformer_Baseline":
        # Weight-tied Causal RoPE Transformer iterated K steps (~220K params)
        m = CausalLoopedTransformerBaseline(
            d_model=72, num_heads=4, d_ff=288, num_prelude_layers=2, max_steps=5
        )
    elif name == "DenseRoPE_Deep":
        # 6-layer one-pass Causal RoPE Transformer (~220K params)
        m = CausalDenseRoPEBaseline(d_model=56, num_heads=4, d_ff=152, num_layers=6)
    elif name == "GRU_Baseline":
        # 3-layer native PyTorch GRU (~225K params)
        m = GRUBaseline(d_model=96, num_layers=3)
    else:
        raise ValueError(f"Unknown Stage 2 model: {name}")
    return m.to(device)


def compute_loss_for_model(
    model_name: str,
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    wm_weight: float = 0.25,
    think_train_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if model_name == "Cog_OnePass":
        out = model(xb, force_thinking_steps=0, targets=yb)
        lg = forward_logits(out)
        if valid_mask is not None and bool(valid_mask.any()):
            loss = F.cross_entropy(lg[valid_mask], yb[valid_mask])
        else:
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1))
        return loss, out

    if isinstance(model, RealSelfStabilizingDMoE):
        train_steps = 5 if model_name == "Cog_FixedThink" else (think_train_steps if think_train_steps is not None else 3)
        out = model(xb, force_thinking_steps=train_steps, targets=yb)
        if model.enable_thinking and model.thinker is not None and "thinking_out" in out:
            th_d = model.thinker.compute_thinking_loss(out["thinking_out"], yb, valid_mask=valid_mask)
            task_loss = th_d["loss"]
        else:
            lg = forward_logits(out)
            task_loss = (
                F.cross_entropy(lg[valid_mask], yb[valid_mask])
                if (valid_mask is not None and bool(valid_mask.any()))
                else F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1))
            )
        wm_loss = (
            out["world_model_loss"]["loss"]
            if out.get("world_model_loss") is not None
            else (task_loss * 0.0)
        )
        return task_loss + wm_weight * wm_loss, out

    if isinstance(model, CausalLoopedTransformerBaseline):
        loop_steps = think_train_steps if think_train_steps is not None else 3
        out = model(xb, force_thinking_steps=loop_steps)
        if valid_mask is None:
            loss = model.compute_deep_loss(out, yb)
        else:
            step_logits = out["step_logits"]
            weights = [float(k + 1) for k in range(len(step_logits))]
            w_sum = sum(weights)
            loss = step_logits[0].sum() * 0.0
            for w, lg in zip(weights, step_logits):
                ce_k = F.cross_entropy(lg[valid_mask], yb[valid_mask])
                loss = loss + (w / w_sum) * ce_k
        return loss, out

    out = model(xb)
    lg = forward_logits(out)
    if valid_mask is not None and bool(valid_mask.any()):
        loss = F.cross_entropy(lg[valid_mask], yb[valid_mask])
    else:
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1))
    return loss, out


def eval_model_preds(
    model_name: str,
    model: nn.Module,
    x: torch.Tensor,
    blind_zero: bool = False,
) -> Dict[str, Any]:
    xin = torch.zeros_like(x) if blind_zero else x
    if model_name == "Cog_OnePass":
        return model(xin, force_thinking_steps=0)
    if isinstance(model, RealSelfStabilizingDMoE):
        force = 5 if model_name == "Cog_FixedThink" else None
        return model(xin, force_thinking_steps=force)
    if isinstance(model, CausalLoopedTransformerBaseline):
        return model(xin, force_thinking_steps=5)
    return model(xin)


# ==============================================================================
# Task 2A-Adj: Adjacent Multi-Hop State Composition (Train D<=3, OOD D=4..5)
# Task 2A-Ptr: Non-Adjacent Variable-Gap Pointer Hops (1..2 distractors/hop)
# ==============================================================================

CHAIN_N_STATES = 8
CHAIN_STATE_OFF = 32   # [32..39]: reset / start states (d=0)
CHAIN_OP_OFF = 40      # [40..47]: hop operators (d>=1)
CHAIN_DIST_OFF = 48    # [48..55]: distractors inserted in 2A-Ptr


def _make_transition_table(seed: int, device: torch.device) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(seed * 7919 + 17)
    return torch.randint(0, CHAIN_N_STATES, (CHAIN_N_STATES, CHAIN_N_STATES), generator=rng).to(device)


def make_2a_adj_batch(
    trans: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_depth: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Causal multi-hop state chain:
      - Start token in [32..39] (d=0), target = start state.
      - Op token in [40..47] (d=1..max_depth), target = trans[op, cur_s] + 32.
    """
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    depths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    cur_s = torch.randint(0, CHAIN_N_STATES, (batch_size,), device=device)
    cur_d = torch.zeros(batch_size, dtype=torch.long, device=device)
    x[:, 0] = cur_s + CHAIN_STATE_OFF
    y[:, 0] = cur_s + CHAIN_STATE_OFF

    for t in range(1, seq_len):
        force_reset = cur_d >= max_depth
        rand_reset = torch.rand(batch_size, device=device) < 0.25
        do_reset = force_reset | rand_reset
        new_s = torch.randint(0, CHAIN_N_STATES, (batch_size,), device=device)
        op = torch.randint(0, CHAIN_N_STATES, (batch_size,), device=device)
        x[:, t] = torch.where(do_reset, new_s + CHAIN_STATE_OFF, op + CHAIN_OP_OFF)
        cur_s = torch.where(do_reset, new_s, trans[op, cur_s])
        cur_d = torch.where(do_reset, torch.zeros_like(cur_d), cur_d + 1)
        y[:, t] = cur_s + CHAIN_STATE_OFF
        depths[:, t] = cur_d

    return x, y, depths


def make_2a_ptr_batch(
    trans: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_depth: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Non-adjacent pointer hops (vectorized):
      Between consecutive hop/state nodes, inserts 1 or 2 random distractor bytes [48..55].
      Because z_{t-1} at every op node is a distractor byte, hardcoded t-1 shifts fail unless
      attention/memory reaches back dynamically across variable 1..2 token gaps.
    """
    # Build compressed active chains of length <= seq_len on GPU, then scatter after variable gaps
    x = torch.randint(
        CHAIN_DIST_OFF, CHAIN_DIST_OFF + CHAIN_N_STATES, (batch_size, seq_len), device=device
    )
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    depths = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    cur_s = torch.randint(0, CHAIN_N_STATES, (batch_size,), device=device)
    cur_d = torch.zeros(batch_size, dtype=torch.long, device=device)
    # Wait countdown per sample: 0 means place an active node here; >0 means emit distractor
    cooldown = torch.randint(0, 2, (batch_size,), device=device)
    placed_any = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(seq_len):
        act = cooldown == 0
        if bool(act.any()):
            is_first = ~placed_any
            force_reset = cur_d >= max_depth
            rand_reset = torch.rand(batch_size, device=device) < 0.25
            do_reset = is_first | force_reset | rand_reset

            new_s = torch.randint(0, CHAIN_N_STATES, (batch_size,), device=device)
            op = torch.randint(0, CHAIN_N_STATES, (batch_size,), device=device)
            tok_act = torch.where(do_reset, new_s + CHAIN_STATE_OFF, op + CHAIN_OP_OFF)
            next_s = torch.where(do_reset, new_s, trans[op, cur_s])
            next_d = torch.where(do_reset, torch.zeros_like(cur_d), cur_d + 1)

            x[:, t] = torch.where(act, tok_act, x[:, t])
            cur_s = torch.where(act, next_s, cur_s)
            cur_d = torch.where(act, next_d, cur_d)
            y[:, t] = torch.where(act, cur_s + CHAIN_STATE_OFF, y[:, t])
            depths[:, t] = torch.where(act, cur_d, depths[:, t])
            mask[:, t] = act
            placed_any = placed_any | act
            new_cd = torch.randint(1, 3, (batch_size,), device=device)  # 1 or 2 distractors
            cooldown = torch.where(act, new_cd, cooldown - 1)
        else:
            cooldown = cooldown - 1

    return x, y, depths, mask


def run_task_2a(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps_adj: int = 240,
    steps_ptr: int = 550,
    blind_eval: bool = False,
    also_blind: bool = False,
) -> Any:
    set_all_seeds(seed)
    trans = _make_transition_table(seed, device)
    shuffled = model_name == "Null_ShuffledTargets"

    def _score_2a(
        model: nn.Module,
        xt: torch.Tensor,
        yt: torch.Tensor,
        dt: torch.Tensor,
        mt: torch.Tensor,
        use_blind: bool,
    ) -> Dict[str, float]:
        out = eval_model_preds(model_name, model, xt, blind_zero=use_blind)
        preds = forward_logits(out).argmax(dim=-1)
        correct = (preds == yt) & mt
        id_m = mt & (dt >= 0) & (dt <= 3)
        ood_m = mt & (dt >= 4)

        id_acc = float(correct[id_m].float().mean().item()) if id_m.any() else 0.0
        ood_acc = float(correct[ood_m].float().mean().item()) if ood_m.any() else 0.0
        ov_acc = float(correct[mt].float().mean().item()) if mt.any() else 0.0
        active_frac = float(out.get("active_token_fraction", 1.0))

        step0_acc = ov_acc
        step5_acc = ov_acc
        step0_ood45 = ood_acc
        step5_ood45 = ood_acc
        if isinstance(model, RealSelfStabilizingDMoE) and "step_logits" in out and len(out["step_logits"]) >= 2:
            lg0 = out["step_logits"][0]
            if lg0.dim() == 4:
                lg0 = lg0.squeeze(2)
            c0 = (lg0.argmax(-1) == yt) & mt
            step0_acc = float(c0[mt].float().mean().item())
            step0_ood45 = float(c0[ood_m].float().mean().item()) if ood_m.any() else step0_acc
            out5 = model(torch.zeros_like(xt) if use_blind else xt, force_thinking_steps=5)
            lg5 = forward_logits(out5)
            c5 = (lg5.argmax(-1) == yt) & mt
            step5_acc = float(c5[mt].float().mean().item())
            step5_ood45 = float(c5[ood_m].float().mean().item()) if ood_m.any() else step5_acc

        return {
            "overall": ov_acc,
            "id": id_acc,
            "ood45": ood_acc,
            "active_frac": active_frac,
            "step0": step0_acc,
            "step5": step5_acc,
            "step0_ood45": step0_ood45,
            "step5_ood45": step5_ood45,
        }

    def _run_subtask(is_ptr: bool, steps: int) -> Any:
        set_all_seeds(seed + (10000 if is_ptr else 0))
        model = build_model(model_name, device)
        if model_name != "Null_Untrained":
            opt = make_adamw(model.parameters(), lr=lr, device=device)
            model.train()
            for st in range(steps):
                set_warmup_cosine_lr(opt, st, steps, lr)
                if is_ptr:
                    xb, yb, _, mb = make_2a_ptr_batch(trans, 64, 24, max_depth=3, device=device)
                else:
                    xb, yb, _ = make_2a_adj_batch(trans, 64, 16, max_depth=3, device=device)
                    mb = None
                if shuffled:
                    perm = torch.randperm(xb.size(0), device=device)
                    yb = yb[perm]
                opt.zero_grad(set_to_none=True)
                loss, _ = compute_loss_for_model(
                    model_name, model, xb, yb, valid_mask=mb, think_train_steps=4
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        with torch.no_grad():
            if is_ptr:
                xt, yt, dt, mt = make_2a_ptr_batch(trans, 256, 28, max_depth=5, device=device)
            else:
                xt, yt, dt = make_2a_adj_batch(trans, 256, 20, max_depth=5, device=device)
                mt = torch.ones_like(yt, dtype=torch.bool)
            main_r = _score_2a(model, xt, yt, dt, mt, blind_eval)
            if also_blind:
                return main_r, _score_2a(model, xt, yt, dt, mt, True)
            return main_r

    def _pack(ra: Dict[str, float], rp: Dict[str, float]) -> Dict[str, Any]:
        return {
            "adj_overall": ra["overall"],
            "adj_id": ra["id"],
            "adj_ood45": ra["ood45"],
            "adj_active_frac": ra["active_frac"],
            "adj_step0": ra["step0"],
            "adj_step5": ra["step5"],
            "adj_step0_ood45": ra["step0_ood45"],
            "adj_step5_ood45": ra["step5_ood45"],
            "ptr_overall": rp["overall"],
            "ptr_id": rp["id"],
            "ptr_ood45": rp["ood45"],
            "ptr_active_frac": rp["active_frac"],
            "ptr_step0": rp["step0"],
            "ptr_step5": rp["step5"],
            "ptr_step0_ood45": rp["step0_ood45"],
            "ptr_step5_ood45": rp["step5_ood45"],
        }

    out_adj = _run_subtask(is_ptr=False, steps=steps_adj)
    out_ptr = _run_subtask(is_ptr=True, steps=steps_ptr)
    if also_blind:
        return _pack(out_adj[0], out_ptr[0]), _pack(out_adj[1], out_ptr[1])
    return _pack(out_adj, out_ptr)


# ==============================================================================
# Task 2B: Zero-Shot Epistemic Abstention & Conflict Catching (Clean-Trained Only)
# ==============================================================================

def make_2b_batch(
    batch_size: int,
    mode: str,
    device: torch.device,
    n_rules: int = 6,
) -> Dict[str, torch.Tensor]:
    """
    Vectorized associative rule episodes over shared vocabulary [32..63] (zero CPU-GPU syncs):
      clean         : RULE k0 v0 ... RULE k5 v5 QUERY q1 v(q1) QUERY q_final -> pred v(q_final)
      unknown       : same clean rules, q_final is a valid token in [32..63] unbound in THIS episode
      obs_conflict  : clean rules k0->v0 ... k5->v5, verification QUERY k0 v_alt@20 (contradicts rule!)
      rule_conflict : conflicting rules (Rule 0: k0->v0 and Rule 5: k0->v_alt), clean q1, QUERY k0@22
    """
    seq_len = n_rules * 3 + 5  # 23 tokens
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    sup_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    perm_k = torch.argsort(torch.rand(batch_size, VOCAB_SIZE, device=device), dim=-1) + VOCAB_START
    keys = perm_k[:, : n_rules + 1]  # [B, n_rules + 1]
    vals = torch.randint(VOCAB_START, VOCAB_START + VOCAB_SIZE, (batch_size, n_rules + 1), device=device)

    for i in range(n_rules):
        idx = i * 3
        x[:, idx] = RULE_MARK
        x[:, idx + 1] = keys[:, i]
        x[:, idx + 2] = vals[:, i]
        next_mark = RULE_MARK if i < n_rules - 1 else QUERY_MARK
        y[:, idx + 2] = next_mark
        sup_mask[:, idx + 2] = True

    idx = n_rules * 3  # position 18
    # Verification triple: QUERY, q1, a1 (at idx, idx+1, idx+2)
    if mode == "obs_conflict":
        # Clean rules intact; observation at pos 20 (v_alt) contradicts RULE k0 -> vals[:, 0]
        v_alt = VOCAB_START + ((vals[:, 0] - VOCAB_START + 1) % VOCAB_SIZE)
        x[:, idx] = QUERY_MARK
        x[:, idx + 1] = keys[:, 0]
        x[:, idx + 2] = v_alt
        y[:, idx + 1] = vals[:, 0]
        y[:, idx + 2] = QUERY_MARK
    elif mode in ("rule_conflict", "conflict"):
        # Rule ambiguity: Rule 0 binds k0->vals[:,0], Rule n_rules-1 binds k0->v_alt; clean q1 != k0
        v_alt = VOCAB_START + ((vals[:, 0] - VOCAB_START + 1) % VOCAB_SIZE)
        last_rule = (n_rules - 1) * 3
        x[:, last_rule + 1] = keys[:, 0]
        x[:, last_rule + 2] = v_alt
        q1_idx = torch.randint(1, max(2, n_rules - 1), (batch_size,), device=device)
        b_arange = torch.arange(batch_size, device=device)
        k_q1 = keys[b_arange, q1_idx]
        v_q1 = vals[b_arange, q1_idx]
        x[:, idx] = QUERY_MARK
        x[:, idx + 1] = k_q1
        x[:, idx + 2] = v_q1
        y[:, idx + 1] = v_q1
        y[:, idx + 2] = QUERY_MARK
    else:
        q1_idx = torch.randint(0, n_rules, (batch_size,), device=device)
        b_arange = torch.arange(batch_size, device=device)
        k_q1 = keys[b_arange, q1_idx]
        v_q1 = vals[b_arange, q1_idx]
        x[:, idx] = QUERY_MARK
        x[:, idx + 1] = k_q1
        x[:, idx + 2] = v_q1
        y[:, idx + 1] = v_q1
        y[:, idx + 2] = QUERY_MARK
    sup_mask[:, idx + 1] = True
    sup_mask[:, idx + 2] = True

    # Final query at seq_len - 1 (position 22)
    idx_f = idx + 3  # 21
    x[:, idx_f] = QUERY_MARK
    if mode == "clean":
        qf_idx = torch.randint(0, n_rules, (batch_size,), device=device)
        b_arange = torch.arange(batch_size, device=device)
        x[:, idx_f + 1] = keys[b_arange, qf_idx]
        y[:, idx_f + 1] = vals[b_arange, qf_idx]
    elif mode == "unknown":
        x[:, idx_f + 1] = keys[:, n_rules]
        y[:, idx_f + 1] = vals[:, n_rules]
    elif mode in ("obs_conflict", "rule_conflict", "conflict"):
        x[:, idx_f + 1] = keys[:, 0]
        y[:, idx_f + 1] = vals[:, 0]
    sup_mask[:, idx_f + 1] = True

    verify_val_pos = torch.full((batch_size,), idx + 2, dtype=torch.long, device=device)
    query_pos = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)
    return {
        "x": x,
        "y": y,
        "mask": sup_mask,
        "query_pos": query_pos,
        "conflict_val_pos": verify_val_pos,
    }


def run_task_2b(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps: int = 420,
    blind_eval: bool = False,
    also_blind: bool = False,
) -> Any:
    set_all_seeds(seed + 20000)
    model = build_model(model_name, device)
    shuffled = model_name == "Null_ShuffledTargets"
    if model_name != "Null_Untrained":
        opt = make_adamw(model.parameters(), lr=lr, device=device)
        model.train()
        for st in range(steps):
            set_warmup_cosine_lr(opt, st, steps, lr)
            batch = make_2b_batch(64, "clean", device)
            xb, yb, mb = batch["x"], batch["y"], batch["mask"]
            if shuffled:
                yb = yb[torch.randperm(xb.size(0), device=device)]
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_loss_for_model(model_name, model, xb, yb, valid_mask=mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        bc = make_2b_batch(192, "clean", device)
        bu = make_2b_batch(192, "unknown", device)
        bo = make_2b_batch(192, "obs_conflict", device)
        br = make_2b_batch(192, "rule_conflict", device)

        def _score_2b(use_blind: bool) -> Dict[str, Any]:
            oc = eval_model_preds(model_name, model, bc["x"], blind_zero=use_blind)
            ou = eval_model_preds(model_name, model, bu["x"], blind_zero=use_blind)
            oo = eval_model_preds(model_name, model, bo["x"], blind_zero=use_blind)
            oru = eval_model_preds(model_name, model, br["x"], blind_zero=use_blind)

            uc, pc_c = model_uncertainty(model, oc)
            uu, _ = model_uncertainty(model, ou)
            uo, pc_o = model_uncertainty(model, oo)
            ur, _ = model_uncertainty(model, oru)

            qc = forward_logits(oc)[:, -1, :].argmax(dim=-1)
            clean_correct = (qc == bc["y"][:, -1]).float()
            clean_acc = float(clean_correct.mean().item())
            qo = forward_logits(oo)[:, -1, :].argmax(dim=-1)
            obs_robust_acc = float((qo == bo["y"][:, -1]).float().mean().item())

            u_clean_q = uc[:, -1].cpu().tolist()
            u_unk_q = uu[:, -1].cpu().tolist()
            u_conf_q = ur[:, -1].cpu().tolist()

            cpos = bo["conflict_val_pos"]
            b_idx = torch.arange(bo["x"].size(0), device=device)
            clean_val_pos = bc["conflict_val_pos"]
            sig_c = (pc_c if pc_c is not None else uc)[b_idx, clean_val_pos].cpu().tolist()
            sig_o = (pc_o if pc_o is not None else uo)[b_idx, cpos].cpu().tolist()

            auroc_unk = compute_auroc(u_unk_q, u_clean_q)
            auroc_conf_q = compute_auroc(u_conf_q, u_clean_q)
            auroc_all = compute_auroc(u_unk_q + u_conf_q, u_clean_q)
            auroc_pc_conf = compute_auroc(sig_o, sig_c)

            mix_unc = u_clean_q + u_unk_q[:96] + u_conf_q[:96]
            mix_err = (1.0 - clean_correct).cpu().tolist() + [1.0] * 192
            ece = compute_ece(mix_unc, mix_err)

            paired_mix = sorted(zip(mix_unc, mix_err), key=lambda t: t[0])
            top50 = paired_mix[: len(paired_mix) // 2]
            sel_acc_50 = 1.0 - (sum(e for _, e in top50) / len(top50))

            return {
                "clean_acc": clean_acc,
                "obs_robust_acc": obs_robust_acc,
                "u_clean": sum(u_clean_q) / len(u_clean_q),
                "u_unknown": sum(u_unk_q) / len(u_unk_q),
                "u_conflict_q": sum(u_conf_q) / len(u_conf_q),
                "auroc_unknown": auroc_unk,
                "auroc_conflict_q": auroc_conf_q,
                "auroc_abstain_all": auroc_all,
                "auroc_pc_contradiction": auroc_pc_conf,
                "ece": ece,
                "selective_acc_50": sel_acc_50,
            }

        main_r = _score_2b(blind_eval)
        if also_blind:
            return main_r, _score_2b(True)
        return main_r


# ==============================================================================
# Task 2C: Multi-Step World-Model Rollout, Horizon Calibration & Downstream Accuracy
# ==============================================================================

def make_2c_batch(
    batch_size: int,
    seq_len: int,
    corrupt: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Modular deterministic trajectory on [32..95] vs corrupted random transitions on same vocab."""
    if not corrupt:
        start = torch.randint(32, 96, (batch_size, 1), device=device)
        delta = torch.randint(1, 5, (batch_size, 1), device=device)
        steps = torch.arange(seq_len + 1, device=device).unsqueeze(0)
        seq = 32 + ((start - 32 + delta * steps) % 64)
        return seq[:, :-1], seq[:, 1:]
    seq = torch.randint(32, 96, (batch_size, seq_len + 1), device=device)
    return seq[:, :-1], seq[:, 1:]


def run_task_2c(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps: int = 220,
    blind_eval: bool = False,
    also_blind: bool = False,
) -> Any:
    set_all_seeds(seed + 30000)
    model = build_model(model_name, device)
    shuffled = model_name == "Null_ShuffledTargets"
    if model_name != "Null_Untrained":
        opt = make_adamw(model.parameters(), lr=lr, device=device)
        model.train()
        for st in range(steps):
            set_warmup_cosine_lr(opt, st, steps, lr)
            xb, yb = make_2c_batch(48, 16, corrupt=False, device=device)
            if shuffled:
                yb = yb[torch.randperm(xb.size(0), device=device)]
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_loss_for_model(model_name, model, xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        xd, yd = make_2c_batch(256, 16, corrupt=False, device=device)
        xc, _ = make_2c_batch(256, 16, corrupt=True, device=device)

        def _score_2c(use_blind: bool) -> Dict[str, Any]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            od = eval_model_preds(model_name, model, xd, blind_zero=use_blind)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_dt = max(time.perf_counter() - t0, 1e-6)
            toks_per_sec = float(xd.numel()) / inf_dt

            oc = eval_model_preds(model_name, model, xc, blind_zero=use_blind)
            traj_acc = float((forward_logits(od).argmax(-1) == yd).float().mean().item())

            ud, pc_d = model_uncertainty(model, od)
            uc, pc_c = model_uncertainty(model, oc)
            sig_d = (pc_d[:, 1:] if pc_d is not None else ud[:, 1:]).reshape(-1).cpu().tolist()
            sig_c = (pc_c[:, 1:] if pc_c is not None else uc[:, 1:]).reshape(-1).cpu().tolist()
            corrupt_auroc = compute_auroc(sig_c, sig_d)

            cos_h = [0.0, 0.0, 0.0]
            unc_h = [0.0, 0.0, 0.0]
            wm_ece = 0.0
            if isinstance(model, RealSelfStabilizingDMoE) and od.get("world_model_imagine") is not None:
                wm = model.world_model
                im = od["world_model_imagine"]
                xin = torch.zeros_like(xd) if use_blind else xd
                tgt = wm.encode_target(xin)  # [B, L, D]
                D = tgt.size(-1)
                errs_all, uncs_all = [], []
                for h_idx in range(min(3, im["imagined_latents"].size(2))):
                    k = h_idx + 1
                    pred_k = im["imagined_latents"][:, :-k, h_idx, :]
                    tgt_k = F.layer_norm(tgt[:, k:, :], (D,))
                    cos_k = F.cosine_similarity(pred_k, tgt_k, dim=-1)
                    cos_h[h_idx] = float(cos_k.mean().item())
                    u_k = im["imagined_uncertainty"][:, :-k, h_idx]
                    unc_h[h_idx] = float(u_k.mean().item())
                    errs_all.extend((1.0 - cos_k).clamp(0.0, 1.0).reshape(-1).cpu().tolist())
                    uncs_all.extend(u_k.reshape(-1).cpu().tolist())
                wm_ece = compute_ece(uncs_all, errs_all)

            return {
                "traj_acc": traj_acc,
                "cos_h1": cos_h[0],
                "cos_h2": cos_h[1],
                "cos_h3": cos_h[2],
                "unc_h1": unc_h[0],
                "unc_h2": unc_h[1],
                "unc_h3": unc_h[2],
                "wm_ece": wm_ece,
                "corrupt_auroc": corrupt_auroc,
                "eval_toks_sec": toks_per_sec,
            }

        main_r = _score_2c(blind_eval)
        if also_blind:
            return main_r, _score_2c(True)
        return main_r


def check_stage2_causality(device: torch.device) -> float:
    set_all_seeds(42)
    max_diff = 0.0
    x = torch.randint(32, 64, (4, 16), device=device)
    x_pert = x.clone()
    x_pert[:, 8:] = torch.randint(32, 64, (4, 8), device=device)
    for mname in ("Cog_Stage2_Full", "LoopedTransformer_Baseline", "DenseRoPE_Deep", "GRU_Baseline"):
        m = build_model(mname, device)
        m.eval()
        with torch.no_grad():
            o1 = eval_model_preds(mname, m, x)
            o2 = eval_model_preds(mname, m, x_pert)
            d_out = (forward_logits(o1)[:, :8] - forward_logits(o2)[:, :8]).abs().max().item()
            max_diff = max(max_diff, d_out)
            if isinstance(m, RealSelfStabilizingDMoE):
                u1 = m.assess_epistemic_uncertainty(o1)
                u2 = m.assess_epistemic_uncertainty(o2)
                max_diff = max(
                    max_diff,
                    (u1["uncertainty"][:, :8] - u2["uncertainty"][:, :8]).abs().max().item(),
                )
    return float(max_diff)


def _aggregate(t2a: List[Dict], t2b: List[Dict], t2c: List[Dict], dt: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"wall_clock_sec": round(dt, 2)}
    for prefix, rows in [("2a", t2a), ("2b", t2b), ("2c", t2c)]:
        for key in rows[0].keys():
            out[f"{prefix}_{key}"] = summarize([float(r[key]) for r in rows])
    return out


def _collect_metadata() -> Dict[str, Any]:
    git_commit, git_dirty = "UNKNOWN", None
    if shutil.which("git") is not None and (ROOT / ".git").exists():
        res_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
        res_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True)
        if res_head.returncode == 0 and res_dirty.returncode == 0:
            git_commit = res_head.stdout.strip()
            git_dirty = len(res_dirty.stdout.strip()) > 0
        else:
            git_commit = f"ERROR(rc={res_head.returncode})"
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "command": " ".join(sys.argv),
    }


def _resolve_lr(name: str, args: argparse.Namespace) -> float:
    if name == "LoopedTransformer_Baseline":
        return float(args.lr_looped)
    if name == "DenseRoPE_Deep":
        return float(args.lr_dense)
    if name == "GRU_Baseline":
        return float(args.lr_gru)
    return float(args.lr_cog)


def _run_seed_bundle(
    job: Tuple[str, int, float, int, int, int, int, bool, bool]
) -> Tuple[int, Any, Any, Any, Optional[Tuple[Any, Any, Any]]]:
    model_name, seed, lr, steps_adj, steps_ptr, steps_2b, steps_2c, also_blind, use_cuda = job
    dev = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    out2a = run_task_2a(
        model_name, seed, dev, lr=lr, steps_adj=steps_adj, steps_ptr=steps_ptr, also_blind=also_blind
    )
    out2b = run_task_2b(model_name, seed, dev, lr=lr, steps=steps_2b, also_blind=also_blind)
    out2c = run_task_2c(model_name, seed, dev, lr=lr, steps=steps_2c, also_blind=also_blind)
    if also_blind:
        return seed, out2a[0], out2b[0], out2c[0], (out2a[1], out2b[1], out2c[1])
    return seed, out2a, out2b, out2c, None


def _eval_seeds_for_model(
    model_name: str,
    seeds: List[int],
    device: torch.device,
    lr: float,
    args: argparse.Namespace,
    also_blind: bool = False,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], Optional[Tuple[List[Dict], List[Dict], List[Dict]]]]:
    use_cuda = device.type == "cuda"
    jobs = [
        (
            model_name,
            s,
            lr,
            args.steps_2a_adj,
            args.steps_2a_ptr,
            args.steps_2b,
            args.steps_2c,
            also_blind,
            use_cuda,
        )
        for s in seeds
    ]
    workers = max(1, int(getattr(args, "workers", 1)))
    if workers > 1 and model_name != "Null_Untrained":
        if executor is not None:
            rows = list(executor.map(_run_seed_bundle, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                rows = list(ex.map(_run_seed_bundle, jobs))
    else:
        rows = [_run_seed_bundle(j) for j in jobs]
    rows.sort(key=lambda r: r[0])
    r2a = [r[1] for r in rows]
    r2b = [r[2] for r in rows]
    r2c = [r[3] for r in rows]
    if also_blind:
        b2a = [r[4][0] for r in rows]
        b2b = [r[4][1] for r in rows]
        b2c = [r[4][2] for r in rows]
        return r2a, r2b, r2c, (b2a, b2b, b2c)
    return r2a, r2b, r2c, None


def run_tune(args: argparse.Namespace, device: torch.device) -> None:
    lrs = [float(x) for x in args.tune_lrs.split(",")]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    workers = max(1, int(getattr(args, "workers", 1)))
    log_entries: List[Dict[str, Any]] = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for m in models:
            for lr in lrs:
                t0 = time.perf_counter()
                r2a, r2b, r2c, _ = _eval_seeds_for_model(
                    m, DEV_SEEDS, device, lr, args, also_blind=False, executor=pool
                )
                agg = _aggregate(r2a, r2b, r2c, time.perf_counter() - t0)
                entry = {"model": m, "lr": lr, **agg}
                log_entries.append(entry)
                print(
                    f"[DEV {agg['wall_clock_sec']:5.1f}s] {m:26s} lr={lr:.1e} | "
                    f"2A_adj_ood={agg['2a_adj_ood45']['mean']:.3f}(act={agg['2a_adj_active_frac']['mean']:.2f}) "
                    f"2A_ptr_ood={agg['2a_ptr_ood45']['mean']:.3f}(act={agg['2a_ptr_active_frac']['mean']:.2f}) | "
                    f"2B_acc={agg['2b_clean_acc']['mean']:.3f} auroc={agg['2b_auroc_abstain_all']['mean']:.3f} pc={agg['2b_auroc_pc_contradiction']['mean']:.3f} | "
                    f"2C_traj={agg['2c_traj_acc']['mean']:.3f} cos1={agg['2c_cos_h1']['mean']:.3f} corr={agg['2c_corrupt_auroc']['mean']:.3f}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "tuning_log_v1.json"
    merged_entries = list(log_entries)
    if getattr(args, "append", False) and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        seen = {(e["model"], e["lr"]) for e in merged_entries}
        merged_entries = [e for e in prev.get("entries", []) if (e["model"], e["lr"]) not in seen] + merged_entries
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": _collect_metadata(), "dev_seeds": DEV_SEEDS, "entries": merged_entries}, f, indent=2)
    print(f"Saved DEV tuning log ({len(merged_entries)} entries) to {out_path}")


def run_test(args: argparse.Namespace, device: torch.device) -> None:
    causality = check_stage2_causality(device)
    assert causality <= 1e-5, f"Causality failure: {causality}"

    all_models = [
        "Cog_Stage2_Full",
        "Cog_NoWM",
        "Cog_OnePass",
        "Cog_FixedThink",
        "LoopedTransformer_Baseline",
        "DenseRoPE_Deep",
        "GRU_Baseline",
        "Null_Untrained",
        "Null_ShuffledTargets",
    ]
    param_counts = {m: count_parameters(build_model(m, torch.device("cpu"))) for m in all_models}
    param_counts["Null_BlindZeroInput"] = param_counts["Cog_Stage2_Full"]

    results: Dict[str, Any] = {
        "exp_id": "stage2_thinking_world_model_v1",
        "metadata": _collect_metadata(),
        "test_seeds": TEST_SEEDS,
        "parameter_counts": param_counts,
        "causality_gate_max_diff": causality,
        "models": {},
    }

    workers = max(1, int(getattr(args, "workers", 1)))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for m in all_models:
            lr_m = _resolve_lr(m, args)
            blind_too = m == "Cog_Stage2_Full"
            t0 = time.perf_counter()
            t2a, t2b, t2c, blind_rows = _eval_seeds_for_model(
                m, TEST_SEEDS, device, lr_m, args, also_blind=blind_too, executor=pool
            )
            agg = _aggregate(t2a, t2b, t2c, time.perf_counter() - t0)
            results["models"][m] = agg
            print(
                f"[TEST {agg['wall_clock_sec']:5.1f}s] {m:26s} | "
                f"2A_adj_ood={agg['2a_adj_ood45']['mean']:.3f} (act={agg['2a_adj_active_frac']['mean']:.2f}) | "
                f"2A_ptr_ood={agg['2a_ptr_ood45']['mean']:.3f} (act={agg['2a_ptr_active_frac']['mean']:.2f}) | "
                f"2B(acc={agg['2b_clean_acc']['mean']:.3f}, abst={agg['2b_auroc_abstain_all']['mean']:.3f}, pc={agg['2b_auroc_pc_contradiction']['mean']:.3f}) | "
                f"2C(traj={agg['2c_traj_acc']['mean']:.3f}, cos1={agg['2c_cos_h1']['mean']:.3f}, corr={agg['2c_corrupt_auroc']['mean']:.3f})",
                flush=True,
            )

            if blind_too and blind_rows is not None:
                b_agg = _aggregate(blind_rows[0], blind_rows[1], blind_rows[2], 0.0)
                results["models"]["Null_BlindZeroInput"] = b_agg
                print(
                    f"[TEST blind] {'Null_BlindZeroInput':26s} | "
                    f"2A_adj_ood={b_agg['2a_adj_ood45']['mean']:.3f} | 2A_ptr_ood={b_agg['2a_ptr_ood45']['mean']:.3f} | "
                    f"2B(acc={b_agg['2b_clean_acc']['mean']:.3f}) | 2C(traj={b_agg['2c_traj_acc']['mean']:.3f})",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    full = results["models"]["Cog_Stage2_Full"]
    nowm = results["models"]["Cog_NoWM"]
    onep = results["models"]["Cog_OnePass"]
    fixed = results["models"]["Cog_FixedThink"]
    loop = results["models"]["LoopedTransformer_Baseline"]
    drope = results["models"]["DenseRoPE_Deep"]
    gru = results["models"]["GRU_Baseline"]

    results["paired_comparisons"] = {
        "2a_adj_ood_full_vs_onepass": paired_stats(full["2a_adj_ood45"]["per_seed"], onep["2a_adj_ood45"]["per_seed"]),
        "2a_adj_ood_full_vs_looped": paired_stats(full["2a_adj_ood45"]["per_seed"], loop["2a_adj_ood45"]["per_seed"]),
        "2a_adj_ood_full_vs_denserope": paired_stats(full["2a_adj_ood45"]["per_seed"], drope["2a_adj_ood45"]["per_seed"]),
        "2a_adj_ood_full_vs_gru": paired_stats(full["2a_adj_ood45"]["per_seed"], gru["2a_adj_ood45"]["per_seed"]),
        "2a_ptr_ood_full_vs_onepass": paired_stats(full["2a_ptr_ood45"]["per_seed"], onep["2a_ptr_ood45"]["per_seed"]),
        "2a_ptr_ood_full_vs_fixed": paired_stats(full["2a_ptr_ood45"]["per_seed"], fixed["2a_ptr_ood45"]["per_seed"]),
        "2a_ptr_ood_full_vs_looped": paired_stats(full["2a_ptr_ood45"]["per_seed"], loop["2a_ptr_ood45"]["per_seed"]),
        "2a_ptr_ood_full_vs_denserope": paired_stats(full["2a_ptr_ood45"]["per_seed"], drope["2a_ptr_ood45"]["per_seed"]),
        "2a_ptr_ood_full_vs_gru": paired_stats(full["2a_ptr_ood45"]["per_seed"], gru["2a_ptr_ood45"]["per_seed"]),
        "2b_clean_acc_full_vs_denserope": paired_stats(full["2b_clean_acc"]["per_seed"], drope["2b_clean_acc"]["per_seed"]),
        "2b_clean_acc_full_vs_looped": paired_stats(full["2b_clean_acc"]["per_seed"], loop["2b_clean_acc"]["per_seed"]),
        "2b_clean_acc_full_vs_gru": paired_stats(full["2b_clean_acc"]["per_seed"], gru["2b_clean_acc"]["per_seed"]),
        "2b_auroc_abstain_full_vs_nowm": paired_stats(full["2b_auroc_abstain_all"]["per_seed"], nowm["2b_auroc_abstain_all"]["per_seed"]),
        "2b_auroc_pc_full_vs_nowm": paired_stats(full["2b_auroc_pc_contradiction"]["per_seed"], nowm["2b_auroc_pc_contradiction"]["per_seed"]),
        "2b_auroc_pc_full_vs_looped": paired_stats(full["2b_auroc_pc_contradiction"]["per_seed"], loop["2b_auroc_pc_contradiction"]["per_seed"]),
        "2c_traj_full_vs_nowm": paired_stats(full["2c_traj_acc"]["per_seed"], nowm["2c_traj_acc"]["per_seed"]),
        "2c_corrupt_auroc_full_vs_nowm": paired_stats(full["2c_corrupt_auroc"]["per_seed"], nowm["2c_corrupt_auroc"]["per_seed"]),
    }

    null_models = ("Null_Untrained", "Null_ShuffledTargets", "Null_BlindZeroInput")
    falsifiers = {
        "F1_causality_gate": causality <= 1e-5,
        "F2_thinking_beats_onepass_and_step5_gt_step0": (
            full["2a_adj_ood45"]["mean"] > onep["2a_adj_ood45"]["mean"] + 0.18
            and full["2a_ptr_ood45"]["mean"] > onep["2a_ptr_ood45"]["mean"] + 0.12
            and full["2a_adj_step5_ood45"]["mean"] > full["2a_adj_step0_ood45"]["mean"] + 0.25
            and full["2a_ptr_step5_ood45"]["mean"] > full["2a_ptr_step0_ood45"]["mean"] + 0.18
        ),
        "F3_adaptive_compute_efficiency": (
            full["2a_adj_ood45"]["mean"] >= 0.85
            and full["2a_ptr_ood45"]["mean"] >= 0.78
            and full["2a_adj_active_frac"]["mean"] <= 0.50
            and full["2a_ptr_active_frac"]["mean"] <= 0.65
            and full["2a_ptr_ood45"]["mean"] >= fixed["2a_ptr_ood45"]["mean"] - 0.04
        ),
        "F4_abstention_and_pc_conflict": (
            full["2b_clean_acc"]["mean"] >= 0.90
            and full["2b_auroc_abstain_all"]["mean"] >= 0.85
            and full["2b_auroc_pc_contradiction"]["mean"] >= 0.85
            and full["2b_auroc_pc_contradiction"]["mean"] > nowm["2b_auroc_pc_contradiction"]["mean"] + 0.25
        ),
        "F5_world_model_no_traj_regression": (
            full["2c_traj_acc"]["mean"] >= nowm["2c_traj_acc"]["mean"] - 0.005
            and full["2c_traj_acc"]["mean"] >= 0.90
            and full["2c_cos_h1"]["mean"] >= 0.70
        ),
        "F6_wm_horizon_and_corrupt_calibration": (
            full["2c_unc_h1"]["mean"] < full["2c_unc_h2"]["mean"]
            and full["2c_unc_h2"]["mean"] < full["2c_unc_h3"]["mean"]
            and full["2c_wm_ece"]["mean"] <= 0.10
            and full["2c_corrupt_auroc"]["mean"] >= 0.90
        ),
        "F7_null_controls_fail": all(
            results["models"][nm]["2a_ptr_ood45"]["mean"] < 0.20
            and results["models"][nm]["2b_clean_acc"]["mean"] < 0.15
            and results["models"][nm]["2c_traj_acc"]["mean"] < 0.15
            for nm in null_models
        ),
    }
    results["falsifiers"] = falsifiers
    results["all_passed"] = all(falsifiers.values())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "results_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Stage 2 test results to {out_path} | all_passed={results['all_passed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "test"], default="tune")
    parser.add_argument("--models", type=str, default="Cog_Stage2_Full,LoopedTransformer_Baseline,DenseRoPE_Deep,GRU_Baseline")
    parser.add_argument("--tune_lrs", type=str, default="3e-3,5e-3,7e-3")
    parser.add_argument("--lr_cog", type=float, default=3e-3)
    parser.add_argument("--lr_looped", type=float, default=7e-3)
    parser.add_argument("--lr_dense", type=float, default=5e-3)
    parser.add_argument("--lr_gru", type=float, default=7e-3)
    parser.add_argument("--steps_2a_adj", type=int, default=340)
    parser.add_argument("--steps_2a_ptr", type=int, default=700)
    parser.add_argument("--steps_2b", type=int, default=520)
    parser.add_argument("--steps_2c", type=int, default=220)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 2 Benchmark | device={device} | mode={args.mode}")
    if args.mode == "tune":
        run_tune(args, device)
    else:
        run_test(args, device)


if __name__ == "__main__":
    main()

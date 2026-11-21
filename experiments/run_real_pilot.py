"""real_pilot_v1: end-to-end multi-domain REAL-data pilot for the Cognitive Dynamic MoE (byte level).

Run:  python experiments/run_real_pilot.py --mode {smoke,dev,test} --seeds 0 1 2 --out_name NAME
Everything printed/stored is produced by this script; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dmoe.continual import ContinualLearner
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.stage1_baselines import CausalDenseRoPEBaseline, count_parameters
from real_pilot_data import DOMAINS, RealCorpora, SEQ_LEN

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = ROOT / "results" / "real_pilot_v1"

# ----------------------------------------------------------------------------------------------
# FROZEN PROTOCOL (see results/real_pilot_v1/PREREG.md). Changing any of this is a NEW experiment.
# ----------------------------------------------------------------------------------------------
PROTO = dict(batch=16, lr=1e-3, wd=0.01, clip=1.0, warmup=30, wm_weight=0.15)

COG_CFG = dict(
    d_model=192, num_heads=6, d_ff=512, initial_layers=6, initial_experts=1, chunk_size=1,
    enable_memory=True, num_ltm_slots=32, enable_three_level_attn=True, local_window=16, chunk_stride=8,
    enable_layer_stm=True, signed_beta=True, use_rope=True, causal_attn=True, auto_write_ltm=False,
    enable_thinking=True, max_thinking_steps=2, skip_threshold=0.08, theta_halt=0.10,
    enable_world_model=True, wm_rollout_horizon=1, enable_pc_rectification=True, pc_thresh=0.65,
    hybrid_mode=True, router_mode="causal_proto", num_router_protos=64, router_sim_threshold=0.55,
    auto_lifecycle=True, max_experts=6, min_mature_steps=40, novel_confirm_steps=3,
    novel_surprise_threshold=0.45, dormant_age=10 ** 9, prune_age=10 ** 9, route_during_train=True,
    use_trunk_routing=True, router_context_blend=0.20, extend_loss_margin=0.25,
    align_wm_target_with_encoder=True,
)

# v6 configuration: resolves GPM saturation, transition overwrite, mixed-stream collapse,
# dormant halting/PC gate, untrained LTM, and eliminates duplicate screen forward pass.
COG_V6_CFG = dict(
    **COG_CFG,
    enable_full_stack_adapters=True,
    adapter_rank=16,
    router_use_idf=True,
    router_center_multi=False,
    train_revisit_conf_margin=0.02,
    preserve_input_novel_on_extend=True,
    defer_on_novel_streak=True,
    enable_mixed_split=False,
    mixed_split_gap=0.18,
    max_coactive_experts=4,
    use_marginal_drop_halt=True,
    halt_use_energy_only=True,
    skip_final_lookahead=True,
    pc_mode="adaptive_rel",
    ltm_null_logit=1.5,
)
COG_V6_CFG["skip_threshold"] = 0.15
COG_V6_CFG["theta_halt"] = 0.22
COG_V6_CFG["min_surprise_drop"] = 0.008
COG_V6_CFG["pc_thresh"] = 0.24
COG_V6_CFG["layer_stm_every"] = 1

V6_PROTO = dict(
    step_rank_ratio=0.26,
    step_rank_decay=0.84,
    ep_ltm_every=6,
    ep_ltm_weight=0.45,
    quarantine_cap=4,
    gpm_cov_every=1,
    tier2_stm_every=4,
    tier2_ep_ltm_every=12,
    tier2_gpm_cov_every=2,
)
DENSE_LAYERS, DENSE_HEADS, DENSE_D = 6, 6, 192
REPLAY_WINDOWS = 512         # equal-memory replay budget (512 x 129 bytes ~ 66 KB)
REPLAY_FRAC = 0.25


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_meta() -> Dict[str, Any]:
    out: Dict[str, Any] = {"commit": "unknown", "dirty": None}
    git = shutil.which("git")
    if git is None:
        out["note"] = "git executable not found"
        return out
    head = subprocess.run([git, "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    st = subprocess.run([git, "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True)
    if head.returncode != 0 or st.returncode != 0:
        out["note"] = f"git failed: {head.stderr.strip()} {st.stderr.strip()}"
        return out
    out["commit"] = head.stdout.strip()
    out["dirty"] = bool(st.stdout.strip())
    return out


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def make_gpm(model: RealSelfStabilizingDMoE, is_v6: bool = False) -> NullSpaceProjector:
    kwargs: Dict[str, Any] = dict(
        energy_threshold=float(PROTO.get("gpm_energy", 0.97)), max_rank_ratio=0.85, track_embeddings=True,
        freeze_shared_norms_on_update=True, freeze_unprojected_shared=True,
    )
    if is_v6:
        kwargs["max_step_rank_ratio"] = float(V6_PROTO["step_rank_ratio"])
        kwargs["step_rank_decay"] = float(V6_PROTO["step_rank_decay"])
        kwargs["defer_cov_commit"] = True
    gpm = NullSpaceProjector(model, **kwargs)
    model.attach_null_space_projector(gpm)
    return gpm


def make_cog(kind: str, consolidate_after: Optional[int] = None) -> Tuple[RealSelfStabilizingDMoE, Optional[NullSpaceProjector], Optional[ContinualLearner]]:
    is_v6 = kind.startswith("Cog_v6")
    cfg = dict(COG_V6_CFG if is_v6 else COG_CFG)
    if kind == "Cog_SingleExpert":
        cfg["auto_lifecycle"] = False
    if kind in ("Cog_v6_Mixed", "Cog_v6_Tier2_Mixed"):
        cfg["enable_mixed_split"] = True
        cfg["router_center_multi"] = True
        cfg["defer_on_novel_streak"] = False
        cfg["preserve_input_novel_on_extend"] = False
        cfg["max_coactive_experts"] = 5
    if "Tier2" in kind:
        cfg["layer_stm_every"] = int(V6_PROTO["tier2_stm_every"])
    m = RealSelfStabilizingDMoE(**cfg).to(DEV)
    if m.lifecycle is not None:
        m.lifecycle.freeze_embed_after_first = True
    gpm = (
        make_gpm(m, is_v6=is_v6)
        if kind in ("Cog_Full", "Cog_Mixed", "Cog_SPGC", "Cog_v6", "Cog_v6_Mixed", "Cog_v6_NoLTM", "Cog_v6_Tier2", "Cog_v6_Tier2_Mixed")
        else None
    )
    learner: Optional[ContinualLearner] = None
    if kind == "Cog_SPGC":
        spgc_lr = float(PROTO.get("spgc_lr", 5e-4))
        learner = ContinualLearner(
            m, enable_spgc=True, enable_null_space=False, spgc_tau_skip=0.12,
            spgc_min_sources=2, spgc_max_bytes=16384, spgc_lr=spgc_lr,
            spgc_wd_trunk=PROTO["wd"], spgc_wd_fast=PROTO["wd"], spgc_wm_loss_weight=PROTO["wm_weight"],
            spgc_min_flush=1, spgc_store_samples=False,
            spgc_min_steps_before_skip=min(40, consolidate_after or 40),
            spgc_auto_consolidate_steps=consolidate_after,
            spgc_auto_consolidate_loss=1e9,
        )
        learner.null_space_projector = gpm
    return m, gpm, learner


def compute_grown_cog_params(n_experts: int, kind: str = "Cog_Full") -> int:
    """Estimate parameter count of RealSelfStabilizingDMoE after spawning `n_experts - 1` experts."""
    m, _, _ = make_cog(kind)
    with torch.no_grad():
        for _ in range(max(0, n_experts - 1)):
            for layer in m.layers:
                layer.moe.spawn_expert(source_idx=0, fresh_init=True)
                if hasattr(layer, "spawn_expert_adapters"):
                    layer.spawn_expert_adapters(source_idx=0, fresh_init=True)
            m.decoder.spawn_expert_head()
            if getattr(m, "thinker", None) is not None and hasattr(m.thinker, "spawn_expert_adapter"):
                m.thinker.spawn_expert_adapter(source_idx=0, fresh_init=True)
    n = count_parameters(m)
    del m
    return n


def make_dense(d_ff: int, layers: int = DENSE_LAYERS) -> nn.Module:
    return CausalDenseRoPEBaseline(d_model=DENSE_D, num_heads=DENSE_HEADS, d_ff=d_ff, num_layers=layers).to(DEV)


def match_dense_dff(target_params: int, layers: int = DENSE_LAYERS) -> int:
    """Smallest search over d_ff giving a Dense RoPE baseline with params closest to the target."""
    best, best_gap = 64, 10 ** 12
    for d_ff in range(64, 4096, 16):
        n = count_parameters(CausalDenseRoPEBaseline(d_model=DENSE_D, num_heads=DENSE_HEADS, d_ff=d_ff, num_layers=layers))
        gap = abs(n - target_params)
        if gap < best_gap:
            best, best_gap = d_ff, gap
    return best


def logits_of(out: Any) -> torch.Tensor:
    lg = out["output"] if isinstance(out, dict) else out
    return lg.reshape(lg.size(0), lg.size(1), -1)


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.numel() < 16:
        return None
    af, bf = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    ac, bc = af - af.mean(), bf - bf.mean()
    denom = float(ac.norm().item() * bc.norm().item())
    return float((ac @ bc).item() / denom) if denom > 1e-9 else None


def summarize_gpm(gpm: Optional[NullSpaceProjector]) -> Dict[str, Any]:
    if gpm is None:
        return {}
    ranks = {k: int(v.size(1)) for k, v in gpm.bases.items() if v is not None}
    ratios = []
    for k, v in gpm.bases.items():
        if v is not None and v.size(0) > 0:
            max_r = max(1, int(v.size(0) * gpm.max_rank_ratio))
            ratios.append(float(v.size(1)) / float(max_r))
    return {
        "num_projected": len(ranks),
        "mean_rank_ratio_of_cap": float(np.mean(ratios)) if ratios else 0.0,
        "max_rank_ratio_of_cap": float(np.max(ratios)) if ratios else 0.0,
        "at_cap_count": int(sum(1 for r in ratios if r >= 0.999)),
        "sample_ranks": dict(list(ranks.items())[:8]),
    }


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def eval_domain(
    model: nn.Module, windows: torch.Tensor, is_cog: bool, bs: int = 64,
    force_thinking_steps: Optional[int] = None, start_pos: int = 0,
) -> Dict[str, Any]:
    model.eval()
    nll = 0.0
    n_tok = 0
    top1 = 0
    routes: List[int] = []
    tok_steps: List[float] = []
    act_fracs: List[float] = []
    unk_fracs: List[float] = []
    pc_trig = 0.0
    pc_tok = 0
    pc_min = 1.0
    wm_cos: List[float] = []
    wm_l1: List[float] = []
    surp_buf: List[torch.Tensor] = []
    nll_buf: List[torch.Tensor] = []

    for i in range(0, windows.size(0), bs):
        w = windows[i:i + bs].to(DEV)
        x, y = w[:, :-1], w[:, 1:]
        if is_cog:
            out = model(x, update_stats=False, force_thinking_steps=force_thinking_steps)
        else:
            out = model(x)
        lg = logits_of(out)
        lg_s = lg[:, start_pos:, :]
        y_s = y[:, start_pos:]
        tok_nll = F.cross_entropy(lg_s.reshape(-1, lg_s.size(-1)), y_s.reshape(-1), reduction="none")
        nll += float(tok_nll.sum().item())
        top1 += int((lg_s.argmax(-1) == y_s).sum().item())
        n_tok += y_s.numel()
        if is_cog:
            routes.extend(out["assigned_experts"].detach().cpu().tolist())
            ts = out.get("token_steps")
            if ts is not None:
                tok_steps.append(float(ts[:, start_pos:].float().mean().item()))
            if "active_token_fraction" in out:
                act_fracs.append(float(out["active_token_fraction"]))
            unk = out.get("is_unknown")
            if unk is not None:
                unk_fracs.append(float(unk[:, start_pos:].float().mean().item()))
            g = out.get("pc_write_gate")
            if g is not None:
                g_s = g[:, start_pos:]
                pc_trig += float((g_s < 0.98).float().sum().item())
                pc_tok += g_s.numel()
                pc_min = min(pc_min, float(g_s.min().item()))
            wmi = out.get("world_model_imagine")
            if wmi is not None and getattr(model, "world_model", None) is not None and x.size(1) > 2:
                tgt = model.world_model.encode_target(x)
                pred_1 = wmi["imagined_latents"][:, :-1, 0, :]
                obs_1 = F.layer_norm(tgt[:, 1:, :], (tgt.size(-1),))
                wm_cos.append(float(F.cosine_similarity(pred_1, obs_1, dim=-1).mean().item()))
                wm_l1.append(float(F.smooth_l1_loss(pred_1, obs_1).item()))
            st = out.get("surprise_trace")
            if st:
                s0 = st[0][:, start_pos:].detach().cpu()
                full_nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), reduction="none").view_as(y)
                surp_buf.append(s0.reshape(-1))
                nll_buf.append(full_nll[:, start_pos:].detach().cpu().reshape(-1))

    res: Dict[str, Any] = {"bpb": nll / n_tok / math.log(2.0), "top1": top1 / n_tok}
    if is_cog:
        res["routes"] = routes
        res["mean_token_steps"] = float(np.mean(tok_steps)) if tok_steps else None
        res["active_token_frac"] = float(np.mean(act_fracs)) if act_fracs else None
        res["unknown_frac"] = float(np.mean(unk_fracs)) if unk_fracs else None
        res["pc_trigger_frac"] = pc_trig / max(1, pc_tok)
        res["pc_gate_min"] = pc_min if pc_tok > 0 else None
        res["wm_cos_1step"] = float(np.mean(wm_cos)) if wm_cos else None
        res["wm_l1_1step"] = float(np.mean(wm_l1)) if wm_l1 else None
        if surp_buf:
            scat = torch.cat(surp_buf)
            res["surprise_nll_corr"] = _pearson_corr(scat, torch.cat(nll_buf))
            qs = torch.quantile(scat.float(), torch.tensor([0.10, 0.50, 0.90])).tolist()
            res["surprise_p10_p50_p90"] = [float(v) for v in qs]
    return res


def eval_all(model: nn.Module, corp: RealCorpora, is_cog: bool) -> Dict[str, Dict[str, Any]]:
    return {d: eval_domain(model, corp.test_windows[d], is_cog) for d in DOMAINS}


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def lr_scale(step_in_stream: int) -> float:
    return min(1.0, (step_in_stream + 1) / PROTO["warmup"])


def set_lr(opt: torch.optim.Optimizer, scale: float) -> None:
    for g in opt.param_groups:
        g["lr"] = PROTO["lr"] * scale


def new_opt(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=PROTO["lr"], weight_decay=PROTO["wd"])


def dense_step(model: nn.Module, opt: torch.optim.Optimizer, w: torch.Tensor, log: Dict[str, Any]) -> Tuple[float, float]:
    model.train()
    x, y = w[:, :-1].to(DEV), w[:, 1:].to(DEV)
    opt.zero_grad(set_to_none=True)
    log["forward_passes"] += 1
    lg = logits_of(model(x))
    loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    log["backward_passes"] += 1
    nn.utils.clip_grad_norm_(model.parameters(), PROTO["clip"])
    opt.step()
    v = float(loss.item())
    return v, v


def cog_step(
    model: RealSelfStabilizingDMoE, gpm: Optional[NullSpaceProjector], opt_box: List[torch.optim.Optimizer],
    w: torch.Tensor, scale: float, log: Dict[str, Any], gstep: int, stream: str,
) -> Tuple[Optional[float], Optional[float]]:
    model.train()
    x, y = w[:, :-1].to(DEV), w[:, 1:].to(DEV)
    saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
    if gpm is not None:
        gpm.is_tracking = False
    if model.lifecycle is not None:
        log["forward_passes"] += 1
        with torch.no_grad():
            out0 = model(x, update_stats=True, targets=y)
        for ev in out0.get("lifecycle_events", []):
            log["events"].append({"step": gstep, "stream": stream, "event": ev})
            if ev.startswith("spawned_expert_"):
                opt_box[0] = new_opt(model)
        if model.lifecycle.should_defer_update:
            log["skipped_steps"] += 1
            return None, None
    if gpm is not None:
        gpm.is_tracking = True
    set_lr(opt_box[0], scale)
    opt_box[0].zero_grad(set_to_none=True)
    log["forward_passes"] += 1
    out = model(x, update_stats=(model.lifecycle is None), targets=y)
    lg = logits_of(out)
    ce_val = float(F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).detach().item())
    if model.enable_thinking and model.thinker is not None and "thinking_out" in out:
        loss = model.compute_total_loss(out, y, wm_weight=PROTO["wm_weight"])["loss"]
    else:
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        wml = out.get("world_model_loss")
        if wml is not None:
            loss = loss + PROTO["wm_weight"] * wml["loss"]
    loss.backward()
    log["backward_passes"] += 1
    if gpm is not None and gpm.p_null:
        gpm.project_gradients()
    nn.utils.clip_grad_norm_(model.parameters(), PROTO["clip"])
    opt_box[0].step()
    if saved is not None:
        gpm.project_parameter_deltas(saved)
    return float(loss.detach().item()), ce_val


def _cog_v6_backward_and_step(
    model: RealSelfStabilizingDMoE,
    gpm: Optional[NullSpaceProjector],
    opt_box: List[torch.optim.Optimizer],
    saved: Optional[Dict[str, torch.Tensor]],
    out: Dict[str, Any],
    w: torch.Tensor,
    y: torch.Tensor,
    scale: float,
    log: Dict[str, Any],
    enable_ep_ltm: bool,
    ep_every_override: Optional[int] = None,
) -> Tuple[float, float]:
    lg = logits_of(out)
    ce_val = float(F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).detach().item())
    if model.enable_thinking and model.thinker is not None and "thinking_out" in out:
        loss = model.compute_total_loss(out, y, wm_weight=PROTO["wm_weight"])["loss"]
    else:
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        wml = out.get("world_model_loss")
        if wml is not None:
            loss = loss + PROTO["wm_weight"] * wml["loss"]

    ep_every = int(V6_PROTO["ep_ltm_every"] if ep_every_override is None else ep_every_override)
    if (
        enable_ep_ltm
        and ep_every > 0
        and (log.get("ep_steps", 0) % ep_every == 0)
        and model.enable_memory
        and model.memory is not None
        and out.get("memory_out") is not None
        and w.size(1) >= 96
    ):
        x_ep = w[:, :64].to(DEV)
        ep_cached: Dict[str, Any] = {
            "output": out["output"][:, :64].detach(),
            "memory_out": {
                k: v[:, :64].detach()
                for k, v in out["memory_out"].items()
                if torch.is_tensor(v) and v.dim() >= 2 and v.size(1) >= 64
            },
        }
        wmi = out.get("world_model_imagine")
        if wmi is not None and "predictive_coding_error" in wmi and wmi["predictive_coding_error"].size(1) >= 64:
            ep_cached["world_model_imagine"] = {
                "predictive_coding_error": wmi["predictive_coding_error"][:, :64].detach()
            }
        ep_mem = model.extract_episode_kv(x_ep, top_k=8, min_nll=0.5, cached_out=ep_cached)
        if ep_mem is not None and bool(ep_mem["ep_mask"].any().item()):
            q_x = w[:, 64:-1].to(DEV)
            q_y = w[:, 65:].to(DEV)
            out_ep = model(q_x, update_stats=False, targets=q_y, episode_memory=ep_mem)
            log["forward_passes"] += 1
            log["ep_ltm_passes"] = log.get("ep_ltm_passes", 0) + 1
            ep_lg = logits_of(out_ep)
            ep_loss = float(V6_PROTO["ep_ltm_weight"]) * F.cross_entropy(ep_lg.reshape(-1, 256), q_y.reshape(-1))
            loss = loss + ep_loss
    if enable_ep_ltm:
        log["ep_steps"] = log.get("ep_steps", 0) + 1

    set_lr(opt_box[0], scale)
    loss.backward()
    log["backward_passes"] += 1
    if gpm is not None and gpm.p_null:
        gpm.project_gradients()
    nn.utils.clip_grad_norm_(model.parameters(), PROTO["clip"])
    opt_box[0].step()
    if saved is not None:
        gpm.project_parameter_deltas(saved)
    if gpm is not None and getattr(gpm, "defer_cov_commit", False):
        gpm.commit_pending()
    return float(loss.detach().item()), ce_val


def cog_v6_step(
    model: RealSelfStabilizingDMoE, gpm: Optional[NullSpaceProjector], opt_box: List[torch.optim.Optimizer],
    w: torch.Tensor, scale: float, log: Dict[str, Any], gstep: int, stream: str,
    quarantine_buf: Optional[List[Tuple[torch.Tensor, float]]] = None,
    enable_ep_ltm: bool = True,
    ep_every_override: Optional[int] = None,
    gpm_cov_every: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    model.train()
    x, y = w[:, :-1].to(DEV), w[:, 1:].to(DEV)
    saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
    if gpm is not None:
        cov_stride = max(1, int(gpm_cov_every))
        gpm.is_tracking = (cov_stride == 1) or (gstep % cov_stride == 0) or (len(gpm.covariances) == 0)
    set_lr(opt_box[0], scale)
    opt_box[0].zero_grad(set_to_none=True)
    log["forward_passes"] += 1
    out = model(x, update_stats=True, targets=y)
    evs = out.get("lifecycle_events", [])
    spawned = False
    for ev in evs:
        log["events"].append({"step": gstep, "stream": stream, "event": ev})
        if ev.startswith("spawned_expert_") or ev.startswith("mixed_split_"):
            spawned = True

    if spawned:
        del out
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.discard_pending()
        opt_box[0] = new_opt(model)
        replay_items = (list(quarantine_buf) if quarantine_buf is not None else []) + [(w.detach().cpu(), float(scale))]
        if quarantine_buf is not None:
            quarantine_buf.clear()
        last_tot, last_ce = None, None
        for qb_w, qb_scale in replay_items:
            saved_q = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
            set_lr(opt_box[0], qb_scale)
            opt_box[0].zero_grad(set_to_none=True)
            log["forward_passes"] += 1
            qx, qy = qb_w[:, :-1].to(DEV), qb_w[:, 1:].to(DEV)
            q_out = model(qx, update_stats=True, targets=qy)
            last_tot, last_ce = _cog_v6_backward_and_step(
                model, gpm, opt_box, saved_q, q_out, qb_w, qy, qb_scale, log, enable_ep_ltm=False,
            )
        return last_tot, last_ce

    if model.lifecycle is not None and model.lifecycle.should_defer_update:
        del out
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.discard_pending()
        log["skipped_steps"] += 1
        if quarantine_buf is not None:
            quarantine_buf.append((w.detach().cpu(), float(scale)))
            if len(quarantine_buf) > int(V6_PROTO["quarantine_cap"]):
                quarantine_buf.pop(0)
        return None, None

    if quarantine_buf is not None and model.lifecycle is not None and model.lifecycle.novel_streak == 0:
        quarantine_buf.clear()
    return _cog_v6_backward_and_step(
        model, gpm, opt_box, saved, out, w, y, scale, log,
        enable_ep_ltm=enable_ep_ltm, ep_every_override=ep_every_override,
    )


def cog_spgc_step(
    model: RealSelfStabilizingDMoE, learner: ContinualLearner, w: torch.Tensor,
    log: Dict[str, Any], gstep: int, stream: str, local_st: int,
) -> Tuple[Optional[float], Optional[float]]:
    x, y = w[:, :-1].to(DEV), w[:, 1:].to(DEV)
    prev_n = len(model.layers[0].moe.experts)
    rep = learner.step(x, y, output_head="byte_chunk", source_id=f"{stream}_{local_st % 2}")
    for ev in getattr(model, "last_lifecycle_events", []):
        log["events"].append({"step": gstep, "stream": stream, "event": ev})
    if len(model.layers[0].moe.experts) != prev_n:
        log["events"].append({"step": gstep, "stream": stream, "event": f"spgc_experts_now_{len(model.layers[0].moe.experts)}"})
    in_acquire = getattr(learner.optimizer, "mode", "") == "acquire"
    early = in_acquire and (learner.global_step - 1) < learner.spgc_min_steps_before_skip
    log["forward_passes"] += 1 if early else (1 if rep["skipped_backward"] else 2)
    if rep["skipped_backward"]:
        log["skipped_steps"] += 1
        return None, None
    log["backward_passes"] += 1
    return float(rep["total_loss"]), float(rep["task_loss"])


class Reservoir:
    def __init__(self, cap: int, rng: np.random.Generator):
        self.cap, self.rng, self.seen = cap, rng, 0
        self.buf: List[np.ndarray] = []

    def add(self, batch: torch.Tensor) -> None:
        for row in batch.numpy().astype(np.uint8):
            self.seen += 1
            if len(self.buf) < self.cap:
                self.buf.append(row)
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.cap:
                    self.buf[j] = row

    def sample(self, n: int) -> Optional[torch.Tensor]:
        if not self.buf:
            return None
        idx = self.rng.integers(0, len(self.buf), size=n)
        return torch.from_numpy(np.stack([self.buf[i] for i in idx]).astype(np.int64))


def run_sequential(
    arm: str, seed: int, corp: RealCorpora, steps: int, dense_dff: int,
) -> Tuple[Dict[str, Any], RealSelfStabilizingDMoE | nn.Module, Optional[NullSpaceProjector]]:
    set_all_seeds(seed)
    is_cog = arm.startswith("Cog")
    is_v6 = arm.startswith("Cog_v6")
    gpm, learner = None, None
    if is_cog:
        model, gpm, learner = make_cog(arm, consolidate_after=steps)
    else:
        model = make_dense(dense_dff)
    params_init = count_parameters(model)
    opt_box = [new_opt(model)]
    log: Dict[str, Any] = {
        "events": [], "skipped_steps": 0, "forward_passes": 0, "backward_passes": 0,
        "ep_ltm_passes": 0, "ep_steps": 0,
    }
    quarantine_buf: List[Tuple[torch.Tensor, float]] = []
    rep = Reservoir(REPLAY_WINDOWS, np.random.default_rng([seed, 99])) if arm == "Dense_Replay" else None
    matrix: List[Dict[str, Any]] = []
    train_loss_tail: Dict[str, float] = {}
    train_ce_tail: Dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gstep = 0
    for di, dom in enumerate(DOMAINS):
        gen = np.random.default_rng([seed, di, 1])
        losses: List[float] = []
        ces: List[float] = []
        for st in range(steps):
            fresh = corp.sample_train(dom, PROTO["batch"], gen)
            if rep is not None and rep.buf:
                n_rep = int(round(PROTO["batch"] * REPLAY_FRAC))
                w = torch.cat([fresh[: PROTO["batch"] - n_rep], rep.sample(n_rep)], dim=0)
            else:
                w = fresh
            if learner is not None:
                tot_l, ce_l = cog_spgc_step(model, learner, w, log, gstep, dom, st)
            elif is_v6:
                is_t2 = "Tier2" in arm
                tot_l, ce_l = cog_v6_step(
                    model, gpm, opt_box, w, lr_scale(st), log, gstep, dom,
                    quarantine_buf=quarantine_buf, enable_ep_ltm=(arm != "Cog_v6_NoLTM"),
                    ep_every_override=int(V6_PROTO["tier2_ep_ltm_every"]) if is_t2 else None,
                    gpm_cov_every=int(V6_PROTO["tier2_gpm_cov_every"]) if is_t2 else int(V6_PROTO["gpm_cov_every"]),
                )
            elif is_cog:
                tot_l, ce_l = cog_step(model, gpm, opt_box, w, lr_scale(st), log, gstep, dom)
            else:
                set_lr(opt_box[0], lr_scale(st))
                tot_l, ce_l = dense_step(model, opt_box[0], w, log)
            if tot_l is not None:
                losses.append(tot_l)
            if ce_l is not None:
                ces.append(ce_l)
            if rep is not None:
                rep.add(fresh)
            gstep += 1
        quarantine_buf.clear()
        train_loss_tail[dom] = float(np.mean(losses[-20:])) if losses else float("nan")
        train_ce_tail[dom] = float(np.mean(ces[-20:])) if ces else float("nan")
        row = eval_all(model, corp, is_cog)
        row["_after_stream"] = dom
        if is_cog:
            row["_num_experts"] = len(model.layers[0].moe.experts)
            row["_expert_states"] = list(model.lifecycle.expert_states) if model.lifecycle is not None else []
            if gpm is not None:
                row["_gpm_summary"] = summarize_gpm(gpm)
        matrix.append(row)
        print(f"  [{arm} s{seed}] after {dom:5s}: " + " ".join(f"{d}={row[d]['bpb']:.3f}" for d in DOMAINS)
              + (f" | exp={row['_num_experts']}" if is_cog else ""), flush=True)
    sec = time.perf_counter() - t0
    res: Dict[str, Any] = {
        "arm": arm, "seed": seed, "steps_per_stream": steps, "params_init": params_init,
        "params_final": count_parameters(model), "train_seconds": sec,
        "tokens_per_sec": (steps * len(DOMAINS) * PROTO["batch"] * SEQ_LEN) / sec,
        "forward_passes": log["forward_passes"], "backward_passes": log["backward_passes"],
        "ep_ltm_passes": log["ep_ltm_passes"],
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
        "matrix": matrix, "train_loss_tail_nats": train_loss_tail, "train_ce_tail_nats": train_ce_tail,
        "events": log["events"], "skipped_steps": log["skipped_steps"],
    }
    if is_cog:
        res["gpm_ranks"] = summarize_gpm(gpm)
        res["ltm_used_slots"] = int(model.memory.long_term.slot_used.sum().item())
        if learner is not None and learner.spgc_gate is not None:
            res["spgc_stats"] = {
                "quarantine_bytes": int(learner.spgc_gate.storage_bytes()),
                "optimizer_mode": getattr(learner.optimizer, "mode", None),
            }
    return res, model, gpm


def run_mixed(
    arm: str, seed: int, corp: RealCorpora, total_steps: int, dense_dff: int,
) -> Tuple[Dict[str, Any], nn.Module, Optional[NullSpaceProjector]]:
    """Fully mixed stream: every window of every batch is drawn from a uniformly random domain."""
    set_all_seeds(seed)
    is_cog = arm.startswith("Cog")
    is_v6 = arm.startswith("Cog_v6")
    gpm = None
    if is_cog:
        model, gpm, _ = make_cog(arm)
    else:
        model = make_dense(dense_dff)
    opt_box = [new_opt(model)]
    log: Dict[str, Any] = {
        "events": [], "skipped_steps": 0, "forward_passes": 0, "backward_passes": 0,
        "ep_ltm_passes": 0, "ep_steps": 0,
    }
    quarantine_buf: List[Tuple[torch.Tensor, float]] = []
    rng = np.random.default_rng([seed, 4242])
    gens = {d: np.random.default_rng([seed, 5000 + i]) for i, d in enumerate(DOMAINS)}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for st in range(total_steps):
        doms = rng.integers(0, len(DOMAINS), size=PROTO["batch"])
        w = torch.cat([corp.sample_train(DOMAINS[int(k)], 1, gens[DOMAINS[int(k)]]) for k in doms], dim=0)
        if is_v6:
            is_t2 = "Tier2" in arm
            cog_v6_step(
                model, gpm, opt_box, w, lr_scale(st), log, st, "mixed",
                quarantine_buf=quarantine_buf, enable_ep_ltm=True,
                ep_every_override=int(V6_PROTO["tier2_ep_ltm_every"]) if is_t2 else None,
                gpm_cov_every=int(V6_PROTO["tier2_gpm_cov_every"]) if is_t2 else int(V6_PROTO["gpm_cov_every"]),
            )
        elif is_cog:
            cog_step(model, gpm, opt_box, w, lr_scale(st), log, st, "mixed")
        else:
            set_lr(opt_box[0], lr_scale(st))
            dense_step(model, opt_box[0], w, log)
    sec = time.perf_counter() - t0
    row = eval_all(model, corp, is_cog)
    print(f"  [{arm} s{seed}] mixed-final: " + " ".join(f"{d}={row[d]['bpb']:.3f}" for d in DOMAINS)
          + (f" | exp={len(model.layers[0].moe.experts)}" if is_cog else ""), flush=True)
    res: Dict[str, Any] = {
        "arm": arm, "seed": seed, "total_steps": total_steps, "final": row, "train_seconds": sec,
        "forward_passes": log["forward_passes"], "backward_passes": log["backward_passes"],
        "ep_ltm_passes": log["ep_ltm_passes"],
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
        "params_final": count_parameters(model), "events": log["events"], "skipped_steps": log["skipped_steps"],
    }
    if is_cog:
        res["num_experts"] = len(model.layers[0].moe.experts)
        res["routing"] = routing_report(row)
    return res, model, gpm


# ----------------------------------------------------------------------------------------------
# Null controls / trivial baselines
# ----------------------------------------------------------------------------------------------
def ngram_baselines(corp: RealCorpora) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {"unigram": {}, "bigram": {}}
    for d in DOMAINS:
        tr = corp.train[d].astype(np.int64)
        uni = np.bincount(tr, minlength=256).astype(np.float64) + 1.0
        uni /= uni.sum()
        bi = np.ones((256, 256), dtype=np.float64)
        np.add.at(bi, (tr[:-1], tr[1:]), 1.0)
        bi /= bi.sum(axis=1, keepdims=True)
        w = corp.test_windows[d].numpy()
        x, y = w[:, :-1].reshape(-1), w[:, 1:].reshape(-1)
        out["unigram"][d] = float(-np.log2(uni[y]).mean())
        out["bigram"][d] = float(-np.log2(bi[x, y]).mean())
    return out


def untrained_eval(corp: RealCorpora, dense_dff: int, seed: int) -> Dict[str, Any]:
    set_all_seeds(seed)
    cog_u, _, _ = make_cog("Cog_SingleExpert")
    rc = eval_all(cog_u, corp, True)
    del cog_u
    set_all_seeds(seed)
    den_u = make_dense(dense_dff)
    rd = eval_all(den_u, corp, False)
    del den_u
    return {"cog_bpb": {d: rc[d]["bpb"] for d in DOMAINS}, "dense_bpb": {d: rd[d]["bpb"] for d in DOMAINS}}


def bag_of_bytes_domain_accuracy(corp: RealCorpora) -> float:
    """Dumbest router: nearest-centroid on 256-d byte histograms, fit with TRUE domain labels."""
    cents = []
    for d in DOMAINS:
        pool = corp.train[d]
        wins = pool[: (len(pool) // SEQ_LEN) * SEQ_LEN].reshape(-1, SEQ_LEN)[:4000]
        h = np.stack([np.bincount(r, minlength=256) for r in wins]).astype(np.float64)
        h /= h.sum(axis=1, keepdims=True)
        cents.append(h.mean(axis=0))
    cents_arr = np.stack(cents)
    correct = tot = 0
    for k, d in enumerate(DOMAINS):
        for w in corp.test_windows[d].numpy()[:, :SEQ_LEN]:
            h = np.bincount(w, minlength=256).astype(np.float64)
            h /= h.sum()
            correct += int(np.argmin(((cents_arr - h) ** 2).sum(axis=1)) == k)
            tot += 1
    return correct / tot


def nmi(labels: List[int], clusters: List[int]) -> float:
    la, cl = np.array(labels), np.array(clusters)
    n = len(la)
    def ent(v):
        _, c = np.unique(v, return_counts=True)
        p = c / n
        return float(-(p * np.log(p)).sum())
    ha, hc = ent(la), ent(cl)
    if ha == 0 or hc == 0:
        return 0.0
    mi = 0.0
    for a in np.unique(la):
        for c in np.unique(cl):
            pj = np.sum((la == a) & (cl == c)) / n
            if pj > 0:
                mi += pj * np.log(pj / ((np.sum(la == a) / n) * (np.sum(cl == c) / n)))
    return float(mi / math.sqrt(ha * hc))


def routing_report(cog_row: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    labels, clusters = [], []
    conf: Dict[str, Dict[int, int]] = {}
    for k, d in enumerate(DOMAINS):
        conf[d] = {}
        for e in cog_row[d]["routes"]:
            labels.append(k)
            clusters.append(int(e))
            conf[d][int(e)] = conf[d].get(int(e), 0) + 1
    return {"confusion_domain_x_expert": conf, "nmi_domain_vs_expert": nmi(labels, clusters)}


# ----------------------------------------------------------------------------------------------
# Behavioural & Mechanistic Probes (labelled OBSERVED in report)
# ----------------------------------------------------------------------------------------------
def decode(t: torch.Tensor) -> str:
    return bytes(t.flatten().numpy().astype(np.uint8)).decode("utf-8", errors="replace")


def _gen_stats(cont: torch.Tensor) -> Dict[str, Any]:
    b = cont.flatten().numpy().astype(np.uint8).tolist()
    n = len(b)
    if n < 4:
        return {"distinct_4gram_frac": 0.0, "max_4gram_repeat": 0, "printable_frac": 0.0, "utf8_valid": True}
    grams = [tuple(b[i:i + 4]) for i in range(n - 3)]
    cnt = Counter(grams)
    try:
        bytes(b).decode("utf-8", errors="strict")
        u_ok = True
    except UnicodeDecodeError:
        u_ok = False
    printable = sum(1 for v in b if (32 <= v <= 126 or v in (9, 10, 13)))
    return {
        "distinct_4gram_frac": len(cnt) / len(grams),
        "max_4gram_repeat": max(cnt.values()),
        "printable_frac": printable / n,
        "utf8_valid": u_ok,
        "unique_bytes": len(set(b)),
    }


@torch.no_grad()
def generate(
    model: nn.Module, prompt: torch.Tensor, n_new: int, is_cog: bool,
    temperature: float = 0.0, gen: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[int]]:
    model.eval()
    x = prompt.clone().to(DEV)
    experts: List[int] = []
    for _ in range(n_new):
        out = model(x, update_stats=False) if is_cog else model(x)
        lg = logits_of(out)[:, -1, :]
        if is_cog:
            experts.append(int(out["assigned_experts"][0].item()))
        if temperature <= 0:
            nxt = lg.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(lg / temperature, dim=-1), 1, generator=gen)
        x = torch.cat([x, nxt], dim=1)
    return x[:, prompt.size(1):].cpu(), experts


def generation_samples(
    model: nn.Module, corp: RealCorpora, is_cog: bool, n_per_domain: int = 3,
    prompt_len: int = 48, n_new: int = 96,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in DOMAINS:
        rows = []
        for k in range(n_per_domain):
            w = corp.test_windows[d][k * 7]
            prompt = w[:prompt_len].unsqueeze(0)
            cont, ex = generate(model, prompt, n_new, is_cog, temperature=0.0)
            cont2, _ = generate(model, prompt, n_new, is_cog, temperature=0.0)
            g = torch.Generator(device=DEV).manual_seed(100 + k)
            cont_t, _ = generate(model, prompt, n_new, is_cog, temperature=0.8, gen=g)
            rows.append({
                "prompt": decode(prompt),
                "generated_greedy": decode(cont),
                "generated_temp08": decode(cont_t),
                "reference": decode(w[prompt_len:prompt_len + n_new]),
                "deterministic_repeat_identical": bool(torch.equal(cont, cont2)),
                "routed_experts_sampled": ex[:: max(1, n_new // 8)] if ex else None,
                "stats_greedy": _gen_stats(cont),
                "stats_temp08": _gen_stats(cont_t),
            })
        out[d] = rows
    return out


def thinking_ablation_probe(model: RealSelfStabilizingDMoE, corp: RealCorpora) -> Dict[str, Any]:
    """Compare adaptive thinking vs forced 0 steps vs forced max steps (2) per domain."""
    out: Dict[str, Any] = {}
    for d in DOMAINS:
        w = corp.test_windows[d][:128]
        r_adp = eval_domain(model, w, True, force_thinking_steps=None)
        r0 = eval_domain(model, w, True, force_thinking_steps=0)
        r2 = eval_domain(model, w, True, force_thinking_steps=2)
        out[d] = {
            "bpb_adaptive": r_adp["bpb"], "bpb_think0": r0["bpb"], "bpb_think2": r2["bpb"],
            "delta_adp_minus_0": r_adp["bpb"] - r0["bpb"],
            "mean_steps_adaptive": r_adp["mean_token_steps"],
            "surprise_nll_corr": r_adp.get("surprise_nll_corr"),
        }
    return out


@torch.no_grad()
def ltm_and_pc_probe(model: RealSelfStabilizingDMoE, corp: RealCorpora) -> Dict[str, Any]:
    """Test explicit LTM consolidation vs distractor control + natural-language PC contradiction gating."""
    model.eval()
    model.reset_ltm()
    passages = torch.cat([corp.test_windows[d][:8] for d in ("wiki", "story", "math")], dim=0)  # 24 x 129
    bpb_no, bpb_match, bpb_dist, slots_list = [], [], [], []
    n = passages.size(0)
    for i in range(n):
        ep = passages[i:i + 1, :64].to(DEV)
        dist = passages[(i + 7) % n: (i + 7) % n + 1, :64].to(DEV)
        q = passages[i:i + 1, 64:]
        model.reset_ltm()
        bpb_no.append(eval_domain(model, q, True)["bpb"])
        model.reset_ltm()
        added = model.consolidate_ltm(ep, top_k=8, min_nll=0.5)
        slots_list.append(int(added))
        bpb_match.append(eval_domain(model, q, True)["bpb"])
        model.reset_ltm()
        model.consolidate_ltm(dist, top_k=8, min_nll=0.5)
        bpb_dist.append(eval_domain(model, q, True)["bpb"])
    model.reset_ltm()

    # Contradiction vs clean minimal-pair PC check
    clean_txt = [
        b"Paris is the capital of France. Question: What is the capital of France? Answer: Paris is the capital city.",
        b"Water boils at 100 degrees Celsius at sea level. Water boils at 100 degrees Celsius when heated in an open pot.",
        b"def add(a, b):\n    return a + b\nassert add(2, 3) == 5\n# Result of adding 2 and 3 is five.",
        b"Alice had 3 apples and bought 4 more apples. In total Alice has 7 apples right now.",
    ]
    contra_txt = [
        b"Paris is the capital of France. Actually wait: Paris is NOT a city, Mars is France. Answer: xyzq 999.",
        b"Water boils at 100 degrees Celsius at sea level. Wait no: water boils at minus 5000 degrees solid metal.",
        b"def add(a, b):\n    return a + b\nassert add(2, 3) == 99999 # syntax crash zzzqqq!",
        b"Alice had 3 apples and bought 4 more apples. Wait: Alice has -800 purple elephant submarines.",
    ]
    def _pc_stats(texts: List[bytes]) -> Dict[str, Any]:
        trigs, gmin, gmean, bpbs = [], [], [], []
        for t in texts:
            xb = torch.tensor([list(t[:96])], dtype=torch.long, device=DEV)
            out = model(xb[:, :-1], update_stats=False)
            lg = logits_of(out)
            ce = float(F.cross_entropy(lg.reshape(-1, 256), xb[:, 1:].reshape(-1)).item())
            bpbs.append(ce / math.log(2.0))
            g = out.get("pc_write_gate")
            if g is not None:
                trigs.append(float((g < 0.98).float().mean().item()))
                gmin.append(float(g.min().item()))
                gmean.append(float(g.mean().item()))
        return {
            "mean_bpb": float(np.mean(bpbs)),
            "pc_trigger_frac": float(np.mean(trigs)) if trigs else 0.0,
            "pc_gate_min_mean": float(np.mean(gmin)) if gmin else 1.0,
            "pc_gate_mean": float(np.mean(gmean)) if gmean else 1.0,
        }

    return {
        "ltm_bpb_no_ltm": float(np.mean(bpb_no)),
        "ltm_bpb_matched_ltm": float(np.mean(bpb_match)),
        "ltm_bpb_distractor_ltm": float(np.mean(bpb_dist)),
        "ltm_mean_slots_added": float(np.mean(slots_list)),
        "pc_clean_pairs": _pc_stats(clean_txt),
        "pc_contradiction_pairs": _pc_stats(contra_txt),
    }


def context_and_noise_probe(model: nn.Module, corp: RealCorpora, is_cog: bool) -> Dict[str, Any]:
    """Compare clean suffix [64:128] BPB vs shuffled-prefix [0:64] vs 10% byte-corrupted prefix."""
    rng = np.random.default_rng(2026)
    res: Dict[str, Any] = {}
    for d in ("wiki", "story", "code", "math"):
        w = corp.test_windows[d][:64].clone()
        clean = eval_domain(model, w, is_cog, start_pos=63)["bpb"]
        w_shuf = w.clone()
        for i in range(w_shuf.size(0)):
            perm = torch.from_numpy(rng.permutation(64))
            w_shuf[i, :64] = w_shuf[i, perm]
        shuf = eval_domain(model, w_shuf, is_cog, start_pos=63)["bpb"]
        w_noi = w.clone()
        mask = torch.from_numpy(rng.random((w_noi.size(0), 64)) < 0.10)
        rand_b = torch.from_numpy(rng.integers(32, 126, size=(w_noi.size(0), 64), dtype=np.int64))
        w_noi[:, :64] = torch.where(mask, rand_b, w_noi[:, :64])
        noi = eval_domain(model, w_noi, is_cog, start_pos=63)["bpb"]
        res[d] = {
            "suffix_clean_bpb": clean, "suffix_shuffled_prefix_bpb": shuf,
            "suffix_noise10_prefix_bpb": noi,
            "delta_shuffle": shuf - clean, "delta_noise10": noi - clean,
        }
    return res


def _load_gsm8k_eval(eval_split: str):
    from datasets import load_dataset
    if eval_split == "test":
        return load_dataset("gsm8k", "main", split="test")
    tr = load_dataset("gsm8k", "main", split="train")
    return tr.select(range(6000, len(tr)))


def compositional_probe(model: nn.Module, is_cog: bool, corp: RealCorpora, eval_split: str) -> Dict[str, Any]:
    """(1) gsm8k <<expr=result>> + step-count stratification, (2) zero-shot var-binding, (3) code nesting."""
    ds = _load_gsm8k_eval(eval_split)
    pat = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
    prompts, targets, meta = [], [], []
    simple_wins, multi_wins = [], []
    for r in ds:
        txt = f"Question: {r['question']}\nAnswer: {r['answer']}"
        n_steps = len(pat.findall(txt))
        enc = list(txt.encode("utf-8", "ignore"))
        if len(enc) >= SEQ_LEN + 1:
            ans_pos = txt.find("\nAnswer:")
            if n_steps <= 2 and len(simple_wins) < 128 and ans_pos >= 16:
                simple_wins.append(enc[:SEQ_LEN + 1])
            elif n_steps >= 4 and len(multi_wins) < 128 and ans_pos >= 16:
                multi_wins.append(enc[:SEQ_LEN + 1])
        for m in pat.finditer(txt):
            if len(prompts) >= 300:
                break
            eq_pos = txt.index("=", m.start())
            pre = txt[: eq_pos + 1].encode("utf-8", "ignore")
            if len(pre) < 96:
                continue
            prompts.append(list(pre[-96:]))
            targets.append(m.group(2))
            meta.append(re.findall(r"\d+\.?\d*", m.group(1)))

    # Calculator greedy exact match
    cur = torch.tensor(prompts, dtype=torch.long)
    produced = [[] for _ in range(len(targets))]
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            nxt_all = []
            for i in range(0, cur.size(0), 100):
                xb = cur[i:i + 100].to(DEV)
                out = model(xb, update_stats=False) if is_cog else model(xb)
                nxt_all.append(logits_of(out)[:, -1, :].argmax(-1).cpu())
            nxt = torch.cat(nxt_all)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            for i, b in enumerate(nxt.tolist()):
                produced[i].append(b)
    hits = small_hits = small_n = copy_first = 0
    for i, tgt in enumerate(targets):
        ok = bytes(produced[i]).decode("utf-8", errors="replace").split(">")[0] == tgt
        hits += ok
        if meta[i] and all(float(n) < 10 for n in meta[i]):
            small_n += 1
            small_hits += ok
        if meta[i] and meta[i][0] == tgt:
            copy_first += 1

    # GSM8K step-depth BPB
    gsm_simple_bpb = eval_domain(model, torch.tensor(simple_wins, dtype=torch.long), is_cog)["bpb"] if simple_wins else None
    gsm_multi_bpb = eval_domain(model, torch.tensor(multi_wins, dtype=torch.long), is_cog)["bpb"] if multi_wins else None

    # Zero-shot multi-hop Python variable-binding probe (unseen variable identifiers + 2-digit values)
    rng = random.Random(999)
    bind_wins, bind_tgts, shuf_wins = [], [], []
    for _ in range(64):
        names = [f"{chr(97 + rng.randrange(26))}{rng.randrange(10, 99)}" for _ in range(4)]
        vals = [str(rng.randrange(10, 99)) for _ in range(4)]
        prog = (f"{names[0]} = {vals[0]}\n{names[3]} = {vals[3]}\n"
                f"{names[1]} = {names[0]}\n{names[2]} = {names[1]}\n# {names[2]} == ").encode("utf-8")
        # Truncate/pad cleanly to common length by right-aligning fixed-length template (exact same length!)
        bind_wins.append(list(prog))
        bind_tgts.append(vals[0])
        shuf_prog = (f"{names[3]} = {vals[3]}\n{names[0]} = {vals[0]}\n"
                     f"{names[1]} = {names[3]}\n{names[2]} = {names[1]}\n# {names[2]} == ").encode("utf-8")
        shuf_wins.append(list(shuf_prog))
    bx = torch.tensor(bind_wins, dtype=torch.long)
    bind_hit = 0
    bind_nlls = []
    with torch.no_grad():
        cur_b = bx.to(DEV)
        b_preds = [[] for _ in range(len(bind_tgts))]
        for step_c in range(2):
            out = model(cur_b, update_stats=False) if is_cog else model(cur_b)
            lg = logits_of(out)[:, -1, :]
            tgt_bytes = torch.tensor([ord(t[step_c]) for t in bind_tgts], dtype=torch.long, device=DEV)
            bind_nlls.append(F.cross_entropy(lg, tgt_bytes).item())
            nxt = lg.argmax(-1)
            for i, v in enumerate(nxt.tolist()):
                b_preds[i].append(v)
            cur_b = torch.cat([cur_b, nxt.unsqueeze(1)], dim=1)
        for i, tgt in enumerate(bind_tgts):
            bind_hit += int(bytes(b_preds[i]).decode("utf-8", errors="replace") == tgt)
        # Check whether model assigns lower BPB to true bound digit vs distractor value
        sx = torch.tensor(shuf_wins, dtype=torch.long, device=DEV)
        out_s = model(sx, update_stats=False) if is_cog else model(sx)
        shuf_nll = float(F.cross_entropy(
            logits_of(out_s)[:, -1, :],
            torch.tensor([ord(t[0]) for t in bind_tgts], dtype=torch.long, device=DEV),
        ).item())

    # Code nesting stratification
    cw = corp.test_windows["code"]
    nest_scores = [sum(1 for b in r.tolist() if b in (40, 41, 91, 93, 123, 125)) for r in cw]
    med = float(np.median(nest_scores))
    shallow_idx = [i for i, s in enumerate(nest_scores) if s <= med]
    deep_idx = [i for i, s in enumerate(nest_scores) if s > med]

    return {
        "arith_n": len(targets), "arith_exact_match": hits / max(1, len(targets)),
        "arith_exact_small": (small_hits / small_n) if small_n else None, "arith_n_small": small_n,
        "arith_copy_first": copy_first / max(1, len(targets)),
        "arith_majority": Counter(targets).most_common(1)[0][1] / max(1, len(targets)),
        "gsm_simple_le2_bpb": gsm_simple_bpb, "gsm_multi_ge4_bpb": gsm_multi_bpb,
        "var_binding_exact2": bind_hit / len(bind_tgts),
        "var_binding_2byte_bpb": float(np.mean(bind_nlls) / math.log(2.0)),
        "var_binding_shuffled_1st_byte_bpb": shuf_nll / math.log(2.0),
        "code_shallow_bpb": eval_domain(model, cw[shallow_idx], is_cog)["bpb"] if shallow_idx else None,
        "code_deep_bpb": eval_domain(model, cw[deep_idx], is_cog)["bpb"] if deep_idx else None,
    }


@torch.no_grad()
def pooled_features(
    model: Optional[nn.Module], texts: List[bytes], is_cog: bool, maxlen: int = 64,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    feats = [None] * len(texts)
    routes: Dict[int, int] = {}
    by_len: Dict[int, List[int]] = {}
    for i, t in enumerate(texts):
        by_len.setdefault(min(len(t), maxlen), []).append(i)
    if model is not None:
        model.eval()
    for L, idxs in by_len.items():
        if L < 4:
            continue
        for s in range(0, len(idxs), 128):
            ii = idxs[s:s + 128]
            xb = torch.tensor([list(texts[i][:L]) for i in ii], dtype=torch.long, device=DEV)
            if model is None:
                h = F.one_hot(xb, 256).float()
            else:
                out = model(xb, update_stats=False) if is_cog else model(xb)
                h = out["latent"].float()
                if is_cog and "assigned_experts" in out:
                    for e in out["assigned_experts"].detach().cpu().tolist():
                        routes[int(e)] = routes.get(int(e), 0) + 1
            p = h.mean(dim=1).cpu()
            for j, i in enumerate(ii):
                feats[i] = p[j]
    d = next(f for f in feats if f is not None).numel()
    return torch.stack([f if f is not None else torch.zeros(d) for f in feats]), routes


def ridge_probe(ftr: torch.Tensor, ytr: torch.Tensor, fte: torch.Tensor, yte: torch.Tensor, n_cls: int) -> float:
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    n = ftr.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    va, tr = perm[: n // 5], perm[n // 5:]
    def fit(F_, y_, lam):
        Y = F.one_hot(y_, n_cls).float()
        A = F_.T @ F_ + lam * torch.eye(F_.size(1))
        return torch.linalg.solve(A, F_.T @ Y)
    best_lam, best_acc = 1.0, -1.0
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        W = fit(ftr[tr], ytr[tr], lam)
        acc = float(((ftr[va] @ W).argmax(-1) == ytr[va]).float().mean())
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    W = fit(ftr, ytr, best_lam)
    return float(((fte @ W).argmax(-1) == yte).float().mean())


class MeaningEvaluator:
    """Caches banking77 split + untrained/bag-of-bytes baselines once per run."""

    def __init__(self, eval_split: str, dense_dff: int, seed: int):
        from datasets import load_dataset
        self.eval_split = eval_split
        tr = load_dataset("banking77", split="train")
        rng = random.Random(0)
        if eval_split == "test":
            te = load_dataset("banking77", split="test")
            tr_idx = rng.sample(range(len(tr)), 4000)
            self.tr_txt = [tr[i]["text"].encode("utf-8", "ignore") for i in tr_idx]
            self.ytr = torch.tensor([tr[i]["label"] for i in tr_idx])
            self.te_txt = [t.encode("utf-8", "ignore") for t in te["text"]]
            self.yte = torch.tensor(te["label"])
        else:
            idx = list(range(len(tr)))
            rng.shuffle(idx)
            tr_i, te_i = idx[:2000], idx[2000:3200]
            self.tr_txt = [tr[i]["text"].encode("utf-8", "ignore") for i in tr_i]
            self.ytr = torch.tensor([tr[i]["label"] for i in tr_i])
            self.te_txt = [tr[i]["text"].encode("utf-8", "ignore") for i in te_i]
            self.yte = torch.tensor([tr[i]["label"] for i in te_i])

        # Build same-intent low-overlap vs diff-intent high-overlap pairs
        words = [set(t.decode("utf-8", "ignore").lower().split()) for t in self.te_txt]
        by_lbl: Dict[int, List[int]] = {}
        for i, lbl in enumerate(self.yte.tolist()):
            by_lbl.setdefault(lbl, []).append(i)
        same_cands, diff_cands = [], []
        for lbl, idxs in by_lbl.items():
            for a_i in range(min(8, len(idxs))):
                for b_i in range(a_i + 1, min(8, len(idxs))):
                    u, v = idxs[a_i], idxs[b_i]
                    jac = len(words[u] & words[v]) / max(1, len(words[u] | words[v]))
                    same_cands.append((jac, u, v))
        same_cands.sort(key=lambda z: z[0])
        self.same_pairs = [(u, v) for _, u, v in same_cands[:200]]
        for i in range(0, min(400, len(self.te_txt))):
            for j in range(i + 1, min(400, len(self.te_txt))):
                if self.yte[i] != self.yte[j]:
                    jac = len(words[i] & words[j]) / max(1, len(words[i] | words[j]))
                    if jac >= 0.25:
                        diff_cands.append((jac, i, j))
        diff_cands.sort(key=lambda z: -z[0])
        self.diff_pairs = [(u, v) for _, u, v in diff_cands[:200]]

        self.refs: Dict[str, Any] = {
            "n_train": len(self.tr_txt), "n_test": len(self.te_txt),
            "chance": 1.0 / 77, "majority": float(torch.bincount(self.yte).max()) / len(self.yte),
            "bag_of_bytes": self.eval_model(None, False),
        }
        set_all_seeds(seed)
        cog_u, _, _ = make_cog("Cog_Full")
        self.refs["Cog_untrained"] = self.eval_model(cog_u, True)
        del cog_u
        set_all_seeds(seed)
        den_u = make_dense(dense_dff)
        self.refs["Dense_untrained"] = self.eval_model(den_u, False)
        del den_u

    def eval_model(self, model: Optional[nn.Module], is_cog: bool) -> Dict[str, Any]:
        ftr, _ = pooled_features(model, self.tr_txt, is_cog)
        fte, te_routes = pooled_features(model, self.te_txt, is_cog)
        acc = ridge_probe(ftr, self.ytr, fte, self.yte, 77)
        fn = F.normalize(fte, p=2, dim=-1)
        pos = [float((fn[u] @ fn[v]).item()) for u, v in self.same_pairs]
        neg = [float((fn[u] @ fn[v]).item()) for u, v in self.diff_pairs]
        # Exact pairwise ROC-AUC (pos vs diff-intent high-overlap pairs)
        auc = float(np.mean([(1.0 if p > q else (0.5 if p == q else 0.0)) for p in pos for q in neg])) if (pos and neg) else None
        out_d: Dict[str, Any] = {
            "intent_77_acc": acc,
            "same_vs_overlap_diff_auc": auc,
            "sim_same_intent": float(np.mean(pos)) if pos else None,
            "sim_diff_high_overlap": float(np.mean(neg)) if neg else None,
        }
        if is_cog:
            out_d["test_expert_routes"] = te_routes
        return out_d


def revisit_probe(
    model: RealSelfStabilizingDMoE, gpm: Optional[NullSpaceProjector], corp: RealCorpora, seed: int, steps_per_dom: int = 12,
) -> Dict[str, Any]:
    """Stream fresh batches across all 5 domains in order AFTER initial continual learning."""
    before = eval_all(model, corp, True)
    opt_box = [new_opt(model)]
    log: Dict[str, Any] = {"events": [], "skipped_steps": 0, "forward_passes": 0, "backward_passes": 0}
    exp_before = len(model.layers[0].moe.experts)
    is_v6 = (
        getattr(model, "lifecycle", None) is not None
        and getattr(model.lifecycle, "defer_on_novel_streak", False)
    )
    quarantine_buf: List[Tuple[torch.Tensor, float]] = []
    gen = np.random.default_rng([seed, 8888])
    rev_routes: Dict[str, Dict[int, int]] = {}
    gstep = 0
    for dom in DOMAINS:
        rev_routes[dom] = {}
        for st in range(steps_per_dom):
            w = corp.sample_train(dom, PROTO["batch"], gen)
            if is_v6:
                cog_v6_step(
                    model, gpm, opt_box, w, 0.25, log, gstep, f"revisit_{dom}",
                    quarantine_buf=quarantine_buf, enable_ep_ltm=False,
                )
            else:
                cog_step(model, gpm, opt_box, w, 0.25, log, gstep, f"revisit_{dom}")
            if model.last_assigned_experts is not None:
                for e in model.last_assigned_experts.detach().cpu().tolist():
                    rev_routes[dom][int(e)] = rev_routes[dom].get(int(e), 0) + 1
            gstep += 1
        quarantine_buf.clear()
    after = eval_all(model, corp, True)
    spawns = [e for e in log["events"] if "spawned_expert_" in e["event"]]
    return {
        "steps_per_domain": steps_per_dom,
        "experts_before": exp_before,
        "experts_after": len(model.layers[0].moe.experts),
        "spawns_on_revisit": len(spawns),
        "revisit_events": log["events"],
        "revisit_train_routes": rev_routes,
        "bpb_before": {d: before[d]["bpb"] for d in DOMAINS},
        "bpb_after": {d: after[d]["bpb"] for d in DOMAINS},
        "delta_bpb": {d: after[d]["bpb"] - before[d]["bpb"] for d in DOMAINS},
    }


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def run_probes_for_model(
    arm: str, model: nn.Module, gpm: Optional[NullSpaceProjector], is_cog: bool,
    corp: RealCorpora, eval_split: str, meaning_eval: MeaningEvaluator, seed: int, res: Dict[str, Any],
) -> None:
    res["meaning_probe_banking77"] = {
        "trained": meaning_eval.eval_model(model, is_cog),
        "references": meaning_eval.refs if arm in ("Cog_Full", "Cog_v6") else {"chance": meaning_eval.refs["chance"]},
    }
    res["context_and_noise_probe"] = context_and_noise_probe(model, corp, is_cog)
    res["compositional_probe"] = compositional_probe(model, is_cog, corp, eval_split)
    if arm in ("Cog_Full", "Cog_v6", "Cog_v6_Mixed", "Cog_v6_NoLTM", "Cog_v6_Tier2", "Cog_v6_Tier2_Mixed", "Dense_NoReplay", "Dense_Replay", "Dense_Large", "Dense_Joint", "Cog_Mixed", "Cog_SPGC"):
        res["generation"] = generation_samples(model, corp, is_cog)
    if is_cog:
        res["thinking_ablation"] = thinking_ablation_probe(model, corp)
        if arm in ("Cog_Full", "Cog_v6", "Cog_v6_Mixed", "Cog_v6_NoLTM", "Cog_v6_Tier2", "Cog_v6_Tier2_Mixed", "Cog_NoGPM", "Cog_SingleExpert", "Cog_SPGC"):
            res["ltm_and_pc_probe"] = ltm_and_pc_probe(model, corp)
        if arm in ("Cog_Full", "Cog_v6", "Cog_v6_Tier2", "Cog_NoGPM"):
            res["revisit_probe"] = revisit_probe(model, gpm, corp, seed)


def run_seed(
    seed: int, corp: RealCorpora, eval_split: str, steps: int, dense_dff: int, dense_large_dff: int,
    arms: List[str], out_dir: Path, do_probes: bool,
) -> None:
    meaning_eval = MeaningEvaluator(eval_split, dense_dff, seed) if do_probes else None
    for arm in arms:
        path = out_dir / f"seed{seed}_{arm}.json"
        if path.exists():
            print(f"[skip] {path.name} exists", flush=True)
            continue
        print(f"=== seed {seed} arm {arm} ===", flush=True)
        is_cog = arm.startswith("Cog")
        if arm in ("Dense_Joint", "Cog_Mixed", "Cog_v6_Mixed", "Cog_v6_Tier2_Mixed"):
            res, model, gpm = run_mixed(arm, seed, corp, steps * len(DOMAINS), dense_dff)
        else:
            dff = dense_large_dff if arm == "Dense_Large" else dense_dff
            res, model, gpm = run_sequential(arm, seed, corp, steps, dff)
            if is_cog and arm != "Cog_SingleExpert":
                res["routing"] = routing_report(res["matrix"][-1])
        if do_probes and meaning_eval is not None:
            run_probes_for_model(arm, model, gpm, is_cog, corp, eval_split, meaning_eval, seed, res)
        path.write_text(json.dumps(res, indent=1))
        print(f"[saved] {path.name}", flush=True)
        del model, gpm
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "dev", "test"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out_name", type=str, required=True)
    ap.add_argument("--results_root", type=str, default="real_pilot_v1")
    ap.add_argument("--arms", type=str, nargs="+",
                    default=["Cog_Full", "Dense_NoReplay", "Dense_Replay", "Dense_Large", "Cog_NoGPM",
                             "Cog_SingleExpert", "Dense_Joint", "Cog_Mixed", "Cog_SPGC"])
    ap.add_argument("--dense_dff", type=int, default=0)
    ap.add_argument("--dense_large_dff", type=int, default=0)
    ap.add_argument("--spgc_lr", type=float, default=5e-4)
    ap.add_argument("--gpm_energy", type=float, default=0.97)
    ap.add_argument("--v6_pc_thresh", type=float, default=COG_V6_CFG["pc_thresh"])
    ap.add_argument("--v6_theta_halt", type=float, default=COG_V6_CFG["theta_halt"])
    ap.add_argument("--v6_skip_thresh", type=float, default=COG_V6_CFG["skip_threshold"])
    ap.add_argument("--v6_min_drop", type=float, default=COG_V6_CFG["min_surprise_drop"])
    ap.add_argument("--v6_step_rank", type=float, default=V6_PROTO["step_rank_ratio"])
    ap.add_argument("--v6_step_decay", type=float, default=V6_PROTO["step_rank_decay"])
    ap.add_argument("--v6_ltm_null", type=float, default=COG_V6_CFG["ltm_null_logit"])
    ap.add_argument("--v6_adapter_rank", type=int, default=COG_V6_CFG["adapter_rank"])
    ap.add_argument("--v6_revisit_margin", type=float, default=COG_V6_CFG["train_revisit_conf_margin"])
    ap.add_argument("--v6_mixed_gap", type=float, default=COG_V6_CFG["mixed_split_gap"])
    ap.add_argument("--v6_ep_ltm_every", type=int, default=V6_PROTO["ep_ltm_every"])
    ap.add_argument("--v6_ep_ltm_weight", type=float, default=V6_PROTO["ep_ltm_weight"])
    ap.add_argument("--tier2_stm_every", type=int, default=V6_PROTO["tier2_stm_every"])
    ap.add_argument("--tier2_ep_ltm_every", type=int, default=V6_PROTO["tier2_ep_ltm_every"])
    ap.add_argument("--tier2_gpm_cov_every", type=int, default=V6_PROTO["tier2_gpm_cov_every"])
    ap.add_argument("--no_probes", action="store_true")
    a = ap.parse_args()
    PROTO["spgc_lr"] = a.spgc_lr
    PROTO["gpm_energy"] = a.gpm_energy
    COG_V6_CFG["pc_thresh"] = a.v6_pc_thresh
    COG_V6_CFG["theta_halt"] = a.v6_theta_halt
    COG_V6_CFG["skip_threshold"] = a.v6_skip_thresh
    COG_V6_CFG["min_surprise_drop"] = a.v6_min_drop
    COG_V6_CFG["adapter_rank"] = a.v6_adapter_rank
    COG_V6_CFG["train_revisit_conf_margin"] = a.v6_revisit_margin
    COG_V6_CFG["mixed_split_gap"] = a.v6_mixed_gap
    COG_V6_CFG["ltm_null_logit"] = a.v6_ltm_null
    V6_PROTO["step_rank_ratio"] = a.v6_step_rank
    V6_PROTO["step_rank_decay"] = a.v6_step_decay
    V6_PROTO["ep_ltm_every"] = a.v6_ep_ltm_every
    V6_PROTO["ep_ltm_weight"] = a.v6_ep_ltm_weight
    V6_PROTO["tier2_stm_every"] = a.tier2_stm_every
    V6_PROTO["tier2_ep_ltm_every"] = a.tier2_ep_ltm_every
    V6_PROTO["tier2_gpm_cov_every"] = a.tier2_gpm_cov_every

    out_dir = ROOT / "results" / a.results_root / a.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_split = "test" if a.mode == "test" else "dev"
    corp = RealCorpora(eval_split=eval_split)
    set_all_seeds(0)
    cog_tmp, _, _ = make_cog("Cog_Full")
    cog_params = count_parameters(cog_tmp)
    del cog_tmp
    cog_v6_tmp, _, _ = make_cog("Cog_v6")
    cog_v6_params = count_parameters(cog_v6_tmp)
    del cog_v6_tmp
    cog_grown_params = compute_grown_cog_params(len(DOMAINS), kind="Cog_Full")
    cog_v6_grown_params = compute_grown_cog_params(len(DOMAINS), kind="Cog_v6")
    dense_dff = a.dense_dff or match_dense_dff(cog_params)
    dense_large_dff = a.dense_large_dff or match_dense_dff(
        cog_v6_grown_params if any(x.startswith("Cog_v6") for x in a.arms) else cog_grown_params
    )

    meta = {
        "exp_id": a.results_root, "mode": a.mode, "eval_split": eval_split,
        "seeds": a.seeds, "steps_per_stream": a.steps,
        "domains": DOMAINS, "proto": PROTO, "cog_cfg": COG_CFG, "cog_v6_cfg": COG_V6_CFG, "v6_proto": V6_PROTO,
        "dense": {
            "d_model": DENSE_D, "layers": DENSE_LAYERS, "heads": DENSE_HEADS,
            "d_ff": dense_dff, "params": count_parameters(make_dense(dense_dff)),
            "d_ff_large": dense_large_dff, "params_large": count_parameters(make_dense(dense_large_dff)),
        },
        "cog_params_init": cog_params, "cog_params_grown_5exp": cog_grown_params,
        "cog_v6_params_init": cog_v6_params, "cog_v6_params_grown_5exp": cog_v6_grown_params,
        "replay": {"windows": REPLAY_WINDOWS, "frac": REPLAY_FRAC}, "arms": a.arms, "git": git_meta(),
        "utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "command": " ".join(sys.argv), "data_meta": corp.meta,
        "note_nondeterminism": "CUDA kernels not forced deterministic; PYTHONHASHSEED not pinned",
    }
    base = {
        "ngram_baselines_bpb": ngram_baselines(corp),
        "untrained_eval": untrained_eval(corp, dense_dff, a.seeds[0]),
        "bag_of_bytes_domain_accuracy_with_true_labels": bag_of_bytes_domain_accuracy(corp),
    }
    (out_dir / "_meta.json").write_text(json.dumps({"meta": meta, "null_and_trivial": base}, indent=1))
    print(f"[meta] split={eval_split} cog_init={cog_params:,} cog_v6_init={cog_v6_params:,} "
          f"cog_5exp={cog_grown_params:,} cog_v6_5exp={cog_v6_grown_params:,} "
          f"dense(d_ff={dense_dff})={count_parameters(make_dense(dense_dff)):,} "
          f"dense_large(d_ff={dense_large_dff})={count_parameters(make_dense(dense_large_dff)):,}", flush=True)
    for s in a.seeds:
        run_seed(s, corp, eval_split, a.steps, dense_dff, dense_large_dff, a.arms, out_dir, do_probes=not a.no_probes)


if __name__ == "__main__":
    main()

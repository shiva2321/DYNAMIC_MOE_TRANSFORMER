"""
CognitiveDMoE v12 Breakthrough Evaluation Runner — `real_pilot_v9`.

Pre-registered in `results/real_pilot_v9/PREREG.md`.
Evaluates `CognitiveDMoE_v12` head-to-head against matched baselines (`CognitiveDMoE_v12_NoProcrustes`,
`CognitiveDMoE_v12_FrozenRandomRouter`, `CognitiveDMoE_v11`, `Dense_Matched`) and mandatory null
controls across Track 1 (5 disjoint test seeds `0..4`) and Track 2 (3 disjoint test seeds `0..2`).

Dev tuning is strictly restricted to disjoint `seed = 100` (`eval_split = "dev"`), max 4 trials per
track, automatically appended to `results/real_pilot_v9/tuning_log.json`.
Supports `--resume` to preserve completed per-seed JSONs without re-rolling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import BeliefMeaningTask, K_CONCEPTS, Q_ATTR, Q_PROP, Q_VAL
from dmoe.cognitive_dmoe import CognitiveDMoE, CognitiveDMoEConfig, CognitiveTrainer
from dmoe.spgc import stablemax_cross_entropy
from experiments.real_pilot_data import DOMAINS, RealCorpora
from experiments.run_v7_cognitive_pilot import (
    DEV,
    SEQ_LEN,
    encode_belief_chunks,
    roc_auc_pairs,
    set_all_seeds,
)
from experiments.run_v8_cognitive_pilot import (
    V8RealBenchmark,
    encode_texts_v8,
    eval_banking77_v8,
    eval_domain_v8,
    eval_routing_diagnostics_v8,
    eval_triples_nuance_v8,
    generate_i7_procedural_batch,
)
from experiments.run_v9_cognitive_pilot import _sample_episodic_honest_batch
from experiments.run_v10_cognitive_pilot import (
    _encode_shot_sequences,
    _pack_episode_memory,
    _perturb_byte_tensor,
    _sample_comp_equiv_batch,
    _sample_unseen_meaning_shift_batch,
    eval_v10_episodic_suite,
    eval_v10_semantic_routing_suite,
)


def _sample_unseen_shift_multiquery_batch(
    task: BeliefMeaningTask,
    n_episodes: int,
    gen: torch.Generator,
    n_shots: int = 8,
    n_queries: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample `n_episodes` cyclic-shift training episodes, each paired with `n_queries` strictly
    unseen-meaning queries reusing the SAME support set (4x richer operator gradient per support encode).
    Returns `(x_sup [E, n_shots, 28], x_q [E*n_queries, 24], y_q [E*n_queries])`.
    """
    phi = task.phi
    inv_phi = task.inv_phi
    n_unique_sup = min(int(n_shots), K_CONCEPTS - max(4, int(n_queries)))
    x_sup_list: List[torch.Tensor] = []
    x_q_list: List[torch.Tensor] = []
    y_q_list: List[int] = []
    for _ in range(n_episodes):
        delta_ep = int(torch.randint(1, K_CONCEPTS, (1,), generator=gen).item())
        m_perm = torch.randperm(K_CONCEPTS, generator=gen)
        sup_m_pool = m_perm[:n_unique_sup]
        unseen_m_pool = m_perm[n_unique_sup:]

        a_l, b_l, y_l = [], [], []
        for si in range(n_shots):
            m_val = int(sup_m_pool[si % n_unique_sup].item())
            qb_s = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
            phi_a = (int(phi[m_val].item()) - 3 * int(phi[qb_s].item())) % K_CONCEPTS
            a_l.append(int(inv_phi[phi_a].item()))
            b_l.append(qb_s)
            y_l.append(int(inv_phi[(int(phi[m_val].item()) + delta_ep) % K_CONCEPTS].item()))
        x_sup_list.append(
            _encode_shot_sequences(
                torch.tensor(a_l, dtype=torch.long),
                torch.tensor(b_l, dtype=torch.long),
                torch.tensor(y_l, dtype=torch.long),
            )
        )
        for qi in range(n_queries):
            m_q = int(unseen_m_pool[qi % unseen_m_pool.numel()].item())
            qb_q = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
            phi_aq = (int(phi[m_q].item()) - 3 * int(phi[qb_q].item())) % K_CONCEPTS
            qa_q = int(inv_phi[phi_aq].item())
            y_q = int(inv_phi[(int(phi[m_q].item()) + delta_ep) % K_CONCEPTS].item())
            xs_q = torch.tensor([[0, qa_q + 4, qb_q + 4, 4, 4, Q_ATTR]], dtype=torch.long)
            x_q_list.append(encode_belief_chunks(xs_q)[0])
            y_q_list.append(y_q)
    return (
        torch.stack(x_sup_list, dim=0),
        torch.stack(x_q_list, dim=0),
        torch.tensor(y_q_list, dtype=torch.long),
    )
from experiments.run_v11_consolidated_eval import (
    DEFAULT_V11_CFG,
    _agg_v11_runs,
    run_track1_arm_v11,
    run_track2_arm_v11,
)

RESULTS_ROOT = ROOT / "results" / "real_pilot_v9"

DEFAULT_V12_CFG: Dict[str, Any] = {
    # Track 1: Scaled StableMax (2.0x logits) + Muon (wd=0.14, Zero Grokfast) + Multi-Plane SO(2)^p Procrustes
    "t1_lr_muon": 1.1e-2,
    "t1_lr_adam": 4.2e-3,
    "t1_wd_trunk": 0.16,
    "t1_ortho_grad": False,
    "t1_use_stablemax": True,
    "t1_stablemax_scale": 2.0,
    "t1_cosine_anneal": True,
    "t1_ep_start_frac": 0.40,
    "t1_op_lr": 1.5e-2,
    "dual_ridge_op_rank": 48,
    "op_procrustes_planes": 8,
    # Track 2: Vectorized Dispatch + Capacity Parity + H-Net Boundary + InfoNCE Continual Semantic Router
    "d_ff_shared": 512,
    "d_ff_spec": 256,
    "enable_hnet_boundary": True,
    "dual_granularity_router": True,
    "enable_surface_whitening": True,
    "use_vectorized_dispatch": True,
    "use_infonce_cris": True,
    "cris_weight": 0.35,
    "exp_orth_weight": 0.0,
    "router_anchor_weight": 0.5,
    "pert_every": 4,
    "shared_dropout_prob": 0.0,
    "specialist_init_scale": 0.06,
    "z_nuance_thresh": 1.6,
    "d_sem_nuance_thresh": 1.6,
    "i7_steps": 150,
}


def _collect_meta(
    mode: str,
    eval_split: str,
    seeds: List[int],
    steps_per_dom: int,
    tracks: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    git_commit = "unknown"
    git_dirty = True
    rev_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=False
    )
    if rev_proc.returncode == 0:
        git_commit = rev_proc.stdout.strip()
    st_proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True, check=False
    )
    if st_proc.returncode == 0:
        git_dirty = len(st_proc.stdout.strip()) > 0
    return {
        "exp_id": "real_pilot_v9",
        "mode": mode,
        "eval_split": eval_split,
        "seeds": seeds,
        "steps_per_dom": steps_per_dom,
        "tracks": tracks,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(DEV),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "command": " ".join(sys.argv),
        "cfg": cfg,
    }


def _build_v12_model(
    arm: str,
    cfg: Dict[str, Any],
    *,
    is_track1: bool = False,
    min_mature_steps: int = 100,
) -> CognitiveDMoE:
    hnet_on = bool(cfg["enable_hnet_boundary"]) and (not is_track1)
    if arm == "CognitiveDMoE_v12":
        mcfg = CognitiveDMoEConfig(
            d_ff_shared=int(cfg["d_ff_shared"]) if not is_track1 else 384,
            d_ff_spec=int(cfg["d_ff_spec"]),
            shift_invariant_pool=False,
            enable_hnet_boundary=hnet_on,
            enable_surface_whitening=(False if is_track1 else bool(cfg.get("enable_surface_whitening", False))),
            enable_dual_ridge_ep=True,
            dual_ridge_op_rank=int(cfg["dual_ridge_op_rank"]),
            enable_procrustes_op=True,
            op_procrustes_planes=int(cfg["op_procrustes_planes"]),
            frozen_exempt_balance=True,
            residual_ortho_router=True,
            dual_granularity_router=(False if is_track1 else bool(cfg["dual_granularity_router"])),
            use_vectorized_dispatch=bool(cfg["use_vectorized_dispatch"]),
            use_infonce_cris=(False if is_track1 else bool(cfg["use_infonce_cris"])),
            router_anchor_weight=(0.0 if is_track1 else float(cfg["router_anchor_weight"])),
            enable_cris_loss=(False if is_track1 else True),
            specialist_init_scale=(0.05 if is_track1 else float(cfg["specialist_init_scale"])),
            shared_dropout_prob=(0.0 if is_track1 else float(cfg["shared_dropout_prob"])),
            z_nuance_thresh=float(cfg["z_nuance_thresh"]),
            d_sem_nuance_thresh=float(cfg["d_sem_nuance_thresh"]),
            min_mature_steps=int(min_mature_steps),
            max_experts=6,
        )
        return CognitiveDMoE(mcfg).to(DEV)
    elif arm == "CognitiveDMoE_v12_NoProcrustes":
        mcfg = CognitiveDMoEConfig(
            d_ff_shared=384,
            d_ff_spec=int(cfg["d_ff_spec"]),
            shift_invariant_pool=False,
            enable_hnet_boundary=False,
            enable_surface_whitening=False,
            enable_dual_ridge_ep=True,
            dual_ridge_op_rank=int(cfg["dual_ridge_op_rank"]),
            dual_ridge_op_rank_low=0,
            enable_procrustes_op=False,
            frozen_exempt_balance=True,
            residual_ortho_router=True,
            use_vectorized_dispatch=bool(cfg["use_vectorized_dispatch"]),
            enable_cris_loss=False,
            specialist_init_scale=0.05,
            shared_dropout_prob=0.0,
            min_mature_steps=int(min_mature_steps),
            max_experts=6,
        )
        return CognitiveDMoE(mcfg).to(DEV)
    elif arm == "CognitiveDMoE_v12_FrozenRandomRouter":
        mcfg = CognitiveDMoEConfig(
            d_ff_shared=int(cfg["d_ff_shared"]),
            d_ff_spec=int(cfg["d_ff_spec"]),
            shift_invariant_pool=False,
            enable_hnet_boundary=hnet_on,
            enable_surface_whitening=False,
            enable_dual_ridge_ep=True,
            enable_procrustes_op=True,
            op_procrustes_planes=int(cfg["op_procrustes_planes"]),
            frozen_exempt_balance=True,
            residual_ortho_router=True,
            dual_granularity_router=bool(cfg["dual_granularity_router"]),
            use_vectorized_dispatch=bool(cfg["use_vectorized_dispatch"]),
            use_infonce_cris=False,
            router_anchor_weight=0.0,
            enable_cris_loss=False,
            specialist_init_scale=float(cfg["specialist_init_scale"]),
            shared_dropout_prob=float(cfg["shared_dropout_prob"]),
            z_nuance_thresh=float(cfg["z_nuance_thresh"]),
            d_sem_nuance_thresh=float(cfg["d_sem_nuance_thresh"]),
            min_mature_steps=int(min_mature_steps),
            max_experts=6,
        )
        model = CognitiveDMoE(mcfg).to(DEV)
        for p in model.context_router.parameters():
            p.requires_grad_(False)
        return model
    elif arm == "Dense_Matched":
        mcfg = CognitiveDMoEConfig(
            d_ff_shared=int(cfg["d_ff_shared"]),
            d_ff_spec=int(cfg["d_ff_spec"]),
            max_experts=1,
            shift_invariant_pool=False,
            enable_hnet_boundary=hnet_on,
            enable_surface_whitening=False,
            enable_dual_ridge_ep=False,
            enable_procrustes_op=False,
            frozen_exempt_balance=True,
            residual_ortho_router=True,
            dual_granularity_router=False,
            use_vectorized_dispatch=bool(cfg["use_vectorized_dispatch"]),
            enable_cris_loss=False,
            specialist_init_scale=0.05,
            shared_dropout_prob=0.0,
            min_mature_steps=int(min_mature_steps) + 5000,
        )
        return CognitiveDMoE(mcfg).to(DEV)
    else:
        raise ValueError(f"Unsupported v12 arm: {arm}")


def run_track1_arm_v12(
    arm: str,
    cfg: Dict[str, Any],
    seed: int,
    steps_scale: float = 1.0,
) -> Dict[str, Any]:
    if arm == "CognitiveDMoE_v11":
        return run_track1_arm_v11("CognitiveDMoE", dict(DEFAULT_V11_CFG), seed, steps_scale=steps_scale)

    set_all_seeds(seed)
    t_start = time.time()
    task = BeliefMeaningTask(seed=seed)
    gen = torch.Generator().manual_seed(seed * 97 + 1)

    n_p1 = max(220, int(1300 * steps_scale))
    n_pb = max(8, int(20 * steps_scale))
    n_p2 = max(40, int(180 * steps_scale))
    n_p3 = max(40, int(180 * steps_scale))

    model = _build_v12_model(arm, cfg, is_track1=True, min_mature_steps=n_p1 + 50)
    cls_head = nn.Linear(model.d_model, K_CONCEPTS).to(DEV)
    use_sm = bool(cfg["t1_use_stablemax"])
    sm_scale = float(cfg.get("t1_stablemax_scale", 2.0))
    ce_fn = (lambda lg, y: stablemax_cross_entropy(lg * sm_scale, y)) if use_sm else F.cross_entropy

    extra_groups = [
        {
            "name": "track1_cls",
            "params": list(cls_head.parameters()),
            "use_muon": False,
            "role": "fast",
            "mode": "acquire",
            "lr": 5e-3,
            "lr_consolidate": 3e-3,
            "weight_decay": 0.02,
            "prec_scale": 0.25,
        }
    ]
    trainer = CognitiveTrainer(
        model,
        seed=seed,
        lr_muon=float(cfg["t1_lr_muon"]),
        lr_adam=float(cfg["t1_lr_adam"]),
        lr_consolidate=2e-3,
        wd_trunk=float(cfg["t1_wd_trunk"]),
        wd_fast=0.02,
        trunk_prec_scale=0.85,
        grokfast_alpha=0.0,
        grokfast_lamb=0.0,
        ortho_grad=bool(cfg["t1_ortho_grad"]),
        bong_prior_pull=0.02,
        bong_max_prec=100.0,
        grad_clip=1.5,
        enable_gpm=True,
        exempt_readout_heads=True,
        extra_param_groups=extra_groups,
    )
    op_params = [p for n, p in model.named_parameters() if "long_term.op_" in n and p.requires_grad]
    op_opt = (
        torch.optim.AdamW(op_params, lr=float(cfg["t1_op_lr"]), weight_decay=1e-4)
        if (arm in ("CognitiveDMoE_v12", "CognitiveDMoE_v12_NoProcrustes") and op_params)
        else None
    )

    D_h = model.d_model
    ep_start_step = int(n_p1 * float(cfg["t1_ep_start_frac"]))
    train_ep = arm in ("CognitiveDMoE_v12", "CognitiveDMoE_v12_NoProcrustes")
    grokking_curve: List[float] = []

    def step_t1(xs: torch.Tensor, y: torch.Tensor, source_id: str, ph: str, step_i: int) -> None:
        xb_raw = encode_belief_chunks(xs, paraphrase=False).to(DEV)
        y_dev = y.to(DEV)
        b_sz = xs.size(0)
        if ph == "p1":
            xb_para = encode_belief_chunks(xs, paraphrase=True).to(DEV)
            xb = torch.cat([xb_raw, xb_para], dim=0)
            y_train = torch.cat([y_dev, y_dev], dim=0)
        else:
            xb = xb_raw
            y_train = y_dev

        if ph == "p1" and step_i >= (n_p1 // 2) and (step_i % 2 == 0):
            with torch.no_grad():
                model.eval()
                o_reg = model(xb_raw, update_stats=False)
                lg_reg = cls_head(o_reg["latent"][:, -1, :])
                trainer.gate.triage_batch(
                    xb_raw.detach(),
                    y_dev.detach(),
                    lg_reg.detach(),
                    o_reg["z_expr_pool"].detach(),
                    source_id=source_id,
                    skip_familiar=False,
                    enable_quarantine=False,
                )

        if ph != "p1":
            with torch.no_grad():
                model.eval()
                o_scr = model(xb, update_stats=False)
                lg_scr = cls_head(o_scr["latent"][:, -1, :])
                xb_act, y_act, flips = trainer.gate.triage_batch(
                    xb.detach(),
                    y_train.detach(),
                    lg_scr.detach(),
                    o_scr["z_expr_pool"].detach(),
                    source_id=source_id,
                    skip_familiar=False,
                    enable_quarantine=True,
                )
            if xb_act is None or y_act is None:
                return
            xb, y_train = xb_act, y_act
            if flips > 0:
                trainer.opt.temper_precision(roles=("fast",), factor=0.25)

        model.train()
        saved = trainer.gpm.snapshot_monitored_weights() if (trainer.gpm is not None and trainer.gpm.p_null) else None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(xb, update_stats=True)
            lg = cls_head(out["latent"][:, -1, :])

            l_cls = ce_fn(lg, y_train)
            l_para = (
                (1.0 - (out["z_expr_pool"][:b_sz].float() * out["z_expr_pool"][b_sz:].float()).sum(dim=-1)).mean()
                if ph == "p1"
                else 0.0
            )
            wm_l = out["world_model_loss"]["loss"] if out.get("world_model_loss") is not None else 0.0
            l_ab_cal = 0.0
            if ph == "p1":
                ab_in = out["abstain_logits"][:b_sz, -1, 0].float()
                l_ab_cal = F.binary_cross_entropy_with_logits(ab_in, torch.zeros_like(ab_in))
                if step_i % 4 == 0:
                    xu_ood = torch.randint(33, 122, (16, 24), device=DEV)
                    ou_ood = model(xu_ood, update_stats=False)
                    ab_ood = ou_ood["abstain_logits"][:, -1, 0].float()
                    l_ab_cal = l_ab_cal + F.binary_cross_entropy_with_logits(ab_ood, torch.ones_like(ab_ood))

            loss = l_cls + 0.06 * l_para + 0.03 * wm_l + 0.08 * l_ab_cal

            # Causal Abstraction Interchange-Intervention Training (IIT)
            if ph == "p1" and step_i >= (n_p1 // 3) and (step_i % 2 == 0) and b_sz >= 64:
                xs_b, xs_d = xs[:32], xs[32:64]
                x_b_in, x_d_in = xb_raw[:32], xb_raw[32:64]
                o_cf_m = model.forward_interchange(
                    x_b_in, x_d_in, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2), force_thinking_steps=1
                )
                y_cf_m, _, _ = task._compute_targets(
                    xs_d[:, 0], xs_d[:, 1] - 4, xs_d[:, 2] - 4, xs_b[:, 3] - 4, xs_b[:, 4] - 4, xs_b[:, 5], "p1"
                )
                o_cf_c = model.forward_interchange(
                    x_b_in, x_d_in, intervene_chunks=(3, 5), dim_slice=(D_h // 2, (3 * D_h) // 4), force_thinking_steps=1
                )
                y_cf_c, _, _ = task._compute_targets(
                    xs_b[:, 0], xs_b[:, 1] - 4, xs_b[:, 2] - 4, xs_d[:, 3] - 4, xs_d[:, 4] - 4, xs_b[:, 5], "p1"
                )
                loss = loss + 0.30 * ce_fn(cls_head(o_cf_m["latent"][:, -1, :]), y_cf_m.to(DEV))
                loss = loss + 0.30 * ce_fn(cls_head(o_cf_c["latent"][:, -1, :]), y_cf_c.to(DEV))

            # Decoupled Post-Thinker Episodic Induction Training (identical cadence across v12 & NoProcrustes)
            ep_loss_detached_step: Optional[torch.Tensor] = None
            if ph == "p1" and train_ep and op_opt is not None and (step_i >= ep_start_step) and (step_i % 2 == 1):
                ep_mode = (step_i // 2) % 3
                n_q_ep = 1
                if ep_mode == 0:
                    x_sup_ep, x_q_ep, y_q_ep, _, _ = _sample_episodic_honest_batch(task, 16, gen, n_shots=8)
                elif ep_mode == 1:
                    x_sup_ep, x_q_ep, y_q_ep = _sample_comp_equiv_batch(task, 16, gen, n_shots=8)
                else:
                    n_q_ep = 4
                    x_sup_ep, x_q_ep, y_q_ep = _sample_unseen_shift_multiquery_batch(
                        task, 16, gen, n_shots=8, n_queries=n_q_ep
                    )
                ep_kv = _pack_episode_memory(model, x_sup_ep, top_k=6)
                if ep_kv is not None:
                    if n_q_ep > 1:
                        ep_kv = {k: v.repeat_interleave(n_q_ep, dim=0) for k, v in ep_kv.items()}
                    o_ep = model(x_q_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                    lg_ep = F.linear(
                        o_ep["latent"][:, -1, :],
                        cls_head.weight.detach(),
                        cls_head.bias.detach(),
                    )
                    # Always use crisp F.cross_entropy on decoupled post-Thinker op_* head
                    ep_loss_detached_step = F.cross_entropy(lg_ep, y_q_ep.to(DEV))
            elif ph == "p1" and (not train_ep) and (step_i % 3 == 1) and step_i >= (n_p1 // 5):
                x_sup_ep, x_ql_ep, y_ql_ep, _, _ = _sample_episodic_honest_batch(task, 16, gen, n_shots=4)
                ep_kv = _pack_episode_memory(model, x_sup_ep, top_k=6)
                if ep_kv is not None:
                    o_ep = model(x_ql_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                    loss = loss + 0.40 * ce_fn(cls_head(o_ep["latent"][:, -1, :]), y_ql_ep.to(DEV))

        trainer.step_loss(
            loss,
            lifecycle_events=(),
            saved_weights=saved,
            fisher_scale=1.0,
            extra_clip_params=list(cls_head.parameters()),
        )
        if ep_loss_detached_step is not None and op_opt is not None:
            op_opt.zero_grad()
            ep_loss_detached_step.backward()
            nn.utils.clip_grad_norm_(op_params, 2.0)
            op_opt.step()

    @torch.no_grad()
    def eval_t1(phase: str, pair_split: str, para: bool = False) -> float:
        model.eval()
        hits, tot = 0, 0
        for q_tok in (Q_VAL, Q_ATTR, Q_PROP):
            xs, ys = task.sample_eval_subset(q_tok, phase, [0, 1, 2], pair_split, None, None, 128, gen)
            xb = encode_belief_chunks(xs, paraphrase=para).to(DEV)
            o = model(xb, update_stats=False)
            pr = cls_head(o["latent"][:, -1, :]).argmax(-1).cpu()
            hits += int((pr == ys).sum().item())
            tot += ys.numel()
        return hits / max(1, tot)

    base_lrs = [float(g["lr"]) for g in trainer.opt.param_groups]
    use_cos_anneal = bool(cfg.get("t1_cosine_anneal", True))
    anneal_start = int(0.55 * n_p1)
    for st in range(n_p1):
        if use_cos_anneal and st >= anneal_start:
            prog = float(st - anneal_start) / float(max(1, n_p1 - anneal_start))
            scale_lr = 0.35 + 0.65 * 0.5 * (1.0 + math.cos(math.pi * prog))
            for g, blr in zip(trainer.opt.param_groups, base_lrs):
                g["lr"] = blr * scale_lr
        xs, y = task.sample_stream_batch("p1", 64, gen)
        step_t1(xs, y, f"s_{st % 2}", "p1", st)
        if (st + 1) % 100 == 0 or st == n_p1 - 1:
            grokking_curve.append(eval_t1("p1", "heldout"))

    trainer.consolidate_domain(0)

    for st in range(n_pb):
        xs, y = task.sample_stream_batch("pb", 64, gen)
        step_t1(xs, y, "adversarial_burst", "pb", st)

    with torch.no_grad():
        model.eval()
        xs_s, ys_s = task.sample_eval_subset(Q_ATTR, "p1", [0, 1, 2], "seen", None, None, 128, gen)
        o_s = model(encode_belief_chunks(xs_s).to(DEV), update_stats=False)
        burst_acc = float((cls_head(o_s["latent"][:, -1, :]).argmax(-1).cpu() == ys_s).float().mean())

    for st in range(n_p2):
        xs, y = task.sample_stream_batch("p2", 64, gen)
        step_t1(xs, y, f"s2_{st % 2}", "p2", st)
    for st in range(n_p3):
        xs, y = task.sample_stream_batch("p3", 64, gen)
        step_t1(xs, y, f"s3_{st % 2}", "p3", st)

    with torch.no_grad():
        model.eval()
        xs_k, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xk_b = encode_belief_chunks(xs_k).to(DEV)
        xu_b = torch.randint(33, 122, (64, 24), device=DEV)
        ok = model(xk_b, update_stats=False)
        ou = model(xu_b, update_stats=False)
        sk = (0.5 * ok["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ok["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        su = (0.5 * ou["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ou["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        abstain_auroc = roc_auc_pairs(su, sk)

        near_ood_lines = [
            f"Q9 C{int(xs_k[i, 1] - 4):02d} C{int(xs_k[i, 2] - 4):02d} C00 C00 ? ".encode("ascii")
            for i in range(64)
        ]
        x_near_ood = encode_texts_v8(near_ood_lines, 24, DEV, pad_id=257)
        o_near_ood = model(x_near_ood, update_stats=False)
        sn = (0.5 * o_near_ood["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (o_near_ood["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        near_ood_auroc = roc_auc_pairs(sn, sk)

        xs_iit_b, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "seen", None, None, 64, gen)
        xs_iit_d, _ = task.sample_eval_subset(Q_ATTR, "p3", [0, 1, 2], "seen", None, None, 64, gen)
        xk_iit_b, xd_iit_b = encode_belief_chunks(xs_iit_b).to(DEV), encode_belief_chunks(xs_iit_d).to(DEV)

        xs_iit_hb, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xs_iit_hd, _ = task.sample_eval_subset(Q_ATTR, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xk_iit_hb, xd_iit_hd = encode_belief_chunks(xs_iit_hb).to(DEV), encode_belief_chunks(xs_iit_hd).to(DEV)

        o_seen_causal = model.forward_interchange(xk_iit_b, xd_iit_b, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2))
        y_cf_seen, _, _ = task._compute_targets(
            xs_iit_d[:, 0], xs_iit_d[:, 1] - 4, xs_iit_d[:, 2] - 4, xs_iit_b[:, 3] - 4, xs_iit_b[:, 4] - 4, xs_iit_b[:, 5], "p3"
        )
        iit_seen_acc = float((cls_head(o_seen_causal["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_seen).float().mean().item())

        o_held_causal = model.forward_interchange(xk_iit_hb, xd_iit_hd, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2))
        y_cf_held, _, _ = task._compute_targets(
            xs_iit_hd[:, 0], xs_iit_hd[:, 1] - 4, xs_iit_hd[:, 2] - 4, xs_iit_hb[:, 3] - 4, xs_iit_hb[:, 4] - 4, xs_iit_hb[:, 5], "p3"
        )
        iit_held_acc = float((cls_head(o_held_causal["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_held).float().mean().item())

        o_seen_rand = model.forward_interchange(xk_iit_b, xd_iit_b, intervene_chunks=(0, 3), dim_slice=(D_h // 4, (3 * D_h) // 4))
        iit_rand_null_acc = float((cls_head(o_seen_rand["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_seen).float().mean().item())

        ep_suite = eval_v10_episodic_suite(model, cls_head, task, gen, shot_counts=(4, 8, 16), n_episodes=64)

    return {
        "train_seconds": time.time() - t_start,
        "seen_acc": eval_t1("p3", "seen"),
        "heldout_unseen_acc": eval_t1("p3", "heldout"),
        "paraphrase_acc": eval_t1("p3", "heldout", para=True),
        "iit_causal_acc": iit_seen_acc,
        "iit_heldout_acc": iit_held_acc,
        "iit_random_slice_null": iit_rand_null_acc,
        "episodic_v10": ep_suite,
        "burst_immunity_acc": burst_acc,
        "abstain_auroc": abstain_auroc,
        "near_ood_abstain_auroc": near_ood_auroc,
        "grokking_curve": grokking_curve,
    }


def run_track2_arm_v12(
    arm: str,
    cfg: Dict[str, Any],
    bench: V8RealBenchmark,
    seed: int,
    steps_per_dom: int = 250,
) -> Dict[str, Any]:
    if arm == "CognitiveDMoE_v11":
        return run_track2_arm_v11("CognitiveDMoE", dict(DEFAULT_V11_CFG), bench, seed, steps_per_dom=steps_per_dom)

    set_all_seeds(seed)
    model = _build_v12_model(arm, cfg, is_track1=False, min_mature_steps=max(20, int(steps_per_dom * 0.40)))
    trainer = CognitiveTrainer(
        model,
        seed=seed,
        lr_muon=8e-3,
        lr_adam=3e-3,
        lr_consolidate=2e-3,
        wd_trunk=0.02,
        wd_fast=0.01,
        trunk_prec_scale=0.85,
        bong_prior_pull=0.02,
        bong_max_prec=100.0,
        grad_clip=2.5,
        enable_gpm=True,
        exempt_readout_heads=True,
        decoder_lr=5e-3,
    )

    i7_steps = int(cfg["i7_steps"])
    use_cris = arm == "CognitiveDMoE_v12"
    cris_w = float(cfg["cris_weight"])
    orth_w = float(cfg["exp_orth_weight"])
    pert_every = max(1, int(cfg.get("pert_every", 4)))

    t_start = time.time()
    gen_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    if i7_steps > 0:
        model.train()
        for st_i7 in range(i7_steps):
            x_i7, eq_i7, c_i7, nt_i7, st_i7_t = generate_i7_procedural_batch(16, rng_py, seq_len=SEQ_LEN, device=DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                o_i7 = model(x_i7, update_stats=False)
                rq_p_i7 = None
                rq_idx_i7 = None
                if use_cris and (st_i7 % pert_every == 0):
                    xb_p = _perturb_byte_tensor(x_i7, rng_py)
                    rq_p_i7, _ = model.extract_router_queries(xb_p)
                    rq_idx_i7 = torch.arange(x_i7.size(0), device=DEV)
                l_i7 = model.compute_cognitive_loss(
                    o_i7,
                    x=x_i7,
                    equiv_labels=eq_i7,
                    contra_pair_idx=c_i7,
                    nuance_targets=nt_i7,
                    struct_targets=st_i7_t,
                    wm_weight=0.25,
                    fsp_weight=0.10,
                    equiv_weight=1.25,
                    contra_weight=1.25,
                    think_weight=0.25,
                    struct_weight=0.35,
                    router_q_perturbed=rq_p_i7,
                    router_q_pert_idx=rq_idx_i7,
                    cris_weight=(cris_w if use_cris else 0.0),
                    exp_orth_weight=(orth_w if use_cris else 0.0),
                )
            trainer.step_loss(l_i7["loss"], fisher_scale=0.5)

    for dom_idx, dom in enumerate(DOMAINS):
        model.train()
        for step in range(steps_per_dom):
            xb_raw, y_eq, contra_pair, hard_pair, nuance_t, struct_t, _ = bench.sample_v8_stream_batch(dom, gen_np, rng_py)
            saved = trainer.gpm.snapshot_monitored_weights() if (trainer.gpm is not None and trainer.gpm.p_null) else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(xb_raw, update_stats=True)
                rq_pert = None
                rq_pert_idx = None
                if use_cris and (step % pert_every == 0):
                    # 8-sample balanced perturbation across raw windows, edit triples, and B77 (<0.5ms under no_grad trunk)
                    pert_idx = torch.tensor([0, 1, 4, 5, 6, 10, 11, 14], dtype=torch.long, device=DEV)
                    xb_pert = _perturb_byte_tensor(xb_raw[pert_idx], rng_py)
                    rq_pert, _ = model.extract_router_queries(xb_pert)
                    rq_pert_idx = pert_idx

                loss_dict = model.compute_cognitive_loss(
                    out,
                    x=xb_raw,
                    equiv_labels=y_eq,
                    contra_pair_idx=contra_pair,
                    hard_neg_pair_idx=hard_pair,
                    nuance_targets=nuance_t,
                    struct_targets=struct_t,
                    router_q_perturbed=rq_pert,
                    router_q_pert_idx=rq_pert_idx,
                    cris_weight=(cris_w if use_cris else 0.0),
                    exp_orth_weight=(orth_w if use_cris else 0.0),
                )
            trainer.step_loss(
                loss_dict["loss"],
                lifecycle_events=out.get("lifecycle_events", ()),
                saved_weights=saved,
                fisher_scale=1.0,
            )
        trainer.consolidate_domain(dom_idx)

    train_sec = time.time() - t_start

    dec_diag: Dict[str, Any] = {}
    if arm == "CognitiveDMoE_v12":
        core_snap = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("decoder.")}
        dec_texts = list(bench.b77_tr_txt[:32])
        x_dec = encode_texts_v8(dec_texts, SEQ_LEN, DEV, pad_id=257)
        model.eval()
        with torch.no_grad():
            out_dec = model(x_dec, update_stats=False)
        nll_start = float(model.compute_detached_decoder_loss(out_dec, x_dec).item())
        for _ in range(30):
            nll_end = trainer.step_decoder(out_dec, x_dec)
        max_core_drift = max(
            float((p.detach() - core_snap[n]).abs().max().item())
            for n, p in model.named_parameters()
            if not n.startswith("decoder.")
        )
        gen_sample = model.generate(dec_texts[0][:16], max_new_chunks=4, greedy=True)
        dec_diag = {
            "decoder_nll_start": nll_start,
            "decoder_nll_end": nll_end,
            "core_weight_max_drift": max_core_drift,
            "generated_byte_len": int(gen_sample["byte_ids"].size(1)),
        }

    model.eval()
    b77_res = eval_banking77_v8(model, bench)
    per_dom = {d: eval_domain_v8(model, bench, d) for d in DOMAINS}
    mean_nf_auc = float(np.mean([per_dom[d]["nuance_flip_auc"] for d in DOMAINS]))
    mean_r5 = float(np.mean([per_dom[d]["concept_r5"] for d in DOMAINS]))
    hf_res = eval_triples_nuance_v8(model, bench.heldout_family_pairs)
    rd = eval_routing_diagnostics_v8(model, bench)
    sem_route = eval_v10_semantic_routing_suite(model, bench)

    k_hist = rd.get("k_dist", {"k0": 0.0, "k1": 1.0, "k2": 0.0})
    mean_k_val = float(k_hist.get("k0", 0.0) * 0.0 + k_hist.get("k1", 1.0) * 1.0 + k_hist.get("k2", 0.0) * 2.0)

    return {
        "arm": arm,
        "teacher_informed": False,
        "seed": seed,
        "train_seconds": train_sec,
        "total_spawns": trainer.total_spawns,
        "arch_summary": model.architecture_summary(),
        "detached_decoder_diag": dec_diag,
        "banking77": b77_res,
        "mean_nuance_flip_auc": mean_nf_auc,
        "heldout_families": hf_res,
        "mean_final_r5": mean_r5,
        "per_domain_final": per_dom,
        "routing_diag": {
            "mid_seq_switch_rate": rd.get("mid_seq_switch_rate", 0.0),
            "k_dist": k_hist,
            "mean_k": mean_k_val,
            **sem_route,
        },
    }


def _append_tuning_log(entry: Dict[str, Any]) -> None:
    log_path = RESULTS_ROOT / "tuning_log.json"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        data = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        data = {"exp_id": "real_pilot_v9", "trials": []}
    data["trials"].append(entry)
    log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "dev", "test"], default="dev")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tracks", choices=["all", "track1", "track2"], default="all")
    ap.add_argument("--out_name", type=str, default=None)
    ap.add_argument("--arms", type=str, nargs="+", default=None)
    ap.add_argument("--t1_lr_muon", type=float, default=None)
    ap.add_argument("--t1_lr_adam", type=float, default=None)
    ap.add_argument("--t1_wd_trunk", type=float, default=None)
    ap.add_argument("--t1_op_lr", type=float, default=None)
    ap.add_argument("--t1_ep_start_frac", type=float, default=None)
    ap.add_argument("--op_procrustes_planes", type=int, default=None)
    ap.add_argument("--cris_weight", type=float, default=None)
    ap.add_argument("--exp_orth_weight", type=float, default=None)
    ap.add_argument("--router_anchor_weight", type=float, default=None)
    ap.add_argument("--specialist_init_scale", type=float, default=None)
    ap.add_argument("--z_nuance_thresh", type=float, default=None)
    ap.add_argument("--pert_every", type=int, default=None)
    ap.add_argument("--no_hnet", action="store_true")
    ap.add_argument("--no_dual_gran", action="store_true")
    ap.add_argument("--use_stablemax", action="store_true")
    ap.add_argument("--no_ortho_grad", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse existing per-seed JSON files in out_dir.")
    ap.add_argument("--log_dev_note", type=str, default=None, help="Append this dev trial to tuning_log.json.")
    args = ap.parse_args()

    cfg = dict(DEFAULT_V12_CFG)
    for k in (
        "t1_lr_muon",
        "t1_lr_adam",
        "t1_wd_trunk",
        "t1_op_lr",
        "t1_ep_start_frac",
        "cris_weight",
        "exp_orth_weight",
        "router_anchor_weight",
        "specialist_init_scale",
    ):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = float(v)
    if args.op_procrustes_planes is not None:
        cfg["op_procrustes_planes"] = int(args.op_procrustes_planes)
    if args.pert_every is not None:
        cfg["pert_every"] = int(args.pert_every)
    if args.z_nuance_thresh is not None:
        cfg["z_nuance_thresh"] = float(args.z_nuance_thresh)
        cfg["d_sem_nuance_thresh"] = float(args.z_nuance_thresh)
    if args.no_hnet:
        cfg["enable_hnet_boundary"] = False
    if args.no_dual_gran:
        cfg["dual_granularity_router"] = False
    if args.use_stablemax:
        cfg["t1_use_stablemax"] = True
    if args.no_ortho_grad:
        cfg["t1_ortho_grad"] = False

    if args.mode == "smoke":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 25
        default_arms = ["CognitiveDMoE_v12"]
        out_name = args.out_name or "smoke_v12"
        t1_scale = 0.2
    elif args.mode == "dev":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 250
        default_arms = (
            ["CognitiveDMoE_v12"]
            if args.tracks == "track1"
            else ["CognitiveDMoE_v12", "CognitiveDMoE_v12_FrozenRandomRouter"]
        )
        out_name = args.out_name or f"dev_{args.tracks}"
        t1_scale = 1.0
    else:
        eval_split = "test"
        if args.tracks == "track1":
            seeds = args.seeds or [0, 1, 2, 3, 4]
            default_arms = [
                "CognitiveDMoE_v12",
                "CognitiveDMoE_v12_NoProcrustes",
                "CognitiveDMoE_v11",
                "Dense_Matched",
            ]
        else:
            seeds = args.seeds or [0, 1, 2]
            default_arms = [
                "CognitiveDMoE_v12",
                "CognitiveDMoE_v12_FrozenRandomRouter",
                "CognitiveDMoE_v11",
                "Dense_Matched",
            ]
        steps_per_dom = args.steps or 250
        out_name = args.out_name or f"test_{args.tracks}"
        t1_scale = 1.0

    arms = args.arms or default_arms
    out_dir = RESULTS_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _collect_meta(args.mode, eval_split, seeds, steps_per_dom, args.tracks, cfg)

    bench: Optional[V8RealBenchmark] = None
    if args.tracks in ("all", "track2"):
        corpora = RealCorpora(seq_len=SEQ_LEN)
        bench = V8RealBenchmark(corpora, eval_split=eval_split)

    all_runs: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    for arm in arms:
        for seed in seeds:
            run_path = out_dir / f"{arm}_seed{seed}.json"
            if args.resume and run_path.exists():
                rec = json.loads(run_path.read_text(encoding="utf-8"))
                print(f"\n=== [v12 {args.mode} | {args.tracks}] Arm={arm} Seed={seed} (RESUMED) ===", flush=True)
            else:
                print(f"\n=== [v12 {args.mode} | {args.tracks}] Arm={arm} Seed={seed} ===", flush=True)
                rec: Dict[str, Any] = {"arm": arm, "seed": seed, "teacher_informed": False}
                if args.tracks in ("all", "track2"):
                    assert bench is not None
                    t2 = run_track2_arm_v12(arm, cfg, bench, seed, steps_per_dom=steps_per_dom)
                    rec.update(t2)
                    rd = t2["routing_diag"]
                    print(
                        f"  [Track 2] b77={t2['banking77']['intent_77_acc']:.3f} "
                        f"b77_auc={t2['banking77']['same_vs_overlap_diff_auc']:.3f} "
                        f"r5={t2['mean_final_r5']:.3f} hf={t2['heldout_families']['auc']:.3f} "
                        f"nf={t2['mean_nuance_flip_auc']:.3f} k={rd['mean_k']:.3f} "
                        f"E={rd['num_experts']} time={t2['train_seconds']:.1f}s",
                        flush=True,
                    )
                    print(
                        f"  [Route]   ami={rd['func_routing_ami']['ami']:.3f} "
                        f"lm_ami={rd['length_matched_role_ami']['ami']:.3f} "
                        f"len_ami={rd['routing_length_ami']:.3f} "
                        f"pert={rd['routing_perturb_agree']:.3f} "
                        f"les_rel={rd['lesion_mean_rel_delta']:.3f} "
                        f"les_drop={rd['lesion_max_acc_drop']:.3f}",
                        flush=True,
                    )
                    if t2.get("detached_decoder_diag"):
                        dd = t2["detached_decoder_diag"]
                        print(
                            f"  [Decoder] nll={dd['decoder_nll_start']:.3f}->{dd['decoder_nll_end']:.3f} "
                            f"core_drift={dd['core_weight_max_drift']:.1e}",
                            flush=True,
                        )
                if args.tracks in ("all", "track1"):
                    t1 = run_track1_arm_v12(arm, cfg, seed, steps_scale=t1_scale)
                    rec["track1"] = t1
                    ep8 = t1["episodic_v10"]["shots_8"]
                    ep16 = t1["episodic_v10"]["shots_16"]
                    print(
                        f"  [Track 1] seen={t1['seen_acc']:.3f} held={t1['heldout_unseen_acc']:.3f} "
                        f"iit_h={t1['iit_heldout_acc']:.3f} burst={t1['burst_immunity_acc']:.3f} "
                        f"near_ood={t1['near_ood_abstain_auroc']:.3f} time={t1['train_seconds']:.1f}s",
                        flush=True,
                    )
                    print(
                        f"  [Ep 8/16] comp={ep8['comp_equiv_induction_acc']:.3f}/{ep16['comp_equiv_induction_acc']:.3f} "
                        f"shift={ep8['unseen_meaning_shift_acc']:.3f}/{ep16['unseen_meaning_shift_acc']:.3f} "
                        f"lu={ep8['lookup_acc']:.3f}/{ep16['lookup_acc']:.3f} "
                        f"null_sh={ep8['shift_shuffled_null_acc']:.3f} null_imp={ep8['impossible_perm_null_acc']:.3f}",
                        flush=True,
                    )
                run_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            all_runs[arm].append(rec)

    summary = {
        "meta": meta,
        "arms": {arm: _agg_v11_runs(runs) for arm, runs in all_runs.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.mode == "dev" and args.log_dev_note is not None:
        _append_tuning_log(
            {
                "note": args.log_dev_note,
                "out_name": out_name,
                "tracks": args.tracks,
                "seeds": seeds,
                "utc": meta["utc"],
                "cfg": cfg,
                "summary": summary["arms"],
            }
        )
    print(f"\nSaved summary -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

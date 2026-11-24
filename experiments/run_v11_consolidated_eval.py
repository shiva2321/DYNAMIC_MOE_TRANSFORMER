"""
Consolidated CognitiveDMoE (v11) Evaluation Runner — `real_pilot_v8`.

Pre-registered in `results/real_pilot_v8/PREREG.md`.
Evaluates `CognitiveDMoE` head-to-head against matched baselines (`CognitiveDMoE_NoDualOp`,
`CognitiveDMoE_FrozenRandomRouter`, `Cog_v10`, `Dense_Matched`) and mandatory null controls
across:
  - Track 1 (5 disjoint test seeds `0..4`): Modular belief composition grokking, causal IIT
    generalization, adversarial burst immunity, near/far-OOD abstention, and the honest
    6-probe zero-weight-update episodic induction suite (lookup, comp_equiv, unseen_meaning_shift,
    and 4 null controls across 4/8/16 shots).
  - Track 2 (3 disjoint test seeds `0..2`): 4-domain continual real text stream (`code`, `math`,
    `prose`, `banking77`), Banking77 intent & overlap-trap AUC, concept retrieval R@5, held-out
    nuance families, compute `mean_k`, train time, honest semantic routing & perturbation suite,
    stratified out-of-sample frozen-probe multi-layer expert lesioning, AND end-to-end detached
    byte-decoder NLL + zero-core-drift verification.

Dev tuning is strictly restricted to disjoint `seed = 100` (`eval_split = "dev"`), max 4 trials
per track, logged to `results/real_pilot_v8/tuning_log.json`.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import BeliefMeaningTask, K_CONCEPTS, Q_ATTR, Q_PROP, Q_VAL
from dmoe.cognitive_dmoe import CognitiveDMoE, CognitiveDMoEConfig, CognitiveTrainer
from dmoe.null_space import NullSpaceProjector
from dmoe.spgc import LatentQuarantineGate, MuonBONGOptimizer, classify_v7_spgc_param_groups
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
    DEFAULT_V10_CFG,
    _pack_episode_memory,
    _perturb_byte_tensor,
    _sample_comp_equiv_batch,
    _sample_unseen_meaning_shift_batch,
    build_v10_arm,
    eval_v10_episodic_suite,
    eval_v10_semantic_routing_suite,
    run_track1_arm_v10,
    run_track2_arm_v10,
)

RESULTS_ROOT = ROOT / "results" / "real_pilot_v8"

DEFAULT_V11_CFG: Dict[str, Any] = {
    # Track 1 Grokfast + SPGC acquisition settings (selected from Dev Trial 2 on seed 100)
    "t1_lr_muon": 9.0e-3,
    "t1_lr_adam": 3.5e-3,
    "t1_wd_trunk": 0.14,
    "t1_grokfast_alpha": 0.98,
    "t1_grokfast_lamb": 2.0,
    "t1_ep_start_frac": 0.50,
    "t1_cooldown": False,
    "dual_ridge_op_rank": 48,
    "dual_ridge_op_rank_low": 12,
    "dual_ridge_v_mlp": True,
    # Track 2 Continual Semantic Routing settings (frozen from Dev Trial 3 on seed 100)
    "cris_weight": 0.25,
    "exp_orth_weight": 0.15,
    "pert_every": 4,
    "shared_dropout_prob": 0.15,
    "specialist_init_scale": 0.25,
    "z_nuance_thresh": 1.6,
    "d_sem_nuance_thresh": 1.6,
    "shift_invariant_pool": False,
    "frozen_exempt_balance": True,
    "residual_ortho_router": True,
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
        "exp_id": "real_pilot_v8",
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


def _build_v11_model(
    arm: str,
    cfg: Dict[str, Any],
    *,
    is_track1: bool = False,
    min_mature_steps: int = 100,
) -> CognitiveDMoE:
    """Build a `CognitiveDMoE` variant for `real_pilot_v8` arms."""
    if arm == "CognitiveDMoE":
        mcfg = CognitiveDMoEConfig(
            shift_invariant_pool=(False if is_track1 else bool(cfg["shift_invariant_pool"])),
            enable_surface_whitening=(False if is_track1 else True),
            enable_dual_ridge_ep=True,
            dual_ridge_op_rank=int(cfg["dual_ridge_op_rank"]),
            dual_ridge_op_rank_low=int(cfg["dual_ridge_op_rank_low"]),
            dual_ridge_v_mlp=bool(cfg["dual_ridge_v_mlp"]),
            frozen_exempt_balance=bool(cfg["frozen_exempt_balance"]),
            residual_ortho_router=bool(cfg["residual_ortho_router"]),
            enable_cris_loss=(False if is_track1 else True),
            specialist_init_scale=(0.05 if is_track1 else float(cfg["specialist_init_scale"])),
            shared_dropout_prob=(0.0 if is_track1 else float(cfg["shared_dropout_prob"])),
            z_nuance_thresh=float(cfg["z_nuance_thresh"]),
            d_sem_nuance_thresh=float(cfg["d_sem_nuance_thresh"]),
            min_mature_steps=int(min_mature_steps),
            max_experts=6,
        )
        return CognitiveDMoE(mcfg).to(DEV)
    elif arm == "CognitiveDMoE_NoDualOp":
        mcfg = CognitiveDMoEConfig(
            shift_invariant_pool=False,
            enable_surface_whitening=False,
            enable_dual_ridge_ep=False,
            frozen_exempt_balance=bool(cfg["frozen_exempt_balance"]),
            residual_ortho_router=bool(cfg["residual_ortho_router"]),
            enable_cris_loss=False,
            specialist_init_scale=0.05,
            shared_dropout_prob=0.0,
            min_mature_steps=int(min_mature_steps),
            max_experts=6,
        )
        return CognitiveDMoE(mcfg).to(DEV)
    elif arm == "CognitiveDMoE_FrozenRandomRouter":
        mcfg = CognitiveDMoEConfig(
            shift_invariant_pool=bool(cfg["shift_invariant_pool"]),
            enable_surface_whitening=False,
            enable_dual_ridge_ep=True,
            frozen_exempt_balance=bool(cfg["frozen_exempt_balance"]),
            residual_ortho_router=bool(cfg["residual_ortho_router"]),
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
            d_ff_shared=512,
            d_ff_spec=256,
            max_experts=1,
            shift_invariant_pool=(False if is_track1 else bool(cfg["shift_invariant_pool"])),
            enable_surface_whitening=False,
            enable_dual_ridge_ep=False,
            frozen_exempt_balance=True,
            residual_ortho_router=True,
            enable_cris_loss=False,
            specialist_init_scale=0.05,
            shared_dropout_prob=0.0,
            min_mature_steps=int(min_mature_steps) + 5000,
        )
        return CognitiveDMoE(mcfg).to(DEV)
    else:
        raise ValueError(f"Unsupported v11 arm: {arm}")


def run_track1_arm_v11(
    arm: str,
    cfg: Dict[str, Any],
    seed: int,
    steps_scale: float = 1.0,
) -> Dict[str, Any]:
    if arm == "Cog_v10":
        return run_track1_arm_v10("Cog_v10", dict(DEFAULT_V10_CFG), seed, steps_scale=steps_scale)

    set_all_seeds(seed)
    t_start = time.time()
    task = BeliefMeaningTask(seed=seed)
    gen = torch.Generator().manual_seed(seed * 97 + 1)

    n_p1 = max(220, int(1300 * steps_scale))
    n_pb = max(8, int(20 * steps_scale))
    n_p2 = max(40, int(180 * steps_scale))
    n_p3 = max(40, int(180 * steps_scale))

    model = _build_v11_model(arm, cfg, is_track1=True, min_mature_steps=n_p1 + 50)
    cls_head = nn.Linear(model.d_model, K_CONCEPTS).to(DEV)

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
        grokfast_alpha=float(cfg["t1_grokfast_alpha"]),
        grokfast_lamb=float(cfg["t1_grokfast_lamb"]),
        bong_prior_pull=0.02,
        bong_max_prec=100.0,
        grad_clip=1.5,
        enable_gpm=True,
        exempt_readout_heads=True,
        extra_param_groups=extra_groups,
    )
    # Dedicated optimizer for decoupled post-Thinker Dual Ridge parameters so large episodic
    # ridge gradients NEVER distort global trunk gradient clipping or grokking dynamics.
    op_params = [
        p for n, p in model.named_parameters()
        if "long_term.op_" in n and p.requires_grad
    ]
    op_opt = torch.optim.AdamW(op_params, lr=6e-3, weight_decay=1e-3) if (arm == "CognitiveDMoE" and op_params) else None

    D_h = model.d_model
    ep_start_step = int(n_p1 * float(cfg["t1_ep_start_frac"]))
    train_v11_ep = arm in ("CognitiveDMoE", "CognitiveDMoE_NoDualOp")
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

            l_cls = F.cross_entropy(lg, y_train)
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
                loss = loss + 0.30 * F.cross_entropy(cls_head(o_cf_m["latent"][:, -1, :]), y_cf_m.to(DEV))
                loss = loss + 0.30 * F.cross_entropy(cls_head(o_cf_c["latent"][:, -1, :]), y_cf_c.to(DEV))

            # Episodic Memory & Rule Induction Training (matched cadence for CognitiveDMoE & NoDualOp)
            ep_loss_detached_step: Optional[torch.Tensor] = None
            if ph == "p1":
                if train_v11_ep and (step_i >= ep_start_step) and (step_i % 2 == 1):
                    ep_mode = (step_i // 2) % 3
                    if ep_mode == 0:
                        x_sup_ep, x_q_ep, y_q_ep, _, _ = _sample_episodic_honest_batch(task, 16, gen, n_shots=8)
                    elif ep_mode == 1:
                        x_sup_ep, x_q_ep, y_q_ep = _sample_comp_equiv_batch(task, 16, gen, n_shots=8)
                    else:
                        x_sup_ep, x_q_ep, y_q_ep = _sample_unseen_meaning_shift_batch(task, 16, gen, n_shots=8)
                    ep_kv = _pack_episode_memory(model, x_sup_ep, top_k=6)
                    if ep_kv is not None:
                        o_ep = model(x_q_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                        if arm == "CognitiveDMoE" and op_opt is not None:
                            lg_ep = F.linear(
                                o_ep["latent"][:, -1, :],
                                cls_head.weight.detach(),
                                cls_head.bias.detach(),
                            )
                            ep_loss_detached_step = F.cross_entropy(lg_ep, y_q_ep.to(DEV))
                        else:
                            loss = loss + 0.50 * F.cross_entropy(cls_head(o_ep["latent"][:, -1, :]), y_q_ep.to(DEV))
                elif (not train_v11_ep) and (step_i % 3 == 1) and step_i >= (n_p1 // 5):
                    x_sup_ep, x_ql_ep, y_ql_ep, _, _ = _sample_episodic_honest_batch(task, 16, gen, n_shots=4)
                    ep_kv = _pack_episode_memory(model, x_sup_ep, top_k=6)
                    if ep_kv is not None:
                        o_ep = model(x_ql_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                        loss = loss + 0.40 * F.cross_entropy(cls_head(o_ep["latent"][:, -1, :]), y_ql_ep.to(DEV))

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

    for st in range(n_p1):
        if bool(cfg.get("t1_cooldown", False)) and st == int(0.75 * n_p1) and trainer.opt.grokfast_lamb > 0.0:
            trainer.opt.grokfast_lamb = 0.0
            for g in trainer.opt.param_groups:
                if g.get("role") == "trunk":
                    g["lr"] = float(g["lr"]) * 0.65
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


def run_track2_arm_v11(
    arm: str,
    cfg: Dict[str, Any],
    bench: V8RealBenchmark,
    seed: int,
    steps_per_dom: int = 250,
) -> Dict[str, Any]:
    if arm == "Cog_v10":
        return run_track2_arm_v10("Cog_v10", dict(DEFAULT_V10_CFG), bench, seed, steps_per_dom=steps_per_dom)

    set_all_seeds(seed)
    model = _build_v11_model(arm, cfg, is_track1=False, min_mature_steps=max(20, int(steps_per_dom * 0.40)))
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
    use_cris = arm == "CognitiveDMoE"
    cris_w = float(cfg["cris_weight"])
    orth_w = float(cfg["exp_orth_weight"])

    t_start = time.time()
    gen_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    if i7_steps > 0:
        model.train()
        for _ in range(i7_steps):
            x_i7, eq_i7, c_i7, nt_i7, st_i7 = generate_i7_procedural_batch(16, rng_py, seq_len=SEQ_LEN, device=DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                o_i7 = model(x_i7, update_stats=False)
                l_i7 = model.compute_cognitive_loss(
                    o_i7,
                    x=x_i7,
                    equiv_labels=eq_i7,
                    contra_pair_idx=c_i7,
                    nuance_targets=nt_i7,
                    struct_targets=st_i7,
                    wm_weight=0.25,
                    fsp_weight=0.10,
                    equiv_weight=1.25,
                    contra_weight=1.25,
                    think_weight=0.25,
                    struct_weight=0.35,
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
                pert_every = max(1, int(cfg.get("pert_every", 8)))
                if use_cris and (step % pert_every == 0):
                    pert_idx = torch.tensor([0, 4, 10, 12], dtype=torch.long, device=DEV)
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

    # End-to-end decoupled byte-decoder verification (strictly behind stop_gradient; 0.0 core drift!)
    dec_diag: Dict[str, Any] = {}
    if arm == "CognitiveDMoE":
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


def _stat_dict(vals: List[float]) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in vals],
    }


def _agg_v11_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_seeds": len(runs)}
    if not runs:
        return out
    if all("banking77" in r for r in runs):
        out["train_seconds"] = _stat_dict([r["train_seconds"] for r in runs])
        out["b77_intent_77_acc"] = _stat_dict([r["banking77"]["intent_77_acc"] for r in runs])
        out["b77_same_vs_overlap_diff_auc"] = _stat_dict([r["banking77"]["same_vs_overlap_diff_auc"] for r in runs])
        out["mean_final_r5"] = _stat_dict([r["mean_final_r5"] for r in runs])
        out["heldout_families_auc"] = _stat_dict([r["heldout_families"]["auc"] for r in runs])
        out["mean_nuance_flip_auc"] = _stat_dict([r["mean_nuance_flip_auc"] for r in runs])
        out["mean_k"] = _stat_dict([r["routing_diag"]["mean_k"] for r in runs])
        out["num_experts"] = _stat_dict([r["routing_diag"]["num_experts"] for r in runs])
        out["func_routing_ami"] = _stat_dict([r["routing_diag"]["func_routing_ami"]["ami"] for r in runs])
        out["length_matched_role_ami"] = _stat_dict([r["routing_diag"]["length_matched_role_ami"]["ami"] for r in runs])
        out["routing_length_ami"] = _stat_dict([r["routing_diag"]["routing_length_ami"] for r in runs])
        out["routing_perturb_agree"] = _stat_dict([r["routing_diag"]["routing_perturb_agree"] for r in runs])
        out["lesion_max_acc_drop"] = _stat_dict([r["routing_diag"]["lesion_max_acc_drop"] for r in runs])
        out["lesion_max_role_drop"] = _stat_dict([r["routing_diag"]["lesion_max_role_drop"] for r in runs])
        out["lesion_mean_rel_delta"] = _stat_dict([r["routing_diag"]["lesion_mean_rel_delta"] for r in runs])
        out["lesion_role_specificity"] = _stat_dict([r["routing_diag"]["lesion_role_specificity"] for r in runs])
    if all("track1" in r for r in runs):
        t1_keys = [
            "train_seconds",
            "seen_acc",
            "heldout_unseen_acc",
            "paraphrase_acc",
            "iit_causal_acc",
            "iit_heldout_acc",
            "iit_random_slice_null",
            "burst_immunity_acc",
            "abstain_auroc",
            "near_ood_abstain_auroc",
        ]
        out["track1"] = {k: _stat_dict([r["track1"][k] for r in runs]) for k in t1_keys}
        out["track1"]["episodic_v10"] = {}
        for sh in ("shots_4", "shots_8", "shots_16"):
            ep_keys = list(runs[0]["track1"]["episodic_v10"][sh].keys())
            out["track1"]["episodic_v10"][sh] = {
                ek: _stat_dict([r["track1"]["episodic_v10"][sh][ek] for r in runs]) for ek in ep_keys
            }
    return out


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
    ap.add_argument("--t1_grokfast_alpha", type=float, default=None)
    ap.add_argument("--t1_grokfast_lamb", type=float, default=None)
    ap.add_argument("--t1_ep_start_frac", type=float, default=None)
    ap.add_argument("--cris_weight", type=float, default=None)
    ap.add_argument("--exp_orth_weight", type=float, default=None)
    ap.add_argument("--shared_dropout_prob", type=float, default=None)
    ap.add_argument("--z_nuance_thresh", type=float, default=None)
    ap.add_argument("--resume", action="store_true", help="Reuse existing per-seed JSON files in out_dir.")
    args = ap.parse_args()

    cfg = dict(DEFAULT_V11_CFG)
    for k in (
        "t1_lr_muon",
        "t1_lr_adam",
        "t1_wd_trunk",
        "t1_grokfast_alpha",
        "t1_grokfast_lamb",
        "t1_ep_start_frac",
        "cris_weight",
        "exp_orth_weight",
        "shared_dropout_prob",
    ):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = float(v)
    if args.z_nuance_thresh is not None:
        cfg["z_nuance_thresh"] = float(args.z_nuance_thresh)
        cfg["d_sem_nuance_thresh"] = float(args.z_nuance_thresh)

    if args.mode == "smoke":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 25
        default_arms = ["CognitiveDMoE"]
        out_name = args.out_name or "smoke_v11"
        t1_scale = 0.2
    elif args.mode == "dev":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 250
        default_arms = (
            ["CognitiveDMoE"]
            if args.tracks == "track1"
            else ["CognitiveDMoE", "CognitiveDMoE_FrozenRandomRouter"]
        )
        out_name = args.out_name or "dev_v11"
        t1_scale = 1.0
    else:
        eval_split = "test"
        if args.tracks == "track1":
            seeds = args.seeds or [0, 1, 2, 3, 4]
            default_arms = ["CognitiveDMoE", "CognitiveDMoE_NoDualOp", "Cog_v10", "Dense_Matched"]
            out_name = args.out_name or "test_track1"
        elif args.tracks == "track2":
            seeds = args.seeds or [0, 1, 2]
            default_arms = ["CognitiveDMoE", "CognitiveDMoE_FrozenRandomRouter", "Cog_v10", "Dense_Matched"]
            out_name = args.out_name or "test_track2"
        else:
            seeds = args.seeds or [0, 1, 2]
            default_arms = ["CognitiveDMoE", "CognitiveDMoE_NoDualOp", "CognitiveDMoE_FrozenRandomRouter", "Cog_v10", "Dense_Matched"]
            out_name = args.out_name or "test_all"
        steps_per_dom = args.steps or 250
        t1_scale = 1.0

    arms = args.arms or default_arms
    out_dir = RESULTS_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _collect_meta(args.mode, eval_split, seeds, steps_per_dom, args.tracks, cfg)

    bench = None
    if args.tracks in ("all", "track2"):
        corpora = RealCorpora(seq_len=SEQ_LEN, eval_split=eval_split)
        bench = V8RealBenchmark(corpora, eval_split=eval_split)

    results_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    for seed in seeds:
        for arm in arms:
            seed_file = out_dir / f"seed{seed}_{arm}.json"
            if args.resume and seed_file.exists():
                res = json.loads(seed_file.read_text())
                results_by_arm[arm].append(res)
                print(f"[{out_name} s{seed} {arm}] RESUMED from {seed_file.name}", flush=True)
                continue
            res: Dict[str, Any] = {"meta": meta, "arm": arm, "seed": seed}
            if args.tracks in ("all", "track2"):
                assert bench is not None
                res.update(run_track2_arm_v11(arm, cfg, bench, seed, steps_per_dom=steps_per_dom))
            if args.tracks in ("all", "track1"):
                res["track1"] = run_track1_arm_v11(arm, cfg, seed, steps_scale=t1_scale)

            results_by_arm[arm].append(res)
            seed_file.write_text(json.dumps(res, indent=2))

            parts = [f"[{out_name} s{seed} {arm}]"]
            if "banking77" in res:
                rd = res["routing_diag"]
                dd = res.get("detached_decoder_diag", {})
                dec_s = (
                    f" decNLL={dd['decoder_nll_start']:.2f}->{dd['decoder_nll_end']:.2f}(drift={dd['core_weight_max_drift']:.0e})"
                    if dd
                    else ""
                )
                parts.append(
                    f"T2({res['train_seconds']:.0f}s) b77={res['banking77']['intent_77_acc']:.3f} "
                    f"r5={res['mean_final_r5']:.3f} k={rd['mean_k']:.2f} E={rd['num_experts']} "
                    f"lm_ami={rd['length_matched_role_ami']['ami']:.3f} "
                    f"len_ami={rd['routing_length_ami']:.3f} "
                    f"pert={rd['routing_perturb_agree']:.3f} "
                    f"les_d={rd['lesion_mean_rel_delta']:.3f}{dec_s}"
                )
            if "track1" in res:
                t1 = res["track1"]
                ep8 = t1["episodic_v10"]["shots_8"]
                ep16 = t1["episodic_v10"]["shots_16"]
                parts.append(
                    f"T1({t1['train_seconds']:.0f}s) seen={t1['seen_acc']:.3f} held={t1['heldout_unseen_acc']:.3f} "
                    f"iit_h={t1['iit_heldout_acc']:.3f} burst={t1['burst_immunity_acc']:.3f} "
                    f"lu8={ep8['lookup_acc']:.3f} comp8={ep8['comp_equiv_induction_acc']:.3f} "
                    f"shift8={ep8['unseen_meaning_shift_acc']:.3f} shift16={ep16['unseen_meaning_shift_acc']:.3f} "
                    f"shuff8={ep8['shift_shuffled_null_acc']:.3f}"
                )
            print(" | ".join(parts), flush=True)

    summary = {
        "meta": meta,
        "arms": {a: _agg_v11_runs(results_by_arm[a]) for a in arms},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

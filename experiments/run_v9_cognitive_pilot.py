"""
real_pilot_v6: Architecture v9 Rigorous Diagnostics and Generalization Pilot.

Implements Stage 0 diagnostics and Stage 1 interventions from:
  docs/superpowers/specs/2026-11-21-v9-diagnostics-and-generalization-plan.md

Diagnostics:
  - D1: Matched controls (Dense_v8_Matched & Cog_v8_FrozenRandomRouter use MuonBONGOptimizer, GPM, QuarantineGate) + AdamW sweep
  - D2: Unbundle NoI5_NoRel into explicit independent flags (use_iit, i7_steps, use_rel_pool, use_v8_core)
  - D3: 2x2 routing x escalation factorial ({dynamic_k, sequence_top1} x {escalation_on, escalation_off})
  - D4: Honest episodic probes: (a) shuffled-binding control, (b) exact-lookup oracle, (c) novel-pair rule induction probe
  - D5: Interchange on held-out concept values and random dimension slice null control
  - D6: 20-seed Track 1 sweep with grokking curve logging
  - D7: GPM exemption ablation (exempt_readout_heads=False) logging per-head basis saturation
  - D8: Adjusted Mutual Information (AMI) + permutation null + per-expert lesion analysis
  - D9: Near-OOD abstain set and metric hygiene
  - D10: Steps scaling for R@5
  - D11: Profiler diagnostics

Stage 1 Interventions:
  - H1a: Constant Fisher scaling in Track 1 (decoupled from acc_batch)
  - H2: Compute-aware routing penalty targeting mean k <= 1.0
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import (
    BeliefMeaningTask,
    BudgetedSurfaceDict,
    K_CONCEPTS,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
)
from dmoe.null_space import NullSpaceProjector, _READOUT_HEAD_NAMES
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import LatentQuarantineGate, MuonBONGOptimizer, classify_v7_spgc_param_groups
from experiments.real_pilot_data import DOMAINS, RealCorpora, SEQ_LEN
from experiments.run_v7_cognitive_pilot import (
    OfflineTeacherGramEncoder,
    bag_of_bytes_features,
    encode_belief_chunks,
)
from experiments.run_v8_cognitive_pilot import (
    DEFAULT_V8_CFG,
    V8RealBenchmark,
    encode_texts_v8,
    roc_auc_pairs,
    compute_nmi,
    eval_banking77_v8,
    eval_triples_nuance_v8,
    eval_domain_v8,
    eval_routing_diagnostics_v8,
    ridge_probe,
    _base_arm_name,
)
from experiments.v8_edit_families import (
    HELDOUT_FAMILIES,
    TRAIN_FAMILIES,
    generate_i7_procedural_batch,
    generate_v7_triple,
    generate_v8_train_triple,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_ROOT = ROOT / "results" / "real_pilot_v6"

DEFAULT_V9_CFG: Dict[str, Any] = dict(
    DEFAULT_V8_CFG,
    use_iit=True,
    use_rel_pool=True,
    i7_steps=150,
    enable_nuance_escalation=True,
    exempt_readout_heads=True,
    use_constant_fisher=False,
    target_mean_k=None,
    compute_penalty_weight=0.0,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ami_with_null(labels_true: List[int], labels_pred: List[int], n_perms: int = 50) -> Dict[str, float]:
    """Compute Adjusted Mutual Information and empirical permutation null."""
    if not labels_true or len(labels_true) != len(labels_pred):
        return {"ami": 0.0, "null_mean": 0.0, "null_max": 0.0, "ami_vs_null": 0.0}
    ami = float(adjusted_mutual_info_score(labels_true, labels_pred))
    
    rng = np.random.default_rng(0)
    arr_pred = np.array(labels_pred)
    null_scores = []
    for _ in range(n_perms):
        shuffled = rng.permutation(arr_pred)
        null_scores.append(float(adjusted_mutual_info_score(labels_true, shuffled)))
    null_mean = float(np.mean(null_scores))
    null_max = float(np.max(null_scores))
    return {
        "ami": ami,
        "null_mean": null_mean,
        "null_max": null_max,
        "ami_vs_null": ami - null_mean,
    }


def build_v9_arm(
    arm: str,
    cfg: Dict[str, Any],
    seed: int,
    steps_per_dom: int = 250,
    dense_adamw_lr: float = 3e-3,
) -> Tuple[Optional[RealSelfStabilizingDMoE], Optional[NullSpaceProjector], Optional[torch.optim.Optimizer], Optional[LatentQuarantineGate]]:
    base_arm = _base_arm_name(arm)
    if base_arm == "BagOfBytes_Null":
        return None, None, None, None
    if base_arm == "Upkeeper_M09":
        return None, None, None, None

    c = dict(cfg)
    c["initial_experts"] = 1
    c["min_mature_steps"] = min(int(cfg["min_mature_steps"]), max(20, int(steps_per_dom * 0.85)))

    # D2: Unbundle NoI5_NoRel into explicit independent flags
    if base_arm == "Cog_v9":
        c["use_rel_pool"] = False
        c["enable_nuance_escalation"] = False
        c["use_constant_fisher"] = True
        c["use_iit"] = True
    elif base_arm == "Cog_v8_NoV8Core":
        c["use_v8_core"] = False
        c["use_v7_core"] = True
        c["use_rel_pool"] = False
        c["halt_use_energy_only"] = False
    elif base_arm == "Cog_v8_NoRel":
        c["use_rel_pool"] = False
    elif base_arm == "Cog_v8_NoEsc":
        c["enable_nuance_escalation"] = False
    elif base_arm == "Cog_v8_SequenceTop1_NoEsc":
        c["enable_nuance_escalation"] = False

    # D1 Matched control setup for Dense
    if base_arm in ("Dense_v8_Matched", "Dense_v8_AdamW"):
        c["auto_lifecycle"] = False
        c["max_experts"] = 1
        c["d_ff_shared"] = 512
        c["d_ff_spec"] = 256

    # Stage 1: H2 compute-aware routing target
    if base_arm == "Cog_v9_H2":
        c["target_mean_k"] = 1.0
        c["compute_penalty_weight"] = 0.5

    model_cfg = {
        k: v
        for k, v in c.items()
        if k
        not in (
            "use_iit",
            "i7_steps",
            "exempt_readout_heads",
            "use_constant_fisher",
            "target_mean_k",
            "compute_penalty_weight",
        )
    }
    model = RealSelfStabilizingDMoE(**model_cfg).to(DEV)

    if base_arm == "Cog_v8_FrozenRandomRouter" and model.context_router is not None:
        for p in model.context_router.parameters():
            p.requires_grad_(False)

    if base_arm in ("Cog_v8_SequenceTop1", "Cog_v8_SequenceTop1_NoEsc") and model.context_router is not None:
        model.context_router.sequence_level = True
        model.context_router.tau_skip = -1.0
        model.context_router.tau_top2 = 2.0
        model.context_router.delta_mix = 0.0

    # D1 Matched controls: Dense_v8_Matched & Cog_v8_FrozenRandomRouter use MuonBONGOptimizer, GPM, QuarantineGate!
    matched_arms = (
        "Cog_v8",
        "Cog_v9",
        "Cog_v8_Mixed",
        "Cog_v8_SequenceTop1",
        "Cog_v8_SequenceTop1_NoEsc",
        "Cog_v8_NoEsc",
        "Cog_v8_NoI5",
        "Cog_v8_NoI7",
        "Cog_v8_NoRel",
        "Cog_v8_NoV8Core",
        "Cog_v8_FrozenRandomRouter",
        "Dense_v8_Matched",
        "Cog_v8_GpmNonExempt",
        "Cog_v9_H1a",
        "Cog_v9_H2",
    )

    if base_arm in matched_arms:
        exempt_heads = (base_arm != "Cog_v8_GpmNonExempt") and bool(c.get("exempt_readout_heads", True))
        gpm = NullSpaceProjector(
            model,
            energy_threshold=0.97,
            max_rank_ratio=0.85,
            max_step_rank_ratio=0.24,
            step_rank_decay=0.84,
            defer_cov_commit=True,
            track_embeddings=True,
            freeze_shared_norms_on_update=True,
            freeze_unprojected_shared=True,
            exempt_readout_heads=exempt_heads,
        )
        model.attach_null_space_projector(gpm)
        groups = classify_v7_spgc_param_groups(
            model,
            shared_mode="acquire",
            specialist_mode="acquire",
            lr_muon=8e-3,
            lr_adam=3e-3,
            lr_consolidate=2e-3,
            trunk_prec_scale=0.85,
        )
        opt = MuonBONGOptimizer(
            groups,
            mode="acquire",
            bong_prior_pull=0.02,
            bong_max_prec=100.0,
            grad_clip=4.5,
        )
        gate = (
            LatentQuarantineGate(latent_dim=c["d_model"], seed=seed, max_bytes=16384, store_samples=False)
            if "Mixed" not in base_arm
            else None
        )
        return model, gpm, opt, gate
    elif base_arm == "Dense_v8_AdamW":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=dense_adamw_lr, weight_decay=0.01)
        return model, None, opt, None
    elif base_arm == "Cog_v6_Ref":
        from experiments.run_v8_cognitive_pilot import build_v8_arm as build_v8_raw
        return build_v8_raw("Cog_v6_Ref", cfg, seed, steps_per_dom=steps_per_dom)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.5e-3, weight_decay=0.01)
        return model, None, opt, None


# --------------------------------------------------------------------------------------------------
# D4 Honest Episodic Probes
# --------------------------------------------------------------------------------------------------
def _sample_episodic_honest_batch(
    task: BeliefMeaningTask,
    n_episodes: int,
    gen: torch.Generator,
    n_shots: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample honest episodic evaluation batches separating lookup from novel-pair induction:
      - Returns: (x_sup_all, x_q_lookup, y_q_lookup, x_q_novel, y_q_novel)
      - x_q_lookup: query pair (qa, qb) is drawn from support shots (associative lookup oracle)
      - x_q_novel: query pair (qa, qb) is strictly NOT in support shots (rule induction probe)
    """
    phi_ep = torch.randperm(K_CONCEPTS, generator=gen)
    inv_ep = torch.argsort(phi_ep)
    g_ep = torch.randperm(K_CONCEPTS, generator=gen)

    x_sup_list: List[torch.Tensor] = []
    x_q_lookup_list: List[torch.Tensor] = []
    y_q_lookup_list: List[int] = []
    x_q_novel_list: List[torch.Tensor] = []
    y_q_novel_list: List[int] = []

    for _ in range(n_episodes):
        a_s = torch.randint(0, K_CONCEPTS, (n_shots,), generator=gen)
        b_s = torch.randint(0, K_CONCEPTS, (n_shots,), generator=gen)
        m_s = inv_ep[(phi_ep[a_s] + 3 * phi_ep[b_s]) % K_CONCEPTS]
        y_s = g_ep[m_s]
        c_z = torch.zeros(n_shots, dtype=torch.long)
        q_a = torch.full((n_shots,), Q_ATTR, dtype=torch.long)
        xs_s = torch.stack([torch.zeros(n_shots, dtype=torch.long), a_s + 4, b_s + 4, c_z + 4, c_z + 4, q_a], dim=-1)
        xb_s = encode_belief_chunks(xs_s)
        sup_seq = []
        seen_pairs = set()
        for si in range(n_shots):
            ans_b = torch.tensor(list(f"Y{int(y_s[si].item()):02d} ".encode("ascii")), dtype=torch.long)
            sup_seq.append(torch.cat([xb_s[si], ans_b], dim=0))
            seen_pairs.add((int(a_s[si].item()), int(b_s[si].item())))
        x_sup_list.append(torch.stack(sup_seq, dim=0))

        # 1. Lookup oracle query: chosen from support
        pick = int(torch.randint(0, n_shots, (1,), generator=gen).item())
        qa_l, qb_l, qy_l = int(a_s[pick].item()), int(b_s[pick].item()), int(y_s[pick].item())
        xs_ql = torch.tensor([[0, qa_l + 4, qb_l + 4, 4, 4, Q_ATTR]], dtype=torch.long)
        x_q_lookup_list.append(encode_belief_chunks(xs_ql)[0])
        y_q_lookup_list.append(qy_l)

        # 2. Honest novel induction query: strictly not in seen_pairs
        qa_n = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
        qb_n = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
        attempts = 0
        while (qa_n, qb_n) in seen_pairs and attempts < 100:
            qa_n = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
            qb_n = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
            attempts += 1
        qm_n = int(inv_ep[(phi_ep[qa_n] + 3 * phi_ep[qb_n]) % K_CONCEPTS].item())
        qy_n = int(g_ep[qm_n].item())
        xs_qn = torch.tensor([[0, qa_n + 4, qb_n + 4, 4, 4, Q_ATTR]], dtype=torch.long)
        x_q_novel_list.append(encode_belief_chunks(xs_qn)[0])
        y_q_novel_list.append(qy_n)

    return (
        torch.stack(x_sup_list, dim=0),
        torch.stack(x_q_lookup_list, dim=0),
        torch.tensor(y_q_lookup_list, dtype=torch.long),
        torch.stack(x_q_novel_list, dim=0),
        torch.tensor(y_q_novel_list, dtype=torch.long),
    )


def eval_honest_episodic_probes(
    model: RealSelfStabilizingDMoE,
    cls_head: nn.Module,
    task: BeliefMeaningTask,
    gen: torch.Generator,
    shot_counts: Tuple[int, ...] = (4, 8, 16),
    n_episodes: int = 64,
) -> Dict[str, Any]:
    """Evaluate honest episodic probes across shot counts."""
    res = {}
    for n_sh in shot_counts:
        x_sup, x_ql, y_ql, x_qn, y_qn = _sample_episodic_honest_batch(task, n_episodes, gen, n_shots=n_sh)
        e_sz, _, l_sh = x_sup.shape
        ep_kv_flat = model.extract_episode_kv(x_sup.view(e_sz * n_sh, l_sh).to(DEV), top_k=6, min_nll=0.0)
        if ep_kv_flat is None:
            res[f"shots_{n_sh}"] = {
                "lookup_acc": 1.0 / K_CONCEPTS,
                "shuffled_null_acc": 1.0 / K_CONCEPTS,
                "novel_induction_acc": 1.0 / K_CONCEPTS,
            }
            continue

        k_sh = ep_kv_flat["ep_keys"].size(1)
        d_kv = ep_kv_flat["ep_keys"].size(-1)
        ep_kv = {
            "ep_keys": ep_kv_flat["ep_keys"].view(e_sz, n_sh * k_sh, d_kv),
            "ep_vals": ep_kv_flat["ep_vals"].view(e_sz, n_sh * k_sh, d_kv),
            "ep_mask": ep_kv_flat["ep_mask"].view(e_sz, n_sh * k_sh),
        }

        # 1. Exact lookup
        o_lookup = model(x_ql.to(DEV), update_stats=False, episode_memory=ep_kv)
        lookup_acc = float((cls_head(o_lookup["latent"][:, -1, :]).argmax(-1).cpu() == y_ql).float().mean().item())

        # 2. Shuffled null control: roll memory indices by 1
        ep_kv_shuffled = {
            "ep_keys": torch.roll(ep_kv["ep_keys"], shifts=1, dims=0),
            "ep_vals": torch.roll(ep_kv["ep_vals"], shifts=1, dims=0),
            "ep_mask": torch.roll(ep_kv["ep_mask"], shifts=1, dims=0),
        }
        o_shuff = model(x_ql.to(DEV), update_stats=False, episode_memory=ep_kv_shuffled)
        shuff_acc = float((cls_head(o_shuff["latent"][:, -1, :]).argmax(-1).cpu() == y_ql).float().mean().item())

        # 3. Novel induction query
        o_novel = model(x_qn.to(DEV), update_stats=False, episode_memory=ep_kv)
        novel_acc = float((cls_head(o_novel["latent"][:, -1, :]).argmax(-1).cpu() == y_qn).float().mean().item())

        res[f"shots_{n_sh}"] = {
            "lookup_acc": lookup_acc,
            "shuffled_null_acc": shuff_acc,
            "novel_induction_acc": novel_acc,
        }
    return res


# --------------------------------------------------------------------------------------------------
# Track 1 Runner (v9 with all diagnostics and honest probes)
# --------------------------------------------------------------------------------------------------
def run_track1_arm_v9(
    arm: str,
    cfg: Dict[str, Any],
    seed: int,
    steps_scale: float = 1.0,
    dense_adamw_lr: float = 3e-3,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    base_arm = _base_arm_name(arm)
    task = BeliefMeaningTask(seed=seed)
    gen = torch.Generator().manual_seed(seed * 97 + 1)

    if base_arm == "Upkeeper_M09":
        upk = BudgetedSurfaceDict(max_total_bytes=16384, min_sources_for_flip=2, min_flip_count=3, flip_threshold_ratio=0.45)
        for ph, n_st, src in [("p1", int(800 * steps_scale), "s0"), ("pb", int(20 * steps_scale), "burst"), ("p2", int(200 * steps_scale), "s1"), ("p3", int(200 * steps_scale), "s2")]:
            for st_i in range(n_st):
                xs, y = task.sample_stream_batch(ph, 64, gen)
                xb = encode_belief_chunks(xs, paraphrase=(ph == "p1" and (st_i % 2 == 1)))
                upk.update(xb, y, source_id=f"{src}_{st_i % 2 if ph in ('p1','p2','p3') else 0}")

        def eval_upk(pair_split: str, para: bool = False) -> float:
            hits, tot = 0, 0
            for q_tok in (Q_VAL, Q_ATTR, Q_PROP):
                xs, ys = task.sample_eval_subset(q_tok, "p3", [0, 1, 2], pair_split, None, None, 128, gen)
                preds = upk.predict_log_probs(encode_belief_chunks(xs, paraphrase=para)).argmax(dim=-1)
                hits += int((preds == ys).sum().item())
                tot += ys.numel()
            return hits / max(1, tot)

        return {
            "seen_acc": eval_upk("seen"),
            "heldout_unseen_acc": eval_upk("heldout"),
            "paraphrase_acc": eval_upk("heldout", para=True),
            "iit_causal_acc": 1.0 / K_CONCEPTS,
            "iit_heldout_acc": 1.0 / K_CONCEPTS,
            "iit_random_slice_null": 1.0 / K_CONCEPTS,
            "episodic_honest": {"shots_4": {"lookup_acc": 1.0 / 16.0, "shuffled_null_acc": 1.0 / 16.0, "novel_induction_acc": 1.0 / 16.0}},
            "burst_immunity_acc": eval_upk("seen"),
            "abstain_auroc": 0.5,
            "near_ood_abstain_auroc": 0.5,
            "grokking_curve": [],
        }

    n_p1 = max(220, int(1300 * steps_scale))
    n_pb = max(8, int(20 * steps_scale))
    n_p2 = max(40, int(180 * steps_scale))
    n_p3 = max(40, int(180 * steps_scale))
    t1_cfg = dict(cfg)
    t1_cfg["min_mature_steps"] = n_p1 + 50
    model, gpm, opt, gate = build_v9_arm(base_arm, t1_cfg, seed, steps_per_dom=n_p1 + n_p2 + n_p3 + 50, dense_adamw_lr=dense_adamw_lr)
    cls_head = nn.Linear(t1_cfg["d_model"], K_CONCEPTS).to(DEV)

    # D2: unbundled use_iit flag
    use_iit = bool(cfg.get("use_iit", True) and base_arm not in ("Cog_v8_NoI5", "Cog_v8_NoV8Core"))

    if isinstance(opt, MuonBONGOptimizer):
        groups = classify_v7_spgc_param_groups(
            model,
            shared_mode="acquire",
            specialist_mode="acquire",
            lr_muon=8e-3,
            lr_adam=3e-3,
            lr_consolidate=2e-3,
            wd_trunk=0.10,
            wd_fast=0.02,
            trunk_prec_scale=0.85,
        )
        groups.append({
            "name": "track1_cls",
            "params": list(cls_head.parameters()),
            "use_muon": False,
            "role": "fast",
            "mode": "acquire",
            "lr": 5e-3,
            "lr_consolidate": 3e-3,
            "weight_decay": 0.02,
            "prec_scale": 0.25,
        })
        opt = MuonBONGOptimizer(groups, mode="acquire", bong_prior_pull=0.02, bong_max_prec=100.0)
    else:
        opt = torch.optim.AdamW(
            list(p for p in model.parameters() if p.requires_grad) + list(cls_head.parameters()),
            lr=dense_adamw_lr,
            weight_decay=0.05,
        )

    D_h = t1_cfg["d_model"]
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

        if ph == "p1" and gate is not None and step_i >= (n_p1 // 2) and (step_i % 2 == 0):
            with torch.no_grad():
                model.eval()
                o_reg = model(xb_raw, update_stats=False)
                lg_reg = cls_head(o_reg["latent"][:, -1, :])
                gate.triage_batch(
                    xb_raw.detach(),
                    y_dev.detach(),
                    lg_reg.detach(),
                    o_reg["z_expr_pool"].detach(),
                    source_id=source_id,
                    skip_familiar=False,
                    enable_quarantine=False,
                )

        if ph != "p1" and gate is not None:
            with torch.no_grad():
                model.eval()
                o_scr = model(xb, update_stats=False)
                lg_scr = cls_head(o_scr["latent"][:, -1, :])
                xb_act, y_act, flips = gate.triage_batch(
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
            if flips > 0 and isinstance(opt, MuonBONGOptimizer):
                opt.temper_precision(roles=("fast",), factor=0.25)

        model.train()
        saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
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
            if out.get("abstain_logits") is not None and ph == "p1":
                ab_in = out["abstain_logits"][:b_sz, -1, 0].float()
                l_ab_cal = F.binary_cross_entropy_with_logits(ab_in, torch.zeros_like(ab_in))
                if step_i % 4 == 0:
                    xu_ood = torch.randint(33, 122, (16, 24), device=DEV)
                    ou_ood = model(xu_ood, update_stats=False)
                    if ou_ood.get("abstain_logits") is not None:
                        ab_ood = ou_ood["abstain_logits"][:, -1, 0].float()
                        l_ab_cal = l_ab_cal + F.binary_cross_entropy_with_logits(ab_ood, torch.ones_like(ab_ood))

            loss = l_cls + 0.06 * l_para + 0.03 * wm_l + 0.08 * l_ab_cal

            # I5 Causal Abstraction Interchange-Intervention Training
            if use_iit and ph == "p1" and step_i >= (n_p1 // 3) and (step_i % 2 == 0) and b_sz >= 64:
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

            # I6 Episodic Memory Training
            if use_iit and ph == "p1" and (step_i % 3 == 1) and step_i >= (n_p1 // 5):
                x_sup_ep, x_ql_ep, y_ql_ep, _, _ = _sample_episodic_honest_batch(task, 16, gen, n_shots=4)
                e_sz, n_sh, l_sh = x_sup_ep.shape
                ep_kv_flat = model.extract_episode_kv(x_sup_ep.view(e_sz * n_sh, l_sh).to(DEV), top_k=6, min_nll=0.0)
                if ep_kv_flat is not None:
                    k_sh = ep_kv_flat["ep_keys"].size(1)
                    d_kv = ep_kv_flat["ep_keys"].size(-1)
                    ep_kv = {
                        "ep_keys": ep_kv_flat["ep_keys"].view(e_sz, n_sh * k_sh, d_kv),
                        "ep_vals": ep_kv_flat["ep_vals"].view(e_sz, n_sh * k_sh, d_kv),
                        "ep_mask": ep_kv_flat["ep_mask"].view(e_sz, n_sh * k_sh),
                    }
                    o_ep = model(x_ql_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                    loss = loss + 0.40 * F.cross_entropy(cls_head(o_ep["latent"][:, -1, :]), y_ql_ep.to(DEV))

            # Stage 1: H1a Constant Fisher scaling vs baseline adaptive
            if cfg.get("use_constant_fisher", False) or base_arm in ("Cog_v9", "Cog_v9_H1a"):
                fisher_sc = 1.0
            else:
                acc_batch = float((lg[:b_sz].detach().argmax(-1) == y_train[:b_sz]).float().mean().item())
                fisher_sc = 1.0 if ph == "pb" else (0.25 + 1.15 * acc_batch)

        opt.zero_grad()
        loss.backward()
        if gpm is not None and gpm.p_null:
            gpm.project_gradients()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(cls_head.parameters()), 1.5)
        if isinstance(opt, MuonBONGOptimizer):
            opt.step(fisher_scale=fisher_sc)
        else:
            opt.step()
        if saved is not None:
            gpm.project_parameter_deltas(saved)
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.commit_pending()

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

    # Train p1 with D6 grokking curve logging
    for st in range(n_p1):
        xs, y = task.sample_stream_batch("p1", 64, gen)
        step_t1(xs, y, f"s_{st % 2}", "p1", st)
        if (st + 1) % 100 == 0 or st == n_p1 - 1:
            held_cur = eval_t1("p1", "heldout")
            grokking_curve.append(held_cur)

    if gpm is not None:
        gpm.update_basis_for_domain(0)
        if isinstance(opt, MuonBONGOptimizer):
            opt.set_group_mode("trunk", "consolidate")

    # Train pb (burst)
    for st in range(n_pb):
        xs, y = task.sample_stream_batch("pb", 64, gen)
        step_t1(xs, y, "adversarial_burst", "pb", st)

    with torch.no_grad():
        model.eval()
        xs_s, ys_s = task.sample_eval_subset(Q_ATTR, "p1", [0, 1, 2], "seen", None, None, 128, gen)
        o_s = model(encode_belief_chunks(xs_s).to(DEV), update_stats=False)
        burst_acc = float((cls_head(o_s["latent"][:, -1, :]).argmax(-1).cpu() == ys_s).float().mean())

    # Train p2 and p3
    for st in range(n_p2):
        xs, y = task.sample_stream_batch("p2", 64, gen)
        step_t1(xs, y, f"s2_{st % 2}", "p2", st)
    for st in range(n_p3):
        xs, y = task.sample_stream_batch("p3", 64, gen)
        step_t1(xs, y, f"s3_{st % 2}", "p3", st)

    with torch.no_grad():
        model.eval()
        # D9: Standard random-byte abstain AUROC
        xs_k, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xk_b = encode_belief_chunks(xs_k).to(DEV)
        xu_b = torch.randint(33, 122, (64, 24), device=DEV)
        ok = model(xk_b, update_stats=False)
        ou = model(xu_b, update_stats=False)
        if ok.get("abstain_logits") is not None and ou.get("abstain_logits") is not None:
            sk = (0.5 * ok["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ok["nuance_score"].mean(dim=-1) / 3.0)).tolist()
            su = (0.5 * ou["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ou["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        else:
            sk = ok["nuance_score"].mean(dim=-1).tolist()
            su = ou["nuance_score"].mean(dim=-1).tolist()
        abstain_auroc = roc_auc_pairs(su, sk)

        # D9: Honest Near-OOD Abstain AUROC (valid syntax but unseen/illegal operation token 'Q9')
        near_ood_lines = []
        for i in range(64):
            near_ood_lines.append(f"Q9 C{int(xs_k[i, 1] - 4):02d} C{int(xs_k[i, 2] - 4):02d} C00 C00 ? ".encode("ascii"))
        x_near_ood = encode_texts_v8(near_ood_lines, 24, DEV, pad_id=257)
        o_near_ood = model(x_near_ood, update_stats=False)
        if ok.get("abstain_logits") is not None and o_near_ood.get("abstain_logits") is not None:
            sn = (0.5 * o_near_ood["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (o_near_ood["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        else:
            sn = o_near_ood["nuance_score"].mean(dim=-1).tolist()
        near_ood_auroc = roc_auc_pairs(sn, sk)

        # D5: Interchange on seen vs held-out concept pairs and random dimension slice null control
        xs_iit_b, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "seen", None, None, 64, gen)
        xs_iit_d, _ = task.sample_eval_subset(Q_ATTR, "p3", [0, 1, 2], "seen", None, None, 64, gen)
        xk_iit_b = encode_belief_chunks(xs_iit_b).to(DEV)
        xd_iit_b = encode_belief_chunks(xs_iit_d).to(DEV)

        xs_iit_hb, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xs_iit_hd, _ = task.sample_eval_subset(Q_ATTR, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xk_iit_hb = encode_belief_chunks(xs_iit_hb).to(DEV)
        xd_iit_hd = encode_belief_chunks(xs_iit_hd).to(DEV)

        if hasattr(model, "forward_interchange") and getattr(model, "use_v7_core", False):
            # 1. Seen pairs, causal slice [0:D//2]
            o_seen_causal = model.forward_interchange(xk_iit_b, xd_iit_b, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2))
            y_cf_seen, _, _ = task._compute_targets(
                xs_iit_d[:, 0], xs_iit_d[:, 1] - 4, xs_iit_d[:, 2] - 4, xs_iit_b[:, 3] - 4, xs_iit_b[:, 4] - 4, xs_iit_b[:, 5], "p3"
            )
            iit_seen_acc = float((cls_head(o_seen_causal["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_seen).float().mean().item())

            # 2. Heldout pairs, causal slice [0:D//2]
            o_held_causal = model.forward_interchange(xk_iit_hb, xd_iit_hd, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2))
            y_cf_held, _, _ = task._compute_targets(
                xs_iit_hd[:, 0], xs_iit_hd[:, 1] - 4, xs_iit_hd[:, 2] - 4, xs_iit_hb[:, 3] - 4, xs_iit_hb[:, 4] - 4, xs_iit_hb[:, 5], "p3"
            )
            iit_held_acc = float((cls_head(o_held_causal["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_held).float().mean().item())

            # 3. Seen pairs, random slice null control [D//4 : 3D//4]
            o_seen_rand = model.forward_interchange(xk_iit_b, xd_iit_b, intervene_chunks=(0, 3), dim_slice=(D_h // 4, (3 * D_h) // 4))
            iit_rand_null_acc = float((cls_head(o_seen_rand["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_seen).float().mean().item())
        else:
            iit_seen_acc = 1.0 / K_CONCEPTS
            iit_held_acc = 1.0 / K_CONCEPTS
            iit_rand_null_acc = 1.0 / K_CONCEPTS

        # D4: Honest Episodic Probes
        ep_probes = eval_honest_episodic_probes(model, cls_head, task, gen, shot_counts=(4, 8, 16), n_episodes=64)

        # D7: Basis Saturation Log
        basis_saturation = {}
        if gpm is not None:
            for head_name in _READOUT_HEAD_NAMES:
                cov_k = gpm.cov_alias.get(head_name, head_name)
                if cov_k in gpm.bases and gpm.bases[cov_k] is not None:
                    max_r = max(1, int(gpm.bases[cov_k].size(0) * gpm.max_rank_ratio))
                    basis_saturation[head_name] = float(gpm.bases[cov_k].size(1) / max_r)
                else:
                    basis_saturation[head_name] = 0.0

    fin_held = eval_t1("p3", "heldout")
    return {
        "seen_acc": eval_t1("p3", "seen"),
        "heldout_unseen_acc": fin_held,
        "paraphrase_acc": eval_t1("p3", "heldout", para=True),
        "iit_causal_acc": iit_seen_acc,
        "iit_heldout_acc": iit_held_acc,
        "iit_random_slice_null": iit_rand_null_acc,
        "episodic_honest": ep_probes,
        "burst_immunity_acc": burst_acc,
        "abstain_auroc": abstain_auroc,
        "near_ood_abstain_auroc": near_ood_auroc,
        "basis_saturation": basis_saturation,
        "grokking_curve": grokking_curve,
    }


# --------------------------------------------------------------------------------------------------
# Track 2 Runner (v9 with D8 AMI, Lesion Analysis, and H2 Compute Control)
# --------------------------------------------------------------------------------------------------
def run_track2_arm_v9(
    arm: str,
    cfg: Dict[str, Any],
    bench: V8RealBenchmark,
    seed: int,
    steps_per_dom: int = 250,
    teacher: Optional[OfflineTeacherGramEncoder] = None,
    dense_adamw_lr: float = 3e-3,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    base_arm = _base_arm_name(arm)
    model, gpm, opt, gate = build_v9_arm(base_arm, cfg, seed, steps_per_dom=steps_per_dom, dense_adamw_lr=dense_adamw_lr)
    is_teacher = teacher is not None and arm.endswith("_I9")

    if model is None:
        if base_arm == "BagOfBytes_Null":
            return bag_of_bytes_features(bench)
        else:
            raise ValueError(f"Unknown null arm: {arm}")

    # Stage 0 / Stage 1: Optional I7 procedural pre-training
    i7_steps = int(cfg.get("i7_steps", 150))
    if base_arm in ("Cog_v8_NoI7", "Cog_v8_NoV8Core"):
        i7_steps = 0

    t_start = time.time()
    gen_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    if i7_steps > 0 and getattr(model, "use_v8_core", False):
        model.train()
        for i7_st in range(i7_steps):
            x_i7, eq_i7, c_i7, nt_i7, st_i7 = generate_i7_procedural_batch(16, rng_py, seq_len=SEQ_LEN, device=DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                o_i7 = model(x_i7, update_stats=False)
                l_i7 = model.compute_v7_non_token_loss(
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
                )
            opt.zero_grad()
            l_i7["loss"].backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
            if isinstance(opt, MuonBONGOptimizer):
                opt.step(fisher_scale=0.5)
            else:
                opt.step()

    # Continual multi-domain training loop
    for dom_idx, dom in enumerate(DOMAINS):
        model.train()
        if gpm is not None and dom_idx > 0:
            if isinstance(opt, MuonBONGOptimizer):
                opt.set_group_mode("shared_trunk", "consolidate")
        for step in range(steps_per_dom):
            xb_raw, y_eq, contra_pair, hard_pair, nuance_t, struct_t, g_mask = bench.sample_v8_stream_batch(dom, gen_np, rng_py)
            teach_gram = teacher.encode(xb_raw) if is_teacher else None

            model.train()
            saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(xb_raw, update_stats=True)
                loss_dict = model.compute_v7_non_token_loss(
                    out,
                    x=xb_raw,
                    equiv_labels=y_eq,
                    contra_pair_idx=contra_pair,
                    hard_neg_pair_idx=hard_pair,
                    nuance_targets=nuance_t,
                    teacher_gram=teach_gram,
                    gram_mask=g_mask,
                    struct_targets=struct_t,
                )
                loss = loss_dict["loss"]

                # Stage 1: H2 compute penalty targeting mean k <= 1.0
                if cfg.get("compute_penalty_weight", 0.0) > 0 and out.get("k_used") is not None:
                    k_mean = out["k_used"].float().mean()
                    target_k = float(cfg.get("target_mean_k", 1.0))
                    pen = F.relu(k_mean - target_k).pow(2)
                    loss = loss + float(cfg["compute_penalty_weight"]) * pen

            opt.zero_grad()
            loss.backward()
            if gpm is not None and gpm.p_null:
                gpm.project_gradients()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.5)
            if isinstance(opt, MuonBONGOptimizer):
                opt.step(fisher_scale=1.0)
            else:
                opt.step()
            if saved is not None:
                gpm.project_parameter_deltas(saved)
            if gpm is not None and getattr(gpm, "defer_cov_commit", False):
                gpm.commit_pending()

        if gpm is not None:
            gpm.update_basis_for_domain(dom_idx)

    train_sec = time.time() - t_start

    # Evaluations
    model.eval()
    b77_res = eval_banking77_v8(model, bench)
    per_dom = {d: eval_domain_v8(model, bench, d) for d in DOMAINS}
    mean_nf_auc = float(np.mean([per_dom[d]["nuance_flip_auc"] for d in DOMAINS]))
    mean_r5 = float(np.mean([per_dom[d]["concept_r5"] for d in DOMAINS]))
    hf_res = eval_triples_nuance_v8(model, bench.heldout_family_pairs)
    rd = eval_routing_diagnostics_v8(model, bench)

    # Routing Diagnostics and D8 AMI + Lesion Analysis
    is_v78 = getattr(model, "use_v7_core", False)
    sw_rate = rd.get("mid_seq_switch_rate", 0.0)
    k_hist = rd.get("k_dist", {"k0": 0.0, "k1": 1.0, "k2": 0.0})
    mean_k_val = float(k_hist.get("k0", 0.0) * 0.0 + k_hist.get("k1", 1.0) * 1.0 + k_hist.get("k2", 0.0) * 2.0)

    with torch.no_grad():
        # D8: AMI with Permutation Null across 6 held-out functional roles
        rf_preds: List[int] = []
        for s in range(0, len(bench.func_role_texts), 64):
            xb_r = encode_texts_v8(bench.func_role_texts[s : s + 64], SEQ_LEN, DEV, pad_id=257 if is_v78 else 32)
            o_r = model(xb_r, update_stats=False)
            rf_preds.extend(o_r["assigned_experts"].detach().cpu().tolist())

        ami_diag = compute_ami_with_null(bench.func_role_labels, rf_preds, n_perms=50)

        # D8: Per-Expert Lesion Analysis (measure role classification drop when expert e is masked)
        lesion_drops: Dict[str, float] = {}
        if hasattr(model, "layers") and model.router_layer < len(model.layers):
            moe_r = model.layers[model.router_layer].moe
            n_exp = len(moe_r.experts)
            if n_exp > 1:
                base_feats = []
                for s in range(0, len(bench.func_role_texts), 64):
                    xb_r = encode_texts_v8(bench.func_role_texts[s : s + 64], SEQ_LEN, DEV, pad_id=257 if is_v78 else 32)
                    base_feats.append(model(xb_r, update_stats=False)["latent"][:, -1, :].cpu())
                base_f = torch.cat(base_feats, dim=0)
                y_roles = torch.tensor(bench.func_role_labels, dtype=torch.long)
                base_acc = ridge_probe(base_f, y_roles, base_f, y_roles, 6)

                for e_idx in range(n_exp):
                    orig_state = moe_r.experts[e_idx].training
                    moe_r.experts[e_idx].eval()
                    saved_w = [p.data.clone() for p in moe_r.experts[e_idx].parameters()]
                    for p in moe_r.experts[e_idx].parameters():
                        p.data.zero_()

                    les_feats = []
                    for s in range(0, len(bench.func_role_texts), 64):
                        xb_r = encode_texts_v8(bench.func_role_texts[s : s + 64], SEQ_LEN, DEV, pad_id=257 if is_v78 else 32)
                        les_feats.append(model(xb_r, update_stats=False)["latent"][:, -1, :].cpu())
                    les_f = torch.cat(les_feats, dim=0)
                    les_acc = ridge_probe(les_f, y_roles, les_f, y_roles, 6)
                    lesion_drops[f"expert_{e_idx}"] = float(base_acc - les_acc)

                    for p, sw in zip(moe_r.experts[e_idx].parameters(), saved_w):
                        p.data.copy_(sw)
                    moe_r.experts[e_idx].train(orig_state)

    return {
        "arm": arm,
        "teacher_informed": is_teacher,
        "seed": seed,
        "train_seconds": train_sec,
        "banking77": b77_res,
        "mean_nuance_flip_auc": mean_nf_auc,
        "heldout_families": hf_res,
        "mean_final_r5": mean_r5,
        "per_domain_final": per_dom,
        "routing_diag": {
            "mid_seq_switch_rate": sw_rate,
            "k_dist": k_hist,
            "mean_k": mean_k_val,
            "func_routing_nmi": compute_nmi(bench.func_role_labels, rf_preds),
            "func_routing_ami": ami_diag,
            "lesion_drops": lesion_drops,
        },
    }


def _agg_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not runs:
        return out
    out["n_seeds"] = len(runs)
    out["teacher_informed"] = runs[0].get("teacher_informed", False)
    out["train_seconds"] = {
        "mean": float(np.mean([r["train_seconds"] for r in runs if "train_seconds" in r])),
        "std": float(np.std([r["train_seconds"] for r in runs if "train_seconds" in r])) if len(runs) > 1 else 0.0,
    }
    if all("banking77" in r for r in runs):
        for k in ("intent_77_acc", "same_vs_overlap_diff_auc"):
            vals = [r["banking77"][k] for r in runs]
            out[f"b77_{k}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(runs) > 1 else 0.0,
                "values": vals,
            }
    if all("mean_final_r5" in r for r in runs):
        vals = [r["mean_final_r5"] for r in runs]
        out["mean_final_r5"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(runs) > 1 else 0.0,
            "values": vals,
        }
    if all("heldout_families" in r for r in runs):
        vals = [r["heldout_families"]["auc"] for r in runs]
        out["heldout_families_auc"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(runs) > 1 else 0.0,
            "values": vals,
        }
    if all("routing_diag" in r for r in runs):
        out["mean_k"] = float(np.mean([r["routing_diag"].get("mean_k", 1.0) for r in runs]))
        out["ami"] = float(np.mean([r["routing_diag"]["func_routing_ami"]["ami"] for r in runs if "func_routing_ami" in r["routing_diag"]]))
    if all("track1" in r for r in runs):
        t1_keys = ["seen_acc", "heldout_unseen_acc", "paraphrase_acc", "iit_causal_acc", "iit_heldout_acc", "iit_random_slice_null", "burst_immunity_acc", "abstain_auroc", "near_ood_abstain_auroc"]
        out["track1"] = {}
        for tk in t1_keys:
            vals = [r["track1"][tk] for r in runs if tk in r["track1"]]
            if vals:
                out["track1"][tk] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "values": vals,
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "dev", "test"], default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tracks", choices=["all", "track1", "track2"], default="all")
    ap.add_argument("--out_name", type=str, default=None)
    ap.add_argument("--arms", type=str, nargs="+", default=None)
    ap.add_argument("--dense_lr", type=float, default=3e-3)
    ap.add_argument("--use_constant_fisher", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULT_V9_CFG)
    cfg["use_constant_fisher"] = bool(args.use_constant_fisher)

    if args.mode == "smoke":
        eval_split = "dev"
        seeds = args.seeds or [0]
        steps_per_dom = args.steps or 25
        default_arms = ["BagOfBytes_Null", "Dense_v8_Matched", "Cog_v8"]
        out_name = args.out_name or "smoke_v9"
    elif args.mode == "dev":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 200
        default_arms = ["Dense_v8_Matched", "Cog_v8_SequenceTop1", "Cog_v8"]
        out_name = args.out_name or f"dev_{int(time.time())}"
    else:
        eval_split = "test"
        seeds = args.seeds or [0, 1, 2]
        steps_per_dom = args.steps or 250
        default_arms = [
            "Dense_v8_Matched",
            "Cog_v8_FrozenRandomRouter",
            "Cog_v8_SequenceTop1",
            "Cog_v8_NoI5",
            "Cog_v8_NoRel",
            "Cog_v8_NoEsc",
            "Cog_v8",
            "Cog_v9",
        ]
        out_name = args.out_name or "test_final"

    arms = args.arms or default_arms
    out_dir = RESULTS_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_arms = {
        "Upkeeper_M09",
        "Dense_v8_Matched",
        "Dense_v8_AdamW",
        "Cog_v8_FrozenRandomRouter",
        "Cog_v8_SequenceTop1",
        "Cog_v8_SequenceTop1_NoEsc",
        "Cog_v8_NoEsc",
        "Cog_v8_NoI5",
        "Cog_v8_NoRel",
        "Cog_v8_NoV8Core",
        "Cog_v8",
        "Cog_v9",
        "Cog_v8_GpmNonExempt",
        "Cog_v9_H1a",
        "Cog_v9_H2",
    }

    corpora = None
    bench = None
    teacher = None
    if args.tracks in ("all", "track2"):
        corpora = RealCorpora(seq_len=SEQ_LEN, eval_split=eval_split)
        bench = V8RealBenchmark(corpora, eval_split=eval_split)
        if any(a.endswith("_I9") for a in arms):
            teacher = OfflineTeacherGramEncoder()

    results_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    t1_scale = 0.1 if args.mode == "smoke" else 1.0

    for seed in seeds:
        for arm in arms:
            res: Dict[str, Any] = {"arm": arm, "seed": seed}
            if args.tracks in ("all", "track2"):
                if arm == "Upkeeper_M09":
                    res["banking77"] = {"intent_77_acc": 1.0 / 77.0, "same_vs_overlap_diff_auc": 0.5}
                else:
                    t2_res = run_track2_arm_v9(arm, cfg, bench, seed, steps_per_dom=steps_per_dom, teacher=teacher, dense_adamw_lr=args.dense_lr)
                    res.update(t2_res)
            if args.tracks in ("all", "track1") and arm in t1_arms:
                t1_res = run_track1_arm_v9(arm, cfg, seed, steps_scale=t1_scale, dense_adamw_lr=args.dense_lr)
                res["track1"] = t1_res

            results_by_arm[arm].append(res)
            (out_dir / f"seed{seed}_{arm}.json").write_text(json.dumps(res, indent=2))

            t1_str = ""
            if "track1" in res:
                t1 = res["track1"]
                t1_str = f" | T1 seen={t1['seen_acc']:.3f} held={t1['heldout_unseen_acc']:.3f} iit_seen={t1['iit_causal_acc']:.3f} iit_held={t1['iit_heldout_acc']:.3f} rand_slice={t1['iit_random_slice_null']:.3f}"

            t2_str = ""
            if "banking77" in res:
                t2_str = f"b77_acc={res['banking77']['intent_77_acc']:.3f} b77_auc={res['banking77']['same_vs_overlap_diff_auc']:.3f} r5={res.get('mean_final_r5', 0):.3f} hf={res.get('heldout_families', {}).get('auc', 0):.3f} ami={res.get('routing_diag', {}).get('func_routing_ami', {}).get('ami', 0):.3f}"

            print(f"[{out_name} s{seed} {arm}] {t2_str}{t1_str}", flush=True)

    summary = {
        "meta": {
            "mode": args.mode,
            "eval_split": eval_split,
            "seeds": seeds,
            "steps_per_dom": steps_per_dom,
            "tracks": args.tracks,
            "dense_lr": args.dense_lr,
            "utc": datetime.now(timezone.utc).isoformat(),
            "device": str(DEV),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cfg": cfg,
        },
        "arms": {a: _agg_runs(results_by_arm[a]) for a in arms},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

"""
real_pilot_v7: Architecture v10 (`Cog_v10`) — Semantic Functional Routing & In-Context Rule Induction.

Pre-registered in: results/real_pilot_v7/PREREG.md
Governing protocol: TRUTHGATE-CONTRACT & research-protocol.md.

Arms:
  Track 1: Cog_v10, Cog_v10_NoDualOp, Cog_v9, Dense_v8_Matched
  Track 2: Cog_v10, Cog_v10_NoSIP, Cog_v10_FrozenRandomRouter, Cog_v9, Dense_v8_Matched
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from sklearn.metrics import adjusted_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import (
    BeliefMeaningTask,
    K_CONCEPTS,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
)
from dmoe.null_space import NullSpaceProjector, _READOUT_HEAD_NAMES
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import LatentQuarantineGate, MuonBONGOptimizer, classify_v7_spgc_param_groups
from experiments.real_pilot_data import DOMAINS, RealCorpora, SEQ_LEN
from experiments.run_v7_cognitive_pilot import encode_belief_chunks
from experiments.run_v8_cognitive_pilot import (
    DEFAULT_V8_CFG,
    V8RealBenchmark,
    compute_nmi,
    encode_texts_v8,
    eval_banking77_v8,
    eval_domain_v8,
    eval_routing_diagnostics_v8,
    eval_triples_nuance_v8,
    roc_auc_pairs,
)
from experiments.run_v9_cognitive_pilot import (
    DEFAULT_V9_CFG,
    _sample_episodic_honest_batch,
    compute_ami_with_null,
    set_all_seeds,
)
from experiments.v8_edit_families import generate_i7_procedural_batch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_ROOT = ROOT / "results" / "real_pilot_v7"

DEFAULT_V10_CFG: Dict[str, Any] = dict(
    DEFAULT_V9_CFG,
    use_rel_pool=False,
    enable_nuance_escalation=False,
    use_constant_fisher=True,
    use_iit=True,
    exempt_readout_heads=True,
    enable_surface_whitening=True,
    enable_cris_loss=True,
    enable_dual_ridge_ep=True,
    dual_ridge_op_rank=48,
    decouple_nuance_lifecycle=True,
    specialist_init_scale=0.25,
    shared_dropout_prob=0.18,
    cris_weight=0.75,
    exp_orth_weight=0.15,
    nuance_cluster_min=16,
    nuance_cluster_cos=0.68,
)


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
        "exp_id": "real_pilot_v7",
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


def _perturb_surface_bytes(texts: List[bytes], rng: random.Random) -> List[bytes]:
    """
    Meaning-preserving surface formatting perturbation:
      - Random leading space shift (1..3 spaces, shifting 4-byte chunk boundaries)
      - Random ASCII case swap / upper / lower
      - Random whitespace expansion or contraction
    """
    out: List[bytes] = []
    for raw in texts:
        s = raw.decode("utf-8", errors="ignore")
        mode = rng.randint(0, 2)
        if mode == 0:
            s = s.swapcase()
        elif mode == 1:
            s = s.lower()
        else:
            s = s.upper()
        parts = s.split()
        sep = "  " if rng.random() < 0.5 else " "
        s = (" " * rng.randint(1, 3)) + sep.join(parts)
        out.append(s.encode("utf-8", errors="ignore"))
    return out


def _perturb_byte_tensor(xb: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Perturb a batch of byte-ID tensors [B, L] with meaning-preserving case + boundary shift."""
    B, L = xb.shape
    rows = xb.detach().cpu().tolist()
    texts = []
    for r in rows:
        b = bytes([v for v in r if 0 <= v <= 255])
        texts.append(b)
    pert = _perturb_surface_bytes(texts, rng)
    return encode_texts_v8(pert, L, xb.device, pad_id=257)


def build_v10_arm(
    arm: str,
    cfg: Dict[str, Any],
    seed: int,
    steps_per_dom: int = 250,
) -> Tuple[RealSelfStabilizingDMoE, Optional[NullSpaceProjector], MuonBONGOptimizer, Optional[LatentQuarantineGate], Dict[str, Any]]:
    c = dict(cfg)
    c["initial_experts"] = 1
    c["min_mature_steps"] = min(int(cfg["min_mature_steps"]), max(20, int(steps_per_dom * 0.70)))

    if arm == "Cog_v10":
        pass  # Full v10 config
    elif arm == "Cog_v10_NoDualOp":
        c["enable_dual_ridge_ep"] = False
    elif arm == "Cog_v10_NoSIP":
        c["enable_surface_whitening"] = False
        c["enable_cris_loss"] = False
        c["shared_dropout_prob"] = 0.0
    elif arm == "Cog_v10_FrozenRandomRouter":
        c["enable_surface_whitening"] = False
        c["enable_cris_loss"] = False
        c["shared_dropout_prob"] = 0.0
    elif arm == "Cog_v9":
        c["enable_surface_whitening"] = False
        c["enable_cris_loss"] = False
        c["enable_dual_ridge_ep"] = False
        c["decouple_nuance_lifecycle"] = False
        c["specialist_init_scale"] = 0.05
        c["shared_dropout_prob"] = 0.0
        c["min_mature_steps"] = min(int(DEFAULT_V9_CFG["min_mature_steps"]), max(20, int(steps_per_dom * 0.85)))
        c["nuance_cluster_min"] = int(DEFAULT_V9_CFG["nuance_cluster_min"])
        c["nuance_cluster_cos"] = float(DEFAULT_V9_CFG["nuance_cluster_cos"])
    elif arm == "Dense_v8_Matched":
        c["auto_lifecycle"] = False
        c["max_experts"] = 1
        c["d_ff_shared"] = 512
        c["d_ff_spec"] = 256
        c["enable_surface_whitening"] = False
        c["enable_cris_loss"] = False
        c["enable_dual_ridge_ep"] = False
        c["decouple_nuance_lifecycle"] = False
        c["specialist_init_scale"] = 0.05
        c["shared_dropout_prob"] = 0.0
        c["use_rel_pool"] = True
        c["enable_nuance_escalation"] = True
        c["use_constant_fisher"] = False
    else:
        raise ValueError(f"Unknown v10 arm: {arm}")

    runner_only = {
        "use_iit",
        "i7_steps",
        "exempt_readout_heads",
        "use_constant_fisher",
        "target_mean_k",
        "compute_penalty_weight",
        "cris_weight",
        "exp_orth_weight",
    }
    model_cfg = {k: v for k, v in c.items() if k not in runner_only}
    model = RealSelfStabilizingDMoE(**model_cfg).to(DEV)

    if arm == "Cog_v10_FrozenRandomRouter" and model.context_router is not None:
        for p in model.context_router.parameters():
            p.requires_grad_(False)

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
        exempt_readout_heads=bool(c.get("exempt_readout_heads", True)),
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
    gate = LatentQuarantineGate(latent_dim=c["d_model"], seed=seed, max_bytes=16384, store_samples=False)
    return model, gpm, opt, gate, c


# --------------------------------------------------------------------------------------------------
# Track 1 Honest 5-Probe Episodic Induction Suite
# --------------------------------------------------------------------------------------------------
def _encode_shot_sequences(
    a_s: torch.Tensor, b_s: torch.Tensor, y_s: torch.Tensor
) -> torch.Tensor:
    """Encode support shots [(op=0, a, b, 0, 0, Q_ATTR) + 'Yxx '] -> [n_shots, 28]."""
    n_shots = a_s.size(0)
    c_z = torch.zeros(n_shots, dtype=torch.long)
    q_a = torch.full((n_shots,), Q_ATTR, dtype=torch.long)
    xs_s = torch.stack([torch.zeros(n_shots, dtype=torch.long), a_s + 4, b_s + 4, c_z + 4, c_z + 4, q_a], dim=-1)
    xb_s = encode_belief_chunks(xs_s)  # [n_shots, 24]
    sup_seq = []
    for si in range(n_shots):
        ans_b = torch.tensor(list(f"Y{int(y_s[si].item()):02d} ".encode("ascii")), dtype=torch.long)
        sup_seq.append(torch.cat([xb_s[si], ans_b], dim=0))
    return torch.stack(sup_seq, dim=0)


def _sample_comp_equiv_batch(
    task: BeliefMeaningTask,
    n_episodes: int,
    gen: torch.Generator,
    n_shots: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compositional Equivalence Novel-Pair Induction:
      - Shared concept geometry (task.phi, task.inv_phi), FRESH random output permutation g_ep.
      - Query pair (qa, qb) is STRICTLY NOT in support shots (both qa != a_match and qb != b_match),
        yet maps to the same compositional meaning m_q as one support shot -> transfers g_ep[m_q]!
    """
    phi = task.phi
    inv_phi = task.inv_phi
    x_sup_list: List[torch.Tensor] = []
    x_q_list: List[torch.Tensor] = []
    y_q_list: List[int] = []

    for _ in range(n_episodes):
        g_ep = torch.randperm(K_CONCEPTS, generator=gen)
        a_s = torch.randint(0, K_CONCEPTS, (n_shots,), generator=gen)
        b_s = torch.randint(0, K_CONCEPTS, (n_shots,), generator=gen)
        m_s = inv_phi[(phi[a_s] + 3 * phi[b_s]) % K_CONCEPTS]
        y_s = g_ep[m_s]
        seen_pairs = {(int(a_s[i].item()), int(b_s[i].item())) for i in range(n_shots)}
        x_sup_list.append(_encode_shot_sequences(a_s, b_s, y_s))

        pick = int(torch.randint(0, n_shots, (1,), generator=gen).item())
        target_m = int(m_s[pick].item())
        target_phi_m = int(phi[target_m].item())
        # Pick a novel (qa, qb) != any support pair with phi[qa] + 3*phi[qb] == phi[target_m] mod 16
        cands = []
        for qb_c in range(K_CONCEPTS):
            phi_qa = (target_phi_m - 3 * int(phi[qb_c].item())) % K_CONCEPTS
            qa_c = int(inv_phi[phi_qa].item())
            if (qa_c, qb_c) not in seen_pairs and qa_c != int(a_s[pick].item()) and qb_c != int(b_s[pick].item()):
                cands.append((qa_c, qb_c))
        qa_q, qb_q = cands[int(torch.randint(0, len(cands), (1,), generator=gen).item())]
        xs_q = torch.tensor([[0, qa_q + 4, qb_q + 4, 4, 4, Q_ATTR]], dtype=torch.long)
        x_q_list.append(encode_belief_chunks(xs_q)[0])
        y_q_list.append(int(g_ep[target_m].item()))

    return (
        torch.stack(x_sup_list, dim=0),
        torch.stack(x_q_list, dim=0),
        torch.tensor(y_q_list, dtype=torch.long),
    )


def _sample_unseen_meaning_shift_batch(
    task: BeliefMeaningTask,
    n_episodes: int,
    gen: torch.Generator,
    n_shots: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    True Unseen-Meaning Rule-Shift Induction (Zero-Weight-Update Operator Deduction):
      - Shared concept geometry (task.phi, task.inv_phi), FRESH secret cyclic shift delta_ep in 1..15.
      - Support shots cover at most min(n_shots, 12) meanings, leaving >= 4 meanings STRICTLY UNSEEN.
      - Query (qa, qb) has a STRICTLY UNSEEN meaning m_q not in any support shot (AND novel pair)!
      - Target y_q = inv_phi[(phi[m_q] + delta_ep) % 16].
      - Because delta_ep in 1..15, default non-episodic prediction is 0% accurate and KV lookup is chance!
    """
    phi = task.phi
    inv_phi = task.inv_phi
    x_sup_list: List[torch.Tensor] = []
    x_q_list: List[torch.Tensor] = []
    y_q_list: List[int] = []

    n_unique_sup = min(int(n_shots), K_CONCEPTS - 4)  # guarantee >= 4 strictly unseen meanings
    for _ in range(n_episodes):
        delta_ep = int(torch.randint(1, K_CONCEPTS, (1,), generator=gen).item())
        m_perm = torch.randperm(K_CONCEPTS, generator=gen)
        sup_m_pool = m_perm[:n_unique_sup]
        unseen_m_pool = m_perm[n_unique_sup:]

        # Build n_shots support pairs whose compositional meanings come strictly from sup_m_pool
        a_l, b_l, y_l = [], [], []
        for si in range(n_shots):
            m_val = int(sup_m_pool[si % n_unique_sup].item())
            qb_s = int(torch.randint(0, K_CONCEPTS, (1,), generator=gen).item())
            phi_a = (int(phi[m_val].item()) - 3 * int(phi[qb_s].item())) % K_CONCEPTS
            qa_s = int(inv_phi[phi_a].item())
            y_val = int(inv_phi[(int(phi[m_val].item()) + delta_ep) % K_CONCEPTS].item())
            a_l.append(qa_s)
            b_l.append(qb_s)
            y_l.append(y_val)
        a_s = torch.tensor(a_l, dtype=torch.long)
        b_s = torch.tensor(b_l, dtype=torch.long)
        y_s = torch.tensor(y_l, dtype=torch.long)
        x_sup_list.append(_encode_shot_sequences(a_s, b_s, y_s))

        # Sample query strictly from UNSEEN meanings!
        m_q = int(unseen_m_pool[int(torch.randint(0, unseen_m_pool.numel(), (1,), generator=gen).item())].item())
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


def _pack_episode_memory(
    model: RealSelfStabilizingDMoE,
    x_sup: torch.Tensor,
    top_k: int = 6,
) -> Optional[Dict[str, torch.Tensor]]:
    """Extract and pack [E, n_shots, L_shot] into episode_memory dict for model forward."""
    e_sz, n_sh, l_sh = x_sup.shape
    kv_flat = model.extract_episode_kv(x_sup.view(e_sz * n_sh, l_sh).to(DEV), top_k=top_k, min_nll=0.0)
    if kv_flat is None:
        return None
    k_sh = kv_flat["ep_keys"].size(1)
    d_kv = kv_flat["ep_keys"].size(-1)
    packed: Dict[str, torch.Tensor] = {
        "ep_keys": kv_flat["ep_keys"].view(e_sz, n_sh * k_sh, d_kv),
        "ep_vals": kv_flat["ep_vals"].view(e_sz, n_sh * k_sh, d_kv),
        "ep_mask": kv_flat["ep_mask"].view(e_sz, n_sh * k_sh),
    }
    if "op_keys" in kv_flat and "op_vals" in kv_flat:
        ok_sh = kv_flat["op_keys"].size(1)
        packed["op_keys"] = kv_flat["op_keys"].view(e_sz, n_sh * ok_sh, kv_flat["op_keys"].size(-1))
        packed["op_vals"] = kv_flat["op_vals"].view(e_sz, n_sh * ok_sh, kv_flat["op_vals"].size(-1))
        packed["op_mask"] = kv_flat["op_mask"].view(e_sz, n_sh * ok_sh)
    return packed


@torch.no_grad()
def eval_v10_episodic_suite(
    model: RealSelfStabilizingDMoE,
    cls_head: nn.Module,
    task: BeliefMeaningTask,
    gen: torch.Generator,
    shot_counts: Tuple[int, ...] = (4, 8, 16),
    n_episodes: int = 64,
) -> Dict[str, Any]:
    """Evaluate all 6 honest episodic probes + controls across shot counts (zero weight updates)."""
    model.eval()
    res: Dict[str, Any] = {}
    for n_sh in shot_counts:
        # 1-3: Lookup, Shuffled Null, Impossible-Entropy Random-Perm Null
        x_sup, x_ql, y_ql, x_qn, y_qn = _sample_episodic_honest_batch(task, n_episodes, gen, n_shots=n_sh)
        ep_kv = _pack_episode_memory(model, x_sup)
        if ep_kv is None:
            res[f"shots_{n_sh}"] = {
                "lookup_acc": 1.0 / K_CONCEPTS,
                "shuffled_null_acc": 1.0 / K_CONCEPTS,
                "impossible_perm_null_acc": 1.0 / K_CONCEPTS,
                "comp_equiv_induction_acc": 1.0 / K_CONCEPTS,
                "unseen_meaning_shift_acc": 1.0 / K_CONCEPTS,
                "shift_shuffled_null_acc": 1.0 / K_CONCEPTS,
                "shift_no_mem_blind_acc": 1.0 / K_CONCEPTS,
            }
            continue

        o_lu = model(x_ql.to(DEV), update_stats=False, episode_memory=ep_kv)
        lookup_acc = float((cls_head(o_lu["latent"][:, -1, :]).argmax(-1).cpu() == y_ql).float().mean().item())

        ep_shuff = {k: torch.roll(v, shifts=1, dims=0) for k, v in ep_kv.items()}
        o_sh = model(x_ql.to(DEV), update_stats=False, episode_memory=ep_shuff)
        shuff_acc = float((cls_head(o_sh["latent"][:, -1, :]).argmax(-1).cpu() == y_ql).float().mean().item())

        o_imp = model(x_qn.to(DEV), update_stats=False, episode_memory=ep_kv)
        imp_acc = float((cls_head(o_imp["latent"][:, -1, :]).argmax(-1).cpu() == y_qn).float().mean().item())

        # 4: Compositional Equivalence Novel-Pair Induction (fresh random g_ep, unseen (qa, qb), m_q in m_sup)
        x_sup_c, x_qc, y_qc = _sample_comp_equiv_batch(task, n_episodes, gen, n_shots=n_sh)
        ep_c = _pack_episode_memory(model, x_sup_c)
        o_c = model(x_qc.to(DEV), update_stats=False, episode_memory=ep_c)
        comp_acc = float((cls_head(o_c["latent"][:, -1, :]).argmax(-1).cpu() == y_qc).float().mean().item())

        # 5: Unseen-Meaning Rule-Shift Induction (fresh secret delta_ep in 1..15, unseen m_q not in m_sup!)
        x_sup_s, x_qs, y_qs = _sample_unseen_meaning_shift_batch(task, n_episodes, gen, n_shots=n_sh)
        ep_s = _pack_episode_memory(model, x_sup_s)
        o_s = model(x_qs.to(DEV), update_stats=False, episode_memory=ep_s)
        shift_acc = float((cls_head(o_s["latent"][:, -1, :]).argmax(-1).cpu() == y_qs).float().mean().item())

        # Controls on the exact same unseen-meaning shift queries:
        ep_s_shuff = {k: torch.roll(v, shifts=1, dims=0) for k, v in ep_s.items()}
        o_s_sh = model(x_qs.to(DEV), update_stats=False, episode_memory=ep_s_shuff)
        shift_shuff_acc = float((cls_head(o_s_sh["latent"][:, -1, :]).argmax(-1).cpu() == y_qs).float().mean().item())

        o_s_blind = model(x_qs.to(DEV), update_stats=False, episode_memory=None)
        shift_blind_acc = float((cls_head(o_s_blind["latent"][:, -1, :]).argmax(-1).cpu() == y_qs).float().mean().item())

        res[f"shots_{n_sh}"] = {
            "lookup_acc": lookup_acc,
            "shuffled_null_acc": shuff_acc,
            "impossible_perm_null_acc": imp_acc,
            "comp_equiv_induction_acc": comp_acc,
            "unseen_meaning_shift_acc": shift_acc,
            "shift_shuffled_null_acc": shift_shuff_acc,
            "shift_no_mem_blind_acc": shift_blind_acc,
        }
    return res


# --------------------------------------------------------------------------------------------------
# Track 1 Runner (v10)
# --------------------------------------------------------------------------------------------------
def run_track1_arm_v10(
    arm: str,
    cfg: Dict[str, Any],
    seed: int,
    steps_scale: float = 1.0,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    t_start = time.time()
    task = BeliefMeaningTask(seed=seed)
    gen = torch.Generator().manual_seed(seed * 97 + 1)

    n_p1 = max(220, int(1300 * steps_scale))
    n_pb = max(8, int(20 * steps_scale))
    n_p2 = max(40, int(180 * steps_scale))
    n_p3 = max(40, int(180 * steps_scale))

    t1_cfg = dict(cfg)
    t1_cfg["min_mature_steps"] = n_p1 + 50
    t1_cfg["specialist_init_scale"] = 0.05
    t1_cfg["enable_surface_whitening"] = False
    model, gpm, opt, gate, arm_cfg = build_v10_arm(arm, t1_cfg, seed, steps_per_dom=n_p1 + n_p2 + n_p3 + 50)
    cls_head = nn.Linear(arm_cfg["d_model"], K_CONCEPTS).to(DEV)

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

    D_h = arm_cfg["d_model"]
    use_iit = bool(arm_cfg.get("use_iit", True))
    use_const_fisher = bool(arm_cfg.get("use_constant_fisher", True))
    # Matched episodic induction training exposure for v10 & v10_NoDualOp (pre-registered §2.2)
    train_v10_ep = arm in ("Cog_v10", "Cog_v10_NoDualOp")
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
            if flips > 0:
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

            # I6 Episodic Memory & Rule Induction Training (identical cadence; v10 & v10_NoDualOp matched!)
            if use_iit and ph == "p1":
                if train_v10_ep and (step_i >= (n_p1 // 2)) and (step_i % 2 == 1):
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
                        loss = loss + 0.45 * F.cross_entropy(cls_head(o_ep["latent"][:, -1, :]), y_q_ep.to(DEV))
                elif (not train_v10_ep) and (step_i % 3 == 1) and step_i >= (n_p1 // 5):
                    x_sup_ep, x_ql_ep, y_ql_ep, _, _ = _sample_episodic_honest_batch(task, 16, gen, n_shots=4)
                    ep_kv = _pack_episode_memory(model, x_sup_ep, top_k=6)
                    if ep_kv is not None:
                        o_ep = model(x_ql_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                        loss = loss + 0.40 * F.cross_entropy(cls_head(o_ep["latent"][:, -1, :]), y_ql_ep.to(DEV))

            if use_const_fisher:
                fisher_sc = 1.0
            else:
                acc_batch = float((lg[:b_sz].detach().argmax(-1) == y_train[:b_sz]).float().mean().item())
                fisher_sc = 1.0 if ph == "pb" else (0.25 + 1.15 * acc_batch)

        opt.zero_grad()
        loss.backward()
        if gpm is not None and gpm.p_null:
            gpm.project_gradients()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(cls_head.parameters()), 1.5)
        opt.step(fisher_scale=fisher_sc)
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

    for st in range(n_p1):
        xs, y = task.sample_stream_batch("p1", 64, gen)
        step_t1(xs, y, f"s_{st % 2}", "p1", st)
        if (st + 1) % 100 == 0 or st == n_p1 - 1:
            grokking_curve.append(eval_t1("p1", "heldout"))

    if gpm is not None:
        gpm.update_basis_for_domain(0)
        opt.set_group_mode("trunk", "consolidate")

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
        if ok.get("abstain_logits") is not None and ou.get("abstain_logits") is not None:
            sk = (0.5 * ok["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ok["nuance_score"].mean(dim=-1) / 3.0)).tolist()
            su = (0.5 * ou["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ou["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        else:
            sk = ok["nuance_score"].mean(dim=-1).tolist()
            su = ou["nuance_score"].mean(dim=-1).tolist()
        abstain_auroc = roc_auc_pairs(su, sk)

        near_ood_lines = [
            f"Q9 C{int(xs_k[i, 1] - 4):02d} C{int(xs_k[i, 2] - 4):02d} C00 C00 ? ".encode("ascii")
            for i in range(64)
        ]
        x_near_ood = encode_texts_v8(near_ood_lines, 24, DEV, pad_id=257)
        o_near_ood = model(x_near_ood, update_stats=False)
        if ok.get("abstain_logits") is not None and o_near_ood.get("abstain_logits") is not None:
            sn = (0.5 * o_near_ood["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (o_near_ood["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        else:
            sn = o_near_ood["nuance_score"].mean(dim=-1).tolist()
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


# --------------------------------------------------------------------------------------------------
# Track 2 Honest Semantic Routing & Frozen-Probe Lesion Diagnostics
# --------------------------------------------------------------------------------------------------
def _fit_frozen_ridge_probe(ftr: torch.Tensor, ytr: torch.Tensor, n_cls: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a ridge classifier strictly on clean train features and return (mu, sd, W)."""
    mu = ftr.mean(0, keepdim=True)
    sd = ftr.std(0, keepdim=True) + 1e-6
    ftr_n = (ftr - mu) / sd
    n = ftr_n.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    va, tr = perm[: max(1, n // 5)], perm[max(1, n // 5) :]

    def fit(F_: torch.Tensor, y_: torch.Tensor, lam: float) -> torch.Tensor:
        Y = F.one_hot(y_, n_cls).float()
        A = F_.T @ F_ + lam * torch.eye(F_.size(1))
        return torch.linalg.solve(A, F_.T @ Y)

    best_lam, best_acc = 10.0, -1.0
    for lam in (1e-1, 1.0, 10.0, 100.0, 1000.0):
        W = fit(ftr_n[tr], ytr[tr], lam)
        acc = float(((ftr_n[va] @ W).argmax(-1) == ytr[va]).float().mean())
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    W = fit(ftr_n, ytr, best_lam)
    return mu, sd, W


@torch.no_grad()
def eval_v10_semantic_routing_suite(
    model: RealSelfStabilizingDMoE,
    bench: V8RealBenchmark,
) -> Dict[str, Any]:
    """
    Honest Semantic Functional Routing Probe Suite:
      1. func_routing_ami (+ 50-perm null) across 6 held-out functional roles
      2. length_matched_role_ami (all 6 roles fixed to exact 48 content bytes -> zero length shortcut!)
      3. routing_length_ami (spurious AMI with byte-length quartiles -> lower is better!)
      4. routing_perturb_agree (routing agreement under case + whitespace + boundary shift)
      5. Frozen-probe out-of-sample lesion test + relative latent perturbation per expert across ALL upper MoE layers!
    """
    model.eval()
    is_v78 = bool(getattr(model, "use_v7_core", False))
    pad_id = 257 if is_v78 else 32

    def get_assigned_and_latents(texts: List[bytes]) -> Tuple[List[int], torch.Tensor]:
        preds: List[int] = []
        feats: List[torch.Tensor] = []
        for s in range(0, len(texts), 64):
            xb = encode_texts_v8(texts[s : s + 64], SEQ_LEN, DEV, pad_id=pad_id)
            o = model(xb, update_stats=False)
            preds.extend(o["assigned_experts"].detach().cpu().tolist())
            feats.append(o["latent"][:, -1, :].float().cpu())
        return preds, torch.cat(feats, dim=0)

    # 1. Raw functional role AMI
    rf_preds, base_feats = get_assigned_and_latents(bench.func_role_texts)
    ami_raw = compute_ami_with_null(bench.func_role_labels, rf_preds, n_perms=50)

    # 2. Length-matched functional role AMI (exact 48 content bytes for EVERY sequence -> zero length cue!)
    lm_texts = [t[:48].ljust(48, b" ") for t in bench.func_role_texts]
    lm_preds, _ = get_assigned_and_latents(lm_texts)
    ami_len_matched = compute_ami_with_null(bench.func_role_labels, lm_preds, n_perms=50)

    # 3. Spurious sequence-length quartile AMI across mixed-length texts (func_role + Banking77)
    len_texts = list(bench.func_role_texts) + list(bench.b77_te_txt[:168])
    byte_lens = np.array([len(t) for t in len_texts], dtype=np.float64)
    q25, q50, q75 = np.quantile(byte_lens, [0.25, 0.50, 0.75])
    len_bins = [int(bl > q25) + int(bl > q50) + int(bl > q75) for bl in byte_lens]
    len_preds, _ = get_assigned_and_latents(len_texts)
    routing_length_ami = float(adjusted_mutual_info_score(len_bins, len_preds))

    # 4. Meaning-preserving surface formatting perturbation agreement
    rng_eval = random.Random(2026)
    pert_texts = _perturb_surface_bytes(bench.func_role_texts, rng_eval)
    pert_preds, _ = get_assigned_and_latents(pert_texts)
    routing_perturb_agree = float(np.mean([int(a == b) for a, b in zip(rf_preds, pert_preds)]))

    # 5. Honest Out-of-Sample Frozen-Probe Multi-Layer Expert Lesion Analysis
    y_roles = torch.tensor(bench.func_role_labels, dtype=torch.long)
    # Stratified 50/50 split across the 6 roles (14 train / 14 test per role)
    tr_idx, te_idx = [], []
    for r in range(6):
        r_idxs = [i for i, lbl in enumerate(bench.func_role_labels) if lbl == r]
        tr_idx.extend(r_idxs[: len(r_idxs) // 2])
        te_idx.extend(r_idxs[len(r_idxs) // 2 :])
    tr_t = torch.tensor(tr_idx, dtype=torch.long)
    te_t = torch.tensor(te_idx, dtype=torch.long)
    te_texts = [bench.func_role_texts[i] for i in te_idx]
    y_te = y_roles[te_t]

    mu_p, sd_p, W_p = _fit_frozen_ridge_probe(base_feats[tr_t], y_roles[tr_t], 6)
    base_te = base_feats[te_t]
    base_preds_te = (((base_te - mu_p) / sd_p) @ W_p).argmax(-1)
    base_test_acc = float((base_preds_te == y_te).float().mean().item())
    base_role_accs = [
        float((base_preds_te[y_te == r] == r).float().mean().item()) for r in range(6)
    ]

    moe_layers = [l.moe for l in model.layers[model.router_layer :] if not getattr(l.moe, "v7_trunk_only", False)]
    n_exp = len(moe_layers[0].experts) if moe_layers else 1
    lesion_acc_drops: Dict[str, float] = {}
    lesion_rel_deltas: Dict[str, float] = {}
    lesion_role_deltas: Dict[str, List[float]] = {}
    lesion_role_acc_drops: Dict[str, List[float]] = {}

    if n_exp > 1 and moe_layers:
        base_norm = base_te.norm(dim=-1).clamp_min(1e-6)
        for e_idx in range(n_exp):
            # Save & zero expert e_idx across ALL upper MoE layers
            saved_all = []
            for moe_l in moe_layers:
                if e_idx < len(moe_l.experts):
                    exp = moe_l.experts[e_idx]
                    saved_all.append((exp, [p.data.clone() for p in exp.parameters()]))
                    for p in exp.parameters():
                        p.data.zero_()

            _, les_te = get_assigned_and_latents(te_texts)
            les_preds_te = (((les_te - mu_p) / sd_p) @ W_p).argmax(-1)
            les_test_acc = float((les_preds_te == y_te).float().mean().item())
            lesion_acc_drops[f"expert_{e_idx}"] = float(base_test_acc - les_test_acc)
            rel_d_vec = (base_te - les_te).norm(dim=-1) / base_norm
            lesion_rel_deltas[f"expert_{e_idx}"] = float(rel_d_vec.mean().item())
            lesion_role_deltas[f"expert_{e_idx}"] = [
                float(rel_d_vec[y_te == r].mean().item()) for r in range(6)
            ]
            lesion_role_acc_drops[f"expert_{e_idx}"] = [
                float(base_role_accs[r] - float((les_preds_te[y_te == r] == r).float().mean().item()))
                for r in range(6)
            ]

            for exp, sw_list in saved_all:
                for p, sw in zip(exp.parameters(), sw_list):
                    p.data.copy_(sw)

    mean_rel_delta = float(np.mean(list(lesion_rel_deltas.values()))) if lesion_rel_deltas else 0.0
    max_acc_drop = float(max(lesion_acc_drops.values())) if lesion_acc_drops else 0.0
    max_role_drop = (
        float(max(max(v) for v in lesion_role_acc_drops.values()))
        if lesion_role_acc_drops
        else 0.0
    )
    # Role specificity: average across experts of (max role delta / mean role delta), 1.0 = uniform, >1.0 = role-selective
    spec_ratios = []
    for rd_vals in lesion_role_deltas.values():
        m_v = float(np.mean(rd_vals))
        if m_v > 1e-5:
            spec_ratios.append(float(np.max(rd_vals)) / m_v)
    role_specificity = float(np.mean(spec_ratios)) if spec_ratios else 0.0

    return {
        "num_experts": int(n_exp),
        "func_routing_nmi": compute_nmi(bench.func_role_labels, rf_preds),
        "func_routing_ami": ami_raw,
        "length_matched_role_ami": ami_len_matched,
        "routing_length_ami": routing_length_ami,
        "routing_perturb_agree": routing_perturb_agree,
        "lesion_base_test_acc": base_test_acc,
        "lesion_acc_drops": lesion_acc_drops,
        "lesion_rel_deltas": lesion_rel_deltas,
        "lesion_role_deltas": lesion_role_deltas,
        "lesion_role_acc_drops": lesion_role_acc_drops,
        "lesion_max_acc_drop": max_acc_drop,
        "lesion_max_role_drop": max_role_drop,
        "lesion_mean_rel_delta": mean_rel_delta,
        "lesion_role_specificity": role_specificity,
    }


# --------------------------------------------------------------------------------------------------
# Track 2 Runner (v10)
# --------------------------------------------------------------------------------------------------
def run_track2_arm_v10(
    arm: str,
    cfg: Dict[str, Any],
    bench: V8RealBenchmark,
    seed: int,
    steps_per_dom: int = 250,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    model, gpm, opt, gate, arm_cfg = build_v10_arm(arm, cfg, seed, steps_per_dom=steps_per_dom)

    i7_steps = int(arm_cfg.get("i7_steps", 150))
    use_cris = bool(arm_cfg.get("enable_cris_loss", False))
    cris_w = float(arm_cfg.get("cris_weight", 0.35))
    orth_w = float(arm_cfg.get("exp_orth_weight", 0.15))
    rebuild_on_spawn = arm in ("Cog_v10", "Cog_v10_NoSIP", "Cog_v10_FrozenRandomRouter")

    t_start = time.time()
    gen_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    total_spawns = 0

    if i7_steps > 0 and getattr(model, "use_v8_core", False):
        model.train()
        for _ in range(i7_steps):
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
                    cris_weight=(cris_w if use_cris else 0.0),
                    exp_orth_weight=(orth_w if use_cris else 0.0),
                )
            opt.zero_grad()
            l_i7["loss"].backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
            opt.step(fisher_scale=0.5)

    for dom_idx, dom in enumerate(DOMAINS):
        model.train()
        shared_mode = "consolidate" if dom_idx > 0 else "acquire"
        if gpm is not None and dom_idx > 0:
            # Fix v9 bug: actual group role is "trunk" (not "shared_trunk")
            opt.set_group_mode("trunk" if rebuild_on_spawn else "shared_trunk", "consolidate")

        for step in range(steps_per_dom):
            xb_raw, y_eq, contra_pair, hard_pair, nuance_t, struct_t, g_mask = bench.sample_v8_stream_batch(dom, gen_np, rng_py)

            model.train()
            saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(xb_raw, update_stats=True)

                # Self-supervised surface formatting perturbation invariance every 4th step (4 items -> fast)
                rq_pert = None
                rq_pert_idx = None
                if use_cris and (step % 4 == 0):
                    # Sample 4 diverse rows: 1 raw corpus, 1 edit, 2 Banking77
                    pert_idx = torch.tensor([0, 4, 10, 12], dtype=torch.long, device=DEV)
                    xb_pert = _perturb_byte_tensor(xb_raw[pert_idx], rng_py)
                    rq_pert, _ = model.extract_router_queries(xb_pert)
                    rq_pert_idx = pert_idx

                loss_dict = model.compute_v7_non_token_loss(
                    out,
                    x=xb_raw,
                    equiv_labels=y_eq,
                    contra_pair_idx=contra_pair,
                    hard_neg_pair_idx=hard_pair,
                    nuance_targets=nuance_t,
                    teacher_gram=None,
                    gram_mask=g_mask,
                    struct_targets=struct_t,
                    router_q_perturbed=rq_pert,
                    router_q_pert_idx=rq_pert_idx,
                    cris_weight=(cris_w if use_cris else 0.0),
                    exp_orth_weight=(orth_w if use_cris else 0.0),
                )
                loss = loss_dict["loss"]

            # Critical v10 fix: rebuild optimizer preserving state whenever lifecycle spawns a new expert!
            if rebuild_on_spawn and any(ev.startswith("spawned_") for ev in out.get("lifecycle_events", [])):
                total_spawns += 1
                new_groups = classify_v7_spgc_param_groups(
                    model,
                    shared_mode=shared_mode,
                    specialist_mode="acquire",
                    lr_muon=8e-3,
                    lr_adam=3e-3,
                    lr_consolidate=2e-3,
                    trunk_prec_scale=0.85,
                )
                opt.rebuild_preserving_state(new_groups)

            opt.zero_grad()
            loss.backward()
            if gpm is not None and gpm.p_null:
                gpm.project_gradients()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.5)
            opt.step(fisher_scale=1.0)
            if saved is not None:
                gpm.project_parameter_deltas(saved)
            if gpm is not None and getattr(gpm, "defer_cov_commit", False):
                gpm.commit_pending()

        if gpm is not None:
            gpm.update_basis_for_domain(dom_idx)

    train_sec = time.time() - t_start

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
        "total_spawns": total_spawns,
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


def _agg_v10_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    ap.add_argument("--cris_weight", type=float, default=None)
    ap.add_argument("--exp_orth_weight", type=float, default=None)
    ap.add_argument("--shared_dropout_prob", type=float, default=None)
    ap.add_argument("--dual_ridge_op_rank", type=int, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_V10_CFG)
    if args.cris_weight is not None:
        cfg["cris_weight"] = float(args.cris_weight)
    if args.exp_orth_weight is not None:
        cfg["exp_orth_weight"] = float(args.exp_orth_weight)
    if args.shared_dropout_prob is not None:
        cfg["shared_dropout_prob"] = float(args.shared_dropout_prob)
    if args.dual_ridge_op_rank is not None:
        cfg["dual_ridge_op_rank"] = int(args.dual_ridge_op_rank)

    if args.mode == "smoke":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 25
        default_arms = ["Cog_v10", "Cog_v10_NoDualOp"] if args.tracks == "track1" else ["Cog_v10", "Cog_v10_NoSIP"]
        out_name = args.out_name or "smoke_v10"
        t1_scale = 0.2
    elif args.mode == "dev":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 250
        default_arms = (
            ["Cog_v10", "Cog_v10_NoDualOp"]
            if args.tracks == "track1"
            else ["Cog_v10", "Cog_v10_NoSIP", "Cog_v10_FrozenRandomRouter"]
        )
        out_name = args.out_name or "dev_v10"
        t1_scale = 1.0
    else:
        eval_split = "test"
        if args.tracks == "track1":
            seeds = args.seeds or [0, 1, 2, 3, 4]
            default_arms = ["Cog_v10", "Cog_v10_NoDualOp", "Cog_v9", "Dense_v8_Matched"]
            out_name = args.out_name or "test_track1"
        elif args.tracks == "track2":
            seeds = args.seeds or [0, 1, 2]
            default_arms = ["Cog_v10", "Cog_v10_NoSIP", "Cog_v10_FrozenRandomRouter", "Cog_v9", "Dense_v8_Matched"]
            out_name = args.out_name or "test_track2"
        else:
            seeds = args.seeds or [0, 1, 2]
            default_arms = ["Cog_v10", "Cog_v10_NoDualOp", "Cog_v10_NoSIP", "Cog_v10_FrozenRandomRouter", "Cog_v9", "Dense_v8_Matched"]
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
            res: Dict[str, Any] = {"meta": meta, "arm": arm, "seed": seed}
            if args.tracks in ("all", "track2"):
                assert bench is not None
                res.update(run_track2_arm_v10(arm, cfg, bench, seed, steps_per_dom=steps_per_dom))
            if args.tracks in ("all", "track1"):
                res["track1"] = run_track1_arm_v10(arm, cfg, seed, steps_scale=t1_scale)

            results_by_arm[arm].append(res)
            (out_dir / f"seed{seed}_{arm}.json").write_text(json.dumps(res, indent=2))

            parts = [f"[{out_name} s{seed} {arm}]"]
            if "banking77" in res:
                rd = res["routing_diag"]
                parts.append(
                    f"T2({res['train_seconds']:.0f}s) b77={res['banking77']['intent_77_acc']:.3f} "
                    f"k={rd['mean_k']:.2f} E={rd['num_experts']} "
                    f"lm_ami={rd['length_matched_role_ami']['ami']:.3f} "
                    f"len_ami={rd['routing_length_ami']:.3f} "
                    f"pert={rd['routing_perturb_agree']:.3f} "
                    f"les_d={rd['lesion_mean_rel_delta']:.3f}"
                )
            if "track1" in res:
                t1 = res["track1"]
                ep8 = t1["episodic_v10"]["shots_8"]
                parts.append(
                    f"T1({t1['train_seconds']:.0f}s) seen={t1['seen_acc']:.3f} held={t1['heldout_unseen_acc']:.3f} "
                    f"iit_h={t1['iit_heldout_acc']:.3f} lu8={ep8['lookup_acc']:.3f} "
                    f"comp8={ep8['comp_equiv_induction_acc']:.3f} shift8={ep8['unseen_meaning_shift_acc']:.3f} "
                    f"shuff8={ep8['shift_shuffled_null_acc']:.3f} imp8={ep8['impossible_perm_null_acc']:.3f}"
                )
            print(" | ".join(parts), flush=True)

    summary = {
        "meta": meta,
        "arms": {a: _agg_v10_runs(results_by_arm[a]) for a in arms},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

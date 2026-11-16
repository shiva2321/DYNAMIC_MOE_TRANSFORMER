"""
Joint Multi-Hop Belief-Revision + Surprise-Driven Latent Thinking Benchmark (Track KR).

Evaluates 9 conditions across 20 seeds on a continual stream of multi-hop state-transition chains
(trained only on depths D <= 3, tested on depths D in 0..5 including zero-shot OOD depths D = 4, 5):
1. Null_Untrained
2. Null_Shuffled
3. Null_Blind
4. Baseline_Dense_1Pass_NoReplay
5. Baseline_Dense_1Pass_Replay128
6. Baseline_Dense_1Pass_Upkeeper (Ablation: Same Upkeeper, 1-Pass Architecture without Thinker)
7. Baseline_NonNeural_Dictionary_Counters (Algorithmic bigram counting dictionary)
8. dMoE_Thinker_NoUpkeeper (Ablation: Same Thinker Architecture, without Upkeeper)
9. dMoE_Thinker_Upkeeper_Adaptive (Proposed Combined System: SurpriseDrivenLatentThinker + Upkeeper)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.upkeeper import EvidenceAccumulatingUpkeeper
from scripts.baselines_track_k import (
    DenseTransformerBaseline,
    NonNeuralDictionaryBaseline,
)


def build_transition_tables(seed: int = 99) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    trans_init = torch.stack([torch.randperm(8, generator=gen) for _ in range(8)])
    trans_burst = (trans_init + 3) % 8
    trans_rev = (trans_init + 5) % 8
    return trans_init, trans_burst, trans_rev


def make_chain_batch(
    trans_table: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_depth: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.zeros(batch_size, seq_len, dtype=torch.long)
    cur_s = torch.randint(0, 8, (batch_size,), generator=g)
    x[:, 0] = cur_s
    y[:, 0] = cur_s
    cur_d = torch.zeros(batch_size, dtype=torch.long)
    for t in range(1, seq_len):
        force_reset = cur_d >= max_depth
        rand_reset = torch.rand(batch_size, generator=g) < 0.25
        do_reset = force_reset | rand_reset
        new_s = torch.randint(0, 8, (batch_size,), generator=g)
        op = torch.randint(0, 8, (batch_size,), generator=g)
        x[:, t] = torch.where(do_reset, new_s, op + 8)
        cur_s = torch.where(do_reset, new_s, trans_table[op, cur_s])
        cur_d = torch.where(do_reset, torch.zeros_like(cur_d), cur_d + 1)
        y[:, t] = cur_s
        depths[:, t] = cur_d
    return x.to(device) + 32, y.to(device) + 32, depths.to(device)


def eval_acc(preds: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> Tuple[float, float]:
    overall = (preds == y).float().mean().item()
    ood_mask = d >= 4
    ood = (preds[ood_mask] == y[ood_mask]).float().mean().item() if ood_mask.any() else 0.0
    return overall, ood


def run_seed(seed: int, train_steps: int, rev_steps: int, device: torch.device) -> Dict[str, Dict[str, float]]:
    trans_init, trans_burst, trans_rev = build_transition_tables(seed=99)
    x_eval_init, y_eval_init, d_eval_init = make_chain_batch(
        trans_init, 256, 16, max_depth=5, seed=80000 + seed, device=device
    )
    x_eval_rev, y_eval_rev, d_eval_rev = make_chain_batch(
        trans_rev, 256, 16, max_depth=5, seed=90000 + seed, device=device
    )

    out: Dict[str, Dict[str, float]] = {}
    sources = ["source_alpha", "source_beta", "source_gamma"]

    # 1. Null_Untrained
    torch.manual_seed(seed)
    m_unt = RealSelfStabilizingDMoE(
        d_model=32, num_heads=2, d_ff=128, initial_layers=1, chunk_size=1,
        enable_thinking=True, max_thinking_steps=5, theta_halt=0.18,
    ).to(device)
    m_unt.eval()
    with torch.no_grad():
        p_init = m_unt(x_eval_init)["output"].squeeze(2).argmax(dim=-1)
        p_rev = m_unt(x_eval_rev)["output"].squeeze(2).argmax(dim=-1)
        ai, aoi = eval_acc(p_init, y_eval_init, d_eval_init)
        ar, aor = eval_acc(p_rev, y_eval_rev, d_eval_rev)
        out["Null_Untrained"] = {
            "acc_initial": ai,
            "acc_initial_ood": aoi,
            "acc_k2_post_burst": ai,
            "k1_retention": 1.0,
            "acc_k3_post_revised": ar,
            "acc_k3_ood_revised": aor,
            "active_token_fraction": 0.0,
        }

    # 2. Null_Shuffled
    torch.manual_seed(seed)
    m_shuf = RealSelfStabilizingDMoE(
        d_model=32, num_heads=2, d_ff=128, initial_layers=1, chunk_size=1,
        enable_thinking=True, max_thinking_steps=5, theta_halt=0.18,
    ).to(device)
    opt_shuf = torch.optim.AdamW(m_shuf.parameters(), lr=4e-3)
    m_shuf.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        yb_s = torch.randint(32, 40, yb.shape, device=device)
        opt_shuf.zero_grad()
        res_s = m_shuf(xb, force_thinking_steps=3)
        loss_s = m_shuf.thinker.compute_thinking_loss(res_s["thinking_out"], yb_s)["loss"]
        loss_s.backward()
        opt_shuf.step()
    m_shuf.eval()
    with torch.no_grad():
        p_init = m_shuf(x_eval_init)["output"].squeeze(2).argmax(dim=-1)
        p_rev = m_shuf(x_eval_rev)["output"].squeeze(2).argmax(dim=-1)
        ai, aoi = eval_acc(p_init, y_eval_init, d_eval_init)
        ar, aor = eval_acc(p_rev, y_eval_rev, d_eval_rev)
        out["Null_Shuffled"] = {
            "acc_initial": ai,
            "acc_initial_ood": aoi,
            "acc_k2_post_burst": ai,
            "k1_retention": 1.0,
            "acc_k3_post_revised": ar,
            "acc_k3_ood_revised": aor,
            "active_token_fraction": 0.0,
        }

    # 3. Baseline_Dense_1Pass_NoReplay
    torch.manual_seed(seed)
    m_d_nr = DenseTransformerBaseline(d_model=32, num_heads=2, d_ff=128, num_layers=2, chunk_size=1).to(device)
    opt_dnr = torch.optim.AdamW(m_d_nr.parameters(), lr=4e-3)
    m_d_nr.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        opt_dnr.zero_grad()
        F.cross_entropy(m_d_nr(xb).reshape(-1, 256), yb.reshape(-1)).backward()
        opt_dnr.step()
    m_d_nr.eval()
    with torch.no_grad():
        ai, aoi = eval_acc(m_d_nr(x_eval_init).argmax(dim=-1), y_eval_init, d_eval_init)
    m_d_nr.train()
    opt_dnr_b = torch.optim.AdamW(m_d_nr.parameters(), lr=4e-3)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=seed * 2000 + st, device=device)
        opt_dnr_b.zero_grad()
        F.cross_entropy(m_d_nr(xb).reshape(-1, 256), yb.reshape(-1)).backward()
        opt_dnr_b.step()
    m_d_nr.eval()
    with torch.no_grad():
        ak2, _ = eval_acc(m_d_nr(x_eval_init).argmax(dim=-1), y_eval_init, d_eval_init)
    m_d_nr.train()
    opt_dnr_r = torch.optim.AdamW(m_d_nr.parameters(), lr=4e-3)
    for st in range(rev_steps):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=seed * 3000 + st, device=device)
        opt_dnr_r.zero_grad()
        F.cross_entropy(m_d_nr(xb).reshape(-1, 256), yb.reshape(-1)).backward()
        opt_dnr_r.step()
    m_d_nr.eval()
    with torch.no_grad():
        ar, aor = eval_acc(m_d_nr(x_eval_rev).argmax(dim=-1), y_eval_rev, d_eval_rev)
    out["Baseline_Dense_1Pass_NoReplay"] = {
        "acc_initial": ai,
        "acc_initial_ood": aoi,
        "acc_k2_post_burst": ak2,
        "k1_retention": ak2 / max(ai, 1e-6),
        "acc_k3_post_revised": ar,
        "acc_k3_ood_revised": aor,
        "active_token_fraction": 0.0,
    }

    # 4. Baseline_Dense_1Pass_Replay128
    random.seed(seed)
    torch.manual_seed(seed)
    m_d_rep = DenseTransformerBaseline(d_model=32, num_heads=2, d_ff=128, num_layers=2, chunk_size=1).to(device)
    opt_drep = torch.optim.AdamW(m_d_rep.parameters(), lr=4e-3)
    replay_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    m_d_rep.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        for i in range(xb.size(0)):
            if len(replay_buf) < 128:
                replay_buf.append((xb[i].detach().clone(), yb[i].detach().clone()))
            else:
                replay_buf[random.randint(0, 127)] = (xb[i].detach().clone(), yb[i].detach().clone())
        opt_drep.zero_grad()
        F.cross_entropy(m_d_rep(xb).reshape(-1, 256), yb.reshape(-1)).backward()
        opt_drep.step()
    m_d_rep.eval()
    with torch.no_grad():
        ai, aoi = eval_acc(m_d_rep(x_eval_init).argmax(dim=-1), y_eval_init, d_eval_init)
    m_d_rep.train()
    opt_drep_b = torch.optim.AdamW(m_d_rep.parameters(), lr=4e-3)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=seed * 2000 + st, device=device)
        r_idx = [random.randint(0, len(replay_buf) - 1) for _ in range(16)]
        rx = torch.stack([replay_buf[i][0] for i in r_idx])
        ry = torch.stack([replay_buf[i][1] for i in r_idx])
        cx = torch.cat([xb, rx], dim=0)
        cy = torch.cat([yb, ry], dim=0)
        opt_drep_b.zero_grad()
        F.cross_entropy(m_d_rep(cx).reshape(-1, 256), cy.reshape(-1)).backward()
        opt_drep_b.step()
    m_d_rep.eval()
    with torch.no_grad():
        ak2, _ = eval_acc(m_d_rep(x_eval_init).argmax(dim=-1), y_eval_init, d_eval_init)
    m_d_rep.train()
    opt_drep_r = torch.optim.AdamW(m_d_rep.parameters(), lr=4e-3)
    for st in range(rev_steps):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=seed * 3000 + st, device=device)
        opt_drep_r.zero_grad()
        F.cross_entropy(m_d_rep(xb).reshape(-1, 256), yb.reshape(-1)).backward()
        opt_drep_r.step()
    m_d_rep.eval()
    with torch.no_grad():
        ar, aor = eval_acc(m_d_rep(x_eval_rev).argmax(dim=-1), y_eval_rev, d_eval_rev)
    out["Baseline_Dense_1Pass_Replay128"] = {
        "acc_initial": ai,
        "acc_initial_ood": aoi,
        "acc_k2_post_burst": ak2,
        "k1_retention": ak2 / max(ai, 1e-6),
        "acc_k3_post_revised": ar,
        "acc_k3_ood_revised": aor,
        "active_token_fraction": 0.0,
    }

    # 5. Baseline_Dense_1Pass_Upkeeper (Critical Control 1: Same Upkeeper, 1-Pass Architecture)
    torch.manual_seed(seed)
    m_d_upk = DenseTransformerBaseline(d_model=32, num_heads=2, d_ff=128, num_layers=2, chunk_size=1).to(device)
    upk_dense = EvidenceAccumulatingUpkeeper()
    opt_dupk = torch.optim.AdamW(m_d_upk.parameters(), lr=4e-3)
    m_d_upk.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        opt_dupk.zero_grad()
        lg = m_d_upk(xb)
        gy, mask = upk_dense.filter_and_gate_targets(
            xb, yb, lg.argmax(dim=-1), source_id="trusted_source", state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            F.cross_entropy(lg.reshape(-1, 256)[mask.reshape(-1)], gy.reshape(-1)[mask.reshape(-1)]).backward()
            opt_dupk.step()
    m_d_upk.eval()
    with torch.no_grad():
        ai, aoi = eval_acc(m_d_upk(x_eval_init).argmax(dim=-1), y_eval_init, d_eval_init)
    m_d_upk.train()
    opt_dupk_b = torch.optim.AdamW(m_d_upk.parameters(), lr=4e-3)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=seed * 2000 + st, device=device)
        opt_dupk_b.zero_grad()
        lg = m_d_upk(xb)
        gy, mask = upk_dense.filter_and_gate_targets(
            xb, yb, lg.argmax(dim=-1), source_id="noisy_source_1", state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            F.cross_entropy(lg.reshape(-1, 256)[mask.reshape(-1)], gy.reshape(-1)[mask.reshape(-1)]).backward()
            opt_dupk_b.step()
    m_d_upk.eval()
    with torch.no_grad():
        ak2, _ = eval_acc(m_d_upk(x_eval_init).argmax(dim=-1), y_eval_init, d_eval_init)
    m_d_upk.train()
    opt_dupk_r = torch.optim.AdamW(m_d_upk.parameters(), lr=4e-3)
    for st in range(rev_steps):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=seed * 3000 + st, device=device)
        opt_dupk_r.zero_grad()
        lg = m_d_upk(xb)
        gy, mask = upk_dense.filter_and_gate_targets(
            xb, yb, lg.argmax(dim=-1), source_id=sources[st % 3], state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            F.cross_entropy(lg.reshape(-1, 256)[mask.reshape(-1)], gy.reshape(-1)[mask.reshape(-1)]).backward()
            opt_dupk_r.step()
    m_d_upk.eval()
    with torch.no_grad():
        ar, aor = eval_acc(m_d_upk(x_eval_rev).argmax(dim=-1), y_eval_rev, d_eval_rev)
    out["Baseline_Dense_1Pass_Upkeeper"] = {
        "acc_initial": ai,
        "acc_initial_ood": aoi,
        "acc_k2_post_burst": ak2,
        "k1_retention": ak2 / max(ai, 1e-6),
        "acc_k3_post_revised": ar,
        "acc_k3_ood_revised": aor,
        "active_token_fraction": 0.0,
    }

    # 6. Baseline_NonNeural_Dictionary_Counters
    dict_store = NonNeuralDictionaryBaseline(flip_threshold_ratio=1.0)
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        dict_store.update(xb, yb, source_id="trusted_source")
    ai, aoi = eval_acc(dict_store.predict(x_eval_init), y_eval_init, d_eval_init)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=seed * 2000 + st, device=device)
        dict_store.update(xb, yb, source_id="noisy_source_1")
    ak2, _ = eval_acc(dict_store.predict(x_eval_init), y_eval_init, d_eval_init)
    for st in range(rev_steps):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=seed * 3000 + st, device=device)
        dict_store.update(xb, yb, source_id=sources[st % 3])
    ar, aor = eval_acc(dict_store.predict(x_eval_rev), y_eval_rev, d_eval_rev)
    out["Baseline_NonNeural_Dictionary_Counters"] = {
        "acc_initial": ai,
        "acc_initial_ood": aoi,
        "acc_k2_post_burst": ak2,
        "k1_retention": ak2 / max(ai, 1e-6),
        "acc_k3_post_revised": ar,
        "acc_k3_ood_revised": aor,
        "active_token_fraction": 0.0,
    }

    # 7. dMoE_Thinker_NoUpkeeper (Critical Control 2: Same Thinker Architecture, without Upkeeper)
    torch.manual_seed(seed)
    m_tnu = RealSelfStabilizingDMoE(
        d_model=32, num_heads=2, d_ff=128, initial_layers=1, chunk_size=1,
        enable_thinking=True, max_thinking_steps=5, skip_threshold=0.05, theta_halt=0.18,
    ).to(device)
    opt_tnu = torch.optim.AdamW(m_tnu.parameters(), lr=4e-3)
    m_tnu.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        opt_tnu.zero_grad()
        res_t = m_tnu(xb, force_thinking_steps=3)
        m_tnu.thinker.compute_thinking_loss(res_t["thinking_out"], yb)["loss"].backward()
        opt_tnu.step()
    m_tnu.eval()
    with torch.no_grad():
        ai, aoi = eval_acc(m_tnu(x_eval_init)["output"].squeeze(2).argmax(dim=-1), y_eval_init, d_eval_init)
    m_tnu.train()
    opt_tnu_b = torch.optim.AdamW(m_tnu.parameters(), lr=4e-3)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=seed * 2000 + st, device=device)
        opt_tnu_b.zero_grad()
        res_t = m_tnu(xb, force_thinking_steps=3)
        m_tnu.thinker.compute_thinking_loss(res_t["thinking_out"], yb)["loss"].backward()
        opt_tnu_b.step()
    m_tnu.eval()
    with torch.no_grad():
        ak2, _ = eval_acc(m_tnu(x_eval_init)["output"].squeeze(2).argmax(dim=-1), y_eval_init, d_eval_init)
    m_tnu.train()
    opt_tnu_r = torch.optim.AdamW(m_tnu.parameters(), lr=4e-3)
    for st in range(rev_steps):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=seed * 3000 + st, device=device)
        opt_tnu_r.zero_grad()
        res_t = m_tnu(xb, force_thinking_steps=3)
        m_tnu.thinker.compute_thinking_loss(res_t["thinking_out"], yb)["loss"].backward()
        opt_tnu_r.step()
    m_tnu.eval()
    with torch.no_grad():
        res_rev_tnu = m_tnu(x_eval_rev)
        ar, aor = eval_acc(res_rev_tnu["output"].squeeze(2).argmax(dim=-1), y_eval_rev, d_eval_rev)
    out["dMoE_Thinker_NoUpkeeper"] = {
        "acc_initial": ai,
        "acc_initial_ood": aoi,
        "acc_k2_post_burst": ak2,
        "k1_retention": ak2 / max(ai, 1e-6),
        "acc_k3_post_revised": ar,
        "acc_k3_ood_revised": aor,
        "active_token_fraction": float(res_rev_tnu["active_token_fraction"]),
    }

    # 8. Proposed Combined System: dMoE_Thinker_Upkeeper_Adaptive
    torch.manual_seed(seed)
    m_joint = RealSelfStabilizingDMoE(
        d_model=32, num_heads=2, d_ff=128, initial_layers=1, chunk_size=1,
        enable_thinking=True, max_thinking_steps=5, skip_threshold=0.05, theta_halt=0.18,
    ).to(device)
    opt_j = torch.optim.AdamW(m_joint.parameters(), lr=4e-3)
    m_joint.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device)
        opt_j.zero_grad()
        res_j = m_joint(xb, force_thinking_steps=3)
        preds_j = res_j["output"].squeeze(2).argmax(dim=-1)
        gy, mask = m_joint.upkeeper.filter_and_gate_targets(
            xb, yb, preds_j, source_id="trusted_source", state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            m_joint.thinker.compute_thinking_loss(
                res_j["thinking_out"], gy, valid_mask=mask
            )["loss"].backward()
            opt_j.step()

    m_joint.eval()
    with torch.no_grad():
        res_init_j = m_joint(x_eval_init)
        ai, aoi = eval_acc(res_init_j["output"].squeeze(2).argmax(dim=-1), y_eval_init, d_eval_init)

    # Phase 2: K2 single-source contradictory burst
    m_joint.train()
    opt_jb = torch.optim.AdamW(m_joint.parameters(), lr=4e-3)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=seed * 2000 + st, device=device)
        opt_jb.zero_grad()
        res_j = m_joint(xb, force_thinking_steps=3)
        preds_j = res_j["output"].squeeze(2).argmax(dim=-1)
        gy, mask = m_joint.upkeeper.filter_and_gate_targets(
            xb, yb, preds_j, source_id="noisy_source_1", state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            m_joint.thinker.compute_thinking_loss(
                res_j["thinking_out"], gy, valid_mask=mask
            )["loss"].backward()
            opt_jb.step()

    m_joint.eval()
    with torch.no_grad():
        res_k2_j = m_joint(x_eval_init)
        ak2, _ = eval_acc(res_k2_j["output"].squeeze(2).argmax(dim=-1), y_eval_init, d_eval_init)

    # Phase 3: K3 persistent multi-source genuine belief revision
    m_joint.train()
    opt_jr = torch.optim.AdamW(m_joint.parameters(), lr=4e-3)
    for st in range(rev_steps):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=seed * 3000 + st, device=device)
        opt_jr.zero_grad()
        res_j = m_joint(xb, force_thinking_steps=3)
        preds_j = res_j["output"].squeeze(2).argmax(dim=-1)
        gy, mask = m_joint.upkeeper.filter_and_gate_targets(
            xb, yb, preds_j, source_id=sources[st % 3], state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            m_joint.thinker.compute_thinking_loss(
                res_j["thinking_out"], gy, valid_mask=mask
            )["loss"].backward()
            opt_jr.step()

    m_joint.eval()
    with torch.no_grad():
        res_rev_j = m_joint(x_eval_rev)
        ar, aor = eval_acc(res_rev_j["output"].squeeze(2).argmax(dim=-1), y_eval_rev, d_eval_rev)

        # 9. Null_Blind on trained joint model
        x_blind = torch.full_like(x_eval_rev, 32)
        p_blind = m_joint(x_blind)["output"].squeeze(2).argmax(dim=-1)
        ab_i, ab_oi = eval_acc(p_blind, y_eval_init, d_eval_init)
        ab_r, ab_or = eval_acc(p_blind, y_eval_rev, d_eval_rev)
        out["Null_Blind"] = {
            "acc_initial": ab_i,
            "acc_initial_ood": ab_oi,
            "acc_k2_post_burst": ab_i,
            "k1_retention": 1.0,
            "acc_k3_post_revised": ab_r,
            "acc_k3_ood_revised": ab_or,
            "active_token_fraction": 0.0,
        }

    out["dMoE_Thinker_Upkeeper_Adaptive"] = {
        "acc_initial": ai,
        "acc_initial_ood": aoi,
        "acc_k2_post_burst": ak2,
        "k1_retention": ak2 / max(ai, 1e-6),
        "acc_k3_post_revised": ar,
        "acc_k3_ood_revised": aor,
        "active_token_fraction": float(res_rev_j["active_token_fraction"]),
        "tentative_store_bytes": float(m_joint.upkeeper.tentative_store_bytes()),
    }
    return out


def summarize(all_seeds: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    n = len(all_seeds)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond in all_seeds[0]:
        summary[cond] = {}
        for m in all_seeds[0][cond]:
            vals = [s[cond][m] for s in all_seeds]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
            sd = math.sqrt(var)
            ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
            summary[cond][m] = {"mean": mean, "sd": sd, "ci95": ci95}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=220)
    parser.add_argument("--rev-steps", type=int, default=240)
    parser.add_argument("--out", type=str, default="results/dmoe_v4_track_kr_joint/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    all_seeds = []
    for s in range(args.seeds):
        seed_id = 100 + s
        res = run_seed(seed=seed_id, train_steps=args.train_steps, rev_steps=args.rev_steps, device=device)
        all_seeds.append(res)
        j = res["dMoE_Thinker_Upkeeper_Adaptive"]
        tnu = res["dMoE_Thinker_NoUpkeeper"]
        dupk = res["Baseline_Dense_1Pass_Upkeeper"]
        print(
            f"[Seed {s+1:02d}/{args.seeds:02d}] "
            f"Joint(Init={j['acc_initial']:.4f}, K2={j['acc_k2_post_burst']:.4f}, K3={j['acc_k3_post_revised']:.4f}, OOD_K3={j['acc_k3_ood_revised']:.4f}, Act={j['active_token_fraction']:.3f}) | "
            f"NoUpk(K2={tnu['acc_k2_post_burst']:.4f}) | "
            f"Dense1PassUpk(Init={dupk['acc_initial']:.4f}, K3={dupk['acc_k3_post_revised']:.4f})",
            flush=True,
        )

    summary = summarize(all_seeds)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "train_steps": args.train_steps,
            "rev_steps": args.rev_steps,
            "elapsed_sec": time.time() - t0,
        },
        "summary": summary,
        "per_seed": all_seeds,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== TRACK KR JOINT BENCHMARK SUMMARY (20 SEEDS) ===")
    for cond, md in summary.items():
        ai = md["acc_initial"]["mean"]
        ak2 = md["acc_k2_post_burst"]["mean"]
        ak3 = md["acc_k3_post_revised"]["mean"]
        ood3 = md["acc_k3_ood_revised"]["mean"]
        af = md["active_token_fraction"]["mean"]
        print(
            f"{cond:39s} | Init={ai:.4f} | K2_Burst={ak2:.4f} | K3_Rev={ak3:.4f} | K3_OOD(D4-5)={ood3:.4f} | ActiveFrac={af:.3f}"
        )


if __name__ == "__main__":
    main()


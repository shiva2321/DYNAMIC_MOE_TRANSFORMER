"""
Verification Benchmark for Surprise-Driven Dynamic Latent Thinking (Phase A2 / Track R).

Evaluates:
1. Null Controls (Mandatory):
   - Null_Untrained: Randomly initialized RealSelfStabilizingDMoE with thinking enabled.
   - Null_Blind: Trained with inputs zeroed out at test time.
   - Null_Shuffled: Trained with shuffled target labels.
2. Baselines:
   - Baseline_1Pass_NextToken: Standard 1-pass RealSelfStabilizingDMoE (enable_thinking=False).
   - Baseline_Unanchored_NoDeepSup: Thinking loop trained only on final output without
     step-wise deep supervision or surprise calibration (analogous to old Th0 loop).
3. Proposed Algorithm (SurpriseDrivenLatentThinker in RealSelfStabilizingDMoE):
   - dMoE_Thinker_Step0: Evaluated at k=0 steps (fast pass only).
   - dMoE_Thinker_Full5: Evaluated at k=5 steps (trained only on depths D <= 3 with k=3,
     tested on D in 0..5 including zero-shot OOD depths D = 4, 5).
   - dMoE_Thinker_Adaptive: Dynamic surprise-gated per-token halting (theta_halt=0.18),
     measuring both accuracy and active token compute fraction across depths D = 0..5.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dmoe.real_model import RealSelfStabilizingDMoE


def build_transition_table(seed: int = 99) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(8, generator=gen) for _ in range(8)])


def make_chain_batch(
    trans: torch.Tensor,
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
        cur_s = torch.where(do_reset, new_s, trans[op, cur_s])
        cur_d = torch.where(do_reset, torch.zeros_like(cur_d), cur_d + 1)
        y[:, t] = cur_s
        depths[:, t] = cur_d
    return x.to(device) + 32, y.to(device) + 32, depths.to(device)


def evaluate_predictions(
    preds: torch.Tensor, y_test: torch.Tensor, d_test: torch.Tensor
) -> Dict[str, float]:
    overall = (preds == y_test).float().mean().item()
    id_mask = d_test <= 3
    ood_mask = d_test >= 4
    id_acc = (preds[id_mask] == y_test[id_mask]).float().mean().item() if id_mask.any() else 0.0
    ood_acc = (preds[ood_mask] == y_test[ood_mask]).float().mean().item() if ood_mask.any() else 0.0
    res = {
        "overall_acc": overall,
        "id_acc_d0_3": id_acc,
        "ood_acc_d4_5": ood_acc,
    }
    for d in range(6):
        m = d_test == d
        res[f"acc_d{d}"] = (preds[m] == y_test[m]).float().mean().item() if m.any() else 0.0
    return res


def run_single_seed(
    seed: int,
    train_steps: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    trans = build_transition_table(seed=99)
    x_test, y_test, d_test = make_chain_batch(
        trans, batch_size=256, seq_len=16, max_depth=5, seed=90000 + seed, device=device
    )

    results: Dict[str, Dict[str, float]] = {}

    # 1. Null_Untrained
    torch.manual_seed(seed)
    m_untrained = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        theta_halt=0.18,
    ).to(device)
    m_untrained.eval()
    with torch.no_grad():
        out_u = m_untrained(x_test)
        preds_u = out_u["output"].squeeze(2).argmax(dim=-1)
        results["Null_Untrained"] = evaluate_predictions(preds_u, y_test, d_test)

    # 2. Null_Shuffled
    torch.manual_seed(seed)
    m_shuf = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        theta_halt=0.18,
    ).to(device)
    opt_shuf = torch.optim.AdamW(m_shuf.parameters(), lr=4e-3)
    m_shuf.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(
            trans, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device
        )
        # Randomly permute labels in [32..39]
        yb_shuf = torch.randint(32, 40, yb.shape, device=device)
        opt_shuf.zero_grad()
        out = m_shuf(xb, force_thinking_steps=3)
        loss_dict = m_shuf.thinker.compute_thinking_loss(out["thinking_out"], yb_shuf)
        loss_dict["loss"].backward()
        opt_shuf.step()
    m_shuf.eval()
    with torch.no_grad():
        out_s = m_shuf(x_test)
        preds_s = out_s["output"].squeeze(2).argmax(dim=-1)
        results["Null_Shuffled"] = evaluate_predictions(preds_s, y_test, d_test)

    # 3. Baseline_1Pass_NextToken (enable_thinking=False)
    torch.manual_seed(seed)
    m_1pass = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=False,
    ).to(device)
    opt_1pass = torch.optim.AdamW(m_1pass.parameters(), lr=4e-3)
    m_1pass.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(
            trans, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device
        )
        opt_1pass.zero_grad()
        out = m_1pass(xb)
        loss = F.cross_entropy(out["output"].reshape(-1, 256), yb.reshape(-1))
        loss.backward()
        opt_1pass.step()
    m_1pass.eval()
    with torch.no_grad():
        out_1p = m_1pass(x_test)
        preds_1p = out_1p["output"].squeeze(2).argmax(dim=-1)
        r_1p = evaluate_predictions(preds_1p, y_test, d_test)
        r_1p["active_token_fraction"] = 0.0
        results["Baseline_1Pass_NextToken"] = r_1p

    # 4. Baseline_Th0_UntrainedLoop (trained only on step-0 CE, evaluated at 5 steps)
    torch.manual_seed(seed)
    m_th0 = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        theta_halt=-1.0,
        skip_threshold=-1.0,
    ).to(device)
    opt_th0 = torch.optim.AdamW(m_th0.parameters(), lr=4e-3)
    m_th0.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(
            trans, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device
        )
        opt_th0.zero_grad()
        out = m_th0(xb, force_thinking_steps=0)
        loss = F.cross_entropy(out["output"].reshape(-1, 256), yb.reshape(-1))
        loss.backward()
        opt_th0.step()
    m_th0.eval()
    with torch.no_grad():
        out_th0 = m_th0(x_test, force_thinking_steps=5)
        preds_th0 = out_th0["output"].squeeze(2).argmax(dim=-1)
        r_th0 = evaluate_predictions(preds_th0, y_test, d_test)
        r_th0["active_token_fraction"] = 1.0
        results["Baseline_Th0_UntrainedLoop"] = r_th0

    # 5. Proposed Surprise-Driven Latent Thinker (trained with compute_thinking_loss on D<=3, k=3)
    torch.manual_seed(seed)
    m_think = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        skip_threshold=0.05,
        theta_halt=0.18,
    ).to(device)
    opt_think = torch.optim.AdamW(m_think.parameters(), lr=4e-3)
    m_think.train()
    for st in range(train_steps):
        xb, yb, _ = make_chain_batch(
            trans, 64, 16, max_depth=3, seed=seed * 1000 + st, device=device
        )
        opt_think.zero_grad()
        out = m_think(xb, force_thinking_steps=3)
        loss_dict = m_think.thinker.compute_thinking_loss(out["thinking_out"], yb)
        loss_dict["loss"].backward()
        opt_think.step()

    m_think.eval()
    with torch.no_grad():
        # Null_Blind (constant token 32 input)
        x_blind = torch.full_like(x_test, 32)
        out_blind = m_think(x_blind)
        preds_blind = out_blind["output"].squeeze(2).argmax(dim=-1)
        results["Null_Blind"] = evaluate_predictions(preds_blind, y_test, d_test)

        # Full 5-step rollout (gives k=0..5 trajectories)
        out_full = m_think(x_test, force_thinking_steps=5)
        for k, lg in enumerate(out_full["step_logits"]):
            preds_k = lg.squeeze(2).argmax(dim=-1)
            rk = evaluate_predictions(preds_k, y_test, d_test)
            rk["active_token_fraction"] = float(k) / 5.0
            results[f"dMoE_Thinker_Step{k}"] = rk

        # Adaptive surprise-gated rollout (theta_halt=0.18)
        out_adapt = m_think(x_test)
        preds_adapt = out_adapt["output"].squeeze(2).argmax(dim=-1)
        r_adapt = evaluate_predictions(preds_adapt, y_test, d_test)
        r_adapt["active_token_fraction"] = float(out_adapt["active_token_fraction"])
        tsteps = out_adapt["token_steps"]
        for d in range(6):
            m = d_test == d
            r_adapt[f"steps_d{d}"] = tsteps[m].mean().item() if m.any() else 0.0
        results["dMoE_Thinker_Adaptive"] = r_adapt

    return results


def summarize(all_seed_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    n = len(all_seed_results)
    systems = list(all_seed_results[0].keys())
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sys_name in systems:
        metrics = list(all_seed_results[0][sys_name].keys())
        summary[sys_name] = {}
        for m in metrics:
            vals = [r[sys_name][m] for r in all_seed_results]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
            sd = math.sqrt(var)
            ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
            summary[sys_name][m] = {
                "mean": mean,
                "sd": sd,
                "ci95": ci95,
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=220)
    parser.add_argument("--out", type=str, default="results/dmoe_v4_surprise_thinking/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    all_seed_results = []
    for s in range(args.seeds):
        seed_id = 100 + s
        res = run_single_seed(seed=seed_id, train_steps=args.train_steps, device=device)
        all_seed_results.append(res)
        print(
            f"[Seed {s+1:02d}/{args.seeds:02d}] "
            f"1Pass={res['Baseline_1Pass_NextToken']['overall_acc']:.4f} | "
            f"Th0={res['Baseline_Th0_UntrainedLoop']['overall_acc']:.4f} | "
            f"Step0={res['dMoE_Thinker_Step0']['overall_acc']:.4f} -> "
            f"Step3={res['dMoE_Thinker_Step3']['overall_acc']:.4f} -> "
            f"Step5={res['dMoE_Thinker_Step5']['overall_acc']:.4f} (OOD={res['dMoE_Thinker_Step5']['ood_acc_d4_5']:.4f}) | "
            f"Adapt={res['dMoE_Thinker_Adaptive']['overall_acc']:.4f} "
            f"(ActiveFrac={res['dMoE_Thinker_Adaptive']['active_token_fraction']:.3f})",
            flush=True,
        )

    summary = summarize(all_seed_results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "train_steps": args.train_steps,
            "train_max_depth": 3,
            "train_thinking_steps": 3,
            "test_max_depth": 5,
            "test_max_thinking_steps": 5,
            "theta_halt": 0.18,
            "elapsed_sec": time.time() - t0,
        },
        "summary": summary,
        "per_seed": all_seed_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== SUMMARY ACROSS SEEDS ===")
    for sys_name, m_dict in summary.items():
        ov = m_dict["overall_acc"]
        id_a = m_dict["id_acc_d0_3"]
        ood_a = m_dict["ood_acc_d4_5"]
        af = m_dict.get("active_token_fraction", {"mean": 0.0})["mean"]
        print(
            f"{sys_name:28s} | Overall: {ov['mean']:.4f} +/- {ov['sd']:.4f} "
            f"| ID(D0-3): {id_a['mean']:.4f} | OOD(D4-5): {ood_a['mean']:.4f} | ActiveFrac: {af:.3f}"
        )


if __name__ == "__main__":
    main()

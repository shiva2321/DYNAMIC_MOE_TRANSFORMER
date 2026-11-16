"""
10-Seed End-to-End Evaluation of UnifiedNoveltyDetector on Trained RealSelfStabilizingDMoE.

Trains a RealSelfStabilizingDMoE (enable_thinking=True) on a base domain (tokens 32..39,
16-token sequences) and calibrates UnifiedNoveltyDetector on in-distribution batches.
Then evaluates Detection Rate & False-Positive Rate across 4 stream types:
1. In-Distribution (must NOT flag: FPR)
2. Coarse Domain Shift (tokens 120..127)
3. Near-Twin Rule Shift (same tokens 32..39, flipped rule -> higher post-thinking surprise & NLL)
4. Local 2-Token Nuance Flip (14/16 tokens familiar in-distribution, 2 tokens [12, 13] flipped/novel)

Compares:
- Old Mean-Only Detector (seq_dist >= 0.45 or mean_surprise >= 0.42)
- New UnifiedNoveltyDetector (coarse + near-twin + local token/tail nuance)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dmoe.novelty import UnifiedNoveltyDetector
from dmoe.real_model import RealSelfStabilizingDMoE


def make_rule_batch(offset: int, shift: int, batch_size: int, seq_len: int, seed: int, device: torch.device):
    g = torch.Generator().manual_seed(seed)
    xb = torch.randint(offset, offset + 8, (batch_size, seq_len), generator=g)
    yb = ((xb - offset + shift) % 8) + offset
    return xb.to(device), yb.to(device)


def run_seed(seed: int, train_steps: int, device: torch.device) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    model = RealSelfStabilizingDMoE(
        d_model=24,
        num_heads=2,
        d_ff=64,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # 1. Train on base domain (tokens 32..39, shift +1)
    model.train()
    for st in range(train_steps):
        xb, yb = make_rule_batch(32, 1, 32, 16, seed=seed * 1000 + st, device=device)
        opt.zero_grad()
        out = model(xb, force_thinking_steps=2)
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
        loss.backward()
        opt.step()

    # 2. Calibrate UnifiedNoveltyDetector on 20 in-distribution batches
    model.eval()
    det = UnifiedNoveltyDetector(d_model=24, tail_fraction=0.15)
    with torch.no_grad():
        for st in range(20):
            xb, yb = make_rule_batch(32, 1, 16, 16, seed=seed * 2000 + st, device=device)
            out = model(xb, force_thinking_steps=2)
            seq_sig = F.normalize(model.encode(xb).mean(dim=1), p=2, dim=-1)
            tok_surp = out["surprise_trace"][-1]
            tok_nll = F.cross_entropy(
                out["output"].reshape(-1, 256), yb.reshape(-1), reduction="none"
            ).view(xb.size(0), xb.size(1))
            ex = model.layers[0].moe.experts[0].exemplars
            det.evaluate(
                expert_idx=0,
                seq_sig=seq_sig,
                expert_centroid=ex if ex.numel() > 0 else None,
                token_latents=out["anchor_latent"],
                token_surprise=tok_surp,
                token_nll=tok_nll,
                expert_is_converged=True,
                update_baseline=True,
            )

    # 3. Evaluate on 10 batches per stream type
    streams = ("in_dist", "coarse", "near_twin", "local_nuance")
    old_hits = {s: 0 for s in streams}
    new_hits = {s: 0 for s in streams}
    n_eval = 10

    with torch.no_grad():
        ex = model.layers[0].moe.experts[0].exemplars
        cent = F.normalize(ex.mean(dim=0, keepdim=True).float(), p=2, dim=-1).to(device)

        for st in range(n_eval):
            for s_name in streams:
                if s_name == "in_dist":
                    xb, yb = make_rule_batch(32, 1, 16, 16, seed=seed * 3000 + st, device=device)
                elif s_name == "coarse":
                    xb, yb = make_rule_batch(120, 3, 16, 16, seed=seed * 4000 + st, device=device)
                elif s_name == "near_twin":
                    xb, yb = make_rule_batch(32, 5, 16, 16, seed=seed * 5000 + st, device=device)
                else:
                    # Local 2-token nuance: 14/16 tokens identical to in_dist, 2 tokens [12, 13] flipped
                    xb, yb = make_rule_batch(32, 1, 16, 16, seed=seed * 6000 + st, device=device)
                    xb_n, yb_n = make_rule_batch(120, 3, 16, 2, seed=seed * 7000 + st, device=device)
                    xb[:, 12:14] = xb_n
                    yb[:, 12:14] = yb_n

                out = model(xb, force_thinking_steps=2)
                seq_sig = F.normalize(model.encode(xb).mean(dim=1), p=2, dim=-1)
                tok_surp = out["surprise_trace"][-1]
                tok_nll = F.cross_entropy(
                    out["output"].reshape(-1, 256), yb.reshape(-1), reduction="none"
                ).view(xb.size(0), xb.size(1))

                # Old Mean-Only detector
                seq_dist = float((1.0 - (seq_sig.float() @ cent.T).squeeze(-1)).mean().item())
                mean_surp = float(tok_surp.mean().item())
                if seq_dist >= 0.45 or mean_surp >= 0.42:
                    old_hits[s_name] += 1

                # New UnifiedNoveltyDetector
                verdict = det.evaluate(
                    expert_idx=0,
                    seq_sig=seq_sig,
                    expert_centroid=ex,
                    token_latents=out["anchor_latent"],
                    token_surprise=tok_surp,
                    token_nll=tok_nll,
                    expert_is_converged=True,
                    update_baseline=False,
                )
                if verdict.is_novel:
                    new_hits[s_name] += 1

    return {
        "Old_MeanOnly": {s: old_hits[s] / n_eval for s in streams},
        "New_Unified": {s: new_hits[s] / n_eval for s in streams},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--out", type=str, default="results/dmoe_v4_unified_novelty/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    all_seeds: List[Dict[str, Dict[str, float]]] = []
    for s in range(args.seeds):
        res = run_seed(seed=100 + s, train_steps=args.train_steps, device=device)
        all_seeds.append(res)
        o = res["Old_MeanOnly"]
        n = res["New_Unified"]
        print(
            f"[Seed {s+1:02d}/{args.seeds:02d}] "
            f"Old(FPR={o['in_dist']:.2f}, Coarse={o['coarse']:.2f}, Twin={o['near_twin']:.2f}, Nuance={o['local_nuance']:.2f}) | "
            f"New(FPR={n['in_dist']:.2f}, Coarse={n['coarse']:.2f}, Twin={n['near_twin']:.2f}, Nuance={n['local_nuance']:.2f})",
            flush=True,
        )

    summary = {}
    for cond in ("Old_MeanOnly", "New_Unified"):
        summary[cond] = {}
        for m in ("in_dist", "coarse", "near_twin", "local_nuance"):
            vals = [r[cond][m] for r in all_seeds]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            sd = math.sqrt(var)
            summary[cond][m] = {"mean": mean, "sd": sd}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"elapsed_sec": time.time() - t0, "summary": summary, "per_seed": all_seeds}, f, indent=2)

    print("\n=== UNIFIED NOVELTY DETECTION SUMMARY (10 SEEDS) ===")
    for cond, md in summary.items():
        print(
            f"{cond:14s} | FPR(InDist)={md['in_dist']['mean']:.3f}+/-{md['in_dist']['sd']:.3f} "
            f"| Coarse={md['coarse']['mean']:.3f} | NearTwin={md['near_twin']['mean']:.3f} "
            f"| LocalNuance={md['local_nuance']['mean']:.3f}+/-{md['local_nuance']['sd']:.3f}"
        )


if __name__ == "__main__":
    main()

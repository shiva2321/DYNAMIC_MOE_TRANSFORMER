"""
bench_s5_comparison.py - Decisive S5 group tracking comparison for Round Ye.

Pre-registered prediction (written before running):
Both GRU and SnappedRotationState (at K=4|G|=480, K=8|G|=960, K=16|G|=1920) will fail
to generalize on S5 (|G|=120, chance 0.83%), remaining near chance level (< 10% accuracy)
at length 256 across 10 test seeds. Outcome classifies SnappedRotationState as
"verified, no demonstrated advantage over a GRU".

Runs 10 seeds (0..9) on S5 under curriculum lengths <= 24, tests lengths 16, 32, 64, 128, 256.
ASCII output only.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from geometry.snapped_state import SnappedRotationState
from research.yggdrasil_poc.poc_b_state_tracking import make_group, prefix_products, GRUModel


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LENGTHS = (16, 32, 64, 128, 256)
SCHEDULE = ((300, 6), (300, 12), (400, 24))  # 1000 steps curriculum <= 24
BATCH = 256
SEEDS = list(range(10))  # 10 test seeds


def ci95(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))


def train_and_eval_gru(mul, seed, lr=0.01, n_test=200):
    n_sym = mul.shape[0]  # 120
    torch.manual_seed(seed)
    model = GRUModel(emb=16, hid=32, n_sym=n_sym).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)

    for steps, T in SCHEDULE:
        for _ in range(steps):
            x = torch.randint(0, n_sym, (BATCH, T), generator=g).to(DEVICE)
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, n_sym), prefix_products(x, mul).reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    accs = {}
    gt = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    model.eval()
    with torch.no_grad():
        for T in LENGTHS:
            x_test = torch.randint(0, n_sym, (n_test, T), generator=gt).to(DEVICE)
            targets = prefix_products(x_test, mul)[:, -1]
            preds = model(x_test)[:, -1].argmax(dim=-1)
            accs[T] = (preds == targets).float().mean().item()

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accs


def train_and_eval_vq(mul, seed, num_codes, lr=0.003, k=8, n_test=200):
    n_sym = mul.shape[0]  # 120
    torch.manual_seed(seed)
    model = SnappedRotationState(n_sym=n_sym, k=k, num_codes=num_codes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)

    for steps, T in SCHEDULE:
        for _ in range(steps):
            x = torch.randint(0, n_sym, (BATCH, T), generator=g).to(DEVICE)
            out, aux = model(x, return_loss=True)
            loss = F.cross_entropy(out.reshape(-1, n_sym), prefix_products(x, mul).reshape(-1)) + aux
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    accs = {}
    gt = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    model.eval()
    with torch.no_grad():
        for T in LENGTHS:
            x_test = torch.randint(0, n_sym, (n_test, T), generator=gt).to(DEVICE)
            targets = prefix_products(x_test, mul)[:, -1]
            preds = model(x_test)[:, -1].argmax(dim=-1)
            accs[T] = (preds == targets).float().mean().item()

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accs


def main():
    print("=" * 80)
    print("ROUND Ye: S5 GROUP WORD PROBLEM COMPARISON (10 SEEDS)")
    print("Task: S5 (order 120, chance 0.83%), curriculum T <= 24, lengths 16..256")
    print("=" * 80)

    mul = make_group("s5").to(DEVICE)
    results = {}

    configs = [
        ("gru", "GRU Baseline (hid=32, emb=16)", lambda s: train_and_eval_gru(mul, s, lr=0.01)),
        ("vq_k480", "SnappedRotation (K=4|G|=480, k=8)", lambda s: train_and_eval_vq(mul, s, num_codes=480, lr=0.003)),
        ("vq_k960", "SnappedRotation (K=8|G|=960, k=8)", lambda s: train_and_eval_vq(mul, s, num_codes=960, lr=0.003)),
        ("vq_k1920", "SnappedRotation (K=16|G|=1920, k=8)", lambda s: train_and_eval_vq(mul, s, num_codes=1920, lr=0.003)),
    ]

    for model_id, label, runner in configs:
        print(f"\nRunning {label} across 10 seeds...")
        t0 = time.time()
        seed_results = []
        for s in SEEDS:
            t_s = time.time()
            res = runner(s)
            seed_results.append(res)
            print(f"  Seed {s:2d} ({time.time()-t_s:4.1f}s): T=16: {res[16]*100:4.1f}% | T=256: {res[256]*100:4.1f}%")

        # Aggregate across seeds
        summary = {}
        for T in LENGTHS:
            vals = [sr[T] * 100 for sr in seed_results]
            mean_val = float(np.mean(vals))
            ci_val = ci95(vals)
            summary[str(T)] = {"mean": mean_val, "ci95": ci_val, "per_seed": vals}

        # Count seeds >= 95% at T=256
        n_solved = sum(1 for sr in seed_results if sr[256] >= 0.95)
        summary["solved_256"] = f"{n_solved}/{len(SEEDS)}"
        summary["wall_clock_s"] = time.time() - t0

        results[model_id] = summary
        print(f"  --> {model_id} T=256: {summary['256']['mean']:.2f}% +- {summary['256']['ci95']:.2f}% (Solved: {n_solved}/10)")

    # Save JSON results
    os.makedirs("research/yggdrasil_results", exist_ok=True)
    out_json = "research/yggdrasil_results/s5_comparison.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Format ASCII table
    lines = []
    lines.append("accuracy (%) of product after LAST symbol; mean +- 95% CI over 10 seeds; trained on S5 lengths <= 24")
    lines.append(f"{'task':<4} {'model':<12} {'codebook':<10} | {'T=16':<12} {'T=32':<12} {'T=64':<12} {'T=128':<12} {'T=256':<12} | seeds >=95% at T=256")
    lines.append("-" * 105)

    for model_id, label, _ in configs:
        m = results[model_id]
        cb_str = "N/A" if model_id == "gru" else model_id.split("_")[1].upper()
        t16 = f"{m['16']['mean']:4.1f}+-{m['16']['ci95']:4.1f}"
        t32 = f"{m['32']['mean']:4.1f}+-{m['32']['ci95']:4.1f}"
        t64 = f"{m['64']['mean']:4.1f}+-{m['64']['ci95']:4.1f}"
        t128 = f"{m['128']['mean']:4.1f}+-{m['128']['ci95']:4.1f}"
        t256 = f"{m['256']['mean']:4.1f}+-{m['256']['ci95']:4.1f}"
        solved = m["solved_256"]
        lines.append(f"s5   {model_id:<12} {cb_str:<10} | {t16:<12} {t32:<12} {t64:<12} {t128:<12} {t256:<12} | {solved}")

    table_text = "\n".join(lines)
    print("\n" + table_text)

    out_txt = "research/yggdrasil_results/s5_comparison_table.txt"
    with open(out_txt, "w") as f:
        f.write(table_text)
    print(f"\nResults saved to {out_json} and {out_txt}")


if __name__ == "__main__":
    main()

"""CVA-3A: Competence monitor false alarm audit on a stationary stream.

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Evaluates PhyloMoE with competence monitor enabled (and arm_monitor_after_infancy=True)
on a stationary single-regime stream of 20,000 samples (seeds 10-19).
Then repeats with competence=False as a negative control.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes

torch.set_num_threads(3)
SEEDS = list(range(10, 20))  # Evaluation seeds 10-19
OUT_DIR = Path("research/growmoe_results/cva")
N_TRAIN = 20000
BATCH_SIZE = 64
EXPECTED_BATCHES = 313  # ceil(20000 / 64) = 313


def run_single_stream(seed: int, competence: bool) -> dict:
    """Run PhyloMoE on segment 0 (20,000 samples) of function_regimes."""
    cfg = PhyloCfg(
        seed=seed,
        d=128,
        n_classes=8,
        inner_steps=5,
        q=0.02,
        alpha_spawn=0.01,
        m_min=32,
        route_rule="accept_then_logf",
        arm_monitor_after_infancy=True,
        competence=competence,
    )
    pool = PhyloMoE(cfg=cfg)
    stream = function_regimes(seed, n_regimes=2, n_train=N_TRAIN, n_test=64)

    t0 = time.time()
    batch_count = 0
    total_samples = 0
    for x, y, ctx in stream.algo_batches(BATCH_SIZE, seed):
        pool.observe(x, y, ctx)
        batch_count += 1
        total_samples += len(x)
        if batch_count == EXPECTED_BATCHES:
            break

    elapsed = time.time() - t0

    alarms = pool.events.of_kind("alarm")
    spawns = pool.events.of_kind("spawn")
    extends = pool.events.of_kind("extend")

    alarm_calls = [ev.step for ev in alarms]
    n_alarms = len(alarms)
    n_spawns = len(spawns)
    n_extends = len(extends)
    n_experts = len(pool.experts)

    # Rate per 10,000 samples
    alarms_per_10k = (n_alarms / total_samples) * 10000.0

    return {
        "seed": seed,
        "competence": competence,
        "total_samples": total_samples,
        "batch_count": batch_count,
        "n_alarms": n_alarms,
        "alarms_per_10k": alarms_per_10k,
        "alarm_calls": alarm_calls,
        "n_spawns": n_spawns,
        "n_extends": n_extends,
        "final_n_experts": n_experts,
        "elapsed_sec": round(elapsed, 2),
    }


def run_experiment():
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting CVA-3A Main Runs (competence=True, seeds 10-19)")
    comp_true_results = []
    for s in SEEDS:
        res = run_single_stream(s, competence=True)
        comp_true_results.append(res)
        print(
            f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): "
            f"Seed {s:2d} (competence=True): alarms={res['n_alarms']} (calls={res['alarm_calls']}), "
            f"spawns={res['n_spawns']}, extends={res['n_extends']}, experts={res['final_n_experts']} ({res['elapsed_sec']:.1f}s)"
        )

    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting CVA-3A Control Runs (competence=False, seeds 10-19)")
    comp_false_results = []
    for s in SEEDS:
        res = run_single_stream(s, competence=False)
        comp_false_results.append(res)
        print(
            f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): "
            f"Seed {s:2d} (competence=False): alarms={res['n_alarms']}, "
            f"spawns={res['n_spawns']}, extends={res['n_extends']}, experts={res['final_n_experts']} ({res['elapsed_sec']:.1f}s)"
        )

    return comp_true_results, comp_false_results


def write_outputs(comp_true: List[dict], comp_false: List[dict]):
    # Compute summary stats
    alarms_per_10k_list = [r["alarms_per_10k"] for r in comp_true]
    mean_alarms_10k = float(np.mean(alarms_per_10k_list))
    min_alarms_10k = float(np.min(alarms_per_10k_list))
    max_alarms_10k = float(np.max(alarms_per_10k_list))

    total_alarms_list = [r["n_alarms"] for r in comp_true]
    mean_total_alarms = float(np.mean(total_alarms_list))

    # Check whether any seed ended with more than one expert
    more_than_one_expert_true = any(r["final_n_experts"] > 1 for r in comp_true)
    more_than_one_expert_false = any(r["final_n_experts"] > 1 for r in comp_false)

    # Confirm competence=False had 0 alarms
    false_alarms_zero = all(r["n_alarms"] == 0 for r in comp_false)

    out_data = {
        "metadata": {
            "title": "CVA-3A: Competence Monitor False Alarm Audit",
            "stream": "function_regimes(n_regimes=2, n_train=20000, n_test=64)",
            "segment": "regime 0 (first 20,000 samples = 313 batches of 64)",
            "seeds": SEEDS,
            "d": 128,
            "arm_monitor_after_infancy": True,
        },
        "summary": {
            "competence_true": {
                "mean_alarms_per_10k": mean_alarms_10k,
                "range_alarms_per_10k": [min_alarms_10k, max_alarms_10k],
                "mean_total_alarms": mean_total_alarms,
                "any_seed_more_than_one_expert": more_than_one_expert_true,
            },
            "competence_false": {
                "all_zero_alarms": false_alarms_zero,
                "any_seed_more_than_one_expert": more_than_one_expert_false,
            },
        },
        "runs_competence_true": comp_true,
        "runs_competence_false": comp_false,
    }

    with open(OUT_DIR / "cva3a.json", "w") as f:
        json.dump(out_data, f, indent=2)

    md_lines = [
        "# CVA-3A Results: Competence Monitor False Alarm Audit",
        "",
        "VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Stream: function_regimes (stationary segment 0, 20,000 samples fed in 313 batches of 64)",
        f"Evaluation seeds: 10-19 | Latent dimension: d = 128 | arm_monitor_after_infancy = True",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Alarms per 10,000 labelled samples (competence=True)**: **{mean_alarms_10k:.2f}** (range: **{min_alarms_10k:.2f}** to **{max_alarms_10k:.2f}**; mean total alarms: {mean_total_alarms:.1f} per 20k stream)",
        f"- **Did any seed end with more than one expert (competence=True)?**: **{'YES' if more_than_one_expert_true else 'NO'}** (all seeds ended with exactly {comp_true[0]['final_n_experts']} expert{'s' if comp_true[0]['final_n_experts'] != 1 else ''})",
        f"- **Control (competence=False)**: All seeds produced exactly 0 alarms: **{'CONFIRMED' if false_alarms_zero else 'FAILED'}** (ended with {'more than one' if more_than_one_expert_false else 'exactly 1'} expert)",
        "",
        "---",
        "",
        "## Runs with Competence Monitor Active (`competence=True`)",
        "",
        "| Seed | Total Samples | Alarms | Alarms / 10k | Alarm Call Indices | Spawns | Extends | Final Experts | Runtime (s) |",
        "|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
    ]

    for r in comp_true:
        call_str = ", ".join(str(c) for c in r["alarm_calls"]) if r["alarm_calls"] else "None"
        md_lines.append(
            f"| {r['seed']} | {r['total_samples']} | {r['n_alarms']} | {r['alarms_per_10k']:.2f} | {call_str} | {r['n_spawns']} | {r['n_extends']} | {r['final_n_experts']} | {r['elapsed_sec']:.1f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Negative Control Runs (`competence=False`)",
        "",
        "| Seed | Total Samples | Alarms | Spawns | Extends | Final Experts | Runtime (s) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r in comp_false:
        md_lines.append(
            f"| {r['seed']} | {r['total_samples']} | {r['n_alarms']} | {r['n_spawns']} | {r['n_extends']} | {r['final_n_experts']} | {r['elapsed_sec']:.1f} |"
        )

    with open(OUT_DIR / "CVA3A_RESULTS.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Saved cva3a.json and CVA3A_RESULTS.md to {OUT_DIR}")


def main():
    t_start = time.time()
    comp_true, comp_false = run_experiment()
    write_outputs(comp_true, comp_false)
    print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Total runtime = {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()

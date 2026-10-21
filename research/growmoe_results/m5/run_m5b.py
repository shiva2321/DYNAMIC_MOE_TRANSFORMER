"""DEV, exploratory, not pre-registered; tuned and scored on the same dev seeds, so optimistic for the tuned arms

M5-B: Banking77 continual learning benchmark on dev seeds 100, 101, 102.
Tuned arms: A_today, B2, B4, B1_128 (grid=(1, 5, 20, 50), factor=2.5, cap=200).
Lookup baseline: kNN_all (steps=1).
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from growmoe.config import PhyloCfg
from growmoe.fairfight import run_fast, tune_steps
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_real import banking_incremental
from research.growmoe_results.m0.run_m0 import HybridFrontend, factory

LABEL = "DEV, exploratory, not pre-registered; tuned and scored on the same dev seeds, so optimistic for the tuned arms"
SEEDS = (100, 101, 102)
TUNED_ARMS = ("A_today", "B2", "B4", "B1_128")
ALL_ARMS = ("A_today", "B2", "B4", "B1_128", "kNN_all")


def make_arm(arm_name: str, st, seed: int, steps: int):
    if arm_name == "A_today":
        cfg = PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe",
            d=st.d,
            n_classes=st.n_classes,
            seed=seed,
            inner_steps=steps,
            route_rule="accept_then_logf",
            q=0.02,
            alpha_spawn=0.01,
            m_min=32,
        )
        return lambda: PhyloMoE(cfg)
    elif arm_name in ("B2", "B4", "B1_128"):
        return factory(arm_name, st, seed, {"steps": steps})
    elif arm_name == "kNN_all":
        return factory("kNN_all", st, seed, {"steps": 1})
    raise ValueError(f"Unknown arm {arm_name}")


def main():
    print("=" * 90)
    print(LABEL)
    print("M5-B: Banking77 Incremental Benchmark (Dev Seeds 100, 101, 102)")
    print("=" * 90)

    t_start = time.perf_counter()

    # 1. Frontend & Streams
    print("\n[1/5] Initializing HybridFrontend(d=128, seed=0) and streams...")
    fe = HybridFrontend(d=128, seed=0)
    streams = {sd: banking_incremental(seed=sd, frontend=fe) for sd in SEEDS}

    regime_intents = {}
    for sd in SEEDS:
        st = streams[sd]
        regime_intents[sd] = {r: len(st.regimes[r].meta["intents"]) for r in st.regimes}

    # 2. Tuning for A_today, B2, B4, B1_128
    print("\n[2/5] Tuning arms on dev seeds (grid=(1, 5, 20, 50), factor=2.5, cap=200)...")
    tuning_results = {}
    tuned_steps = {}

    for arm in TUNED_ARMS:
        print(f"  Tuning {arm}...")
        t_arm0 = time.perf_counter()

        def score_fn(steps: int, arm_name=arm) -> float:
            accs = []
            for sd in SEEDS:
                st = streams[sd]
                pool_fn = make_arm(arm_name, st, sd, steps)
                res = run_fast(pool_fn, st, seed=sd, eval_every="final")
                accs.append(res.acc)
            mean_acc = float(np.mean(accs))
            print(f"    {arm_name} @ steps={steps:<3d} -> dev mean acc: {mean_acc * 100:6.2f}%")
            return mean_acc

        res_tune = tune_steps(score_fn, grid=(1, 5, 20, 50), factor=2.5, cap=200)
        tuned_steps[arm] = res_tune.best
        tuning_results[arm] = {
            "best_steps": res_tune.best,
            "at_edge": res_tune.at_edge,
            "table": {int(k): float(v) for k, v in res_tune.table.items()},
            "seconds": time.perf_counter() - t_arm0,
        }
        print(f"  -> {arm} tuned to steps={res_tune.best} (at_edge={res_tune.at_edge}) in {tuning_results[arm]['seconds']:.1f}s")

    tuned_steps["kNN_all"] = 1

    # 3. Final Runs at Tuned Steps
    print("\n[3/5] Running final evaluations at tuned steps...")
    final_runs = {arm: {} for arm in ALL_ARMS}

    for arm in ALL_ARMS:
        steps = tuned_steps[arm]
        print(f"  Evaluating {arm} (steps={steps})...")
        for sd in SEEDS:
            st = streams[sd]
            t0 = time.perf_counter()
            pool_fn = make_arm(arm, st, sd, steps)
            res = run_fast(pool_fn, st, seed=sd, eval_every="final")
            elapsed = time.perf_counter() - t0

            final_runs[arm][sd] = {
                "acc": float(res.acc),
                "n_experts": int(res.report.n_experts),
                "params": int(res.report.params),
                "seconds": float(elapsed),
                "per_regime": [float(x) for x in res.per_regime],
            }
            print(f"    Seed {sd}: acc={res.acc * 100:6.2f}%, experts={res.report.n_experts:2d}, "
                  f"params={res.report.params:6d}, time={elapsed:5.2f}s")

    # 4. Assignment Diagnostic for A_today at tuned steps
    print("\n[4/5] Running assignment diagnostic for A_today at tuned steps...")
    assignment_tables = {}
    steps_a = tuned_steps["A_today"]

    for sd in SEEDS:
        st = streams[sd]
        cfg_a = PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe",
            d=st.d,
            n_classes=st.n_classes,
            seed=sd,
            inner_steps=steps_a,
            route_rule="accept_then_logf",
            q=0.02,
            alpha_spawn=0.01,
            m_min=32,
        )
        pool = PhyloMoE(cfg_a)

        seg_data_list = []
        current_k = None
        current_seg_dict = None

        for k, seg, idx in st._walk(64, sd):
            if k != current_k:
                if current_seg_dict is not None:
                    seg_data_list.append(current_seg_dict)
                current_k = k
                current_seg_dict = {
                    "segment": k,
                    "regime": int(seg.regime),
                    "first_visit": bool(seg.first_visit),
                    "total": 0,
                    "novel": 0,
                    "untrained": 0,
                    "assigned_per_expert": defaultdict(int),
                }

            rec = pool.observe(seg.x[idx], y=seg.y[idx])
            current_seg_dict["total"] += len(idx)
            current_seg_dict["novel"] += int(rec.novel.sum().item())
            current_seg_dict["untrained"] += int((rec.assigned == -1).sum().item())
            for eid in rec.assigned[rec.assigned >= 0].tolist():
                current_seg_dict["assigned_per_expert"][int(eid)] += 1

        if current_seg_dict is not None:
            seg_data_list.append(current_seg_dict)

        all_eids = sorted(list(pool.experts.keys()))
        assignment_tables[sd] = {
            "experts": all_eids,
            "segments": [
                {
                    "segment": row["segment"],
                    "regime": row["regime"],
                    "first_visit": row["first_visit"],
                    "total": row["total"],
                    "novel": row["novel"],
                    "untrained": row["untrained"],
                    "expert_counts": {e: row["assigned_per_expert"][e] for e in all_eids},
                }
                for row in seg_data_list
            ],
        }

    # 5. Compile summaries and differences
    summary_table = {}
    for arm in ALL_ARMS:
        runs = [final_runs[arm][sd] for sd in SEEDS]
        summary_table[arm] = {
            "tuned_steps": tuned_steps[arm],
            "per_seed": {sd: final_runs[arm][sd] for sd in SEEDS},
            "mean_acc": float(np.mean([r["acc"] for r in runs])),
            "mean_experts": float(np.mean([r["n_experts"] for r in runs])),
            "mean_params": float(np.mean([r["params"] for r in runs])),
            "mean_seconds": float(np.mean([r["seconds"] for r in runs])),
        }

    differences = {}
    diff_pairs = [
        ("A_today - B2", lambda sd: final_runs["A_today"][sd]["acc"] - final_runs["B2"][sd]["acc"]),
        ("B4 - B2", lambda sd: final_runs["B4"][sd]["acc"] - final_runs["B2"][sd]["acc"]),
        ("B4 - A_today", lambda sd: final_runs["B4"][sd]["acc"] - final_runs["A_today"][sd]["acc"]),
        ("kNN_all - A_today", lambda sd: final_runs["kNN_all"][sd]["acc"] - final_runs["A_today"][sd]["acc"]),
        ("B1_128 - B4", lambda sd: final_runs["B1_128"][sd]["acc"] - final_runs["B4"][sd]["acc"]),
    ]

    for name, calc in diff_pairs:
        per_seed_diff = {sd: float(calc(sd)) for sd in SEEDS}
        mean_diff = float(np.mean(list(per_seed_diff.values())))
        differences[name] = {"per_seed": per_seed_diff, "mean": mean_diff}

    # Print Report Section 2: Tuning tables
    print("\n" + "=" * 90)
    print(LABEL)
    print("SECTION 2: TUNING TABLES (steps -> mean dev score)")
    print("=" * 90)
    for arm in TUNED_ARMS:
        t_info = tuning_results[arm]
        tbl_str = ", ".join([f"{st}: {sc * 100:.2f}%" for st, sc in sorted(t_info["table"].items())])
        print(f"{arm:<10} | Best: {t_info['best_steps']:<3d} | at_edge: {str(t_info['at_edge']):<5} | Grid: {{{tbl_str}}}")

    # Print Report Section 3: Final Table per seed and mean
    print("\n" + "=" * 90)
    print(LABEL)
    print("SECTION 3: FINAL EVALUATION TABLE")
    print("=" * 90)
    header = f"{'Arm':<10} | {'Seed':<6} | {'Steps':<5} | {'Acc (%)':<9} | {'Experts':<7} | {'Params':<8} | {'Seconds':<7}"
    print(header)
    print("-" * len(header))
    for arm in ALL_ARMS:
        for sd in SEEDS:
            r = final_runs[arm][sd]
            print(f"{arm:<10} | {sd:<6} | {tuned_steps[arm]:<5d} | {r['acc'] * 100:8.2f}% | {r['n_experts']:<7d} | {r['params']:<8d} | {r['seconds']:6.2f}s")
        s = summary_table[arm]
        print(f"{arm + ' (mean)':<10} | {'ALL':<6} | {tuned_steps[arm]:<5d} | {s['mean_acc'] * 100:8.2f}% | {s['mean_experts']:<7.1f} | {int(s['mean_params']):<8d} | {s['mean_seconds']:6.2f}s")
        print("-" * len(header))

    # Print Report Section 4: Differences per seed
    print("\n" + "=" * 90)
    print(LABEL)
    print("SECTION 4: ACCURACY DIFFERENCES (percentage points)")
    print("=" * 90)
    diff_header = f"{'Comparison':<22} | {'Seed 100':<10} | {'Seed 101':<10} | {'Seed 102':<10} | {'Mean':<10}"
    print(diff_header)
    print("-" * len(diff_header))
    for name in differences:
        d = differences[name]
        print(f"{name:<22} | {d['per_seed'][100] * 100:+9.2f}% | {d['per_seed'][101] * 100:+9.2f}% | {d['per_seed'][102] * 100:+9.2f}% | {d['mean'] * 100:+9.2f}%")

    # Print Report Section 5: Per-regime accuracy per seed for A_today and B4
    print("\n" + "=" * 90)
    print(LABEL)
    print("SECTION 5: PER-REGIME ACCURACY PER SEED FOR A_today AND B4")
    print("=" * 90)
    for sd in SEEDS:
        print(f"\n--- Seed {sd} ---")
        print(f"{'Regime':<8} | {'Intents':<8} | {'A_today Acc':<12} | {'B4 Acc':<12}")
        print("-" * 46)
        a_reg = final_runs["A_today"][sd]["per_regime"]
        b4_reg = final_runs["B4"][sd]["per_regime"]
        for r_id in range(len(a_reg)):
            n_ints = regime_intents[sd][r_id]
            print(f"Regime {r_id:<2d} | {n_ints:<8d} | {a_reg[r_id] * 100:10.2f}% | {b4_reg[r_id] * 100:10.2f}%")

    # Print Report Section 6: Assignment Tables of A_today
    print("\n" + "=" * 90)
    print(LABEL)
    print("SECTION 6: ASSIGNMENT DIAGNOSTIC TABLES FOR A_today")
    print("=" * 90)
    for sd in SEEDS:
        tbl = assignment_tables[sd]
        eids = tbl["experts"]
        print(f"\n--- Seed {sd} Assignment Table (Experts: {eids}) ---")
        exp_headers = " | ".join([f"E{e:<3}" for e in eids])
        asgn_hdr = f"{'Seg':<4} | {'Reg':<4} | {'Visit':<7} | {'Total':<5} | {'Novel':<5} | {'Untrain':<7} | {exp_headers}"
        print(asgn_hdr)
        print("-" * len(asgn_hdr))
        for row in tbl["segments"]:
            v_str = "First" if row["first_visit"] else "Revisit"
            exp_counts = " | ".join([f"{row['expert_counts'][e]:<4d}" for e in eids])
            print(f"{row['segment']:<4d} | {row['regime']:<4d} | {v_str:<7} | {row['total']:<5d} | {row['novel']:<5d} | {row['untrained']:<7d} | {exp_counts}")

    # Write m5b.json
    out_dict = {
        "label": LABEL,
        "seeds": list(SEEDS),
        "tuned_steps": tuned_steps,
        "tuning": tuning_results,
        "final_runs": final_runs,
        "summary": summary_table,
        "differences": differences,
        "regime_intents": regime_intents,
        "assignment_tables": assignment_tables,
        "total_seconds": time.perf_counter() - t_start,
    }

    out_json = HERE / "m5b.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved results to {out_json} (total runtime: {out_dict['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()

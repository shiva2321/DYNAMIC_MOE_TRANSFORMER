"""DEV, exploratory, not pre-registered

M5 Banking77 smoke run on dev seeds 100, 101, 102.
Arms:
  (a) PhyloMoE (today's pool defaults + inner_steps=200, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32)
  (b) Single network (max_experts=1, forgetting baseline)
  (c) kNN_all (lookup baseline via factory("kNN_all", ...))
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from growmoe.config import PhyloCfg
from growmoe.fairfight import run_fast
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_real import banking_incremental
from research.growmoe_results.m0.run_m0 import HybridFrontend, factory

SEEDS = (100, 101, 102)


def get_peak_mb() -> float:
    info = psutil.Process(os.getpid()).memory_info()
    return getattr(info, "peak_wset", info.rss) / (1024 * 1024)


def main():
    print("=" * 80)
    print("DEV, exploratory, not pre-registered")
    print("M5 Smoke: Banking77 Incremental on Seeds 100, 101, 102")
    print("=" * 80)

    # Initialize frozen frontend once (uses local cache)
    print("Initializing HybridFrontend(d=128, seed=0)...")
    fe = HybridFrontend(d=128, seed=0)

    results = {
        "label": "DEV, exploratory, not pre-registered",
        "seeds": list(SEEDS),
        "arms": {
            "a_pool": {"name": "PhyloMoE (today's pool)", "runs": []},
            "b_single": {"name": "Single network (max_experts=1)", "runs": []},
            "c_knn": {"name": "kNN_all (lookup baseline)", "runs": []},
        },
        "stream_meta_seed100": None,
    }

    streams = {}
    for sd in SEEDS:
        print(f"Building stream for seed {sd}...")
        st = banking_incremental(seed=sd, frontend=fe)
        streams[sd] = st
        if sd == 100:
            reg_train_stream = {r_id: sum(len(seg.y) for seg in st.segments if seg.regime == r_id) for r_id in st.regimes}
            reg_test_counts = {r_id: len(st.test_sets()[r_id][1]) for r_id in st.regimes}
            results["stream_meta_seed100"] = {
                "schedule": st.meta.get("schedule"),
                "groups": st.meta.get("groups"),
                "regime_stream_train_counts": reg_train_stream,
                "regime_test_counts": reg_test_counts,
            }

    # Execute arms
    for arm_key, arm_name in [
        ("a_pool", "PhyloMoE (today)"),
        ("b_single", "Single network"),
        ("c_knn", "kNN_all"),
    ]:
        print(f"\n--- Running Arm: {arm_name} ---")
        for sd in SEEDS:
            st = streams[sd]
            t0 = time.perf_counter()

            if arm_key == "a_pool":
                cfg = PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe",
                    d=st.d,
                    n_classes=st.n_classes,
                    seed=sd,
                    inner_steps=200,
                    route_rule="accept_then_logf",
                    q=0.02,
                    alpha_spawn=0.01,
                    m_min=32,
                )
                make_pool = lambda cfg=cfg: PhyloMoE(cfg)
            elif arm_key == "b_single":
                cfg = PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe",
                    d=st.d,
                    n_classes=st.n_classes,
                    seed=sd,
                    inner_steps=200,
                    route_rule="accept_then_logf",
                    q=0.02,
                    alpha_spawn=0.01,
                    m_min=32,
                    max_experts=1,
                    cap_policy="drop",          # pinned (stage 4.1 made "share" the default): this baseline was run with the cluster dropped, so its numbers keep their meaning (E-036)
                )
                make_pool = lambda cfg=cfg: PhyloMoE(cfg)
            elif arm_key == "c_knn":
                make_pool = factory("kNN_all", st, sd, {"steps": 200})

            res = run_fast(make_pool, st, seed=sd, eval_every="final")
            elapsed = time.perf_counter() - t0
            peak_mb = get_peak_mb()

            run_record = {
                "seed": sd,
                "acc": float(res.acc),
                "n_experts": int(res.report.n_experts),
                "seconds": float(elapsed),
                "peak_mb": float(peak_mb),
            }
            if arm_key == "a_pool":
                run_record["per_regime"] = [float(x) for x in res.per_regime]

            results["arms"][arm_key]["runs"].append(run_record)
            print(f"  Seed {sd:3d}: acc={res.acc * 100:6.2f}%, experts={res.report.n_experts:2d}, "
                  f"time={elapsed:6.2f}s, peak_mem={peak_mb:6.1f} MB")

    # Compute summaries
    summary_table = {}
    for arm_key in ("a_pool", "b_single", "c_knn"):
        runs = results["arms"][arm_key]["runs"]
        accs = [r["acc"] for r in runs]
        experts = [r["n_experts"] for r in runs]
        secs = [r["seconds"] for r in runs]
        mems = [r["peak_mb"] for r in runs]
        summary_table[arm_key] = {
            "name": results["arms"][arm_key]["name"],
            "mean_acc": float(np.mean(accs)),
            "mean_experts": float(np.mean(experts)),
            "mean_seconds": float(np.mean(secs)),
            "mean_peak_mb": float(np.mean(mems)),
        }

    # Per-regime accuracy for PhyloMoE
    a_runs = results["arms"]["a_pool"]["runs"]
    per_regime_matrix = np.array([r["per_regime"] for r in a_runs])
    mean_per_regime = np.mean(per_regime_matrix, axis=0).tolist()
    summary_table["a_pool"]["mean_per_regime"] = mean_per_regime

    results["summary"] = summary_table

    # Print Seed 100 metadata
    meta100 = results["stream_meta_seed100"]
    print("\n" + "=" * 80)
    print("SEED 100 REGIME COMPOSITION & DOCUMENT COUNTS")
    print("=" * 80)
    print(f"Schedule: {meta100['schedule']}")
    for r_id, group in enumerate(meta100["groups"]):
        train_cnt = meta100["regime_stream_train_counts"][r_id]
        test_cnt = meta100["regime_test_counts"][r_id]
        print(f"Regime {r_id} ({len(group)} intents, {train_cnt} stream train docs, {test_cnt} test docs):")
        print(f"  Intents: {sorted(group)}")

    # Print Table
    print("\n" + "=" * 80)
    print("DEV, exploratory, not pre-registered")
    print("SMOKE RESULTS SUMMARY (Mean across dev seeds 100, 101, 102)")
    print("=" * 80)
    print(f"{'Arm':<32} | {'Final Acc':<10} | {'Experts':<8} | {'Seconds':<8} | {'Peak MB':<8}")
    print("-" * 80)
    for arm_key in ("a_pool", "b_single", "c_knn"):
        s = summary_table[arm_key]
        print(f"{s['name']:<32} | {s['mean_acc'] * 100:8.2f}% | {s['mean_experts']:8.1f} | {s['mean_seconds']:8.2f} | {s['mean_peak_mb']:8.1f}")
    print("-" * 80)

    print("\nArm (a) PhyloMoE Mean Accuracy per Regime:")
    for r_idx, r_acc in enumerate(mean_per_regime):
        print(f"  Regime {r_idx}: {r_acc * 100:6.2f}%")

    # Write smoke_m5.json
    out_json = HERE / "smoke_m5.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved smoke results to {out_json}")


if __name__ == "__main__":
    main()

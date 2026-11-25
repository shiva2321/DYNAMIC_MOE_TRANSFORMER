"""Aggregate all 3 seeds x 9 arms from results/real_pilot_v1/test_v1 verbatim to stdout & test_v1_summary.json."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List
import numpy as np

RES = pathlib.Path("results/real_pilot_v1/test_v1")
DOMS = ["wiki", "story", "code", "math", "forum"]
ARMS = [
    "Cog_Full", "Dense_NoReplay", "Dense_Replay", "Dense_Large",
    "Cog_NoGPM", "Cog_SingleExpert", "Dense_Joint", "Cog_Mixed", "Cog_SPGC",
]
SEEDS = [0, 1, 2]


def ms(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "vals": [float(x) for x in a]}


meta = json.loads((RES / "_meta.json").read_text())
print("=== META ===")
print("commit:", meta["meta"]["git"])
print("eval_split:", meta["meta"]["eval_split"], "steps:", meta["meta"]["steps_per_stream"],
      "cog_init:", meta["meta"]["cog_params_init"], "cog_5exp:", meta["meta"]["cog_params_grown_5exp"])
print("dense:", meta["meta"]["dense"])
print("data_meta:", meta["meta"]["data_meta"])
print("\n=== NULL & TRIVIAL BASELINES ===")
nb = meta["null_and_trivial"]["ngram_baselines_bpb"]
print("unigram:", {d: round(nb["unigram"][d], 4) for d in DOMS}, "mean=", round(sum(nb["unigram"].values()) / 5, 4))
print("bigram :", {d: round(nb["bigram"][d], 4) for d in DOMS}, "mean=", round(sum(nb["bigram"].values()) / 5, 4))
ue = meta["null_and_trivial"]["untrained_eval"]
print("untrained_cog  :", {d: round(ue["cog_bpb"][d], 4) for d in DOMS}, "mean=", round(sum(ue["cog_bpb"].values()) / 5, 4))
print("untrained_dense:", {d: round(ue["dense_bpb"][d], 4) for d in DOMS}, "mean=", round(sum(ue["dense_bpb"].values()) / 5, 4))
print("bag_of_bytes_domain_acc (with true labels):", meta["null_and_trivial"]["bag_of_bytes_domain_accuracy_with_true_labels"])

summary: Dict[str, Any] = {"meta": meta["meta"], "null_and_trivial": meta["null_and_trivial"], "arms": {}}

print("\n=== MAIN ARMS (3 SEEDS MEAN +/- STD) ===")
for arm in ARMS:
    recs = [json.loads((RES / f"seed{s}_{arm}.json").read_text()) for s in SEEDS]
    arm_s: Dict[str, Any] = {
        "train_seconds": ms([r["train_seconds"] for r in recs]),
        "forward_passes": ms([r["forward_passes"] for r in recs]),
        "backward_passes": ms([r["backward_passes"] for r in recs]),
        "skipped_steps": ms([r["skipped_steps"] for r in recs]),
        "peak_vram_mb": ms([r["peak_vram_mb"] for r in recs]),
        "params_final": ms([r["params_final"] for r in recs]),
    }

    is_seq = "matrix" in recs[0]
    if is_seq:
        final_dom, peak_dom, fgt_dom, fgt_pct_dom = {}, {}, {}, {}
        for di, d in enumerate(DOMS):
            finals = [r["matrix"][-1][d]["bpb"] for r in recs]
            peaks = [r["matrix"][di][d]["bpb"] for r in recs]  # immediately after domain d
            fgts = [f - p for f, p in zip(finals, peaks)]
            fgt_pcts = [100.0 * (f - p) / p for f, p in zip(finals, peaks)]
            final_dom[d] = ms(finals)
            peak_dom[d] = ms(peaks)
            fgt_dom[d] = ms(fgts)
            fgt_pct_dom[d] = ms(fgt_pcts)
        arm_s["final_bpb"] = final_dom
        arm_s["peak_after_own_bpb"] = peak_dom
        arm_s["forgetting_bpb"] = fgt_dom
        arm_s["forgetting_pct"] = fgt_pct_dom
        arm_s["mean_final_bpb"] = ms([np.mean([r["matrix"][-1][d]["bpb"] for d in DOMS]) for r in recs])
        arm_s["mean_forgetting_4prior_bpb"] = ms([
            np.mean([r["matrix"][-1][d]["bpb"] - r["matrix"][di][d]["bpb"] for di, d in enumerate(DOMS[:-1])])
            for r in recs
        ])
        arm_s["mean_forgetting_4prior_pct"] = ms([
            np.mean([100.0 * (r["matrix"][-1][d]["bpb"] - r["matrix"][di][d]["bpb"]) / r["matrix"][di][d]["bpb"] for di, d in enumerate(DOMS[:-1])])
            for r in recs
        ])
    else:
        final_dom = {d: ms([r["final"][d]["bpb"] for r in recs]) for d in DOMS}
        arm_s["final_bpb"] = final_dom
        arm_s["mean_final_bpb"] = ms([np.mean([r["final"][d]["bpb"] for d in DOMS]) for r in recs])

    if "routing" in recs[0]:
        arm_s["nmi"] = ms([r["routing"]["nmi_domain_vs_expert"] for r in recs])
    if is_seq and "_num_experts" in recs[0]["matrix"][-1]:
        arm_s["experts_final"] = ms([r["matrix"][-1]["_num_experts"] for r in recs])
        arm_s["experts_trajectory"] = [[row["_num_experts"] for row in r["matrix"]] for r in recs]
    if "gpm_ranks" in recs[0] and recs[0]["gpm_ranks"]:
        arm_s["gpm_final"] = {
            "mean_cap_ratio": ms([r["gpm_ranks"]["mean_rank_ratio_of_cap"] for r in recs]),
            "at_cap_count": ms([r["gpm_ranks"]["at_cap_count"] for r in recs]),
        }
        arm_s["gpm_at_cap_trajectory"] = [[row.get("_gpm_summary", {}).get("at_cap_count", 0) for row in r["matrix"]] for r in recs]
        arm_s["gpm_mean_cap_trajectory"] = [[round(row.get("_gpm_summary", {}).get("mean_rank_ratio_of_cap", 0), 3) for row in r["matrix"]] for r in recs]

    # Meaning probe
    mp = [r["meaning_probe_banking77"]["trained"] for r in recs]
    arm_s["meaning_banking77"] = {
        "intent_77_acc": ms([x["intent_77_acc"] for x in mp]),
        "same_vs_overlap_diff_auc": ms([x["same_vs_overlap_diff_auc"] for x in mp]),
        "sim_same_intent": ms([x["sim_same_intent"] for x in mp]),
        "sim_diff_high_overlap": ms([x["sim_diff_high_overlap"] for x in mp]),
    }
    if "references" in recs[0]["meaning_probe_banking77"] and "bag_of_bytes" in recs[0]["meaning_probe_banking77"]["references"]:
        refs_all = [r["meaning_probe_banking77"]["references"] for r in recs]
        summary["meaning_references"] = {
            "n_train": refs_all[0]["n_train"], "n_test": refs_all[0]["n_test"],
            "chance": refs_all[0]["chance"], "majority": refs_all[0]["majority"],
            "bag_of_bytes": refs_all[0]["bag_of_bytes"],
            "Cog_untrained_acc": ms([rf["Cog_untrained"]["intent_77_acc"] for rf in refs_all]),
            "Cog_untrained_auc": ms([rf["Cog_untrained"]["same_vs_overlap_diff_auc"] for rf in refs_all]),
            "Dense_untrained_acc": ms([rf["Dense_untrained"]["intent_77_acc"] for rf in refs_all]),
            "Dense_untrained_auc": ms([rf["Dense_untrained"]["same_vs_overlap_diff_auc"] for rf in refs_all]),
        }

    # Context & noise probe
    arm_s["ctx_noise"] = {
        d: {
            k: ms([r["context_and_noise_probe"][d][k] for r in recs])
            for k in ["suffix_clean_bpb", "delta_shuffle", "delta_noise10"]
        }
        for d in ["wiki", "story", "code", "math"]
    }

    # Compositional probe
    cp_keys = ["arith_exact_match", "arith_exact_small", "gsm_simple_le2_bpb", "gsm_multi_ge4_bpb",
               "var_binding_exact2", "var_binding_2byte_bpb", "var_binding_shuffled_1st_byte_bpb",
               "code_shallow_bpb", "code_deep_bpb"]
    arm_s["compositional"] = {k: ms([r["compositional_probe"][k] for r in recs]) for k in cp_keys}
    arm_s["compositional"]["arith_baselines"] = {
        "copy_first": recs[0]["compositional_probe"]["arith_copy_first"],
        "majority": recs[0]["compositional_probe"]["arith_majority"],
        "arith_n": recs[0]["compositional_probe"]["arith_n"],
        "arith_n_small": recs[0]["compositional_probe"]["arith_n_small"],
    }

    # Thinking, LTM/PC, Revisit, WM/PC/Surprise from final row
    if "thinking_ablation" in recs[0]:
        arm_s["thinking_ablation"] = {
            d: {k: ms([r["thinking_ablation"][d][k] for r in recs])
                for k in ["bpb_adaptive", "bpb_think0", "delta_adp_minus_0", "mean_steps_adaptive", "surprise_nll_corr"]}
            for d in DOMS
        }
    if is_seq and arm.startswith("Cog"):
        arm_s["eval_internals"] = {
            d: {
                "wm_cos_1step": ms([r["matrix"][-1][d]["wm_cos_1step"] for r in recs]),
                "pc_trigger_frac": ms([r["matrix"][-1][d]["pc_trigger_frac"] for r in recs]),
                "surprise_p10": ms([r["matrix"][-1][d]["surprise_p10_p50_p90"][0] for r in recs]),
                "surprise_p50": ms([r["matrix"][-1][d]["surprise_p10_p50_p90"][1] for r in recs]),
                "surprise_p90": ms([r["matrix"][-1][d]["surprise_p10_p50_p90"][2] for r in recs]),
            }
            for d in DOMS
        }
    if "ltm_and_pc_probe" in recs[0]:
        lp = [r["ltm_and_pc_probe"] for r in recs]
        arm_s["ltm_and_pc"] = {
            "no_ltm": ms([x["ltm_bpb_no_ltm"] for x in lp]),
            "matched_ltm": ms([x["ltm_bpb_matched_ltm"] for x in lp]),
            "distractor_ltm": ms([x["ltm_bpb_distractor_ltm"] for x in lp]),
            "slots_added": ms([x["ltm_mean_slots_added"] for x in lp]),
            "pc_clean_trig": ms([x["pc_clean_pairs"]["pc_trigger_frac"] for x in lp]),
            "pc_contra_trig": ms([x["pc_contradiction_pairs"]["pc_trigger_frac"] for x in lp]),
            "pc_clean_bpb": ms([x["pc_clean_pairs"]["mean_bpb"] for x in lp]),
            "pc_contra_bpb": ms([x["pc_contradiction_pairs"]["mean_bpb"] for x in lp]),
        }
    if "revisit_probe" in recs[0]:
        rv = [r["revisit_probe"] for r in recs]
        arm_s["revisit"] = {
            "spawns_on_revisit": ms([x["spawns_on_revisit"] for x in rv]),
            "delta_bpb": {d: ms([x["delta_bpb"][d] for x in rv]) for d in DOMS},
        }
    if "generation" in recs[0]:
        arm_s["gen_stats_greedy"] = {
            d: {
                "distinct_4gram_frac": ms([g["stats_greedy"]["distinct_4gram_frac"] for r in recs for g in r["generation"][d]]),
                "max_4gram_repeat": ms([g["stats_greedy"]["max_4gram_repeat"] for r in recs for g in r["generation"][d]]),
                "det_identical_rate": float(np.mean([g["deterministic_repeat_identical"] for r in recs for g in r["generation"][d]])),
                "utf8_valid_rate": float(np.mean([g["stats_greedy"]["utf8_valid"] for r in recs for g in r["generation"][d]])),
            }
            for d in DOMS
        }
        arm_s["gen_stats_temp08"] = {
            d: {
                "distinct_4gram_frac": ms([g["stats_temp08"]["distinct_4gram_frac"] for r in recs for g in r["generation"][d]]),
                "max_4gram_repeat": ms([g["stats_temp08"]["max_4gram_repeat"] for r in recs for g in r["generation"][d]]),
                "utf8_valid_rate": float(np.mean([g["stats_temp08"]["utf8_valid"] for r in recs for g in r["generation"][d]])),
            }
            for d in DOMS
        }

    summary["arms"][arm] = arm_s
    row_str = " ".join(f"{d}={arm_s['final_bpb'][d]['mean']:.3f}+/-{arm_s['final_bpb'][d]['std']:.3f}" for d in DOMS)
    fgt_str = f" fgt4={arm_s['mean_forgetting_4prior_bpb']['mean']:+.3f}" if is_seq else ""
    print(f"{arm:17s} | mean={arm_s['mean_final_bpb']['mean']:.3f}+/-{arm_s['mean_final_bpb']['std']:.3f}{fgt_str} | {row_str} | {arm_s['train_seconds']['mean']:.1f}s vram={arm_s['peak_vram_mb']['mean']:.0f}MB")

out_p = pathlib.Path("results/real_pilot_v1/test_v1_summary.json")
out_p.write_text(json.dumps(summary, indent=2))
print("\nWrote", out_p)

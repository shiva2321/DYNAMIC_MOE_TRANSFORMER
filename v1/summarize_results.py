import json
import glob
from pathlib import Path
import numpy as np

def summarize():
    print("=== PART A: CLOSED-FORM HEADS SUMMARY ===")
    models = ["ClassMean", "KnnMemory", "MahalanobisProto", "RandomFeatureRidge", "ExactLda", "StreamingLDA"]
    r_sets = ["R1", "R2", "R3", "R4"]
    seeds = [800, 801, 802]

    for m in models:
        stat_diffs = []
        weight_diffs = []
        prob_diffs = []
        agreements = []
        for rid in r_sets:
            for s in seeds:
                f = Path(f"results/v1/heads_{rid.lower()}_s{s}.json")
                d = json.loads(f.read_text(encoding="utf-8"))
                res = d["models"][m]
                stat_diffs.append(res["max_stat_diff"])
                weight_diffs.append(res["max_weight_diff"])
                prob_diffs.append(res["max_prob_diff"])
                agreements.append(res["argmax_agreement"])
        print(f"{m:20s}: stat_max={max(stat_diffs):.2e}, weight_max={max(weight_diffs):.2e}, prob_max={max(prob_diffs):.2e}, min_agree={min(agreements):.6f}")

    print("\n=== STREAMING LDA vs EXACT LDA ===")
    cov_frob = []
    acc_diffs = []
    agrees = []
    for rid in r_sets:
        for s in seeds:
            f = Path(f"results/v1/heads_{rid.lower()}_s{s}.json")
            d = json.loads(f.read_text(encoding="utf-8"))
            comp = d["streaming_lda_vs_exact_lda"]
            cov_frob.append(comp["covariance_frobenius_diff"])
            acc_diffs.append(comp["accuracy_diff"])
            agrees.append(comp["agreement"])
    print(f"Cov Frobenius diff: mean={np.mean(cov_frob):.6f}, min={min(cov_frob):.6f}, max={max(cov_frob):.6f}")
    print(f"Accuracy diff: mean={np.mean(acc_diffs):.6f}, min={min(acc_diffs):.6f}, max={max(acc_diffs):.6f}")
    print(f"Argmax agreement: mean={np.mean(agrees):.6f}, min={min(agrees):.6f}, max={max(agrees):.6f}")

    print("\n=== PART B: POOL REMOVAL SUMMARY ===")
    for rid in r_sets:
        print(f"\n--- Removal Set {rid} ---")
        for s in seeds:
            f = Path(f"results/v1/pool_{rid.lower()}_s{s}.json")
            d = json.loads(f.read_text(encoding="utf-8"))
            exp = d["exposure"]
            comp = d["comparison"]
            cert = d["certificate"]
            rem_eval = comp["removed_class_evaluation"]
            match = comp["expert_weight_matching"]
            time_s = d.get("timing_seconds", {}).get("total_experiment_seconds", 0.0)
            print(
                f"Seed {s} ({time_s:.2f}s): "
                f"dropped={len(exp['dropped_experts'])}, "
                f"retained={len(exp['retained_experts'])}, "
                f"collateral_lost={exp['n_collateral_classes_lost']}, "
                f"P_full={comp['mean_accuracy_retained']['P_full']:.4f}, "
                f"P_rem={comp['mean_accuracy_retained']['P_removed']:.4f}, "
                f"P_nev={comp['mean_accuracy_retained']['P_never']:.4f}, "
                f"route_agree={comp['routing_agreement_retained']:.4f}, "
                f"argmax_agree={comp['argmax_agreement_retained']:.4f}, "
                f"matched_exp={match['n_matched']}/{match['n_matched']+match['n_unmatched']}, "
                f"rem_pred={rem_eval['predictions_predicting_removed_label']}/{rem_eval['total_removed_test_samples']}, "
                f"traces={cert['trace_audit']['total_traces_found']}"
            )

if __name__ == "__main__":
    summarize()

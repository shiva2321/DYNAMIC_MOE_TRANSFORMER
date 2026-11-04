import json
from pathlib import Path
import numpy as np

def run():
    r_sets = ["R1", "R2", "R3", "R4"]
    seeds = [800, 801, 802]

    heads_data = {r: {s: json.loads(Path(f"results/v1/heads_{r.lower()}_s{s}.json").read_text(encoding="utf-8")) for s in seeds} for r in r_sets}
    pool_data = {r: {s: json.loads(Path(f"results/v1/pool_{r.lower()}_s{s}.json").read_text(encoding="utf-8")) for s in seeds} for r in r_sets}

    print("--- HEADS STATS ---")
    models = ["ClassMean", "KnnMemory", "MahalanobisProto", "RandomFeatureRidge", "ExactLda", "StreamingLDA"]
    for m in models:
        for r in r_sets:
            stat_d = [heads_data[r][s]["models"][m]["max_stat_diff"] for s in seeds]
            prob_d = [heads_data[r][s]["models"][m]["max_prob_diff"] for s in seeds]
            w_d = [heads_data[r][s]["models"][m]["max_weight_diff"] for s in seeds]
            agree = [heads_data[r][s]["models"][m]["argmax_agreement"] for s in seeds]
            print(f"{m} {r}: stat={max(stat_d):.2e}, weight={max(w_d):.2e}, prob={max(prob_d):.2e}, agree_min={min(agree):.6f}")

    print("\n--- POOL ACCURACY & ROUTING ---")
    for r in r_sets:
        p_full = [pool_data[r][s]["comparison"]["mean_accuracy_retained"]["P_full"] for s in seeds]
        p_rem = [pool_data[r][s]["comparison"]["mean_accuracy_retained"]["P_removed"] for s in seeds]
        p_nev = [pool_data[r][s]["comparison"]["mean_accuracy_retained"]["P_never"] for s in seeds]
        r_agree = [pool_data[r][s]["comparison"]["routing_agreement_retained"] for s in seeds]
        a_agree = [pool_data[r][s]["comparison"]["argmax_agreement_retained"] for s in seeds]
        drop = [len(pool_data[r][s]["exposure"]["dropped_experts"]) for s in seeds]
        ret = [len(pool_data[r][s]["exposure"]["retained_experts"]) for s in seeds]
        coll = [pool_data[r][s]["exposure"]["n_collateral_classes_lost"] for s in seeds]
        print(f"{r}: dropped={drop}, retained={ret}, collateral={coll}")
        print(f"   P_full: {[round(x, 4) for x in p_full]} (mean {np.mean(p_full):.4f})")
        print(f"   P_rem : {[round(x, 4) for x in p_rem]} (mean {np.mean(p_rem):.4f})")
        print(f"   P_nev : {[round(x, 4) for x in p_nev]} (mean {np.mean(p_nev):.4f})")
        print(f"   route : {[round(x, 4) for x in r_agree]} (mean {np.mean(r_agree):.4f})")
        print(f"   agree : {[round(x, 4) for x in a_agree]} (mean {np.mean(a_agree):.4f})")

    print("\n--- EXPERT WEIGHT MATCHING ---")
    for r in r_sets:
        for s in seeds:
            m = pool_data[r][s]["comparison"]["expert_weight_matching"]
            diffs = [v["max_weight_diff"] for v in m["details"].values()] if m["details"] else [0.0]
            print(f"Pool {r} s{s}: matched={m['n_matched']}, unmatched={m['n_unmatched']}, max_diff={max(diffs):.2e}")

    print("\n--- REMOVED CLASS EVALUATION ---")
    for r in r_sets:
        for s in seeds:
            rem = pool_data[r][s]["comparison"]["removed_class_evaluation"]
            print(f"Pool {r} s{s}: samples={rem['total_removed_test_samples']}, pred_rem={rem['predictions_predicting_removed_label']}, acc={rem['accuracy_on_removed_classes']}")

    print("\n--- TIMINGS ---")
    for r in r_sets:
        for s in seeds:
            t = pool_data[r][s]["timing_seconds"]
            print(f"Pool {r} s{s}: {t['total']:.2f}s (rem: {t['removal']:.2f}s, nev: {t['never_train']:.2f}s, comp: {t['comparison']:.2f}s)")

if __name__ == "__main__":
    run()

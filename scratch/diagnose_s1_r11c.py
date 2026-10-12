"""R11c diagnostic script: S1 Owner-Set Misrouting, D8 Failure Mode Analysis, and Candidate Routing Rules.

Addresses Review Section I (GrowMoE R11c):
- I2: Redefine owner set = experts that received >= 10% of regime's training samples.
      Report mis-routing relative to owner set (test seeds 0-9).
- I3 (D8): Measure winner and owner conformal p (median, IQR), log-density margin,
           and p-value saturation/noise on misrouted samples.
- I1: Candidate routing rules on dev seeds 100-109 (and test seeds 0-9):
      (1) default
      (2) argmax_logf
      (3) nearest_proto
      (4) accept_then_logf (eligible = p >= q, max logf; if none eligible -> argmax logf or unknown)
      and non-regression across streams S1, S2, S3, S4.
"""

import sys
sys.path.insert(0, ".")
import os
import math
import numpy as np
import torch
from collections import defaultdict, Counter

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted, s3_drift, s4_outliers
from growmoe.metrics import mean_ci, paired_diff, auroc, false_novelty_rate, novel_detection_rate
from growmoe.bench import run_stream


def run_d7_d8_owner_analysis(seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), owner_threshold=0.10):
    print("=" * 80)
    print("GrowMoE R11c: D7 Owner-Set & D8 Argmax Diagnostic on Test Seeds (0-9)")
    print(f"Owner set threshold: >= {owner_threshold*100:.0f}% of regime's training samples")
    print("=" * 80)

    n_regimes = 7
    max_e = 20

    total_train_assign = np.zeros((n_regimes, max_e), dtype=np.int64)
    total_test_route = np.zeros((n_regimes, max_e), dtype=np.int64)
    regime_test_counts = np.zeros(n_regimes, dtype=np.int64)
    regime_train_counts = np.zeros(n_regimes, dtype=np.int64)

    # Historical single-majority metrics
    test_mismatch_rates_maj = []
    # Redefined owner-set metrics
    test_mismatch_rates_set = []

    misrouted_total_maj = 0
    misrouted_maj_owner_accepts = 0
    misrouted_maj_owner_rejects = 0

    misrouted_total_set = 0
    misrouted_set_any_accepts = 0
    misrouted_set_all_rejects = 0

    per_seed_maj_accept_frac = []
    per_seed_set_accept_frac = []

    # D8 measurements on misrouted samples (both majority and owner-set)
    d8_p_winner = []
    d8_p_owner_maj = []
    d8_p_owner_set_max = []
    d8_logf_margin_maj = []       # logf(winner) - logf(owner_maj)
    d8_logf_margin_set_max = []   # logf(winner) - max_{e in owner_set} logf(e)

    # Breakdown of why winner was chosen over owner (for set-misrouted samples):
    # Case 1: Both owner and winner eligible (p >= q), winner had higher logf
    # Case 2: Winner eligible (p >= q), owner not eligible (p < q)
    # Case 3: Neither eligible (all p < q), winner won on argmax p fallback
    # Case 4: Owner eligible (p >= q), winner NOT eligible (p < q) [should be impossible in default]
    set_case_counts = Counter()

    # Saturation counts
    sat_winner_p90 = 0
    sat_owner_p90 = 0
    sat_multi_p90 = 0   # >= 2 experts have p >= 0.90
    sat_multi_p50 = 0   # >= 2 experts have p >= 0.50
    sat_all_reject = 0  # all experts p < q

    regime_names = [
        "R0: digits (image)",
        "R1: wine (tabular)",
        "R2: breast_cancer (tab)",
        "R3: sinusoid (series)",
        "R4: ar2 (series)",
        "R5: pydoc_text (text)",
        "R6: stdlib_code (code)",
    ]

    for seed in seeds:
        s1 = s1_types(seed=seed, n_train=400, n_test=150)
        cfg = PhyloCfg(seed=seed, q=0.02, alpha_spawn=0.01, m_min=32)
        pool = PhyloMoE(cfg)

        # Train on stream (test-then-train protocol)
        train_counts_seed = defaultdict(Counter)
        for _, seg, idx in s1._walk(batch_size=64, seed=seed):
            bx = seg.x[idx]
            by = seg.y[idx]
            bctx = seg.ctx[idx] if seg.ctx is not None else None
            _ = pool.predict(bx, ctx=bctx)
            rec = pool.observe(bx, by, ctx=bctx)
            assigned = rec.assigned.cpu().numpy()
            for e in assigned:
                if e >= 0:
                    train_counts_seed[seg.regime][e] += 1
                    total_train_assign[seg.regime, e] += 1
                    regime_train_counts[seg.regime] += 1

        # Identify training owners for each regime
        maj_owners = {}
        owner_sets = {}
        for r in range(n_regimes):
            counts = train_counts_seed[r]
            total_r = sum(counts.values())
            if total_r > 0:
                maj_e = max(counts, key=counts.get)
                maj_owners[r] = maj_e
                owner_sets[r] = {e for e, c in counts.items() if (c / total_r) >= owner_threshold}
            else:
                maj_owners[r] = 0
                owner_sets[r] = {0}

        # Evaluate test sets under default rule
        test_sets = s1.test_sets()
        seed_test_total = 0
        seed_mis_maj = 0
        seed_mis_set = 0
        seed_maj_accept = 0
        seed_set_accept = 0

        for r, (x_te, y_te) in test_sets.items():
            n_te = len(y_te)
            seed_test_total += n_te
            regime_test_counts[r] += n_te
            x_te_norm = pool._normalize(x_te.float())

            eids = list(pool.experts.keys())
            p_mat = torch.stack([pool.experts[e].meta["sig"].pvalue(x_te_norm) for e in eids], dim=1).cpu().numpy()
            logf_mat = torch.stack([pool.experts[e].meta["sig"].logpdf(x_te_norm) for e in eids], dim=1).cpu().numpy()

            pool.route_rule = "default"
            pred = pool.predict(x_te)
            routed = pred.expert.cpu().numpy()

            for i in range(n_te):
                e_routed = routed[i]
                if 0 <= e_routed < max_e:
                    total_test_route[r, e_routed] += 1

                p_row = p_mat[i]
                logf_row = logf_mat[i]

                idx_winner = eids.index(e_routed) if e_routed in eids else -1
                p_win = p_row[idx_winner] if idx_winner >= 0 else 0.0
                logf_win = logf_row[idx_winner] if idx_winner >= 0 else -1e9

                # Majority owner
                maj_e = maj_owners[r]
                idx_maj = eids.index(maj_e) if maj_e in eids else -1
                p_maj = p_row[idx_maj] if idx_maj >= 0 else 0.0
                logf_maj = logf_row[idx_maj] if idx_maj >= 0 else -1e9

                # Owner set
                o_set = owner_sets[r]
                o_indices = [eids.index(e) for e in o_set if e in eids]
                p_set_max = max([p_row[idx] for idx in o_indices]) if o_indices else 0.0
                logf_set_max = max([logf_row[idx] for idx in o_indices]) if o_indices else -1e9

                # 1. Check majority owner misrouting
                if e_routed != maj_e:
                    seed_mis_maj += 1
                    misrouted_total_maj += 1
                    if p_maj >= cfg.q:
                        seed_maj_accept += 1
                        misrouted_maj_owner_accepts += 1
                    else:
                        misrouted_maj_owner_rejects += 1

                # 2. Check owner set misrouting
                if e_routed not in o_set:
                    seed_mis_set += 1
                    misrouted_total_set += 1

                    # D8 measurements
                    d8_p_winner.append(p_win)
                    d8_p_owner_maj.append(p_maj)
                    d8_p_owner_set_max.append(p_set_max)
                    d8_logf_margin_maj.append(logf_win - logf_maj)
                    d8_logf_margin_set_max.append(logf_win - logf_set_max)

                    if p_win >= 0.90:
                        sat_winner_p90 += 1
                    if p_set_max >= 0.90:
                        sat_owner_p90 += 1
                    if (p_row >= 0.90).sum() >= 2:
                        sat_multi_p90 += 1
                    if (p_row >= 0.50).sum() >= 2:
                        sat_multi_p50 += 1
                    if (p_row < cfg.q).all():
                        sat_all_reject += 1

                    # Accept vs reject for owner set
                    if p_set_max >= cfg.q:
                        seed_set_accept += 1
                        misrouted_set_any_accepts += 1
                    else:
                        misrouted_set_all_rejects += 1

                    # Case breakdown
                    win_eligible = (p_win >= cfg.q)
                    owner_eligible = (p_set_max >= cfg.q)
                    if win_eligible and owner_eligible:
                        set_case_counts["case1_both_accept_winner_higher_logf"] += 1
                    elif win_eligible and not owner_eligible:
                        set_case_counts["case2_winner_accepts_owner_rejects"] += 1
                    elif not win_eligible and not owner_eligible:
                        set_case_counts["case3_all_reject_fallback_argmax_p"] += 1
                    else:
                        set_case_counts["case4_owner_accepts_winner_rejects"] += 1

        test_mismatch_rates_maj.append(seed_mis_maj / max(1, seed_test_total))
        test_mismatch_rates_set.append(seed_mis_set / max(1, seed_test_total))
        per_seed_maj_accept_frac.append(seed_maj_accept / max(1, seed_mis_maj) if seed_mis_maj > 0 else 0.0)
        per_seed_set_accept_frac.append(seed_set_accept / max(1, seed_mis_set) if seed_mis_set > 0 else 0.0)

    # Print summary
    print("\n--- Misrouting Rates: Single Majority vs Redefined Owner Set (>= 10%) ---")
    m_maj, lo_maj, hi_maj = mean_ci(test_mismatch_rates_maj)
    m_set, lo_set, hi_set = mean_ci(test_mismatch_rates_set)
    print(f"Majority Owner Misrouting: {m_maj*100:.2f}% [{lo_maj*100:.2f}%, {hi_maj*100:.2f}%]")
    print(f"Owner Set Misrouting:     {m_set*100:.2f}% [{lo_set*100:.2f}%, {hi_set*100:.2f}%]")

    print("\n--- Gate Behavior on Misrouted Test Samples ---")
    acc_maj_m, acc_maj_lo, acc_maj_hi = mean_ci(per_seed_maj_accept_frac)
    acc_set_m, acc_set_lo, acc_set_hi = mean_ci(per_seed_set_accept_frac)
    print(f"Majority Misrouted: Total = {misrouted_total_maj}")
    print(f"  Majority Owner ACCEPTS (p >= q) and loses: {acc_maj_m*100:.2f}% [{acc_maj_lo*100:.2f}%, {acc_maj_hi*100:.2f}%]")
    print(f"  Majority Owner REJECTS (p < q):            {(1-acc_maj_m)*100:.2f}%")
    print(f"Owner Set Misrouted: Total = {misrouted_total_set}")
    print(f"  Owner Set ACCEPTS (any p >= q) and loses:  {acc_set_m*100:.2f}% [{acc_set_lo*100:.2f}%, {acc_set_hi*100:.2f}%]")
    print(f"  Owner Set REJECTS (all p < q):             {(1-acc_set_m)*100:.2f}%")

    print("\n" + "=" * 80)
    print("D8: NUMERICAL MECHANISM ANALYSIS ON MISROUTED SAMPLES (Owner-Set)")
    print(f"Total misrouted samples analyzed: {len(d8_p_winner)}")
    print("=" * 80)

    def stats_summary(arr):
        a = np.array(arr)
        med = float(np.median(a))
        q25 = float(np.percentile(a, 25))
        q75 = float(np.percentile(a, 75))
        iqr = q75 - q25
        return med, q25, q75, iqr

    w_med, w_q25, w_q75, w_iqr = stats_summary(d8_p_winner)
    o_med, o_q25, o_q75, o_iqr = stats_summary(d8_p_owner_set_max)
    m_med, m_q25, m_q75, m_iqr = stats_summary(d8_logf_margin_set_max)

    print(f"{'Quantity':<35} | {'Median':<10} | {'25th Pct':<10} | {'75th Pct':<10} | {'IQR':<10}")
    print("-" * 80)
    print(f"{'Winner conformal p':<35} | {w_med:<10.4f} | {w_q25:<10.4f} | {w_q75:<10.4f} | {w_iqr:<10.4f}")
    print(f"{'Owner set max conformal p':<35} | {o_med:<10.4f} | {o_q25:<10.4f} | {o_q75:<10.4f} | {o_iqr:<10.4f}")
    print(f"{'Log-density margin (win - owner)':<35} | {m_med:<10.4f} | {m_q25:<10.4f} | {m_q75:<10.4f} | {m_iqr:<10.4f}")

    n_tot = len(d8_p_winner)
    print("\n--- P-Value Saturation & Degeneracy Rates on Misrouted Samples ---")
    print(f"  Winner p >= 0.90:                        {sat_winner_p90}/{n_tot} ({sat_winner_p90/n_tot*100:.1f}%)")
    print(f"  Owner set p >= 0.90:                     {sat_owner_p90}/{n_tot} ({sat_owner_p90/n_tot*100:.1f}%)")
    print(f"  Multi-expert saturation (>= 2 p >= 0.90): {sat_multi_p90}/{n_tot} ({sat_multi_p90/n_tot*100:.1f}%)")
    print(f"  Multi-expert inlier (>= 2 p >= 0.50):     {sat_multi_p50}/{n_tot} ({sat_multi_p50/n_tot*100:.1f}%)")
    print(f"  All-reject fallback (all p < q):          {sat_all_reject}/{n_tot} ({sat_all_reject/n_tot*100:.1f}%)")

    print("\n--- Case Breakdown: Why Winner Was Chosen Over Owner Set ---")
    for k, v in sorted(set_case_counts.items()):
        print(f"  {k}: {v}/{n_tot} ({v/n_tot*100:.2f}%)")

    return {
        "m_maj": m_maj, "lo_maj": lo_maj, "hi_maj": hi_maj,
        "m_set": m_set, "lo_set": lo_set, "hi_set": hi_set,
        "acc_set_m": acc_set_m, "acc_set_lo": acc_set_lo, "acc_set_hi": acc_set_hi,
        "w_stats": (w_med, w_q25, w_q75, w_iqr),
        "o_stats": (o_med, o_q25, o_q75, o_iqr),
        "m_stats": (m_med, m_q25, m_q75, m_iqr),
        "saturation": {
            "winner_p90": sat_winner_p90 / n_tot,
            "owner_p90": sat_owner_p90 / n_tot,
            "multi_p90": sat_multi_p90 / n_tot,
            "multi_p50": sat_multi_p50 / n_tot,
            "all_reject": sat_all_reject / n_tot,
        },
        "cases": {k: v / n_tot for k, v in set_case_counts.items()},
    }


if __name__ == "__main__":
    res = run_d7_d8_owner_analysis(seeds=range(10), owner_threshold=0.10)

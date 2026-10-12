"""Diagnostic script for S1 performance and gate decomposition in PhyloMoE (Review Items D1 & D7).

Addresses Claude's review D7:
(a) Training-time assignment matrix (regime x expert) next to test-time routing matrix,
    and the share of test samples routed to an expert other than their regime's training majority expert.
(b) For mis-routed test samples, whether the owner expert accepts (conformal p >= q) and loses the argmax,
    or rejects (p < q, signature under-fit).
(c) Offline evaluation on the same trained pool under alternative routing rules:
    - default (current conformal-gated max logf)
    - argmax_p (pure argmax conformal p-value)
    - argmax_logf (pure argmax log-density)
    - zscore_logf (per-expert z-scored log-density against prototype scores)
    - nearest_proto (nearest-prototype cosine similarity)
    - oracle (test-time routing to training-time majority expert)
(d) Per-expert signature sizes (prototypes, samples, kappa) and foreign test samples captured
    to test the "broad expert wins" hypothesis.
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
from growmoe.streams import s1_types
from growmoe.metrics import mean_ci, paired_diff


def run_s1_diagnostic_d7(seeds: int = 10):
    print("=" * 80)
    print("GrowMoE S1 Routing Gate Decomposition Diagnostic (Review Item D7)")
    print(f"Evaluating across {seeds} seeds on heterogeneous types stream (S1)...")
    print("=" * 80)

    n_regimes = 7
    max_e = 20
    total_train_assign = np.zeros((n_regimes, max_e), dtype=np.int64)
    total_test_route = np.zeros((n_regimes, max_e), dtype=np.int64)
    regime_test_counts = np.zeros(n_regimes, dtype=np.int64)
    regime_train_counts = np.zeros(n_regimes, dtype=np.int64)

    test_mismatch_rates = []

    misrouted_total = 0
    misrouted_owner_accepts = 0  # p >= q, lost argmax
    misrouted_owner_rejects = 0  # p < q, underfit

    per_seed_owner_accept_frac = []

    rules = ["default", "argmax_p", "argmax_logf", "zscore_logf", "nearest_proto", "oracle"]
    accs_by_rule = {r: [] for r in rules}
    per_regime_by_rule = {r: defaultdict(list) for r in rules}

    expert_proto_counts = defaultdict(list)
    expert_sample_counts = defaultdict(list)
    expert_kappas = defaultdict(list)
    expert_stolen_counts = defaultdict(int)
    expert_owner_regimes = defaultdict(set)

    regime_names = [
        "R0: digits (image)",
        "R1: wine (tabular)",
        "R2: breast_cancer (tab)",
        "R3: sinusoid (series)",
        "R4: ar2 (series)",
        "R5: pydoc_text (text)",
        "R6: stdlib_code (code)",
    ]

    for seed in range(seeds):
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

        # Identify training majority owner expert for each regime
        owner_expert = {}
        for r in range(n_regimes):
            if train_counts_seed[r]:
                maj_e = max(train_counts_seed[r], key=train_counts_seed[r].get)
                owner_expert[r] = maj_e
                expert_owner_regimes[maj_e].add(r)
            else:
                owner_expert[r] = 0

        # Record expert signature properties for part (d)
        for eid, rec in pool.experts.items():
            sig = rec.meta["sig"]
            expert_proto_counts[eid].append(len(sig.protos))
            expert_sample_counts[eid].append(sig.seen)
            expert_kappas[eid].append(sig.kappa)

        # Evaluate test sets across alternative routing rules and track misrouting details
        test_sets = s1.test_sets()
        seed_test_total = 0
        seed_misrouted = 0
        seed_mis_accept = 0

        seed_correct = {r: 0 for r in rules}

        for r, (x_te, y_te) in test_sets.items():
            n_te = len(y_te)
            seed_test_total += n_te
            regime_test_counts[r] += n_te
            x_te_norm = pool._normalize(x_te.float())

            owner_e = owner_expert[r]
            owner_sig = pool.experts[owner_e].meta["sig"] if owner_e in pool.experts else None

            # Test under each rule
            for rule in rules:
                if rule == "oracle":
                    if owner_e in pool.experts:
                        probs = pool.experts[owner_e].module.proba(x_te_norm).detach()
                        corr = (probs.argmax(dim=-1) == y_te).sum().item()
                    else:
                        corr = 0
                else:
                    pool.route_rule = rule
                    pred = pool.predict(x_te)
                    corr = (pred.probs.argmax(dim=-1) == y_te).sum().item()

                    if rule == "default":
                        # Record test routing matrix
                        routed = pred.expert.cpu().numpy()
                        for e in routed:
                            if 0 <= e < max_e:
                                total_test_route[r, e] += 1

                        # Breakdown of misrouted samples: part (b)
                        p_owner = owner_sig.pvalue(x_te_norm).cpu().numpy() if owner_sig is not None else np.zeros(n_te)
                        for i in range(n_te):
                            e_routed = routed[i]
                            if e_routed != owner_e:
                                seed_misrouted += 1
                                misrouted_total += 1
                                expert_stolen_counts[e_routed] += 1
                                if p_owner[i] >= cfg.q:
                                    seed_mis_accept += 1
                                    misrouted_owner_accepts += 1
                                else:
                                    misrouted_owner_rejects += 1

                seed_correct[rule] += corr
                per_regime_by_rule[rule][r].append(corr / n_te)

        test_mismatch_rates.append(seed_misrouted / max(1, seed_test_total))
        per_seed_owner_accept_frac.append(seed_mis_accept / max(1, seed_misrouted) if seed_misrouted > 0 else 0.0)

        for rule in rules:
            accs_by_rule[rule].append(seed_correct[rule] / seed_test_total)

    # ----------------------------------------------------------------------- Output Results
    # Trim matrices to used experts
    max_used = max(
        int(np.where(total_train_assign.sum(axis=0) > 0)[0][-1] + 1) if (total_train_assign.sum(axis=0) > 0).any() else 1,
        int(np.where(total_test_route.sum(axis=0) > 0)[0][-1] + 1) if (total_test_route.sum(axis=0) > 0).any() else 1,
    )
    train_mat = total_train_assign[:, :max_used]
    test_mat = total_test_route[:, :max_used]

    print("\n" + "=" * 80)
    print("D7 (a): TRAINING ASSIGNMENT vs TEST ROUTING MATRICES")
    print("=" * 80)
    hdr = f"{'Regime':<24} | " + " | ".join(f"E{e:<2}" for e in range(max_used)) + " | Total"
    print("\n--- Training-Time Assignment Matrix (regime x expert trained) ---")
    print(hdr)
    print("-" * len(hdr))
    for r in range(n_regimes):
        row_str = f"R{r} ({regime_names[r][4:14]:<10}) | " + " | ".join(f"{train_mat[r, e]:<4}" for e in range(max_used))
        print(f"{row_str} | {regime_train_counts[r]}")

    print("\n--- Test-Time Routing Matrix (regime x expert routed at test) ---")
    print(hdr)
    print("-" * len(hdr))
    for r in range(n_regimes):
        row_str = f"R{r} ({regime_names[r][4:14]:<10}) | " + " | ".join(f"{test_mat[r, e]:<4}" for e in range(max_used))
        print(f"{row_str} | {regime_test_counts[r]}")

    m_mismatch, lo_mis, hi_mis = mean_ci(test_mismatch_rates)
    print(f"\nShare of test samples routed AWAY from training majority expert: {m_mismatch*100:.2f}% [{lo_mis*100:.2f}%, {hi_mis*100:.2f}%]")

    print("\n" + "=" * 80)
    print("D7 (b): MIS-ROUTED TEST SAMPLES GATE BEHAVIOR")
    print("=" * 80)
    acc_frac_m, acc_lo, acc_hi = mean_ci(per_seed_owner_accept_frac)
    rej_frac_m = 1.0 - acc_frac_m
    print(f"Total mis-routed test samples across {seeds} seeds: {misrouted_total}")
    print(f"  Owner expert ACCEPTS (p >= q) and LOSES argmax: {misrouted_owner_accepts} ({acc_frac_m*100:.2f}% [{acc_lo*100:.2f}%, {acc_hi*100:.2f}%])")
    print(f"  Owner expert REJECTS (p < q, signature underfit): {misrouted_owner_rejects} ({rej_frac_m*100:.2f}%)")

    print("\n" + "=" * 80)
    print("D7 (c): OFFLINE S1 ACCURACY UNDER ALTERNATIVE ROUTING RULES")
    print("=" * 80)
    print(f"{'Routing Rule':<20} | {'Mean Acc':<18} | {'Diff vs Default':<20} | {'Gap to Oracle (74.30%)':<22}")
    print("-" * 84)
    def_accs = accs_by_rule["default"]
    orc_accs = accs_by_rule["oracle"]
    for rule in rules:
        r_accs = accs_by_rule[rule]
        rm, rlo, rhi = mean_ci(r_accs)
        diff_def = paired_diff(r_accs, def_accs)
        diff_orc = paired_diff(r_accs, orc_accs)
        print(f"{rule:<20} | {rm*100:.2f}% [{rlo*100:.2f}, {rhi*100:.2f}] | {diff_def['mean_diff']*100:+.2f}% (p={diff_def['p_t']:.3e}) | {diff_orc['mean_diff']*100:+.2f}%")

    print("\n--- Per-Regime Breakdown Across Alternative Rules ---")
    hdr_rules = f"{'Regime':<24} | " + " | ".join(f"{r:<14}" for r in rules)
    print(hdr_rules)
    print("-" * len(hdr_rules))
    for r in range(n_regimes):
        row_vals = []
        for rule in rules:
            m_r = mean_ci(per_regime_by_rule[rule][r])[0]
            row_vals.append(f"{m_r*100:.1f}%")
        print(f"R{r} ({regime_names[r][4:14]:<10}) | " + " | ".join(f"{v:<14}" for v in row_vals))

    print("\n" + "=" * 80)
    print("D7 (d): PER-EXPERT SIGNATURE SIZE & FOREIGN SAMPLES CAPTURED ('Broad Expert Wins')")
    print("=" * 80)
    print(f"{'Expert':<8} | {'Owner Regime(s)':<22} | {'Prototypes':<12} | {'Samples Seen':<14} | {'Kappa':<10} | {'Foreign Test Samples Stolen':<28}")
    print("-" * 100)
    for eid in range(max_used):
        if eid in expert_proto_counts and len(expert_proto_counts[eid]) > 0:
            p_m = np.mean(expert_proto_counts[eid])
            s_m = np.mean(expert_sample_counts[eid])
            k_m = np.mean(expert_kappas[eid])
            reg_owned = ", ".join(f"R{rg}" for rg in sorted(expert_owner_regimes[eid])) if expert_owner_regimes[eid] else "None"
            stolen = expert_stolen_counts[eid]
            print(f"E{eid:<7} | {reg_owned:<22} | {p_m:<12.1f} | {s_m:<14.1f} | {k_m:<10.1f} | {stolen:<28}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_s1_diagnostic_d7(seeds=10)

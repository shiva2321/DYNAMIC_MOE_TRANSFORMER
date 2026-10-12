"""Evaluate candidate routing rules on dev seeds 100-109 across streams S1, S2, S3, S4.

Candidate rules (at most 4 pre-registered per Review Section I1):
1. default: current conformal-gated max logf (with supersede filter)
2. argmax_logf: pure argmax log-density across active experts
3. nearest_proto: nearest prototype cosine similarity
4. accept_then_logf: eligible = {k: p_k >= q}; max logf among eligible; fallback to argmax logf if none eligible
(Also checks accept_then_logf_prior: logf + log(seen_k / total_seen))
"""

import sys
sys.path.insert(0, ".")
import math
import numpy as np
import torch
from collections import defaultdict

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted, s3_drift, s4_outliers
from growmoe.metrics import mean_ci, paired_diff, accuracy_stats, false_novelty_rate, novel_detection_rate, auroc


def evaluate_rules_on_stream(stream_fn, stream_name, dev_seeds=range(100, 110), cfg_kwargs=None):
    if cfg_kwargs is None:
        cfg_kwargs = {}

    rules = ["default", "argmax_logf", "nearest_proto", "accept_then_logf", "accept_then_logf_prior"]
    accs = {r: [] for r in rules}
    regimes_acc = {r: defaultdict(list) for r in rules}

    # For novelty and spawn metrics, record stream metrics
    fnr_list = []
    ndr_list = []
    recalls = []

    print(f"\n--- Evaluating {stream_name} across dev seeds {min(dev_seeds)}-{max(dev_seeds)} ---")

    for seed in dev_seeds:
        stream = stream_fn(seed=seed)
        cfg = PhyloCfg(seed=seed, **cfg_kwargs)
        pool = PhyloMoE(cfg)

        # Train on stream (standard protocol)
        for _, seg, idx in stream._walk(batch_size=64, seed=seed):
            bx = seg.x[idx]
            by = seg.y[idx]
            bctx = seg.ctx[idx] if seg.ctx is not None else None
            _ = pool.predict(bx, ctx=bctx)
            _ = pool.observe(bx, by, ctx=bctx)

        # Evaluate test sets under each candidate rule
        test_sets = stream.test_sets()
        eids = list(pool.experts.keys())
        total_test = 0
        correct = {r: 0 for r in rules}

        for r_id, (x_te, y_te) in test_sets.items():
            n_te = len(y_te)
            total_test += n_te
            x_norm = pool._normalize(x_te.float())

            # Precompute expert outputs
            p_list = []
            logf_list = []
            seen_list = []
            for eid in eids:
                rec = pool.experts[eid]
                sig = rec.meta["sig"]
                p_k = sig.pvalue(x_norm)
                if rec.state == "infant":
                    p_k = torch.clamp(p_k * cfg.infant_relax, max=1.0)
                logf_k = sig.logpdf(x_norm)
                p_list.append(p_k)
                logf_list.append(logf_k)
                seen_list.append(max(1, sig.seen))

            P = torch.stack(p_list, dim=1)        # (B, K)
            LogF = torch.stack(logf_list, dim=1)  # (B, K)
            p_nov = torch.max(P, dim=1).values
            tot_seen = sum(seen_list)
            log_priors = torch.tensor([math.log(s / tot_seen) for s in seen_list], device=LogF.device)
            LogF_prior = LogF + log_priors.unsqueeze(0)

            # Nearest proto similarities
            sim_list = []
            for eid in eids:
                pr = pool.experts[eid].meta["sig"].protos
                if pr is not None and len(pr) > 0:
                    pr_norm = pr / pr.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    sim_k = (x_norm @ pr_norm.T).max(dim=1).values
                else:
                    sim_k = torch.full((n_te,), -1.0, device=x_norm.device)
                sim_list.append(sim_k)
            Sim = torch.stack(sim_list, dim=1)

            # Route each sample under each rule
            for rule in rules:
                routed = torch.zeros(n_te, dtype=torch.long)
                for i in range(n_te):
                    if rule == "default":
                        threshold_i = min(cfg.q, p_nov[i].item())
                        elig = [k for k in range(len(eids)) if P[i, k].item() >= threshold_i]
                        if not elig:
                            elig = [int(torch.argmax(P[i]).item())]
                        el_eids = [eids[k] for k in elig]
                        act = pool._filter_superseded(el_eids) or el_eids
                        routed[i] = max(act, key=lambda e: LogF[i, eids.index(e)].item())
                    elif rule == "argmax_logf":
                        routed[i] = eids[int(torch.argmax(LogF[i]).item())]
                    elif rule == "nearest_proto":
                        routed[i] = eids[int(torch.argmax(Sim[i]).item())]
                    elif rule == "accept_then_logf":
                        # Eligible = conformal p >= q
                        elig = [k for k in range(len(eids)) if P[i, k].item() >= cfg.q]
                        if elig:
                            el_eids = [eids[k] for k in elig]
                            routed[i] = max(el_eids, key=lambda e: LogF[i, eids.index(e)].item())
                        else:
                            # Fallback: highest log-density among all experts
                            routed[i] = eids[int(torch.argmax(LogF[i]).item())]
                    elif rule == "accept_then_logf_prior":
                        elig = [k for k in range(len(eids)) if P[i, k].item() >= cfg.q]
                        if elig:
                            el_eids = [eids[k] for k in elig]
                            routed[i] = max(el_eids, key=lambda e: LogF_prior[i, eids.index(e)].item())
                        else:
                            routed[i] = eids[int(torch.argmax(LogF_prior[i]).item())]

                # Compute accuracy for routed experts
                probs = torch.zeros((n_te, cfg.n_classes), dtype=torch.float32)
                for eid in eids:
                    mask_k = (routed == eid)
                    if mask_k.any():
                        probs[mask_k] = pool.experts[eid].module.proba(x_norm[mask_k]).detach()

                corr_k = (probs.argmax(dim=-1).cpu() == y_te.cpu()).sum().item()
                correct[rule] += corr_k
                regimes_acc[rule][r_id].append(corr_k / n_te)

        for rule in rules:
            accs[rule].append(correct[rule] / total_test)

    # Print summary table
    print(f"{'Rule':<24} | {'Mean Acc':<20} | {'Diff vs Default':<24} | {'p-value':<10}")
    print("-" * 84)
    def_accs = accs["default"]
    for rule in rules:
        m, lo, hi = mean_ci(accs[rule])
        diff = paired_diff(accs[rule], def_accs)
        print(f"{rule:<24} | {m*100:.2f}% [{lo*100:.2f}%, {hi*100:.2f}%] | {diff['mean_diff']*100:+.2f}% [{diff['ci_lo']*100:+.2f}%, {diff['ci_hi']*100:+.2f}%] | {diff['p_t']:.4e}")

    return accs


if __name__ == "__main__":
    dev_seeds = list(range(100, 110))
    print("=" * 80)
    print("GrowMoE R11c: Pre-Registered Routing Rule Selection on DEV SEEDS (100-109)")
    print("=" * 80)

    # 1. Evaluate S1 (Heterogeneous Types)
    s1_accs = evaluate_rules_on_stream(
        lambda seed: s1_types(seed=seed, n_train=400, n_test=150),
        "S1 Types",
        dev_seeds=dev_seeds,
        cfg_kwargs={"q": 0.02, "alpha_spawn": 0.01, "m_min": 32},
    )

    # 2. Evaluate S2 (Permuted Digits)
    s2_accs = evaluate_rules_on_stream(
        lambda seed: s2_permuted(seed=seed, n_train=400, n_test=150),
        "S2 Permuted",
        dev_seeds=dev_seeds,
        cfg_kwargs={"q": 0.05, "alpha_spawn": 0.01, "m_min": 32},
    )

    # 3. Evaluate S3 (Rotational Drift)
    s3_accs = evaluate_rules_on_stream(
        lambda seed: s3_drift(seed=seed, n_samples=2000, n_test=150),
        "S3 Drift",
        dev_seeds=dev_seeds,
        cfg_kwargs={"q": 0.05, "alpha_spawn": 0.01, "m_min": 32},
    )

    # 4. Evaluate S4 (Outliers)
    s4_accs = evaluate_rules_on_stream(
        lambda seed: s4_outliers(seed=seed, n_train=500, n_test=150),
        "S4 Outliers",
        dev_seeds=dev_seeds,
        cfg_kwargs={"q": 0.05, "alpha_spawn": 0.01, "m_min": 32},
    )

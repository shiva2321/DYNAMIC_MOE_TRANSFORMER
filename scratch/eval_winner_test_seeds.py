"""Evaluate the winning rule (accept_then_logf) vs default ONCE on TEST SEEDS 0-9 across S1-S4.

Per Section I1:
- Evaluate winner on test seeds 0-9 across S1, S2, S3, S4.
- Report mean +- 95% CI and paired diff vs default.
"""

import sys
sys.path.insert(0, ".")
import math
import time
import numpy as np
import torch
from collections import defaultdict

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted, s3_drift, s4_outliers
from growmoe.metrics import mean_ci, paired_diff, false_novelty_rate, novel_detection_rate, spawn_stats, regime_owner


def eval_test_seeds(stream_builder, stream_name, test_seeds=range(10), cfg_kwargs=None):
    if cfg_kwargs is None:
        cfg_kwargs = {}

    rules = ["default", "accept_then_logf"]
    accs = {r: [] for r in rules}

    fnr_stream = []
    ndr_stream = []
    recall_stream = []

    print(f"\n========================================================", flush=True)
    print(f"Evaluating {stream_name} on TEST SEEDS {min(test_seeds)}-{max(test_seeds)}...", flush=True)
    print(f"========================================================", flush=True)

    t0 = time.perf_counter()

    for seed in test_seeds:
        stream = stream_builder(seed)
        cfg = PhyloCfg(seed=seed, **cfg_kwargs)
        pool = PhyloMoE(cfg)

        all_p_nov = []
        all_unknown = []
        all_assigned = []
        all_true_regimes = []
        all_gen_novel = []
        all_known = []
        has_revisits = bool(len(stream.segments) > len(stream.regimes))

        onsets = {}
        for _, seg, idx in stream._walk(batch_size=64, seed=seed):
            bx = seg.x[idx]
            by = seg.y[idx]
            bctx = seg.ctx[idx] if seg.ctx is not None else None
            b = bx.shape[0]

            if seg.first_visit and seg.regime not in onsets:
                onsets[seg.regime] = pool.step

            pred = pool.predict(bx, ctx=bctx)
            rec = pool.observe(bx, by, ctx=bctx)

            is_first_batch = bool(seg.first_visit and (idx[0] == 0))
            outlier_flags = seg.is_outlier[idx].detach().cpu().numpy().tolist() if hasattr(seg, "is_outlier") and seg.is_outlier is not None else [False] * b
            gen_novel = [is_first_batch or bool(o) for o in outlier_flags]
            if has_revisits:
                known = [(not seg.first_visit and not bool(o)) for o in outlier_flags]
            else:
                known = [(not is_first_batch and not bool(o)) for o in outlier_flags]

            all_p_nov.extend(pred.p_nov.detach().cpu().numpy().tolist())
            all_unknown.extend(pred.unknown.detach().cpu().numpy().tolist())
            all_assigned.extend(rec.assigned.detach().cpu().numpy().tolist())
            all_true_regimes.extend([seg.regime] * b)
            all_gen_novel.extend(gen_novel)
            all_known.extend(known)

        unknown_arr = np.array(all_unknown, dtype=bool)
        known_mask = np.array(all_known, dtype=bool)
        try:
            fnr_stream.append(false_novelty_rate(unknown_arr, known_mask))
        except ValueError:
            fnr_stream.append(0.0)
        try:
            ndr_stream.append(novel_detection_rate(unknown_arr, known_mask))
        except ValueError:
            ndr_stream.append(1.0)

        assigned_arr = np.array(all_assigned)
        true_reg_arr = np.array(all_true_regimes)
        owner_of = regime_owner(assigned_arr, true_reg_arr)
        spwn_s = spawn_stats(pool.events, owner_of, onsets)
        recall_stream.append(spwn_s.get("recall", 1.0))

        # Evaluate test sets
        test_sets = stream.test_sets()
        eids = list(pool.experts.keys())
        total_test = 0
        correct = {r: 0 for r in rules}

        for r_id, (x_te, y_te) in test_sets.items():
            n_te = len(y_te)
            total_test += n_te
            x_norm = pool._normalize(x_te.float())

            p_list = []
            logf_list = []
            for eid in eids:
                rec_e = pool.experts[eid]
                sig = rec_e.meta["sig"]
                p_k = sig.pvalue(x_norm)
                if rec_e.state == "infant":
                    p_k = torch.clamp(p_k * cfg.infant_relax, max=1.0)
                logf_k = sig.logpdf(x_norm)
                p_list.append(p_k)
                logf_list.append(logf_k)

            P = torch.stack(p_list, dim=1)        # (B, K)
            LogF = torch.stack(logf_list, dim=1)  # (B, K)
            p_nov = torch.max(P, dim=1).values

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
                    elif rule == "accept_then_logf":
                        elig = [k for k in range(len(eids)) if P[i, k].item() >= cfg.q]
                        if elig:
                            el_eids = [eids[k] for k in elig]
                            routed[i] = max(el_eids, key=lambda e: LogF[i, eids.index(e)].item())
                        else:
                            routed[i] = eids[int(torch.argmax(LogF[i]).item())]

                probs = torch.zeros((n_te, cfg.n_classes), dtype=torch.float32)
                for eid in eids:
                    mask_k = (routed == eid)
                    if mask_k.any():
                        probs[mask_k] = pool.experts[eid].module.proba(x_norm[mask_k]).detach()

                corr_k = (probs.argmax(dim=-1).cpu() == y_te.cpu()).sum().item()
                correct[rule] += corr_k

        for rule in rules:
            accs[rule].append(correct[rule] / total_test)
        print(f"  Seed {seed} done (acc: default={accs['default'][-1]*100:.1f}%, accept_then_logf={accs['accept_then_logf'][-1]*100:.1f}%)", flush=True)

    dt = time.perf_counter() - t0
    print(f"Completed {stream_name} in {dt:.1f}s ({dt/len(test_seeds):.1f}s/seed).", flush=True)

    print(f"{'Rule':<24} | {'Mean Acc':<20} | {'Diff vs Default':<24} | {'p-value':<10}", flush=True)
    print("-" * 84, flush=True)
    def_accs = accs["default"]
    for rule in rules:
        m, lo, hi = mean_ci(accs[rule])
        diff = paired_diff(accs[rule], def_accs)
        print(f"{rule:<24} | {m*100:.2f}% [{lo*100:.2f}%, {hi*100:.2f}%] | {diff['mean_diff']*100:+.2f}% [{diff['ci_lo']*100:+.2f}%, {diff['ci_hi']*100:+.2f}%] | {diff['p_t']:.4e}", flush=True)

    fnr_m, fnr_lo, fnr_hi = mean_ci(fnr_stream)
    ndr_m, ndr_lo, ndr_hi = mean_ci(ndr_stream)
    rec_m, rec_lo, rec_hi = mean_ci(recall_stream)
    print(f"  Stream Gate: FNR={fnr_m*100:.2f}% [{fnr_lo*100:.2f}%, {fnr_hi*100:.2f}%], NDR={ndr_m*100:.2f}% [{ndr_lo*100:.2f}%, {ndr_hi*100:.2f}%], Spawn Recall={rec_m*100:.2f}% [{rec_lo*100:.2f}%, {rec_hi*100:.2f}%]", flush=True)

    return accs


if __name__ == "__main__":
    test_seeds = list(range(10))
    print("=" * 80, flush=True)
    print("GrowMoE R11c: EVALUATING WINNER (accept_then_logf) ON TEST SEEDS (0-9)", flush=True)
    print("=" * 80, flush=True)

    s1_accs = eval_test_seeds(
        lambda sd: s1_types(seed=sd, n_train=400, n_test=150),
        "S1 Types",
        test_seeds=test_seeds,
        cfg_kwargs={"q": 0.02, "alpha_spawn": 0.01, "m_min": 32},
    )

    s2_accs = eval_test_seeds(
        lambda sd: s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100),
        "S2 Permuted",
        test_seeds=test_seeds,
        cfg_kwargs={"q": 0.05, "alpha_spawn": 0.01, "m_min": 32},
    )

    s3_accs = eval_test_seeds(
        lambda sd: s3_drift(seed=sd, n_samples=2000, n_swaps=24, n_segments=16, n_test=150),
        "S3 Drift",
        test_seeds=test_seeds,
        cfg_kwargs={"q": 0.05, "alpha_spawn": 0.01, "m_min": 32},
    )

    s4_accs = eval_test_seeds(
        lambda sd: s4_outliers(seed=sd, outlier_rate=0.01, label_noise=0.1, n_regimes=5, n_train=400, n_test=100),
        "S4 Outliers",
        test_seeds=test_seeds,
        cfg_kwargs={"q": 0.05, "alpha_spawn": 0.01, "m_min": 32},
    )

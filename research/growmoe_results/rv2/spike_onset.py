"""THROWAWAY spike on DEV seeds (100-102): WHY does the pool merge regimes that overlap in input space, and would a BATCH-level fingerprint see them?

A. Regime onsets (live pool, whole stream). For the first batch of every segment: the fraction the per-sample gate flags as novel (Benjamini-Hochberg at the pool's q),
   the prequential cross-entropy of the expert the batch is routed to, and the lifecycle events fired during the segment (spawn / extend / alarm).
B. Detection power after the pool has seen ONLY regime 0: the per-sample conformal rejection rate on the other regimes versus a batch-level test on the expert's
   fingerprint (distance of a 64-sample batch mean from the mean of the expert's stored prototypes; threshold = 99.9 % quantile of the same statistic on bootstrap
   batches of the expert's own regime).
C. How much of the regime identity is in x alone: nearest-centroid (cosine) accuracy over the 10 regimes, centroids from each regime's first training segment.

    python research/growmoe_results/rv2/spike_onset.py
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "m2a"))
sys.path.insert(0, str(HERE.parent / "m0"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402
import torch.nn.functional as F                                                         # noqa: E402

import run_m2a as R                                                                     # noqa: E402
from growmoe.conformal import benjamini_hochberg                                        # noqa: E402
from spike_overlap import overlap_stream                                                # noqa: E402

torch.set_num_threads(3)
RHOS = (0.0, 0.3, 0.6, 0.85)
SEEDS = (100, 101, 102)
BS = 64


def seg_index(st, seed):
    return [k for k, _, _ in st._walk(BS, seed)]


def make_pool(st, seed):
    return R.factory("A_new", st, seed, {"steps": 5})()


def onset_run(rho, seed):
    st, _ = overlap_stream(seed, rho)
    pool = make_pool(st, seed)
    ks = seg_index(st, seed)
    seg_rows = {}
    for bi, (x, y, ctx) in enumerate(st.algo_batches(BS, seed)):
        k = ks[bi]
        rec = seg_rows.setdefault(k, {"seg": k, "regime": st.segments[k].regime, "first_visit": bool(st.segments[k].first_visit), "events": []})
        if "novel_frac" not in rec and pool.experts:
            eids = list(pool.experts)
            xn = pool._normalize(x.float())
            P, LogF = pool._gate_matrices(xn, eids)
            nov = benjamini_hochberg(P.max(1).values, pool.cfg.q)
            owner = LogF.argmax(1)
            with torch.no_grad():
                ce = torch.stack([F.cross_entropy(pool.experts[e].module(xn), y, reduction="none") for e in eids], 1)
            rec.update(novel_frac=float(nov.float().mean()), preq_ce=float(ce[torch.arange(len(y)), owner].mean()),
                       n_experts_before=len(eids), owner=int(eids[int(torch.mode(owner).values)]))
        before = len(pool.events)
        pool.observe(x, y, ctx)
        rec["events"] += [e.kind for e in list(pool.events)[before:]]
    return [seg_rows[k] for k in sorted(seg_rows)], len(pool.experts)


def detect_run(rho, seed, n_boot=2000, n_batch=100):
    st, _ = overlap_stream(seed, rho)
    pool = make_pool(st, seed)
    ks = seg_index(st, seed)
    for bi, (x, y, ctx) in enumerate(st.algo_batches(BS, seed)):
        if ks[bi] != 0:
            break
        pool.observe(x, y, ctx)
    eids = list(pool.experts)
    q = pool.cfg.q
    mu = torch.cat([pool.experts[e].meta["sig"].protos for e in eids]).mean(0)           # the expert's fingerprint: mean of its stored prototypes
    g = torch.Generator().manual_seed(seed)
    regs = list(st.regimes.values())
    own = pool._normalize(regs[0].x_te.float())

    def batches(xn, n, replace):
        idx = torch.stack([torch.randint(0, len(xn), (BS,), generator=g) if replace else torch.randperm(len(xn), generator=g)[:BS] for _ in range(n)])
        return xn[idx]

    def stat(bat):
        return (bat.mean(1) - mu).norm(dim=1)

    def per_sample_rejection(bat):
        return float(np.mean([float(benjamini_hochberg(pool._gate_matrices(b, eids)[0].max(1).values, q).float().mean()) for b in bat]))

    null = stat(batches(own, n_boot, True))
    thr = float(torch.quantile(null, 0.999))
    own_bat = batches(own, n_batch, False)
    res = {"rho": rho, "seed": seed, "experts_after_seg0": len(eids), "thr": thr, "null_mean": float(null.mean()),
           "own_batch_false_alarm": float((stat(own_bat) > thr).float().mean()), "own_sample_rejection": per_sample_rejection(own_bat),
           "foreign_batch_detect": [], "foreign_sample_rejection": []}
    for r in range(1, len(regs)):
        bat = batches(pool._normalize(regs[r].x_te.float()), n_batch, False)
        res["foreign_batch_detect"].append(float((stat(bat) > thr).float().mean()))
        res["foreign_sample_rejection"].append(per_sample_rejection(bat))
    return res


def xonly_ceiling(rho, seed):
    st, _ = overlap_stream(seed, rho)
    regs = list(st.regimes.values())
    cents = torch.stack([F.normalize(next(s for s in st.segments if s.regime == r and s.first_visit).x.mean(0), dim=0) for r in range(len(regs))])
    x = torch.cat([r.x_te for r in regs])
    lab = torch.cat([torch.full((len(r.y_te),), i, dtype=torch.long) for i, r in enumerate(regs)])
    return float(((x @ cents.T).argmax(1) == lab).float().mean())


if __name__ == "__main__":
    out = {"onset": [], "detect": [], "xonly": []}
    for rho in RHOS:
        for seed in SEEDS:
            segs, n_exp = onset_run(rho, seed)
            out["onset"].append({"rho": rho, "seed": seed, "n_experts": n_exp, "segments": segs})
            out["detect"].append(detect_run(rho, seed))
            out["xonly"].append({"rho": rho, "seed": seed, "acc": xonly_ceiling(rho, seed)})
            d = out["detect"][-1]
            print(f"rho {rho} seed {seed}: experts {n_exp}; per-sample rejection foreign {100 * np.mean(d['foreign_sample_rejection']):.1f} % vs own {100 * d['own_sample_rejection']:.1f} %; "
                  f"batch test detects {100 * np.mean(d['foreign_batch_detect']):.1f} % (false alarm {100 * d['own_batch_false_alarm']:.1f} %); x-only regime id {100 * out['xonly'][-1]['acc']:.1f} %", flush=True)
    (HERE / "spike_onset.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\nONSETS of first-visit regimes 1..9 (mean over seeds)")
    print("rho   novel frac first batch   preq CE at onset   segments with a spawn   segments with an alarm   segments with an extend   experts at end")
    for rho in RHOS:
        rr = [o for o in out["onset"] if o["rho"] == rho]
        on = [s for o in rr for s in o["segments"] if s["first_visit"] and s["seg"] > 0]
        print(f"{rho:4.2f} {100 * np.mean([s['novel_frac'] for s in on]):20.1f} % {np.mean([s['preq_ce'] for s in on]):16.2f} "
              f"{np.mean(['spawn' in s['events'] for s in on]) * 100:19.0f} % {np.mean(['alarm' in s['events'] for s in on]) * 100:21.0f} % "
              f"{np.mean(['extend' in s['events'] for s in on]) * 100:22.0f} % {np.mean([o['n_experts'] for o in rr]):14.1f}")

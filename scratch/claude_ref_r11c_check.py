"""REFERENCE ONLY (written by Claude while reviewing R11c). Re-implement; do not copy numbers. Run from the repo root:
    python scratch/claude_ref_r11c_check.py --seeds dev|test --q 0.02 --out <file>.json
It evaluates the library predict() under route_rule overrides at the benchmark configuration (q=0.02 for ALL streams) and separates the supersede filter from the rule."""
import sys, json, time, argparse
sys.path.insert(0, ".")
import numpy as np
import torch
from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted, s3_drift, s4_outliers
from growmoe.metrics import mean_ci, paired_diff

BUILD = {
    "s1": lambda sd: s1_types(seed=sd, n_train=400, n_test=150),
    "s2": lambda sd: s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100),
    "s3": lambda sd: s3_drift(seed=sd, n_samples=2000, n_swaps=24, n_segments=16, n_test=150),
    "s4": lambda sd: s4_outliers(seed=sd, outlier_rate=0.01, label_noise=0.1, n_regimes=5, n_train=400, n_test=100),
}
VARIANTS = ["default", "default_nofilter", "accept_then_logf", "atl_filter", "argmax_logf", "nearest_proto"]


def train(stream, seed, q):
    pool = PhyloMoE(PhyloCfg(seed=seed, q=q, alpha_spawn=0.01, m_min=32))
    for k, seg, idx in stream._walk(64, seed):
        bx = seg.x[idx]
        by = seg.y[idx]
        bctx = seg.ctx[idx] if seg.ctx is not None else None
        pool.predict(bx, ctx=bctx)
        pool.observe(bx, by, ctx=bctx)
    return pool


def acc_lib(pool, test_sets, rule, nofilter=False):
    pool.route_rule = rule
    if nofilter:
        pool._filter_superseded = lambda c: list(c)
    try:
        accs = []
        unk = []
        for r_id, (x_te, y_te) in test_sets.items():
            pr = pool.predict(x_te)
            accs.append(float((pr.probs.argmax(-1) == y_te).float().mean().item()))
            unk.append(pr.unknown.clone())
        return float(np.mean(accs)), torch.cat(unk)
    finally:
        if nofilter:
            del pool._filter_superseded


def acc_atl_filter(pool, test_sets, q):
    eids = list(pool.experts.keys())
    accs = []
    removed = 0
    elig_total = 0
    n = 0
    for r_id, (x_te, y_te) in test_sets.items():
        x = pool._normalize(x_te.float())
        P, L = [], []
        for e in eids:
            rec = pool.experts[e]
            sig = rec.meta["sig"]
            p = sig.pvalue(x)
            if rec.state == "infant":
                p = torch.clamp(p * pool.cfg.infant_relax, max=1.0)
            P.append(p)
            L.append(sig.logpdf(x))
        P = torch.stack(P, 1)
        L = torch.stack(L, 1)
        routed = torch.zeros(len(y_te), dtype=torch.long)
        for i in range(len(y_te)):
            elig = [eids[k] for k in range(len(eids)) if P[i, k].item() >= q]
            if elig:
                act = pool._filter_superseded(elig)
                elig_total += 1
                removed += int(len(act) < len(elig))
                routed[i] = max(act, key=lambda e: L[i, eids.index(e)].item())
            else:
                routed[i] = eids[int(torch.argmax(L[i]).item())]
        n += len(y_te)
        probs = torch.zeros((len(y_te), pool.cfg.n_classes))
        for e in eids:
            m = routed == e
            if m.any():
                probs[m] = pool.experts[e].module.proba(x[m]).detach()
        accs.append(float((probs.argmax(-1) == y_te).float().mean().item()))
    return float(np.mean(accs)), removed, elig_total


def filter_stats(pool, test_sets, q):
    """Under the default rule's eligibility (p >= min(q,p_nov)), how often does the supersede filter remove an eligible expert?"""
    eids = list(pool.experts.keys())
    removed = 0
    total = 0
    for r_id, (x_te, y_te) in test_sets.items():
        x = pool._normalize(x_te.float())
        P = []
        for e in eids:
            rec = pool.experts[e]
            p = rec.meta["sig"].pvalue(x)
            if rec.state == "infant":
                p = torch.clamp(p * pool.cfg.infant_relax, max=1.0)
            P.append(p)
        P = torch.stack(P, 1)
        pn = P.max(1).values
        for i in range(len(y_te)):
            thr = min(q, pn[i].item())
            elig = [eids[k] for k in range(len(eids)) if P[i, k].item() >= thr]
            act = pool._filter_superseded(elig)
            total += 1
            removed += int(len(act) < len(elig))
    return removed, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--streams", default="s1,s2,s3,s4")
    ap.add_argument("--q", type=float, default=0.02)
    ap.add_argument("--out", default="r11c_check_test.json")
    a = ap.parse_args()
    seeds = list(range(10)) if a.seeds == "test" else list(range(100, 110))
    res = {}
    for sname in a.streams.split(","):
        t0 = time.perf_counter()
        rows = []
        for sd in seeds:
            st = BUILD[sname](sd)
            pool = train(st, sd, a.q)
            ts = st.test_sets()
            row = {"seed": sd, "n_experts": len(pool.experts),
                   "n_superseding": sum(1 for e in pool.experts.values() if e.meta.get("supersedes") is not None)}
            unk_ref = None
            for v, rule, nf in [("default", "default", False), ("default_nofilter", "default", True),
                                ("accept_then_logf", "accept_then_logf", False),
                                ("argmax_logf", "argmax_logf", False), ("nearest_proto", "nearest_proto", False)]:
                acc, unk = acc_lib(pool, ts, rule, nf)
                row[v] = acc
                if unk_ref is None:
                    unk_ref = unk
                else:
                    row["unknown_identical_" + v] = bool(torch.equal(unk_ref, unk))
            row["atl_filter"], row["atl_filter_removed"], row["atl_filter_elig"] = acc_atl_filter(pool, ts, a.q)
            row["default_filter_removed"], row["default_filter_total"] = filter_stats(pool, ts, a.q)
            rows.append(row)
            print(f"{sname} seed {sd}: " + " ".join(f"{v}={row[v]*100:.1f}" for v in VARIANTS)
                  + f" | experts={row['n_experts']} superseding={row['n_superseding']}"
                  + f" | filter removed {row['default_filter_removed']}/{row['default_filter_total']}", flush=True)
        res[sname] = rows
        print(f"[{sname}] {time.perf_counter()-t0:.0f}s", flush=True)
        base = [r["default"] for r in rows]
        print(f"=== {sname} (q={a.q}, seeds {seeds[0]}-{seeds[-1]}) ===")
        for v in VARIANTS:
            m, lo, hi = mean_ci([r[v] for r in rows])
            d = paired_diff([r[v] for r in rows], base)
            print(f"{v:18s} {m*100:6.2f}% [{lo*100:6.2f},{hi*100:6.2f}]  diff vs default {d['mean_diff']*100:+6.2f} [{d['ci_lo']*100:+6.2f},{d['ci_hi']*100:+6.2f}] p={d['p_t']:.2e}", flush=True)
        ident = all(all(v for k, v in r.items() if k.startswith("unknown_identical")) for r in rows)
        print(f"unknown flags identical across rules: {ident}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

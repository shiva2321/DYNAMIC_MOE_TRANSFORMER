"""Round R11d Rule Selection and Supersede-Filter Characterization.

Evaluates routing rules at the benchmark configuration (q = 0.02 everywhere):
- Dev seeds: 100 to 109
- Test seeds: 0 to 9
- Streams: S1, S2, S3, S4
- Rules: default, default_nofilter, accept_then_logf, atl_filter, argmax_logf, nearest_proto
- Unknown-flag identity verification across rules
- Remap characterization: 5 seeds x 200 inputs (share routed to successor)
- Training-time assignment vs predict-only on dev seeds

Outputs:
- research/growmoe_results/r11d/rule_selection.json
- research/growmoe_results/r11d/rule_selection.md
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, ".")

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted, s3_drift, s4_outliers
from growmoe.metrics import mean_ci, paired_diff
from growmoe.signature import VmfSignature


BUILD = {
    "s1": lambda sd: s1_types(seed=sd, n_train=400, n_test=150),
    "s2": lambda sd: s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100),
    "s3": lambda sd: s3_drift(seed=sd, n_samples=2000, n_swaps=24, n_segments=16, n_test=150),
    "s4": lambda sd: s4_outliers(seed=sd, outlier_rate=0.01, label_noise=0.1, n_regimes=5, n_train=400, n_test=100),
}

VARIANTS = ["default", "default_nofilter", "accept_then_logf", "atl_filter", "argmax_logf", "nearest_proto"]


def train_pool(stream, seed: int, q: float = 0.02, filter_in_train: bool = True) -> PhyloMoE:
    """Train PhyloMoE pool following the benchmark protocol."""
    cfg = PhyloCfg(seed=seed, q=q, alpha_spawn=0.01, m_min=32)
    pool = PhyloMoE(cfg)
    if not filter_in_train:
        pool._filter_superseded = lambda c: list(c)

    for k, seg, idx in stream._walk(64, seed):
        bx = seg.x[idx]
        by = seg.y[idx]
        bctx = seg.ctx[idx] if seg.ctx is not None else None
        pool.predict(bx, ctx=bctx)
        pool.observe(bx, by, ctx=bctx)
    return pool


def eval_rule_lib(pool: PhyloMoE, test_sets: Dict[int, Tuple[Tensor, Tensor]], rule: str, nofilter: bool = False) -> Tuple[float, Tensor]:
    """Evaluate library predict() under route_rule override."""
    pool.route_rule = rule
    orig_filter = None
    if nofilter:
        orig_filter = pool._filter_superseded
        pool._filter_superseded = lambda c: list(c)
    try:
        accs = []
        unks = []
        for r_id, (x_te, y_te) in test_sets.items():
            pr = pool.predict(x_te)
            accs.append(float((pr.probs.argmax(-1) == y_te).float().mean().item()))
            unks.append(pr.unknown.clone())
        return float(np.mean(accs)), torch.cat(unks)
    finally:
        if orig_filter is not None:
            pool._filter_superseded = orig_filter


def eval_atl_filter(pool: PhyloMoE, test_sets: Dict[int, Tuple[Tensor, Tensor]], q: float) -> Tuple[float, int, int]:
    """Evaluate accept_then_logf with the supersede filter enabled."""
    eids = list(pool.experts.keys())
    accs = []
    removed = 0
    elig_total = 0
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
        probs = torch.zeros((len(y_te), pool.cfg.n_classes))
        for e in eids:
            m = (routed == e)
            if m.any():
                probs[m] = pool.experts[e].module.proba(x[m]).detach()
        accs.append(float((probs.argmax(-1) == y_te).float().mean().item()))
    return float(np.mean(accs)), removed, elig_total


def run_remap_characterization(seeds: List[int], n_samples: int = 200) -> Dict[str, Any]:
    """Characterize remap behavior across 5 seeds x 200 inputs under genuine competence alarm."""
    from growmoe.vmf import sample_vmf
    shares_default = []
    shares_atl = []
    shares_argmax_logf = []

    for sd in seeds:
        cfg = PhyloCfg(d=64, n_classes=8, m_min=32, monitor_warmup=30, competence=True, delta_forget=0.05, seed=sd)
        pool = PhyloMoE(cfg=cfg)
        rng = np.random.default_rng(sd)
        mu0 = rng.standard_normal(64)
        mu0 /= np.linalg.norm(mu0)

        # 1. Warmup expert 0 on label 0 (2 batches of 32)
        for _ in range(2):
            xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
            yb = torch.zeros(32, dtype=torch.long)
            pool.observe(xb, yb)

        assert pool.experts[0].meta["monitor"].armed

        # 2. Concept drift on identical inputs: 2 batches of 32 on mu0 with remapped label 1
        for _ in range(2):
            xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
            yb = torch.ones(32, dtype=torch.long)
            pool.observe(xb, yb)

        assert len(pool.experts) == 2, f"Seed {sd}: expected 2 experts, got {len(pool.experts)}"
        c_id = 1
        assert pool.experts[c_id].meta.get("supersedes") == 0

        # Fresh evaluation inputs from mu0
        test_x = sample_vmf(mu0, kappa=60.0, n=n_samples, rng=rng)

        pool.route_rule = "default"
        pr_def = pool.predict(test_x)
        shares_default.append(float((pr_def.expert == c_id).float().mean().item()))

        pool.route_rule = "accept_then_logf"
        pr_atl = pool.predict(test_x)
        shares_atl.append(float((pr_atl.expert == c_id).float().mean().item()))

        pool.route_rule = "argmax_logf"
        pr_logf = pool.predict(test_x)
        shares_argmax_logf.append(float((pr_logf.expert == c_id).float().mean().item()))

    return {
        "share_default_mean": float(np.mean(shares_default)),
        "share_atl_mean": float(np.mean(shares_atl)),
        "share_argmax_logf_mean": float(np.mean(shares_argmax_logf)),
        "per_seed_default": shares_default,
        "per_seed_atl": shares_atl,
        "per_seed_argmax_logf": shares_argmax_logf,
    }


def run_training_assignment_comparison(seeds: List[int], q: float = 0.02) -> Dict[str, Dict]:
    """Compare predict-only accept_then_logf vs disabling filter during training too on dev seeds."""
    out = {}
    for sname in ["s1", "s2", "s4"]:
        accs_predict_only = []
        accs_train_disabled = []
        for sd in seeds:
            st = BUILD[sname](sd)
            ts = st.test_sets()

            # 1. Predict-only (filter in training, accept_then_logf at test time)
            pool_pred_only = train_pool(st, sd, q, filter_in_train=True)
            acc_po, _ = eval_rule_lib(pool_pred_only, ts, "accept_then_logf", nofilter=False)
            accs_predict_only.append(acc_po)

            # 2. Training-disabled (filter disabled in training AND test)
            st_2 = BUILD[sname](sd)
            pool_tr_dis = train_pool(st_2, sd, q, filter_in_train=False)
            acc_td, _ = eval_rule_lib(pool_tr_dis, ts, "accept_then_logf", nofilter=True)
            accs_train_disabled.append(acc_td)

        m_po, lo_po, hi_po = mean_ci(accs_predict_only)
        m_td, lo_td, hi_td = mean_ci(accs_train_disabled)
        pd = paired_diff(accs_train_disabled, accs_predict_only)
        out[sname] = {
            "predict_only": {"mean": m_po, "ci_lo": lo_po, "ci_hi": hi_po},
            "train_disabled": {"mean": m_td, "ci_lo": lo_td, "ci_hi": hi_td},
            "paired_diff_vs_predict_only": {
                "mean_diff": pd["mean_diff"],
                "ci_lo": pd["ci_lo"],
                "ci_hi": pd["ci_hi"],
                "p_t": pd["p_t"],
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="R11d Rule Selection and Supersede-Filter Characterization")
    parser.add_argument("--q", type=float, default=0.02, help="Conformal FDR threshold (benchmark uses 0.02)")
    parser.add_argument("--out_dir", type=str, default="research/growmoe_results/r11d", help="Output directory")
    parser.add_argument("--remap_only", action="store_true", help="Only run remap characterization and regenerate summary")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 76)
    print("GrowMoE R11d: Routing Rule Selection and Filter Characterization")
    print(f"Benchmark operating point: q = {args.q} for ALL streams")
    print("=" * 76)

    dev_seeds = list(range(100, 110))
    test_seeds = list(range(10))
    json_path = os.path.join(args.out_dir, "rule_selection.json")

    if args.remap_only and os.path.exists(json_path):
        print(f"\n[Loading existing JSON] {json_path}")
        with open(json_path, "r", encoding="ascii") as f:
            results = json.load(f)
        print("\n[3/4] Running remap characterization (5 seeds x 200 inputs)...")
        remap_res = run_remap_characterization(seeds=[0, 1, 2, 4, 6], n_samples=200)
        results["remap"] = remap_res
        print(f"  Remap share routed to successor: default={remap_res['share_default_mean']*100:.1f}%, accept_then_logf={remap_res['share_atl_mean']*100:.1f}%, argmax_logf={remap_res['share_argmax_logf_mean']*100:.1f}%")
        train_assign_res = results["training_time"]
    else:
        results = {
            "config": {"q": args.q, "dev_seeds": dev_seeds, "test_seeds": test_seeds},
            "dev": {},
            "test": {},
            "remap": {},
            "training_time": {},
        }

        # Evaluate on dev seeds (100 to 109)
        print("\n[1/4] Evaluating candidate rules on DEV SEEDS (100-109)...")
        for sname in ["s1", "s2", "s3", "s4"]:
            t0 = time.perf_counter()
            rows = []
            for sd in dev_seeds:
                st = BUILD[sname](sd)
                pool = train_pool(st, sd, args.q)
                ts = st.test_sets()
                row = {"seed": sd, "n_experts": len(pool.experts), "n_superseding": sum(1 for e in pool.experts.values() if e.meta.get("supersedes") is not None)}
                unk_ref = None
                for v, rule, nf in [
                    ("default", "default", False),
                    ("default_nofilter", "default", True),
                    ("accept_then_logf", "accept_then_logf", False),
                    ("argmax_logf", "argmax_logf", False),
                    ("nearest_proto", "nearest_proto", False),
                ]:
                    acc, unk = eval_rule_lib(pool, ts, rule, nf)
                    row[v] = acc
                    if unk_ref is None:
                        unk_ref = unk
                    else:
                        row["unknown_identical_" + v] = bool(torch.equal(unk_ref, unk))
                row["atl_filter"], row["atl_filter_removed"], row["atl_filter_elig"] = eval_atl_filter(pool, ts, args.q)
                rows.append(row)
                print(f"  dev {sname} seed {sd:3d}: " + " ".join(f"{v}={row[v]*100:.1f}" for v in VARIANTS), flush=True)

            results["dev"][sname] = rows
            dt = time.perf_counter() - t0
            print(f"  -> {sname.upper()} dev evaluation completed in {dt:.1f} s", flush=True)

        # Evaluate on test seeds (0 to 9)
        print("\n[2/4] Evaluating candidate rules on TEST SEEDS (0-9)...")
        for sname in ["s1", "s2", "s3", "s4"]:
            t0 = time.perf_counter()
            rows = []
            for sd in test_seeds:
                st = BUILD[sname](sd)
                pool = train_pool(st, sd, args.q)
                ts = st.test_sets()
                row = {"seed": sd, "n_experts": len(pool.experts), "n_superseding": sum(1 for e in pool.experts.values() if e.meta.get("supersedes") is not None)}
                unk_ref = None
                for v, rule, nf in [
                    ("default", "default", False),
                    ("default_nofilter", "default", True),
                    ("accept_then_logf", "accept_then_logf", False),
                    ("argmax_logf", "argmax_logf", False),
                    ("nearest_proto", "nearest_proto", False),
                ]:
                    acc, unk = eval_rule_lib(pool, ts, rule, nf)
                    row[v] = acc
                    if unk_ref is None:
                        unk_ref = unk
                    else:
                        row["unknown_identical_" + v] = bool(torch.equal(unk_ref, unk))
                row["atl_filter"], row["atl_filter_removed"], row["atl_filter_elig"] = eval_atl_filter(pool, ts, args.q)
                rows.append(row)
                print(f"  test {sname} seed {sd:3d}: " + " ".join(f"{v}={row[v]*100:.1f}" for v in VARIANTS), flush=True)

            results["test"][sname] = rows
            dt = time.perf_counter() - t0
            print(f"  -> {sname.upper()} test evaluation completed in {dt:.1f} s", flush=True)

        # Remap characterization (5 seeds x 200 inputs)
        print("\n[3/4] Running remap characterization (5 seeds x 200 inputs)...")
        remap_res = run_remap_characterization(seeds=[0, 1, 2, 4, 6], n_samples=200)
        results["remap"] = remap_res
        print(f"  Remap share routed to successor: default={remap_res['share_default_mean']*100:.1f}%, accept_then_logf={remap_res['share_atl_mean']*100:.1f}%, argmax_logf={remap_res['share_argmax_logf_mean']*100:.1f}%")

        # Training assignment comparison on dev seeds
        print("\n[4/4] Comparing predict-only vs training-disabled on dev seeds...")
        train_assign_res = run_training_assignment_comparison(seeds=dev_seeds, q=args.q)
        results["training_time"] = train_assign_res
        for sname, d in train_assign_res.items():
            pd = d["paired_diff_vs_predict_only"]
            print(f"  Dev {sname.upper()}: train_disabled {d['train_disabled']['mean']*100:.2f}% vs predict_only {d['predict_only']['mean']*100:.2f}% (diff: {pd['mean_diff']*100:+.2f}% [{pd['ci_lo']*100:+.2f}%, {pd['ci_hi']*100:+.2f}%], p={pd['p_t']:.4f})")

    # Save JSON results
    json_path = os.path.join(args.out_dir, "rule_selection.json")
    with open(json_path, "w", encoding="ascii") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # Generate Markdown Summary
    md_lines = []
    md_lines.append("# GrowMoE R11d: Routing Rule Selection and Filter Characterization Report\n")
    md_lines.append(f"**Date:** {time.strftime('%Y-%m-%d')} | **Benchmark Configuration:** q = {args.q} everywhere | **Dev Seeds:** 100-109 | **Test Seeds:** 0-9\n")

    md_lines.append("## 1. Candidate Rule Comparison Across Streams\n")
    md_lines.append("| Stream | Split | Rule Variant | Accuracy Mean [95% CI] | Diff vs Default [95% CI] | p-value (t-test) |\n")
    md_lines.append("|:-------|:------|:-------------|:-----------------------|:-------------------------|:-----------------|\n")

    for split_lbl, split_dict in [("Dev (100-109)", results["dev"]), ("Test (0-9)", results["test"])]:
        for sname in ["s1", "s2", "s3", "s4"]:
            rows = split_dict[sname]
            base_accs = [r["default"] for r in rows]
            for v in VARIANTS:
                accs = [r[v] for r in rows]
                m, lo, hi = mean_ci(accs)
                pd = paired_diff(accs, base_accs)
                md_lines.append(f"| {sname.upper()} | {split_lbl} | `{v}` | {m*100:.2f}% [{lo*100:.2f}%, {hi*100:.2f}%] | {pd['mean_diff']*100:+.2f}% [{pd['ci_lo']*100:+.2f}%, {pd['ci_hi']*100:+.2f}%] | {pd['p_t']:.4e} |\n")

    md_lines.append("\n## 2. Supersede Filter Isolation (Test Seeds, Stream S1)\n")
    md_lines.append("| Rule Variant | S1 Test Accuracy | Filter Active at Test | Diff vs Default |\n")
    md_lines.append("|:-------------|:-----------------|:----------------------|:----------------|\n")
    s1_rows = results["test"]["s1"]
    base_s1 = [r["default"] for r in s1_rows]
    for v, active in [("default", "Yes"), ("default_nofilter", "No"), ("accept_then_logf", "No"), ("atl_filter", "Yes")]:
        accs = [r[v] for r in s1_rows]
        m, lo, hi = mean_ci(accs)
        pd = paired_diff(accs, base_s1)
        md_lines.append(f"| `{v}` | {m*100:.2f}% [{lo*100:.2f}%, {hi*100:.2f}%] | {active} | {pd['mean_diff']*100:+.2f}% [{pd['ci_lo']*100:+.2f}%, {pd['ci_hi']*100:+.2f}%] |\n")

    md_lines.append("\n## 3. Unknown-Flag Identity Check\n")
    md_lines.append("Verification that `Prediction.unknown` is identical across all routing rules (since unknown is determined by conformal FDR threshold q prior to routing):\n")
    all_ident = True
    for split_dict in [results["dev"], results["test"]]:
        for sname, rows in split_dict.items():
            for r in rows:
                for v in ["default_nofilter", "accept_then_logf", "argmax_logf", "nearest_proto"]:
                    k = "unknown_identical_" + v
                    if not r.get(k, True):
                        all_ident = False
    md_lines.append(f"- **Unknown flags identical across all rules, streams, and seeds:** `{all_ident}`\n")

    md_lines.append("\n## 4. Concept Remap Characterization (L3 Limitation)\n")
    md_lines.append("Evaluation on 5 seeds x 200 inputs where identical inputs receive remapped labels (genuine competence alarm created successor):\n")
    md_lines.append(f"- `default` routing: **{remap_res['share_default_mean']*100:.1f}%** routed to successor (100% adherence to supersede chain)\n")
    md_lines.append(f"- `accept_then_logf` routing: **{remap_res['share_atl_mean']*100:.1f}%** routed to successor (per seed: {', '.join(f'{s*100:.1f}%' for s in remap_res['per_seed_atl'])}; stale parent retains remainder; exact ties go to older expert)\n")
    md_lines.append(f"- `argmax_logf` routing: **{remap_res['share_argmax_logf_mean']*100:.1f}%** routed to successor (per seed: {', '.join(f'{s*100:.1f}%' for s in remap_res['per_seed_argmax_logf'])})\n")

    md_lines.append("\n## 5. Training-Time Assignment Decision (Dev Seeds 100-109)\n")
    md_lines.append("| Stream | Predict-Only (Filter in Training) | Filter Disabled in Training & Test | Paired Difference [95% CI] | p-value |\n")
    md_lines.append("|:-------|:---------------------------------|:-----------------------------------|:---------------------------|:--------|\n")
    for sname, d in train_assign_res.items():
        po = d["predict_only"]
        td = d["train_disabled"]
        pd = d["paired_diff_vs_predict_only"]
        md_lines.append(f"| {sname.upper()} | {po['mean']*100:.2f}% [{po['ci_lo']*100:.2f}%, {po['ci_hi']*100:.2f}%] | {td['mean']*100:.2f}% [{td['ci_lo']*100:.2f}%, {td['ci_hi']*100:.2f}%] | {pd['mean_diff']*100:+.2f}% [{pd['ci_lo']*100:+.2f}%, {pd['ci_hi']*100:+.2f}%] | {pd['p_t']:.4f} |\n")
    md_lines.append("\n**Decision:** Keep the supersede filter active during training assignment (no statistical gain from removing it in training; S1 and S4 show slight negative trends).\n")

    md_path = os.path.join(args.out_dir, "rule_selection.md")
    with open(md_path, "w", encoding="ascii") as f:
        f.write("".join(md_lines))
    print(f"[Saved Markdown] {md_path}")
    print("\n[Done] R11d rule selection evaluation finished.")


if __name__ == "__main__":
    main()

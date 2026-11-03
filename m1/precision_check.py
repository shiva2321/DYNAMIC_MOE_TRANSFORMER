"""Precision check of the timed pass (D-97 as corrected by D-100): is the timing of seeds 810-812 precise enough to stop at 3 timed seeds?

For each time-based held-default rule of m1/verdict.py the script takes the quantity the rule thresholds, forms the per-seed relative paired difference
(candidate minus reference over reference) and the half-width of its 95 % t-interval over the timed seeds present (n-1 degrees of freedom), in percentage points.
It looks ONLY at the width of the intervals; the decision never depends on the sign or size of any difference.
  growth    H_shipped vs H_shipped+gap   update wall-clock, pooled over all beds per seed          threshold 15 %  -> limit 7.5 points
  skip      H_shipped vs H_shipped+skip  update wall-clock, on each of B2, B3seq, B3mix, B3drift   threshold 10 %  -> limit 5 points
  lda_head  H_U-real  vs H_ULDA          answer latency at batch 1, on each of B3seq, B3mix, B3drift  threshold 10 % -> limit 5 points
Every limit is half of the rule's own threshold. All intervals within their limits -> STOP at 3 timed seeds; otherwise CONTINUE with seeds 813 and 814.
Usage:  python precision_check.py --cells results/m1/scored --seeds 810 811 812 --out results/m1/precision_check_after_812.json
        python precision_check.py --selftest
"""

import argparse
import glob
import json
import math
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

LIMIT_FRACTION_OF_THRESHOLD = 0.5
RULES = {
    "growth": {"ref": "H_shipped", "cand": "H_shipped+gap", "quantity": "wall", "threshold": 0.15, "beds": "pooled"},
    "skip": {"ref": "H_shipped", "cand": "H_shipped+skip", "quantity": "wall", "threshold": 0.10, "beds": ["B2", "B3seq", "B3mix", "B3drift"]},
    "lda_head": {"ref": "H_U-real", "cand": "H_ULDA", "quantity": "lat1", "threshold": 0.10, "beds": ["B3seq", "B3mix", "B3drift"]},
}


def bed_key(rec: Dict[str, Any]) -> str:
    return f"B1_n{rec['n_regimes']}" if rec["bed"] == "B1" else str(rec["bed"])


def quantity(rec: Dict[str, Any], kind: str) -> Optional[float]:
    if kind == "wall":
        v = rec.get("update_cost", {}).get("wall_clock_total")
    else:
        v = rec.get("answer_cost", {}).get("1", {}).get("latency_median_s")
    return None if v is None else float(v)


def load(cells_dir: str, seeds: List[int]) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for f in glob.glob(f"{cells_dir}/*.json"):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("seed") in seeds and r.get("split") == "report" and str(r.get("arm", "")).startswith("H_"):
            out[(bed_key(r), r["arm"], r["seed"])] = r
    return out


def half_width_points(rel: List[float]) -> float:
    n = len(rel)
    if n < 2:
        return float("inf")
    return float(stats.t.ppf(0.975, n - 1) * np.std(rel, ddof=1) / math.sqrt(n) * 100.0)


def check(cells: Dict[tuple, Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    incomplete: List[str] = []
    for name, rule in RULES.items():
        limit = rule["threshold"] * LIMIT_FRACTION_OF_THRESHOLD * 100.0
        if rule["beds"] == "pooled":
            beds_present = sorted({k[0] for k in cells})
            rel = []
            for s in seeds:
                cand = ref = 0.0
                for b in beds_present:
                    rc, rr = cells.get((b, rule["cand"], s)), cells.get((b, rule["ref"], s))
                    if rc is None or rr is None:
                        incomplete.append(f"{name} {b} seed {s}")
                        continue
                    cand += quantity(rc, rule["quantity"]) or 0.0
                    ref += quantity(rr, rule["quantity"]) or 0.0
                rel.append((cand - ref) / ref if ref > 0 else float("nan"))
            groups = {"pooled over beds": rel}
        else:
            groups = {}
            for b in rule["beds"]:
                rel = []
                for s in seeds:
                    rc, rr = cells.get((b, rule["cand"], s)), cells.get((b, rule["ref"], s))
                    if rc is None or rr is None or quantity(rc, rule["quantity"]) is None or quantity(rr, rule["quantity"]) in (None, 0.0):
                        incomplete.append(f"{name} {b} seed {s}")
                        continue
                    rel.append((quantity(rc, rule["quantity"]) - quantity(rr, rule["quantity"])) / quantity(rr, rule["quantity"]))
                groups[b] = rel
        for label, rel in groups.items():
            hw = half_width_points(rel)
            results.append({"rule": name, "bed": label, "n_seeds": len(rel), "half_width_points": hw, "limit_points": limit, "within_limit": bool(hw <= limit)})
    decided = not incomplete and all(r["n_seeds"] == len(seeds) for r in results)
    stop = decided and all(r["within_limit"] for r in results)
    return {"seeds": seeds, "intervals": results, "incomplete": sorted(set(incomplete)),
            "decision": ("INCOMPLETE: no decision" if not decided else ("STOP at 3 timed seeds" if stop else "CONTINUE with seeds 813 and 814"))}


def selftest() -> None:
    # relative differences 0.10, 0.12, 0.08: sd 0.02, t(0.975, 2) = 4.302653, half-width = 4.302653 * 0.02 / sqrt(3) * 100 = 4.968 points
    assert abs(half_width_points([0.10, 0.12, 0.08]) - 4.9686) < 1e-3
    assert half_width_points([0.1]) == float("inf")

    def rec(bed, n, arm, seed, wall, lat):
        return {"bed": bed, "n_regimes": n, "arm": arm, "seed": seed, "split": "report",
                "update_cost": {"wall_clock_total": wall}, "answer_cost": {"1": {"latency_median_s": lat}}}
    cells = {}
    seeds = [1, 2, 3]
    ref_w = {1: 100.0, 2: 110.0, 3: 90.0}
    for s in seeds:
        for b, n in (("B1", 100), ("B2", 6), ("B3seq", 22), ("B3mix", 22), ("B3drift", 22)):
            key = f"B1_n{n}" if b == "B1" else b
            for arm, f in (("H_shipped", 1.0), ("H_shipped+gap", 0.80), ("H_shipped+skip", 0.90), ("H_U-real", 1.0), ("H_ULDA", 1.05)):
                cells[(key, arm, s)] = rec(b, n, arm, s, ref_w[s] * f, 0.001 * f)
    out = check(cells, seeds)
    assert out["decision"] == "STOP at 3 timed seeds", out          # perfectly constant ratios: zero width
    cells[("B3seq", "H_shipped+skip", 3)] = rec("B3seq", 22, "H_shipped+skip", 3, 90.0 * 0.60, 0.001)   # one noisy bed
    out2 = check(cells, seeds)
    assert out2["decision"] == "CONTINUE with seeds 813 and 814", out2
    del cells[("B3mix", "H_ULDA", 2)]
    assert check(cells, seeds)["decision"].startswith("INCOMPLETE")
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/m1/scored")
    ap.add_argument("--seeds", type=int, nargs="+", default=[810, 811, 812])
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    res = check(load(a.cells, a.seeds), a.seeds)
    for r in res["intervals"]:
        print(f"{r['rule']:<9}{r['bed']:<18} n={r['n_seeds']} half-width {r['half_width_points']:6.2f} pts (limit {r['limit_points']:.1f}) {'ok' if r['within_limit'] else 'TOO WIDE'}")
    print("incomplete:", res["incomplete"])
    print("DECISION:", res["decision"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()

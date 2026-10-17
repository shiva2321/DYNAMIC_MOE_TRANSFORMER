"""RV2-D (routing v2 stage 4): the statements S4-1 to S4-9 on per-cell run records, and the decision rule (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 8).

A record carries ``family``, ``stream``, ``arm``, ``seed``, ``acc`` (router off), ``router`` ({"off": {"acc"}, "<mode>-<margin>": {"acc", ...}}), ``n_experts``, ``params``,
``grow_events``, ``final_hidden``, ``forgetting`` (operator-by-operator arithmetic), ``grow_stats``, ``inner_steps``. Differences are in points (x100), paired by seed.
A multi-stream statement is PASS when it holds on all, FAIL on none, PARTIAL otherwise, NOT MEASURABLE when data are missing (the rule of the earlier stages).
"""

import math
from typing import Optional, Sequence

import numpy as np

from .fairfight import _combine
from .metrics import paired_diff

_POINTS = 100.0
IDS = tuple(f"S4-{i}" for i in range(1, 10))
EXPECTED_RV2D = {k: "PASS" for k in IDS}
S3_STREAMS = ("F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5")
GUARD_STREAMS = tuple(s for s in S3_STREAMS if s != "OV-0.85")
BARS_RV2D = {"S4_1_lo": 5.0, "S4_2_lo": 4.0, "S4_3": -0.5, "S4_4_gain": 5.0, "S4_4_forgetting": 9.0, "S4_4_min_experts": 5, "S4_4_seeds_below_allowed": 1,
             "S4_5_gain": -0.5, "S4_5_ratio": 1.2, "S4_6_frac": 0.9, "S4_6_seed_share": 13 / 15, "S4_6_width": (64, 256), "S4_6_cost": 0.05,
             "S4_7_gain": -0.5, "S4_7_grow": 1.0, "S4_7_params": 0.25, "S4_8_acc": 44.0, "S4_8_forgetting": 8.0}


def _fin(x: float) -> Optional[float]:
    return float(x) if math.isfinite(x) else None


def _index(runs: Sequence[dict]) -> dict:
    ix: dict = {}
    for r in runs:
        ix.setdefault((r["family"], r["stream"], r["arm"]), {})[int(r["seed"])] = r
    return ix


def _diff(a: Optional[dict], b: Optional[dict], fa, fb) -> Optional[dict]:
    """Paired difference fa(record of a) - fb(record of b) in points over the seeds both have; None when there is no common seed."""
    if not a or not b:
        return None
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = paired_diff([fa(a[s]) * _POINTS for s in seeds], [fb(b[s]) * _POINTS for s in seeds])
    return {"n": d["n"], "mean_diff_points": d["mean_diff"], "ci_lo": _fin(d["ci_lo"]), "ci_hi": _fin(d["ci_hi"])}


def _holds(d: Optional[dict], condition) -> Optional[dict]:
    if d is not None:
        d["holds"] = bool(condition(d))
    return d


def _mean(recs: dict, f) -> float:
    return float(np.mean([f(r) for r in recs.values()]))


def rv2d_verdicts(runs: Sequence[dict], bars: dict, golden: Optional[dict] = None, expected: Optional[dict] = None) -> dict:
    bars = {**BARS_RV2D, **bars}
    sel = bars["router_setting"]
    on, off, acc = (lambda r: r["router"][sel]["acc"]), (lambda r: r["router"]["off"]["acc"]), (lambda r: r["acc"])
    ix = _index(runs)
    g = lambda family, stream, arm: ix.get((family, stream, arm))                                                   # noqa: E731
    detail = {}

    def router_gain(family, stream):
        recs = g(family, stream, "S3")
        return _diff(recs, recs, on, off)

    detail["S4-1"] = _holds(router_gain("arith-seq", "-"), lambda d: d["ci_lo"] is not None and d["ci_lo"] >= bars["S4_1_lo"])
    detail["S4-2"] = _holds(router_gain("s3", "OV-0.85"), lambda d: d["ci_lo"] is not None and d["ci_lo"] >= bars["S4_2_lo"])
    detail["S4-3"] = {f"{f}:{s}": _holds(router_gain(f, s), lambda d: d["mean_diff_points"] >= bars["S4_3"]) for f, s in [("s3", s) for s in GUARD_STREAMS] + [("banking", "-")]}

    s3, t = g("arith-seq", "-", "S3"), g("arith-seq", "-", "S3+T")
    d4 = _diff(t, s3, acc, acc)
    if d4 is not None:
        forget = _POINTS * _mean(t, lambda r: r["forgetting"])
        below = sum(r["n_experts"] < bars["S4_4_min_experts"] for r in t.values())
        d4.update(forgetting_points=forget, seeds_below_min_experts=below, n_seeds=len(t))
        d4["holds"] = bool(d4["mean_diff_points"] >= bars["S4_4_gain"] and forget <= bars["S4_4_forgetting"] and below <= bars["S4_4_seeds_below_allowed"])
    detail["S4-4"] = d4

    def transfer_guard(family, stream):
        """Amendment A4-1: the experts of S3+T are counted against the number of true functions (the stream's ``regimes``) when every record has it, else against S3's count. The shipped pool
        under-spawns on several streams, so a ratio to S3 calls the transfer rule's correct count a cascade; the purpose of the statement is no false spawns."""
        a, b = g(family, stream, "S3+T"), g(family, stream, "S3")
        d = _diff(a, b, acc, acc)
        if d is not None:
            truth = [r.get("regimes") for r in a.values()]
            known = all(isinstance(x, (int, float)) and x > 0 for x in truth)
            mean_t, mean_s3 = _mean(a, lambda r: r["n_experts"]), max(_mean(b, lambda r: r["n_experts"]), 1e-9)
            ratio = mean_t / float(np.mean(truth)) if known else mean_t / mean_s3
            d.update(expert_ratio=ratio, ratio_against="true functions" if known else "S3's count", expert_ratio_vs_s3=mean_t / mean_s3)
            d["holds"] = bool(d["mean_diff_points"] >= bars["S4_5_gain"] and ratio <= bars["S4_5_ratio"])
        return d
    detail["S4-5"] = {f"{f}:{s}": transfer_guard(f, s) for f, s in [("s3", s) for s in GUARD_STREAMS] + [("banking", "-")]}

    small, big = g("arith-mixed16", "-", "S3+G"), g("arith-mixed128", "-", "S3")
    d6 = None
    if small and big and set(small) & set(big):
        seeds = sorted(set(small) & set(big))
        reach = sum(small[s]["acc"] >= bars["S4_6_frac"] * big[s]["acc"] for s in seeds)
        need = math.ceil(bars["S4_6_seed_share"] * len(seeds) - 1e-9)
        width = float(np.mean([np.mean(small[s]["final_hidden"]) for s in seeds]))
        cost = float(np.mean([small[s]["grow_stats"]["post_rows"] / max(small[s]["grow_stats"]["train_rows"] * small[s]["inner_steps"] * 6, 1) for s in seeds]))
        lo, hi = bars["S4_6_width"]
        d6 = {"n": len(seeds), "seeds_reaching": reach, "needed": need, "mean_final_width": width, "extra_forward_cost": cost,
              "holds": bool(reach >= need and lo <= width <= hi and cost <= bars["S4_6_cost"])}
    detail["S4-6"] = d6

    def growth_guard(family, stream):
        a, b = g(family, stream, "S3+G"), g(family, stream, "S3")
        d = _diff(a, b, acc, acc)
        if d is not None:
            seeds = sorted(set(a) & set(b))
            grows = float(np.mean([len(a[s]["grow_events"]) for s in seeds]))
            relp = float(np.mean([a[s]["params"] / b[s]["params"] - 1.0 for s in seeds]))
            d.update(growth_events_per_run=grows, rel_params=relp)
            d["holds"] = bool(d["mean_diff_points"] >= bars["S4_7_gain"] and grows <= bars["S4_7_grow"] and relp <= bars["S4_7_params"])
        return d
    detail["S4-7"] = {f"{f}:{s}": growth_guard(f, s) for f, s in [("arith-mixed128", "-")] + [("s3", s) for s in S3_STREAMS]}

    full = g("arith-seq", "-", "S3+T+G")
    d8 = None
    if full:
        a8, f8 = _POINTS * _mean(full, on), _POINTS * _mean(full, lambda r: r["forgetting"])
        d8 = {"n": len(full), "exact_match_points": a8, "forgetting_points": f8, "holds": bool(a8 >= bars["S4_8_acc"] and f8 <= bars["S4_8_forgetting"])}
    detail["S4-8"] = d8
    detail["S4-9"] = None if golden is None else {"differences": list(golden["differences"]), "holds": not golden["differences"]}

    exp = {**EXPECTED_RV2D, **(expected or {})}
    rules = {
        "S4-1": f"arithmetic operator-by-operator, S3 pools: exact match with the router minus without, interval lower bound >= +{bars['S4_1_lo']:.1f} points",
        "S4-2": f"OV-0.85, S3 pools: router minus no router, interval lower bound >= +{bars['S4_2_lo']:.1f} points",
        "S4-3": f"each other stage-3 stream and banking77, S3 pools: router minus no router, mean >= {bars['S4_3']:+.1f} points",
        "S4-4": f"arithmetic operator-by-operator: S3+T minus S3 mean >= +{bars['S4_4_gain']:.1f} points, mean forgetting <= {bars['S4_4_forgetting']:.1f} points, at most "
                f"{bars['S4_4_seeds_below_allowed']} seed with fewer than {bars['S4_4_min_experts']} experts",
        "S4-5": f"each other stage-3 stream and banking77: S3+T minus S3 mean >= {bars['S4_5_gain']:+.1f} points and mean experts <= {bars['S4_5_ratio']} x the number of true functions of the stream (S3's count where there is none; amendment A4-1)",
        "S4-6": f"arithmetic mixed from 16 units, S3+G: accuracy >= {bars['S4_6_frac']} x the 128-unit S3 of the same seed in at least {bars['S4_6_seed_share']:.2f} of the seeds, mean final width in "
                f"{bars['S4_6_width']}, extra forward cost <= {100 * bars['S4_6_cost']:.0f} % of training",
        "S4-7": f"arithmetic mixed from 128 units and every stage-3 stream: S3+G minus S3 mean >= {bars['S4_7_gain']:+.1f} points, growth events per run <= {bars['S4_7_grow']}, "
                f"relative parameter increase <= {bars['S4_7_params']}",
        "S4-8": f"arithmetic operator-by-operator, S3+T+G with the router: exact match >= {bars['S4_8_acc']:.0f} points and forgetting <= {bars['S4_8_forgetting']:.0f} points",
        "S4-9": "all stage-4 flags off: the pool is bit-identical to the recorded one (golden fingerprints of rv2c/golden.py)",
    }
    verdict = {k: _combine(v) for k, v in detail.items()}
    return {k: {"rule": rules[k], "verdict": verdict[k], "detail": detail[k], "expected": exp[k], "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == exp[k]} for k in IDS}


def rv2d_decision(result: dict) -> dict:
    """What the pre-registration says to do: ``{"4a": bool, "4b": bool, "4c": bool, "text": str}``. 4a needs (S4-1 or S4-2) and S4-3, 4b needs S4-4 and S4-5, 4c needs S4-6 and S4-7;
    S4-9 failing is a bug, not a result; S4-8 (the full stack) flags the adoption when it misses."""
    none = {"4a": False, "4b": False, "4c": False}
    v = {k: result[k]["verdict"] for k in IDS}
    missing = [k for k in IDS if v[k] == "NOT MEASURABLE"]
    if missing:
        return {**none, "text": "The RV2-D decision cannot be applied: " + ", ".join(missing) + " not measurable."}
    if v["S4-9"] != "PASS":
        return {**none, "text": "S4-9 FAIL: with all stage-4 flags off the pool is not bit-identical to the recorded one. Stop: this is a bug, not a result."}
    a = (v["S4-1"] == "PASS" or v["S4-2"] == "PASS") and v["S4-3"] == "PASS"
    b = v["S4-4"] == "PASS" and v["S4-5"] == "PASS"
    c = v["S4-6"] == "PASS" and v["S4-7"] == "PASS"
    parts = [f"4a (router) {'adopt: router=True becomes the default' if a else 'NOT adopted: stays off'}",
             f"4b (transfer rule) {'adopt: spawn_rule=transfer becomes the default' if b else 'NOT adopted: stays off'}",
             f"4c (growth from the fit gap) {'adopt: growth_rule=gap becomes the default' if c else 'NOT adopted: stays off'}"]
    if (a or b or c) and v["S4-8"] != "PASS":
        parts.append(f"the full stack missed S4-8 ({v['S4-8']}): adoption is flagged, say so everywhere")
    return {"4a": a, "4b": b, "4c": c, "text": "; ".join(parts) + "."}

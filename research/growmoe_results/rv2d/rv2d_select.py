"""The declared selection rules of the dev constants (spec sections 6.7 and 8). Pure functions of per-cell rows (dicts with family, stream, arm, seed, gap, tmargin, acc, params,
n_grow, n_experts, router). Nothing here reads a confirmatory seed. Every function returns the chosen value, the whole curve and the reason, so the choice can be audited."""

import numpy as np

GAP_GRID = (0.14, 0.17, 0.20, 0.23, 0.26)
TRANSFER_GRID = (0.15, 0.25, 0.35)
GROUPS = {"arith16": [("arith-mixed16", "-")], "arith128": [("arith-mixed128", "-")], "stage3": [("s3", "F-15"), ("s3", "OV-0.6")]}
GUARD_GROUPS = ("arith128", "stage3")


def _by(rows, family, stream, arm, **where):
    return {r["seed"]: r for r in rows if r["family"] == family and r["stream"] == stream and r["arm"] == arm and all(r.get(k) == v for k, v in where.items())}


def _paired(rows, family, stream, arm, **where):
    base, other = _by(rows, family, stream, "S3"), _by(rows, family, stream, arm, **where)
    seeds = sorted(set(base) & set(other))
    return [(base[s], other[s]) for s in seeds]


def choose_grow_gap(rows) -> dict:
    """Score(gamma) = mean over the three groups of [paired accuracy gain over S3 - 0.1 x relative parameter increase], subject to the guards on the arithmetic-128 and stage-3
    groups: accuracy gain >= -0.005, at most 1 growth event per run on average, relative parameter increase <= 0.25. The largest score among the gammas that satisfy every guard."""
    curve = {}
    for g in GAP_GRID:
        per_group, ok = {}, True
        for name, members in GROUPS.items():
            pairs = [p for fam, st in members for p in _paired(rows, fam, st, "S3+G", gap=g)]
            if not pairs:
                per_group[name] = None
                ok = False
                continue
            gain = float(np.mean([o["acc"] - b["acc"] for b, o in pairs]))
            relp = float(np.mean([o["params"] / b["params"] - 1.0 for b, o in pairs]))
            grows = float(np.mean([o["n_grow"] for _, o in pairs]))
            per_group[name] = {"gain": gain, "rel_params": relp, "grow_events": grows, "score": gain - 0.1 * relp}
            if name in GUARD_GROUPS and not (gain >= -0.005 and grows <= 1.0 and relp <= 0.25):
                ok = False
        score = float(np.mean([v["score"] for v in per_group.values() if v])) if all(per_group.values()) else float("-inf")
        curve[g] = {"groups": per_group, "score": score, "guards_ok": ok}
    feasible = {g: c["score"] for g, c in curve.items() if c["guards_ok"]}
    if not feasible:
        return {"chosen": None, "curve": curve, "reason": "no single value satisfies the guards: declare a per-family value in the pre-registration and add a self-normalising threshold to the open questions"}
    best = max(feasible, key=lambda g: (feasible[g], -g))
    return {"chosen": best, "curve": curve, "reason": f"largest score {feasible[best]:.4f} among the values that satisfy the guards"}


def choose_router_setting(rows) -> dict:
    """Among the four post hoc settings: the best mean gain on the two target streams (arithmetic operator-by-operator, OV-0.85) among the settings whose mean difference on
    every other stream is at least -0.005 (the -0.5 point guard). Ties go to restrict-0.01."""
    settings = ["restrict-0.01", "restrict-0.03", "free-0.01", "free-0.03"]
    targets = {("arith-seq", "-"), ("s3", "OV-0.85")}
    curve = {}
    for s in settings:
        by_stream = {}
        for r in rows:
            if r["arm"] != "S3" or s not in r.get("router", {}) or "off" not in r["router"]:
                continue
            by_stream.setdefault((r["family"], r["stream"]), []).append(r["router"][s]["acc"] - r["router"]["off"]["acc"])
        means = {k: float(np.mean(v)) for k, v in by_stream.items()}
        target = [v for k, v in means.items() if k in targets]
        guards = {k: v for k, v in means.items() if k not in targets}
        curve[s] = {"target_gain": float(np.mean(target)) if target else float("nan"), "worst_guard": min(guards.values()) if guards else 0.0, "per_stream": means,
                    "guards_ok": all(v >= -0.005 for v in guards.values())}
    feasible = {s: c["target_gain"] for s, c in curve.items() if c["guards_ok"] and c["target_gain"] == c["target_gain"]}
    if not feasible:
        return {"chosen": None, "curve": curve, "reason": "every setting breaks a guard stream: the router stays off"}
    best = max(feasible, key=lambda s: (round(feasible[s], 6), -settings.index(s)))
    return {"chosen": best, "curve": curve, "reason": f"best mean gain {feasible[best]:.4f} on the target streams among the settings that hurt no guard stream by more than 0.5 points"}


def choose_transfer_margin(rows) -> dict:
    """Among the declared margins: the best mean paired accuracy gain over S3 on the target (arithmetic operator-by-operator) subject to the guards on the other streams: mean
    difference >= -0.005 and mean expert count <= 1.2 x the number of true functions of the stream (amendment A4-1; S3's count for a stream that records none). Ties go to the default 0.25."""
    curve = {}
    for t in TRANSFER_GRID:
        gains, ok = {}, True
        for fam, st in sorted({(r["family"], r["stream"]) for r in rows if r["arm"] == "S3+T"}):
            pairs = _paired(rows, fam, st, "S3+T", tmargin=t)
            if not pairs:
                continue
            gain = float(np.mean([o["acc"] - b["acc"] for b, o in pairs]))
            truth = [o.get("regimes") for _, o in pairs]
            denominator = float(np.mean(truth)) if all(isinstance(x, (int, float)) and x > 0 for x in truth) else max(float(np.mean([b["n_experts"] for b, _ in pairs])), 1e-9)
            ratio = float(np.mean([o["n_experts"] for _, o in pairs]) / denominator)
            gains[f"{fam}:{st}"] = {"gain": gain, "expert_ratio": ratio}
            if fam != "arith-seq" and not (gain >= -0.005 and ratio <= 1.2):
                ok = False
        target = gains.get("arith-seq:-", {}).get("gain", float("nan"))
        curve[t] = {"target_gain": target, "per_stream": gains, "guards_ok": ok}
    feasible = {t: c["target_gain"] for t, c in curve.items() if c["guards_ok"] and c["target_gain"] == c["target_gain"]}
    if not feasible:
        return {"chosen": None, "curve": curve, "reason": "no margin satisfies the guards: the transfer rule stays off"}
    top = max(round(v, 6) for v in feasible.values())
    tied = [t for t, v in feasible.items() if round(v, 6) == top]
    best = 0.25 if 0.25 in tied else min(tied)
    return {"chosen": best, "curve": curve, "reason": f"best target gain {top:.4f} among the margins that satisfy the guards (ties go to 0.25)"}

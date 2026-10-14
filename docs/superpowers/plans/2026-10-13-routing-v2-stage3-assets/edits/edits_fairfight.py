"""Append to growmoe/fairfight.py (stage 3 verdicts). An entry whose `old` is None is appended to the end of the file."""

FILE = "growmoe/fairfight.py"

EDITS = [
    ("fairfight-rv2c",
     None,
     '''

# ------------------------------------------------------------------------------------------------ RV2-C verdicts (stage 3: batch-level gate and group answering)
# research/growmoe_results/rv2c/PREREG.md; spec docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md section 7.
# A run record is one row of `run_record` plus: `regimes` (number of regimes of the stream), `fn_known` / `fn_called_novel` (A_set only: known single-source
# batches and how many of them the set gate called new), and `groups` = {kind: {"single": accuracy, "group": accuracy}} for the kinds
# pure64, pure16, mixed32_32 (50/50) and mixed58_6 (90/10), both measured on the same pool and the same inputs.
BARS_RV2C = {"S3_1": {"OV-0.6": 8.0, "OV-0.85": 2.5}, "S3_2_mean": -1.0, "S3_2_lo": -2.0, "S3_3": -1.0, "S3_4": 0.5, "S3_5": 0.02,
             "S3_6": {"OV-0.85": 6.0, "OV-0.6": 0.0, "other": -0.5}, "S3_7": {"mixed32_32": -5.0, "mixed58_6": -1.0}, "S3_8": 5.0}
EXPECTED_RV2C = {k: "PASS" for k in ("S3-1", "S3-2", "S3-3", "S3-4", "S3-5", "S3-6", "S3-7", "S3-8", "S3-9")}
_GUARDS_RV2C = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3")
_SINGLE_SOURCE_RV2C = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.6", "OV-0.85")      # streams whose batches come from one regime
_ALL_RV2C = _SINGLE_SOURCE_RV2C + ("OV-0.3-MIXED",)
_GROUP_OTHERS_RV2C = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.3-MIXED")


def _paired_field(index: dict, stream: str, method: str, path_a: tuple, path_b: tuple, margin: float):
    """Paired difference (field a - field b) of two fractions stored in the SAME run record (one pool, one set of inputs), in points."""
    runs = index.get((stream, method))
    if not runs:
        return None

    def get(r, path):
        for key in path:
            r = r[key]
        return r

    seeds = sorted(runs)
    try:
        a = [get(runs[s], path_a) * _POINTS for s in seeds]
        b = [get(runs[s], path_b) * _POINTS for s in seeds]
    except (KeyError, TypeError):                                                       # a record without the group measurements (or with None for a kind that could not be built): not measurable, not a crash
        return None
    d = paired_diff(a, b)
    return {"n": d["n"], "mean_diff_points": d["mean_diff"], "ci_lo": _finite(d["ci_lo"]), "ci_hi": _finite(d["ci_hi"]), "p_t": _finite(d["p_t"]),
            "holds": bool(d["mean_diff"] >= margin)}


def _rules_rv2c(bars: dict, cand: str, ref: str) -> dict:
    b1, b6, b7 = bars["S3_1"], bars["S3_6"], bars["S3_7"]
    return {
        "S3-1": f"OV-0.6 and OV-0.85: mean paired {cand} - {ref} >= +{b1['OV-0.6']:.1f} and >= +{b1['OV-0.85']:.1f} points, interval lower bounds above 0",
        "S3-2": f"F-15, F-30, S1, S2-5, S2-30, OV-0.0, OV-0.3: mean paired {cand} - {ref} >= {bars['S3_2_mean']:+.1f} points and the interval lower bound >= {bars['S3_2_lo']:+.1f}",
        "S3-3": f"OV-0.3-MIXED: mean paired {cand} - {ref} >= {bars['S3_3']:+.1f} points",
        "S3-4": f"mean experts of {cand} <= regimes + {bars['S3_4']} on every one of the ten regular streams",
        "S3-5": f"known single-source batches called new as a set, pooled over the nine single-source streams: <= {100 * bars['S3_5']:.0f} %",
        "S3-6": (f"{cand} pools, pure groups of 64, group answer (vote + veto) minus per-input answer: OV-0.85 >= +{b6['OV-0.85']:.1f}, OV-0.6 >= {b6['OV-0.6']:+.1f}, "
                 f"every other stream >= {b6['other']:+.1f} points"),
        "S3-7": f"{cand} pools, mixed groups with the veto minus per input, every stream: 50/50 >= {b7['mixed32_32']:+.1f} and 90/10 >= {b7['mixed58_6']:+.1f} points",
        "S3-8": f"{cand} pools, OV-0.85, groups of 16 with the veto minus per input >= +{bars['S3_8']:.1f} points",
        "S3-9": "both flags off: the pool is bit-identical to the one recorded before stage 3 (golden fingerprints of growmoe/../rv2c/golden.py)",
    }


def rv2c_verdicts(runs: Sequence[dict], golden: dict = None, bars: dict = None, expected: dict = None, candidate: str = "A_set", reference: str = "A_vote") -> dict:
    """The RV2-C statements S3-1 to S3-9 on per-seed run records. Multi-stream statements follow the M0 rule (PASS on all, FAIL on none, PARTIAL otherwise,
    NOT MEASURABLE when data are missing). ``golden`` = {"differences": [...]} from the flag-off check (None = not measured)."""
    bars = {**BARS_RV2C, **(bars or {})}
    exp = {**EXPECTED_RV2C, **(expected or {})}
    ix = _index(runs)
    detail = {"S3-1": {s: _paired_points(ix, (s, candidate), (s, reference), m, need_interval=True) for s, m in bars["S3_1"].items()}}
    s32 = {}
    for s in _GUARDS_RV2C:
        d = _paired_points(ix, (s, candidate), (s, reference), bars["S3_2_mean"])
        if d is not None:
            d["holds"] = bool(d["holds"] and d["ci_lo"] is not None and d["ci_lo"] >= bars["S3_2_lo"])
        s32[s] = d
    detail["S3-2"] = s32
    detail["S3-3"] = _paired_points(ix, ("OV-0.3-MIXED", candidate), ("OV-0.3-MIXED", reference), bars["S3_3"])
    s34 = {}
    for s in _ALL_RV2C:
        rows = ix.get((s, candidate))
        if not rows or any("regimes" not in r for r in rows.values()):
            s34[s] = None
            continue
        mean_experts, regimes = float(np.mean([r["n_experts"] for r in rows.values()])), float(np.mean([r["regimes"] for r in rows.values()]))
        s34[s] = {"n": len(rows), "mean_experts": mean_experts, "regimes": regimes, "holds": bool(mean_experts <= regimes + bars["S3_4"])}
    detail["S3-4"] = s34
    known = called = 0
    present = [s for s in _SINGLE_SOURCE_RV2C if ix.get((s, candidate))]
    for s in present:
        for r in ix[(s, candidate)].values():
            known += int(r.get("fn_known", 0))
            called += int(r.get("fn_called_novel", 0))
    detail["S3-5"] = (None if len(present) < len(_SINGLE_SOURCE_RV2C) or known == 0 else
                      {"streams": len(present), "known_batches": known, "called_novel": called, "rate": called / known, "holds": bool(called / known <= bars["S3_5"])})
    s36 = {}
    for s in ("OV-0.85", "OV-0.6") + _GROUP_OTHERS_RV2C:
        margin = bars["S3_6"].get(s, bars["S3_6"]["other"])
        s36[s] = _paired_field(ix, s, candidate, ("groups", "pure64", "group"), ("groups", "pure64", "single"), margin)
    detail["S3-6"] = s36
    detail["S3-7"] = {f"{s}/{kind}": _paired_field(ix, s, candidate, ("groups", kind, "group"), ("groups", kind, "single"), bars["S3_7"][kind])
                      for s in _ALL_RV2C for kind in ("mixed32_32", "mixed58_6")}
    detail["S3-8"] = _paired_field(ix, "OV-0.85", candidate, ("groups", "pure16", "group"), ("groups", "pure16", "single"), bars["S3_8"])
    detail["S3-9"] = None if golden is None else {"differences": list(golden["differences"]), "holds": not golden["differences"]}
    verdict = {k: _combine(v) for k, v in detail.items()}
    rules = _rules_rv2c(bars, candidate, reference)
    return {k: {"rule": rules[k], "verdict": verdict[k], "detail": detail[k], "expected": exp[k],
                "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == exp[k]} for k in EXPECTED_RV2C}


def rv2c_decision(result: dict) -> str:
    """What the pre-registration says to do, for 3a (flag set_gate) and 3b (flag group_vote). The kill rules are S3-1 at OV-0.6, S3-2, S3-4 and S3-5."""
    ids = list(EXPECTED_RV2C)
    if any(result[k]["verdict"] == "NOT MEASURABLE" for k in ids):
        return "The RV2-C decision cannot be applied: " + ", ".join(k for k in ids if result[k]["verdict"] == "NOT MEASURABLE") + " not measurable."
    if result["S3-9"]["verdict"] != "PASS":
        return "S3-9 FAIL: with both flags off the pool is not bit-identical to the recorded one. Stop: this is a bug, not a result."
    killed = [k for k in ("S3-2", "S3-4", "S3-5") if result[k]["verdict"] != "PASS"]
    ov6 = result["S3-1"]["detail"]["OV-0.6"]["holds"]
    if killed:
        a = f"3a is NOT adopted ({', '.join(killed)} did not pass: kill rule); set_gate stays False"
    elif result["S3-1"]["verdict"] == "PASS" and result["S3-3"]["verdict"] == "PASS":
        a = "adopt 3a: set_gate=True becomes the PhyloMoE default, test-first, with the historical entry points pinned (set_gate=False)"
    elif ov6 and result["S3-3"]["verdict"] == "PASS":
        a = "3a passes at OV-0.6 only (S3-1 PARTIAL): it stays opt-in (set_gate=False) and every document says so"
    else:
        a = f"3a is NOT adopted (S3-1 {result['S3-1']['verdict']}, S3-3 {result['S3-3']['verdict']}); set_gate stays False"
    if all(result[k]["verdict"] == "PASS" for k in ("S3-6", "S3-7", "S3-8")):
        b = ("adopt 3b: group_vote=True becomes the default (it changes nothing unless predict is given groups); it was measured on the set-gate pools, "
             "the vote pools are reported in the results")
    else:
        b = "3b is NOT adopted (" + ", ".join(f"{k} {result[k]['verdict']}" for k in ("S3-6", "S3-7", "S3-8") if result[k]["verdict"] != "PASS") + "); group_vote stays False"
    return f"3a: {a}. 3b: {b}."
'''),
]

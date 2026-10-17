"""RV2-D verdicts: the statements S4-1 to S4-9 and the decision rule (spec section 8). Records are synthetic; the point is that every bar is read from the bars and that one broken
effect breaks exactly its own statement."""

import copy

import pytest

from growmoe.verdicts_rv4 import IDS, rv2d_decision, rv2d_verdicts

SEL = "restrict-0.01"
BARS = {"router_setting": SEL}
S3_STREAMS = ("F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5")


def rec(family, stream, arm, seed, acc, on=None, experts=4, params=1000, grow=0, forgetting=None, hidden=128, post=0, train=1):
    return {"family": family, "stream": stream, "arm": arm, "seed": seed, "acc": acc, "router": {"off": {"acc": acc}, SEL: {"acc": acc if on is None else on}}, "n_experts": experts,
            "params": params, "grow_events": [{}] * grow, "final_hidden": [hidden], "forgetting": forgetting, "grow_stats": {"post_rows": post, "train_rows": train}, "inner_steps": 5}


def good_runs():
    """A run set in which every statement holds with room to spare: 15 seeds for the stage-3 streams and the mixed arithmetic, 6 for operator-by-operator arithmetic, 5 for banking77."""
    runs = []
    for s in range(85, 100):
        runs += [rec("arith-mixed128", "-", "S3", s, 0.34), rec("arith-mixed128", "-", "S3+G", s, 0.34, params=1100, grow=1),
                 rec("arith-mixed16", "-", "S3+G", s, 0.32, hidden=128, post=3, train=100)]
        for st in S3_STREAMS:
            runs += [rec("s3", st, "S3", s, 0.40, on=0.50 if st == "OV-0.85" else 0.40), rec("s3", st, "S3+T", s, 0.40), rec("s3", st, "S3+G", s, 0.40, params=1050, grow=1),
                     rec("s3", st, "S3+T+G", s, 0.40)]
    for s in range(85, 91):
        runs += [rec("arith-seq", "-", "S3", s, 0.27, on=0.40, forgetting=0.15, experts=3), rec("arith-seq", "-", "S3+T", s, 0.36, forgetting=0.07, experts=6),
                 rec("arith-seq", "-", "S3+T+G", s, 0.36, on=0.46, forgetting=0.07, experts=6)]
    for s in range(85, 90):
        runs += [rec("banking", "-", "S3", s, 0.70, experts=6), rec("banking", "-", "S3+T", s, 0.70, experts=7), rec("banking", "-", "S3+G", s, 0.70)]
    return runs


def _edit(runs, where, **change):
    out = copy.deepcopy(runs)
    for r in out:
        if where(r):
            for k, v in change.items():
                if k == "on":
                    r["router"][SEL]["acc"] = v
                elif k == "off":
                    r["router"]["off"]["acc"] = v
                elif k == "experts":
                    r["n_experts"] = v
                elif k == "grow":
                    r["grow_events"] = [{}] * v
                else:
                    r[k] = v
    return out


BREAKS = {
    "S4-1": lambda runs: _edit(runs, lambda r: r["family"] == "arith-seq" and r["arm"] == "S3", on=0.29),
    "S4-2": lambda runs: _edit(runs, lambda r: r["stream"] == "OV-0.85" and r["arm"] == "S3", on=0.40),
    "S4-3": lambda runs: _edit(runs, lambda r: r["stream"] == "S1" and r["arm"] == "S3", on=0.30),
    "S4-4": lambda runs: _edit(runs, lambda r: r["family"] == "arith-seq" and r["arm"] == "S3+T", acc=0.30),
    "S4-5": lambda runs: _edit(runs, lambda r: r["family"] == "banking" and r["arm"] == "S3+T", experts=12),
    "S4-6": lambda runs: _edit(runs, lambda r: r["family"] == "arith-mixed16", acc=0.20),
    "S4-7": lambda runs: _edit(runs, lambda r: r["family"] == "arith-mixed128" and r["arm"] == "S3+G", grow=3),
    "S4-8": lambda runs: _edit(runs, lambda r: r["family"] == "arith-seq" and r["arm"] == "S3+T+G", on=0.40),
}


def test_every_statement_passes_when_every_effect_is_large():
    res = rv2d_verdicts(good_runs(), BARS, {"differences": []})
    assert {k: v["verdict"] for k, v in res.items()} == {k: "PASS" for k in IDS}
    assert all(v["rule"] for v in res.values())
    d = rv2d_decision(res)
    assert d["4a"] and d["4b"] and d["4c"]


@pytest.mark.parametrize("sid", sorted(BREAKS))
def test_breaking_one_effect_breaks_exactly_its_statement(sid):
    res = rv2d_verdicts(BREAKS[sid](good_runs()), BARS, {"differences": []})
    assert {k: v["verdict"] != "PASS" for k, v in res.items()} == {k: k == sid for k in IDS}


def test_s4_5_counts_experts_against_the_true_number_of_functions_when_the_stream_has_one():
    """Amendment A4-1: the shipped pool under-spawns (S2-5: 3 experts for 5 regimes), so '1.2 x S3's experts' called the transfer rule's correct 5 a cascade."""
    runs = _edit(good_runs(), lambda r: r["stream"] == "S2-5", regimes=5)
    runs = _edit(runs, lambda r: r["stream"] == "S2-5" and r["arm"] == "S3", experts=3)
    runs = _edit(runs, lambda r: r["stream"] == "S2-5" and r["arm"] == "S3+T", experts=5)
    res = rv2d_verdicts(runs, BARS, {"differences": []})
    d = res["S4-5"]["detail"]["s3:S2-5"]
    assert res["S4-5"]["verdict"] == "PASS" and d["ratio_against"] == "true functions" and d["expert_ratio"] == pytest.approx(1.0) and d["expert_ratio_vs_s3"] == pytest.approx(5 / 3)
    cascade = _edit(runs, lambda r: r["stream"] == "S2-5" and r["arm"] == "S3+T", experts=7)                       # 7 experts for 5 functions is a cascade whatever S3 did
    assert rv2d_verdicts(cascade, BARS, {"differences": []})["S4-5"]["verdict"] != "PASS"


def test_s4_5_falls_back_to_the_s3_count_for_records_without_a_ground_truth():
    res = rv2d_verdicts(good_runs(), BARS, {"differences": []})                                                       # the synthetic records carry no 'regimes'
    assert res["S4-5"]["detail"]["banking:-"]["ratio_against"] == "S3's count"


def test_the_decision_rule_adopts_each_component_on_its_own_statements():
    d = rv2d_decision(rv2d_verdicts(BREAKS["S4-3"](good_runs()), BARS, {"differences": []}))
    assert (d["4a"], d["4b"], d["4c"]) == (False, True, True)                      # the router guard failed: 4a stays off, the others proceed
    d = rv2d_decision(rv2d_verdicts(BREAKS["S4-1"](good_runs()), BARS, {"differences": []}))
    assert (d["4a"], d["4b"], d["4c"]) == (True, True, True)                       # S4-2 alone carries 4a with S4-3 ("S4-1 or S4-2")
    d = rv2d_decision(rv2d_verdicts(BREAKS["S4-7"](good_runs()), BARS, {"differences": []}))
    assert (d["4a"], d["4b"], d["4c"]) == (True, True, False)


def test_a_failing_golden_check_stops_everything():
    res = rv2d_verdicts(good_runs(), BARS, {"differences": ["default|S1-smoke"]})
    d = rv2d_decision(res)
    assert res["S4-9"]["verdict"] == "FAIL" and not (d["4a"] or d["4b"] or d["4c"]) and "bug, not a result" in d["text"]


def test_missing_data_is_not_measurable_and_blocks_the_decision():
    res = rv2d_verdicts([r for r in good_runs() if r["family"] != "banking"], BARS, {"differences": []})
    assert res["S4-3"]["verdict"] != "PASS" and res["S4-5"]["verdict"] != "PASS"
    d = rv2d_decision(rv2d_verdicts([r for r in good_runs() if r["arm"] != "S3+T+G"], BARS, {"differences": []}))
    assert "cannot" in d["text"] and not (d["4a"] or d["4b"] or d["4c"])

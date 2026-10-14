"""RV3: the pre-registered statements S3-1 to S3-9 and the decision rule, on synthetic run records (spec section 7)."""

import copy

import numpy as np
import pytest

from growmoe.fairfight import BARS_RV2C, EXPECTED_RV2C, rv2c_decision, rv2c_verdicts

STREAMS = ["F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.3-MIXED", "OV-0.6", "OV-0.85"]
REGIMES = {"F-15": 15, "F-30": 30, "S1": 5, "S2-5": 5, "S2-30": 30, "OV-0.0": 10, "OV-0.3": 10, "OV-0.3-MIXED": 10, "OV-0.6": 10, "OV-0.85": 10}
SEEDS = list(range(60, 70))
KINDS = ("pure64", "pure16", "mixed32_32", "mixed58_6")
GOOD_SET = {"OV-0.6": 17.0, "OV-0.85": 5.0}                                              # points, A_set - A_vote
GOOD_GROUP = {"pure64": {"OV-0.85": 11.0, "OV-0.6": 1.0}, "pure16": {"OV-0.85": 9.0}, "mixed32_32": {}, "mixed58_6": {}}      # points, group - single, on A_set pools


def _runs(set_effect=None, group_effect=None, experts=None, called_novel=0.001, jitter=0.3):
    rng = np.random.default_rng(0)
    set_effect = {**GOOD_SET, **(set_effect or {})}
    group_effect = copy.deepcopy(group_effect) if group_effect is not None else copy.deepcopy(GOOD_GROUP)
    rows = []
    for s in STREAMS:
        for seed in SEEDS:
            base = 0.40 + 0.01 * rng.standard_normal()
            for method in ("A_vote", "A_set"):
                eff = set_effect.get(s, 0.0) / 100.0 if method == "A_set" else 0.0
                acc = base + eff + 0.001 * jitter * rng.standard_normal()
                groups = {}
                for k in KINDS:
                    gain = group_effect.get(k, {}).get(s, 0.0) / 100.0 if method == "A_set" else 0.0
                    single = acc + 0.001 * rng.standard_normal()
                    groups[k] = {"single": single, "group": single + gain + 0.001 * jitter * rng.standard_normal()}
                n_exp = REGIMES[s] + (experts or {}).get(s, 0.0) if method == "A_set" else REGIMES[s]
                rows.append({"stream": s, "method": method, "seed": seed, "acc": acc, "n_experts": n_exp, "regimes": REGIMES[s], "groups": groups,
                             "fn_known": 100, "fn_called_novel": round(100 * called_novel) if method == "A_set" else 0})
    return rows


GOLDEN_OK = {"differences": []}


def test_the_spec_s_dev_effects_pass_every_statement_and_adopt_both_levels():
    res = rv2c_verdicts(_runs(), GOLDEN_OK)
    assert list(res) == list(EXPECTED_RV2C)
    assert {k: v["verdict"] for k, v in res.items()} == {k: "PASS" for k in EXPECTED_RV2C}
    assert all(v["as_expected"] for v in res.values())
    d = rv2c_decision(res)
    assert d.startswith("3a: adopt 3a") and "3b: adopt 3b" in d


def test_the_rules_name_the_candidate_and_the_bars_are_the_specs():
    res = rv2c_verdicts(_runs(), GOLDEN_OK, candidate="A_set", reference="A_vote")
    assert "A_set - A_vote" in res["S3-1"]["rule"] and "+8.0" in res["S3-1"]["rule"] and "+2.5" in res["S3-1"]["rule"]
    assert BARS_RV2C["S3_1"] == {"OV-0.6": 8.0, "OV-0.85": 2.5} and BARS_RV2C["S3_5"] == 0.02 and BARS_RV2C["S3_8"] == 5.0
    assert "-5.0" in res["S3-7"]["rule"] and "-1.0" in res["S3-7"]["rule"]


def test_a_gain_at_ov_06_only_makes_3a_opt_in():
    res = rv2c_verdicts(_runs(set_effect={"OV-0.85": 1.0}), GOLDEN_OK)
    assert res["S3-1"]["verdict"] == "PARTIAL"
    d = rv2c_decision(res)
    assert "OV-0.6 only" in d and "opt-in" in d


def test_no_gain_at_ov_06_means_not_adopted():
    res = rv2c_verdicts(_runs(set_effect={"OV-0.6": 5.0, "OV-0.85": 1.0}), GOLDEN_OK)
    assert res["S3-1"]["verdict"] == "FAIL" and "3a is NOT adopted" in rv2c_decision(res)


def _pair_rows(stream, diffs_points):
    """A_vote / A_set records of one stream whose per-seed accuracy difference is exactly the given list (points)."""
    rows = []
    for seed, d in zip(SEEDS, diffs_points):
        for method, acc in (("A_vote", 0.40), ("A_set", 0.40 + d / 100.0)):
            rows.append({"stream": stream, "method": method, "seed": seed, "acc": acc, "n_experts": 10, "regimes": 10, "groups": {}, "fn_known": 100, "fn_called_novel": 0})
    return rows


def test_the_interval_must_exclude_zero_not_just_the_mean_pass_the_bar():
    noisy = _pair_rows("OV-0.85", [10.0, -4.0] * 5)                                         # mean +3.0 (above +2.5) but the 95 % interval reaches below 0
    res = rv2c_verdicts(noisy, None)["S3-1"]["detail"]["OV-0.85"]
    assert res["mean_diff_points"] == pytest.approx(3.0) and res["ci_lo"] < 0.0 and res["holds"] is False
    steady = _pair_rows("OV-0.85", [3.0, 2.0, 4.0, 3.0, 3.5, 2.5, 3.0, 3.5, 2.5, 3.0])
    ok = rv2c_verdicts(steady, None)["S3-1"]["detail"]["OV-0.85"]
    assert ok["mean_diff_points"] >= 2.5 and ok["ci_lo"] > 0.0 and ok["holds"] is True


def test_a_guard_stream_losing_more_than_a_point_is_a_kill_rule():
    res = rv2c_verdicts(_runs(set_effect={"F-15": -3.0}), GOLDEN_OK)
    assert res["S3-2"]["verdict"] in ("FAIL", "PARTIAL") and res["S3-2"]["detail"]["F-15"]["holds"] is False
    d = rv2c_decision(res)
    assert "3a is NOT adopted" in d and "S3-2" in d


def test_the_interval_floor_of_the_guards_is_enforced_separately_from_the_mean():
    res = rv2c_verdicts(_runs(set_effect={"S1": -0.9}, jitter=60.0), GOLDEN_OK)             # mean above -1.0, interval reaching below -2.0
    d = res["S3-2"]["detail"]["S1"]
    assert d["mean_diff_points"] >= -1.0 and d["holds"] == (d["ci_lo"] >= -2.0)


def test_the_mixed_stream_guard_is_a_mean_only_bar():
    res = rv2c_verdicts(_runs(set_effect={"OV-0.3-MIXED": -2.0}), GOLDEN_OK)
    assert res["S3-3"]["verdict"] == "FAIL" and "3a is NOT adopted" in rv2c_decision(res)


def test_spurious_experts_on_any_stream_fail_s3_4():
    res = rv2c_verdicts(_runs(experts={"F-30": 3.0}), GOLDEN_OK)
    assert res["S3-4"]["verdict"] == "PARTIAL" and res["S3-4"]["detail"]["F-30"]["holds"] is False and "S3-4" in rv2c_decision(res)
    ok = rv2c_verdicts(_runs(experts={"F-30": 0.4}), GOLDEN_OK)
    assert ok["S3-4"]["verdict"] == "PASS"                                                  # within regimes + 0.5


def test_false_novelty_is_pooled_over_the_nine_single_source_streams_and_capped_at_two_percent():
    assert rv2c_verdicts(_runs(called_novel=0.02), GOLDEN_OK)["S3-5"]["verdict"] == "PASS"                  # exactly 2 % is allowed
    bad = rv2c_verdicts(_runs(called_novel=0.03), GOLDEN_OK)["S3-5"]
    assert bad["verdict"] == "FAIL" and bad["detail"]["streams"] == 9 and bad["detail"]["known_batches"] == 9 * len(SEEDS) * 100
    d = rv2c_decision(rv2c_verdicts(_runs(called_novel=0.03), GOLDEN_OK))
    assert "3a is NOT adopted" in d and "S3-5" in d                                         # false novelty is a kill rule


def test_group_answering_is_judged_on_the_set_gate_pools_against_the_per_input_answer_of_the_same_pool():
    res = rv2c_verdicts(_runs(group_effect={**GOOD_GROUP, "pure64": {"OV-0.85": 4.0, "OV-0.6": 1.0}}), GOLDEN_OK)
    assert res["S3-6"]["verdict"] == "PARTIAL" and res["S3-6"]["detail"]["OV-0.85"]["holds"] is False
    d = rv2c_decision(res)
    assert d.startswith("3a: adopt 3a") and "3b is NOT adopted" in d and "S3-6" in d


def test_a_group_answer_that_is_worse_than_per_input_on_another_stream_fails_s3_6_by_more_than_half_a_point():
    ge = copy.deepcopy(GOOD_GROUP)
    ge["pure64"]["F-30"] = -0.8
    assert rv2c_verdicts(_runs(group_effect=ge), GOLDEN_OK)["S3-6"]["detail"]["F-30"]["holds"] is False
    ge["pure64"]["F-30"] = -0.3
    assert rv2c_verdicts(_runs(group_effect=ge), GOLDEN_OK)["S3-6"]["detail"]["F-30"]["holds"] is True


def test_mixed_groups_have_their_own_floors():
    ge = copy.deepcopy(GOOD_GROUP)
    ge["mixed32_32"] = {"F-15": -6.0}
    ge["mixed58_6"] = {"S1": -1.5}
    res = rv2c_verdicts(_runs(group_effect=ge), GOLDEN_OK)
    assert res["S3-7"]["detail"]["F-15/mixed32_32"]["holds"] is False and res["S3-7"]["detail"]["S1/mixed58_6"]["holds"] is False
    assert res["S3-7"]["verdict"] == "PARTIAL"


def test_small_groups_at_ov_085_need_five_points():
    ge = copy.deepcopy(GOOD_GROUP)
    ge["pure16"] = {"OV-0.85": 4.0}
    assert rv2c_verdicts(_runs(group_effect=ge), GOLDEN_OK)["S3-8"]["verdict"] == "FAIL"


def test_a_recorded_difference_in_the_flag_off_fingerprints_is_a_bug_and_stops_everything():
    res = rv2c_verdicts(_runs(), {"differences": ["default|S1-smoke"]})
    assert res["S3-9"]["verdict"] == "FAIL" and rv2c_decision(res).startswith("S3-9 FAIL")


def test_missing_data_is_not_measurable_and_the_decision_says_what_is_missing():
    rows = [r for r in _runs() if r["stream"] != "OV-0.85"]
    res = rv2c_verdicts(rows, None)
    assert res["S3-9"]["verdict"] == "NOT MEASURABLE" and res["S3-8"]["verdict"] == "NOT MEASURABLE" and res["S3-5"]["verdict"] == "NOT MEASURABLE"
    assert rv2c_decision(res).startswith("The RV2-C decision cannot be applied")
    assert rv2c_verdicts([], None)["S3-1"]["verdict"] == "NOT MEASURABLE"


def test_the_expected_table_marks_results_against_expectation():
    res = rv2c_verdicts(_runs(set_effect={"OV-0.6": 1.0}), GOLDEN_OK, expected={"S3-1": "PASS"})
    assert res["S3-1"]["as_expected"] is False and res["S3-2"]["as_expected"] is True


def test_a_group_kind_that_could_not_be_built_is_not_measurable_not_a_crash():
    rows = _runs()
    for r in rows:
        if r["stream"] == "OV-0.85":
            r["groups"]["pure16"] = None
    res = rv2c_verdicts(rows, GOLDEN_OK)
    assert res["S3-8"]["verdict"] == "NOT MEASURABLE" and res["S3-6"]["verdict"] == "PASS"

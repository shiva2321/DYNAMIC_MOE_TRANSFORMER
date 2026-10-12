"""M0/M1: the pre-registered decision rules (research/growmoe_results/m0/PREREG.md section 6), computed from per-seed run records.

A run record is a dict with at least ``stream``, ``method``, ``seed`` and ``acc`` (final mean accuracy in [0, 1]); resource statements
also read ``params``, ``naive_params`` and ``predict_macs``. Differences are paired on the seeds both methods ran.
"""

import json

import pytest

from growmoe.fairfight import EXPECTED, kill_rule, verdicts

SEEDS = range(10)
BASE = [0.40 + 0.01 * i for i in SEEDS]                                                 # A's accuracy on every stream, seed by seed


def _runs(stream, method, accs, **extra):
    return [dict(stream=stream, method=method, seed=i, acc=a, **extra) for i, a in zip(SEEDS, accs)]


def _vs_a(stream, method, diff_pts, wobble=0.5, seeds=SEEDS):
    """Records for A plus for ``method``, whose paired difference to A (method - A) is diff_pts, +/- wobble alternating by seed."""
    other = [x + (diff_pts + (wobble if i % 2 else -wobble)) / 100.0 for i, x in enumerate(BASE)]
    return _runs(stream, "A", BASE) + [r for r in _runs(stream, method, other) if r["seed"] in seeds]


def test_a1_holds_when_the_tuned_replay_net_is_within_a_point_of_a_on_both_streams():
    runs = _vs_a("S2-5", "B2", -0.5) + _vs_a("S1", "B2", +3.0)
    assert verdicts(runs)["A1"]["verdict"] == "PASS"


def test_a1_is_partial_when_a_still_beats_the_replay_net_on_one_stream_and_fails_when_it_does_on_both():
    one = verdicts(_vs_a("S2-5", "B2", -0.5) + _vs_a("S1", "B2", -6.0))
    both = verdicts(_vs_a("S2-5", "B2", -3.0) + _vs_a("S1", "B2", -6.0))
    assert one["A1"]["verdict"] == "PARTIAL" and both["A1"]["verdict"] == "FAIL"


def test_a2_needs_the_lookup_to_lead_by_five_points_on_both_streams():
    ok = verdicts(_vs_a("S2-5", "kNN_all", 8.0) + _vs_a("S1", "kNN_all", 5.5))
    short = verdicts(_vs_a("S2-5", "kNN_all", 8.0) + _vs_a("S1", "kNN_all", 4.0))
    assert ok["A2"]["verdict"] == "PASS" and short["A2"]["verdict"] == "PARTIAL"


def test_b1_needs_both_the_five_point_margin_and_an_interval_that_excludes_zero():
    tight = _vs_a("F-15", "B2star", -6.0) + _vs_a("F-30", "B2star", -6.0)
    assert verdicts(tight)["B1"]["verdict"] == "PASS"
    noisy = _vs_a("F-15", "B2star", -6.0, wobble=14.0) + _vs_a("F-30", "B2star", -6.0)  # +/-14 per seed: the interval contains 0
    out = verdicts(noisy)["B1"]
    assert out["verdict"] == "PARTIAL" and out["detail"]["F-15"]["holds"] is False and out["detail"]["F-30"]["holds"] is True
    small = _vs_a("F-15", "B2star", -3.0) + _vs_a("F-30", "B2star", -4.0)
    assert verdicts(small)["B1"]["verdict"] == "FAIL"


def test_b1_reports_the_numbers_it_decided_on():
    d = verdicts(_vs_a("F-15", "B2star", -6.0) + _vs_a("F-30", "B2star", -6.0))["B1"]["detail"]["F-15"]
    assert d["n"] == 10 and d["mean_diff_points"] == pytest.approx(6.0) and 0 < d["ci_lo"] < 6.0 < d["ci_hi"]


def test_b2_and_b3_use_ten_and_five_point_margins():
    runs = _vs_a("F-15", "kNN_all", -12.0) + _vs_a("F-30", "kNN_all", -11.0) + _vs_a("F-30", "B4", +5.5)
    v = verdicts(runs)
    assert v["B2"]["verdict"] == "PASS" and v["B3"]["verdict"] == "PASS"
    weak = verdicts(_vs_a("F-15", "kNN_all", -12.0) + _vs_a("F-30", "kNN_all", -9.0) + _vs_a("F-30", "B4", +2.0))
    assert weak["B2"]["verdict"] == "PARTIAL" and weak["B3"]["verdict"] == "FAIL"


def test_pairs_use_only_the_seeds_both_methods_ran():
    runs = _vs_a("F-15", "B2star", -6.0, seeds=range(5)) + _vs_a("F-30", "B2star", -6.0)
    assert verdicts(runs)["B1"]["detail"]["F-15"]["n"] == 5


def test_a_statement_with_a_missing_stream_or_method_is_not_measurable_rather_than_guessed():
    assert verdicts(_vs_a("F-15", "B2star", -6.0))["B1"]["verdict"] == "NOT MEASURABLE"
    assert verdicts([])["A1"]["verdict"] == "NOT MEASURABLE"
    assert set(verdicts([])) == set(EXPECTED)


def test_a3_compares_unique_parameters_and_flags_the_double_count():
    a = _runs("S2-5", "A", [0.5] * 5, params=100_000)
    ok = verdicts(a + _runs("S2-5", "C", [0.4] * 5, params=110_000, naive_params=300_000))["A3"]
    assert ok["verdict"] == "PASS" and ok["detail"]["ratio"] == pytest.approx(1.1)
    inflated = verdicts(a + _runs("S2-5", "C", [0.4] * 5, params=110_000, naive_params=150_000))["A3"]      # naive < 2 x unique
    far = verdicts(a + _runs("S2-5", "C", [0.4] * 5, params=40_000, naive_params=300_000))["A3"]            # ratio 0.4
    assert inflated["verdict"] == "FAIL" and far["verdict"] == "FAIL"


def test_c1_reports_whether_the_pool_saves_compute_against_a_lookup():
    knn = _runs("S2-30", "kNN_all", [0.6] * 10, predict_macs=1_000_000)
    assert verdicts(_runs("S2-30", "A", [0.5] * 10, predict_macs=900_000) + knn)["C1"]["verdict"] == "PASS"   # 0.9 >= 0.5: no saving
    saving = verdicts(_runs("S2-30", "A", [0.5] * 10, predict_macs=300_000) + knn)["C1"]
    assert saving["verdict"] == "FAIL" and saving["detail"]["ratio"] == pytest.approx(0.3)


def test_c2_compares_the_same_method_on_two_streams():
    runs = _runs("S1", "A", BASE) + _runs("S1-PRE", "A", [x + 0.03 for x in BASE])
    out = verdicts(runs)["C2"]
    assert out["verdict"] == "PASS" and out["detail"]["mean_diff_points"] == pytest.approx(3.0)


def test_every_verdict_carries_its_expectation_and_whether_the_result_matched_it():
    v = verdicts(_vs_a("F-15", "B2star", -1.0) + _vs_a("F-30", "B2star", -1.0))
    assert v["B1"]["expected"] == EXPECTED["B1"] == "PASS" and v["B1"]["as_expected"] is False
    assert v["A1"]["as_expected"] is None                                                                   # not measurable: no claim either way


def test_the_kill_rule_follows_b1():
    fail = verdicts(_vs_a("F-15", "B2star", -1.0) + _vs_a("F-30", "B2star", -1.0))
    partial = verdicts(_vs_a("F-15", "B2star", -6.0) + _vs_a("F-30", "B2star", -1.0))
    ok = verdicts(_vs_a("F-15", "B2star", -6.0) + _vs_a("F-30", "B2star", -6.0))
    assert "stops claiming an accuracy advantage" in kill_rule(fail)
    assert "continues only to resolve" in kill_rule(partial)
    assert "not triggered" in kill_rule(ok)
    assert "cannot be applied" in kill_rule(verdicts([]))


def test_verdicts_survive_a_json_round_trip():
    runs = _vs_a("F-15", "B2star", -6.0) + _vs_a("F-30", "B2star", -6.0)
    assert verdicts(json.loads(json.dumps(runs))) == verdicts(runs)
    json.dumps(verdicts(runs))                                                                              # plain data only

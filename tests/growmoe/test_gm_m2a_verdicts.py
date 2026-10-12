"""M2-A: the pre-registered decision rules (research/growmoe_results/m2a/PREREG.md), computed from per-seed run records.

A run record has ``stream``, ``method``, ``seed``, ``acc`` (final mean accuracy in [0, 1]) and ``event_counts``. Methods: A_old (M0 frozen
pool), A_new (monitor armed after infancy, re-tuned budget), B2star (M0's best single network), B4, kNN_all. Differences are paired on
the seeds both methods ran and expressed in percentage points.
"""

import json

import pytest

from growmoe.fairfight import EXPECTED_M2A, m2a_decision, m2a_verdicts

SEEDS = range(10)
BASE = [0.40 + 0.01 * i for i in SEEDS]


def _runs(stream, method, accs, alarms=None):
    return [dict(stream=stream, method=method, seed=i, acc=a, event_counts={} if alarms is None else {"alarm": alarms[i]})
            for i, a in zip(SEEDS, accs)]


def _pair(stream, new, ref, diff_pts, wobble=0.5, **kw):
    """``new`` scores ``diff_pts`` (+/- wobble, alternating by seed) points above ``ref`` on every seed."""
    new_accs = [x + (diff_pts + (wobble if i % 2 else -wobble)) / 100.0 for i, x in enumerate(BASE)]
    return _runs(stream, ref, BASE) + _runs(stream, new, new_accs, **kw)


def _world(p1_f15=9.0, p1_f30=7.0, guard_s1=0.0, alarms=3):
    """A complete set of records: A_new leads B2* by p1_* points, A_old sits 5 points below B2*, guards move by guard_s1 on S1."""
    runs = []
    for stream, over_b2 in (("F-15", p1_f15), ("F-30", p1_f30)):
        runs += _runs(stream, "B2star", BASE)
        runs += _runs(stream, "A_old", [x - 0.05 for x in BASE])
        runs += _runs(stream, "A_new", [x + (over_b2 + (0.5 if i % 2 else -0.5)) / 100.0 for i, x in enumerate(BASE)])
    for stream, diff in (("S1", guard_s1), ("S2-5", 0.0), ("S2-30", 0.0), ("S4", 1.0)):
        runs += _pair(stream, "A_new", "A_old", diff, wobble=2.0)
    return runs + _runs("S7", "A_new", [0.3] * 10, alarms=[alarms] * 10)


# ------------------------------------------------------------------------------------------------------------ statements
def test_p1_is_the_b1_rule_five_points_over_the_best_single_network_with_an_interval_above_zero():
    assert m2a_verdicts(_pair("F-15", "A_new", "B2star", 9.0) + _pair("F-30", "A_new", "B2star", 7.0))["P1"]["verdict"] == "PASS"
    assert m2a_verdicts(_pair("F-15", "A_new", "B2star", 9.0) + _pair("F-30", "A_new", "B2star", 3.0))["P1"]["verdict"] == "PARTIAL"
    assert m2a_verdicts(_pair("F-15", "A_new", "B2star", 4.0) + _pair("F-30", "A_new", "B2star", 3.0))["P1"]["verdict"] == "FAIL"
    noisy = _pair("F-15", "A_new", "B2star", 6.0, wobble=14.0) + _pair("F-30", "A_new", "B2star", 7.0)       # +/-14 per seed: interval contains 0
    out = m2a_verdicts(noisy)["P1"]
    assert out["verdict"] == "PARTIAL" and out["detail"]["F-15"]["holds"] is False


def test_p2_asks_that_the_fix_itself_moves_the_pool_by_three_points_on_both_function_streams():
    ok = _pair("F-15", "A_new", "A_old", 7.0) + _pair("F-30", "A_new", "A_old", 3.5)
    short = _pair("F-15", "A_new", "A_old", 7.0) + _pair("F-30", "A_new", "A_old", 2.0)
    assert m2a_verdicts(ok)["P2"]["verdict"] == "PASS"
    assert m2a_verdicts(short)["P2"]["verdict"] == "PARTIAL"


@pytest.mark.parametrize("gid, stream", [("G1", "S1"), ("G2", "S2-5"), ("G3", "S2-30"), ("G4", "S4")])
def test_a_guard_holds_unless_the_new_pool_is_three_points_worse_or_significantly_worse(gid, stream):
    assert m2a_verdicts(_pair(stream, "A_new", "A_old", 0.0, wobble=2.0))[gid]["verdict"] == "PASS"             # no change
    assert m2a_verdicts(_pair(stream, "A_new", "A_old", -2.0, wobble=5.0))[gid]["verdict"] == "PASS"           # -2 but the interval spans 0
    assert m2a_verdicts(_pair(stream, "A_new", "A_old", -3.5, wobble=5.0))[gid]["verdict"] == "FAIL"           # more than 3 points worse
    assert m2a_verdicts(_pair(stream, "A_new", "A_old", -1.0, wobble=0.2))[gid]["verdict"] == "FAIL"           # small but clearly below 0
    assert m2a_verdicts(_pair(stream, "A_new", "A_old", +2.0, wobble=0.2))[gid]["verdict"] == "PASS"           # better is fine


def test_g5_checks_that_the_alarm_still_fires_on_a_label_remap_stream():
    ok = m2a_verdicts(_runs("S7", "A_new", [0.3] * 10, alarms=[4, 3, 3, 4, 3, 4, 3, 3, 4, 3]))["G5"]
    assert ok["verdict"] == "PASS" and ok["detail"]["mean_alarms"] == pytest.approx(3.4)
    silent = m2a_verdicts(_runs("S7", "A_new", [0.3] * 10, alarms=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]))["G5"]
    assert silent["verdict"] == "FAIL"


def test_missing_streams_or_methods_are_not_measurable_not_guessed():
    out = m2a_verdicts(_pair("F-15", "A_new", "B2star", 9.0))
    assert out["P1"]["verdict"] == "NOT MEASURABLE" and out["P2"]["verdict"] == "NOT MEASURABLE"
    assert all(v["verdict"] == "NOT MEASURABLE" for v in m2a_verdicts([]).values())
    assert set(m2a_verdicts([])) == set(EXPECTED_M2A)


def test_every_verdict_says_whether_it_matched_the_expectation_written_beforehand():
    v = m2a_verdicts(_world(p1_f15=1.0, p1_f30=1.0))
    assert v["P1"]["expected"] == EXPECTED_M2A["P1"] and v["P1"]["as_expected"] is False
    assert v["G1"]["as_expected"] is True
    assert m2a_verdicts([])["G1"]["as_expected"] is None


# ------------------------------------------------------------------------------------------------------------ decision
def test_the_decision_follows_p1_and_the_guards():
    assert "adopt" in m2a_decision(m2a_verdicts(_world()))
    assert "keep it as an option" in m2a_decision(m2a_verdicts(_world(guard_s1=-6.0)))           # P1 passes, a guard fails
    assert "continues only to resolve" in m2a_decision(m2a_verdicts(_world(p1_f30=1.0)))         # P1 PARTIAL
    assert "stops claiming an accuracy advantage" in m2a_decision(m2a_verdicts(_world(p1_f15=1.0, p1_f30=1.0)))   # P1 FAIL
    assert "keep it as an option" in m2a_decision(m2a_verdicts(_world(alarms=0)))                # the alarm no longer fires on remaps
    assert "cannot be applied" in m2a_decision(m2a_verdicts([]))


def test_verdicts_survive_a_json_round_trip():
    runs = _world()
    assert m2a_verdicts(json.loads(json.dumps(runs))) == m2a_verdicts(runs)
    json.dumps(m2a_verdicts(runs))

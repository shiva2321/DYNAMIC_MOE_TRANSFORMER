"""RV2-A: the pre-registered decision rules (research/growmoe_results/rv2a/PREREG.md), computed from per-seed run records.

Arms: A_new (M2-A default, per-sample assignment), A_vote (batch vote), A_voter (restricted vote, only if the fallback runs). Differences are
paired on the seeds both arms ran and expressed in percentage points. MIX-* streams carry the accuracy on the minority regimes in ``acc``.
"""

import json

import pytest

from growmoe.fairfight import EXPECTED_RV2A, rv2a_decision, rv2a_verdicts

SEEDS = range(20)
BASE = [0.40 + 0.005 * i for i in SEEDS]


def _runs(stream, method, accs, alarms=None):
    return [dict(stream=stream, method=method, seed=i, acc=a, event_counts={} if alarms is None else {"alarm": alarms[i]}) for i, a in zip(SEEDS, accs)]


def _pair(stream, cand, ref, diff_pts, wobble=0.5):
    new = [x + (diff_pts + (wobble if i % 2 else -wobble)) / 100.0 for i, x in enumerate(BASE)]
    return _runs(stream, ref, BASE) + _runs(stream, cand, new)


def _world(cand="A_vote", v1=12.0, v2_f15=3.0, v2_f30=2.5, v3=6.0, guard=0.0, mix=0.0, alarms=4):
    runs = []
    for stream, diff in (("OV-0.3", v1), ("F-15", v2_f15), ("F-30", v2_f30), ("OV-0.6", v3)):
        runs += _pair(stream, cand, "A_new", diff)
    for stream in ("S1", "S2-5", "S2-30", "S4", "OV-0.0", "OV-0.3-MIXED"):
        runs += _pair(stream, cand, "A_new", guard, wobble=2.0)
    for stream in ("MIX-0.0", "MIX-0.3"):
        runs += _pair(stream, cand, "A_new", mix, wobble=2.0)
    return runs + _runs("S7", cand, [0.3] * 20, alarms=[alarms] * 20)


def test_v1_needs_the_bar_and_an_interval_above_zero_on_the_overlap_stream():
    assert rv2a_verdicts(_pair("OV-0.3", "A_vote", "A_new", 12.0))["V1"]["verdict"] == "PASS"
    assert rv2a_verdicts(_pair("OV-0.3", "A_vote", "A_new", 5.0))["V1"]["verdict"] == "FAIL"           # below +8
    assert rv2a_verdicts(_pair("OV-0.3", "A_vote", "A_new", 9.0, wobble=40.0))["V1"]["verdict"] == "FAIL"   # mean fine, interval contains 0


def test_v2_is_a_both_streams_statement():
    both = _pair("F-15", "A_vote", "A_new", 3.0) + _pair("F-30", "A_vote", "A_new", 2.0)
    one = _pair("F-15", "A_vote", "A_new", 3.0) + _pair("F-30", "A_vote", "A_new", 0.2)
    none = _pair("F-15", "A_vote", "A_new", 0.2) + _pair("F-30", "A_vote", "A_new", 0.3)
    assert [rv2a_verdicts(r)["V2"]["verdict"] for r in (both, one, none)] == ["PASS", "PARTIAL", "FAIL"]


def test_v3_asks_for_three_points_at_high_overlap():
    assert rv2a_verdicts(_pair("OV-0.6", "A_vote", "A_new", 6.0))["V3"]["verdict"] == "PASS"
    assert rv2a_verdicts(_pair("OV-0.6", "A_vote", "A_new", 2.0))["V3"]["verdict"] == "FAIL"


def test_the_bars_can_be_lowered_by_the_preregistration():
    runs = _pair("OV-0.3", "A_vote", "A_new", 5.0)
    assert rv2a_verdicts(runs, bars={"V1": 4.0})["V1"]["verdict"] == "PASS"
    assert "+4.0" in rv2a_verdicts(runs, bars={"V1": 4.0})["V1"]["rule"]


@pytest.mark.parametrize("gid, stream", [("G1", "S1"), ("G2", "S2-5"), ("G3", "S2-30"), ("G4", "S4"), ("G5", "OV-0.0"), ("G6", "OV-0.3-MIXED")])
def test_a_guard_holds_unless_the_candidate_is_three_points_worse_or_significantly_worse(gid, stream):
    assert rv2a_verdicts(_pair(stream, "A_vote", "A_new", 0.0, wobble=2.0))[gid]["verdict"] == "PASS"
    assert rv2a_verdicts(_pair(stream, "A_vote", "A_new", -2.0, wobble=8.0))[gid]["verdict"] == "PASS"       # -2 but the interval spans 0
    assert rv2a_verdicts(_pair(stream, "A_vote", "A_new", -3.5, wobble=5.0))[gid]["verdict"] == "FAIL"       # more than 3 points worse
    assert rv2a_verdicts(_pair(stream, "A_vote", "A_new", -1.0, wobble=0.2))[gid]["verdict"] == "FAIL"       # small but clearly below 0
    assert rv2a_verdicts(_pair(stream, "A_vote", "A_new", +2.0, wobble=0.2))[gid]["verdict"] == "PASS"


def test_g7_checks_that_the_alarm_still_fires_on_a_label_remap_stream():
    ok = rv2a_verdicts(_runs("S7", "A_vote", [0.3] * 20, alarms=[4, 3] * 10))["G7"]
    assert ok["verdict"] == "PASS" and ok["detail"]["mean_alarms"] == pytest.approx(3.5)
    assert rv2a_verdicts(_runs("S7", "A_vote", [0.3] * 20, alarms=[0, 1] * 10))["G7"]["verdict"] == "FAIL"


def test_gm_covers_both_mixed_tail_streams():
    ok = _pair("MIX-0.0", "A_vote", "A_new", -1.0, wobble=3.0) + _pair("MIX-0.3", "A_vote", "A_new", 8.0, wobble=3.0)
    bad = _pair("MIX-0.0", "A_vote", "A_new", -6.0, wobble=3.0) + _pair("MIX-0.3", "A_vote", "A_new", 8.0, wobble=3.0)
    assert rv2a_verdicts(ok)["GM"]["verdict"] == "PASS"
    assert rv2a_verdicts(bad)["GM"]["verdict"] == "PARTIAL"


def test_missing_data_is_not_measurable_and_the_ids_are_the_expected_ones():
    assert all(v["verdict"] == "NOT MEASURABLE" for v in rv2a_verdicts([]).values())
    assert set(rv2a_verdicts([])) == set(EXPECTED_RV2A) and len(EXPECTED_RV2A) == 11


def test_every_verdict_says_whether_it_matched_the_expectation_written_beforehand():
    v = rv2a_verdicts(_world(v1=1.0))
    assert v["V1"]["expected"] == "PASS" and v["V1"]["as_expected"] is False and v["G1"]["as_expected"] is True
    assert rv2a_verdicts(_world(v1=1.0), expected={"V1": "FAIL"})["V1"]["as_expected"] is True
    assert rv2a_verdicts([])["G1"]["as_expected"] is None


def test_the_candidate_can_be_the_restricted_vote():
    v = rv2a_verdicts(_world(cand="A_voter"), candidate="A_voter")
    assert v["V1"]["verdict"] == "PASS" and v["G7"]["verdict"] == "PASS"


def test_the_decision_follows_v1_and_the_guards_with_the_restricted_fallback():
    good, bad_guard = rv2a_verdicts(_world()), rv2a_verdicts(_world(guard=-6.0))
    assert "adopt the plain vote" in rv2a_decision(good)
    assert "run the restricted vote" in rv2a_decision(bad_guard)
    assert "Stop" in rv2a_decision(rv2a_verdicts(_world(v1=1.0)))
    assert "cannot be applied" in rv2a_decision(rv2a_verdicts([]))
    restricted_ok = rv2a_verdicts(_world(cand="A_voter"), candidate="A_voter")
    restricted_bad = rv2a_verdicts(_world(cand="A_voter", v1=1.0), candidate="A_voter")
    assert "adopt the restricted vote" in rv2a_decision(bad_guard, restricted_ok)
    assert "adopt neither" in rv2a_decision(bad_guard, restricted_bad)


def test_verdicts_survive_a_json_round_trip():
    runs = _world()
    assert rv2a_verdicts(json.loads(json.dumps(runs))) == rv2a_verdicts(runs)
    json.dumps(rv2a_verdicts(runs))

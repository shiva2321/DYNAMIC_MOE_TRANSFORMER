"""RV2-B: the pre-registered decision rules (research/growmoe_results/rv2b/PREREG.md), computed from per-seed run records and scaling_check.json."""

import json

import pytest

from growmoe.fairfight import EXPECTED_RV2B, rv2b_decision, rv2b_verdicts

SEEDS = range(20)


def _rec(stream, method, seed, acc, macs=1000.0, spawn=10, n_experts=10):
    return dict(stream=stream, method=method, seed=seed, acc=acc, gate_macs_pred=macs, event_counts={"spawn": spawn}, n_experts=n_experts)


def _world(gap=0.0, wobble=0.2, ratios=(0.10, 0.12, 0.15), spawn_gap=0, scaling_time=0.35, g_differs=False):
    runs = []
    for stream, ratio in (("F-30", ratios[0]), ("S2-30", ratios[1]), ("F-15", ratios[2]), ("OV-0.3", 0.5)):
        for s in SEEDS:
            base = 0.40 + 0.005 * s
            runs.append(_rec(stream, "A_exact", s, base))
            runs.append(_rec(stream, "A_cascade", s, base - gap / 100.0 + (wobble if s % 2 else -wobble) / 100.0, macs=1000.0 * ratio, spawn=10 + spawn_gap))
    for stream in ("S1", "S2-5"):
        for s in SEEDS:
            r = _rec(stream, "A_exact", s, 0.7 + 0.01 * (s % 3))
            runs += [r, dict(r, method="A_cascade", acc=r["acc"] + (0.01 if g_differs else 0.0))]
    scaling = {"rows": {"500": {"64": {"ratio_macs": 0.03, "ratio_time": 0.3}, "1024": {"ratio_macs": 0.03, "ratio_time": scaling_time}}}}
    return runs, scaling


def test_c1_is_a_non_inferiority_statement_on_four_streams():
    ok, sc = _world(gap=0.3)
    assert rv2b_verdicts(ok, sc)["C1"]["verdict"] == "PASS"
    worse, sc = _world(gap=3.0)
    assert rv2b_verdicts(worse, sc)["C1"]["verdict"] == "FAIL"
    noisy, sc = _world(gap=0.5, wobble=15.0)                                            # mean fine, interval lower bound far below -2
    assert rv2b_verdicts(noisy, sc)["C1"]["verdict"] == "FAIL"


def test_c2_compares_the_number_of_spawns():
    assert rv2b_verdicts(*_world(spawn_gap=0))["C2"]["verdict"] == "PASS"
    assert rv2b_verdicts(*_world(spawn_gap=2))["C2"]["verdict"] == "FAIL"


def test_c3_reads_the_measured_gate_cost_ratio_per_stream():
    assert rv2b_verdicts(*_world())["C3"]["verdict"] == "PASS"
    out = rv2b_verdicts(*_world(ratios=(0.30, 0.12, 0.15)))["C3"]                      # F-30 above its 0.20 bar
    assert out["verdict"] == "PARTIAL" and out["detail"]["F-30"]["holds"] is False


def test_c4_reads_the_scaling_check():
    assert rv2b_verdicts(*_world())["C4"]["verdict"] == "PASS"
    assert rv2b_verdicts(*_world(scaling_time=0.8))["C4"]["verdict"] == "FAIL"
    assert rv2b_verdicts(_world()[0])["C4"]["verdict"] == "NOT MEASURABLE"


def test_g_requires_bit_identical_accuracy_where_the_pool_is_too_small_for_the_cascade():
    assert rv2b_verdicts(*_world())["G"]["verdict"] == "PASS"
    assert rv2b_verdicts(*_world(g_differs=True))["G"]["verdict"] == "FAIL"


def test_missing_data_is_not_measurable_and_the_ids_are_the_expected_ones():
    assert all(v["verdict"] == "NOT MEASURABLE" for v in rv2b_verdicts([]).values())
    assert set(rv2b_verdicts([])) == set(EXPECTED_RV2B)


def test_every_verdict_says_whether_it_matched_the_expectation_written_beforehand():
    v = rv2b_verdicts(*_world(gap=3.0))
    assert v["C1"]["as_expected"] is False and v["G"]["as_expected"] is True


def test_the_decision_adopts_only_when_correctness_and_cost_both_pass():
    assert "adopt the cascade" in rv2b_decision(rv2b_verdicts(*_world()), k=4, min_experts=16)
    assert "opt-in" in rv2b_decision(rv2b_verdicts(*_world(scaling_time=0.8)))          # C1 passes, C4 fails
    assert "opt-in" in rv2b_decision(rv2b_verdicts(*_world(ratios=(0.30, 0.12, 0.15))))   # C3 not PASS
    assert "not adopted" in rv2b_decision(rv2b_verdicts(*_world(gap=3.0)))
    assert "Stop" in rv2b_decision(rv2b_verdicts(*_world(g_differs=True)))
    assert "cannot be applied" in rv2b_decision(rv2b_verdicts([]))


def test_verdicts_survive_a_json_round_trip():
    runs, sc = _world()
    assert rv2b_verdicts(json.loads(json.dumps(runs)), sc) == rv2b_verdicts(runs, sc)

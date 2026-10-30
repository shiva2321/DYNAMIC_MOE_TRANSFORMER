"""Outcome-calibrated readers (routing_general/outcome.py): splitting the stream's known messages, drawing the verified sample, and scoring a system calibrated on it."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general"):
    sys.path.insert(0, str(p))

import outcome as OC  # noqa: E402


def test_the_split_is_stratified_disjoint_complete_and_reproducible():
    owner = [0] * 20 + [1] * 10 + [2] * 3
    a, b = OC.split_pool_eval(owner, np.random.default_rng(1))
    assert sorted(a + b) == list(range(33)) and not set(a) & set(b)
    for c, n in ((0, 20), (1, 10)):
        assert sum(owner[i] == c for i in a) == sum(owner[i] == c for i in b) == n // 2
    assert sum(owner[i] == 2 for i in a) + sum(owner[i] == 2 for i in b) == 3
    assert OC.split_pool_eval(owner, np.random.default_rng(1)) == (a, b)
    assert OC.split_pool_eval(owner, np.random.default_rng(2)) != (a, b)


def test_the_verified_sample_has_only_verified_messages_in_a_random_order_and_never_more_than_asked():
    verified = np.array([True, False, True, True, False, True, True, False])
    s = OC.verified_sample(list(range(8)), verified, 3, np.random.default_rng(0))
    assert len(s) == 3 and all(verified[i] for i in s) and len(set(s)) == 3
    assert sorted(OC.verified_sample(list(range(8)), verified, 99, np.random.default_rng(0))) == [0, 2, 3, 5, 6]        # fewer verified than asked: all of them
    assert OC.verified_sample([1, 4, 7], verified, 2, np.random.default_rng(0)) == []                                    # none verified
    assert OC.verified_sample(list(range(8)), verified, 3, np.random.default_rng(0)) == s                                # reproducible
    assert OC.verified_sample(list(range(8)), verified, 3, np.random.default_rng(5)) != s


def synthetic(shift, rng, n=400):
    """Two readers' novelty scores: 'fixed' sees a shift in the stream (known messages score higher than in calibration), 'steady' does not. Never-taught messages score high in both."""
    known_cal = {"steady": rng.normal(0, 1, n), "fixed": rng.normal(0, 1, n)}
    known = {"steady": rng.normal(0, 1, n), "fixed": rng.normal(shift, 1, n)}
    unknown = {"steady": rng.normal(3, 1, n), "fixed": rng.normal(3 + shift, 1, n)}
    foreign = {"steady": rng.normal(5, 1, n), "fixed": rng.normal(5 + shift, 1, n)}
    return known_cal, known, unknown, foreign


def test_calibrating_on_the_streams_own_wording_restores_the_false_flag_rate_a_fixed_calibration_loses():
    rng = np.random.default_rng(0)
    cal, known, unknown, foreign = synthetic(2.0, rng)
    ev = {"known": known, "unknown": unknown, "foreign": foreign}
    fixed = OC.system_metrics(cal, ev, ["fixed"], "either", 0.05)
    assert fixed["known_flagged"] > 0.4                                                       # the old calibration flags a lot of the shifted known messages
    sample = {r: known[r][:60] for r in known}                                                 # 60 verified messages of the stream's own wording
    ev2 = {"known": {r: known[r][60:] for r in known}, "unknown": unknown, "foreign": foreign}
    adapted = OC.system_metrics(sample, ev2, ["fixed"], "either", 0.05)
    assert adapted["known_flagged"] <= 0.10 and 0.0 < adapted["unknown_found"] < 1.0


def test_fusing_two_calibrated_readers_keeps_the_false_flag_rate_at_the_level_and_the_power_of_the_better_one():
    rng = np.random.default_rng(1)
    cal, known, unknown, foreign = synthetic(0.0, rng)
    ev = {"known": known, "unknown": unknown, "foreign": foreign}
    single = OC.system_metrics(cal, ev, ["steady"], "either", 0.05)
    either = OC.system_metrics(cal, ev, ["steady", "fixed"], "either", 0.05)
    both = OC.system_metrics(cal, ev, ["steady", "fixed"], "both", 0.05)
    assert abs(either["known_flagged"] - 0.05) < 0.04 and abs(single["known_flagged"] - 0.05) < 0.04
    assert either["unknown_found"] >= single["unknown_found"] - 0.15 and abs(both["known_flagged"] - 0.05) < 0.04                # both are recalibrated to the same false-flag rate
    assert set(single) >= {"known_flagged", "unknown_found", "foreign_found", "auroc_unknown"}


def test_a_calibration_sample_too_small_to_flag_anything_flags_nothing_and_is_reported():
    rng = np.random.default_rng(2)
    cal, known, unknown, foreign = synthetic(0.0, rng)
    ev = {"known": known, "unknown": unknown, "foreign": foreign}
    tiny = OC.system_metrics({r: v[:10] for r, v in cal.items()}, ev, ["steady"], "either", 0.05)
    assert tiny["known_flagged"] == 0.0 and tiny["unknown_found"] == 0.0
    with pytest.raises(ValueError):
        OC.system_metrics({}, ev, ["steady"], "either", 0.05)


def test_averaging_draws_ignores_nothing_and_reports_the_number_of_draws():
    rows = [{"a": 0.2, "b": float("nan")}, {"a": 0.4, "b": 0.6}]
    m = OC.mean_over_draws(rows)
    assert m["a"] == pytest.approx(0.3) and m["b"] == pytest.approx(0.6) and m["draws"] == 2


def test_verified_order_is_the_sample_before_it_is_cut_and_keeps_the_old_sample_unchanged():
    verified = np.array([True, False, True, True, False, True, True, False])
    order = OC.verified_order(list(range(8)), verified, np.random.default_rng(3))
    assert sorted(order) == [0, 2, 3, 5, 6]
    assert OC.verified_sample(list(range(8)), verified, 3, np.random.default_rng(3)) == order[:3]
    assert OC.verified_order([1, 4, 7], verified, np.random.default_rng(3)) == []

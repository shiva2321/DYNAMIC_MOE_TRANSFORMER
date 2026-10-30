"""Measures of the general routing lab (routing_general/measures.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
sys.path.insert(0, str(_KS / "routing_general"))

import measures as MS  # noqa: E402
import route as RT  # noqa: E402

STATES = ["solid", "solid"]


def row(*p):
    return np.array(list(p) + [1.0 - sum(p)])


def test_auroc_counts_ties_as_half_and_checks_inputs():
    assert MS.auroc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert MS.auroc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert MS.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert MS.auroc([0.1, 0.5], [0.5, 0.9]) == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    with pytest.raises(ValueError):
        MS.auroc([], [1.0])
    with pytest.raises(ValueError):
        MS.auroc([1.0], [])


def test_flag_rate_is_the_share_strictly_above_the_threshold():
    assert MS.flag_rate([0.1, 0.5, 0.9], 0.5) == pytest.approx(1 / 3)
    assert MS.flag_rate([0.1, 0.5], float("inf")) == 0.0
    assert np.isnan(MS.flag_rate([], 0.5))


def test_claim_rates_split_the_three_kinds_of_message():
    known = RT.decide(np.stack([row(0.9, 0.05), row(0.05, 0.9), row(0.1, 0.1), row(0.9, 0.05)]), STATES, tau=0.5)
    truth = [0, 0, 1, 1]                                                          # claimed rightly, claimed wrongly, unclaimed, claimed wrongly
    unknown = RT.decide(np.stack([row(0.1, 0.1), row(0.9, 0.0)]), STATES, tau=0.5)
    foreign = RT.decide(np.stack([row(0.1, 0.1)]), STATES, tau=0.5)
    r = MS.claim_rates(known, truth, unknown, foreign)
    assert r["known_claimed"] == pytest.approx(0.75)
    assert r["known_right"] == pytest.approx(0.25)
    assert r["right_given_claimed"] == pytest.approx(1 / 3)
    assert r["unknown_unclaimed"] == pytest.approx(0.5)
    assert r["foreign_unclaimed"] == 1.0
    with pytest.raises(ValueError):
        MS.claim_rates(known, truth[:2], unknown, foreign)


def test_right_given_claimed_is_nan_when_nothing_is_claimed():
    known = RT.decide(np.stack([row(0.1, 0.1)]), STATES, tau=0.5)
    r = MS.claim_rates(known, [0], known, known)
    assert np.isnan(r["right_given_claimed"]) and r["known_claimed"] == 0.0


def test_operating_point_uses_the_calibration_threshold_and_reports_every_kind():
    cal = np.linspace(0.0, 0.2, 99)                                                # held-out known messages
    known, unknown, foreign = np.linspace(0.0, 0.2, 100), np.full(10, 0.9), np.array([0.9, 0.9, 0.0, 0.0])
    r = MS.operating_point(cal, known, unknown, foreign, q=0.05)
    assert r["auroc_unknown"] == 1.0
    assert r["auroc_foreign"] == pytest.approx((2 * 1.0 + 2 * 0.5 / 100) / 4)                  # the two 0.9 beat everything; each 0.0 only ties the one known 0.0
    assert r["known_flagged"] <= 0.07                                              # about q
    assert r["unknown_found"] == 1.0
    assert r["foreign_found"] == 0.5
    inf = MS.operating_point(cal[:5], known, unknown, foreign, q=0.05)             # too few calibration scores to ever flag at 5 %
    assert inf["unknown_found"] == 0.0 and inf["known_flagged"] == 0.0

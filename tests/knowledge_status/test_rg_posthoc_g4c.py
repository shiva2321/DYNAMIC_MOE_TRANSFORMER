"""Post-hoc analysis of lab G4c (routing_general/posthoc_g4c.py): the corrected flag-rate formula and the clean false-flag / found curve the contaminated points are read against."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general"):
    sys.path.insert(0, str(p))

import outcome as OC  # noqa: E402
import posthoc_g4c as PH  # noqa: E402
import robust as RB  # noqa: E402


@pytest.mark.parametrize("n,k", [(30, 0), (30, 3), (40, 0), (40, 1), (40, 2), (40, 3), (60, 0), (60, 1), (60, 3), (60, 6)])
def test_the_corrected_flag_rate_matches_the_cut_the_code_really_sets(n, k):
    rng = np.random.default_rng(100 + n + k)
    rates = []
    for _ in range(300):
        ev = {"known": {"r": rng.normal(0, 1, 2000)}, "unknown": {"r": rng.normal(3, 1, 10)}, "foreign": {"r": rng.normal(6, 1, 10)}}
        rates.append(OC.system_metrics(RB.trim_top({"r": rng.normal(0, 1, n)}, k), ev, ["r"], "either", 0.05)["known_flagged"])
    assert np.mean(rates) == pytest.approx(PH.corrected_flag_rate(n, k, 0.05), abs=0.012)


def test_the_corrected_formula_differs_from_k_plus_one_over_n_plus_one_exactly_where_the_conformal_rank_moves():
    assert PH.corrected_flag_rate(30, 2, 0.05) == pytest.approx(3 / 31)                            # n - k <= 38: the cut is the (k+1)-th largest
    assert PH.corrected_flag_rate(60, 0, 0.05) == pytest.approx(3 / 61)                            # above that the rank gives a lower cut: 3 of 61, not 1 of 61
    assert PH.corrected_flag_rate(40, 1, 0.05) == PH.corrected_flag_rate(40, 2, 0.05)              # trimming one more changes nothing here (the same cut)
    assert PH.corrected_flag_rate(10, 0, 0.05) == 0.0                                              # too few scores: the cut is infinite, nothing is flagged


def test_the_curve_interpolates_the_clean_points_between_the_corners_and_never_falls():
    c = PH.Curve([(0.1, 0.2), (0.2, 0.5)])
    assert c(0.0) == 0.0 and c(1.0) == 1.0 and c(0.1) == pytest.approx(0.2) and c(0.15) == pytest.approx(0.35) and c(0.05) == pytest.approx(0.1)
    assert PH.Curve([(0.1, 0.2), (0.1, 0.4)])(0.1) == pytest.approx(0.3)                          # points at the same false-flag rate are averaged
    noisy = PH.Curve([(0.1, 0.5), (0.2, 0.4)])                                                     # a dip from noise is flattened, not followed
    assert noisy(0.2) >= noisy(0.1)
    assert PH.Curve([(0.1, 0.2)])(1.5) == 1.0 and PH.Curve([(0.1, 0.2)])(-0.1) == 0.0            # outside [0, 1] the ends are used


def test_power_lost_at_matched_false_flags_reads_a_point_against_the_curve():
    c = PH.Curve([(0.1, 0.2), (0.2, 0.5)])
    assert PH.power_lost(c, 0.15, 0.30) == pytest.approx(0.05)                                     # the curve would find 0.35 at this false-flag rate
    assert PH.power_lost(c, 0.15, 0.35) == pytest.approx(0.0) and PH.power_lost(c, 0.15, 0.40) == pytest.approx(-0.05)

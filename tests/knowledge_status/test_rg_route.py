"""The domain-free decision (routing_general/route.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_general"))

import route as RT  # noqa: E402

STATES = ["solid", "thin", "solid"]


def row(*p):
    p = list(p)
    p.append(1.0 - sum(p))                                   # the last column is "none of these"
    return np.array(p)


def test_confident_card_is_claimed_with_its_state():
    d = RT.decide(np.stack([row(0.97, 0.01, 0.01)]), STATES, tau=0.5)[0]
    assert (d.status, d.card) == ("solid", 0)
    assert d.p == pytest.approx(0.97)
    d = RT.decide(np.stack([row(0.02, 0.9, 0.03)]), STATES, tau=0.5)[0]
    assert (d.status, d.card) == ("thin", 1)


def test_none_on_top_or_a_weak_top_card_is_unclaimed_with_no_team():
    for r in (row(0.1, 0.1, 0.1), row(0.4, 0.3, 0.2), row(0.49, 0.0, 0.0)):
        d = RT.decide(np.stack([r]), STATES, tau=0.5)[0]
        assert d.status == "unclaimed" and d.card is None and d.team == ()
    d = RT.decide(np.stack([row(0.5, 0.0, 0.0)]), STATES, tau=0.5)[0]                         # exactly tau counts as claimed
    assert d.status == "solid"
    assert RT.decide(np.stack([row(0.4, 0.3, 0.2)]), STATES, tau=0.3)[0].status == "solid"   # a lower tau claims what a higher one leaves


def test_unclaimed_still_reports_the_best_card_probability():
    d = RT.decide(np.stack([row(0.3, 0.2, 0.1)]), STATES, tau=0.5)[0]
    assert d.p == pytest.approx(0.3)


def test_team_grows_until_it_holds_the_mass_and_never_passes_the_cap():
    P = np.stack([row(0.60, 0.30, 0.05)])
    assert RT.decide(P, STATES, tau=0.5, mass=0.5)[0].team == (0,)                            # 0.60 / 0.95 = 0.63 already holds half
    assert RT.decide(P, STATES, tau=0.5, mass=0.9)[0].team == (0, 1)                          # 0.90 / 0.95 = 0.947
    assert RT.decide(P, STATES, tau=0.5, mass=0.999)[0].team == (0, 1, 2)
    assert RT.decide(P, STATES, tau=0.5, mass=0.999, cap=2)[0].team == (0, 1)
    P2 = np.stack([row(0.3, 0.6, 0.05)])
    assert RT.decide(P2, STATES, tau=0.5, mass=0.5)[0].team == (1,)                           # ordered by probability, not by index


def test_rows_are_independent_and_shapes_are_checked():
    P = np.stack([row(0.9, 0.0, 0.0), row(0.0, 0.0, 0.0), row(0.0, 0.8, 0.1)])
    out = RT.decide(P, STATES, tau=0.5)
    assert [d.status for d in out] == ["solid", "unclaimed", "thin"]
    with pytest.raises(ValueError):
        RT.decide(P[:, :3], STATES, tau=0.5)                                                  # one column per card plus "none"
    with pytest.raises(ValueError):
        RT.decide(P, ["solid", "grey", "solid"], tau=0.5)
    with pytest.raises(ValueError):
        RT.decide(P, STATES, tau=0.0)
    with pytest.raises(ValueError):
        RT.decide(P, STATES, tau=0.5, mass=1.5)
    with pytest.raises(ValueError):
        RT.decide(P * 2.0, STATES, tau=0.5)                                                   # rows must be probabilities
    assert RT.decide(np.zeros((0, 4)), STATES, tau=0.5) == []


def test_claim_scores_rank_inputs_by_how_unclaimed_they_are():
    P = np.stack([row(0.95, 0.0, 0.0), row(0.3, 0.1, 0.0), row(0.0, 0.0, 0.0)])
    s = RT.unclaimed_score(P)
    assert s[0] < s[1] < s[2]                                                                  # 1 minus the best card probability

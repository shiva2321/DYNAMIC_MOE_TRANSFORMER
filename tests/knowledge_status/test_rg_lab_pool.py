"""Scoring of the pool lab (routing_general/lab_pool.py): pure functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general"):
    sys.path.insert(0, str(p))

import lab_pool as LP  # noqa: E402
import pool as PL  # noqa: E402

A = PL.Answer


def test_known_messages_are_scored_by_what_was_answered_what_was_right_and_who_owned_it():
    answers = [A(5, 0, "answered"), A(6, 1, "answered"), A(None, None, "abstained"), A(None, 1, "unreadable"), A(7, 0, "answered")]
    owner, value, solid = [0, 0, 1, 1, 0], [5, 9, 3, 4, 7], [True, True, False, False, True]
    s = LP.score_known(answers, owner, value, solid)
    assert s["answered"] == pytest.approx(0.6) and s["correct"] == pytest.approx(0.4) and s["wrong"] == pytest.approx(0.2)
    assert s["abstained"] == pytest.approx(0.2) and s["unreadable"] == pytest.approx(0.2)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["routed_right"] == pytest.approx(3 / 4)                                  # routed: 0, 1, 1 (unreadable), 0 against owners 0, 0, 1, 0
    assert s["correct_solid"] == pytest.approx(2 / 3) and s["correct_thin"] == 0.0 and s["answered_thin"] == 0.0
    none = LP.score_known([A(None, None, "abstained")] * 2, [0, 1], [1, 2], [True, True])
    assert none["answered"] == 0.0 and np.isnan(none["precision"]) and np.isnan(none["routed_right"]) and np.isnan(none["correct_thin"])


def test_unowned_messages_count_only_the_share_that_got_an_answer():
    assert LP.answered_share([A(1, 0, "answered"), A(None, None, "abstained"), A(None, 0, "unreadable"), A(2, 1, "answered")]) == pytest.approx(0.5)
    assert np.isnan(LP.answered_share([]))


def test_net_is_right_minus_wrong_minus_the_mean_unowned_answer_share():
    known = {"correct": 0.5, "wrong": 0.3}
    assert LP.net(known, {"a": 0.2, "b": 0.0, "c": float("nan")}) == pytest.approx(0.5 - 0.3 - 0.1)
    assert LP.net(known, {}) == pytest.approx(0.2)


def test_the_oracle_gate_routes_to_the_owner_and_refuses_the_rest():
    g = LP.OracleGate({"x": 2, "y": None, "z": 0})
    route, abstain = g.decide(["x", "y", "z"])
    assert route.tolist()[0] == 2 and route.tolist()[2] == 0 and abstain.tolist() == [False, True, False]


def test_the_parser_reads_exactly_two_numbers():
    assert LP.parse_pair("17 plus 5") == (17, 5) and LP.parse_pair("just words") is None and LP.parse_pair("1 2 3") is None


def row(ask, vmf):
    g = lambda n: {"known": {"correct": 0.0, "wrong": 0.0}, "unowned": {}, "net": n}
    return {"gates": {"ask": g(ask), "vmf": g(vmf), "oracle": g(0), "mix": g(0)}}


def test_the_verdict_needs_the_declared_margin():
    assert LP.verdict([row(0.10, -0.10)])["C7_pass"] and not LP.verdict([row(0.10, -0.04)])["C7_pass"]
    assert LP.verdict([row(0.0, 0.0), row(0.4, 0.0)])["difference"] == pytest.approx(0.2)

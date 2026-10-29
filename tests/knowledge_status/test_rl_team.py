"""Teams of experts (routing_lab/team.py): who is in the team, and what the team settles on."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_lab"))

import ask_reader as AR  # noqa: E402
import team as TM  # noqa: E402

KNOWN = ["add", "sub", "max", "none"]


def prow(**kw):
    p = np.zeros(len(AR.OPS))
    for k, v in kw.items():
        p[AR.OPS.index(k)] = v
    rest = [i for i in range(len(AR.OPS)) if AR.OPS[i] not in kw]
    p[rest] = (1.0 - p.sum()) / len(rest)
    return p


def test_policies_order_by_probability_and_respect_who_is_usable():
    p = prow(add=0.5, sub=0.3, max=0.1)
    cov = np.array([True, True, False, False])
    assert TM.build_team("top1", p, cov, KNOWN, ["add", "sub", "max"]) == ["add"]
    assert TM.build_team("top2", p, cov, KNOWN, ["add", "sub", "max"]) == ["add", "sub"]
    assert TM.build_team("top3", p, cov, KNOWN, ["add", "sub", "max"]) == ["add", "sub", "max"]
    assert TM.build_team("all", p, cov, KNOWN, ["add", "sub", "max"]) == ["add", "sub", "max"]
    assert TM.build_team("set", p, cov, KNOWN, ["add", "sub", "max"]) == ["add", "sub"]
    assert TM.build_team("top2", p, cov, KNOWN, ["sub", "max"]) == ["sub", "max"]               # an expert that is not usable is never in the team
    assert TM.build_team("set", p, np.array([False] * 4), KNOWN, ["add", "sub", "max"]) == []    # nothing covers it: novel, no expert
    assert TM.build_team("set", p, np.array([True, True, True, True]), KNOWN, ["add", "sub", "max"], cap=2) == ["add", "sub"]
    with pytest.raises(ValueError):
        TM.build_team("nope", p, cov, KNOWN, ["add"])


def test_the_none_class_is_never_an_expert():
    p = prow(none=0.9)
    assert "none" not in TM.build_team("all", p, np.array([False, False, False, True]), KNOWN, ["add", "sub", "max", "none"])


def test_a_stated_answer_picks_the_member_that_reproduces_it():
    p = prow(add=0.6, sub=0.3)
    assert TM.choose(["add", "sub"], 17, 5, 12, p, ["add", "sub"]) == "sub"                      # 17 - 5 = 12; the most probable member (add) does not reproduce it
    assert TM.choose(["add", "sub"], 17, 5, None, p, ["add", "sub"]) == "add"                    # no answer: the most probable member
    assert TM.choose(["add", "sub"], 17, 5, 99, p, ["add", "sub"]) == "add"                      # nobody reproduces it: the most probable member
    assert TM.choose(["add", "sub"], 17, 5, 12, p, ["add"]) == "add"                             # only members that may verify are checked
    assert TM.choose([], 1, 2, 3, p, ["add"]) is None


def test_two_members_that_both_reproduce_the_answer_go_to_the_more_probable_one():
    p = prow(add=0.2, max=0.6)
    assert TM.choose(["add", "max"], 5, 0, 5, p, ["add", "max"]) == "max"                        # add(5, 0) == max(5, 0) == 5: a coincidence, the more probable member wins


def test_a_zero_divisor_is_never_computed_for_div_or_mod():
    p = prow(div=0.5, add=0.4)
    assert TM.choose(["div", "add"], 7, 0, 7, p, ["div", "add"]) == "add"


def test_evaluate_counts_recall_size_and_right():
    p = prow(add=0.5, sub=0.4)
    rows = [{"p_row": p, "covered_row": np.array([True, True, False, False]), "a": 17, "b": 5, "answer": 12, "skill": "sub"}]
    top1 = TM.evaluate("top1", rows, KNOWN, ["add", "sub", "max"], ["add", "sub", "max"], True)
    top2 = TM.evaluate("top2", rows, KNOWN, ["add", "sub", "max"], ["add", "sub", "max"], True)
    assert top1 == {"recall": 0.0, "size": 1.0, "right": 0.0} and top2 == {"recall": 1.0, "size": 2.0, "right": 1.0}
    assert TM.evaluate("top2", rows, KNOWN, ["add", "sub", "max"], ["add", "sub", "max"], False)["right"] == 0.0


def test_the_mass_policy_adds_experts_until_the_probability_is_held_and_respects_the_cap():
    sure = prow(add=0.95, sub=0.03, max=0.01)
    torn = prow(add=0.45, sub=0.40, max=0.10)
    cov = np.array([True, True, True, False])
    allowed = ["add", "sub", "max"]
    assert TM.build_team("mass0.9", sure, cov, KNOWN, allowed) == ["add"]
    assert TM.build_team("mass0.9", torn, cov, KNOWN, allowed) == ["add", "sub", "max"][:3] or True
    assert TM.build_team("mass0.9", torn, cov, KNOWN, allowed) == ["add", "sub", "max"]            # 0.45 + 0.40 holds 0.85/0.95 of the taught mass: not enough, the third joins
    assert TM.build_team("mass0.9", torn, cov, KNOWN, allowed, cap=2) == ["add", "sub"]
    assert TM.build_team("mass0.99", sure, cov, KNOWN, allowed) == ["add", "sub", "max"]

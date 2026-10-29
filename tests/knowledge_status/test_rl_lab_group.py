"""The Lab B runner helpers (routing_lab/lab_group.py): windows and applying the changes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_lab"))

import lab_group as LG  # noqa: E402
import evaluate as EV  # noqa: E402
import status as ST  # noqa: E402


def case(skill):
    return EV.Case("solid", "S4", skill, "arith")


def test_mixed_windows_cover_every_flagged_part_once_in_chunks_of_g():
    cases = [case("add")] * 10
    w = LG.windows_for(list(range(10)), cases, "mixed", 4, np.random.default_rng(0))
    assert [len(x) for x in w] == [4, 4, 2] and sorted(sum(w, [])) == list(range(10))
    assert LG.windows_for(list(range(10)), cases, "mixed", 4, np.random.default_rng(0)) == w            # deterministic in the generator


def test_same_skill_windows_keep_a_skill_together():
    cases = [case("add"), case("mul"), case("add"), case("mul"), case("add"), case("mul")]
    w = LG.windows_for(list(range(6)), cases, "same_skill", 3, np.random.default_rng(1))
    assert [sorted(cases[i].skill for i in x) for x in w] == [["add"] * 3, ["mul"] * 3]
    with pytest.raises(ValueError):
        LG.windows_for([0], cases, "other", 1, np.random.default_rng(0))


def test_apply_turns_a_changed_part_solid_with_the_skill_and_leaves_the_rest():
    a = ST.Assessment("unseen", None, 0.01, 0, float("nan"), 1, "sub")
    b = ST.Assessment("thin", "sub", 0.5, 20, 0.3, 1, "sub")
    out = LG.apply([a, b], {0: "add"}, {"add": 0.93})
    assert (out[0].status, out[0].skill, out[0].competence, out[0].nearest) == ("solid", "add", 0.93, "sub") and out[1] is b

import numpy as np
import pytest

from fsmoe.domains import code as CO
from fsmoe.domains.base import rng_for
from fsmoe.e2e import Clusters, ToolChannel

SKILLS = [f"code:{s}" for s in CO.SKILLS]
TAUGHT = [s for s in CO.SKILLS if s not in CO.WITHHELD]


def test_the_tool_labels_known_families_in_every_style_and_opens_one_class_per_new_behaviour():
    tool = ToolChannel(SKILLS, TAUGHT)
    rng = rng_for(3, "e2e")
    new_sigs = {}
    for skill in CO.SKILLS:
        for style in CO.TEACH + CO.HELDOUT:
            status, payload, ran = tool.verify(CO.render(skill, style, rng)[0])
            assert ran
            if skill in TAUGHT:
                assert status == "known" and payload == SKILLS.index(f"code:{skill}"), (skill, style)
            else:
                assert status == "new"
                new_sigs.setdefault(skill, set()).add(payload)
    assert all(len(v) == 1 for v in new_sigs.values()) and len({next(iter(v)) for v in new_sigs.values()}) == len(CO.WITHHELD)
    assert tool.calls == len(CO.SKILLS) * 9


def test_the_tool_never_runs_a_text_that_is_not_code():
    tool = ToolChannel(SKILLS, TAUGHT)
    for text in ["I lost my card", "17 + 5", "Tom scored 3 points and Ana scored 9. What is the highest score?"]:
        assert tool.verify(text) == ("none", None, False)
    assert tool.ran == 0 and tool.calls == 3


def test_clusters_give_one_id_per_behaviour_cap_their_slots_and_map_to_the_majority_truth():
    cl = Clusters(100, slots=2)
    a = cl.assign(("list", ("1",)), 5)
    b = cl.assign(("list", ("2",)), 6)
    assert (a, b) == (100, 101) and cl.assign(("list", ("1",)), 5) == 100 and cl.assign(("list", ("1",)), 7) == 100
    assert cl.assign(("list", ("3",)), 8) is None                                      # no slot left
    assert cl.mapping() == {100: 5, 101: 6} and abs(cl.purity() - 3 / 4) < 1e-9
    assert Clusters(10).purity() != Clusters(10).purity()                              # nan when empty


def test_the_tool_holds_one_reference_behaviour_per_taught_family_and_none_for_withheld_ones():
    tool = ToolChannel(SKILLS, TAUGHT)
    assert len(tool.refs) == len(TAUGHT) == 11 and set(tool.refs.values()) == {SKILLS.index(f"code:{s}") for s in TAUGHT}

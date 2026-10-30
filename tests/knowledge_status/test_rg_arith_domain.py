"""The arithmetic domain world of the general lab (routing_general/arith_domain.py): the sets are matched in wording."""

import sys
from pathlib import Path

import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general"):
    sys.path.insert(0, str(p))

import arith_domain as AD  # noqa: E402
import run_e2 as R  # noqa: E402


@pytest.fixture(scope="module")
def built():
    world, foreign, extras = AD.build(150)
    truth = R.build_world(150, "dev", R.SIZES)
    return world, foreign, extras, truth


def cases_of(truth):
    return {t: c for t, c in zip(truth.test_texts, truth.test_cases)}


def test_known_and_never_taught_are_both_in_the_new_wordings(built):
    world, foreign, extras, truth = built
    by_text = cases_of(truth)
    assert world.known_test and world.unknown_test
    assert all(by_text[t].family in AD.HELD and by_text[t].skill == truth.taught[i] for t, i in world.known_test)
    assert all(by_text[t].family in AD.HELD and by_text[t].skill not in truth.taught for t in world.unknown_test)
    assert world.unknown_labels and all(s not in truth.taught for s in world.unknown_labels)


def test_never_taught_messages_in_the_teaching_wordings_are_kept_apart_not_dropped(built):
    world, foreign, extras, truth = built
    by_text = cases_of(truth)
    early = extras["unknown_teaching_wording"]
    assert early and all(by_text[t].family not in AD.HELD and by_text[t].skill not in truth.taught for t in early)
    assert not set(early) & set(world.unknown_test)
    n_untaught = sum(1 for c in truth.test_cases if c.kind == "arith" and c.status == "unseen")
    assert len(early) + len(world.unknown_test) == n_untaught


def test_calibration_is_taught_skills_in_taught_wordings_and_no_set_shares_a_message(built):
    world, foreign, extras, truth = built
    cal = {t for t, _ in world.known_cal}
    full = len(truth.taught) * R.SIZES.cal
    assert 0.85 * full <= len(world.known_cal) <= full                                                  # a few calibration requests repeat a teaching request in the test bed; they are dropped
    teach = {t for msgs in world.teach.values() for t in msgs}
    sets = {"cal": cal, "known": {t for t, _ in world.known_test}, "unknown": set(world.unknown_test), "foreign": set(foreign), "trap": set(extras["trap"])}
    names = list(sets)
    for i, a in enumerate(names):
        assert not sets[a] & teach, f"{a} shares a message with the teaching"
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], f"{a} and {b} share a message"
    assert world.states == [("solid" if truth.assignment[s] == "solid" else "thin") for s in truth.taught]
    assert [c.name for c in world.cards] == world.taught

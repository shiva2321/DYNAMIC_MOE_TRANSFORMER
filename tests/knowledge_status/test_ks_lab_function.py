"""The function-check lab: when a part is called unseen by its look, does the answer that comes with it overturn that call? Tiny worlds and exact solvers; no language model."""

import math

import numpy as np
import pytest

import assignments
import generator as G
import lab_function as LF
import run_e2 as R
import skills
import status as ST
from test_ks_function_check import ExactSolver

SOLVERS = {s: ExactSolver(s) for s in skills.SKILLS}
NAN = float("nan")
TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05,))


@pytest.fixture
def tiny_thresholds(monkeypatch):
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        monkeypatch.setitem(assignments.THRESHOLDS, k, v)


def fake_competence(pool, skill, count, seed):
    return {"accuracy": 0.97 if count >= 60 else 0.5, "chance": 0.1}


def A(status, skill=None, nearest="", evidence=0, competence=NAN):
    return ST.Assessment(status, skill, 0.5, evidence, competence, 1, nearest)


ACC = {"add": 0.97, "sub": 0.95, "max": 0.99}


def test_an_unseen_call_is_overturned_when_a_competent_skill_explains_the_example():
    out, which = LF.apply_function_check([A("unseen", None, "add")], ["I had 17 apples and got 5 more."], [22], SOLVERS, ["add", "sub", "max"], ACC)
    assert which == ["add"] and out[0].status == "solid" and out[0].skill == "add" and out[0].competence == 0.97 and out[0].nearest == "add"


def test_nothing_else_changes():
    texts = ["17 times 5", "17 plus 5", "17 plus 5", "17 plus 5"]
    given = [A("unseen", None, "add"), A("solid", "add", "add", 5, 0.97), A("thin", "mod", "mod", 3, 0.4), A("other", None, "other")]
    out, which = LF.apply_function_check(given, texts, [85, 22, 22, 22], SOLVERS, ["add", "sub", "max"], ACC)
    assert out == given and which == [None, None, None, None]                      # not explained, and the covered ones are never touched


def test_a_call_without_an_answer_is_left_alone():
    out, which = LF.apply_function_check([A("unseen", None, "add")], ["17 plus 5"], [None], SOLVERS, ["add"], ACC)
    assert out[0].status == "unseen" and which == [None]


def test_when_several_skills_explain_the_nearest_one_wins():
    out, which = LF.apply_function_check([A("unseen", None, "mod"), A("unseen", None, "add")], ["5 and 5", "5 and 5"], [0, 0], SOLVERS, ["sub", "mod", "add"], {"sub": 0.9, "mod": 0.9, "add": 0.9})
    assert which == ["mod", "sub"]                                                    # 0 is explained by sub and mod: the nearest one when it explains (mod), otherwise the first (add is nearest but cannot)


def test_the_function_check_can_only_remove_unseen_calls():
    rng = np.random.default_rng(0)
    given, texts, answers = [], [], []
    for _ in range(200):
        skill = skills.SKILLS[int(rng.integers(0, 6))]
        a, b = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        texts.append(f"{a} and {b}")
        answers.append(skills.truth(skill, a, b))
        given.append(A(["unseen", "solid", "thin", "other"][int(rng.integers(0, 4))], None, skills.SKILLS[int(rng.integers(0, 6))]))
    out, _ = LF.apply_function_check(given, texts, answers, SOLVERS, ["add", "sub", "max"], ACC)
    assert all(not (y.status == "unseen" and x.status != "unseen") for x, y in zip(given, out))
    assert all(x == y for x, y in zip(given, out) if x.status != "unseen")


def test_lengths_must_agree():
    with pytest.raises(ValueError, match="same length"):
        LF.apply_function_check([A("unseen")], ["a", "b"], [1, 2], SOLVERS, ["add"], ACC)


def test_the_answers_of_the_test_parts_are_regenerated_exactly(tiny_thresholds):
    world = R.build_world(100, "dev", TINY, fake_competence)
    ex = LF.test_examples(world)
    arith = [(t, c) for t, c in zip(world.test_texts, world.test_cases) if c.kind == "arith"]
    assert arith and all(t in ex for t, _ in arith)
    for t, c in arith:
        e = ex[t]
        assert e.skill == c.skill and e.family == c.family and skills.truth(e.skill, e.a, e.b) == e.answer
        assert sorted(__import__("function_check").parse_numbers(t)) == sorted([e.a, e.b]) or c.family == "S2"


def test_competence_is_measured_on_problems_the_solver_never_saw(tiny_thresholds):
    world = R.build_world(100, "dev", TINY, fake_competence)
    solvers, acc, competent = LF.train_solvers(world, epochs=40)
    assert set(solvers) == set(world.taught) == set(acc)
    assert all(0.0 <= v <= 1.0 for v in acc.values())
    assert set(competent) <= set(world.taught)
    for s in competent:
        assert world.assignment[s] == "solid" and acc[s] >= assignments.THRESHOLDS["solid_min_acc"] - 1e-9          # competent = a Solid state AND measured accuracy at the bar

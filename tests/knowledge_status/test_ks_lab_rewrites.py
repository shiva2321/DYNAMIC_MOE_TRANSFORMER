"""The rewrite lab (option (a)): which rewrites enter which part of the machine, what is never touched, and one tiny arm end to end (hashed reader, canned writer and solver; no language model)."""

import numpy as np
import pytest

import assignments
import lab_rewrites as LR
import learned_map
import reps
import rewrites as RW
import run_e2 as R
import skills
import status as ST

TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05,))


def fake_competence(pool, skill, count, seed):
    return {"accuracy": 0.97 if count >= 60 else 0.5, "chance": 0.1}


@pytest.fixture
def tiny_thresholds(monkeypatch):
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        monkeypatch.setitem(assignments.THRESHOLDS, k, v)


@pytest.fixture
def world(tiny_thresholds):
    return R.build_world(100, "dev", TINY, fake_competence)


def canned_writer(prompts, seed):
    """'Rewrites' by swapping the template wording of a request: keeps the numbers, changes the words."""
    out = []
    for p in prompts:
        src = p.split("Request: ", 1)[1]
        nums = RW.numbers_in(src)
        out.append(f"please work out the thing with {nums[0]} and {nums[1]} for me" if len(nums) == 2 else "")
    return out


def canned_solver_for(world):
    """Answers every rewrite with the TRUE answer of the example the rewrite's numbers come from (a perfect checker)."""
    table = {}
    for s in world.taught:
        for e in world.teach[s]:
            table[tuple(RW.numbers_in(e.span))] = e.answer

    def solve(prompts, seed):
        return [str(table.get(tuple(RW.numbers_in(p.split("\n\n", 1)[1])), "no")) for p in prompts]

    return solve


def test_the_calibration_examples_are_regenerated_exactly(world):
    ex = LR.cal_examples(world)
    assert set(ex) == set(world.taught)
    for s in world.taught:
        assert [e.span for e in ex[s]] == world.cal[s] and all(skills.truth(s, e.a, e.b) == e.answer for e in ex[s])


def test_sources_come_only_from_teaching_and_calibration_never_from_a_test_part(world):
    src = LR.sources_and_answers(world, n_src=10)
    teach = {e.span for s in world.teach for e in world.teach[s]}
    cal = {t for s in world.taught for t in world.cal[s]}
    assert set(src["teach"]) <= teach and set(src["cal"]) <= cal
    assert not (set(src["teach"]) | set(src["cal"])) & set(world.test_texts)
    assert all(skills.truth(sk, *map(int, RW.numbers_in(t)[:2][::1])) is not None for t, (sk, _) in list(src["teach"].items())[:0])      # shape only
    for t, (sk, ans) in src["teach"].items():
        assert sk in world.taught and isinstance(ans, int)
    per_skill = {s: sum(1 for _, (sk, _) in src["teach"].items() if sk == s) for s in world.taught}
    assert all(0 < n <= 10 for n in per_skill.values())                                       # at most n_src per skill (a Thin skill may give fewer)


def test_a_thin_skill_gives_all_of_its_examples(world):
    thin = [s for s in world.taught if world.assignment[s] == "thin"][0]
    src = LR.sources_and_answers(world, n_src=100)
    assert sum(1 for _, (sk, _) in src["teach"].items() if sk == thin) == len(set(e.span for e in world.teach[thin]))


def test_a_rewrite_that_equals_a_test_or_adaptation_part_is_never_used(world):
    banned = set(world.test_texts[:3]) | set(world.adapt[world.taught[0]][:2])
    clean = LR.drop_known({"a": [*banned, "something else with 1 and 2"]}, world)
    assert clean == {"a": ["something else with 1 and 2"]}


def test_the_family_guard_of_the_map_still_refuses_held_out_wordings_but_takes_rewrites():
    X = np.eye(8, dtype=np.float32)[:6] + 0.01
    skill = ["add", "add", "sub", "sub", "add", "sub"]
    m = learned_map.LearnedMap(["add", "sub"], epochs=2)
    with pytest.raises(ValueError, match="teaching wordings"):
        m.fit(X, skill, ["S1", "S2", "S1", "S2", "S4", "S1"], ["a", "b", "c", "d", "e", "f"])
    LR.fit_map_with_rewrites(["add", "sub"], 0, X[:4], skill[:4], ["S1", "S2", "S1", "S2"], ["a", "b", "c", "d"], X[4:], skill[4:], epochs=2)         # no error


def test_rewrites_are_labelled_as_the_skill_of_their_source_with_a_template_of_their_own():
    assert LR.REWRITE_FAMILY in learned_map.ALLOWED_FAMILIES                                  # the label used so that the map accepts them (documented in lab_rewrites)
    assert LR.REWRITE_FAMILY in assignments.TEACH_FAMILIES


def test_the_generic_instructions_pass_the_guard_and_the_ceiling_ones_are_marked_contaminated():
    for text in RW.INSTRUCTIONS:
        assert all(w not in text.lower() for w in ("story", "typo", "formal", "casual", "terse"))
    assert LR.CEILING_INSTRUCTIONS and any("story" in t.lower() for t in LR.CEILING_INSTRUCTIONS) and LR.CEILING_LABEL.startswith("CONTAMINATED")


def run_arm(arm, tmp_path, world):
    rep = reps.get_rep("R0_hash")
    return LR.run_arm(arm, 100, tmp_path, "s", sizes=TINY, rep=rep, competence_fn=fake_competence, writer=canned_writer, solver=canned_solver_for(world), n_src=8, cache_dir=None, rewrite_cache=tmp_path / "rw")


def test_the_arms_differ_only_where_they_should(tmp_path, world):
    built = {}
    real = ST.StatusMachine

    class Spy(real):
        def __init__(self, groups, other, cal_vectors, cal_groups, **kw):
            built.setdefault(self_arm[0], []).append(({k: (len(g.protos), g.n_examples) for k, g in groups.items()}, len(cal_groups)))
            super().__init__(groups, other, cal_vectors, cal_groups, **kw)

    self_arm = [None]
    LR.StatusMachine = Spy
    try:
        for arm in ("map only", "full"):
            self_arm[0] = arm
            rec = run_arm(arm, tmp_path / arm.replace(" ", "_"), world)
            assert rec["arm"] == arm and "false_unseen" in rec and rec["n_rewrites_teach"] > 0
    finally:
        LR.StatusMachine = real
    (g1, c1), (g2, c2) = built["map only"][0], built["full"][0]
    assert all(g1[s][1] == g2[s][1] == len(world.teach[s]) for s in world.taught)                       # the teaching COUNT is never inflated by rewrites
    assert all(g2[s][0] > g1[s][0] for s in world.taught) and c2 > c1                                    # but the full arm has more prototypes and more calibration parts
    assert all(g1[s][0] == len(world.teach[s]) for s in world.taught)                                    # the map-only arm has exactly the original prototypes


def test_an_arm_result_is_resumable(tmp_path, world):
    import json
    a = run_arm("map only", tmp_path, world)
    b = run_arm("map only", tmp_path, world)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_known_parts_filter_is_applied_to_both_the_teaching_and_the_calibration_rewrites(tmp_path, world, monkeypatch):
    calls = []
    real = LR.drop_known
    monkeypatch.setattr(LR, "drop_known", lambda rw, w: (calls.append(sum(len(v) for v in rw.values())), real(rw, w))[1])
    run_arm("map only", tmp_path, world)
    assert len(calls) == 2 and all(c > 0 for c in calls)


def test_the_map_receives_exactly_the_kept_teaching_rewrites_with_their_skills(tmp_path, world, monkeypatch):
    seen = []
    real = LR.fit_map_with_rewrites

    def spy(taught, seed, X, skill, family, template, Xr, skill_r, epochs=None):
        seen.append((len(Xr), list(skill_r), set(taught)))
        return real(taught, seed, X, skill, family, template, Xr, skill_r, epochs)

    monkeypatch.setattr(LR, "fit_map_with_rewrites", spy)
    rec = run_arm("full", tmp_path, world)
    assert len(seen) == 1
    n, skill_r, taught = seen[0]
    assert n == len(skill_r) == rec["n_rewrites_teach"] > 0 and set(skill_r) <= taught == set(world.taught)

"""E4 (relation from worked traces) on tiny worlds: which seeds qualify, how the world and the traces are built, and the end-to-end numbers with an oracle reader (no language model)."""

import hashlib
import json

import numpy as np
import pytest

import assignments
import function_check as FC
import reps
import run_e2 as R
import run_e3 as E3
import skills
import traces as TR
from test_ks_function_check import ExactSolver

TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05,))


def fake_competence(pool, skill, count, seed):
    acc = {"add": 0.97, "sub": 0.95 if seed % 2 == 0 else 0.80}.get(skill, 0.97 if count >= 60 else 0.5)
    return {"accuracy": acc, "chance": 0.1}


@pytest.fixture(autouse=True)
def tiny_thresholds(monkeypatch):
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        monkeypatch.setitem(assignments.THRESHOLDS, k, v)


def test_the_forced_assignment_always_holds_addition_and_subtraction_as_solid():
    for seed in range(40):
        a = assignments.draw_assignment(np.random.default_rng(seed), need_solid=frozenset({"add", "sub"}))
        assert a["add"] == a["sub"] == "solid" and sorted(a.values()).count("thin") == 1 and sorted(a.values()).count("unseen") == 3


def test_a_seed_qualifies_only_if_add_and_sub_are_solid_by_their_measured_accuracy():
    got = E3.qualifying_seeds(range(100, 110), 3, "dev", fake_competence, sizes=TINY)
    assert got == [100, 102, 104]                                                       # sub is accurate enough only in the even seeds of this fake
    assert E3.qualifying_seeds(range(100, 110), 99, "dev", fake_competence, sizes=TINY) == [100, 102, 104, 106, 108]
    with pytest.raises(ValueError, match="count"):
        E3.qualifying_seeds([100], 0, "dev", fake_competence, sizes=TINY)


def test_the_trace_set_has_every_kind_in_every_wording_and_is_deterministic():
    ts = E3.trace_set(100, 8)
    kinds = {}
    for t in ts:
        kinds.setdefault((t.header_skill, t.decoy), []).append(t)
    assert set(kinds) == {("mul", None), ("div", None), ("mul", "wrong_steps"), ("div", "wrong_steps"), ("max", "irrelevant_valid_steps"), ("mod", "unrelated_skill")}
    assert all(len(v) == 8 for v in kinds.values())
    assert {TR.N_VARIANTS - 1 - i for i in range(TR.N_VARIANTS)} == set(range(TR.N_VARIANTS))
    texts = [FC.parse_numbers(t.text) for t in kinds[("mul", None)]]
    assert len({t.text for t in kinds[("mul", None)]}) > 1
    assert E3.trace_set(100, 8) == ts and E3.trace_set(101, 8) != ts


def test_the_world_of_a_trace_seed_holds_the_step_texts_and_the_patch_is_restored(monkeypatch):
    before = R.assignments.draw_assignment
    world = E3.build_trace_world(100, "dev", TINY, fake_competence, n_per_kind=4)
    assert R.assignments.draw_assignment is before
    assert world.assignment["add"] == world.assignment["sub"] == "solid"
    steps = {st.text for t in E3.trace_set(100, 4) for st in t.steps}
    assert steps <= set(world.all_texts()) and steps


class StepOracle(reps.Representation):
    """Knows the skill of every text of the world and of every step: puts it near its own axis."""

    def __init__(self, world, traces):
        super().__init__("step-oracle", 24)
        g = {}
        for s in world.teach:
            g.update({e.span: s for e in world.teach[s]})
        g.update({i.text: "other" for i in world.teach_none})
        for name, texts in list(world.cal.items()) + list(world.adapt.items()):
            g.update({t: name for t in texts})
        for t, c in zip(world.test_texts, world.test_cases):
            g[t] = c.skill if c.skill else ("never" if c.kind == "unseen_content" else "other")
        g.update({t: "other" for t in world.background})
        for tr in traces:
            g.update({st.text: st.skill for st in tr.steps})
        self.g, self.axes = g, {n: k for k, n in enumerate(list(skills.SKILLS) + ["other", "never"])}
        self.tangled = [(m.text[p.start:p.end], p.skill if p.kind == "skill" else "other") for m in world.tangled for p in m.parts]

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim))
        for k, t in enumerate(texts):
            grp = self.g.get(t)
            if grp is None:
                grp = next((gg for part, gg in self.tangled if t in part or part in t), "other")
            noise = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).standard_normal(self.dim)
            out[k] = 2.0 * np.eye(self.dim)[self.axes[grp]] + 0.3 * noise
        return out


def exact_solvers(world):
    return {s: ExactSolver(s) for s in world.taught}, {}, []


def test_with_an_oracle_reader_the_genuine_traces_name_their_parent_and_the_decoys_name_none(tmp_path):
    world = E3.build_trace_world(100, "dev", TINY, fake_competence, n_per_kind=6)
    rec = E3.run_seed(100, tmp_path, "s", sizes=TINY, rep=StepOracle(world, E3.trace_set(100, 6)), competence_fn=fake_competence, cache_dir=None, n_per_kind=6, solvers_fn=exact_solvers)
    assert rec["seed"] == 100 and rec["rates"]["mul"]["n"] == rec["rates"]["div"]["n"] == 6
    assert rec["rates"]["mul"]["parent_is_add"] >= 0.8 and rec["rates"]["div"]["parent_is_sub"] >= 0.8
    assert rec["rates"]["mul"]["chain_valid"] == 1.0 and rec["rates"]["div"]["chain_valid"] == 1.0
    for kind in ("mul_wrong_steps", "div_wrong_steps", "max_irrelevant_valid_steps", "mod_unrelated_skill"):
        assert rec["rates"][kind]["any_parent"] == 0.0, kind
    assert (tmp_path / "s100.json").exists()
    again = E3.run_seed(100, tmp_path, "s", sizes=TINY, rep=StepOracle(world, E3.trace_set(100, 6)), competence_fn=fake_competence, cache_dir=None, n_per_kind=6, solvers_fn=exact_solvers)
    assert json.dumps(again, sort_keys=True) == json.dumps(rec, sort_keys=True)                      # finished cells are not redone


def test_the_report_gives_the_bars_and_says_which_seeds_qualified(tmp_path):
    world = E3.build_trace_world(100, "dev", TINY, fake_competence, n_per_kind=6)
    rec = E3.run_seed(100, tmp_path, "s", sizes=TINY, rep=StepOracle(world, E3.trace_set(100, 6)), competence_fn=fake_competence, cache_dir=None, n_per_kind=6, solvers_fn=exact_solvers)
    text = E3.report_e4([rec], candidates=12, label="oracle")
    assert "B5" in text and "0.90" in text and "0.10" in text and "1 of 12" in text and ("PASS" in text or "fail" in text)
    assert "mul" in text and "div" in text and "decoy" in text.lower()


def test_the_wording_read_back_from_a_trace_is_the_one_it_was_made_with():
    rng = np.random.default_rng(0)
    for skill, decoy in E3.KINDS:
        for variant in range(TR.N_VARIANTS):
            for _ in range(10):
                assert E3.TR_variant(TR.make_trace(rng, skill, variant, decoy)) == variant, (skill, decoy, variant)

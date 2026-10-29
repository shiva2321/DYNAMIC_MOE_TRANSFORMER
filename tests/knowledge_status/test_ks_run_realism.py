"""B7 plumbing on tiny worlds with an oracle reader (no language model): the cell, its balanced accuracies, the leak guard and the report."""

import hashlib
import json

import numpy as np
import pytest

import realism as RL
import reps
import run_e2 as R
import run_realism as RR
import skills
from test_ks_run_e3 import TINY, fake_competence  # noqa: F401  (the tiny sizes and a fake competence table; the autouse threshold fixture is imported with them)
from test_ks_run_e3 import tiny_thresholds  # noqa: F401


def rows_for(n=40):
    rows = RL.build_realism_set(n, seed=5)
    return rows


class Oracle(reps.Representation):
    """Puts every text of the world, and every realism problem, near the axis of its skill (or of Other)."""

    def __init__(self, world, rows):
        super().__init__("realism-oracle", 24)
        g = {}
        for s in world.teach:
            g.update({e.span: s for e in world.teach[s]})
        g.update({i.text: "other" for i in world.teach_none})
        for name, texts in list(world.cal.items()) + list(world.adapt.items()):
            g.update({t: name for t in texts})
        for t, c in zip(world.test_texts, world.test_cases):
            g[t] = c.skill if c.skill else ("never" if c.kind == "unseen_content" else "other")
        g.update({t: "other" for t in world.background})
        g.update({r["text"]: r["skill"] for r in rows})
        self.g, self.axes = g, {n: k for k, n in enumerate(list(skills.SKILLS) + ["other", "never"])}
        self.tangled = [(m.text[p.start:p.end], p.skill if p.kind == "skill" else "other") for m in world.tangled for p in m.parts]

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim))
        for k, t in enumerate(texts):
            grp = self.g.get(t) or next((gg for part, gg in self.tangled if t in part or part in t), "other")
            noise = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).standard_normal(self.dim)
            out[k] = 2.0 * np.eye(self.dim)[self.axes[grp]] + 0.3 * noise
        return out


def test_balanced_is_the_mean_of_the_two_kinds_and_none_when_one_is_missing():
    assert RR.balanced([False, False, True, False], [True, True, False, True]) == pytest.approx(0.5 * (0.75 + 0.75))
    assert RR.balanced([True, True], [False]) == 0.0 and RR.balanced([False], [True]) == 1.0
    assert RR.balanced([], [True]) is None and RR.balanced([False], []) is None


def test_with_an_oracle_reader_the_machine_separates_solid_from_unseen_on_the_realism_problems_as_on_the_templates(tmp_path):
    rows = rows_for()
    world = RR.build_realism_world(100, "dev", TINY, fake_competence, rows)
    rec = RR.run_seed(100, tmp_path, "s", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=rows)
    assert rec["seed"] == 100 and rec["pool"] == "dev" and (tmp_path / "s100.json").exists()
    assert rec["template_bal"] is not None and rec["real_bal"] is not None
    assert rec["real_bal"] >= 0.8 and rec["template_bal"] >= 0.8 and abs(rec["diff"]) <= 0.2
    assert rec["diff"] == pytest.approx(rec["real_bal"] - rec["template_bal"])
    assert set(rec["per_skill"]) <= {"add", "sub", "mul", "div"} and sum(d["n"] for d in rec["per_skill"].values()) == sum(rec["n_real"].values())
    again = RR.run_seed(100, tmp_path, "s", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=rows)
    assert json.dumps(again, sort_keys=True) == json.dumps(rec, sort_keys=True)                  # a finished cell is not redone
    other = RR.run_seed(100, tmp_path, "t", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=rows)
    assert other["code_stamp"] == "t"                                                         # a cell written by other code is redone


def test_realism_problems_are_added_to_the_texts_to_read_and_never_to_the_teaching():
    rows = rows_for(20)
    world = RR.build_realism_world(100, "dev", TINY, fake_competence, rows)
    assert {r["text"] for r in rows} <= set(world.all_texts())
    taught = {e.span for s in world.teach for e in world.teach[s]} | {i.text for i in world.teach_none}
    assert not ({r["text"] for r in rows} & taught)
    leak = [{**rows[0], "text": next(iter(taught))}] + rows[1:]
    with pytest.raises(AssertionError, match="realism"):
        RR.build_realism_world(100, "dev", TINY, fake_competence, leak)


def test_report_states_both_balanced_accuracies_the_gap_and_handles_unscorable_seeds():
    cells = [{"diff": 0.05, "template_bal": 0.80, "real_bal": 0.85, "real_false_unseen": 0.1, "real_unseen_found": 0.9},
             {"diff": -0.03, "template_bal": 0.70, "real_bal": 0.67, "real_false_unseen": 0.2, "real_unseen_found": 0.8},
             {"diff": None, "template_bal": None, "real_bal": None, "real_false_unseen": None, "real_unseen_found": None}]
    text = RR.report_b7(cells)
    assert "2 of 3 seeds" in text and "0.750" in text and "0.760" in text and "+0.010" in text
    assert "No seed could be scored" in RR.report_b7([{"diff": None}])


def test_the_template_set_is_exactly_the_held_out_wording_parts_of_solid_and_unseen_skills(tmp_path):
    import assignments
    rows = rows_for()
    world = RR.build_realism_world(100, "dev", TINY, fake_competence, rows)
    rec = RR.run_seed(100, tmp_path, "s", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=rows)
    want = {k: sum(1 for c in world.test_cases if c.kind == "arith" and c.family in assignments.TEST_FAMILIES and c.status == k and c.skill in ("add", "sub", "mul", "div")) for k in ("solid", "unseen")}
    assert rec["n_template"] == want and want["solid"] > 0 and want["unseen"] > 0
    assert any(c.kind != "arith" or c.family not in assignments.TEST_FAMILIES or c.status not in ("solid", "unseen") for c in world.test_cases)        # so the filter is not vacuous


def test_a_seed_without_a_scorable_realism_pair_has_no_difference(tmp_path):
    rows = rows_for(60)
    world = RR.build_realism_world(100, "dev", TINY, fake_competence, rows)
    solid_only = [r for r in rows if world.truth.get(r["skill"]) == "solid"]
    assert solid_only and len(solid_only) < len(rows)
    rec = RR.run_seed(100, tmp_path, "s", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=solid_only)
    assert rec["real_bal"] is None and rec["template_bal"] is not None and rec["diff"] is None and rec["real_unseen_found"] is None and rec["real_false_unseen"] is not None


def test_the_report_averages_the_realism_rates_over_the_seeds_that_have_them():
    cells = [{"diff": 0.1, "template_bal": 0.8, "real_bal": 0.9, "real_false_unseen": 0.2, "real_unseen_found": None},
             {"diff": -0.1, "template_bal": 0.8, "real_bal": 0.7, "real_false_unseen": None, "real_unseen_found": 0.6},
             {"diff": 0.0, "template_bal": 0.8, "real_bal": 0.8, "real_false_unseen": 0.4, "real_unseen_found": 0.8}]
    text = RR.report_b7(cells)
    assert "Solid skills called Unseen 0.300" in text and "Unseen skills found 0.700" in text
    none = [{"diff": 0.0, "template_bal": 0.8, "real_bal": 0.8, "real_false_unseen": None, "real_unseen_found": None}]
    assert "called Unseen nan" in RR.report_b7(none) and "found nan" in RR.report_b7(none)


def test_the_template_side_holds_only_the_four_operations_the_realism_set_has(tmp_path):
    import assignments
    rows = rows_for()
    world = RR.build_realism_world(100, "dev", TINY, fake_competence, rows)
    rec = RR.run_seed(100, tmp_path, "s", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=rows)
    six = {k: sum(1 for c in world.test_cases if c.kind == "arith" and c.family in assignments.TEST_FAMILIES and c.status == k) for k in ("solid", "unseen")}
    assert sum(rec["n_template"].values()) < sum(six.values())                                # mod and max parts are not counted on the template side
    assert RR.REALISM_SKILLS == ("add", "sub", "mul", "div")


def test_the_small_operand_row_keeps_only_problems_whose_operands_are_both_two_digits_at_most(tmp_path):
    rows = rows_for(120)
    world = RR.build_realism_world(100, "dev", TINY, fake_competence, rows)
    rec = RR.run_seed(100, tmp_path, "s", sizes=TINY, rep=Oracle(world, rows), competence_fn=fake_competence, cache_dir=None, rows=rows)
    eligible = [r for r in rows if world.truth.get(r["skill"]) in ("solid", "unseen")]
    small = [r for r in eligible if max(r["operands"]) <= 99]
    assert 0 < len(small) < len(eligible) and sum(rec["n_real_small"].values()) == len(small)
    assert rec["real_bal_small"] is not None and rec["diff_small"] == pytest.approx(rec["real_bal_small"] - rec["template_bal"])
    cells = [{"diff": 0.1, "template_bal": 0.8, "real_bal": 0.9, "real_false_unseen": 0.1, "real_unseen_found": 0.9, "real_bal_small": 0.7, "diff_small": -0.1},
             {"diff": 0.0, "template_bal": 0.8, "real_bal": 0.8, "real_false_unseen": 0.1, "real_unseen_found": 0.9, "real_bal_small": None, "diff_small": None}]
    text = RR.report_b7(cells)
    assert "operands are both at most 99" in text and "1 seeds can be scored" in text and "0.700 against 0.800" in text and "-0.100" in text
    assert "operands are both at most" not in RR.report_b7([{**cells[1]}])


def test_the_report_gives_the_realism_rates_when_every_seed_has_them():
    cells = [{"diff": 0.05, "template_bal": 0.8, "real_bal": 0.85, "real_false_unseen": 0.1, "real_unseen_found": 0.9},
             {"diff": -0.03, "template_bal": 0.7, "real_bal": 0.67, "real_false_unseen": 0.3, "real_unseen_found": 0.7}]
    text = RR.report_b7(cells)
    assert "Solid skills called Unseen 0.200" in text and "Unseen skills found 0.800" in text

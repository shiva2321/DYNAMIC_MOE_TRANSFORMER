"""The final run of the chosen wording design on the report seeds: the ordinary E2 runner with the chosen reader and machine swapped in (tiny sizes, the hashed reader, no language model)."""

import json

import numpy as np
import pytest

import assignments
import reps
import run_e2 as R
import run_e2w as RW
import status as ST
import wording as W

TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05, 0.10))


def fake_competence(pool, skill, count, seed):
    return {"accuracy": 0.97 if count >= 60 else 0.5, "chance": 0.1}


@pytest.fixture(autouse=True)
def small_thresholds(monkeypatch):
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        monkeypatch.setitem(assignments.THRESHOLDS, k, v)


def test_the_choice_file_is_read_and_checked(tmp_path):
    path = tmp_path / "choice.json"
    path.write_text(json.dumps({"reader": "clean", "machine_name": "lofo + margin 0.25", "machine": {"lofo": True, "margin_q": 0.25}}), encoding="utf-8")
    assert RW.load_choice(path) == {"reader": "clean", "machine_name": "lofo + margin 0.25", "machine": {"lofo": True, "margin_q": 0.25}}
    path.write_text(json.dumps({"reader": "nope", "machine_name": "as is", "machine": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="reader"):
        RW.load_choice(path)
    path.write_text(json.dumps({"reader": "plain", "machine_name": "x", "machine": {"colour": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="machine"):
        RW.load_choice(path)
    with pytest.raises(FileNotFoundError):
        RW.load_choice(tmp_path / "missing.json")


def test_the_stamp_depends_on_the_choice(tmp_path):
    path = tmp_path / "choice.json"
    path.write_text(json.dumps({"reader": "plain", "machine_name": "as is", "machine": {}}), encoding="utf-8")
    a = RW.stamp_for(path)
    path.write_text(json.dumps({"reader": "plain", "machine_name": "lofo", "machine": {"lofo": True}}), encoding="utf-8")
    assert RW.stamp_for(path) != a


def test_the_chosen_machine_is_used_for_every_machine_of_the_cell_and_the_runner_is_restored(tmp_path):
    built = []
    real = W.WordingMachine

    class Spy(real):
        def __init__(self, groups, other, cal_vectors, cal_groups, **kw):
            built.append((type(self).__name__, kw.get("lofo", False), kw.get("margin_q"), len(cal_groups), kw.get("cal_families") is not None and len(kw["cal_families"]) == len(cal_groups)))
            super().__init__(groups, other, cal_vectors, cal_groups, **kw)

    choice = {"reader": "plain", "machine_name": "lofo + margin 0.25", "machine": {"lofo": True, "margin_q": 0.25}}
    before = ST.StatusMachine
    rec = RW.run_cell(choice, 100, "dev", tmp_path, "s", TINY, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None, machine_cls=Spy)
    assert ST.StatusMachine is before and R.ST.StatusMachine is before                          # nothing stays patched
    assert built and all(name == "Spy" and lofo and mq == 0.25 and ok for name, lofo, mq, _, ok in built)
    assert rec["rep"] == RW.REP_ID and rec["machine"]["q0.05"]["standard"] and rec["machine"]["q0.05"]["adapted"]
    assert (tmp_path / f"{RW.REP_ID}__s100.json").exists()


def test_with_the_machine_as_it_is_the_cell_equals_the_ordinary_one(tmp_path):
    rep = reps.get_rep("R0_hash")
    plain = R.run_cell("R5_R0_hash", 100, "dev", tmp_path / "a", "s", TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    same = RW.run_cell({"reader": "plain", "machine_name": "as is", "machine": {}}, 100, "dev", tmp_path / "b", "s", TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    assert same["machine"] == plain["machine"] and same["baselines"] == plain["baselines"] and same["tangled"] == plain["tangled"]


def test_the_mitigated_machine_changes_the_numbers_it_should(tmp_path):
    rep = reps.get_rep("R0_hash")
    plain = R.run_cell("R5_R0_hash", 100, "dev", tmp_path / "a", "s", TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    gated = RW.run_cell({"reader": "plain", "machine_name": "margin 0.50", "machine": {"margin_q": 0.50}}, 100, "dev", tmp_path / "b", "s", TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    a, b = plain["machine"]["q0.05"]["standard"], gated["machine"]["q0.05"]["standard"]
    assert b["false_unseen_heldout"] <= a["false_unseen_heldout"] + 1e-12 and b["unseen_recall"] <= a["unseen_recall"] + 1e-12            # the gate only removes unseen calls
    assert gated["baselines"] == plain["baselines"]                                                                                      # the baselines do not depend on the machine

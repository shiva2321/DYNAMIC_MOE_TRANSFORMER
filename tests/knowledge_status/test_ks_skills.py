import json
import os

import numpy as np
import pytest

import runlib
import skills
from growmoe import arith


def test_truth_matches_growmoe_arith():
    rng = np.random.default_rng(0)
    for skill in skills.SKILLS:
        for _ in range(200):
            a, b = int(rng.integers(0, 100)), int(rng.integers(1, 100))     # b >= 1 so div and mod are defined
            op = np.array([skills.ARITH_OP_ID[skill]])
            want = int(arith.answers(op, np.array([a]), np.array([b]))[0])
            assert skills.truth(skill, a, b) == want, (skill, a, b)


def test_parent_graph_is_acyclic_and_rooted_at_add_and_sub():
    assert skills.PARENTS["add"] == frozenset() and skills.PARENTS["sub"] == frozenset()

    def ancestors(s, seen=()):
        assert s not in seen, f"cycle through {s}"
        out = set()
        for p in skills.PARENTS[s]:
            out |= {p} | ancestors(p, seen + (s,))
        return out

    for s in skills.SKILLS:
        anc = ancestors(s)
        assert anc <= {"add", "sub", "div"}                                   # mod reaches div, which reaches sub; nothing else
        assert s not in anc


def test_mul_parent_is_add_and_div_parent_is_sub():
    assert skills.PARENTS["mul"] == frozenset({"add"})
    assert skills.PARENTS["div"] == frozenset({"sub"})
    assert skills.PARENTS["mod"] == frozenset({"div", "sub"})
    assert skills.PARENTS["max"] == frozenset({"sub"})


def test_commutative_skills_give_the_same_answer_when_swapped():
    for s in skills.COMMUTATIVE:
        assert skills.truth(s, 17, 5) == skills.truth(s, 5, 17)
    assert skills.COMMUTATIVE == frozenset({"add", "mul", "max"})


def test_non_commutative_skills_do_not():
    for s in set(skills.SKILLS) - skills.COMMUTATIVE:
        assert skills.truth(s, 17, 5) != skills.truth(s, 5, 17), s


def test_div_and_mod_refuse_a_zero_divisor():
    for s in ("div", "mod"):
        with pytest.raises(ValueError):
            skills.truth(s, 7, 0)


def test_cell_done_ignores_truncated_and_foreign_stamp_files(tmp_path):
    good = tmp_path / "good.json"
    runlib.atomic_write_text(good, json.dumps({"code_stamp": "abc", "x": 1}))
    foreign = tmp_path / "foreign.json"
    runlib.atomic_write_text(foreign, json.dumps({"code_stamp": "zzz"}))
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"code_stamp": "abc", "x"', encoding="utf-8")
    assert runlib.cell_done(good, "abc")
    assert not runlib.cell_done(foreign, "abc")
    assert not runlib.cell_done(truncated, "abc")
    assert not runlib.cell_done(tmp_path / "missing.json", "abc")


def test_atomic_write_leaves_no_partial_file_when_the_write_fails(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    runlib.atomic_write_text(target, "old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(runlib.os, "replace", boom)
    with pytest.raises(OSError):
        runlib.atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]               # no .tmp left behind


def test_code_stamp_is_the_same_for_crlf_and_lf_and_changes_with_content(tmp_path):
    f = tmp_path / "same.py"
    f.write_bytes(b"x = 1\ny = 2\n")
    lf = runlib.code_stamp([f])
    assert len(lf) == 16
    f.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert runlib.code_stamp([f]) == lf                                       # CRLF folded to LF
    f.write_bytes(b"x = 1\ny = 3\n")
    assert runlib.code_stamp([f]) != lf                                       # content matters


def test_set_threads_sets_torch_threads():
    import torch
    old = torch.get_num_threads()
    try:
        runlib.set_threads(2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(old)

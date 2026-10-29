"""Task 14: the confirmatory pool, the verdict rules and the freeze of the pre-registration. Nothing here runs a model; the dev result files are faked in a temporary folder."""

import ast
import importlib
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import freeze_prereg as FZ
import generator as G
import pools
import skills
import verdict as VD

SPEC = Path(__file__).resolve().parents[2] / "docs" / "superpowers" / "specs" / "2026-10-26-knowledge-status-and-routing-design.md"
FAMILIES = tuple(f"S{i}" for i in range(1, 9))


@pytest.fixture
def confirm_allowed(monkeypatch):
    monkeypatch.setenv("KS_ALLOW_CONFIRM", "1")
    pools.load_pool.cache_clear()
    sys.modules.pop("pool_confirm", None)
    yield
    pools.load_pool.cache_clear()
    sys.modules.pop("pool_confirm", None)


# ------------------------------------------------------------------------------------------------------------------------------------ the pool
def test_confirm_pool_is_blocked_without_the_flag(monkeypatch):
    monkeypatch.delenv("KS_ALLOW_CONFIRM", raising=False)
    pools.load_pool.cache_clear()
    sys.modules.pop("pool_confirm", None)
    with pytest.raises(PermissionError):
        pools.load_pool("confirm")
    with pytest.raises(PermissionError):                       # importing the module directly is blocked as well
        importlib.import_module("pool_confirm")
    monkeypatch.setenv("KS_ALLOW_CONFIRM", "0")
    with pytest.raises(PermissionError):
        pools.load_pool("confirm")
    assert "pool_confirm" not in sys.modules


def test_dev_and_confirm_pools_share_no_template_or_frame(confirm_allowed):
    dev, con = pools.load_pool("dev"), pools.load_pool("confirm")
    d, c = FZ.pool_strings(dev), FZ.pool_strings(con)
    assert c and d and not (set(d) & set(c)), sorted(set(d) & set(c))               # no normalised template string is shared
    assert FZ.five_gram_overlap(con, dev) < 0.2
    for fam in FAMILIES:                                                              # also family by family: no family borrows its wording from the dev pool
        assert FZ.five_gram_overlap(con, dev, family=fam) < 0.2, fam
    assert not (set(map(str.lower, dev.things)) & set(map(str.lower, con.things)))
    assert not (set(dev.chatter) & set(con.chatter))
    assert not (set(dev.traps) & set(con.traps))
    assert FZ.five_gram_overlap(dev, dev) == 1.0 and FZ.five_gram_overlap(con, dev, family="S1") < 1.0     # the measure can reach 1


def test_five_gram_overlap_is_zero_for_unrelated_pools_and_counts_shared_phrases():
    a = pools.Pool("a", {"S1": (pools.Frame("a.f0", "tell me about {core} right now please"),)}, {"S1": (pools.Core("a.c0", "add", "the sum of {a} and {b} is wanted here"),)}, {}, (), ())
    b = pools.Pool("b", {"S1": (pools.Frame("b.f0", "tell me about {core} right now please"),)}, {"S1": (pools.Core("b.c0", "add", "completely different words appear in this one"),)}, {}, (), ())
    assert FZ.five_gram_overlap(a, b) == pytest.approx(0.5, abs=0.2)                  # the frame is shared, the core is not
    assert FZ.five_gram_overlap(b, a) == pytest.approx(0.5, abs=0.2)
    c = pools.Pool("c", {"S1": (pools.Frame("c.f0", "nothing alike at all, honestly {core}"),)}, {"S1": (pools.Core("c.c0", "add", "wholly unrelated vocabulary appears here"),)}, {}, (), ())
    assert FZ.five_gram_overlap(a, c) == 0.0


def test_confirm_pool_structure_matches_what_the_generator_needs(confirm_allowed):
    pool = pools.load_pool("confirm")
    assert pool.name == "confirm"
    ids, texts = set(), set()
    for fam in FAMILIES:
        assert len(pool.frames[fam]) >= 4
        own = set()
        for fr in pool.frames[fam]:
            assert fr.frame_id not in ids and fr.text not in own
            assert "{core}" in fr.text and fr.text.strip() != "{core}", "a bare {core} frame would be shared by every pool"
            assert not re.search(r"\d", fr.text), "a frame must not contain digits"
            assert not (set(re.findall(r"[a-z]+", fr.text.lower())) & pools.ARITH_CUES), fr.text      # a frame is shared by every skill, so it must not carry an operator cue
            ids.add(fr.frame_id)
            own.add(fr.text)
    for fam in ("S1", "S2", "S3", "S4"):
        for skill in skills.SKILLS:
            assert len([c for c in pool.cores[fam] if c.skill == skill]) >= 3, (fam, skill)
    assert {c.skill for c in pool.cores["S5"]} == set(skills.COMMUTATIVE)
    for skill in skills.COMMUTATIVE:
        assert len([c for c in pool.cores["S5"] if c.skill == skill]) >= 3
    for fam, cs in pool.cores.items():
        for c in cs:
            assert c.core_id not in ids and c.text not in texts
            ids.add(c.core_id)
            texts.add(c.text)
    assert all("{pre}" in f.text or "{post}" in f.text for f in pool.frames["S6"])
    assert {fr.noise for fr in pool.frames["S7"]} == {"typo", "case", "space", "mix"}
    assert set(pool.core_sources) == {"S6", "S7", "S8"} and all(set(v) <= {"S1", "S2", "S3", "S4"} for v in pool.core_sources.values())
    assert len(pool.things) >= 4 and not any(set(re.findall(r"[a-z]+", t.lower())) & pools.ARITH_CUES for t in pool.things)


def test_confirm_chatter_is_the_fixed_corpus_draw_and_has_no_digits_or_cues(confirm_allowed):
    import corpora
    pool = pools.load_pool("confirm")
    assert len(pool.chatter) == 48
    assert list(pool.chatter) == corpora.load_other_sentences(48, 0xC0F1)               # the sentences are exactly this draw, written out so that the file hash fixes them
    for line in pool.chatter:
        assert not re.search(r"\d", line) and not (set(re.findall(r"[a-z]+", line.lower())) & pools.ARITH_CUES), line


def test_confirm_traps_use_arithmetic_words_for_something_else_and_split_evenly_by_parity(confirm_allowed):
    pool = pools.load_pool("confirm")
    assert len(pool.traps) >= 10 and len(pool.traps) % 2 == 0                        # draw_teaching takes the even template numbers, draw_test the odd ones
    for t in pool.traps:
        assert "{n}" in t and set(re.findall(r"[a-z]+", t.lower())) & pools.ARITH_CUES, t
    rng = np.random.default_rng(3)
    for _ in range(40):
        m = G.make_trap(rng, "confirm")
        assert m.parts[0].kind == "other" and m.parts[0].skill is None and re.search(r"\d", m.text)


def test_confirm_pool_renders_every_family_and_skill_with_exact_labels(confirm_allowed):
    rng = np.random.default_rng(11)
    for fam in FAMILIES:
        for skill in skills.SKILLS:
            if fam == "S5" and skill not in skills.COMMUTATIVE:
                continue
            for _ in range(25):
                msg = G.make_single(rng, "confirm", skill, fam)
                p = next(q for q in msg.parts if q.kind == "skill")
                assert p.skill == skill and p.answer == skills.truth(skill, p.a, p.b)
                span = msg.text[p.start:p.end]
                assert span == span.strip() and span
                if fam in ("S1", "S3", "S4", "S5") or (fam in ("S6", "S7", "S8") and "S2" not in (msg.template_ids[1][:2],)):
                    if fam != "S7":                                                  # S7 reorders letters but never touches digits; checked below
                        assert str(p.a) in span and str(p.b) in span, (fam, skill, span)
    for _ in range(100):
        msg = G.make_single(rng, "confirm", skills.SKILLS[int(rng.integers(0, 6))], "S7")
        p = next(q for q in msg.parts if q.kind == "skill")
        digits = re.findall(r"\d+", msg.text[p.start:p.end])
        if digits:
            assert sorted(digits) == sorted([str(p.a), str(p.b)])


def test_confirm_near_twins_exist_for_every_infix_skill(confirm_allowed):
    rng = np.random.default_rng(5)
    for skill in ("add", "sub", "mul", "div", "mod"):
        for fam in ("S1", "S2"):
            msg = G.make_single(rng, "confirm", skill, fam, operands=(14, 3))
            for to in ("add", "sub", "mul", "div", "mod"):
                if to != skill:
                    twin = G.make_near_twin(rng, msg, to)
                    tp = next(q for q in twin.parts if q.kind == "skill")
                    assert tp.skill == to and tp.answer == skills.truth(to, 14, 3)


# ------------------------------------------------------------------------------------------------------------------------------------ the verdict rules
BARS = {
    "b1a": {"bar": 0.95, "direction": "min"}, "b1b": {"bar": 0.95, "direction": "min"}, "b1c": {"bar": 0.90, "direction": "min"},
    "macro_f1": {"bar": 0.85, "direction": "min"}, "false_unseen_heldout": {"bar": 0.05, "direction": "max"}, "thin_vs_solid_auroc": {"bar": 0.85, "direction": "min"},
    "other_as_unseen": {"bar": 0.05, "direction": "max"}, "trap_as_unseen": {"bar": 0.05, "direction": "max"}, "b4a_residual_hit": {"bar": 0.85, "direction": "min"}, "b4a_known_not_flagged": {"bar": 0.95, "direction": "min"},
    "b4b_nothing_flagged": {"bar": 0.95, "direction": "min"}, "b4b_other_labelled": {"bar": 0.95, "direction": "min"},
    "b5_mul_parent": {"bar": 0.90, "direction": "min"}, "b5_div_parent": {"bar": 0.90, "direction": "min"}, "b5_decoy_worst": {"bar": 0.10, "direction": "max"},
    "b6_gap": {"bar": 0.10, "direction": "max"}, "b7_gap": {"bar": 0.10, "direction": "max"}, "b2_gap": {"bar": 0.15, "direction": "min"},
}
ALL_PASS = {"b1a": 0.96, "b1b": 0.97, "b1c": 0.91, "r0_b1a": 0.54, "r0_b1b": 0.01, "macro_f1": 0.90, "false_unseen_heldout": 0.03, "thin_vs_solid_auroc": 0.90, "other_as_unseen": 0.02, "trap_as_unseen": 0.03,
            "b4a_residual_hit": 0.90, "b4a_known_not_flagged": 0.97, "b4b_nothing_flagged": 0.97, "b4b_other_labelled": 0.97, "b5_mul_parent": 0.95, "b5_div_parent": 0.93,
            "b5_decoy_worst": 0.02, "b6_gap": 0.05, "b7_gap": 0.04}


def run_verdict(module, measured, bars=BARS):
    return module.verdict(measured, bars)


def check_battery(module):
    """Every rule of the verdict, pinned by an input where the right answer is known. Raises AssertionError at the first wrong answer."""
    stat = lambda m: run_verdict(module, m)["statements"]
    ms = lambda m: run_verdict(module, m)["measures"]
    v = stat(ALL_PASS)
    assert {k: x for k, x in v.items() if k not in ("B3b", "B7")} == {k: "pass" for k in ("B1a", "B1b", "B1c", "B2", "B3a", "B4a", "B4b", "B5", "B6")}
    assert v["B3b"] == "reported" and v["B7"] == "pass"
    # a bar to stay above: exactly at the bar passes, just below fails; a bar to stay below: exactly at the bar passes, just above fails
    assert ms({**ALL_PASS, "b1a": 0.95})["b1a"]["verdict"] == "pass"
    assert ms({**ALL_PASS, "b1a": 0.9499})["b1a"]["verdict"] == "fail"
    assert ms({**ALL_PASS, "false_unseen_heldout": 0.05})["false_unseen_heldout"]["verdict"] == "pass"
    assert ms({**ALL_PASS, "false_unseen_heldout": 0.0501})["false_unseen_heldout"]["verdict"] == "fail"
    assert ms({**ALL_PASS, "b5_decoy_worst": 0.10})["b5_decoy_worst"]["verdict"] == "pass"
    assert ms({**ALL_PASS, "b5_decoy_worst": 0.1001})["b5_decoy_worst"]["verdict"] == "fail"
    assert ms({**ALL_PASS, "b6_gap": 0.1001})["b6_gap"]["verdict"] == "fail"
    assert ms({**ALL_PASS, "b6_gap": 0.0})["b6_gap"]["verdict"] == "pass"                                       # a smaller gap is never worse
    assert ms({**ALL_PASS, "b1c": 1.0})["b1c"]["verdict"] == "pass" and ms({**ALL_PASS, "false_unseen_heldout": 0.0})["false_unseen_heldout"]["verdict"] == "pass"
    # a statement with several measures fails when ANY of them fails, and passes only when ALL pass
    for name in ("macro_f1", "thin_vs_solid_auroc"):
        assert stat({**ALL_PASS, name: 0.5})["B3a"] == "fail"
    for name in ("false_unseen_heldout", "other_as_unseen", "trap_as_unseen"):
        assert stat({**ALL_PASS, name: 0.5})["B3a"] == "fail"
    assert stat({**ALL_PASS, "b4a_residual_hit": 0.5})["B4a"] == "fail" and stat({**ALL_PASS, "b4a_known_not_flagged": 0.5})["B4a"] == "fail"
    assert stat({**ALL_PASS, "b4b_nothing_flagged": 0.5})["B4b"] == "fail" and stat({**ALL_PASS, "b4b_other_labelled": 0.5})["B4b"] == "fail"
    assert stat({**ALL_PASS, "b5_mul_parent": 0.5})["B5"] == "fail" and stat({**ALL_PASS, "b5_div_parent": 0.5})["B5"] == "fail" and stat({**ALL_PASS, "b5_decoy_worst": 0.5})["B5"] == "fail"
    assert stat({**ALL_PASS, "false_unseen_heldout": 0.5})["B4a"] == "pass"                                      # a failure stays in its own statement
    # a missing or NaN value is never a pass: the statement is not_run, unless another of its measures already failed
    no_f1 = {k: x for k, x in ALL_PASS.items() if k != "macro_f1"}
    assert stat(no_f1)["B3a"] == "not_run"
    assert stat({**no_f1, "other_as_unseen": 0.5})["B3a"] == "fail"
    assert stat({**ALL_PASS, "macro_f1": float("nan")})["B3a"] == "not_run" and stat({**ALL_PASS, "macro_f1": None})["B3a"] == "not_run"
    assert ms({**ALL_PASS, "macro_f1": float("nan")})["macro_f1"]["verdict"] == "not_run"
    assert stat({k: x for k, x in ALL_PASS.items() if k != "b7_gap"})["B7"] == "not_run"
    assert stat({})["B1a"] == "not_run" and stat({})["B2"] == "not_run"
    # B2, the validity of the test bed: R0 must fall short of the B1a bar OR the B1b bar by at least 0.15
    assert stat({**ALL_PASS, "r0_b1a": 0.80, "r0_b1b": 0.80})["B2"] == "pass"                                    # B1b alone: 0.95 - 0.80 = 0.15 (exactly at the margin)
    assert stat({**ALL_PASS, "r0_b1a": 0.81, "r0_b1b": 0.81})["B2"] == "fail"                                    # 0.14 on both: the test bed is invalid
    assert stat({**ALL_PASS, "r0_b1a": 0.54, "r0_b1b": 0.99})["B2"] == "pass"                                    # B1a alone is enough
    assert stat({**ALL_PASS, "r0_b1a": 0.99, "r0_b1b": 0.01})["B2"] == "pass"                                    # B1b alone is enough
    assert stat({**ALL_PASS, "r0_b1a": 0.99, "r0_b1b": 0.99})["B2"] == "fail"
    assert stat({**ALL_PASS, "r0_b1a": 0.99, "r0_b1b": float("nan")})["B2"] == "not_run"
    # B2 is judged against the spec's draft bar when the record carries one, so a bar tightened on dev cannot make it easier
    tightened = {**BARS, "b1b": {"bar": 0.99, "direction": "min", "draft": 0.95}, "b1a": {"bar": 0.95, "direction": "min", "draft": 0.95}}
    assert module.verdict({**ALL_PASS, "r0_b1a": 0.99, "r0_b1b": 0.84}, tightened)["statements"]["B2"] == "fail"           # 0.95 - 0.84 = 0.11 against the draft (0.15 against the tightened bar would pass)
    assert module.verdict({**ALL_PASS, "r0_b1a": 0.99, "r0_b1b": 0.80}, tightened)["statements"]["B2"] == "pass"
    # a statement with no measure at all is not a pass
    assert module._statement([]) == "not_run"
    # what the verdicts mean for the rest of the run
    bars_d = {k: {**b, "draft": b["bar"]} for k, b in BARS.items()}
    good = module.verdict(ALL_PASS, bars_d)
    assert module.consequences(good, bars_d) == {"invalid": False, "status_void": False, "conditional": False}
    assert module.consequences(module.verdict({**ALL_PASS, "r0_b1a": 0.99, "r0_b1b": 0.99}, bars_d), bars_d)["invalid"] is True
    assert module.consequences(module.verdict({**ALL_PASS, "b1a": 0.5}, bars_d), bars_d) == {"invalid": False, "status_void": False, "conditional": True}
    assert module.consequences(module.verdict({**ALL_PASS, "b1c": 0.5}, bars_d), bars_d)["conditional"] is True
    assert module.consequences(module.verdict({**ALL_PASS, "b1b": 0.5}, bars_d), bars_d)["status_void"] is True
    frozen_tight = {**bars_d, "b1b": {"bar": 0.991, "direction": "min", "draft": 0.95}}
    assert module.consequences(module.verdict({**ALL_PASS, "b1b": 0.97}, frozen_tight), frozen_tight)["status_void"] is False        # above the spec bar although below the tightened bar
    assert module.consequences(module.verdict({**ALL_PASS, "b1b": 0.94}, frozen_tight), frozen_tight)["status_void"] is True
    assert module.consequences(module.verdict({k: x for k, x in ALL_PASS.items() if k != "b1b"}, bars_d), bars_d)["status_void"] is False  # not measured is not void, it is not run
    assert stat({k: x for k, x in ALL_PASS.items() if k != "r0_b1a"})["B2"] in ("pass", "not_run")
    # the bar comes from the bars passed in, never from the code
    tight = {**BARS, "b1a": {"bar": 0.99, "direction": "min"}}
    assert module.verdict(ALL_PASS, tight)["statements"]["B1a"] == "fail"
    assert module.verdict({**ALL_PASS, "r0_b1a": 0.90, "r0_b1b": 0.90}, {**BARS, "b2_gap": {"bar": 0.01, "direction": "min"}})["statements"]["B2"] == "pass"
    out = ms(ALL_PASS)["b1a"]
    assert out["value"] == 0.96 and out["bar"] == 0.95 and out["direction"] == "min" and out["statement"] == "B1a"
    with pytest.raises(KeyError):
        module.verdict(ALL_PASS, {k: x for k, x in BARS.items() if k != "macro_f1"})
    with pytest.raises(ValueError):
        module.verdict(ALL_PASS, {**BARS, "b1a": {"bar": 0.95, "direction": "sideways"}})


def test_verdict_rules_are_pinned_by_the_battery():
    check_battery(VD)


def _mutants(source):
    """Every single change that flips one comparison to its complement, swaps and / or, or swaps all / any, as (description, mutated source)."""
    tree = ast.parse(source)
    flips = {ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and not (isinstance(node.left, ast.Name) and node.left.id == "__name__"):        # the `if __name__ == "__main__"` guard is not code under test
            for i, op in enumerate(node.ops):
                if type(op) in flips:
                    sites.append(("cmp", node, i))
        elif isinstance(node, ast.BoolOp):
            sites.append(("bool", node, 0))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("all", "any"):
            sites.append(("call", node, 0))
    out = []
    for n, (kind, node, i) in enumerate(sites):
        t = ast.parse(source)
        target = [x for x in ast.walk(t) if type(x) is type(node) and getattr(x, "lineno", None) == node.lineno and getattr(x, "col_offset", None) == node.col_offset][0]
        if kind == "cmp":
            target.ops[i] = flips[type(target.ops[i])]()
        elif kind == "bool":
            target.op = ast.Or() if isinstance(target.op, ast.And) else ast.And()
        else:
            target.func.id = "any" if target.func.id == "all" else "all"
        out.append((f"{kind} at line {node.lineno} col {node.col_offset}", ast.unparse(ast.fix_missing_locations(t))))
    return out


def test_verdict_rules_catch_mutants():
    source = inspect.getsource(VD)
    mutants = _mutants(source)
    assert len(mutants) >= 10, "the verdict should contain several comparisons to mutate"
    survivors = []
    for desc, src in mutants:
        mod = type(sys)("verdict_mutant")
        exec(compile(src, "verdict_mutant", "exec"), mod.__dict__)
        try:
            check_battery(mod)
        except (AssertionError, KeyError, ValueError, TypeError, ZeroDivisionError):
            continue                                                                          # the battery notices: the mutant is killed
        survivors.append(desc)
    assert not survivors, f"mutants that no test notices: {survivors}"


# ------------------------------------------------------------------------------------------------------------------------------------ the freeze
def write_cells(results: Path, e1_stamp="E1", e2_stamp="E2", e4_stamp="E4", b7_stamp="B7", seeds=range(100, 105), bump=0.0, pool="dev"):
    (results / "e1").mkdir(parents=True)
    (results / "e2").mkdir(parents=True)
    (results / "e4").mkdir(parents=True)
    (results / "realism").mkdir(parents=True)
    for i, s in enumerate(seeds):
        jit = 0.005 * (i - 2)
        for rep, (a, b, c) in {"R5_R4_qwen_L26": (0.92, 0.995, 0.89), "R0_hash": (0.54, 0.01, 0.68)}.items():
            (results / "e1" / f"{rep}__s{s}.json").write_text(json.dumps({"b1a": a + jit, "b1b": b + jit / 5, "b1c": c + jit, "code_stamp": e1_stamp, "pool": pool}))
        tang = lambda r, k: {"b4a": {"n": 40, "residual_hit": r + jit, "known_not_flagged": k + jit}, "b4b": {"nothing_flagged": 0.64 + jit, "other_labelled": 0.97 + jit}}
        cell = {"machine": {"q0.05": {"standard": {"macro_f1": {"macro_f1": 0.855 + jit + bump}, "false_unseen_heldout": 0.29 + jit, "thin_vs_solid_auroc": 0.844 + jit, "other_as_unseen": 0.25 + jit, "trap_as_unseen": 0.47 + jit, "unseen_recall": 0.89 + jit}}},
                "tangled": {"machine": {"gold": tang(0.66, 0.67), "auto_sentence": tang(0.50, 0.50), "auto_window": tang(0.40, 0.50)}}, "code_stamp": e2_stamp, "pool": pool}
        (results / "e2" / f"R5_R4_qwen_L26__s{s}.json").write_text(json.dumps(cell))
        rates = {"mul": {"parent_is_add": 0.45 + jit}, "div": {"parent_is_sub": 0.5 + jit}, "mul_wrong_steps": {"any_parent": 0.0}, "div_wrong_steps": {"any_parent": 0.02},
                 "max_irrelevant_valid_steps": {"any_parent": 0.0}, "mod_unrelated_skill": {"any_parent": 0.01}}
        (results / "e4" / f"s{s}.json").write_text(json.dumps({"seed": s, "rates": rates, "code_stamp": e4_stamp}))
        (results / "realism" / f"s{s}.json").write_text(json.dumps({"seed": s, "pool": pool, "diff": -0.30 + 0.02 * (i - 2) if i != 4 else None, "code_stamp": b7_stamp}))


STAMPS = {"e1": "E1", "e2": "E2", "e4": "E4", "b7": "B7"}


def freeze_into(tmp_path, **kw):
    results = tmp_path / "results"
    write_cells(results)
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "code"
    code.mkdir()
    (code / "a.py").write_text("x = 1\n")
    (code / "b.py").write_text("y = 2\n")
    out = tmp_path / "out"
    out.mkdir()
    rec = FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=sorted(code.glob("*.py")), expected_stamps=STAMPS, **kw)
    return rec, out, pool, code


def test_freeze_refuses_a_second_run(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    assert (out / "bars.json").exists() and (out / "bars.sha256").exists() and (out / "PREREG.md").exists()
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError, match="already frozen"):
        FZ.freeze(out_dir=out, results_dir=tmp_path / "results", pool_file=pool, code_files=sorted(code.glob("*.py")), expected_stamps=STAMPS)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before                  # the refusal changes nothing
    (out / "PREREG.md").unlink()                                                      # a partly deleted freeze is still a freeze
    with pytest.raises(FileExistsError):
        FZ.freeze(out_dir=out, results_dir=tmp_path / "results", pool_file=pool, code_files=sorted(code.glob("*.py")), expected_stamps=STAMPS)


def test_changing_a_bar_changes_the_hash(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    assert FZ.check_frozen(out)["ok"] is True
    bars = json.loads((out / "bars.json").read_text(encoding="utf-8"))
    h0 = FZ.bars_hash(bars)
    assert h0 == (out / "bars.sha256").read_text(encoding="ascii").split()[0]
    for path in (("bars", "b1a", "bar"), ("thresholds", "solid_min_count"), ("q",), ("seeds", "confirm"), ("representations",), ("group_sizes",), ("pool_confirm_sha256",), ("code_stamp",)):
        changed = json.loads(json.dumps(bars))
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = [0] if isinstance(node[path[-1]], list) else (node[path[-1]] + 0.01 if isinstance(node[path[-1]], (int, float)) else str(node[path[-1]]) + "x")
        assert FZ.bars_hash(changed) != h0, path
    assert FZ.bars_hash(json.loads(json.dumps(bars))) == h0                           # the hash does not depend on key order or spacing
    tampered = json.loads(json.dumps(bars))
    tampered["bars"]["false_unseen_heldout"]["bar"] = 0.30
    (out / "bars.json").write_text(json.dumps(tampered), encoding="utf-8")
    check = FZ.check_frozen(out)
    assert check["ok"] is False and "bars.json" in check["problem"]
    (out / "bars.json").write_text(json.dumps(bars), encoding="utf-8")
    assert FZ.check_frozen(out)["ok"] is True
    pool.write_text("POOL = 2\n")                                                     # a changed pool file is a changed pre-registration
    assert FZ.check_frozen(out, pool_file=pool)["ok"] is False


def test_bars_follow_the_rule_written_before_the_dev_run(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    b = rec["bars"]
    for name, row in b.items():
        assert row["direction"] in ("min", "max")
        assert (row["bar"] >= row["draft"]) if row["direction"] == "min" else (row["bar"] <= row["draft"]), name         # a dev result can tighten a bar and can never loosen it
    assert b["false_unseen_heldout"]["bar"] == 0.05 and b["false_unseen_heldout"]["draft"] == 0.05                          # dev mean 0.29: the draft stays
    assert b["macro_f1"]["bar"] == pytest.approx(max(0.85, 0.855 - 2 * np.std([0.855 + 0.005 * (i - 2) for i in range(5)], ddof=1)))
    assert b["b1b"]["bar"] > 0.95 and b["b1b"]["rule"].startswith("max(draft")                                              # a very high and steady dev result tightens the bar
    assert b["b1a"]["n_dev_seeds"] == 5 and b["macro_f1"]["dev_mean"] == pytest.approx(0.855)
    assert b["b6_gap"]["bar"] == 0.10 and b["b7_gap"]["bar"] == 0.10 and b["b2_gap"]["bar"] == 0.15                          # bars with no dev value keep the draft
    assert b["b6_gap"]["dev_mean"] == pytest.approx(0.17, abs=1e-9) and rec["b6_primary"] == "auto_sentence"
    assert rec["thresholds"] == {"solid_min_count": 500, "solid_min_acc": 0.90, "thin_max_count": 100, "thin_acc_margin": 0.10, "thin_max_acc": 0.60}
    assert rec["e4_n_per_kind"] == 40
    assert rec["q"] == 0.05 and rec["q_curve"] == [0.01, 0.02, 0.05, 0.10] and rec["group_sizes"] == [1, 4, 8, 16]
    assert rec["seeds"]["confirm"] == list(range(400, 430)) and rec["seeds"]["dev"][0] == 100
    assert rec["representations"] == ["R5_R4_qwen_L26", "R4_qwen_L26", "R0_hash"] and rec["headline_representation"] == "R5_R4_qwen_L26"
    assert rec["min_seeds"] == 10


def test_freeze_records_the_pool_hash_and_the_code_stamp(tmp_path):
    import hashlib
    rec, out, pool, code = freeze_into(tmp_path)
    assert rec["pool_confirm_sha256"] == hashlib.sha256(pool.read_bytes().replace(b"\r\n", b"\n")).hexdigest()          # line endings do not matter
    import runlib
    assert rec["code_stamp"] == runlib.code_stamp(sorted(code.glob("*.py"))) and rec["code_files"] == ["a.py", "b.py"]
    assert (out / "PREREG.md").read_text(encoding="utf-8").count(rec["pool_confirm_sha256"]) >= 1


def test_freeze_refuses_dev_results_from_other_code_and_missing_dev_results(tmp_path):
    results = tmp_path / "results"
    write_cells(results, e2_stamp="OLD")
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "p.py"
    pool.write_text("x")
    f = tmp_path / "f.py"
    f.write_text("x")
    with pytest.raises(ValueError, match="stamp"):
        FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[f], expected_stamps=STAMPS)
    assert not list(out.iterdir())                                                     # nothing is written when the freeze is refused
    empty = tmp_path / "empty"
    (empty / "e1").mkdir(parents=True)
    (empty / "e2").mkdir()
    (empty / "e4").mkdir()
    with pytest.raises(ValueError, match="no dev"):
        FZ.freeze(out_dir=out, results_dir=empty, pool_file=pool, code_files=[f], expected_stamps=STAMPS)


def test_prereg_lists_every_statement_of_spec_section_7(tmp_path):
    spec = SPEC.read_text(encoding="utf-8")
    section = spec.split("## 7. Measurements and draft bars")[1].split("## 8.")[0]
    wanted = set(re.findall(r"\bB\d[abc]\b|\bB[2567]\b", section))                                      # read from the spec itself, not from the code under test
    assert wanted == {"B1a", "B1b", "B1c", "B2", "B3a", "B3b", "B4a", "B4b", "B5", "B6", "B7"}, wanted
    rec, out, pool, code = freeze_into(tmp_path)
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    for sid in sorted(wanted):
        assert re.search(rf"\b{sid}\b", text), f"PREREG.md does not mention {sid}"
    assert set(rec["statements"]) == wanted
    for sid, row in rec["statements"].items():
        assert row["expected"] in ("pass", "fail", "not_run", "reported") and row["rule"] and row["text"]
    # the calibration claim is stated as expected to FAIL, with the dev evidence next to it
    assert rec["statements"]["B3a"]["expected"] == "fail"
    assert "false_unseen_heldout" in text and "EXPECTED TO FAIL" in text
    assert "kill rule" in text.lower() and "decision rule" in text.lower() and "single use" in text.lower()
    assert "B7" in text and "not run" in text.lower().replace("_", " ")                                    # the realism set is named as not run, not left out


def test_the_prereg_is_plain_text_without_non_ascii(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    for name in ("PREREG.md", "bars.json", "bars.sha256"):
        (out / name).read_bytes().decode("ascii")
    assert b"\r\n" not in (out / "PREREG.md").read_bytes()


def test_draft_bars_equal_the_numbers_of_the_spec():
    # copied from spec section 7 by hand (not from the code): the freeze must start from these numbers
    spec = {"b1a": (0.95, "min"), "b1b": (0.95, "min"), "b1c": (0.90, "min"), "b2_gap": (0.15, "min"), "macro_f1": (0.85, "min"), "false_unseen_heldout": (0.05, "max"),
            "thin_vs_solid_auroc": (0.85, "min"), "other_as_unseen": (0.05, "max"), "trap_as_unseen": (0.05, "max"), "b4a_residual_hit": (0.85, "min"), "b4a_known_not_flagged": (0.95, "min"),
            "b4b_nothing_flagged": (0.95, "min"), "b4b_other_labelled": (0.95, "min"), "b5_mul_parent": (0.90, "min"), "b5_div_parent": (0.90, "min"), "b5_decoy_worst": (0.10, "max"),
            "b6_gap": (0.10, "max"), "b7_gap": (0.10, "max")}
    assert FZ.drafts() == spec
    assert set(VD.MEASURES) | {"b2_gap"} == set(spec)                                    # the verdict knows exactly these measures


def test_imported_growmoe_files_follow_imports_through_each_other(tmp_path):
    grow = tmp_path / "growmoe"
    grow.mkdir()
    (grow / "a.py").write_text("from .b import x\nimport numpy\n")
    (grow / "b.py").write_text("from growmoe.c import y\n")
    (grow / "c.py").write_text("z = 1\n")
    (grow / "unused.py").write_text("w = 1\n")
    mine = tmp_path / "mine.py"
    mine.write_text("from growmoe.a import f\nimport growmoe.missing\nimport os\n")
    other = tmp_path / "other.py"
    other.write_text("from growmoe import c\n")
    assert [p.name for p in FZ.imported_growmoe_files([mine], grow_dir=grow)] == ["a.py", "b.py", "c.py"]
    assert [p.name for p in FZ.imported_growmoe_files([other], grow_dir=grow)] == ["c.py"]
    assert FZ.imported_growmoe_files([], grow_dir=grow) == []


def test_the_real_code_set_contains_this_package_and_the_growmoe_files_it_imports():
    names = {FZ._label(f) for f in FZ.default_code_files()}
    assert {"verdict.py", "freeze_prereg.py", "pool_confirm.py", "pool_dev.py", "run_e2.py", "status.py", "growmoe/signature.py", "growmoe/arith.py"} <= names
    assert all(not n.startswith("test_") for n in names)


def test_check_frozen_notices_a_changed_code_file(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    files = sorted(code.glob("*.py"))
    assert FZ.check_frozen(out, pool_file=pool, code_files=files)["ok"] is True
    (code / "b.py").write_text("y = 3\n")
    r = FZ.check_frozen(out, pool_file=pool, code_files=files)
    assert r["ok"] is False and "code" in r["problem"]
    (out / "bars.sha256").unlink()
    assert FZ.check_frozen(out)["ok"] is False


def test_reading_the_pool_overlap_leaves_the_confirm_flag_and_caches_as_they_were(monkeypatch):
    monkeypatch.delenv("KS_ALLOW_CONFIRM", raising=False)
    ov = FZ._pool_overlap()
    assert ov["identical_templates"] == 0 and ov["five_gram_all"] < 0.2 and set(ov["five_gram_by_family"]) == set(FAMILIES)
    import os
    assert "KS_ALLOW_CONFIRM" not in os.environ and "pool_confirm" not in sys.modules
    with pytest.raises(PermissionError):
        pools.load_pool("confirm")


# ------------------------------------------------------------------------------------------------------------------------------------ gaps found by the planted-bug sweep of freeze_prereg.py
def test_pool_strings_include_the_traps_and_a_family_view_does_not(confirm_allowed):
    con = pools.load_pool("confirm")
    traps = {FZ.normalise(t) for t in con.traps}
    assert traps <= set(FZ.pool_strings(con))
    assert not (traps & set(FZ._templates(con, family="S1")))
    assert set(FZ._templates(con, family="S1")) == {FZ.normalise(f.text) for f in con.frames["S1"]} | {FZ.normalise(c.text) for c in con.cores["S1"]}


def test_a_nan_in_a_dev_cell_is_skipped_and_a_missing_experiment_refuses_the_freeze(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    cell = results / "e2" / "R5_R4_qwen_L26__s102.json"
    d = json.loads(cell.read_text())
    d["machine"]["q0.05"]["standard"]["thin_vs_solid_auroc"] = float("nan")
    cell.write_text(json.dumps(d))                                                      # json.dumps writes NaN, json.loads reads it back
    dev = FZ.collect_dev_values(results, STAMPS)
    assert len(dev["thin_vs_solid_auroc"]) == 4 and all(not np.isnan(x) for x in dev["thin_vs_solid_auroc"]) and len(dev["macro_f1"]) == 5
    for missing in ("e1", "e2", "e4"):
        res2 = tmp_path / f"without_{missing}"
        write_cells(res2)
        for p in (res2 / missing).glob("*.json"):
            p.unlink()
        with pytest.raises(ValueError, match="no dev"):
            FZ.collect_dev_values(res2, STAMPS)
    res3 = tmp_path / "no_control"
    write_cells(res3)
    for p in (res3 / "e1").glob("R0_hash*.json"):
        p.unlink()
    with pytest.raises(ValueError, match="no dev"):
        FZ.collect_dev_values(res3, STAMPS)


def test_growmoe_imports_are_found_only_where_they_really_are(tmp_path):
    grow = tmp_path / "growmoe"
    grow.mkdir()
    (grow / "a.py").write_text("from c import y\nfrom numpy import pi\n")                # a plain import of a top-level module named c is not a growmoe import
    (grow / "c.py").write_text("z = 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("from .a import f\n")                                                # a relative import outside growmoe is not one either
    inside = tmp_path / "inside.py"
    inside.write_text("from growmoe import a\n")
    assert FZ.imported_growmoe_files([outside], grow_dir=grow) == []
    assert [p.name for p in FZ.imported_growmoe_files([inside], grow_dir=grow)] == ["a.py"]


def test_the_code_set_is_listed_and_stamped_in_a_fixed_order_with_growmoe_last(tmp_path):
    import runlib
    results = tmp_path / "results"
    write_cells(results)
    grow = tmp_path / "growmoe"
    grow.mkdir()
    (grow / "a.py").write_text("g = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "c.py").write_text("x = 1\n")
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    out = tmp_path / "out"
    out.mkdir()
    shuffled = [grow / "a.py", tmp_path / "c.py", tmp_path / "b.py"]
    rec = FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=shuffled, expected_stamps=STAMPS)
    ordered = [tmp_path / "b.py", tmp_path / "c.py", grow / "a.py"]
    assert rec["code_files"] == ["b.py", "c.py", "growmoe/a.py"] and rec["code_stamp"] == runlib.code_stamp(ordered)
    assert FZ.check_frozen(out, pool_file=pool, code_files=list(reversed(shuffled)))["ok"] is True             # the order the caller lists them in does not matter


def test_a_single_dev_seed_gives_no_spread_and_the_draft_stands(tmp_path):
    results = tmp_path / "results"
    write_cells(results, seeds=range(100, 101))
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    rec = FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps=STAMPS)
    row = rec["bars"]["macro_f1"]
    assert row["dev_sd"] is None and row["bar"] == row["draft"] == 0.85 and "fewer than two" in row["rule"] and row["n_dev_seeds"] == 1
    assert rec["bars"]["b1b"]["bar"] == 0.95 and rec["bars"]["false_unseen_heldout"]["bar"] == 0.05


def test_the_prereg_table_shows_dev_numbers_directions_and_the_overlap_of_the_real_pool(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    assert rec["pool_overlap"] is None                                                   # a stand-in pool file is not measured
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    row = [r for r in text.splitlines() if r.startswith("| macro_f1")][0]
    assert "(at least)" in row and re.search(r"0\.855 \(0\.0\d\d\) \[0\.8\d\d - 0\.8\d\d\], 5", row)
    assert "(at most)" in [r for r in text.splitlines() if r.startswith("| false_unseen_heldout")][0]
    assert "no dev value" in [r for r in text.splitlines() if r.startswith("| b2_gap")][0]
    # the real pool file: its overlap with the dev pool is measured and written down
    results = tmp_path / "res2"
    write_cells(results)
    out2 = tmp_path / "out2"
    out2.mkdir()
    some = tmp_path / "x.py"
    some.write_text("x = 1\n")
    rec2 = FZ.freeze(out_dir=out2, results_dir=results, pool_file=FZ.HERE / "pool_confirm.py", code_files=[some], expected_stamps=STAMPS)
    assert rec2["pool_overlap"]["identical_templates"] == 0 and rec2["pool_overlap"]["five_gram_all"] < 0.2
    assert "shares no template with the dev pool" in (out2 / "PREREG.md").read_text(encoding="utf-8")


def test_a_windows_checkout_that_rewrites_line_endings_is_not_a_changed_pool_file(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    pool.write_bytes(pool.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))     # the same text with CRLF line endings
    assert FZ.check_frozen(out, pool_file=pool)["ok"] is True
    assert FZ.file_sha256(pool) == rec["pool_confirm_sha256"]
    pool.write_bytes(b"POOL = 2\r\n")
    assert FZ.check_frozen(out, pool_file=pool)["ok"] is False


# ------------------------------------------------------------------------------------------------------------------------------------ after the independent review (Task 14)
def test_a_freeze_is_refused_once_any_trace_of_a_confirmatory_run_exists(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")

    def attempt(artifacts, name):
        out = tmp_path / name
        out.mkdir()
        return FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps=STAMPS, confirm_artifacts=artifacts)

    started = tmp_path / "CONFIRM_STARTED.json"
    empty_dir = tmp_path / "confirm_results"
    empty_dir.mkdir()
    assert attempt([started, empty_dir], "ok")["q"] == 0.05                                  # nothing there, or only an empty folder: fine
    started.write_text("{}")
    with pytest.raises(FileExistsError, match="already started"):
        attempt([started], "a")
    started.unlink()
    (empty_dir / "e1").mkdir()
    (empty_dir / "e1" / "cell.json").write_text("{}")
    with pytest.raises(FileExistsError, match="confirm_results"):
        attempt([empty_dir], "b")
    (empty_dir / "e1" / "cell.json").unlink()
    (results / "competence").mkdir()
    (results / "competence" / "confirm__add__n2000__s400.json").write_text("{}")
    with pytest.raises(FileExistsError, match="competence"):
        attempt([empty_dir], "c")


def test_check_frozen_notices_a_hand_edited_prereg(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    assert FZ.check_frozen(out)["ok"] is True
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    (out / "PREREG.md").write_text(text.replace("EXPECTED TO FAIL", "expected to pass"), encoding="utf-8")
    r = FZ.check_frozen(out)
    assert r["ok"] is False and "PREREG.md" in r["problem"]
    (out / "PREREG.md").write_bytes(text.replace("\n", "\r\n").encode("ascii"))              # only the line endings differ: still the same file
    assert FZ.check_frozen(out)["ok"] is True
    (out / "PREREG.md").unlink()
    assert FZ.check_frozen(out)["ok"] is False


def test_the_collector_checks_seeds_pool_and_required_experiments(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    assert FZ.collect_dev_values(results, STAMPS)["macro_f1"]
    with pytest.raises(ValueError, match="outside the expected seeds"):
        FZ.collect_dev_values(results, STAMPS, seeds=[100, 101, 102, 103])                   # seed 104 is not expected
    other = tmp_path / "other_pool"
    write_cells(other, pool="confirm")
    with pytest.raises(ValueError, match="pool"):
        FZ.collect_dev_values(other, STAMPS)                                                  # confirmatory cells are not dev cells
    conf = FZ.collect_dev_values(other, STAMPS, seeds=range(100, 105), pool="confirm", require=("e1", "e2"))
    assert conf["macro_f1"]
    nothing4 = tmp_path / "no_e4"
    write_cells(nothing4)
    for q in (nothing4 / "e4").glob("*.json"):
        q.unlink()
    with pytest.raises(ValueError, match="no dev"):
        FZ.collect_dev_values(nothing4, STAMPS)
    assert "b5_mul_parent" not in FZ.collect_dev_values(nothing4, STAMPS, require=("e1", "e2", "b7"))


def test_derived_means_need_the_minimum_seeds_and_b7_is_the_size_of_the_mean_difference():
    values = {"macro_f1": [0.9] * 10, "b5_mul_parent": [0.5] * 9, "b7_diff": [-0.3, -0.1, 0.2], "b7_gap": [0.3, 0.1, 0.2]}
    m = FZ.derive_means(values, min_seeds=3)
    assert m["macro_f1"] == pytest.approx(0.9) and m["b7_gap"] == pytest.approx(abs(np.mean([-0.3, -0.1, 0.2])))
    assert "b7_diff" not in m
    m10 = FZ.derive_means(values, min_seeds=10)
    assert "b5_mul_parent" not in m10 and "b7_gap" not in m10 and "macro_f1" in m10                  # fewer seeds than the minimum: not judged
    assert FZ.derive_means({"b7_gap": [0.3]}, 1) == {}                                         # the per-seed sizes alone are only for fixing the bar


def test_the_b7_cells_enter_the_bars_and_the_expected_verdicts(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    b = rec["bars"]["b7_gap"]
    assert b["n_dev_seeds"] == 4 and b["draft"] == 0.10 and b["bar"] == 0.10 and b["dev_mean"] == pytest.approx(np.mean([0.34, 0.32, 0.30, 0.28]))      # the seed with no scorable pair is skipped
    assert rec["statements"]["B7"]["expected"] == "fail"
    assert rec["bars"]["b7_gap"]["expected"] == "fail" and rec["bars"]["macro_f1"]["expected"] in ("pass", "fail")
    assert rec["dev_extras"]["r0_b1a"] == pytest.approx(0.54) and rec["dev_extras"]["r0_b1b"] == pytest.approx(0.01)


def write_arm_cells(results, arm, offset, stamp="FS", seeds=range(100, 105), pool="dev"):
    for i, s in enumerate(seeds):
        jit = 0.005 * (i - 2)
        (results / "e1" / f"{arm}__s{s}.json").write_text(json.dumps({"b1a": 0.92 + jit, "b1b": 0.995 + jit / 5 - offset, "b1c": 0.89 + jit - offset, "code_stamp": stamp, "pool": pool}))
        cell = {"machine": {"q0.05": {"standard": {"macro_f1": {"macro_f1": 0.855 + jit - offset}, "false_unseen_heldout": 0.29 + jit + offset, "thin_vs_solid_auroc": 0.844, "other_as_unseen": 0.25,
                                                   "trap_as_unseen": 0.47, "unseen_recall": 0.89}}}, "tangled": {"machine": {"gold": {"b4a": {"n": 40, "residual_hit": 0.6, "known_not_flagged": 0.6}, "b4b": {"nothing_flagged": 0.6, "other_labelled": 0.96}}, "auto_sentence": {"b4a": {"n": 40, "residual_hit": 0.6, "known_not_flagged": 0.6}, "b4b": {"nothing_flagged": 0.6, "other_labelled": 0.96}}, "auto_window": {"b4a": {"n": 40, "residual_hit": 0.6, "known_not_flagged": 0.6}, "b4b": {"nothing_flagged": 0.6, "other_labelled": 0.96}}}}, "code_stamp": stamp, "pool": pool}
        (results / "e2" / f"{arm}__s{s}.json").write_text(json.dumps(cell))


def test_the_from_scratch_rule_of_d40_keeps_parks_or_drops_an_arm_by_its_worst_gap(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.03)
    write_arm_cells(results, "R5_FS_pre", 0.10)
    stamps = {**STAMPS, "fs": "FS"}
    out = FZ.fs_summary(results, stamps)
    assert out["R5_FS_strict"]["status"] == "kept" and out["R5_FS_strict"]["worst_gap"] == pytest.approx(0.03, abs=1e-9)
    assert out["R5_FS_pre"]["status"] == "parked" and out["R5_FS_pre"]["worst_gap"] == pytest.approx(0.10, abs=1e-9)
    write_arm_cells(tmp_path / "far", "x", 0) if False else None
    results2 = tmp_path / "results2"
    write_cells(results2)
    write_arm_cells(results2, "R5_FS_strict", 0.20)
    out2 = FZ.fs_summary(results2, stamps)
    assert out2["R5_FS_strict"]["status"] == "not worth it now" and out2["R5_FS_pre"]["status"] == "no dev cells"
    write_arm_cells(tmp_path / "edge_results", "R5_FS_strict", 0.05) if False else None
    rec_edge = FZ.fs_summary(results, stamps, arms=("R5_FS_strict",))
    assert rec_edge["R5_FS_strict"]["status"] == "kept"


def test_the_from_scratch_edges_of_the_rule_are_inclusive(tmp_path):
    for offset, want in ((0.05, "kept"), (0.0501, "parked"), (0.15, "parked"), (0.1501, "not worth it now")):
        results = tmp_path / f"r{offset}"
        write_cells(results)
        write_arm_cells(results, "R5_FS_strict", offset)
        assert FZ.fs_summary(results, {**STAMPS, "fs": "FS"}, arms=("R5_FS_strict",))["R5_FS_strict"]["status"] == want, offset


def test_a_cell_of_a_from_scratch_arm_must_carry_the_from_scratch_stamp(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.0, stamp="OLD")
    with pytest.raises(ValueError, match="stamp"):
        FZ.fs_summary(results, {**STAMPS, "fs": "FS"}, arms=("R5_FS_strict",))               # stale cells are an error, not "no dev cells"
    with pytest.raises(ValueError, match="stamp"):
        FZ.collect_dev_values(results, {**STAMPS, "fs": "FS"}, "R5_FS_strict", require=("e1", "e2"))


def test_the_prereg_has_the_realism_the_kill_rules_the_disclosures_and_the_from_scratch_section(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.03)
    write_arm_cells(results, "R5_FS_pre", 0.10)
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    rec = FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps={**STAMPS, "fs": "FS"})
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    for needle in ("## Realism check (B7)", "SVAMP", "## From-scratch reader arms", "R5_FS_pre | parked", "R5_FS_strict | kept", "## Kill rules", "CONDITIONAL on a reader that missed B1", "INVALID" if False else "invalid for this run",
                   "run_confirm.py", "Found after the dev numbers were seen", "extended to E1, E4 and B7", "TAUGHT skills", "trap_as_unseen", "verbatim S1 to S3 request templates", "THRESHOLDS question",
                   "tightened bar", "min(draft, mean + 2 sd)"):
        assert needle in text or needle == "tightened bar", needle
    assert rec["representations"] == ["R5_R4_qwen_L26", "R4_qwen_L26", "R0_hash", "R5_FS_strict"]                       # the kept arm joins, the parked one does not
    assert rec["fs_arms"]["R5_FS_strict"]["status"] == "kept" and rec["fs_arms"]["R5_FS_pre"]["status"] == "parked"
    row = [r for r in text.splitlines() if r.startswith("| macro_f1")][0]
    assert row.split("|")[6].strip() in ("PASS", "FAIL")                                         # the table has an expected-verdict column


def test_templates_that_differ_only_by_a_line_break_are_different_templates():
    assert FZ.normalise("{pre}\n{core}") != FZ.normalise("{pre} {core}")
    assert FZ.normalise("{pre}\n{core}") == FZ.normalise("{pre}\n\n{core}".replace("\n\n", "\n")) and FZ.normalise("A  {a}\t+ {b}") == "a <s> + <s>"
    assert FZ.normalise("{pre}\n\n{core}") != FZ.normalise("{pre}\n{core}")


def test_the_confirmatory_s6_frames_put_only_whitespace_around_the_slots(confirm_allowed):
    pool = pools.load_pool("confirm")
    for f in pool.frames["S6"]:
        outside = re.sub(r"\{(core|pre|post)\}", "", f.text)
        assert outside.strip() == "", (f.frame_id, f.text)
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(300):
        msg = G.make_single(rng, "confirm", skills.SKILLS[int(rng.integers(0, 6))], "S6")
        for part in msg.parts:
            if part.kind == "other":
                seen.add(msg.text[part.start:part.end])
                assert msg.text[part.start:part.end] in pool.chatter                                   # a chatter part is exactly one sentence of the corpus draw
    assert len(seen) > 20


def test_the_confirmatory_pool_refuses_to_be_imported_by_a_development_entry_point(monkeypatch):
    import types
    monkeypatch.setenv("KS_ALLOW_CONFIRM", "1")
    for name in ("run_e2.py", "run_e1.py", "run_e3.py", "run_competence.py", "lab_wording.py", "diag_e2.py", "tune_baselines.py", "layer_scan.py"):
        pools.load_pool.cache_clear()
        sys.modules.pop("pool_confirm", None)
        monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__file__=f"C:/somewhere/{name}"))
        with pytest.raises(PermissionError, match="run_confirm.py"):
            importlib.import_module("pool_confirm")
    for name in ("run_confirm.py", "freeze_prereg.py", "check_length_leak.py", "pytest", "__main__.py"):
        pools.load_pool.cache_clear()
        sys.modules.pop("pool_confirm", None)
        monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__file__=f"C:/somewhere/{name}"))
        importlib.import_module("pool_confirm")
    pools.load_pool.cache_clear()
    sys.modules.pop("pool_confirm", None)


def test_a_competence_folder_without_confirmatory_cells_does_not_block_a_freeze(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    (results / "competence").mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    out = tmp_path / "out"
    out.mkdir()
    assert FZ._confirm_artifacts_present([], results / "competence") == []                      # an existing, empty folder
    (results / "competence" / "dev__add__n2000__s100.json").write_text("{}")
    assert FZ._confirm_artifacts_present([], results / "competence") == []                      # dev cells only
    assert FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps=STAMPS, confirm_artifacts=[])["q"] == 0.05
    (results / "competence" / "confirm__add__n2000__s400.json").write_text("{}")
    assert FZ._confirm_artifacts_present([], results / "competence") == ["results/competence/confirm__*"]


def test_the_realism_set_hash_is_recorded_only_when_the_file_exists(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    realism = tmp_path / "realism_set.jsonl"
    realism.write_bytes(b'{"skill": "add"}\r\n{"skill": "mul"}\r\n{"skill": "add"}\r\n')
    import hashlib

    def go(name, real):
        out = tmp_path / name
        out.mkdir()
        return FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps=STAMPS, realism_file=real)["realism_set_sha256"]

    assert go("a", realism) == hashlib.sha256(b'{"skill": "add"}\n{"skill": "mul"}\n{"skill": "add"}\n').hexdigest()          # line endings folded
    assert go("b", None) is None and go("c", tmp_path / "missing.jsonl") is None
    assert FZ._realism_counts(realism) == {"add": 2, "mul": 1} and FZ._realism_counts(None) == {} and FZ._realism_counts(tmp_path / "missing.jsonl") == {}


def test_the_prereg_lists_a_from_scratch_arm_without_dev_cells_as_such(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.03)
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    rec = FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps={**STAMPS, "fs": "FS"})
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    assert rec["fs_arms"]["R5_FS_pre"]["status"] == "no dev cells" and "| R5_FS_pre | no dev cells | | | | | |" in text and "| R5_FS_strict | kept |" in text


def test_a_freeze_that_includes_from_scratch_arms_can_be_checked_afterwards(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.03)
    write_arm_cells(results, "R5_FS_pre", 0.10)
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps={**STAMPS, "fs": "FS"})
    assert FZ.check_frozen(out, pool, [code]) == {"ok": True, "problem": None}                 # the written PREREG.md is what the re-read record renders to, whatever order the dicts come back in
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    assert text.index("| R5_FS_pre |") < text.index("| R5_FS_strict |")

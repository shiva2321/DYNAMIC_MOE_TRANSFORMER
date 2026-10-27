"""Task 11, the E2/E3 runner: the world it builds (no leakage, labels from the assignment and the measured competence), and one full cell on the hashed n-gram reader with tiny
sizes (no language model). The real run on the Qwen reader is a command, not a test."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import assignments
import generator as G
import reps
import run_e2 as R
import skills

TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05, 0.10))


def fake_competence(pool, skill, count, seed):
    return {"accuracy": 0.97 if count >= 60 else 0.5, "chance": 0.1, "status": "x"}


@pytest.fixture(scope="module", autouse=True)
def small_thresholds():
    """The sizes of the tiny world are far below the real ones, so the Solid / Thin rule is scaled to match (the dict is mutated in place, so the machine and the truth both see it).
    Module scope, so that the module-scoped world below is built under the same rule."""
    mp = pytest.MonkeyPatch()
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        mp.setitem(assignments.THRESHOLDS, k, v)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def world():
    return R.build_world(100, "dev", TINY, fake_competence)


def fps(texts):
    return {R.fingerprint(t) for t in texts}


def test_fingerprint_is_the_one_assignments_uses():
    ex = assignments.draw_teaching(np.random.default_rng(0), "dev", "add", 3)
    assert {R.fingerprint(e.text) for e in ex} == set(assignments.text_keys(ex))


def test_world_is_deterministic_and_does_not_depend_on_the_reader():
    a, b = R.build_world(101, "dev", TINY, fake_competence), R.build_world(101, "dev", TINY, fake_competence)
    assert a.assignment == b.assignment and a.all_texts() == b.all_texts() and [c for c in a.test_cases] == [c for c in b.test_cases]
    assert R.build_world(102, "dev", TINY, fake_competence).all_texts() != a.all_texts()


def test_assignment_and_competence_decide_the_truth(world):
    taught = [s for s in skills.SKILLS if world.assignment[s] != "unseen"]
    assert len(taught) == 3 and sum(v == "solid" for v in world.assignment.values()) == 2 and sum(v == "thin" for v in world.assignment.values()) == 1
    for s in skills.SKILLS:
        if world.assignment[s] == "unseen":
            assert world.truth[s] == "unseen" and s not in world.teach and s not in world.cal and s not in world.adapt
        else:
            assert world.truth[s] == ("solid" if world.assignment[s] == "solid" else "thin") and len(world.teach[s]) == (60 if world.assignment[s] == "solid" else 8)
    for c in world.test_cases:
        if c.kind in ("arith", "twin"):
            assert c.status == world.truth[c.skill], c


def test_nothing_taught_is_calibrated_adapted_or_tested_on(world):
    teach = fps(e.text for s in world.teach for e in world.teach[s]) | fps(i.text for i in world.teach_none)
    cal = fps(t for texts in world.cal.values() for t in texts)
    adapt = fps(t for texts in world.adapt.values() for t in texts)
    tests = fps(t for t, c in zip(world.test_texts, world.test_cases) if c.kind in ("arith", "twin", "trap", "other"))
    assert not (teach & cal) and not (teach & adapt) and not (cal & adapt)
    assert not (teach & tests) and not (cal & tests) and not (adapt & tests)


def test_fresh_draws_never_repeat_a_text_that_was_already_used():
    rng_a, rng_b = R._stream(5, 4, 0), R._stream(5, 4, 0)                                    # the same stream twice: the second draw would repeat the first one exactly
    first = R._fresh_taught_wording(rng_a, "dev", "add", 25, set())
    used = {R.fingerprint(e.text) for e in first}
    again = R._fresh_taught_wording(rng_b, "dev", "add", 25, set(used))
    assert not (used & {R.fingerprint(e.text) for e in again}) and len({R.fingerprint(e.text) for e in again}) == 25


def test_teaching_is_the_taught_wordings_and_the_wording_claim_is_tested_on_the_others(world):
    assert {e.family for s in world.teach for e in world.teach[s]} <= set(assignments.TEACH_FAMILIES)
    heldout = {c.family for c in world.test_cases if c.kind == "arith"} - set(assignments.TEACH_FAMILIES)
    assert heldout and heldout <= set(assignments.TEST_FAMILIES)
    taught_wording = [c for c in world.test_cases if c.kind == "arith" and c.family in assignments.TEACH_FAMILIES]
    assert taught_wording and all(c.status != "unseen" or c.skill in [s for s in skills.SKILLS if world.assignment[s] == "unseen"] for c in taught_wording)
    assert {i.family for i in world.teach_none} <= {"other", "trap"}
    assert all(fam in assignments.TEST_FAMILIES for fam in world.adapt_families)


def test_other_content_kinds_have_their_own_truth(world):
    kinds = {c.kind for c in world.test_cases}
    assert {"arith", "twin", "trap", "other", "unseen_content"} <= kinds
    assert all(c.status == "other" for c in world.test_cases if c.kind in ("trap", "other"))
    assert all(c.status == "unseen" for c in world.test_cases if c.kind == "unseen_content")
    traps = [i for i in world.teach_none if i.family == "trap"]
    assert traps and all(int(i.template_ids[0].split(".")[1]) % 2 == 0 for i in traps)               # teaching traps and test traps never share a template


def test_tangled_messages_pair_a_known_skill_with_an_unseen_one_or_other_content(world):
    kinds = {"b4a": 0, "b4b": 0, "b4c": 0}
    for m in world.tangled:
        states = sorted(R.part_status(p, world.truth) for p in m.parts)
        key = {("solid", "unseen"): "b4a", ("thin", "unseen"): "b4a", ("other", "solid"): "b4b", ("other", "thin"): "b4b", ("other", "unseen"): "b4c"}[tuple(states)]
        kinds[key] += 1
    assert all(v == TINY.tangled for v in kinds.values())


def test_tangled_requests_are_not_copies_of_taught_texts(world):
    used = fps(e.text for s in world.teach for e in world.teach[s]) | fps(t for texts in world.cal.values() for t in texts)
    for m in world.tangled:
        for p in m.parts:
            if p.kind == "skill":
                assert R.fingerprint(m.text[p.start:p.end]) not in used


def test_every_text_the_cell_needs_is_listed_once(world):
    texts = world.all_texts()
    assert len(texts) == len(set(texts))
    needed = set(world.test_texts) | set(world.background) | {i.text for i in world.teach_none} | {e.span for s in world.teach for e in world.teach[s]}
    for m in world.tangled:
        for spans in R.segment_sets(m).values():
            needed |= {m.text[a:b] for a, b in spans}
    assert needed <= set(texts)


# ------------------------------------------------------------------------------------------------------------------------------ one cell
def stamp_of(rep_id):
    return R.stamp_for(rep_id)


def test_a_cell_on_the_hash_reader_has_every_section_and_is_resumable(tmp_path):
    rep = reps.get_rep("R0_hash")
    rec = R.run_cell("R0_hash", 100, "dev", tmp_path, stamp_of("R0_hash"), TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    assert rec["code_stamp"] == stamp_of("R0_hash") and rec["rep"] == "R0_hash" and rec["seed"] == 100
    assert set(rec["machine"]) == {"q0.05", "q0.1"} and set(rec["machine"]["q0.05"]) == {"standard", "adapted"}
    s = rec["machine"]["q0.05"]["standard"]
    assert {"macro_f1", "false_unseen_heldout", "unseen_recall", "other_as_unseen", "thin_vs_solid_auroc", "confusion"} <= set(s)
    assert set(rec["baselines"]) == set(R.BL.BASELINE_NAMES)
    assert {"standard", "adapted"} == set(rec["baselines"]["knn_distance"]["q0.05"]) and "auroc_taught_vs_untaught" in rec["baselines"]["knn_distance"]["q0.05"]["standard"]
    assert {"b4a", "b4b", "b4c", "mode"} <= set(rec["tangled"]["machine"]["gold"]) and {"gold", "auto_sentence", "auto_window"} == set(rec["tangled"]["machine"])
    assert set(rec["tangled"]["b6"]) == {"auto_sentence", "auto_window"}
    assert set(rec["groups"]["machine"]) == {"1", "4"} and set(rec["groups"]["baseline"]) == {"1", "4"}
    assert "estimate" in rec["consequence"]["text"].lower() and rec["skill_confusion"]["rows"] == list(skills.SKILLS)
    path = tmp_path / "R0_hash__s100.json"
    assert path.exists() and json.loads(path.read_text(encoding="utf-8"))["seed"] == 100
    mtime = path.stat().st_mtime_ns
    again = R.run_cell("R0_hash", 100, "dev", tmp_path, stamp_of("R0_hash"), TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    assert path.stat().st_mtime_ns == mtime and again["seed"] == 100                                  # finished with the same code: not redone
    R.run_cell("R0_hash", 100, "dev", tmp_path, "another-stamp", TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    assert json.loads(path.read_text(encoding="utf-8"))["code_stamp"] == "another-stamp"            # a different code stamp is redone


def test_a_damaged_cell_file_is_recomputed(tmp_path):
    (tmp_path / "R0_hash__s100.json").write_text('{"seed": 100, "code_st', encoding="utf-8")
    rec = R.run_cell("R0_hash", 100, "dev", tmp_path, stamp_of("R0_hash"), TINY, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None)
    assert rec["machine"]


def test_the_learned_map_sees_only_taught_skills_and_taught_wordings(tmp_path):
    seen = []

    def spy(X, items, taught):
        seen.append(([i.label for i in items], [i.family for i in items], list(taught)))
        return lambda Z: np.asarray(Z, dtype=np.float32)

    w = R.build_world(100, "dev", TINY, fake_competence)
    R.run_cell("R5_R0_hash", 100, "dev", tmp_path, "s", TINY, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None, mapper=spy)
    assert len(seen) == 1
    labels, families, taught = seen[0]
    assert set(labels) <= set(taught) | {"none"} and set(taught) == {s for s in skills.SKILLS if w.assignment[s] != "unseen"}
    assert set(families) <= set(assignments.TEACH_FAMILIES) | {"other", "trap"}
    assert max(sum(1 for l in labels if l == s) for s in taught) <= TINY.map_cap


def test_the_status_machine_is_built_only_from_skill_groups_and_held_out_calibration(tmp_path, monkeypatch):
    built = []
    real = R.ST.StatusMachine

    class Spy(real):
        def __init__(self, groups, other, cal_vectors, cal_groups, **kw):
            built.append((dict(groups), other, np.asarray(cal_groups), kw))
            super().__init__(groups, other, cal_vectors, cal_groups, **kw)

    monkeypatch.setattr(R.ST, "StatusMachine", Spy)
    R.run_cell("R0_hash", 100, "dev", tmp_path, "s", TINY, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None)
    assert built
    w = R.build_world(100, "dev", TINY, fake_competence)
    taught = {s for s in skills.SKILLS if w.assignment[s] != "unseen"}
    n_std = sum(len(v) for v in w.cal.values())
    n_ada = n_std + sum(len(v) for v in w.adapt.values())
    assert n_ada > n_std
    by_q = {}
    for _, _, cg, kw in built:
        by_q.setdefault(kw["q"], []).append(len(cg))
    assert set(by_q) == set(TINY.qs)
    for q in TINY.qs:
        assert by_q[q][:2] == [n_std, n_ada], (q, by_q[q])           # per level: the standard machine (no S4 to S8 part in its calibration) first, then the adapted one
    for groups, other, cal_groups, kw in built:
        assert set(groups) == taught and other.name == "other" and set(cal_groups.tolist()) == taught | {"other"}
        assert {g.name: g.n_examples for g in groups.values()} == {s: len(w.teach[s]) for s in taught}
        assert all(g.solver is None for g in groups.values())


def test_the_baselines_are_fitted_on_the_teaching_parts_only_and_calibrated_on_held_out_ones(tmp_path, monkeypatch):
    fitted, scored = [], []
    real = R.BL.make_baseline

    def spy(name, background=None, **kw):
        det = real(name, background=background, **kw)
        orig_fit, orig_score = det.fit, det.novelty_score

        def fit(X, y):
            fitted.append((name, len(X), sorted(set(np.asarray(y).tolist()))))
            return orig_fit(X, y)

        def score(X):
            scored.append((name, len(X)))
            return orig_score(X)

        det.fit, det.novelty_score = fit, score
        return det

    monkeypatch.setattr(R.BL, "make_baseline", spy)
    R.run_cell("R0_hash", 100, "dev", tmp_path, "s", TINY, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None)
    w = R.build_world(100, "dev", TINY, fake_competence)
    n_teach = sum(len(v) for v in w.teach.values()) + len(w.teach_none)
    assert len(fitted) == len(R.BL.BASELINE_NAMES) and all(n == n_teach for _, n, _ in fitted)
    assert all(classes == sorted([s for s in skills.SKILLS if w.assignment[s] != "unseen"] + ["other"]) for _, _, classes in fitted)
    n_cal = sum(len(v) for v in w.cal.values())
    assert n_cal in {n for _, n in scored} and n_cal + sum(len(v) for v in w.adapt.values()) in {n for _, n in scored}


def test_each_baseline_threshold_comes_from_the_matching_calibration_set(tmp_path, monkeypatch):
    calls = []
    real = R.BL.conformal_threshold

    def spy(scores, q):
        calls.append((len(scores), q))
        return real(scores, q)

    monkeypatch.setattr(R.BL, "conformal_threshold", spy)
    R.run_cell("R0_hash", 100, "dev", tmp_path, "s", TINY, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None)
    w = R.build_world(100, "dev", TINY, fake_competence)
    n_cal = sum(len(v) for v in w.cal.values())
    n_ada = n_cal + sum(len(v) for v in w.adapt.values())
    for q in TINY.qs:
        sizes = [n for n, qq in calls if qq == q]
        assert sizes.count(n_cal) == len(R.BL.BASELINE_NAMES) and sizes.count(n_ada) == len(R.BL.BASELINE_NAMES), (q, sizes)


# ------------------------------------------------------------------------------------------------------------------------------ the report
def test_the_report_marks_bars_and_keeps_the_adapted_condition_apart(tmp_path):
    rep = reps.get_rep("R0_hash")
    for seed in (100, 101):
        R.run_cell("R0_hash", seed, "dev", tmp_path, stamp_of("R0_hash"), TINY, rep=rep, competence_fn=fake_competence, cache_dir=None)
    text = R.report_e2(tmp_path, ["R0_hash"], [100, 101], stamp_of, "dev", sizes=TINY)
    assert "E2" in text and "R0_hash" in text and "B3a" in text
    assert ("PASS" in text) or ("fail" in text)
    assert "adapted" in text.lower() and "not gated" in text.lower()
    assert "estimate" in text.lower()
    assert "draft bar" in text.lower() and "dev-fixed" in text.lower()
    assert "0.85" in text                                                  # a draft bar is shown next to the dev-fixed one
    missing = R.report_e2(tmp_path, ["R0_hash", "R5_R0_hash"], [100, 101], stamp_of, "dev", sizes=TINY)
    assert "R5_R0_hash" in missing and "missing" in missing.lower()


# ------------------------------------------------------------------------------------------------------------------------------ the reader and the oracle
def test_centring_uses_the_teaching_vectors_and_not_the_other_texts():
    w = R.build_world(100, "dev", TINY, fake_competence)
    texts = w.all_texts()
    index = {t: k for k, t in enumerate(texts)}
    rng = np.random.default_rng(0)
    teach = {e.span for s in w.teach for e in w.teach[s]} | {i.text for i in w.teach_none}
    offset_teach, offset_rest = 5.0 * np.eye(16)[0], -5.0 * np.eye(16)[1]
    X = np.array([(offset_teach if t in teach else offset_rest) + 0.3 * rng.standard_normal(16) for t in texts], dtype=np.float32)
    P = R._reader(w, X, index, 100, False, None)
    in_teach = np.array([t in teach for t in texts])
    assert np.linalg.norm(P[in_teach].mean(axis=0)) < 0.35                  # the teaching vectors are centred on their own mean ...
    assert np.linalg.norm(P[~in_teach].mean(axis=0)) > 0.9                  # ... so everything else lies on the other side of it
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-5)


def test_cell_formatting_marks_the_right_side_of_a_bar():
    assert R._cell([0.9, 0.95], (0.85, "min")).endswith("PASS") and R._cell([0.8], (0.85, "min")).endswith("fail")
    assert R._cell([0.04], (0.05, "max")).endswith("PASS") and R._cell([0.2], (0.05, "max")).endswith("fail")
    assert R._cell([], (0.05, "max")) == "n/a"


class OracleRep(reps.Representation):
    """Knows the true group of every text of the world (a skill, Other content or never-taught content) and puts it near its own axis, with noise that depends only on the text."""

    def __init__(self, world):
        super().__init__("oracle", 24)
        groups = {}
        for s in world.teach:
            groups.update({e.span: s for e in world.teach[s]})
        groups.update({i.text: "other" for i in world.teach_none})
        for name, texts in list(world.cal.items()) + list(world.adapt.items()):
            groups.update({t: name for t in texts})
        for t, c in zip(world.test_texts, world.test_cases):
            groups[t] = c.skill if c.skill else ("never" if c.kind == "unseen_content" else "other")
        groups.update({t: "other" for t in world.background})
        self.groups, self.axes = groups, {n: k for k, n in enumerate(list(skills.SKILLS) + ["other", "never"])}
        self.tangled = [(m.text[p.start:p.end], p.skill if p.kind == "skill" else "other") for m in world.tangled for p in m.parts]

    def group_of(self, text):
        if text in self.groups:
            return self.groups[text]
        for part, g in self.tangled:
            if text in part or part in text:
                return g
        return "other"

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim))
        for k, t in enumerate(texts):
            noise = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).standard_normal(self.dim)
            out[k] = 2.0 * np.eye(self.dim)[self.axes[self.group_of(t)]] + 0.35 * noise
        return out


def test_an_oracle_reader_gives_the_statuses_the_assignment_says(tmp_path):
    w = R.build_world(100, "dev", TINY, fake_competence)
    rec = R.run_cell("oracle", 100, "dev", tmp_path, "s", TINY, rep=OracleRep(w), competence_fn=fake_competence, cache_dir=None)
    s = rec["machine"]["q0.05"]["standard"]
    assert s["macro_f1"]["macro_f1"] >= 0.8 and s["false_unseen_heldout"] <= 0.25 and s["unseen_recall"] >= 0.9 and s["unseen_content_flagged"] >= 0.6
    assert s["other_as_unseen"] <= 0.25 and s["other_labelled"] >= 0.7 and s["thin_vs_solid_auroc"] >= 0.8
    assert rec["machine"]["q0.05"]["adapted"]["false_unseen_heldout"] <= 0.25
    t = rec["tangled"]["machine"]["gold"]
    assert t["b4a"]["residual_hit"] >= 0.8 and t["b4a"]["known_not_flagged"] >= 0.8 and t["b4b"]["other_labelled"] >= 0.8 and t["b4c"]["unseen_found"] >= 0.8
    assert rec["groups"]["machine"]["4"]["unseen_recall"] >= 0.9 and rec["groups"]["machine"]["4"]["false_unseen"] <= 0.5
    for name, b in rec["baselines"].items():
        assert b["q0.05"]["standard"]["auroc_taught_vs_untaught"] >= 0.75, name               # every method ranks untaught parts as newer (the tiny world is hard for the autoencoder: 8 examples)
        assert 0.0 <= b["q0.05"]["standard"]["unseen_recall"] <= 1.0 and 0.0 <= b["q0.05"]["standard"]["false_unseen_heldout"] <= 1.0
    # the skills a part is nearest to are its own: a diagonal confusion for the taught skills
    mat, cols = np.array(rec["skill_confusion"]["matrix"]), rec["skill_confusion"]["cols"]
    for k, name in enumerate(rec["skill_confusion"]["rows"]):
        if name in rec["n_teach"]:
            assert mat[k].argmax() == cols.index(name)

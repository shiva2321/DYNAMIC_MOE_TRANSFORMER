import json
import zlib

import numpy as np
import pytest

import assignments
import ceiling as C
import reps
import run_e1
import skills

SMALL = C.Sizes(teach=30, test=20, pairs=40, assignments=4)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_pair_auroc_is_one_for_separated_and_half_for_random():
    rng = np.random.default_rng(0)
    centres = unit(rng.standard_normal((2, 16)))
    X = unit(np.vstack([centres[0] + 0.05 * rng.standard_normal((20, 16)), centres[1] + 0.05 * rng.standard_normal((20, 16))]))
    same = [(i, j) for i in range(0, 20, 2) for j in range(i + 1, 20, 3)]
    other = [(i, 20 + j) for i in range(0, 20, 2) for j in range(0, 20, 3)]
    assert C.pair_auroc(X, same, other) == pytest.approx(1.0)
    assert C.pair_auroc(X, other, same) == pytest.approx(0.0)
    R = unit(rng.standard_normal((400, 32)))
    idx = rng.integers(0, 400, size=(4000, 2))
    assert abs(C.pair_auroc(R, [tuple(p) for p in idx[:2000]], [tuple(p) for p in idx[2000:]]) - 0.5) < 0.05


def test_probe_accuracy_matches_chance_on_noise():
    rng = np.random.default_rng(1)
    accs = [C.probe_accuracy(rng.standard_normal((700, 32)), rng.integers(0, 7, 700), rng.standard_normal((700, 32)), rng.integers(0, 7, 700)) for _ in range(5)]
    assert abs(np.mean(accs) - 1 / 7) < 0.05
    y = np.repeat(np.arange(7), 40)
    means = 4.0 * rng.standard_normal((7, 32))
    sep = lambda: means[y] + rng.standard_normal((280, 32))
    assert C.probe_accuracy(sep(), y, sep(), y) > 0.95
    groups = np.tile(np.array(["a", "b", "c", "d"]), 70)                                   # the leave-one-group-out choice of regularisation also runs
    assert C.probe_accuracy(sep(), y, sep(), y, groups=groups) > 0.95


def test_taught_auroc_is_one_when_untaught_parts_are_far():
    rng = np.random.default_rng(2)
    basis = np.eye(8, dtype=np.float32)
    protos = basis[:2]
    taught = unit(basis[rng.integers(0, 2, 60)] + 0.05 * rng.standard_normal((60, 8)))
    untaught = unit(basis[rng.integers(4, 6, 60)] + 0.05 * rng.standard_normal((60, 8)))
    X = np.vstack([taught, untaught])
    is_taught = np.array([True] * 60 + [False] * 60)
    assert C.taught_auroc(protos, X, is_taught) == pytest.approx(1.0)
    assert C.taught_auroc(protos, X, ~is_taught) == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------------------------------------------ the test bed
def test_items_are_labelled_disjoint_and_aligned():
    items = C.build_items(100, "dev", SMALL)
    again = C.build_items(100, "dev", SMALL)
    assert [i.text for i in items.all] == [i.text for i in again.all]                    # the same seed gives the same test bed
    assert [i.text for i in items.all] != [i.text for i in C.build_items(101, "dev", SMALL).all]
    teach, test = items.role("teach"), items.role("test")
    for lab in list(skills.SKILLS) + ["none"]:
        assert sum(i.label == lab for i in teach) == SMALL.teach and sum(i.label == lab for i in test) == SMALL.test
    assert {i.family for i in teach if i.label != "none"} <= set(assignments.TEACH_FAMILIES)
    assert {i.family for i in test if i.label != "none"} <= set(assignments.TEST_FAMILIES)
    assert {i.family for i in test if i.label == "none"} == {"other", "trap", "unseen"}
    assert not ({i.text for i in teach} & {i.text for i in test})
    for i in teach + test:
        assert i.span and i.span in i.text
    anchors, twins, partners = items.role("anchor"), items.role("twin"), items.role("partner")
    assert len(anchors) == len(twins) == len(partners) == SMALL.pairs
    for a, t, p in zip(anchors, twins, partners):
        assert t.label != a.label and p.label == a.label and p.family != a.family       # a twin changes the skill only; a partner rewords the same skill
        assert t.template_ids[0] == a.template_ids[0] and p.template_ids[0] != a.template_ids[0]
        assert a.operands == t.operands == p.operands and a.label != "max"


class Stub(reps.Representation):
    """A fake representation: each text maps to one fixed column (plus a little deterministic noise so no two texts coincide)."""

    def __init__(self, table, dim):
        super().__init__("stub", dim, tag="stub")
        self.table = table

    def _encode(self, texts):
        z = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            z[i] = np.random.default_rng(zlib.crc32(t.encode("utf-8"))).normal(0, 0.01, self.dim)
            z[i, self.table[t]] += 1.0
        return z


def stub_for(items, key):
    classes = {name: k for k, name in enumerate(sorted({key(i) for i in items.all}))}
    table = {}
    for i in items.all:
        table[i.text] = table[i.span] = classes[key(i)]
    return Stub(table, len(classes))


def test_run_ceiling_on_a_fake_oracle_representation():
    items = C.build_items(100, "dev", SMALL)
    # a representation that encodes the skill perfectly: every measure must see it (the spans of a twin are not needed: B1b uses whole messages)
    res = C.run_ceiling("oracle", 100, "dev", sizes=SMALL, rep=stub_for(items, lambda i: i.label))
    assert res["b1a"] == pytest.approx(1.0) and res["b1a_span"] == pytest.approx(1.0)
    assert res["b1b"] > 0.99 and res["b1c"] > 0.99 and res["b1c_raw"] > 0.99
    assert res["chance"] == {"b1a": pytest.approx(1 / 7), "b1b": 0.5, "b1c": 0.5}
    # a representation that encodes only the wrapper (the frame): the test bed must not leak the skill through anything else
    res = C.run_ceiling("wrapper_only", 100, "dev", sizes=SMALL, rep=stub_for(items, lambda i: i.template_ids[0]))
    assert abs(res["b1a"] - 1 / 7) < 0.10 and abs(res["b1c"] - 0.5) < 0.15
    assert res["b1b"] < 0.25                                                              # a twin shares the wrapper, its reworded partner does not: a word-matcher fails B1b


def test_resume_skips_complete_cells_and_ignores_truncated_files(tmp_path, monkeypatch):
    calls = []
    original = run_e1.ceiling.run_ceiling

    def counting(rep_id, seed, *a, **k):
        calls.append((rep_id, seed))
        return original(rep_id, seed, *a, **k)

    monkeypatch.setattr(run_e1.ceiling, "run_ceiling", counting)
    stamp = "teststamp"
    for seed in (100, 101):
        run_e1.run_cell("R0_hash", seed, "dev", tmp_path, stamp, sizes=SMALL)
    assert calls == [("R0_hash", 100), ("R0_hash", 101)]
    run_e1.run_cell("R0_hash", 100, "dev", tmp_path, stamp, sizes=SMALL)                  # a finished cell is not run again
    assert len(calls) == 2
    bad = tmp_path / "R0_hash__s101.json"
    bad.write_text(bad.read_text()[: len(bad.read_text()) // 2])                          # a power cut in the middle of a write
    run_e1.run_cell("R0_hash", 101, "dev", tmp_path, stamp, sizes=SMALL)
    assert calls[2:] == [("R0_hash", 101)]
    assert json.loads(bad.read_text())["code_stamp"] == stamp
    run_e1.run_cell("R0_hash", 101, "dev", tmp_path, "another-code-version", sizes=SMALL)  # results made by other code are not trusted either
    assert len(calls) == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_b2_validity_rule_needs_r0_to_fail_by_at_least_015():
    ok, _ = run_e1.b2_validity({"b1a": 0.60, "b1b": 0.97})
    assert ok                                                                             # fails B1a by 0.35
    ok, _ = run_e1.b2_validity({"b1a": 0.96, "b1b": 0.96})
    assert not ok                                                                         # R0 passes both: the test bed cannot tell a word-matcher from a reader
    assert run_e1.b2_validity({"b1a": 0.96, "b1b": 0.80})[0]                              # exactly 0.15 below the bar counts
    assert not run_e1.b2_validity({"b1a": 0.96, "b1b": 0.81})[0]


# ------------------------------------------------------------------------------------------------------------------------------ the hook for a learned map (Task 9)
class Spy:
    """A mapper that does nothing but remembers what it was asked to learn from."""

    def __init__(self):
        self.calls = []

    def __call__(self, X, items, taught):
        self.calls.append((X.shape, list(items), tuple(taught)))
        return lambda Z: Z


def test_an_identity_mapper_changes_no_number():
    items = C.build_items(100, "dev", SMALL)
    stub = stub_for(items, lambda i: i.label)
    plain = C.run_ceiling("oracle", 100, "dev", sizes=SMALL, rep=stub)
    same = C.run_ceiling("oracle", 100, "dev", sizes=SMALL, rep=stub, mapper=Spy())
    for k in ("b1a", "b1a_span", "b1b", "b1b_raw", "b1c", "b1c_raw", "b1c_min", "b1b_taught", "b1b_story"):
        assert same[k] == pytest.approx(plain[k]), k


def test_the_mapper_only_sees_taught_skills_taught_wordings_and_no_evaluation_text():
    items = C.build_items(100, "dev", SMALL)
    spy = Spy()
    C.run_ceiling("oracle", 100, "dev", sizes=SMALL, rep=stub_for(items, lambda i: i.label), mapper=spy)
    assert len(spy.calls) == 1 + SMALL.assignments                                        # one map for B1a and B1b, one per teaching assignment for B1c
    shape, train, taught = spy.calls[0]
    assert set(taught) == set(skills.SKILLS) and shape[0] == len(train)
    evaluation_texts = {i.text for i in items.all if i.role in ("anchor", "twin", "partner", "test")}
    for _, train, taught in spy.calls:
        assert {i.role for i in train} == {"teach"} and not ({i.text for i in train} & evaluation_texts)
        assert {i.label for i in train} <= set(taught) | {"none"}                         # an untaught skill never reaches the map
        assert {i.family for i in train} <= set(assignments.TEACH_FAMILIES) | {"other", "trap"}
    sizes_taught = {len(t) for _, _, t in spy.calls[1:]}
    assert sizes_taught == {3}                                                            # the default assignment: two Solid and one Thin


def test_b1b_is_also_reported_for_partners_in_taught_wordings_and_for_stories():
    e = np.eye(4)
    # six triples: taught-wording partners that sit on the anchor and twins that do not, then story partners that do not and twins that sit on the anchor
    rows = [e[0], e[0], e[2]] * 3 + [e[0], e[1], e[0]] * 3                                 # anchor, partner, twin for each triple
    X = np.array(rows)
    n = 6
    anchors, partners, twins = X[0::3], X[1::3], X[2::3]
    Xb = np.vstack([anchors, partners, twins])
    same, other = [(k, n + k) for k in range(n)], [(k, 2 * n + k) for k in range(n)]
    res = C.b1b_by_partner_family(Xb, same, other, ["S3", "S1", "S2", "S4", "S4", "S4"])
    assert res == {"b1b_taught": pytest.approx(1.0), "b1b_story": pytest.approx(0.0)}
    assert np.isnan(C.b1b_by_partner_family(Xb, same, other, ["S1"] * 6)["b1b_story"])     # no story partner at all: not a number, not a crash


def test_run_ceiling_reports_the_split_b1b_and_it_is_one_for_an_oracle():
    items = C.build_items(100, "dev", SMALL)
    res = C.run_ceiling("oracle", 100, "dev", sizes=SMALL, rep=stub_for(items, lambda i: i.label))
    assert res["b1b_taught"] > 0.99 and res["b1b_story"] > 0.99

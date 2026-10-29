"""The asking reader (routing_lab/ask_reader.py): the decision on the asked probabilities, the cache, the plain naming rule."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_lab"))

import ask_reader as AR  # noqa: E402
import lab_ask as LA  # noqa: E402


def row(**kw):
    """A probability row over OPS: the named options get their mass, the rest share what is left."""
    p = np.zeros(len(AR.OPS))
    for k, v in kw.items():
        p[AR.OPS.index(k)] = v
    rest = [i for i in range(len(AR.OPS)) if AR.OPS[i] not in kw]
    p[rest] = (1.0 - p.sum()) / len(rest)
    return p


def machine(q=0.05, n_cal=40):
    rng = np.random.default_rng(0)
    cal = {"add": np.stack([row(add=0.9 + 0.09 * rng.random()) for _ in range(n_cal)]), "sub": np.stack([row(sub=0.9 + 0.09 * rng.random()) for _ in range(n_cal)]),
           "none": np.stack([row(none=0.9 + 0.09 * rng.random()) for _ in range(n_cal)])}
    return AR.AskMachine(cal, {"add": "solid", "sub": "thin"}, {"add": 0.95, "sub": 0.4}, {"add": 2000, "sub": 20}, q=q)


def test_softmax_is_a_distribution_and_stable():
    p = AR.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 1 and np.isfinite(p).all()


def test_a_class_covers_a_part_through_its_own_probability_only():
    m = machine()
    cov = m.covering(np.stack([row(add=0.97), row(sub=0.97), row(none=0.97), row(add=0.5, sub=0.45)]))
    assert cov[0].tolist() == [True, False, False] and cov[1].tolist() == [False, True, False] and cov[2].tolist() == [False, False, True]
    assert not cov[3].any()                                                                    # torn between two classes, neither is confident enough: nothing covers it


def test_a_confidently_named_taught_skill_is_covered_and_gets_its_state():
    m = machine()
    a = m.assess_many(np.stack([row(add=0.95), row(sub=0.95)]))
    assert (a[0].status, a[0].skill, a[0].evidence, a[0].competence) == ("solid", "add", 2000, 0.95)
    assert (a[1].status, a[1].skill, a[1].evidence) == ("thin", "sub", 20)


def test_an_operation_nobody_taught_is_unseen_even_when_the_model_is_sure():
    a = machine().assess_many(np.stack([row(mul=0.97)]))[0]
    assert a.status == "unseen" and a.skill is None and math.isnan(a.competence) and a.nearest in ("add", "sub", "none")


def test_none_is_other_content_and_is_never_unseen_arithmetic():
    a = machine().assess_many(np.stack([row(none=0.96)]))[0]
    assert a.status == "other" and a.skill is None


def test_an_unsure_model_is_unseen_because_no_known_class_covers_the_part():
    a = machine().assess_many(np.stack([np.full(len(AR.OPS), 1.0 / len(AR.OPS))]))[0]
    assert a.status == "unseen"


def test_the_decision_does_not_depend_on_what_else_is_assessed():
    m = machine()
    P = np.stack([row(add=0.95), row(mul=0.9), row(sub=0.6, add=0.3)])
    together = m.assess_many(P)
    alone = [m.assess_many(P[i:i + 1])[0] for i in range(3)]
    assert together == alone


def test_a_smaller_q_flags_fewer_parts():
    P = np.stack([row(add=0.80), row(add=0.90), row(sub=0.85)])
    loose = sum(a.status == "unseen" for a in machine(q=0.30).assess_many(P))
    strict = sum(a.status == "unseen" for a in machine(q=0.02).assess_many(P))
    assert strict <= loose


def test_the_machine_refuses_a_missing_calibration_class_and_a_bad_q():
    with pytest.raises(ValueError, match="no calibration parts"):
        AR.AskMachine({"add": np.stack([row(add=0.9)]), "none": np.stack([row(none=0.9)])}, {"add": "solid", "sub": "thin"}, {"add": 1, "sub": 1}, {"add": 1, "sub": 1})
    with pytest.raises(ValueError, match="q must lie"):
        machine(q=1.5)


def test_probabilities_cache_the_answers_and_ask_only_for_new_texts(tmp_path):
    calls = []

    def fake(texts):
        calls.append(list(texts))
        return np.stack([row(add=0.8) for _ in texts])

    path = tmp_path / "c" / "probs.json"
    a = AR.probabilities(["x", "y", "x"], path, fake)
    assert a.shape == (3, 7) and calls == [["x", "y"]]
    b = AR.probabilities(["y", "z"], path, fake)
    assert calls[-1] == ["z"] and b.shape == (2, 7)
    assert AR.probabilities(["x", "z"], path, fake).shape == (2, 7) and len(calls) == 2
    assert len(json.loads(path.read_text())) == 3


def test_probabilities_refuse_a_reader_that_returns_the_wrong_shape(tmp_path):
    with pytest.raises(RuntimeError, match="the reader returned"):
        AR.probabilities(["x"], tmp_path / "p.json", lambda ts: np.zeros((len(ts), 3)))


def test_top1_accuracy_ignores_rows_without_a_truth():
    P = np.stack([row(add=0.9), row(sub=0.9), row(mul=0.9)])
    assert AR.top1_accuracy(P, ["add", "add", None]) == pytest.approx(0.5)
    assert math.isnan(AR.top1_accuracy(P, [None, None, None]))


def test_the_plain_naming_rule_calls_an_untaught_top_option_unseen():
    P = np.stack([row(add=0.9), row(mul=0.9), row(none=0.9)])
    assert LA.naming_rule(P, ["add", "sub"]) == ["known", "unseen", "known"]


def test_the_prompt_names_the_message_and_all_seven_options():
    class Tok:
        def __call__(self, text, add_special_tokens=False, truncation=True, max_length=10):
            return {"input_ids": list(range(min(len(text.split()), max_length)))}

        def decode(self, ids):
            return " ".join(f"w{i}" for i in ids)

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

    out = AR.build_prompt(Tok(), "a b c")
    assert "Message: w0 w1 w2" in out and all(f"{ch}. " in out for ch in AR.LETTERS) and out.endswith("Answer with one letter.")


def test_letters_must_be_single_distinct_tokens():
    class Good:
        def __call__(self, ch, add_special_tokens=False):
            return {"input_ids": [ord(ch)]}

    assert AR.letter_ids(Good()) == [ord(c) for c in AR.LETTERS]

    class Multi:
        def __call__(self, ch, add_special_tokens=False):
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="single token"):
        AR.letter_ids(Multi())


def test_the_covering_set_is_empty_for_a_novel_part_single_for_a_clear_one_and_matches_the_assessment():
    m = machine()
    P = np.stack([row(add=0.97), row(mul=0.97), np.full(len(AR.OPS), 1.0 / len(AR.OPS))])
    cov = m.covering(P)
    assert cov.shape == (3, 3) and m.known == ["add", "sub", "none"]
    assert cov[0].tolist() == [True, False, False] and not cov[1].any() and not cov[2].any()
    assert [a.status == "unseen" for a in m.assess_many(P)] == [not c.any() for c in cov]


def test_set_stats_counts_coverage_empty_single_and_multi():
    cov = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=bool)
    s = LA.set_stats(cov, [0, 1, 2, 3], [0, 0, 1, 0])
    assert s["n"] == 4 and s["coverage"] == pytest.approx(0.5) and s["empty"] == pytest.approx(0.25) and s["single"] == pytest.approx(0.5) and s["multi"] == pytest.approx(0.25)
    assert s["mean_size_nonempty"] == pytest.approx(4 / 3)
    never = LA.set_stats(cov, [2], [None])
    assert never["empty"] == 1.0 and math.isnan(never["coverage"]) and math.isnan(never["mean_size_nonempty"])
    assert LA.set_stats(cov, [], [])["n"] == 0

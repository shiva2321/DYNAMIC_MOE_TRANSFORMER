"""The card reader (routing_general/reader.py): prompt building and the cache, with a stand-in for the language model."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
sys.path.insert(0, str(_KS / "routing_general"))

import cards as CD  # noqa: E402
import reader as RD  # noqa: E402

CARDS = [CD.Card("addition", ("17 plus 5",)), CD.Card("card arrival", ("when will my card get here",))]


class FakeTok:
    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=200):
        return {"input_ids": text.split()[:max_length]}

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert len(msgs) == 1 and msgs[0]["role"] == "user" and not tokenize and add_generation_prompt
        return "<user>" + msgs[0]["content"] + "<assistant>"


def test_chat_prompt_wraps_the_question_and_cuts_a_long_message_by_tokens():
    p = RD.build_chat(FakeTok(), "when does it come", CARDS, with_exemplars=False)
    assert p.startswith("<user>") and p.endswith("<assistant>")
    assert "when does it come" in p and "A. addition" in p and "C. none of these" in p
    long = " ".join(f"w{i}" for i in range(RD.MAX_TEXT_TOKENS + 50))
    q = RD.build_chat(FakeTok(), long, CARDS, with_exemplars=False)
    assert f"w{RD.MAX_TEXT_TOKENS - 1}" in q and f"w{RD.MAX_TEXT_TOKENS}" not in q
    assert '(examples: "17 plus 5")' in RD.build_chat(FakeTok(), "x", CARDS, with_exemplars=True)


class Counting:
    def __init__(self, k):
        self.k, self.calls = k, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = np.full((len(texts), self.k + 1), 0.0)
        for i, t in enumerate(texts):
            out[i, len(t) % (self.k + 1)] = 1.0
        return out


def test_answers_are_cached_per_question_and_only_new_texts_are_asked(tmp_path):
    ask = Counting(2)
    r = RD.CardReader(CARDS, with_exemplars=False, cache_dir=tmp_path, ask=ask)
    P = r.probabilities(["aa", "bbb", "aa"])
    assert P.shape == (3, 3) and np.allclose(P.sum(axis=1), 1.0)
    assert ask.calls == [["aa", "bbb"]]                                                  # a repeated text is asked once
    r.probabilities(["aa", "cccc"])
    assert ask.calls[-1] == ["cccc"]
    again = RD.CardReader(CARDS, with_exemplars=False, cache_dir=tmp_path, ask=Counting(2))
    assert np.allclose(again.probabilities(["aa"]), P[:1])                               # a new reader finds the file
    ask2 = Counting(2)
    RD.CardReader(CARDS, with_exemplars=True, cache_dir=tmp_path, ask=ask2).probabilities(["aa"])
    assert ask2.calls == [["aa"]]                                                        # another question never reads this question's answers
    ask3 = Counting(2)
    RD.CardReader(CARDS[::-1], with_exemplars=False, cache_dir=tmp_path, ask=ask3).probabilities(["aa"])
    assert ask3.calls == [["aa"]]


def test_empty_input_and_a_reader_of_the_wrong_width_are_handled_loudly(tmp_path):
    r = RD.CardReader(CARDS, with_exemplars=False, cache_dir=tmp_path, ask=Counting(2))
    assert r.probabilities([]).shape == (0, 3)
    bad = RD.CardReader(CARDS, with_exemplars=False, cache_dir=tmp_path / "bad", ask=Counting(5))
    with pytest.raises(RuntimeError):
        bad.probabilities(["x"])

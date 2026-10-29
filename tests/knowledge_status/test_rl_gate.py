"""The boundary question (routing_lab/lab_gate.py, ask_reader gate parts)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_lab"))

import ask_reader as AR  # noqa: E402
import lab_gate as LG  # noqa: E402
import status as ST  # noqa: E402
from test_rl_ask import machine, row  # noqa: E402


def test_the_threshold_leaves_at_most_alpha_of_the_calibration_requests_below_it():
    v = np.linspace(0.5, 1.0, 199)
    th = LG.gate_threshold(v, 0.05)
    assert (v < th).mean() <= 0.05 and (v <= th).mean() >= 0.04
    assert LG.gate_threshold(v, 0.01) <= th


def test_p_yes_is_the_share_of_yes_and_the_gate_cache_has_width_two(tmp_path):
    assert AR.p_yes(np.array([[0.6, 0.2], [0.1, 0.3]])).tolist() == pytest.approx([0.75, 0.25])
    calls = []
    out = AR.gate_probabilities(["a", "b"], tmp_path / "g.json", lambda ts: calls.append(ts) or np.array([[0.9, 0.1]] * len(ts)))
    assert out.shape == (2, 2) and calls == [["a", "b"]]
    assert AR.gate_probabilities(["a"], tmp_path / "g.json", lambda ts: 1 / 0).shape == (1, 2)


def test_two_stage_turns_low_yes_parts_into_other_and_leaves_the_rest_to_the_rule():
    m = machine()
    P = np.stack([row(mul=0.97), row(add=0.97), row(mul=0.97)])
    out = LG.two_stage(P, np.array([0.99, 0.99, 0.02]), 0.5, ["add", "sub"], {"add": "solid", "sub": "thin"}, {"add": 0.9, "sub": 0.3}, {"add": 2000, "sub": 20}, m)
    assert [a.status for a in out] == ["unseen", "solid", "other"]


def test_the_gate_prompt_names_the_message():
    class Tok:
        def __call__(self, text, add_special_tokens=False, truncation=True, max_length=10):
            return {"input_ids": list(range(min(len(text.split()), max_length)))}

        def decode(self, ids):
            return " ".join(f"w{i}" for i in ids)

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

    out = AR.build_gate_prompt(Tok(), "a b")
    assert "Message: w0 w1" in out and out.endswith("Answer with Yes or No.")

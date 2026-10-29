"""A reader that ASKS instead of comparing: the language model is shown the message and a multiple-choice question ("which operation does it ask for?"), and the probabilities of the answer letters
are the reader's output. Wording is handled by the model's own language understanding, not by a distance in an embedding space.

A message gets a probability for each of the six operations and for "none". The status of a part is decided as the look-based machine decides it (class-conditional conformal p-values,
calibration on held-out parts of the taught skills in wordings S1 to S3 only, level q): the conformity of a part to a known class (a taught skill, or "none") is the probability the model gives
that class, and its p-value is the rank of that probability among the held-out parts OF THAT CLASS (small p = novel, as in status.conformal_p). The classes whose p-value is above q form the
COVERING SET of the part: empty = nothing taught explains it (unseen), one class = a confident route, several = ambiguous or a part that needs more than one expert.

Declared limits (they are the point of the lab, not a footnote): the reader relies on the model's prior knowledge of the operation vocabulary, so it can name a skill nobody taught here (that is what
makes "never-taught" detectable) but it can NOT recognise a skill outside that vocabulary; and "none" lumps every non-arithmetic content together, so unseen content types are not told apart from
taught Other content by this reader (the look-based machine keeps that job).
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_KS = Path(__file__).resolve().parents[1]
for _p in (str(_KS.parents[1]), str(_KS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import runlib  # noqa: E402
import status as ST  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OPS = ("add", "sub", "mul", "div", "mod", "max", "none")          # option letters A to G, in this order
LETTERS = "ABCDEFG"
BATCH = 16
MAX_TEXT_TOKENS = 200

QUESTION = ("Read the message below and say which arithmetic operation it asks for.\n\n"
            "Message: {text}\n\n"
            "Options:\nA. addition\nB. subtraction\nC. multiplication\nD. division (the whole-number quotient)\nE. the remainder after a division\n"
            "F. the larger of two numbers\nG. none of these (not an arithmetic request)\n\n"
            "Answer with one letter.")


def build_prompt(tok, text: str) -> str:
    """The chat-formatted prompt for one message (the model's own template, generation prompt added). Long messages are cut by tokens, not characters."""
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_TEXT_TOKENS)["input_ids"]
    cut = tok.decode(ids)
    return tok.apply_chat_template([{"role": "user", "content": QUESTION.format(text=cut)}], tokenize=False, add_generation_prompt=True)


def letter_ids(tok) -> List[int]:
    ids = []
    for ch in LETTERS:
        t = tok(ch, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise RuntimeError(f"option letter {ch!r} is not a single token for this tokenizer: {t}")
        ids.append(t[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError("option letters share a token")
    return ids


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _ask_model(texts: Sequence[str], gpu: bool = True) -> np.ndarray:
    """(n, 7) probabilities over the option letters; one forward pass per message, no generation."""
    import torch
    import reps
    tok, model = reps._hf_model(MODEL, "AutoModelForCausalLM", gpu, gpu)
    ids_l = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    out = []
    for s in range(0, len(texts), BATCH):
        enc = [tok(build_prompt(tok, t), add_special_tokens=False)["input_ids"] for t in texts[s:s + BATCH]]
        n = max(len(x) for x in enc)
        batch = torch.tensor([x + [pad] * (n - len(x)) for x in enc], device=model.device)
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in enc], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=batch, attention_mask=mask).logits
        last = torch.tensor([len(x) - 1 for x in enc], device=logits.device)
        sel = logits[torch.arange(len(enc), device=logits.device), last][:, ids_l].float().cpu().numpy()
        out.append(softmax(sel))
    return np.concatenate(out) if out else np.zeros((0, len(OPS)))


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def probabilities(texts: Sequence[str], cache_path: Path, ask=_ask_model) -> np.ndarray:
    """Probabilities for the texts, from a cache file (json, one entry per distinct text, written atomically) and the model for the texts not yet in it."""
    cache: Dict[str, List[float]] = {}
    if Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    todo = list(dict.fromkeys(t for t in texts if _key(t) not in cache))
    if todo:
        P = ask(todo)
        if len(P) != len(todo) or P.shape[1] != len(OPS):
            raise RuntimeError(f"the reader returned {P.shape} for {len(todo)} texts")
        for t, row in zip(todo, P):
            cache[_key(t)] = [float(x) for x in row]
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        runlib.atomic_write_text(Path(cache_path), json.dumps(cache))
    return np.array([cache[_key(t)] for t in texts], dtype=np.float64).reshape(len(texts), len(OPS))


class AskMachine:
    """The status decision on the asked probabilities. ``cal`` maps each known class ("none" for Other content) to the probabilities of its held-out calibration parts;
    ``states`` maps each taught skill to "solid" or "thin" and ``competence`` / ``counts`` give its evidence fields."""

    def __init__(self, cal: Mapping[str, np.ndarray], states: Mapping[str, str], competence: Mapping[str, float], counts: Mapping[str, int], q: float = 0.05) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must lie in (0, 1), got {q}")
        self.known = list(states) + ["none"]
        for k in self.known:
            if k not in cal or len(cal[k]) == 0:
                raise ValueError(f"no calibration parts for the known class {k!r}")
        self.states, self.competence, self.counts, self.q = dict(states), dict(competence), dict(counts), q
        self._cal = {k: np.sort(np.asarray(cal[k], dtype=np.float64)[:, OPS.index(k)]) for k in self.known}

    def p_values(self, P: np.ndarray) -> np.ndarray:
        """(n, known classes): the conformal p-value of each part against each known class, from the probability the model gives that class; a class COVERS a part when its p-value is above q."""
        P = np.asarray(P, dtype=np.float64)
        return np.stack([ST.conformal_p(self._cal[k], P[:, OPS.index(k)]) for k in self.known], axis=1)

    def covering(self, P: np.ndarray) -> np.ndarray:
        return self.p_values(P) > self.q

    def assess_many(self, P: np.ndarray) -> List[ST.Assessment]:
        P = np.asarray(P, dtype=np.float64)
        cols = [OPS.index(k) for k in self.known]
        p = self.p_values(P)                                                                       # (n, known): p-value of the part against each known class
        covered = p > self.q
        named = np.where(covered, P[:, cols], -np.inf).argmax(axis=1)                              # the covering class the model gives the highest probability
        nearest = P[:, cols].argmax(axis=1)
        out: List[ST.Assessment] = []
        for i in range(len(P)):
            if not covered[i].any():
                out.append(ST.Assessment("unseen", None, float(p[i].max()), 0, math.nan, 1, self.known[int(nearest[i])]))
                continue
            k = self.known[int(named[i])]
            if k == "none":
                out.append(ST.Assessment("other", None, float(p[i, int(named[i])]), 0, 1.0, 1, self.known[int(nearest[i])]))
            else:
                out.append(ST.Assessment(self.states[k], k, float(p[i, int(named[i])]), int(self.counts[k]), float(self.competence[k]), 1, self.known[int(nearest[i])]))
        return out


def top1_accuracy(P: np.ndarray, truth: Sequence[Optional[str]]) -> float:
    """How often the model's most probable option is the true operation (rows whose truth is None are ignored)."""
    rows = [i for i, t in enumerate(truth) if t is not None]
    if not rows:
        return float("nan")
    return float(np.mean([OPS[int(np.argmax(P[i]))] == truth[i] for i in rows]))

"""Test messages with exact labels (spec section 5): one arithmetic request in one of eight surface families, with the request's character span and the true answer.

Families: S1 symbols, S2 number words, S3 verbal phrasing, S4 short stories, S5 operands in the other order (commutative skills only), S6 chatter around the request,
S7 typos / casing / spacing noise, S8 register (formal, casual, terse). A message is a frame (tone, chatter slots) around a core (the request); only the core names the skill.
"""

import string
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import pools
import skills
from growmoe.arith import draw_numbers

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


@dataclass(frozen=True)
class Part:
    start: int
    end: int
    kind: str                       # "skill" | "other" | "unseen_content"
    skill: Optional[str] = None
    a: Optional[int] = None
    b: Optional[int] = None
    answer: Optional[int] = None


@dataclass(frozen=True)
class Message:
    text: str
    parts: Tuple[Part, ...]
    family: str
    template_ids: Tuple[str, ...]   # (frame id, core id); the frame id alone is opaque and shared across skills
    pool: str
    seed: int = -1


def number_to_words(n: int) -> str:
    """English number words for -9999..9999 ("twenty-one", "one thousand two hundred thirty-four", "minus eight")."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + number_to_words(n % 100) if n % 100 else "")
    if n < 10000:
        return _ONES[n // 1000] + " thousand" + (" " + number_to_words(n % 1000) if n % 1000 else "")
    raise ValueError(f"{n} is outside -9999..9999")


def apply_noise(text: str, kind: str, rng: np.random.Generator) -> str:
    """Typos reorder two inner letters of some longer words; case flips some letters; space doubles some spaces. Digits are never touched."""
    if kind == "mix":
        return apply_noise(apply_noise(apply_noise(text, "typo", rng), "case", rng), "space", rng)
    chars = list(text)
    if kind == "typo":
        spans, i = [], 0
        while i < len(chars):
            if chars[i].isalpha():
                j = i
                while j < len(chars) and chars[j].isalpha():
                    j += 1
                if j - i >= 5:
                    spans.append((i, j))
                i = j
            else:
                i += 1
        for s, e in [spans[k] for k in rng.permutation(len(spans))[:max(1, len(spans) // 3)]] if spans else []:
            options = [k for k in range(s + 1, e - 2) if chars[k] != chars[k + 1]]
            if options:
                k = options[int(rng.integers(0, len(options)))]
                chars[k], chars[k + 1] = chars[k + 1], chars[k]
        return "".join(chars)
    if kind == "case":
        letters = [k for k, c in enumerate(chars) if c.isalpha()]
        if not letters:
            return text
        flip = [k for k in letters if rng.random() < 0.3] or [letters[int(rng.integers(0, len(letters)))]]
        for k in flip:
            chars[k] = chars[k].upper() if chars[k].islower() else chars[k].lower()
        return "".join(chars)
    if kind == "space":
        gaps = [k for k, c in enumerate(chars) if c == " "]
        if not gaps:
            return text
        pick = set(int(g) for g in rng.permutation(gaps)[:max(1, len(gaps) // 4)])
        return "".join(c + " " if k in pick else c for k, c in enumerate(chars))
    raise KeyError(f"unknown noise kind {kind!r}")


def _assemble(frame_text: str, core_text: str, pre: str, post: str) -> Tuple[str, dict]:
    """Fill the frame; return the text and the (start, end) of each of 'core', 'pre', 'post' that appears in it."""
    out, spans = "", {}
    for literal, field, _spec, _conv in string.Formatter().parse(frame_text):
        out += literal
        if field is not None:
            value = {"core": core_text, "pre": pre, "post": post}[field]
            spans[field] = (len(out), len(out) + len(value))
            out += value
    return out, spans


def make_single(rng: np.random.Generator, pool: str, skill: str, family: str, digits: Tuple[int, int] = (1, 2), seed: int = -1) -> Message:
    """One request of ``skill`` in surface ``family``. Operands have between digits[0] and digits[1] digits; div and mod never get a zero divisor."""
    if family == "S5" and skill not in skills.COMMUTATIVE:
        raise ValueError("S5 (operands in the other order) exists only for commutative skills")
    P = pools.load_pool(pool)
    source = family if family in P.cores else P.core_sources[family][int(rng.integers(0, len(P.core_sources[family])))]
    cores = [c for c in P.cores[source] if c.skill == skill]
    core = cores[int(rng.integers(0, len(cores)))]
    frame = P.frames[family][int(rng.integers(0, len(P.frames[family])))]
    la, lb = (int(rng.integers(digits[0], digits[1] + 1)) for _ in range(2))
    a = int(draw_numbers(rng, np.array([la]))[0])
    b = int(draw_numbers(rng, np.array([lb]), nonzero=skill in ("div", "mod"))[0])
    things = P.things[int(rng.integers(0, len(P.things)))]
    core_text = core.text.format(a=a, b=b, aw=number_to_words(a), bw=number_to_words(b), things=things)
    if frame.noise:
        core_text = apply_noise(core_text, frame.noise, rng)
    pre = post = ""
    if "{pre}" in frame.text or "{post}" in frame.text:
        i, j = rng.permutation(len(P.chatter))[:2]
        pre, post = P.chatter[int(i)], P.chatter[int(j)]
    text, spans = _assemble(frame.text, core_text, pre, post)
    parts: List[Part] = [Part(*spans["core"], "skill", skill, a, b, skills.truth(skill, a, b))]
    parts += [Part(*spans[k], "other") for k in ("pre", "post") if k in spans]
    return Message(text, tuple(sorted(parts, key=lambda q: q.start)), family, (frame.frame_id, core.core_id), pool, seed)

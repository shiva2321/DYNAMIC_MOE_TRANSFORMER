"""Test messages with exact labels (spec section 5): one arithmetic request in one of eight surface families, with the request's character span and the true answer.

Families: S1 symbols, S2 number words, S3 verbal phrasing, S4 short stories, S5 operands in the other order (commutative skills only), S6 chatter around the request,
S7 typos / casing / spacing noise, S8 register (formal, casual, terse). A message is a frame (tone, chatter slots) around a core (the request); only the core names the skill.
"""

import re
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import corpora
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


def make_single(rng: np.random.Generator, pool: str, skill: str, family: str, digits: Tuple[int, int] = (1, 2), seed: int = -1,
                operands: Optional[Tuple[int, int]] = None) -> Message:
    """One request of ``skill`` in surface ``family``. Operands have between digits[0] and digits[1] digits; div and mod never get a zero divisor.
    ``operands=(a, b)`` fixes the numbers instead (to render the same request in two wordings); a zero divisor is then a ValueError."""
    if family == "S5" and skill not in skills.COMMUTATIVE:
        raise ValueError("S5 (operands in the other order) exists only for commutative skills")
    P = pools.load_pool(pool)
    source = family if family in P.cores else P.core_sources[family][int(rng.integers(0, len(P.core_sources[family])))]
    cores = [c for c in P.cores[source] if c.skill == skill]
    core = cores[int(rng.integers(0, len(cores)))]
    frame = P.frames[family][int(rng.integers(0, len(P.frames[family])))]
    if operands is not None:
        a, b = operands
        if b == 0 and skill in ("div", "mod"):
            raise ValueError(f"{skill} with a zero divisor")
    else:
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


# ------------------------------------------------------------------------------------------------------------------------------------ controls
_WORD_CUE = re.compile(r"divided by|plus|minus|times|mod|max")
_SYMBOL_CUE = re.compile(r"[+\-*/%]")
_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}
_WORD = {"add": "plus", "sub": "minus", "mul": "times", "div": "divided by", "mod": "mod", "max": "max"}


def make_near_twin(rng: np.random.Generator, msg: Message, to_skill: str) -> Message:
    """The same message with only the operator cue changed (a word stays a word, a symbol stays a symbol), operands untouched. Defined for the infix cores of S1 and S2
    (and the frames that wrap them); verbal, story and max cores have no parallel form and raise ValueError, as does a twin that would divide by zero."""
    p = next(x for x in msg.parts if x.kind == "skill")
    if not msg.template_ids[1].startswith(("S1.", "S2.")) or p.skill == "max":
        raise ValueError("a near-twin exists only for the infix cores of S1 and S2 (not max)")
    if to_skill == p.skill:
        raise ValueError("a near-twin must change the skill")
    if to_skill in ("div", "mod") and p.b == 0:
        raise ValueError("the twin would divide by zero")
    req = msg.text[p.start:p.end]
    hit = _WORD_CUE.search(req)
    if hit:
        cue = _WORD[to_skill]
    else:
        hit = _SYMBOL_CUE.search(req)
        if not hit:
            raise ValueError(f"no operator cue found in {req!r}")
        cue = _SYM.get(to_skill, "max")
    new_req = req[:hit.start()] + cue + req[hit.end():]
    delta = len(new_req) - len(req)
    parts = []
    for x in msg.parts:
        if x is p:
            parts.append(Part(x.start, x.end + delta, "skill", to_skill, x.a, x.b, skills.truth(to_skill, x.a, x.b)))
        elif x.start >= p.end:
            parts.append(Part(x.start + delta, x.end + delta, x.kind, x.skill, x.a, x.b, x.answer))
        else:
            parts.append(x)
    return Message(msg.text[:p.start] + new_req + msg.text[p.end:], tuple(parts), msg.family, (msg.template_ids[0], msg.template_ids[1] + "->" + to_skill), msg.pool, msg.seed)


# ------------------------------------------------------------------------------------------------------------------------------------ tangled messages
SEPARATORS = (" ", " Also: ", " By the way: ", " Another thing: ", " Separately: ", " One more thing: ", " Next: ", " New topic: ", " Different question: ")   # end in a colon, so the next segment may start with a capital
_TANGLED_FAMILIES = ("S1", "S3", "S4", "S5")      # digit forms, so a span's operands can be read straight off its text


def _segment(text: str, closing: str) -> str:
    text = text.strip()
    return text if text[-1] in ".?!" else text + closing


def make_tangled(rng: np.random.Generator, pool: str, kinds: Tuple[str, str], skill_labels: Tuple[Optional[str], Optional[str]] = (None, None),
                 families: Tuple[str, ...] = _TANGLED_FAMILIES, digits: Tuple[int, int] = (1, 2), seed: int = -1) -> Message:
    """Two parts in one message, each either a request ("skill") or an everyday sentence ("other"), in a random order, joined by a separator from SEPARATORS.
    The parts are exactly the two segments: the first starts at 0, the second ends at len(text), and the separator belongs to neither. A request segment is the
    request's core only (a frame would add words that belong to no part); ``skill_labels[i]`` names the skill of ``kinds[i]`` (None draws one), and two requests always
    use two different skills. ``families`` are the surface families the requests are drawn from."""
    if len(kinds) != 2 or any(k not in ("skill", "other") for k in kinds):
        raise ValueError(f"kinds must be two of 'skill' / 'other', got {kinds!r}")
    if kinds == ("other", "other"):
        raise ValueError("a tangled message needs at least one request")
    chosen: List[Optional[str]] = [skill_labels[i] if kinds[i] == "skill" else None for i in range(2)]
    if chosen[0] is not None and chosen[0] == chosen[1]:
        raise ValueError("the two requests of a tangled message must use different skills")
    for i in range(2):
        if kinds[i] == "skill" and chosen[i] is None:
            taken = {s for s in chosen if s is not None}
            options = [s for s in skills.SKILLS if s not in taken]
            chosen[i] = options[int(rng.integers(0, len(options)))]
    segments = []                  # (text, Part with start/end relative to the segment, template id)
    for i in range(2):
        if kinds[i] == "other":
            text = _segment(corpora.sample_other(rng), ".")
            segments.append((text, Part(0, len(text), "other"), "other"))
            continue
        fams = [f for f in families if f != "S5" or chosen[i] in skills.COMMUTATIVE]
        if not fams:
            raise ValueError(f"none of the families {families} can express the skill {chosen[i]!r}")
        one = make_single(rng, pool, chosen[i], fams[int(rng.integers(0, len(fams)))], digits)
        p = next(x for x in one.parts if x.kind == "skill")
        text = _segment(one.text[p.start:p.end], "?")
        segments.append((text, Part(0, len(text), "skill", p.skill, p.a, p.b, p.answer), one.template_ids[1]))
    if rng.random() < 0.5:
        segments.reverse()
    sep = SEPARATORS[int(rng.integers(0, len(SEPARATORS)))]
    (t1, p1, id1), (t2, p2, id2) = segments
    offset = len(t1) + len(sep)
    parts = (p1, Part(p2.start + offset, p2.end + offset, p2.kind, p2.skill, p2.a, p2.b, p2.answer))
    return Message(t1 + sep + t2, parts, "tangled", (id1, id2), pool, seed)


def _single_other(text: str, kind: str, family: str, template_id: str, pool: str, seed: int) -> Message:
    return Message(text, (Part(0, len(text), kind),), family, (template_id,), pool, seed)


def make_trap(rng: np.random.Generator, pool: str, seed: int = -1) -> Message:
    """Arithmetic words or digits used for something else: a word-matcher calls it a request, but its skill is None and its kind is 'other'."""
    P = pools.load_pool(pool)
    k = int(rng.integers(0, len(P.traps)))
    return _single_other(P.traps[k].format(n=int(rng.integers(2, 99))), "other", "trap", f"trap.{k}", pool, seed)


def make_other(rng: np.random.Generator, pool: str, seed: int = -1) -> Message:
    """An everyday sentence (banking77 or AG News): known "Other content", never a skill."""
    return _single_other(corpora.sample_other(rng), "other", "other", "other", pool, seed)


def make_unseen_content(rng: np.random.Generator, pool: str, seed: int = -1) -> Message:
    """A sentence of a content type that is never taught (a science question): the easy cross-type novelty control."""
    return _single_other(corpora.sample_unseen(rng), "unseen_content", "unseen", "unseen", pool, seed)

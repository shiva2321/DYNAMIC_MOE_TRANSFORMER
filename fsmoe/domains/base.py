"""The common shape of a test-bed domain: skills x wording families x near-twins x withheld skills, with an outside anchor.

An ``Item`` is one message (or one snippet). Its ``skill`` is the thing to recognise, its ``family`` the way it is worded (teaching families are the ones a system may learn from, held-out families
are never trained on), and ``check`` is the outside anchor: a checker the system never sees as a feature (the generator's own answer, or the execution of the code).
"""

import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Item:
    text: str
    domain: str
    skill: str
    family: str
    meta: tuple = ()


def rng_for(*keys) -> np.random.Generator:
    """A generator that depends on every key (ints are used as they are, anything else through crc32)."""
    return np.random.default_rng([int(k) if isinstance(k, (int, np.integer)) else zlib.crc32(str(k).encode()) for k in keys])


class Domain:
    name: str = ""
    skills: Tuple[str, ...] = ()
    twins: Dict[str, str] = {}                      # skill -> its nearest confusable skill
    teach_families: Tuple[str, ...] = ()
    heldout_families: Tuple[str, ...] = ()
    withheld: Tuple[str, ...] = ()                  # skills a system is never taught
    far_withheld: Tuple[str, ...] = ()              # the withheld skills whose twin is NOT a taught skill's near copy (the "far" ones); the rest are near-twins

    @property
    def taught(self) -> Tuple[str, ...]:
        return tuple(s for s in self.skills if s not in self.withheld)

    @property
    def families(self) -> Tuple[str, ...]:
        return self.teach_families + self.heldout_families

    def near_withheld(self) -> Tuple[str, ...]:
        return tuple(s for s in self.withheld if s not in self.far_withheld)

    def make(self, skill: str, family: str, n: int, seed: int, split: str = "teach") -> List[Item]:
        raise NotImplementedError

    def check(self, item: Item) -> bool:
        raise NotImplementedError


def table(dom: Domain, seed: int, teach_n: int, eval_n: int) -> List[Tuple[Item, str]]:
    """Every (item, split) of a domain: ``teach_n`` items per taught skill and teaching family for the split "teach"; ``eval_n`` items per skill (taught or withheld) and family for the split "eval"
    (teaching families included, so that new items in a taught wording are measured too). Withheld skills have no "teach" rows."""
    rows: List[Tuple[Item, str]] = []
    for s in dom.skills:
        if s not in dom.withheld:
            for f in dom.teach_families:
                rows += [(it, "teach") for it in dom.make(s, f, teach_n, seed, "teach")]
        for f in dom.families:
            rows += [(it, "eval") for it in dom.make(s, f, eval_n, seed + 1, "eval")]
    return rows


def noisy(text: str, rng: np.random.Generator, kinds: Sequence[str] = ("typo", "case", "space")) -> str:
    """Typos reorder two inner letters of some longer words; case flips some letters; space doubles some spaces. Digits are never touched."""
    chars = list(text)
    for kind in kinds:
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
            order = [spans[k] for k in rng.permutation(len(spans))[:max(1, len(spans) // 3)]] if spans else []
            for s, e in order:
                options = [k for k in range(s + 1, e - 2) if chars[k] != chars[k + 1]]
                if options:
                    k = options[int(rng.integers(0, len(options)))]
                    chars[k], chars[k + 1] = chars[k + 1], chars[k]
        elif kind == "case":
            letters = [k for k, c in enumerate(chars) if c.isalpha()]
            if letters:
                flip = [k for k in letters if rng.random() < 0.3] or [letters[int(rng.integers(0, len(letters)))]]
                for k in flip:
                    chars[k] = chars[k].upper() if chars[k].islower() else chars[k].lower()
        elif kind == "space":
            gaps = [k for k, c in enumerate(chars) if c == " "]
            if gaps:
                pick = set(int(g) for g in rng.permutation(gaps)[:max(1, len(gaps) // 4)])
                chars = list("".join(c + " " if k in pick else c for k, c in enumerate(chars)))
        else:
            raise KeyError(f"unknown noise kind {kind!r}")
    return "".join(chars)

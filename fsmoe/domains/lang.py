"""Language as intent recognition: the 77 banking intents, five wording families, near-twin intents, twenty intents never taught.

Every intent's utterances are split once into a teaching pool and an evaluation pool (disjoint, fixed), so a held-out wording is never a re-wording of an utterance the system learned. Twins are
found from the data (nearest intent by tf-idf cosine over the intents' words); the ten intents with the closest twin whose twin stays taught are withheld as NEAR novelty, and ten intents with the
farthest twin are withheld as FAR-within-domain novelty. No class names are read for this choice, so it does not depend on what an intent is called.
Teaching family: the original utterance. Held-out families: typos (typos, casing, spacing), chatter, register (formal, casual, terse), framing (quoted inside a short story of why it was asked).
"""

import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from unimoe.domains import _raw
from .base import Domain, Item, noisy, rng_for

STOP = frozenset("i my me to a the is it do how can what why when where of in on for and or be am are was with have has not no this that you your we our an at as by if so but just get got there".split())
TEACH = ("original",)
HELDOUT = ("typos", "chatter", "register", "framing")
PRE = ("hi there, ", "quick question, ", "sorry to bother you but ", "hello! ", "so, ", "um ", "hey, ")
POST = (" thanks!", " please help.", " thank you so much.", " any ideas?", " cheers", "")
REGISTER = ("Good day. I would like to ask the following: {u}", "hey so {l} lol", "{u} pls", "Dear support team, {u} Kind regards.")
FRAMING = ("My friend sent me this and I am not sure what to do about it: \"{u}\"", "Someone on a forum wrote: \"{u}\" and I have the same problem.", "A customer wrote in: \"{u}\"")
N_NEAR, N_FAR = 10, 10


@lru_cache(maxsize=1)
def _load() -> dict:
    tr_x, tr_y, _, _, names = _raw("banking")
    rng = rng_for(0, "lang-split")
    per: Dict[int, List[str]] = {c: [] for c in range(len(names))}
    for t, y in zip(tr_x, tr_y):
        per[int(y)].append(t)
    teach, ev = {}, {}
    for c, texts in per.items():
        order = rng.permutation(len(texts))
        texts = [texts[i] for i in order]
        half = len(texts) // 2
        teach[names[c]], ev[names[c]] = texts[:half], texts[half:]
    toks = [Counter(w for t in teach[n] + ev[n] for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP) for n in names]
    vocab = sorted({w for c in toks for w in c})
    index = {w: k for k, w in enumerate(vocab)}
    M = np.zeros((len(names), len(vocab)), dtype=np.float64)
    for r, c in enumerate(toks):
        tot = sum(c.values())
        for w, k in c.items():
            M[r, index[w]] = k / tot
    df = (M > 0).sum(0)
    M *= np.log(len(names) / np.maximum(df, 1))
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    cos = M @ M.T
    np.fill_diagonal(cos, -1.0)
    twin = cos.argmax(1)
    closeness = cos.max(1)
    near: List[int] = []
    for k in np.argsort(-closeness):
        k = int(k)
        if len(near) == N_NEAR:
            break
        if int(twin[k]) not in near and k not in {int(twin[j]) for j in near}:
            near.append(k)
    blocked = set(near) | {int(twin[j]) for j in near}
    far: List[int] = []
    for k in np.argsort(closeness):
        k = int(k)
        if len(far) == N_FAR:
            break
        if k not in blocked and int(twin[k]) not in far and int(twin[k]) not in near and k not in {int(twin[j]) for j in far}:
            far.append(k)
    return {"names": names, "teach": teach, "eval": ev, "twins": {names[k]: names[int(twin[k])] for k in range(len(names))},
            "withheld": tuple(names[k] for k in near + far), "far": tuple(names[k] for k in far)}


class LangDomain(Domain):
    name = "lang"
    teach_families = TEACH
    heldout_families = HELDOUT

    def __init__(self) -> None:
        d = _load()
        self._d = d
        self.skills = tuple(d["names"])
        self.twins = d["twins"]
        self.withheld = d["withheld"]
        self.far_withheld = d["far"]

    def make(self, skill: str, family: str, n: int, seed: int, split: str = "teach") -> List[Item]:
        if skill not in self._d["teach"] or family not in self.families:
            raise ValueError(f"unknown skill {skill!r} or family {family!r}")
        if split == "teach" and family not in TEACH:
            raise ValueError("a held-out wording is never taught")
        pool = self._d["teach" if split == "teach" else "eval"][skill]
        rng = rng_for(seed, "lang", skill, family, split)
        order = rng.permutation(len(pool))
        out = []
        for k in range(n):
            u = pool[int(order[k % len(pool)])]
            if family == "typos":
                text = noisy(u, rng)
            elif family == "chatter":
                text = str(rng.choice(PRE)) + u + str(rng.choice(POST))
            elif family == "register":
                text = str(rng.choice(REGISTER)).format(u=u, l=u.lower())
            elif family == "framing":
                text = str(rng.choice(FRAMING)).format(u=u)
            else:
                text = u
            out.append(Item(text, self.name, skill, family, (self.skills.index(skill),)))
        return out

    def check(self, item: Item) -> bool:
        return self.skills[item.meta[0]] == item.skill

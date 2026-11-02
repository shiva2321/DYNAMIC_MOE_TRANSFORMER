"""The embedded test bed: every row of the three domains with its labels and the shared reader's raw vectors, plus the far-novelty sets.

``build_bed(seed)`` is deterministic: the rows come from the domain generators (seeded), the vectors from the cached reader. A row is labelled (domain, skill, family, split) and flagged ``taught`` (its skill is
taught) and ``heldout`` (its wording family is never taught). Teaching rows are rows of taught skills in teaching families; nothing else may be used to learn from.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .domains import NAMES, all_domains, table
from .domains.far import far_texts
from .reader import CachedReader, MiniLMReader, QwenReader

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "fsmoe" / "_cache"
SIZES = {"arith": (100, 40), "code": (100, 40), "lang": (60, 12)}          # (teaching items per taught skill and teaching family, evaluation items per skill and family)
FAR_N = 300


@dataclass
class Bed:
    seed: int
    domains: dict
    skills: List[str]                                   # global skill names "domain:skill"
    domain: np.ndarray
    skill: np.ndarray
    family: np.ndarray
    split: np.ndarray
    taught: np.ndarray
    heldout: np.ndarray
    text: List[str]
    raw: np.ndarray                                     # float16 (N, 1536 + 384): the Qwen block then the MiniLM block
    y: np.ndarray                                       # index into skills
    twin: Dict[str, str]
    withheld: frozenset
    far_withheld: frozenset
    far: Dict[str, Tuple[List[str], np.ndarray]] = field(default_factory=dict)

    def mask(self, domain: Optional[str] = None, split: Optional[str] = None, taught: Optional[bool] = None, heldout: Optional[bool] = None, family: Optional[str] = None,
             skill: Optional[str] = None) -> np.ndarray:
        m = np.ones(len(self.text), dtype=bool)
        if domain is not None:
            m &= self.domain == domain
        if split is not None:
            m &= self.split == split
        if taught is not None:
            m &= self.taught == taught
        if heldout is not None:
            m &= self.heldout == heldout
        if family is not None:
            m &= self.family == family
        if skill is not None:
            m &= self.skill == skill
        return m

    def teaching(self, domain: Optional[str] = None) -> np.ndarray:
        """Rows a system may learn from: taught skill, teaching wording, split teach."""
        return self.mask(domain=domain, split="teach", taught=True, heldout=False)


def rows_of(seed: int, sizes: Dict[str, Tuple[int, int]] = SIZES, verify: bool = True) -> dict:
    """The rows (no vectors) of every domain; with ``verify`` every row is checked by its domain's outside anchor and any failure raises."""
    doms = all_domains()
    cols = {k: [] for k in ("domain", "skill", "family", "split", "taught", "heldout", "text")}
    for name in NAMES:
        dom = doms[name]
        teach_n, eval_n = sizes[name]
        for it, sp in table(dom, seed, teach_n, eval_n):
            if verify and not dom.check(it):
                raise AssertionError(f"the outside anchor rejects a generated item: {name} {it.skill} {it.family}: {it.text!r}")
            cols["domain"].append(name)
            cols["skill"].append(f"{name}:{it.skill}")
            cols["family"].append(it.family)
            cols["split"].append(sp)
            cols["taught"].append(it.skill not in dom.withheld)
            cols["heldout"].append(it.family in dom.heldout_families)
            cols["text"].append(it.text)
    return {"doms": doms, **cols}


def default_readers() -> List[CachedReader]:
    return [QwenReader(CACHE), MiniLMReader(CACHE)]


def _encode(readers: Sequence[CachedReader], texts: Sequence[str], name: str) -> np.ndarray:
    """The blocks of every reader side by side: float16 (n, sum of the readers' widths)."""
    return np.concatenate([rd.encode(texts, name) for rd in readers], axis=1)


def build_bed(seed: int, readers: Optional[Sequence[CachedReader]] = None, sizes: Dict[str, Tuple[int, int]] = SIZES, far_n: int = FAR_N, verify: bool = True,
              with_far: bool = True) -> Bed:
    readers = list(readers) if readers is not None else default_readers()
    width = sum(rd.dim for rd in readers)
    r = rows_of(seed, sizes, verify)
    doms = r["doms"]
    domain = np.asarray(r["domain"])
    raw = np.zeros((len(domain), width), dtype=np.float16)
    for name in NAMES:
        idx = np.flatnonzero(domain == name)
        raw[idx] = _encode(readers, [r["text"][i] for i in idx], f"{name}_s{seed}")
    skills = [f"{n}:{s}" for n in NAMES for s in doms[n].skills]
    skill = np.asarray(r["skill"])
    lookup = {s: i for i, s in enumerate(skills)}
    far = {}
    if with_far:
        for k, texts in far_texts(far_n, 0).items():
            far[k] = (texts, _encode(readers, texts, f"far_{k}"))
    return Bed(seed, doms, skills, domain, skill, np.asarray(r["family"]), np.asarray(r["split"]), np.asarray(r["taught"], dtype=bool), np.asarray(r["heldout"], dtype=bool), r["text"], raw,
               np.asarray([lookup[s] for s in skill]), {f"{n}:{s}": f"{n}:{t}" for n in NAMES for s, t in doms[n].twins.items()},
               frozenset(f"{n}:{s}" for n in NAMES for s in doms[n].withheld), frozenset(f"{n}:{s}" for n in NAMES for s in doms[n].far_withheld), far)

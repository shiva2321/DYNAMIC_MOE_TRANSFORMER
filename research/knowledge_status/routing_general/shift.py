"""Shift diagnostics: how far can a reader be trusted when the wording of a known message changes?

The roster's own held-out messages are rewritten by the same language model with GENERIC instructions (different words, a different sentence structure; they name no domain and no wording
family of any test bed). A reader that flags many of the rewrites of messages it knows is shift-sensitive. The rewrites inherit the class of their source. This is a DIAGNOSTIC; earlier work
(PROGRESS.md, 2026-10-28) showed that rewrites as extra calibration or prototypes do not cure the look-based reader.
"""

import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import rewrites as RW  # noqa: E402
import runlib  # noqa: E402

MAX_WORDS = RW.MAX_WORDS
INSTRUCTIONS = RW.INSTRUCTIONS
Labelled = List[Tuple[str, int]]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def keep_general(source: str, rewrite: str) -> bool:
    """A usable rewrite: one line, not too long, not chatter, not the source again, and every digit string of the source still there (digits are the one thing any domain can check)."""
    r = rewrite.strip()
    if not r or "\n" in r or len(r.split()) > MAX_WORDS:
        return False
    if _norm(r) == _norm(source) or _norm(r).startswith(RW._CHATTER_START):
        return False
    return sorted(re.findall(r"\d+", source)) == sorted(re.findall(r"\d+", r))


def split_calibration(known_cal: Sequence[Tuple[str, int]]) -> Tuple[Labelled, Labelled]:
    """Per class, alternate the held-out messages between a half that sets the cut (A) and a half that is rewritten (B), so a rewrite is never scored against its own source's cut."""
    a: Labelled = []
    b: Labelled = []
    seen: Dict[int, int] = {}
    for t, c in known_cal:
        k = seen.get(c, 0)
        seen[c] = k + 1
        (a if k % 2 == 0 else b).append((t, c))
    return a, b


def cap_per_class(items: Sequence[Tuple[str, int]], n: int) -> Labelled:
    return [x for c in sorted({c for _, c in items}) for x in [y for y in items if y[1] == c][:n]]


def shift_set(sources: Sequence[Tuple[str, int]], generate: RW.Generator, cache_dir, seed: int, keep: Callable[[str, str], bool] = keep_general,
              instructions: Sequence[str] = INSTRUCTIONS, batch: int = 64) -> Labelled:
    """The kept rewrites of the sources, each with the class of its source. Generation is cached in blocks on disk (written atomically); the filter runs when the cache is read."""
    label: Dict[str, int] = {}
    for t, c in sources:
        label.setdefault(t, c)
    uniq = list(label)
    seen = {s: {_norm(s)} for s in uniq}
    out: Labelled = []
    if not uniq:
        return out
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for instruction in instructions:
        for start in range(0, len(uniq), batch):
            block = uniq[start:start + batch]
            path = RW._block_path(Path(cache_dir), instruction, seed, "shift", block)
            raw = RW._read_block(path, len(block))
            if raw is None:
                raw = [str(x) for x in generate([RW.build_prompt(instruction, s) for s in block], seed)]
                if len(raw) != len(block):
                    raise RuntimeError(f"the generator returned {len(raw)} texts for {len(block)} prompts")
                runlib.atomic_write_text(path, json.dumps({"raw": raw}))
            for s, r in zip(block, raw):
                if keep(s, r) and _norm(r) not in seen[s]:
                    seen[s].add(_norm(r))
                    out.append((r.strip(), label[s]))
    return out


def trusted(shift_flags: Mapping[str, float], limit: float) -> List[str]:
    """The readers whose flag rate on rewrites of known messages is at most `limit`; if there is none, the least shifted one (a layer without a reader cannot route)."""
    if not shift_flags:
        raise ValueError("no readers to choose from")
    ok = [r for r, v in shift_flags.items() if v <= limit]
    return ok if ok else [min(shift_flags, key=lambda r: shift_flags[r])]


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=np.float64)
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("spearman needs two vectors of the same length")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

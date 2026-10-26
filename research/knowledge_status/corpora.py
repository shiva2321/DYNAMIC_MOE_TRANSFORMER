"""Offline everyday sentences for the knowledge-status test bed: chatter and "Other content" (banking77 and AG News) and one content type that is never taught
(ARC science questions). Every sentence is clean: no digits, no arithmetic cue word (whole-word match, so "modern" is fine), no arithmetic symbol, 5 to 40 words.
Only the local Hugging Face cache is read; a missing dataset is a loud FileNotFoundError that names it.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import pools
import runlib

ARITH_CUES = pools.ARITH_CUES
HUB = Path(os.environ.get("KS_HF_HUB", Path.home() / ".cache" / "huggingface" / "hub"))
CACHE_DIR = Path(__file__).resolve().parent / "_cache"
_MEMO: Dict[str, List[str]] = {}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ALLOWED = re.compile(r"^[A-Za-z][A-Za-z ,.'?!\"]*$")


def is_clean(sentence: str) -> bool:
    words = sentence.split()
    if not 5 <= len(words) <= 40:
        return False
    if re.search(r"\d", sentence) or (set(sentence) & pools.ARITH_SYMBOLS):
        return False
    return not (set(re.findall(r"[a-z]+", sentence.lower())) & ARITH_CUES)


def _snapshot(dataset_dir: str, sub: str, pattern: str) -> Path:
    base = HUB / dataset_dir / "snapshots"
    hits = sorted(base.glob(f"*/{sub}/{pattern}")) if base.exists() else []
    if not hits:
        raise FileNotFoundError(f"dataset {dataset_dir.replace('datasets--', '')} (looked for {sub}/{pattern}) is not in the local cache {HUB}; this project never downloads")
    return hits[0]


def _sentences_from(texts) -> List[str]:
    out, seen = [], set()
    for t in texts:
        for s in _SENTENCE_END.split(" ".join(str(t).split())):
            s = s.strip()
            if _ALLOWED.match(s) and is_clean(s) and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _read_cache(path: Path):
    """The cached list, or None when the file is missing, unreadable or empty (it is then rebuilt)."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cached if isinstance(cached, list) and cached else None


def _candidates(name: str) -> List[str]:
    """All clean sentences of one source, in a fixed order, cached in memory and on disk."""
    if name in _MEMO:
        return _MEMO[name]
    path = CACHE_DIR / f"candidates_{name}.json"
    out = _read_cache(path)
    if out is None:
        if name == "banking77":
            texts = pd.read_parquet(_snapshot("datasets--banking77", "data", "train-*.parquet"))["text"].tolist()
        elif name == "agnews":
            texts = pd.read_parquet(_snapshot("datasets--ag_news", "data", "train-*.parquet"))["text"].tolist()
        elif name == "arc":
            texts = pd.read_parquet(_snapshot("datasets--ai2_arc", "ARC-Challenge", "train-*.parquet"))["question"].tolist()
        else:
            raise KeyError(name)
        out = _sentences_from(texts)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        runlib.atomic_write_text(path, json.dumps(out))
    _MEMO[name] = out
    return out


def _union(source: str) -> List[str]:
    key = "union:" + source
    if key not in _MEMO:
        seen, pool = set(), []
        for name in source.split("|"):
            for s in _candidates(name):
                if s not in seen:
                    seen.add(s)
                    pool.append(s)
        _MEMO[key] = pool
    return _MEMO[key]


def _pick(pool: List[str], n: int, seed: int) -> List[str]:
    if n > len(pool):
        raise ValueError(f"asked for {n} sentences but only {len(pool)} clean ones exist")
    idx = np.random.default_rng([seed, 0x4B53]).permutation(len(pool))[:n]
    return [pool[int(i)] for i in idx]


def load_other_sentences(n: int, seed: int, source: str = "banking77|agnews") -> List[str]:
    """n distinct clean everyday sentences ("Other content" and chatter); source is banking77, agnews or both joined by '|'."""
    return _pick(_union(source), n, seed)


def load_unseen_content(n: int, seed: int) -> List[str]:
    """n distinct clean sentences of a content type that is never taught (science questions)."""
    return _pick(_union("arc"), n, seed)


def sample_other(rng: np.random.Generator, source: str = "banking77|agnews") -> str:
    pool = _union(source)
    return pool[int(rng.integers(0, len(pool)))]


def sample_unseen(rng: np.random.Generator) -> str:
    pool = _union("arc")
    return pool[int(rng.integers(0, len(pool)))]

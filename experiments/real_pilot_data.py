"""Real multi-domain byte corpora for the pilot (experiment `real_pilot_v1`).

Domains (all REAL text, loaded from the local HuggingFace cache, offline):
  wiki   : wikitext-103-raw-v1            (encyclopedic prose)         train / test splits as shipped
  story  : roneneldan/TinyStories         (simple fiction)             train / validation(as test)
  code   : python code-instructions 18k   (Python source + English)    deterministic row split + dedup
  math   : gsm8k main                     (grade-school word problems) train / test splits as shipped
  forum  : SetFit/20_newsgroups           (informal email/forum text)  train / test splits as shipped

Disjointness from the DEVELOPMENT streams (ag_news, ai2_arc, repo .py files, synthetic JSON) is by source.
Train and test windows come from disjoint dataset splits (or a deterministic disjoint row split for code).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DOMAINS = ["wiki", "story", "code", "math", "forum"]
MAX_TRAIN_BYTES = 3_000_000
TEST_SEQS = 256          # fixed windows per domain, identical for every arm and seed
SEQ_LEN = 128
TEST_WINDOW_SEED = 777
CODE_SPLIT_SEED = 12345


def _clean_ws(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", s)


def _join(docs: List[str], cap: int) -> bytes:
    out, n = [], 0
    for d in docs:
        b = d.encode("utf-8", errors="ignore")
        out.append(b)
        n += len(b) + 2
        if n >= cap:
            break
    return b"\n\n".join(out)[:cap]


def _load_wiki(split: str = "test") -> Tuple[bytes, bytes]:
    from datasets import load_dataset
    tr = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    if split == "dev":
        va = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        tr_docs = [t.strip() for t in tr.select(range(60000, 90000))["text"] if len(t.strip()) > 40]
        te_docs = [t.strip() for t in va["text"] if len(t.strip()) > 40]
        return _join(tr_docs, MAX_TRAIN_BYTES), _join(te_docs, MAX_TRAIN_BYTES)
    te = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    tr_docs = [t.strip() for t in tr.select(range(0, 60000))["text"] if len(t.strip()) > 40]
    te_docs = [t.strip() for t in te["text"] if len(t.strip()) > 40]
    return _join(tr_docs, MAX_TRAIN_BYTES), _join(te_docs, MAX_TRAIN_BYTES)


def _load_story(split: str = "test") -> Tuple[bytes, bytes]:
    from datasets import load_dataset
    tr = load_dataset("roneneldan/TinyStories", split="train")
    va = load_dataset("roneneldan/TinyStories", split="validation")
    if split == "dev":
        tr_docs = list(tr.select(range(12000, 20000))["text"])
        te_docs = list(va.select(range(4000, 8000))["text"])
        return _join(tr_docs, MAX_TRAIN_BYTES), _join(te_docs, MAX_TRAIN_BYTES)
    tr_docs = list(tr.select(range(0, 12000))["text"])
    te_docs = list(va.select(range(0, 4000))["text"])
    return _join(tr_docs, MAX_TRAIN_BYTES), _join(te_docs, MAX_TRAIN_BYTES)


def _load_code(split: str = "test") -> Tuple[bytes, bytes, Dict[str, int]]:
    from datasets import load_dataset
    ds = None
    load_errors: Dict[str, str] = {}
    for name in ("iamtarun/python_code_instructions_18k_alpaca",
                 "HydraLM/python-code-instructions-18k-alpaca-standardized",
                 "AdapterOcean/python-code-instructions-18k-alpaca-standardized_cluster_2_alpaca"):
        try:
            ds = load_dataset(name, split="train")
            src = name
            break
        except (OSError, ConnectionError, FileNotFoundError, ValueError, KeyError) as err:
            load_errors[name] = f"{type(err).__name__}: {err}"
            continue
    if ds is None:
        raise RuntimeError(f"no cached python code dataset available. Errors encountered: {load_errors}")
    cols = ds.column_names
    rows = []
    for r in ds:
        if "output" in r and "instruction" in r:
            rows.append(f"### Instruction:\n{r['instruction']}\n### Output:\n{r['output']}")
        elif "output" in r and "input" in r:
            rows.append(f"{r['input']}\n### Output:\n{r['output']}")
        else:
            rows.append(" ".join(str(v) for v in r.values()))
    idx = list(range(len(rows)))
    random.Random(CODE_SPLIT_SEED).shuffle(idx)
    n_te = max(200, len(rows) // 10)
    te_idx, tr_idx = idx[:n_te], idx[n_te:]
    if split == "dev":
        n_va = min(200, len(tr_idx) // 5)
        va_idx, dev_tr_idx = tr_idx[-n_va:], tr_idx[:-n_va]
        tr_hash = {hashlib.md5(rows[i].encode("utf-8", "ignore")).hexdigest() for i in dev_tr_idx}
        va_keep = [i for i in va_idx if hashlib.md5(rows[i].encode("utf-8", "ignore")).hexdigest() not in tr_hash]
        stats = {"rows": len(rows), "train_rows": len(dev_tr_idx), "val_rows_after_dedup": len(va_keep)}
        print(f"[data:dev] code source={src} cols={cols} {stats}")
        return (_join([rows[i] for i in dev_tr_idx], MAX_TRAIN_BYTES),
                _join([rows[i] for i in va_keep], MAX_TRAIN_BYTES), stats)
    tr_hash = {hashlib.md5(rows[i].encode("utf-8", "ignore")).hexdigest() for i in tr_idx}
    te_keep = [i for i in te_idx if hashlib.md5(rows[i].encode("utf-8", "ignore")).hexdigest() not in tr_hash]
    stats = {"rows": len(rows), "train_rows": len(tr_idx), "test_rows_before_dedup": len(te_idx),
             "test_rows_after_exact_dedup": len(te_keep)}
    print(f"[data] code source={src} cols={cols} {stats}")
    return (_join([rows[i] for i in tr_idx], MAX_TRAIN_BYTES),
            _join([rows[i] for i in te_keep], MAX_TRAIN_BYTES), stats)


def _load_math(split: str = "test") -> Tuple[bytes, bytes]:
    from datasets import load_dataset
    tr = load_dataset("gsm8k", "main", split="train")
    f = lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}"
    if split == "dev":
        tr_sub = tr.select(range(0, 6000))
        va_sub = tr.select(range(6000, len(tr)))
        return _join([f(r) for r in tr_sub], MAX_TRAIN_BYTES), _join([f(r) for r in va_sub], MAX_TRAIN_BYTES)
    te = load_dataset("gsm8k", "main", split="test")
    return _join([f(r) for r in tr], MAX_TRAIN_BYTES), _join([f(r) for r in te], MAX_TRAIN_BYTES)


def _load_forum(split: str = "test") -> Tuple[bytes, bytes]:
    from datasets import load_dataset
    tr = load_dataset("SetFit/20_newsgroups", split="train")
    tr_docs = [_clean_ws(t) for t in tr["text"] if len(t) > 80]
    if split == "dev":
        cut = int(len(tr_docs) * 0.8)
        return _join(tr_docs[:cut], MAX_TRAIN_BYTES), _join(tr_docs[cut:], MAX_TRAIN_BYTES)
    te = load_dataset("SetFit/20_newsgroups", split="test")
    te_docs = [_clean_ws(t) for t in te["text"] if len(t) > 80]
    return _join(tr_docs, MAX_TRAIN_BYTES), _join(te_docs, MAX_TRAIN_BYTES)


class RealCorpora:
    """Holds per-domain train byte pools and fixed evaluation windows (`dev` or `test` split)."""

    def __init__(self, seq_len: int = SEQ_LEN, eval_split: str = "test"):
        if eval_split not in ("dev", "test"):
            raise ValueError(f"eval_split must be 'dev' or 'test', got {eval_split}")
        self.seq_len = seq_len
        self.eval_split = eval_split
        self.train: Dict[str, np.ndarray] = {}
        self.test_windows: Dict[str, torch.Tensor] = {}
        self.test_pool: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, object] = {"eval_split": eval_split}
        loaders = {"wiki": _load_wiki, "story": _load_story, "math": _load_math, "forum": _load_forum}
        for d in DOMAINS:
            if d == "code":
                tr, te, st = _load_code(split=eval_split)
                self.meta["code_split_stats"] = st
            else:
                tr, te = loaders[d](split=eval_split)
            self.train[d] = np.frombuffer(tr, dtype=np.uint8).copy()
            self.test_pool[d] = np.frombuffer(te, dtype=np.uint8).copy()
            self.test_windows[d] = self._fixed_test_windows(self.test_pool[d])
            self.meta[f"{d}_train_bytes"] = int(len(tr))
            self.meta[f"{d}_eval_bytes"] = int(len(te))
        self.meta["leakage_audit"] = self.leakage_audit()

    def _fixed_test_windows(self, pool: np.ndarray) -> torch.Tensor:
        # non-overlapping windows at fixed, seeded offsets -> identical for all arms/seeds
        L = self.seq_len + 1
        n_slots = len(pool) // L
        rng = random.Random(TEST_WINDOW_SEED)
        slots = rng.sample(range(n_slots), min(TEST_SEQS, n_slots))
        arr = np.stack([pool[s * L:(s + 1) * L] for s in sorted(slots)])
        return torch.from_numpy(arr.astype(np.int64))  # [N, L+1]: x = [:, :-1], y = [:, 1:]

    def sample_train(self, domain: str, batch: int, gen: np.random.Generator) -> torch.Tensor:
        pool = self.train[domain]
        L = self.seq_len + 1
        offs = gen.integers(0, len(pool) - L, size=batch)
        arr = np.stack([pool[o:o + L] for o in offs]).astype(np.int64)
        return torch.from_numpy(arr)

    def leakage_audit(self) -> Dict[str, float]:
        """Exact 64-byte shingle overlap between each domain's test windows and its own train pool."""
        res = {}
        for d in DOMAINS:
            pool = self.train[d]
            grams = set(pool[i:i + 64].tobytes() for i in range(0, len(pool) - 64, 16))
            hit = tot = 0
            for w in self.test_windows[d]:
                b = w.numpy().astype(np.uint8)
                for i in range(0, len(b) - 64, 16):
                    tot += 1
                    hit += b[i:i + 64].tobytes() in grams
            res[d] = hit / max(1, tot)
        return res


def bits_per_byte_from_nll(nll_nats: float) -> float:
    return nll_nats / float(np.log(2.0))

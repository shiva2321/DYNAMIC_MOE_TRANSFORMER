"""Real-data domains for the UniMoE suite. Local Hugging Face cache only (this project never downloads; a missing dataset is a loud FileNotFoundError).

A ``Domain`` is one kind of data already on the sphere: train and test vectors with LOCAL labels and a regime for every example. In the text domains a regime is a group of classes (class-
incremental: the classes of one regime arrive together, found by spherical k-means on the class means); in arithmetic a regime is an operator (the same twelve digit labels, a different
function). Text is capped per class so that every domain has a similar order of size; the arithmetic domain is built from ``growmoe.arith`` (no arithmetic knowledge, only the data format).
"""

import json
import math
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from growmoe import arith
from growmoe.fingerprint import spherical_kmeans

HUB = Path(os.environ.get("KS_HF_HUB", Path.home() / ".cache" / "huggingface" / "hub"))
MAX_WORDS = 60
N_GROUPS = {"banking": 6, "agnews": 2, "newsgroups": 4, "emotion": 2, "sst2": 1, "svamp": 1}
TEXT_DOMAINS = tuple(N_GROUPS)
ALL_DOMAINS = TEXT_DOMAINS + ("arith",)
ARITH_LENS = (1, 2, 3)
EMOTION = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@dataclass
class Domain:
    name: str
    kind: str                                       # "text" | "digits"
    x_train: torch.Tensor
    y_train: torch.Tensor                           # local labels 0..n_classes-1
    r_train: torch.Tensor                           # local regime of every example
    x_test: torch.Tensor
    y_test: torch.Tensor
    r_test: torch.Tensor
    n_classes: int
    class_names: List[str]
    regime_names: List[str]
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------------------------------------ helpers
def stratified(y: Sequence[int], cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of at most ``cap`` examples with every class represented as evenly as the data allows (all of them when the cap is above the data)."""
    y = np.asarray(y)
    if cap >= len(y):
        return np.arange(len(y))
    classes = sorted(set(y.tolist()))
    per = int(math.ceil(cap / len(classes)))
    keep = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        keep.append(rng.permutation(idx)[:per])
    out = np.concatenate(keep)
    if len(out) > cap:
        out = rng.permutation(out)[:cap]
    return np.sort(out)


def group_classes(class_vectors: torch.Tensor, n_groups: int, seed: int, restarts: int = 12) -> List[List[int]]:
    """A partition of the classes into at most ``n_groups`` groups by spherical k-means on the unit class vectors (best of several restarts); groups and members in ascending order."""
    v = torch.nn.functional.normalize(class_vectors.float(), dim=1)
    n = v.shape[0]
    k = max(1, min(int(n_groups), n))
    best, best_score = None, -1e9
    for r in range(restarts):
        c = spherical_kmeans(v, k, iters=20, seed=seed * 1000 + r)
        sim = v @ c.T
        score = float(sim.max(dim=1).values.sum())
        if score > best_score:
            best, best_score = sim.argmax(dim=1), score
    groups = [sorted(i for i in range(n) if int(best[i]) == j) for j in range(k)]
    return sorted((g for g in groups if g), key=lambda g: g[0])


def _file(dataset_dir: str, pattern: str) -> Path:
    base = HUB / dataset_dir / "snapshots"
    hits = sorted(base.glob(pattern)) if base.exists() else []
    if not hits:
        raise FileNotFoundError(f"dataset {dataset_dir.replace('datasets--', '')} ({pattern}) is not in the local cache {HUB}; this project never downloads")
    return hits[0]


def _clip(text) -> str:
    return " ".join(str(text).split()[:MAX_WORDS])


def _readme_names(dataset_dir: str, pattern: str, n: int) -> List[str]:
    text = _file(dataset_dir, "*/README.md").read_text(encoding="utf-8")
    names = [name.strip() for _, name in sorted(re.findall(pattern, text, flags=re.M), key=lambda p: int(p[0]))]
    if len(names) != n:
        raise RuntimeError(f"expected {n} label names in {dataset_dir}, found {len(names)}")
    return names


# ------------------------------------------------------------------------------------------------------------------------------ raw text data
def _raw(name: str) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    """(train texts, train labels, test texts, test labels, class names) of one text domain."""
    def parquet(ds, pat, text_col="text", label_col="label"):
        df = pd.read_parquet(_file(ds, pat))
        return [_clip(t) for t in df[text_col]], [int(v) for v in df[label_col]]

    if name == "banking":
        a, b = parquet("datasets--banking77", "*/data/train-*.parquet"), parquet("datasets--banking77", "*/data/test-*.parquet")
        return a[0], a[1], b[0], b[1], [s.replace("_", " ") for s in _readme_names("datasets--banking77", r"^\s+'(\d+)': (\S+)\s*$", 77)]
    if name == "agnews":
        a, b = parquet("datasets--ag_news", "*/data/train-*.parquet"), parquet("datasets--ag_news", "*/data/test-*.parquet")
        return a[0], a[1], b[0], b[1], _readme_names("datasets--ag_news", r"^\s+'(\d+)': (.+)$", 4)
    if name == "emotion":
        a, b = parquet("datasets--emotion", "*/split/train-*.parquet"), parquet("datasets--emotion", "*/split/test-*.parquet")
        return a[0], a[1], b[0], b[1], list(EMOTION)
    if name == "sst2":
        a = parquet("datasets--glue", "*/sst2/train-*.parquet", "sentence")
        b = parquet("datasets--glue", "*/sst2/validation-*.parquet", "sentence")                  # the GLUE test labels are hidden
        return a[0], a[1], b[0], b[1], ["negative", "positive"]
    if name == "newsgroups":
        parts, names = [], {}
        for split in ("train", "test"):
            texts, labels = [], []
            for line in _file("datasets--SetFit--20_newsgroups", f"*/{split}.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                t = _clip(row["text"])
                if len(t.split()) >= 5:
                    texts.append(t)
                    labels.append(int(row["label"]))
                    names[int(row["label"])] = str(row["label_text"])
            parts.append((texts, labels))
        return parts[0][0], parts[0][1], parts[1][0], parts[1][1], [names[i].replace(".", " ") for i in sorted(names)]
    if name == "svamp":
        frames = [pd.read_parquet(_file("datasets--ChilleD--SVAMP", f"*/data/{s}-*.parquet")) for s in ("train", "test")]
        kinds = sorted({t.replace("Divison", "Division") for f in frames for t in f["Type"]})
        out = []
        for f in frames:
            out += [[_clip(f"{b} {q}") for b, q in zip(f["Body"], f["Question"])], [kinds.index(t.replace("Divison", "Division")) for t in f["Type"]]]
        return out[0], out[1], out[2], out[3], kinds
    raise ValueError(f"unknown text domain {name!r}; the domains are {list(ALL_DOMAINS)}")


# ------------------------------------------------------------------------------------------------------------------------------ the domains
def _text_domain(name: str, fe, seed: int, cap_train: int, cap_test: int) -> Domain:
    rng = np.random.default_rng([seed, 0x444F4D, zlib.crc32(name.encode())])
    tr_x, tr_y, te_x, te_y, names = _raw(name)
    it, ie = stratified(tr_y, cap_train, rng), stratified(te_y, cap_test, rng)
    tr_x, tr_y = [tr_x[i] for i in it], np.asarray(tr_y)[it]
    te_x, te_y = [te_x[i] for i in ie], np.asarray(te_y)[ie]
    x_tr, x_te = fe.text(tr_x), fe.text(te_x)
    n_cls = len(names)
    means = torch.stack([torch.nn.functional.normalize(x_tr[torch.from_numpy(tr_y == c)].mean(0), dim=0) if (tr_y == c).any() else torch.zeros(x_tr.shape[1]) for c in range(n_cls)])
    groups = group_classes(means, N_GROUPS[name], seed)
    regime_of = np.zeros(n_cls, dtype=np.int64)
    for r, g in enumerate(groups):
        regime_of[g] = r
    return Domain(name, "text", x_tr, torch.from_numpy(tr_y.astype(np.int64)), torch.from_numpy(regime_of[tr_y]), x_te, torch.from_numpy(te_y.astype(np.int64)),
                  torch.from_numpy(regime_of[te_y]), n_cls, names, [f"{name}_g{r}" for r in range(len(groups))],
                  {"groups": groups, "texts_train": tr_x, "texts_test": te_x})


def _arith_domain(fe, seed: int, cap_train: int, cap_test: int) -> Domain:
    rng = np.random.default_rng([seed, 0x444F4D, zlib.crc32(b"arith")])
    n_ops = len(arith.OPS)
    parts = {"train": [], "test": []}
    problems = {"train": [], "test": []}
    for split, cap in (("train", cap_train), ("test", cap_test)):
        per_op = max(1, cap // n_ops)
        for oi in range(n_ops):
            n_prob = int(math.ceil(per_op / arith.WIDTH[arith.OPS[oi]]))
            p = arith.sample_problems(rng, n_prob, (oi,), ARITH_LENS, ARITH_LENS, split)
            ps = arith.ProblemSet(**p)
            problems[split].append(p)
            parts[split].append((ps, oi))
    out = {}
    for split, cap in (("train", cap_train), ("test", cap_test)):
        xs, ys, rs, pid = [], [], [], []
        base = 0
        for ps, oi in parts[split]:
            take = ps.n_samples if split == "test" else min(ps.n_samples, max(1, cap // n_ops))      # test keeps whole problems (exact match needs them)
            xs.append(fe.digits(ps.x(np.arange(take))))
            ys.append(torch.from_numpy(ps.y[:take].astype(np.int64)))
            rs.append(torch.full((take,), oi, dtype=torch.long))
            pid.append(torch.from_numpy(base + ps.pi[:take]))
            base += len(ps)
        out[split] = (torch.cat(xs), torch.cat(ys), torch.cat(rs), torch.cat(pid))
    cat = lambda split: {k: np.concatenate([p[k] for p in problems[split]]) for k in ("op", "a", "b", "la", "lb")}
    return Domain("arith", "digits", out["train"][0], out["train"][1], out["train"][2], out["test"][0], out["test"][1], out["test"][2], arith.N_CLASS,
                  [str(i) for i in range(10)] + ["PAD", "NEG"], list(arith.OPS),
                  {"problems_train": cat("train"), "problems_test": cat("test"), "problem_id_test": out["test"][3]})


def load_domain(name: str, fe, seed: int, cap_train: int = 4000, cap_test: int = 1500) -> Domain:
    if name == "arith":
        return _arith_domain(fe, seed, cap_train, cap_test)
    if name in TEXT_DOMAINS:
        return _text_domain(name, fe, seed, cap_train, cap_test)
    raise ValueError(f"unknown domain {name!r}; the domains are {list(ALL_DOMAINS)}")


def ood_sets(fe, seed: int, n: int = 300) -> Dict[str, torch.Tensor]:
    """Text the pool never trains on, of three registers that none of the domains has: children's stories, encyclopedic sentences, school science questions."""
    rng = np.random.default_rng([seed, 0x4F4F44])
    sources = {
        "stories": [_clip(t) for t in pd.read_parquet(_file("datasets--roneneldan--TinyStories", "*/data/validation-*.parquet"))["text"] if len(str(t).split()) >= 12],
        "wiki": [_clip(t) for t in pd.read_parquet(_file("datasets--wikitext", "*/wikitext-2-raw-v1/test-*.parquet"))["text"]
                 if len(str(t).split()) >= 12 and not str(t).strip().startswith("=")],
        "science_questions": [_clip(t) for t in pd.read_parquet(_file("datasets--ai2_arc", "*/ARC-Challenge/test-*.parquet"))["question"] if len(str(t).split()) >= 5],
    }
    out = {}
    for name, texts in sources.items():
        pick = rng.permutation(len(texts))[:n]
        out[name] = fe.text([texts[i] for i in pick])
    return out

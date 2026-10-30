"""Domain worlds for the general routing lab.

A domain world is what the router is given and tested on, built from any labelled corpus by one pure function: a few classes are TAUGHT (their teaching messages, a held-out calibration set
and a test set of known messages), the rest are never taught (a test set of unknown messages). The teacher's name for a class is its label name made readable; nothing else about the
domain enters. Loaders read the local Hugging Face cache only (this project never downloads); a missing dataset is a loud FileNotFoundError.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import cards as CD

HUB = Path(os.environ.get("KS_HF_HUB", Path.home() / ".cache" / "huggingface" / "hub"))
N_EXEMPLARS = 2
MAX_WORDS = 60
Labelled = Tuple[List[str], List[int]]


@dataclass
class DomainWorld:
    name: str
    taught: List[str]                               # the cards' names, in roster order
    cards: List[CD.Card]
    teach: Dict[str, List[str]]                     # name -> teaching messages
    known_cal: List[Tuple[str, int]]                # held-out messages of the taught classes (index into taught): calibration, never teaching, never test
    known_test: List[Tuple[str, int]]
    unknown_test: List[str]                         # messages of never-taught classes
    unknown_labels: List[str]                       # the never-taught class of each unknown message (for analysis only)
    states: List[str]


def readable(label: str) -> str:
    return " ".join(label.replace("_", " ").replace(".", " ").split())


def build_labelled_world(name: str, teach: Labelled, test: Labelled, label_names: Sequence[str], n_taught: int, n_teach: int, n_cal: int, n_known: int, n_unknown: int,
                         seed: int) -> DomainWorld:
    n_cls = len(label_names)
    if not 1 <= n_taught < n_cls:
        raise ValueError(f"need between 1 and {n_cls - 1} taught classes, got {n_taught}")
    rng = np.random.default_rng([seed, 0x47454E])
    tr_x, tr_y = np.asarray(teach[0], dtype=object), np.asarray(teach[1])
    te_x, te_y = np.asarray(test[0], dtype=object), np.asarray(test[1])
    taught_ids = [int(c) for c in rng.permutation(n_cls)[:n_taught]]
    names = [readable(label_names[c]) for c in taught_ids]
    teach_msgs: Dict[str, List[str]] = {}
    cal: List[Tuple[str, int]] = []
    for i, c in enumerate(taught_ids):
        pool = np.flatnonzero(tr_y == c)
        if len(pool) < n_teach + n_cal:
            raise ValueError(f"class {label_names[c]!r} has {len(pool)} teaching messages, need {n_teach} to teach and {n_cal} to calibrate")
        pool = pool[rng.permutation(len(pool))]
        teach_msgs[names[i]] = [str(tr_x[j]) for j in pool[:n_teach]]
        cal += [(str(tr_x[j]), i) for j in pool[n_teach:n_teach + n_cal]]
    per, extra = divmod(n_known, n_taught)
    known: List[Tuple[str, int]] = []
    for i, c in enumerate(taught_ids):
        want = per + (1 if i < extra else 0)
        pool = np.flatnonzero(te_y == c)
        if len(pool) < want:
            raise ValueError(f"class {label_names[c]!r} has {len(pool)} test messages, need {want}")
        known += [(str(te_x[j]), i) for j in rng.permutation(pool)[:want]]
    known = [known[int(j)] for j in rng.permutation(len(known))]
    rest = np.flatnonzero(~np.isin(te_y, taught_ids))
    if len(rest) < n_unknown:
        raise ValueError(f"only {len(rest)} test messages of never-taught classes, need {n_unknown}")
    pick = rest[rng.permutation(len(rest))[:n_unknown]]
    cards = [CD.Card(n, tuple(teach_msgs[n][:N_EXEMPLARS])) for n in names]
    return DomainWorld(name, names, cards, teach_msgs, cal, known, [str(te_x[j]) for j in pick], [str(label_names[int(te_y[j])]) for j in pick], ["solid"] * n_taught)


# ------------------------------------------------------------------------------------------------------------------------------------ loaders
def _file(dataset_dir: str, pattern: str) -> Path:
    base = HUB / dataset_dir / "snapshots"
    hits = sorted(base.glob(pattern)) if base.exists() else []
    if not hits:
        raise FileNotFoundError(f"dataset {dataset_dir.replace('datasets--', '')} ({pattern}) is not in the local cache {HUB}; this project never downloads")
    return hits[0]


def _clip(text: str) -> str:
    return " ".join(str(text).split()[:MAX_WORDS])


def _label_names(readme: Path, pattern: str) -> List[str]:
    pairs = re.findall(pattern, readme.read_text(encoding="utf-8"), flags=re.M)
    return [name.strip() for _, name in sorted(pairs, key=lambda p: int(p[0]))]


def _parquet(dataset_dir: str, readme_pattern: str, n_names: int) -> Tuple[Labelled, Labelled, List[str]]:
    parts = []
    for split in ("train", "test"):
        df = pd.read_parquet(_file(dataset_dir, f"*/data/{split}-*.parquet"))
        parts.append(([_clip(t) for t in df["text"]], [int(v) for v in df["label"]]))
    names = _label_names(_file(dataset_dir, "*/README.md"), readme_pattern)
    if len(names) != n_names:
        raise RuntimeError(f"expected {n_names} label names for {dataset_dir}, found {len(names)}")
    return parts[0], parts[1], names


def load_banking() -> Tuple[Labelled, Labelled, List[str]]:
    return _parquet("datasets--banking77", r"^\s+'(\d+)': (\S+)\s*$", 77)


def load_agnews() -> Tuple[Labelled, Labelled, List[str]]:
    return _parquet("datasets--ag_news", r"^\s+'(\d+)': (.+)$", 4)


def load_newsgroups() -> Tuple[Labelled, Labelled, List[str]]:
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
    return parts[0], parts[1], [names[i] for i in sorted(names)]

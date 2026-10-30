"""Domain-level routing: which domain does a message belong to? Three answers compared on the text domains of the suite.

* the pool's own routing: the expert a test message is sent to belongs to the domain it trained on most (``expert_domains``);
* a centroid rule: the nearest domain centroid of the training data on the sphere (look only);
* the Router layer's asking reader: a language model is shown the message and one card per domain (the teacher's name and two training messages) and answers "which of these, or none".
The asking reader reads words, so only the text domains take part (digit codes are not messages). Nothing here trains anything.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from .runner import RunRecord

_ROUTING_GENERAL = Path(__file__).resolve().parents[1] / "research" / "knowledge_status" / "routing_general"

TEACHER_NAMES = {
    "banking": "a customer's question or request to a bank about their account, card or payments",
    "agnews": "a news headline or short news story",
    "newsgroups": "a post from an online discussion group",
    "emotion": "a short personal statement that expresses an emotion",
    "sst2": "a sentence from a movie review",
    "svamp": "a short arithmetic word problem",
}


def expert_domains(rec: RunRecord, regimes) -> Dict[int, str]:
    """The domain each expert trained on most, from the training assignments."""
    reg, asg = rec.arrays["train_regime"], rec.arrays["train_assigned"]
    out: Dict[int, str] = {}
    for e in np.unique(asg[asg >= 0]):
        out[int(e)] = Counter(regimes[int(r)].domain for r in reg[asg == e]).most_common(1)[0][0]
    return out


def _summarise(hits: Dict[str, list]) -> dict:
    allv = [v for vs in hits.values() for v in vs]
    return {"accuracy": float(np.mean(allv)) if allv else float("nan"), "per_domain": {d: float(np.mean(v)) for d, v in hits.items() if v}}


def pool_domain_routing(rec: RunRecord, regimes, text_domains: Sequence[str]) -> dict:
    ed = expert_domains(rec, regimes)
    hits = {d: [] for d in text_domains}
    for r, e in zip(rec.arrays["test_regime"], rec.arrays["test_expert"]):
        d = regimes[int(r)].domain
        if d in hits:
            hits[d].append(ed.get(int(e)) == d)
    return _summarise(hits)


def centroid_domain_routing(stream, text_domains: Sequence[str]) -> dict:
    sums: Dict[str, torch.Tensor] = {}
    for ph in stream.phases:
        for r in ph.regime.unique().tolist():
            d = stream.regimes[r].domain
            s = ph.x[ph.regime == r].sum(0)
            sums[d] = sums[d] + s if d in sums else s
    names = [d for d in text_domains if d in sums]
    C = torch.nn.functional.normalize(torch.stack([sums[d] for d in names]), dim=1)
    hits = {d: [] for d in names}
    for reg in stream.regimes.values():
        if reg.domain in hits:
            pred = (reg.x_te @ C.T).argmax(1)
            hits[reg.domain] += (pred == names.index(reg.domain)).tolist()
    return _summarise(hits)


def domain_cards(domains, names: Dict[str, str], order: Sequence[str]):
    if str(_ROUTING_GENERAL) not in sys.path:
        sys.path.insert(0, str(_ROUTING_GENERAL))
        sys.path.insert(0, str(_ROUTING_GENERAL.parent))
        sys.path.insert(0, str(_ROUTING_GENERAL.parents[2]))
    import cards as CD
    return [CD.Card(names[d], tuple(str(t) for t in domains[d].meta["texts_train"][:2])) for d in order]


def asking_domain_routing(stream, domains, reader, order: Sequence[str], per_regime: int = 100) -> dict:
    """``reader.probabilities(texts)`` gives (n, len(order) + 1) probabilities, the last column being 'none of these'. A none answer is a miss and is counted as a refusal."""
    hits = {d: [] for d in order}
    refused = []
    for reg in stream.regimes.values():
        if reg.domain not in hits:
            continue
        idx = np.asarray(reg.meta["test_index"])[:per_regime]
        texts = [domains[reg.domain].meta["texts_test"][int(i)] for i in idx]
        pred = np.asarray(reader.probabilities(texts)).argmax(1)
        hits[reg.domain] += (pred == list(order).index(reg.domain)).tolist()
        refused += (pred == len(order)).tolist()
    out = _summarise(hits)
    out["refused"] = float(np.mean(refused)) if refused else float("nan")
    return out


def ood_texts(seed: int, n: int = 300) -> Dict[str, list]:
    """The texts of ``domains.ood_sets`` (same sources, same draw), for readers that read words rather than vectors."""
    import pandas as pd
    from . import domains as DM
    rng = np.random.default_rng([seed, 0x4F4F44])
    sources = {
        "stories": [DM._clip(t) for t in pd.read_parquet(DM._file("datasets--roneneldan--TinyStories", "*/data/validation-*.parquet"))["text"] if len(str(t).split()) >= 12],
        "wiki": [DM._clip(t) for t in pd.read_parquet(DM._file("datasets--wikitext", "*/wikitext-2-raw-v1/test-*.parquet"))["text"]
                 if len(str(t).split()) >= 12 and not str(t).strip().startswith("=")],
        "science_questions": [DM._clip(t) for t in pd.read_parquet(DM._file("datasets--ai2_arc", "*/ARC-Challenge/test-*.parquet"))["question"] if len(str(t).split()) >= 5],
    }
    return {name: [texts[i] for i in rng.permutation(len(texts))[:n]] for name, texts in sources.items()}


def classifier_domain_routing(stream, text_domains: Sequence[str]) -> dict:
    """A logistic regression on the sphere, trained to name the DOMAIN of a message (the best look-only domain router these vectors allow)."""
    from sklearn.linear_model import LogisticRegression
    names = [d for d in text_domains if any(r.domain == d for r in stream.regimes.values())]
    xs, ys = [], []
    for ph in stream.phases:
        for r in ph.regime.unique().tolist():
            d = stream.regimes[r].domain
            if d in names and bool(ph.first[ph.regime == r].any()):
                m = (ph.regime == r) & ph.first
                xs.append(ph.x[m])
                ys += [names.index(d)] * int(m.sum())
    clf = LogisticRegression(max_iter=2000, C=10.0).fit(torch.cat(xs).numpy(), np.asarray(ys))
    hits = {d: [] for d in names}
    for reg in stream.regimes.values():
        if reg.domain in hits:
            hits[reg.domain] += (clf.predict(reg.x_te.numpy()) == names.index(reg.domain)).tolist()
    return _summarise(hits)

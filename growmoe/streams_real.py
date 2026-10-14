"""Real-data continual learning streams (docs/superpowers/handoff/GROWMOE_HANDOFF.md M5).

banking_incremental: banking77 text classification grouped into class-incremental regimes
via spherical k-means on intent embeddings, with revisits.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from growmoe.fingerprint import spherical_kmeans
from growmoe.frontend import HashFrontend
from growmoe.streams import Regime, Segment, Stream, _segment

_PARQUET_CACHE = None


def _load_banking77() -> Tuple[pd.DataFrame, pd.DataFrame]:
    global _PARQUET_CACHE
    if _PARQUET_CACHE is not None:
        return _PARQUET_CACHE
    base = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--banking77"
    if not base.exists():
        raise FileNotFoundError(f"Local Hugging Face cache for banking77 not found at {base}")
    train_files = list(base.glob("snapshots/*/data/train-00000-of-00001.parquet"))
    test_files = list(base.glob("snapshots/*/data/test-00000-of-00001.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"banking77 parquet files missing in {base}")
    df_train = pd.read_parquet(train_files[0])
    df_test = pd.read_parquet(test_files[0])
    _PARQUET_CACHE = (df_train, df_test)
    return _PARQUET_CACHE


def banking_incremental(
    seed: int = 0,
    frontend=None,
    n_regimes: int = 6,
    n_train: int = 400,
    n_test: int = 150,
    revisit: Sequence[int] = (2, 0, 4),
    revisit_size: int = 200,
) -> Stream:
    """Class-incremental stream from banking77 with disjoint label blocks and revisits.

    Args:
        seed: random seed for clustering and sample ordering.
        frontend: callable converting string list to (N, d) unit-length tensor. Default HashFrontend().
        n_regimes: number of clustered intent groups.
        n_train: number of training documents sampled per first visit.
        n_test: maximum number of test documents per regime test set.
        revisit: tuple of regime IDs to revisit after initial passes.
        revisit_size: number of documents sampled per revisit.
    """
    fe = frontend if frontend is not None else HashFrontend()
    df_train, df_test = _load_banking77()

    # 1. Embed training texts and compute unit mean embedding per intent
    z_train = fe(df_train["text"].tolist())
    d = z_train.shape[1]
    y_train_orig = torch.tensor(df_train["label"].values, dtype=torch.long)

    intent_means = torch.stack([
        F.normalize(z_train[y_train_orig == i].mean(dim=0), dim=0)
        for i in range(77)
    ])

    # 2. Cluster 77 intent means into n_regimes groups via spherical k-means
    centroids = spherical_kmeans(intent_means, n_regimes, seed=seed)
    assign = (intent_means @ centroids.T).argmax(dim=1)
    groups = [
        sorted([i for i in range(77) if assign[i].item() == r])
        for r in range(n_regimes)
    ]

    # 3. Disjoint class-incremental label blocks (n_classes = 77)
    intent_order = [intent for g in groups for intent in g]
    intent_to_label = {intent: pos for pos, intent in enumerate(intent_order)}

    # 4. Embed test texts
    z_test = fe(df_test["text"].tolist())

    # 5. Build regimes with official test split
    regimes = {}
    for r in range(n_regimes):
        te_indices = np.array([idx for idx, lbl in enumerate(df_test["label"]) if lbl in set(groups[r])], dtype=np.int64)
        rng_te = np.random.default_rng([seed, 0x5445, r])
        perm_te = rng_te.permutation(len(te_indices))
        n_te = min(n_test, len(te_indices))
        pick_te = te_indices[perm_te[:n_te]]

        x_te_r = z_test[pick_te]
        y_te_r = torch.tensor([intent_to_label[df_test["label"].iloc[idx]] for idx in pick_te], dtype=torch.long)
        ids_te_r = torch.tensor([len(df_train) + idx for idx in pick_te], dtype=torch.long)
        labels_r = [intent_to_label[i] for i in groups[r]]

        regimes[r] = Regime(
            id=r,
            name=f"banking_r{r}",
            x_te=x_te_r,
            y_te=y_te_r,
            te_ids=ids_te_r,
            meta={"intents": groups[r], "labels": labels_r},
        )

    # 6. Build training pools and segments (schedule = range(n_regimes) + revisit)
    pools = {}
    for r in range(n_regimes):
        tr_indices = np.array([idx for idx, lbl in enumerate(df_train["label"]) if lbl in set(groups[r])], dtype=np.int64)
        x_tr_r = z_train[tr_indices]
        y_tr_r = torch.tensor([intent_to_label[df_train["label"].iloc[idx]] for idx in tr_indices], dtype=torch.long)
        rng_tr = np.random.default_rng([seed, 0x504F, r])
        order = rng_tr.permutation(len(tr_indices))
        pools[r] = (x_tr_r, y_tr_r, tr_indices, order)

    schedule = list(range(n_regimes)) + list(revisit)
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        n = n_train if v == 0 else revisit_size
        start = 0 if v == 0 else n_train + (v - 1) * revisit_size
        x_tr_r, y_tr_r, ids_tr_r, order = pools[r]
        pick = order[(start + np.arange(n)) % len(order)]
        segments.append(_segment(r, v == 0, x_tr_r[pick], y_tr_r[pick], ids_tr_r[pick]))

    return Stream(
        name="banking_incremental",
        d=d,
        n_classes=77,
        regimes=regimes,
        segments=segments,
        meta={"schedule": schedule, "groups": groups},
    )

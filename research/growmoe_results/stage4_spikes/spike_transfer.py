"""Stage-4 spike B: does "the parent already predicts the new cluster's labels" separate a new function from a known one?  (throwaway; no product file touched)

The stage-4 candidate for the spawn-or-extend decision is a held-out, label-aware transfer test: before any training, compare the parent's cross-entropy on the cluster's labelled
samples with the parent's own honest (prequential-like) losses on fresh samples of its regime; if the cluster's mean loss is significantly higher (exact permutation test), the
parent does not explain the cluster: spawn; otherwise extend. This spike measures its power and its false-spawn rate on the saved pools of the operator-by-operator runs: expert 0
holds max, add and sub; clusters of size 32 / 64 / 128 are drawn from held-out test examples of each operator. Reference = 512 held-out example rows of the expert's own operators.

    python spike_transfer.py
"""

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / "diagnosis"
sys.path.insert(0, str(DIAG))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402

ALPHA = 0.01
N_PERM = 999


def perm_p_greater(cluster, ref, rng):
    """One-sided exact-style permutation p-value that mean(cluster) is larger than mean(ref) under exchangeability."""
    obs = cluster.mean() - ref.mean()
    pooled = np.concatenate([cluster, ref])
    n = len(cluster)
    c = 0
    for _ in range(N_PERM):
        perm = rng.permutation(len(pooled))
        c += (pooled[perm[:n]].mean() - pooled[perm[n:]].mean()) >= obs
    return (1 + c) / (1 + N_PERM)


def main():
    test = L.build_test(200)
    x = torch.from_numpy(D.unit(test.x()))
    y = torch.from_numpy(test.y).long()
    op = test.op[test.pi]
    out = []
    for f in sorted((DIAG / "seqprobe3").glob("default_s*.pool.pkl")):
        seed = int(f.name.split("_s")[1].split(".")[0])
        meta = next(r for r in json.loads((DIAG / "router.json").read_text(encoding="utf-8")) if r["variant"] == "default" and r["seed"] == seed)
        own = [A.OPS.index(o) for o, e in meta["operator_to_expert"].items() if e == 0]       # operators held by expert 0 (max, add, sub)
        with open(f, "rb") as fh:                                       # written by diagnosis/run_seqprobe.py in this project; never loaded from outside
            pool = pickle.load(fh)
        net = pool.experts[0].module.eval()
        with torch.no_grad():
            loss = F.cross_entropy(net(x), y, reduction="none").numpy()
        rng = np.random.default_rng(seed)
        known_rows = np.flatnonzero(np.isin(op, own))
        ref_idx = rng.choice(known_rows, 512, replace=False)
        ref = loss[ref_idx]
        pool_rows = np.setdiff1d(np.arange(len(op)), ref_idx)
        row = {"seed": seed, "reference_mean": float(ref.mean()), "per_operator_mean_loss": {A.OPS[o]: float(loss[op == o].mean()) for o in range(A.N_OPS)}, "cells": {}}
        print(f"== default seed {seed}: expert 0 holds {[A.OPS[o] for o in own]}; reference loss {ref.mean():.3f}; mean loss per operator " + " ".join(f"{A.OPS[o]} {loss[op == o].mean():.2f}" for o in range(A.N_OPS)), flush=True)
        for o in range(A.N_OPS):
            rows_o = np.intersect1d(np.flatnonzero(op == o), pool_rows)
            for n in (32, 64, 128):
                ps = []
                for _ in range(100):
                    c = loss[rng.choice(rows_o, n, replace=False)]
                    ps.append(perm_p_greater(c, ref, rng))
                ps = np.array(ps)
                row["cells"][f"{A.OPS[o]}-{n}"] = {"known": o in own, "spawn_rate": float((ps < ALPHA).mean()), "median_p": float(np.median(ps))}
            print(f"   cluster of {A.OPS[o]:<4s} ({'KNOWN' if o in own else 'new  '}): spawn decision at alpha {ALPHA}: " + "  ".join(f"n={n}: {100 * row['cells'][f'{A.OPS[o]}-{n}']['spawn_rate']:5.1f}%" for n in (32, 64, 128)), flush=True)
        out.append(row)
    D.atomic_write_text(HERE / "spike_transfer.json", json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

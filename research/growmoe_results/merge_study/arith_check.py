"""Merge study, realism check on REAL arithmetic experts (throwaway; no product file touched).

The saved operator-by-operator pool (diagnosis/seqprobe3/forced_spawn_s1: six experts of 512 hidden units, one per operator, trained under the pool's recipe) provides
  * distinct pairs: every ordered pair of experts that learned different operators (merging them would be wrong), and
  * duplicates: an expert plus a shrink-perturbed copy of it (a false spawn), both then trained on different fresh problems of the same operator.
Archived experts in the pool have no labelled replay any more, so the "stored labelled samples" of a pair are 256 fresh generated problems of the expert's operator and its honest
reference is its loss on 512 other fresh problems (what a live prequential window would hold). Nothing here has been seen by any expert.

    python arith_check.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "research" / "growmoe_results" / "arith"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import arith_lib as L  # noqa: E402
from growmoe import arith as A  # noqa: E402
from growmoe.settest import generator  # noqa: E402
from growmoe.transfer import cluster_conflict  # noqa: E402

POOL = ROOT / "research" / "growmoe_results" / "diagnosis" / "seqprobe3" / "forced_spawn_s1.pool.pkl"
OPS = list(range(A.N_OPS))


def unit(x):
    return F.normalize(torch.from_numpy(np.ascontiguousarray(x)).float(), dim=1)


def op_rows(train, o):
    return np.flatnonzero(train.op[train.pi] == o)


def take(train, idx):
    return unit(train.x(idx)), torch.from_numpy(train.y[idx]).long()


def acc(m, x, y):
    with torch.no_grad():
        return float((m(x).argmax(1) == y).float().mean())


def loss_each(m, x, y):
    with torch.no_grad():
        return F.cross_entropy(m(x), y, reduction="none")


def direction(src, dst_x, dst_y, ref_losses, seed):
    """src's loss on dst's stored labelled samples against dst's honest reference: excess, exact permutation p, one-sided 95 % upper bound."""
    losses = loss_each(src, dst_x, dst_y)
    ref = torch.as_tensor(ref_losses, dtype=torch.float32)
    _, p, diff = cluster_conflict(losses, ref, 0.01, 999, 0.0, generator(4243, seed))
    se = float(torch.sqrt(losses.var() / len(losses) + ref.var() / len(ref)))
    return {"diff": diff, "p": p, "ucb": diff + 1.645 * se, "loss_src_on_dst": float(losses.mean()), "ref_dst": float(ref.mean())}


def agreement(m1, m2, x):
    with torch.no_grad():
        return float((m1(x).argmax(1) == m2(x).argmax(1)).float().mean())


def train_more(module, x_all, y_all, n_batches, seed, inner=5, lr=3e-3):
    """The pool's recipe on fresh problems of one operator: batch of 64 + an equal replay sample from a growing reservoir, 5 steps; honest per-sample losses before training."""
    from growmoe.replay import Reservoir
    g = np.random.default_rng(seed)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    res = Reservoir(2000, x_all.shape[1], np.random.default_rng([seed, 1]))
    ref = []
    order = g.permutation(len(x_all))                                     # a stream never repeats a sample: batch k takes the next 64 rows (sampling WITH replacement made a third of the
    assert n_batches * 64 <= len(order)                                   # 'honest' losses come from samples the expert had already trained on, and its reference too good)
    for k in range(n_batches):
        idx = torch.from_numpy(order[k * 64:(k + 1) * 64])
        x, y = x_all[idx], y_all[idx]
        ref.extend(loss_each(module, x, y).tolist())
        rep = res.sample(64) if res.n else None
        for _ in range(inner):
            module.train_step(x, y, opt, replay=rep)
        res.add(x, y)
    return res, ref


def main() -> None:
    torch.set_num_threads(2)
    with open(POOL, "rb") as fh:                                          # written by this project's diagnosis runs; local file
        pool = pickle.load(fh)
    experts = {e: r for e, r in pool.experts.items()}
    train = L.build_train("FULL", 777, 3000)                              # fresh problems no pool has seen
    rows = {o: op_rows(train, o) for o in OPS}
    rng = np.random.default_rng(5)
    # which operator did each expert learn?
    fit = {}
    for e, r in experts.items():
        r.module.eval()
        fit[e] = [acc(r.module, *take(train, rng.choice(rows[o], 1500, replace=False))) for o in OPS]
    op_of = {e: int(np.argmax(fit[e])) for e in experts}
    print("expert -> operator:", {e: (A.OPS[op_of[e]], round(max(fit[e]), 2)) for e in experts}, flush=True)
    out = {"op_of": {str(e): A.OPS[o] for e, o in op_of.items()}, "fit": {str(e): v for e, v in fit.items()}, "distinct": [], "dups": []}
    data = {}
    for o in set(op_of.values()):
        idx = rng.permutation(rows[o])
        data[o] = {"store": take(train, idx[:256]), "ref": take(train, idx[256:768]), "train_a": take(train, idx[768:7000]), "train_b": take(train, idx[7000:13000])}
    # distinct pairs
    for ea in experts:
        for eb in experts:
            if ea == eb or op_of[ea] == op_of[eb]:
                continue
            ob = op_of[eb]
            ref_losses = loss_each(experts[eb].module, *data[ob]["ref"]).tolist()                      # eb's honest loss on fresh problems of its own operator
            d = direction(experts[ea].module, *data[ob]["store"], ref_losses, ea * 10 + eb)
            protos = torch.cat([experts[ea].meta["sig"].protos, experts[eb].meta["sig"].protos])
            d.update(a=ea, b=eb, op_a=A.OPS[op_of[ea]], op_b=A.OPS[ob], agree_protos=agreement(experts[ea].module, experts[eb].module, protos))
            out["distinct"].append(d)
    print("distinct pairs:", len(out["distinct"]), "median ucb %.2f min ucb %.2f max p %.3f" % (np.median([d["ucb"] for d in out["distinct"]]), min(d["ucb"] for d in out["distinct"]), max(d["p"] for d in out["distinct"])), flush=True)
    # duplicates of the experts that learned something
    for e in experts:
        if max(fit[e]) < 0.15:
            continue
        o = op_of[e]
        base = experts[e].module
        a = base.clone()
        b = base.shrink_perturb(lam=0.9, seed=e)
        res_a, ref_a = train_more(a, *data[o]["train_a"], 80, seed=e)
        res_b, ref_b = train_more(b, *data[o]["train_b"], 80, seed=e + 100)
        xa, ya = res_a.data
        xb, yb = res_b.data
        # "stored replay" of each = its reservoir (trained on); honest reference = its own prequential losses of the last 512 samples
        da = direction(b, xa[:256], ya[:256], ref_a[-512:], e)                      # B on A's stored data
        db = direction(a, xb[:256], yb[:256], ref_b[-512:], e + 1)                  # A on B's stored data
        protos = torch.cat([experts[e].meta["sig"].protos])
        fresh_x, fresh_y = data[o]["ref"]
        out["dups"].append({"expert": e, "op": A.OPS[o], "B_on_A_data": da, "A_on_B_data": db, "agree_fresh": agreement(a, b, fresh_x), "agree_protos": agreement(a, b, protos),
                            "acc_A": acc(a, fresh_x, fresh_y), "acc_B": acc(b, fresh_x, fresh_y)})
        print("dup of expert", e, A.OPS[o], "ucb B->A %.2f A->B %.2f  p %.3f / %.3f  agree fresh %.2f  acc %.2f / %.2f" % (da["ucb"], db["ucb"], da["p"], db["p"], out["dups"][-1]["agree_fresh"], out["dups"][-1]["acc_A"], out["dups"][-1]["acc_B"]), flush=True)
    tmp = HERE / "arith_check.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, HERE / "arith_check.json")
    d_ucb = [d["ucb"] for d in out["distinct"]]
    u = [x for dup in out["dups"] for x in (dup["B_on_A_data"]["ucb"], dup["A_on_B_data"]["ucb"])]
    print("\nSUMMARY  duplicates: upper bounds %s | distinct pairs: min %.2f median %.2f" % (np.round(u, 2).tolist(), min(d_ucb), np.median(d_ucb)))
    for delta in (0.1, 0.2, 0.3, 0.4, 0.6, 1.0):
        print("  delta %.1f: duplicates accepted %d/%d, distinct accepted %d/%d" % (delta, sum(x <= delta for x in u), len(u), sum(x <= delta for x in d_ucb), len(d_ucb)))


if __name__ == "__main__":
    main()

"""Study 3b: how much of the sequential pool's loss is routing, and does a trivial LEARNED router recover it?

Loads the final pool of each instrumented sequential run (``seqprobe3/*.pool.pkl``), reconstructs which expert trained on which operator from the recorded training assignment, trains a
multinomial logistic-regression router on (unit-length input -> expert that was trained on it) using the training examples (what a pool could keep as a by-product of its own
assignments), and answers every test example with the expert the router picks. Compared with: the pool's own gate (the shipped rule), and the routing oracle (best expert per operator).

    python analyse_router.py
"""

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402


def train_router(x, y, n_class, seed=0, steps=400, lr=0.1):
    torch.manual_seed(seed)
    w = torch.zeros(x.shape[1], n_class, requires_grad=True)
    b = torch.zeros(n_class, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).long()
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (4096,))
        loss = F.cross_entropy(xt[idx] @ w + b, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def main():
    test = L.build_test(200)
    xt_all = D.unit(test.x())
    op_te = test.op[test.pi]
    rows = []
    for f in sorted((HERE / "seqprobe3").glob("*_s*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        pk = f.with_suffix(".pool.pkl")
        if not pk.exists():
            continue
        # The pickle is an artefact that run_seqprobe.py wrote into this project's seqprobe3/ folder; it is loaded only from there and never from any outside source.
        with open(pk, "rb") as fh:
            pool = pickle.load(fh)
        eids = sorted(pool.experts)
        # which expert trained on each block (operator): the one that received most of the block's training examples
        blk = r["train_assignment_per_block"]
        op_expert = {}
        for k, o in enumerate(L.OP_ORDER):
            counts = {int(e): c for e, c in blk[k].items() if e != "-1"}
            op_expert[o] = max(counts, key=counts.get)
        idx_of = {e: i for i, e in enumerate(eids)}
        tr = L.build_train("FULL", r["seed"], 8000)
        x_tr = D.unit(tr.x())
        y_tr = np.array([idx_of[op_expert[o]] for o in tr.op[tr.pi]])
        w, b = train_router(x_tr, y_tr, len(eids), seed=r["seed"])
        with torch.no_grad():
            xt = torch.from_numpy(xt_all)
            choice = (xt @ w + b).argmax(1).numpy()
            preds = np.zeros(len(xt_all), dtype=np.int64)
            per_expert_pred = {}
            for e in eids:
                m = pool.experts[e].module
                m.eval()
                per_expert_pred[e] = m(xt).argmax(-1).numpy()
            for e in eids:
                sel = choice == idx_of[e]
                preds[sel] = per_expert_pred[e][sel]
        exact = test.exact(preds).reshape(A.N_OPS, -1).mean(-1)
        own = [float((choice[op_te == o] == idx_of[op_expert[o]]).mean()) for o in range(A.N_OPS)]
        row = {"variant": r["variant"], "seed": r["seed"], "operator_to_expert": {A.OPS[o]: int(e) for o, e in op_expert.items()},
               "shipped_gate_mean_exact": r["summary"]["mean_exact"], "oracle_mean_exact": float(np.mean(r["oracle_routed_exact_by_op"])),
               "learned_router_mean_exact": float(exact.mean()), "learned_router_exact_by_op": {A.OPS[o]: float(exact[o]) for o in range(A.N_OPS)},
               "learned_router_to_own_expert": {A.OPS[o]: own[o] for o in range(A.N_OPS)},
               "gate_rules_mean_exact": {k: float(np.mean(v["exact_by_op"])) for k, v in r["route_rules"].items()}}
        rows.append(row)
        print(f"[{r['variant']} s{r['seed']}] shipped gate {100 * row['shipped_gate_mean_exact']:.1f}  oracle {100 * row['oracle_mean_exact']:.1f}  learned router {100 * row['learned_router_mean_exact']:.1f}   "
              f"router sends to own expert: " + " ".join(f"{A.OPS[o]} {100 * own[o]:.0f}" for o in range(A.N_OPS)), flush=True)
    D.atomic_write_text((HERE / "router.json"), json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

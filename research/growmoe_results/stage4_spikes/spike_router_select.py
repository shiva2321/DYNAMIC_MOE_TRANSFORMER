"""Stage-4 spike A3: a router that CHOOSES its own rule by cross-validation on the stored prototypes.  (throwaway; no product file touched)

Spike A2 showed that an LDA router reaches the routing oracle on operator-by-operator arithmetic but is much WORSE than the pool's density gate on S1 and on real text. No single rule
wins everywhere, so the design question is whether the pool can tell, from what it already stores, which rule to trust. Candidates (all fitted on nothing but the experts' signature
prototypes, owner = the expert that holds the prototype): density (the pool's own vMF kernel density, argmax log-density), LDA (shared covariance), softmax regression (balanced classes),
5-nearest-prototype vote. Selection: 5-fold cross-validation over the prototypes (hold out a fifth of every expert's prototypes, fit on the rest, predict their owner); the candidate
with the best held-out owner accuracy wins, density wins ties and is the default unless another candidate is better by more than a margin.
Reported per unit: each candidate's CV owner accuracy and END-TO-END exact match (the experts answer what the candidate sends them), the selected candidate, the shipped gate, the oracle.

    python spike_router_select.py --streams arith,F-15,OV-0.6,OV-0.85,S1,S2-5,banking --seeds 100,101
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIAG = HERE.parent / "diagnosis"
for p in (HERE.parent / "rv2c", HERE.parent / "rv2a", HERE.parent / "m2a", HERE.parent / "m0", DIAG, ROOT):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import diag_lib as D  # noqa: E402
import run_rv2c as R  # noqa: E402
from growmoe.fairfight import run_fast  # noqa: E402
from growmoe.signature import VmfSignature  # noqa: E402

OUT = HERE / "router_select"
ARM = "A_set@12-0.5-100"
MARGIN = 0.01                                                     # another candidate must beat density's cross-validated owner accuracy by more than this


# ------------------------------------------------------------------------------------------------ candidate routers: fit(X, y, k) -> score(Z) (n, k), higher = more likely owner
def fit_density(X, y, k):
    sigs = []
    for c in range(k):
        s = VmfSignature(d=X.shape[1], max_protos=256, seed=c)
        s.update(torch.from_numpy(X[y == c]))
        s.fit()
        sigs.append(s)
    return lambda Z: torch.stack([s.score(torch.from_numpy(Z)) + s._log_c[s._kappa] for s in sigs], 1).numpy()


def fit_lda(X, y, k, reg=1e-3):
    X = X.astype(np.float64)
    mu = np.stack([X[y == c].mean(0) for c in range(k)])
    xc = X - mu[y]
    cov = xc.T @ xc / max(len(X) - k, 1) + reg * np.eye(X.shape[1])
    prec = np.linalg.inv(cov)
    prior = np.log(np.bincount(y, minlength=k) / len(y))
    w = prec @ mu.T
    b = -0.5 * np.einsum("kd,dk->k", mu, w) + prior
    return lambda Z: Z.astype(np.float64) @ w + b


def fit_softmax(X, y, k, steps=500, lr=0.05, l2=1e-3):
    g = torch.Generator().manual_seed(0)
    w = torch.zeros(X.shape[1], k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    cls_w = torch.from_numpy(len(y) / (k * np.maximum(np.bincount(y, minlength=k), 1))).float()           # balanced classes
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (min(2048, len(xt)),), generator=g)
        loss = F.cross_entropy(xt[idx] @ w + b, yt[idx], weight=cls_w) + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w, b = w.detach(), b.detach()
    return lambda Z: (torch.from_numpy(Z) @ w + b).numpy()


def fit_knn(X, y, k, nn=5):
    Xt = torch.from_numpy(X)

    def score(Z):
        sim = torch.from_numpy(Z) @ Xt.T
        top = sim.topk(min(nn, len(X)), dim=1)
        out = torch.zeros(len(Z), k)
        out.scatter_add_(1, torch.from_numpy(y)[top.indices], top.values.clamp_min(0) + 1e-6)
        return out.numpy()
    return score


CANDIDATES = {"density": fit_density, "lda": fit_lda, "softmax": fit_softmax, "knn5": fit_knn}


def cv_owner_accuracy(protos, owner, k, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(owner), dtype=int)
    for c in range(k):
        idx = np.flatnonzero(owner == c)
        fold_of[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % folds
    acc = {name: [] for name in CANDIDATES}
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        for name, fit in CANDIDATES.items():
            sc = fit(protos[tr], owner[tr], k)(protos[te])
            acc[name].append(float((sc.argmax(1) == owner[te]).mean()))
    return {name: float(np.mean(v)) for name, v in acc.items()}


def end_to_end(pool, eids, xs, ys, scorers):
    """Mean over test sets of accuracy when each candidate's choice is answered by the chosen expert ('free' mode)."""
    idx_of = {e: i for i, e in enumerate(eids)}
    res = {name: [] for name in scorers}
    res["oracle"], res["shipped"] = [], []
    for x, y in zip(xs, ys):
        xn = pool._normalize(x.float())
        with torch.no_grad():
            hits = torch.stack([(pool.experts[e].module.proba(xn).argmax(-1).cpu() == y).float() for e in eids], 1)
        for name, sc in scorers.items():
            choice = torch.from_numpy(sc(xn.numpy()).argmax(1))
            res[name].append(float(hits[torch.arange(len(y)), choice].mean()))
        res["oracle"].append(float(hits.mean(0).max()))
        res["shipped"].append(float((pool.predict(x).probs.argmax(-1).cpu() == y).float().mean()))
    return {k: float(np.mean(v)) for k, v in res.items()}


def unit_from_pool(label, pool, xs, ys, extra):
    eids = sorted(pool.experts)
    idx_of = {e: i for i, e in enumerate(eids)}
    protos = np.concatenate([pool.experts[e].meta["sig"].protos.numpy() for e in eids]).astype(np.float32)
    owner = np.concatenate([np.full(len(pool.experts[e].meta["sig"].protos), idx_of[e]) for e in eids])
    cv = cv_owner_accuracy(protos, owner, len(eids))
    scorers = {name: fit(protos, owner, len(eids)) for name, fit in CANDIDATES.items()}
    e2e = end_to_end(pool, eids, xs, ys, scorers)
    best = max(CANDIDATES, key=lambda n: cv[n] - (0 if n == "density" else MARGIN))
    selected = best if cv[best] > cv["density"] + MARGIN else "density"
    out = {"unit": label, "experts": len(eids), "cv_owner_accuracy": cv, "end_to_end": e2e, "selected": selected, "selected_exact": e2e["shipped"] if selected == "density" else e2e[selected], **extra}
    print(f"[{label}] experts {len(eids)} | CV " + " ".join(f"{n} {100 * v:.0f}" for n, v in cv.items()) + f" -> {selected} | exact: shipped {100 * e2e['shipped']:.1f} " + " ".join(f"{n} {100 * e2e[n]:.1f}" for n in CANDIDATES)
          + f" oracle {100 * e2e['oracle']:.1f} | selected {100 * out['selected_exact']:.1f}", flush=True)
    return out


def run_arith():
    from growmoe import arith as A
    from diag_lib import L
    test = L.build_test(100)
    xt = D.unit(test.x())
    for f in sorted((DIAG / "seqprobe3").glob("*_s*.pool.pkl")):
        variant, seed = f.name.split(".")[0].rsplit("_s", 1)
        stem = OUT / f"arith-{variant}_s{seed}"
        if D.valid_json(stem.with_suffix(".json")):
            continue
        with open(f, "rb") as fh:                                       # written by diagnosis/run_seqprobe.py in this project; never loaded from outside
            pool = pickle.load(fh)
        # per-OPERATOR test sets (the unit of evaluation here is the exact-match grid, so score per example and average over operators)
        op = test.op[test.pi]
        eids = sorted(pool.experts)
        idx_of = {e: i for i, e in enumerate(eids)}
        protos = np.concatenate([pool.experts[e].meta["sig"].protos.numpy() for e in eids]).astype(np.float32)
        owner = np.concatenate([np.full(len(pool.experts[e].meta["sig"].protos), idx_of[e]) for e in eids])
        cv = cv_owner_accuracy(protos, owner, len(eids))
        with torch.no_grad():
            xt_t = torch.from_numpy(xt)
            pred_e = {e: pool.experts[e].module.eval()(xt_t).argmax(-1).numpy() for e in eids}
        e2e = {}
        for name, fit in CANDIDATES.items():
            choice = fit(protos, owner, len(eids))(xt).argmax(1)
            pred = np.zeros(len(xt), dtype=np.int64)
            for e in eids:
                m = choice == idx_of[e]
                pred[m] = pred_e[e][m]
            e2e[name] = float(test.exact(pred).reshape(A.N_OPS, -1).mean())
        shipped = np.concatenate([pool.predict(torch.from_numpy(xt[i:i + 4096])).probs.argmax(-1).numpy() for i in range(0, len(xt), 4096)])
        e2e["shipped"] = float(test.exact(shipped).reshape(A.N_OPS, -1).mean())
        per_op_best = {e: test.exact(pred_e[e]).reshape(A.N_OPS, -1).mean(-1) for e in eids}
        e2e["oracle"] = float(np.mean([max(per_op_best[e][o] for e in eids) for o in range(A.N_OPS)]))
        best = max(CANDIDATES, key=lambda n: cv[n] - (0 if n == "density" else MARGIN))
        selected = best if cv[best] > cv["density"] + MARGIN else "density"
        out = {"unit": f"arith-{variant}-s{seed}", "experts": len(eids), "cv_owner_accuracy": cv, "end_to_end": e2e, "selected": selected, "selected_exact": e2e["shipped"] if selected == "density" else e2e[selected]}
        D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))
        print(f"[arith {variant} s{seed}] experts {len(eids)} | CV " + " ".join(f"{n} {100 * v:.0f}" for n, v in cv.items()) + f" -> {selected} | exact: shipped {100 * e2e['shipped']:.1f} "
              + " ".join(f"{n} {100 * e2e[n]:.1f}" for n in CANDIDATES) + f" oracle {100 * e2e['oracle']:.1f} | selected {100 * out['selected_exact']:.1f}", flush=True)


def run_stream(stream, seed):
    stem = OUT / f"{stream.replace(chr(46), chr(112))}_s{seed}"
    if D.valid_json(stem.with_suffix(".json")):
        return
    if stream == "banking":
        from growmoe.streams_real import banking_incremental
        from run_m0 import HybridFrontend
        from growmoe.config import PhyloCfg
        st = banking_incremental(seed=seed, frontend=HybridFrontend(d=128, seed=0))
        cfg = PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", d=st.d, n_classes=st.n_classes, seed=seed, inner_steps=20, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32, set_gate=True)
    else:
        st = R.build_stream(stream, seed, False)
        cfg = R.arm_cfg(ARM, st, seed, R.steps_for(stream))
    holder = []

    def make():
        holder.append(R.LoggingPool(cfg))
        return holder[0]

    torch.set_num_threads(3)
    run_fast(make, st, seed, eval_every="final")
    xs = [x for _, (x, y) in st.test_sets().items()]
    ys = [y.cpu() for _, (x, y) in st.test_sets().items()]
    out = unit_from_pool(f"{stream}-s{seed}", holder[0], xs, ys, {"regimes": len(st.regimes)})
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="arith,F-15,OV-0.6,OV-0.85,S1,S2-5,banking")
    ap.add_argument("--seeds", default="100,101")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for stream in a.streams.split(","):
        if stream == "arith":
            run_arith()
            continue
        for seed in [int(s) for s in a.seeds.split(",")]:
            run_stream(stream, seed)


if __name__ == "__main__":
    main()

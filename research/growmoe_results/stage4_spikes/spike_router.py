"""Stage-4 spike A: what does a REALISTIC learned router need?  (throwaway; no product file touched)

The diagnosis (E-030) trained its router on ALL 298,000 training examples. A pool does not have those: it has each expert's stored inputs (the 256 prototypes of its signature, and
the replay reservoir if the expert is not archived). This spike asks, on the saved final pools of the operator-by-operator runs:
  1. how accurate is routing when the router sees ONLY the signature prototypes (16 ... 256 per expert), or the reservoirs;
  2. which router: softmax regression, streaming LDA (closed form, shared covariance), nearest class mean, against the pool's own density rule;
  3. what exact match results end to end when the experts answer what the router sends them.

    python spike_router.py
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


def fit_softmax(x, y, k, seed=0, steps=400, lr=0.1, l2=1e-4):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(x.shape[1], k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).long()
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (min(2048, len(xt)),), generator=g)
        loss = F.cross_entropy(xt[idx] @ w + b, yt[idx]) + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w, b = w.detach(), b.detach()
    return lambda z: (torch.from_numpy(z) @ w + b).argmax(1).numpy()


def fit_lda(x, y, k, reg=1e-3):
    """Linear discriminant analysis: class means, ONE shared within-class covariance, class priors (closed form; sufficient statistics only)."""
    mu = np.stack([x[y == c].mean(0) for c in range(k)])
    xc = x - mu[y]
    cov = xc.T @ xc / max(len(x) - k, 1) + reg * np.eye(x.shape[1])
    prec = np.linalg.inv(cov)
    prior = np.log(np.bincount(y, minlength=k) / len(y))
    w = prec @ mu.T                                                    # (d, k)
    b = -0.5 * np.einsum("kd,dk->k", mu, w) + prior
    return lambda z: (z @ w + b).argmax(1)


def fit_ncm(x, y, k):
    mu = np.stack([x[y == c].mean(0) for c in range(k)])
    return lambda z: (z @ mu.T - 0.5 * (mu ** 2).sum(1)).argmax(1)


def main():
    test = L.build_test(100)
    xt = D.unit(test.x())
    op_te = test.op[test.pi]
    out = []
    for f in sorted((DIAG / "seqprobe3").glob("*_s*.pool.pkl")):
        variant, seed = f.name.split(".")[0].rsplit("_s", 1)
        meta = next(r for r in json.loads((DIAG / "router.json").read_text(encoding="utf-8")) if r["variant"] == variant and r["seed"] == int(seed))
        op_expert = {A.OPS.index(o): e for o, e in meta["operator_to_expert"].items()}
        with open(f, "rb") as fh:                                       # written by diagnosis/run_seqprobe.py in this project; never loaded from outside
            pool = pickle.load(fh)
        eids = sorted(pool.experts)
        idx_of = {e: i for i, e in enumerate(eids)}
        with torch.no_grad():
            xt_t = torch.from_numpy(xt)
            pred_e = {e: pool.experts[e].module.eval()(xt_t).argmax(-1).numpy() for e in eids}
        owner = np.array([idx_of[op_expert[o]] for o in op_te])        # index of the expert that owns each test example's operator

        def end_to_end(choice):
            pred = np.zeros(len(xt), dtype=np.int64)
            for e in eids:
                m = choice == idx_of[e]
                pred[m] = pred_e[e][m]
            return float(test.exact(pred).reshape(A.N_OPS, -1).mean())

        def data(kind, n):
            xs, ys = [], []
            rng = np.random.default_rng(7)
            for e in eids:
                rec = pool.experts[e]
                if kind == "protos":
                    v = rec.meta["sig"].protos.numpy()
                else:
                    v = rec.replay.data[0].numpy() if rec.replay is not None and rec.replay.n > 0 else rec.meta["sig"].protos.numpy()
                v = v[rng.permutation(len(v))[:n]]
                xs.append(v)
                ys.append(np.full(len(v), idx_of[e]))
            return np.concatenate(xs).astype(np.float32), np.concatenate(ys)

        row = {"variant": variant, "seed": int(seed), "experts": len(eids), "shipped_exact": float(test.exact(
            np.concatenate([pool.predict(torch.from_numpy(xt[i:i + 4096])).probs.argmax(-1).numpy() for i in range(0, len(xt), 4096)])).reshape(A.N_OPS, -1).mean()), "rows": []}
        for kind, n in (("protos", 16), ("protos", 32), ("protos", 64), ("protos", 128), ("protos", 256), ("reservoir", 20000)):
            x, y = data(kind, n)
            for name, fit in (("softmax", lambda x, y: fit_softmax(x, y, len(eids))), ("lda", lambda x, y: fit_lda(x, y, len(eids))), ("ncm", lambda x, y: fit_ncm(x, y, len(eids)))):
                pr = fit(x, y)
                choice = pr(xt)
                row["rows"].append({"data": f"{kind}-{n}", "router": name, "train_rows": int(len(x)), "routing_acc": float((choice == owner).mean()), "exact": end_to_end(choice)})
        row["oracle_exact"] = end_to_end(owner)
        out.append(row)
        print(f"== {variant} seed {seed}: {len(eids)} experts; shipped gate {100 * row['shipped_exact']:.1f}  oracle {100 * row['oracle_exact']:.1f}", flush=True)
        for r in row["rows"]:
            print(f"   {r['data']:<14s} {r['router']:<8s} train rows {r['train_rows']:6d}  routes right {100 * r['routing_acc']:5.1f} %   exact {100 * r['exact']:5.1f}", flush=True)
    D.atomic_write_text(HERE / "spike_router.json", json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

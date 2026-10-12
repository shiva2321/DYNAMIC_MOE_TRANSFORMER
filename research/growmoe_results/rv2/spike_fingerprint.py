"""THROWAWAY spike on DEV seeds (100-102): can a cheap per-expert fingerprint ("resonance") replace the exact gate scan?

The exact gate scores an input against EVERY stored prototype of EVERY expert (E x 256 x d multiply-adds). Candidates tested here, all
computed from what the trained pool already holds (prototypes, replay samples), on the repaired pool (arm_monitor_after_infancy=True):

  centroid     one von Mises-Fisher mean + concentration per expert (E x d)                       -- the simplest "resonance" key
  subproto-k   k spherical k-means centroids per expert, score = best cosine (E x k x d)          -- VSARouter-style
  wave-m       random Fourier ("wave") features, expert fingerprint = mean wave vector (m x d + E x m)
  ridge-lin    one linear "hypervector" per expert trained discriminatively on the stored exemplars (E x d)
  ridge-rff    the same on 1024 random Fourier features (1024 x d + 1024 x E)

Each is used (a) alone and (b) as a stage-1 SHORTLIST (top-k experts) followed by the exact gate on the shortlist only.
Reported: accuracy, recall of the exact choice inside the shortlist, multiply-adds per input, and the headroom (accuracy if every regime were answered by its own owner expert).

    python research/growmoe_results/rv2/spike_fingerprint.py
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "m2a"))
sys.path.insert(0, str(HERE.parent / "m0"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402
import torch.nn.functional as F                                                         # noqa: E402

import run_m2a as R                                                                     # noqa: E402
from growmoe.vmf import log_vmf_norm                                                    # noqa: E402

torch.set_num_threads(3)
TUNING = json.loads((HERE.parent / "m2a" / "tuning_m2a.json").read_text(encoding="utf-8"))
STREAMS = ["F-15", "F-30", "S2-5", "S2-30", "S1"]
SEEDS = (100, 101, 102)
KS = (1, 2, 3)


def train(stream, seed):
    st = R.get_stream(stream, seed, False)
    steps = TUNING["A_new"][R.FAMILY[stream]]["steps"]
    pool = R.factory("A_new", st, seed, {"steps": steps})()
    for x, y, ctx in st.algo_batches(64, seed):
        pool.observe(x, y, ctx)
    return st, pool


def spherical_kmeans(p, k, iters=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = p[torch.randperm(len(p), generator=g)[:k]].clone()
    for _ in range(iters):
        a = (p @ c.T).argmax(1)
        for j in range(len(c)):
            if (a == j).any():
                c[j] = F.normalize(p[a == j].mean(0), dim=0)
    return c


def build_scorers(pool, eids, xn, d, seed):
    protos = [pool.experts[e].meta["sig"].protos for e in eids]
    n_e = torch.tensor([len(p) for p in protos], dtype=torch.float32)
    S, macs = {}, {}
    E = len(eids)
    # --- centroid (single vMF)
    mean = torch.stack([p.mean(0) for p in protos])
    rbar = mean.norm(dim=1).clamp(1e-6, 1 - 1e-6)
    kap = (rbar * (d - rbar ** 2) / (1 - rbar ** 2)).clamp(1.0, 5000.0)
    mu = F.normalize(mean, dim=1)
    logc = torch.tensor([log_vmf_norm(d, float(k)) for k in kap])
    S["centroid"] = (xn @ mu.T) * kap + logc
    macs["centroid"] = E * d
    # --- sub-prototypes
    for k in (4, 8):
        cents = [spherical_kmeans(p, min(k, len(p)), seed=seed) for p in protos]
        S[f"subproto-{k}"] = torch.stack([(xn @ c.T).max(1).values for c in cents], dim=1)
        macs[f"subproto-{k}"] = int(sum(len(c) for c in cents) * d)
    # --- random "wave" features of the expert's data (broad kernel; mean embedding = fingerprint)
    g = torch.Generator().manual_seed(seed + 7)
    for m, kt in ((512, 2.0),):
        Om = torch.randn(m, d, generator=g) * (kt ** 0.5)
        b = torch.rand(m, generator=g) * 2 * np.pi
        phi = lambda z: (2.0 / m) ** 0.5 * torch.cos(z @ Om.T + b)                        # noqa: E731
        fp = torch.stack([phi(p).mean(0) for p in protos])
        S[f"wave-{m}"] = phi(xn) @ fp.T
        macs[f"wave-{m}"] = m * d + E * m
    # --- discriminative linear router on stored exemplars (prototypes + replay), one hypervector per expert
    xs, ys = [], []
    for i, e in enumerate(eids):
        rec = pool.experts[e]
        parts = [protos[i]]
        if rec.replay is not None and rec.replay.n:
            parts.append(rec.replay.data[0])
        xe = torch.cat(parts)
        xs.append(xe)
        ys.append(torch.full((len(xe),), i, dtype=torch.long))
    X, Y = torch.cat(xs), torch.cat(ys)
    Yh = F.one_hot(Y, E).float()
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    W = torch.linalg.solve(Xb.T @ Xb + 1.0 * torch.eye(d + 1), Xb.T @ Yh)
    S["ridge-lin"] = torch.cat([xn, torch.ones(len(xn), 1)], 1) @ W
    macs["ridge-lin"] = E * d
    M = 1024
    Om2 = torch.randn(M, d, generator=g) * (3.0 ** 0.5)
    b2 = torch.rand(M, generator=g) * 2 * np.pi
    phi2 = lambda z: (2.0 / M) ** 0.5 * torch.cos(z @ Om2.T + b2)                         # noqa: E731
    Z = phi2(X)
    W2 = torch.linalg.solve(Z.T @ Z + 1e-3 * torch.eye(M), Z.T @ Yh)
    S["ridge-rff"] = phi2(xn) @ W2
    macs["ridge-rff"] = M * d + M * E
    return S, macs, float(n_e.mean())


def run_one(stream, seed):
    st, pool = train(stream, seed)
    eids = list(pool.experts)
    E = len(eids)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    xn = pool._normalize(x.float())
    n = len(y)
    P, LogF = pool._gate_matrices(xn, eids)
    q = pool.cfg.q
    with torch.no_grad():
        pred_e = torch.stack([pool.experts[e].module.proba(xn).argmax(1) for e in eids])       # (E, n)
    correct = (pred_e == y[None])                                                            # (E, n)
    rows = torch.arange(n)

    def arbitrate(mask):
        elig = (P.double() >= float(q)) & mask
        best = LogF.masked_fill(~elig, float("-inf")).argmax(1)
        fallback = LogF.masked_fill(~mask, float("-inf")).argmax(1)
        return torch.where(elig.any(1), best, fallback)

    idx_exact = arbitrate(torch.ones(n, E, dtype=torch.bool))
    regime = torch.cat([torch.full((len(r.y_te),), i, dtype=torch.long) for i, r in enumerate(st.regimes.values())])
    owner = torch.stack([torch.mode(idx_exact[regime == r]).values for r in range(len(st.regimes))])   # the expert each regime's inputs mostly reach
    out = {"stream": stream, "seed": seed, "E": E, "acc_exact": float(correct[idx_exact, rows].float().mean()),
           "acc_owner": float(correct[owner[regime], rows].float().mean())}                          # perfect routing to the pool's own experts
    S, macs, mean_protos = build_scorers(pool, eids, xn, st.d, seed)
    exact_macs = mean_protos * E * st.d
    out["exact_macs"] = float(exact_macs)
    for name, s in S.items():
        direct = s.argmax(1)
        res = {"acc_direct": float(correct[direct, rows].float().mean()), "agree_exact": float((direct == idx_exact).float().mean()),
               "macs_stage1": int(macs[name])}
        order = s.argsort(1, descending=True)
        for k in KS:
            kk = min(k, E)
            mask = torch.zeros(n, E, dtype=torch.bool)
            mask.scatter_(1, order[:, :kk], True)
            idx = arbitrate(mask)
            res[f"recall@{k}"] = float(mask[rows, idx_exact].float().mean())
            res[f"acc_cascade@{k}"] = float(correct[idx, rows].float().mean())
            res[f"macs_cascade@{k}"] = float(macs[name] + kk * mean_protos * st.d)
        out[name] = res
    return out


if __name__ == "__main__":
    rows = []
    t0 = time.time()
    for stream in STREAMS:
        for seed in SEEDS:
            rows.append(run_one(stream, seed))
            print(f"{stream} seed {seed} done ({time.time() - t0:.0f}s): E={rows[-1]['E']} exact {100 * rows[-1]['acc_exact']:.1f} owner-routed {100 * rows[-1]['acc_owner']:.1f}", flush=True)
    (HERE / "spike_fingerprint.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    scorers = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    for stream in STREAMS:
        rr = [r for r in rows if r["stream"] == stream]
        print(f"\n=== {stream}: experts {np.mean([r['E'] for r in rr]):.1f}; exact gate {100 * np.mean([r['acc_exact'] for r in rr]):.1f} %, "
              f"owner-routed bound {100 * np.mean([r['acc_owner'] for r in rr]):.1f} %, exact gate {np.mean([r['exact_macs'] for r in rr]) / 1000:.0f} kMACs ===")
        print(f"{'stage 1':12s} {'alone %':>8s} {'agree %':>8s} {'kMACs':>7s} | " + " | ".join(f"top-{k}: recall  acc  kMACs" for k in KS))
        for sc in scorers:
            m = lambda key: np.mean([r[sc][key] for r in rr])                                  # noqa: E731
            print(f"{sc:12s} {100 * m('acc_direct'):8.1f} {100 * m('agree_exact'):8.1f} {m('macs_stage1') / 1000:7.1f} | "
                  + " | ".join(f"{100 * m(f'recall@{k}'):6.1f} {100 * m(f'acc_cascade@{k}'):5.1f} {m(f'macs_cascade@{k}') / 1000:5.0f}" for k in KS))

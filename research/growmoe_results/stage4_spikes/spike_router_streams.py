"""Stage-4 spike A2: does a prototype-trained LDA router hold up on the streams of stages 1-3 and on real text?  (throwaway; no product file touched)

For each (stream, dev seed) a stage-3 pool (A_set, the breaker chosen on dev seeds) is trained exactly as the pre-registered runs do (``run_fast``). The SAME trained pool is then
answered under: its shipped gate (``res.acc``), a routing oracle (the best existing expert per regime), and an LDA router fitted on nothing but the experts' signature prototypes,
in two modes: ``free`` (the router decides among all experts) and ``restrict`` (among the experts whose density gate accepts the input, the router decides; if none accepts, free).

    python spike_router_streams.py --streams F-15,OV-0.6,OV-0.85,S1,S2-5,banking --seeds 100,101
Each unit writes router_streams/<stream>_s<seed>.json atomically; finished units are skipped (safe to rerun after a power cut).
"""

import argparse
import json
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

import diag_lib as D  # noqa: E402
import run_rv2c as R  # noqa: E402
from growmoe.fairfight import run_fast  # noqa: E402

OUT = HERE / "router_streams"
ARM = "A_set@12-0.5-100"


def fit_lda(x, y, k, reg=1e-3):
    mu = np.stack([x[y == c].mean(0) for c in range(k)])
    xc = x - mu[y]
    cov = xc.T @ xc / max(len(x) - k, 1) + reg * np.eye(x.shape[1])
    prec = np.linalg.inv(cov)
    prior = np.log(np.bincount(y, minlength=k) / len(y))
    w = prec @ mu.T
    b = -0.5 * np.einsum("kd,dk->k", mu, w) + prior
    return w.astype(np.float32), b.astype(np.float32)


def evaluate(pool, st):
    eids = sorted(pool.experts)
    idx_of = {e: i for i, e in enumerate(eids)}
    xs, ys = [], []
    for r, (x, y) in st.test_sets().items():
        xs.append(x)
        ys.append(y.cpu())
    protos, owner = [], []
    for e in eids:
        v = pool.experts[e].meta["sig"].protos.numpy()
        protos.append(v)
        owner.append(np.full(len(v), idx_of[e]))
    w, b = fit_lda(np.concatenate(protos).astype(np.float64), np.concatenate(owner), len(eids))
    w_t, b_t = torch.from_numpy(w), torch.from_numpy(b)
    per = {"free": [], "restrict": [], "oracle": [], "shipped": []}
    for x, y in zip(xs, ys):
        xn = pool._normalize(x.float())
        with torch.no_grad():
            probs = {e: pool.experts[e].module.proba(xn) for e in eids}                  # every expert answers every input
            hit = {e: (probs[e].argmax(-1).cpu() == y).float() for e in eids}
            score = xn @ w_t + b_t
            free = score.argmax(1)
            P, _ = pool._gate_matrices(xn, eids)
            accepted = P >= pool.cfg.q
            restricted = torch.where(accepted.any(1), score.masked_fill(~accepted, float("-inf")).argmax(1), free)
        for name, choice in (("free", free), ("restrict", restricted)):
            h = torch.stack([hit[e] for e in eids], 1)
            per[name].append(float(h[torch.arange(len(y)), choice.cpu()].mean()))
        per["oracle"].append(max(float(hit[e].mean()) for e in eids))
        per["shipped"].append(float((pool.predict(x).probs.argmax(-1).cpu() == y).float().mean()))
    return {k: float(np.mean(v)) for k, v in per.items()}, {k: [float(a) for a in v] for k, v in per.items()}


def run_unit(stream: str, seed: int):
    stem = OUT / f"{stream.replace(chr(46), chr(112))}_s{seed}"
    if D.valid_json(stem.with_suffix(".json")):
        return
    t0 = time.time()
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
    res = run_fast(make, st, seed, eval_every="final")
    pool = holder[0]
    mean, per = evaluate(pool, st)
    out = {"stream": stream, "seed": seed, "experts": len(pool.experts), "regimes": len(st.regimes), "run_fast_acc": float(res.acc), "mean": mean, "per_regime": per, "seconds": time.time() - t0}
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))
    print(f"[{stream} s{seed}] experts {out['experts']}/{out['regimes']} regimes | shipped {100 * mean['shipped']:.1f} (run_fast {100 * res.acc:.1f})  router free {100 * mean['free']:.1f}  "
          f"router restrict {100 * mean['restrict']:.1f}  oracle {100 * mean['oracle']:.1f}   {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="F-15,OV-0.6,OV-0.85,S1,S2-5,banking")
    ap.add_argument("--seeds", default="100,101")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for stream in a.streams.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            run_unit(stream, seed)


if __name__ == "__main__":
    main()

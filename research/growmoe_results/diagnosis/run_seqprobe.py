"""Study 3: operators arrive one block after the other. What do the lifecycle decisions (extend the old expert or spawn a new one) see, and what happens afterwards?

Same stream as the benchmark's sequential protocol (max, add, sub, mod, div, mul; 8,000 problems per operator; the tuned pool h512, 5 steps, replay 20,000, stage-3 gate on).
Instrumentation only, no product code changed: (1) the interference probe of ``PhyloMoE._handle_cluster`` is logged (the four held-out losses it compares), (2) every event of the
pool is kept with its fields, (3) the pool is scored after every block, (4) the step counter at each block boundary is recorded.

Variants (config fields only):
  default        the shipped lifecycle (probe_steps=60, delta_forget=0.05 nats)
  probe600       the same with a ten times longer probe (600 steps): does a longer look-ahead see the interference?
  forced_spawn   delta_forget=-1: an extend can never be accepted, every cluster founds a new expert (one expert per operator, whatever the gate flags)

    python run_seqprobe.py --variant default --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
import growmoe.pools.phylo as phylo  # noqa: E402

OUT = HERE / "seqprobe"
VARIANTS = {"default": {}, "probe600": {"probe_steps": 600}, "forced_spawn": {"delta_forget": -1.0}}


class _FProxy:
    """Stands in for ``torch.nn.functional`` inside growmoe.pools.phylo and records the held-out losses of the interference probe (the only no-grad cross-entropies there)."""

    def __init__(self):
        self.record = None

    def __getattr__(self, name):
        return getattr(TF, name)

    def cross_entropy(self, *a, **kw):
        out = TF.cross_entropy(*a, **kw)
        if self.record is not None and not torch.is_grad_enabled() and out.ndim == 0:
            self.record.append(float(out))
        return out


def install_probe_logger(log: list):
    proxy = _FProxy()
    phylo.F = proxy
    orig = phylo.PhyloMoE._handle_cluster

    def wrapped(self, x_taken, y_taken, *args, **kw):
        proxy.record = []
        n_before = len(self.events)
        res = orig(self, x_taken, y_taken, *args, **kw)
        rec, proxy.record = proxy.record, None
        entry = {"step": self.step, "n_cluster": int(len(x_taken)), "force_parent": kw.get("force_parent"), "outcome": [e.kind for e in list(self.events)[n_before:]]}
        if len(rec) >= 4:
            entry.update(old_loss_shadow=rec[0], old_loss_parent=rec[1], new_loss_shadow=rec[2], new_loss_child=rec[3], d_old=rec[0] - rec[1], gap=rec[2] - rec[3])
        log.append(entry)
        return res

    phylo.PhyloMoE._handle_cluster = wrapped


def forgetting(blocks):
    """Mean over operators of (best exact after any block - final exact), as in the benchmark report."""
    arr = np.array([b["exact_by_op"] for b in blocks])             # (n_blocks, 6)
    return float(np.mean(arr.max(0) - arr[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--seq-cell", type=int, default=40)
    ap.add_argument("--n-per-op", type=int, default=8000)
    ap.add_argument("--out", default="seqprobe")
    a = ap.parse_args()
    global OUT
    OUT = HERE / a.out
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"{a.variant}_s{a.seed}"
    if stem.with_suffix(".json").exists():
        return
    probe_log = []
    install_probe_logger(probe_log)
    train = L.build_train("FULL", a.seed, a.n_per_op)
    test, small = L.build_test(a.per_cell), L.build_test(a.seq_cell)
    seen = L.build_seen(train, min(500, a.n_per_op), a.seed)
    learner = L.Pool(5, True, name=f"seqprobe-{a.variant}", hidden=512, replay_size=20000, **VARIANTS[a.variant])
    blocks, boundaries = [], []

    def after_block(k, oi):
        out = learner.predict(small)
        ex = small.exact(out["pred"]).reshape(A.N_OPS, -1).mean(-1)
        blocks.append({"block": k, "op_added": A.OPS[oi], "exact_by_op": [float(v) for v in ex], "n_experts": len(learner.pool.experts), "step": learner.pool.step})
        boundaries.append(learner.pool.step)
        print(f"    after block {k} (+{A.OPS[oi]}): experts {len(learner.pool.experts)}  " + " ".join(f"{A.OPS[o]} {100 * ex[o]:5.1f}" for o in L.OP_ORDER), flush=True)

    t0 = time.time()
    info = learner.fit_sequential(train, a.seed, after_block)
    res, raw = L.evaluate(learner, test, a.per_cell, seen)
    events = [{"kind": e.kind, "step": e.step, "fields": e.fields} for e in learner.pool.events]
    res.update(variant=a.variant, seed=a.seed, train=info, blocks=blocks, block_steps=boundaries, probe_log=probe_log, events=events,
               summary={**D.summarise(res), "forgetting": forgetting(blocks), "experts": len(learner.pool.experts)}, total_seconds=time.time() - t0)
    np.savez_compressed(stem.with_suffix(".npz"), **raw)
    stem.with_suffix(".json").write_text(json.dumps(res), encoding="utf-8")
    s = res["summary"]
    print(f"[{a.variant} s{a.seed}] mean exact {100 * s['mean_exact']:.1f}  forgetting {100 * s['forgetting']:.1f}  experts {s['experts']}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

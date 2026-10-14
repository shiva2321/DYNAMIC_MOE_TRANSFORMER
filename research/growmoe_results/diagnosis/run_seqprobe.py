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
import os
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


ASSIGN_LOG = []


def install_assign_logger():
    """Records, for every ``observe`` call, how many of the batch's samples were handed to each expert for TRAINING (-1 = trained by nobody: flagged, held or unlabelled)."""
    orig = phylo.PhyloMoE.observe

    def observe_logged(self, x, y=None, ctx=None):
        rec = orig(self, x, y, ctx)
        a = rec.assigned
        ASSIGN_LOG.append((int(self.step), {int(e): int((a == e).sum()) for e in torch.unique(a).tolist()}))
        return rec

    phylo.PhyloMoE.observe = observe_logged


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
    if D.valid_json(stem.with_suffix(".json")):                                       # done (a truncated file does not count)
        return
    ckpt = stem.with_suffix(".ckpt")                                                  # state after the last finished block: a power cut costs at most one block
    probe_log = []
    install_probe_logger(probe_log)
    install_assign_logger()
    train = L.build_train("FULL", a.seed, a.n_per_op)
    test, small = L.build_test(a.per_cell), L.build_test(a.seq_cell)
    seen = L.build_seen(train, min(500, a.n_per_op), a.seed)
    learner = L.Pool(5, True, name=f"seqprobe-{a.variant}", hidden=512, replay_size=20000, **VARIANTS[a.variant])
    blocks, boundaries = [], []

    def after_block(k, oi):
        out = learner.predict(small)
        ex = small.exact(out["pred"]).reshape(A.N_OPS, -1).mean(-1)
        op_of, correct = small.op[small.pi], out["pred"] == small.y
        eids = sorted(learner.pool.experts)
        routing = {A.OPS[o]: {int(e): int(((op_of == o) & (out["expert"] == e)).sum()) for e in eids} for o in range(A.N_OPS)}          # examples of operator o sent to expert e
        acc_given = {A.OPS[o]: {int(e): (float(correct[(op_of == o) & (out["expert"] == e)].mean()) if ((op_of == o) & (out["expert"] == e)).any() else None) for e in eids} for o in range(A.N_OPS)}
        blocks.append({"block": k, "op_added": A.OPS[oi], "exact_by_op": [float(v) for v in ex], "n_experts": len(learner.pool.experts), "step": learner.pool.step,
                       "routing": routing, "slot_acc_given_expert": acc_given, "unknown_rate": float(out["unknown"].mean()),
                       "expert_states": {int(e): learner.pool.experts[e].state for e in eids}})
        boundaries.append(learner.pool.step)
        print(f"    after block {k} (+{A.OPS[oi]}): experts {len(learner.pool.experts)}  " + " ".join(f"{A.OPS[o]} {100 * ex[o]:5.1f}" for o in L.OP_ORDER), flush=True)
        for o in L.OP_ORDER:
            tot = max(sum(routing[A.OPS[o]].values()), 1)
            print(f"        {A.OPS[o]:<4s} routed to " + "  ".join(f"e{e}:{100 * c / tot:4.0f}%" for e, c in routing[A.OPS[o]].items() if c), flush=True)

    torch.set_num_threads(3)
    seconds_before, done_block = 0.0, -1
    if ckpt.exists():
        try:
            import pickle
            with open(ckpt, "rb") as fh:                                              # written by this script (atomic_pickle) in this folder; never loaded from anywhere else
                state = pickle.load(fh)
            learner.pool = state["pool"]
            probe_log.extend(state["probe_log"])
            blocks.extend(state["blocks"])
            boundaries.extend(state["boundaries"])
            ASSIGN_LOG.extend(state["assign_log"])
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["np_rng"])
            import random
            random.setstate(state["py_rng"])
            seconds_before, done_block = state["seconds"], state["done_block"]
            print(f"RESUMED from the checkpoint after block {done_block} ({seconds_before:.0f} s of work already done)", flush=True)
        except Exception as exc:                                                      # unreadable checkpoint: start over rather than trust it
            print("checkpoint unreadable, starting from the beginning:", exc, flush=True)
            probe_log.clear(), blocks.clear(), boundaries.clear(), ASSIGN_LOG.clear()
            seconds_before, done_block = 0.0, -1
    if done_block < 0:
        learner._new(a.seed)
    t0 = time.time() - seconds_before
    for k, (oi, idx) in enumerate(L.op_blocks(train, a.seed)):
        if k <= done_block:
            continue
        learner._feed(train, idx)
        after_block(k, oi)
        import random
        D.atomic_pickle(ckpt, {"pool": learner.pool, "probe_log": probe_log, "blocks": blocks, "boundaries": boundaries, "assign_log": ASSIGN_LOG, "done_block": k,
                               "seconds": time.time() - t0, "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate()})
        if os.environ.get("SEQPROBE_CRASH_AFTER_BLOCK") == str(k):                    # test hook: simulate a power cut right after the checkpoint of block k
            print(f"simulated power cut after block {k}", flush=True)
            os._exit(3)
    info = learner._info(t0)
    res, raw = L.evaluate(learner, test, a.per_cell, seen)
    xt = torch.from_numpy(D.unit(test.x()))                                          # every expert answers EVERY test example: the best expert per operator is the routing oracle
    per_expert = {}
    with torch.no_grad():
        for e, rec in learner.pool.experts.items():
            rec.module.eval()
            per_expert[int(e)] = [float(v) for v in test.exact(rec.module(xt).argmax(-1).numpy()).reshape(A.N_OPS, -1).mean(-1)]
    res["expert_exact_by_op"] = per_expert
    res["oracle_routed_exact_by_op"] = [max(v[o] for v in per_expert.values()) for o in range(A.N_OPS)]
    rules = {}                                                                         # the SAME trained pool answered under the routing rules the product offers (prediction-time only)
    saved_rule = learner.pool.cfg.route_rule
    for rule in ("accept_then_logf", "argmax_logf", "argmax_p", "nearest_proto", "zscore_logf"):
        learner.pool.cfg.route_rule = rule
        o = learner.predict(test)
        rules[rule] = {"exact_by_op": [float(v) for v in test.exact(o["pred"]).reshape(A.N_OPS, -1).mean(-1)],
                       "routing_to_own": float(np.mean([(o["expert"][test.op[test.pi] == k] == int(np.argmax([per_expert[e][k] for e in sorted(per_expert)]))).mean() for k in range(A.N_OPS)]))}
    learner.pool.cfg.route_rule = saved_rule
    res["route_rules"] = rules
    try:
        D.atomic_pickle(stem.with_suffix(".pool.pkl"), learner.pool)
    except Exception as exc:                                                           # the analysis does not depend on it
        print("could not pickle the pool:", exc, flush=True)
    events = [{"kind": e.kind, "step": e.step, "fields": e.fields} for e in learner.pool.events]
    per_block, lo = [], 0                                                         # training assignment per block: samples handed to each expert (-1: trained by nobody)
    for hi in boundaries:
        tot = {}
        for st, cnt in ASSIGN_LOG:
            if lo < st <= hi:
                for e, c in cnt.items():
                    tot[e] = tot.get(e, 0) + c
        per_block.append({str(e): c for e, c in sorted(tot.items())})
        lo = hi
    res.update(variant=a.variant, seed=a.seed, train=info, blocks=blocks, block_steps=boundaries, probe_log=probe_log, events=events, train_assignment_per_block=per_block,
               summary={**D.summarise(res), "forgetting": forgetting(blocks), "experts": len(learner.pool.experts)}, total_seconds=time.time() - t0)
    D.atomic_savez(stem.with_suffix(".npz"), **raw)
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(res))                    # LAST: this file is what marks the run as done
    if ckpt.exists():
        os.remove(ckpt)
    s = res["summary"]
    print(f"[{a.variant} s{a.seed}] mean exact {100 * s['mean_exact']:.1f}  forgetting {100 * s['forgetting']:.1f}  experts {s['experts']}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

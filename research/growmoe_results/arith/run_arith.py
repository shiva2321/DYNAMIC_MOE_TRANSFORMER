"""Arithmetic benchmark: can GrowMoE and its competitors LEARN +, -, *, %, // and max on 1-5 digit numbers from examples alone, and does it generalise?

No arithmetic algorithm, table or helper is given to anyone: the learners see (problem, output slot) -> symbol examples (growmoe/arith.py) and nothing else.

Protocols
  iid   all operators and lengths mixed, one pass of the data (the standard supervised setting; for GrowMoE a single stationary stream).
  seq   the operators arrive one block after the other (max, add, sub, mod, div, mul): the continual-learning setting; scored after every block (forgetting).
Training length regimes (what the learners may see): FULL = lengths 1-5, HOLE = lengths 1,2,4,5 (no operand ever has 3 digits), SHORT = lengths 1-3.
Test grid (identical for every learner, regime and seed): held-out problems (never trained on), PER_CELL per (operator, length of A, length of B), plus a sample of the
learner's own training problems ("seen"). A problem counts as right only if every output slot is right (exact match).

    python research/growmoe_results/arith/run_arith.py run --protocol iid --regimes FULL,HOLE,SHORT --seeds 0,1 --learners MLP-L,GrowMoE
    python research/growmoe_results/arith/run_arith.py report
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import arith_lib as L  # noqa: E402
from growmoe import arith as A  # noqa: E402

RES = HERE / "results"

REGISTRY = {
    "MLP-S": lambda: L.DenseMLP(256, 2, epochs=40, name="MLP-S"),
    "MLP-L": lambda: L.DenseMLP(1024, 3, epochs=40, name="MLP-L"),
    "Transformer": lambda: L.Transformer(epochs=100),
    "kNN": lambda: L.Knn(),
    "Oracle-op": lambda: L.OracleExperts("op"),
    "Oracle-oppos": lambda: L.OracleExperts("oppos"),
    "GrowMoE": lambda: L.Pool(5, False),
    "GrowMoE+s3": lambda: L.Pool(5, True),
    "GrowMoE-s20": lambda: L.Pool(20, False, name="GrowMoE-s20"),
    "MLP-L-seq": lambda: L.DenseMLP(1024, 3, epochs=40, replay=0, name="MLP-L-seq"),
    "MLP-L-replay": lambda: L.DenseMLP(1024, 3, epochs=40, replay=20000, name="MLP-L-replay"),
    "ReplayNet": lambda: L.SmallNet("replay"),
    "FineTuneNet": lambda: L.SmallNet("finetune"),
}
_TEST = {}


def make(name: str):
    """A learner by name; GrowMoE variants are parsed from the name: GrowMoE[-h<expert width>][-s<steps per batch>][-r<replay size per expert>][+s3]
    (defaults: width 128, 5 steps, replay 256, stage 3 off)."""
    if name in REGISTRY:
        return REGISTRY[name]()
    m = re.fullmatch(r"GrowMoE(-h\d+)?(-s\d+)?(-r\d+)?(\+s3)?", name)
    if not m:
        raise KeyError(name)
    kw = {}
    if m.group(1):
        kw["hidden"] = int(m.group(1)[2:])
    if m.group(3):
        kw["replay_size"] = int(m.group(3)[2:])
    return L.Pool(int(m.group(2)[2:]) if m.group(2) else 5, bool(m.group(4)), name=name, **kw)


def test_grid(per_cell: int):
    if per_cell not in _TEST:
        _TEST[per_cell] = L.build_test(per_cell)
    return _TEST[per_cell]


def in_length_mask(regime: str) -> np.ndarray:
    """(5, 5) bool over (la - 1, lb - 1): True where BOTH operand lengths were available in training."""
    ok = np.zeros((5, 5), dtype=bool)
    for la in L.LENGTHS[regime]:
        for lb in L.LENGTHS[regime]:
            ok[la - 1, lb - 1] = True
    return ok


def summarize(res: dict, regime: str) -> str:
    ex, dg, ok = np.array(res["grid_exact"]), np.array(res["grid_digit"]), in_length_mask(regime)
    lines = [f"    {'op':<5s} {'seen':>6s} {'unseen/in-len':>14s} {'unseen/new-len':>15s} {'digit/in':>9s} {'digit/new':>10s}"]
    for oi, name in enumerate(A.OPS):
        new = f"{100 * ex[oi][~ok].mean():5.1f}" if (~ok).any() else "  -  "
        newd = f"{100 * dg[oi][~ok].mean():5.1f}" if (~ok).any() else "  -  "
        lines.append(f"    {name:<5s} {100 * res['seen_exact_by_op'][oi]:6.1f} {100 * ex[oi][ok].mean():14.1f} {new:>15s} {100 * dg[oi][ok].mean():9.1f} {newd:>10s}")
    return "\n".join(lines)


def run_iid(name, regime, seed, a):
    stem = RES / f"iid_{regime}_{name}_s{seed}"
    if stem.with_suffix(".json").exists():
        return
    train = L.build_train(regime, seed, a.n_per_op)
    test, seen = test_grid(a.per_cell), L.build_seen(train, a.seen_per_op, seed)
    learner = make(name)
    t0 = time.time()
    info = learner.fit(train, seed)
    res, raw = evaluate_and_pack(learner, test, a, seen)
    res.update(learner=name, protocol="iid", regime=regime, seed=seed, train=info, train_problems=len(train), train_samples=train.n_samples, n_per_op=a.n_per_op,
               per_cell=a.per_cell, total_seconds=time.time() - t0)
    save(stem, res, raw)
    print(f"[iid {regime} {name} seed {seed}] fit {info['seconds']:.0f}s  experts {info.get('experts', '-')}  params {info.get('params')}  total {time.time() - t0:.0f}s", flush=True)
    print(summarize(res, regime), flush=True)


def run_seq(name, regime, seed, a):
    stem = RES / f"seq_{regime}_{name}_s{seed}"
    if stem.with_suffix(".json").exists():
        return
    train = L.build_train(regime, seed, a.n_per_op)
    test, seen = test_grid(a.per_cell), L.build_seen(train, a.seen_per_op, seed)
    small = test_grid(a.seq_cell)
    learner = make(name)
    after = []

    def after_block(k, oi):
        out = learner.predict(small)
        ex = small.exact(out["pred"]).reshape(A.N_OPS, -1).mean(-1)
        after.append({"block": k, "op_added": A.OPS[oi], "exact_by_op": [float(v) for v in ex]})
        print(f"    after block {k} (+{A.OPS[oi]}): " + " ".join(f"{A.OPS[o]} {100 * ex[o]:5.1f}" for o in L.OP_ORDER), flush=True)

    t0 = time.time()
    info = learner.fit_sequential(train, seed, after_block)
    res, raw = evaluate_and_pack(learner, test, a, seen)
    res.update(learner=name, protocol="seq", regime=regime, seed=seed, train=info, blocks=after, train_problems=len(train), train_samples=train.n_samples, n_per_op=a.n_per_op,
               per_cell=a.per_cell, total_seconds=time.time() - t0)
    save(stem, res, raw)
    print(f"[seq {regime} {name} seed {seed}] fit {info['seconds']:.0f}s experts {info.get('experts', '-')} total {time.time() - t0:.0f}s", flush=True)
    print(summarize(res, regime), flush=True)


def evaluate_and_pack(learner, test, a, seen):
    return L.evaluate(learner, test, a.per_cell, seen)


def save(stem: Path, res: dict, raw: dict):
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), **raw)
    stem.with_suffix(".json").write_text(json.dumps(res), encoding="utf-8")


def cmd_run(a):
    global RES
    if a.out:
        RES = HERE / a.out
    for regime in a.regimes.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            for name in a.learners.split(","):
                (run_iid if a.protocol == "iid" else run_seq)(name, regime, seed, a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--protocol", choices=("iid", "seq"), default="iid")
    r.add_argument("--regimes", default="FULL")
    r.add_argument("--seeds", default="0")
    r.add_argument("--learners", default="MLP-L")
    r.add_argument("--n-per-op", type=int, default=8000)
    r.add_argument("--per-cell", type=int, default=200)
    r.add_argument("--seq-cell", type=int, default=40)
    r.add_argument("--seen-per-op", type=int, default=500)
    r.add_argument("--out", default="")
    args = ap.parse_args()
    {"run": cmd_run}[args.cmd](args)

"""SHOOTOUT (exploratory, dev seeds 100-104): at EQUAL MEMORY, does a growing pool of experts beat a single network and a lookup on streams where every
regime has its own hidden rule? What does it cost per prediction?

Written down BEFORE the first run (2026-10-14):

  Question.  The repaired pool beat the best single network by +10 to +12 points on 15 and 30 function regimes (M2-A, fresh seeds), but the pool used
             1.4x the single network's memory, and the single network's width grid stopped at 4096. Is the advantage still there when every arm gets the SAME
             number of bytes (4 x parameters + 4 x d x stored vectors, the project's accounting), and does it grow with the number of regimes?
  Streams.   function_regimes with N = 10, 30 and 100 regimes (hidden rule per input region), batches of 64, final accuracy averaged over all regimes.
  Arms.      POOL      PhyloMoE with the product defaults, 5 steps per batch (the budget tuned for this family in M2-A) and the fingerprint shortlist forced on
                       (gate_shortlist_min_experts = 2) so that its compute is the efficient configuration; its accuracy equals the exact-scan pool's within 0.1 point
                       (RV2-B, C1);
             REPLAY_EQ one network + replay buffer, widths 1024 / 2048 / 4096 / 8192 / 16384, 5 and 20 steps; the buffer takes whatever bytes the POOL used
                       (capped at the number of training samples); the BEST cell of the grid is reported, chosen on the evaluation seeds themselves
                       (favourable to the baseline);
             KNN_EQ    nearest neighbours with the POOL's bytes; KNN_ALL stores everything (reference);
             ORACLE    one expert per regime with the regime id given (reference: the routing ceiling);
             JOINT     one network trained on everything at once, widths 1024 and 4096 (reference: the single-network ceiling; it needs the whole stream in memory).
  Metrics.   final accuracy, bytes, multiply-adds per prediction (POOL: measured gate multiply-adds with the shortlist + the answering expert), seconds per run.
  Pass line (declared in advance, for the verdict on viability on this stream family): POOL beats the best REPLAY_EQ and KNN_EQ cell by >= 5 points at N = 100 AND
             uses <= 1/3 of the best REPLAY_EQ cell's multiply-adds per prediction. Anything else is reported as it is.

    python research/growmoe_results/shootout/shootout.py run --n 30 --seeds 100-104
    python research/growmoe_results/shootout/shootout.py report
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
from growmoe import bench  # noqa: E402
from growmoe.accounting import measured_gate_macs_per_input, pool_report, weight_macs  # noqa: E402
from growmoe.baselines import JointOracle, OracleRouted, ReplayNet  # noqa: E402
from growmoe.config import CommonCfg, PhyloCfg  # noqa: E402
from growmoe.experts import MLPExpert  # noqa: E402
from growmoe.fairfight import matched_capacity, run_fast  # noqa: E402
from growmoe.memory_baselines import KnnMemory  # noqa: E402
from growmoe.metrics import paired_diff  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402

RUNS = HERE / "runs"
WIDTHS, STEPS = (1024, 2048, 4096, 8192, 16384), (5, 20)
POOL_STEPS = 5


def parse_seeds(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def _common(st, seed, steps, **kw):
    return CommonCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps, **kw)


def total_samples(st):
    return int(sum(len(seg.x) for seg in st.segments))


def net_params(st, width):
    return sum(p.numel() for p in MLPExpert(st.d, width, 1, st.n_classes).parameters())


def record(arm, n, seed, res, pool=None, **extra):
    rep = res.report
    return dict(arm=arm, n_regimes=n, seed=seed, acc=res.acc, experts=rep.n_experts, params=rep.params, stored_vectors=rep.stored_vectors, bytes=rep.memory_bytes,
                predict_macs=rep.predict_macs, seconds=res.wall_clock, **extra)


def save(path, rec):
    path.write_text(json.dumps(rec), encoding="utf-8")
    print(f"  {rec['arm']:<26s} N={rec['n_regimes']:<3d} seed {rec['seed']}: acc {rec['acc']:.4f}  bytes {rec['bytes'] / 1e6:6.2f} MB  macs/pred {rec['predict_macs'] / 1e3:8.1f} k  {rec['seconds']:5.0f}s", flush=True)


def run_one(n, seed):
    out = RUNS / f"N{n}"
    out.mkdir(parents=True, exist_ok=True)
    todo = lambda name: not (out / f"{name}__{seed}.json").exists()           # noqa: E731
    st = function_regimes(seed, n)
    # ---- the pool
    pool_path = out / f"pool__{seed}.json"
    if pool_path.exists():
        pool_rec = json.loads(pool_path.read_text(encoding="utf-8"))
    else:
        holder = []

        def make():
            holder.append(PhyloMoE(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=POOL_STEPS, q=0.02, alpha_spawn=0.01, m_min=32,
                                            route_rule="accept_then_logf", gate_shortlist_min_experts=2)))
            return holder[0]
        res = run_fast(make, st, seed, eval_every="final")
        pool = holder[0]
        gate = measured_gate_macs_per_input(pool, "predict")
        rep = res.report
        pool_rec = record("pool", n, seed, res, steps=POOL_STEPS, gate_macs_measured=gate, expert_macs=rep.expert_macs)
        pool_rec["predict_macs"] = gate + rep.expert_macs                         # the shortlist's measured gate cost + the one answering expert
        save(pool_path, pool_rec)
    budget = pool_rec["bytes"]
    samples = total_samples(st)
    # ---- replay network with the pool's bytes, over a small grid
    for w in WIDTHS:
        params_bytes = 4 * net_params(st, w)
        buffer = min(samples, int((budget - params_bytes) // (4 * st.d)))
        if buffer < 512:
            continue
        for steps in STEPS:
            name = f"replay_eq_w{w}_s{steps}"
            if todo(name):
                res = run_fast(lambda: ReplayNet(_common(st, seed, steps, hidden=w), replay_size=buffer), st, seed, eval_every="final")
                save(out / f"{name}__{seed}.json", record(name, n, seed, res, width=w, buffer=buffer, steps=steps))
    # ---- lookups and references
    cap = max(int(budget // (4 * st.d)), 16)
    refs = {"knn_eq": lambda: KnnMemory(_common(st, seed, 1), capacity=cap), "knn_all": lambda: KnnMemory(_common(st, seed, 1)),
            "oracle_routed": lambda: OracleRouted(bench.make_regime_of(st, 64), _common(st, seed, 5)),
            "joint_w1024": lambda: JointOracle(_common(st, seed, 5, hidden=1024)), "joint_w4096": lambda: JointOracle(_common(st, seed, 5, hidden=4096))}
    for name, make in refs.items():
        if todo(name):
            res = run_fast(make, st, seed, eval_every="final")
            save(out / f"{name}__{seed}.json", record(name, n, seed, res))


def cmd_run(args):
    for seed in parse_seeds(args.seeds):
        t0 = time.time()
        print(f"=== N={args.n} seed {seed}", flush=True)
        run_one(args.n, seed)
        print(f"    done in {time.time() - t0:.0f}s", flush=True)


def load():
    rows = []
    for p in sorted(RUNS.glob("N*/*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def cmd_report(args):
    rows = load()
    ns = sorted({r["n_regimes"] for r in rows})
    lines = ["# Shootout results (exploratory, dev seeds; generated by shootout.py report)\n"]
    for n in ns:
        rs = [r for r in rows if r["n_regimes"] == n]
        seeds = sorted({r["seed"] for r in rs if r["arm"] == "pool"})
        by = {}
        for r in rs:
            by.setdefault(r["arm"], {})[r["seed"]] = r
        replay_cells = sorted(a for a in by if a.startswith("replay_eq"))
        # best replay cell per N (mean accuracy over the seeds on which the whole grid ran)
        full = [s for s in seeds if all(s in by[c] for c in replay_cells)]
        best = max(replay_cells, key=lambda c: np.mean([by[c][s]["acc"] for s in full])) if full and replay_cells else None
        lines += [f"\n## N = {n} regimes (seeds {', '.join(map(str, seeds))})\n", "| arm | accuracy % | bytes (MB) | multiply-adds per prediction (k) | seconds per run | experts |", "|---|---|---|---|---|---|"]
        order = ["pool"] + ([best] if best else []) + ["knn_eq", "knn_all", "oracle_routed", "joint_w1024", "joint_w4096"]
        for a in order:
            ss = [s for s in seeds if a in by and s in by[a]]
            if not ss:
                continue
            acc = [100 * by[a][s]["acc"] for s in ss]
            lines.append(f"| {a} | {np.mean(acc):.1f} ± {np.std(acc, ddof=1) / np.sqrt(len(acc)) if len(acc) > 1 else 0:.1f} | {np.mean([by[a][s]['bytes'] for s in ss]) / 1e6:.1f} | "
                         f"{np.mean([by[a][s]['predict_macs'] for s in ss]) / 1e3:.0f} | {np.mean([by[a][s]['seconds'] for s in ss]):.0f} | {np.mean([by[a][s]['experts'] for s in ss]):.1f} |")
        if best:
            for ref in (best, "knn_eq"):
                ss = [s for s in full if s in by.get(ref, {})]
                if len(ss) >= 2:
                    d = paired_diff([100 * by["pool"][s]["acc"] for s in ss], [100 * by[ref][s]["acc"] for s in ss])
                    lines.append(f"\n* pool - {ref}: {d['mean_diff']:+.2f} points [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}] (n = {len(ss)})")
            ratio = np.mean([by["pool"][s]["predict_macs"] / by[best][s]["predict_macs"] for s in full])
            lines.append(f"* multiply-adds per prediction, pool / {best}: {ratio:.2f}")
        grid = [(c, np.mean([100 * by[c][s]["acc"] for s in full])) for c in replay_cells] if full else []
        if grid:
            lines.append("* replay grid (mean accuracy %): " + ", ".join(f"{c.replace('replay_eq_', '')} {v:.1f}" for c, v in grid))
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS_SHOOTOUT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--seeds", default="100-104")
    sub.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)

"""GrowMoE demo: learn a long sequence of different tasks without forgetting, and say "I don't know" to something new.

    python demos/demo_growmoe_continual_learning.py                       # 12 tasks, about 1-2 minutes on a laptop CPU, writes demos/growmoe_demo.png
    python demos/demo_growmoe_continual_learning.py --tasks 30 --seed 3

What it shows. Every learner gets exactly the same data in the same order, one task after the other (a task = a different hidden rule that maps a
128-number input to one of 8 classes; chance is 12.5 %):
  * plain network             just keeps training                                  -> forgets the old tasks
  * network + replay buffer   the usual fix, small                                 -> forgets less
  * network + replay (same memory)  a big network and a big buffer with exactly as many bytes as GrowMoE uses (the fair comparison)
  * GrowMoE                   a pool that adds one small expert per new task, answers each input with ONE expert
and at the end: how often GrowMoE says "I don't know" to inputs from tasks it never saw.

Use it on your own data (any feature vectors: embeddings, sensor windows, ...; labels 0..C-1):

    from growmoe.config import PhyloCfg
    from growmoe.pools.phylo import PhyloMoE
    pool = PhyloMoE(PhyloCfg(d=128, n_classes=10, inner_steps=5, route_rule="accept_then_logf"))     # d = length of your feature vectors
    for x, y in batches:                                              # x: float tensor (batch, d); y: long tensor (batch,)
        pool.observe(x, y)                                            # learns; adds an expert when the data is new
    out = pool.predict(x_new)                                         # out.probs, out.expert (who answered), out.unknown (True = "nothing like this seen before")

The inputs are normalised inside (only their direction matters). Since 2026-10-21 two of the stage-4 repairs are on by default: a spawn-or-extend test that asks the labels, and a learned
router that picks the expert at answer time. The router needs the routing rule route_rule="accept_then_logf" that every measured pool used, so set it as above. To run the pool as it was
before stage 4: PhyloCfg(..., **growmoe.config.PRE_STAGE4). The router refits itself when the set of experts changes (about 2 s per refit at 12 experts, most of it one candidate); if the
tasks never overlap it changes no answer, and router=False saves that time. (Growth of a starved expert from the fit gap, growth_rule="gap", is tested but not the default.)

Honest limits (measured; see research/growmoe_results/shootout, m5 and rv2d): GrowMoE wins when the tasks are DIFFERENT RULES over the input: at the same memory it beats the
best replay network by 6 points on 10 hidden-rule tasks, 9 on 30 and 13 on 100 (dev seeds, measured before stage 4 became the default; the transfer rule moves this pool by about
-0.3 points on such streams, fresh seeds), and a nearest-neighbour lookup fails there. This demo gives the pool its own random seed (the stream's seed + 7): with the same seed the expert founded
for task r would start with task r's own hidden units, a head start worth about 8 points that an earlier version of this demo included. When the tasks are categories that a lookup
can recognise (real text intents, banking77), GrowMoE with the stage-3 gate is level with a replay network of the same memory (82.3 against 83.4 %, fresh seeds) and a plain kNN
(86.5 %) is ahead; the transfer rule lifts the pool from 81.8 to 84.3 % on five other fresh seeds, which closes some of that gap (the replay and kNN numbers are from the earlier seeds).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
from growmoe.accounting import measured_gate_macs_per_input  # noqa: E402
from growmoe.baselines import FineTune, ReplayNet  # noqa: E402
from growmoe.config import CommonCfg, PhyloCfg  # noqa: E402
from growmoe.experts import MLPExpert  # noqa: E402
from growmoe.fairfight import run_fast  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402


def curves(R, stream):
    """After each training segment: mean accuracy over the tasks learned so far, and accuracy on the very first task."""
    seen, avg, first = set(), [], []
    for i, row in enumerate(R):
        seen.add(stream.segments[i].regime)
        avg.append(float(np.mean([row[r] for r in seen])))
        first.append(float(row[0]))
    return np.array(avg), np.array(first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "demos" / "growmoe_demo.png"))
    a = ap.parse_args()
    n, seed = a.tasks, a.seed
    t0 = time.time()
    print(f"Building a stream of {n} tasks (each task = a different hidden rule: 128 numbers in, 8 classes out; chance = 12.5 %) ...", flush=True)
    st = function_regimes(seed, n, n_train=400, n_test=200)
    st_new = function_regimes(seed + 777, 2, n_train=400, n_test=200)           # other random tasks, never trained on: the "something new" for the unknown test
    samples = int(sum(len(s.x) for s in st.segments))

    def common(steps, **kw):
        return CommonCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps, **kw)

    def train(name, make):
        holder = []

        def keep():
            holder.append(make())
            return holder[0]
        res = run_fast(keep, st, seed, eval_every="segments")
        print(f"  trained {name:<34s} ({res.wall_clock:4.0f}s)", flush=True)
        return res, holder[0]

    results, pools = {}, {}
    results["GrowMoE"], pools["GrowMoE"] = train("GrowMoE", lambda: PhyloMoE(PhyloCfg(
        seed=seed + 7, d=st.d, n_classes=st.n_classes, inner_steps=5, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", gate_shortlist_min_experts=2)))      # seed + 7: not the stream's seed (E-035)
    budget = results["GrowMoE"].report.memory_bytes                              # the fair comparison gets exactly GrowMoE's bytes
    width = 2048
    buffer = max(512, min(samples, int((budget - 4 * MLPExpert(st.d, width, 1, st.n_classes).num_params()) // (4 * st.d))))
    results["plain network"], pools["plain network"] = train("plain network", lambda: FineTune(common(5)))
    results["network + replay buffer"], pools["network + replay buffer"] = train("network + replay buffer (small)", lambda: ReplayNet(common(20)))
    results["network + replay (same memory)"], pools["network + replay (same memory)"] = train(
        f"network + replay, same memory (w{width}, {buffer} kept)", lambda: ReplayNet(common(20, hidden=width), replay_size=buffer))

    names = list(results)
    curve = {k: curves(results[k].R, st) for k in names}
    nseg = len(curve[names[0]][0])
    pick = sorted(set(list(range(0, nseg, max(1, nseg // 6))) + [nseg - 1]))
    short = {"plain network": "plain", "network + replay buffer": "replay (small)", "network + replay (same memory)": "replay (same mem)", "GrowMoE": "GrowMoE"}
    print("\nAverage accuracy (%) over the tasks learned so far, after each training segment:")
    print(f"  {'segment':>8s} " + " ".join(f"{short[k]:>18s}" for k in names))
    for i in pick:
        print(f"  {i + 1:>8d} " + " ".join(f"{100 * curve[k][0][i]:>18.1f}" for k in names))
    print("\nAccuracy on the FIRST task at the end (did it forget?):")
    for k in names:
        print(f"  {k:<40s} {100 * curve[k][1][-1]:5.1f} %   (right after learning it: {100 * curve[k][1][0]:5.1f} %)")

    pool = pools["GrowMoE"]
    print(f"\nGrowMoE built {pool.summary()['n_experts']} experts for {n} tasks and answers each input with ONE of them.")
    owner = []
    for r in range(n):
        owner.append(int(torch.mode(pool.predict(st.test_sets()[r][0]).expert).values))
    print("  task -> expert that answers it:", ", ".join(f"{r}->{e}" for r, e in enumerate(owner)))

    print("\nCost at the end (same accounting for all: bytes = 4 x parameters + 4 x 128 x stored examples):")
    print(f"  {'':<40s} {'accuracy %':>11s} {'memory MB':>10s} {'multiply-adds per answer':>26s}")
    for k in names:
        rep = results[k].report
        macs = rep.predict_macs
        if k == "GrowMoE":
            macs = rep.expert_macs + measured_gate_macs_per_input(pool, "predict")      # the fingerprint shortlist's measured gate cost + the answering expert
        print(f"  {k:<40s} {100 * results[k].acc:>11.1f} {rep.memory_bytes / 1e6:>10.1f} {macs:>26,.0f}")

    print("\n'I don't know' test (GrowMoE only; the networks always answer):")
    x_old = torch.cat([st.test_sets()[r][0] for r in range(n)])
    x_new = st_new.test_sets()[0][0]
    print(f"  inputs from tasks it learned : {100 * float(pool.predict(x_old).unknown.float().mean()):5.1f} % flagged   (the design target is at most about 2 %)")
    print(f"  inputs from an unrelated task: {100 * float(pool.predict(x_new).unknown.float().mean()):5.1f} % flagged")
    print("  (an easy case: unrelated tasks. Tasks that overlap a lot are much harder to flag one input at a time; that is what the batch test (stage 3) is for.)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        colors = {"plain network": "#c0392b", "network + replay buffer": "#e67e22", "network + replay (same memory)": "#2e86c1", "GrowMoE": "#1e8449"}
        for k in names:
            xs = np.arange(1, nseg + 1)
            ax[0].plot(xs, 100 * curve[k][0], label=k, color=colors[k], lw=2)
            ax[1].plot(xs, 100 * curve[k][1], label=k, color=colors[k], lw=2)
        ax[0].set_title("Average accuracy on all tasks learned so far")
        ax[1].set_title("Accuracy on the first task (forgetting)")
        for axis in ax:
            axis.axhline(12.5, color="k", lw=0.8, ls=":")
            axis.set_xlabel("training segments seen")
            axis.set_ylabel("accuracy %")
            axis.grid(alpha=0.3)
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(a.out, dpi=130)
        print(f"\nPlot written to {a.out}")
    except Exception as e:                                                       # the numbers above are the result; the picture is a bonus
        print(f"(no plot: {e})")
    print(f"Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

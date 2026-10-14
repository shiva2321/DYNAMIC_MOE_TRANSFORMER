"""M5-D: the real-text check of stage 3 on FRESH seeds 80-84, nothing tuned on them.

Written down before the run (2026-10-14). On dev seeds 100-102 the amended stage-3 gate (set_gate=True, persistence 2) lifted GrowMoE on the banking77
class-incremental stream from 71.8 % to 83.7 %, equal to the small-buffer replay network (83.7 %); kNN stored 87.3 %, the joint ceiling 86.2 % (m5c_stage3.json).
Prediction: on seeds 80-84 with the SAME settings (20 steps per batch for every learner, set_gate defaults) A_set beats A_today by >= 5 points on at least 4 of 5 seeds,
and the mean of A_set is within 5 points of the replay network that gets A_set's bytes (the fair comparison). Seeds 80-84 are used up by this run.

    python research/growmoe_results/m5/run_m5d_fresh.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(3)

from growmoe.baselines import JointOracle, ReplayNet  # noqa: E402
from growmoe.config import CommonCfg, PhyloCfg  # noqa: E402
from growmoe.experts import MLPExpert  # noqa: E402
from growmoe.fairfight import run_fast  # noqa: E402
from growmoe.memory_baselines import KnnMemory  # noqa: E402
from growmoe.metrics import paired_diff  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.streams_real import banking_incremental  # noqa: E402
from research.growmoe_results.m0.run_m0 import HybridFrontend  # noqa: E402

SEEDS, STEPS = (80, 81, 82, 83, 84), 20


def main():
    fe = HybridFrontend(d=128, seed=0)
    rows = {}
    for sd in SEEDS:
        st = banking_incremental(seed=sd, frontend=fe)
        samples = int(sum(len(s.x) for s in st.segments))

        def common(**kw):
            return CommonCfg(seed=sd, d=st.d, n_classes=st.n_classes, inner_steps=STEPS, **kw)

        def pool(set_gate):
            return lambda: PhyloMoE(PhyloCfg(d=st.d, n_classes=st.n_classes, seed=sd, inner_steps=STEPS, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32, set_gate=set_gate))
        r_set = run_fast(pool(True), st, sd, eval_every="final")
        budget = r_set.report.memory_bytes
        out = {"A_set": r_set, "A_today": run_fast(pool(False), st, sd, eval_every="final"),
               "replay_default": run_fast(lambda: ReplayNet(common()), st, sd, eval_every="final"),
               "joint_128": run_fast(lambda: JointOracle(common(hidden=128)), st, sd, eval_every="final"),
               "kNN_all": run_fast(lambda: KnnMemory(CommonCfg(seed=sd, d=st.d, n_classes=st.n_classes, inner_steps=1)), st, sd, eval_every="final")}
        for w in (128, 512, 2048):
            buf = max(256, min(samples, int((budget - 4 * MLPExpert(st.d, w, 1, st.n_classes).num_params()) // (4 * st.d))))
            out[f"replay_same_bytes_w{w}"] = run_fast(lambda w=w, buf=buf: ReplayNet(common(hidden=w), replay_size=buf), st, sd, eval_every="final")
        rows[sd] = out
        print(f"seed {sd}: " + "  ".join(f"{k} {100 * v.acc:5.1f}" for k, v in out.items()) + f"   (A_set {r_set.report.n_experts} experts, {budget / 1e6:.2f} MB; stream {samples} samples)", flush=True)
    names = list(rows[SEEDS[0]])
    summary = {k: [100 * rows[s][k].acc for s in SEEDS] for k in names}
    print("\nmean over seeds 80-84:", {k: round(float(np.mean(v)), 1) for k, v in summary.items()})
    d = paired_diff(summary["A_set"], summary["A_today"])
    wins = sum(a - b >= 5 for a, b in zip(summary["A_set"], summary["A_today"]))
    best_same = max((k for k in names if k.startswith("replay_same_bytes")), key=lambda k: np.mean(summary[k]))
    gap = float(np.mean(summary[best_same]) - np.mean(summary["A_set"]))
    print(f"A_set - A_today: {d['mean_diff']:+.1f} points [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}], >= +5 on {wins} of {len(SEEDS)} seeds (prediction: >= 4)")
    print(f"best replay with A_set's bytes: {best_same}; its mean minus A_set's = {gap:+.1f} points (prediction: A_set within 5 points)")
    (HERE / "m5d_fresh.json").write_text(json.dumps({"seeds": list(SEEDS), "steps": STEPS, "acc": summary,
                                                      "experts_A_set": [rows[s]["A_set"].report.n_experts for s in SEEDS],
                                                      "bytes": {k: [rows[s][k].report.memory_bytes for s in SEEDS] for k in names}}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f}s")

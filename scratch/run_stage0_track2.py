import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import time
from experiments.real_pilot_data import RealCorpora, SEQ_LEN
from experiments.run_v8_cognitive_pilot import V8RealBenchmark
from experiments.run_v9_cognitive_pilot import (
    DEFAULT_V9_CFG,
    run_track2_arm_v9,
    RESULTS_ROOT,
    _agg_runs,
)

out_dir = RESULTS_ROOT / "stage0_track2"
out_dir.mkdir(parents=True, exist_ok=True)

corpora = RealCorpora(seq_len=SEQ_LEN, eval_split="test")
bench = V8RealBenchmark(corpora, eval_split="test")

arms = [
    "Cog_v8",
    "Cog_v8_NoEsc",
    "Cog_v8_SequenceTop1",
    "Cog_v8_SequenceTop1_NoEsc",
    "Dense_v8_Matched",
    "Cog_v8_NoRel",
    "Cog_v8_NoI7",
    "Cog_v8_NoV8Core",
    "Cog_v8_FrozenRandomRouter",
]

seeds = [0]
results_by_arm = {a: [] for a in arms}

print(f"Starting Stage 0 Track 2 across {len(arms)} arms and {len(seeds)} seeds...", flush=True)

for seed in seeds:
    for arm in arms:
        t0 = time.time()
        res = run_track2_arm_v9(arm, DEFAULT_V9_CFG, bench, seed=seed, steps_per_dom=250)
        dt = time.time() - t0
        results_by_arm[arm].append(res)
        (out_dir / f"seed{seed}_{arm}.json").write_text(json.dumps(res, indent=2))
        
        b77_acc = res['banking77']['intent_77_acc']
        b77_auc = res['banking77']['same_vs_overlap_diff_auc']
        r5 = res.get('mean_final_r5', 0.0)
        hf = res.get('heldout_families', {}).get('auc', 0.0)
        mean_k = res.get('routing_diag', {}).get('mean_k', 1.0)
        ami = res.get('routing_diag', {}).get('func_routing_ami', {}).get('ami', 0.0)
        
        print(
            f"[Stage0 T2 s{seed} {arm}] b77_acc={b77_acc:.3f} b77_auc={b77_auc:.3f} "
            f"r5={r5:.3f} hf={hf:.3f} mean_k={mean_k:.2f} ami={ami:.3f} ({dt:.1f}s)",
            flush=True,
        )

summary = {
    "meta": {
        "seeds": seeds,
        "steps_per_dom": 250,
        "arms": arms,
    },
    "arms": {a: _agg_runs(results_by_arm[a]) for a in arms},
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print("Finished Stage 0 Track 2! Wrote summary.json", flush=True)

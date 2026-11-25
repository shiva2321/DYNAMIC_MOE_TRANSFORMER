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
)

corpora = RealCorpora(seq_len=SEQ_LEN, eval_split="test")
bench = V8RealBenchmark(corpora, eval_split="test")

print("Running Cog_v9 on Track 2 (seed 0)...", flush=True)
t0 = time.time()
res_v9 = run_track2_arm_v9("Cog_v9", DEFAULT_V9_CFG, bench, seed=0, steps_per_dom=250)
dt = time.time() - t0

b77_acc = res_v9['banking77']['intent_77_acc']
b77_auc = res_v9['banking77']['same_vs_overlap_diff_auc']
r5 = res_v9.get('mean_final_r5', 0.0)
hf = res_v9.get('heldout_families', {}).get('auc', 0.0)
mean_k = res_v9.get('routing_diag', {}).get('mean_k', 1.0)
ami = res_v9.get('routing_diag', {}).get('func_routing_ami', {}).get('ami', 0.0)

print(
    f"Cog_v9 T2 seed 0: b77_acc={b77_acc:.3f}, b77_auc={b77_auc:.3f}, "
    f"r5={r5:.3f}, hf={hf:.3f}, mean_k={mean_k:.2f}, ami={ami:.3f} ({dt:.1f}s)",
    flush=True,
)

out_file = RESULTS_ROOT / "stage0_track2" / "seed0_Cog_v9.json"
out_file.write_text(json.dumps(res_v9, indent=2))
print(f"Wrote {out_file}", flush=True)

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

out_dir = RESULTS_ROOT / "stage0_d10_steps"
out_dir.mkdir(parents=True, exist_ok=True)

corpora = RealCorpora(seq_len=SEQ_LEN, eval_split="test")
bench = V8RealBenchmark(corpora, eval_split="test")

print("Running D10 steps scaling: Cog_v8 with 500 steps per domain...", flush=True)
t0 = time.time()
res_500 = run_track2_arm_v9("Cog_v8", DEFAULT_V9_CFG, bench, seed=0, steps_per_dom=500)
dt = time.time() - t0

r5_500 = res_500.get('mean_final_r5', 0.0)
b77_acc = res_500['banking77']['intent_77_acc']
print(f"500 steps: r5={r5_500:.3f}, b77_acc={b77_acc:.3f} ({dt:.1f}s)", flush=True)

(out_dir / "seed0_Cog_v8_500steps.json").write_text(json.dumps(res_500, indent=2))
print("Finished D10 steps scaling test!", flush=True)

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from experiments.run_v9_cognitive_pilot import run_track1_arm_v9, DEFAULT_V9_CFG, RESULTS_ROOT

out_dir = RESULTS_ROOT / "stage0_d1_dense_sweep"
out_dir.mkdir(parents=True, exist_ok=True)

results = {}
for lr in (1e-3, 3e-3, 8e-3):
    arm_name = f"Dense_AdamW_lr_{lr}"
    print(f"Running Track 1 {arm_name} seed 0...", flush=True)
    res = run_track1_arm_v9("Dense_v8_AdamW", DEFAULT_V9_CFG, seed=0, steps_scale=1.0, dense_adamw_lr=lr)
    results[arm_name] = res
    print(f"  -> seen={res['seen_acc']:.3f}, held={res['heldout_unseen_acc']:.3f}, iit={res['iit_causal_acc']:.3f}", flush=True)

print("Running Track 1 Dense_v8_Matched (MuonBONGOptimizer) seed 0...", flush=True)
res_matched = run_track1_arm_v9("Dense_v8_Matched", DEFAULT_V9_CFG, seed=0, steps_scale=1.0)
results["Dense_v8_Matched_MuonBONG"] = res_matched
print(f"  -> seen={res_matched['seen_acc']:.3f}, held={res_matched['heldout_unseen_acc']:.3f}, iit={res_matched['iit_causal_acc']:.3f}", flush=True)

print("Running Track 1 Cog_v8_FrozenRandomRouter (MuonBONGOptimizer) seed 0...", flush=True)
res_frr = run_track1_arm_v9("Cog_v8_FrozenRandomRouter", DEFAULT_V9_CFG, seed=0, steps_scale=1.0)
results["Cog_v8_FrozenRandomRouter"] = res_frr
print(f"  -> seen={res_frr['seen_acc']:.3f}, held={res_frr['heldout_unseen_acc']:.3f}, iit={res_frr['iit_causal_acc']:.3f}", flush=True)

(out_dir / "d1_results.json").write_text(json.dumps(results, indent=2))
print("Finished D1 sweep! Wrote results to d1_results.json", flush=True)

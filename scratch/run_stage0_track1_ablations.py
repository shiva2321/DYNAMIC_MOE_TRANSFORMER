import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from experiments.run_v9_cognitive_pilot import run_track1_arm_v9, DEFAULT_V9_CFG, RESULTS_ROOT

out_dir = RESULTS_ROOT / "stage0_track1_seed0"
out_dir.mkdir(parents=True, exist_ok=True)

arms = [
    "Cog_v8_NoI5",
    "Cog_v8_NoRel",
    "Cog_v8_NoV8Core",
    "Cog_v8_SequenceTop1",
    "Cog_v8_GpmNonExempt",
]

results = {}
for arm in arms:
    print(f"Running Track 1 {arm} seed 0...", flush=True)
    res = run_track1_arm_v9(arm, DEFAULT_V9_CFG, seed=0, steps_scale=1.0)
    results[arm] = res
    print(
        f"  -> seen={res['seen_acc']:.3f}, held={res['heldout_unseen_acc']:.3f}, "
        f"iit_seen={res['iit_causal_acc']:.3f}, iit_held={res['iit_heldout_acc']:.3f}, "
        f"rand_slice={res['iit_random_slice_null']:.3f}, near_ood_auc={res['near_ood_abstain_auroc']:.3f}",
        flush=True,
    )

(out_dir / "track1_seed0_ablations.json").write_text(json.dumps(results, indent=2))
print("Finished Track 1 ablations! Wrote track1_seed0_ablations.json", flush=True)

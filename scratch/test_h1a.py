import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from experiments.run_v9_cognitive_pilot import run_track1_arm_v9, DEFAULT_V9_CFG

print("Running Track 1 Cog_v9_H1a (Constant Fisher scale = 1.0)...", flush=True)
res_h1a = run_track1_arm_v9("Cog_v9_H1a", DEFAULT_V9_CFG, seed=0, steps_scale=1.0)
print(
    f"Cog_v9_H1a seed 0: seen={res_h1a['seen_acc']:.3f}, held={res_h1a['heldout_unseen_acc']:.3f}, "
    f"iit_seen={res_h1a['iit_causal_acc']:.3f}, iit_held={res_h1a['iit_heldout_acc']:.3f}",
    flush=True,
)
print("Grokking curve:", [round(x, 3) for x in res_h1a['grokking_curve']], flush=True)

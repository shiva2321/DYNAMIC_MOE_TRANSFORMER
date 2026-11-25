import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from experiments.run_v9_cognitive_pilot import run_track1_arm_v9, DEFAULT_V9_CFG

cfg = dict(DEFAULT_V9_CFG)
cfg["use_constant_fisher"] = True
cfg["use_rel_pool"] = False

print("Running Track 1 Cog_v9_Candidate (Constant Fisher scale = 1.0 + NoRel)...", flush=True)
res_cand = run_track1_arm_v9("Cog_v8_NoRel", cfg, seed=0, steps_scale=1.0)
print(
    f"Cog_v9_Candidate seed 0: seen={res_cand['seen_acc']:.3f}, held={res_cand['heldout_unseen_acc']:.3f}, "
    f"iit_seen={res_cand['iit_causal_acc']:.3f}, iit_held={res_cand['iit_heldout_acc']:.3f}, "
    f"rand_slice={res_cand['iit_random_slice_null']:.3f}",
    flush=True,
)
print("Grokking curve:", [round(x, 3) for x in res_cand['grokking_curve']], flush=True)

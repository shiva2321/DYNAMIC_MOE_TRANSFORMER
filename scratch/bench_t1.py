import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
import torch
from experiments.run_v8_cognitive_pilot import run_track1_arm_v8, DEFAULT_V8_CFG

t0 = time.time()
res = run_track1_arm_v8('Cog_v8', DEFAULT_V8_CFG, 0, steps_scale=1.0)
dt = time.time() - t0
print(f"Track 1 Cog_v8 seed 0 took {dt:.1f}s")
print(f"seen={res['seen_acc']:.3f}, held={res['heldout_unseen_acc']:.3f}, iit={res['iit_causal_acc']:.3f}, ep={res['episodic_fewshot_acc']:.3f}")

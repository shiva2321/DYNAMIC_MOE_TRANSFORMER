import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from experiments.run_v8_cognitive_pilot import build_v8_arm, DEFAULT_V8_CFG, DEV
from dmoe.spgc import classify_v7_spgc_param_groups, MuonBONGOptimizer

model, gpm, opt, gate = build_v8_arm("Cog_v8", DEFAULT_V8_CFG, seed=0, steps_per_dom=200)
x = torch.randint(33, 122, (16, 128), device=DEV)

# Warmup
for _ in range(3):
    out = model(x, update_stats=True)
    loss = out["latent"].sum()
    loss.backward()
    opt.zero_grad()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_step"):
        out = model(x, update_stats=True)
        loss = out["latent"].sum()
        loss.backward()
        opt.step(fisher_scale=1.0)
        opt.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

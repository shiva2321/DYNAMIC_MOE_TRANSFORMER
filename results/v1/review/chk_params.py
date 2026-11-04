import os, sys
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0, "D:/spherical_hyperspace_memory")
from costlab.env import fixed_threads
import m1.beds, v0.system
with fixed_threads(n=2, cpus=[8, 10]):
    bed = m1.beds.build_bed("B3seq", 22, 800)
    arm = v0.system.build_v0(bed)
    pool = arm.model
    print("pool type", type(pool).__name__)
    # drive a few batches so at least one expert exists
    n = 0
    for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
        arm.observe(x_b, y_b); n += 1
        if n >= 3: break
    print("experts", list(pool.experts.keys()))
    rec = next(iter(pool.experts.values()))
    print("module type", type(rec.module).__name__)
    names = [nm for nm, _ in rec.module.named_parameters()]
    print("n named params", len(names), names[:8])
    print("hasattr(module, name) for each:", [hasattr(rec.module, nm) for nm in names[:8]])
    print("pred.expert dtype/sample:", pool.predict(x_b[:4]).expert)
    print("birth_step attr:", rec.birth_step, "id", rec.id)

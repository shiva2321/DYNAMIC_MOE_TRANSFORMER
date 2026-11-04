import copy, os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "D:/spherical_hyperspace_memory")
import torch
from costlab.env import fixed_threads
import m1.beds, v0.system
from v1.pool_removal import ExposureTracker, remove_labels
from v1.run_pool_experiments import REMOVAL_SETS

with fixed_threads(n=2, cpus=[8, 10]):
    bed = m1.beds.build_bed("B3seq", 22, 801)
    arm = v0.system.build_v0(bed)
    tr = ExposureTracker(arm.model)
    stream = []
    for x, y, *_ in bed.walk_full(64):
        stream.append((x, y)); arm.observe(x, y)
    pool = arm.model
    # runtime inventory
    keys = {}
    for e, rec in pool.experts.items():
        for k, v in rec.meta.items():
            keys.setdefault(k, set()).add(type(v).__name__)
    print("expert meta keys:", {k: sorted(v) for k, v in keys.items()})
    print("pool attrs:", sorted(k for k in vars(pool).keys() if not k.startswith("__")))
    kinds = {}
    for ev in pool.events._events:
        kinds.setdefault(ev.kind, set()).update(ev.fields.keys())
    print("event kinds:", {k: sorted(v) for k, v in kinds.items()})
    print("arm attrs:", sorted(vars(arm).keys()))
    print("expert 0 replay labels unique:", torch.unique(pool.experts[0].replay.y[:pool.experts[0].replay.n]).tolist()[:40])
    print("route eids before:", pool._route.eids if hasattr(pool, "_route") else None, "live:", sorted(pool.experts))
    labels = REMOVAL_SETS["R1"]
    x_rem = torch.cat([x[torch.isin(y, torch.tensor(labels))] for x, y in stream]); y_rem = torch.cat([y[torch.isin(y, torch.tensor(labels))] for x, y in stream])
    for attempt in (1, 2):
        arm_r = copy.deepcopy(arm)
        cert = remove_labels(arm_r.model, labels, data=(x_rem, y_rem), tracker=tr)
        print(f"attempt {attempt}: dropped {cert['exposure_summary']['dropped_experts']} route eids {arm_r.model._route.eids} live {sorted(arm_r.model.experts)} clean {cert['certified_clean']} router stale {cert['trace_audit']['pool_router_stale_eids']}")
    # who exposed expert 12 directly?
    print("direct labels of 12 (first 30):", sorted(tr.expert_direct_labels.get(12, []))[:30], "contains removed:", bool(set(labels) & tr.expert_direct_labels.get(12, set())))

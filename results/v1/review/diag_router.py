import copy, os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "D:/spherical_hyperspace_memory")
import numpy as np, torch
from costlab.env import fixed_threads
import m1.beds, v0.system
from v1.pool_removal import ExposureTracker, remove_labels
from v1.run_pool_experiments import REMOVAL_SETS

seed, rid = int(sys.argv[1]), sys.argv[2]


def mean_regime_acc(model, bed, labels):
    accs = []
    for r in sorted(bed.test_sets.keys()):
        tx, ty = bed.test_sets[r]
        m = ~torch.isin(ty, torch.tensor(labels))
        if bool(m.any()):
            p = model.predict(tx[m]).probs.argmax(-1).cpu()
            accs.append(float((p == ty[m].cpu()).float().mean()))
    return float(np.mean(accs))


with fixed_threads(n=2, cpus=[8, 10]):
    bed = m1.beds.build_bed("B3seq", 22, seed)
    arm = v0.system.build_v0(bed)
    tr = ExposureTracker(arm.model)
    stream = []
    for x, y, *_ in bed.walk_full(64):
        stream.append((x, y)); arm.observe(x, y)
    labels = REMOVAL_SETS[rid]
    x_rem = torch.cat([x[torch.isin(y, torch.tensor(labels))] for x, y in stream]); y_rem = torch.cat([y[torch.isin(y, torch.tensor(labels))] for x, y in stream])
    inst = [k for k in ("_lda_absorb", "_refit_router", "observe", "train_expert") if k in vars(arm.model)]
    print("instance-level callables on pool:", inst)
    arm_r = copy.deepcopy(arm)
    print("copy's wrapped methods are the originals' (same function object):", {k: getattr(arm_r.model, k).__code__ is getattr(arm.model, k).__code__ for k in inst})
    cert = remove_labels(arm_r.model, labels, data=(x_rem, y_rem), tracker=tr)
    print("route eids of copy right after remove_labels:", arm_r.model._route.eids, "| original's:", arm.model._route.eids, "live in copy:", sorted(arm_r.model.experts))
    a_gem = mean_regime_acc(arm_r.model, bed, labels)
    rule_gem = arm_r.model._route.name
    # honest copy: drop the closure-bound wrappers so the class methods act on the copy
    for k in inst:
        if k in vars(arm_r.model):
            del arm_r.model.__dict__[k]
    arm_r.model._route.eids = None
    arm_r.model._refit_router()
    a_fix = mean_regime_acc(arm_r.model, bed, labels)
    print(f"seed {seed} {rid}: P_removed retained accuracy as Gemini evaluated it (router not refit in the copy, rule {rule_gem!r}): {a_gem:.4f}; "
          f"with the copy's own router refit (rule {arm_r.model._route.name!r}, eids {arm_r.model._route.eids}): {a_fix:.4f}")
    print("P_full:", f"{mean_regime_acc(arm.model, bed, labels):.4f}", "rule", arm.model._route.name)

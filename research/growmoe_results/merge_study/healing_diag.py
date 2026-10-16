"""Merge study, diagnosis of the first healing study (throwaway; no product file touched).

Two questions about healing_study.py:
  1. In the injection arms the pool ends with 25 experts, not 12 + 24 = 36: did the FIRST injection of every regime found the regime's real expert (pre-empting the natural
     founding), so that the 'duplicates' were not all duplicates and the accuracy loss came from late, short training and not from duplication?
  2. Which injected experts could the redirect merge even look at (state, replay size, reference length), and which did it merge?

The walk below repeats healing_study.run() batch by batch and writes down, for every expert, when it was founded (segment, batch inside the segment), by whom (natural / injection),
and at the end its state. It also logs the pool's honest accuracy on every incoming batch BEFORE it learns it (prequential), which says when a regime counts as known.

    python healing_diag.py [seed]
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import healing_study as H  # noqa: E402
from growmoe.fairfight import _Scorer  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402


def walk(arm: str, seed: int, every: int = 3) -> dict:
    st = function_regimes(seed, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=640, n_test=200)
    pool = H.make_pool(arm, seed, None)
    founded, preq, last_k, within = {}, [], None, 0
    seen_ids = set()
    for k, seg, idx in st._walk(64, seed):
        if k != last_k:
            last_k, within = k, 0
        within += 1
        x, y = seg.x[idx], seg.y[idx]
        if len(pool.experts):
            with torch.no_grad():
                pred = pool.predict(x).probs.argmax(1)
            acc_before = float((pred == y).float().mean())
        else:
            acc_before = 0.25
        before = set(pool.experts)
        pool.observe(x, y)
        for e in set(pool.experts) - before:
            founded[e] = {"segment": k, "batch": within, "by": "natural", "regime": seg.regime, "first_visit": bool(seg.first_visit)}
        did_inject = False
        if arm != "clean" and seg.first_visit and within >= 4 and within % every == 0 and len(pool.experts) > 0:
            before = set(pool.experts)
            H.inject(pool, x, y)
            for e in set(pool.experts) - before:
                founded[e] = {"segment": k, "batch": within, "by": "injection", "regime": seg.regime, "first_visit": bool(seg.first_visit), "acc_before": acc_before}
                did_inject = True
        preq.append({"segment": k, "batch": within, "regime": seg.regime, "acc_before": acc_before, "n_experts": len(pool.experts), "inject": did_inject})
    row = _Scorer(pool, st).row()
    final = {}
    for e, r in pool.experts.items():
        final[e] = {"state": r.state, "parent": r.parent, "replay_n": int(r.replay.n) if r.replay is not None else None, "ref_n": len(r.meta.get("ref", ())),
                    "n_proto": int(r.meta["sig"].n)}
    healed = [{"step": s, "kept": i, "removed": j} for s, i, j in getattr(pool, "healed", [])]
    return {"arm": arm, "seed": seed, "founded": {str(e): v for e, v in founded.items()}, "preq": preq, "final": {str(e): v for e, v in final.items()}, "healed": healed,
            "acc": float(row.mean()), "per_regime": row.tolist(), "alive": sorted(pool.experts)}


def summarise(r: dict) -> None:
    f = r["founded"]
    nat = [e for e, v in f.items() if v["by"] == "natural"]
    inj = [e for e, v in f.items() if v["by"] == "injection"]
    print(f"\n=== arm {r['arm']} seed {r['seed']}: acc {100 * r['acc']:.1f}  experts founded {len(f)} (natural {len(nat)}, injected {len(inj)}), alive {len(r['alive'])}, healed {len(r['healed'])}")
    by_seg = {}
    for e, v in f.items():
        by_seg.setdefault(v["segment"], []).append((v["batch"], v["by"][0], int(e)))
    for k in sorted(by_seg):
        print(f"  segment {k:2d}: founded at batch/by/id: {sorted(by_seg[k])}")
    print("  prequential accuracy before learning each batch (segment: batches 1..10), mean over segments of batch index b:")
    for b in range(1, 11):
        a = [p["acc_before"] for p in r["preq"] if p["batch"] == b and p["segment"] < H.REGIMES]
        print(f"    batch {b:2d}: {100 * np.mean(a):5.1f}", end="")
    print()
    alive_inj = [e for e in inj if int(e) in r["alive"]]
    print(f"  injected experts still alive: {len(alive_inj)} of {len(inj)}")
    for e in inj:
        fin = r["final"].get(e)
        tag = "alive" if fin else "gone "
        extra = f"state {fin['state']:8s} replay {fin['replay_n']} ref {fin['ref_n']}" if fin else ""
        print(f"    id {e:>3s} segment {f[e]['segment']:2d} batch {f[e]['batch']:2d} pool acc on that batch before {100 * f[e]['acc_before']:5.1f}  {tag} {extra}")


def main() -> None:
    torch.set_num_threads(2)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = HERE / f"healing_diag_s{seed}.json"
    res = []
    for arm in ("clean", "none", "redirect"):
        r = walk(arm, seed)
        res.append(r)
        summarise(r)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


if __name__ == "__main__":
    main()

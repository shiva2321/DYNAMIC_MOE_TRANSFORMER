"""Audit of the merge routine (stage-4 spec section 5.5): can PhyloMoE._attempt_merges clean up a forced duplicate of ONE function?   (throwaway; no product file touched)

Setting (the easiest case there is): a pool learns one linear teacher (hidden 32, 5 inner steps, 30 batches of 64); the probe is told it can never accept an extend
(delta_forget = -1), a cluster of the same function is then handed to it, which founds a duplicate; ten more batches of the same function settle the infant; then the numbers
inside the merge acceptance test are recomputed exactly as ``_attempt_merges`` computes them and the routine itself is run.

    python audit_merge.py            # writes audit_merge.json next to this file
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests" / "growmoe"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rv4_helpers import regime_batch, trained_pool  # noqa: E402


def accuracy(module, x, y) -> float:
    v = y >= 0
    with torch.no_grad():
        return float((module(x[v]).argmax(1) == y[v]).float().mean())


def audit(seed: int) -> dict:
    pool, W = trained_pool(n_batches=30, seed=seed, spawn_rule="probe", delta_forget=-1.0)
    assert pool._handle_cluster(*regime_batch(W, 48, 9100 + seed), evidence=50.0) == 1
    for k in range(10):
        pool.observe(*regime_batch(W, 64, 9200 + 10 * seed + k))
    ri, rj = pool.experts[0], pool.experts[1]
    si, sj = ri.meta["sig"], rj.meta["sig"]
    overlap = {"p_j_on_i": float(sj.pvalue(si.protos).mean()), "p_i_on_j": float(si.pvalue(sj.protos).mean()), "needed": pool.cfg.merge_overlap}
    rx_i, ry_i = ri.replay.data
    rx_j, ry_j = rj.replay.data
    hi, hj = len(rx_i) // 2, len(rx_j) // 2                                        # the routine's split: fit half / validation half
    ux, uy = torch.cat([rx_i[:hi], rx_j[:hj]]), torch.cat([ry_i[:hi], ry_j[:hj]])
    clone = ri.module.clone()
    clone.mark_seen(uy[uy >= 0])
    opt = torch.optim.Adam(clone.parameters(), lr=pool.cfg.lr)
    bs = min(pool.cfg.batch_size, len(ux))
    for s in range(pool.cfg.probe_steps):
        gen = torch.Generator().manual_seed(pool.cfg.seed + 3000 * pool.step + s)
        sel = torch.randint(0, len(ux), (bs,), generator=gen)
        clone.train_step(ux[sel], uy[sel], opt, replay=None)
    pre_i, post_i = accuracy(ri.module, rx_i[hi:], ry_i[hi:]), accuracy(clone, rx_i[hi:], ry_i[hi:])
    pre_j, post_j = accuracy(rj.module, rx_j[hj:], ry_j[hj:]), accuracy(clone, rx_j[hj:], ry_j[hj:])
    own_replay = {"expert_i_on_its_replay": accuracy(ri.module, rx_i, ry_i), "expert_j_on_its_replay": accuracy(rj.module, rx_j, ry_j)}
    n0 = len(pool.events)
    pool._attempt_merges()
    return {"seed": seed, "overlap": overlap, "drop_i": pre_i - post_i, "drop_j": pre_j - post_j, "delta_merge": pool.cfg.delta_merge, **own_replay,
            "merged": any(e.kind == "merge" for e in pool.events.since(n0)), "experts_after": len(pool.experts)}


def main() -> None:
    rows = [audit(s) for s in range(4)]
    for r in rows:
        print(f"seed {r['seed']}: overlap {r['overlap']['p_j_on_i']:.2f}/{r['overlap']['p_i_on_j']:.2f} (needs {r['overlap']['needed']})  drop i {r['drop_i']:+.3f}  drop j {r['drop_j']:+.3f}  "
              f"(tolerance {r['delta_merge']})  replay accuracy {r['expert_i_on_its_replay']:.2f}/{r['expert_j_on_its_replay']:.2f}  merged {r['merged']}")
    print(f"{sum(r['merged'] for r in rows)} of {len(rows)} forced duplicates merged")
    out = HERE / "audit_merge.json"
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rows, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


if __name__ == "__main__":
    main()

"""Merge study, end to end (throwaway; no product file touched): does a pool HEAL when wrong experts are founded, and what happens when it cannot?

A function-regime stream (12 regimes, each its own label function, plus two revisits) is learned by a pool in which false spawns are INJECTED on purpose: every few batches inside a known
regime the current batch is handed to ``_handle_cluster`` with the probe told it may never extend, which is exactly how the merge audit forced a duplicate. Arms:

    clean         no injection (what the pool does when nothing goes wrong)
    none          injections, merge never runs                    (today's reality: the shipped merge cannot repair a duplicate)
    shipped       injections, the shipped ``_attempt_merges``
    redirect      injections, ``_attempt_merges`` replaced by the transfer test BETWEEN experts: retire X when Y already predicts X's stored labelled samples as well as X honestly did
                  (exact permutation test + a tight margin, signature coverage as a prefilter, no weight is touched, X's signature / replay / reference move into Y)

With ``--cap`` the pool may hold only a few more experts than there are regimes: the arms then show what the silent stop looks like.

    python healing_study.py [--cap]
"""

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from growmoe.config import PhyloCfg  # noqa: E402
from growmoe.fairfight import _Scorer  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.settest import generator  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402
from growmoe.transfer import cluster_conflict  # noqa: E402

REGIMES, D, C = 12, 32, 4


class HealingPhylo(PhyloMoE):
    """PhyloMoE whose merge is a redirect: retire an expert that another expert already predicts as well as it did itself (honest comparison, no training)."""

    def __init__(self, cfg, margin_k=0.10):
        super().__init__(cfg)
        self.margin_k, self.healed = margin_k, []

    def _redundant(self):
        cfg, best = self.cfg, None
        ok = [e for e, r in self.experts.items() if r.state != "infant" and r.replay is not None and r.replay.n >= 64 and len(r.meta.get("ref", ())) >= cfg.transfer_min_ref]
        for i in ok:
            for j in ok:
                if i == j:
                    continue
                ri, rj = self.experts[i], self.experts[j]
                if float(ri.meta["sig"].pvalue(rj.meta["sig"].protos).mean()) < cfg.merge_overlap:          # the cheap prefilter: do the signatures cover each other at all?
                    continue
                rx, ry = rj.replay.data
                with torch.no_grad():
                    losses = torch.nn.functional.cross_entropy(ri.module(rx), ry, reduction="none")
                ref = torch.tensor(list(rj.meta["ref"]))
                margin = self.margin_k * math.log(max(int(rj.module.seen.sum()), 2))
                conflict, p, diff = cluster_conflict(losses, ref, cfg.transfer_alpha, cfg.transfer_perms, margin, generator(cfg.seed, 0x4D47, self.step, i, j))
                if conflict:
                    continue
                score = (float(np.mean(ri.meta["ref"])), i)                                                   # when both directions pass keep the one with the lower honest loss, then the older
                if best is None or (diff, score) < best[0]:
                    best = ((diff, score), i, j)
        return None if best is None else (best[1], best[2])

    def _attempt_merges(self) -> None:
        for _ in range(4):
            pair = self._redundant()
            if pair is None:
                return
            self._absorb(*pair)

    def _absorb(self, i: int, j: int) -> None:
        ri, rj = self.experts[i], self.experts[j]
        ri.meta["sig"].merge_from(rj.meta["sig"])
        ri.meta["sig"].fit()
        rx, ry = rj.replay.data
        ri.replay.add(rx, ry)
        if rj.meta.get("windows") is not None:
            self._windows_of(ri).merge_from(rj.meta["windows"])
        self._merge_refs(ri, rj)
        for c_id, c_rec in list(self.experts.items()):
            if c_rec.parent == j:
                self.set_parent(c_id, rj.parent if c_id == i else i)           # the survivor may be the retired expert's own child: it then takes its place in the lineage
        for k_id, k_rec in list(self.experts.items()):
            if k_rec.meta.get("supersedes") == j:
                k_rec.meta.pop("supersedes", None)
        self.remove_expert(j)
        self.healed.append((self.step, i, j))
        self.events.add(kind="merge", step=self.step, kept=i, removed=j, change_kept=0.0, change_removed=0.0)


def make_pool(arm: str, seed: int, cap):
    cfg = PhyloCfg(router=False, growth_rule="probe", d=D, hidden=32, n_classes=C, seed=seed, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False,
                   spawn_rule="transfer", transfer_min_ref=64, merge_every=(10 ** 9 if arm in ("clean", "none") else 10), max_experts=cap if cap else 256)
    return HealingPhylo(cfg) if arm == "redirect" else PhyloMoE(cfg)


def inject(pool, x, y) -> bool:
    saved = pool.cfg.spawn_rule, pool.cfg.delta_forget
    pool.cfg.spawn_rule, pool.cfg.delta_forget = "probe", -1.0                      # the probe can never accept an extend: this cluster founds an expert, known regime or not
    before = len(pool.experts)
    try:
        pool._handle_cluster(x, y, evidence=50.0)
    finally:
        pool.cfg.spawn_rule, pool.cfg.delta_forget = saved
    return len(pool.experts) > before


def run(arm: str, seed: int, every: int = 3, cap=None) -> dict:
    st = function_regimes(seed, n_regimes=REGIMES, d=D, n_classes=C, n_train=640, n_test=200)
    pool = make_pool(arm, seed, cap)
    counts, within, last_k, injected, cap_step = [], 0, None, 0, None
    for k, seg, idx in st._walk(64, seed):
        if k != last_k:
            if last_k is not None:
                counts.append(len(pool.experts))
            last_k, within = k, 0
        within += 1
        x, y = seg.x[idx], seg.y[idx]
        pool.observe(x, y)
        if arm != "clean" and seg.first_visit and within >= 4 and within % every == 0 and len(pool.experts) > 0:
            injected += int(inject(pool, x, y))
        if cap and cap_step is None and len(pool.experts) >= cap:
            cap_step = k
    counts.append(len(pool.experts))
    row = _Scorer(pool, st).row()
    first = {}
    for k, seg, _ in st._walk(64, seed):
        if seg.first_visit and seg.regime not in first:
            first[seg.regime] = k
    late = [r for r, k in first.items() if cap_step is not None and k > cap_step]
    return {"arm": arm, "seed": seed, "cap": cap, "final_experts": len(pool.experts), "experts_after_each_segment": counts, "injected": injected, "merges": len(getattr(pool, "healed", [])) or sum(e.kind == "merge" for e in pool.events),
            "acc": float(row.mean()), "per_regime": row.tolist(), "regimes": REGIMES, "cap_reached_in_segment": cap_step, "late_regimes": late,
            "acc_late": float(np.mean([row[r] for r in late])) if late else None}


def main() -> None:
    torch.set_num_threads(2)
    cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else None
    out = HERE / ("healing_cap.json" if cap else "healing.json")
    arms = ["clean", "none", "shipped", "redirect"]
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["arm"], r["seed"]) for r in res}
    for seed in range(3):
        for arm in arms:
            if (arm, seed) in done:
                continue
            res.append(run(arm, seed, cap=cap))
            tmp = out.with_name(out.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(res, indent=1))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, out)
            r = res[-1]
            print(f"seed {seed} {arm:9s} experts {r['final_experts']:3d} (injected {r['injected']:2d}, merged {r['merges']:2d}) acc {100 * r['acc']:.1f}" + (f" | late regimes {100 * r['acc_late']:.1f}" if r["acc_late"] is not None else ""), flush=True)
    print("\narm        final experts  accuracy   (mean of 3 seeds; regimes = %d)" % REGIMES)
    for arm in arms:
        rs = [r for r in res if r["arm"] == arm]
        late = [r["acc_late"] for r in rs if r["acc_late"] is not None]
        print(f"{arm:9s}  {np.mean([r['final_experts'] for r in rs]):6.1f}      {100 * np.mean([r['acc'] for r in rs]):5.1f}" + (f"   late-regime accuracy {100 * np.mean(late):.1f}" if late else ""))


if __name__ == "__main__":
    main()

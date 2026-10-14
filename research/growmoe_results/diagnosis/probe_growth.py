"""Study 4: why does the pool's capacity growth fire in some runs and not in others, and is its decision based on enough evidence?

Instrumentation only. (1) Every call of ``GrowthPolicy.step`` is logged with the size of the data it decided on, its candidates' gains, p-values and its verdict; (2) every time the
plateau trigger returns True is logged; (3) the batch-loss curve of the training run is kept, so the trigger can be replayed offline; (4) at the end the same policy is asked again on
the FINAL expert with a much larger training and validation set drawn from the stream, to see what it would have decided with enough evidence.

    python probe_growth.py --config shipped --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
import growmoe.growth as growth  # noqa: E402
import growmoe.pools.base as base  # noqa: E402
import growmoe.pools.phylo as phylo  # noqa: E402
from growmoe.config import GrowthCfg  # noqa: E402

OUT = HERE / "growth"
CONFIGS = {"shipped": {"name": "GrowMoE", "kw": {}}, "tuned": {"name": "GrowMoE-h512-s5-r20000", "kw": {"hidden": 512, "replay_size": 20000}}}
LOG = {"plateau_true": [], "calls": [], "losses": []}


def install():
    orig_plateau = phylo.plateau

    def plateau_logged(history, window=40, tol=0.01):
        r = orig_plateau(history, window, tol)
        if r:
            h = np.asarray(history, dtype=np.float64)
            LOG["plateau_true"].append({"n_batches_seen": len(LOG["losses"]), "prev_mean": float(h[-2 * window:-window].mean()), "recent_mean": float(h[-window:].mean())})
        return r

    phylo.plateau = plateau_logged

    orig_step = growth.GrowthPolicy.step

    def step_logged(self, expert, x_tr, y_tr, x_va, y_va, rng):
        d = orig_step(self, expert, x_tr, y_tr, x_va, y_va, rng)
        LOG["calls"].append({"n_batches_seen": len(LOG["losses"]), "n_train": int(len(x_tr)), "n_valid": int(len(x_va)), "hidden": int(expert.hidden), "depth": len(expert.blocks), "reason": d.reason,
                             "kind": d.kind, "best_gain": float(d.gain), "best_p": float(d.p_value), "tried": d.tried, "next_cooldown": self.cooldown, "backoff": self.backoff})
        return d

    growth.GrowthPolicy.step = step_logged

    orig_train = base.BasePool.train_expert

    def train_logged(self, eid, x, y, replay=True, project=False):
        loss = orig_train(self, eid, x, y, replay=replay, project=project)
        LOG["losses"].append(float(loss))
        return loss

    base.BasePool.train_expert = train_logged


def counterfactual(learner, train, seed):
    """Ask the growth policy again, on the final expert, with training/validation sets of different sizes drawn from the stream (no test data)."""
    rec = next(iter(learner.pool.experts.values()))
    expert = rec.module
    x, y = D.unit(train.x()), train.y
    rng = np.random.default_rng([seed, 0xC0F])
    out = {}
    for n_tr, n_va in ((205, 51), (4000, 1000), (16000, 4000)):
        idx = rng.permutation(len(x))[: n_tr + n_va]
        x_tr, y_tr = torch.from_numpy(x[idx[:n_tr]]), torch.from_numpy(y[idx[:n_tr]]).long()
        x_va, y_va = torch.from_numpy(x[idx[n_tr:]]), torch.from_numpy(y[idx[n_tr:]]).long()
        pol = growth.GrowthPolicy(GrowthCfg())
        t0 = time.time()
        d = pol.step(expert.clone(), x_tr, y_tr, x_va, y_va, np.random.default_rng(seed))
        out[f"{n_tr}/{n_va}"] = {"reason": d.reason, "kind": d.kind, "best_gain": float(d.gain), "best_p": float(d.p_value), "tried": d.tried, "seconds": time.time() - t0}
        LOG["calls"].pop()                                                    # the counterfactual calls are not part of the run's own log
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="shipped", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"{a.config}_s{a.seed}"
    if stem.with_suffix(".json").exists():
        return
    install()
    cfg = CONFIGS[a.config]
    train = L.build_train("FULL", a.seed, 8000)
    test = L.build_test(a.per_cell)
    seen = L.build_seen(train, 500, a.seed)
    learner = L.Pool(5, False, name=cfg["name"], **cfg["kw"])
    t0 = time.time()
    info = learner.fit(train, a.seed)
    res, _ = L.evaluate(learner, test, a.per_cell, seen)
    s = D.summarise(res)
    run_calls, run_plateaus = list(LOG["calls"]), list(LOG["plateau_true"])
    cf = counterfactual(learner, train, a.seed)
    # replay the trigger offline on the recorded loss curve (no clearing): how often WOULD the plateau test be true?
    losses = np.asarray(LOG["losses"])
    offline = []
    for t in range(80, len(losses)):
        prev, recent = losses[t - 80:t - 40].mean(), losses[t - 40:t].mean()
        offline.append(bool((prev - recent) < 0.01 * max(abs(prev), 1e-12)))
    out = {"config": a.config, "seed": a.seed, "summary": s, "train": info, "n_batches": len(losses), "run_calls": run_calls, "run_plateaus": run_plateaus,
           "offline_plateau_fraction": float(np.mean(offline)) if offline else None, "loss_curve_per_100_batches": [float(losses[i:i + 100].mean()) for i in range(0, len(losses), 100)],
           "counterfactual_final_expert": cf, "total_seconds": time.time() - t0}
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))
    print(f"[{a.config} s{a.seed}] unseen {100 * s['mean_exact']:.1f}  seen {100 * s['seen_exact']:.1f}  params {info['params']:,}  growth calls {len(run_calls)}  plateau-true {len(run_plateaus)}  "
          f"offline plateau fraction {out['offline_plateau_fraction']:.2f}  events {info['events']}", flush=True)
    for c in run_calls:
        print(f"   call at batch {c['n_batches_seen']}: n_train {c['n_train']} n_valid {c['n_valid']} hidden {c['hidden']} -> {c['reason']} best gain {100 * c['best_gain']:+.1f}% p {c['best_p']:.3f}", flush=True)
    for k, v in cf.items():
        print(f"   counterfactual {k}: {v['reason']} best gain {100 * v['best_gain']:+.1f}% p {v['best_p']:.3f}", flush=True)


if __name__ == "__main__":
    main()

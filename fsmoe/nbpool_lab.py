"""Runs the notebook-first pool against the shipped pool on real-pool test beds (``python -m fsmoe.nbpool_lab run --bed f30 --seeds 700 --arms U8 U32 U128 NB``); one JSON per (bed, seed, arm) in
results/fsmoe/N, written atomically, finished files skipped.

Beds: ``f10`` / ``f30``: growmoe's function-regime stream with 10 / 30 regimes (the viability-shootout stream, d = 128, 8 classes, each regime a small teacher network), read after the whole stream
(`growmoe.fairfight.run_fast`); ``t``: the FS-MoE three-domain teaching bed (real text embeddings, 101 skills), taught domain by domain and read on known skills in the teaching and in new wordings.
``g``: the growth lab's mixed-needs world (E-060: 16 regimes on a sphere, 8 of them needing a wide network, d = 32, 6 classes, 128-sample batches) run through the real pool, accuracy per regime on held test sets.
Arms: ``U<h>`` the shipped ULDA pool (a network of h units at birth, h = 8, 32 or 128 = the default), ``NB`` the notebook-first pool (candidate sizes judged on held-out samples)."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from growmoe.fairfight import run_fast
from growmoe.streams_m1 import function_regimes
from unimoe.systems2 import make_system2

from .bed import ROOT, build_bed
from .nbpool import NbPool, nb_cfg
from .system_lab import OUT as S_OUT, Prep, teach

OUT = ROOT / "results" / "fsmoe" / "N"
F_RECIPE = {"inner_steps": 5, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32}            # the pool recipe of the viability shootout
G_RECIPE = {"inner_steps": 2, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32}
NB_VARIANTS = {"NB": {}, "NBa": {"nb_steps": 100}, "NBb": {"nb_margin": 0.01}, "NBc": {"nb_sizes": ((8, 1), (32, 1))}, "NBd": {"nb_min": 100}, "NBn": {"nb_min": 10 ** 9}}      # NBn: exploratory, never judged = notebook experts only


def make_pool(arm: str, n_classes: int, seed: int, recipe: dict, d: int):
    if arm.startswith("U") and arm.endswith("w"):                      # E-062: the shipped pool with growmoe's transfer_chance flag, weak-parent-only (U8w, U128w)
        return make_system2("ULDA", n_classes, seed, {**recipe, "hidden": int(arm[1:-1]), "transfer_chance": 1.0, "transfer_chance_weak_only": True}, d=d)
    if arm.startswith("U") and arm.endswith("x"):                      # exploratory ablation: the shipped pool (networks at birth) with ONLY the worse-than-chance rule of the notebook-first pool
        return NbPool(nb_cfg(n_classes, seed, {**recipe, "hidden": int(arm[1:-1])}, d=d, nb_first=False))
    if arm.startswith("U"):
        return make_system2("ULDA", n_classes, seed, {**recipe, "hidden": int(arm[1:])}, d=d)
    return NbPool(nb_cfg(n_classes, seed, recipe, d=d, **NB_VARIANTS[arm]))


def _nb_part(pool) -> Optional[dict]:
    s = pool.summary().get("nb")
    if s is None:
        return None
    return {**s, "net_sizes": {str(k): list(v) for k, v in s["net_sizes"].items()}}


def run_f(n: int, seed: int, arm: str) -> dict:
    st = function_regimes(seed, n)
    holder = []

    def make():
        holder.append(make_pool(arm, st.n_classes, seed, F_RECIPE, st.d))
        return holder[0]
    res = run_fast(make, st, seed, eval_every="final")
    pool, rep = holder[0], res.report
    return {"bed": f"f{n}", "seed": seed, "arm": arm, "acc": res.acc, "min_regime_acc": float(np.min(res.per_regime)), "experts": rep.n_experts, "params": rep.params, "stored_vectors": rep.stored_vectors,
            "bytes": rep.memory_bytes, "predict_macs": rep.predict_macs, "expert_macs": rep.expert_macs, "seconds": res.wall_clock, "events": dict(Counter(e.kind for e in pool.events)),
            "nb": _nb_part(pool), "judging_macs": float(getattr(pool, "nb_macs", 0.0))}


def run_g(seed: int, arm: str) -> dict:
    from growmoe.accounting import pool_report
    from .grow.world import SCENARIOS, GrowthWorld
    w = GrowthWorld(seed, SCENARIOS["mixed"])
    pool = make_pool(arm, 6, seed, G_RECIPE, 32)
    t0 = time.perf_counter()
    for b in w.stream():
        pool.observe(b.x, b.y)
    acc = {rid: float((pool.predict(x).probs.argmax(1) == y).float().mean()) for rid, (x, y) in w.test.items()}
    rep = pool_report(pool, 32)
    return {"bed": "g", "seed": seed, "arm": arm, "acc": float(np.mean(list(acc.values()))), "min_regime_acc": float(min(acc.values())), "per_regime": {str(k): v for k, v in acc.items()}, "experts": rep.n_experts, "params": rep.params,
            "stored_vectors": rep.stored_vectors, "bytes": rep.memory_bytes, "predict_macs": rep.predict_macs, "expert_macs": rep.expert_macs, "seconds": time.perf_counter() - t0, "events": dict(Counter(e.kind for e in pool.events)),
            "nb": _nb_part(pool), "judging_macs": float(getattr(pool, "nb_macs", 0.0))}


def run_t(seed: int, arm: str) -> dict:
    bed = build_bed(seed)
    prep = Prep(bed)
    recipe = json.loads((S_OUT / "tuning.json").read_text(encoding="utf-8"))["FS_best"]
    pool = make_pool(arm, prep.n_classes, seed, recipe, prep.d)
    r = teach(pool, prep, seed)
    return {"bed": "t", "seed": seed, "arm": arm, "mean_tw": r["mean_tw"], "mean_ho": r["mean_ho"], "final": r["final"], "forgetting": r["forgetting"], "experts": r["n_experts"], "params": r["params"],
            "stored_vectors": r["stored_vectors"], "bytes": r["memory_bytes"], "seconds": r["seconds"], "events": r["events"], "nb": _nb_part(pool), "judging_macs": float(getattr(pool, "nb_macs", 0.0))}


def run(bed: str, seed: int, arm: str, out_dir: Path = OUT) -> dict:
    path = out_dir / f"{bed}_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = run_t(seed, arm) if bed == "t" else (run_g(seed, arm) if bed == "g" else run_f(int(bed[1:]), seed, arm))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    acc = f"tw {res['mean_tw']:.3f} ho {res['mean_ho']:.3f}" if bed == "t" else f"acc {res['acc']:.4f}"
    print(f"{bed} s{seed} {arm}: {acc} experts {res['experts']} params {res['params']} bytes {res['bytes'] / 1e6:.2f} MB nb {res['nb']} {res['seconds']:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--bed", default="f10", choices=["f10", "f30", "t", "g"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    ap.add_argument("--arms", nargs="+", default=["U8", "U32", "U128", "NB"])
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    for s in a.seeds:
        for arm in a.arms:
            run(a.bed, s, arm)

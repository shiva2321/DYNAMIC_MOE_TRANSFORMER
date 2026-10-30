"""Run one system on one stream under a strict test-then-train protocol and keep everything the analysis needs.

For every batch the pool first PREDICTS it, then OBSERVES it (the pool is handed inputs and labels only; an oracle baseline that declares ``uses_regime`` is also handed the regime id in ``ctx``).
After every phase the pool is read on the test set of every regime (the accuracy matrix, no gaps), on the out-of-distribution sets (novelty), and its structure and cost are snapshotted. At the end
every test sample's answer, routed expert and novelty flag are logged, together with the answer the regime's OWNER expert (the one that trained on most of the regime) would have given: the
oracle-routed answer among the same experts. A cell is one (system, stream, seed): written atomically with a code stamp, reused when the stamp matches.
"""

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from growmoe.accounting import pool_report
from growmoe.metrics import auroc
from growmoe.telemetry import _plain

from . import runlib

TRAIN_KEYS = ("phase", "regime", "first", "y", "pred", "expert", "unknown", "p_nov", "assigned")
TEST_KEYS = ("regime", "y", "pred", "expert", "unknown", "p_nov")


@dataclass
class RunRecord:
    system: str
    stream: str
    seed: int
    R: np.ndarray
    per_phase: List[dict]
    ood: List[dict]
    report: dict
    summary: dict
    events: List[dict]
    extra: dict
    seconds: float
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)


def owner_of_regime(assigned: np.ndarray, regime: np.ndarray) -> Dict[int, int]:
    """For each regime, the expert that trained on most of its samples (samples no expert trained on, assigned < 0, do not count)."""
    owner = {}
    for r in np.unique(regime):
        a = assigned[(regime == r) & (assigned >= 0)]
        if len(a):
            owner[int(r)] = int(Counter(a.tolist()).most_common(1)[0][0])
    return owner


def _oracle_predictions(pool, owner: Dict[int, int], x: torch.Tensor, regime: np.ndarray) -> np.ndarray:
    out = np.full(len(regime), -1, dtype=np.int64)
    experts = getattr(pool, "experts", None)
    if not isinstance(experts, dict):
        return out
    for r, e in owner.items():
        rec = experts.get(e)
        m = np.flatnonzero(regime == r)
        if rec is None or not len(m):
            continue
        rec.module.eval()
        with torch.no_grad():
            out[m] = rec.module.proba(x[torch.from_numpy(m)]).argmax(1).numpy()
    return out


def _cat(parts) -> np.ndarray:
    return np.concatenate([np.asarray(p) for p in parts]) if parts else np.zeros(0)


def run_system(make_pool: Callable[[], object], stream, seed: int, batch_size: int = 64, system: str = "") -> RunRecord:
    t0 = time.perf_counter()
    pool = make_pool()
    uses_regime = bool(getattr(pool, "uses_regime", False))
    rids = sorted(stream.regimes)
    tests = stream.test_sets()
    logs = {k: [] for k in TRAIN_KEYS}
    R, per_phase, ood_rows, seen = [], [], [], set()
    for k, phase in enumerate(stream.phases):
        for x, y, gt in stream.batches(k, batch_size, seed):
            ctx = gt["regime"] if uses_regime else None
            pred = pool.predict(x, ctx=ctx)                                                      # test first ...
            rec = pool.observe(x, y, ctx=ctx)                                                    # ... then train
            n = len(y)
            for key, val in (("phase", np.full(n, k)), ("regime", gt["regime"].numpy()), ("first", gt["first"].numpy()), ("y", y.numpy()), ("pred", pred.probs.argmax(1).numpy()),
                             ("expert", pred.expert.numpy()), ("unknown", pred.unknown.numpy()), ("p_nov", pred.p_nov.numpy()), ("assigned", rec.assigned.numpy())):
                logs[key].append(val)
        seen |= set(phase.regime.unique().tolist())
        row, kn_p, kn_u = [], [], []
        for rid in rids:
            xt, yt = tests[rid]
            p = pool.predict(xt, ctx=torch.full((len(xt),), rid, dtype=torch.long) if uses_regime else None)
            row.append(float((p.probs.argmax(1) == yt).float().mean()))
            if rid in seen:
                kn_p.append(p.p_nov)
                kn_u.append(p.unknown)
        R.append(row)
        kn_p, kn_u = torch.cat(kn_p), torch.cat(kn_u)
        o = {"phase": k, "known_unknown_rate": float(kn_u.float().mean())}
        for name, xo in stream.ood.items():
            if uses_regime:                                                                      # a regime-told baseline has no regime for a message it never saw
                o[f"{name}_unknown_rate"], o[f"{name}_auroc"] = float("nan"), float("nan")
                continue
            p = pool.predict(xo)
            o[f"{name}_unknown_rate"] = float(p.unknown.float().mean())
            try:
                o[f"{name}_auroc"] = auroc(torch.cat([1 - kn_p, 1 - p.p_nov]), torch.cat([torch.zeros(len(kn_p)), torch.ones(len(p.p_nov))]))
            except ValueError:
                o[f"{name}_auroc"] = float("nan")
        ood_rows.append(o)
        s = pool.summary()
        rep = pool_report(pool, stream.d)
        per_phase.append({"phase": k, "name": phase.name, "n_experts": int(s.get("n_experts", 0)), "n_params": int(rep.params), "stored_vectors": int(rep.stored_vectors),
                          "memory_bytes": int(rep.memory_bytes), "events": dict(Counter(e.kind for e in pool.events)), "seconds": time.perf_counter() - t0})
    train = {f"train_{k}": _cat(v) for k, v in logs.items()}
    owner = owner_of_regime(train["train_assigned"], train["train_regime"])
    tl = {k: [] for k in TEST_KEYS}
    xs = []
    for rid in rids:
        xt, yt = tests[rid]
        p = pool.predict(xt, ctx=torch.full((len(xt),), rid, dtype=torch.long) if uses_regime else None)
        xs.append(xt)
        for key, val in (("regime", np.full(len(yt), rid)), ("y", yt.numpy()), ("pred", p.probs.argmax(1).numpy()), ("expert", p.expert.numpy()), ("unknown", p.unknown.numpy()), ("p_nov", p.p_nov.numpy())):
            tl[key].append(val)
    test = {f"test_{k}": _cat(v) for k, v in tl.items()}
    test["test_oracle_pred"] = _oracle_predictions(pool, owner, torch.cat(xs), test["test_regime"])
    arrays = {**train, **test, "R": np.asarray(R)}
    summary = json.loads(json.dumps(_plain(pool.summary()), default=str))
    return RunRecord(system or getattr(pool, "name", "pool"), stream.name, seed, np.asarray(R), per_phase, ood_rows, asdict(pool_report(pool, stream.d)), summary,
                     [{"kind": e.kind, "step": e.step, **e.fields} for e in pool.events], {"owner_of_regime": {str(r): e for r, e in owner.items()}, "n_samples": int(len(train["train_y"]))},
                     time.perf_counter() - t0, arrays)


# ------------------------------------------------------------------------------------------------------------------------------ cells
def _paths(out_dir, stream_name: str, system: str, seed: int):
    base = Path(out_dir) / stream_name
    return base / f"{system}_s{seed}.json", base / f"{system}_s{seed}.npz"


def save_record(rec: RunRecord, out_dir, stamp: str) -> None:
    jp, zp = _paths(out_dir, rec.stream, rec.system, rec.seed)
    runlib.atomic_save_npz(zp, **rec.arrays)
    head = {k: v for k, v in asdict(rec).items() if k not in ("arrays", "R")}
    head["code_stamp"] = stamp
    runlib.atomic_write_text(jp, json.dumps(head, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def load_record(out_dir, stream_name: str, system: str, seed: int) -> RunRecord:
    jp, zp = _paths(out_dir, stream_name, system, seed)
    head = json.loads(jp.read_text(encoding="utf-8"))
    with np.load(zp) as z:
        arrays = {k: z[k] for k in z.files}
    head.pop("code_stamp", None)
    return RunRecord(R=arrays["R"], arrays=arrays, **head)


def run_cell(out_dir, system: str, make_pool: Callable[[], object], stream, seed: int, stamp: str, batch_size: int = 64) -> RunRecord:
    jp, _ = _paths(out_dir, stream.name, system, seed)
    if runlib.cell_done(jp, stamp):
        return load_record(out_dir, stream.name, system, seed)
    rec = run_system(make_pool, stream, seed, batch_size, system)
    save_record(rec, out_dir, stamp)
    return rec

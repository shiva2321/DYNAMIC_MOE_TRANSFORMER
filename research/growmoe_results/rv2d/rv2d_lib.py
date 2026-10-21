"""Shared pieces of the RV2-D runner (routing v2 stage 4). Nothing here changes product code. Built for a PC that loses power daily (D-30): every result is written atomically,
an unreadable file counts as missing, long cells checkpoint."""

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (HERE.parent / "rv2c", HERE.parent / "rv2a", HERE.parent / "m2a", HERE.parent / "m0", HERE.parent / "arith", ROOT):
    sys.path.insert(0, str(_p))

SMOKE = bool(os.environ.get("RV2D_SMOKE"))               # tiny streams and a separate result folder: a check that every family runs end to end, the numbers mean nothing
CELLS = HERE / ("cells_smoke" if SMOKE else "cells")
N_PER_OP, PER_CELL, SEQ_CELL = (300, 10, 5) if SMOKE else (8000, 200, 40)
S3_ARM = "A_set@12-0.5-100"                    # the stage-3 arm of the stage-4 spikes (batch-level gate, breaker len 12, rate 0.5, pause 100)
POOL_SEED_OFFSET = 7                           # function-regime streams only (amendment A4-2, E-035): the pool seed is the stream seed + 7. With the same seed the expert founded for regime r starts with
#                                                regime r's teacher units (MLPExpert(seed = s*1000 + r) on both sides); the dev1 cells of this family carry that head start, the unaliased dev cells and the confirmatory run do not
STREAMS = ["F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5"]      # the ten pre-registered stage-3 streams
ARMS = {"S3": {}, "S3+T": {"spawn_rule": "transfer"}, "S3+G": {"growth_rule": "gap"}, "S3+T+G": {"spawn_rule": "transfer", "growth_rule": "gap"}}
ROUTER_SETTINGS = [("restrict", 0.01), ("restrict", 0.03), ("free", 0.01), ("free", 0.03)]
GAP_GRID = (0.14, 0.17, 0.20, 0.23, 0.26)     # declared in spec section 6.7
TRANSFER_GRID = (0.15, 0.25, 0.35)            # declared in spec section 8


def pool_seed(cell) -> int:
    """The seed of a cell's pool: the stream seed + POOL_SEED_OFFSET for the function-regime family ("s3"), the stream seed for the others (arithmetic and banking have no teacher)."""
    return cell.seed + POOL_SEED_OFFSET if cell.family == "s3" else cell.seed


@dataclass(frozen=True)
class Cell:
    family: str                  # "arith-seq" | "arith-mixed16" | "arith-mixed128" | "s3" | "banking"
    stream: str = "-"            # stage-3 stream name for family "s3", else "-"
    arm: str = "S3"              # key of ARMS
    seed: int = 100
    gap: float = 0.0             # grow_gap of the growth arms
    tmargin: float = 0.0         # transfer_margin of the transfer arms
    phase: str = "dev"

    @property
    def cid(self) -> str:
        num = lambda v: str(v).replace(".", "p")                       # noqa: E731 - never a dot in a file name (Path.with_suffix collapses names with dots)
        return f"{self.family}__{self.stream.replace('.', 'p')}__{self.arm.replace('+', '_')}__s{self.seed}__g{num(self.gap)}__t{num(self.tmargin)}"


def arm_flags(cell: Cell) -> dict:
    kw = dict(ARMS[cell.arm])
    if "growth_rule" in kw:
        if cell.gap <= 0.0:
            raise ValueError(f"{cell.cid}: a growth arm needs grow_gap")
        kw["grow_gap"] = cell.gap
    if "spawn_rule" in kw:
        if cell.tmargin <= 0.0:
            raise ValueError(f"{cell.cid}: a transfer arm needs transfer_margin")
        kw["transfer_margin"] = cell.tmargin
    from growmoe.config import PRE_STAGE4
    return {**PRE_STAGE4, **kw}              # the arms are defined against the pool before stage 4 (the router and the transfer rule are the default only since the adoption of 2026-10-21, E-043): an arm lists what it turns ON


def code_stamp(root: Path = None, files=None) -> str:
    """Short sha256 of the source a cell result depends on (every .py file of the growmoe package and this module; CRLF and LF hash alike). A result is valid only for the code
    that produced it: ``Job.done`` accepts a cell only when its stamp is the current one, and the pre-registration records the stamp it was frozen at."""
    import hashlib
    root = ROOT if root is None else Path(root)
    if files is None:
        files = sorted(str(f.relative_to(root)).replace("\\", "/") for f in (root / "growmoe").rglob("*.py")) + ["research/growmoe_results/rv2d/rv2d_lib.py"]
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode("utf-8"))
        h.update((root / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(text.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def cell_done(path, stamp: str = None) -> bool:
    """A cell is done only when its result parses AND was produced by the current code (its ``code_stamp`` equals the given or the current stamp)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except Exception:
        return False
    return rec.get("code_stamp") == (stamp if stamp is not None else code_stamp())


def valid_json(path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ evaluation of one trained pool
def router_variants(pool, evaluate) -> dict:
    """The SAME trained pool answered without the router and under each declared router setting (the router acts at answer time only, so these are paired by construction).
    ``evaluate()`` returns the accuracy of the pool as it is now."""
    from growmoe.router import RouteRule
    saved = (pool.cfg.router, pool.cfg.router_mode, pool.cfg.router_margin)
    out = {}
    try:
        pool.cfg.router = False
        out["off"] = {"acc": float(evaluate())}
        for mode, margin in ROUTER_SETTINGS:
            pool.cfg.router, pool.cfg.router_mode, pool.cfg.router_margin = True, mode, margin
            pool._route, pool._route_age, pool._route_dirty = RouteRule(), 0, False               # a fresh state: one setting's selection must not inherit another's
            acc = float(evaluate())
            out[f"{mode}-{margin}"] = {"acc": acc, "selected": pool._route.name, "cv": pool._route.cv, "rows": pool._route.rows}
    finally:
        pool.cfg.router, pool.cfg.router_mode, pool.cfg.router_margin = saved
        pool._route, pool._route_age, pool._route_dirty = RouteRule(), 0, False
    return out


def _record(cell, pool, evaluate, d, inner_steps, seconds, extra=None) -> dict:
    from collections import Counter
    from growmoe.accounting import pool_report
    events = dict(Counter(e.kind for e in pool.events))                       # counted BEFORE the router variants below add their own refit events: these are the events of training
    grow_events = [{"step": e.step, **e.fields} for e in pool.events.of_kind("grow")]
    variants = router_variants(pool, evaluate)
    rep = pool_report(pool, d)
    rec = {"cid": cell.cid, "cell": asdict(cell), "acc": variants["off"]["acc"], "router": variants, "n_experts": rep.n_experts, "params": rep.params, "memory_bytes": rep.memory_bytes,
           "events": events, "grow_events": grow_events, "final_hidden": sorted(int(r.module.hidden) for r in pool.experts.values()), "grow_stats": dict(pool._grow_stats()),
           "inner_steps": inner_steps, "seconds": seconds}
    rec.update(extra or {})
    return rec


# ------------------------------------------------------------------------------------------------ stage-3 streams and real text
def run_s3(cell: Cell) -> dict:
    import dataclasses
    import time
    import run_rv2c as R3
    from growmoe.fairfight import _Scorer, run_fast
    t0 = time.perf_counter()
    st = R3.build_stream(cell.stream, cell.seed, SMOKE)
    steps = R3.steps_for(cell.stream)
    cfg = dataclasses.replace(R3.arm_cfg(S3_ARM, st, cell.seed, steps), seed=pool_seed(cell), **arm_flags(cell))
    holder = []

    def make():
        holder.append(R3.LoggingPool(cfg))
        return holder[0]

    run_fast(make, st, cell.seed, eval_every="final")
    pool = holder[0]
    return _record(cell, pool, lambda: _Scorer(pool, st).row().mean(), st.d, steps, time.perf_counter() - t0, {"regimes": len(st.regimes)})


def run_banking(cell: Cell) -> dict:
    import dataclasses
    import time
    import run_m0 as M0
    from growmoe.config import PhyloCfg
    from growmoe.fairfight import _Scorer, run_fast
    from growmoe.pools.phylo import PhyloMoE
    from growmoe.streams_real import banking_incremental
    t0 = time.perf_counter()
    st = banking_incremental(seed=cell.seed, frontend=M0.HybridFrontend(d=128, seed=0))
    cfg = PhyloCfg(d=st.d, n_classes=st.n_classes, seed=cell.seed, inner_steps=20, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32, set_gate=True,
                   set_gate_breaker_len=12, set_gate_breaker_rate=0.5, set_gate_breaker_pause=100, set_gate_confirm=2, group_veto_q=0.2)
    cfg = dataclasses.replace(cfg, **arm_flags(cell))
    holder = []

    def make():
        holder.append(PhyloMoE(cfg))
        return holder[0]

    run_fast(make, st, cell.seed, eval_every="final")
    pool = holder[0]
    return _record(cell, pool, lambda: _Scorer(pool, st).row().mean(), st.d, 20, time.perf_counter() - t0, {"regimes": len(st.regimes)})


# ------------------------------------------------------------------------------------------------ arithmetic (checkpointed: a power cut costs one block / one chunk)
def _ckpt_load(path):
    import pickle
    try:
        with open(path, "rb") as fh:                                   # written by this module (atomic) in this folder; never loaded from anywhere else
            return pickle.load(fh)
    except Exception:
        return None


def _ckpt_save(path, obj) -> None:
    import pickle
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(pickle.dumps(obj, protocol=4))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _exact_mean(learner, test, per_cell):
    import arith_lib as L
    ex, _ = L.cell_tables(test, learner.predict(test)["pred"], per_cell)
    return float(ex.mean())


def run_arith_seq(cell: Cell) -> dict:
    """Operator-by-operator arithmetic (the stage-3 pool, hidden 512, replay 20,000, 5 inner steps): exact match, forgetting, experts. One checkpoint per block."""
    import time
    import numpy as np
    import torch
    import arith_lib as L
    from growmoe import arith as A
    t0 = time.perf_counter()
    train, test, small = L.build_train("FULL", cell.seed, N_PER_OP), L.build_test(PER_CELL), L.build_test(SEQ_CELL)
    learner = L.Pool(5, True, name=cell.arm, hidden=512, replay_size=20000, **arm_flags(cell))
    ckpt = CELLS / f"{cell.cid}.ckpt"
    state = _ckpt_load(ckpt) if ckpt.exists() else None
    snaps, done = [], -1
    torch.set_num_threads(3)
    if state is not None:
        learner.pool, snaps, done = state["pool"], state["snaps"], state["done"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
    else:
        learner._new(cell.seed)
    for k, (oi, idx) in enumerate(L.op_blocks(train, cell.seed)):
        if k <= done:
            continue
        learner._feed(train, idx)
        out = learner.predict(small)
        snaps.append([float(v) for v in small.exact(out["pred"]).reshape(A.N_OPS, -1).mean(-1)])
        _ckpt_save(ckpt, {"pool": learner.pool, "snaps": snaps, "done": k, "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()})
    arr = np.array(snaps)
    forgetting = float(np.mean(arr.max(0) - arr[-1]))
    rec = _record(cell, learner.pool, lambda: _exact_mean(learner, test, PER_CELL), A.D_MODEL, 5, time.perf_counter() - t0, {"forgetting": forgetting, "snapshots": snaps})
    if ckpt.exists():
        ckpt.unlink()
    return rec


def run_arith_mixed(cell: Cell) -> dict:
    """Mixed-operator arithmetic in random order from a small (16) or ordinary (128) expert. One checkpoint per 150 batches."""
    import time
    import numpy as np
    import torch
    import arith_lib as L
    from growmoe import arith as A
    t0 = time.perf_counter()
    hidden = 16 if cell.family == "arith-mixed16" else 128
    train, test = L.build_train("FULL", cell.seed, N_PER_OP), L.build_test(PER_CELL)
    learner = L.Pool(5, False, name=cell.arm, hidden=hidden, replay_size=20000, **arm_flags(cell))
    order = np.random.default_rng([cell.seed, 0x5F1]).permutation(train.n_samples)
    chunks = [order[i:i + 9600] for i in range(0, len(order), 9600)]          # 150 batches of 64
    ckpt = CELLS / f"{cell.cid}.ckpt"
    state = _ckpt_load(ckpt) if ckpt.exists() else None
    done = -1
    torch.set_num_threads(3)
    if state is not None:
        learner.pool, done = state["pool"], state["done"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
    else:
        learner._new(cell.seed)
    for k, idx in enumerate(chunks):
        if k <= done:
            continue
        learner._feed(train, idx)
        _ckpt_save(ckpt, {"pool": learner.pool, "done": k, "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()})
    rec = _record(cell, learner.pool, lambda: _exact_mean(learner, test, PER_CELL), A.D_MODEL, 5, time.perf_counter() - t0)
    if ckpt.exists():
        ckpt.unlink()
    return rec


RUNNERS = {"arith-seq": run_arith_seq, "arith-mixed16": run_arith_mixed, "arith-mixed128": run_arith_mixed, "s3": run_s3, "banking": run_banking}


def run_cell(cell: Cell) -> dict:
    return RUNNERS[cell.family](cell)

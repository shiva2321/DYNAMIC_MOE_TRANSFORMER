"""M0 fair-fight tooling: a fast runner that reproduces the benchmark's accuracy exactly, and the training-budget search.

``run_fast`` replays a stream the way ``growmoe.bench.run_stream`` does (same batches in the same order, ``observe`` once per
batch, the same held-out sets) and drops only what a novelty table needs and an accuracy table does not:

* the per-batch ``predict`` that precedes every ``observe`` - no pool in this package keeps state from ``predict``, which
  tests/growmoe/test_gm_m0_fairfight.py checks by comparing the accuracy matrix AND the event log with ``run_stream``;
* one ``predict`` call per held-out regime - when no pool hook or context is involved, all held-out sets go through one call.

``eval_every="final"`` also skips the intermediate rows (only forgetting / backward transfer need them).

``tune_steps`` gives a baseline its fair training budget: it widens the grid until the best setting is no longer the largest one
tried, because a comparison against an under-trained baseline measures the budget, not the method.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from .accounting import BYTES_PER_FLOAT, PoolReport, pool_report
from .metrics import accuracy_stats, paired_diff
from .streams import Stream

_EVAL_CHUNK = 8192                                                                      # rows per batched predict call (bounds peak memory)
_EVAL_MODES = ("segments", "final")


def _finite(x: float):
    return float(x) if math.isfinite(x) else None


@dataclass
class FastResult:
    name: str
    R: np.ndarray                  # (rows, regimes): one row per segment, or only the last row with eval_every="final"
    acc: float                     # mean over regimes of the final accuracy
    stats: dict                    # accuracy_stats(R) when every segment was evaluated, otherwise {"avg": acc}
    per_regime: np.ndarray         # final accuracy of each regime
    report: PoolReport
    events: list
    summary: dict
    wall_clock: float

    @property
    def n_experts(self) -> int:
        return self.report.n_experts


class _Scorer:
    """Accuracy of ``pool`` on every held-out set of ``stream``, as a row indexed by regime id."""

    def __init__(self, pool, stream: Stream) -> None:
        self.pool = pool
        ctxs = stream.test_ctx()
        self.sets = [(r, x, y.cpu(), ctxs.get(r)) for r, (x, y) in stream.test_sets().items()]
        self.n_regimes = len(stream.regimes)
        self.batched = not hasattr(pool, "set_eval_regime") and all(c is None for _, _, _, c in self.sets)
        if self.batched:
            self.x = torch.cat([x for _, x, _, _ in self.sets])
            self.y = torch.cat([y for _, _, y, _ in self.sets])
            self.bounds = np.cumsum([0] + [len(y) for _, _, y, _ in self.sets])

    def _hits(self, x: torch.Tensor, y: torch.Tensor, ctx=None) -> torch.Tensor:
        return (self.pool.predict(x, ctx=ctx).probs.argmax(dim=-1).cpu() == y).float()

    def row(self) -> np.ndarray:
        row = np.zeros(self.n_regimes, dtype=np.float64)
        if self.batched:
            hit = torch.cat([self._hits(self.x[lo:lo + _EVAL_CHUNK], self.y[lo:lo + _EVAL_CHUNK])
                             for lo in range(0, len(self.y), _EVAL_CHUNK)])
            for j, (r, _, _, _) in enumerate(self.sets):
                row[r] = float(hit[self.bounds[j]:self.bounds[j + 1]].mean().item())
        else:
            for r, x, y, ctx in self.sets:
                if hasattr(self.pool, "set_eval_regime"):
                    self.pool.set_eval_regime(r)
                row[r] = float(self._hits(x, y, ctx).mean().item())
        return row


def run_fast(make_pool: Callable[[], object], stream: Stream, seed: int = 0, batch_size: int = 64,
             eval_every: str = "segments", d: int = None) -> FastResult:
    """Train a fresh pool on ``stream`` and report its held-out accuracy (see the module docstring for what is skipped)."""
    if eval_every not in _EVAL_MODES:
        raise ValueError(f"eval_every must be one of {_EVAL_MODES}, got {eval_every!r}")
    t0 = time.perf_counter()
    pool = make_pool()
    scorer = _Scorer(pool, stream)
    rows, learned_at, first_seen = [], [0] * len(stream.regimes), set()
    current = -1
    for k, seg, idx in stream._walk(batch_size, seed):
        if k != current:
            if current >= 0 and eval_every == "segments":
                rows.append(scorer.row())
            current = k
            if seg.first_visit and seg.regime not in first_seen:
                first_seen.add(seg.regime)
                learned_at[seg.regime] = len(rows)
        pool.observe(seg.x[idx], y=seg.y[idx], ctx=seg.ctx[idx] if seg.ctx is not None else None)
    rows.append(scorer.row())

    R = np.array(rows)
    final = R[-1]
    if eval_every == "segments":
        stats = accuracy_stats(R, learned_at=learned_at, chance=1.0 / stream.n_classes)
    else:
        stats = {"avg": float(final.mean())}
    return FastResult(name=getattr(pool, "name", "unknown_pool"), R=R, acc=float(final.mean()), stats=stats, per_regime=final.copy(),
                      report=pool_report(pool, stream.d if d is None else d), events=list(pool.events), summary=pool.summary(),
                      wall_clock=time.perf_counter() - t0)


def matched_capacity(report: PoolReport, d: int) -> int:
    """How many stored vectors give a lookup the same memory in bytes as the pool that produced ``report`` (never fewer than 16)."""
    return max(int(report.memory_bytes // (BYTES_PER_FLOAT * d)), 16)


def run_record(stream: str, method: str, seed: int, res: FastResult, **extra) -> dict:
    """One JSON-able row per (stream, method, seed): what the verdict script and the result tables read."""
    rep = res.report
    forgetting = res.stats.get("forgetting")
    return {"stream": stream, "method": method, "seed": int(seed), "acc": res.acc, "per_regime": res.per_regime.tolist(),
            "forgetting": None if forgetting is None else _finite(forgetting), "n_experts": rep.n_experts, "params": rep.params,
            "naive_params": rep.naive_params, "stored_vectors": rep.stored_vectors, "memory_bytes": rep.memory_bytes,
            "gate_macs": rep.gate_macs, "expert_macs": rep.expert_macs, "predict_macs": rep.predict_macs,
            "wall_clock": res.wall_clock, "event_counts": dict(Counter(e.kind for e in res.events)), **extra}


# ------------------------------------------------------------------------------------------------ budget search
@dataclass
class TuneResult:
    best: int
    at_edge: bool                  # True when the best setting is the largest one tried (the cap stopped the search)
    table: Dict[int, float]


def tune_steps(score: Callable[[int], float], grid: Sequence[int] = (1, 5, 20, 50), factor: float = 2.5, cap: int = 200,
               tol: float = 1e-12) -> TuneResult:
    """Choose the training budget (optimizer steps per batch) that maximises ``score(steps)``, measured on dev data.

    Ties go to the cheaper setting. If the best setting is the largest one tried, the grid is extended by ``factor`` (up to
    ``cap``) until an extension no longer helps, so the reported optimum is interior unless the cap itself was the limit.
    """
    table: Dict[int, float] = {}

    def evaluate(steps: int) -> None:
        if steps not in table:
            value = float(score(steps))
            if not math.isfinite(value):
                raise ValueError(f"score({steps}) is not finite: {value}")
            table[steps] = value

    def argbest() -> int:
        best, best_value = None, -math.inf
        for steps in sorted(table):
            if table[steps] > best_value + tol:
                best, best_value = steps, table[steps]
        return best

    for steps in sorted({int(g) for g in grid}):
        evaluate(steps)
    best = argbest()
    while best == max(table) and best < cap:
        evaluate(min(int(round(best * factor)), cap))
        better = argbest()
        if better == best:
            break
        best = better
    return TuneResult(best=best, at_edge=best == max(table), table=dict(table))


# ------------------------------------------------------------------------------------------------ pre-registered verdicts
EXPECTED = {"A1": "PASS", "A2": "PASS", "A3": "PASS", "B1": "PASS", "B2": "PASS", "B3": "PASS", "C1": "PASS", "C2": "PASS"}
_RULES = {
    "A1": "S2-5 and S1: mean(B2 tuned) - mean(A tuned) >= -1.0 point (PASS = Phase 1's 'A beats replay' is not supported)",
    "A2": "S2-5 and S1: mean(kNN-all) - mean(A tuned) >= +5.0 points",
    "A3": "S2-5: unique(C) / unique(A) in [0.6, 1.2] and reported(C) > 2 x unique(C)",
    "B1": "F-15 and F-30: mean paired A tuned - B2* >= +5.0 points and the 95% interval excludes 0",
    "B2": "F-15 and F-30: A tuned - kNN-all >= +10.0 points",
    "B3": "F-30: B4 - A >= +5.0 points (reported; no gate)",
    "C1": "S2-30: predict multiply-adds(A) >= 0.5 x multiply-adds(kNN-all) (PASS = no compute saving; reported, no gate)",
    "C2": "A tuned: S1-PRE - S1 >= +1.0 point (reported)",
}
_POINTS = 100.0


def _index(runs: Sequence[dict]) -> dict:
    index: dict = {}
    for r in runs:
        index.setdefault((r["stream"], r["method"]), {})[int(r["seed"])] = r
    return index


def _common(index: dict, key_a: tuple, key_b: tuple):
    a, b = index.get(key_a), index.get(key_b)
    if not a or not b:
        return None
    seeds = sorted(set(a) & set(b))
    return (a, b, seeds) if seeds else None


def _paired_points(index: dict, key_a: tuple, key_b: tuple, margin: float, need_interval: bool = False):
    """Paired difference key_a - key_b of the final accuracy, in points, with the verdict of ``difference >= margin``."""
    pair = _common(index, key_a, key_b)
    if pair is None:
        return None
    a, b, seeds = pair
    d = paired_diff([a[s]["acc"] * _POINTS for s in seeds], [b[s]["acc"] * _POINTS for s in seeds])
    holds = d["mean_diff"] >= margin and (d["ci_lo"] > 0.0 if need_interval else True)
    return {"n": d["n"], "mean_diff_points": d["mean_diff"], "ci_lo": _finite(d["ci_lo"]), "ci_hi": _finite(d["ci_hi"]),
            "p_t": _finite(d["p_t"]), "holds": bool(holds)}


def _combine(details) -> str:
    """One verdict from the per-stream details (a dict), or from a single detail (None when the data are missing)."""
    parts = list(details.values()) if isinstance(details, dict) and "holds" not in details else [details]
    if any(v is None for v in parts):
        return "NOT MEASURABLE"
    holds = [v["holds"] for v in parts]
    return "PASS" if all(holds) else ("FAIL" if not any(holds) else "PARTIAL")


def verdicts(runs: Sequence[dict]) -> dict:
    """The pre-registered statements A1-C2 evaluated on per-seed run records (``stream``, ``method``, ``seed``, ``acc``, and for
    the resource statements ``params``, ``naive_params``, ``predict_macs``). A statement over several streams is PASS when it holds
    on all, FAIL when it holds on none, PARTIAL otherwise, and NOT MEASURABLE when any stream or method is missing."""
    ix = _index(runs)
    out = {}

    def per_stream(streams, a, b, margin, need_interval=False):
        return {s: _paired_points(ix, (s, a), (s, b), margin, need_interval) for s in streams}

    detail = {
        "A1": per_stream(("S2-5", "S1"), "B2", "A", -1.0),
        "A2": per_stream(("S2-5", "S1"), "kNN_all", "A", 5.0),
        "B1": per_stream(("F-15", "F-30"), "A", "B2star", 5.0, need_interval=True),
        "B2": per_stream(("F-15", "F-30"), "A", "kNN_all", 10.0),
        "B3": _paired_points(ix, ("F-30", "B4"), ("F-30", "A"), 5.0),
        "C2": _paired_points(ix, ("S1-PRE", "A"), ("S1", "A"), 1.0),
    }
    verdict = {k: _combine(v) for k, v in detail.items()}

    pair = _common(ix, ("S2-5", "C"), ("S2-5", "A"))                                    # A3: Pool C's parameter count
    if pair is None:
        detail["A3"], verdict["A3"] = None, "NOT MEASURABLE"
    else:
        c, a, seeds = pair
        ratio = float(np.mean([c[s]["params"] / a[s]["params"] for s in seeds]))
        inflation = float(np.mean([c[s]["naive_params"] for s in seeds]) / np.mean([c[s]["params"] for s in seeds]))
        detail["A3"] = {"n": len(seeds), "ratio": ratio, "reported_over_unique": inflation}
        verdict["A3"] = "PASS" if 0.6 <= ratio <= 1.2 and inflation > 2.0 else "FAIL"

    pair = _common(ix, ("S2-30", "A"), ("S2-30", "kNN_all"))                            # C1: compute against a lookup
    if pair is None:
        detail["C1"], verdict["C1"] = None, "NOT MEASURABLE"
    else:
        a, k, seeds = pair
        ratio = float(np.mean([a[s]["predict_macs"] for s in seeds]) / np.mean([k[s]["predict_macs"] for s in seeds]))
        detail["C1"] = {"n": len(seeds), "ratio": ratio}
        verdict["C1"] = "PASS" if ratio >= 0.5 else "FAIL"

    return {k: {"rule": _RULES[k], "verdict": verdict[k], "detail": detail[k], "expected": EXPECTED[k],
                "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == EXPECTED[k]} for k in EXPECTED}


def kill_rule(result: dict) -> str:
    """What the pre-registration says to do, given B1."""
    b1 = result["B1"]["verdict"]
    if b1 == "NOT MEASURABLE":
        return "The kill rule cannot be applied: B1 was not measurable."
    if b1 == "FAIL":
        return ("B1 FAIL: the dynamic pool is not shown to beat a dense network on streams that need computed answers; the MoE track "
                "stops claiming an accuracy advantage and re-scopes to a compute saver and abstention layer over an episodic memory.")
    if b1 == "PARTIAL":
        return ("B1 PARTIAL: the track continues only to resolve it (more regimes, a memory-matched network) before any other work.")
    return "B1 PASS: the kill rule is not triggered; the track continues."


# ------------------------------------------------------------------------------------------------ M2-A verdicts
# research/growmoe_results/m2a/PREREG.md: does repairing the newborn false alarm make the pool beat the best single network on fresh
# seeds, without hurting the streams that need the alarm?
EXPECTED_M2A = {"P1": "PASS", "P2": "PASS", "G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS", "G5": "PASS"}
_RULES_M2A = {
    "P1": "F-15 and F-30: mean paired A_new - B2* >= +5.0 points and the 95% interval excludes 0 (the B1 rule)",
    "P2": "F-15 and F-30: mean paired A_new - A_old >= +3.0 points (the repair itself moves the pool)",
    "G1": "S1: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0",
    "G2": "S2-5: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0",
    "G3": "S2-30: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0",
    "G4": "S4: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0",
    "G5": "S7: A_new raises at least 3 alarms per run on average (the alarm still catches a label remap)",
}
_GUARD_STREAMS = {"G1": "S1", "G2": "S2-5", "G3": "S2-30", "G4": "S4"}


def m2a_verdicts(runs: Sequence[dict]) -> dict:
    """The M2-A statements on per-seed run records. Multi-stream statements follow the M0 rule (PASS on all, FAIL on none, PARTIAL
    otherwise, NOT MEASURABLE when data are missing)."""
    ix = _index(runs)
    detail = {
        "P1": {s: _paired_points(ix, (s, "A_new"), (s, "B2star"), 5.0, need_interval=True) for s in ("F-15", "F-30")},
        "P2": {s: _paired_points(ix, (s, "A_new"), (s, "A_old"), 3.0) for s in ("F-15", "F-30")},
    }
    for gid, stream in _GUARD_STREAMS.items():
        d = _paired_points(ix, (stream, "A_new"), (stream, "A_old"), -3.0)
        if d is not None:                                                               # a regression that is clearly below 0 fails even if small
            d["holds"] = bool(d["holds"] and not (d["ci_hi"] is not None and d["ci_hi"] < 0.0))
        detail[gid] = d
    verdict = {k: _combine(v) for k, v in detail.items()}

    s7 = ix.get(("S7", "A_new"))
    if not s7:
        detail["G5"], verdict["G5"] = None, "NOT MEASURABLE"
    else:
        mean_alarms = float(np.mean([r.get("event_counts", {}).get("alarm", 0) for r in s7.values()]))
        detail["G5"] = {"n": len(s7), "mean_alarms": mean_alarms}
        verdict["G5"] = "PASS" if mean_alarms >= 3.0 else "FAIL"

    return {k: {"rule": _RULES_M2A[k], "verdict": verdict[k], "detail": detail[k], "expected": EXPECTED_M2A[k],
                "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == EXPECTED_M2A[k]} for k in EXPECTED_M2A}


def m2a_decision(result: dict) -> str:
    """What the pre-registration says to do, given P1 and the guards."""
    p1 = result["P1"]["verdict"]
    if p1 == "NOT MEASURABLE":
        return "The M2-A decision cannot be applied: P1 was not measurable."
    if p1 == "FAIL":
        return ("P1 FAIL: the repaired pool is still not shown to beat a dense network by the pre-registered margin. Second kill: the MoE track "
                "stops claiming an accuracy advantage and re-scopes to a compute saver and abstention layer over an episodic memory.")
    if p1 == "PARTIAL":
        return "P1 PARTIAL: the track continues only to resolve it (more regimes, a memory-matched network) before any other work."
    bad = [g for g in ("G1", "G2", "G3", "G4", "G5") if result[g]["verdict"] != "PASS"]
    if bad:
        return (f"P1 PASS but guard(s) {', '.join(bad)} did not pass: keep it as an option (arm_monitor_after_infancy stays off by default) "
                "and fold the cause into the routing-v2 design.")
    return ("P1 PASS and every guard holds: adopt arm_monitor_after_infancy=True as the PhyloMoE default (a separate, reviewed change) and "
            "carry the lesson into the routing-v2 design.")


# ------------------------------------------------------------------------------------------------ RV2-A verdicts (stage 1: batch vote)
# research/growmoe_results/rv2a/PREREG.md: does letting a batch train one expert fix training contamination under overlap without hurting anything else?
BARS_RV2A = {"V1": 8.0, "V2": 1.0, "V3": 3.0}
EXPECTED_RV2A = {k: "PASS" for k in ("V1", "V2", "V3", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "GM")}
_GUARDS_RV2A = {"G1": "S1", "G2": "S2-5", "G3": "S2-30", "G4": "S4", "G5": "OV-0.0", "G6": "OV-0.3-MIXED"}
_MIX_STREAMS = ("MIX-0.0", "MIX-0.3")
_GUARD_IDS_RV2A = tuple(_GUARDS_RV2A) + ("G7", "GM")


def _rules_rv2a(bars: dict, cand: str, ref: str) -> dict:
    guard = f"{cand} - {ref} >= -3.0 points and the 95% interval is not entirely below 0"
    rules = {
        "V1": f"OV-0.3: mean paired {cand} - {ref} >= +{bars['V1']:.1f} points and the 95% interval excludes 0",
        "V2": f"F-15 and F-30: mean paired {cand} - {ref} >= +{bars['V2']:.1f} points and the 95% interval excludes 0",
        "V3": f"OV-0.6: mean paired {cand} - {ref} >= +{bars['V3']:.1f} points",
        "G7": f"S7: {cand} raises at least 3 alarms per run on average (the alarm still catches a label remap)",
        "GM": f"MIX-0.0 and MIX-0.3, accuracy on the minority regimes: {guard}",
    }
    rules.update({gid: f"{stream}: {guard}" for gid, stream in _GUARDS_RV2A.items()})
    return rules


def _guard_points(ix: dict, stream: str, cand: str, ref: str):
    d = _paired_points(ix, (stream, cand), (stream, ref), -3.0)
    if d is not None:                                                                   # a regression that is clearly below 0 fails even if small
        d["holds"] = bool(d["holds"] and not (d["ci_hi"] is not None and d["ci_hi"] < 0.0))
    return d


def rv2a_verdicts(runs: Sequence[dict], bars: dict = None, expected: dict = None, candidate: str = "A_vote", reference: str = "A_new") -> dict:
    """The RV2-A statements on per-seed run records. Multi-stream statements follow the M0 rule (PASS on all, FAIL on none, PARTIAL otherwise,
    NOT MEASURABLE when data are missing). ``bars`` / ``expected`` come from the pre-registration (bars_rv2a.json)."""
    bars = {**BARS_RV2A, **(bars or {})}
    exp = {**EXPECTED_RV2A, **(expected or {})}
    ix = _index(runs)
    detail = {
        "V1": _paired_points(ix, ("OV-0.3", candidate), ("OV-0.3", reference), bars["V1"], need_interval=True),
        "V2": {s: _paired_points(ix, (s, candidate), (s, reference), bars["V2"], need_interval=True) for s in ("F-15", "F-30")},
        "V3": _paired_points(ix, ("OV-0.6", candidate), ("OV-0.6", reference), bars["V3"]),
    }
    for gid, stream in _GUARDS_RV2A.items():
        detail[gid] = _guard_points(ix, stream, candidate, reference)
    detail["GM"] = {s: _guard_points(ix, s, candidate, reference) for s in _MIX_STREAMS}
    verdict = {k: _combine(v) for k, v in detail.items()}
    s7 = ix.get(("S7", candidate))
    if not s7:
        detail["G7"], verdict["G7"] = None, "NOT MEASURABLE"
    else:
        mean_alarms = float(np.mean([r.get("event_counts", {}).get("alarm", 0) for r in s7.values()]))
        detail["G7"] = {"n": len(s7), "mean_alarms": mean_alarms}
        verdict["G7"] = "PASS" if mean_alarms >= 3.0 else "FAIL"
    rules = _rules_rv2a(bars, candidate, reference)
    return {k: {"rule": rules[k], "verdict": verdict[k], "detail": detail[k], "expected": exp[k],
                "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == exp[k]} for k in EXPECTED_RV2A}


def _clears_rv2a(result: dict):
    return result["V1"]["verdict"], [g for g in _GUARD_IDS_RV2A if result[g]["verdict"] != "PASS"]


def rv2a_decision(plain: dict, restricted: dict = None) -> str:
    """What the pre-registration says to do, given the plain-vote verdicts and (if the fallback ran) the restricted-vote verdicts."""
    v1, bad = _clears_rv2a(plain)
    if v1 == "NOT MEASURABLE":
        return "The RV2-A decision cannot be applied: V1 was not measurable."
    if v1 != "PASS":
        return (f"V1 {v1}: the training-contamination mechanism is not supported at the pre-registered margin. Stop: keep batch_vote off, "
                "re-diagnose, and do not start stage 3 on this premise.")
    if not bad:
        return ("V1 PASS and every guard holds: adopt the plain vote (batch_vote=True) as the PhyloMoE default, test-first, with the historical "
                "entry points pinned.")
    if restricted is None:
        return (f"V1 PASS but guard(s) {', '.join(bad)} did not pass: run the restricted vote (a sample moves only if the modal expert's own gate "
                "accepts it) as the pre-declared fallback on the same seeds.")
    r1, rbad = _clears_rv2a(restricted)
    if r1 == "PASS" and not rbad:
        return "The plain vote failed a guard and the restricted vote passes V1 and every guard: adopt the restricted vote, test-first, with the historical entry points pinned."
    return ("Neither the plain nor the restricted vote passes V1 and every guard: adopt neither; batch_vote stays off and the cause goes into "
            "the stage-3 design.")

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


# ------------------------------------------------------------------------------------------------ RV2-B verdicts (stage 2: fingerprint shortlist)
# research/growmoe_results/rv2b/PREREG.md: does the shortlist cut the gate cost without costing accuracy?
BARS_RV2B = {"C1_mean": -1.0, "C1_lo": -2.0, "C2": 0.5, "C3": {"F-30": 0.20, "S2-30": 0.25, "F-15": 0.25}, "C4_macs": 0.05, "C4_time": 0.5}
EXPECTED_RV2B = {k: "PASS" for k in ("C1", "C2", "C3", "C4", "G")}
_C1_STREAMS = ("F-15", "F-30", "S2-30", "OV-0.3")
_G_STREAMS = ("S1", "S2-5")


def _rules_rv2b(bars: dict, cand: str, ref: str) -> dict:
    return {
        "C1": f"F-15, F-30, S2-30, OV-0.3: mean paired {cand} - {ref} >= {bars['C1_mean']:+.1f} points and the interval lower bound >= {bars['C1_lo']:+.1f}",
        "C2": f"the mean absolute difference in spawns per run between {cand} and {ref} is <= {bars['C2']} on those streams",
        "C3": "measured gate multiply-adds per input, " + ", ".join(f"{cand}/{ref} <= {v:.2f} on {s}" for s, v in bars["C3"].items()),
        "C4": f"synthetic pool of 500 experts: gate MACs ratio <= {bars['C4_macs']} and gate wall-clock ratio <= {bars['C4_time']} at B = 64 and B = 1024",
        "G": "S1 and S2-5 (too few experts for the cascade): accuracy and expert count bit-identical to the exact gate",
    }


def rv2b_verdicts(runs: Sequence[dict], scaling: dict = None, bars: dict = None, expected: dict = None,
                  candidate: str = "A_cascade", reference: str = "A_exact") -> dict:
    bars = {**BARS_RV2B, **(bars or {})}
    exp = {**EXPECTED_RV2B, **(expected or {})}
    ix = _index(runs)
    c1, c2, c3, g = {}, {}, {}, {}
    for s in _C1_STREAMS:
        d = _paired_points(ix, (s, candidate), (s, reference), bars["C1_mean"])
        if d is not None:
            d["holds"] = bool(d["holds"] and d["ci_lo"] is not None and d["ci_lo"] >= bars["C1_lo"])
        c1[s] = d
        pair = _common(ix, (s, candidate), (s, reference))
        if pair is None:
            c2[s] = None
        else:
            a, b, seeds = pair
            gap = float(np.mean([abs(a[i]["event_counts"].get("spawn", 0) - b[i]["event_counts"].get("spawn", 0)) for i in seeds]))
            c2[s] = {"n": len(seeds), "mean_abs_spawn_gap": gap, "holds": gap <= bars["C2"]}
    for s, bar in bars["C3"].items():
        pair = _common(ix, (s, candidate), (s, reference))
        if pair is None:
            c3[s] = None
        else:
            a, b, seeds = pair
            ratio = float(np.mean([a[i]["gate_macs_pred"] for i in seeds]) / np.mean([b[i]["gate_macs_pred"] for i in seeds]))
            c3[s] = {"n": len(seeds), "ratio": ratio, "bar": bar, "holds": ratio <= bar}
    for s in _G_STREAMS:
        pair = _common(ix, (s, candidate), (s, reference))
        if pair is None:
            g[s] = None
        else:
            a, b, seeds = pair
            same = all(a[i]["acc"] == b[i]["acc"] and a[i]["n_experts"] == b[i]["n_experts"] for i in seeds)
            g[s] = {"n": len(seeds), "identical": same, "holds": bool(same)}
    detail = {"C1": c1, "C2": c2, "C3": c3, "G": g}
    row = (scaling or {}).get("rows", {}).get("500")
    if not row or not all(b in row for b in ("64", "1024")):
        detail["C4"] = None
    else:
        ok = all(row[b]["ratio_macs"] <= bars["C4_macs"] and row[b]["ratio_time"] <= bars["C4_time"] for b in ("64", "1024"))
        detail["C4"] = {"holds": bool(ok), **{f"B{b}": {"ratio_macs": row[b]["ratio_macs"], "ratio_time": row[b]["ratio_time"]} for b in ("64", "1024")}}
    verdict = {k: _combine(v) for k, v in detail.items()}
    rules = _rules_rv2b(bars, candidate, reference)
    return {k: {"rule": rules[k], "verdict": verdict[k], "detail": detail[k], "expected": exp[k],
                "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == exp[k]} for k in EXPECTED_RV2B}


def rv2b_decision(result: dict, k: int = None, min_experts: int = None) -> str:
    """What the pre-registration says to do. C1 is the correctness statement; C3 and C4 decide default vs opt-in; G failing is a bug."""
    if result["C1"]["verdict"] == "NOT MEASURABLE":
        return "The RV2-B decision cannot be applied: C1 was not measurable."
    if result["G"]["verdict"] == "FAIL":
        return "G FAIL: a pool too small for the cascade does not match the exact gate bit for bit. Stop: this is a bug, not a result."
    if result["C1"]["verdict"] != "PASS":
        return f"C1 {result['C1']['verdict']}: the cascade is not shown to be non-inferior; it is not adopted and gate_shortlist stays 0."
    if result["C3"]["verdict"] == "PASS" and result["C4"]["verdict"] == "PASS":
        return (f"C1, C3 and C4 PASS: adopt the cascade as the PhyloMoE default (gate_shortlist=2, gate_fingerprint_k={k}, "
                f"gate_shortlist_min_experts={min_experts}), test-first, with the historical entry points pinned.")
    return ("C1 PASS but the cost statements C3/C4 did not both pass: keep the cascade opt-in (gate_shortlist stays 0 by default) and say so "
            "wherever it is documented.")


# ------------------------------------------------------------------------------------------------ RV2-C verdicts (stage 3: batch-level gate and group answering)
# research/growmoe_results/rv2c/PREREG.md; spec docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md section 7.
# A run record is one row of `run_record` plus: `regimes` (number of regimes of the stream), `fn_known` / `fn_called_novel` (A_set only: known single-source
# batches and how many of them the set gate acted on at once; optional `fn_doubtful`: how many it doubted at all, acted on or held under persistence (S3-5 reads it
# when present), and `groups` = {kind: {"single": accuracy, "group": accuracy}} for the kinds
# pure64, pure16, mixed32_32 (50/50) and mixed58_6 (90/10), both measured on the same pool and the same inputs.
BARS_RV2C = {"S3_1": {"OV-0.6": 8.0, "OV-0.85": 2.5}, "S3_2_mean": -1.0, "S3_2_lo": -2.0, "S3_3": -1.0, "S3_4": 0.5, "S3_5": 0.02,
             "S3_6": {"OV-0.85": 6.0, "OV-0.6": 0.0, "other": -0.5}, "S3_7": {"mixed32_32": -5.0, "mixed58_6": -1.0}, "S3_8": 5.0}
EXPECTED_RV2C = {k: "PASS" for k in ("S3-1", "S3-2", "S3-3", "S3-4", "S3-5", "S3-6", "S3-7", "S3-8", "S3-9")}
_GUARDS_RV2C = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3")
_SINGLE_SOURCE_RV2C = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.6", "OV-0.85")      # streams whose batches come from one regime
_ALL_RV2C = _SINGLE_SOURCE_RV2C + ("OV-0.3-MIXED",)
_GROUP_OTHERS_RV2C = ("F-15", "F-30", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.3-MIXED")


def _paired_field(index: dict, stream: str, method: str, path_a: tuple, path_b: tuple, margin: float):
    """Paired difference (field a - field b) of two fractions stored in the SAME run record (one pool, one set of inputs), in points."""
    runs = index.get((stream, method))
    if not runs:
        return None

    def get(r, path):
        for key in path:
            r = r[key]
        return r

    seeds = sorted(runs)
    try:
        a = [get(runs[s], path_a) * _POINTS for s in seeds]
        b = [get(runs[s], path_b) * _POINTS for s in seeds]
    except (KeyError, TypeError):                                                       # a record without the group measurements (or with None for a kind that could not be built): not measurable, not a crash
        return None
    d = paired_diff(a, b)
    return {"n": d["n"], "mean_diff_points": d["mean_diff"], "ci_lo": _finite(d["ci_lo"]), "ci_hi": _finite(d["ci_hi"]), "p_t": _finite(d["p_t"]),
            "holds": bool(d["mean_diff"] >= margin)}


def _rules_rv2c(bars: dict, cand: str, ref: str) -> dict:
    b1, b6, b7 = bars["S3_1"], bars["S3_6"], bars["S3_7"]
    return {
        "S3-1": f"OV-0.6 and OV-0.85: mean paired {cand} - {ref} >= +{b1['OV-0.6']:.1f} and >= +{b1['OV-0.85']:.1f} points, interval lower bounds above 0",
        "S3-2": f"F-15, F-30, S1, S2-5, S2-30, OV-0.0, OV-0.3: mean paired {cand} - {ref} >= {bars['S3_2_mean']:+.1f} points and the interval lower bound >= {bars['S3_2_lo']:+.1f}",
        "S3-3": f"OV-0.3-MIXED: mean paired {cand} - {ref} >= {bars['S3_3']:+.1f} points",
        "S3-4": f"mean experts of {cand} <= regimes + {bars['S3_4']} on every one of the ten regular streams",
        "S3-5": f"known single-source batches the set test called new (acted on at once or held for confirmation), pooled over the nine single-source streams: <= {100 * bars['S3_5']:.0f} %",
        "S3-6": (f"{cand} pools, pure groups of 64, group answer (vote + veto) minus per-input answer: OV-0.85 >= +{b6['OV-0.85']:.1f}, OV-0.6 >= {b6['OV-0.6']:+.1f}, "
                 f"every other stream >= {b6['other']:+.1f} points"),
        "S3-7": f"{cand} pools, mixed groups with the veto minus per input, every stream: 50/50 >= {b7['mixed32_32']:+.1f} and 90/10 >= {b7['mixed58_6']:+.1f} points",
        "S3-8": f"{cand} pools, OV-0.85, groups of 16 with the veto minus per input >= +{bars['S3_8']:.1f} points",
        "S3-9": "both flags off: the pool is bit-identical to the one recorded before stage 3 (golden fingerprints of growmoe/../rv2c/golden.py)",
    }


def rv2c_verdicts(runs: Sequence[dict], golden: dict = None, bars: dict = None, expected: dict = None, candidate: str = "A_set", reference: str = "A_vote") -> dict:
    """The RV2-C statements S3-1 to S3-9 on per-seed run records. Multi-stream statements follow the M0 rule (PASS on all, FAIL on none, PARTIAL otherwise,
    NOT MEASURABLE when data are missing). ``golden`` = {"differences": [...]} from the flag-off check (None = not measured)."""
    bars = {**BARS_RV2C, **(bars or {})}
    exp = {**EXPECTED_RV2C, **(expected or {})}
    ix = _index(runs)
    detail = {"S3-1": {s: _paired_points(ix, (s, candidate), (s, reference), m, need_interval=True) for s, m in bars["S3_1"].items()}}
    s32 = {}
    for s in _GUARDS_RV2C:
        d = _paired_points(ix, (s, candidate), (s, reference), bars["S3_2_mean"])
        if d is not None:
            d["holds"] = bool(d["holds"] and d["ci_lo"] is not None and d["ci_lo"] >= bars["S3_2_lo"])
        s32[s] = d
    detail["S3-2"] = s32
    detail["S3-3"] = _paired_points(ix, ("OV-0.3-MIXED", candidate), ("OV-0.3-MIXED", reference), bars["S3_3"])
    s34 = {}
    for s in _ALL_RV2C:
        rows = ix.get((s, candidate))
        if not rows or any("regimes" not in r for r in rows.values()):
            s34[s] = None
            continue
        mean_experts, regimes = float(np.mean([r["n_experts"] for r in rows.values()])), float(np.mean([r["regimes"] for r in rows.values()]))
        s34[s] = {"n": len(rows), "mean_experts": mean_experts, "regimes": regimes, "holds": bool(mean_experts <= regimes + bars["S3_4"])}
    detail["S3-4"] = s34
    known = called = 0
    present = [s for s in _SINGLE_SOURCE_RV2C if ix.get((s, candidate))]
    for s in present:
        for r in ix[(s, candidate)].values():
            known += int(r.get("fn_known", 0))
            called += int(r.get("fn_doubtful", r.get("fn_called_novel", 0)))              # A-15: doubted = acted on at once or held; the first build's records only have the former
    detail["S3-5"] = (None if len(present) < len(_SINGLE_SOURCE_RV2C) or known == 0 else
                      {"streams": len(present), "known_batches": known, "called_novel": called, "rate": called / known, "holds": bool(called / known <= bars["S3_5"])})
    s36 = {}
    for s in ("OV-0.85", "OV-0.6") + _GROUP_OTHERS_RV2C:
        margin = bars["S3_6"].get(s, bars["S3_6"]["other"])
        s36[s] = _paired_field(ix, s, candidate, ("groups", "pure64", "group"), ("groups", "pure64", "single"), margin)
    detail["S3-6"] = s36
    detail["S3-7"] = {f"{s}/{kind}": _paired_field(ix, s, candidate, ("groups", kind, "group"), ("groups", kind, "single"), bars["S3_7"][kind])
                      for s in _ALL_RV2C for kind in ("mixed32_32", "mixed58_6")}
    detail["S3-8"] = _paired_field(ix, "OV-0.85", candidate, ("groups", "pure16", "group"), ("groups", "pure16", "single"), bars["S3_8"])
    detail["S3-9"] = None if golden is None else {"differences": list(golden["differences"]), "holds": not golden["differences"]}
    verdict = {k: _combine(v) for k, v in detail.items()}
    rules = _rules_rv2c(bars, candidate, reference)
    return {k: {"rule": rules[k], "verdict": verdict[k], "detail": detail[k], "expected": exp[k],
                "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == exp[k]} for k in EXPECTED_RV2C}


def rv2c_decision(result: dict) -> str:
    """What the pre-registration says to do, for 3a (flag set_gate) and 3b (flag group_vote). The kill rules are S3-1 at OV-0.6, S3-2, S3-4 and S3-5."""
    ids = list(EXPECTED_RV2C)
    if any(result[k]["verdict"] == "NOT MEASURABLE" for k in ids):
        return "The RV2-C decision cannot be applied: " + ", ".join(k for k in ids if result[k]["verdict"] == "NOT MEASURABLE") + " not measurable."
    if result["S3-9"]["verdict"] != "PASS":
        return "S3-9 FAIL: with both flags off the pool is not bit-identical to the recorded one. Stop: this is a bug, not a result."
    killed = [k for k in ("S3-2", "S3-4", "S3-5") if result[k]["verdict"] != "PASS"]
    ov6 = result["S3-1"]["detail"]["OV-0.6"]["holds"]
    if killed:
        a = f"3a is NOT adopted ({', '.join(killed)} did not pass: kill rule); set_gate stays False"
    elif result["S3-1"]["verdict"] == "PASS" and result["S3-3"]["verdict"] == "PASS":
        a = "adopt 3a: set_gate=True becomes the PhyloMoE default, test-first, with the historical entry points pinned (set_gate=False)"
    elif ov6 and result["S3-3"]["verdict"] == "PASS":
        a = "3a passes at OV-0.6 only (S3-1 PARTIAL): it stays opt-in (set_gate=False) and every document says so"
    else:
        a = f"3a is NOT adopted (S3-1 {result['S3-1']['verdict']}, S3-3 {result['S3-3']['verdict']}); set_gate stays False"
    if all(result[k]["verdict"] == "PASS" for k in ("S3-6", "S3-7", "S3-8")):
        b = ("adopt 3b: group_vote=True becomes the default (it changes nothing unless predict is given groups); it was measured on the set-gate pools, "
             "the vote pools are reported in the results")
    else:
        b = "3b is NOT adopted (" + ", ".join(f"{k} {result[k]['verdict']}" for k in ("S3-6", "S3-7", "S3-8") if result[k]["verdict"] != "PASS") + "); group_vote stays False"
    return f"3a: {a}. 3b: {b}."

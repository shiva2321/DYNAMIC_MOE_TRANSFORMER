"""Lab G2: the same trained experts behind different gates (spec section 4; PROGRESS.md 2026-10-30).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_pool.py [first_seed last_seed]

Arithmetic in everyday language, seeds 150-152. The experts are the lab's own arithmetic solvers (one per taught skill, trained on its teaching examples; the numbers are read from the message by
a fixed parser, a declared simplification). Four gates decide who answers and whether anyone does: the pool's own vMF density gate, the general layer's asking gate, a mix of the two, and an
oracle that knows who owns a message (the ceiling of any gate). Messages: known skills in NEW wordings (a right answer exists), and messages nobody owns (never-taught skills, everyday
sentences, look-alike traps), where any answer is wrong. Because the experts are identical, any difference is the gate.
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_KS / "routing_lab"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import arith_domain as AD  # noqa: E402
import function_check as FC  # noqa: E402
import gates as GT  # noqa: E402
import lab_function as LF  # noqa: E402
import lab_general as LG  # noqa: E402
import pool as PL  # noqa: E402
import reader as RD  # noqa: E402
import runlib  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g2"
STAMP_FILES = [_HERE / n for n in ("gates.py", "pool.py", "lab_pool.py", "arith_domain.py", "measures.py", "route.py", "cards.py", "reader.py", "lab_general.py")] + [
    _KS / "baselines.py", _KS / "lab_function.py", _KS / "function_check.py", _KS / "routing_lab" / "ask_reader.py"]
GATES = ("oracle", "ask", "vmf", "mix")
UNOWNED = ("unknown", "teaching_wording", "foreign", "trap")
NET_MARGIN = 0.15                                                                       # rule C7
SEEDS = list(range(150, 153))


def parse_pair(text: str) -> Optional[tuple]:
    nums = FC.parse_numbers(text)
    return (nums[0], nums[1]) if len(nums) == 2 else None


def solver_expert(solver) -> Callable[[tuple], int]:
    return lambda f: int(solver.predict(np.array([f[0]], dtype=np.int64), np.array([f[1]], dtype=np.int64))[0])


class OracleGate:
    """Knows who owns each message (None = nobody): the best any gate can do."""

    def __init__(self, owner: Dict[str, Optional[int]]) -> None:
        self.owner = owner

    def decide(self, texts):
        o = [self.owner[t] for t in texts]
        return np.array([0 if x is None else x for x in o], dtype=int), np.array([x is None for x in o], dtype=bool)


def score_known(answers: Sequence[PL.Answer], owner: Sequence[int], value: Sequence[int], solid: Sequence[bool]) -> Dict[str, float]:
    """Rates over the known messages (each has a right owner and a right answer): answered, correct, wrong, abstained, unreadable, routed right among those routed, and the same split by the
    state of the owner."""
    ans = np.array([a.reason == "answered" for a in answers])
    cor = np.array([a.reason == "answered" and a.value == v for a, v in zip(answers, value)])
    routed = np.array([a.expert is not None for a in answers])
    right = np.array([a.expert == o for a, o in zip(answers, owner)])
    sol = np.asarray(solid, dtype=bool)
    out = {"answered": float(ans.mean()), "correct": float(cor.mean()), "wrong": float((ans & ~cor).mean()), "abstained": float(np.mean([a.reason == "abstained" for a in answers])),
           "unreadable": float(np.mean([a.reason == "unreadable" for a in answers])), "routed_right": float(right[routed].mean()) if routed.any() else float("nan"),
           "precision": float(cor[ans].mean()) if ans.any() else float("nan")}
    for tag, m in (("solid", sol), ("thin", ~sol)):
        out[f"correct_{tag}"] = float(cor[m].mean()) if m.any() else float("nan")
        out[f"answered_{tag}"] = float(ans[m].mean()) if m.any() else float("nan")
    return out


def answered_share(answers: Sequence[PL.Answer]) -> float:
    return float(np.mean([a.reason == "answered" for a in answers])) if answers else float("nan")


def net(known: Dict[str, float], unowned: Dict[str, float]) -> float:
    """Right answers minus wrong answers on the known messages, minus the share of messages nobody owns that got an answer (mean over the nonempty unowned sets): one point each."""
    rates = [v for v in unowned.values() if not np.isnan(v)]
    return known["correct"] - known["wrong"] - (float(np.mean(rates)) if rates else 0.0)


def run_cell(seed: int, ask=None, cache_dir=LG.ASK_DIR, encode: Callable = LG.default_encode, solvers_fn: Callable = LF.train_solvers, q: float = 0.05, tau: float = 0.5) -> dict:
    dw, foreign, extras, w = AD.build_full(seed)
    ex = LF.test_examples(w)
    solvers, acc, competent = solvers_fn(w)
    taught = list(w.taught)
    pool_args = (dw.taught, [solver_expert(solvers[s]) for s in taught], parse_pair)
    known_t = [t for t, i in dw.known_test if t in ex]
    owner = [i for t, i in dw.known_test if t in ex]
    value = [ex[t].answer for t in known_t]
    solid = [dw.states[i] == "solid" for i in owner]
    sets = {"unknown": list(dw.unknown_test), "teaching_wording": list(extras["unknown_teaching_wording"]), "foreign": list(foreign), "trap": list(extras["trap"])}
    X, y = [], []
    for i, n in enumerate(dw.taught):
        X += dw.teach[n]
        y += [i] * len(dw.teach[n])
    ask_gate = GT.AskGate(RD.CardReader(dw.cards, with_exemplars=True, cache_dir=cache_dir, ask=ask), dw.states, tau)
    vmf_gate = GT.VmfGate(encode, q).fit(encode(X), np.asarray(y), encode([t for t, _ in dw.known_cal]))
    owners = {t: i for t, i in zip(known_t, owner)}
    owners.update({t: None for ts in sets.values() for t in ts})
    gates = {"oracle": OracleGate(owners), "ask": ask_gate, "vmf": vmf_gate, "mix": GT.MixGate(ask_gate, vmf_gate)}
    rec: dict = {"n": {"known": len(known_t), **{k: len(v) for k, v in sets.items()}}, "taught": dw.taught, "states": dw.states, "expert_accuracy": {s: acc[s] for s in taught}, "gates": {}}
    for name, gate in gates.items():
        pool = PL.GatedPool(*pool_args, gate)
        known = score_known(pool.predict(known_t), owner, value, solid)
        unowned = {k: answered_share(pool.predict(v)) for k, v in sets.items()}
        rec["gates"][name] = {"known": known, "unowned": unowned, "net": net(known, unowned)}
    return rec


def _mean(rows: List[dict], pick: Callable[[dict], float]) -> float:
    return float(np.nanmean([pick(r) for r in rows]))


def verdict(rows: List[dict]) -> dict:
    """C7 (declared before the run): the asking gate's net score is at least NET_MARGIN above the vMF gate's."""
    a, v = _mean(rows, lambda r: r["gates"]["ask"]["net"]), _mean(rows, lambda r: r["gates"]["vmf"]["net"])
    return {"ask_net": a, "vmf_net": v, "difference": a - v, "C7_pass": bool(a - v >= NET_MARGIN)}


def report(rows: List[dict], seeds: Sequence[int]) -> str:
    n = rows[0]["n"]
    L = [f"# Lab G2, the same experts behind four gates (arithmetic in everyday language, seeds {seeds[0]}-{seeds[-1]})", "",
         f"Per seed about {n['known']} known messages in new wordings (a right answer exists) and messages nobody owns ({n['unknown']} never-taught, {n['teaching_wording']} never-taught in teaching wordings, "
         f"{n['foreign']} everyday, {n['trap']} look-alike traps), where any answer is wrong. Experts: one solver per taught skill, numbers read by a fixed parser (declared simplification). "
         "Net = correct - wrong on the known messages - the mean share of unowned messages that got an answer.", "",
         "| gate | answered | correct | wrong | precision (correct / answered) | routed right | correct, Solid owner | correct, Thin owner | NET |", "|---|---|---|---|---|---|---|---|---|"]
    for g in GATES:
        m = lambda k: _mean(rows, lambda r: r["gates"][g]["known"][k])
        L.append(f"| {g} | {m('answered'):.3f} | {m('correct'):.3f} | {m('wrong'):.3f} | {m('precision'):.3f} | {m('routed_right'):.3f} | {m('correct_solid'):.3f} | {m('correct_thin'):.3f} | "
                 f"{_mean(rows, lambda r: r['gates'][g]['net']):+.3f} |")
    L += ["", "Share of messages nobody owns that got an answer (lower is better):", "", "| gate | " + " | ".join(UNOWNED) + " |", "|---|" + "---|" * len(UNOWNED)]
    for g in GATES:
        L.append(f"| {g} | " + " | ".join(f"{_mean(rows, lambda r: r['gates'][g]['unowned'][k]):.3f}" for k in UNOWNED) + " |")
    acc = {s: [] for s in ("solid", "thin")}
    for r in rows:
        for st, a in zip(r["states"], r["expert_accuracy"].values()):
            acc[st].append(a)
    v = verdict(rows)
    L += ["", f"Expert accuracy on problems it never saw: Solid experts {np.mean(acc['solid']):.3f}, Thin experts {np.mean(acc['thin']):.3f}.", "",
          f"**C7** (asking gate NET at least {NET_MARGIN} above the vMF gate's): ask {v['ask_net']:+.3f}, vmf {v['vmf_net']:+.3f}, difference {v['difference']:+.3f}. **{'PASS' if v['C7_pass'] else 'FAIL'}**", ""]
    return "\n".join(L)


def run_seed(seed: int, out_dir, stamp: str) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    rec = run_cell(seed)
    rec.update({"seed": seed, "code_stamp": stamp})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(LG._native(rec)))
    return rec


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows = []
    for seed in range(lo, hi + 1):
        r = run_seed(seed, OUT, stamp)
        rows.append(r)
        print(f"seed {seed}: " + "; ".join(f"{g} net {r['gates'][g]['net']:+.3f} (correct {r['gates'][g]['known']['correct']:.2f}, wrong {r['gates'][g]['known']['wrong']:.2f})" for g in GATES), flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(_KS / "results" / f"ROUTING_GENERAL_G2_{lo}_{hi}.md", text)
    print(text)

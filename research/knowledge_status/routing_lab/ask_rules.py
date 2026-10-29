"""Rules for turning the asked probabilities into known / unseen, compared offline on the cached answers (no model is run), chosen on the TUNING seeds by the recorded costs of a wrong call
(a missed new skill about 7 pool points, a false alarm about 4; E-0xx cost table), and only then applied to report seeds.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/ask_rules.py [first_seed last_seed]

Rules: ``plain`` (unseen iff the top option is a skill nobody taught), ``tau<x>`` (also unseen when the top option has probability below x), ``conf<q>`` (class-conditional conformal, calibration on
S1-S3 held-out parts only, level q), ``hyb<x>`` (a part the calibrated "none" class covers is Other content, otherwise the ``tau<x>`` rule). Real problems are shown but never used in the choice.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent
KS = HERE.parent
ROOT = KS.parents[1]
for _p in (str(ROOT), str(KS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ask_reader as AR  # noqa: E402
import lab_ask as LA  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "ask"
COST_FALSE_ALARM, COST_MISS = 4.0, 7.0
RULES = ("plain", "tau0.3", "tau0.5", "tau0.7", "tau0.9", "conf0.05", "conf0.02", "conf0.01", "hyb0.0", "hyb0.5", "hyb0.7")
HELD = ("S4", "S5", "S6", "S7", "S8")


def unseen_flags(rule: str, P: np.ndarray, taught: Sequence[str], machine: "AR.AskMachine") -> np.ndarray:
    """Boolean per row: called unseen under the rule."""
    P = np.asarray(P, dtype=np.float64)
    known = set(taught) | {"none"}
    top = P.argmax(axis=1)
    named_unknown = np.array([AR.OPS[int(t)] not in known for t in top])
    if rule == "plain":
        return named_unknown
    if rule.startswith("tau"):
        return named_unknown | (P.max(axis=1) < float(rule[3:]))
    if rule.startswith("hyb"):                                                                       # the calibrated "none" class first, then the model's naming
        machine.q = 0.05
        other = machine.covering(P)[:, machine.known.index("none")]
        return ~other & (named_unknown | (P.max(axis=1) < float(rule[3:])))
    if rule.startswith("conf"):
        machine.q = float(rule[4:])
        return ~machine.covering(P).any(axis=1)
    raise ValueError(f"unknown rule {rule!r}")


def assess_by_rule(rule: str, P: np.ndarray, taught: Sequence[str], states: Dict[str, str], competence: Dict[str, float], counts: Dict[str, int],
                   machine: "AR.AskMachine") -> List[ST.Assessment]:
    """Full assessments under a rule: unseen where the rule says so, Other content where the top option is "none" (or, in the hybrid, where the calibrated none class covers the part),
    otherwise the state of the taught skill the model names. A conformal rule uses the machine's own assessment."""
    P = np.asarray(P, dtype=np.float64)
    if rule.startswith("conf"):
        machine.q = float(rule[4:])
        return machine.assess_many(P)
    flags = unseen_flags(rule, P, taught, machine)
    machine.q = 0.05
    none_cov = machine.covering(P)[:, machine.known.index("none")] if rule.startswith("hyb") else np.zeros(len(P), dtype=bool)
    out: List[ST.Assessment] = []
    for i in range(len(P)):
        top = AR.OPS[int(P[i].argmax())]
        if flags[i]:
            out.append(ST.Assessment("unseen", None, 0.0, 0, float("nan"), 1, top))
        elif none_cov[i] or top == "none":
            out.append(ST.Assessment("other", None, 1.0, 0, 1.0, 1, top))
        else:
            out.append(ST.Assessment(states[top], top, 1.0, int(counts[top]), float(competence[top]), 1, top))
    return out


def cost(false_new: float, found: float) -> float:
    return COST_FALSE_ALARM * false_new + COST_MISS * (1.0 - found)


def evaluate_seed(seed: int, out_dir=OUT, pool: str = "dev", with_realism: bool = True) -> Dict[str, dict]:
    """Every rule on one seed, from the cached probabilities; the cache must already hold every text (lab_ask.py ran this seed)."""
    def never(_texts):
        raise RuntimeError("the cache is missing answers; run lab_ask.py for this seed first")
    world = R.build_world(seed, pool, R.SIZES)
    cache = Path(out_dir) / f"probs_s{seed}.json"
    P = lambda ts: AR.probabilities(ts, cache, never)
    states = {s: ST.group_state(world.comp[s]["count"], world.comp[s]["accuracy"]) for s in world.taught}
    machine = AR.AskMachine({**{s: P(world.cal[s]) for s in world.taught}, "none": P(world.cal["other"])}, states,
                            {s: world.comp[s]["accuracy"] for s in world.taught}, {s: world.comp[s]["count"] for s in world.taught}, q=0.05)
    cases = world.test_cases
    Pt = P(world.test_texts)
    held = [i for i, c in enumerate(cases) if c.kind == "arith" and c.family in HELD and c.skill in world.taught]
    nev = [i for i, c in enumerate(cases) if c.kind == "arith" and c.skill not in world.taught]
    traps = [i for i, c in enumerate(cases) if c.kind == "trap"]
    others = [i for i, c in enumerate(cases) if c.kind == "other"]
    rows = LA.realism_rows() if with_realism else []
    Pr = P([r["text"] for r in rows]) if rows else None
    out: Dict[str, dict] = {}
    for rule in RULES:
        f = unseen_flags(rule, Pt, world.taught, machine)
        rec = {"false_new": float(f[held].mean()), "found": float(f[nev].mean()), "traps_unseen": float(f[traps].mean()), "others_unseen": float(f[others].mean())}
        rec["cost"] = cost(rec["false_new"], rec["found"])
        if rows:
            fr = unseen_flags(rule, Pr, world.taught, machine)
            kn = [i for i, r in enumerate(rows) if r["skill"] in world.taught]
            nw = [i for i, r in enumerate(rows) if r["skill"] not in world.taught]
            rec["real_false_new"] = float(fr[kn].mean()) if kn else float("nan")
            rec["real_found"] = float(fr[nw].mean()) if nw else float("nan")
        out[rule] = rec
    return out


def report(per_seed: List[Dict[str, dict]], seeds: Sequence[int]) -> str:
    L = [f"# Rules on the asked probabilities (seeds {seeds[0]}-{seeds[-1]}), from the cached answers", "",
         f"Cost per part = {COST_FALSE_ALARM:g} x (known skill in a new wording called unseen) + {COST_MISS:g} x (never-taught skill missed); real problems are shown, never used in the choice.", "",
         "| rule | false-new S4-S8 | never-taught found | cost | traps unseen | everyday unseen | real: taught called unseen | real: never-taught found |", "|---|---|---|---|---|---|---|---|"]
    mean = lambda rule, k: float(np.nanmean([r[rule].get(k, float("nan")) for r in per_seed]))
    for rule in RULES:
        L.append(f"| {rule} | {mean(rule, 'false_new'):.3f} | {mean(rule, 'found'):.3f} | {mean(rule, 'cost'):.3f} | {mean(rule, 'traps_unseen'):.3f} | {mean(rule, 'others_unseen'):.3f} | "
                 f"{mean(rule, 'real_false_new'):.3f} | {mean(rule, 'real_found'):.3f} |")
    best = min(RULES, key=lambda r: mean(r, "cost"))
    L += ["", f"Chosen on cost: **{best}**.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (150, 154)
    per = [evaluate_seed(s) for s in range(lo, hi + 1)]
    text = report(per, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_ASK_RULES_{lo}_{hi}.md", text)
    print(text)

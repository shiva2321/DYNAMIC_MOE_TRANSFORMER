"""Lab E (the boundary question). Stage one asks the model "does this message ask for a calculation at all?" (Yes / No); its threshold is calibrated on the held-out ARITHMETIC requests of the
taught skills in the wordings S1-S3 only (so the false "Other" rate on taught arithmetic is the chosen level), and a part below it is Other content. Stage two is the rule of lab A (tau0.5).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_gate.py [first_seed last_seed]

Needs the asked-operation answers of lab A in the cache (lab_ask.py or lab_ask_report.py ran the seed). One small result file per seed, written atomically.
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
from sklearn.metrics import roc_auc_score  # noqa: E402

import ask_reader as AR  # noqa: E402
import ask_rules as RU  # noqa: E402
import evaluate as EV  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "gate"
ASK_CACHE = KS / "results" / "routing_lab" / "ask"
STAMP_FILES = [HERE / "ask_reader.py", HERE / "ask_rules.py", HERE / "lab_gate.py", KS / "run_e2.py", KS / "status.py", KS / "evaluate.py"]
SEEDS = list(range(150, 155))
LEVELS = (0.01, 0.05)
HELD = ("S4", "S5", "S6", "S7", "S8")


def gate_threshold(p_yes_cal: np.ndarray, alpha: float) -> float:
    """The level-alpha lower quantile of the Yes-probabilities of held-out arithmetic requests: a request below it is declared Other content. About alpha of the taught arithmetic in the
    calibrated wordings falls below (the finite-sample rank rule floor(alpha * (n + 1)) is used, so no more than alpha of them)."""
    v = np.sort(np.asarray(p_yes_cal, dtype=np.float64))
    k = int(np.floor(alpha * (len(v) + 1)))
    return float(v[k - 1]) if k >= 1 else -np.inf if len(v) == 0 else float(v[0]) - 1e-12


def two_stage(P: np.ndarray, pyes: np.ndarray, theta: float, taught: Sequence[str], states: Dict[str, str], comp: Dict[str, float], counts: Dict[str, int],
              machine: "AR.AskMachine", rule: str = "tau0.5") -> List[ST.Assessment]:
    """Stage one (below the threshold = Other content), then the rule of lab A for the rest."""
    base = RU.assess_by_rule(rule, P, taught, states, comp, counts, machine)
    return [ST.Assessment("other", None, 1.0, 0, 1.0, 1, a.nearest) if py < theta else a for a, py in zip(base, np.asarray(pyes))]


def run_seed(seed: int, out_dir, stamp: str, ask=AR._gate_model) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, "dev", R.SIZES)
    never = lambda ts: (_ for _ in ()).throw(RuntimeError("the asked-operation answers are missing; run lab A for this seed first"))
    gcache = Path(out_dir) / f"gate_s{seed}.json"
    texts = list(world.test_texts) + [t for g in world.cal.values() for t in g]
    AR.gate_probabilities(texts, gcache, ask)
    Pg = lambda ts: AR.p_yes(AR.gate_probabilities(ts, gcache, ask))
    Pa = lambda ts: AR.probabilities(ts, ASK_CACHE / f"probs_s{seed}.json", never)
    states = {s: ST.group_state(world.comp[s]["count"], world.comp[s]["accuracy"]) for s in world.taught}
    comp = {s: world.comp[s]["accuracy"] for s in world.taught}
    counts = {s: world.comp[s]["count"] for s in world.taught}
    machine = AR.AskMachine({**{s: Pa(world.cal[s]) for s in world.taught}, "none": Pa(world.cal["other"])}, states, comp, counts, q=0.05)
    cases, Pt, py = world.test_cases, Pa(world.test_texts), Pg(world.test_texts)
    cal_py = np.concatenate([Pg(world.cal[s]) for s in world.taught])
    arith = [i for i, c in enumerate(cases) if c.kind == "arith"]
    held = [i for i in arith if cases[i].family in HELD and cases[i].skill in world.taught]
    nev = [i for i in arith if cases[i].skill not in world.taught]
    traps = [i for i, c in enumerate(cases) if c.kind == "trap"]
    others = [i for i, c in enumerate(cases) if c.kind == "other"]

    def auc(neg):
        y = [0] * len(arith) + [1] * len(neg)
        return float(roc_auc_score(y, list(1 - py[arith]) + list(1 - py[neg]))) if neg else float("nan")

    rec = {"seed": seed, "code_stamp": stamp, "auc_trap": auc(traps), "auc_other": auc(others), "levels": {}}
    base = EV.status_scores(RU.assess_by_rule("tau0.5", Pt, world.taught, states, comp, counts, machine), cases)
    rec["tau0.5_alone"] = {"false_new": base["false_unseen_heldout"], "found": base["unseen_recall"], "traps_unseen": base["trap_as_unseen"], "others_unseen": base["other_as_unseen"],
                           "other_labelled": base["other_labelled"], "trap_labelled_other": base["trap_labelled_other"]}
    for a in LEVELS:
        theta = gate_threshold(cal_py, a)
        sc = EV.status_scores(two_stage(Pt, py, theta, world.taught, states, comp, counts, machine), cases)
        rec["levels"][str(a)] = {"theta": theta, "false_other_held": float(np.mean(py[held] < theta)), "false_other_new": float(np.mean(py[nev] < theta)) if nev else float("nan"),
                                 "trap_other": float(np.mean(py[traps] < theta)), "everyday_other": float(np.mean(py[others] < theta)),
                                 "false_new": sc["false_unseen_heldout"], "found": sc["unseen_recall"], "traps_unseen": sc["trap_as_unseen"], "others_unseen": sc["other_as_unseen"],
                                 "other_labelled": sc["other_labelled"], "trap_labelled_other": sc["trap_labelled_other"]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(R._native(rec)))
    return rec


def _m(rows, f):
    v = [f(r) for r in rows]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def report(rows: List[dict], seeds: Sequence[int]) -> str:
    L = [f"# Lab E, the boundary question (seeds {seeds[0]}-{seeds[-1]})", "",
         "Stage one: the model is asked whether the message asks for a calculation at all; the threshold is calibrated on held-out arithmetic requests (taught skills, wordings S1-S3) so that about alpha of them are "
         "called Other. Stage two: the rule tau0.5 of lab A.", "",
         f"Separating arithmetic requests from look-alike traps by the Yes-probability: AUROC {_m(rows, lambda r: r['auc_trap']):.3f}; from everyday sentences: {_m(rows, lambda r: r['auc_other']):.3f}.", "",
         "| | tau0.5 alone | two-stage, alpha 0.01 | two-stage, alpha 0.05 |", "|---|---|---|---|"]
    keys = [("known skills in new wordings called unseen", "false_new"), ("never-taught skills found", "found"), ("look-alike traps called unseen", "traps_unseen"),
            ("everyday sentences called unseen", "others_unseen"), ("everyday sentences labelled Other", "other_labelled"), ("traps labelled Other", "trap_labelled_other")]
    for name, k in keys:
        L.append(f"| {name} | {_m(rows, lambda r: r['tau0.5_alone'][k]):.3f} | " + " | ".join(f"{_m(rows, lambda r, a=a: r['levels'][str(a)][k]):.3f}" for a in LEVELS) + " |")
    for name, k in (("real arithmetic in new wordings wrongly called Other", "false_other_held"), ("never-taught arithmetic wrongly called Other", "false_other_new"),
                    ("traps called Other", "trap_other"), ("everyday sentences called Other", "everyday_other")):
        L.append(f"| {name} | | " + " | ".join(f"{_m(rows, lambda r, a=a: r['levels'][str(a)][k]):.3f}" for a in LEVELS) + " |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows = []
    for seed in range(lo, hi + 1):
        r = run_seed(seed, OUT, stamp)
        rows.append(r)
        print(f"seed {seed}: auc trap {r['auc_trap']:.3f} everyday {r['auc_other']:.3f}; alpha 0.01 everyday unseen {r['levels']['0.01']['others_unseen']:.3f} traps unseen {r['levels']['0.01']['traps_unseen']:.3f}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_GATE_{lo}_{hi}.md", text)
    print(text)

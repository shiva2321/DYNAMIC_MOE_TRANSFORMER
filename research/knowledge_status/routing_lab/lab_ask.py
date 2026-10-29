"""Lab A (asking reader). Does a reader that ASKS the language model "which operation is this?" keep a known skill known when the wording changes, and still find the skills nobody taught?

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_ask.py [first_seed last_seed] [pool]

Tuning seeds 150 to 154 by default; one small result file per seed, written atomically; the model's answers are cached per seed so a restart after a power cut costs nothing.
The prompt is fixed in ask_reader.py before any run (no tuning on test texts). The realism set (real SVAMP problems) is read once per run, for a report line only, never for a choice.
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
import evaluate as EV  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import skills  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "ask"
REALISM = KS / "results" / "realism_set.jsonl"
STAMP_FILES = [HERE / "ask_reader.py", HERE / "lab_ask.py", KS / "run_e2.py", KS / "status.py", KS / "evaluate.py", KS / "assignments.py", KS / "generator.py"]
SEEDS = list(range(150, 155))
HELD = ("S4", "S5", "S6", "S7", "S8")
TAUGHT_FAMILIES = ("S1", "S2", "S3")


def naming_rule(P: np.ndarray, taught: Sequence[str]) -> List[str]:
    """The plain rule with no calibration: known if the model's top option is a taught skill or "none", unseen otherwise."""
    known = set(taught) | {"none"}
    return ["known" if AR.OPS[int(np.argmax(r))] in known else "unseen" for r in P]


def set_stats(cov: np.ndarray, rows: Sequence[int], true_col: Sequence) -> dict:
    """The covering sets of the rows: how often the true class is in the set, how many are empty (nothing explains the part), singletons, larger sets, and the mean size of the non-empty sets."""
    if not rows:
        return {"n": 0, "coverage": float("nan"), "empty": float("nan"), "single": float("nan"), "multi": float("nan"), "mean_size_nonempty": float("nan")}
    sz = cov[list(rows)].sum(axis=1)
    hit = [bool(cov[i, c]) for i, c in zip(rows, true_col) if c is not None]
    return {"n": len(rows), "coverage": float(np.mean(hit)) if hit else float("nan"), "empty": float(np.mean(sz == 0)), "single": float(np.mean(sz == 1)), "multi": float(np.mean(sz >= 2)),
            "mean_size_nonempty": float(sz[sz > 0].mean()) if (sz > 0).any() else float("nan")}


def realism_rows(path: Path = REALISM) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_seed(seed: int, out_dir, stamp: str, pool: str = "dev", q: float = 0.05, ask=AR._ask_model, with_realism: bool = True) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, pool, R.SIZES)
    cache = Path(out_dir) / f"probs_s{seed}.json"
    rows = realism_rows() if with_realism else []
    texts = list(world.test_texts) + [t for g in world.cal.values() for t in g] + [r["text"] for r in rows]
    AR.probabilities(texts, cache, ask)
    P = lambda ts: AR.probabilities(ts, cache, ask)
    states = {s: ST.group_state(world.comp[s]["count"], world.comp[s]["accuracy"]) for s in world.taught}
    competence = {s: world.comp[s]["accuracy"] for s in world.taught}
    counts = {s: world.comp[s]["count"] for s in world.taught}
    cal = {s: P(world.cal[s]) for s in world.taught}
    cal["none"] = P(world.cal["other"])
    machine = AR.AskMachine(cal, states, competence, counts, q=q)
    Pt = P(world.test_texts)
    assessed = machine.assess_many(Pt)
    cases = world.test_cases
    scores = EV.status_scores(assessed, cases)
    rule = naming_rule(Pt, world.taught)
    arith = [i for i, c in enumerate(cases) if c.kind == "arith"]
    held = [i for i in arith if cases[i].family in HELD and cases[i].skill in world.taught]
    never = [i for i in arith if cases[i].skill not in world.taught]
    truth_op = [cases[i].skill if cases[i].kind in ("arith", "twin") else None for i in range(len(cases))]
    naming = {"all": AR.top1_accuracy(Pt, [t if cases[i].kind == "arith" else None for i, t in enumerate(truth_op)])}
    for f in sorted({cases[i].family for i in arith}):
        naming[f] = AR.top1_accuracy(Pt, [truth_op[i] if i in set(arith) and cases[i].family == f else None for i in range(len(cases))])
    by_skill = {s: AR.top1_accuracy(Pt, [truth_op[i] if cases[i].kind == "arith" and cases[i].skill == s else None for i in range(len(cases))]) for s in skills.SKILLS}
    rate = lambda idx, flags: float(np.mean([flags[i] for i in idx])) if idx else float("nan")
    unseen_flag = [a.status == "unseen" for a in assessed]
    rule_flag = [r == "unseen" for r in rule]
    cov = machine.covering(Pt)
    taught_rows = [i for i in arith if cases[i].family in TAUGHT_FAMILIES and cases[i].skill in world.taught]
    sets = {"held": set_stats(cov, held, [machine.known.index(cases[i].skill) for i in held]), "taught_wordings": set_stats(cov, taught_rows, [machine.known.index(cases[i].skill) for i in taught_rows]),
            "never": set_stats(cov, never, [None] * len(never))}
    rec = {"seed": seed, "code_stamp": stamp, "q": q, "assignment": world.assignment, "scores": R._native(scores), "sets": sets,
           "false_new_conformal": rate(held, unseen_flag), "found_conformal": rate(never, unseen_flag),
           "false_new_rule": rate(held, rule_flag), "found_rule": rate(never, rule_flag),
           "false_new_taught_wordings": rate([i for i in arith if cases[i].family in TAUGHT_FAMILIES and cases[i].skill in world.taught], unseen_flag),
           "naming_accuracy": naming, "naming_by_skill": by_skill,
           "false_new_by_family": {f: rate([i for i in held if cases[i].family == f], unseen_flag) for f in HELD}}
    if rows:
        Pr = P([r["text"] for r in rows])
        ar = machine.assess_many(Pr)
        rr = naming_rule(Pr, world.taught)
        known_rows = [i for i, r in enumerate(rows) if r["skill"] in world.taught]
        new_rows = [i for i, r in enumerate(rows) if r["skill"] not in world.taught]
        rec["realism"] = {"n_known": len(known_rows), "n_new": len(new_rows),
                          "false_new_conformal": rate(known_rows, [a.status == "unseen" for a in ar]), "found_conformal": rate(new_rows, [a.status == "unseen" for a in ar]),
                          "false_new_rule": rate(known_rows, [x == "unseen" for x in rr]), "found_rule": rate(new_rows, [x == "unseen" for x in rr]),
                          "naming_accuracy": AR.top1_accuracy(Pr, [r["skill"] for r in rows])}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def _m(rows: List[dict], f) -> float:
    v = [f(r) for r in rows]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def report(rows: List[dict], seeds: Sequence[int], pool: str) -> str:
    L = [f"# Lab A, the asking reader (pool {pool}, seeds {seeds[0]}-{seeds[-1]}, q = 0.05)", "",
         "The language model is asked which operation a message wants; known and unseen are decided on its answer probabilities (calibration on S1-S3 held-out parts only). The look-based machine on the same seeds "
         "(tuning seeds 150-154, results/FUNCTION_LAB.md): known skills in new wordings called unseen 0.256, never-taught skills found 0.903.", "",
         "| measure | conformal (calibrated on S1-S3) | plain naming rule |", "|---|---|---|",
         f"| known skills in new wordings S4-S8 called unseen (bar 0.05) | {_m(rows, lambda r: r['false_new_conformal']):.3f} | {_m(rows, lambda r: r['false_new_rule']):.3f} |",
         f"| never-taught skills found | {_m(rows, lambda r: r['found_conformal']):.3f} | {_m(rows, lambda r: r['found_rule']):.3f} |",
         f"| known skills in the taught wordings S1-S3 called unseen | {_m(rows, lambda r: r['false_new_taught_wordings']):.3f} | |",
         f"| macro F1 (Solid / Thin / Unseen) | {_m(rows, lambda r: r['scores']['macro_f1']['macro_f1']):.3f} | |",
         f"| Thin against Solid, AUROC | {_m(rows, lambda r: r['scores'].get('thin_vs_solid_auroc', float('nan'))):.3f} | |",
         f"| everyday sentences called unseen | {_m(rows, lambda r: r['scores'].get('other_as_unseen', float('nan'))):.3f} | |",
         f"| look-alike traps called unseen | {_m(rows, lambda r: r['scores'].get('trap_as_unseen', float('nan'))):.3f} | |", ""]
    L += ["Operation named correctly (top option = the true operation), by wording: " + ", ".join(f"{f} {_m(rows, lambda r: r['naming_accuracy'].get(f, float('nan'))):.3f}" for f in sorted(rows[0]['naming_accuracy']) if f != 'all'),
          "By skill: " + ", ".join(f"{s} {_m(rows, lambda r: r['naming_by_skill'][s]):.3f}" for s in skills.SKILLS),
          "Known skills called unseen, by new wording: " + ", ".join(f"{f} {_m(rows, lambda r: r['false_new_by_family'][f]):.3f}" for f in HELD), ""]
    for key, label in (("taught_wordings", "known skills, taught wordings S1-S3"), ("held", "known skills, new wordings S4-S8"), ("never", "never-taught skills")):
        L.append(f"Covering set, {label}: true skill in the set {_m(rows, lambda r: r['sets'][key]['coverage']):.3f}, empty {_m(rows, lambda r: r['sets'][key]['empty']):.3f}, "
                 f"one expert {_m(rows, lambda r: r['sets'][key]['single']):.3f}, two or more {_m(rows, lambda r: r['sets'][key]['multi']):.3f}, mean size when not empty {_m(rows, lambda r: r['sets'][key]['mean_size_nonempty']):.2f}")
    L.append("")
    real = [r for r in rows if "realism" in r]
    if real:
        L += ["Real problems (SVAMP, 300 one-step problems, add / sub / mul / div; reported once, never used to choose anything):",
              f"- operation named correctly {_m(real, lambda r: r['realism']['naming_accuracy']):.3f}",
              f"- problems of a taught skill called unseen {_m(real, lambda r: r['realism']['false_new_conformal']):.3f} (plain rule {_m(real, lambda r: r['realism']['false_new_rule']):.3f}); the look-based machine called 0.98 of them unseen",
              f"- problems of a never-taught skill called unseen {_m(real, lambda r: r['realism']['found_conformal']):.3f} (plain rule {_m(real, lambda r: r['realism']['found_rule']):.3f})", ""]
    return "\n".join(L)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    pool = args.pop() if args and args[-1] in ("dev",) else "dev"
    lo, hi = (int(args[0]), int(args[1])) if len(args) >= 2 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows = []
    for seed in range(lo, hi + 1):
        r = run_seed(seed, OUT, stamp, pool)
        rows.append(r)
        print(f"seed {seed}: false-new {r['false_new_conformal']:.3f} (rule {r['false_new_rule']:.3f})  found {r['found_conformal']:.3f}  naming {r['naming_accuracy']['all']:.3f}", flush=True)
    text = report(rows, list(range(lo, hi + 1)), pool)
    runlib.atomic_write_text(KS / "results" / f"ROUTING_ASK_{lo}_{hi}.md", text)
    print(text)

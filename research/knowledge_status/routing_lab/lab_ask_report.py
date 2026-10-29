"""Lab A on REPORT seeds: the rules chosen on the tuning seeds, applied to seeds nobody tuned on, next to the look-based machine's cells for the same seeds.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_ask_report.py [first_seed last_seed]

Rules (declared in PROGRESS.md before this run): tau0.5 (chosen on cost on seeds 150-154), hyb0.5 (the calibrated none class first) and conf0.05 (the first design). Per seed one small file,
written atomically; the model's answers are cached. Includes the tangled messages (gold parts, one asked expert-team per part) and the real problems (SVAMP, report only).
"""

import json
import sys
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent
KS = HERE.parent
ROOT = KS.parents[1]
for _p in (str(ROOT), str(KS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ask_reader as AR  # noqa: E402
import ask_rules as RU  # noqa: E402
import evaluate as EV  # noqa: E402
import lab_ask as LA  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "ask_report"
CACHE_DIR = KS / "results" / "routing_lab" / "ask"
STAMP_FILES = [HERE / "ask_reader.py", HERE / "ask_rules.py", HERE / "lab_ask.py", HERE / "lab_ask_report.py", KS / "run_e2.py", KS / "status.py", KS / "evaluate.py"]
REPORT_RULES = ("tau0.5", "hyb0.5", "conf0.05")
SEEDS = list(range(100, 110))


def gold_texts(world) -> List[List[str]]:
    return [[m.text[a:b] for a, b in R.segment_sets(m)["gold"]] for m in world.tangled]


def run_seed(seed: int, out_dir, stamp: str, ask=AR._ask_model) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, "dev", R.SIZES)
    cache = CACHE_DIR / f"probs_s{seed}.json"
    rows = LA.realism_rows()
    tang_texts = gold_texts(world)
    flat = [t for ts in tang_texts for t in ts]
    AR.probabilities(list(world.test_texts) + [t for g in world.cal.values() for t in g] + flat + [r["text"] for r in rows], cache, ask)
    P = lambda ts: AR.probabilities(ts, cache, ask)
    states = {s: ST.group_state(world.comp[s]["count"], world.comp[s]["accuracy"]) for s in world.taught}
    comp = {s: world.comp[s]["accuracy"] for s in world.taught}
    counts = {s: world.comp[s]["count"] for s in world.taught}
    machine = AR.AskMachine({**{s: P(world.cal[s]) for s in world.taught}, "none": P(world.cal["other"])}, states, comp, counts, q=0.05)
    cases, Pt, Pr = world.test_cases, P(world.test_texts), P([r["text"] for r in rows])
    segs = [R.segment_sets(m)["gold"] for m in world.tangled]
    solid_rows = [i for i, r in enumerate(rows) if states.get(r["skill"]) == "solid"]
    taught_rows = [i for i, r in enumerate(rows) if r["skill"] in world.taught]
    new_rows = [i for i, r in enumerate(rows) if r["skill"] not in world.taught]
    rec: dict = {"seed": seed, "code_stamp": stamp, "assignment": world.assignment, "rules": {}}
    for rule in REPORT_RULES:
        assessed = RU.assess_by_rule(rule, Pt, world.taught, states, comp, counts, machine)
        sc = EV.status_scores(assessed, cases)
        by_part = []
        for i in range(len(world.tangled)):
            a = RU.assess_by_rule(rule, P(tang_texts[i]), world.taught, states, comp, counts, machine)
            by_part.append(list(zip(segs[i], a)))
        tg = EV.tangled_scores(world.tangled, by_part, "gold", world.truth)
        machine.q = 0.05
        fr = ~machine.covering(Pr).any(axis=1) if rule.startswith("conf") else RU.unseen_flags(rule, Pr, world.taught, machine)
        rate = lambda idx: float(fr[idx].mean()) if idx else float("nan")
        rec["rules"][rule] = {"false_new": sc["false_unseen_heldout"], "found": sc["unseen_recall"], "macro_f1": sc["macro_f1"]["macro_f1"], "other_unseen": sc["other_as_unseen"],
                              "trap_unseen": sc["trap_as_unseen"], "twin_accuracy": sc["twin_accuracy"], "b4a": tg["b4a"], "b4b": tg["b4b"], "b4c": tg["b4c"],
                              "real_false_new_solid": rate(solid_rows), "real_false_new_taught": rate(taught_rows), "real_found": rate(new_rows)}
    look = json.loads((KS / "results" / "e2" / f"R5_R4_qwen_L26__s{seed}.json").read_text(encoding="utf-8"))
    ls = look["machine"]["q0.05"]["standard"]
    lt = look["tangled"]["machine"]["gold"]
    lp = KS / "results" / "realism" / f"s{seed}.json"
    lr = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    rec["look"] = {"false_new": ls["false_unseen_heldout"], "found": ls["unseen_recall"], "macro_f1": ls["macro_f1"]["macro_f1"], "other_unseen": ls["other_as_unseen"], "trap_unseen": ls["trap_as_unseen"],
                   "twin_accuracy": ls["twin_accuracy"], "b4a": lt["b4a"], "b4b": lt["b4b"], "b4c": lt["b4c"], "real_false_new_solid": lr.get("real_false_unseen"), "real_found": lr.get("real_unseen_found")}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(R._native(rec)))
    return rec


def _m(rows: List[dict], getter) -> float:
    v = []
    for r in rows:
        try:
            x = getter(r)
        except (KeyError, TypeError):
            continue
        if x is not None and x == x:
            v.append(float(x))
    return float(np.mean(v)) if v else float("nan")


def report(rows: List[dict], seeds: Sequence[int]) -> str:
    L = [f"# Lab A on report seeds {seeds[0]}-{seeds[-1]}: the asking reader against the look-based machine (same seeds, q = 0.05 for the conformal parts)", "",
         "Bars: false-new 0.05, traps 0.05, B4a 0.85 / 0.95, B4b 0.95 / 0.95 (spec section 7). The rule tau0.5 was chosen on the tuning seeds 150-154 by cost; nothing here was used to choose.", "",
         "| measure | look-based machine | tau0.5 (asks) | hyb0.5 (none class first) | conf0.05 (asks, calibrated) |", "|---|---|---|---|---|"]

    def line(name, f):
        L.append(f"| {name} | {_m(rows, lambda r: f(r['look'])):.3f} | " + " | ".join(f"{_m(rows, lambda r, k=k: f(r['rules'][k])):.3f}" for k in REPORT_RULES) + " |")

    line("known skills in new wordings called unseen", lambda d: d["false_new"])
    line("never-taught skills found", lambda d: d["found"])
    line("macro F1 (Thin / Solid labels come from the assignment here: read it as identification only)", lambda d: d["macro_f1"])
    line("everyday sentences called unseen", lambda d: d["other_unseen"])
    line("look-alike traps called unseen", lambda d: d["trap_unseen"])
    line("near-twins labelled right", lambda d: d["twin_accuracy"])
    line("tangled: new part found (B4a residual hit)", lambda d: d["b4a"]["residual_hit"])
    line("tangled: known part left alone (B4a)", lambda d: d["b4a"]["known_not_flagged"])
    line("tangled: nothing flagged when known + Other (B4b)", lambda d: d["b4b"]["nothing_flagged"])
    line("tangled: Other part labelled Other (B4b)", lambda d: d["b4b"]["other_labelled"])
    line("real problems: Solid skills called unseen (B7)", lambda d: d.get("real_false_new_solid"))
    line("real problems: never-taught skills found", lambda d: d.get("real_found"))
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
        t = r["rules"]["tau0.5"]
        print(f"seed {seed}: tau0.5 false-new {t['false_new']:.3f} found {t['found']:.3f} real {t['real_false_new_solid']} | look {r['look']['false_new']:.3f} {r['look']['found']:.3f}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_ASK_REPORT_{lo}_{hi}.md", text)
    print(text)

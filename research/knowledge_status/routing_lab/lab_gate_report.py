"""Lab E on REPORT seeds: the two-stage rule (boundary question, then tau0.5) against tau0.5 alone and the look-based machine, same seeds as lab_ask_report.py.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_gate_report.py [first_seed last_seed]

The levels (alpha 0.05 primary, 0.01 for comparison) were declared before this run in PROGRESS.md; alpha 0.05 is the one that met the rule on the tuning seeds. The asked-operation answers come from the
cache of lab_ask_report.py; only the Yes / No question is asked here. One small file per seed, written atomically.
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
import lab_ask_report as LAR  # noqa: E402
import lab_gate as LG  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "gate_report"
ASK_CACHE = KS / "results" / "routing_lab" / "ask"
STAMP_FILES = [HERE / "ask_reader.py", HERE / "ask_rules.py", HERE / "lab_gate.py", HERE / "lab_gate_report.py", HERE / "lab_ask_report.py", KS / "run_e2.py", KS / "status.py", KS / "evaluate.py"]
ARMS = (("tau0.5", None), ("gate0.05", 0.05), ("gate0.01", 0.01))
SEEDS = list(range(100, 110))


def run_seed(seed: int, out_dir, stamp: str, ask_gate=AR._gate_model) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, "dev", R.SIZES)
    rows = LA.realism_rows()
    tang_texts = LAR.gold_texts(world)
    flat = [t for ts in tang_texts for t in ts]
    real_texts = [r["text"] for r in rows]
    gcache = Path(out_dir) / f"gate_s{seed}.json"
    AR.gate_probabilities(list(world.test_texts) + [t for g in world.cal.values() for t in g] + flat + real_texts, gcache, ask_gate)
    Pg = lambda ts: AR.p_yes(AR.gate_probabilities(ts, gcache, ask_gate))
    never = lambda ts: (_ for _ in ()).throw(RuntimeError("the asked-operation answers are missing; run lab_ask_report.py for this seed first"))
    Pa = lambda ts: AR.probabilities(ts, ASK_CACHE / f"probs_s{seed}.json", never)
    states = {s: ST.group_state(world.comp[s]["count"], world.comp[s]["accuracy"]) for s in world.taught}
    comp = {s: world.comp[s]["accuracy"] for s in world.taught}
    counts = {s: world.comp[s]["count"] for s in world.taught}
    machine = AR.AskMachine({**{s: Pa(world.cal[s]) for s in world.taught}, "none": Pa(world.cal["other"])}, states, comp, counts, q=0.05)
    cal_py = np.concatenate([Pg(world.cal[s]) for s in world.taught])
    segs = [R.segment_sets(m)["gold"] for m in world.tangled]
    solid_rows = [i for i, r in enumerate(rows) if states.get(r["skill"]) == "solid"]
    new_rows = [i for i, r in enumerate(rows) if r["skill"] not in world.taught]
    rec: dict = {"seed": seed, "code_stamp": stamp, "arms": {}}
    for name, alpha in ARMS:
        theta = LG.gate_threshold(cal_py, alpha) if alpha else -np.inf

        def assess(texts):
            return LG.two_stage(Pa(texts), Pg(texts), theta, world.taught, states, comp, counts, machine)

        sc = EV.status_scores(assess(world.test_texts), world.test_cases)
        tg = EV.tangled_scores(world.tangled, [list(zip(segs[i], assess(tang_texts[i]))) for i in range(len(world.tangled))], "gold", world.truth)
        fr = np.array([a.status == "unseen" for a in assess(real_texts)])
        rate = lambda idx: float(fr[idx].mean()) if idx else float("nan")
        rec["arms"][name] = {"false_new": sc["false_unseen_heldout"], "found": sc["unseen_recall"], "macro_f1": sc["macro_f1"]["macro_f1"], "other_unseen": sc["other_as_unseen"],
                             "trap_unseen": sc["trap_as_unseen"], "twin_accuracy": sc["twin_accuracy"], "other_labelled": sc["other_labelled"], "trap_labelled_other": sc["trap_labelled_other"],
                             "b4a": tg["b4a"], "b4b": tg["b4b"], "b4c": tg["b4c"], "real_false_new_solid": rate(solid_rows), "real_found": rate(new_rows)}
    look = json.loads((KS / "results" / "e2" / f"R5_R4_qwen_L26__s{seed}.json").read_text(encoding="utf-8"))
    ls, lt = look["machine"]["q0.05"]["standard"], look["tangled"]["machine"]["gold"]
    lp = KS / "results" / "realism" / f"s{seed}.json"
    lr = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    rec["look"] = {"false_new": ls["false_unseen_heldout"], "found": ls["unseen_recall"], "macro_f1": ls["macro_f1"]["macro_f1"], "other_unseen": ls["other_as_unseen"], "trap_unseen": ls["trap_as_unseen"],
                   "twin_accuracy": ls["twin_accuracy"], "other_labelled": ls["other_labelled"], "trap_labelled_other": ls["trap_labelled_other"], "b4a": lt["b4a"], "b4b": lt["b4b"], "b4c": lt["b4c"],
                   "real_false_new_solid": lr.get("real_false_unseen"), "real_found": lr.get("real_unseen_found")}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(R._native(rec)))
    return rec


def report(rows: List[dict], seeds: Sequence[int]) -> str:
    m = LAR._m
    L = [f"# Lab E on report seeds {seeds[0]}-{seeds[-1]}: the two-stage asking reader (boundary question first) against the look-based machine", "",
         "Bars: false-new 0.05, traps 0.05, everyday 0.05, B4a 0.85 / 0.95, B4b 0.95 / 0.95, B7 real problems called unseen at most 0.30 (my line for the claim). gate0.05 is the arm that met the rule on the tuning seeds.", "",
         "| measure | look-based machine | tau0.5 alone | two-stage, alpha 0.05 | two-stage, alpha 0.01 |", "|---|---|---|---|---|"]

    def line(name, f):
        L.append(f"| {name} | {m(rows, lambda r: f(r['look'])):.3f} | " + " | ".join(f"{m(rows, lambda r, k=k: f(r['arms'][k])):.3f}" for k, _ in ARMS) + " |")

    line("known skills in new wordings called unseen", lambda d: d["false_new"])
    line("never-taught skills found", lambda d: d["found"])
    line("everyday sentences called unseen", lambda d: d["other_unseen"])
    line("look-alike traps called unseen", lambda d: d["trap_unseen"])
    line("everyday sentences labelled Other", lambda d: d["other_labelled"])
    line("look-alike traps labelled Other", lambda d: d["trap_labelled_other"])
    line("near-twins labelled right", lambda d: d["twin_accuracy"])
    line("tangled: new part found (B4a residual hit)", lambda d: d["b4a"]["residual_hit"])
    line("tangled: known part left alone (B4a)", lambda d: d["b4a"]["known_not_flagged"])
    line("tangled: nothing flagged when known + Other (B4b)", lambda d: d["b4b"]["nothing_flagged"])
    line("tangled: Other part labelled Other (B4b)", lambda d: d["b4b"]["other_labelled"])
    line("tangled: new part found next to Other (B4c)", lambda d: d["b4c"]["unseen_found"])
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
        g = r["arms"]["gate0.05"]
        print(f"seed {seed}: gate0.05 false-new {g['false_new']:.3f} found {g['found']:.3f} everyday unseen {g['other_unseen']:.3f} traps {g['trap_unseen']:.3f} real {g['real_false_new_solid']}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_GATE_REPORT_{lo}_{hi}.md", text)
    print(text)

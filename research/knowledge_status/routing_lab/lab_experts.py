"""Lab C (experts that share). Does a skill taught only a few examples become competent when its expert shares the number encoder with experts that have many?

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_experts.py [first_seed last_seed]

For each target skill and taught count: (separate) the solver of assignments.py alone; (shared) the target and the helper skills (2000 examples each) trained together with a shared encoder.
Accuracy is on held-out problems (arith's held-out split). One small result file per (seed, skill, count), written atomically.
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

import experts as EX  # noqa: E402
import runlib  # noqa: E402
import skills  # noqa: E402
from assignments import Solver  # noqa: E402
from growmoe import arith  # noqa: E402

OUT = KS / "results" / "routing_lab" / "experts"
STAMP_FILES = [HERE / "experts.py", HERE / "lab_experts.py", KS / "assignments.py", ROOT / "growmoe" / "arith.py"]
SEEDS = [150, 151, 152]
TARGETS = ("sub", "mul", "div", "mod", "max")
COUNTS = (20, 100, 500)
HELPERS = ("add", "sub", "max")                  # the helpers are Solid skills that carry digit structure; a target is never its own helper
N_HELP = 2000


def draw(seed: int, skill: str, n: int):
    p = arith.sample_problems(np.random.default_rng([seed, 0xC0, skills.SKILLS.index(skill), n]), n, (skills.ARITH_OP_ID[skill],), (1, 2), (1, 2), "train")
    return p["a"], p["b"], arith.answers(p["op"], p["a"], p["b"])


def run_cell(seed: int, target: str, count: int, stamp: str, out_dir=OUT, steps: int = 6000, epochs: int = 1000) -> dict:
    path = Path(out_dir) / f"s{seed}__{target}__n{count}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    op = skills.ARITH_OP_ID[target]
    a, b, y = draw(seed, target, count)
    sep = Solver(seed=seed, epochs=epochs).fit(a, b, y)
    held = lambda f: EX.held_out_accuracy(f, op, np.random.default_rng([seed, 0xC1, op]))
    helpers = [h for h in HELPERS if h != target]
    data = {target: (a, b, y), **{h: draw(seed, h, N_HELP) for h in helpers}}
    sh = EX.SharedExperts([target] + helpers, seed=seed, steps=steps).fit(data)
    helper_accuracy = {}
    for h in helpers:
        problems = np.random.default_rng([seed, 0xC1, skills.ARITH_OP_ID[h]])                  # which held-out problems are drawn, not a result
        helper_accuracy[h] = EX.held_out_accuracy(sh.solver(h).predict, skills.ARITH_OP_ID[h], problems)
    rec = {"seed": seed, "skill": target, "count": count, "helpers": helpers, "separate": held(sep.predict), "shared": held(sh.solver(target).predict),
           "helper_accuracy": helper_accuracy, "code_stamp": stamp}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def report(rows: List[dict], seeds: Sequence[int]) -> str:
    L = [f"# Lab C, experts that share a number encoder (tuning seeds {seeds[0]}-{seeds[-1]}, 1-2 digit operands, held-out problems)", "",
         "Accuracy on problems never seen: the skill alone (separate) against the skill trained together with helper skills at 2000 examples each (shared encoder, own head). "
         "Chance for div is about 0.46 and for mod 0.24 (their answers are mostly 0 and small).", "",
         "| skill | taught | separate | shared | helpers (shared net) |", "|---|---|---|---|---|"]
    for t in TARGETS:
        for c in COUNTS:
            r = [x for x in rows if x["skill"] == t and x["count"] == c]
            if r:
                L.append(f"| {t} | {c} | {np.mean([x['separate'] for x in r]):.2f} | {np.mean([x['shared'] for x in r]):.2f} | "
                         + ", ".join(f"{h} {np.mean([x['helper_accuracy'][h] for x in r]):.2f}" for h in r[0]["helpers"]) + " |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows: List[dict] = []
    for seed in range(lo, hi + 1):
        for t in TARGETS:
            for c in COUNTS:
                r = run_cell(seed, t, c, stamp)
                rows.append(r)
                print(f"seed {seed} {t} n={c}: separate {r['separate']:.2f} shared {r['shared']:.2f}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_EXPERTS_{lo}_{hi}.md", text)
    print(text)

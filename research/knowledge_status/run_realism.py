"""B7, the realism check (spec 7): does the status machine separate Solid from Unseen on real, human-written one-step problems as it does on the templates?

    PYTHONPATH=<repo root> python run_realism.py --seeds 100-129          # dev cells only; the confirmatory cells are made by run_confirm.py

For each seed the same world is built as in E2 (same assignment of the skills to Solid / Thin / Unseen, same teaching, same calibration, same learned-map reader). Then two sets of parts are
assessed by the same machine: the template parts of the held-out wordings S4 to S8, and the realism problems (add, sub, mul, div only). On each set the balanced accuracy of the call
"Unseen or not" is taken over parts whose skill is Solid (should not be called Unseen) and parts whose skill is Unseen (should be). B7 compares the two numbers.
Realism problems never enter teaching, calibration or tuning (``realism.assert_not_realism``).
"""

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

import assignments
import realism as RL
import reps
import run_e2 as R
import runlib
import status as ST

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "realism"
STAMP_FILES = R.STAMP_FILES + [HERE / "run_realism.py", HERE / "realism.py", R.LEARNED_MAP]
REALISM_N, REALISM_SEED = 300, 100
RealismRows = List[dict]


@dataclass
class RealismWorld(R.World):
    extra_texts: List[str] = field(default_factory=list)

    def all_texts(self) -> List[str]:
        return list(dict.fromkeys(super().all_texts() + self.extra_texts))


def build_realism_world(seed: int, pool: str, sizes: R.Sizes, competence_fn: Optional[Callable], rows: RealismRows) -> RealismWorld:
    world = R.build_world(seed, pool, sizes, competence_fn)
    seen = [e.span for s in world.teach for e in world.teach[s]] + [i.text for i in world.teach_none] + [t for g in list(world.cal.values()) + list(world.adapt.values()) for t in g]
    RL.assert_not_realism(seen, rows)
    return RealismWorld(**{f.name: getattr(world, f.name) for f in fields(R.World)}, extra_texts=[r["text"] for r in rows])


def balanced(solid_flags: List[bool], unseen_flags: List[bool]) -> Optional[float]:
    """Mean of the share of Solid parts NOT called Unseen and the share of Unseen parts called Unseen; None if either kind is missing."""
    if not solid_flags or not unseen_flags:
        return None
    return float(0.5 * (np.mean([not f for f in solid_flags]) + np.mean(unseen_flags)))


def run_seed(seed: int, out_dir, stamp: str, pool: str = "dev", sizes: R.Sizes = R.SIZES, rep: Optional[reps.Representation] = None, competence_fn: Optional[Callable] = None,
             cache_dir=R.CACHE, rows: Optional[RealismRows] = None, rep_id: str = "R5_R4_qwen_L26") -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    rows = rows if rows is not None else RL.build_realism_set(REALISM_N, REALISM_SEED)
    world = build_realism_world(seed, pool, sizes, competence_fn, rows)
    P, index = R.prepare(world, rep_id, rep, cache_dir)

    def V(texts):
        return P[[index[t] for t in texts]]

    groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
    other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
    cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
    machine = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
    cache: Dict[str, ST.Assessment] = {}

    def unseen(text: str) -> bool:
        if text not in cache:
            cache[text] = machine.assess(P[index[text]])
        return cache[text].status == "unseen"

    tmpl = {"solid": [], "unseen": []}
    for text, case in zip(world.test_texts, world.test_cases):
        if case.kind == "arith" and case.family in assignments.TEST_FAMILIES and case.status in tmpl:
            tmpl[case.status].append(unseen(text))
    real = {"solid": [], "unseen": []}
    per_skill: Dict[str, dict] = {}
    for r in rows:
        truth = world.truth.get(r["skill"])
        if truth in real:
            flag = unseen(r["text"])
            real[truth].append(flag)
            d = per_skill.setdefault(r["skill"], {"truth": truth, "n": 0, "called_unseen": 0})
            d["n"] += 1
            d["called_unseen"] += int(flag)
    t_bal, r_bal = balanced(tmpl["solid"], tmpl["unseen"]), balanced(real["solid"], real["unseen"])
    rec = {"seed": seed, "pool": pool, "rep": rep_id, "assignment": world.assignment, "truth": world.truth, "template_bal": t_bal, "real_bal": r_bal,
           "diff": None if t_bal is None or r_bal is None else r_bal - t_bal, "n_template": {k: len(v) for k, v in tmpl.items()}, "n_real": {k: len(v) for k, v in real.items()},
           "real_false_unseen": float(np.mean(real["solid"])) if real["solid"] else None, "real_unseen_found": float(np.mean(real["unseen"])) if real["unseen"] else None,
           "per_skill": per_skill, "code_stamp": stamp}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def report_b7(cells: List[dict], label: str = "dev") -> str:
    ok = [c for c in cells if c["diff"] is not None]
    L = [f"# B7, realism check ({label}; {len(ok)} of {len(cells)} seeds have both a Solid and an Unseen skill among add, sub, mul, div)", "",
         "Balanced accuracy of the call 'Unseen or not' over Solid and Unseen parts; template = held-out wordings S4 to S8; realism = SVAMP one-step problems (never used for teaching or tuning).", ""]
    if not ok:
        return "\n".join(L + ["No seed could be scored."]) + "\n"
    t, r = float(np.mean([c["template_bal"] for c in ok])), float(np.mean([c["real_bal"] for c in ok]))
    fu = float(np.mean([c["real_false_unseen"] for c in ok if c["real_false_unseen"] is not None])) if any(c["real_false_unseen"] is not None for c in ok) else float("nan")
    uf = float(np.mean([c["real_unseen_found"] for c in ok if c["real_unseen_found"] is not None])) if any(c["real_unseen_found"] is not None for c in ok) else float("nan")
    L += ["| set | balanced accuracy |", "|---|---|", f"| templates (S4 to S8) | {t:.3f} |", f"| realism problems | {r:.3f} |", f"| gap (realism - templates), mean over seeds | {r - t:+.3f} |", "",
          f"On the realism problems: Solid skills called Unseen {fu:.3f}; Unseen skills found {uf:.3f}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-129")
    args = ap.parse_args()
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    cells = []
    for seed in R.seed_range(args.seeds):
        c = run_seed(seed, OUT, stamp)
        cells.append(c)
        print(f"seed {seed}: template {c['template_bal']} realism {c['real_bal']} diff {c['diff']}", flush=True)
    text = report_b7(cells)
    runlib.atomic_write_text(HERE / "results" / "B7_TABLE.md", text)
    print(text)

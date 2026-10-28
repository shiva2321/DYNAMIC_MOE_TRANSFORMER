"""Rewrite lab (option (a), wording diversity in teaching; user decision 2026-10-28). Tuning seeds 150 to 154; the plan, the predictions and the rule for going on are in PROGRESS.md
(2026-10-28), fixed before this was run. One small result file per (arm, seed), written atomically.

    PYTHONPATH=<repo root> python lab_rewrites.py [--seeds 150-154] [--arms "map only,full,ceiling"]

Arms (the reader is always the learned map on Qwen layer 26; the machine is the ordinary one):
  map only  generic rewrites of teaching sentences are extra positives for the learned map; prototypes and calibration are the originals, as before.
  full      the same, and the rewrites are also prototypes of their skill and the rewrites of the calibration parts are calibration parts. The taught COUNT (what makes a skill Solid or Thin)
            stays the number of ORIGINAL teaching examples: rewrites are not new evidence.
  ceiling   as ``full`` but the writer is told to produce stories, typos and registers: a CONTAMINATED DIAGNOSTIC. It names the held-out wordings, so its S4 / S7 / S8 numbers are not a claim
            of wording invariance; it only shows how much diversity could buy at best.
Every rewrite is kept only if it has the same numbers as its source AND the writer's own model, asked to solve it, gives the true answer of the teaching example (``rewrites.verify``).
Sources are teaching and calibration parts only; a rewrite equal to a test, adaptation, teaching or calibration part is dropped.
"""

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import assignments
import evaluate as EV
import learned_map
import reps
import rewrites as RW
import run_e2 as R
import runlib
import skills
import status as ST

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "rewrite_lab"
REWRITE_CACHE = HERE / "_cache" / "rewrites"
STAMP_FILES = R.STAMP_FILES + [HERE / "lab_rewrites.py", HERE / "rewrites.py", HERE / "function_check.py", R.LEARNED_MAP]
SEEDS = list(range(150, 155))
# The map's guard accepts only the teaching wordings. Rewrites are teaching-side data now (user decision 2026-10-28), so they carry a teaching label; the guard against S4-S8 labels is untouched.
REWRITE_FAMILY = "S3"
CEILING_LABEL = "CONTAMINATED DIAGNOSTIC: the writer is told to produce stories, typos and registers, which are the held-out wordings of the test bed; never a claim of wording invariance"
CEILING_INSTRUCTIONS = (
    "Rewrite the following request as a very short story problem that asks the same thing. Keep every number and the exact meaning. Reply with the rewritten request only.",
    "Rewrite the following with a few spelling mistakes and odd capital letters, as if typed in a hurry. Keep every number and the exact meaning. Reply with the rewritten request only.",
    "Rewrite the following as a very formal, polite request. Keep every number and the exact meaning. Reply with the rewritten request only.",
    "Rewrite the following as a short, casual text message. Keep every number and the exact meaning. Reply with the rewritten request only.",
)
ARMS: Dict[str, Tuple[str, bool]] = {"map only": ("generic", False), "full": ("generic", True), "ceiling": ("ceiling", True)}
StatusMachine = ST.StatusMachine


@dataclass
class RewriteWorld(R.World):
    extra_texts: List[str] = field(default_factory=list)

    def all_texts(self) -> List[str]:
        return list(dict.fromkeys(super().all_texts() + self.extra_texts))


def cal_examples(world: R.World) -> Dict[str, List[assignments.Example]]:
    """The calibration examples (with their numbers and answers); ``build_world`` keeps only their texts, so the draws are repeated with the same streams (deterministic) and checked."""
    used = {R.fingerprint(e.text) for s in world.teach for e in world.teach[s]}
    out: Dict[str, List[assignments.Example]] = {}
    for s in world.taught:
        ex = R._fresh_taught_wording(R._stream(world.seed, 4, skills.SKILLS.index(s)), world.pool, s, world.sizes.cal, used)
        if [e.span for e in ex] != world.cal[s]:
            raise RuntimeError(f"the calibration draw of {s} could not be reproduced")
        out[s] = ex
    return out


def sources_and_answers(world: R.World, n_src: int = 150, n_cal_src: int = 60) -> Dict[str, Dict[str, Tuple[str, int]]]:
    """{"teach": {text: (skill, true answer)}, "cal": {...}}: up to ``n_src`` distinct teaching parts and ``n_cal_src`` calibration parts per taught skill (a Thin skill gives all it has)."""
    out: Dict[str, Dict[str, Tuple[str, int]]] = {"teach": {}, "cal": {}}
    cal = cal_examples(world)
    for s in world.taught:
        for key, ex, cap in (("teach", world.teach[s], n_src), ("cal", cal[s], n_cal_src)):
            uniq = list({e.span: e for e in ex}.values())
            order = np.random.default_rng([world.seed, 0x5C, skills.SKILLS.index(s), len(key)]).permutation(len(uniq))
            for i in order[:cap]:
                out[key][uniq[int(i)].span] = (s, int(uniq[int(i)].answer))
    return out


def drop_known(rewrites: Dict[str, List[str]], world: R.World) -> Dict[str, List[str]]:
    """Drop a rewrite that equals (case and spacing ignored) any part the world already holds: test, adaptation, teaching or calibration."""
    held = {R.fingerprint(t) for t in world.test_texts}
    for texts in list(world.adapt.values()) + list(world.cal.values()):
        held |= {R.fingerprint(t) for t in texts}
    held |= {R.fingerprint(e.span) for s in world.teach for e in world.teach[s]}
    return {src: [r for r in rs if R.fingerprint(r) not in held] for src, rs in rewrites.items()}


def fit_map_with_rewrites(taught: Sequence[str], seed: int, X: np.ndarray, skill: Sequence[str], family: Sequence[str], template: Sequence[str], Xr: np.ndarray, skill_r: Sequence[str],
                          epochs: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    """The learned map fitted on the originals AND the rewrites (each rewrite: the skill of its source, a teaching label, a template of its own so that it is a positive of its source)."""
    params = dict(learned_map.HYPER)
    if epochs is not None:
        params["epochs"] = epochs
    m = learned_map.LearnedMap(taught, seed=seed, **params)
    m.fit(np.vstack([X, Xr]) if len(Xr) else X, [*skill, *skill_r], [*family, *([REWRITE_FAMILY] * len(skill_r))], [*template, *[f"rw{k}" for k in range(len(skill_r))]])
    return m.transform


def run_arm(arm: str, seed: int, out_dir, stamp: str, sizes: R.Sizes = R.SIZES, rep: Optional[reps.Representation] = None, competence_fn: Optional[Callable] = None,
            writer: Optional[Callable] = None, solver: Optional[Callable] = None, n_src: int = 150, n_cal_src: int = 60, cache_dir=R.CACHE, rewrite_cache=REWRITE_CACHE) -> dict:
    path = Path(out_dir) / f"{arm.replace(' ', '_')}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    which, in_machine = ARMS[arm]
    instructions = RW.INSTRUCTIONS if which == "generic" else CEILING_INSTRUCTIONS
    world = R.build_world(seed, "dev", sizes, competence_fn)
    src = sources_and_answers(world, n_src, n_cal_src)
    writer = writer or RW.make_qwen_generator()
    solver = solver or RW.make_qwen_solver()
    rw: Dict[str, Dict[str, List[str]]] = {}
    raw_counts = {}
    for key in ("teach", "cal"):
        raw = RW.rewrites_for(list(src[key]), instructions, writer, rewrite_cache, seed=seed, tag=f"{which}-{key}")
        ok = RW.verify(raw, {t: a for t, (_, a) in src[key].items()}, solver, rewrite_cache, seed=seed, tag=f"{which}-{key}-solve")
        rw[key] = drop_known(ok, world)
        raw_counts[key] = (sum(len(v) for v in raw.values()), sum(len(v) for v in rw[key].values()))
    by_skill = {key: {s: [r for t, rs in rw[key].items() if src[key][t][0] == s for r in rs] for s in world.taught} for key in ("teach", "cal")}
    extra = [r for key in ("teach", "cal") for s in world.taught for r in by_skill[key][s]]
    rw_world = RewriteWorld(**{f.name: getattr(world, f.name) for f in fields(R.World)}, extra_texts=extra)
    teach_r = [(r, s) for s in world.taught for r in by_skill["teach"][s]]

    def mapper(X, items, taught):
        Xr = (reps.cached_encode(rep_obj, [t for t, _ in teach_r], cache_dir) if cache_dir is not None else rep_obj.encode([t for t, _ in teach_r])) if teach_r else np.zeros((0, X.shape[1]), np.float32)
        return fit_map_with_rewrites(taught, seed, X, [i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items], Xr, [s for _, s in teach_r])

    rep_obj = rep or reps.get_rep("R4_qwen_L26")
    P, index = R.prepare(rw_world, "R5_R4_qwen_L26", rep_obj, cache_dir, mapper)

    def V(texts):
        return P[[index[t] for t in texts]]

    groups = {}
    for s in world.taught:
        texts = [e.span for e in world.teach[s]] + (by_skill["teach"][s] if in_machine else [])
        groups[s] = ST.SkillGroup(s, V(texts), len(world.teach[s]), None, world.comp[s]["accuracy"])
    other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
    cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
    if in_machine:
        extra_cal = [(r, s) for s in world.taught for r in by_skill["cal"][s]]
        if extra_cal:
            cal_vec = np.vstack([cal_vec, V([r for r, _ in extra_cal])])
            cal_grp = np.concatenate([cal_grp, np.array([s for _, s in extra_cal])])
    m = StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
    assessed = m.assess_many(V(world.test_texts))
    s = EV.status_scores(assessed, world.test_cases)
    rec = {"arm": arm, "seed": seed, "code_stamp": stamp, "false_unseen": s["false_unseen_heldout"], "taught_wording": s["false_unseen_taught_wording"], "found": s["unseen_recall"],
           "macro_f1": s["macro_f1"]["macro_f1"], "thin_vs_solid": s["thin_vs_solid_auroc"], "other_as_unseen": s["other_as_unseen"], "by_family": s["false_unseen_by_family"],
           "n_rewrites_teach": raw_counts["teach"][1], "n_rewrites_cal": raw_counts["cal"][1], "n_sources_teach": len(src["teach"]),
           "yield_after_number_filter_and_answer_check": {k: raw_counts[k] for k in raw_counts}, "label": CEILING_LABEL if which == "ceiling" else "generic rewrites"}
    rec = json.loads(json.dumps(rec))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def report(arms: Sequence[str], seeds: Sequence[int], out_dir, stamp: str, base_rows: List[dict]) -> str:
    mean = lambda rows, k: float(np.nanmean([r[k] for r in rows]))
    fam = lambda rows, f: float(np.nanmean([r["by_family"].get(f, np.nan) for r in rows]))
    L = [f"# Rewrite lab (tuning seeds {seeds[0]}-{seeds[-1]}, learned-map reader on Qwen layer 26, status machine q = 0.05)", "",
         "| arm | known skills in new wordings called unseen (bar 0.05) | stories S4 | noisy text S7 | register S8 | never-taught skills found | macro F1 | rewrites kept (teaching) |", "|---|---|---|---|---|---|---|---|",
         f"| nothing (the machine as it is) | {mean(base_rows, 'false_unseen'):.3f} | {fam(base_rows, 'S4'):.3f} | {fam(base_rows, 'S7'):.3f} | {fam(base_rows, 'S8'):.3f} | {mean(base_rows, 'found'):.3f} | {mean(base_rows, 'macro_f1'):.3f} | 0 |"]
    for arm in arms:
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in [Path(out_dir) / f"{arm.replace(' ', '_')}__s{s}.json" for s in seeds] if runlib.cell_done(p, stamp)]
        if len(rows) != len(seeds):
            continue
        name = arm + (" (CONTAMINATED DIAGNOSTIC)" if arm == "ceiling" else "")
        L.append(f"| {name} | {mean(rows, 'false_unseen'):.3f} | {fam(rows, 'S4'):.3f} | {fam(rows, 'S7'):.3f} | {fam(rows, 'S8'):.3f} | {mean(rows, 'found'):.3f} | {mean(rows, 'macro_f1'):.3f} | {mean(rows, 'n_rewrites_teach'):.0f} per seed |")
    L += ["", "The ceiling arm tells the writer to produce stories, typos and registers, so its numbers on those wordings are not a wording-invariance claim."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="150-154")
    ap.add_argument("--arms", default="map only,full,ceiling")
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds = R.seed_range(args.seeds)
    stamp = runlib.code_stamp(STAMP_FILES)
    arms = args.arms.split(",")
    for arm in arms:
        for seed in seeds:
            r = run_arm(arm, seed, OUT, stamp)
            print(f"{arm:8s} seed {seed}: false alarm {r['false_unseen']:.3f}  found {r['found']:.3f}  macro F1 {r['macro_f1']:.3f}  rewrites kept {r['n_rewrites_teach']} of {r['yield_after_number_filter_and_answer_check']['teach'][0]}", flush=True)
        reps.clear_models()
    import lab_wording as LAB
    lstamp = runlib.code_stamp(LAB.STAMP_FILES)
    base = [LAB.run_variant("plain", "as is", s, LAB.OUT, lstamp) for s in seeds]
    text = report(arms, seeds, OUT, stamp, base)
    runlib.atomic_write_text(HERE / "results" / "REWRITE_LAB.md", text)
    print(text)

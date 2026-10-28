"""Mitigations lab for the wording shift (Task 11 follow-up). Tuning seeds 150 to 154 only; the plan, the predictions and the rule for choosing a winner are in PROGRESS.md (2026-10-28), fixed
before this was run. One small result file per (variant, seed), written atomically, so the whole command can be run again after a power cut.

    PYTHONPATH=<repo root> python lab_wording.py --stage 1
    PYTHONPATH=<repo root> python lab_wording.py --stage 2 --reader <winner of stage 1>

Stage 1 changes the READER (plain, clean-up, two closing prompts, clean-up with each prompt) with the status machine as it is. Stage 2 changes the MACHINE (as it is, leave-one-wording-out
calibration, a margin gate at 0.10 / 0.25 / 0.50, both) on the stage-1 winner, and, for comparison only, on the plain reader. The learned map is always on (the headline reader).
Winner rule: the lowest false alarm on S4-S8 among variants that find at least 0.80 of the never-taught skills and keep macro F1 at 0.80 or more; within 0.01 the simpler variant wins.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import evaluate as EV
import reps
import run_e2 as R
import runlib
import skills
import status as ST
import wording as W

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "wording_lab"
STAMP_FILES = R.STAMP_FILES + [HERE / "lab_wording.py", HERE / "wording.py", R.LEARNED_MAP]
SEEDS = list(range(150, 155))
MIN_FOUND, MIN_F1, TIE = 0.80, 0.80, 0.01
P1 = "\nIgnoring any story or extra words, the arithmetic operation asked for is"
P2 = "\nIn one word, the arithmetic operation that this asks for is"
QWEN, LAYER = "Qwen/Qwen2.5-1.5B-Instruct", 26


def reader_variants() -> Dict[str, Callable[[], reps.Representation]]:
    """name -> function building the representation; the order is the order of simplicity (the tie rule)."""
    base = lambda: reps.get_rep("R4_qwen_L26")
    p1 = lambda: W.PromptedRep("R4_qwen_L26_p1", QWEN, LAYER, True, P1)
    p2 = lambda: W.PromptedRep("R4_qwen_L26_p2", QWEN, LAYER, True, P2)
    return {"plain": base, "clean": lambda: W.CleanRep(base()), "p1": p1, "p2": p2, "clean+p1": lambda: W.CleanRep(p1()), "clean+p2": lambda: W.CleanRep(p2())}


MACHINE_VARIANTS: Dict[str, dict] = {"as is": {}, "lofo": dict(lofo=True), "margin 0.10": dict(margin_q=0.10), "margin 0.25": dict(margin_q=0.25), "margin 0.50": dict(margin_q=0.50),
                                     "lofo + margin 0.25": dict(lofo=True, margin_q=0.25)}


def cal_family_labels(world: R.World) -> Dict[str, np.ndarray]:
    """The wording (S1, S2 or S3) of every calibration part. ``build_world`` keeps only the texts, so the draw is repeated with the same streams (it is deterministic) and checked against them."""
    used = {R.fingerprint(e.text) for s in world.teach for e in world.teach[s]}
    out: Dict[str, np.ndarray] = {}
    for s in world.taught:
        ex = R._fresh_taught_wording(R._stream(world.seed, 4, skills.SKILLS.index(s)), world.pool, s, world.sizes.cal, used)
        if [e.span for e in ex] != world.cal[s]:
            raise RuntimeError(f"the calibration draw of {s} could not be reproduced")
        out[s] = np.array([e.family for e in ex])
    return out


def family_arrays(world: R.World) -> dict:
    """The wording labels the leave-one-wording-out calibration needs: per taught group one per prototype, and one per calibration vector in the order of ``calibration_arrays``."""
    cal = cal_family_labels(world)
    names = list(world.taught) + ["other"]
    return {"proto": {s: np.array([e.family for e in world.teach[s]]) for s in world.taught}, "cal": np.concatenate([cal[n] if n in cal else np.array(["other"] * len(world.cal[n])) for n in names])}


def choose_winner(rows: Sequence[dict]) -> dict:
    """Each row: name, false_unseen, found, macro_f1, rank (0 is the simplest). See the module docstring for the rule; if no variant qualifies, the simplest one (no change) is returned."""
    if not rows:
        raise ValueError("no variants to choose from")
    ok = [r for r in rows if r["found"] >= MIN_FOUND - 1e-9 and r["macro_f1"] >= MIN_F1 - 1e-9]
    if not ok:
        return min(rows, key=lambda r: r["rank"])
    best = min(r["false_unseen"] for r in ok)
    return min((r for r in ok if r["false_unseen"] <= best + TIE + 1e-12), key=lambda r: r["rank"])


def evaluate_variant(world: R.World, P: np.ndarray, index: Dict[str, int], seed: int, machine_kw: dict) -> dict:
    def V(texts):
        return P[[index[t] for t in texts]]

    groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
    other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
    cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
    if machine_kw:
        fam = family_arrays(world)
        m = W.WordingMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed, proto_families=fam["proto"], cal_families=fam["cal"], **machine_kw)
    else:
        m = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
    assessed = m.assess_many(V(world.test_texts))
    s = EV.status_scores(assessed, world.test_cases)
    return {"false_unseen": s["false_unseen_heldout"], "taught_wording": s["false_unseen_taught_wording"], "found": s["unseen_recall"], "macro_f1": s["macro_f1"]["macro_f1"],
            "thin_vs_solid": s["thin_vs_solid_auroc"], "other_as_unseen": s["other_as_unseen"], "by_family": s["false_unseen_by_family"], "other_labelled": s["other_labelled"],
            "twin_unseen_recall": s["twin_unseen_recall"], "overridden": int(getattr(m, "last_overridden", np.zeros(0)).sum())}


def run_variant(reader_name: str, machine_name: str, seed: int, out_dir, stamp: str, rep: Optional[reps.Representation] = None) -> dict:
    path = Path(out_dir) / f"{reader_name}__{machine_name.replace(' ', '_').replace('+', 'and')}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, "dev", R.SIZES)
    rep = rep or reader_variants()[reader_name]()
    P, index = R.prepare(world, "R5_R4_qwen_L26", rep, R.CACHE)
    rec = evaluate_variant(world, P, index, seed, MACHINE_VARIANTS[machine_name])
    rec.update(reader=reader_name, machine=machine_name, seed=seed, code_stamp=stamp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def table(rows: List[dict], title: str, key: str) -> str:
    L = [f"### {title}", "", f"| {key} | false alarm S4-S8 | stories S4 | noisy text S7 | swapped order S5 | taught wordings | never-taught skills found | macro F1 | Other called unseen | passed by the margin gate |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bf = r["by_family"]
        L.append(f"| {r['name']} | {r['false_unseen']:.3f} | {bf.get('S4', float('nan')):.3f} | {bf.get('S7', float('nan')):.3f} | {bf.get('S5', float('nan')):.3f} | {r['taught_wording']:.3f} | "
                 f"{r['found']:.3f} | {r['macro_f1']:.3f} | {r['other_as_unseen']:.3f} | {r['overridden']:.0f} |")
    return "\n".join(L) + "\n"


def summarise(reader_names: Sequence[str], machine_names: Sequence[str], out_dir, stamp: str, by: str) -> List[dict]:
    rows = []
    for rank, (rn, mn) in enumerate((r, m) for r in reader_names for m in machine_names):
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in [Path(out_dir) / f"{rn}__{mn.replace(' ', '_').replace('+', 'and')}__s{s}.json" for s in SEEDS] if runlib.cell_done(p, stamp)]
        if len(recs) != len(SEEDS):
            continue
        mean = lambda k: float(np.mean([r[k] for r in recs]))
        fams = sorted({f for r in recs for f in r["by_family"]})
        rows.append({"name": rn if by == "reader" else mn, "reader": rn, "machine": mn, "rank": rank, **{k: mean(k) for k in ("false_unseen", "taught_wording", "found", "macro_f1", "other_as_unseen", "overridden")},
                     "by_family": {f: float(np.nanmean([r["by_family"].get(f, np.nan) for r in recs])) for f in fams}})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--reader", default="plain")
    args = ap.parse_args()
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    if args.stage == 1:
        names = list(reader_variants())
        for rn in names:
            for seed in SEEDS:
                r = run_variant(rn, "as is", seed, OUT, stamp)
                print(f"{rn:9s} seed {seed}: false alarm {r['false_unseen']:.3f}  found {r['found']:.3f}  macro F1 {r['macro_f1']:.3f}", flush=True)
            reps.clear_models()
        rows = summarise(names, ["as is"], OUT, stamp, "reader")
        win = choose_winner(rows)
        text = "# Mitigations lab, stage 1 (reader variants, tuning seeds 150-154)\n\n" + table(rows, "Reader variants, machine as it is", "reader") + f"\n**Winner by the declared rule: `{win['name']}`**\n"
        runlib.atomic_write_text(HERE / "results" / "WORDING_LAB_STAGE1.md", text)
    else:
        readers = [args.reader] + ([] if args.reader == "plain" else ["plain"])
        for rn in readers:
            for mn in MACHINE_VARIANTS:
                for seed in SEEDS:
                    run_variant(rn, mn, seed, OUT, stamp)
            reps.clear_models()
        text = "# Mitigations lab, stage 2 (machine variants, tuning seeds 150-154)\n\n"
        for rn in readers:
            rows = summarise([rn], list(MACHINE_VARIANTS), OUT, stamp, "machine")
            text += table(rows, f"Machine variants on the `{rn}` reader" + (" (the stage-1 winner; the choice is made here)" if rn == args.reader else " (for comparison only)"), "machine")
            if rn == args.reader:
                text += f"\n**Winner by the declared rule: `{choose_winner(rows)['name']}`**\n\n"
        runlib.atomic_write_text(HERE / "results" / "WORDING_LAB_STAGE2.md", text)
    print(text)

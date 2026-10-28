"""The chosen wording design (reader and machine from the mitigations lab) on the report seeds: the ordinary E2 runner with the chosen reader and machine swapped in. Run ONCE, after the
lab has chosen on the tuning seeds and written ``results/WORDING_CHOICE.json``; results go next to the ordinary ones as ``R5_R4_qwen_L26+wording__s<seed>.json`` and a comparison table.

    PYTHONPATH=<repo root> python run_e2w.py --seeds 100-129

The ordinary cells (``R5_R4_qwen_L26``) are not touched. With a leave-one-wording-out machine the ADAPTED condition equals the standard one (the few S4-S8 parts of the adapted calibration
are not taught wordings and are left out of the leave-one-out scores); the table says so.
"""

import argparse
import json
from pathlib import Path
from unittest import mock

import numpy as np

import lab_wording as LAB
import reps
import run_e2 as R
import runlib
import wording as W

HERE = Path(__file__).resolve().parent
CHOICE = HERE / "results" / "WORDING_CHOICE.json"
REP_ID = "R5_R4_qwen_L26+wording"
MACHINE_OPTIONS = {"lofo", "margin_q"}


def load_choice(path=CHOICE) -> dict:
    choice = json.loads(Path(path).read_text(encoding="utf-8"))
    if choice.get("reader") not in LAB.reader_variants():
        raise ValueError(f"unknown reader {choice.get('reader')!r}; the readers are {list(LAB.reader_variants())}")
    if not isinstance(choice.get("machine"), dict) or set(choice["machine"]) - MACHINE_OPTIONS:
        raise ValueError(f"machine options must be a dict with keys among {sorted(MACHINE_OPTIONS)}, got {choice.get('machine')!r}")
    if not isinstance(choice.get("machine_name"), str):
        raise ValueError("machine_name must be a string")
    return choice


def stamp_for(choice_path=CHOICE) -> str:
    return runlib.code_stamp(R.STAMP_FILES + [R.LEARNED_MAP, HERE / "wording.py", HERE / "lab_wording.py", HERE / "run_e2w.py", Path(choice_path)])


def run_cell(choice: dict, seed: int, pool: str, out_dir, stamp: str, sizes: R.Sizes = R.SIZES, rep=None, competence_fn=None, cache_dir=R.CACHE, machine_cls=W.WordingMachine) -> dict:
    path = Path(out_dir) / f"{REP_ID}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    fam = LAB.family_arrays(R.build_world(seed, pool, sizes, competence_fn)) if choice["machine"].get("lofo") else None
    rep = rep or LAB.reader_variants()[choice["reader"]]()

    def build(groups, other, cal_vectors, cal_groups, **kw):
        extra = {}
        if fam is not None:
            cal_fam = fam["cal"]
            if len(cal_groups) > len(cal_fam):                                  # the adapted condition adds S4 to S8 parts: not taught wordings, labelled so and left out
                cal_fam = np.concatenate([cal_fam, np.array(["adapt"] * (len(cal_groups) - len(cal_fam)))])
            extra = dict(proto_families=fam["proto"], cal_families=cal_fam)
        return machine_cls(groups, other, cal_vectors, cal_groups, **extra, **choice["machine"], **kw)

    with mock.patch.object(R.ST, "StatusMachine", build):
        return R.run_cell(REP_ID, seed, pool, out_dir, stamp, sizes, rep=rep, competence_fn=competence_fn, cache_dir=cache_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-129")
    ap.add_argument("--pool", default="dev")
    args = ap.parse_args()
    runlib.set_threads(3)
    choice = load_choice()
    seeds = R.seed_range(args.seeds)
    stamp = stamp_for()
    for seed in seeds:
        r = run_cell(choice, seed, args.pool, R.OUT, stamp)
        s = r["machine"]["q0.05"]["standard"]
        print(f"{REP_ID} seed {seed}: macro F1 {s['macro_f1']['macro_f1']:.3f}  false-unseen(S4-S8) {s['false_unseen_heldout']:.3f}  unseen found {s['unseen_recall']:.3f}  {r['secs']}s", flush=True)
    reps.clear_models()
    old = "R5_R4_qwen_L26"
    text = R.report_e2(R.OUT, [REP_ID, old], seeds, lambda rid: stamp if rid == REP_ID else R.stamp_for(rid), args.pool)
    note = (f"\n> The design under test is `{REP_ID}`: reader `{choice['reader']}`, machine `{choice['machine_name']}` {choice['machine']}; chosen on the tuning seeds 150-154 by the rule in PROGRESS.md. "
            "With leave-one-wording-out calibration the adapted condition equals the standard one. The bars table below is computed from the first reader.\n")
    runlib.atomic_write_text(HERE / "results" / "E2W_TABLE.md", note + text)
    print(note + text)

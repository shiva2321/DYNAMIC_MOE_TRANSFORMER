"""The wording-confound test (follow-up design, addendum): does a new skill learned from stream-wording examples take over the old skills' items in those wordings?

A closed-form classifier (the LDA head: fixed, no training schedule, no pool) is fitted on the teaching rows of the taught skills plus ``m`` examples of every withheld skill, and read on the taught skills in HELD-OUT
wordings (the rows no fit used). Only the WORDING of the new skills' examples changes between conditions:
* none     no new skill (the reference);
* tw       the new skills' examples are in teaching wordings;
* ho       in held-out wordings;
* mixed    in all wordings;
* balanced mixed, plus ``m`` examples of every taught skill in held-out wordings (what verifying known items in the stream wordings would give the old skills).
``python -m fsmoe.confound --seeds 710 ...`` writes results/fsmoe/L/CONFOUND.md.
"""

import argparse
from typing import Sequence

import numpy as np
import torch

from unimoe.heads import LdaHead

from .bed import ROOT, build_bed
from .detectors import auroc  # noqa: F401  (kept for symmetry with the other labs)
from .report_s import fmt
from .system_lab import Prep

OUT = ROOT / "results" / "fsmoe" / "L"
CONDITIONS = ("none", "tw", "ho", "mixed", "balanced")


def run_seed(seed: int, m: int) -> dict:
    prep = Prep(build_bed(seed))
    bed = prep.bed
    rng = np.random.default_rng([seed, 0x434F4E, m])
    teach = np.flatnonzero(bed.teaching())
    test_nov = prep.rows("test", None, "novel")
    test_ho = prep.rows("test", None, "known_ho")
    ev_ho, ev_nov = prep.rows("cal", None, "known_ho"), prep.rows("cal", None, "novel")

    def pick(rows: np.ndarray, per_class: int) -> np.ndarray:
        out = []
        for c in np.unique(prep.y[rows].numpy()):
            r = rows[prep.y[rows].numpy() == c]
            out.append(rng.permutation(r)[:per_class])
        return np.concatenate(out) if out else np.zeros(0, dtype=int)

    held = bed.heldout[test_nov]
    sets = {"none": np.zeros(0, dtype=int), "tw": pick(test_nov[~held], m), "ho": pick(test_nov[held], m), "mixed": pick(test_nov, m)}
    sets["balanced"] = np.concatenate([sets["mixed"], pick(test_ho, m)])
    out = {}
    for name in CONDITIONS:
        idx = np.concatenate([teach, sets[name]])
        head = LdaHead(prep.d, prep.n_classes, 0.1)
        head.update(prep.X[idx], prep.y[idx])
        pho = head.proba(prep.X[ev_ho]).argmax(1)
        pnov = head.proba(prep.X[ev_nov]).argmax(1)
        withheld_ids = {bed.skills.index(s) for s in bed.withheld}
        stolen = float(np.mean([int(c) in withheld_ids for c, ok in zip(pho.tolist(), (pho == prep.y[ev_ho]).tolist()) if not ok])) if bool((pho != prep.y[ev_ho]).any()) else 0.0
        out[name] = {"known_ho": float((pho == prep.y[ev_ho]).float().mean()), "novel": float((pnov == prep.y[ev_nov]).float().mean()), "errors_to_new_skills": stolen}
    return out


def build(seeds: Sequence[int], ms=(10, 20)) -> str:
    L = ["# Wording-confound test (closed-form classifier, no pool)", "",
         "A classifier fitted on the taught skills' teaching rows plus m examples of each of the 28 withheld skills; read on taught skills in held-out wordings (known ho) and on the withheld skills (novel); "
         "`errors to new skills` = share of the known-ho errors that land on a withheld skill. Mean [95 % interval over seeds].", ""]
    for m in ms:
        runs = [run_seed(s, m) for s in seeds]
        L += [f"## m = {m} examples per new skill ({len(runs)} seeds)", "", "| new skills' examples | known ho | novel | known-ho errors that land on a new skill |", "|---|---|---|---|"]
        for c in CONDITIONS:
            L.append(f"| {c} | {fmt([r[c]['known_ho'] for r in runs])} | {fmt([r[c]['novel'] for r in runs])} | {fmt([r[c]['errors_to_new_skills'] for r in runs], 2)} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CONFOUND.md").write_text(text, encoding="utf-8")
    print(text)

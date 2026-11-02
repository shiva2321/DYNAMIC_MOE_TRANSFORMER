"""Does a character-aware front end help the look-based detectors with known skills in new wordings (E-064)?

    python char_front_check.py --seeds 150-154

The nuance lab (run_e2.py) reads each part with one representation. Here the same world, the same teaching, calibration and test parts and the same detectors (vMF conformal, 1-NN, Mahalanobis,
the settings of BASELINE_PARAMS, calibrated at q = 0.05 on held-out S1 to S3 parts) are run on four readers: R1_minilm384 (a sentence-meaning model), R0_hash (word and letter-group counts: it sees
spelling, not meaning), and two concatenations [R1, w x R0] with w = 0.5 and 1.0 (both parts centred on the taught examples and unit length first). Nothing is tuned on the test parts and no wording
is added to the teaching. One JSON per seed in results/char_front, written atomically. Seeds 150-154 are the tuning seeds of the lab."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines as BL                    # noqa: E402
import evaluate as EV                     # noqa: E402
import reps                               # noqa: E402
import run_e2 as R                        # noqa: E402
import runlib                             # noqa: E402

OUT = HERE / "results" / "char_front"
READERS = {"R1": (("R1_minilm384",), (1.0,)), "R0": (("R0_hash",), (1.0,)), "R1+0.5R0": (("R1_minilm384", "R0_hash"), (1.0, 0.5)), "R1+1R0": (("R1_minilm384", "R0_hash"), (1.0, 1.0))}
DETECTORS = ("vmf_conformal", "knn_distance", "mahalanobis")
Q = 0.05


def vectors(world, rep_ids, weights):
    texts = world.all_texts()
    index = {t: k for k, t in enumerate(texts)}
    parts = []
    for rid, w in zip(rep_ids, weights):
        X = reps.cached_encode(reps.get_rep(rid), texts, R.CACHE)
        parts.append(w * R._reader(world, X, index, world.seed, False, None))
    P = np.hstack(parts)
    return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12), index


def one_seed(seed: int) -> dict:
    path = OUT / f"s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, "dev")
    cases = world.test_cases
    rec = {"seed": seed, "assignment": world.assignment, "readers": {}}
    for name, (rep_ids, weights) in READERS.items():
        P, index = vectors(world, rep_ids, weights)

        def V(texts):
            return P[[index[t] for t in texts]]
        X_fit, y_fit = R.fit_arrays(world, V)
        cal_vec, _, _, _ = R.calibration_arrays(world, V)
        tvec = V(world.test_texts)
        rec["readers"][name] = {}
        for det_name in DETECTORS:
            det = BL.make_baseline(det_name, background=V(world.background), **R.BASELINE_PARAMS.get(det_name, {})).fit(X_fit, y_fit)
            thr = BL.conformal_threshold(det.novelty_score(cal_vec), Q)
            s = det.novelty_score(tvec)
            rec["readers"][name][det_name] = EV.flag_scores(s > thr, cases, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def table(seeds) -> str:
    recs = [one_seed(s) for s in seeds]
    L = [f"# Character-aware front end: known skills in new wordings called unseen, at q = {Q} (seeds {seeds[0]}-{seeds[-1]}; bar 0.05)", "",
         "| reader | detector | false unseen S4-S8 | S4 stories | S5 | S6 | S7 noisy text | S8 | unseen skills found | AUROC taught/untaught | Other content called unseen |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in READERS:
        for det in DETECTORS:
            def m(f):
                v = [f(r["readers"][name][det]) for r in recs]
                return float(np.nanmean(v))
            fam = lambda k: m(lambda d: d["false_unseen_by_family"].get(k, float("nan")))
            L.append(f"| {name} | {det} | {m(lambda d: d['false_unseen_heldout']):.3f} | " + " | ".join(f"{fam(k):.3f}" for k in ("S4", "S5", "S6", "S7", "S8")) +
                     f" | {m(lambda d: d['unseen_recall']):.3f} | {m(lambda d: d['auroc_taught_vs_untaught']):.3f} | {m(lambda d: d['other_as_unseen']):.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="150-154")
    a = ap.parse_args()
    seeds = R.seed_range(a.seeds)
    for s in seeds:
        one_seed(s)
        print(f"seed {s} done", flush=True)
    text = table(seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(OUT / f"TABLE_{seeds[0]}_{seeds[-1]}.md", text)
    print(text)

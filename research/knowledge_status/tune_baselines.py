"""Tuning of the standard detectors (the rule of this project: every arm is tuned, and its operating point is reported). One setting is chosen per detector on seeds that are NOT report
seeds, by the highest mean AUROC of "a part of an untaught skill is newer than a part of a taught skill" (no threshold involved), so the comparison with the status machine is not
a tuned machine against untuned baselines. One small result file per (reader, detector, setting, seed), written atomically, so the whole command can be run again after a power cut.

    python tune_baselines.py --reader R5_R4_qwen_L26 --seeds 150-154

The status machine has no setting to tune: q is fixed (0.05, with the curve shown) and the kernel width is chosen by the signature's own leave-one-out likelihood.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import baselines as BL
import evaluate as EV
import reps
import run_e2 as R
import runlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "baseline_tuning"
STAMP_FILES = R.STAMP_FILES + [HERE / "tune_baselines.py"]
REPORT_SEEDS = range(100, 130)
TIE = 0.002                                                       # a setting within this AUROC of the best loses to the default

GRID: Dict[str, List[dict]] = {
    "knn_distance": [{"k": k} for k in (1, 5, 10, 20, 50)],
    "vmf_conformal": [{"max_protos": m} for m in (64, 256, 1024)],
    "mahalanobis": [{}],
    "relative_mahalanobis": [{}],
    "classifier_confidence": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "energy": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "autoencoder_descriptor": [{"hidden": h, "epochs": e} for h in (8, 16, 32, 64) for e in (100, 300)],
}
DEFAULTS: Dict[str, dict] = {"knn_distance": {"k": 10}, "vmf_conformal": {"max_protos": 256}, "classifier_confidence": {"C": 1.0}, "energy": {"C": 1.0},
                             "autoencoder_descriptor": {"hidden": 32, "epochs": 300}}


def setting_name(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items())) or "default"


def stamp_for(reader: str) -> str:
    return runlib.code_stamp(STAMP_FILES + ([R.LEARNED_MAP] if reader.startswith("R5_") else []))


def _path(out_dir, reader: str, name: str, params: dict, seed: int) -> Path:
    return Path(out_dir) / f"{reader}__{name}__{setting_name(params)}__s{seed}.json"


def run_tuning(reader: str, seeds: Sequence[int], pool: str, grid: Dict[str, List[dict]], sizes: R.Sizes, out_dir, rep: Optional[reps.Representation] = None,
               competence_fn: Optional[Callable] = None, cache_dir=R.CACHE, mapper: Optional[Callable] = None) -> Dict[str, List[dict]]:
    bad = [s for s in seeds if s in REPORT_SEEDS or s >= 400]
    if bad:
        raise ValueError(f"seeds {bad} are report seeds (100-129) or confirmatory seeds (400 and above); tune on other dev seeds such as 150-154")
    stamp = stamp_for(reader)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        pending = [(n, p) for n, ps in grid.items() for p in ps if not runlib.cell_done(_path(out_dir, reader, n, p, seed), stamp)]
        if not pending:
            continue
        world = R.build_world(seed, pool, sizes, competence_fn)
        P, index = R.prepare(world, reader, rep, cache_dir, mapper)

        def V(texts):
            return P[[index[t] for t in texts]]

        X_fit, y_fit = R.fit_arrays(world, V)
        cal_vec = R.calibration_arrays(world, V)[0]
        background, tvec = V(world.background), V(world.test_texts)
        for name, params in pending:
            det = BL.make_baseline(name, background=background, **params).fit(X_fit, y_fit)
            s_test = det.novelty_score(tvec)
            thr = BL.conformal_threshold(det.novelty_score(cal_vec), R.HEADLINE_Q)
            r = EV.flag_scores(s_test > thr, world.test_cases, s_test)
            rec = {"reader": reader, "name": name, "params": params, "seed": seed, "code_stamp": stamp, **{k: r[k] for k in ("auroc_taught_vs_untaught", "unseen_recall", "false_unseen_heldout", "other_as_unseen")}}
            runlib.atomic_write_text(_path(out_dir, reader, name, params, seed), json.dumps(rec))
            print(f"{reader} seed {seed} {name:24s} {setting_name(params):22s} AUROC {rec['auroc_taught_vs_untaught']:.3f}  found {rec['unseen_recall']:.3f}  false-unseen {rec['false_unseen_heldout']:.3f}", flush=True)
    return summarise(reader, seeds, grid, out_dir)


def summarise(reader: str, seeds: Sequence[int], grid: Dict[str, List[dict]], out_dir) -> Dict[str, List[dict]]:
    stamp = stamp_for(reader)
    out: Dict[str, List[dict]] = {}
    for name, settings in grid.items():
        rows = []
        for params in settings:
            recs = [json.loads(_path(out_dir, reader, name, params, s).read_text(encoding="utf-8")) for s in seeds if runlib.cell_done(_path(out_dir, reader, name, params, s), stamp)]
            mean = lambda k: float(np.nanmean([r[k] for r in recs])) if recs else float("nan")
            rows.append({"setting": setting_name(params), "params": params, "n_seeds": len(recs), "mean_auroc": mean("auroc_taught_vs_untaught"), "mean_recall": mean("unseen_recall"),
                         "mean_false_unseen": mean("false_unseen_heldout"), "is_default": params == DEFAULTS.get(name, {})})
        out[name] = rows
    return out


def choose(summary: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Per detector: the parameters to override (an empty dict = keep the default). The setting with the highest mean AUROC wins; one within ``TIE`` of the best loses to the default."""
    chosen: Dict[str, dict] = {}
    for name, rows in summary.items():
        rows = [r for r in rows if r["n_seeds"] and not np.isnan(r["mean_auroc"])]
        if not rows:
            chosen[name] = {}
            continue
        best = max(rows, key=lambda r: r["mean_auroc"])
        default = next((r for r in rows if r["is_default"]), None)
        pick = default if default is not None and best["mean_auroc"] - default["mean_auroc"] <= TIE else best
        chosen[name] = {} if pick["is_default"] else dict(pick["params"])
    return chosen


def report_tuning(reader: str, seeds: Sequence[int], summary: Dict[str, List[dict]], chosen: Dict[str, dict]) -> str:
    L = [f"# Tuning of the standard detectors on `{reader}`, seeds {seeds[0]}-{seeds[-1]} (not report seeds)", "",
         "Criterion: mean AUROC of parts of untaught skills against parts of taught skills (no threshold). A setting within 0.002 of the best loses to the default. 'found' and 'false unseen' are "
         "at q = 0.05 with the shared held-out calibration and are shown for information only.", "", "| detector | setting | seeds | mean AUROC | found | false unseen | chosen |", "|---|---|---|---|---|---|---|"]
    for name, rows in summary.items():
        picked = chosen.get(name) or DEFAULTS.get(name, {})
        for r in rows:
            mark = "chosen" if r["params"] == picked else ""
            L.append(f"| {name} | {r['setting']}{' (default)' if r['is_default'] else ''} | {r['n_seeds']} | {r['mean_auroc']:.3f} | {r['mean_recall']:.3f} | {r['mean_false_unseen']:.3f} | {mark} |")
    L += ["", "Chosen overrides (an empty dict keeps the default): `BASELINE_PARAMS = " + json.dumps(chosen) + "`", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="R5_R4_qwen_L26")
    ap.add_argument("--seeds", default="150-154")
    ap.add_argument("--pool", default="dev")
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds = R.seed_range(args.seeds)
    summary = run_tuning(args.reader, seeds, args.pool, GRID, R.SIZES, OUT)
    chosen = choose(summary)
    text = report_tuning(args.reader, seeds, summary, chosen)
    runlib.atomic_write_text(HERE / "results" / "BASELINE_TUNING.md", text)
    print(text)

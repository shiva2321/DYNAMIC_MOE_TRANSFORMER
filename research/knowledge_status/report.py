"""The verdict of the confirmatory run: the cells in ``results/confirm`` against the FROZEN bars (``bars.json``), by ``verdict.py`` and nothing else.

    PYTHONPATH=<repo root> python report.py            # after run_confirm.py closed the run; writes results/confirm/CONFIRM_RESULT.md and verdict.json

Written and tested before the freeze, so the way the numbers are made is part of the pre-registration. The full tables of E1, E2/E3 and E4 are appended from the same report functions that
produced the dev tables.
"""

import json
import math
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

import evaluate as EV
import freeze_prereg as FZ
import runlib
import verdict as VD

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "confirm"


def measured_means(values: Mapping[str, list]) -> dict:
    return {k: float(np.mean(v)) for k, v in values.items() if v}


def adapted_false_unseen(results_dir: Path, rep: str, stamp: str, q: float) -> Optional[float]:
    """The not-gated adapted-condition number (a few S4 to S8 parts in the calibration), mean over seeds, or None."""
    vals = []
    for p in sorted((Path(results_dir) / "e2").glob(f"{rep}__s*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        if c.get("code_stamp") != stamp:
            continue
        try:
            v = c["machine"][f"q{q:g}"]["adapted"]["false_unseen_heldout"]
        except KeyError:
            continue
        if isinstance(v, (int, float)) and not math.isnan(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def surprise(name: str, mean: Optional[float], bar_row: Mapping) -> bool:
    """A confirmatory mean outside [dev min - dev sd, dev max + dev sd] (the registered surprise rule); False when there is no dev range."""
    if mean is None or bar_row.get("dev_min") is None:
        return False
    sd = bar_row.get("dev_sd") or 0.0
    return bool(mean < bar_row["dev_min"] - sd or mean > bar_row["dev_max"] + sd)


def build(rec: Mapping, results_dir: Path = OUT, stamps: Optional[Mapping] = None, full: bool = True) -> tuple:
    """(markdown text, result dict). ``full`` appends the E1, E2/E3 and E4 tables of the confirmatory cells."""
    stamps = stamps if stamps is not None else FZ.current_stamps()
    values = FZ.collect_dev_values(results_dir, stamps, rec["headline_representation"])             # the same reading of the cells as at the freeze
    means = measured_means(values)
    bars = {k: {"bar": r["bar"], "direction": r["direction"]} for k, r in rec["bars"].items()}
    v = VD.verdict(means, bars)
    adapted = adapted_false_unseen(results_dir, rec["headline_representation"], FZ._stamp(stamps["e2"], rec["headline_representation"]), rec["q"])
    claim = EV.gate(means.get("false_unseen_heldout", float("nan")), adapted if adapted is not None else float("nan"), rec["bars"]["false_unseen_heldout"]["bar"])
    L = [f"# Confirmatory result (single use), frozen bars sha256 `{FZ.bars_hash(rec)}`", "",
         f"Cells read: {', '.join(f'{k} {n}' for k, n in sorted((k, len(x)) for k, x in values.items()))} (values per measure over the seeds that have it). Verdicts are made by `verdict.py` from the "
         "frozen `bars.json`; the expected verdict is the one registered before the run.", "",
         "| measure | statement | confirmatory mean | bar | verdict | dev mean | surprise |", "|---|---|---|---|---|---|---|"]
    for name, row in v["measures"].items():
        dev = rec["bars"][name]["dev_mean"]
        sup = surprise(name, row["value"], rec["bars"][name])
        L.append(f"| {name} | {row['statement']} | {'n/a' if row['value'] is None else format(row['value'], '.3f')} | {'at least' if row['direction'] == 'min' else 'at most'} {row['bar']:.3f} | "
                 f"{row['verdict'].replace('_', ' ').upper()} | {'n/a' if dev is None else format(dev, '.3f')} | {'SURPRISE' if sup else ''} |")
    L += ["", "| statement | verdict | registered expectation | as expected |", "|---|---|---|---|"]
    for sid, got in v["statements"].items():
        want = rec["statements"][sid]["expected"]
        L.append(f"| {sid} | {got.replace('_', ' ').upper()} | {want.replace('_', ' ').upper()} | {'yes' if got == want else 'NO'} |")
    L += ["", f"**Wording claim** (false 'unseen' on S4 to S8, standard calibration, bar {rec['bars']['false_unseen_heldout']['bar']:.2f}): {claim['claim']}; standard "
          f"{means.get('false_unseen_heldout', float('nan')):.3f}, adapted {adapted if adapted is None else format(adapted, '.3f')} (adapted is reported, never gated).", ""]
    if full:
        import run_e1
        import run_e2
        import run_e3
        seeds = rec["seeds"]["confirm"]
        L += ["", run_e1.report_e1(Path(results_dir) / "e1", rec["representations"], seeds, stamps["e1"], "confirm"), run_e2.report_e2(Path(results_dir) / "e2", rec["representations"], seeds,
                                                                                                                                      stamps["e2"], "confirm")]
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(results_dir) / "e4").glob("s*.json"))]
        if rows:
            L += [run_e3.report_e4(rows, len(seeds), "learned-map reader, confirmatory pool")]
    return "\n".join(L) + "\n", {"verdict": v, "means": means, "adapted_false_unseen": adapted, "claim": claim, "bars_sha256": FZ.bars_hash(rec)}


def main() -> int:
    if not (HERE / "bars.json").exists():
        print("REFUSED: nothing is frozen yet (no bars.json)")
        return 2
    rec = json.loads((HERE / "bars.json").read_text(encoding="utf-8"))
    chk = FZ.check_frozen(HERE, HERE / "pool_confirm.py", FZ.default_code_files())
    if not chk["ok"]:
        print(f"REFUSED: {chk['problem']}")
        return 2
    if not (HERE / "CONFIRM_CLOSED.json").exists():
        print("REFUSED: the confirmatory run is not closed; run run_confirm.py to completion first")
        return 2
    text, result = build(rec)
    runlib.atomic_write_text(OUT / "CONFIRM_RESULT.md", text)
    runlib.atomic_write_text(OUT / "verdict.json", json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

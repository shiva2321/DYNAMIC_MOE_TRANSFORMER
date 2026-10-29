"""The verdict of the confirmatory run: the cells in ``results/confirm`` against the FROZEN bars (``bars.json``), by ``verdict.py`` and nothing else.

    PYTHONPATH=<repo root> python report.py            # after run_confirm.py closed the run; writes results/confirm/CONFIRM_RESULT.md and verdict.json

Written and tested before the freeze, so the way the numbers are made is part of the pre-registration. The full tables of E1, E2/E3 and E4 are appended from the same report functions that
produced the dev tables, with their PASS / fail marks (made against the DRAFT bars) and their "dev-fixed bars" section removed: the verdict is the table at the top, made from the frozen bars.
"""

import json
import math
import re
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

import evaluate as EV
import freeze_prereg as FZ
import runlib
import verdict as VD

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "confirm"


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


def strip_marks(text: str) -> str:
    """The appended experiment tables without what could contradict the frozen verdict: their PASS / fail marks (against the DRAFT bars), the B2 line, and the 'draft and dev-fixed bars' section,
    which the dev report computes from whatever cells it is given."""
    cut = text.find("## Bars: draft and dev-fixed")
    if cut >= 0:
        text = text[:cut]
    out = []
    for line in text.splitlines():
        if line.startswith("**B2 validity") or "PASS / fail is the mean against the DRAFT bar" in line or line.startswith("B5: the parent set of a multiplication trace is exactly"):
            continue
        if line.startswith("|"):
            line = re.sub(r":? (?:PASS|fail)(?=\s*\|)", "", line)
            line = re.sub(r" at (?:least|most) \d\.\d\d(?=\s*\|)", "", line)               # a bar printed from the DRAFT constants (the frozen bar is in the table at the top)
        out.append(line)
    return "\n".join(out)


def banners(cons: Mapping) -> list:
    out = []
    if cons["invalid"]:
        out.append("**INVALID: B2 failed. The test bed is broken for this run (anti-goal A1); none of the results below counts.**")
    if cons["status_void"]:
        out.append("**STATUS CLAIMS VOID: B1b is below its bar for the headline reader, which cannot see an operator change; no status claim (B3a to B7) built on it counts.**")
    if cons["conditional"]:
        out.append("**STATUS CLAIMS CONDITIONAL: the headline reader missed B1a or B1c (the spec says such a reader is not used for status; the run carried it on by the user's decision D-40). "
                   "Every pass of B3a to B7 below is conditional on that, and not a spec pass.**")
    return out


def build(rec: Mapping, results_dir: Path = OUT, stamps: Optional[Mapping] = None, full: bool = True) -> tuple:
    """(markdown text, result dict). ``full`` appends the E1, E2/E3 and E4 tables of the confirmatory cells."""
    import run_confirm as RC
    stamps = stamps if stamps is not None else RC.stamps(rec)
    head = rec["headline_representation"]
    seeds = rec["seeds"]["confirm"]
    values = FZ.collect_dev_values(results_dir, stamps, head, seeds=seeds, pool="confirm", require=("e1", "e2"))
    means = FZ.derive_means(values, rec["min_seeds"])
    bars = {k: {"bar": r["bar"], "direction": r["direction"], "draft": r["draft"]} for k, r in rec["bars"].items()}
    v = VD.verdict(means, bars)
    cons = VD.consequences(v, bars)
    adapted = adapted_false_unseen(results_dir, head, FZ._stamp(stamps["e2"], head), rec["q"])
    claim = EV.gate(means.get("false_unseen_heldout", float("nan")), adapted if adapted is not None else float("nan"), rec["bars"]["false_unseen_heldout"]["bar"])
    L = [f"# Confirmatory result (single use), frozen bars sha256 `{FZ.bars_hash(rec)}`", ""] + banners(cons) + [""] * bool(any(cons.values()))
    L += [f"Cells read: {', '.join(f'{k} {n}' for k, n in sorted((k, len(x)) for k, x in values.items()))} (values per measure over the seeds that have it; a measure with fewer than "
          f"{rec['min_seeds']} seeds is not judged). Verdicts are made by `verdict.py` from the frozen `bars.json`; the expected verdict is the one registered before the run.", "",
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
    r0 = {k: means.get(k) for k in ("r0_b1a", "r0_b1b")}
    small = means.get("b7_diff_small")
    L += ["", "**B2, the word-matcher control R0** (gap = the spec's draft bar minus R0's value; B2 needs one gap of at least "
          f"{rec['bars']['b2_gap']['bar']:.2f}): " + "; ".join(f"{k} {('n/a' if x is None else format(x, '.3f'))}" + ("" if x is None else f", gap {rec['bars'][b]['draft'] - x:+.3f}")
                                                                for (k, x), b in zip(r0.items(), ("b1a", "b1b"))) + ".", "",
          f"**Wording claim** (false 'unseen' on S4 to S8, standard calibration, bar {rec['bars']['false_unseen_heldout']['bar']:.2f}): {claim['claim']}; standard "
          f"{means.get('false_unseen_heldout', float('nan')):.3f}, adapted {adapted if adapted is None else format(adapted, '.3f')} (adapted is reported, never gated).", "",
          "**B7 on the problems whose operands are both at most 99** (reported, not gated; dev " + ("n/a" if rec['dev_extras'].get('b7_diff_small') is None else format(rec['dev_extras']['b7_diff_small'], '+.3f'))
          + "): " + ("n/a" if small is None else format(small, '+.3f')) + " (realism minus templates, mean over the seeds that can be scored).", ""]
    fs_rows = {}
    for arm in rec["representations"]:
        if arm in FZ.FS_ARMS:
            try:
                fv = FZ.collect_dev_values(results_dir, stamps, arm, seeds=seeds, pool="confirm", require=("e1", "e2"))
            except ValueError:
                continue
            fs_rows[arm] = {k: float(np.mean(fv[k])) for k in ("b1a", "b1b", "b1c", "macro_f1", "false_unseen_heldout") if k in fv}
    if fs_rows:
        L += ["**From-scratch reader arms** (reported, never gated), confirmatory means next to the headline reader:", "", "| reader | B1a | B1b | B1c | macro F1 | false-new |", "|---|---|---|---|---|---|"]
        for name, r in [(head, {k: means.get(k) for k in ("b1a", "b1b", "b1c", "macro_f1", "false_unseen_heldout")}), *fs_rows.items()]:
            L.append(f"| {name} | " + " | ".join('n/a' if r.get(k) is None else format(r[k], '.3f') for k in ("b1a", "b1b", "b1c", "macro_f1", "false_unseen_heldout")) + " |")
        L.append("")
    if full:
        import run_e1
        import run_e2
        import run_e3
        import run_realism
        diag = ["## Experiment tables (diagnostic)", "", "Appended from the experiment reports. Their PASS / fail marks (against the DRAFT bars) and the dev-fixed-bars section were removed: "
                "the verdict is the table at the top, made from the frozen bars.", ""]
        L += diag + [strip_marks(run_e1.report_e1(Path(results_dir) / "e1", list(rec["representations"]) + list(rec.get("e1_only_readers", [])), seeds, stamps["e1"], "confirm")),
                     strip_marks(run_e2.report_e2(Path(results_dir) / "e2", rec["representations"], seeds, stamps["e2"], "confirm"))]
        e4 = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(results_dir) / "e4").glob("s*.json"))]
        if e4:
            L += [strip_marks(run_e3.report_e4(e4, len(seeds), "learned-map reader, confirmatory pool"))]
        b7 = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(results_dir) / "realism").glob("s*.json"))]
        if b7:
            L += [run_realism.report_b7(b7, "confirmatory")]
    return "\n".join(L) + "\n", {"verdict": v, "consequences": cons, "means": means, "adapted_false_unseen": adapted, "claim": claim, "bars_sha256": FZ.bars_hash(rec)}


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

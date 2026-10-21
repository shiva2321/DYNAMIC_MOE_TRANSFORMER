"""Independent audit of the RV2-D (routing v2 stage 4) verdicts: recompute S4-1 to S4-8 from the raw cell records, written from the spec text (docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md,
section 8) and amendment A4-1, WITHOUT importing growmoe.verdicts_rv4 or report_rv2d. Also checks the integrity of the record set (stamps, phases, seeds, duplicates) and, when the report has been
written, compares every verdict with verdicts_rv2d.json.

    python audit_rv2d.py                                         # confirmatory records in cells/ (phase confirm, seeds 85-99, the stamp frozen in bars_rv2d.json)
    python audit_rv2d.py --dry-run-dev                           # the same code on cells_devview/ (phase dev, seeds >= 100, their own stamp): checks this script against the main verdict code on data that exists
    python audit_rv2d.py --write                                 # also writes AUDIT_RV2D.md (confirm) or AUDIT_RV2D_DEV.md (dev)

S4-9 (golden) is not recomputed here: it needs `python research/growmoe_results/rv2c/golden.py check` (about six minutes); the audit reports the value stored in the verdicts file.

Since the independent review (E-042, 2026-10-21) the audit (1) reads S4-5 over all eleven units the spec names ("each other stream and banking77": the ten stage-3 streams and banking77;
the verdict code of the run, growmoe/verdicts_rv4.py, uses ten and leaves out OV-0.85, review finding SF-4; the verdict is the same) and (2) compares the record set with cells_confirm.json.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ALL_S3 = ("F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5")
GUARDS = tuple(s for s in ALL_S3 if s != "OV-0.85")


def t_interval(a, b, level=0.95):
    """mean of a - b and its two-sided t interval over the paired seeds (the definition of growmoe.metrics.paired_diff, written again here)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    m = float(d.mean())
    if n < 2:
        return n, m, float("-inf"), float("inf")
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1) * d.std(ddof=1) / math.sqrt(n))
    return n, m, m - half, m + half


class Records:
    def __init__(self, folder: Path, phase: str, seeds, stamp: str):
        self.by = defaultdict(dict)
        self.problems, self.n = [], 0
        seen = set()
        for f in sorted(folder.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                cell = rec["cell"]
            except (OSError, ValueError, KeyError) as exc:
                self.problems.append(f"{f.name}: unreadable record ({type(exc).__name__}: {exc})")                  # an audit reports what it cannot read; it never skips it silently
                continue
            if cell.get("phase") != phase or int(cell["seed"]) not in seeds:
                continue
            if rec.get("code_stamp") != stamp:
                self.problems.append(f"{f.name}: code stamp {rec.get('code_stamp')} is not {stamp}")
                continue
            cid = rec.get("cid", f.stem)
            if cid in seen:
                self.problems.append(f"{cid}: duplicate record")
                continue
            seen.add(cid)
            if not (isinstance(rec.get("acc"), (int, float)) and math.isfinite(rec["acc"])):
                self.problems.append(f"{cid}: accuracy is not a finite number")
                continue
            self.by[(cell["family"], cell["stream"], cell["arm"])][int(cell["seed"])] = rec
            self.n += 1

    def get(self, family, stream, arm):
        return self.by.get((family, stream, arm), {})


def paired(a: dict, b: dict, fa, fb):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    return t_interval([100.0 * fa(a[s]) for s in seeds], [100.0 * fb(b[s]) for s in seeds])


def audit(R: Records, setting: str, bars: dict) -> dict:
    on = lambda r: r["router"][setting]["acc"]          # noqa: E731
    off = lambda r: r["router"]["off"]["acc"]           # noqa: E731
    acc = lambda r: r["acc"]                            # noqa: E731
    out = {}

    def router_gain(family, stream):
        recs = R.get(family, stream, "S3")
        return paired(recs, recs, on, off)

    # S4-1, S4-2: lower bound of the 95 % interval of the router gain
    g = router_gain("arith-seq", "-")
    out["S4-1"] = {"value": f"{g[1]:+.2f} [{g[2]:+.2f}, {g[3]:+.2f}] n={g[0]}" if g else None, "pass": None if g is None else g[2] >= bars["S4_1_lo"], "bar": f"lower bound >= +{bars['S4_1_lo']}"}
    g = router_gain("s3", "OV-0.85")
    out["S4-2"] = {"value": f"{g[1]:+.2f} [{g[2]:+.2f}, {g[3]:+.2f}] n={g[0]}" if g else None, "pass": None if g is None else g[2] >= bars["S4_2_lo"], "bar": f"lower bound >= +{bars['S4_2_lo']}"}

    # S4-3: the router must not hurt any other stream or banking77 (mean >= -0.5)
    units = [("s3", s) for s in GUARDS] + [("banking", "-")]
    gains = {f"{f}:{s}": router_gain(f, s) for f, s in units}
    ok = None if any(v is None for v in gains.values()) else all(v[1] >= bars["S4_3"] for v in gains.values())
    worst = None if any(v is None for v in gains.values()) else min(v[1] for v in gains.values())
    out["S4-3"] = {"value": "worst guard mean " + ("n/a" if worst is None else f"{worst:+.2f}") + f" over {sum(v is not None for v in gains.values())}/{len(units)} units", "pass": ok, "bar": f"every mean >= {bars['S4_3']}"}

    # S4-4: transfer rule on operator-by-operator arithmetic
    s3, t = R.get("arith-seq", "-", "S3"), R.get("arith-seq", "-", "S3+T")
    d = paired(t, s3, acc, acc)
    if d is None or not t:
        out["S4-4"] = {"value": None, "pass": None, "bar": "gain >= +5, forgetting <= 9, at most 1 seed with < 5 experts"}
    else:
        forget = 100.0 * float(np.mean([r["forgetting"] for r in t.values()]))
        few = sum(r["n_experts"] < bars["S4_4_min_experts"] for r in t.values())
        out["S4-4"] = {"value": f"gain {d[1]:+.2f} (n={d[0]}), forgetting {forget:.2f}, seeds with < {bars['S4_4_min_experts']} experts {few}",
                       "pass": bool(d[1] >= bars["S4_4_gain"] and forget <= bars["S4_4_forgetting"] and few <= bars["S4_4_seeds_below_allowed"]), "bar": "gain >= +5, forgetting <= 9, at most 1 seed with < 5 experts"}

    # S4-5 (amendment A4-1): accuracy and expert count of S3+T on every other stream, experts against the number of true functions
    units5 = [("s3", s) for s in ALL_S3] + [("banking", "-")]                        # all eleven: OV-0.85 is S4-2's target, not S4-5's exception (review SF-4)
    rows, complete = [], True
    for f, s in units5:
        a, b = R.get(f, s, "S3+T"), R.get(f, s, "S3")
        d = paired(a, b, acc, acc)
        if d is None:
            complete = False
            continue
        truth = [r.get("regimes") for r in a.values()]
        known = all(isinstance(x, (int, float)) and x > 0 for x in truth)
        denominator = float(np.mean(truth)) if known else float(np.mean([r["n_experts"] for r in b.values()]))
        ratio = float(np.mean([r["n_experts"] for r in a.values()])) / max(denominator, 1e-9)
        rows.append((f"{f}:{s}", d[1], ratio, "true" if known else "S3"))
    if not rows:
        out["S4-5"] = {"value": None, "pass": None, "bar": "every mean >= -0.5 and experts <= 1.2 x true functions"}
    else:
        good = all(m >= bars["S4_5_gain"] and r <= bars["S4_5_ratio"] for _, m, r, _ in rows)
        worst_m, worst_r = min(m for _, m, _, _ in rows), max(r for _, _, r, _ in rows)
        out["S4-5"] = {"value": f"worst accuracy mean {worst_m:+.2f}, largest expert ratio {worst_r:.2f}, {len(rows)}/{len(units5)} units", "pass": bool(good) if complete else None,
                       "bar": "every mean >= -0.5 and experts <= 1.2 x true functions (A4-1)"}

    # S4-6: growth from 16 units reaches 90 % of the fixed 128-unit expert in enough seeds, width in the band, forward cost small
    small, big = R.get("arith-mixed16", "-", "S3+G"), R.get("arith-mixed128", "-", "S3")
    seeds = sorted(set(small) & set(big))
    if not seeds:
        out["S4-6"] = {"value": None, "pass": None, "bar": ">= 13 of 15 seeds reach 0.9, width in [64, 256], extra forward <= 5 %"}
    else:
        reach = sum(small[s]["acc"] >= bars["S4_6_frac"] * big[s]["acc"] for s in seeds)
        need = math.ceil(bars["S4_6_seed_share"] * len(seeds) - 1e-9)
        width = float(np.mean([np.mean(small[s]["final_hidden"]) for s in seeds]))
        cost = float(np.mean([small[s]["grow_stats"]["post_rows"] / max(small[s]["grow_stats"]["train_rows"] * small[s]["inner_steps"] * 6, 1) for s in seeds]))
        lo, hi = bars["S4_6_width"]
        out["S4-6"] = {"value": f"{reach}/{len(seeds)} seeds reach 0.9 (need {need}), width {width:.0f}, extra forward {100 * cost:.2f} %", "pass": bool(reach >= need and lo <= width <= hi and cost <= bars["S4_6_cost"]),
                       "bar": ">= 13 of 15 seeds reach 0.9, width in [64, 256], extra forward <= 5 %"}

    # S4-7: growth must not hurt where it is not needed
    rows = []
    for f, s in [("arith-mixed128", "-")] + [("s3", x) for x in ALL_S3]:
        a, b = R.get(f, s, "S3+G"), R.get(f, s, "S3")
        d = paired(a, b, acc, acc)
        if d is None:
            continue
        seeds = sorted(set(a) & set(b))
        grows = float(np.mean([len(a[x]["grow_events"]) for x in seeds]))
        relp = float(np.mean([a[x]["params"] / b[x]["params"] - 1.0 for x in seeds]))
        rows.append((f"{f}:{s}", d[1], grows, relp))
    expected_units = 1 + len(ALL_S3)
    if not rows:
        out["S4-7"] = {"value": None, "pass": None, "bar": "every mean >= -0.5, growth events per run <= 1, parameters <= +25 %"}
    else:
        good = all(m >= bars["S4_7_gain"] and g <= bars["S4_7_grow"] and p <= bars["S4_7_params"] for _, m, g, p in rows)
        out["S4-7"] = {"value": f"worst mean {min(m for _, m, _, _ in rows):+.2f}, most growth events per run {max(g for _, _, g, _ in rows):.2f}, largest parameter increase {100 * max(p for _, _, _, p in rows):+.1f} %, {len(rows)}/{expected_units} units",
                       "pass": bool(good) if len(rows) == expected_units else None, "bar": "every mean >= -0.5, growth events per run <= 1, parameters <= +25 %"}

    # S4-8: the full stack on operator-by-operator arithmetic
    full = R.get("arith-seq", "-", "S3+T+G")
    if not full:
        out["S4-8"] = {"value": None, "pass": None, "bar": "exact match with the router >= 44, forgetting <= 8"}
    else:
        a8 = 100.0 * float(np.mean([on(r) for r in full.values()]))
        f8 = 100.0 * float(np.mean([r["forgetting"] for r in full.values()]))
        out["S4-8"] = {"value": f"exact match {a8:.2f} (n={len(full)}), forgetting {f8:.2f}", "pass": bool(a8 >= bars["S4_8_acc"] and f8 <= bars["S4_8_forgetting"]), "bar": "exact match >= 44, forgetting <= 8"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-dev", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    bars_file = json.loads((HERE / "bars_rv2d.json").read_text(encoding="utf-8"))
    bars = {k: (tuple(v) if isinstance(v, list) else v) for k, v in bars_file["bars"].items()}
    setting = bars_file["constants"]["router_setting"]
    if a.dry_run_dev:
        folder, phase, seeds = HERE / "cells_devview", "dev", set(range(100, 200))
        stamps = {json.loads(f.read_text(encoding="utf-8")).get("code_stamp") for f in folder.glob("*.json")}
        assert len(stamps) == 1, stamps
        stamp, reported_name, out_name = stamps.pop(), None, "AUDIT_RV2D_DEV.md"
    else:
        folder, phase, seeds = HERE / "cells", "confirm", set(range(85, 100))
        stamp, reported_name, out_name = bars_file["code_stamp"], "verdicts_rv2d.json", "AUDIT_RV2D.md"
    R = Records(folder, phase, seeds, stamp)
    if not a.dry_run_dev:                                                                  # the record set must be exactly the cells that were listed before the run (review N-4)
        listed = {tuple(sorted(c.items())) for c in json.loads((HERE / "cells_confirm.json").read_text(encoding="utf-8"))}
        found = {tuple(sorted(rec["cell"].items())) for recs in R.by.values() for rec in recs.values()}
        R.problems += [f"cell listed in cells_confirm.json but without a record: {dict(c)}" for c in sorted(listed - found)[:10]]
        R.problems += [f"record that cells_confirm.json does not list: {dict(c)}" for c in sorted(found - listed)[:10]]
    res = audit(R, setting, bars)
    lines = [f"# Independent audit of the RV2-D verdicts ({'DRY RUN on the dev records' if a.dry_run_dev else 'confirmatory records'})\n",
             f"Records read: {R.n} of phase {phase} with the code stamp {stamp}; integrity problems: {len(R.problems)}.\n"]
    lines += [f"- {p}" for p in R.problems[:20]]
    lines += ["", "| ID | recomputed value | bar | recomputed verdict |", "|---|---|---|---|"]
    for k in sorted(res):
        v = res[k]
        lines.append(f"| {k} | {v['value']} | {v['bar']} | {'NOT MEASURABLE' if v['pass'] is None else ('PASS' if v['pass'] else 'FAIL')} |")
    status = 0
    reported = None
    if reported_name and (HERE / reported_name).exists():
        reported = json.loads((HERE / reported_name).read_text(encoding="utf-8"))
    if reported:
        lines += ["", "## Against the reported verdicts (`verdicts_rv2d.json`)", "", "| ID | audit | report | agree |", "|---|---|---|---|"]
        for k in sorted(res):
            mine = "NOT MEASURABLE" if res[k]["pass"] is None else ("PASS" if res[k]["pass"] else "FAIL")
            theirs = reported[k]["verdict"]
            lines.append(f"| {k} | {mine} | {theirs} | {'yes' if mine == theirs else '**NO**'} |")
            status |= int(mine != theirs)
        lines.append(f"| S4-9 | (not recomputed) | {reported['S4-9']['verdict']} | |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.write:
        (HERE / out_name).write_text(text, encoding="utf-8", newline="\n")
    return status | int(bool(R.problems))


if __name__ == "__main__":
    sys.exit(main())

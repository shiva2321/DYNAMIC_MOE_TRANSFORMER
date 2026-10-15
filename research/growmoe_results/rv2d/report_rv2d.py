"""Assemble the RV2-D report (routing v2 stage 4) from the cell records only: every number below is computed here from cells/*.json, none is typed by hand.

    python report_rv2d.py dev        # RESULTS_DEV.md: router, transfer, growth, full stack and cost tables on the dev seeds (cells of dev1 and dev2)
    python report_rv2d.py confirm    # RESULTS_RV2D.md: the pre-registered statements S4-1 to S4-9 with verdicts, then the same tables on seeds 85-99
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import rv2d_lib as R  # noqa: E402
from growmoe.metrics import paired_diff  # noqa: E402

PHASE_SEEDS = {"dev": lambda s: s >= 100, "confirm": lambda s: 85 <= s <= 99}
ARM_ORDER = ["S3", "S3+T", "S3+G", "S3+T+G"]


STALE = {"cells": 0}


def load(phase: str, stamp: str = None) -> list:
    """All cell records of the phase that were produced by the code with this stamp (the current one by default), as flat rows (the fields of rv2d_select plus the raw record).
    Records of other code are left out and counted in STALE."""
    keep = PHASE_SEEDS[phase]
    stamp = R.code_stamp() if stamp is None else stamp
    STALE["cells"] = 0
    rows = []
    for f in sorted(R.CELLS.glob("*.json")):
        if not R.valid_json(f):
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        c = rec["cell"]
        if not keep(int(c["seed"])):
            continue
        if rec.get("code_stamp") != stamp:
            STALE["cells"] += 1
            continue
        rows.append({"family": c["family"], "stream": c["stream"], "arm": c["arm"], "seed": int(c["seed"]), "gap": c["gap"], "tmargin": c["tmargin"], "acc": rec["acc"],
                     "params": rec["params"], "n_grow": len(rec["grow_events"]), "n_experts": rec["n_experts"], "router": rec["router"], "forgetting": rec.get("forgetting"),
                     "grow_events": rec["grow_events"], "final_hidden": rec["final_hidden"], "grow_stats": rec["grow_stats"], "inner_steps": rec["inner_steps"], "events": rec["events"],
                     "regimes": rec.get("regimes"), "seconds": rec["seconds"]})
    return rows


def pm(d: dict, digits: int = 1) -> str:
    lo, hi = d["ci_lo"], d["ci_hi"]
    ci = "n/a" if not np.isfinite(lo) else f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"
    return f"{d['mean_diff']:+.{digits}f} {ci} (n={d['n']})"


def paired(rows, family, stream, arm_a, arm_b, fa=lambda r: r["acc"], fb=lambda r: r["acc"], gap=None, tmargin=None):
    """Mean paired difference arm_a - arm_b in points over the seeds both have, or None."""
    def pick(arm, **where):
        return {r["seed"]: r for r in rows if r["family"] == family and r["stream"] == stream and r["arm"] == arm and all(r.get(k) == v for k, v in where.items())}
    a = pick(arm_a, **({"gap": gap} if gap and "G" in arm_a else {}), **({"tmargin": tmargin} if tmargin and "T" in arm_a else {}))
    b = pick(arm_b)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    return paired_diff([100 * fa(a[s]) for s in seeds], [100 * fb(b[s]) for s in seeds])


def units(rows):
    return sorted({(r["family"], r["stream"]) for r in rows}, key=lambda u: (["arith-seq", "arith-mixed16", "arith-mixed128", "s3", "banking"].index(u[0]), u[1]))


def label(u):
    return u[0] if u[1] == "-" else f"{u[0]}:{u[1]}"


def section_router(rows, setting):
    out = [f"## Router (setting {setting}), S3 pools: the SAME trained pool answered without and with the router\n",
           "| unit | seeds | accuracy without | with the router | with minus without (points, 95 % interval) | rule selected (cells) | CV owner accuracy density / LDA / softmax / 5-NN (mean) |", "|---|---|---|---|---|---|---|"]
    for u in units(rows):
        rs = [r for r in rows if (r["family"], r["stream"]) == u and r["arm"] == "S3" and setting in r["router"]]
        if not rs:
            continue
        d = paired_diff([100 * r["router"][setting]["acc"] for r in rs], [100 * r["router"]["off"]["acc"] for r in rs]) if len(rs) > 0 else None
        sel = {}
        for r in rs:
            sel[r["router"][setting]["selected"]] = sel.get(r["router"][setting]["selected"], 0) + 1
        cvs = [r["router"][setting]["cv"] for r in rs if r["router"][setting]["cv"]]
        cv = " / ".join(f"{100 * np.mean([c[k] for c in cvs]):.0f}" for k in ("density", "lda", "softmax", "knn5") if all(k in c for c in cvs)) if cvs else "-"
        out.append(f"| {label(u)} | {len(rs)} | {100 * np.mean([r['router']['off']['acc'] for r in rs]):.1f} | {100 * np.mean([r['router'][setting]['acc'] for r in rs]):.1f} | {pm(d)} | "
                   f"{', '.join(f'{k} {v}' for k, v in sorted(sel.items()))} | {cv} |")
    return out + [""]


def section_arm(rows, arm, title, extra):
    out = [f"## {title}\n", "| unit | seeds | S3 accuracy | " + arm + " accuracy | difference (points, 95 % interval) | " + extra[0] + " |", "|---|---|---|---|---|---|"]
    for u in units(rows):
        a = [r for r in rows if (r["family"], r["stream"]) == u and r["arm"] == arm]
        b = [r for r in rows if (r["family"], r["stream"]) == u and r["arm"] == "S3"]
        if not a or not b:
            continue
        d = paired(rows, u[0], u[1], arm, "S3")
        if d is None:
            continue
        out.append(f"| {label(u)} | {d['n']} | {100 * np.mean([r['acc'] for r in b]):.1f} | {100 * np.mean([r['acc'] for r in a]):.1f} | {pm(d)} | {extra[1](a, b)} |")
    return out + [""]


def transfer_extra(a, b):
    spawns = np.mean([r["events"].get("spawn", 0) for r in a])
    extends = np.mean([r["events"].get("extend", 0) for r in a])
    fa = [r["forgetting"] for r in a if r.get("forgetting") is not None]
    fb = [r["forgetting"] for r in b if r.get("forgetting") is not None]
    forg = f"; forgetting {100 * np.mean(fb):.1f} -> {100 * np.mean(fa):.1f}" if fa and fb else ""
    return f"experts {np.mean([r['n_experts'] for r in b]):.1f} -> {np.mean([r['n_experts'] for r in a]):.1f}; spawns {spawns:.1f}, extends {extends:.1f}{forg}"


def growth_extra(a, b):
    g = np.mean([r["n_grow"] for r in a])
    rev = np.mean([r["events"].get("grow_revert", 0) for r in a])
    relp = np.mean([r["params"] / bb["params"] - 1.0 for r in a for bb in b if bb["seed"] == r["seed"]])
    width = np.mean([np.mean(r["final_hidden"]) for r in a])
    cost = np.mean([r["grow_stats"]["post_rows"] / max(r["grow_stats"]["train_rows"] * r["inner_steps"] * 6, 1) for r in a])
    return f"growth events/run {g:.2f}, reverts/run {rev:.2f}, parameters {100 * relp:+.0f} %, mean final width {width:.0f}, extra forward {100 * cost:.2f} % of training"


def section_stack(rows, setting):
    out = ["## Full stack on arithmetic operator-by-operator (S3+T+G, router at its setting)\n", "| arm | seeds | exact match (router off) | exact match with the router | forgetting | experts |", "|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        rs = [r for r in rows if r["family"] == "arith-seq" and r["arm"] == arm]
        if rs:
            on = np.mean([r["router"][setting]["acc"] for r in rs if setting in r["router"]]) if any(setting in r["router"] for r in rs) else float("nan")
            out.append(f"| {arm} | {len(rs)} | {100 * np.mean([r['acc'] for r in rs]):.1f} | {100 * on:.1f} | {100 * np.mean([r['forgetting'] for r in rs]):.1f} | {np.mean([r['n_experts'] for r in rs]):.1f} |")
    return out + [""]


def section_gamma(rows):
    out = ["## Growth threshold curve (S3+G minus S3, dev seeds): the declared grid\n", "| grow_gap | " + " | ".join(f"{f}{'' if s == '-' else ':' + s}" for f, s in [("arith-mixed16", "-"), ("arith-mixed128", "-"), ("s3", "F-15"), ("s3", "OV-0.6")]) + " |", "|---|---|---|---|---|"]
    for g in R.GAP_GRID:
        cells = []
        for fam, st in [("arith-mixed16", "-"), ("arith-mixed128", "-"), ("s3", "F-15"), ("s3", "OV-0.6")]:
            a = [r for r in rows if r["family"] == fam and r["stream"] == st and r["arm"] == "S3+G" and r["gap"] == g]
            d = paired(rows, fam, st, "S3+G", "S3", gap=g)
            cells.append("-" if not a or d is None else f"{d['mean_diff']:+.1f} pts, width {np.mean([np.mean(r['final_hidden']) for r in a]):.0f}, growths {np.mean([r['n_grow'] for r in a]):.1f}")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    return out + [""]


def section_cost(rows):
    out = ["## Wall-clock per cell (seconds, mean)\n", "| unit | " + " | ".join(ARM_ORDER) + " |", "|---|---|---|---|---|"]
    for u in units(rows):
        out.append(f"| {label(u)} | " + " | ".join(f"{np.mean([r['seconds'] for r in rows if (r['family'], r['stream']) == u and r['arm'] == a]):.0f}" if any((r['family'], r['stream']) == u and r['arm'] == a for r in rows) else "-" for a in ARM_ORDER) + " |")
    return out + [""]


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "dev"
    frozen = json.loads((HERE / "bars_rv2d.json").read_text(encoding="utf-8")).get("code_stamp") if phase == "confirm" and (HERE / "bars_rv2d.json").exists() else None
    rows = load(phase, frozen)
    if not rows:
        sys.exit(f"no cell records for phase {phase} in {R.CELLS} with the code stamp {frozen or R.code_stamp()}")
    consts = {}
    cpath = HERE / ("bars_rv2d.json" if phase == "confirm" else "dev_constants.json")
    if cpath.exists():
        c = json.loads(cpath.read_text(encoding="utf-8"))
        consts = c.get("constants", c)
    setting = consts.get("router_setting", "restrict-0.01")
    md = [f"# RV2-D {phase} results (generated by report_rv2d.py from {len(rows)} cell records; do not edit by hand)\n", f"Constants in force: {json.dumps(consts, sort_keys=True)}\n",
          f"Code stamp {frozen or R.code_stamp()}; cells of other code left out: {STALE['cells']}.\n"]
    if phase == "confirm":
        from growmoe.verdicts_rv4 import rv2d_decision, rv2d_verdicts
        bars = json.loads((HERE / "bars_rv2d.json").read_text(encoding="utf-8"))
        res = rv2d_verdicts(rows, {**bars["bars"], "router_setting": setting}, bars["golden"], bars.get("expected"))
        md += ["## Statements\n", "| ID | Rule | Verdict | Expected |", "|---|---|---|---|"] + [f"| {k} | {v['rule']} | **{v['verdict']}** | {v['expected']} |" for k, v in res.items()]
        md += ["", "Decision: " + rv2d_decision(res)["text"], ""]
        (HERE / "verdicts_rv2d.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md += section_router(rows, setting)
    md += section_arm(rows, "S3+T", "Transfer rule (S3+T minus S3)", ("experts and events", transfer_extra))
    md += section_arm(rows, "S3+G", "Growth from the fit gap (S3+G minus S3)", ("growth", growth_extra))
    md += section_arm(rows, "S3+T+G", "Both together (S3+T+G minus S3)", ("experts and events", transfer_extra))
    md += section_stack(rows, setting)
    if phase == "dev":
        md += section_gamma(rows)
    md += section_cost(rows)
    out = HERE / ("RESULTS_DEV.md" if phase == "dev" else "RESULTS_RV2D.md")
    R.atomic_write_text(out, "\n".join(md) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()

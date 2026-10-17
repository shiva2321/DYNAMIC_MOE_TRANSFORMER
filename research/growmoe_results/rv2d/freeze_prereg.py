"""Freeze the RV2-D pre-registration (routing v2 stage 4): bars_rv2d.json (+ its sha256) and PREREG.md. Written from the dev results BEFORE any confirmatory seed (85-99) has run.

    python freeze_prereg.py            # refuses to run when bars_rv2d.sha256 exists: a change after the freeze is a recorded amendment, not a rerun

What is frozen: the numeric bars of the statements S4-1 to S4-9 (they come from the spec, not from dev), the constants chosen on dev seeds by the declared rules (dev_constants.json, plus the
optional constants_override.json, whose every entry needs a written reason), the arms, streams and seeds, the dev verdicts as the expected outcomes, the golden check, and the code stamp:
the confirmatory cells are only accepted when they were produced by exactly this code.
"""

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rv2d_lib as R  # noqa: E402
from growmoe.verdicts_rv4 import BARS_RV2D, rv2d_verdicts  # noqa: E402
from report_rv2d import folder_stamp, load  # noqa: E402
import make_devview  # noqa: E402
import run_m0 as M0  # noqa: E402

ROOT = R.ROOT
SPEC = "docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md"
AMENDMENTS = "docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md"
PROVENANCE = ["growmoe/pools/phylo.py", "growmoe/router.py", "growmoe/transfer.py", "growmoe/gapgrow.py", "growmoe/config.py", "growmoe/accounting.py", "growmoe/verdicts_rv4.py",
              "growmoe/experts.py", "research/growmoe_results/rv2d/rv2d_lib.py", "research/growmoe_results/rv2d/rv2d_select.py", "research/growmoe_results/rv2d/make_cells.py",
              "research/growmoe_results/rv2d/report_rv2d.py", "research/growmoe_results/rv2d/make_devview.py", AMENDMENTS, "research/growmoe_results/rv2c/golden_flags_off.json"]


def sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12] if p.exists() else "missing"


def golden_now() -> dict:
    spec = importlib.util.spec_from_file_location("golden", ROOT / "research" / "growmoe_results" / "rv2c" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"differences": mod.differences(), "recorded_sha256": sha("research/growmoe_results/rv2c/golden_flags_off.json")}


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE           # --out DIR: rehearse the freeze in a scratch folder (the real one can be done only once)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "bars_rv2d.sha256").exists():
        sys.exit("bars_rv2d.json is already frozen (bars_rv2d.sha256 exists): freeze_prereg cannot rewrite it. A change after the freeze is a recorded amendment, not a rerun.")
    dev = json.loads((HERE / "dev_constants.json").read_text(encoding="utf-8"))
    constants, overrides = dict(dev["constants"]), {}
    if (HERE / "constants_override.json").exists():
        overrides = json.loads((HERE / "constants_override.json").read_text(encoding="utf-8"))
        for k, v in overrides.items():
            assert isinstance(v, dict) and v.get("reason"), f"override {k} needs a written reason"
            constants[k] = v["value"]
    if "router_setting" in overrides:
        constants["router_mode"], constants["router_margin"] = constants["router_setting"].split("-")[0], float(constants["router_setting"].split("-")[1])
    bars = {k: list(v) if isinstance(v, tuple) else v for k, v in BARS_RV2D.items()}
    make_devview.main()                                                                         # the dev rows: dev1 and dev2 records of the arithmetic and banking families, the function-regime family without the head start (A4-2)
    R.CELLS = make_devview.VIEW
    dev_stamp = folder_stamp(R.CELLS)                                                           # the dev cells were produced by the stage-4 build before stage 4.1 and A4-2 were merged: their own recorded stamp
    rows = load("dev", dev_stamp)
    golden = golden_now()
    dev_verdicts = rv2d_verdicts(rows, {**BARS_RV2D, "router_setting": constants["router_setting"]}, golden)
    expected = {k: ("PASS" if v["verdict"] == "NOT MEASURABLE" else v["verdict"]) for k, v in dev_verdicts.items()}          # dev cannot measure what needs the fresh run: the expectation is PASS
    stamp = R.code_stamp()
    payload = {"bars": bars, "constants": constants, "overrides": overrides, "arms": list(R.ARMS), "streams": R.STREAMS,
               "seeds": {"s3": list(range(85, 100)), "arith-seq": list(range(85, 91)), "banking": list(range(85, 90)), "arith-mixed": list(range(85, 100))},
               "expected": expected, "dev_verdicts": {k: v["verdict"] for k, v in dev_verdicts.items()}, "golden": golden, "code_stamp": stamp, "dev_stamp": dev_stamp, "dev_cells": len(rows),
               "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    bars_path = out / "bars_rv2d.json"
    bars_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    digest = M0.freeze_check(bars_path)
    rules = {k: v["rule"] for k, v in dev_verdicts.items()}
    c = constants
    lines = [
        "# RV2-D pre-registration (generated by freeze_prereg.py; written before any confirmatory seed ran)\n",
        f"Spec: `{SPEC}` (sections 4 to 9 and 13) with the amendments of `{AMENDMENTS}` (A4-1 changes the denominator of S4-5; A4-2 the pool seed of the function-regime streams). "
        f"Plan: `docs/superpowers/plans/2026-10-15-routing-v2-stage4.md`. Build record: notebook E-033; dev runs (code stamp {dev_stamp}): `RESULTS_DEV_REVIEW.md`, `DEV_SELECTION.md`. "
        f"Confirmatory seeds 85-99 (single use): the ten stage-3 streams 15 seeds, arithmetic operator-by-operator 85-90, banking77 85-89, arithmetic mixed from 16 and from 128 units 15 seeds. "
        f"Dev seeds are 100 and above.\n",
        "## Arms\n",
        "`S3` = the stage-3 pool as it stands (batch-level gate on, breaker 12 / 0.5 / 100, persistence 2, group veto 0.2; the budget tuned for A_vote); `S3+T` adds `spawn_rule=\"transfer\"`; `S3+G` adds "
        "`growth_rule=\"gap\"`; `S3+T+G` both. The router (`router=True`) is evaluated on the SAME trained pools of every arm (it acts at answer time only and never influences training), so router off "
        "and on are paired by construction. Arithmetic: hidden 512, replay 20,000, 5 inner steps (mixed streams: hidden 16 or 128). The function-regime streams use the pool seed `stream seed + 7` "
        "(A4-2); arithmetic and banking77 keep the stream seed.\n",
        "## Constants (chosen on dev seeds by the declared rules; the curves are in `DEV_SELECTION.md`)\n",
        f"- `grow_gap` = {c['grow_gap']} (rule: spec 6.7, score = accuracy gain minus 0.1 x relative parameter increase over three groups, subject to the guards of S4-7); "
        f"`grow_guard_eps` = {c.get('grow_guard_eps', 0.02)}, `grow_guard_window` = {c.get('grow_guard_window', 100)}\n"
        f"- `transfer_margin` = {c['transfer_margin']} (rule: best arithmetic gain among the margins that keep the guard streams within -0.5 points and 1.2 x the experts; ties to 0.25), `transfer_alpha` = 0.01\n"
        f"- router: mode `{c['router_mode']}`, margin {c['router_margin']} (rule: best mean gain on arithmetic and OV-0.85 among the settings that hurt no guard stream by more than 0.5 points)\n"
        f"- overrides applied (each with its reason in `constants_override.json`): {json.dumps(overrides) if overrides else 'none'}\n"
        f"- `bars_rv2d.json` sha256 {digest[:16]}; code stamp {stamp}\n",
        "## Statements (bars from the spec; expected = the dev verdicts, NOT MEASURABLE counted as PASS)\n",
        "| ID | Rule | Dev verdict | Expected |", "|---|---|---|---|"] + [f"| {k} | {rules[k]} | {payload['dev_verdicts'][k]} | {expected[k]} |" for k in rules] + [
        "",
        "## Decision rules (verbatim from the spec, section 8)\n",
        "Each component is adopted on its own: 4a if S4-1 or S4-2 and S4-3 pass; 4b if S4-4 and S4-5 pass; 4c if S4-6 and S4-7 pass; adopting means the default flips, the old entry points are pinned in the "
        "experiment scripts, and notes go wherever the old behaviour was documented. Kill rules: if a guard fails the component stays OFF and the others proceed; if the router's selection picks a worse "
        "rule than density on any guard stream by more than 0.5 points the margin is raised to 0.03 and the guard rerun once; the stage-3 statements S3-1 to S3-9 must still pass with every adopted "
        "component (golden and the stage-3 tests). S4-9 failing is a bug, not a result: stop. S4-8 (the full stack) flags the adoption when it misses.\n",
        "## Reported, not gated\n",
        "CPU seconds per arm; experts, spawns and extends per stream; growth events, reverts and widths per run; the router rule selected per stream and its cross-validated accuracies; the cost of the extra "
        "forward pass of the growth rule; the merge audit (E-033, F-28); a transfer-rule diagnostic on the one-function arithmetic stream (dev only).\n",
        "## What is claimed, and what is not\n",
        "Only what the tested streams show: the router's gain is a property of operator-like and heavily overlapping regimes; the growth threshold is an operating point; the transfer rule cannot rely on merge "
        "to clean up false spawns (a wrongly founded expert is permanent), which is why the expert counts are gated (S4-5). Nothing here concerns streams, widths or budgets that were not run.\n",
        "## Provenance (sha256, first 12 hex digits)\n",
        "; ".join(f"{p} {sha(p)}" for p in PROVENANCE) + f"; this script {sha('research/growmoe_results/rv2d/freeze_prereg.py')}.\n"]
    (out / "PREREG.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"bars_rv2d.json frozen, sha256 {digest[:16]}, code stamp {stamp}")
    print("expected", expected)


if __name__ == "__main__":
    main()

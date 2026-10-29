"""The audit of the confirmatory run (Task 16), written from the SPEC and the frozen files, not from the code under test (anti-goal A4). It lives outside the frozen code stamp (this folder is not
globbed) because it reads results and never makes one.

    PYTHONPATH=<repo root> python research/knowledge_status/post_run/audit.py

It lists problems, one per line, and exits 1 if there are any:
  * every statement id that spec section 7 names has a verdict, and each statement is judged on exactly the measures the FROZEN record lists for it (not the measures the verdict code knows);
  * what was frozen (bars.json, bars.sha256, PREREG.md, pool, code) is still what is on disk after the run;
  * the run was started and closed under that freeze; every cell exists, carries the stamp of that freeze, and belongs to seeds 400 to 429 (and nothing else is there); an unreadable cell is named;
  * every measure, every measure verdict, every statement verdict (all must pass, any fail fails, B2 by its rule, B3b reported) and the three consequences of the kill rule are RECOMPUTED here from the raw cell
    files and the frozen bars with code of this file, and compared with verdict.json.
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-10-26-knowledge-status-and-routing-design.md"
TOL = 1e-12
DECOYS = ("mul_wrong_steps", "div_wrong_steps", "max_irrelevant_valid_steps", "mod_unrelated_skill")


def spec_statements(spec: Path = SPEC) -> List[str]:
    section = spec.read_text(encoding="utf-8").split("## 7. Measurements and draft bars")[1].split("## 8.")[0]
    return sorted(set(re.findall(r"\bB\d[abc]\b|\bB[2567]\b", section)))


def _read(path: Path, problems: List[str]) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        problems.append(f"cannot read cell {path.name}: {e}")
        return None


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x) else None


def _mean(v: List[float]) -> Optional[float]:
    return sum(v) / len(v) if v else None


def raw_measures(out: Path, rec: dict, problems: List[str]) -> Dict[str, Optional[float]]:
    """Every measure of the pre-registration, from the raw cell files, by the definitions of spec section 7 (written here, not imported)."""
    head, q = rec["headline_representation"], f"q{rec['q']:g}"
    vals: Dict[str, List[float]] = {}

    def add(name, v):
        v = _num(v)
        if v is not None:
            vals.setdefault(name, []).append(v)

    for p in sorted((out / "e1").glob(f"{head}__s*.json")):
        c = _read(p, problems)
        if c:
            for k in ("b1a", "b1b", "b1c"):
                add(k, c.get(k))
    for p in sorted((out / "e1").glob("R0_hash__s*.json")):
        c = _read(p, problems)
        if c:
            add("r0_b1a", c.get("b1a"))
            add("r0_b1b", c.get("b1b"))
    for p in sorted((out / "e2").glob(f"{head}__s*.json")):
        c = _read(p, problems)
        if not c:
            continue
        try:
            s = c["machine"][q]["standard"]
            t = c["tangled"]["machine"]
            add("macro_f1", s["macro_f1"]["macro_f1"])
            for k in ("false_unseen_heldout", "thin_vs_solid_auroc", "other_as_unseen", "trap_as_unseen"):
                add(k, s.get(k))
            add("b4a_residual_hit", t["gold"]["b4a"]["residual_hit"])
            add("b4a_known_not_flagged", t["gold"]["b4a"]["known_not_flagged"])
            add("b4b_nothing_flagged", t["gold"]["b4b"]["nothing_flagged"])
            add("b4b_other_labelled", t["gold"]["b4b"]["other_labelled"])
            auto = t[rec["b6_primary"]]["b4a"]
            add("b6_gap", max(abs(t["gold"]["b4a"][k] - auto[k]) for k in ("residual_hit", "known_not_flagged")))
        except (KeyError, TypeError) as e:
            problems.append(f"cell {p.name} lacks {e}")
    for p in sorted((out / "e4").glob("s*.json")):
        c = _read(p, problems)
        if not c:
            continue
        try:
            r = c["rates"]
            add("b5_mul_parent", r["mul"]["parent_is_add"])
            add("b5_div_parent", r["div"]["parent_is_sub"])
            add("b5_decoy_worst", max(r[k]["any_parent"] for k in DECOYS))
        except (KeyError, TypeError) as e:
            problems.append(f"cell {p.name} lacks {e}")
    for p in sorted((out / "realism").glob("s*.json")):
        c = _read(p, problems)
        if c and _num(c.get("diff")) is not None:
            add("b7_diff", c["diff"])
    out_m: Dict[str, Optional[float]] = {}
    for k, v in vals.items():
        if len(v) >= rec["min_seeds"]:
            out_m[k] = _mean(v)
    if "b7_diff" in out_m:
        out_m["b7_gap"] = abs(out_m.pop("b7_diff"))
    return out_m


def judge(value: Optional[float], bar: float, direction: str) -> str:
    if value is None:
        return "not_run"
    ok = value >= bar - TOL if direction == "min" else value <= bar + TOL
    return "pass" if ok else "fail"


def audit(root: Path = HERE, spec: Path = SPEC, pool_file: Optional[Path] = None, code_files=None, stamps=None) -> List[str]:
    import freeze_prereg as FZ
    import run_confirm as RC
    import runlib
    problems: List[str] = []
    root = Path(root)
    out = root / "results" / "confirm"
    try:
        rec = json.loads((root / "bars.json").read_text(encoding="utf-8"))
        result = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"cannot read bars.json or results/confirm/verdict.json: {e}"]
    wanted = spec_statements(spec)
    got = sorted(result["verdict"]["statements"])
    if got != wanted:
        problems.append(f"statements judged {got} differ from the statements of spec section 7 {wanted}")
    # each statement is judged on exactly the measures the frozen record lists for it
    frozen_by_statement = {sid: sorted(m for m in st["measures"] if m in rec["bars"] and m != "b2_gap") for sid, st in rec["statements"].items()}
    judged_by_statement: Dict[str, List[str]] = {}
    for name, row in result["verdict"]["measures"].items():
        judged_by_statement.setdefault(row["statement"], []).append(name)
    for sid, names in frozen_by_statement.items():
        if sid in ("B2", "B3b"):
            continue
        if sorted(judged_by_statement.get(sid, [])) != names:
            problems.append(f"{sid} was judged on {sorted(judged_by_statement.get(sid, []))}, the frozen record lists {names}")
    chk = FZ.check_frozen(root, pool_file if pool_file is not None else root / "pool_confirm.py", code_files if code_files is not None else FZ.default_code_files())
    if not chk["ok"]:
        problems.append(f"the frozen files changed: {chk['problem']}")
    for name in ("CONFIRM_STARTED.json", "CONFIRM_CLOSED.json"):
        if not (root / name).exists():
            problems.append(f"{name} is missing")
    try:
        started = json.loads((root / "CONFIRM_STARTED.json").read_text(encoding="utf-8"))
        if started.get("bars_sha256") != FZ.bars_hash(rec):
            problems.append("the run was started under another freeze")
        e4 = started.get("e4_seeds") or []
    except (OSError, ValueError):
        e4 = []
    st = stamps if stamps is not None else RC.stamps(rec)
    seeds = rec["seeds"]["confirm"]
    for job in RC.jobs(rec, e4):
        path = RC.cell_path(out, job)
        s = st[job[0]]
        s = s(job[1]) if callable(s) else s
        if not path.exists():
            problems.append(f"missing cell {path.relative_to(out)}")
        elif not runlib.cell_done(path, s):
            problems.append(f"cell {path.relative_to(out)} is not stamped with this freeze")
    known = {RC.cell_path(out, j) for j in RC.jobs(rec, e4)}
    for p in out.rglob("*.json"):
        if p not in known and p.name != "verdict.json":
            problems.append(f"unexpected file {p.relative_to(out)}")
    for p in out.glob("*/*.json"):
        m = re.search(r"s(\d+)\.json$", p.name)
        if m and int(m.group(1)) not in seeds:
            problems.append(f"{p.relative_to(out)} is outside seeds {seeds[0]} to {seeds[-1]}")
    # recompute everything from the raw cells and the frozen bars
    means = raw_measures(out, rec, problems)
    measures: Dict[str, str] = {}
    for name, row in result["verdict"]["measures"].items():
        bar, direction = rec["bars"][name]["bar"], rec["bars"][name]["direction"]
        v = means.get(name)
        measures[name] = judge(v, bar, direction)
        if row["verdict"] != measures[name]:
            problems.append(f"{name}: verdict.json says {row['verdict']}, the cells and the frozen bar ({direction} {bar}) give {measures[name]} (value {v})")
        if v is not None and row["value"] is not None and abs(row["value"] - v) > 1e-9:
            problems.append(f"{name}: verdict.json holds the value {row['value']}, the cells give {v}")
    for sid in frozen_by_statement:
        if sid == "B3b":
            want = "reported"
        elif sid == "B2":
            gaps = [None if means.get(r0) is None else min(rec["bars"][b]["bar"], rec["bars"][b]["draft"]) - means[r0] for r0, b in (("r0_b1a", "b1a"), ("r0_b1b", "b1b"))]
            margin = rec["bars"]["b2_gap"]["bar"]
            want = "pass" if any(g is not None and g >= margin - TOL for g in gaps) else ("not_run" if any(g is None for g in gaps) else "fail")
        else:
            parts = [measures[m] for m in frozen_by_statement[sid] if m in measures]
            want = "fail" if "fail" in parts else ("pass" if parts and all(p == "pass" for p in parts) else "not_run")
        if result["verdict"]["statements"].get(sid) != want:
            problems.append(f"{sid}: verdict.json says {result['verdict']['statements'].get(sid)}, recomputing gives {want}")
    st_v = result["verdict"]["statements"]
    b1b = means.get("b1b")
    expect = {"invalid": st_v.get("B2") == "fail", "status_void": b1b is not None and b1b < min(rec["bars"]["b1b"]["draft"], rec["bars"]["b1b"]["bar"]) - TOL,
              "conditional": "fail" in (st_v.get("B1a"), st_v.get("B1c"))}
    if result.get("consequences") != expect:
        problems.append(f"consequences in verdict.json {result.get('consequences')} differ from {expect} recomputed from the cells")
    return problems


if __name__ == "__main__":
    found = audit()
    for line in found:
        print("PROBLEM:", line)
    print(f"{len(found)} problems")
    sys.exit(1 if found else 0)

"""The audit of the confirmatory run (Task 16), written from the SPEC and the frozen files, not from the code under test (anti-goal A4). It lives outside the frozen code stamp (this folder is not
globbed) because it reads results and never makes one.

    PYTHONPATH=<repo root> python research/knowledge_status/post_run/audit.py

It lists problems, one per line, and exits 1 if there are any:
  * every statement id that spec section 7 names has a verdict, and nothing else is judged;
  * what was frozen (bars.json, bars.sha256, PREREG.md, pool, code) is still what is on disk after the run;
  * the run was started and closed under that freeze; every cell of the run exists, carries the stamp of that freeze, and belongs to seeds 400 to 429 (and to nothing else);
  * the verdicts in verdict.json are what the frozen bars and the cells give when recomputed here from the cell files, by the rule of the spec (at or above / at or below the frozen bar).
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-10-26-knowledge-status-and-routing-design.md"


def spec_statements(spec: Path = SPEC) -> List[str]:
    section = spec.read_text(encoding="utf-8").split("## 7. Measurements and draft bars")[1].split("## 8.")[0]
    return sorted(set(re.findall(r"\bB\d[abc]\b|\bB[2567]\b", section)))


def audit(root: Path = HERE, spec: Path = SPEC, pool_file: Optional[Path] = None, code_files=None, stamps=None) -> List[str]:
    import freeze_prereg as FZ
    import run_confirm as RC
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
    import runlib
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
    # recompute the verdict from the frozen bars with the plain rule of the spec
    means = FZ.derive_means(FZ.collect_dev_values(out, st, rec["headline_representation"], seeds=seeds, pool="confirm", require=("e1", "e2")), rec["min_seeds"])
    for name, row in result["verdict"]["measures"].items():
        bar, direction = rec["bars"][name]["bar"], rec["bars"][name]["direction"]
        v = means.get(name)
        want = "not_run" if v is None else ("pass" if (v >= bar - 1e-12 if direction == "min" else v <= bar + 1e-12) else "fail")
        if row["verdict"] != want:
            problems.append(f"{name}: verdict.json says {row['verdict']}, the cells and the frozen bar ({direction} {bar}) give {want} (value {v})")
    return problems


if __name__ == "__main__":
    found = audit()
    for line in found:
        print("PROBLEM:", line)
    print(f"{len(found)} problems")
    sys.exit(1 if found else 0)

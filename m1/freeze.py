"""The freeze of the M1 preregistration: the report seeds open only after docs/roadmap/M1_FREEZE.json names the exact bytes of the files that fix the scored run.

The freeze file holds the SHA-256 of the preregistration, of m1/config_m1.py and of the tuning files (the operating points), plus the code files for the record.
A missing freeze file, a frozen file that changed afterwards, or a seed outside the report seeds refuses the run. The freeze file is created once and never
overwritten: a change needs a logged decision made before the results it would judge are seen (preregistration, header).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from costlab.env import environment
from m1.config_m1 import DEV_SEEDS, REPORT_SEEDS, RESERVE_SEEDS

REPO_ROOT = Path(__file__).resolve().parent.parent
FREEZE_FILE = "docs/roadmap/M1_FREEZE.json"

# Changing any of these after the freeze refuses the run.
FROZEN_FILES = (
    "docs/roadmap/M1_PREREG_DRAFT.md",
    "m1/config_m1.py",
    "results/m1/tuning/B1.json",
    "results/m1/tuning/B2.json",
    "results/m1/tuning/B3seq.json",
    "results/m1/tuning/B3mix.json",
    "results/m1/tuning/B3drift.json",
)

# Recorded for the audit trail; a repair of the report script after the freeze is a logged decision, not a refusal.
RECORDED_FILES = (
    "m1/arms.py",
    "m1/beds.py",
    "m1/cell.py",
    "m1/queue.py",
    "m1/report.py",
    "m1/verdict.py",
    "m1/freeze.py",
    "m1/scored_run.sh",
)


class FreezeError(ValueError):
    """The report seeds are not open (no freeze, a changed frozen file, or a seed that is not a report seed)."""


def file_digest(path: Path) -> str:
    """SHA-256 of the file with CRLF read as LF, so a git checkout on Windows cannot change it."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _digests(root: Path, names: Iterable[str]) -> Dict[str, str]:
    return {name: file_digest(root / name) for name in names}


def freeze_digest(root: Path = REPO_ROOT) -> Optional[str]:
    """SHA-256 of the freeze file itself (stored in every report cell), or None when there is no freeze."""
    path = root / FREEZE_FILE
    return file_digest(path) if path.exists() else None


def write_freeze(root: Path = REPO_ROOT, readings: Optional[Dict[str, Any]] = None, run_plan: Optional[Dict[str, Any]] = None) -> Path:
    """Create the freeze file exclusively; an existing freeze is never overwritten."""
    path = root / FREEZE_FILE
    missing = [n for n in FROZEN_FILES if not (root / n).exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze: missing {missing}")
    record = {
        "schema": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": environment().get("git_commit") or "unknown",
        "frozen": _digests(root, FROZEN_FILES),
        "recorded": _digests(root, [n for n in RECORDED_FILES if (root / n).exists()]),
        "dev_seeds": list(DEV_SEEDS),
        "report_seeds": list(REPORT_SEEDS),
        "reserve_seeds": list(RESERVE_SEEDS),
        "readings": readings or {},
        "run_plan": run_plan or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return path


def check_frozen(seeds: Optional[Iterable[int]] = None, root: Path = REPO_ROOT) -> List[str]:
    """Problems that keep the report seeds closed (empty list: open)."""
    path = root / FREEZE_FILE
    if not path.exists():
        return [f"the preregistration is not frozen ({FREEZE_FILE} does not exist)"]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        frozen = dict(record["frozen"])
    except (OSError, ValueError, KeyError) as exc:
        return [f"{FREEZE_FILE} cannot be read ({exc})"]
    problems = []
    for name in FROZEN_FILES:
        if name not in frozen:
            problems.append(f"{name} is not named in the freeze")
        elif not (root / name).exists():
            problems.append(f"{name} no longer exists")
        elif file_digest(root / name) != frozen[name]:
            problems.append(f"{name} changed after the freeze")
    if seeds is not None:
        problems += [f"seed {s} is not a report seed (report: {REPORT_SEEDS[0]}-{REPORT_SEEDS[-1]}; {RESERVE_SEEDS[0]}-{RESERVE_SEEDS[-1]} stay in reserve)" for s in sorted(set(seeds)) if s not in REPORT_SEEDS]
    return problems


def require_frozen(seeds: Optional[Iterable[int]] = None, root: Path = REPO_ROOT) -> None:
    problems = check_frozen(seeds, root)
    if problems:
        raise FreezeError("Stage B refuses --split report: " + "; ".join(problems) + ". Report seeds are reserved until the freeze exists and matches.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create or check the freeze of the M1 preregistration.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="create docs/roadmap/M1_FREEZE.json (refuses to overwrite)")
    w.add_argument("--readings", required=True, help="JSON file with the readings logged at the freeze")
    w.add_argument("--run-plan", required=True, help="JSON file with the run plan (timed and deterministic seeds, profiles)")
    sub.add_parser("check", help="verify the frozen files against the freeze")
    args = parser.parse_args(argv)
    if args.cmd == "write":
        path = write_freeze(
            readings=json.loads(Path(args.readings).read_text(encoding="utf-8")),
            run_plan=json.loads(Path(args.run_plan).read_text(encoding="utf-8")),
        )
        print(f"froze: {path}")
        return
    problems = check_frozen(REPORT_SEEDS)
    print("freeze OK" if not problems else "freeze NOT OK:\n  " + "\n  ".join(problems))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

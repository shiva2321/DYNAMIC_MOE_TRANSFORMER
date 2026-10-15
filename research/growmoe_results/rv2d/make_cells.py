"""Write the cell lists. dev1: baselines + the calibration sweeps. dev2: every arm at the chosen constants. confirm: seeds 85-99 (single use), constants frozen in bars_rv2d.json.

    python make_cells.py dev1 | dev2 | confirm
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rv2d_lib import ARMS, GAP_GRID, STREAMS, TRANSFER_GRID, Cell  # noqa: E402

DEV_ARITH, DEV_MIXED, DEV_S3, DEV_BANK = (100, 101), (100, 101, 102), (100, 101, 102), (100, 101)
CALIBRATION_STREAMS = ("F-15", "OV-0.6")                                   # the two stage-3 families of the gamma calibration (spec 6.7)
TRANSFER_STREAMS = ("F-15", "OV-0.6", "OV-0.85", "S1", "S2-5")             # guard streams for the transfer margin


def dev1():
    cells = [Cell("arith-seq", "-", "S3", s) for s in DEV_ARITH] + [Cell(f, "-", "S3", s) for f in ("arith-mixed16", "arith-mixed128") for s in DEV_MIXED]
    cells += [Cell("s3", st, "S3", s) for st in STREAMS for s in DEV_S3] + [Cell("banking", "-", "S3", s) for s in DEV_BANK]
    cells += [Cell("arith-seq", "-", "S3+T", s, tmargin=t) for t in TRANSFER_GRID for s in DEV_ARITH]
    cells += [Cell("s3", st, "S3+T", s, tmargin=t) for st in TRANSFER_STREAMS for t in TRANSFER_GRID for s in DEV_S3]
    cells += [Cell("banking", "-", "S3+T", s, tmargin=t) for t in TRANSFER_GRID for s in DEV_BANK]
    cells += [Cell(f, "-", "S3+G", s, gap=g) for f in ("arith-mixed16", "arith-mixed128") for g in GAP_GRID for s in DEV_MIXED]
    cells += [Cell("s3", st, "S3+G", s, gap=g) for st in CALIBRATION_STREAMS for g in GAP_GRID for s in DEV_S3]
    return cells


def dev2():
    c = json.loads((HERE / "dev_constants.json").read_text(encoding="utf-8"))
    g, t = c["grow_gap"], c["transfer_margin"]
    cells = []
    for arm in ("S3+T", "S3+G", "S3+T+G"):
        kw = dict(gap=g if "G" in arm else 0.0, tmargin=t if "T" in arm else 0.0)
        cells += [Cell("arith-seq", "-", arm, s, **kw) for s in DEV_ARITH]
        cells += [Cell("s3", st, arm, s, **kw) for st in STREAMS for s in DEV_S3]
        cells += [Cell("banking", "-", arm, s, **kw) for s in DEV_BANK]
        if "T" not in arm:
            cells += [Cell(f, "-", arm, s, **kw) for f in ("arith-mixed16", "arith-mixed128") for s in DEV_MIXED]
        else:
            cells += [Cell("arith-mixed128", "-", arm, s, **kw) for s in DEV_MIXED]            # diagnostic (dev only): does the transfer rule found false experts on a one-function stream?
    return cells


def confirm():
    c = json.loads((HERE / "bars_rv2d.json").read_text(encoding="utf-8"))["constants"]                # frozen before any confirmatory seed
    g, t = c["grow_gap"], c["transfer_margin"]
    all15, six, five = tuple(range(85, 100)), tuple(range(85, 91)), tuple(range(85, 90))
    cells = []
    for arm in ARMS:
        kw = dict(gap=g if "G" in arm else 0.0, tmargin=t if "T" in arm else 0.0, phase="confirm")
        cells += [Cell("arith-seq", "-", arm, s, **kw) for s in six]
        cells += [Cell("s3", st, arm, s, **kw) for st in STREAMS for s in all15]
        cells += [Cell("banking", "-", arm, s, **kw) for s in five]
        if "T" not in arm:
            cells += [Cell(f, "-", arm, s, **kw) for f in ("arith-mixed16", "arith-mixed128") for s in all15]
    return cells


def smoke():
    """One cell of every family (run with RV2D_SMOKE=1: tiny streams, a separate result folder)."""
    return [Cell("arith-seq", "-", "S3+T+G", 100, gap=0.20, tmargin=0.25), Cell("arith-mixed16", "-", "S3+G", 100, gap=0.20), Cell("arith-mixed128", "-", "S3", 100),
            Cell("s3", "F-15", "S3+T+G", 100, gap=0.20, tmargin=0.25), Cell("s3", "OV-0.6", "S3", 100)]


if __name__ == "__main__":
    which = sys.argv[1]
    cells = {"dev1": dev1, "dev2": dev2, "confirm": confirm, "smoke": smoke}[which]()
    seen, uniq = set(), []
    for c in cells:
        if c.cid not in seen:
            seen.add(c.cid)
            uniq.append(c)
    (HERE / f"cells_{which}.json").write_text(json.dumps([asdict(c) for c in uniq], indent=0), encoding="utf-8")
    print(f"{which}: {len(uniq)} cells")

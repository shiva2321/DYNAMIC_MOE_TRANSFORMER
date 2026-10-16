"""Run ONE cell and write its result atomically. Safe to run again: a finished cell is skipped.   python run_cell.py --phase dev1 --cid <cid>"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rv2d_lib as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--cid", required=True)
    a = ap.parse_args()
    out = R.CELLS / f"{a.cid}.json"
    stamp = R.code_stamp()                                            # taken before the run: the result is valid for the code that was there when the cell started
    if R.cell_done(out, stamp):
        return
    cells = {R.Cell(**c).cid: R.Cell(**c) for c in json.loads((HERE / f"cells_{a.phase}.json").read_text(encoding="utf-8"))}
    R.CELLS.mkdir(exist_ok=True)
    cpu0 = time.process_time()                                        # CPU seconds of this process (all threads): a cost measure that a sleeping PC or busy neighbours cannot distort,
    rec = R.run_cell(cells[a.cid])                                    # unlike the wall-clock "seconds" (a cell that was frozen overnight reported 78,000 of them). A resumed cell counts its own part only.
    rec["cpu_seconds"] = time.process_time() - cpu0
    rec["code_stamp"] = stamp
    R.atomic_write_text(out, json.dumps(rec))


if __name__ == "__main__":
    main()

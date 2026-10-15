"""Run ONE cell and write its result atomically. Safe to run again: a finished cell is skipped.   python run_cell.py --phase dev1 --cid <cid>"""

import argparse
import json
import sys
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
    if R.valid_json(out):
        return
    cells = {R.Cell(**c).cid: R.Cell(**c) for c in json.loads((HERE / f"cells_{a.phase}.json").read_text(encoding="utf-8"))}
    R.CELLS.mkdir(exist_ok=True)
    R.atomic_write_text(out, json.dumps(R.run_cell(cells[a.cid])))


if __name__ == "__main__":
    main()

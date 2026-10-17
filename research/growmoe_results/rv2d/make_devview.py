"""The dev review looks at the function-regime cells WITHOUT the seed-aliasing head start (E-035, E-038). This builds `cells_devview/`: every cell record of `cells/` (dev1 and dev2) whose family is not
`s3`, plus every record of `cells_s3u/` (the function-regime family run with the pool seed shifted by 7). The folder is a copy (regenerate it, never edit it); the report is then

    python make_devview.py
    python report_rv2d.py dev --cells cells_devview --out RESULTS_DEV_REVIEW.md
"""

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEW = HERE / "cells_devview"


def main() -> None:
    VIEW.mkdir(exist_ok=True)
    for old in VIEW.glob("*"):                                         # empty the folder instead of removing it (Windows refuses to remove a folder some other program has open)
        old.unlink()
    kept = dropped = 0
    for f in sorted((HERE / "cells").glob("*.json")):
        if f.name.startswith("s3__"):
            dropped += 1
            continue
        shutil.copy2(f, VIEW / f.name)
        kept += 1
    added = 0
    for f in sorted((HERE / "cells_s3u").glob("*.json")):
        shutil.copy2(f, VIEW / f.name)
        added += 1
    print(f"cells_devview: {kept} records of other families from cells/, {added} unaliased function-regime records from cells_s3u/; {dropped} aliased function-regime records left out")


if __name__ == "__main__":
    main()

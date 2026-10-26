"""Run helpers for a PC that loses power: atomic writes, a code stamp, and a "is this cell finished" check (pattern: research/growmoe_results/rv2d/rv2d_lib.py)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


def atomic_write_text(path, text: str) -> None:
    """Write ``text`` so that a reader sees either the old file or the complete new one, never half of it."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def code_stamp(files: Sequence[Path]) -> str:
    """Short sha256 of the given source files (CRLF folded to LF, so Windows and Unix checkouts agree). A result is valid only for the code that made it."""
    h = hashlib.sha256()
    for f in files:
        f = Path(f)
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def cell_done(path, stamp: str) -> bool:
    """A cell is finished only if its file exists, parses as a JSON object and was written by the code with this stamp."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(rec, dict) and rec.get("code_stamp") == stamp


def set_threads(n: int = 3) -> None:
    """Keep the PC usable: a few torch threads per job."""
    import torch
    torch.set_num_threads(n)

"""Small helpers for cells that survive a power cut: atomic writes and a code stamp that invalidates a cached cell when the code that made it changed."""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def code_stamp(paths: Iterable) -> str:
    h = hashlib.sha1()
    for p in sorted(str(p) for p in paths):
        h.update(p.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def cell_done(json_path, stamp: str) -> bool:
    p = Path(json_path)
    if not p.exists() or not p.with_suffix(".npz").exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("code_stamp") == stamp
    except (OSError, ValueError):
        return False

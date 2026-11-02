"""Far novelty: text from registers that no domain of the test bed teaches (local Hugging Face cache only, this project never downloads).

Used as the "everyday or off-topic input" every recognizer must refuse: children's stories, encyclopedic sentences, school science questions, news, newsgroup posts, emotion sentences.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from unimoe.domains import _clip, _file, _raw
from .base import rng_for


def stdlib_code(n: int, seed: int = 0, min_lines: int = 4, max_lines: int = 18) -> List[str]:
    """Real functions from the local Python standard library (short ones): code the test bed never teaches. The data files of the cached code datasets are not on this PC, this is what is."""
    import ast
    import sysconfig
    import textwrap
    from pathlib import Path
    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.glob("*.py") if not p.name.startswith(("test", "this", "antigravity", "__")))
    rng = rng_for(seed, "stdlib")
    out: List[str] = []
    for k in rng.permutation(len(files)):
        try:
            src = files[int(k)].read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                seg = ast.get_source_segment(src, node)
                if seg and min_lines <= seg.count("\n") + 1 <= max_lines and len(seg) < 900:
                    out.append(textwrap.dedent(seg) + "\n")
        if len(out) >= 6 * n:
            break
    return [out[int(i)] for i in rng.permutation(len(out))[:n]]


def far_texts(n: int, seed: int = 0) -> Dict[str, List[str]]:
    rng = rng_for(seed, "far")
    sources: Dict[str, List[str]] = {
        "stories": [_clip(t) for t in pd.read_parquet(_file("datasets--roneneldan--TinyStories", "*/data/validation-*.parquet"))["text"] if len(str(t).split()) >= 12],
        "wiki": [_clip(t) for t in pd.read_parquet(_file("datasets--wikitext", "*/wikitext-2-raw-v1/test-*.parquet"))["text"]
                 if len(str(t).split()) >= 12 and not str(t).strip().startswith("=")],
        "science": [_clip(t) for t in pd.read_parquet(_file("datasets--ai2_arc", "*/ARC-Challenge/test-*.parquet"))["question"] if len(str(t).split()) >= 5],
    }
    for name in ("agnews", "newsgroups", "emotion"):
        sources[name] = _raw(name)[2]
    out = {k: [v[int(i)] for i in rng.permutation(len(v))[:n]] for k, v in sources.items()}
    out["stdlib_code"] = stdlib_code(n, seed)
    return out

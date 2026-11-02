"""Pilot: do the shared readers carry the skills at all, and which frontend (projection mode, size) keeps them? A closed-form LDA fitted on teaching rows, read on taught skills in every family.
Dev seed only: this decides the frontend once (recorded in the notebook), nothing else. ``python -m fsmoe.pilot [--seed 700]``. (The first run, Qwen alone, found language at 0.24 on held-out wordings;
MiniLM was added as a second block, see fsmoe/reader.py.)"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .bed import ROOT, build_bed
from .domains import NAMES
from .frontend import Frontend
from unimoe.heads import LdaHead


def run(seed: int, out: Path) -> str:
    bed = build_bed(seed, with_far=False)
    lines = [f"pilot seed {seed}: LDA accuracy on taught skills, by wording (eval rows); frontend fitted on the teaching rows of all three domains"]
    tr = bed.teaching()
    for mode in ("rp", "pca"):
        for d in (128, 256):
            fe = Frontend(d, mode, 0).fit(bed.raw[tr], np.ones(int(tr.sum()), dtype=bool))
            X = fe(bed.raw)
            y = torch.from_numpy(bed.y)
            for name in NAMES:
                m = bed.teaching(name)
                head = LdaHead(fe.out_dim, len(bed.skills), 0.1)
                head.update(X[m], y[m])
                ev = bed.mask(domain=name, split="eval", taught=True)
                pred = head.proba(X[ev]).argmax(1).numpy()
                ok = pred == bed.y[ev]
                fam = bed.family[ev]
                by = {f: float(ok[fam == f].mean()) for f in sorted(set(fam))}
                held = float(ok[bed.heldout[ev]].mean()) if bed.heldout[ev].any() else float("nan")
                lines.append(f"{mode:3s} d={d:3d} {name:5s} all={ok.mean():.3f} teaching-wording={float(ok[~bed.heldout[ev]].mean()):.3f} held-out={held:.3f}  " + "  ".join(f"{k}={v:.2f}" for k, v in by.items()))
    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=700)
    a = ap.parse_args()
    print(run(a.seed, ROOT / "results" / "fsmoe" / f"pilot_{a.seed}.txt"))

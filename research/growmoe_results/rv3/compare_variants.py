"""Side-by-side table of the integration variants on the streams that decided the design (reads the integration JSON files; nothing typed by hand).   (entry E-023)

    python research/growmoe_results/rv3/compare_variants.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = [r for f in sorted(HERE.glob("spike_set_gate_g*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
streams = ["OV-0.85", "OV-0.6", "S1", "S2-5", "S2-30", "OV-0.3-MIXED"]
variants = ["vote", "md_stored", "md_recent", "md_recent_keep", "md_recent_loss", "md_windows_loss"]
label = {"vote": "vote (current default)", "md_stored": "stored prototypes, nearest owner", "md_recent": "one recent window, nearest owner", "md_recent_keep": "one recent window, vote keeps ownership",
         "md_recent_loss": "one recent window, owner by label loss", "md_windows_loss": "source windows, owner by label loss"}
print("accuracy % (mean of seeds 100-102) / experts / known batches called novel")
print(f"{'variant':44s} " + " ".join(f"{s:>20s}" for s in streams))
for v in variants:
    cells = []
    for s in streams:
        rr = [r for r in rows if r["stream"] == s and r["variant"] == v]
        if not rr:
            cells.append("-")
            continue
        kn = sum(r["known_novel"] + r["known_known"] + r["known_skip"] + r["known_none"] for r in rr)
        cells.append(f"{100 * np.mean([r['acc'] for r in rr]):4.1f}/{np.mean([r['experts'] for r in rr]):4.1f}/{sum(r['known_novel'] for r in rr)}of{kn}")
    print(f"{label[v]:44s} " + " ".join(f"{c:>20s}" for c in cells))

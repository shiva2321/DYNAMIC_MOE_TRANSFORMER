"""Markdown tables of the group-routing runs (reads spike_arbitration_{vote_v2,windows_loss}.json; nothing typed by hand).   (entry E-023)

    python research/growmoe_results/rv3/routing_table.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STREAMS = ["OV-0.85", "OV-0.6", "OV-0.3", "OV-0.0", "OV-0.3-MIXED", "F-15", "S1", "S2-5", "S2-30"]
for tag, title in (("windows_loss", "pools trained with the final set gate (md_windows_loss)"), ("vote_v2", "pools trained with today's default (vote)")):
    rows = json.loads((HERE / f"spike_arbitration_{tag}.json").read_text(encoding="utf-8"))
    m = lambda s, k: 100 * float(np.mean([r[k] for r in rows if r["stream"] == s]))                  # noqa: E731
    print(f"\nGroup routing on {title}; accuracy %, mean of seeds 100-102")
    print("| stream | experts | per sample | group vote | group vote + veto | 50/50 mixed: per sample / vote / vote+veto | 90/10 mixed: per sample / vote / vote+veto | group of 8 / 16 / 32 |")
    print("|---|---|---|---|---|---|---|---|")
    for s in STREAMS:
        e = np.mean([r["experts"] for r in rows if r["stream"] == s])
        print(f"| {s} | {e:.1f} | {m(s, 'hard_default'):.1f} | {m(s, 'vote_pure'):.1f} ({m(s, 'vote_pure') - m(s, 'hard_default'):+.1f}) | {m(s, 'vote_pure_veto'):.1f} ({m(s, 'vote_pure_veto') - m(s, 'hard_default'):+.1f}) | "
              f"{m(s, 'mixed_perSample'):.1f} / {m(s, 'mixed_vote'):.1f} / {m(s, 'mixed_vote_veto'):.1f} | {m(s, 'mixed9010_perSample'):.1f} / {m(s, 'mixed9010_vote'):.1f} / {m(s, 'mixed9010_vote_veto'):.1f} | "
              f"{m(s, 'vote_b8'):.1f} / {m(s, 'vote_b16'):.1f} / {m(s, 'vote_b32'):.1f} |")

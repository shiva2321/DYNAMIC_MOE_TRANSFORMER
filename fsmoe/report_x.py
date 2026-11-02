"""End-to-end report: ``python -m fsmoe.report_x --seeds 710 ...`` reads results/fsmoe/X/X_s*.json and writes results/fsmoe/X/REPORT_X.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .domains import NAMES
from .e2e import ARMS, OUT
from .report_s import fmt


def load(seeds: Sequence[int], out: Path = OUT):
    return [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"X_s{s}.json").exists()]


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    runs = load(seeds, out)
    first = runs[0]["arms"]["A0"]
    L = [f"# End-to-end report (seeds {[r['seed'] for r in runs]})", "",
         "One stream over arithmetic, code and language (the stream of E-055: known skills in all wordings, never-taught skills, far texts). "
         "A0 pool alone, human labels on 10 % of all inputs; A1 pool + global closed-form head, same labels; A2 as A1 with the sandbox tool on 10 % of the code inputs and human labels on 10 % of the others; "
         "A3 tool on ALL code inputs; A4 as A3 with human labels on 20 % of the others. Never-taught skills: 3 arithmetic, 5 code, 20 language. "
         "A skill is learned when accuracy on rows the stream never contained is at least 0.5; for the code skills the tool opened, an id counts as that skill's class by the majority truth of its items (evaluation only).", "",
         f"Stream: {first['stream_items']} items, {first['code_items']} of them code-looking ({first['far_code_in_stream']} far code snippets from the standard library). "
         f"After teaching only (class slots 121): pool alone {fmt([r['teaching']['pool_alone']['mean_tw'] for r in runs])} taught / {fmt([r['teaching']['pool_alone']['mean_ho'] for r in runs])} held-out wordings; "
         f"pool + global head {fmt([r['teaching']['pool_global_head']['mean_tw'] for r in runs])} / {fmt([r['teaching']['pool_global_head']['mean_ho'] for r in runs])}.", "",
         "## Never-taught skills learned, and the cost", "",
         "| arm | human labels used | tool calls | classes the tool opened (5 true new) | purity of those classes | learned (all 28) | arithmetic (3) | code (5) | language (20) | novel accuracy (all) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, _ in ARMS:
        a = [r["arms"][name] for r in runs]
        L.append(f"| {name} | {fmt([x['human_labels'] for x in a], 0)} | {fmt([x['tool_calls'] for x in a], 0)} | {fmt([x['clusters_opened'] for x in a], 1)} | {fmt([x['cluster_purity'] for x in a], 2)} | "
                 f"{fmt([x['after']['learned']['all'] for x in a], 2)} | " + " | ".join(fmt([x["after"]["learned"][d] for x in a], 2) for d in NAMES) + f" | {fmt([x['after']['novel_acc']['all'] for x in a])} |")
    L += ["", "## Known skills after the stream (taught skills in held-out wordings; before the stream in the last row)", "",
          "| arm | all | arithmetic | code | language | taught wordings, all |", "|---|---|---|---|---|---|"]
    for name, _ in ARMS:
        a = [r["arms"][name] for r in runs]
        L.append(f"| {name} | " + " | ".join(fmt([x["after"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['after']['known_tw']['all'] for x in a])} |")
    a = [r["arms"]["A0"] for r in runs]
    L.append("| before the stream | " + " | ".join(fmt([x["before"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['before']['known_tw']['all'] for x in a])} |")
    L += ["", "## The tool", "",
          "| arm | known-family labels | new-behaviour labels | no outcome | slot overflow | non-code texts run | far code snippets run in the sandbox (of the stream's) | classes opened beyond the 5 true |", "|---|---|---|---|---|---|---|---|"]
    for name, cfg in ARMS:
        if cfg["tool"] is None:
            continue
        a = [r["arms"][name] for r in runs]
        L.append(f"| {name} | {fmt([x['tool_known'] for x in a], 0)} | {fmt([x['tool_new'] for x in a], 0)} | {fmt([x['tool_none'] for x in a], 0)} | {fmt([x['tool_slot_overflow'] for x in a], 1)} | "
                 f"{fmt([x['tool_ran_far_noncode'] for x in a], 0)} | {fmt([x['tool_ran_far_code'] for x in a], 1)} ({fmt([x['far_code_in_stream'] for x in a], 0)}) | {fmt([x['clusters_opened'] - 5 for x in a], 1)} |")
    ex = [json.loads((out / f"X_extra_s{r['seed']}.json").read_text(encoding="utf-8")) for r in runs if (out / f"X_extra_s{r['seed']}.json").exists()]
    if ex:
        L += ["", "## Diagnostic arms (added after the first five arms; see the addendum of the design)", "",
              f"A1r and A3r re-run A1 and A3 with the missing counters; they reproduced the first runs in {sum(bool(r['arms'][k].get('reproduced')) for r in ex for k in ('A1r', 'A3r'))} of {2 * len(ex)} cells. "
              "A5 = A3 with the tool's KNOWN items trained on at a rate of 10 % (the new-behaviour items always).", "",
              "| arm | human labels (arith / code / lang) | tool-labelled known items trained | learned all | arithmetic | code | language | known ho all | arithmetic | code | language |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in ("A1r", "A3r", "A5"):
            a = [r["arms"][name] for r in ex]
            L.append(f"| {name} | {fmt([x['human_by_domain']['arith'] for x in a], 0)} / {fmt([x['human_by_domain']['code'] for x in a], 0)} / {fmt([x['human_by_domain']['lang'] for x in a], 0)} | "
                     f"{fmt([x['tool_known_trained'] for x in a], 0)} | {fmt([x['after']['learned']['all'] for x in a], 2)} | " + " | ".join(fmt([x["after"]["learned"][d] for x in a], 2) for d in NAMES) + " | "
                     + " | ".join(fmt([x["after"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_X.md").write_text(text, encoding="utf-8")
    print(text)

"""Balance report: ``python -m fsmoe.report_b --seeds 710 ...`` reads results/fsmoe/B/B_s*.json and writes results/fsmoe/B/REPORT_B.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .domains import NAMES
from .e2e import B_OUT, BALANCE_ARMS
from .report_s import fmt

LABEL = {"A3b": "A3b arrival order (the E-057 A3)", "D1": "D1 sub: equal shares, surplus unused", "D2": "D2 over: equal shares, scarce domains replayed"}


def load(seeds: Sequence[int], out: Path = B_OUT):
    return [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"B_s{s}.json").exists()]


def build(seeds: Sequence[int], out: Path = B_OUT) -> str:
    runs = load(seeds, out)
    L = [f"# Domain-balanced training report (seeds {[r['seed'] for r in runs]})", "",
         "The E-057 A3 setting (pool + global head, tool on ALL code inputs, human labels on 10 % of the other inputs); only how the verified items enter the pool changes. "
         f"A3b reproduced E-057's A3 in {sum(bool(r['arms']['A3b'].get('reproduced')) for r in runs)} of {len(runs)} seeds. Never-taught skills: 3 arithmetic, 5 code, 20 language.", "",
         "## Never-taught skills learned", "", "| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | novel accuracy (all) | chunks | items trained: arithmetic | code | language |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, _ in BALANCE_ARMS:
        a = [r["arms"][name] for r in runs]
        tr = lambda d: fmt([x["trained_by_domain"].get(d, 0) for x in a], 0) if "trained_by_domain" in a[0] else "n/a (all arrivals)"
        L.append(f"| {LABEL[name]} | {fmt([x['after']['learned']['all'] for x in a], 2)} | " + " | ".join(fmt([x["after"]["learned"][d] for x in a], 2) for d in NAMES)
                 + f" | {fmt([x['after']['novel_acc']['all'] for x in a])} | {fmt([x['chunks'] for x in a], 0) if 'chunks' in a[0] else 'n/a'} | " + " | ".join(tr(d) for d in NAMES) + " |")
    L += ["", "## Known skills after the stream (taught skills in held-out wordings; before the stream in the last row)", "", "| arm | all | arithmetic | code | language | taught wordings, all |", "|---|---|---|---|---|---|"]
    for name, _ in BALANCE_ARMS:
        a = [r["arms"][name] for r in runs]
        L.append(f"| {LABEL[name]} | " + " | ".join(fmt([x["after"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['after']['known_tw']['all'] for x in a])} |")
    a = [r["arms"]["A3b"] for r in runs]
    L.append("| before the stream | " + " | ".join(fmt([x["before"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['before']['known_tw']['all'] for x in a])} |")
    d1 = [r["arms"]["D1"] for r in runs]
    L += ["", f"Domain head: {fmt([x['domain_head_accuracy'] for x in d1])} of the human-labelled items placed in the right domain (arithmetic against language; code items come from the tool).", ""]
    pr = [json.loads((out / f"B_priors_s{r['seed']}.json").read_text(encoding="utf-8")) for r in runs if (out / f"B_priors_s{r['seed']}.json").exists()]
    if pr:
        L += ["## The class priors of the global head (addendum of the design)", "",
              "The global closed-form head answers with its class counts removed (uniform priors); `A3u` keeps the arrival-order chunks of A3, `D2u` uses the `over` chunks of D2. "
              f"After teaching only, the head with uniform priors reads taught skills {fmt([r['teaching']['uniform_mean_tw'] for r in pr])} and held-out wordings {fmt([r['teaching']['uniform_mean_ho'] for r in pr])} "
              f"(with its priors: {fmt([r['teaching']['prior_mean_tw'] for r in pr])} and {fmt([r['teaching']['prior_mean_ho'] for r in pr])}).", "",
              "| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | known ho: all | arithmetic | code | language | taught wordings, all |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, label in (("A3u", "A3u arrival order, uniform priors"), ("D2u", "D2u over, uniform priors")):
            a = [r["arms"][name] for r in pr]
            L.append(f"| {label} | {fmt([x['after']['learned']['all'] for x in a], 2)} | " + " | ".join(fmt([x["after"]["learned"][d] for x in a], 2) for d in NAMES) + " | "
                     + " | ".join(fmt([x["after"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['after']['known_tw']['all'] for x in a])} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (B_OUT / "REPORT_B.md").write_text(text, encoding="utf-8")
    print(text)

"""E2: domain-level routing on the six text domains, four routers on the same test messages.

    python -m unimoe.p3_routing            # seeds 500-502, writes results/unimoe/E2_domain_routing.md

The pool's own routing (U-real, sequential: the expert a message is sent to counts as right when that expert trained mostly on the message's domain), a nearest-centroid rule, a logistic-regression
domain classifier on the sphere (the best look-only router these vectors allow), and the asking reader of the Router layer (a language model shown six teacher-named cards with two training
messages each, answering "which of these, or none"). The asking reader also reads the three out-of-distribution sets: how often it says "none of these" to text that belongs to no domain, against how
often it says it to messages that do.
"""

import sys
from pathlib import Path

import numpy as np

from . import domain_routing as DR
from . import domains as DM
from . import experiment as E
from . import runlib
from .runner import load_record

SEEDS = (500, 501, 502)
TEXT = ("banking", "agnews", "newsgroups", "emotion", "sst2", "svamp")


def asking_reader(cards, cache_dir):
    import reader as RD                                                                            # routing_general/reader.py; domain_cards() put it on the path
    return RD.CardReader(cards, with_exemplars=True, cache_dir=cache_dir)


def main() -> None:
    rows = {m: [] for m in ("pool (U-real routing)", "nearest centroid", "logistic-regression domain classifier", "asking reader (six cards + none)")}
    per_domain = {m: [] for m in rows}
    ood_refused, known_refused = [], []
    cache = E.OUT / "ask"
    for seed in SEEDS:
        st, doms = E.streams(seed)["seq"], E.domains_for(seed)
        rec = load_record(E.OUT / "runs", st.name, "U-real", seed)
        res = {"pool (U-real routing)": DR.pool_domain_routing(rec, st.regimes, TEXT), "nearest centroid": DR.centroid_domain_routing(st, TEXT),
               "logistic-regression domain classifier": DR.classifier_domain_routing(st, TEXT)}
        cards = DR.domain_cards(doms, DR.TEACHER_NAMES, TEXT)
        reader = asking_reader(cards, cache)
        res["asking reader (six cards + none)"] = ask = DR.asking_domain_routing(st, doms, reader, TEXT, per_regime=100)
        known_refused.append(ask["refused"])
        ood = DR.ood_texts(seed, 100)
        ood_refused.append({k: float((np.asarray(reader.probabilities(v)).argmax(1) == len(TEXT)).mean()) for k, v in ood.items()})
        for m, r in res.items():
            rows[m].append(r["accuracy"])
            per_domain[m].append(r["per_domain"])
        print(f"seed {seed}: " + ", ".join(f"{m.split(' (')[0]} {r['accuracy']:.3f}" for m, r in res.items()) + f"; asking reader refuses {ask['refused']:.3f} of known, OOD {ood_refused[-1]}", flush=True)
    L = ["# E2: domain-level routing on the six text domains (sequential schedule, seeds 500-502; generated)", "",
         "| router | accuracy | " + " | ".join(TEXT) + " |", "|" + "---|" * (len(TEXT) + 2)]
    for m in rows:
        L.append(f"| {m} | {np.mean(rows[m]):.3f} (seeds {', '.join(f'{x:.3f}' for x in rows[m])}) | " + " | ".join(f"{np.mean([p.get(d, np.nan) for p in per_domain[m]]):.2f}" for d in TEXT) + " |")
    L += ["", f"Asking reader says \"none of these\" to {np.mean(known_refused):.3f} of known text messages (false refusals) and to out-of-distribution text: "
          + ", ".join(f"{k} {np.mean([o[k] for o in ood_refused]):.3f}" for k in ood_refused[0]) + ".", ""]
    text = "\n".join(L)
    runlib.atomic_write_text(E.OUT / "E2_domain_routing.md", text)
    print(text)


if __name__ == "__main__":
    main()

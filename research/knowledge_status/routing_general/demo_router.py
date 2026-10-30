"""Demo: the router as a working system, end to end, on real models and real data.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/demo_router.py [--domain arithmetic] [--seed 150] [--ask "a message" ...]

A pool of experts is taught a few things (arithmetic: three operations, each with its own trained solver; text domains: a few classes). A mixed stream then arrives: messages of the taught things in
wordings nobody taught, messages of things nobody taught, and everyday sentences. For each message the ROUTER decides who answers or "not mine", the claimed expert answers, and, for a random share
of the messages, an outcome arrives (a label, or a stated answer that a taught expert may reproduce); verified outcomes recalibrate the router. The outcomes arrive independently of the router's
decision: calibrating only on messages the router already claimed would be circular (lab G4c, the second opinion). Seeds 150-152 are tuning seeds; this is a demonstration, not evidence.
"""

import argparse
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_KS / "routing_lab"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import lab_general as LG  # noqa: E402
import lab_outcome as LO  # noqa: E402
import lab_pool as LP  # noqa: E402
import reader as RD  # noqa: E402
import router as RO  # noqa: E402
import runlib  # noqa: E402

BATCH, OUTCOME_RATE = 30, 0.30


def build_router(domain: str, seed: int, q: float = 0.05):
    world, foreign, verifier, _ = LO.build(domain, seed)
    experts = list(verifier.experts) if isinstance(verifier, LO.AnswerVerifier) else None
    router = RO.Router(world.taught, world.teach, LG.default_encode, lambda cards: RD.CardReader(cards, True, LG.ASK_DIR), initial_cal=[t for t, _ in world.known_cal], states=world.states, q=q,
                       experts=experts, featurize=LP.parse_pair if experts else None)
    return world, foreign, verifier, router


def stream(world, foreign, seed: int):
    items = [(t, "known", i) for t, i in world.known_test] + [(t, "unknown", -1) for t in world.unknown_test] + [(t, "foreign", -1) for t in foreign]
    return [items[int(j)] for j in np.random.default_rng([seed, 0xDE30]).permutation(len(items))]


def show(router: RO.Router, texts: List[str]) -> None:
    for t, r in zip(texts, router.respond(texts) if router.experts else router.route(texts)):
        route, value = (r.route, r.value) if isinstance(r, RO.Response) else (r, None)
        who = f"-> {route.name} [{route.state} expert] (p={route.p_card:.2f})" + (f" answers {value}" if value is not None else "") if route.status == "claimed" else f"-> NOT MINE (said so: {', '.join(route.flagged_by)})"
        print(f"  {t[:90]!r}\n      {who}   [{route.mode}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="arithmetic", choices=LG.ALL_DOMAINS)
    ap.add_argument("--seed", type=int, default=150)
    ap.add_argument("--q", type=float, default=0.05, help="the false-refusal rate the cut is aimed at: the knob between refusing too much and answering what nobody owns")
    ap.add_argument("--ask", action="append", default=[], help="a message to route after the stream (needs the language model, loaded once)")
    a = ap.parse_args()
    runlib.set_threads(3)
    world, foreign, verifier, router = build_router(a.domain, a.seed, a.q)
    print(f"Domain {a.domain}, seed {a.seed}, q = {a.q}. The pool's experts (cards, state): {list(zip(world.taught, world.states))}")
    items = stream(world, foreign, a.seed)
    print(f"Stream: {len(items)} messages ({sum(k == 'known' for _, k, _ in items)} of taught things in new wordings, {sum(k == 'unknown' for _, k, _ in items)} of untaught things, "
          f"{sum(k == 'foreign' for _, k, _ in items)} everyday sentences). Outcomes arrive for {OUTCOME_RATE:.0%} of the messages, independently of the router.\n")
    rng = np.random.default_rng([a.seed, 0xDE31])
    stated = getattr(verifier, "stated", {})
    tally = {m: {k: [0, 0] for k in ("known", "unknown", "foreign")} for m in ("uncalibrated", "initial", "calibrated")}      # mode -> kind -> [refused, total]
    answers = {m: [0, 0] for m in tally}                                                                                          # mode -> [correct, answered] over known messages
    print(f"{'batch':>5} {'mode':>12} {'verified':>8} | {'known answered':>14} {'untaught refused':>16} {'everyday refused':>16}")
    for b in range(0, len(items), BATCH):
        chunk = items[b:b + BATCH]
        texts = [t for t, _, _ in chunk]
        out = router.respond(texts) if router.experts else router.route(texts)
        per = {k: [0, 0] for k in ("known", "unknown", "foreign")}
        for (t, kind, _), x in zip(chunk, out):
            route = x.route if isinstance(x, RO.Response) else x
            refused = route.status == "unclaimed"
            per[kind][0] += refused
            per[kind][1] += 1
            tally[route.mode][kind][0] += refused
            tally[route.mode][kind][1] += 1
            if kind == "known" and isinstance(x, RO.Response) and x.reason == "answered":
                answers[route.mode][1] += 1
                answers[route.mode][0] += int(x.value == stated.get(t))
        mode = out[0].route.mode if isinstance(out[0], RO.Response) else out[0].mode
        # outcomes: a random share of the messages gets one, whatever the router decided
        got = [(t, kind) for t, kind, _ in chunk if rng.random() < OUTCOME_RATE]
        ok = [t for t, kind in got if (kind == "known" and verifier.known([t])[0]) or (kind == "unknown" and verifier.unowned([t])[0])]
        router.observe(ok)
        r = lambda k, share: f"{per[k][1] - per[k][0]:>3}/{per[k][1]:<3}" if share == "answered" else f"{per[k][0]:>3}/{per[k][1]:<3}"
        print(f"{b // BATCH + 1:>5} {mode:>12} {router.status()['verified']:>8} | {r('known', 'answered'):>14} {r('unknown', 'refused'):>16} {r('foreign', 'refused'):>16}")
    print("\nBy mode (messages the router handled while in that mode):")
    for m, t in tally.items():
        n = sum(v[1] for v in t.values())
        if not n:
            continue
        rate = lambda k: f"{t[k][0] / t[k][1]:.3f}" if t[k][1] else "n/a"
        acc = f"{answers[m][0] / answers[m][1]:.3f} of {answers[m][1]} answered" if answers[m][1] else "n/a"
        print(f"  {m:>12}: {n:>3} messages; taught-thing messages refused {rate('known')} (false refusals), untaught refused {rate('unknown')}, everyday refused {rate('foreign')}; answers right {acc}")
    if a.ask:
        print("\nYour messages, routed by the router as it stands now:")
        show(router, a.ask)


if __name__ == "__main__":
    main()

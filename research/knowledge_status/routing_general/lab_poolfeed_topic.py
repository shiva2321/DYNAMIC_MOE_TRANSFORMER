"""Lab G6: the reader-fed pool in TOPICAL domains (E-066; Q-70 of the pool-feed lab G5).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_poolfeed_topic.py run --domain banking --seeds 150 151 --arms emb both:c,e0.5

G5 (lab_poolfeed.py) fed the real pool the asking reader's verdict next to the sentence vector on arithmetic requests, where the sentence vector follows the wording. The nuance lab had found that in topical
domains (banking intents, news topics) plain sentence vectors already separate classes better than the asking reader. Does the fused key still help there, or only cost? Same stream, same pool, same arms:
  phase 1  three taught classes one after another, natural wording (banking 64 messages each, news 96);
  phase 2  the same classes again, new messages, half of them in a SHIFTED wording (48 / 96 each);
  phase 3  a fourth class nobody taught, half shifted, mixed with the three (96 and 30 / 32 each).
The shifted wording is label-preserving and domain-neutral: typos, case flips and double spaces (the nuance lab's noisy text) plus one polite chatter sentence before and one after ("Thanks for helping me out."),
taken from the test bed's chatter list. The label is the class; the experts see the sentence vector and/or the reader's verdict (probabilities over the taught classes, named by their label, with two taught messages
each, and "none of these"). Tests are new messages: taught classes plain / shifted, the new class plain / shifted. Arms and options are those of lab_poolfeed (c, e, w, k, t, m, n, u, s)."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_KS / "routing_lab"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

import cards as CD  # noqa: E402
import domains as DM  # noqa: E402
import generator as G  # noqa: E402
import lab_poolfeed as LP  # noqa: E402
import pools  # noqa: E402
import reps  # noqa: E402
import runlib  # noqa: E402
from growmoe.accounting import pool_report  # noqa: E402
from unimoe import systems2 as S2  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g6"
LP.CACHE_READER = OUT / "reader_cache"
DOMAINS = {"banking": DM.load_banking, "agnews": DM.load_agnews, "newsgroups": DM.load_newsgroups}
GROUPS = ("taught_plain", "taught_shift", "novel_plain", "novel_shift")
SIZES = {"banking": {"n1": 64, "n2": 48, "n3_novel": 96, "n3_taught": 30, "n_test": 20},                      # banking77 classes have 35-187 teaching messages: one batch per class in phase 1
         "agnews": {"n1": 96, "n2": 96, "n3_novel": 96, "n3_taught": 32, "n_test": 40},                          # news messages are long: a smaller stream keeps the language-model passes affordable
         "newsgroups": {"n1": 96, "n2": 96, "n3_novel": 96, "n3_taught": 32, "n_test": 40}}
MAX_WORDS = 40                                                                                  # messages are cut to their first 40 words (banking messages are shorter anyway)
RECIPE = dict(LP.RECIPE)


class TMsg:
    __slots__ = ("span", "cls", "shifted")

    def __init__(self, span: str, cls: int, shifted: bool) -> None:
        self.span, self.cls, self.shifted = span, cls, shifted


def shift_text(text: str, rng: np.random.Generator, chatter: Sequence[str]) -> str:
    """The shifted wording: typos, case flips and double spaces in the message, one polite sentence before and one after."""
    body = G.apply_noise(text, "mix", rng)
    pre, post = (chatter[int(i)] for i in rng.permutation(len(chatter))[:2])
    return f"{pre} {body} {post}"


def build_topic_stream(domain: str, seed: int, n1: int = 30, n2: int = 40, n3_novel: int = 40, n3_taught: int = 15, n_test: int = 20) -> dict:
    """Deterministic in (domain, seed). Four classes are drawn among those with enough teaching messages; three are taught, one is new."""
    (tr_x, tr_y), (te_x, te_y), names = DOMAINS[domain]()
    tr_x, tr_y, te_x, te_y = np.asarray(tr_x, dtype=object), np.asarray(tr_y), np.asarray(te_x, dtype=object), np.asarray(te_y)
    rng = np.random.default_rng([seed, 0xF6, len(domain)])
    need_taught, need_novel = n1 + n2 + n3_taught, n3_novel
    ok = [c for c in range(len(names)) if (tr_y == c).sum() >= need_taught and (te_y == c).sum() >= 2 * n_test]
    ok_novel = [c for c in range(len(names)) if (tr_y == c).sum() >= need_novel and (te_y == c).sum() >= 2 * n_test]
    taught = [int(c) for c in rng.permutation(ok)[:3]]
    novel = int(next(c for c in rng.permutation(ok_novel) if int(c) not in taught))
    chatter = list(pools.load_pool("dev").chatter)

    def pick(x, y, c, n, offset=0):
        idx = np.flatnonzero(y == c)
        idx = idx[np.random.default_rng([seed, 0xC1, c]).permutation(len(idx))]
        return [" ".join(str(x[j]).split()[:MAX_WORDS]) for j in idx[offset:offset + n]]

    def maybe_shift(texts, c, half):
        out = []
        for i, s in enumerate(texts):
            sh = i % 2 == 1 if half else False
            out.append(TMsg(shift_text(s, rng, chatter) if sh else s, c, sh))
        return out
    p1, p2, p3 = [], [], []
    for c in taught:
        p1 += maybe_shift(pick(tr_x, tr_y, c, n1), c, False)
        p2 += maybe_shift(pick(tr_x, tr_y, c, n2, n1), c, True)
        p3 += maybe_shift(pick(tr_x, tr_y, c, n3_taught, n1 + n2), c, True)
    p3 += maybe_shift(pick(tr_x, tr_y, novel, n3_novel), novel, True)
    phases = [p1, [p2[int(i)] for i in rng.permutation(len(p2))], [p3[int(i)] for i in rng.permutation(len(p3))]]
    test = {"taught_plain": [], "taught_shift": [], "novel_plain": [], "novel_shift": []}
    for c in taught + [novel]:
        kind = "taught" if c in taught else "novel"
        t = pick(te_x, te_y, c, 2 * n_test)
        test[f"{kind}_plain"] += [TMsg(s, c, False) for s in t[:n_test]]
        test[f"{kind}_shift"] += [TMsg(shift_text(s, rng, chatter), c, True) for s in t[n_test:2 * n_test]]
    cards = [CD.Card(DM.readable(names[c]), tuple(m.span for m in p1 if m.cls == c)[:DM.N_EXEMPLARS]) for c in taught]
    return {"domain": domain, "seed": seed, "taught": taught, "novel": novel, "names": [DM.readable(n) for n in names], "n_classes": len(names), "phases": phases, "test": test, "cards": cards}


def all_spans(stream: dict) -> List[str]:
    out = [m.span for ph in stream["phases"] for m in ph] + [m.span for g in stream["test"].values() for m in g]
    return list(dict.fromkeys(out))


def key_parts(msgs: Sequence[TMsg], stream: dict, arm: str, emb: Dict[str, np.ndarray], probs: Dict[str, np.ndarray], opt: dict) -> np.ndarray:
    spans = [m.span for m in msgs]
    parts = []
    if arm in ("emb", "both"):
        parts.append(opt["we"] * np.stack([emb[s] for s in spans]))
    if arm in ("key", "both"):
        P = np.stack([probs[s] for s in spans]).astype(np.float32)
        if opt["hard"]:
            P = np.eye(P.shape[1], dtype=np.float32)[P.argmax(1)]
        parts.append(opt["wk"] * P)
    if arm == "oracle":
        order = stream["taught"] + [stream["novel"]]
        oh = np.zeros((len(msgs), len(order)), dtype=np.float32)
        for i, m in enumerate(msgs):
            oh[i, order.index(m.cls)] = 1.0
        parts.append(opt["wk"] * oh)
    return np.hstack(parts)


def run_arm(domain: str, seed: int, arm_full: str) -> dict:
    arm, opt = LP.parse_arm(arm_full)
    stream = build_topic_stream(domain, seed, **SIZES[domain])
    spans = all_spans(stream)
    uses_reader = arm in ("key", "both")
    emb = LP.embeddings(spans) if (arm in ("emb", "both") or (uses_reader and (opt["cache"] or opt["head"]))) else {}
    cached = bool(uses_reader and (opt["cache"] or opt["head"]))
    static_probs, rcost = (LP.reader_probs(spans, stream["cards"]) if uses_reader and not cached else ({}, {"reader_texts": 0, "reader_seconds": 0.0}))
    if cached:
        truth, _ = LP.reader_probs(spans, stream["cards"])
        roster = LP.VerdictHead(stream["cards"], emb, opt["head"], opt["msim"], truth, opt["warm"], opt["sharp"]) if opt["head"] else LP.VerdictCache(stream["cards"], emb, opt["cache"], truth)
    else:
        roster = LP.Cards(stream["cards"], False)
    if opt["shuffle"] and static_probs:
        static_probs = LP.shuffle_verdicts(static_probs, seed)
    recipe = dict(RECIPE)
    if opt["chance"]:
        recipe["transfer_chance"] = 1.0
    if opt["alpha"] is not None:
        recipe["alpha_spawn"] = opt["alpha"]
    if opt["mmin"] is not None:
        recipe["m_min"] = opt["mmin"]
    d = (len(stream["cards"]) + 1) * uses_reader + 128 * (arm in ("emb", "both")) + 4 * (arm == "oracle")
    pool = S2.make_system2("ULDA", stream["n_classes"], seed + 7, recipe, d=d)

    def feats(msgs, learn: bool = True):
        probs = roster.probs([m.span for m in msgs], learn) if cached else static_probs
        return key_parts(msgs, stream, arm, emb, probs, opt)

    def labels(msgs):
        return np.array([m.cls for m in msgs], dtype=np.int64)
    t0 = time.perf_counter()
    per_phase, assigned, cls_of = [], [], []
    for k, ph in enumerate(stream["phases"]):
        n_ev = len(pool.events)
        for lo in range(0, len(ph), LP.BATCH):
            msgs = ph[lo:lo + LP.BATCH]
            rec = pool.observe(torch.from_numpy(feats(msgs)), torch.from_numpy(labels(msgs)))
            assigned.append(rec.assigned.numpy())
            cls_of += [m.cls for m in msgs]
        kinds = Counter(e.kind for e in list(pool.events)[n_ev:])
        accs = {}
        for g in GROUPS:
            ms = stream["test"][g]
            p = pool.predict(torch.from_numpy(feats(ms, learn=False)))
            accs[g] = float((p.probs.argmax(1).numpy() == labels(ms)).mean())
        per_phase.append({"phase": k + 1, "experts": len(pool.experts), "events": dict(kinds), "acc": accs})
    seconds = time.perf_counter() - t0
    if cached:
        rcost = {"reader_texts": roster.asked, "reader_seconds": roster.seconds, "cache_served": roster.served, "cache_hits": roster.hits, "cache_wrong": roster.wrong, "asked_unique": len(roster.asked_set),
                 "stream_served": roster.s_served, "stream_hits": roster.s_hits, "stream_wrong": roster.s_wrong}
    a, o = np.concatenate(assigned), np.array(cls_of)
    sel = a >= 0
    sel23 = sel.copy()
    sel23[:len(stream["phases"][0])] = False
    nmi = float(normalized_mutual_info_score(o[sel23], a[sel23])) if sel23.sum() > 1 else float("nan")
    rep = pool_report(pool, d)
    return {"domain": domain, "seed": seed, "arm": arm_full, "taught": stream["taught"], "novel": stream["novel"], "d": d, "per_phase": per_phase, "final": per_phase[-1]["acc"], "experts": len(pool.experts),
            "false_spawns": int(per_phase[1]["events"].get("spawn", 0)), "spawns_p3": int(per_phase[2]["events"].get("spawn", 0)), "nmi_expert_skill": nmi, "unassigned_share": float(1.0 - sel.mean()),
            "params": int(rep.params), "predict_macs": float(rep.predict_macs), "pool_seconds": seconds, **rcost}


def cell(domain: str, seed: int, arm: str) -> dict:
    path = OUT / f"{domain}_{arm.replace(':', '_').replace(',', '-')}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = run_arm(domain, seed, arm)
    OUT.mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(res))
    return res


def line(r: dict) -> str:
    f = r["final"]
    return (f"{r['domain']} s{r['seed']} {r['arm']:<24s} experts {r['experts']:>2d} (p1 {r['per_phase'][0]['experts']}, p2 {r['per_phase'][1]['experts']}) false spawns {r['false_spawns']} spawns p3 {r['spawns_p3']} NMI {r['nmi_expert_skill']:.2f} "
            f"| taught plain {f['taught_plain']:.2f} shift {f['taught_shift']:.2f} | novel plain {f['novel_plain']:.2f} shift {f['novel_shift']:.2f} | reader {r['reader_texts']} texts"
            + (f" | answered by cheap reader {r['cache_hits']}/{r['cache_served']} wrong {r['cache_wrong']}" if "cache_hits" in r else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--domain", default="banking", choices=list(DOMAINS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[150])
    ap.add_argument("--arms", nargs="+", default=list(LP.ARMS))
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    for s in a.seeds:
        for arm in a.arms:
            print(line(cell(a.domain, s, arm)), flush=True)

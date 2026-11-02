"""Lab G5: the real growmoe pool with a reader in front of its gate (E-065).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_poolfeed.py run --seeds 150 151 --arms emb key

One stream per seed, arithmetic requests in everyday wording (the knowledge-status test bed's generator), single-digit operands so that one expert can learn one operation from a few hundred messages:
  phase 1  three taught skills in the TEACHING wordings (S1-S3);
  phase 2  the same three skills in NEW wordings (S4-S8): nothing new to learn, so a pool that spawns here is wrong;
  phase 3  a fourth skill nobody taught appears, in every wording, mixed with the three taught ones.
The label of a message is its answer (a class from -9 to 81); the pool's experts see the digits of the request and a "key" that says what the request is about. Four arms differ ONLY in the key:
  emb     the sentence-meaning vector of the request (MiniLM, 128 numbers): the pool as it is today, the gate follows the look of the text;
  key     the probabilities an asking reader gives to the taught skills and "none of these" (the closed list of cards.py, teacher names plus two taught messages): the gate follows what the reader understands;
  oracle  the true skill as a one-hot (the ceiling of any key);
  both    the sentence vector and the reader's probabilities side by side.
The numbers are read by the generator (a declared simplification: the lab is about the gate, not about parsing). Test messages are never in the stream (new texts, problems the stream also uses),
in four groups: taught skills in teaching wordings, taught skills in new wordings, the new skill in teaching wordings, the new skill in new wordings."""

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

import arith_domain as AD  # noqa: E402
import assignments as AS  # noqa: E402
import cards as CD  # noqa: E402
import reader as RD  # noqa: E402
import reps  # noqa: E402
import runlib  # noqa: E402
import skills  # noqa: E402
from growmoe.accounting import pool_report  # noqa: E402
from unimoe import systems2 as S2  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g5"
CACHE_READER = OUT / "reader_cache"
CACHE_REPS = _KS / "_cache" / "reps"
RECIPE = {"inner_steps": 5, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32, "hidden": 128}        # the pool recipe of the viability shootout, 128-unit experts
N_CLASSES = 91                                                                                  # answers -9 ... 81, shifted by 9
BATCH = 64
DIGITS = (1, 1)
ARMS = ("emb", "key", "oracle", "both")
GROUPS = ("taught_teach", "taught_new", "novel_teach", "novel_new")
W_DIGITS = 0.7                                                                                  # the two one-hot digits have norm sqrt(2): 0.7 x that is about 1, the norm of the key part


def parse_arm(arm: str) -> Tuple[str, dict]:
    """``key:c,h,w2,d0.35,y,e0.5`` = the key arm with the worse-than-chance flag (c), a hard (one-hot) key (h), key weight 2 (w), digit weight 0.35 (d), a new card for every expert born after
    the teaching phase (y), embedding weight 0.5 (e), a verdict cache that answers from the nearest earlier verdict when the message is that similar (k, cosine 0.95), a CONTROL that shuffles the reader's verdicts across messages (s), a trained head on the reader's verdicts (t, confidence) that abstains below a nearest-neighbour cosine (m), answers only after n confident verdicts were learned (n) and gives LLM-like sharp verdicts (u), the pool's spawn level alpha_spawn (a) and its smallest cluster m_min (p)."""
    base, _, opt = arm.partition(":")
    o = {"chance": False, "hard": False, "wk": 1.0, "wd": W_DIGITS, "dyn": False, "we": 1.0, "cache": None, "shuffle": False, "head": None, "msim": 0.6, "warm": 0, "sharp": False, "alpha": None, "mmin": None}
    for tok in [x for x in opt.split(",") if x]:
        if tok == "c":
            o["chance"] = True
        elif tok == "h":
            o["hard"] = True
        elif tok == "y":
            o["dyn"] = True
        elif tok == "s":
            o["shuffle"] = True
        elif tok[0] == "e":
            o["we"] = float(tok[1:])
        elif tok[0] == "k":
            o["cache"] = float(tok[1:])
        elif tok[0] == "t":
            o["head"] = float(tok[1:])
        elif tok[0] == "m":
            o["msim"] = float(tok[1:])
        elif tok[0] == "n":
            o["warm"] = int(tok[1:])
        elif tok == "u":
            o["sharp"] = True
        elif tok[0] == "a":
            o["alpha"] = float(tok[1:])
        elif tok[0] == "p":
            o["mmin"] = int(tok[1:])
        elif tok[0] == "w":
            o["wk"] = float(tok[1:])
        elif tok[0] == "d":
            o["wd"] = float(tok[1:])
        else:
            raise ValueError(f"unknown option {tok!r} in arm {arm!r}")
    if base not in ARMS:
        raise ValueError(f"unknown arm {base!r}")
    return base, o


class Msg:
    __slots__ = ("span", "skill", "a", "b", "answer", "family")

    def __init__(self, e: AS.Example) -> None:
        self.span, self.skill, self.a, self.b, self.answer, self.family = e.span, e.skill, e.a, e.b, e.answer, e.family


def build_stream(seed: int, n1: int = 200, n2: int = 300, n3_novel: int = 300, n3_taught: int = 100, n_test: int = 40) -> dict:
    """Deterministic in the seed. Phases are lists of messages (shuffled); tests are four groups; ``cards`` the closed list of the taught skills (teacher name and two taught messages each)."""
    rng = np.random.default_rng([seed, 0xF5])
    order = [int(i) for i in rng.permutation(len(skills.SKILLS))]
    taught, novel = [skills.SKILLS[i] for i in order[:3]], skills.SKILLS[order[3]]
    used: set = set()

    def draw(skill, n, families):
        ex = AS._draw(rng, "dev", skill, n, families, False, DIGITS, frozenset(used))
        used.update(AS.text_keys(ex))
        return [Msg(e) for e in ex]
    allf = AS.TEACH_FAMILIES + AS.TEST_FAMILIES
    p1 = [m for s in taught for m in draw(s, n1, AS.TEACH_FAMILIES)]
    p2 = [m for s in taught for m in draw(s, n2, AS.TEST_FAMILIES)]
    p3 = draw(novel, n3_novel, allf) + [m for s in taught for m in draw(s, n3_taught, allf)]
    phases = []
    for k, p in enumerate((p1, p2, p3)):
        if k == 0:                                                                              # phase 1 teaches the skills one after another (a mixed stream makes one expert for everything)
            phases.append([m for s in taught for m in (p[int(i)] for i in rng.permutation(len(p))) if m.skill == s])
        else:
            phases.append([p[int(i)] for i in rng.permutation(len(p))])
    test = {"taught_teach": [m for s in taught for m in draw(s, n_test, AS.TEACH_FAMILIES)], "taught_new": [m for s in taught for m in draw(s, n_test, AS.TEST_FAMILIES)],
            "novel_teach": draw(novel, n_test, AS.TEACH_FAMILIES), "novel_new": draw(novel, n_test, AS.TEST_FAMILIES)}
    cards = []
    for s in taught:
        ex = [m.span for m in phases[0] if m.skill == s][:2]
        cards.append(CD.Card(AD.TEACHER_NAMES[s], tuple(ex)))
    return {"seed": seed, "taught": taught, "novel": novel, "phases": phases, "test": test, "cards": cards}


def digit_features(msgs: Sequence[Msg], wd: float = W_DIGITS) -> np.ndarray:
    X = np.zeros((len(msgs), 20), dtype=np.float32)
    for i, m in enumerate(msgs):
        X[i, m.a] = 1.0
        X[i, 10 + m.b] = 1.0
    return X * wd


def key_parts(msgs: Sequence[Msg], stream: dict, arm: str, emb: Dict[str, np.ndarray], probs: Dict[str, np.ndarray], opt: Optional[dict] = None) -> np.ndarray:
    opt = opt or {"hard": False, "wk": 1.0, "wd": W_DIGITS, "we": 1.0}
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
        oh = np.zeros((len(msgs), len(skills.SKILLS)), dtype=np.float32)
        for i, m in enumerate(msgs):
            oh[i, skills.SKILLS.index(m.skill)] = 1.0
        parts.append(opt["wk"] * oh)
    return np.hstack(parts + [digit_features(msgs, opt["wd"])])


def all_spans(stream: dict) -> List[str]:
    out = [m.span for ph in stream["phases"] for m in ph] + [m.span for g in stream["test"].values() for m in g]
    return list(dict.fromkeys(out))


def embeddings(spans: Sequence[str]) -> Dict[str, np.ndarray]:
    rep = reps.get_rep("R1_minilm128")
    X = reps.cached_encode(rep, list(spans), CACHE_REPS).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return {s: X[i] for i, s in enumerate(spans)}


def reader_probs(spans: Sequence[str], cards: Sequence[CD.Card]) -> Tuple[Dict[str, np.ndarray], dict]:
    t0 = time.perf_counter()
    P = RD.CardReader(cards, True, CACHE_READER).probabilities(list(spans))
    return {s: P[i] for i, s in enumerate(spans)}, {"reader_texts": len(spans), "reader_seconds": time.perf_counter() - t0}


KMAX = 6                                                                                        # card slots of a key with dynamic cards (one per skill of the test bed)


class Cards:
    """The roster the reader is asked about. Static: the taught skills. Dynamic: also one card per expert born after the teaching phase, made of two messages the pool routes to it."""

    def __init__(self, base: Sequence[CD.Card], dynamic: bool) -> None:
        self.cards, self.dynamic = list(base), dynamic
        self.pending: Dict[int, List[str]] = {}
        self.asked = 0
        self.seconds = 0.0

    def probs(self, spans: Sequence[str], learn: bool = True) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        P = RD.CardReader(self.cards, True, CACHE_READER).probabilities(list(spans))
        self.seconds += time.perf_counter() - t0
        self.asked += len(spans)
        if self.dynamic:                                                                        # a fixed width: card slots 0 ... KMAX - 1, then "none of these"
            Q = np.zeros((len(spans), KMAX + 1), dtype=np.float32)
            Q[:, :len(self.cards)] = P[:, :-1]
            Q[:, KMAX] = P[:, -1]
            P = Q
        return {s: P[i] for i, s in enumerate(spans)}

    def after_batch(self, pool, xb: torch.Tensor, spans: Sequence[str], new_ids: Sequence[int], phase: int) -> None:
        """Dynamic cards: a pool expert born after the teaching phase gets a card once it has two messages routed to it."""
        if not self.dynamic or phase < 1:
            return
        for e in new_ids:
            self.pending.setdefault(int(e), [])
        if not self.pending:
            return
        routed = pool.predict(xb).expert.numpy()
        for e in list(self.pending):
            if e not in pool.experts:
                del self.pending[e]
                continue
            self.pending[e] += [spans[i] for i in np.flatnonzero(routed == e)]
            if len(self.pending[e]) >= 2 and len(self.cards) < KMAX:
                self.cards.append(CD.Card(f"another kind of request ({len(self.cards) + 1})", tuple(dict.fromkeys(self.pending[e]))[:2]))
                del self.pending[e]


class VerdictCache(Cards):
    """A cheap reader in front of the asking reader: a message whose sentence vector is within ``sim`` (cosine) of a message the language model already judged gets that message's verdict; any other message
    is sent to the language model and remembered. The language model's verdicts for the messages served from the cache are known (cached on disk) and used ONLY to count wrong hits."""

    def __init__(self, base, emb: Dict[str, np.ndarray], sim: float, truth: Dict[str, np.ndarray]) -> None:
        super().__init__(base, False)
        self.emb, self.sim, self.truth = emb, sim, truth
        self.E = np.zeros((0, 128), dtype=np.float32)
        self.P = np.zeros((0, len(base) + 1), dtype=np.float32)
        self.hits = self.wrong = self.served = 0
        self.asked_set: set = set()                                                           # distinct texts sent to the language model (a deployed system asks each message once)
        self.s_served = self.s_hits = self.s_wrong = 0                                         # the same counts over the stream only (learn=True), not the repeated test evaluations

    def probs(self, spans: Sequence[str], learn: bool = True) -> Dict[str, np.ndarray]:
        n = len(spans)
        E = np.stack([self.emb[s] for s in spans])
        out = np.zeros((n, len(self.cards) + 1), dtype=np.float32)
        hit = np.zeros(n, dtype=bool)
        if len(self.E):
            S = E @ self.E.T
            nn = S.argmax(1)
            hit = S[np.arange(n), nn] >= self.sim
            out[hit] = self.P[nn[hit]]
        miss = np.flatnonzero(~hit)
        if len(miss):
            t0 = time.perf_counter()
            L = RD.CardReader(self.cards, True, CACHE_READER).probabilities([spans[i] for i in miss])
            self.seconds += time.perf_counter() - t0
            self.asked += len(miss)
            self.asked_set.update(spans[i] for i in miss)
            out[miss] = L
            if learn:
                self.E, self.P = np.vstack([self.E, E[miss]]), np.vstack([self.P, L.astype(np.float32)])
        wrong = int(sum(int(out[i].argmax()) != int(self.truth[spans[i]].argmax()) for i in np.flatnonzero(hit)))
        self.served += n
        self.hits += int(hit.sum())
        self.wrong += wrong
        if learn:
            self.s_served += n
            self.s_hits += int(hit.sum())
            self.s_wrong += wrong
        return {s: out[i] for i, s in enumerate(spans)}


class VerdictHead(Cards):
    """A small trained head in front of the asking reader: logistic regression on the sentence vector, trained on the reader's CONFIDENT verdicts (probability at least 0.9) for the messages it had to ask
    about. It answers a message only when its own probability is at least ``conf`` AND the message is at least ``msim`` (cosine) close to some message it was trained on; any other message goes to the
    language model and is learned from. The full reader's verdicts for the messages the head answered are known (cached on disk) and used ONLY to count wrong answers."""

    LEARN_AT = 0.9

    SHARP = 0.97                                                                                # the probability the language model typically gives the option it is sure of

    def __init__(self, base, emb: Dict[str, np.ndarray], conf: float, msim: float, truth: Dict[str, np.ndarray], warm: int = 0, sharp: bool = False) -> None:
        super().__init__(base, False)
        self.emb, self.conf, self.msim, self.truth, self.warm, self.sharp = emb, conf, msim, truth, warm, sharp
        self.E = np.zeros((0, 128), dtype=np.float32)
        self.y: List[int] = []
        self.clf = None
        self.fits = self.hits = self.wrong = self.served = 0
        self.asked_set: set = set()
        self.s_served = self.s_hits = self.s_wrong = 0

    def _refit(self) -> None:
        from sklearn.linear_model import LogisticRegression
        if len(set(self.y)) >= 2:
            self.clf = LogisticRegression(C=10.0, max_iter=300).fit(self.E, np.array(self.y))
            self.fits += 1

    def _head_probs(self, E: np.ndarray) -> np.ndarray:
        H = np.zeros((len(E), len(self.cards) + 1), dtype=np.float32)
        H[:, self.clf.classes_] = self.clf.predict_proba(E)
        return H

    def probs(self, spans: Sequence[str], learn: bool = True) -> Dict[str, np.ndarray]:
        n = len(spans)
        E = np.stack([self.emb[s] for s in spans])
        out = np.zeros((n, len(self.cards) + 1), dtype=np.float32)
        use = np.zeros(n, dtype=bool)
        if self.clf is not None and len(self.y) >= self.warm:
            H = self._head_probs(E)
            use = (H.max(1) >= self.conf) & ((E @ self.E.T).max(1) >= self.msim)
            if self.sharp:                                                                       # hand the pool a verdict shaped like the language model's: the head's choice, sure of itself
                S = np.full_like(H, (1.0 - self.SHARP) / (H.shape[1] - 1))
                S[np.arange(len(H)), H.argmax(1)] = self.SHARP
                H = S
            out[use] = H[use]
        miss = np.flatnonzero(~use)
        if len(miss):
            t0 = time.perf_counter()
            L = RD.CardReader(self.cards, True, CACHE_READER).probabilities([spans[i] for i in miss])
            self.seconds += time.perf_counter() - t0
            self.asked += len(miss)
            self.asked_set.update(spans[i] for i in miss)
            out[miss] = L
            if learn:
                sure = L.max(1) >= self.LEARN_AT
                if sure.any():
                    self.E = np.vstack([self.E, E[miss][sure]])
                    self.y += [int(v) for v in L[sure].argmax(1)]
                    self._refit()
        wrong = int(sum(int(out[i].argmax()) != int(self.truth[spans[i]].argmax()) for i in np.flatnonzero(use)))
        self.served += n
        self.hits += int(use.sum())
        self.wrong += wrong
        if learn:
            self.s_served += n
            self.s_hits += int(use.sum())
            self.s_wrong += wrong
        return {s: out[i] for i, s in enumerate(spans)}


def shuffle_verdicts(probs: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    """The control of the `s` option: the reader's verdicts are kept (same set of vectors) but handed to different messages, so any gain that survives is not about what the reader understood."""
    keys = list(probs)
    perm = np.random.default_rng([seed, 0x5F]).permutation(len(keys))
    vals = [probs[k] for k in keys]
    return {k: vals[int(i)] for k, i in zip(keys, perm)}


def run_arm(seed: int, arm_full: str) -> dict:
    arm, opt = parse_arm(arm_full)
    stream = build_stream(seed)
    spans = all_spans(stream)
    uses_reader = arm in ("key", "both")
    emb = embeddings(spans) if (arm in ("emb", "both") or (uses_reader and (opt["cache"] or opt["head"]))) else {}
    cached = bool(uses_reader and (opt["cache"] or opt["head"]))
    static_probs, rcost = (reader_probs(spans, stream["cards"]) if uses_reader and not (opt["dyn"] or cached) else ({}, {"reader_texts": 0, "reader_seconds": 0.0}))
    if cached:
        truth, _ = reader_probs(spans, stream["cards"])                                          # the full reader's verdicts, used only to count wrong hits
        roster = VerdictHead(stream["cards"], emb, opt["head"], opt["msim"], truth, opt["warm"], opt["sharp"]) if opt["head"] else VerdictCache(stream["cards"], emb, opt["cache"], truth)
    else:
        roster = Cards(stream["cards"], opt["dyn"] and uses_reader)
    if opt["shuffle"] and static_probs:                                                          # control: the same verdicts, given to the wrong messages
        static_probs = shuffle_verdicts(static_probs, seed)
    recipe = dict(RECIPE)
    if opt["chance"]:
        recipe["transfer_chance"] = 1.0                                                          # E-062: a parent that does worse than chance on a cluster cannot vouch for it
    if opt["alpha"] is not None:
        recipe["alpha_spawn"] = opt["alpha"]
    if opt["mmin"] is not None:
        recipe["m_min"] = opt["mmin"]
    d = (len(stream["cards"]) + 1 if not roster.dynamic else KMAX + 1) * uses_reader + (128 if arm in ("emb", "both") else 0) + (len(skills.SKILLS) if arm == "oracle" else 0) + 20
    pool = S2.make_system2("ULDA", N_CLASSES, seed + 7, recipe, d=d)

    def feats(msgs, learn: bool = True):
        probs = roster.probs([m.span for m in msgs], learn) if (roster.dynamic or cached) else static_probs
        return key_parts(msgs, stream, arm, emb, probs, opt)

    def answers(msgs):
        return np.array([m.answer + 9 for m in msgs], dtype=np.int64)
    t0 = time.perf_counter()
    per_phase, assigned, ops = [], [], []
    for k, ph in enumerate(stream["phases"]):
        n_ev, kinds_before = len(pool.events), set(pool.experts)
        for lo in range(0, len(ph), BATCH):
            msgs = ph[lo:lo + BATCH]
            xb, yb = torch.from_numpy(feats(msgs)), torch.from_numpy(answers(msgs))
            before = set(pool.experts)
            rec = pool.observe(xb, yb)
            assigned.append(rec.assigned.numpy())
            ops += [m.skill for m in msgs]
            roster.after_batch(pool, xb, [m.span for m in msgs], sorted(set(pool.experts) - before), k)
        kinds = Counter(e.kind for e in list(pool.events)[n_ev:])
        accs = {}
        for g in GROUPS:
            ms = stream["test"][g]
            p = pool.predict(torch.from_numpy(feats(ms, learn=False)))
            accs[g] = float((p.probs.argmax(1).numpy() == answers(ms)).mean())
        per_phase.append({"phase": k + 1, "experts": len(pool.experts), "events": dict(kinds), "acc": accs, "cards": len(roster.cards)})
    seconds = time.perf_counter() - t0
    if roster.dynamic or cached:
        rcost = {"reader_texts": roster.asked, "reader_seconds": roster.seconds}
    if cached:
        rcost.update({"cache_served": roster.served, "cache_hits": roster.hits, "cache_wrong": roster.wrong, "asked_unique": len(roster.asked_set),
                      "stream_served": roster.s_served, "stream_hits": roster.s_hits, "stream_wrong": roster.s_wrong})
    n_p1 = len(stream["phases"][0])
    a, o = np.concatenate(assigned), np.array(ops)
    sel = a >= 0
    sel23 = sel.copy()
    sel23[:n_p1] = False
    nmi = float(normalized_mutual_info_score(o[sel23], a[sel23])) if sel23.sum() > 1 else float("nan")
    rep = pool_report(pool, d)
    return {"seed": seed, "arm": arm_full, "taught": stream["taught"], "novel": stream["novel"], "d": d, "per_phase": per_phase, "final": per_phase[-1]["acc"], "experts": len(pool.experts),
            "false_spawns": int(per_phase[1]["events"].get("spawn", 0)), "spawns_p3": int(per_phase[2]["events"].get("spawn", 0)), "nmi_expert_skill": nmi, "cards_end": len(roster.cards),
            "unassigned_share": float(1.0 - sel.mean()), "params": int(rep.params), "predict_macs": float(rep.predict_macs), "pool_seconds": seconds, **rcost}


def cell(seed: int, arm: str, tag: str = "") -> dict:
    path = OUT / f"{arm.replace(':', '_').replace(',', '-')}{tag}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = run_arm(seed, arm)
    OUT.mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(res))
    return res


def line(r: dict) -> str:
    f = r["final"]
    return (f"s{r['seed']} {r['arm']:<12s} experts {r['experts']:>2d} (p1 {r['per_phase'][0]['experts']}, p2 {r['per_phase'][1]['experts']}) false spawns {r['false_spawns']} spawns p3 {r['spawns_p3']} NMI {r['nmi_expert_skill']:.2f} "
            f"| taught/teach {f['taught_teach']:.2f} taught/new {f['taught_new']:.2f} novel/teach {f['novel_teach']:.2f} novel/new {f['novel_new']:.2f} | reader {r['reader_texts']} texts {r['reader_seconds']:.0f}s pool {r['pool_seconds']:.0f}s"
            + (f" | cache hits {r['cache_hits']}/{r['cache_served']} wrong {r['cache_wrong']}" if "cache_hits" in r else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[150])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), help="emb | key | oracle | both, with options after a colon, e.g. key:c,h,w2,d0.35")
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    for s in a.seeds:
        for arm in a.arms:
            print(line(cell(s, arm)), flush=True)

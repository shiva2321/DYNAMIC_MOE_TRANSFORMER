"""Layer L: the new-skill learning path (design: docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md, part 2).

``VerifiedLearner`` decides HOW verified items reach the pool; the pool, the stream and the verification policies are those of fsmoe.system_lab, so the arms differ in this one thing:
* ``interleaved``    verified items go to the pool in chunks of 16, whatever their skills (E-055);
* ``focused``        verified items of a NEW skill wait in a buffer per skill; every 10 items the skill is taught as its own batch (its newest items cycled to 32, 5 passes);
* ``focused_replay`` as ``focused``, but every batch is half new-skill items, half replay of known items from a bounded memory (2,000 items: teaching rows sampled at random plus every known item verified so far).
Known skills' verified items always take the interleaved path. The diagnosis (Q-46) is read from the arms themselves: which known held-out-wording items turn from right to wrong, and into what.
``python -m fsmoe.learn_lab run --seeds 700`` (dev smoke test) or ``--seeds 710 711 ...`` writes results/fsmoe/L/L_s<seed>.json.
"""

import argparse
import copy
import json
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .bed import ROOT, build_bed
from .detectors import Maha
from .system_lab import OUT as S_OUT, Prep, TUNE_SEED, VERIFY_CHUNK, accuracy, build_stream, diverse_pick, make_pool, teach, zlib_policy

OUT = ROOT / "results" / "fsmoe" / "L"
MODES = ("interleaved", "focused", "focused_replay")
ARMS = (("oracle", 0.10), ("diverse", 0.10))


class VerifiedLearner:
    def __init__(self, observe: Callable[[torch.Tensor, torch.Tensor], object], mode: str, taught_ids: set, mem_x: torch.Tensor, mem_y: torch.Tensor, rng: np.random.Generator,
                 m: int = 10, passes: int = 5, chunk: int = VERIFY_CHUNK, mem_cap: int = 2000, batch: int = 32) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.observe, self.mode, self.taught, self.rng = observe, mode, set(taught_ids), rng
        self.m, self.passes, self.chunk, self.mem_cap, self.batch = m, passes, chunk, mem_cap, batch
        self.mem_x, self.mem_y = mem_x[:mem_cap].clone(), mem_y[:mem_cap].clone()
        self.buf_x: List[torch.Tensor] = []
        self.buf_y: List[torch.Tensor] = []
        self.cls: Dict[int, List[torch.Tensor]] = {}
        self.done: Dict[int, int] = {}
        self.n_consolidations = 0
        self.n_chunks = 0

    # ---- replay memory
    def _remember(self, x: torch.Tensor, label: int) -> None:
        if len(self.mem_y) < self.mem_cap:
            self.mem_x, self.mem_y = torch.cat([self.mem_x, x[None]]), torch.cat([self.mem_y, torch.tensor([label])])
        else:
            j = int(self.rng.integers(0, self.mem_cap))
            self.mem_x[j], self.mem_y[j] = x, label

    def memory_bytes(self) -> int:
        return int(4 * self.mem_x.numel())

    # ---- the paths
    def _flush_chunk(self) -> None:
        if self.buf_y:
            self.observe(torch.stack(self.buf_x), torch.stack(self.buf_y))
            self.buf_x, self.buf_y = [], []
            self.n_chunks += 1

    def _consolidate(self, label: int) -> None:
        items = torch.stack(self.cls[label][-4 * self.m:])
        cyc = items[torch.arange(self.batch) % len(items)]
        for _ in range(self.passes):
            xb, yb = cyc, torch.full((self.batch,), label, dtype=torch.long)
            if self.mode == "focused_replay" and len(self.mem_y):
                idx = torch.from_numpy(self.rng.choice(len(self.mem_y), self.batch, replace=len(self.mem_y) < self.batch))
                xb, yb = torch.cat([xb, self.mem_x[idx]]), torch.cat([yb, self.mem_y[idx]])
            self.observe(xb, yb)
        self.done[label] = len(self.cls[label])
        self.n_consolidations += 1

    def add(self, x: torch.Tensor, label: int) -> None:
        if self.mode == "interleaved" or label in self.taught:
            if label in self.taught:
                self._remember(x, label)
            self.buf_x.append(x)
            self.buf_y.append(torch.tensor(label))
            if len(self.buf_y) >= self.chunk:
                self._flush_chunk()
            return
        self.cls.setdefault(label, []).append(x)
        if len(self.cls[label]) - self.done.get(label, 0) >= self.m:
            self._consolidate(label)
            for xx in self.cls[label][-self.m:]:
                self._remember(xx, label)

    def finish(self) -> None:
        self._flush_chunk()
        for label, items in self.cls.items():
            if len(items) - self.done.get(label, 0) >= 3:
                self._consolidate(label)


def deploy_learn(pool, prep: Prep, seed: int, policy: str, budget: float, mode: str) -> dict:
    X, y, kind = build_stream(prep, seed)
    bed = prep.bed
    rng = np.random.default_rng([seed, 0x4C4E, zlib_policy(policy, budget)])
    teach_rows = np.flatnonzero(bed.teaching())
    scorer = Maha().fit(prep.X[teach_rows], prep.y[teach_rows]) if policy in ("novelty", "diverse", "balanced") else None
    taught_ids = {int(i) for i in np.unique(prep.y[teach_rows].numpy())}
    mem = rng.choice(teach_rows, min(2000, len(teach_rows)), replace=False)
    learner = VerifiedLearner(pool.observe, mode, taught_ids, prep.X[mem], prep.y[mem], rng)
    rows_ho = prep.rows("cal", None, "known_ho")
    p0 = pool.predict(prep.X[rows_ho]).probs.argmax(1)
    ver = {"total": 0, "novel": 0, "known": 0, "far": 0}
    skills_met = set()
    online = {"novel": [], "known": []}
    n, half = len(y), len(y) // 2
    for lo in range(0, n, 64):
        sl = slice(lo, min(lo + 64, n))
        x, yy, kk = X[sl], y[sl], kind[sl]
        probs = pool.predict(x).probs
        pred = probs.argmax(1)
        for c in ("novel", "known"):
            msk = torch.from_numpy(kk == c)
            if lo >= half and bool(msk.any()):
                online[c].append((pred[msk] == yy[msk]).float())
        k = int(round(budget * len(x)))
        if k <= 0:
            continue
        if policy == "oracle":
            pick = np.flatnonzero(kk == "novel")[:k]
        elif policy == "diverse":
            pick = diverse_pick(scorer.score(x), x, k)
        elif policy == "balanced":                                                      # half diversity-ranked novelty, half random KNOWN items (they carry the stream's wordings)
            k_nov = (k + 1) // 2
            nov = diverse_pick(scorer.score(x), x, k_nov)
            rest = np.setdiff1d(np.flatnonzero(kk == "known"), nov)
            pick = np.concatenate([nov, rng.permutation(rest)[:k - k_nov]]).astype(int)
        elif policy == "novelty":
            pick = np.argsort(-scorer.score(x))[:k]
        else:
            raise ValueError(policy)
        for j in pick:
            ver["total"] += 1
            if kk[j] == "far":
                ver["far"] += 1
                continue
            ver["novel"] += int(kk[j] == "novel")
            ver["known"] += int(kk[j] == "known")
            if kk[j] == "novel":
                skills_met.add(int(yy[j]))
            learner.add(x[j], int(yy[j]))
    learner.finish()
    p1 = pool.predict(prep.X[rows_ho]).probs.argmax(1)
    truth = prep.y[rows_ho]
    flipped = (p0 == truth) & (p1 != truth)
    withheld_ids = {bed.skills.index(s) for s in bed.withheld}
    twin_of = {bed.skills.index(s): bed.skills.index(t) for s, t in bed.twin.items()}
    new_cls = torch.tensor([int(c) in withheld_ids for c in p1.tolist()])
    twin_cls = torch.tensor([twin_of.get(int(t)) == int(c) for t, c in zip(truth.tolist(), p1.tolist())])
    cal_nov = prep.rows("cal", None, "novel")
    per_skill = {}
    for s in sorted({int(v) for v in prep.y[cal_nov].tolist()}):
        r = cal_nov[prep.y[cal_nov].numpy() == s]
        per_skill[s] = accuracy(pool, prep, r)
    mean_online = lambda c: float(torch.cat(online[c]).mean()) if online[c] else float("nan")
    nflip = int(flipped.sum())
    return {"policy": policy, "budget": budget, "mode": mode, "verified": ver["total"], "verified_novel": ver["novel"], "verified_known": ver["known"], "novel_skills_verified": len(skills_met),
            "n_withheld": len(per_skill), "learned_fraction": float(np.mean([a >= 0.5 for a in per_skill.values()])), "novel_acc_final": float(np.mean(list(per_skill.values()))),
            "novel_acc_online": mean_online("novel"), "known_ho_before": float((p0 == truth).float().mean()), "known_ho_final": float((p1 == truth).float().mean()),
            "known_tw_final": accuracy(pool, prep, prep.rows("cal", None, "known_tw")), "flipped": nflip, "flipped_to_new_skill": int((flipped & new_cls).sum()),
            "flipped_to_twin": int((flipped & twin_cls & ~new_cls).sum()), "flipped_to_other_known": int((flipped & ~new_cls & ~twin_cls).sum()),
            "n_known_ho": int(len(truth)), "n_experts": int(pool.summary().get("n_experts", 1)), "consolidations": learner.n_consolidations, "chunks": learner.n_chunks,
            "replay_memory_bytes": learner.memory_bytes() if mode == "focused_replay" else 0}


def run_seed(seed: int, tuning: dict) -> dict:
    path = OUT / f"L_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    fs_pool = make_pool("FS", prep, seed, tuning["FS_best"])
    t = teach(fs_pool, prep, seed)
    res = {"seed": seed, "teaching_mean_tw": t["mean_tw"], "teaching_mean_ho": t["mean_ho"], "deploy": []}
    main = S_OUT / f"S_s{seed}.json"
    if main.exists():
        m = json.loads(main.read_text(encoding="utf-8"))
        res["teaching_reproduced"] = bool(abs(t["mean_tw"] - m["teaching"]["FS"]["mean_tw"]) < 1e-9 and abs(t["mean_ho"] - m["teaching"]["FS"]["mean_ho"]) < 1e-9)
    for policy, budget in ARMS:
        for mode in MODES:
            r = deploy_learn(copy.deepcopy(fs_pool), prep, seed, policy, budget, mode)
            res["deploy"].append(r)
            print(f"seed {seed} {policy} {budget} {mode}: learned {r['learned_fraction']:.2f} known_ho {r['known_ho_before']:.3f}->{r['known_ho_final']:.3f} flipped {r['flipped']} "
                  f"(to new {r['flipped_to_new_skill']}, twin {r['flipped_to_twin']}, other {r['flipped_to_other_known']})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


class GlobalFast:
    """The pool plus ONE global closed-form head over all classes: no routing, one pass, never forgets. Updated by every observed item; the answer is the mixture p = (1 - lam) p_pool + lam p_lda
    (a mixture, not a log-linear blend, so a class the pool has never seen can still win). Same protocol as the pool (observe, predict, summary)."""

    def __init__(self, pool, d: int, n_classes: int, lam: float = 0.5, shrink: float = 0.1, uniform_priors: bool = False) -> None:
        from unimoe.heads import LdaHead
        self.pool, self.lam, self.uniform_priors = pool, lam, uniform_priors
        self.head = LdaHead(d, n_classes, shrink)
        self.name = f"{getattr(pool, 'name', 'pool')}+global_fast"

    @property
    def events(self):
        return self.pool.events

    def observe(self, x, y=None, ctx=None):
        if y is not None:
            self.head.update(torch.nn.functional.normalize(x.float(), dim=1), y)
        return self.pool.observe(x, y, ctx)

    def predict(self, x, ctx=None):
        from growmoe.types import Prediction
        p = self.pool.predict(x, ctx)
        if self.head.total == 0:
            return p
        xn = torch.nn.functional.normalize(x.float(), dim=1)
        if self.uniform_priors:                                              # drop the log class counts the LDA adds to every class: a class met 10 times counts as much as one met 2,000 times
            z, seen = self.head.logits(xn), self.head.seen()
            z[:, seen] -= torch.log(self.head.n[seen] / self.head.total).float()
            pf = torch.softmax(z, dim=-1)
        else:
            pf = self.head.proba(xn)
        probs = (1.0 - self.lam) * p.probs + self.lam * pf
        return Prediction(probs=probs, expert=p.expert, p_nov=p.p_nov, unknown=p.unknown, meta=p.meta)

    def summary(self) -> dict:
        return self.pool.summary()


EXTRA_ARMS = (("balanced", 0.10, "interleaved"), ("balanced", 0.10, "focused_replay"))
FAST_ARMS = (("oracle", 0.10, "interleaved"), ("diverse", 0.10, "interleaved"), ("balanced", 0.10, "interleaved"))


def run_fast(seed: int, tuning: dict) -> dict:
    """The global-closed-form-head arms (addendum 2 of the follow-up design). Writes L_fast_s<seed>.json."""
    path = OUT / f"L_fast_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prep = Prep(build_bed(seed))
    pool = GlobalFast(make_pool("FS", prep, seed, tuning["FS_best"]), prep.d, prep.n_classes)
    t = teach(pool, prep, seed)
    main = json.loads((OUT / f"L_s{seed}.json").read_text(encoding="utf-8"))
    res = {"seed": seed, "teaching_mean_tw": t["mean_tw"], "teaching_mean_ho": t["mean_ho"], "pool_alone_mean_tw": main["teaching_mean_tw"], "pool_alone_mean_ho": main["teaching_mean_ho"], "deploy": []}
    for policy, budget, mode in FAST_ARMS:
        r = deploy_learn(copy.deepcopy(pool), prep, seed, policy, budget, mode)
        res["deploy"].append(r)
        print(f"seed {seed} global_fast {policy} {budget} {mode}: learned {r['learned_fraction']:.2f} known_ho {r['known_ho_before']:.3f}->{r['known_ho_final']:.3f} flipped {r['flipped']} (to new {r['flipped_to_new_skill']})", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


def run_extra(seed: int, tuning: dict) -> dict:
    """The wording-balance arms added after the first cells (see the addendum of the follow-up design). Writes L_extra_s<seed>.json."""
    path = OUT / f"L_extra_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prep = Prep(build_bed(seed))
    fs_pool = make_pool("FS", prep, seed, tuning["FS_best"])
    t = teach(fs_pool, prep, seed)
    main = json.loads((OUT / f"L_s{seed}.json").read_text(encoding="utf-8"))
    res = {"seed": seed, "teaching_reproduced": bool(abs(t["mean_tw"] - main["teaching_mean_tw"]) < 1e-9 and abs(t["mean_ho"] - main["teaching_mean_ho"]) < 1e-9), "deploy": []}
    for policy, budget, mode in EXTRA_ARMS:
        r = deploy_learn(copy.deepcopy(fs_pool), prep, seed, policy, budget, mode)
        res["deploy"].append(r)
        print(f"seed {seed} {policy} {budget} {mode}: learned {r['learned_fraction']:.2f} known_ho {r['known_ho_before']:.3f}->{r['known_ho_final']:.3f} flipped {r['flipped']} (to new {r['flipped_to_new_skill']})", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "extra", "fast"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[TUNE_SEED])
    a = ap.parse_args()
    tuning = json.loads((S_OUT / "tuning.json").read_text(encoding="utf-8"))
    for s in a.seeds:
        {"run": run_seed, "extra": run_extra, "fast": run_fast}[a.cmd](s, tuning)

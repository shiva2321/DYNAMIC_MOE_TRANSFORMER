"""THROWAWAY spike on DEV seeds (100-102): put the CALIBRATED set-level novelty test into the live pool and apply the spec's stage-3 acceptance criteria.   (entry E-023)

Patched copy of PhyloMoE.observe (no product file touched). After the batch vote and before the probation buffer, the samples the per-sample gate ACCEPTED are tested as a set:
  - candidates = the (up to) three experts that best explain the accepted samples (largest summed log-density);
  - each candidate with a usable reference gets an exact permutation test of the accepted samples against that reference (999 permutations, smallest p-value 0.001);
      reference "stored"  = the expert's <= 256 stored prototypes (what the first spike used);
      reference "recent"  = the last five batches' worth of samples the expert itself was trained on (>= 48 samples needed), a change detector;
  - if ALL candidates are testable and every one rejects (p <= alpha): the accepted samples are novel as a SET (covert novelty) and go to the probation buffer with the flagged
    ones (a candidate without a usable reference cannot be rejected, so it blocks the novelty call; this stops a cascade of experts spawned for data a young expert explains);
  - otherwise, if some testable candidate does not reject, the best-explaining such candidate trains on all accepted samples. Samples the per-sample gate flagged are left
    alone (they go to probation as before), so a mixed batch (known + new) keeps its minority;
  - if nothing can be decided (an untestable candidate and every testable one rejects) nothing changes (the plain vote).

Variants: vote (the current default), md_stored, md_recent, md_recent_01 (alpha 0.01), mmd_recent.
Acceptance (spec section 6, on dev seeds, against `vote`):
  (a) expert counts unchanged (within 0.5) on F-15, S1, S2-5, S2-30, OV-0.0;
  (b) at OV-0.6 and OV-0.85: at least 8 experts for 10 regimes and at least +8 points accuracy.
Also reported: OV-0.3, OV-0.3-MIXED (50/50 batches, the guard for the single-source assumption).

    python research/growmoe_results/rv3/spike_set_gate.py [--streams F-15,S1 --tag a]      (one process per stream group; `--merge` prints the combined table)
"""

import argparse
import inspect
import json
import os
import sys
import textwrap
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rv2a"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_rv2a as A                                                                    # noqa: E402
import spike_set_tests as T                                                             # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.pools import phylo as phylo_mod                                            # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402

torch.set_num_threads(3)
STEPS = {"func": 5, "perm": 20, "types": 200}
STREAMS = ["F-15", "S1", "S2-5", "S2-30", "OV-0.0", "OV-0.3", "OV-0.3-MIXED", "OV-0.6", "OV-0.85"]
SEEDS = (100, 101, 102)
# variant -> (statistic, reference, alpha)
VARIANTS = {"vote": None, "md_stored": ("meandist", "stored", 0.001), "md_recent": ("meandist", "recent", 0.001), "md_recent_01": ("meandist", "recent", 0.01),
            "mmd_recent": ("mmd", "recent", 0.001), "md_both": ("meandist", "both", 0.001)}
MIN_RECENT = 48

_PATCH = """\
if getattr(self, "set_cfg", None):
    _acc = assigned >= 0
    if int(_acc.sum()) >= self.cfg.batch_vote_min:
        _xb = x[_acc]
        _own = torch.bincount(torch.searchsorted(torch.tensor(eids), assigned[_acc]), minlength=len(eids))
        _score = _own.double() * 1e6 + LogF[_acc].sum(0).double()                 # the per-sample gate's owners first, then by summed log-density
        _cands = _score.topk(min(3, len(eids))).indices.tolist()
        _ps = [self._set_p(eids[_c], _xb) for _c in _cands]
        _use = [i for i, p in enumerate(_ps) if p == p]
        _ok = [i for i in _use if _ps[i] > self.set_cfg[2]]
        if len(_use) == len(_cands) and not _ok:
            novel = novel | _acc
            assigned = torch.where(_acc, torch.full_like(assigned, -1), assigned)
            self.set_events.append("novel")
        elif _ok:
            _best = max(_ok, key=lambda i: float(_score[_cands[i]]))
            assigned = torch.where(_acc, torch.full_like(assigned, eids[_cands[_best]]), assigned)
            self.set_events.append("known")
        else:
            self.set_events.append("skip")
"""


def _patched_observe():
    lines = textwrap.dedent(inspect.getsource(PhyloMoE.observe)).split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip() == "spawned = []")
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i] = [indent + l if l else l for l in _PATCH.rstrip("\n").split("\n")]
    ns: dict = {}
    exec("\n".join(lines), vars(phylo_mod), ns)
    return ns["observe"]


class SetGatePool(PhyloMoE):
    _observe_patched = _patched_observe()

    def __init__(self, *a, set_cfg=None, **k):
        super().__init__(*a, **k)
        self.set_cfg, self.set_events, self.recent = set_cfg, [], {}
        self._g = torch.Generator().manual_seed(0)

    def _set_p(self, eid, xb):
        stat, ref_kind, _ = self.set_cfg
        test = T.meandist_p if stat == "meandist" else T.mmd_p
        p_stored = test(self.experts[eid].meta["sig"].protos, xb, self._g) if ref_kind in ("stored", "both") else None
        if ref_kind == "stored":
            return p_stored
        hist = self.recent.get(eid)
        if not hist or sum(len(t) for t in hist) < MIN_RECENT:
            return p_stored if ref_kind == "both" else float("nan")
        p_recent = test(torch.cat(list(hist)), xb, self._g)
        return p_recent if ref_kind == "recent" else max(p_stored, p_recent)           # "both": reject only if BOTH references reject

    def observe(self, x, y=None, ctx=None):
        out = self._observe_patched(x, y, ctx)
        if self.set_cfg is not None and self.set_cfg[1] in ("recent", "both"):
            xn = self._normalize(x.float())
            for eid in torch.unique(out.assigned).tolist():
                if eid >= 0:
                    self.recent.setdefault(eid, deque(maxlen=5)).append(xn[out.assigned == eid])
        return out


def run(name: str, seed: int, variant: str) -> dict:
    t0 = time.time()
    st = A.get_stream(name, seed, False)
    pool = SetGatePool(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=STEPS[A.family(name)], q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"),
                       set_cfg=VARIANTS[variant])
    lab_counts = {f"{lab}_{ev}": 0 for lab in ("onset", "known") for ev in ("novel", "known", "skip", "none")}
    last_seg, within = None, 0
    for (k, seg, _), (x, y, ctx) in zip(st._walk(64, seed), st.algo_batches(64, seed)):
        within = within + 1 if k == last_seg else 0
        last_seg = k
        lab = "onset" if (seg.first_visit and within == 0 and k > 0) else ("known" if ((seg.first_visit and within >= 3) or not seg.first_visit) else None)
        n0 = len(pool.set_events)
        pool.observe(x, y, ctx)
        if lab:
            lab_counts[f"{lab}_{pool.set_events[n0] if len(pool.set_events) > n0 else 'none'}"] += 1
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    ev = {k: sum(1 for e in pool.events if e.kind == k) for k in ("spawn", "extend", "alarm", "abstain")}
    return {"stream": name, "seed": seed, "variant": variant, "acc": float((pool.predict(x).probs.argmax(1) == y).float().mean()), "experts": len(pool.experts),
            "regimes": len(st.regimes), "set_novel": pool.set_events.count("novel"), "set_known": pool.set_events.count("known"), "set_skip": pool.set_events.count("skip"), **ev,
            **lab_counts, "secs": round(time.time() - t0, 1)}


def table(rows: list) -> None:
    m = lambda s, v, k: float(np.mean([r[k] for r in rows if r["stream"] == s and r["variant"] == v]))     # noqa: E731
    streams = [s for s in STREAMS if any(r["stream"] == s for r in rows)]
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    has_ab = "abstain" in rows[0]
    print("\nstream          variant         accuracy %   experts   spawn  extend  alarm" + (" abstain" if has_ab else "") + "  set-novel  set-known  set-skip", flush=True)
    for s in streams:
        for v in variants:
            if any(r["stream"] == s and r["variant"] == v for r in rows):
                print(f"{s:15s} {v:14s} {100 * m(s, v, 'acc'):9.1f} {m(s, v, 'experts'):9.1f} {m(s, v, 'spawn'):7.1f} {m(s, v, 'extend'):6.1f} {m(s, v, 'alarm'):6.1f} "
                      + (f"{m(s, v, 'abstain'):7.1f} " if has_ab else "") + f"{m(s, v, 'set_novel'):10.1f} {m(s, v, 'set_known'):10.1f} {m(s, v, 'set_skip'):9.1f}", flush=True)
    for v in variants:
        if v == "vote":
            continue
        guards = [s for s in ("F-15", "S1", "S2-5", "S2-30", "OV-0.0") if s in streams]
        merges = [s for s in ("OV-0.6", "OV-0.85") if s in streams]
        if guards:
            a = all(abs(m(s, v, "experts") - m(s, "vote", "experts")) <= 0.5 for s in guards)
            print(f"{v}: (a) no spurious experts on {guards}: {'MET' if a else 'NOT MET'}", flush=True)
        if merges:
            b = all(m(s, v, "experts") >= 8 and 100 * (m(s, v, "acc") - m(s, "vote", "acc")) >= 8.0 for s in merges)
            print(f"{v}: (b) merges cured on {merges} (>= 8 experts and >= +8 points): {'MET' if b else 'NOT MET'}", flush=True)


def table_labels(rows: list) -> None:
    """Batch-level behaviour of the set test, with ground truth: KNOWN batches (should not be called novel) and ONSET batches (first batch of a new regime)."""
    if not rows or "known_novel" not in rows[0]:
        return
    streams = [s for s in STREAMS if any(r["stream"] == s for r in rows)]
    variants = [v for v in VARIANTS if v != "vote" and any(r["variant"] == v for r in rows)]
    print("\nbatch-level (sum over seeds): KNOWN batches called novel as a set (false alarm) | ONSET batches: set test called novel / known / skip / did not run (per-sample gate decided)", flush=True)
    for s in streams:
        for v in variants:
            rr = [r for r in rows if r["stream"] == s and r["variant"] == v]
            if not rr:
                continue
            tot = lambda k: sum(r[k] for r in rr)                                           # noqa: E731
            kn = sum(tot(f"known_{e}") for e in ("novel", "known", "skip", "none"))
            on = sum(tot(f"onset_{e}") for e in ("novel", "known", "skip", "none"))
            print(f"{s:14s} {v:13s} known batches {kn:4d}: called novel {tot('known_novel'):3d} ({100 * tot('known_novel') / max(kn, 1):4.1f}%), skip {tot('known_skip'):3d}, test not run {tot('known_none'):3d} | "
                  f"onset batches {on:3d}: novel {tot('onset_novel'):3d} known {tot('onset_known'):3d} skip {tot('onset_skip'):3d} not run {tot('onset_none'):3d}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=",".join(STREAMS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        rows = [r for f in sorted(HERE.glob("spike_set_gate_*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
        table(rows)
        table_labels(rows)
        sys.exit(0)
    rows = []
    for name in args.streams.split(","):
        for variant in args.variants.split(","):
            for seed in SEEDS:
                rows.append(run(name, seed, variant))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        (HERE / f"spike_set_gate_{args.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    table(rows)
    table_labels(rows)

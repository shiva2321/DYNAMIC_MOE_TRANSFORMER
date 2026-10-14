"""HISTORICAL since amendments A-15 and A-16 (2026-10-14): persistence is now product code (config `set_gate_confirm`, default 2) and the group veto has its own level
(`group_veto_q`, default 0.2). This file keeps its own exploratory implementations of both, and pins `set_gate_confirm=1` and a veto level equal to `q` in the arms it builds,
so that rerunning it reproduces the study it was written for (`SENS_RV3.md`) with either version of the product code.

EXPLORATORY, dev seeds 100-109 only, NOT pre-registered: why do the guard streams lose a little with the batch-level gate, and which small changes would fix it?

Context (RV3 dev output): OV-0.6 +18.1 and OV-0.85 +6.4 points, but F-30 (-0.5, interval floor -2.6) and OV-0.0 (-1.0, floor -2.6) miss the guard rule S3-2, and on S1 the
50/50 mixed groups miss S3-7 (-5.2 against -5.0). The per-seed differences are 0 in most seeds and -3 to -6 in a few, each time with ONE extra expert: a false "novel as a set"
verdict on a known batch (level 0.001 per test) sends the batch to probation and the evidence test then often spawns a duplicate expert.

Variants of the gate (everything else exactly as in the dev run; breaker (12, 0.35, 100)):
  a1e-4          level 0.0001 with 9999 permutations (false alarms ten times rarer; a rejecting test costs ten times more permutations, a known batch the same)
  a3e-4          level 0.0003 with 3999 permutations
  persist        a doubtful batch is acted on only when the NEXT batch is doubtful too; the first one is skipped (not trained, not sent to probation)
  a1e-4+persist  both
Group answering (post-hoc on the same trained pools): the per-input veto level swept over 0.02 (today), 0.05, 0.10, 0.20, 0.40.

    python research/growmoe_results/rv2c/sens_dev.py run     [--smoke] [--variants a1e-4,persist,a1e-4+persist] [--force]
    python research/growmoe_results/rv2c/sens_dev.py report  [--smoke] [--variants ...]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_rv2c as R                                                                    # noqa: E402  (also puts rv2a, m0 and the repository root on sys.path)
from growmoe.fairfight import rv2c_decision, rv2c_verdicts                              # noqa: E402
from growmoe.groupeval import evaluate_groups, make_groups                              # noqa: E402
from growmoe.metrics import paired_diff                                                 # noqa: E402

M = R.M
BASE_ARM = "A_set@12-0.35-100"
VARIANTS = {"a1e-4": dict(alpha=1e-4, perms=9999), "a3e-4": dict(alpha=3e-4, perms=3999), "persist": dict(persist=1), "persist2": dict(persist=2), "persist3": dict(persist=3),
            "a1e-4+persist": dict(alpha=1e-4, perms=9999, persist=1)}
VETO_Q = (0.02, 0.05, 0.10, 0.20, 0.40)
_HELD_SENS = "held_sens"                                                                # the product's own "held" verdict (amendment A-15) diverts samples; this study's hold must not trigger that

# ---- patches applied at import time, so that worker processes (which re-import this file) have them too
_parse, _cfg, _group_eval = R.parse_arm, R.arm_cfg, R.group_eval


def _split(arm: str):
    base, _, tag = arm.partition("~")
    return base, tag


def parse_arm(arm: str):
    return _parse(_split(arm)[0])


def arm_cfg(arm: str, st, seed: int, steps: int):
    base, tag = _split(arm)
    cfg = _cfg(base, st, seed, steps)
    if hasattr(cfg, "set_gate_confirm"):                                              # since amendments A-15/A-16 the product has persistence and a veto level of its own; these exploratory
        cfg = dataclasses.replace(cfg, set_gate_confirm=1, group_veto_q=cfg.q)         # arms implement both themselves, so they pin the first-build behaviour (act at once, veto at q)
    if tag:
        v = VARIANTS[tag]
        if "alpha" in v:
            cfg = dataclasses.replace(cfg, set_gate_alpha=v["alpha"], set_gate_perms=v["perms"])
        cfg.persist = int(v.get("persist", 0))                                        # 1: skip the first doubtful batch; 2: keep its accepted samples aside and send them to probation with the second
    return cfg


class SensPool(R.LoggingPool):
    """LoggingPool plus the exploratory persistence rules and a sweepable group veto level (nothing of this is product code).
    persist 1: the first doubtful batch is skipped. persist 2: its accepted samples wait and join the cluster of the second doubtful batch.
    persist 3: the WHOLE first doubtful batch (flagged and accepted samples) waits, so no spawn can start from a biased subset of the new regime."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._persist, self._prev_doubtful, self.veto_q = int(getattr(cfg, "persist", 0)), False, cfg.q
        self._pending, self._pending_all, self._pending_step = None, [], -10
        self._hold_now, self._held_this_call, self._stash = False, False, []
        if self._persist == 3:
            self._probation_add = self.probation.add
            self.probation.add = self._stash_or_add

    def _stash_or_add(self, x, y, arrival):
        if self._hold_now:                                                              # the flagged samples of a held batch are diverted into the stash
            self._stash.append((x.clone(), None if y is None else y.clone(), arrival.clone()))
        else:
            self._probation_add(x, y, arrival=arrival)

    def observe(self, x, y=None, ctx=None):
        self._hold_now, self._held_this_call, self._stash = False, False, []
        rec = super().observe(x, y, ctx)
        if self._held_this_call:
            self._pending_all.extend(self._stash)
        self._hold_now = False
        if rec.meta.get("set_verdict") == _HELD_SENS:                                   # report it under the name this study always used
            rec.meta["set_verdict"] = "held"
            self.verdict_log[-1] = "held"
        return rec

    def _set_gate_step(self, x, y_t, assigned, eids):
        out, mask, verdict = super()._set_gate_step(x, y_t, assigned, eids)
        if not self._persist:
            return out, mask, verdict
        if verdict == "novel":
            if self._prev_doubtful:                                                     # second doubtful batch in a row: act
                if self._persist == 2 and self._pending is not None:
                    px, py, pa, pstep = self._pending
                    self._pending = None
                    if self.step - pstep <= 3:
                        self.probation.add(px, py, arrival=pa)
                if self._persist == 3:
                    pend, self._pending_all = self._pending_all, []
                    if self.step - self._pending_step <= 3:
                        for px, py, pa in pend:                                         # the whole first batch joins the cluster
                            self._probation_add(px, py, arrival=pa)
                return out, mask, verdict
            self._prev_doubtful = True
            acc = assigned >= 0
            skipped = assigned.clone()
            skipped[acc] = -1                                                           # first doubtful batch: its accepted samples are neither trained nor sent to probation now
            start = self.samples_seen - x.shape[0]
            if self._persist == 2:
                self._pending = (x[acc].clone(), y_t[acc].clone() if y_t is not None else None, start + torch.where(acc)[0], self.step)
            if self._persist == 3:
                self._pending_all = [(x[acc].clone(), y_t[acc].clone() if y_t is not None else None, start + torch.where(acc)[0])]
                self._pending_step, self._hold_now, self._held_this_call = self.step, True, True
            return skipped, None, _HELD_SENS
        if verdict == "known":
            self._prev_doubtful = False
            self._pending, self._pending_all = None, []
        return out, mask, verdict

    def _answer_groups(self, x, pred, groups):
        field = "group_veto_q" if hasattr(self.cfg, "group_veto_q") else "q"             # the veto reads q before amendment A-16 and group_veto_q after it
        if self.veto_q == getattr(self.cfg, field):
            return super()._answer_groups(x, pred, groups)
        saved = getattr(self.cfg, field)
        setattr(self.cfg, field, self.veto_q)                                           # predict() already ran with the true q; only the veto reads this from here on
        try:
            return super()._answer_groups(x, pred, groups)
        finally:
            setattr(self.cfg, field, saved)


def group_eval(pool, st, seed: int) -> dict:
    out = _group_eval(pool, st, seed)                                                   # the standard measurement (veto level = q), keys as the verdict function expects
    sweep = {}
    for qv in VETO_Q:
        pool.veto_q = qv
        sweep[str(qv)] = {}
        for kind in ("pure64", "pure16", "mixed32_32", "mixed58_6"):
            size, minor = R.KINDS[kind]
            try:
                x, y, g = make_groups(st.test_sets(), size, minor, seed)
            except ValueError:
                continue
            sweep[str(qv)][kind] = evaluate_groups(pool, x, y, g)
    pool.veto_q = pool.cfg.q
    out["sweep"] = sweep
    return out


R.parse_arm, R.arm_cfg, R.LoggingPool, R.group_eval = parse_arm, arm_cfg, SensPool, group_eval


# ---- driver
def out_dir(args) -> Path:
    d = R.out_dir(args)
    (d / "sens").mkdir(exist_ok=True)
    return d


def cmd_run(args) -> None:
    out = out_dir(args)
    seeds = R.SMOKE_DEV if args.smoke else R.DEV_SEEDS
    assert all(s >= 100 for s in seeds), "exploratory runs use dev seeds only"
    arms = [f"{BASE_ARM}~{v}" for v in args.variants.split(",")]
    specs = R.specs_for(R.STREAMS + [R.SORTED], seeds, arms, args.smoke)
    M._parallel(R.run_task, specs, R.path_of(out, "sens"), args.workers, args.threads, "sens", args.force, R.tag_of)


def effect(a: dict, b: dict) -> str:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return "n/a"
    d = paired_diff([100 * a[i]["acc"] for i in common], [100 * b[i]["acc"] for i in common])
    return f"{d['mean_diff']:+6.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]"


def cmd_report(args) -> None:
    out = out_dir(args)
    dev = json.loads((out / "dev_runs.json").read_text(encoding="utf-8"))
    vote = [r for r in dev if r["method"] == "A_vote"]
    base = [r for r in dev if r["method"] == "A_set"]
    sets = {"baseline (level 0.001, 999 permutations)": base}
    for v in args.variants.split(","):
        sets[v] = R.records([json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "sens").glob(f"*~{v}.json"))])
    streams = R.STREAMS + [R.SORTED]
    L = ["# RV3 dev sensitivity (EXPLORATORY, dev seeds 100-109, not pre-registered)\n"]
    for name, rows in sets.items():
        if not rows:
            L.append(f"## {name}: no runs found\n")
            continue
        by = {}
        for r in rows:
            by.setdefault(r["stream"], {})[r["seed"]] = r
        byv = {}
        for r in vote:
            byv.setdefault(r["stream"], {})[r["seed"]] = r
        L += [f"## {name}\n", "| Stream | A_set - A_vote (points) | experts set / vote | known batches called new | doubtful (held) | breaker opens |", "|---|---|---|---|---|---|"]
        for s in streams:
            if s in by and s in byv:
                a, b = by[s], byv[s]
                held = sum(r["verdict_counts"].get("held", 0) for r in a.values())
                L.append(f"| {s} | {effect(a, b)} | {np.mean([r['n_experts'] for r in a.values()]):.1f} / {np.mean([r['n_experts'] for r in byv[s].values()]):.1f} | "
                         f"{sum(r['fn_called_novel'] for r in a.values())} of {sum(r['fn_known'] for r in a.values())} | {held} | {sum(r['breaker_opens'] for r in a.values())} |")
        res = rv2c_verdicts(vote + rows, {"differences": []})
        L.append("\nStatements on these dev runs: " + ", ".join(f"{k} {v['verdict']}" for k, v in res.items()))
        L.append("\n" + rv2c_decision(res) + "\n")
    for name, rows in sets.items():                                                     # the sweep only exists in sensitivity runs (the dev baseline did not record it)
        if not any(r.get("groups", {}).get("sweep") for r in rows):
            continue
        L += [f"## Group veto level, {name} pools (post-hoc; group answer minus per-input answer, points; mean over seeds)\n",
              "| veto level | OV-0.85 pure64 | OV-0.6 pure64 | OV-0.85 pure16 | worst mixed 50/50 (stream) | worst mixed 90/10 (stream) | S1 mixed 50/50 |", "|---|---|---|---|---|---|---|"]
        for qv in VETO_Q:
            def delta(stream, kind):
                vals = [100 * (r["groups"]["sweep"][str(qv)][kind]["group"] - r["groups"]["sweep"][str(qv)][kind]["single"]) for r in rows
                        if r["stream"] == stream and r.get("groups", {}).get("sweep", {}).get(str(qv), {}).get(kind)]
                return float(np.mean(vals)) if vals else float("nan")
            worst = {}
            for kind in ("mixed32_32", "mixed58_6"):
                cand = [(delta(s, kind), s) for s in R.STREAMS]
                worst[kind] = min((c for c in cand if c[0] == c[0]), default=(float("nan"), "-"))
            L.append(f"| {qv} | {delta('OV-0.85', 'pure64'):+.2f} | {delta('OV-0.6', 'pure64'):+.2f} | {delta('OV-0.85', 'pure16'):+.2f} | {worst['mixed32_32'][0]:+.2f} ({worst['mixed32_32'][1]}) | "
                     f"{worst['mixed58_6'][0]:+.2f} ({worst['mixed58_6'][1]}) | {delta('S1', 'mixed32_32'):+.2f} |")
        L.append("")
    (out / "SENS_RV3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", default="a1e-4,persist,a1e-4+persist")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    assert all(v in VARIANTS for v in args.variants.split(",")), f"variants must be among {list(VARIANTS)}"
    torch.set_num_threads(args.threads)
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

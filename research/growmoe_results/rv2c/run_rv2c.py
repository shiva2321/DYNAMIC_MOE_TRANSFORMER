"""RV2-C (routing v2 stage 3: the batch-level gate while learning, group answering while predicting). Pre-registration: PREREG.md (written by `prereg`, read-only afterwards).

    python research/growmoe_results/rv2c/run_rv2c.py trace  [--smoke]        # sanity check on 3 streams x 3 dev seeds before the expensive dev step
    python research/growmoe_results/rv2c/run_rv2c.py dev    [--smoke]        # breaker grid (spec section 8), then A_vote vs A_set on dev seeds 100-109
    python research/growmoe_results/rv2c/run_rv2c.py prereg [--smoke]        # freeze bars_rv2c.json (sha256) and write PREREG.md from the dev results
    python research/growmoe_results/rv2c/run_rv2c.py run    [--smoke] [--seeds 60-79]
    python research/growmoe_results/rv2c/run_rv2c.py m5     [--smoke] [--seeds 60-79]      # reported arm on the real-text stream (not pre-registered)
    python research/growmoe_results/rv2c/run_rv2c.py report [--smoke]

Arms: A_vote = today's PhyloMoE defaults (batch vote and, above 64 experts, the cascade) with the budget tuned for A_vote; A_set = the same with set_gate=True and the
breaker chosen on dev seeds from the grid declared in the spec. Group answering is evaluated on the trained pools of both arms (group_vote switched on after training,
so the training is identical to the arm without it), per input and per group on the same inputs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (HERE.parent / "rv2a", HERE.parent / "m2a", HERE.parent / "m0", ROOT):
    sys.path.insert(0, str(_p))

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m0 as M                                                                      # noqa: E402
import run_rv2a as A                                                                    # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.fairfight import BARS_RV2C, rv2c_decision, rv2c_verdicts, run_fast, run_record   # noqa: E402
from growmoe.groupeval import evaluate_groups, make_groups                              # noqa: E402
from growmoe.metrics import paired_diff                                                 # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.streams_order import label_sorted                                          # noqa: E402

BASE = dict(q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", arm_monitor_after_infancy=True)   # M0's A_KW without its historical pins
STREAMS = ["F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5"]      # the ten pre-registered streams, heaviest first
SORTED = "F-15-SORTED"                                                                   # reported guard: the same F-15 data with every batch ordered by label
GRID = [(12, 0.35, 100), (12, 0.50, 100), (20, 0.30, 100), (30, 0.30, 100)]            # (len, rate, pause): declared in spec section 8
CONFIRM, GROUP_VETO_Q = 2, 0.2                                                           # amendments A-15 (persistence) and A-16 (group veto level), fixed on dev seeds 100-109 before any confirmatory seed
NEVER_FIRE = ["OV-0.85", "OV-0.6", "S1", "S2-30", "F-30"]                                # the breaker must not fire on these dev runs
PRESSURE = [SORTED, "OV-0.0"]                                                            # ... and should hold the expert count down on these
MMD_STREAMS = ["OV-0.6", "OV-0.85", "S1"]                                                # reported arm: MMD instead of distance of means
DEV_SEEDS, SMOKE_DEV = tuple(range(100, 110)), (100,)
SMOKE_SEEDS = (60, 61)
KINDS = {"pure64": (64, 0), "pure16": (16, 0), "pure8": (8, 0), "mixed32_32": (64, 32), "mixed58_6": (64, 6)}   # (group size, minority rows)
METHOD = {"A_vote": "A_vote", "A_set": "A_set", "A_set_nobreaker": "A_set_nobreaker", "A_set_mmd": "A_set_mmd"}
LABEL = {"A_vote": "A_vote (today's default)", "A_set": "A_set (batch-level gate)", "A_set_nobreaker": "A_set without the breaker (reported)",
         "A_set_mmd": "A_set with MMD (reported)"}


def out_dir(args) -> Path:
    d = HERE / "_smoke" if args.smoke else HERE
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_arm(arm: str):
    """'A_set@12-0.35-100' -> ('A_set', (12, 0.35, 100)); 'A_vote' -> ('A_vote', None)."""
    name, _, tail = arm.partition("@")
    if not tail:
        return name, None
    n, r, p = tail.split("-")
    return name, (int(n), float(r), int(p))


def steps_for(name: str) -> int:
    """The budget tuned for A_vote (stage 1's default arm) on the stream's family; the label-sorted stream uses F-15's."""
    fam = A.family(name.replace("-SORTED", ""))
    return json.loads((ROOT / "research/growmoe_results/rv2a/tuning_rv2a.json").read_text(encoding="utf-8"))["A_vote"][fam]["steps"]


def build_stream(name: str, seed: int, smoke: bool):
    st = A.get_stream(name.replace("-SORTED", ""), seed, smoke)
    return label_sorted(st) if name.endswith("-SORTED") else st


def arm_cfg(arm: str, st, seed: int, steps: int) -> PhyloCfg:
    name, breaker = parse_arm(arm)
    kw = dict(BASE, seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps)
    if name == "A_vote":
        return PhyloCfg(set_gate=False, group_vote=False, group_veto_q=GROUP_VETO_Q, **kw)       # explicit: A_vote stays today's pool even after a default is flipped (the group veto level is the same in both arms)
    n, rate, pause = breaker
    if name == "A_set_nobreaker":
        n = 10 ** 6                                                                     # the memory never fills, so the breaker never opens
    return PhyloCfg(set_gate=True, set_gate_breaker_len=n, set_gate_breaker_rate=rate, set_gate_breaker_pause=pause, set_gate_confirm=CONFIRM, group_veto_q=GROUP_VETO_Q,
                    set_gate_stat="mmd" if name == "A_set_mmd" else "meandist", **kw)


class LoggingPool(PhyloMoE):
    """PhyloMoE that remembers the set-gate verdict of every observe call and the seconds spent in observe."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.verdict_log, self.observe_seconds = [], 0.0

    def observe(self, x, y=None, ctx=None):
        t0 = time.perf_counter()
        rec = super().observe(x, y, ctx)
        self.observe_seconds += time.perf_counter() - t0
        self.verdict_log.append(rec.meta.get("set_verdict"))
        return rec


def batch_truth(st, seed: int) -> list:
    """Ground truth of every observe call of run_fast (same walk): 'onset' = first batch of a new regime, 'known' = batch 3 or later of a first visit or any batch of a
    revisit, None otherwise. Only meaningful for single-source streams."""
    out, last, within = [], None, 0
    for k, seg, _ in st._walk(64, seed):
        within = within + 1 if k == last else 0
        last = k
        out.append("onset" if (seg.first_visit and within == 0 and k > 0) else ("known" if ((seg.first_visit and within >= 3) or not seg.first_visit) else None))
    return out


def group_eval(pool, st, seed: int) -> dict:
    """Per-input and per-group accuracy of the trained pool on groups built from the held-out sets; group answering is switched on only here, after training."""
    pool.cfg.group_vote, pool.cfg.group_veto = True, True
    out = {}
    for kind, (size, minor) in KINDS.items():
        try:
            x, y, g = make_groups(st.test_sets(), size, minor, seed)
        except ValueError:
            out[kind] = None
            continue
        out[kind] = evaluate_groups(pool, x, y, g)
    return out


def run_task(spec: dict) -> dict:
    name, seed, arm, smoke = spec["stream"], spec["seed"], spec["arm"], spec["smoke"]
    st = build_stream(name, seed, smoke)
    steps, t0 = steps_for(name), time.perf_counter()
    cfg = arm_cfg(arm, st, seed, steps)
    holder = []

    def make():
        holder.append(LoggingPool(cfg))
        return holder[0]

    res = run_fast(make, st, seed, eval_every="final")
    pool = holder[0]
    truth = batch_truth(st, seed)
    verdicts = pool.verdict_log
    assert len(verdicts) == len(truth), "the pool saw a different number of batches than the stream walk"
    extra = {"regimes": len(st.regimes), "steps": steps, "arm": arm, "observe_seconds": pool.observe_seconds,
             "fn_known": sum(t == "known" for t in truth), "fn_called_novel": sum(t == "known" and v == "novel" for t, v in zip(truth, verdicts)),
             "fn_doubtful": sum(t == "known" and v in ("novel", "held") for t, v in zip(truth, verdicts)),                # amendment A-15: "novel" = acted on at once, "held" = waiting for confirmation
             "onset_total": sum(t == "onset" for t in truth), "onset_called_novel": sum(t == "onset" and v == "novel" for t, v in zip(truth, verdicts)),
             "onset_doubtful": sum(t == "onset" and v in ("novel", "held") for t, v in zip(truth, verdicts)),
             "verdict_counts": {str(v): verdicts.count(v) for v in set(verdicts)}, "set_stats": res.summary.get("set_stats"),
             "breaker_opens": sum(e.kind == "breaker_open" for e in pool.events)}
    extra["sig_vectors"] = sum(r.meta["sig"].n for r in pool.experts.values())                                   # where the stored vectors sit (they add up to run_record's stored_vectors)
    extra["replay_vectors"] = sum(r.replay.n for r in pool.experts.values() if r.replay is not None)
    extra["window_vectors"] = sum(r.meta["windows"].n_samples() for r in pool.experts.values() if r.meta.get("windows") is not None)
    extra["groups"] = group_eval(pool, st, seed)
    rec = run_record(name, METHOD[parse_arm(arm)[0]], seed, res, **extra)
    return {"spec": {k: spec[k] for k in ("stream", "seed", "arm", "smoke")}, "records": [rec], "seconds": time.perf_counter() - t0, "peak_mb": M._peak_mb()}


def specs_for(streams, seeds, arms, smoke) -> list:
    return [dict(stream=s, seed=sd, arm=a, smoke=smoke) for s in streams for sd in seeds for a in arms]


def path_of(out: Path, sub: str):
    return lambda s: out / sub / f"{s['stream']}__{s['seed']}__{s['arm'].replace('@', '_at_')}.json"


def tag_of(s: dict) -> str:
    return f"{s['stream']} seed {s['seed']} {s['arm']}"


def records(tasks) -> list:
    return [r for t in tasks for r in t["records"]]


def mean_of(rows, key) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


# ------------------------------------------------------------------------------------------------ dev
def cmd_dev(args) -> None:
    out = out_dir(args)
    seeds = SMOKE_DEV if args.smoke else DEV_SEEDS
    assert all(s >= 100 for s in seeds), "dev seeds are 100-109"
    pressure_seeds = seeds[:3]
    # step 1 (cheap): every breaker setting on the two pressure streams
    arms = [f"A_set@{n}-{r}-{p}" for n, r, p in GRID]
    tasks = M._parallel(run_task, specs_for(PRESSURE, pressure_seeds, arms, args.smoke), path_of(out, "dev"), args.workers, args.threads, "dev-pressure", args.force, tag_of)
    runs = records(tasks)
    score = {}
    for g in GRID:
        arm = f"A_set@{g[0]}-{g[1]}-{g[2]}"
        rows = [r for r in runs if r["arm"] == arm]
        score[g] = sum(mean_of([r for r in rows if r["stream"] == s], "n_experts") for s in PRESSURE)
    order = sorted(GRID, key=lambda g: (score[g], -g[0]))                               # fewest experts, ties to the larger memory
    # step 2 (expensive, lazy): the best setting must never fire on the five dev streams
    chosen, checks = None, {}
    for g in order:
        arm = f"A_set@{g[0]}-{g[1]}-{g[2]}"
        t2 = M._parallel(run_task, specs_for(NEVER_FIRE, seeds, [arm], args.smoke), path_of(out, "dev"), args.workers, args.threads, "dev-never-fire", args.force, tag_of)
        opens = sum(r["breaker_opens"] for r in records(t2))
        checks[str(g)] = opens
        if opens == 0:
            chosen = g
            break
    reason = "first setting in the order (fewest experts on the pressure streams, ties to the larger len) that never fired"
    if chosen is None:
        chosen = min(order, key=lambda g: (checks.get(str(g), 10 ** 9), score[g]))
        reason = "NO setting satisfied 'never fires'; the one with the fewest openings was taken"
    (out / "dev_breaker.json").write_text(json.dumps({"grid": [list(g) for g in GRID], "pressure_experts": {str(g): score[g] for g in GRID}, "order": [list(g) for g in order],
                                                      "openings_on_never_fire_streams": checks, "chosen": list(chosen), "reason": reason, "dev_seeds": list(seeds)}, indent=1), encoding="utf-8")
    print("breaker grid, experts on pressure streams:", {str(g): round(v, 1) for g, v in score.items()}, "-> chosen", chosen, "|", reason, flush=True)
    # step 3: the dev comparison on every stream (steps 1 and 2 are reused from disk)
    arm = f"A_set@{chosen[0]}-{chosen[1]}-{chosen[2]}"
    t3 = M._parallel(run_task, specs_for(STREAMS + [SORTED], seeds, ["A_vote", arm], args.smoke), path_of(out, "dev"), args.workers, args.threads, "dev-compare", args.force, tag_of)
    dev = [r for r in records(t3) if r["arm"] in ("A_vote", arm)]
    (out / "dev_runs.json").write_text(json.dumps(dev, indent=1), encoding="utf-8")
    for s in STREAMS + [SORTED]:
        a = {r["seed"]: r for r in dev if r["stream"] == s and r["method"] == "A_set"}
        b = {r["seed"]: r for r in dev if r["stream"] == s and r["method"] == "A_vote"}
        common = sorted(set(a) & set(b))
        if len(common) >= 2:
            d = paired_diff([100 * a[i]["acc"] for i in common], [100 * b[i]["acc"] for i in common])
            print(f"  {s:13s} A_set - A_vote {d['mean_diff']:+6.2f} pts [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]  experts {mean_of(list(a.values()), 'n_experts'):5.1f} vs "
                  f"{mean_of(list(b.values()), 'n_experts'):5.1f}  breaker opens {sum(r['breaker_opens'] for r in a.values())}  known batches doubted "
                  f"{sum(r['fn_doubtful'] for r in a.values())} (acted on {sum(r['fn_called_novel'] for r in a.values())}) of {sum(r['fn_known'] for r in a.values())}  "
                  f"stored vectors {mean_of(list(a.values()), 'stored_vectors'):.0f} vs {mean_of(list(b.values()), 'stored_vectors'):.0f}", flush=True)


def chosen_arm(out: Path) -> str:
    g = json.loads((out / "bars_rv2c.json").read_text(encoding="utf-8"))["breaker"]
    return f"A_set@{g[0]}-{g[1]}-{g[2]}"


def golden_now() -> dict:
    spec_path = HERE / "golden.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("golden", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"differences": mod.differences(), "recorded_sha256": A._sha(HERE / "golden_flags_off.json")}


def cmd_prereg(args) -> None:
    out = out_dir(args)
    if (out / "bars_rv2c.sha256").exists():
        sys.exit("bars_rv2c.json is already frozen (bars_rv2c.sha256 exists): prereg cannot rewrite it. A change after the freeze is a recorded amendment, not a rerun.")
    br = json.loads((out / "dev_breaker.json").read_text(encoding="utf-8"))
    dev = json.loads((out / "dev_runs.json").read_text(encoding="utf-8"))
    golden = golden_now() if not args.smoke else {"differences": [], "recorded_sha256": "smoke"}
    expected = {k: v["verdict"] for k, v in rv2c_verdicts(dev, golden).items()}
    expected = {k: ("PASS" if v == "NOT MEASURABLE" else v) for k, v in expected.items()}      # dev cannot measure what needs the fresh run; the pre-registered expectation is PASS
    bars_path = out / "bars_rv2c.json"
    bars_path.write_text(json.dumps({"bars": BARS_RV2C, "breaker": br["chosen"], "breaker_reason": br["reason"], "confirm": CONFIRM, "group_veto_q": GROUP_VETO_Q, "expected": expected, "golden": golden,
                                     "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    rules = {k: v["rule"] for k, v in rv2c_verdicts([], None).items()}
    L = ["# RV2-C pre-registration (generated by run_rv2c.py prereg; written before any confirmatory seed ran)\n",
         f"Spec: docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md section 7 and its amendments. Arms: `A_vote` (today's defaults, budget tuned for A_vote) and `A_set` (`set_gate=True`, persistence `set_gate_confirm` = {CONFIRM} "
         f"(a doubtful batch is acted on only when the next decided batch is doubtful too), group veto level `group_veto_q` = {GROUP_VETO_Q}, breaker "
         f"(len, rate, pause) = {tuple(br['chosen'])}: {br['reason']}). Confirmatory seeds 60-79; dev seeds 100-109; batch 64. Group answering is evaluated on the trained pools of both arms "
         "(vote + veto, groups of 64 / 16 / 8 single-source test inputs; mixed groups 32+32 and 58+6 of adjacent regimes), per input and per group on the same inputs. "
         "A known batch counts as `called new` (S3-5, kill rule at 2 %) if the set test doubted it, whether the pool acted on it at once (verdict `novel`) or held it for confirmation "
         "(verdict `held`); the acted-on count is reported next to it.\n",
         "## Statements\n", "| ID | Rule | Expected (from dev) |", "|---|---|---|"] + [f"| {k} | {r} | {expected[k]} |" for k, r in rules.items()] + [
         "", "## Decision rule (verbatim from the spec)\n",
         "Adopt 3a (flip `set_gate`) iff S3-1 to S3-5 and S3-9 pass; adopt 3b (flip `group_vote`) iff S3-6 to S3-8 pass. S3-1 passing only at OV-0.6 keeps 3a opt-in and says so. Any of S3-2 "
         "to S3-5 failing: 3a is not adopted. Kill rules: a guard stream losing more than 1.0 point, mean experts above regimes + 0.5, false novelty above 2 %, or less than +8.0 at OV-0.6 "
         "on fresh seeds.\n",
         "## Reported, not gated\n", "Wall-clock of observe with and without 3a; permutations and the early-stop share; breaker openings per run; the label-sorted F-15 with and without the breaker; "
         "MMD against distance of means (OV-0.6, OV-0.85, S1); groups of 8; experts and parameter counts of both arms; the real-text stream M5.\n",
         "## Provenance\n", f"phylo.py {A._sha(ROOT / 'growmoe/pools/phylo.py')}; setgate.py {A._sha(ROOT / 'growmoe/setgate.py')}; settest.py {A._sha(ROOT / 'growmoe/settest.py')}; "
         f"sources.py {A._sha(ROOT / 'growmoe/sources.py')}; groupvote.py {A._sha(ROOT / 'growmoe/groupvote.py')}; groupeval.py {A._sha(ROOT / 'growmoe/groupeval.py')}; "
         f"config.py {A._sha(ROOT / 'growmoe/config.py')}; fairfight.py {A._sha(ROOT / 'growmoe/fairfight.py')}; this script {A._sha(Path(__file__))}; "
         f"golden_flags_off.json {golden['recorded_sha256']}; bars_rv2c.json {A._sha(bars_path)}.\n"]
    (out / "PREREG.md").write_text("\n".join(L), encoding="utf-8")
    print(f"bars_rv2c.json frozen, sha256 {M.freeze_check(bars_path)[:16]}")
    print("breaker", br["chosen"], "expected", expected)


# ------------------------------------------------------------------------------------------------ trace check (before the expensive dev step)
TRACE_ARM = "A_set@12-0.35-100"
TRACE_STREAMS = ["OV-0.6", "OV-0.85", "S1"]


def cmd_trace(args) -> None:
    """Three streams, three dev seeds, the first breaker setting of the grid: a sanity check that the product code behaves like the spike did (dev, exploratory: OV-0.6 about +17.8
    points, OV-0.85 about +5.3, S1 about +0.3; with persistence (A-15) about +17.3, +5.9 and -0.05 on dev seeds 100-109; expert counts close to the regime counts) BEFORE the expensive dev step. It is not a statistical test; it only catches an implementation mistake."""
    out = out_dir(args)
    seeds = SMOKE_DEV if args.smoke else DEV_SEEDS[:3]
    tasks = M._parallel(run_task, specs_for(TRACE_STREAMS, seeds, ["A_vote", TRACE_ARM], args.smoke), path_of(out, "trace"), args.workers, args.threads, "trace", args.force, tag_of)
    runs = records(tasks)
    ok, lines = True, []
    for s in TRACE_STREAMS:
        a = {r["seed"]: r for r in runs if r["stream"] == s and r["method"] == "A_set"}
        b = {r["seed"]: r for r in runs if r["stream"] == s and r["method"] == "A_vote"}
        common = sorted(set(a) & set(b))
        diff = float(np.mean([100 * (a[i]["acc"] - b[i]["acc"]) for i in common]))
        regimes, experts = mean_of(list(a.values()), "regimes"), mean_of(list(a.values()), "n_experts")
        known, called = sum(r["fn_known"] for r in a.values()), sum(r["fn_doubtful"] for r in a.values())
        lines.append(f"  {s:8s} A_set - A_vote {diff:+6.2f} pts | experts A_set {experts:5.1f} A_vote {mean_of(list(b.values()), 'n_experts'):5.1f} (regimes {regimes:.0f}) | "
                     f"known batches doubted {called}/{known} | breaker opens {sum(r['breaker_opens'] for r in a.values())} | "
                     f"observe seconds ratio {mean_of(list(a.values()), 'observe_seconds') / max(mean_of(list(b.values()), 'observe_seconds'), 1e-9):.2f}")
        ok &= experts <= regimes + 3.0 and (called <= 0.05 * max(known, 1))
        if s == "OV-0.6":
            ok &= diff >= 8.0
        if s == "S1":
            ok &= diff >= -1.5
    print("\n".join(lines))
    (out / "trace.json").write_text(json.dumps({"arm": TRACE_ARM, "seeds": list(seeds), "lines": lines, "ok": bool(ok)}, indent=1), encoding="utf-8")
    verdict = "TRACE CHECK PASSED" if ok else "TRACE CHECK FAILED (OV-0.6 gain < +8.0 points, S1 below -1.5, experts above regimes + 3, or more than 5 % false novelty): STOP and report"
    print(verdict + ("  [smoke: tiny streams, the numbers mean nothing]" if args.smoke else ""))


# ------------------------------------------------------------------------------------------------ confirmatory run
def cmd_run(args) -> None:
    out = out_dir(args)
    bars_path = out / "bars_rv2c.json"
    if args.smoke and not bars_path.exists():
        bars_path.write_text(json.dumps({"bars": BARS_RV2C, "breaker": [12, 0.35, 100], "expected": {}, "golden": {"differences": []}}), encoding="utf-8")
    if not bars_path.exists():
        sys.exit("bars_rv2c.json is missing: run dev and prereg first")
    print(f"bars_rv2c.json frozen, sha256 {M.freeze_check(bars_path)[:16]}", flush=True)
    seeds = SMOKE_SEEDS if args.smoke else M.parse_seeds(args.seeds)
    if not args.smoke:
        assert all(60 <= s <= 79 for s in seeds), "the stage-3 confirmatory seeds are 60-79"
    arm = chosen_arm(out)
    nb, mm = arm.replace("A_set@", "A_set_nobreaker@"), arm.replace("A_set@", "A_set_mmd@")
    order = {n: i for i, n in enumerate(STREAMS + [SORTED])}
    specs = specs_for(STREAMS + [SORTED], seeds, ["A_vote", arm], args.smoke) + specs_for([SORTED], seeds, [nb], args.smoke) + specs_for(MMD_STREAMS, seeds, [mm], args.smoke)
    specs.sort(key=lambda s: (order[s["stream"]], s["seed"], s["arm"]))
    M._parallel(run_task, specs, path_of(out, "runs"), args.workers, args.threads, "run", args.force, tag_of)


# ------------------------------------------------------------------------------------------------ reported arm on the real-text stream
_FRONTEND = None                                                                         # one frozen frontend per worker process: it caches the embeddings of the 13,000 texts


def m5_task(spec: dict) -> dict:
    global _FRONTEND
    from growmoe.streams_real import banking_incremental
    from run_m0 import HybridFrontend
    seed, arm, steps = spec["seed"], spec["arm"], spec["steps"]
    t0 = time.perf_counter()
    _FRONTEND = _FRONTEND or HybridFrontend(d=128, seed=0)
    st = banking_incremental(seed=seed, frontend=_FRONTEND)
    cfg = arm_cfg(arm, st, seed, steps)
    holder = []

    def make():
        holder.append(LoggingPool(cfg))
        return holder[0]

    res = run_fast(make, st, seed, eval_every="final")
    pool = holder[0]
    rec = run_record("M5-banking", METHOD[parse_arm(arm)[0]], seed, res, regimes=len(st.regimes), steps=steps, arm=arm, observe_seconds=pool.observe_seconds,
                     set_stats=res.summary.get("set_stats"), breaker_opens=sum(e.kind == "breaker_open" for e in pool.events), verdict_counts={str(v): pool.verdict_log.count(v) for v in set(pool.verdict_log)})
    return {"spec": {k: spec[k] for k in ("seed", "arm", "steps")}, "records": [rec], "seconds": time.perf_counter() - t0, "peak_mb": M._peak_mb()}


def cmd_m5(args) -> None:
    out = out_dir(args)
    m5b = json.loads((HERE.parent / "m5" / "m5b.json").read_text(encoding="utf-8"))
    steps = int(m5b["tuned_steps"]["A_today"])
    seeds = SMOKE_SEEDS if args.smoke else M.parse_seeds(args.seeds)
    arm = chosen_arm(out)
    specs = [dict(seed=s, arm=a, steps=steps) for s in seeds for a in ("A_vote", arm)]
    M._parallel(m5_task, specs, lambda s: out / "m5" / f"{s['seed']}__{s['arm'].replace('@', '_at_')}.json", args.workers, args.threads, "m5", args.force,
                tag_of=lambda s: f"M5 seed {s['seed']} {s['arm']}")


# ------------------------------------------------------------------------------------------------ report
def cmd_report(args) -> None:
    out = out_dir(args)
    runs = records([json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))])
    m5 = records([json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "m5").glob("*.json"))]) if (out / "m5").exists() else []
    bp = out / "bars_rv2c.json"
    doc = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {"expected": {}, "breaker": None, "golden": None}
    golden = doc.get("golden")
    result = rv2c_verdicts(runs, golden, expected=doc["expected"])
    decision = rv2c_decision(result)
    (out / "rv2c_runs.json").write_text(json.dumps({"runs": runs, "m5": m5, "bars": doc, "verdicts": result, "decision": decision}, indent=1), encoding="utf-8")
    by: dict = {}
    for r in runs:
        by.setdefault((r["stream"], r["method"]), []).append(r)
    L = ["# RV2-C results (generated by run_rv2c.py; do not edit by hand)\n"]
    if args.smoke:
        L.append("> **SMOKE RUN. Tiny streams. These numbers mean nothing.**\n")
    L.append(f"* PREREG.md sha256 {A._sha(out / 'PREREG.md') if (out / 'PREREG.md').exists() else '-'}; bars_rv2c.json {A._sha(bp) if bp.exists() else '-'}; this script {A._sha(Path(__file__))}; "
             f"phylo.py {A._sha(ROOT / 'growmoe/pools/phylo.py')}; setgate.py {A._sha(ROOT / 'growmoe/setgate.py')}; settest.py {A._sha(ROOT / 'growmoe/settest.py')}; "
             f"sources.py {A._sha(ROOT / 'growmoe/sources.py')}; groupvote.py {A._sha(ROOT / 'growmoe/groupvote.py')}; fairfight.py {A._sha(ROOT / 'growmoe/fairfight.py')}.\n")
    L += ["## Verdicts (computed by `growmoe.fairfight.rv2c_verdicts`)\n", "| ID | Rule | Verdict | Expected | Detail |", "|---|---|---|---|---|"]
    for k, v in result.items():
        mark = {True: "as expected", False: "**AGAINST expectation**", None: "-"}[v["as_expected"]]
        L.append(f"| {k} | {v['rule']} | **{v['verdict']}** | {v['expected']} ({mark}) | `{json.dumps(v['detail'])}` |")
    L += [f"\n**Decision.** {decision}\n", "## Results per stream (final accuracy %; mean [95% t interval] over seeds)\n",
          "| Stream | Arm | Accuracy % | Experts | Params | Seconds per run | Observe seconds | n |", "|---|---|---|---|---|---|---|---|"]
    for s in STREAMS + [SORTED]:
        for m in METHOD:
            rows = by.get((s, m))
            if rows:
                L.append(f"| {s} | {LABEL[m]} | {M._fmt_ci([r['acc'] for r in rows])} | {mean_of(rows, 'n_experts'):.1f} | {mean_of(rows, 'params'):.0f} | {mean_of(rows, 'wall_clock'):.0f} | "
                         f"{mean_of(rows, 'observe_seconds'):.1f} | {len(rows)} |")
    L += ["", "## Paired differences A_set - A_vote (points)\n"]
    for s in STREAMS + [SORTED]:
        a, b = {r["seed"]: r for r in by.get((s, "A_set"), [])}, {r["seed"]: r for r in by.get((s, "A_vote"), [])}
        common = sorted(set(a) & set(b))
        if len(common) >= 2:
            d = paired_diff([100 * a[i]["acc"] for i in common], [100 * b[i]["acc"] for i in common])
            L.append(f"* {s}: {d['mean_diff']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}], p = {d['p_t']:.3g}, n = {d['n']}")
    L += ["", "## Group answering (accuracy %, per input -> per group with the veto; mean over seeds)\n", "| Stream | Arm | " + " | ".join(KINDS) + " |", "|---|---|" + "---|" * len(KINDS)]
    for s in STREAMS + [SORTED]:
        for m in ("A_vote", "A_set"):
            rows = by.get((s, m))
            if rows:
                cells = []
                for k in KINDS:
                    g = [r["groups"][k] for r in rows if r.get("groups") and r["groups"].get(k)]
                    cells.append(f"{100 * np.mean([x['single'] for x in g]):.1f} -> {100 * np.mean([x['group'] for x in g]):.1f}" if g else "-")
                L.append(f"| {s} | {LABEL[m]} | " + " | ".join(cells) + " |")
    L += ["", "## Reported: cost and behaviour of the batch-level gate\n", "| Stream | observe seconds A_set / A_vote | tests per run | permutations per test | early-stop share | breaker openings per run | known batches doubted (acted on) | onsets doubted (acted on) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in STREAMS + [SORTED]:
        a, b = by.get((s, "A_set")), by.get((s, "A_vote"))
        if a and b:
            st_ = [r["set_stats"] for r in a if r.get("set_stats")]
            tests = sum(x["tests"] for x in st_)
            L.append(f"| {s} | {mean_of(a, 'observe_seconds') / max(mean_of(b, 'observe_seconds'), 1e-9):.2f} | {tests / len(a):.0f} | {sum(x['perms'] for x in st_) / max(tests, 1):.0f} | "
                     f"{sum(x['early_stops'] for x in st_) / max(tests, 1):.2f} | {mean_of(a, 'breaker_opens'):.2f} | {sum(r['fn_doubtful'] for r in a)} ({sum(r['fn_called_novel'] for r in a)}) of {sum(r['fn_known'] for r in a)} | "
                     f"{sum(r['onset_doubtful'] for r in a)} ({sum(r['onset_called_novel'] for r in a)}) of {sum(r['onset_total'] for r in a)} |")
    L += ["", "## Reported: where the stored vectors sit (mean per run; accounted bytes = 4 x parameters + 4 x d x stored vectors, as in growmoe.accounting)\n",
          "| Stream | Arm | experts | signature prototypes | replay | source windows | stored vectors total | accounted MB |", "|---|---|---|---|---|---|---|---|"]
    for s in STREAMS + [SORTED]:
        for m in ("A_vote", "A_set"):
            rows = by.get((s, m))
            if rows and all("sig_vectors" in r for r in rows):
                L.append(f"| {s} | {LABEL[m]} | {mean_of(rows, 'n_experts'):.1f} | {mean_of(rows, 'sig_vectors'):.0f} | {mean_of(rows, 'replay_vectors'):.0f} | {mean_of(rows, 'window_vectors'):.0f} | "
                         f"{mean_of(rows, 'stored_vectors'):.0f} | {mean_of(rows, 'memory_bytes') / 1e6:.2f} |")
    L += ["", "## Reported: the label-sorted guard and the MMD variant\n"]
    for s, ms in ((SORTED, ("A_vote", "A_set", "A_set_nobreaker")), ("OV-0.6", ("A_set", "A_set_mmd")), ("OV-0.85", ("A_set", "A_set_mmd")), ("S1", ("A_set", "A_set_mmd"))):
        cells = [f"{LABEL[m]}: {M._fmt_ci([r['acc'] for r in by[(s, m)]])} %, {mean_of(by[(s, m)], 'n_experts'):.1f} experts" for m in ms if by.get((s, m))]
        if cells:
            L.append(f"* {s}: " + "; ".join(cells))
    if m5:
        L += ["", "## Reported: the real-text stream M5 (banking77, class-incremental; not pre-registered)\n"]
        for m in ("A_vote", "A_set"):
            rows = [r for r in m5 if r["method"] == m]
            if rows:
                L.append(f"* {LABEL[m]}: {M._fmt_ci([r['acc'] for r in rows])} %, {mean_of(rows, 'n_experts'):.1f} experts, n = {len(rows)}")
    (out / "RESULTS_RV2C.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'RESULTS_RV2C.md'}")
    for k, v in result.items():
        print(f"  {k}: {v['verdict']:15s} (expected {v['expected']})")
    print(" ", decision)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["trace", "dev", "prereg", "run", "m5", "report"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seeds", default="60-79")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"trace": cmd_trace, "dev": cmd_dev, "prereg": cmd_prereg, "run": cmd_run, "m5": cmd_m5, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

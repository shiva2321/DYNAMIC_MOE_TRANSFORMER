"""M2-A: dev re-tuning of the repaired pool (seeds 100-102), the confirmatory run on fresh seeds 10-19, verdicts, RESULTS_M2A.md.

Pre-registration: research/growmoe_results/m2a/PREREG.md (read-only). Every number in the output is written by this script from
per-seed records; nothing is typed by hand. Run from the repository root:

    python research/growmoe_results/m2a/run_m2a.py tune   [--smoke] [--workers 2] [--threads 3]
    python research/growmoe_results/m2a/run_m2a.py run    [--smoke] [--only F-15,S1] [--seeds 10-19]
    python research/growmoe_results/m2a/run_m2a.py report [--smoke]

``--smoke`` shrinks every stream and grid so the whole pipeline runs in a few minutes; its outputs go to ``_smoke/`` and mean nothing.
Finished tasks are stored as files and skipped on a re-run. The M0 machinery (parallel runner, stream builders, method factories,
freeze check) is imported from research/growmoe_results/m0/run_m0.py, which stays untouched.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "m0"))
sys.path.insert(0, str(ROOT))

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m0 as M                                                                      # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.fairfight import m2a_decision, m2a_verdicts, run_fast, run_record, tune_steps  # noqa: E402
from growmoe.metrics import paired_diff                                                 # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.streams import s4_outliers, s7_mapping                                     # noqa: E402

# ------------------------------------------------------------------------------------------------ pre-registered settings
TEST_SEEDS = tuple(range(10, 20))
FAMILY = dict(M.FAMILY, S4="perm", S7="perm")
PRIMARY, GUARDS = ("F-30", "F-15"), ("S2-30", "S1", "S4", "S7", "S2-5")
METHODS = {**{s: ["A_old", "A_new", "B2star", "B4", "kNN_all"] for s in PRIMARY}, **{s: ["A_old", "A_new"] for s in GUARDS}}
STREAM_ORDER = list(PRIMARY) + list(GUARDS)
LABEL = {"A_old": "A old (M0 frozen)", "A_new": "A new (monitor armed after infancy)", "B2star": "B2* best single network (M0 choice)",
         "B4": "B4 oracle-routed experts", "kNN_all": "kNN, stores everything"}
SHORT = {"A_old": "A_old", "A_new": "A_new", "B2star": "B2*", "B4": "B4", "kNN_all": "kNN-all"}
SMOKE_SEEDS = (10, 11)
_STREAMS: dict = {}


# ------------------------------------------------------------------------------------------------ streams and methods
def build_stream(name: str, seed: int, smoke: bool):
    if name == "S4":
        kw = dict(n_regimes=3, n_train=128, n_test=50) if smoke else dict(n_regimes=5, n_train=400, n_test=100)
        return s4_outliers(seed=seed, outlier_rate=0.01, label_noise=0.1, **kw)
    if name == "S7":
        return s7_mapping(seed=seed, with_ctx=False, **(dict(n_train=200, n_test=80) if smoke else {}))
    return M.build_stream(name, seed, smoke)


def get_stream(name: str, seed: int, smoke: bool):
    key = (name, seed, smoke)
    if key not in _STREAMS:
        if len(_STREAMS) >= 4:                                                          # bound the memory held by one worker
            _STREAMS.pop(next(iter(_STREAMS)))
        _STREAMS[key] = build_stream(name, seed, smoke)
    return _STREAMS[key]


def factory(method: str, st, seed: int, P: dict):
    if method in ("A_old", "A_new"):
        kw = {**M.A_KW, "arm_monitor_after_infancy": method == "A_new"}
        return lambda: PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=P["steps"], **kw))
    return M.factory(method, st, seed, P)


def params_for(method: str, family: str, tm0: dict, tm2: dict) -> dict:
    if method == "A_old":
        return {"steps": tm0["A"][family]["steps"]}
    if method == "A_new":
        return {"steps": tm2["A_new"][family]["steps"]}
    return M.params_for(method, family, tm0)                                           # B2star / B4 from M0's frozen tuning; kNN needs none


def smoke_m0_tuning() -> dict:
    return {"A": {f: {"steps": 5} for f in ("perm", "types", "func")}, "B4": {"func": {"steps": 5}},
            "B2star": {"func": {"steps": 5, "width": 128, "buffer": 2048}}}


# ------------------------------------------------------------------------------------------------ dev tuning
def tune_task(spec: dict) -> dict:
    family, smoke = spec["family"], spec["smoke"]
    seeds = M.SMOKE_DEV if smoke else M.DEV_SEEDS
    assert all(s >= 100 for s in seeds), "tuning must use dev seeds only"
    name = M.TUNE_STREAM[family]
    streams = {s: get_stream(name, s, smoke) for s in seeds}

    def score(steps: int) -> float:
        return float(np.mean([run_fast(factory("A_new", streams[s], s, {"steps": steps}), streams[s], s, eval_every="final").acc for s in seeds]))

    t0 = time.perf_counter()
    res = tune_steps(score, grid=M.SMOKE_GRID if smoke else M.STEP_GRID, factor=M.STEP_FACTOR, cap=M.STEP_CAP)
    return {"family": family, "method": "A_new", "best_steps": res.best, "at_edge": res.at_edge, "table": {str(k): v for k, v in res.table.items()},
            "stream": name, "dev_seeds": list(seeds), "seconds": time.perf_counter() - t0, "peak_mb": M._peak_mb()}


def cmd_tune(args) -> None:
    out = out_dir(args)
    specs = [dict(family=f, smoke=args.smoke) for f in ("func", "perm", "types")]
    results = M._parallel(tune_task, specs, lambda s: out / "tune" / f"{s['family']}.json", args.workers, args.threads, "tune", args.force,
                          tag_of=lambda s: s["family"])
    tuning = {"A_new": {r["family"]: {"steps": r["best_steps"], "at_edge": r["at_edge"], "table": r["table"], "stream": r["stream"],
                                      "dev_seeds": r["dev_seeds"]} for r in results}}
    (out / "tuning_m2a.json").write_text(json.dumps(tuning, indent=1), encoding="utf-8")
    for f, v in tuning["A_new"].items():
        print(f"  A_new {f:6s} steps={v['steps']:>3d}" + ("   ** at the edge of the search **" if v["at_edge"] else ""))


# ------------------------------------------------------------------------------------------------ confirmatory run
def run_task(spec: dict) -> dict:
    name, seed, smoke = spec["stream"], spec["seed"], spec["smoke"]
    tm0 = smoke_m0_tuning() if smoke else json.loads(Path(spec["tuning_m0"]).read_text(encoding="utf-8"))
    tm2 = json.loads(Path(spec["tuning_m2a"]).read_text(encoding="utf-8"))
    family, st = FAMILY[name], get_stream(name, seed, smoke)
    t0, recs = time.perf_counter(), []
    for m in METHODS[name]:
        P = params_for(m, family, tm0, tm2)
        res = run_fast(factory(m, st, seed, P), st, seed, eval_every="final")
        recs.append(run_record(name, m, seed, res, **P))
    return {"spec": {k: spec[k] for k in ("stream", "seed", "smoke")}, "records": recs, "seconds": time.perf_counter() - t0, "peak_mb": M._peak_mb()}


def cmd_run(args) -> None:
    out = out_dir(args)
    tuning_path, tuning_m0 = out / "tuning_m2a.json", ROOT / "research" / "growmoe_results" / "m0" / "tuning.json"
    if not tuning_path.exists():
        sys.exit("tuning_m2a.json is missing: run the tune step first (the test runs must use frozen budgets)")
    print(f"tuning_m2a.json frozen, sha256 {M.freeze_check(tuning_path)[:16]}", flush=True)
    streams = args.only.split(",") if args.only else STREAM_ORDER
    seeds = SMOKE_SEEDS if args.smoke else M.parse_seeds(args.seeds)
    if not args.smoke:
        assert all(10 <= s <= 19 for s in seeds), "the confirmatory seeds are 10-19"
    order = {n: i for i, n in enumerate(STREAM_ORDER)}
    specs = sorted([dict(stream=n, seed=s, smoke=args.smoke, tuning_m0=str(tuning_m0), tuning_m2a=str(tuning_path)) for n in streams for s in seeds],
                   key=lambda s: (order[s["stream"]], s["seed"]))
    M._parallel(run_task, specs, lambda s: out / "runs" / f"{s['stream']}__{s['seed']}.json", args.workers, args.threads, "run", args.force,
                tag_of=lambda s: f"{s['stream']} seed {s['seed']}")


# ------------------------------------------------------------------------------------------------ report
def out_dir(args) -> Path:
    d = HERE / "_smoke" if args.smoke else HERE
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def _pts(d: dict) -> str:
    lo = "-inf" if d["ci_lo"] is None else f"{d['ci_lo']:+.1f}"
    hi = "+inf" if d["ci_hi"] is None else f"{d['ci_hi']:+.1f}"
    return f"{d['mean_diff_points']:+.1f} pts [{lo}, {hi}] (n={d['n']})"


def cmd_report(args) -> None:
    out = out_dir(args)
    tm2 = json.loads((out / "tuning_m2a.json").read_text(encoding="utf-8"))
    tm0_path = ROOT / "research" / "growmoe_results" / "m0" / "tuning.json"
    tm0 = smoke_m0_tuning() if args.smoke else json.loads(tm0_path.read_text(encoding="utf-8"))
    tasks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))]
    runs = [r for t in tasks for r in t["records"]]
    result = m2a_verdicts(runs)
    decision = m2a_decision(result)
    (out / "m2a_runs.json").write_text(json.dumps({"runs": runs, "tuning": tm2, "verdicts": result, "decision": decision}, indent=1), encoding="utf-8")

    by: dict = {}
    for r in runs:
        by.setdefault((r["stream"], r["method"]), []).append(r)
    seeds = sorted({r["seed"] for r in runs})
    L, w = [], None
    w = L.append
    w("# M2-A results (generated by run_m2a.py; do not edit by hand)\n")
    if args.smoke:
        w("> **SMOKE RUN. Tiny streams and grids. These numbers mean nothing.**\n")
    w(f"* Pre-registration `PREREG.md` sha256 {_sha(HERE / 'PREREG.md')}; this script {_sha(Path(__file__))}; `fairfight.py` {_sha(ROOT / 'growmoe/fairfight.py')}; "
      f"`phylo.py` {_sha(ROOT / 'growmoe/pools/phylo.py')}; `config.py` {_sha(ROOT / 'growmoe/config.py')}.")
    w(f"* Frozen tuning: `tuning_m2a.json` sha256 {_sha(out / 'tuning_m2a.json')}; M0 tuning for A_old, B2* and B4: `tuning.json` sha256 {_sha(tm0_path)}.")
    w(f"* Confirmatory seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds); dev seeds {list(M.DEV_SEEDS)}. Compute summed over tasks: "
      f"{sum(t['seconds'] for t in tasks) / 60:.0f} min; largest worker peak memory {max(t['peak_mb'] for t in tasks):.0f} MB.\n")

    w("## 1. Verdicts (computed by `growmoe.fairfight.m2a_verdicts` from the per-seed records)\n")
    w("| ID | Rule | Verdict | Expected | Numbers |")
    w("|---|---|---|---|---|")
    for k, v in result.items():
        d = v["detail"]
        if v["verdict"] == "NOT MEASURABLE":
            nums = "missing data"
        elif k == "P1":
            nums = "(A_new - B2*) " + "; ".join(f"{s}: {_pts(x)}" for s, x in d.items())
        elif k == "P2":
            nums = "(A_new - A_old) " + "; ".join(f"{s}: {_pts(x)}" for s, x in d.items())
        elif k == "G5":
            nums = f"mean alarms per run {d['mean_alarms']:.2f} (n={d['n']})"
        else:
            nums = "(A_new - A_old) " + _pts(d)
        mark = {True: "as expected", False: "**AGAINST expectation**", None: "-"}[v["as_expected"]]
        w(f"| {k} | {v['rule']} | **{v['verdict']}** | {v['expected']} ({mark}) | {nums} |")
    w(f"\n**Decision.** {decision}\n")

    w("## 2. Training budgets (steps per batch)\n")
    w("| Arm | Family | Dev stream | Steps | At the edge? | Dev score by steps (%) |")
    w("|---|---|---|---|---|---|")
    for f in ("func", "perm", "types"):
        v0 = tm0["A"][f]
        w(f"| A_old (M0) | {f} | {M.TUNE_STREAM[f]} | {v0['steps']} | {'yes' if v0.get('at_edge') else 'no'} | "
          + (", ".join(f"{k}: {100 * s:.1f}" for k, s in sorted(v0["table"].items(), key=lambda kv: int(kv[0]))) if "table" in v0 else "(smoke)") + " |")
        v = tm2["A_new"][f]
        w(f"| A_new | {f} | {v['stream']} | {v['steps']} | {'yes' if v['at_edge'] else 'no'} | "
          + ", ".join(f"{k}: {100 * s:.1f}" for k, s in sorted(v["table"].items(), key=lambda kv: int(kv[0]))) + " |")
    w("")

    w("## 3. Results per stream (final mean accuracy over regimes; mean [95% t interval] over seeds)\n")
    for name in STREAM_ORDER:
        present = [m for m in LABEL if (name, m) in by]
        if not present:
            continue
        w(f"### {name}\n")
        w("| Method | Accuracy % | Experts | Alarms / run | Extends / run | Spawns / run | Seconds | n |")
        w("|---|---|---|---|---|---|---|---|")
        for m in present:
            rows = by[(name, m)]
            ev = {k: float(np.mean([r["event_counts"].get(k, 0) for r in rows])) for k in ("alarm", "extend", "spawn")}
            w(f"| {LABEL[m]} | {M._fmt_ci([r['acc'] for r in rows])} | {M._mean(rows, 'n_experts'):.1f} | {ev['alarm']:.1f} | {ev['extend']:.1f} | "
              f"{ev['spawn']:.1f} | {M._mean(rows, 'wall_clock'):.0f} | {len(rows)} |")
        w("")
        lines = []
        for a, b in (("A_new", "A_old"), ("A_new", "B2star"), ("B4", "A_new")):
            ra, rb = {r["seed"]: r for r in by.get((name, a), [])}, {r["seed"]: r for r in by.get((name, b), [])}
            common = sorted(set(ra) & set(rb))
            if len(common) >= 2:
                d = paired_diff([100 * ra[s]["acc"] for s in common], [100 * rb[s]["acc"] for s in common])
                lines.append(f"| {SHORT[a]} - {SHORT[b]} | {d['mean_diff']:+.1f} [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}] | {d['p_t']:.3g} | {d['n']} |")
        if lines:
            w("Paired differences, percentage points:\n")
            w("| Pair | Mean difference [95% CI] | p (paired t) | n |")
            w("|---|---|---|---|")
            for ln in lines:
                w(ln)
            w("")
    if all((s, m) in by for s in PRIMARY for m in ("A_old", "A_new", "B2star", "B4")):
        w("## 4. Resources on the function streams (descriptive; no statement reads this table)\n")
        w("| Stream | Method | Memory KB | Predict kMACs | Gate share of MACs | Experts | Seconds per run |")
        w("|---|---|---|---|---|---|---|")
        for s in PRIMARY[::-1]:
            for m in ("A_old", "A_new", "B2star", "B4", "kNN_all"):
                rows = by[(s, m)]
                macs = M._mean(rows, "predict_macs")
                gate = f"{M._mean(rows, 'gate_macs') / macs:.0%}" if macs and M._mean(rows, "gate_macs") else "-"
                w(f"| {s} | {SHORT[m]} | {M._mean(rows, 'memory_bytes') / 1024:.0f} | {macs / 1000:.0f} | {gate} | {M._mean(rows, 'n_experts'):.1f} | "
                  f"{M._mean(rows, 'wall_clock'):.0f} |")
        w("")
    w("Unit tests: `tests/growmoe/test_gm_m2a_arming.py` (7) and `test_gm_m2a_verdicts.py` (11); full GrowMoE suite green before the run.")
    (out / "RESULTS_M2A.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'RESULTS_M2A.md'} and {out / 'm2a_runs.json'}")
    for k, v in result.items():
        print(f"  {k}: {v['verdict']:15s} (expected {v['expected']})")
    print(" ", decision)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tune", "run", "report"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"tune": cmd_tune, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

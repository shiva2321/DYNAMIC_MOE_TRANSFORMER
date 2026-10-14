"""M0 + M1 confirmatory run: dev tuning (seeds 100-102), test runs (seeds 0-9), verdicts, RESULTS_M0.md.

Pre-registration: research/growmoe_results/m0/PREREG.md (read-only). Every number in the output is written by this script from
per-seed records; nothing is typed by hand. Run from the repository root:

    python research/growmoe_results/m0/run_m0.py tune         [--smoke] [--workers 2] [--threads 3]
    python research/growmoe_results/m0/run_m0.py run          [--smoke] [--only S2-5,F-15] [--skip-b1] [--seeds 0-9]
    python research/growmoe_results/m0/run_m0.py equivalence  [--smoke]
    python research/growmoe_results/m0/run_m0.py report       [--smoke]

``--smoke`` shrinks every stream and grid so the whole pipeline runs in a few minutes; its outputs go to ``_smoke/`` and mean nothing.
Finished tasks are stored as files and skipped on a re-run, so an interrupted job resumes where it stopped.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_THREADS = os.environ.get("M0_THREADS", "3")                                            # set before numpy / torch are imported
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np                                                                      # noqa: E402
import psutil                                                                           # noqa: E402
import torch                                                                            # noqa: E402

from growmoe import bench                                                               # noqa: E402
from growmoe.baselines import JointOracle, OracleRouted, ReplayNet                      # noqa: E402
from growmoe.config import CommonCfg, KeyedCfg, PhyloCfg                                # noqa: E402
from growmoe.fairfight import kill_rule, matched_capacity, run_fast, run_record, tune_steps, verdicts  # noqa: E402
from growmoe.frontend import HashFrontend                                               # noqa: E402
from growmoe.memory_baselines import ClassMean, KnnMemory, StreamingLDA                 # noqa: E402
from growmoe.metrics import mean_ci, paired_diff                                        # noqa: E402
from growmoe.pools.keyed import KeyedDeltaMoE                                           # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.streams import s1_types, s2_permuted                                       # noqa: E402
from growmoe.streams_m1 import function_regimes                                         # noqa: E402

# ------------------------------------------------------------------------------------------------ pre-registered settings
A_KW = dict(q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf",     # Phase 1 frozen settings for A
            arm_monitor_after_infancy=False,                                     # M0 and its diagnostics predate the repair (default ON since 2026-10-10)
            batch_vote=False,                                                    # ... and the batch vote (default ON since 2026-10-12, RV2-A)
            gate_shortlist=0)                                                    # ... and the cascade gate (default ON since 2026-10-13, RV2-B)
C_KW = dict(c=2.0, rank=16, n_stm=32)                                                   # Phase 1 tuned setting for C
DEV_SEEDS, TEST_SEEDS = (100, 101, 102), tuple(range(10))
STEP_GRID, STEP_FACTOR, STEP_CAP = (1, 5, 20, 50), 2.5, 200
B2STAR_GRID = [(w, b) for w in (128, 512, 2048, 4096) for b in (2048, 8192, 32768)]       # extended by AMENDMENT.md #1
FAMILY = {"S2-5": "perm", "S2-30": "perm", "S1": "types", "S1-PRE": "types", "F-5": "func", "F-15": "func", "F-30": "func"}
TUNE_STREAM = {"perm": "S2-5", "types": "S1", "func": "F-15"}
TUNED = {"perm": ("A", "B2"), "types": ("A", "B2"), "func": ("A", "B2", "B4", "B1_128")}
SEGMENT_EVAL = {"S2-5", "S1", "S1-PRE", "F-5"}                                          # keep the per-segment matrix (forgetting columns)
STREAM_ORDER = ["F-30", "F-15", "S2-30", "S1-PRE", "S1", "S2-5", "F-5"]               # heaviest first, for balanced workers
METHODS_MAIN = {
    "perm": ["A", "B2", "kNN_all", "kNN_cap", "NCM", "SLDA"],
    "types": ["A", "B2", "kNN_all", "kNN_cap", "NCM", "SLDA"],
    "func": ["A", "B2", "B2star", "B4", "kNN_all", "kNN_cap"],
}
LABEL = {"A": "A  PhyloMoE (tuned budget)", "B2": "B2 replay net (tuned)", "B2star": "B2* best single net (tuned)",
         "B1_128": "B1 joint ceiling, width 128", "B1_512": "B1 joint ceiling, width 512", "B4": "B4 oracle-routed experts",
         "kNN_all": "kNN, stores everything", "kNN_cap": "kNN, same bytes as A", "NCM": "nearest class mean",
         "SLDA": "streaming LDA", "C": "C keyed delta experts"}

SHORT = {"A": "A", "B2": "B2", "B2star": "B2*", "B1_128": "B1-128", "B1_512": "B1-512", "B4": "B4", "kNN_all": "kNN-all",
         "kNN_cap": "kNN-cap", "NCM": "NCM", "SLDA": "SLDA", "C": "C"}
DIRECTION = {"A1": "B2 - A", "A2": "kNN-all - A", "B1": "A - B2*", "B2": "A - kNN-all", "B3": "B4 - A", "C2": "S1-PRE - S1"}
SMOKE_DEV, SMOKE_SEEDS, SMOKE_GRID = (100,), (0, 1), (1, 5)
SMOKE_B2STAR = [(128, 2048), (512, 2048)]


# ------------------------------------------------------------------------------------------------ streams
class HybridFrontend:
    """Numeric arrays -> the usual hash frontend. Lists of strings -> frozen all-MiniLM-L6-v2 (local cache, offline) -> a fixed
    random projection to 128 -> unit length. Same class as research/moe_diligence_2026-10/spike_frontend.py."""

    def __init__(self, d: int = 128, seed: int = 0) -> None:
        from sentence_transformers import SentenceTransformer
        self.d = d
        self.hash = HashFrontend(d=d, seed=seed)
        self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        rng = np.random.default_rng([seed, 0xE1B])
        self.P = torch.from_numpy((rng.standard_normal((384, d)) / np.sqrt(d)).astype(np.float32))
        self.cache = {}

    def __call__(self, items):
        if not isinstance(items, (np.ndarray, torch.Tensor)) and len(items) and isinstance(items[0], (str, bytes, bytearray)):
            texts = [s if isinstance(s, str) else bytes(s).decode("utf-8", "replace") for s in items]
            miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if miss:
                emb = self.st.encode(miss, batch_size=128, convert_to_tensor=True, show_progress_bar=False,
                                     normalize_embeddings=True).cpu().float()
                for t, e in zip(miss, emb):
                    self.cache[t] = e
            z = torch.stack([self.cache[t] for t in texts]) @ self.P
            return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.hash(items)


_HYBRID = None
_STREAMS: dict = {}


def build_stream(name: str, seed: int, smoke: bool):
    global _HYBRID
    if name.startswith("F-"):
        n = int(name[2:])
        return function_regimes(seed, {5: 3, 15: 4, 30: 5}[n] if smoke else n, **(dict(n_train=128, n_test=64) if smoke else {}))
    if name == "S2-5":
        kw = dict(n_regimes=3, schedule=[0, 1, 2, 0], n_train=128, n_test=50) if smoke else \
            dict(n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100)
        return s2_permuted(seed, **kw)
    if name == "S2-30":
        kw = dict(n_regimes=4, schedule=[0, 1, 2, 3, 1], n_train=128, n_test=50) if smoke else dict(n_regimes=30, n_train=400, n_test=100)
        return s2_permuted(seed, **kw)
    sizes = dict(n_train=100, n_test=50) if smoke else dict(n_train=400, n_test=150)
    if name == "S1":
        return s1_types(seed, **sizes)
    if name == "S1-PRE":
        if _HYBRID is None:
            _HYBRID = HybridFrontend(d=128, seed=0)
        return s1_types(seed, frontend=_HYBRID, **sizes)
    raise KeyError(name)


def get_stream(name: str, seed: int, smoke: bool):
    key = (name, seed, smoke)
    if key not in _STREAMS:
        if len(_STREAMS) >= 4:                                                          # bound the memory held by one worker
            _STREAMS.pop(next(iter(_STREAMS)))
        _STREAMS[key] = build_stream(name, seed, smoke)
    return _STREAMS[key]


# ------------------------------------------------------------------------------------------------ methods
def _common(st, seed, steps, **kw) -> CommonCfg:
    return CommonCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps, **kw)


def factory(method: str, st, seed: int, P: dict):
    """A zero-argument constructor of a fresh pool. ``P`` carries steps / width / buffer / capacity."""
    steps = P.get("steps", 1)
    if method == "A":
        return lambda: PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps, **A_KW))
    if method == "B2":
        return lambda: ReplayNet(_common(st, seed, steps))
    if method == "B2star":
        return lambda: ReplayNet(_common(st, seed, steps, hidden=P["width"]), replay_size=P["buffer"])
    if method in ("B1_128", "B1_512"):
        return lambda: JointOracle(_common(st, seed, steps, hidden=int(method.split("_")[1])))
    if method == "B4":
        return lambda: OracleRouted(bench.make_regime_of(st, 64), _common(st, seed, steps))
    if method == "kNN_all":
        return lambda: KnnMemory(_common(st, seed, 1))
    if method == "kNN_cap":
        return lambda: KnnMemory(_common(st, seed, 1), capacity=P["capacity"])
    if method == "NCM":
        return lambda: ClassMean(_common(st, seed, 1))
    if method == "SLDA":
        return lambda: StreamingLDA(_common(st, seed, 1))
    if method == "C":
        return lambda: KeyedDeltaMoE(KeyedCfg(seed=seed, d=st.d, n_classes=st.n_classes, **C_KW))
    raise KeyError(method)


# ------------------------------------------------------------------------------------------------ tuning (dev seeds only)
def tune_task(spec: dict) -> dict:
    family, method, extra, smoke = spec["family"], spec["method"], spec.get("extra", {}), spec["smoke"]
    seeds = SMOKE_DEV if smoke else DEV_SEEDS
    assert all(s >= 100 for s in seeds), "tuning must use dev seeds only"
    streams = {s: get_stream(TUNE_STREAM[family], s, smoke) for s in seeds}

    def score(steps: int) -> float:
        P = dict(extra, steps=steps)
        return float(np.mean([run_fast(factory(method, streams[s], s, P), streams[s], s, eval_every="final").acc for s in seeds]))

    t0 = time.perf_counter()
    res = tune_steps(score, grid=SMOKE_GRID if smoke else STEP_GRID, factor=STEP_FACTOR, cap=STEP_CAP)
    return {"family": family, "method": method, "extra": extra, "best_steps": res.best, "at_edge": res.at_edge,
            "table": {str(k): v for k, v in res.table.items()}, "dev_seeds": list(seeds), "stream": TUNE_STREAM[family],
            "seconds": time.perf_counter() - t0, "peak_mb": _peak_mb()}


def tune_specs(smoke: bool) -> list:
    specs = [dict(family=f, method=m, smoke=smoke) for f, ms in TUNED.items() for m in ms]
    specs += [dict(family="func", method="B2star", extra=dict(width=w, buffer=b), smoke=smoke)
              for w, b in (SMOKE_B2STAR if smoke else B2STAR_GRID)]
    cost = {"B2star": 3, "B1_128": 2, "A": 1, "B4": 1, "B2": 0}
    return sorted(specs, key=lambda s: (-(s["family"] == "func"), -cost[s["method"]], -s.get("extra", {}).get("width", 0)))


def assemble_tuning(results: list) -> dict:
    tuned: dict = {}
    for r in results:
        if r["method"] != "B2star":
            tuned.setdefault(r["method"], {})[r["family"]] = {"steps": r["best_steps"], "at_edge": r["at_edge"],
                                                              "table": r["table"], "stream": r["stream"], "dev_seeds": r["dev_seeds"]}
    grid = [r for r in results if r["method"] == "B2star"]
    if grid:                                                                            # best score; ties go to the smaller network
        def key(r):
            return (-r["table"][str(r["best_steps"])], r["extra"]["width"], r["extra"]["buffer"], r["best_steps"])
        best = min(grid, key=key)
        tuned["B2star"] = {"func": {"steps": best["best_steps"], "width": best["extra"]["width"], "buffer": best["extra"]["buffer"],
                                    "at_edge": best["at_edge"], "stream": best["stream"], "dev_seeds": best["dev_seeds"],
                                    "grid": [{"width": r["extra"]["width"], "buffer": r["extra"]["buffer"], "steps": r["best_steps"],
                                              "score": r["table"][str(r["best_steps"])], "at_edge": r["at_edge"],
                                              "table": r["table"]} for r in sorted(grid, key=lambda r: (r["extra"]["width"], r["extra"]["buffer"]))]}}
    if "B1_128" in tuned:
        tuned["B1_512"] = {"func": dict(tuned["B1_128"]["func"], note="steps tuned at width 128 and reused (PREREG section 8)")}
    return tuned


# ------------------------------------------------------------------------------------------------ test runs
def params_for(method: str, family: str, tuning: dict) -> dict:
    if method in ("A", "B2", "B4", "B1_128", "B1_512", "B2star"):
        return {k: v for k, v in tuning[method][family].items() if k in ("steps", "width", "buffer")}
    return {}


def run_task(spec: dict) -> dict:
    name, seed, group, smoke = spec["stream"], spec["seed"], spec["group"], spec["smoke"]
    tuning = json.loads(Path(spec["tuning_path"]).read_text(encoding="utf-8"))
    family, st = FAMILY[name], get_stream(name, seed, spec["smoke"])
    ev = "segments" if name in SEGMENT_EVAL else "final"
    methods = ["B1_128", "B1_512"] if group == "b1" else list(METHODS_MAIN[family])
    if group == "main" and name == "S2-5" and seed < 5:                                  # Pool C: parameter count only, seeds 0-4
        methods.append("C")
    t0, recs, res_a = time.perf_counter(), [], None
    for m in methods:
        P = params_for(m, family, tuning)
        if m == "kNN_cap":
            P = {"capacity": matched_capacity(res_a.report, st.d)}
        res = run_fast(factory(m, st, seed, P), st, seed, eval_every=ev)
        if m == "A":
            res_a = res
        recs.append(run_record(name, m, seed, res, **P))
    return {"spec": {k: spec[k] for k in ("stream", "seed", "group", "smoke")}, "records": recs,
            "seconds": time.perf_counter() - t0, "peak_mb": _peak_mb()}


def run_specs(streams, seeds, skip_b1: bool, smoke: bool, tuning_path: Path) -> list:
    specs = []
    for name in streams:
        for seed in seeds:
            specs.append(dict(stream=name, seed=seed, group="main", smoke=smoke, tuning_path=str(tuning_path)))
    if not skip_b1:
        for name in streams:
            if FAMILY[name] == "func":
                for seed in seeds:
                    specs.append(dict(stream=name, seed=seed, group="b1", smoke=smoke, tuning_path=str(tuning_path)))
    order = {n: i for i, n in enumerate(STREAM_ORDER)}
    return sorted(specs, key=lambda s: (s["group"] == "b1", order[s["stream"]], s["seed"]))


# ------------------------------------------------------------------------------------------------ infrastructure
def _peak_mb() -> float:
    info = psutil.Process(os.getpid()).memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20                                  # Windows reports the peak working set


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)
    import warnings
    warnings.filterwarnings("ignore")


def _parallel(fn, specs: list, path_of, workers: int, threads: int, label: str, force: bool = False, tag_of=str) -> list:
    """Run ``fn`` over ``specs`` in worker processes; each result is stored at ``path_of(spec)`` and finished ones are skipped."""
    done, todo = [], []
    for s in specs:
        p = path_of(s)
        if p.exists() and not force:
            try:
                done.append(json.loads(p.read_text(encoding="utf-8")))
            except ValueError:                                                          # a result cut short by a power loss does not count as finished: run it again
                print(f"[{label}] {p.name} is unreadable (interrupted write?); it will be run again", flush=True)
                todo.append(s)
        else:
            todo.append(s)
    print(f"[{label}] {len(done)} finished earlier, {len(todo)} to run, {workers} worker(s) x {threads} thread(s)", flush=True)
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(threads,)) as pool:
            futures = {pool.submit(fn, s): s for s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                p = path_of(futures[fut])
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + ".tmp")                                      # write, flush to disk, then rename: a power cut never leaves a half-written result behind
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, indent=1))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, p)
                done.append(r)
                print(f"[{label}] {i}/{len(todo)} {tag_of(futures[fut])} {r['seconds']:.0f}s, peak {r['peak_mb']:.0f} MB "
                      f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return done


def out_dir(args) -> Path:
    d = HERE / "_smoke" if args.smoke else HERE
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_seeds(text: str) -> tuple:
    if "-" in text:
        lo, hi = text.split("-")
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(t) for t in text.split(","))


def cmd_tune(args) -> None:
    out = out_dir(args)
    specs = tune_specs(args.smoke)

    def path_of(s):
        return out / "tune" / f"{s['family']}__{s['method']}__{s.get('extra', {}).get('width', 0)}_{s.get('extra', {}).get('buffer', 0)}.json"

    results = _parallel(tune_task, specs, path_of, args.workers, args.threads, "tune", args.force,
                        tag_of=lambda s: f"{s['family']}/{s['method']} {s.get('extra', '')}")
    tuning = assemble_tuning(results)
    (out / "tuning.json").write_text(json.dumps(tuning, indent=1), encoding="utf-8")
    for m, fams in tuning.items():
        for f, v in fams.items():
            print(f"  {m:8s} {f:6s} steps={v['steps']:>3d}" + (f" width={v['width']} buffer={v['buffer']}" if "width" in v else "")
                  + ("   ** at the edge of the search **" if v.get("at_edge") else ""))


def freeze_check(tuning_path: Path) -> str:
    """The test runs must all use the same frozen budgets: remember the hash of tuning.json and refuse to continue if it changes."""
    digest = hashlib.sha256(tuning_path.read_bytes()).hexdigest()
    mark = tuning_path.with_suffix(".sha256")
    if mark.exists() and mark.read_text().strip() != digest:
        sys.exit("tuning.json changed after the freeze; delete tuning.sha256 only if the change is recorded in AMENDMENT.md")
    mark.write_text(digest + "\n")
    return digest


def cmd_run(args) -> None:
    out = out_dir(args)
    tuning_path = out / "tuning.json"
    if not tuning_path.exists():
        sys.exit("tuning.json is missing: run the tune step first (the test runs must use frozen budgets)")
    print(f"tuning.json frozen, sha256 {freeze_check(tuning_path)[:16]}", flush=True)
    streams = args.only.split(",") if args.only else STREAM_ORDER
    seeds = SMOKE_SEEDS if args.smoke else parse_seeds(args.seeds)
    if not args.smoke:
        assert all(0 <= s <= 9 for s in seeds), "test seeds are 0-9"
    specs = run_specs(streams, seeds, args.skip_b1, args.smoke, tuning_path)

    def path_of(s):
        return out / "runs" / f"{s['stream']}__{s['seed']}__{s['group']}.json"

    _parallel(run_task, specs, path_of, args.workers, args.threads, "run", args.force,
              tag_of=lambda s: f"{s['stream']} seed {s['seed']} {s['group']}")


# ------------------------------------------------------------------------------------------------ equivalence at scale
def cmd_equivalence(args) -> None:
    """The unit test compares run_fast with bench.run_stream on a tiny stream; this repeats it on dev-seed streams of real size."""
    out = out_dir(args)
    torch.set_num_threads(args.threads)
    tuning = json.loads((out / "tuning.json").read_text(encoding="utf-8")) if (out / "tuning.json").exists() else {}
    rows, seed = [], (SMOKE_DEV if args.smoke else DEV_SEEDS)[0]
    for name in ("S2-5", "F-15"):
        st = get_stream(name, seed, args.smoke)
        fam = FAMILY[name]
        for m in ("A", "B2", "kNN_all"):
            P = params_for(m, fam, tuning) if tuning and m in tuning else ({"steps": 20} if m != "kNN_all" else {})
            make = factory(m, st, seed, P)
            old = bench.run_stream(make, st, seed=seed)
            new = run_fast(make, st, seed=seed, eval_every="segments")
            fin = run_fast(make, st, seed=seed, eval_every="final")
            rows.append({"stream": name, "method": m, "steps": P.get("steps"), "max_abs_diff_matrix": float(np.abs(old.R - new.R).max()),
                         "final_row_equal": bool(np.array_equal(fin.R[-1], old.R[-1])), "events_equal": new.events == list(old.events),
                         "seconds_run_stream": old.wall_clock, "seconds_fast_all_rows": new.wall_clock, "seconds_fast_final_only": fin.wall_clock})
            print(rows[-1], flush=True)
    (out / "equivalence.json").write_text(json.dumps({"seed": seed, "smoke": bool(args.smoke), "rows": rows}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ report
def _fmt_ci(values, scale=100.0, digits=1) -> str:
    if len(values) == 0:
        return "-"
    m, lo, hi = mean_ci(values)
    if len(values) == 1 or not math.isfinite(lo):
        return f"{m * scale:.{digits}f}"
    return f"{m * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def _mean(rows, key) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def _git(*cmd) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                                                    # noqa: BLE001
        return "unavailable"


def cmd_report(args) -> None:
    out = out_dir(args)
    tuning = json.loads((out / "tuning.json").read_text(encoding="utf-8"))
    tasks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))]
    runs = [r for t in tasks for r in t["records"]]
    result = verdicts(runs)
    combined = {"runs": runs, "tuning": tuning, "verdicts": result, "kill_rule": kill_rule(result)}
    (out / "m0_runs.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")

    by = {}
    for r in runs:
        by.setdefault((r["stream"], r["method"]), []).append(r)
    L = []
    w = L.append
    seeds = sorted({r["seed"] for r in runs})
    w("# M0 + M1 results (generated by run_m0.py; do not edit by hand)\n")
    if args.smoke:
        w("> **SMOKE RUN. Tiny streams and grids. These numbers mean nothing.**\n")
    wall = sum(t["seconds"] for t in tasks)
    w(f"* Pre-registration: `PREREG.md` (sha256 {_sha(HERE / 'PREREG.md')}); this script `{_sha(Path(__file__))}`; "
      f"`fairfight.py` {_sha(ROOT / 'growmoe/fairfight.py')}; `phylo.py` {_sha(ROOT / 'growmoe/pools/phylo.py')}.")
    w(f"* Git HEAD {_git('rev-parse', '--short', 'HEAD')}; the M0 code is not committed. python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"numpy {np.__version__}.")
    w(f"* Frozen tuning: `tuning.json` sha256 {hashlib.sha256((out / 'tuning.json').read_bytes()).hexdigest()[:16]}.")
    w(f"* Test seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds); dev seeds {list(DEV_SEEDS)}. Compute time summed over tasks: {wall / 60:.0f} min; "
      f"largest worker peak memory {max(t['peak_mb'] for t in tasks):.0f} MB.\n")

    w("## 1. Verdicts (computed by `growmoe.fairfight.verdicts` from the per-seed records)\n")
    w("| ID | Rule | Verdict | Expected | Numbers |")
    w("|---|---|---|---|---|")
    for k, v in result.items():
        d = v["detail"]
        if v["verdict"] == "NOT MEASURABLE":
            nums = "missing data"
        elif k in ("A1", "A2", "B1", "B2"):
            nums = f"({DIRECTION[k]}) " + "; ".join(f"{s}: {x['mean_diff_points']:+.1f} pts [{x['ci_lo']:+.1f}, {x['ci_hi']:+.1f}] (n={x['n']})"
                                                    for s, x in d.items())
        elif k in ("B3", "C2"):
            nums = f"({DIRECTION[k]}) {d['mean_diff_points']:+.1f} pts [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}] (n={d['n']})"
        elif k == "A3":
            nums = f"unique(C)/unique(A) = {d['ratio']:.2f}; reported/unique for C = {d['reported_over_unique']:.2f} (n={d['n']})"
        else:
            nums = f"A / kNN-all multiply-adds = {d['ratio']:.2f} (n={d['n']})"
        mark = {True: "as expected", False: "**AGAINST expectation**", None: "-"}[v["as_expected"]]
        w(f"| {k} | {v['rule']} | **{v['verdict']}** | {v['expected']} ({mark}) | {nums} |")
    w(f"\n**Kill rule.** {combined['kill_rule']}\n")

    w("## 2. Tuned training budgets (dev seeds only)\n")
    w("| Method | Family | Dev stream | Chosen steps per batch | At the edge of the search? | Dev score by steps |")
    w("|---|---|---|---|---|---|")
    for m, fams in tuning.items():
        for f, v in fams.items():
            table = v.get("table") or (v.get("grid") and [g for g in v["grid"] if g["width"] == v["width"] and g["buffer"] == v["buffer"]][0]["table"]) or {}
            extra = f" (width {v['width']}, buffer {v['buffer']})" if "width" in v else ""
            w(f"| {LABEL.get(m, m)}{extra} | {f} | {v.get('stream', '')} | {v['steps']} | {'yes' if v.get('at_edge') else 'no'} | "
              + ", ".join(f"{k}: {100 * s:.1f}" for k, s in sorted(table.items(), key=lambda kv: int(kv[0]))) + " |")
    if "B2star" in tuning:
        w("\nB2* search on `F-15` dev seeds (best steps per configuration):\n")
        w("| Width | Buffer | Best steps | Dev accuracy % | Edge |")
        w("|---|---|---|---|---|")
        for g in tuning["B2star"]["func"]["grid"]:
            w(f"| {g['width']} | {g['buffer']} | {g['steps']} | {100 * g['score']:.1f} | {'yes' if g['at_edge'] else 'no'} |")
    w("")

    w("## 3. Results per stream (final mean accuracy over regimes; mean [95% t interval] over seeds)\n")
    for name in STREAM_ORDER[::-1]:
        present = [m for m in LABEL if (name, m) in by]
        if not present:
            continue
        w(f"### {name}\n")
        w("| Method | Accuracy % | Experts | Memory KB | Predict kMACs | Seconds | n |")
        w("|---|---|---|---|---|---|---|")
        for m in present:
            rows = by[(name, m)]
            w(f"| {LABEL[m]} | {_fmt_ci([r['acc'] for r in rows])} | {_mean(rows, 'n_experts'):.1f} | {_mean(rows, 'memory_bytes') / 1024:.0f} | "
              f"{_mean(rows, 'predict_macs') / 1000:.0f} | {_mean(rows, 'wall_clock'):.0f} | {len(rows)} |")
        w("")
        pairs = [("A", "B2"), ("A", "B2star"), ("A", "kNN_all"), ("A", "kNN_cap"), ("B4", "A")]
        lines = []
        for a, b in pairs:
            ra, rb = {r["seed"]: r for r in by.get((name, a), [])}, {r["seed"]: r for r in by.get((name, b), [])}
            common = sorted(set(ra) & set(rb))
            if len(common) >= 2:
                d = paired_diff([100 * ra[s]["acc"] for s in common], [100 * rb[s]["acc"] for s in common])
                lines.append(f"| {SHORT[a]} - {SHORT[b]} | {d['mean_diff']:+.1f} [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}] | "
                             f"{d['p_t']:.3g} | {d['n']} |")
        if lines:
            w("Paired differences, percentage points:\n")
            w("| Pair | Mean difference [95% CI] | p (paired t) | n |")
            w("|---|---|---|---|")
            for ln in lines:
                w(ln)
            w("")
        if name in SEGMENT_EVAL:
            fg = {m: _mean(by[(name, m)], "forgetting") for m in present if any(r.get("forgetting") is not None for r in by[(name, m)])}
            if fg:
                w("Forgetting (best accuracy since learned minus final, points): " + ", ".join(f"{SHORT[m]}: {100 * v:.1f}" for m, v in fg.items()) + "\n")
        a_runs = by.get((name, "A"), [])
        kinds = sorted({k for r in a_runs for k in r["event_counts"]})
        if kinds:                                                                       # mean over ALL runs of A (a missing kind counts as 0)
            line = ("Structural events of A per run (mean over all runs; in how many runs it fired at least once): "
                    + ", ".join(f"{k} {np.mean([r['event_counts'].get(k, 0) for r in a_runs]):.1f} "
                                f"({sum(1 for r in a_runs if r['event_counts'].get(k, 0) > 0)}/{len(a_runs)})" for k in kinds) + ".")
            idle = [k for k in ("merge", "archive", "dormant", "prune") if k not in kinds]
            w(line + (f" Never fired: {', '.join(idle)}." if idle else "") + "\n")

    if all((n, m) in by for n in ("F-5", "F-15", "F-30") for m in ("A", "B2star", "B4")):
        w("### Derived quantities (computed by this script from the tables above, not a pre-registered statement)\n")
        w("| Stream | A - B2* (pts) | B4 - B2* (pts) | Share of the oracle-routing gain A realises | Memory A / B2* | Predict MACs A / B2* | Gate share of A's MACs |")
        w("|---|---|---|---|---|---|---|")
        for n in ("F-5", "F-15", "F-30"):
            acc = {m: _mean(by[(n, m)], "acc") for m in ("A", "B2star", "B4")}
            gain, ceiling = 100 * (acc["A"] - acc["B2star"]), 100 * (acc["B4"] - acc["B2star"])
            w(f"| {n} | {gain:+.1f} | {ceiling:+.1f} | {gain / ceiling:.0%} | {_mean(by[(n, 'A')], 'memory_bytes') / _mean(by[(n, 'B2star')], 'memory_bytes'):.2f} | "
              f"{_mean(by[(n, 'A')], 'predict_macs') / _mean(by[(n, 'B2star')], 'predict_macs'):.2f} | "
              f"{_mean(by[(n, 'A')], 'gate_macs') / _mean(by[(n, 'A')], 'predict_macs'):.0%} |")
        w("")

    if ("S2-5", "C") in by:
        w("## 4. Pool C parameter count (S2-5)\n")
        c, a = by[("S2-5", "C")], by[("S2-5", "A")]
        w(f"* C: unique parameters {_mean(c, 'params'):.0f}; per-expert sum that repeats the shared base {_mean(c, 'naive_params'):.0f} "
          f"({_mean(c, 'naive_params') / _mean(c, 'params'):.2f}x); experts {_mean(c, 'n_experts'):.1f}; accuracy {_fmt_ci([r['acc'] for r in c])} %.")
        w(f"* A: unique parameters {_mean(a, 'params'):.0f}; experts {_mean(a, 'n_experts'):.1f}.\n")

    eq = out / "equivalence.json"
    w("## 5. Integrity checks\n")
    if eq.exists():
        data = json.loads(eq.read_text(encoding="utf-8"))
        w(f"`run_fast` against the original `bench.run_stream` on dev seed {data['seed']} streams of "
          f"{'smoke (tiny)' if data.get('smoke') else 'full'} size:\n")
        w("| Stream | Method | Max abs difference of the accuracy matrix | Final row equal | Event log equal | run_stream s | fast, all rows s | fast, final only s |")
        w("|---|---|---|---|---|---|---|---|")
        for r in data["rows"]:
            w(f"| {r['stream']} | {r['method']} | {r['max_abs_diff_matrix']:.3g} | {r['final_row_equal']} | {r['events_equal']} | "
              f"{r['seconds_run_stream']:.1f} | {r['seconds_fast_all_rows']:.1f} | {r['seconds_fast_final_only']:.1f} |")
    else:
        w("`equivalence.json` not found: run the `equivalence` step.")
    w("\nUnit tests: `tests/growmoe/test_gm_m0_*.py` and `test_gm_m1_function_regimes.py` (vectorised predict bit-identical to the original loop for every routing "
      "rule; fast runner identical to `bench.run_stream` on a tiny stream for six pools; verdict rules; accounting).")

    sp, d1, ab = HERE / "speed.json", HERE / "diagnostics_D1.json", HERE / "ablation_dev.json"
    if not args.smoke and (sp.exists() or d1.exists() or ab.exists()):
        w("\n## 6. Exploratory extras (run after the verdicts were known; NOT pre-registered; no verdict reads them)\n")
    if not args.smoke and sp.exists():
        data = json.loads(sp.read_text(encoding="utf-8"))
        w(f"### 6.1 Speed: one run of A, original code path against the new path (dev seed {data['seed']}, tuned steps)\n")
        w("| Stream | Steps | Original path s | New path s | Speed-up | Same final accuracy |")
        w("|---|---|---|---|---|---|")
        for r in data["rows"]:
            w(f"| {r['stream']} | {r['steps']} | {r['seconds_original']:.1f} | {r['seconds_new']:.1f} | {r['speedup']:.1f}x | {r['final_accuracy_equal']} |")
        w("\nOriginal path = `bench.run_stream` with the per-sample `PhyloMoE._predict_reference`; new path = `run_fast(eval_every=\"final\")` with the "
          "vectorised predict. Pools whose cost is mostly evaluation (kNN, class means) gain much more.\n")
    if not args.smoke and d1.exists():
        rows = json.loads(d1.read_text(encoding="utf-8"))["rows"]
        w("### 6.2 Where A loses to oracle routing (fresh seeds 200-201; original runner so that routing statistics exist)\n")
        w("| Stream | Seed | Method | Accuracy % | Test samples sent to their regime's expert % | Experts / regimes | Spawn precision | Spawn recall | "
          "Regimes within 3 pts of the oracle's accuracy | Mean gap on the other regimes (pts) |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        ref = {(r["stream"], r["seed"]): r for r in rows if r["method"] == "B4"}
        for r in rows:
            if r["method"] == "A":
                o = ref[(r["stream"], r["seed"])]
                ra = np.array([r["final_regime_acc"][str(i)] for i in range(r["n_regimes"])])
                rb = np.array([o["final_regime_acc"][str(i)] for i in range(o["n_regimes"])])
                near = (rb - ra) <= 0.03
                gap = f"{near.mean():.0%}", f"{100 * (rb - ra)[~near].mean():.1f}" if (~near).any() else "-"
            else:
                gap = ("-", "-")
            w(f"| {r['stream']} | {r['seed']} | {LABEL[r['method']]} | {100 * r['accuracy']:.1f} | {100 * r['routing_acc']:.1f} | "
              f"{r['n_experts']} / {r['n_regimes']} | {r['spawn_precision']:.2f} | {r['spawn_recall']:.2f} | {gap[0]} | {gap[1]} |")
        w("")
    if not args.smoke and ab.exists():
        data = json.loads(ab.read_text(encoding="utf-8"))
        w(f"### 6.3 Dev-seed ablation of A's expert-training recipe ({data['stream']}, dev seeds {data['dev_seeds']}; hypothesis-generating only)\n")
        w("| Variant of A (tuned 50 steps unless stated) | Dev accuracy % | Experts |")
        w("|---|---|---|")
        for v in dict.fromkeys(r["variant"] for r in data["rows"]):
            rr = [r for r in data["rows"] if r["variant"] == v]
            w(f"| {v.strip()} | {100 * np.mean([r['acc'] for r in rr]):.1f} | {np.mean([r['n_experts'] for r in rr]):.1f} |")
        w("")
    (out / "RESULTS_M0.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'RESULTS_M0.md'} and {out / 'm0_runs.json'}")
    for k, v in result.items():
        print(f"  {k}: {v['verdict']:15s} (expected {v['expected']})")
    print(" ", combined["kill_rule"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tune", "run", "equivalence", "report"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=int(_THREADS))
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--skip-b1", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"tune": cmd_tune, "run": cmd_run, "equivalence": cmd_equivalence, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

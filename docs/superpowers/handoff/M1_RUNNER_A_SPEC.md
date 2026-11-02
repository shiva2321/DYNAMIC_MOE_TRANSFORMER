# M1 runner, stage A: one bed, four existing arms, one cell at a time. Task spec for Gemini

**STOP RULE (also the first line of the kickoff message).** When `docs/superpowers/handoff/M1_RUNNER_A_RESULT.md` is written, STOP. Do not run any report seed, do not build stage B, do not commit or push.

## 1. Why and where this sits

M1 (road plan `docs/roadmap/MILESTONES.md`, protocol `docs/roadmap/M1_PREREG_DRAFT.md`, read both) measures accuracy and cost of the pool and its baselines with one meter, `costlab` (already in this repository; read `costlab/README.md` and `docs/superpowers/handoff/COSTLAB_RESULT.md` for its real API and limits). Stage A builds the harness for **bed B1 (function regimes, N = 10, 30, 100)** and the four arms that already exist. Stage B (new arms, beds B2 and B3, report and verdict scripts) comes later and is not your task.

## 2. Environment and traps

Same as `COSTLAB_SPEC.md` section 2 (Windows, Python 3.12, torch 2.6.0+cu124, ASCII console, no network, no `pip install`, memory-lean, one GPU process at a time, pin timing to logical processors 0-3 with 2 torch threads). Run the Rust-engine tests only if you need them; they need `cargo build --release` in `rust_yggdrasil`, which you must not do. Run only the tests your change can affect (`tests/m1`, plus the tests of a module you import if you suspect an interaction), never the whole suite.

## 3. Files you may touch

Create only: `m1/` (package), `tests/m1/`, `docs/superpowers/handoff/M1_RUNNER_A_RESULT.md`, and result files under `results/m1/dev/` (development seeds only). Everything else is read-only, in particular `growmoe/`, `unimoe/`, `costlab/` and `research/`. No `git add/commit/push/stash/reset`, no `--no-verify`.

## 4. What to build

1. `m1/beds.py`: `build_bed("B1", n_regimes, seed) -> Bed`. It calls the **existing** generators exactly as `research/growmoe_results/shootout/shootout_unaliased.py` does (find the calls; do not copy their logic into new code beyond the call). `Bed` exposes the stream as an ordered iterator of `(x, y, regime_id)` batches, the per-regime test sets, the feature dimension, the number of labels and the regime count. Pool seed = stream seed + 7.
2. `m1/arms.py`: a uniform adapter `Arm` with `observe(x, y)`, `predict(x)`, `memory_bytes()` and `describe()`, and adapters for **A1** the Baseline v1 pool (`unimoe` system `shipped`, or `growmoe` `PhyloCfg(route_rule="accept_then_logf")` when the bed is B1; read `docs/roadmap/BASELINE_V1.md` section 2 and use exactly what it says), **A3** `ReplayNet` at the pool's memory, **A7** `KnnMemory` at the pool's memory, **A8** `JointOracle`. Memory in bytes comes from one existing function (`growmoe/accounting.py`), never recomputed by hand.
3. `m1/cell.py`: the record of one cell (`bed`, `n_regimes`, `arm`, `seed`, `split`, `environment` from `costlab.env`, `commit`, final accuracy per regime and mean, forgetting as defined in `growmoe/metrics.py`, update cost: wall-clock per observe call, per new regime and in total, counted multiply-adds via `costlab`, peak memory; answer cost at batch 1 and 32 via `costlab`: latency stats, items per second, counted multiply-adds, active bytes; memory in bytes against the number of regimes seen so far). Written atomically (temporary file, then `os.replace`). A record carries its schema version.
4. `m1/run.py`: `python -m m1.run --bed B1 --n 10 --arm A1 --seed 800 --split dev --out results/m1/dev --cpus 0-3 --threads 2` runs one cell and writes one JSON. It refuses `--split report`.
5. `m1/queue.py`: `python -m m1.queue --split dev --beds B1 --ns 10 --arms A1,A3,A7,A8 --seeds 800` lists the cells, skips those whose record is complete and valid, runs the rest one after the other, and can be relaunched after a power cut with the same command (idempotent). Cells are short; an incomplete cell is simply rerun.
6. Tests in `tests/m1/`: tiny streams, under 60 s in all: adapters obey the interface; the record schema; atomic write (a crash leaves no half file); idempotent queue; same seed gives the same accuracy; `--split report` is refused; pool seed offset is +7.

## 5. Harness check (the only run you do)

Run the four arms on B1 with N = 10 on seed **800** with the real settings of the shootout, and the pool and `ReplayNet` also on the old development seeds **100-104** as a **harness check only** (these seeds may not appear in any M1 result). Compare against the unaliased shootout results in `research/growmoe_results/shootout/` (find the results file): accuracy of A1 and A3 within 2 points of the old means, memory bytes equal to the old accounting. Report the numbers; if they differ by more, find out why and say so; never adjust a tolerance or a setting to make them agree.

## 6. Forbidden

Typed or hard-coded results; copying numbers from old result files into new records; modifying existing modules; mocks; swallowing exceptions; touching seeds 810-829 or 400-429 or 600-604; a second GPU process; leftover debug prints. Clarity beats cleverness; type hints; a docstring on every public function that says what each number means; under about 600 lines of code.

## 7. Deliverable: `docs/superpowers/handoff/M1_RUNNER_A_RESULT.md`

(1) files created, (2) commands to run the tests, one cell and the queue, (3) the harness-check table (new against old numbers, both copied from generated files, with their paths), (4) deviations from this spec and why, (5) anything in the existing code that looked wrong or surprising (report, do not fix), (6) the line `STOP`.

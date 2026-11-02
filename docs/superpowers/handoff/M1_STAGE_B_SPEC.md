# M1 runner, stage B: all arms, all beds, tuning, report and verdict scripts. Task spec for Gemini (three jobs)

**STOP RULES (also the first line of every kickoff message).** Each job ends when ITS result file is written (B-1: `docs/superpowers/handoff/M1_STAGE_B1_RESULT.md`, B-2: `M1_STAGE_B2_RESULT.md`, B-3: `M1_STAGE_B3_RESULT.md`). Then STOP: no next job, no `git add/commit/push/stash/reset`, no report-split run, no long run (section 8), no edit outside the files of your job (section 3). A job starts only on the user's explicit go; a go covers ONE job.

## 0. Order of work

B-1 (shared machinery, new arms, tuning driver, bed B1) -> Claude reviews, then runs the long tuning and dev cells for B1 in the background -> B-3 (report and verdict scripts, new files only, can run while those long runs go on) -> B-2 (beds B2 and B3, extra pool candidates) -> Claude reviews and runs B2 and B3 -> Claude dry-runs the report on dev cells, then hash-freezes the preregistration and `m1/config_m1.py`. Only after that may report seeds run, and only Claude unlocks them. The jobs are separate so that each one can be reviewed against this spec line by line.

## 1. Why and where this sits

M1 (`docs/roadmap/M1_PREREG_DRAFT.md`, approved as written, D-77; read it all, especially sections 3, 6 and 7) measures accuracy and cost of Baseline v1 and its baselines on three beds with one meter (`costlab`). Stage A (reviewed, accepted, D-83; read `docs/superpowers/handoff/M1_RUNNER_A_RESULT.md` section 7 and lab notebook entry E-068) built the package `m1/` for bed B1 with arms A1, A3, A7, A8: `beds.py`, `arms.py`, `cell.py` (record schema 1.1.0, `record_matches_config`, `read_pool_budget`), `queue.py`, `run.py`, `harness_check.py`, `diag_switches.py`, 18 tests. **Do not break it:** after every change `python -m pytest tests/m1 -q` passes and `python -m m1.harness_check --seeds 100` (3 threads, defaults) still reproduces the old shootout exactly (pool 43.325 %, replay 36.550 %, bytes equal).

What stage A taught, which this spec builds in:
1. **Operating point.** Baseline v1 as built (class defaults, q 0.05, transfer rule) makes 61-67 experts for 10 regimes on B1 (E-068, C-195). The preregistration counts steps and thresholds as part of every arm's operating point, so every arm is TUNED on development seeds 800-804 by one declared rule (section 4.3) and the operating point is stored in every record. A1 is also reported as built.
2. **Baselines are sized to the pool they are compared with**, not only to A1 (section 4.1, reference pools R1 = A1 and R2 = A2).
3. **Threads.** M1 runs use 2 threads on logical processors 0 and 2 (`costlab.env.fixed_threads`), the pinning is stored (D-84). The replay arm's accuracy moves by about a point with the thread count (C-194): never compare numbers from different pinnings.
4. **Your reports have omitted deviations twice** (F-108). Every result file therefore ends with a COMPLIANCE MATRIX: one row per numbered requirement of your job in this spec, status DONE / PARTIAL / NOT DONE, the file and test that show it, and every place where you chose differently from the spec or simplified, with the reason. "No deviations" is accepted only with a full matrix. I rerun everything.

## 2. Environment and traps

Windows, Python 3.12, torch 2.6.0+cu124, ASCII console, **no network** (set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for B2 and B3; the MiniLM cache and the embedding cache `results/unimoe/emb` are local; beds B2 and B3 were checked to build offline for seed 800: B2 in about 60 s, B3 in about 15 s), no `pip install`, memory-lean (one process, small tensors; the machine has hit 91 % RAM before), M1 runs on the CPU (the GPU is not used unless a section says so). Run only the tests your change can affect: `python -m pytest tests/m1 -q` (and a test file of a module you import if you suspect an interaction), never the whole suite; the Rust-engine tests are not needed. A commit gate (not yours to run) flags numeric literals assigned to names that look like results (`acc`, `loss`, `f1`...): name bars and grids `SPEC_BAR`, `GRID_*`, `THRESHOLD_*`, not after metrics. No bare `except ... pass`; an unreadable file is reported.

## 3. Files you may touch

Create or edit only: `m1/` (package), `tests/m1/`, `results/m1/` (smoke outputs under `results/m1/smoke/` only), your job's result file. Everything else is read-only, in particular `growmoe/`, `unimoe/`, `costlab/`, `research/`, `fsmoe/`. Importing from them is expected; patching their classes at run time is allowed only through instance-level wrappers inside `m1/` (section 5, item 6), never by editing the files. No clones or copies of the repository (D-82). Seeds: development 800-804 only; never 810-829, 400-429, 600-604, 100-104 (except the harness check).

## 4. Shared design (applies to all three jobs)

### 4.1 Arms (ids appear in filenames and records)

| Id | What | Tuned over (section 4.3) | Memory budget |
|---|---|---|---|
| A1 | Baseline v1 as built: `PhyloCfg(route_rule="accept_then_logf")` class defaults, q 0.05 | steps {1, 5, 20, 50} | its own, final memory = reference R1 |
| A1q | the same pool, q also tuned | steps {1, 5, 20, 50} x q {0.02, 0.05} | its own |
| A2 | best flags: the best candidate of the list in 4.4 by the same rule, at q in {0.02, 0.05} each | candidate x q, at the steps A1q chose | its own, final memory = reference R2 |
| A3, A3-2 | `ReplayNet` sized to R1 / R2 memory (buffer rule of stage A) | width {256, 512, 1024, 2048, 4096} x steps {5, 20} | R1 / R2 |
| A4, A4-2 | `ReplayNet` at the same memory as A3 / A3-2 whose whole-stream training counted multiply-adds do not exceed the pool's (A1 / A2) own (section 5, item 3) | width, steps set by the budget rule | R1 / R2 |
| A5a | `StreamingLDA` (exists) | shrinkage {1e-3, 1e-2, 1e-1} | intrinsic, reported |
| A5b | Mahalanobis prototype classifier, FeCAM style (new, item 4) | shrinkage {0.01, 0.1, 0.5} | intrinsic, reported |
| A6 | random-feature ridge head (new, item 5) | features D {1024, 4096} x ridge {1e-3, 1e-2, 1e-1, 1} | intrinsic, reported |
| A7, A7-2 | `KnnMemory` with capacity from R1 / R2 memory | k {1, 5, 15} | R1 / R2 |
| A7all | `KnnMemory` storing everything | k {1, 5, 15} | all stored, reported |
| A8 | `JointOracle`, the offline ceiling (never a method) | width {1024, 4096} x steps {5, 20} | reported |

R1 / R2 memory is the final memory of A1 / A2 on the same bed and seed, read only from a current record (`read_pool_budget` logic, generalised). Pool seed = stream seed + 7 everywhere (C-75).

### 4.2 Beds

| Bed | Source | Notes |
|---|---|---|
| B1 | stage A (`function_regimes`), N = 10, 30, 100 | tuning is done at N = 30 and applied to all N (section 4.3) |
| B2 | `growmoe.streams_real.banking_incremental(seed, frontend=HybridFrontend(d=128, seed=0))` with its defaults (6 regimes, 9 segments, 3,000 training samples, 150 test samples per regime); `HybridFrontend` is in `research/growmoe_results/m0/run_m0.py` | one frontend instance per process; same `Stream` class as B1 |
| B3 | `unimoe.experiment.streams(seed)` gives `UStream` for schedules `seq`, `mix`, `drift` (22 regimes, 125 classes, 28 / 5 / 10 phases, 10,000-12,700 training samples) | beds named `B3seq`, `B3mix`, `B3drift`; phases are the evaluation boundaries; tuning on `seq` only, applied to all three schedules (as `unimoe.experiment` does) |

Common bed interface (job B-1 defines it for B1 and B2, B-2 adds B3): `d`, `n_classes`, `n_regimes`, `seed`, `pool_seed`, `total_samples`, `test_sets` (regime -> (x, y)), `regime_info`, and `walk(batch_size=64)` yielding `(x, y, regime_id, first_visit, phase_index)` with the order and batch composition of the existing generators (`Stream._walk`, `UStream.batches`). A row of the accuracy matrix is computed after each phase (for B1 and B2 a phase is a segment). Protocol is train-only (observe), evaluate at phase ends; there is no test-then-train.

### 4.3 Tuning protocol (declared now, before any number exists)

- Tuning seeds: 800-804, all five, for every arm; trials use the light profile (item 2 of section 5): accuracy matrix, memory, number of experts, update wall-clock; no answer-latency passes.
- Selection: highest mean over the five seeds of final accuracy (mean over regimes of the last row). Ties (difference under 0.001) go to lower final memory, then lower mean update wall-clock. Same rule for every arm.
- Grids as in 4.1. If the best value is the last element of a numeric grid (steps, width, D), extend once by a factor (steps x 2.5 up to 200, width and D x 2 up to 8192) and repeat while the new end wins; store `at_edge` true when the cap itself was the limit.
- One tuning file per bed: `results/m1/tuning/<bed>.json` with, per arm, the chosen configuration, every trial (config, per-seed accuracy, mean, memory), `at_edge`, the selection rule text and the code stamp; written atomically; a trial file per (arm, config hash, seed) under `results/m1/tuning/trials/<bed>/` so that a power cut loses at most one trial and a rerun skips finished trials. Records of cells carry `tuning_ref` = sha256 of the chosen entry used.
- A1 and A1q are tuned first; every other pool candidate then runs at the steps A1q chose. A4 reads the pool's counted training multiply-adds of the same seed from a cached count-only pass (`results/m1/tuning/macs/`). A3-2, A4-2, A7-2 are tuned after A2 is chosen.
- B1 is tuned at N = 30; B2 on its own; B3 on `seq`.

### 4.4 Frozen configuration file `m1/config_m1.py`

One module holds everything the freeze must pin: the seeds (dev 800-804, report 810-819), the grids of 4.1, the selection rule constants, the A2 candidate list, the held-default pairs, the thresholds of preregistration 6.1 and 6.2 (copied exactly from `M1_PREREG_DRAFT.md`), the batch size and the pinning. No other module defines such a number. Candidate list for A2 (each candidate is a dict of `PhyloCfg` overrides on the base operating point; each is run at q in {0.02, 0.05}): `shipped` {}; `shipped+twin` {sig_dedupe True, infant_relax 1.0} (added by the reviewer after E-070, which showed that this repair of the founding-cluster flaw removes the chain reaction of duplicate experts that Baseline v1 shows at q 0.05; D-87); `shipped+gap` {growth_rule "gap"}; `shipped+skip` {skip_known 0.1}; `shipped+chance` {transfer_chance 1.0}; `U-real` = `unimoe.systems.VARIANT_SWITCHES["U-real"]`; `U-deep` = `VARIANT_SWITCHES["U-deep"]`; `U-real+skip`; `U-real+chance` {transfer_chance 1.0}; `U-real+chance-weak` {transfer_chance 1.0, transfer_chance_weak_only True}; on B3 also the `UniPool` systems `UG1`, `ULDA`, `UPLUS` of `unimoe/systems2.py` (`uni_cfg(name, n_classes, seed, recipe, blocks)`, `UniPool`). The stage-4 configuration (q 0.02 with the set gate on) is covered by `U-real` at q 0.02. Held-default pairs (reference, candidate), both at the base operating point (q and steps of A1q): LDA head: `U-real` vs `ULDA` (B3 only); growth: `shipped` vs `shipped+gap`; skip: `shipped` vs `shipped+skip`; chance: `shipped` vs `shipped+chance`.

### 4.5 Record schema 1.2.0 (additions to 1.1.0)

`profile` ("full" or "light"; validators accept the fields of the profile), `bed_info` (d, classes, regimes, phases, total samples, regime names), `tuning_ref`, `reference_pool` (for A3/A4/A7 variants: arm id and its bytes), `pool_time_split` (pool arms only: seconds inside `observe`, inside `train_expert`, inside router refits), `uncounted_note` (true for arms whose main work is outside torch matrix products, namely A5a, A5b, A6, the pool's router refit and numpy parts; the report then shows wall-clock next to every counted-MACs figure). Everything else as in 1.1.0; old records of schema 1.1.0 are rerun by the queue.

### 4.6 The report split stays locked

`--split report` is refused in code in all three jobs (as in stage A). Claude later adds the unlock, which requires a freeze file with matching hashes of `M1_PREREG_DRAFT.md` and `m1/config_m1.py`.

## 5. Job B-1: machinery, new arms, tuning driver, bed B1

1. **Generalise** `beds.py` to the interface of 4.2 for B1 and B2 (`build_bed("B2", n_regimes=6, seed)`; the n argument is fixed to 6 and checked) without changing any B1 behaviour (every test of the earlier suite stays; regression: harness check and the 18 tests).
2. **Light profile and schema 1.2.0** in `cell.py` as in 4.5. `execute_cell(bed, arm, pool_memory_bytes, split, pinning, profile, tuning)`; `build_arm(arm_id, bed, pool_memory_bytes=None, tuning=None, **kw)` reads the chosen configuration from the tuning entry; without a tuning entry A1 keeps the stage-A default (5 steps) so stage-A cells stay reproducible.
3. **A4 budget rule.** For width w: count the whole-stream training multiply-adds of a `ReplayNet` at 1 inner step with `costlab.count.count_macs` (cached); steps s(w) = floor(budget / count(w)); verify with one counted run at s(w) and decrease s until the count is at most the budget; skip widths with s < 1. The budget is the counted whole-stream training multiply-adds of the pool arm (A1 for A4, A2 for A4-2) on the same bed and seed. The chosen width is tuned like any other grid; s(w) is recomputed per seed.
4. **A5b** `MahalanobisProto` in `m1/baselines_extra.py`, same interface and return types as `StreamingLDA` (read `growmoe/memory_baselines.py` and `growmoe/types.py`): per-class mean and per-class covariance from incremental sufficient statistics (count, sum, scatter), shrunk towards the pooled covariance by the grid value, class score = negative squared Mahalanobis distance, probabilities by softmax with a fixed scale. Memory comes from `growmoe.accounting.pool_report` only: expose `stored_vectors` as the number of d-vectors it holds (1 + d per class seen).
5. **A6** `RandomFeatureRidge` in the same file: fixed Gaussian projection from `d` to `D` (regenerated from the seed, not counted as stored), ReLU features, incremental `G = phi^T phi` and `B = phi^T Y` (one-hot targets), weights by Cholesky solve of `(G + ridge I)` computed lazily at the first `predict` after an `observe`, probabilities by softmax of the logits times a fixed scale. Memory: `D*D + D*classes` floats exposed so that `pool_report` counts them.
6. **Instrumentation of pool arms** (`A1PoolArm`; the pattern is `instrument()` in `fsmoe/skip_check.py`, and `train_expert` is inherited from `growmoe/pools/base.py`): instance-level wrappers around `model.observe`, `model.train_expert` and `model._refit_router` that add `perf_counter` time to counters, recorded as `pool_time_split`; measure the wrapper overhead once on a B1 N = 10 cell (with and without wrappers, same seed) and report it in the result file.
7. **Tuning driver** `python -m m1.tune --bed B1 --tune-n 30 --arms A1 A1q A2 A3 A4 A5a A5b A6 A7 A7all A8 A3-2 A4-2 A7-2 --seeds 800-804 [--resume]` implementing 4.3, idempotent, pinned (2 threads, 0 and 2), printing one line per trial. The order enforces the dependencies (A1, A1q, A2, then the rest; the `-2` arms after A2). It must be possible to run a subset of arms and seeds for smoke tests.
8. **Queue**: `m1.queue` accepts the new arm ids and `--tuning results/m1/tuning`, builds each arm from its tuning entry, takes R1 / R2 budgets from current records, keeps the stale-record guard (the configuration includes the tuning choices) and the arm order A1, A1q, A2, then the rest; `--profile full|light`.
9. **Tests** (`tests/m1/`, fast, tiny beds, whole suite under 3 minutes): every new arm obeys the interface and its `config()` is plain JSON; A4 never exceeds the budget; A5b and A6 give a hand-computable answer on a two-class toy; memory of A5b and A6 equals the `pool_report` figure; selection rule (ties, edge extension) on synthetic trial tables; trial files make a rerun skip finished trials; stale tuning invalidates cells; light records validate; schema 1.1.0 records are rejected; B2 interface on a synthetic `Stream`; report split refused.
10. **`m1/config_m1.py`** exactly as 4.4 describes (including the thresholds of 6.1 and 6.2 copied from the preregistration, although job B-3 is the one that uses them); `queue.py`, `tune.py`, `arms.py` and `cell.py` import every grid, seed list and candidate from it.
11. **Smoke runs you do execute** (each at most 5 minutes, all at 2 threads on 0 and 2, all on seed 800, output under `results/m1/smoke/`): tuning of every arm on B1 N = 10 with a one-value grid per arm; one full-profile cell of each arm id on B1 N = 10; the harness check. **You do not run** the real tuning or the dev queue: Claude runs them after review (they take hours).

## 6. Job B-2: beds B2 and B3, pool candidates for B2/B3

1. `build_bed("B2", 6, seed)` complete and tested offline; `build_bed("B3seq" | "B3mix" | "B3drift", 22, seed)` wrapping `UStream` with the interface of 4.2 (`first_visit` from the phase's `first` flags, regime info from `UStream.regimes`, the label blocks kept in `bed_info`). Evaluation boundaries are the phases. Check on a synthetic `UStream` in the tests and, in the smoke run, on the real one for seed 800 (compare `total_samples` with the figures in 4.2).
2. The pool factory builds the A2 candidates of 4.4 on B1, B2 and B3 identically through `PhyloCfg` overrides, and the three `UniPool` candidates on B3 through `unimoe.systems2.uni_cfg` with the stream's label blocks; the config of each candidate is stored in full.
3. `m1.tune` and `m1.queue` accept every bed name of 4.2 (`--bed B2`, `--bed B3seq`; `--tune-n` applies to B1 only) and run the protocol of 4.3 (B3 tuned on `seq`, then `mix` and `drift` use the same choices). Time estimates are yours to report: run one trial of each arm on each bed at seed 800 and write the measured seconds in the result file so that the long runs can be scheduled.
4. Smoke runs as in B-1 item 10 on B2 and B3seq (seed 800, one-value grids, at most 5 minutes each). Regression: the stage-A harness check (write its output with `--out results/m1/smoke`, never over the committed evidence in `results/m1/harness/`) and all earlier tests, none removed.

## 7. Job B-3: report and verdict scripts (new files only; synthetic fixtures)

Files: `m1/report.py`, `m1/verdict.py`, `tests/m1/test_verdict.py`, `m1/mutate.py`. They read cells from a folder and write only under a folder given by `--out`. They must run on development cells (labelled DEV in every title and table) and on report cells (labelled REPORT), never mixing the two; the label comes from the `split` field of the records.

1. **Scoreboard** per bed (B1 N = 10, 30, 100; B2; B3seq, B3mix, B3drift): one row per arm, mean and standard deviation over seeds and the 95 % interval of the mean; columns: final accuracy, forgetting, update wall-clock (total, per new regime), update counted multiply-adds, peak RAM, answer median and p95 latency at batch 1, throughput at batch 32, answer counted multiply-adds, active bytes, memory bytes, experts; the operating point of every arm next to it; a marker for arms with `uncounted_note`. Figures: accuracy against update cost and against answer cost (Pareto fronts) per bed, and one summary figure. Figure conventions: one y-axis per chart (never two scales), thin marks, a legend whenever there are two or more series, a colour-blind-safe palette with the same colour for the same arm in every figure, titles built from the generated numbers (never typed claims), matplotlib only.
2. **Verdicts** as pure functions of the record tables, with the thresholds imported from `m1/config_m1.py`: rule 6.1 per bed (a baseline among A3, A4, A5a, A5b, A6, A7, A7all, and the `-2` variants, never A8, is within 1 point of the pool's accuracy at lower update cost and lower answer cost); computed for R1 and R2, and in two readings: primary = update wall-clock total and answer latency at batch 1, secondary = counted multiply-adds; adoption rules 6.2 for the four held defaults from the held-default pairs, with paired report-seed differences from `growmoe.metrics.paired_diff`; predictions M1-1 to M1-9, each as a function returning PASS / FAIL / NOT APPLICABLE with the numbers it used (definitions: M1-4 share = (observe time - train_expert time + router refit time) / (observe time + router refit time); M1-8 = linear fit of memory against regimes seen, R squared at least 0.95 and slope within 25 % of final bytes per expert; "the pool" in M1-1 to M1-3, M1-5, M1-6 is A2, with A1 printed beside it; "the best-tuned replay network" is A3-2).
3. **Top three cost drivers** from `pool_time_split` (update) and from `growmoe.accounting` gate and expert multiply-adds (answer); **reconciliation** table of `growmoe.accounting` counts against costlab counts per arm, with the ratio.
4. **Output**: `REPORT_M1.md` (every number from the tables, none typed), `verdicts.json`, figures; a plain-English first page with one summary figure.
5. **Tests and mutation check**: unit tests of every verdict function on hand-made tables with known answers (including boundary cases at the thresholds); `m1/mutate.py` applies at least 20 single-edit mutants (flip a comparison, change a threshold, swap R1 and R2, drop a seed) to the source of `m1/verdict.py` in memory and shows that the tests fail for each; list the survivors. No file copies of the repository.
6. **Smoke run**: build fixtures programmatically (synthetic cells with the real schema) and run the whole report on them; do not use or invent real-looking results; fixtures are labelled SYNTHETIC in the output and never written under `results/m1/dev`.

## 8. What Claude does after each job (for your information)

Reads the code against this spec, reruns tests and the smoke commands, runs the real tuning and dev cells in the background (idempotent, relaunchable after a power cut), checks the compliance matrix, and only then lets the next job start. Dev cells are full-profile for seeds 800 and 801 on every bed; tuning uses all five dev seeds.

## 9. Forbidden

Typed or hard-coded results; copying numbers from old result files into new records (the harness check may read the old files to compare, as in stage A); editing existing modules outside `m1/`; mocks that replace the thing being measured; swallowing exceptions; touching report seeds or the seeds named in section 3; a second GPU process; leftover debug prints; changing a grid, a threshold, a tolerance or a candidate to make a test or a result look better (propose it in the result file instead); clones. Clarity beats cleverness; type hints; a docstring on every public function that says what each number means. Size guide: B-1 about 900 lines of code including tests, B-2 about 400, B-3 about 700.

## 10. Deliverable per job: `M1_STAGE_B<n>_RESULT.md`

(1) files created and changed, (2) commands for the tests, the smoke runs and (B-1) the real tuning and dev-cell commands for Claude to run, (3) the smoke outputs with their file paths, (4) the COMPLIANCE MATRIX of section 1 item 4, (5) anything in the existing code that looked wrong or surprising (report, do not fix), (6) the line `STOP`.

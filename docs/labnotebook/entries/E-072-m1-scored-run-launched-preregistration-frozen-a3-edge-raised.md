# E-072 - M1 scored run launched: preregistration frozen, A3 edge raised, timed pass on seeds 810-814 started

- **ID:** E-072
- **Date:** 2026-11-03
- **Track:** W
- **Type:** experiment
- **Status:** in progress (timed pass running; the deterministic pass and the report follow; results go into E-073)
- **Evidence grade:** B for the A3 edge probe (script-generated, five development seeds); the scored run itself is graded when its report exists
- **Code / commit:** code and tuning a1464ee, 0b0b3bb, 334e41e; freeze 1177f9f (`docs/roadmap/M1_FREEZE.json`)
- **People:** the user chose option A of Q-82 ("continue then to A"); Claude built the freeze gate and the launcher, ran the A3 probe and launched the run; no Gemini job

## Question
Is the M1 pipeline ready to run on the report seeds, and what has to be fixed or fixed in writing first? A wrong answer after the freeze would cost a re-freeze or a flawed scored run.

## Why we asked it
E-071 rehearsed the whole pipeline on development seeds and left four items for the user: the size of the scored run (Q-82), the grid-edge baselines, the freeze, and the unlock. The user answered on 2026-11-03: option A (five timed seeds, five deterministic seeds, about 37 h).

## Prediction (written BEFORE the numbers, if there was one)
For the A3 probe: none stated beforehand. I expected a gain of about one point per doubling of the width and a pool lead that stays far above it. For the scored run the predictions are the nine of the frozen preregistration (section 7).

## Method (how)
1. **Edge probe.** A3 (replay network at the pool's memory) was at the edge of its width cap on B1 (8192). One more doubling (16384, 20 steps) was run on the development seeds 800-804 through the tuner's own trial runner and cache (the first run's output is `results/m1/diag/a3_edge_probe_B1.json`; `m1/diag_a3_edge.py` re-derives it from the cache and is to be run after the timed pass, because it pins logical processors 0 and 2). The selection rule (best mean accuracy, ties under 0.001 go to the cheaper) then picks 16384, so the cap for A3 only was raised to 16384 (`EDGE_EXTEND_CAP_WIDTH_BY_ARM` in `m1/config_m1.py`, a raised cap makes that entry stale in `grid_ref`), and A3 and A3-2 were tuned again on all five beds from cached trials: only A3 on B1 changed.
2. **Freeze gate** (`m1/freeze.py`, 83 tests in `tests/m1` pass): the report split opens only when `docs/roadmap/M1_FREEZE.json` names the SHA-256 (CRLF read as LF) of the preregistration, `m1/config_m1.py` and the five tuning files, every report cell carries the digest of the freeze file, seeds outside 810-819 and a report run without `--tuning` are refused. The code files are recorded in the freeze, not enforced. The freeze was written exclusively and holds the readings and the run plan.
3. **Run.** `bash m1/scored_run.sh timed` runs seeds 810-814 with the full profile, one process at a time on logical processors 0 and 2; then `bash m1/scored_run.sh det` runs seeds 815-819 with four workers of two threads on logical processors 0,2 / 4,6 / 8,10 / 1,3 (never at the same time as the timed pass); then `bash m1/scored_run.sh report`. Cells land in `results/m1/scored`, logs in `results/m1/scored_logs`. Every step is idempotent, so the same command resumes after a power loss.

## Results (what came out)
**A3 on B1 (development seeds 800-804, N = 30, 20 steps), from `results/m1/diag/a3_edge_probe_B1.json` and the tuning trials:** width 4096 33.79 %, 8192 34.58 %, 16384 35.27 % (+0.70 points over 8192; per seed 35.73, 34.90, 35.11, 35.27, 35.36 against 34.66, 34.62, 34.31, 34.82, 34.47). Memory 21.54 MB at 8192 and 29.96 MB at 16384, against 12.1 MB for the tuned pool. The update wall-clock is 215-267 s per seed at 16384. The pool (A2) scores 42.19 % on the same seeds, so its lead over A3 is 7.6 points at width 8192 and 6.9 points at 16384. A3 stays at its (raised) edge: the doubling slope (+0.8, then +0.7) says a further doubling would add roughly a point, which is not enough to change the B1 verdict but is a stated limit.
**Tuning files:** only `results/m1/tuning/B1.json` changed in its choices (A3: hidden 16384, at_edge true); the other four files changed only in the `grid_ref` of A3 and A3-2.
**Freeze hashes** are in `docs/roadmap/M1_FREEZE.json` (`frozen`); the scored run started at 07:15:03.

## Interpretation (what we understood, and on what basis)
Plain words: the strongest replay baseline on the many-regime bed gets about 0.7 points better when its network is made twice as wide, so we let it, because a baseline that is under-tuned makes the pool look better than it is. It does not change the story (the pool still leads by about seven points on B1) and it is written down as a limit because the baseline is still at its edge. The freeze is now enforced by code, not only by promise: an edit to the preregistration, the config or any tuning file after the freeze stops the run at its start and at every cell.

Measured: the probe numbers above. Assumed: that the slope of about a point per doubling continues (not measured beyond 16384).

## Caveats and what went wrong
- Wall-clock verdicts rest on the five timed seeds, accuracy and counted multiply-adds on all ten.
- The GPU repeat of the wide networks (D-89) is not built; every cell runs on the CPU. That is open as Q-83.
- A6 and A8 stay at the edge of their grids on B1 and the B3 schedules, A3 at its raised edge on B1; the report prints the `at_edge` notes.
- A3 at 16384 has 30 MB of weights against the pool's 12 MB: the "equal-memory" baseline is no longer equal on this bed, which is stated in the report next to the memory column.
- A bug in my first launcher (`python -I -m m1.freeze` cannot import the repo) was found by running the check before launching, not by the run.
- The notebook work and the git commits during the timed pass use one short burst of CPU each; the user must not use the PC heavily while the timed pass runs.
- The truthgate WARN `HARDCODED_EXPECTED_VALUE` on `m1/freeze.py:100` is a false positive (an error message for an unreadable freeze file; no test asserts it).

## Consequence
- D-92, Q-83. The scored run is under way; the next entry (E-073) holds its results and the report.
- If the machine reboots: run `bash m1/scored_run.sh timed` (resumes), then `det`, then `report`.

## Evidence
- `docs/roadmap/M1_FREEZE.json` - the frozen hashes, the readings and the run plan
- `docs/roadmap/M1_PREREG_DRAFT.md` - section 11 (fixed at the freeze)
- `results/m1/tuning/B1.json` - A3 entry (hidden 16384, at_edge true) and its trials (regenerated on purpose, so the manifest hash of the tuning files changed; the freeze names the new hashes)
- `results/m1/diag/a3_edge_probe_B1.json` - the first run of the width probe
- `m1/diag_a3_edge.py` - the script that re-derives it from the trial cache
- `results/m1/scored_logs/` - queue logs of the scored run (cells in `results/m1/scored/`)

## Reproduce
```
python -m m1.freeze check
bash m1/scored_run.sh timed
bash m1/scored_run.sh det
bash m1/scored_run.sh report
```

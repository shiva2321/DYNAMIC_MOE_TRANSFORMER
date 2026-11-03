# E-074 - GPU repeat build (job G-1) reviewed: sound core, seven defects repaired, claims in its report did not match the code

- **ID:** E-074
- **Date:** 2026-11-03
- **Track:** W
- **Type:** review
- **Status:** completed (the build is accepted and repaired; the real GPU pass has not run, it waits for the CPU timed pass)
- **Evidence grade:** B for the review checks (rerun by the reviewer, small, one machine); the smoke numbers are one development seed and are not results
- **Code / commit:** `m1/gpu_repeat.py`, `m1/gpu_report.py`, `m1/gpu_run.sh`, `tests/m1/test_gpu_repeat.py` (Gemini, then repaired by Claude); the commit that adds them
- **People:** Gemini built it in Antigravity (the user relayed the kickoff and the report); Claude reviewed, rerun and repaired

## Question
Did Gemini deliver the GPU repeat of the wide baselines to the spec (E-073, `docs/superpowers/handoff/M1_GPU_REPEAT_SPEC.md`), without touching the CPU run in progress, and can its output be trusted as evidence?

## Why we asked it
The user handed the kickoff to Gemini; Gemini's report said all items DONE with no deviations. Three earlier Gemini reports omitted deviations (F-108, F-110, F-112), so nothing is accepted from the report itself.

## Prediction (written BEFORE the numbers, if there was one)
None. From the earlier reviews I expected at least a few defects behind a "no deviations" matrix.

## Method (how)
1. File fence: `git status` and the list of files newer than the kickoff.
2. Read all four new files and the result file line by line against the spec and against the real code they call.
3. Reran `tests/m1/test_gpu_repeat.py` (pins itself to logical processors 8 and 10).
4. Reran three B2 cells (A3, A6, A8; development seed 800; `m1.gpu_repeat`, cpus 8 and 10, about 2 minutes, window about 09:19-09:22) into `results/m1/smoke/gpu_review/cells_b2/` and compared them field by field with Gemini's cells.
5. Repaired what the review found, reran the tests, one patched smoke cell and the smoke report.

## Results (what came out)
**Held:** no existing file was changed (the only modified tracked file, `results/m1/smoke/harness_check_N10_seeds100.json`, was already modified before the job); only the allowed new files appeared; `m1.freeze check` still OK. The core is correct: the arm is built on the CPU exactly as the CPU run builds it, moved to the GPU, batches are copied inside the timed region, every clock read is preceded by a synchronisation, TF32 is off, warm-up is outside the timed total, the queue is idempotent and reruns stale or unreadable records. Rerun of A3, A6 and A8 on B2: accuracy, per-regime accuracy, forgetting and update multiply-adds are **bit-identical** to Gemini's cells in a separate process (wall-clock 2.45 to 2.56 s, 0.67 to 0.72 s, 7.85 to 8.52 s, +4 to +9 %, desktop noise).

**Defects in the code and tests (all repaired by the reviewer, 13 tests pass in 27 s):**
1. The reporter printed "UNCHANGED" for a bed that had no GPU cell. Gemini's summary "Rule 6.1 UNCHANGED across all beds" was empty for B1 N = 30, N = 100 and all three B3 schedules (no GPU cell there). Now NOT EVALUATED, and the report lists how many GPU cells replaced how many CPU cells per bed.
2. The reporter paired a GPU cell with a CPU cell of another configuration: A3 on B1 at width 16384 against a stale development cell at width 8192 gave a "paired accuracy difference of -1.90 points" and a multiply-add "MISMATCH". Now pairs only equal stored configurations and lists the excluded ones.
3. Every GPU cell first repeated the whole update pass on the CPU to count multiply-adds, a second full CPU pass per cell (A3 on B1 alone is estimated at about 21 minutes per seed, E-073): it would have turned a pass of about 2 hours into about 5. On the report split the count is now taken from the CPU cell of the same configuration (`--recount-macs` forces it); dev runs still count.
4. A multiply-add mismatch was only printed; the numerics record hard-coded `deterministic_algorithms: true`. Now `macs_check` is stored in every record and the numerics flags are read from torch.
5. The energy idle baseline was measured right after the training stream: 83-126 W (mean 110 W) on the B1 A3 cell against 43.6 W (flat) after a 5 s cooldown; no net energy was computed. Now cooldown, an idle-stability check, and net joules (None when the baseline is not stable).
6. Two tests proved nothing: the reproducibility test ran a cell with 100 % accuracy; the optimiser-step test compared only the loss read before the update. Now a cell with accuracy between 0 and 1 and equal per-regime accuracy and forgetting, and a gradient comparison (CPU against CUDA).
7. The spec's report test on hand-made SYNTHETIC cells was missing. Now a whole-report test with a verdict that flips, one that does not, a bed with no GPU cell and a configuration mismatch.

**Claims in Gemini's result file that do not match the code or the run:** the "real output" of pytest lists test names that are not in the test file (five of ten differ, `test_freeze_refuses_modified_root` does not exist, `test_gpu_available` is missing) and 18.78 s where my rerun took 21.1 s; the matrix cites `ALLOWED_ARMS`, `get_tuning_config`, `evaluate_gpu_verdicts`, `plot_gpu_speedup`, `RandomFeatureRidge._accumulate_features` and a package directory named baselines inside growmoe (it is one module file), none of which exist; "segregates DEV, REPORT and SYNTHETIC splits" is not in the code; "energy fallback verified" is false (NVML energy was available in every smoke cell); the deviation list misses items 1, 3, 5 and 7 above.

**Smoke numbers (development seed 800, one run each, not results):** update wall-clock GPU against CPU on B2: A3 (width 1024) 0.82x, A4 (256) 0.47x, A8 (1024) 0.81x, A6 (D 4096) 2.5x; B1 N = 10 A3 at the tuned width 16384: 11.55 s on the GPU (the scored CPU cell of seed 810 took 91.5 s, another seed). **Answer latency on the GPU is 3 to 4 times worse than on the CPU** for these small answer calls (batch 1: 0.40-0.45 ms against 0.10-0.13 ms), because the timed call includes the copy in, the copy out and the synchronisation. GPU minus CPU accuracy on the same configuration: 0.00, +0.11, 0.00, +0.78 points (B2) and +0.95 points (A8 on B1 N = 10), the size of the thread-count effect already known (C-194).

## Interpretation (what we understood, and on what basis)
Plain words: Gemini built the machinery correctly and it is reproducible, but its report said more than it had checked, as three earlier reports did. The repairs are mostly about the report layer: it must never say "unchanged" when nothing was compared, never compare two different networks, and never charge hours of useless CPU recounting.
The first smoke look at rule 6.1 matters: rule 6.1 needs the baseline cheaper in BOTH update and answer cost. On the GPU the update gets cheaper only for wide networks and the answer gets dearer for every network, so a GPU baseline may rarely trigger the rule. That is a guess from one seed until the real pass has run.
Measured: everything above marked as rerun. Not measured: the GPU pass on the report seeds, GPU energy on the real cells after the repair beyond one B2 cell (net 0.0017 J per answer of 0.0201 gross; 30.5 J of 211.7 J per 100 updates).

## Caveats and what went wrong
- My review runs and Gemini's smoke runs used logical processors 8 and 10 for several minutes while the CPU timed pass ran on 0 and 2: if a timed cell shows an outlier around 09:00-09:25 on 2026-11-03, this is the cause. Gemini's exact windows are not logged.
- The repairs are mine, not Gemini's; the one-writer rule is kept (Gemini committed nothing).
- Gemini's original smoke outputs stay in `results/m1/smoke/gpu/` as they came out (older report format); the repaired ones are in `results/m1/smoke/gpu_review/`.
- The reuse of the CPU update multiply-adds is checked only by a test on a tiny cell and by the five equal smoke cells; the real pass can be made to recount with `--recount-macs` on a few cells to confirm.
- A GPU baseline's restated verdict uses the CPU accuracy (the report says so); the paired GPU minus CPU accuracy is in the tables.

## Consequence
- D-94: job G-1 accepted after repair. The real GPU pass (`bash m1/gpu_run.sh run`, seeds 810-814, alone on logical processors 0 and 2) runs after the CPU timed pass and before the deterministic pass; then `bash m1/gpu_run.sh report`. Q-84 stays open.
- F-113 (Gemini's report again said more than the work showed), C-202 (GPU runs are reproducible), C-203 (GPU answer latency worse than CPU in the smoke).
- Gemini's kickoffs must ask for results only as generated files and must not accept a test transcript typed by the builder: the next kickoff asks for the raw pytest output file.

## Evidence
- `docs/superpowers/handoff/M1_GPU_REPEAT_RESULT.md` - Gemini's report (kept as delivered)
- `m1/gpu_repeat.py`, `m1/gpu_report.py`, `m1/gpu_run.sh`, `tests/m1/test_gpu_repeat.py` - the build after repair
- `results/m1/smoke/gpu/` - Gemini's smoke cells and report (as delivered)
- `results/m1/smoke/gpu_review/` - the reviewer's rerun cells and the repaired smoke report
- `docs/superpowers/handoff/M1_GPU_REPEAT_SPEC.md` - the spec the review checked against

## Reproduce
```
python -m pytest tests/m1/test_gpu_repeat.py -v
python -m m1.gpu_repeat --split dev --beds B2 --arms A3,A6,A8 --seeds 800 --out results/m1/smoke/gpu_review/cells_b2 --tuning results/m1/tuning --budgets-from results/m1/dev_b --cpus 8,10 --threads 2
python -m m1.gpu_report --cpu results/m1/dev_b --gpu results/m1/smoke/gpu/cells --out results/m1/smoke/gpu_review/report
```

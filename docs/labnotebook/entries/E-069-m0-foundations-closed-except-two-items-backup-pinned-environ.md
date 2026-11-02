# E-069 - M0 foundations closed except two items: backup, pinned environment, Baseline v1, a validated cost meter and a first prior-art check

- **ID:** E-069
- **Date:** 2026-11-02
- **Track:** W
- **Type:** process
- **Status:** completed (open: a fresh install from the lock file, Q-81)
- **Evidence grade:** B for the cost-meter validation (script-generated, two passing runs, reviewer reruns); H for the rest (records of what exists)
- **Code / commit:** `costlab/` merged at 74fe792; tag `baseline-v1` at a96bffc; `requirements-lock.txt`; the pushed branch `growmoe-lab`
- **People:** Claude planned, reviewed and patched; Gemini built the cost meter (first Gemini job, D-76); the user decided the backup (D-81, D-86) and approved the plan (D-75)

## Question
Is everything the later milestones stand on in place: work that cannot be lost, an environment that can be rebuilt, a named baseline, one trusted way to measure cost, and a first honest look at what is already published?

## Why we asked it
Road plan M0 (docs/roadmap/MILESTONES.md). It follows E-067. M1 (the efficiency scoreboard) cannot start without a validated meter.

## Prediction (written BEFORE the numbers, if there was one)
The five bars of M0 (a) a restore of an archive works and hashes match, (b) the suite passes from a clean clone, (c) counted multiply-adds within 5 % of analytic on three toy models, (d) wall-clock coefficient of variation at most 5 % over 10 repeats at fixed threads, (e) GPU energy readings behave sensibly.

## Method (how)
- Safety: all work committed on one branch and pushed to `growmoe-lab` (D-73, D-81); the folder is also backed up to Google Drive (D-86).
- Environment: `docs/roadmap/make_env_lock.py` wrote `requirements-lock.txt` (58 pins); a clean clone of the pushed branch ran the whole suite (BASELINE_V1.md section 5).
- Baseline v1: `docs/roadmap/make_baseline_v1.py` read every default from the code and wrote BASELINE_V1.md; tag `baseline-v1`.
- Cost meter: `costlab/` (spec docs/superpowers/handoff/COSTLAB_SPEC.md), built by Gemini, reviewed and patched by Claude (COSTLAB_RESULT.md section 6), validated by `python -m costlab.validate` (bars V1 to V5).
- Prior art: about 16 searches and 6 page fetches (docs/roadmap/PRIOR_ART.md), abstract level.

## Results (what came out)
- (a) Partly met: a fresh clone of the pushed branch restores and runs; the evidence hashes were not re-checked there.
- (b) Met with a documented build step: 1915 passed, 27 failed, 1 skipped, 2 xfailed; all 27 failures need `yggdrasil_engine.dll`, which is not in git; after `cargo build --release --offline` the five files pass (BASELINE_V1.md section 5). A fresh install from the lock file was not tested.
- (c) Met: largest error 0.00 % over two runs (C-197).
- (d) Met after pinning one logical processor per physical core: CV 0.50-0.61 % (CPU) and 0.36-0.50 % (GPU); it failed at 11.08-18.52 % with both SMT siblings pinned (C-198).
- (e) Met: loaded 169.6-170.1 W against 60.8-67.6 W at rest, linearity within 0.83-1.25 %, counter against integrated joules within 4.21-4.56 % (C-199).
- Prior art (first pass): every ingredient of the combination is published, including exact removal by ridge downdating (ACU 2025, Quan et al. 2026); the combination of a calibrated novelty gate, closed-form heads in isolated experts, a lifecycle with removal and a compute-matched cost account was not found. Nothing may be called new before M8 repeats the check with a fuller search.

## Interpretation (what we understood, and on what basis)
Plain words: we can now count and time work with a meter that has been checked against known answers, and the machine's one trap (two threads on the two halves of one core jitter by 10 to 20 %) is understood and avoided. The meter counts matrix work only, so closed-form arms and numpy parts need their wall-clock reported next to the count. Technical: C-197 to C-199 are measured; the SMT explanation of the timing jitter is a measured effect, not a hardware-level proof.

## Caveats and what went wrong
- The first Gemini build said "Deviations: None" and had six (F-110); the patched build passed only after the reviewer changed code and the pinning.
- GPU idle readings are 58-68 W because the desktop shares the card.
- A fresh environment from the lock file is untested (Q-81).

## Consequence
M0 is closed apart from Q-81. M1 stage A (E-068) and stage B (docs/superpowers/handoff/M1_STAGE_B_SPEC.md) build on the meter. D-80 recorded the acceptance of the meter.

## Evidence
- `docs/superpowers/handoff/costlab_validation/final_run1.json`
- `docs/superpowers/handoff/costlab_validation/final_run2.json`
- `docs/superpowers/handoff/costlab_validation/patched_run_pin0123_V2_fail.json`
- `docs/superpowers/handoff/COSTLAB_RESULT.md`
- `docs/roadmap/BASELINE_V1.md`
- `docs/roadmap/PRIOR_ART.md`

## Reproduce
```
python -m pytest tests/costlab -q
python -m costlab.validate
```

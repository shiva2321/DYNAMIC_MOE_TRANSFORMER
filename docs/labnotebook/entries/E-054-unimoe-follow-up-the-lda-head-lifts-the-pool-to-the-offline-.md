# E-054 - UniMoE follow-up: the LDA head lifts the pool to the offline ceiling, growth works only where an expert lives long enough, the refusal layer is calibrated but short of its bar

- **ID:** E-054
- **Date:** 2026-10-30
- **Track:** G
- **Type:** experiment
- **Status:** completed on development seeds; Stage 2 (token-level language-model MoE) not started
- **Evidence grade:** B (script-generated; tuning seed 499, evaluation seeds 500-502, confirmation seeds 503-504, all dev seeds; every prediction declared before the cells that test it and scored; NOT a pre-registration; no independent reviewer yet)
- **Code / commit:** branch `unimoe-stage1` (worktree D:/shm_um); new modules `unimoe/heads.py`, `pools.py`, `density.py`, `capacity.py`, `refusal.py`, `systems2.py`, drivers `p4.py`, `p4_refusal.py`, `p4_report.py`; tests `tests/unimoe/test_um_pools.py`, `test_um_p4.py` (77 pass in the package); `growmoe/` untouched; commits 074fd1e (protocol and predictions), e89555e, b6b9450, 2544413 and the commits after it
- **People:** Claude built and ran it on the user's request of 2026-10-30 ("test and run and analyze and improve all 4 and report once done"); no reviewer yet

## Question
The Stage 1 analysis (E-053) ranked four changes: (1) growth read from a statistic that does not depend on the training recipe, tested on a stream that needs capacity, (2) a supervised domain head, (3) a refusal layer with calibration on verified outcomes wired to the pool, (4) a better expert fit. Does each help, by how much, and where does the combined pool stand against its baselines and ceilings?

## Why we asked it
The user wanted a working dynamic MoE and a full analysis of where it is short (2026-10-30). E-053 found growth dead, the pool's own refusal silent, experts weak and the pool level with simple baselines.

## Prediction (written BEFORE the numbers)
`research/unimoe/PROGRESS.md`: 23 numeric predictions for the four changes (commit 074fd1e, before any new cell), then 4 for the density reader (B1), 5 for growth from a small start (B2), and 6 corrected ones after a sign error in my first grid (B2b'; the void sign-based ones are excluded): 38 in all, **17 held, 21 missed (45 %)**. Stage 1's rate was 31 of 63 (49 %).

## Method (how)
- Protocol: seed 499 tunes (growth threshold on the capacity stream; per schedule the domain power beta in {0.5, 1, 2} and the LDA weight lambda in {0.25, 0.5, 0.75}; per schedule the small-start threshold), seeds 500-502 evaluate, seeds 503-504 confirm the final combined system; control UG0 = U-real (reproduces it cell for cell).
- Variants (`systems2.py`, all U-real plus switches): UG1 growth from a one-step probe with fast maturity; UDOM / UDOMS domain head hard / soft; ULDA an LDA head blended into every expert; UPLUS all of them; `oracle_lda` experts told the regime, each with an LDA head.
- Capacity stream (`capacity.py`): six arithmetic operators, 10,000 examples each, no text, no OOD sets. Real suite: the 22 regimes of E-053.
- Refusal (`p4_refusal.py`): a pool trained on the sequential stream, then every known text test message and 300 messages of each foreign set read by the pool (1 - p-value), the asking reader (six cards + none) and a density reader (Mahalanobis distance to the nearest taught class); calibration on 60 verified known messages (5 draws), q = 0.05, fusion by "either", evaluation on the other known messages.
- Commands: `python -m unimoe.p4 {capacity-widths,capacity-tune,capacity-eval,oracle-lda,main-tune SCH,main-run,small-start,small-tune SCH,small-eval}`, `python -m unimoe.p4_refusal run|report`, `python -m unimoe.p4_report`.

## Results (what came out)
Generated tables: `results/unimoe/p4/TABLES.md`, `results/unimoe/p4/REFUSAL.md`, `results/unimoe/p4/REFUSAL_503_504.md`, `results/unimoe/p4/ladder.json`; analysis `results/unimoe/ANALYSIS_P4.md`. Final mean accuracy over the 22 regimes.

**Change 4, LDA head.** Under oracle routing (500-502) lambda 0 / 0.3 / 0.5 / 0.7 / 1.0: sequential 0.669 / 0.714 / 0.731 / 0.745 / 0.750, mixed 0.663 / 0.737 / 0.750 / 0.752 / 0.748, drifting 0.604 / 0.706 / 0.737 / 0.751 / 0.749. In the pool, paired difference to U-real over 5 seeds: ULDA sequential +0.022 [+0.010, +0.033], mixed +0.069 [+0.057, +0.082], drifting +0.126 [+0.100, +0.153]; all 15 paired differences positive; confirmation seeds +0.013 / +0.069 / +0.135.

**Where the combined pool stands (5 seeds).** UPLUS 0.646 / 0.663 / 0.642 (sequential / mixed / drifting) against U-real 0.624 / 0.594 / 0.516, equal-memory replay 0.642 / 0.627 / 0.597, equal-memory kNN 0.644 / 0.502 / 0.531, the offline joint network 0.667 / 0.669 / 0.670. UPLUS minus joint: -0.021 [-0.036, -0.007], -0.006 [-0.021, +0.010], -0.028 [-0.085, +0.029]. UPLUS minus replay: +0.004, +0.036, +0.045; minus kNN: +0.002, +0.161, +0.112. Experts told the regime and carrying LDA heads: 0.745 / 0.752 / 0.751 (the ceiling for a system that is told the task).

**Change 2, domain head** (5 seeds; hard / soft): sequential -0.001 / +0.001, mixed +0.010 / +0.011, drifting +0.032 / +0.032; wrong-domain 10.8 -> 9.9 %, 9.6 -> 7.0 %, 14.2 -> 7.1 %; beta had no effect (saturated posterior); UPLUS = ULDA +- 0.001.

**Change 1, growth.** Capacity stream (500-502), fixed 16 / 128 / 512 units: sequential 0.725 / 0.806 / 0.829, mixed 0.678 / 0.753 / 0.768. From 16 units: Stage 1 rule with fast maturity 0.725 and 0.678 (0 events); one-step probe at theta 0.2 (tuned on 499: 0.02, 0.05 no growth; 0.1: 2 events; 0.2: 7 events; 0.4 the same) 0.782 (7.7 events, 475 k parameters against 803 k for 512 units) and 0.718; at 5 inner steps 0.779 (vs 0.736 fixed) and 0.769 (vs 0.707). Real suite from 16 units (fixed 16 / 128: 0.581 / 0.622, 0.442 / 0.596, 0.386 / 0.512): grown 0.581 (events 0, 1, 0), 0.478 (1, 1, 1), 0.383 (1, 1, 2); theta 0.3 to 0.9 identical. The sequential experts get a median of 5 batches (min 4, max 43) of training; the rule needs age >= 30 and two windows of >= 10 (enforced by `PhyloCfg`). With the tuned 128 units the rule fires nothing and equals U-real.

**Change 3, refusal** (sequential, 60 verified messages, q = 0.05; refused share of science questions / stories / wiki; false refusal): pool built-in flag 0 / 0 / 0 at 0.000; pool calibrated 0.46 / 0.21 / 0.37 at 0.056; asking reader calibrated 0.17 / 0.29 / 0.13 at 0.053; pool + reader 0.45 / 0.33 / 0.34 at 0.073; density reader 0.50 / 0.07 / 0.39 at 0.036; all three 0.45 / 0.22 / 0.32 at 0.052 (confirmation 0.57 / 0.23 / 0.40 at 0.061). AUROC density 0.93 / 0.81 / 0.89, pool 0.85 / 0.79 / 0.83, reader 0.83 / 0.84 / 0.79. At q = 0.10 all three: 0.71 / 0.42 / 0.53 at 0.107. Accuracy on the answered messages +0.011 to +0.021 over answering all.

## Interpretation (what we understood, and on what basis)
Plain words first. The experts were the weak part of the pool, not the routing: a classifier trained by a few gradient steps per batch on frozen sentence vectors is worse than the exact closed-form fit (LDA), by 0.08 when the task is known. Giving every expert an LDA head closes most of that and brings the pool to the offline ceiling in the mixed and drifting schedules. Growth is a measurement problem: the one-step statistic works on a stream where an expert lives for hundreds of batches and fails by construction where tasks are a few batches long, because the rule needs a window of history that does not exist. The refusal layer does what calibration can do (a controlled false-refusal rate); the amount it catches is limited by how well the scores separate known from foreign text (AUROC about 0.85).
Technical statement. Measured: all numbers above. Assumed: that the LDA head's dominance is a property of the frozen embedding (not tested with a trained encoder); that the task-ambiguity floor explains the gap between the ceiling for a system not told the task (0.667-0.670) and the told-the-regime LDA experts (0.745-0.752).

## Caveats and what went wrong
- F-62: a one-line mutation-test leftover in `unimoe/runner.py` (the per-phase known-flag rate hard-coded to 0.0) sat in the working tree during all of Stage 1 and P4; its test catches it; restored; the published end-of-stream value recomputed from the stored arrays is 0.0 for every system, so no published number changed (the per-phase maximum is unverified, nothing cites it).
- F-63: my first small-start threshold grid (0.005-0.1) had the sign backwards (growth fires when the gap is below theta); void, redone as 0.3-0.9.
- F-64: the small-start test repeated Stage 1's E6 (16 units, fast maturity, mixed: 0.442 -> 0.478, one event) which I had not looked up; predictions about it missed.
- F-65: the prediction tally, above.
- Seeds are development seeds; lambda reached the top of its grid in two schedules (plateau in the oracle curve); the threshold was tuned on one seed; the refusal evaluation covers the six text domains only and takes its verified messages from the same test sets as its evaluation messages (disjoint). The repository's best-of-seeds warning on `p4.py` is a false positive (every seed run and reported).

## Consequence
D-43: the LDA head becomes the default of the next UniMoE system (recommended configuration ULDA, lambda 0.5-0.75, domain head optional); U-real stays pinned as the Stage 1 reference. Next: per-visit right-sizing of experts (Q-41), a better representation for refusal (Q-42), the transfer question to a trained representation (Q-43) and Stage 2.

## Evidence
- `results/unimoe/ANALYSIS_P4.md` - the analysis in plain words with the scorecard
- `results/unimoe/p4/TABLES.md`, `results/unimoe/p4/REFUSAL.md`, `results/unimoe/p4/REFUSAL_503_504.md`, `results/unimoe/p4/ladder.json`, `results/unimoe/p4/figure_ladder.png` - generated tables and figure
- `results/unimoe/known_flag_rate_repaired.json` - the recomputed known-flag rate of every Stage 1 system (F-62)
- `results/unimoe/p4_tuning.json` - the tuned operating points
- `research/unimoe/PROGRESS.md` - protocol and every prediction before its cells, results in order

## Reproduce
```
cd D:/shm_um
PYTHONPATH=D:/shm_um python -m unimoe.p4 capacity-widths; ... capacity-tune; ... capacity-eval; ... oracle-lda
PYTHONPATH=D:/shm_um python -m unimoe.p4 main-tune seq   # then mix, drift
PYTHONPATH=D:/shm_um python -m unimoe.p4 main-run [--seeds 503 504]
PYTHONPATH=D:/shm_um python -m unimoe.p4 small-start; ... small-tune seq   # then mix, drift; ... small-eval
PYTHONPATH=D:/shm_um python -m unimoe.p4_refusal run --systems UG0 UPLUS [--seeds 503 504]
PYTHONPATH=D:/shm_um python -m unimoe.p4_refusal report; python -m unimoe.p4_report
python -m pytest tests/unimoe
```

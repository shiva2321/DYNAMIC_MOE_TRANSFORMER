# E-040 - RV2-D pre-registration frozen: routing v2 stage 4, confirmatory run waits for the user's go

- **ID:** E-040
- **Date:** 2026-10-17
- **Track:** G
- **Type:** process (pre-registration)
- **Status:** completed (frozen); the confirmatory run on seeds 85-99 has NOT started
- **Evidence grade:** n/a (a record made before any confirmatory seed ran)
- **Code / commit:** freeze commit 44edd74; `research/growmoe_results/rv2d/bars_rv2d.json` sha256 f445f86018dc1034 (`bars_rv2d.sha256`), `PREREG.md`; code stamp ff3438232c5f0dfa (the merged code of stage 4, stage 4.1 and amendments A4-1, A4-2)
- **People:** Claude wrote and froze it; the user accepted amendment A4-1 on 2026-10-17; the confirmatory run needs the user's explicit go

## Question
What exactly will be tested on the fresh seeds 85-99, with which bars and constants, and what do we expect before we look?

## Why we asked it
Plan task 7. The spec (approved 2026-10-15) fixes the statements S4-1 to S4-9 and their bars; the dev review (E-039) fixed the constants by the declared rules and led to amendments A4-1 and A4-2, which are in force from this freeze.

## Prediction (written BEFORE the numbers, if there was one)
All nine statements pass: the dev verdicts are PASS for S4-1, S4-2, S4-3, S4-4, S4-6, S4-8, S4-9; S4-5 and S4-7 cannot be measured in dev (they need the transfer and growth arms on every guard stream) and are expected to pass. Where I think the risk is, most likely first:
1. **S4-5** on the guard streams that dev never ran with the transfer rule (F-30 and S2-30 with 30 regimes, OV-0.0, OV-0.3, OV-0.3-MIXED): a cascade of false spawns there would fail the expert count, an accuracy loss the accuracy half.
2. **S4-8**: forgetting of the full stack is 7.5 points in dev against a bar of 8, with two seeds.
3. **S4-6**: the 16-unit expert with growth must reach 90 % of the 128-unit pool in at least 13 of 15 seeds with a final width of 64 to 256; dev grows to 128 at the chosen threshold (the band is 64 to 256).
4. **S4-7**: at most one growth event per run on the one-function stream (dev: 0.33 per run at the chosen threshold).
S4-1 (dev lower bound +8.7 against the bar +5), S4-2 (+7.0 against +4), S4-3 (all guard streams 0.0 or above), S4-4 (+8.2, forgetting 7.4) have room.

## Method (how)
`python research/growmoe_results/rv2d/freeze_prereg.py` (once; it refuses to run again) after the smoke test of the merged runner (six cells including banking77, pool seed 107 confirmed for stream seed 100) and the mutation check (27 of 27 mutants killed). Arms: S3, S3+T, S3+G, S3+T+G; the router is evaluated on the same trained pools. Seeds: ten stage-3 streams x 15 seeds (85-99), arithmetic operator-by-operator 85-90, banking77 85-89, arithmetic mixed from 16 and from 128 units 15 seeds each; 704 cells in `cells_confirm.json`. Constants from the dev rules: `grow_gap` 0.17, `transfer_margin` 0.25, router `free` with margin 0.01. Function-regime streams use the pool seed stream seed + 7.

## Results (what came out)
Nothing yet. The freezer's golden check (S4-9) passed at freeze time: with every stage-4 flag off the pool is bit-identical to the recorded one.

## Interpretation (what we understood, and on what basis)
Plain words. The bars and constants are now fixed in a file whose hash is stored; if anything has to change after the first confirmatory seed runs it is a recorded amendment, not a rerun, and the seeds are single-use. Cost estimate for the run, from the recorded CPU seconds of the dev cells: about 36 CPU-hours (600 function-regime cells of 70 to 90 CPU-seconds = 13 h, 24 operator-by-operator arithmetic cells of 30 to 60 CPU-minutes = 18.5 h, 60 mixed-arithmetic cells of 3 to 5 minutes = 3.8 h, 20 banking77 cells of 3 minutes = 1 h), which is about 6 to 10 hours of wall-clock on 9 threads (three cells at a time; the long arithmetic cells set the tail), resumable after a power cut (`RESUME_RV2D.cmd confirm`). The PC must not sleep while it runs: a sleeping PC stalls the jobs between turns.

## Caveats and what went wrong
- The code stamp is part of the contract: editing anything under `growmoe/` or `rv2d_lib.py` after this point makes new cells incompatible with the frozen file. Documents, tests and report scripts may change.
- Dev evidence for S4-5 and S4-7 is partial by construction.

## Consequence
STOP. The user decides when to spend seeds 85-99 (decision D-36). Then: `RESUME_RV2D.cmd confirm`, `report_rv2d.py confirm`, an audit by a fresh reviewer, and adoption of each component on its own statements.

## Evidence
- `research/growmoe_results/rv2d/PREREG.md`, `bars_rv2d.json`, `bars_rv2d.sha256`, `cells_confirm.json`
- `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`
- `research/growmoe_results/rv2d/RESULTS_DEV_REVIEW.md`

## Reproduce
```
cd research/growmoe_results/rv2d
python make_cells.py confirm            # the 704-cell list (deterministic)
python jobqueue.py status --phase confirm
RESUME_RV2D.cmd confirm                 # ONLY after the user's go: seeds 85-99 are single use
python report_rv2d.py confirm
```

# E-071 - M1 stage B: all arms tuned on dev seeds, the dry-runs of the report find five defects, the scored run costed at about 37 hours

- **ID:** E-071
- **Date:** 2026-11-03
- **Track:** W
- **Type:** experiment
- **Status:** completed (dev rehearsal only; nothing frozen, no scored run)
- **Evidence grade:** B for the tuning (script-generated, five development seeds, a trial cache shared by every process); C for the dry-run verdicts (development seeds 800 and 801, wall-clock from seed 800 only); the cost estimate is an extrapolation and says so
- **Code / commit:** tuning data 5377beb; report and cell repairs 03c9a55 and aedee65; dev cells and dry-run report committed with this entry
- **People:** Gemini built stage B (jobs B-1, B-2, B-3), Claude reviewed, repaired, ran the tuning and the dev cells and found the defects; the user was asleep and said "you do you" for the already-approved dev work

## Question
Is the M1 pipeline (tuning of 14 arms, cell runner, report) ready for the scored run, and what will the scored run cost on this machine?

## Why we asked it
The M1 preregistration (D-77) is approved but not frozen. Freezing it fixes the arms, the operating points and the report; a defect found after the freeze would cost a re-freeze or a flawed scored run. The user asked for a rehearsal on development seeds first (E-068, D-87..D-90).

## Prediction (written BEFORE the numbers, if there was one)
None for the tuning (it chooses an operating point). For the rehearsal: the dry-run of the report would show defects, as the first stage-B reviews did (D-87, D-90); I did not expect the pool to lose to a ridge head on the banking-only beds.

## Method (how)
- Tuning on development seeds 800-804: best mean final accuracy per arm, a trial cache keyed by (arm, config hash, seed) shared by the driver and parallel workers; B1 tuned at N = 30, B2 at N = 6, B3seq at N = 22 (B3mix and B3drift reuse B3seq). Results in `results/m1/tuning/{B1,B2,B3seq,B3mix,B3drift}.json`.
- Dev cells (`results/m1/dev_b`, 233 files): 175 `det` cells (deterministic metrics, no timing, made in parallel shards) and 58 `full` cells (the isolated timing pass: one process at a time on logical processors 0 and 2, seed 800: B2, B3seq, B1 N = 30 with all arms and H_* cells, B1 N = 100 for A2 and A3-2). The det cells made before commit 03c9a55 lack the `accounting` block and carry the overwritten `memory_bytes_by_regime`; the timing pass rewrote the seed-800 cells it covers.
- Report: `python -m m1.report --cells results/m1/dev_b --out results/m1/dev_b_report --split dev` (the report seeds 810-829 stay locked).

## Results (what came out)
**Tuned operating points (best mean final accuracy on seeds 800-804, from the tuning files).**

| Arm | B1 (N = 30) | B2 | B3seq |
| :--- | :--- | :--- | :--- |
| A1 as built, steps tuned | 20 steps, 29.05 % | 50 steps, 79.23 % | 50 steps, 52.80 % |
| A2 reference pool | steps 5, q 0.02, 42.19 % | steps 50, q 0.05, U-real + chance-weak, 85.11 % | steps 50, q 0.02, ULDA, 63.95 % |
| A3 replay at the pool's memory | hidden 8192, 34.58 % (at the edge) | hidden 1024, 84.99 % | hidden 512, 61.80 % |
| A5b Mahalanobis | shrinkage 0.5, 26.03 % | 91.31 % | 64.02 % |
| A6 random-feature ridge | D 8192, 25.45 %, 268.7 MB (at the edge) | D 4096, 91.47 %, 68 MB | D 8192, 70.18 %, 272.5 MB (at the edge) |
| A7 kNN | k 15, 25.70 % | k 1, 88.49 % | k 5, 61.25 % |

**Dry-run of the report on the dev cells** (`results/m1/dev_b_report/REPORT_M1.md`): M1-1 PASS, M1-2 FAIL (latency ratio pool to replay 10.1 at batch 1 and 36.8 at batch 32, so the gap widens), M1-3 PASS (training wall-clock of the pool 0.21 of the replay arm at N = 30 and 0.42 at N = 100), M1-4 PASS (gate, router and bookkeeping 0.82 and 0.83 of the pool's observe time on B2 and B3), M1-5 PASS, M1-6 FAIL (streaming LDA is 0.032, 0.035 and 0.022 below the LDA-head pool on B3seq, B3mix and B3drift, the bar is 0.02), M1-7 PASS, M1-8 PASS (R^2 0.993, slope 359,015 bytes per regime against 352,535 bytes per expert), M1-9 FAIL on one part (LDA head adopted as predicted; the growth rule is not adopted: +0.16 % wall-clock against the bar of -15 %; skip and chance are not adopted, as predicted).

**Pool against baselines on the dev seeds** (final accuracy, two seeds): B1 N = 30, A2 42.13 % against the best baseline A3 34.64 % (memory 12.1 MB against 21.5 MB); B1 N = 100, A2 41.35 % against A3-2 27.20 % (the only wide baseline timed there); B2, A2 86.11 % against A6 91.78 % (68 MB), A7 88.78 %, A5a 87.11 %; B3seq, A2 64.95 % against A6 70.58 % (272 MB), A7all 64.88 %, A5b 64.27 %.

**Reconciliation of costlab and `growmoe.accounting`** (report section 5, one bed per family): equal for the pool without shortlist, the replay nets, A7 and A5a (ratio 1.000); costlab counts 0.034 of the accounting for A2 at B1 N = 100 (112,128 against 3,310,592 multiply-adds: the gate shortlist skips most of the scan); A5b is 128 times higher in costlab (1,261,568 against 9,856: accounting has no model for a per-class d x d covariance); A6 is 0 in accounting (random-feature ridge falls through to the default), 839,680 in costlab.

**Cost of the scored run, measured and extrapolated.** Cell run times of the timing pass: B2 4.1 min per seed, B3seq 28.9 min (three schedules about 87 min), B1 N = 30 42.3 min, B1 N = 100: A1 56 min (det, as built it fragments to 256 experts), A2 13 min, A3-2 7 min. The other N = 100 cells are N = 30 times 100/30 and the four H_* cells are counted as one A2 run each: B1 N = 100 about 207 min, B1 N = 10 about 14 min. One seed serial on two cores: about 354 min (5.9 h). Ten seeds with five of them timed alone and the other five in parallel det shards: about 37 h of machine time (29 h for the timed seeds, 7 h for the rest), with a spread of perhaps +-30 % because most N = 100 numbers are extrapolated.

## Interpretation (what we understood, and on what basis)
Plain words: the machinery now works end to end on real cells, and the dry-runs paid for themselves: five defects of ours or Gemini's were in the report and cell code that the tests and Gemini's compliance matrix had not caught. On development seeds the pool clearly beats every baseline on the many-regime bed B1 and is level with kNN and LDA on the banking-only beds B2 and B3, but behind a wide random-feature ridge head that uses 18 to 22 times its memory (A6 is at the edge of its grid on B1 and B3seq, so its true ceiling is unknown). The pool's answer-time gain comes from the shortlist and appears only once the pool is large (N = 100); at 30 experts the pool answers more slowly than any replay net (4.3 ms against 0.2 ms at batch 1). The dominant update cost is the router refit (71 % of the pool's update time in the dev cells), not the expert training (6.6 %).

The five defects: (1) the reconciliation and answer-driver tables were guessed formulas and typed text; (2) the memory-by-regime points that M1-8 fits were overwritten by B1's revisit phases (R^2 0.62 instead of 0.99); (3) the report crashed on every full cell (peak memory is a dict); (4) the reconciliation averaged different cells on its two sides; (5) the summary figure drew one Pareto front through beds whose accuracies are not comparable. All five are repaired with tests (`tests/m1`: 80 passed) and written into `docs/superpowers/handoff/M1_STAGE_B3_RESULT.md` sections 6.6 and 6.7.

## Caveats and what went wrong
- Dev seeds only, two seeds for accuracy and one (800) for wall-clock; the dry-run verdicts are a rehearsal, not a result, and the dev data feed nothing in the scored report.
- The det cells before 03c9a55 lack the accounting block; the timing pass covers only seed 800, so the reconciliation rows read "2 (1)" cells for B2 and B3seq. Every scored cell will carry the block.
- Mixing det and full cells means accuracy means run over more seeds than wall-clock means; the report skips cells without timing (the key is missing), it does not average zeros. The scored report will say how many seeds each column rests on.
- A1 as built needs 56 min per N = 100 cell because it fragments to the 256-expert cap (E-070, C-200); the scored run needs a decision on whether that cell is run at all seeds.
- The "A8 joint ceiling" is below the pool on B1 (32.3 % against 42.2 %): the known streaming-recipe artifact (E-068), not a true ceiling.
- Gemini's compliance matrix was wrong again (row 3 of B-3 said "No deviations").

## Consequence
- C-201, F-112, D-91, Q-82. Freeze of the preregistration and of `m1/config_m1.py`, the report-split unlock and the scored run wait for the user's go; no new Gemini job was started.
- Open for the go: the cost (about 37 h, which needs the daily-outage plan: idempotent shards, one relaunch command) and the A1 N = 100 cell.

## Evidence
- `results/m1/tuning/B1.json`, `B2.json`, `B3seq.json`, `B3mix.json`, `B3drift.json` - the chosen operating points and the trial summaries
- `results/m1/dev_b/*.json` - the 233 dev cells (175 det, 58 full)
- `results/m1/dev_b_report/REPORT_M1.md`, `verdicts.json`, `figures/` - the dry-run report
- `docs/superpowers/handoff/M1_STAGE_B3_RESULT.md` - sections 6.6 and 6.7, the five defects

## Reproduce
```
python -m m1.report --cells results/m1/dev_b --out results/m1/dev_b_report --split dev
python -m pytest tests/m1 -q
python -m m1.queue --beds B2 --arms "A1,A1q,A2,A3,A4,A5a,A5b,A6,A7,A7all,A8,A3-2,A4-2,A7-2,H" --split dev --seeds 800 --out results/m1/dev_b --profile full --tuning results/m1/tuning --shard 0/1 --cpus 0,2 --threads 2
```

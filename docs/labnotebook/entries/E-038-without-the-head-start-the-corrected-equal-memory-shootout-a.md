# E-038 - Without the head start: the corrected equal-memory shootout and the function-regime dev cells

- **ID:** E-038
- **Date:** 2026-10-16
- **Track:** G
- **Type:** experiment (re-measurement after the seed-aliasing audit E-035)
- **Status:** completed (dev seeds; the pre-registered confirmatory runs are not affected: they have not been run)
- **Evidence grade:** B (dev seeds 100-104 for the shootout, 100-102 for the dev cells; script-generated; exploratory)
- **Code / commit:** the commit that adds this entry; code stamp `e468a8911472d079` for the dev cells; `research/growmoe_results/shootout/shootout_unaliased.py`, `research/growmoe_results/rv2d/run_s3_unaliased.py`, `report_s3u.py`
- **People:** the choice to re-measure was mine (the user left the head-start decision to me); run and read by Claude; not reviewed by anyone else

## Question
E-035 found that in the function-regime streams the expert founded for regime r starts with regime r's teacher units (pool seed = stream seed). How much of the headline equal-memory result (C-40 to C-42) and of the stage-4 dev cells is that head start?

## Why we asked it
The user's question "is GrowMoE working as intended now or not" needs the headline numbers to be honest. The head start was measured on the dev runner for two arms (E-035); the shootout and the other arms were not re-run.

## Prediction (written BEFORE the numbers, if there was one)
None. From E-035 I expected a loss of several points on the pool at the function-regime streams and nothing on S1, S2-5 and the arithmetic and text streams.

## Method (how)
- Shootout (`shootout_unaliased.py`): the pool and the oracle-routed bank re-run with pool seed = stream seed + 7, N = 10 and 30 (seeds 100-104) and N = 100 (seeds 100-102); the baselines (best same-memory replay network, kNN at the pool's memory, joint network) are the old records because they do not use the pool seed. Same budgets, streams and settings as `shootout.py`.
- Dev cells (`run_s3_unaliased.py`, `cells_s3u/`): the same 60 cells as the dev runs of the function-regime family (streams F-15, OV-0.6, OV-0.85, S1, S2-5; arms S3, S3+T, S3+G, S3+T+G; seeds 100-102; constants of `dev_constants.json`) through the unchanged `rv2d_lib.run_cell`, with only the pool seed shifted by 7. Compared with `cells/` by `report_s3u.py`.

## Results (what came out)
Shootout (`RESULTS_SHOOTOUT_UNALIASED.md`; accuracy percent, mean over seeds):

| N regimes | pool with the head start | pool without it | best same-memory replay network | pool minus replay, with / without the head start | kNN at the pool's memory |
|---|---|---|---|---|---|
| 10 | 50.9 | 43.0 | 36.8 | +14.1 [+12.4, +15.9] / +6.2 [+4.5, +7.9] | 21.8 |
| 30 | 49.7 | 42.2 | 33.1 | +16.7 [+15.7, +17.7] / +9.1 [+8.0, +10.3] | 21.9 |
| 100 | 46.6 | 41.6 | 28.6 | +18.0 [+10.5, +25.5] / +13.0 [+11.7, +14.2] | 21.4 |

The head start itself on the pool: +7.9 [+6.0, +9.8] (N = 10), +7.6 [+6.9, +8.2] (N = 30), +5.0 [-1.6, +11.6] (N = 100, 3 seeds). The oracle-routed bank loses more (48.1 -> 38.3, 47.9 -> 38.0, 46.5 -> 36.7), so without the head start the pool is 4-5 points ABOVE the oracle-routed bank (earlier the two were equal); the oracle-routed bank is a separate baseline with its own training setup, not an upper bound. Memory (bytes) and multiply-adds per prediction are unchanged by the seed (pool 35.2 MB, about 175-180 k multiply-adds at N = 100; replay network 35.2 MB).

Dev cells, S3 arm, accuracy percent with / without the head start: F-15 50.4 / 42.7 (+7.7), OV-0.6 42.9 / 41.1 (+1.8), OV-0.85 29.0 / 29.3 (-0.2), S1 78.0 / 79.2 (-1.2), S2-5 90.9 / 89.5 (+1.4). What each stage-4 part adds without the head start (points over S3, paired over 3 seeds): transfer rule F-15 +0.2 [-0.5, +0.9], OV-0.6 -0.0 [-1.8, +1.7], OV-0.85 -0.5 [-0.9, -0.1], S1 +1.8 [-3.1, +6.7], S2-5 +2.7 [-10.1, +15.5]; growth from the gap exactly 0.0 on every one of these streams (it never fires there); the router on OV-0.85 +10.0 [+7.0, +13.0] (with the head start +9.5 [+6.1, +12.9]).

## Interpretation (what we understood, and on what basis)
Plain words. About half of the advertised margin over the best single network at 10 and 30 regimes was the head start; what is left is real and grows with the number of regimes: +6 points at 10 regimes, +9 at 30, +13 at 100, at 0.08-0.20 of the compute per prediction. The pool's accuracy now falls by 1.4 points from 10 to 100 regimes (43.0 -> 41.6) while the best single network's falls by 8.2 (36.8 -> 28.6). The three stage-4 parts do almost nothing on these streams, which is what they are for (guard streams: they must not hurt); the router is the one that matters under heavy overlap and it does not depend on the head start. Measured: all numbers above. Assumed: that streams outside this synthetic family behave similarly (not tested; real text is in C-44).

## Caveats and what went wrong
- Dev seeds only; three seeds at N = 100 give a head-start interval that includes zero.
- Only the shootout and the s3-family dev cells were re-run. The earlier entries that quote function-regime numbers (E-013, E-014, E-015, E-017, E-019, E-021, E-023, E-032) are NOT re-run (Q-23 stays open); their conclusions on arithmetic, real text and routing mechanisms do not use the aliased streams.
- The pass line pre-declared in E-025 (at least +5 points and at most one third of the multiply-adds at N = 100) is still met without the head start (+13.0, 0.08 of the multiply-adds); the N = 10 margin is +6.2.

## Consequence
C-40 to C-42 get corrected numbers (C-76); C-75's reach is now measured on the S3 family. The confirmatory run of stage 4 will use a pool seed offset (decision D-34). The user's question is answered in the report with the corrected values.

## Evidence
- `research/growmoe_results/shootout/RESULTS_SHOOTOUT_UNALIASED.md`, `runs_unaliased/`
- `research/growmoe_results/rv2d/RESULTS_S3_UNALIASED.md`, `cells_s3u/`

## Reproduce
```
cd research/growmoe_results/shootout
python shootout_unaliased.py run --n 10 --seeds 100-104
python shootout_unaliased.py run --n 30 --seeds 100-104
python shootout_unaliased.py run --n 100 --seeds 100-102
python shootout_unaliased.py report
cd ../rv2d
python run_s3_unaliased.py run --phase core --workers 2
python report_s3u.py
```

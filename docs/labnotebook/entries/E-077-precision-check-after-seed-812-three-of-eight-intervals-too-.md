# E-077 - Precision check after seed 812: three of eight intervals too wide, timed seeds 813 and 814 follow

- **ID:** E-077
- **Date:** 2026-11-04
- **Track:** W
- **Type:** review
- **Status:** completed (the rule fired; seeds 813 and 814 launched)
- **Evidence grade:** B (a pre-registered stopping rule applied once to complete cells; the width of intervals only, no difference was read)
- **Code / commit:** `m1/precision_check.py` (sha256 of the LF file 30334e094946154eaed2dcd7cdbdc6681836dde5ff2d3ed730f7b10cde9c9c21, fixed before seeds 811 and 812 were read)
- **People:** Claude ran the script, once, on the finished timed cells

## Question
D-97 as corrected by D-100: is the timing of the three timed seeds 810, 811, 812 precise enough to stop at three, or must seeds 813 and 814 be timed?

## Why we asked it
The held defaults of M1 are decided on update wall-clock and answer latency with thresholds of 10 to 15 %; a confidence interval of the relative paired difference wider than half the threshold cannot support a decision at that threshold.

## Prediction (written BEFORE the numbers, if there was one)
None; the rule decides by width only.

## Method (how)
`python m1/precision_check.py --cells results/m1/scored --seeds 810 811 812 --out results/m1/precision_check_after_812.json`, run once after the last cell of seed 812 (132 of 132 cells for each of the three seeds; the script's hash was checked before the run). Half-width = t(0.975, n-1) x sd / sqrt(n) x 100 percentage points over the three seeds.

## Results (what came out)
growth (pooled over beds) 6.86 points (limit 7.5, ok); skip: B2 4.00 (limit 5, ok), B3seq 9.86 (TOO WIDE), B3mix 11.52 (TOO WIDE), B3drift 1.20 (ok); LDA head answer latency at batch 1: B3seq 0.52 (ok), B3mix 3.01 (ok), B3drift 12.34 (TOO WIDE). No incomplete cell. Decision of the rule: CONTINUE with seeds 813 and 814.

## Interpretation (what we understood, and on what basis)
Plain words: three of the eight comparisons that the held-default rules depend on are still too noisy at three seeds, so two more timed seeds are run, as decided beforehand. The widths may include disturbance from my review runs and Gemini's jobs on logical processors 8 and 10 (the machine was not idle: E-075, E-076), which is a reason to keep those runs short and pinned, not a reason to change the rule.

## Caveats and what went wrong
- The widths are the widths of three seeds; five seeds make them smaller by a factor of 1.24/2.48 = 0.5 only if the spread stays the same.
- The launcher of seeds 813 and 814 waited for a quiet machine because my reruns of the V-3 review were still running (log lines "quiet samples 0/3").

## Consequence
- D-104: timed seeds 813 and 814 (`TIMED_SEEDS="813 814" bash m1/scored_run.sh timed`, about 5.5 h), then the deterministic seeds 815-819, as D-100 says; the GPU probe and the GPU pass come between the timed pass and the deterministic pass (D-94).

## Evidence
- `results/m1/precision_check_after_812.json`, `m1/precision_check.py`, `results/m1/scored_timed_driver.log`

## Reproduce
```
python m1/precision_check.py --cells results/m1/scored --seeds 810 811 812
```

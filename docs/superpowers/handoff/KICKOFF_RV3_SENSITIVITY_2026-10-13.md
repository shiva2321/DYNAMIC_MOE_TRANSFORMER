# Kickoff message to Gemini, 2026-10-13: RV3-S (dev-only sensitivity of the batch-level gate) - after Claude's audit of the RV3 build and dev report

Why: the dev output (seeds 100-109) shows the two big effects replicate (OV-0.6 +18.1, OV-0.85 +6.4) but two guard streams (F-30, OV-0.0) miss the pre-registered guard rule and one group statement (S1, 50/50 mixed groups, -5.2 against -5.0) misses its bar. In every bad seed the batch-level gate ended with ONE extra expert: a false "new as a set" verdict on a known batch (level 0.001 per test) sends it to probation and the evidence test often spawns a duplicate. Claude wrote an exploratory script that tries three small remedies on the dev seeds, nothing else changed: a stricter level (a1e-4: 0.0001 with 9999 permutations), persistence (act only when two batches in a row are doubtful), both, and a sweep of the group veto level. The fresh seeds 60-79 stay untouched until the design is settled and re-frozen.

```
STOP RULE (read first): this message authorises ONLY the two commands below, in this order. After the REPORT you STOP. You do not edit any file, you do not run `prereg`, `run`, `m5` or `report` of run_rv2c.py, you do not use seeds 60-79, you write NO documentation, and you do not commit. "STOP IF" means: stop at once and paste what you have; do not try to repair it.

All commands run from the repository root D:\spherical_hyperspace_memory . Keep RAM below 85 % and use the defaults (2 workers x 3 threads).

1. `python research/growmoe_results/rv2c/sens_dev.py run`
   (300 runs of 10 streams + the label-sorted stream x 10 dev seeds x 3 variants; resumable: finished tasks are skipped if you must rerun; expect 30-60 minutes; do not run other heavy jobs meanwhile.)
2. `python research/growmoe_results/rv2c/sens_dev.py report`
   (prints the markdown report and writes research/growmoe_results/rv2c/SENS_RV3.md.)

REPORT (paste, do not summarise): (1) the complete printed report of step 2; (2) total minutes of step 1 and the peak memory seen; (3) `git status --short research/growmoe_results/rv2c`; (4) anything odd (a crash, a variant with no runs, a stream missing from a table).
STOP IF: step 1 crashes (paste the traceback, rerun the same command ONCE; if it crashes again, STOP); RAM above 85 %.
Then STOP. Claude reads the report when you paste it back.
```

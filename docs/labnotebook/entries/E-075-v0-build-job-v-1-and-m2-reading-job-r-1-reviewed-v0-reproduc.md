# E-075 - V0 build (job V-1) and M2 reading (job R-1) reviewed: V0 reproduces M1 exactly, six defects; 18 paper facts verified, the fidelity audit audited the wrong settings

- **ID:** E-075
- **Date:** 2026-11-04
- **Track:** W
- **Type:** review
- **Status:** in progress (V0 awaits the fix round V-2; the reading notes are accepted with the reviewer's corrections)
- **Evidence grade:** B for the reproduction checks (separate processes, development seed 800, two schedules); the smoke numbers are one development seed and are not results
- **Code / commit:** `v0/system.py`, `v0/demo.py`, `v0/auroc.py`, `v0/report.py`, `tests/v0/test_v0.py` (Gemini); `docs/superpowers/handoff/M2_READING_NOTES.md` (Gemini, section 11 by Claude)
- **People:** Gemini built and read (Antigravity, relayed by the user); Claude reviewed and reran

## Question
Did the V0 build (E-073's successor job V-1, spec `docs/superpowers/handoff/V0_BUILD_SPEC.md`) deliver an assembled model that really is the M1 cell `H_ULDA`, and can the reading notes (job R-1) carry the M2 preregistration?

## Why we asked it
The user approved the walking-skeleton order (D-98) and handed both kickoffs to Gemini. Four earlier Gemini reports said more than the work showed (F-108, F-110, F-112, F-113), so nothing is taken from the reports.

## Prediction (written BEFORE the numbers, if there was one)
None written. Expectation from the earlier reviews: a sound core and several overstatements.

## Method (how)
1. Fence: `git status` and the diff: no existing file changed (only the already modified harness check file and the allowed new files).
2. Read `v0/system.py`, `v0/auroc.py`, the summary logic and the comparator code of `v0/demo.py`, and the three result files.
3. Compared the V0 accuracy with the stored M1 dev cell `results/m1/dev_b/B3seq_n22_H_ULDA__seed800.json` (a number that existed before the job).
4. Reran the demo for seq and mix (development seed 800, logical processors 8 and 10, about 2.5 minutes) into `results/v0/review_seq_800` and `results/v0/review_mix_800` and compared every non-timing field with Gemini's records; reran six of the seven tests (67 s).
5. For the reading notes: extracted the text of five of the six papers from the PDFs and searched them for 18 facts.

## Results (what came out)
**V0 build, held:** no existing file touched. V0 accuracy 0.6501575735482302 on B3seq seed 800 equals the stored M1 dev cell exactly. The reruns of seq and mix are bit-identical to Gemini's records in every non-timing field (accuracy, per-regime and per-phase accuracy, experts, memory, forgetting, comparator numbers, unknown-flag rates, AUROC); V0 update seconds 41.95 against 37.03 (seq), 15.72 against 14.96 (mix). Six of seven tests pass (67 s); the seventh (assembly) is accepted through the stored-cell comparison.
**V0 build, defects (to be fixed in job V-2, `docs/superpowers/handoff/V0_FIX_SPEC.md`):** (1) the comparators are A3, A7 and A5a only, and the summary says V0 beat "all three honest comparators"; the strongest simple baselines of M1 (A5b, A6, A7all) are missing: the spec asked for the three, so this one is the reviewer's fault. (2) On seq the memory-matched comparators got 6.77 MB (A3) and 6.20 MB (A7) against V0's 14.01 MB: they cannot use more because they already hold the whole training stream (about 6.5 MB), the summary does not say so, test (d) checked a boundary of a short run, and the matrix said "None". (3) The unknown flag fired on 0.00 % of known and of out-of-domain examples in all three schedules while the novelty AUROC is 0.68 (seq), 0.85 (mix), 0.88 (drift); the summary does not say that the flag is silent. (4) The assembly test takes 6 minutes against the specified 2 and was marked compliant. (5) `build_v0` silently falls back to 5 steps and q 0.05 if the tuned entry is missing. (6) The commands file has 5 lines for dozens of commands and was edited by hand although the result says it was redirected.
**Smoke numbers (development seed 800, one run, not results):** final mean accuracy V0 against A3, A7, A5a: seq 65.02 against 60.07, 64.54, 61.34 (20 experts, 14.01 MB, V0 answer latency 4.64 ms against 0.11 ms for A3); mix 65.21 against 58.79, 53.92, 61.50 (1 expert, 1.06 MB); drift 63.19 against 57.21, 54.74, 61.26 (1 expert, 1.18 MB).
**Reading notes, verified against the full texts (18 facts, all match):** 115,000 and 7,600 examples; 33 classes with Yelp and Amazon merged; the four task orders; MbPA++ Table 1 as printed; 70.6 against 60.7; LAMOL's "2-3% worse than multitasking" and the decaNLP tables (81.5; 76.6 and 74.3) behind it; LAMOL Table 5 and Table 2 rows; RanPAC M = 10000, the 80:20 split, the sweep 10^-8 to 10^8, Table 1 and Table A5; FeCAM shrinkage, correlation normalization and the Tukey rule; SLDA epsilon 10^-4, OAS initialization, Table 1.
**Reading notes, corrections (written into the notes as section 11):** the fidelity audit describes the constructor defaults, not the arms as M1 ran them. As run: A5a shrinkage 0.001 (the lowest value of its grid; the published 1e-4 was never tried), A5b shrinkage 0.5 (the highest value of its grid), A6 D = 8192 and ridge 1.0 on B1 and B3seq (both at the top of their grids) and D = 4096, ridge 0.1 on B2. The "HURTS" labels are expectations, not measurements. Two numbers are derived, not printed (LAMOL Table 2 averages; "RanPAC M = 1024 loses 2.3 points"). MbPA++ prints a Replay average (57.8) that is not the mean of its four orders (60.3).

## Interpretation (what we understood, and on what basis)
Plain words: V0 is the M1 pool with the LDA head run on real text, and that is now proven number for number; the demo around it was built correctly but compared it only with the weaker baselines, did not say that the comparators could not use the memory they were promised, and did not say that the unknown flag stays silent. The reading notes are reliable on the papers and unreliable on our own baselines: they checked what the code does by default, not what the scored run used. The more important consequence is for M1: our "FeCAM-style" and "RanPAC-style" baselines sit at the edges of their tuning grids, so they may be somewhat weaker than their published versions; that is a limit to name, and it cuts toward the pool looking better than it is, not worse.
Measured: everything marked rerun. Not measured: the effect of any of the structural differences between our baselines and the published methods.

## Caveats and what went wrong
- My review runs and Gemini's tests and smoke runs used logical processors 8 and 10 for several minutes (Gemini's test run alone took 6 minutes, windows not logged) while the CPU timed pass ran on 0 and 2; a timed cell that finished with an outlier at any time since about 01:00 on 2026-11-04 may be affected. Cells can be rerun with one command.
- The reviewer did not repeat Gemini's five literature searches; "none found" is not "none exist".
- The comparator flaw in V-1 was in my spec (three comparators chosen); the review caught it before any number was shown to anyone as a claim.

## Consequence
- D-99: V0 is accepted conditionally on job V-2; the reading notes are accepted with section 11. M1 stays frozen; the grid-edge and fidelity limits of A5a, A5b and A6 are named in the M1 report; faithful baselines are decided for M2 (Q-85).
- F-114, C-204, C-205.
- Next: the user hands the V-2 prompt to Gemini; Claude reviews; then V1 (exact removal, probe P2).

## Evidence
- `docs/superpowers/handoff/V0_BUILD_RESULT.md`, `V0_BUILD_TEST_OUTPUT.txt`, `V0_BUILD_COMMANDS.txt` - Gemini's report, raw test output, command log (incomplete)
- `results/v0/smoke_seq_800/record.json` and `summary.txt` - Gemini's smoke run
- `results/v0/review_seq_800/record.json`, `results/v0/review_mix_800/record.json` - the reviewer's reruns
- `results/m1/dev_b/B3seq_n22_H_ULDA__seed800.json` - the stored M1 cell V0 is compared with
- `docs/superpowers/handoff/M2_READING_NOTES.md` - the notes with the reviewer's section 11
- `docs/superpowers/handoff/V0_FIX_SPEC.md` - the fix round

## Reproduce
```
python -m pytest tests/v0 -v -k "not assembly"
python -m v0.demo --schedule seq --seed 800 --out results/v0/review_seq_800
```

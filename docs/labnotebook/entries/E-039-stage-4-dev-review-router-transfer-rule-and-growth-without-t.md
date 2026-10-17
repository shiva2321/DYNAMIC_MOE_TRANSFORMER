# E-039 - Stage 4 dev review: router, transfer rule and growth without the head start; amendments A4-1 to A4-7

- **ID:** E-039
- **Date:** 2026-10-17
- **Track:** G
- **Type:** audit (the dev review of the stage-4 plan, task 6)
- **Status:** completed; the amendments wait for the user's OK (A4-1) and the pre-registration is rehearsed, not frozen
- **Evidence grade:** B (dev seeds 100-102 for the function-regime and mixed-arithmetic streams, 100-101 for operator-by-operator arithmetic and banking77; script-generated; exploratory; none of this is a confirmatory result)
- **Code / commit:** dev cells produced with code stamp `e468a8911472d079` (the stage-4 build of 2026-10-15 before stage 4.1 and the amendments); reports and amendments in the commit that adds this entry; merged code stamp `ff3438232c5f0dfa`
- **People:** Claude ran and read it; the choices that change a pre-registered statement are flagged for the user; not reviewed by anyone else

## Question
With the function-regime streams run without the seed-aliasing head start (E-035, E-038), do the stage-4 parts (router 4a, transfer rule 4b, growth from the fit gap 4c) still do what the spec predicts, do the declared selection rules still choose the same constants, and does anything in the approved spec or in the pre-registration machinery need an amendment before the freeze?

## Why we asked it
Plan task 6 (dev review) comes before the pre-registration is frozen. The user left the head-start handling to me on 2026-10-16 (D-34), so the dev picture had to be rebuilt without it before it could be reviewed.

## Prediction (written BEFORE the numbers, if there was one)
The spec's statements S4-1 to S4-9 are the predictions; their dev verdicts are the "expected" column of the pre-registration. For the review itself: no part should hurt a guard stream, and the arithmetic results (no teacher, no head start) should be unchanged from dev1.

## Method (how)
- Cells: the dev1 and dev2 records of the arithmetic and banking77 families (`cells/`) and the function-regime family without the head start (`cells_s3u/`: core 60 cells, growth-threshold grid 24, the five remaining streams 15 at the dev-time code stamp from a second worktree), 165 records in all; `make_devview.py` builds `cells_devview/`, `report_rv2d.py dev --cells cells_devview --out RESULTS_DEV_REVIEW.md` reports it (the records are read with their own recorded stamp, `folder_stamp`).
- The declared selection rules (`rv2d_select_cli.py --cells cells_devview`) were re-run on the same view.
- Amendments are in `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`; the pre-registration was rehearsed with `freeze_prereg.py --out <scratch folder>`.

## Results (what came out)
- **Router** (free-0.01; the same trained pool answered with and without it), points with 95 % interval: arith-seq **+14.1** [+8.7, +19.5] (2 seeds; cross-validated owner accuracy density 87 / LDA 100 / softmax 100 / 5-NN 85 %, LDA chosen in 2 of 2); OV-0.85 **+10.0** [+7.0, +13.0] (59 / 89 / 88 / 64 %); OV-0.6 +2.0 [-0.7, +4.7]; exactly +0.0 on F-15, F-30, S1, S2-5, S2-30, banking77 and both mixed-arithmetic streams (density kept); OV-0.0 +0.5, OV-0.3 +0.5, OV-0.3-MIXED +0.6 (all intervals contain 0, none negative).
- **Transfer rule** (S3+T minus S3): arith-seq **+8.2** [+5.3, +11.1], experts 4.0 -> 6.0 (six operators), forgetting 15.1 -> 7.4; experts against the true number of functions: S1 4.0 -> 7.3 (7), S2-5 3.0 -> 5.0 (5), banking77 5.0 -> 6.0 (6), F-15 15.0 -> 15.0 (15), OV-0.6 and OV-0.85 10.3 -> 10.3 (10), the one-function stream 1 -> 1; accuracy: S1 +1.8 [-3.1, +6.7], S2-5 +2.7 [-10.1, +15.5], banking77 +1.4 [-12.7, +15.6] (2 seeds), F-15 +0.2, OV-0.6 -0.0, OV-0.85 -0.5 [-0.9, -0.1].
- **Growth from the fit gap** at the rule-chosen threshold 0.17: arith-mixed16 +1.6, arith-mixed128 -0.1, no growth event and exactly +0.0 on every other cell; the grid 0.20 to 0.26 gives +3.2 to +3.6 (mixed16) and +1.1 to +2.0 (mixed128) at +94 % to +444 % parameters, which fails the parameter guard; reverts per run 0.33 (mixed16 at every value, mixed128 at 0.17 where it is the only growth).
- **Full stack** on arithmetic operator-by-operator (router at its setting): S3 26.7 (router off) / 40.7 (router on), S3+T 34.9 / 47.0, S3+T+G 35.0 / 47.2, forgetting 15.1 -> 7.5.
- **Selection rules on the unaliased view**: grow_gap 0.17 (curve identical to the aliased dev1 curve, including the stage-3 group now measured at every value), router free-0.01 (mean gain 12.0), transfer margin 0.25, which is now chosen BY the rule: with the guard of A4-1 it is feasible, before it found no feasible margin (the S3-relative expert guard failed on S1 and S2-5, where S3 under-spawns).
- **Pre-registration rehearsal** (merged code, golden check inside): bars unchanged; dev verdicts S4-1, S4-2, S4-3, S4-4, S4-6, S4-8, S4-9 PASS (S4-3 is measurable now that all ten function-regime streams exist without the head start); S4-5 and S4-7 are NOT MEASURABLE in dev because they need the transfer and growth arms on every guard stream, and are counted as PASS in the expectation, as before.

## Interpretation (what we understood, and on what basis)
Plain words. Without the head start the picture of stage 4 is the same: the router fixes the large routing loss on the two kinds of stream built to need it (operator-like inputs and heavy overlap) and does nothing where the shipped gate is already right; the transfer rule makes the pool found the right number of experts where the shipped pool founds too few, and that is where its gain and its forgetting reduction come from; growth is nearly inert at the threshold the rule picks. Measured: all of the above on dev seeds. Not measured: any of it on fresh seeds (that is the pre-registered run), the transfer rule on streams with short regimes (Q-25), and growth where the starved expert really needs width at the rule's threshold (the rule picked the threshold that protects the parameter budget).

The one thing that needs a decision is A4-1: the pre-registered statement S4-5 compares the transfer rule's expert count with S3's, and S3 is the pool that under-spawns. Taken literally S4-5 would fail on S1 (7.3 against 4.0) and S2-5 (5.0 against 3.0) and the transfer rule would not be adopted for finding the right count. I amended the denominator to the true number of functions; that changes the approved text and is flagged for the user.

## Caveats and what went wrong
- Two seeds for arithmetic operator-by-operator and banking77: wide intervals; the router's arith-seq interval lower bound (+8.7) is the number S4-1 will be judged on (bar +5.0).
- I first wrote the amendments note saying that no pre-registered statement changed; reading `growmoe/verdicts_rv4.py` showed that S4-5 itself carries the S3-relative guard, so the note now says so.
- The queue rewrites `RESULTS_DEV.md` after each cell (it still contains the aliased function-regime cells); the review file is `RESULTS_DEV_REVIEW.md`.
- New dev cells had to be produced at the dev-time code stamp after the merge changed the stamp; a second worktree at the dev commit did that (`D:/shm_dev_e468`, removable).

## Consequence
D-35: amend S4-5 (A4-1), pool seed offset in the confirmatory run (A4-2), twin fix outside the confirmatory run (A4-7); stage 4.1 merged before the freeze. Next: the user's OK of A4-1, then freeze, then the user's explicit go before seeds 85-99. Claims C-80 to C-83.

## Evidence
- `research/growmoe_results/rv2d/RESULTS_DEV_REVIEW.md`, `DEV_SELECTION.md`, `dev_constants.json`
- `research/growmoe_results/rv2d/cells/` (dev1, dev2), `cells_s3u/` (function-regime family without the head start, 99 cells)
- `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`

## Reproduce
```
cd research/growmoe_results/rv2d
python make_devview.py
python report_rv2d.py dev --cells cells_devview --out RESULTS_DEV_REVIEW.md
python rv2d_select_cli.py --cells cells_devview
python freeze_prereg.py --out <scratch folder>      # rehearsal; the real freeze is done once
```

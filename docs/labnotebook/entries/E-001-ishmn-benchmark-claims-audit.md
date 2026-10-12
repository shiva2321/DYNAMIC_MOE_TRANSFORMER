# E-001 - Audit of the ISHMN headline benchmark (GSM8K / ARC): the claimed gain does not survive

- **ID:** E-001
- **Date:** 2026-09-21
- **Track:** I
- **Type:** audit
- **Status:** completed (re-confirmed 2026-10-09)
- **Evidence grade:** B
- **Code / commit:** audit of stored outputs `artifacts/benchmark_*_results.json` (local, gitignored); grader in the benchmark script of that name
- **People:** Claude (audit); user asked for the check

## Question
Is the headline result of the older ISHMN system (GSM8K 43.33 % -> 53.33 %, "+23 % relative", attributed to the Layer-14 "manifold sandwich") real?

## Why we asked it
Every roadmap decision about the sandwich rested on that number. A claim that large from n = 30 questions needed an independent look before anything was built on it.

## Prediction
None written; this was an audit of existing outputs.

## Method (how)
Re-grade every stored answer with a numeric comparison instead of the benchmark's string comparison; run an exact paired test (McNemar) on the discordant questions; list what differs between the two arms besides the claimed mechanism.

## Results (what came out)
- The benchmark graded with a **string** compare (`pred_num == gold_num`): a correct `$26.00` became `"26.0"` and failed against gold `"26"`. Two baseline items (idx 25, 28) were mis-graded that way; no grounded item was.
- Numeric grading: baseline 15/30 (50.0 %), grounded 16/30 (53.3 %); McNemar exact p = 1.000; 95 % CI of the difference [-14.0, +20.6] points. ARC: 1 discordant pair in 50, p = 1.000.
- Re-done 2026-10-09 from the files: as published 13/30 vs 16/30 (discordant 2 vs 5, McNemar p = 0.45); numeric 15/30 vs 16/30 (3 vs 4, p = 1.0); ARC 35/50 vs 36/50 (p = 1.0).
- Both arms were confounded: the grounded GSM8K arm also enabled a regex-plus-Python-AST calculator, the grounded ARC arm a `ChoiceLogitsProcessor`; the baselines got neither, and neither is the manifold sandwich.
- n = 30 resolves only effects of about +/-18 points; a 5-point paired effect needs n of about 400.

## Interpretation
The "+10 points" was a grading artifact on top of an unfair comparison, on a sample far too small to see the effect even if real. This does not show that the sandwich is useless; it shows the benchmark could not have told us.

## Caveats and what went wrong
We had accepted the number for some time before auditing it. The audit used the stored outputs, not a fresh run, so it settles the grading and the statistics, not the underlying model quality.

## Consequence
Standing rules adopted for every later experiment (see `DECISIONS.md` D-02): numeric or exact grading only; arms must differ in exactly one bit (a 4-arm factorial: sandwich x calculator); n >= 400 for small effects; never cite the +10 figure. The decisive ISHMN test (4-arm, numeric, n >= 400) was offered to the user and not run (see `OPEN_QUESTIONS.md` Q-12).

## Evidence
- `artifacts/benchmark_*_results.json` - the stored per-question outputs (local; not in git)
- `docs/COMPONENT_CARDS.md` - card 8 records ISHMN as UNVERIFIED

## Reproduce
Load the stored results, compare `float(pred) == float(gold)` per item, build the 2x2 table of correct/incorrect for the two arms, run an exact binomial test on the discordant pairs.

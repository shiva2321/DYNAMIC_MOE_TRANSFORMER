# E-011 - Phase 1: build, benchmark and the review trail (what held and what was an artifact)

- **ID:** E-011
- **Date:** 2026-10-04 to 2026-10-06
- **Track:** G
- **Type:** experiment
- **Status:** completed; headline claims later restated (see E-012, E-013)
- **Evidence grade:** B (reproducible; several predictions failed; the "beats replay" claim was later refuted)
- **Code / commit:** `growmoe/` at commit `7b37e0a` (later state); Phase-1 results regenerate with `python -m growmoe.bench --phase 1 --seeds 10 --out research/growmoe_results/phase1`
- **People:** Gemini (rounds R1-R12); Claude (review after every round); user (mediator)

## Question
Does the designed system (E-009) work, and which of the three architectures is best, under the pre-registered predictions?

## Why we asked it
It is the test of the user's bet on the first full implementation.

## Prediction
P1-P10 and C1-C6 of E-009, written before any code.

## Method (how)
Rounds R1-R10 built signatures, probation, spawn / extend, infants, competence, merge, growth, architectures B and C, streams and the benchmark runner (test-then-train protocol, hidden ground truth, dev seeds >= 100, test seeds 0-9, 10 seeds, 728 s). Each round ended with a report and a Claude review (`APPROVE`, `CHANGES`, `ANSWER`). Later rounds: R11 (stack, oracle-routing diagnostic D1, routing rule studies), R11c / R11d (rule selection at the benchmark's own configuration), R12 (learning loop).

## Results (what came out)
Phase 1 table (10 seeds; `research/growmoe_results/phase1/RESULTS.md`): PhyloMoE (A) average accuracy S1 69.82 %, S2 87.02 %, S3 85.47 %, S4 71.36 %; false-novelty rate 0.00 % everywhere. Predictions: **P1 PASS** (false-novelty <= 2q), **P8 PASS** (spurious spawn rate 1.00 %), C2 PASS, C6 PASS; **P2 FAIL** (spawn recall 0.517), **P3 FAIL** (revisit routing 89.5 %), **P4 FAIL** (S1 69.8 % vs oracle 78.1 %), C1 FAIL, C3 FAIL (parameter ratio 188.6 %, later found to be a double count: E-012), C4 FAIL, C5 FAIL (B's gate AUROC 0.91), P10 FAIL, P5 INCONCLUSIVE, P6 NOT MEASURABLE. Routing rule `accept_then_logf` lifted S1 from 55.96 % to 69.82 % (+13.86 [+6.02, +21.69]) with the gain entirely from skipping the test-time supersede filter (card 10).

Review findings (the reviewer's checks are the value of this entry):
- **R10 CHANGES:** the benchmark computed false-novelty as `1 - p_nov >= 0.5` (p <= 0.5, about 50 % by construction) instead of the pool's own `unknown` flag at level q; the reported FNR of 40-52 % was an artifact (true value 0.000). R10b: APPROVE after the fix.
- **R11 CHANGES:** results files had been **edited by hand** (a verdict printed NOT MEASURABLE in the file but FAIL from the code; missing rows; prose that the S4 data contradicted). R11b: APPROVE after regeneration; Claude reproduced D1 exactly.
- **R11c CHANGES:** the S2-S4 routing-rule evidence was run at q = 0.05 while the benchmark uses q = 0.02; at the real setting the claimed "+1.5 point S4 gain" is +0.06 [-1.00, +1.12]. R11d: APPROVE with fixes.
- **R12 CHANGES (R12b):** the learning loop was verified (trained infant vs frozen identity +0.1399 [+0.0502, +0.2297], with a valid control showing the gain is training, not capacity), but growth triggering, the pre-registration and several report numbers were not. R12b was paused by the user (E-012).
- Test counts: 240 at hand-over, 301 at R10, 311 at R11b, 319 at R11d, 328 at the 2026-10-09 catch-up.

## Interpretation
Phase 1 produced a working, reproducible system and honest failures of several predictions. It did **not** establish that the architecture beats simple alternatives; that question was reopened in E-012. The charter's "VERIFIED (R10b approved)" meant "the results are reproducible and correctly generated", not "the hypotheses held".

## Caveats and what went wrong
Three recurring implementer failure modes were caught only by review: metric definitions that make a result true by construction, hand-edited or hard-coded result prose, and comparisons run at a configuration other than the benchmark's own (`FAILURES.md` F-05, F-06, F-07). None would have been found by reading the final report alone.

## Consequence
Charter drift rules 3, 8, 9, 10 (numbers from generated files, benchmark's own configuration, determinism, failing tests are information). The register row for Phase 1 now carries the restatement. E-012 re-examined the central claim.

## Evidence
- `research/growmoe_results/phase1/RESULTS.md`, `research/growmoe_results/phase1/phase1_results.json`
- `research/growmoe_results/r11d/rule_selection.json`, `research/growmoe_results/r11d/rule_selection.md`
- `docs/superpowers/handoff/REVIEW_2026-10-05_R10_AND_Y1.md` - sections A-N, the reviews
- `docs/superpowers/handoff/GROWMOE_HANDOFF.md` - the rounds; `docs/superpowers/handoff/SCOPE_CHARTER.md` - drift rules

## Reproduce
`python -m growmoe.bench --phase 1 --seeds 10 --out research/growmoe_results/phase1` (reproduces every delivered number except wall-clock strings; the entry points pin the original behaviour via `bench._phase1_cfg`).

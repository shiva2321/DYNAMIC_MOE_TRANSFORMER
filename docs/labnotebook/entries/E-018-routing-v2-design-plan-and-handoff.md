# E-018 - Routing v2: the design, the plan and the hand-off, and why they look the way they do

- **ID:** E-018
- **Date:** 2026-10-10
- **Track:** G
- **Type:** design
- **Status:** completed (stage 1 executed in E-019; stage 2 in E-021)
- **Evidence grade:** H (design) built on B (E-017)
- **Code / commit:** `docs/superpowers/specs/2026-10-10-routing-v2-design.md`, `docs/superpowers/plans/2026-10-10-routing-v2.md`, `docs/superpowers/handoff/ROUTING_V2_HANDOFF.md` (commit `08c1a7a`)
- **People:** Claude (design, plan, hand-off); user approved approach A, then the spec ("sure continue"), then handed the plan to Gemini

## Question
How do we turn the spikes of E-017 into tested components without repeating the mistakes of Phase 1?

## Why we asked it
The user's two priorities (cheaper gate; messier data) and the standing requirement of pre-registration, fresh seeds, and no unverified headline.

## Prediction
Bars and expected outcomes written into the spec before any implementation: stage 1 V1 (overlap stream, >= +8.0 points, interval excludes 0), V2 (F-15 / F-30 >= +1.0), V3 (OV-0.6 >= +3.0) and guards (>= -3.0 and not significantly negative on S1, S2-5, S2-30, S4, no-overlap, 50/50 mixed, and known regimes arriving 70/30); stage 2 C1 (non-inferior accuracy: >= -1.0 mean and interval lower bound >= -2.0), C3 (gate multiply-adds <= 0.20x on F-30), C4 (500 experts: <= 0.05x multiply-adds and <= 0.5x wall-clock).

## Method (how)
Brainstorming flow: approaches A (three stages, recommended), B (stage 1 only), C (labels as evidence, rejected by E-017); written spec; self-review; then a code-complete 15-task plan (34 Python blocks syntax-checked, four bugs found by reading it against the real helpers); then a hand-off file following the three-party workflow.

## Results (what came out)
Design decisions and their basis:
- **Stage 1 = batch vote**, primary rule plain (best on 5 of 5 dev streams), the restricted vote a pre-declared fallback built only if a guard fails (E-017).
- **Stage 2 = fingerprint shortlist** with k-means sub-centroids (landmarks worse at equal cost), a top-2 shortlist and an **exact second look** for every sample whose restricted p-value is <= q. This gives a property, not just a measurement: *the novelty flag of every input equals the exact gate's*. Proof sketch (in the spec): Benjamini-Hochberg rejects using only p-values <= q; the restricted p-value is <= the exact one; every sample with restricted p <= q is recomputed exactly; so the entries <= q coincide and all others exceed q in both. Only the choice among accepting experts is approximate (recall 99.6-99.9 % in the spike). To be pinned by a property test.
- **Stage 3 = calibrated batch-level novelty**, conditional on a spike with hard acceptance criteria, because the first attempt over-fired.
- Seeds: stage 1 fresh 20-39, stage 2 fresh 40-59 (0-19 are spent); bars may only be lowered by a written rule when the dev effect is below 1.5x the bar; tuning, bars and pre-registration hash-frozen before the first fresh seed.
- `observe` does not use `_gate_matrices` today (it has its own loop), so stage 2 first routes it through the shared function, bit-identically.
- One spec inconsistency resolved in the plan: the restricted vote is not built up front (YAGNI).
- Plan self-review found: a spy test that never reached the code under test (one pool per batch), a tune task missing `peak_mb`, a counter assertion on the wrong side of a spawn, a self-contradictory unfreeze instruction.

## Interpretation
The design buys three things Phase 1 lacked: a decision rule per stage that is computed by script, a guarantee (identical novelty decisions) rather than a hope, and a reproducibility plan for every default flip.

## Caveats and what went wrong
The plan was code-complete but unexecuted when handed over; Gemini was warned to expect small slips.

## Consequence
Gemini executed rounds RV2-1 to RV2-3 (E-019); stage 2 handed over on 2026-10-12 (E-021).

## Evidence
- `docs/superpowers/specs/2026-10-10-routing-v2-design.md`, `docs/superpowers/plans/2026-10-10-routing-v2.md`, `docs/superpowers/handoff/ROUTING_V2_HANDOFF.md`

## Reproduce
n/a (design). The plan's commands are in the plan.

# E-021 - Routing v2 stage 2: the fingerprint shortlist gate (in progress)

- **ID:** E-021
- **Date:** 2026-10-12 (started)
- **Track:** G
- **Type:** experiment
- **Status:** in progress
- **Evidence grade:** H until the pre-registered test has run (dev-seed spikes: B)
- **Code / commit:** to be filled when round RV2-5 is reviewed (new: growmoe/fingerprint.py, not yet created; modified: `growmoe/pools/phylo.py`, `growmoe/config.py`, `growmoe/accounting.py`)
- **People:** Gemini (implementer, rounds RV2-5 to RV2-8); Claude (review, documentation); user (mediator)

## Question
Can the gate be made at least 4x cheaper at about 30 experts, with non-inferior accuracy and **identical novelty decisions** to the exact gate?

## Why we asked it
The gate scan is 97 % of the multiply-adds of a prediction (E-015); the user's first priority was a cheaper gate.

## Prediction
Written in the spec (E-018) and to be frozen in research/growmoe_results/rv2b/PREREG.md (not yet created) before any fresh seed: C1 non-inferiority on F-15, F-30, S2-30, OV-0.3 (mean >= -1.0, interval lower bound >= -2.0); C2 mean absolute spawn-count difference <= 0.5; C3 measured gate multiply-adds <= 0.20x (F-30), <= 0.25x (S2-30, F-15); C4 at 500 synthetic experts <= 0.05x multiply-adds and <= 0.5x wall-clock at B = 64 and 1024; G the pools too small for the cascade (S1, S2-5) bit-identical to the exact gate. Fresh seeds 40-59. Decision: adopt iff C1, C3 and C4 pass; C1 alone passing keeps it opt-in; G failing is a bug.

## Method (how)
Plan tasks 7-14 (`docs/superpowers/plans/2026-10-10-routing-v2.md`): fingerprint module, `observe` routed through `_gate_matrices` bit-identically, two-stage gate with a second look, gate-cost counters and `measured_gate_macs_per_input`, a 500-expert scaling check, verdict functions, a run script that generates its own pre-registration. Kickoff message to Gemini (2026-10-12) puts the stop rule first: stop after `prereg` and wait for approval before any fresh seed.

## Results (what came out)
Pending.

## Interpretation
Pending. If the cascade passes, the "efficiency" half of the dynamic-MoE bet gets its first pre-registered test.

## Caveats and what went wrong
Pending.

## Consequence
Pending.

## Evidence
- `docs/superpowers/specs/2026-10-10-routing-v2-design.md` (section 5), `docs/superpowers/plans/2026-10-10-routing-v2.md` (Tasks 7-14), `research/growmoe_results/rv2/RESULTS_RV2_SPIKES.md` (R1, R7)

## Reproduce
Pending.

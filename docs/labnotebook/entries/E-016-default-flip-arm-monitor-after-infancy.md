# E-016 - Making the confirmed repair the default (and keeping the old results reproducible)

- **ID:** E-016
- **Date:** 2026-10-10
- **Track:** G
- **Type:** decision
- **Status:** completed
- **Evidence grade:** A (the decision follows from E-015's pre-registered rule)
- **Code / commit:** `growmoe/config.py`, `growmoe/bench.py` (`_phase1_cfg`), `growmoe/competence.py` (docstring), `research/growmoe_results/m0/run_m0.py` (`A_KW`); in commit `7b37e0a`
- **People:** user decided; Claude implemented test-first

## Question
Should `arm_monitor_after_infancy` be ON by default, and how do we keep Phase 1, M0 and M2-A reproducible?

## Why we asked it
The M2-A decision rule said "adopt as default (a separate, reviewed change)". The user's instruction: "if it really helps then keep it on by default and make note of it and mention wherever it matters".

## Prediction
n/a (a decision following a pre-registered rule).

## Method (how)
Test-first: a failing test for the default, a test that the default pool equals the explicit flag-on pool at 50 steps and raises no alarm (RED before the flip), pin tests for `bench._phase1_cfg`; then the flip; full suite 432 passed. The historical entry points pin the old value (`bench._phase1_cfg`, `run_m0.A_KW`, and through it M2-A's `A_old`).

## Results (what came out)
Default ON. Documented in component cards 13 and the Phase-1 card, the handoff addenda (13f / 13g), the config and competence docstrings, both AMENDMENT files and the report boxes. Rule saved for future work: confirmed helpful fixes become the default, old entry points are pinned, and the change is noted everywhere it matters (`DECISIONS.md` D-09).

## Interpretation
A default is a claim about what the system is; it must be changed only on a pre-registered pass and with the reproducibility of earlier results protected.

## Caveats and what went wrong
A pin only protects entry points we know about; scripts that build `PhyloCfg` without the pin now get the new behaviour.

## Consequence
The same procedure was used for the batch vote (E-020).

## Evidence
- `research/moe_diligence_2026-10/README_default_flip.md`, `docs/COMPONENT_CARDS.md` (card 13), `research/growmoe_results/m0/AMENDMENT.md` (Note 2), `research/growmoe_results/m2a/AMENDMENT.md` (Note 1)

## Reproduce
`python -m pytest tests/growmoe/test_gm_m2a_arming.py -q`

# E-020 - Adopting the batch vote as the default, documenting it, and committing the GrowMoE track

- **ID:** E-020
- **Date:** 2026-10-12
- **Track:** G
- **Type:** decision
- **Status:** completed
- **Evidence grade:** A (follows E-019's pre-registered decision line)
- **Code / commit:** `growmoe/config.py`, `growmoe/bench.py`, `research/growmoe_results/m0/run_m0.py`, `tests/growmoe/test_gm_rv2a_vote.py`, `tests/growmoe/test_gm_rv2a_pins.py`; branch `track-g-growmoe`, commits `7b37e0a` (package and tests), `f881cb3` (research), `08c1a7a` (docs)
- **People:** Claude (implemented test-first); user: "yes commit it and start stage 2"

## Question
Is the batch vote ready to be the default, and is the work safe in version control?

## Why we asked it
The pre-registered decision line read "V1 PASS and every guard holds: adopt the plain vote as the default, test-first, with the historical entry points pinned", and the user's standing rule is to turn confirmed repairs on and document them. Separately, the whole GrowMoE track (about 245 files) had been untracked since 2026-10-04, i.e. not backed up.

## Prediction
n/a.

## Method (how)
Red first: the default-on test and a pin test for `run_m0.A_KW` failed for the stated reasons (default still False; no pin). Then the flip, the pins (`bench._phase1_cfg`, `run_m0.A_KW`), the full suite (475 passed, 170 s), then documentation (component card 14, handoff 13k and the config list, report Appendix F with the table copied by script, routing-research status, both AMENDMENT notes, the charter register, the hand-off state note) and a three-commit history on a new branch (the repository had been on `main` with one commit).

## Results (what came out)
`batch_vote` is ON by default; Phase 1, M0 and M2-A reproduce exactly because their entry points pin it OFF. No existing test needed changing. Scope of the commit: GrowMoE only; the geometry, Yggdrasil and VSA work, `pyproject.toml` and the edited ISHMN test remain uncommitted, deliberately.

## Interpretation
A default flip is a claim; it is made only on a pre-registered pass, with reproducibility protected and the change written down where a reader will meet it.

## Caveats and what went wrong
The register still showed Phase 1 as "VERIFIED" without qualification, which M0 had undercut; corrected in the same pass. Line-ending warnings (LF to CRLF) appeared on commit because of the Windows setting `core.autocrlf=true`; harmless but noisy.

## Consequence
Stage 2 handed to Gemini (E-021). This notebook created.

## Evidence
- `docs/COMPONENT_CARDS.md` - card 14; `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` - Appendix F; `docs/superpowers/handoff/SCOPE_CHARTER.md` - register

## Reproduce
`python -m pytest tests/growmoe -q` (475 passed); `git log --oneline` on branch `track-g-growmoe`.

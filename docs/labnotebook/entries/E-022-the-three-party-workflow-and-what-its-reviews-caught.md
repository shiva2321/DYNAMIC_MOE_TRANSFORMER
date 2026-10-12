# E-022 - The three-party workflow (user, Claude, Gemini) and what its reviews caught

- **ID:** E-022
- **Date:** 2026-10-04 to 2026-10-12
- **Track:** W
- **Type:** process
- **Status:** completed (ongoing practice)
- **Evidence grade:** B
- **Code / commit:** `docs/superpowers/handoff/*.md`, `GEMINI.md`
- **People:** all three

## Question
Does splitting the work (Gemini implements, Claude designs and reviews, the user decides and relays) produce results we can trust, and where does it fail?

## Why we asked it
The user wanted to conserve Claude's usage while keeping quality. Because the lab notebook is meant to support a publication, the process that produced each result is part of the evidence.

## Prediction
None. A process record.

## Method (how)
A hand-off file per track (goal, constraints, environment traps, exact API, measured decisions, round-by-round specification, report template); one round at a time; a Claude review after every round with a fixed checklist (tests read before code, red evidence plausible, no vacuous or seed-lucky assertions, thresholds backed by measured margins, forbidden things absent, deviations justified, suite re-run by Claude, generated files regenerated and compared); verdicts `APPROVE`, `CHANGES: 1) ...`, `ANSWER`. A scope charter with fences, drift rules and a component register.

## Results (what came out)
What the reviews caught (details in `FAILURES.md`): a novelty metric that was true by construction (F-05); result files and prose edited by hand (F-06); a comparison run at a different operating point than the benchmark's (F-07); a pre-registered 1-D test silently replaced by a 2-D one (F-09); a speedup defined against an over-trained baseline (F-08); a parameter count that double-counted a shared module (F-11); numbers in prose that contradicted tables (the Y4 table); a stop rule skipped (F-16). What the process did well: hash-frozen pre-registrations, byte-exact regeneration of reports, every default flip pinned for reproducibility, the audit that cleared the skipped-stop case in four tool calls.

## Interpretation
The implementer is fast and mostly right on code, and reliably wrong in a small number of characteristic ways that only independent regeneration catches. The review checklist is the control; "the suite passes" is not.

## Caveats and what went wrong
The reviewer is also fallible (F-11: my own review missed the double count). Stop rules written only in a hand-off file were not followed once; they now go in the first line of the kickoff message.

## Consequence
Charter drift rules 1-10; the audit recipe in `DECISIONS.md` D-14.

## Evidence
- `docs/superpowers/handoff/SCOPE_CHARTER.md`, `docs/superpowers/handoff/GROWMOE_HANDOFF.md`, `docs/superpowers/handoff/REVIEW_2026-10-05_R10_AND_Y1.md`, `docs/superpowers/handoff/ROUTING_V2_HANDOFF.md`

## Reproduce
n/a.

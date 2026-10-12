# E-012 - Diligence round: do the Phase 1 claims hold against fair baselines?

- **ID:** E-012
- **Date:** 2026-10-09
- **Track:** G
- **Type:** audit
- **Status:** completed
- **Evidence grade:** B
- **Code / commit:** scripts and raw outputs in `research/moe_diligence_2026-10/`; no repository code changed
- **People:** Claude; user asked "does dynamic MoE help at all for continual learning?"

## Question
Is the Phase 1 story ("the growing pool beats replay and stays cheap") true when every arm is tuned and a no-learning memory baseline is present?

## Why we asked it
The user wanted the honest state before deciding direction. A lesson from the geometry track (E-007: untuned single-setting comparisons give confident wrong answers) applied directly: Phase 1 had compared at one fixed training budget.

## Prediction
None; a diligence audit.

## Method (how)
Reproduce Phase 1 exactly (frozen settings q 0.02, alpha_spawn 0.01, m_min 32, `accept_then_logf`); then add kNN storing everything and at equal memory, nearest class mean, streaming LDA, and replay tuned over 1-50 steps; scale test over 5 / 15 / 30 permuted regimes; resource accounting; frontend check (hash vs frozen MiniLM); other agent's claims re-checked.

## Results (what came out)
- kNN storing everything beats the pool on its own streams: S2 95.60 vs A 87.04, S1 77.95 vs 69.81 (10 seeds); equal-memory kNN also wins; NCM 42.6 / 57.5, SLDA 61.2 / 60.6.
- Phase 1's "A beats replay by about 19 points" exists **only at 1 optimiser step per batch** (replay under-trained). At 5-20 steps replay ties or beats A (S1: replay 79.26 vs A 75.81 at 20 steps, intervals separated).
- The official S2 stream has 5 regimes, not 30. Scale test (3 seeds): replay (20 steps) 92.1 / 84.4 / 72.3 at 5 / 15 / 30 regimes; A(20) 91.2 / 87.5 / 82.0; A(1) 87.7 / 86.7 / 85.7; kNN flat at about 95. Supports the user's bet that growth helps as regimes multiply, but kNN is still ahead.
- Cost: at 30 regimes A uses 9.3 MB vs 6.0 MB of raw data; the gate scan costs about 38x the expert it selects; total about 0.84x of a full kNN lookup.
- Growth, merge, archive and revive never fire in 20 benchmark runs.
- Pool C's `summary()` double-counted the shared base (371,392 reported vs 120,640 unique): C3's "189 %" was really 80-100 %.
- Conformal floor: the smallest possible p-value is 1/(n+1); an expert with 32 stored scores at q 0.02 can never flag novelty.
- Frozen MiniLM instead of hashing for text / code in S1: A 69.81 -> 72.62.
- Pilots: with hidden-function labels, kNN reaches 21-26 % where a small trained network reaches 45-78 % from the same data (chance 12.5 %) - **lookup fails when the answer must be computed**; on a function-regime stream the growing pool beats the single replay net and kNN, with the gap growing with the number of regimes.

## Interpretation
The right mechanism, but our streams could not show that experts add value over memorising. That is a statement about the benchmark as much as about the system: tasks solvable by lookup do not need experts. The next test therefore had to be one where lookup cannot win (E-013).

## Caveats and what went wrong
My own October 5-6 review of Pool C had missed the parameter double count (`FAILURES.md` F-11). The closest published relatives (MaRS, SEMA, LLaVA-CMoE) were found only at this point, so "statistically controlled growth" is not novel.

## Consequence
User replies (same day): pause R12b / vertical growth ("let it sit"); frozen encoder only for realistic text streams later; episodic memory tier undecided, to be tested as an optional arm after M1; Claude does light rounds, Gemini only deep ones. Plan M0-M6 proposed; M0 + M1 run next (E-013).

## Evidence
- `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` - the report (sections 1-8, Appendices A-C)
- `research/moe_diligence_2026-10/spike_baselines_out.txt`, `research/moe_diligence_2026-10/spike_budget10_out.txt`, `research/moe_diligence_2026-10/spike_scale_out.txt`, `research/moe_diligence_2026-10/spike_efficiency_out.txt`, `research/moe_diligence_2026-10/spike_frontend_out.txt`, `research/moe_diligence_2026-10/spike_lookup_fails_out.txt`, `research/moe_diligence_2026-10/function_stream_3seeds_out.txt`

## Reproduce
Each `spike_*.py` in `research/moe_diligence_2026-10/` writes the `*_out.txt` beside it; run from the repository root.

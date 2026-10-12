# E-013 - M0 + M1: the pre-registered fair fight on streams where lookup cannot win (B1 FAILED)

- **ID:** E-013
- **Date:** 2026-10-09
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** A
- **Code / commit:** `growmoe/fairfight.py`, `growmoe/accounting.py`, `growmoe/memory_baselines.py`, `growmoe/streams_m1.py` (`function_regimes`); driver `research/growmoe_results/m0/run_m0.py` (tune / run / equivalence / report; resumable; `tuning.sha256` freezes the budgets)
- **People:** Claude (design, code, run); user approved ("sure go ahead")

## Question
Is the dynamic pool better than the best single network, and than simple memories, on streams where the answer must be computed (the "function-regime" streams F-15, F-30), and what does it cost?

## Why we asked it
E-012 left the user's bet open: the mechanism is right but the old streams could not show an advantage.

## Prediction
`PREREG.md` (section 8 written before any run): statements A1-A3, B1-B3, C1-C2 with expected outcomes and a **kill rule**: if B1 fails, stop claiming an accuracy advantage for the current pool and re-scope to compute saver plus abstention layer over an episodic memory. B1: the pool beats the best single network by >= +5.0 points with the 95 % interval excluding 0.

## Method (how)
10 test seeds; every arm's training budget tuned on dev seeds with `tune_steps` (grid 1 / 5 / 20 / 50, extended by x2.5 while the optimum sits at the edge); the best single network B2* tuned over width and buffer (grid extended by AMENDMENT #1 because the dev optimum sat on the corner: chosen width 4096, buffer 8192, 20 steps); kNN storing everything and at equal memory; an oracle-routed expert arm B4; measured multiply-adds and memory; a frozen MiniLM text stream S1-PRE.

## Results (what came out)
| ID | Statement | Verdict | Numbers |
|---|---|---|---|
| A1 | tuned replay is not worse than A on S2-5 / S1 | PASS | replay - A = +1.5 [+0.3, +2.7] / +6.7 [+3.3, +10.1] |
| A2 | kNN-all beats A by >= 5 | PARTIAL | +4.6 / +5.4 (S2-5 just under) |
| B1 | A beats best single net by >= 5 | **FAIL** | F-15 +4.1 [+2.9, +5.4]; F-30 +2.6 [+0.7, +4.5] |
| B2 | A beats kNN-all by >= 10 | PASS | +18.6 / +11.9 |
| B3 | oracle - A >= 5 at F-30 | PASS | +13.5 [+11.8, +15.3] |
| C1 | compute vs kNN | FAIL (i.e. a saving) | A uses 0.38x kNN's multiply-adds on S2-30 |
| C2 | frozen encoder helps | PASS | +3.8 [+0.7, +6.9] |

Oracle-routed experts (B4) beat the best single network by +6.6 / +11.3 / +16.2 points at 5 / 15 / 30 regimes with a quarter of the memory and about 30x fewer multiply-adds, so the structure is valuable; A realised only 6 % / 37 % / 16 % of that. Test-time routing was fine (92-99 % of held-out inputs reach their regime's expert) but A's experts were 10-15 points weaker per regime than oracle-trained ones. Gate scan = 83-96 % of A's compute. Growth fired in 3 of 70 A runs; merge, archive, dormant, prune never.

## Interpretation
The pre-registered kill rule fired for the **current pool**: not for the idea. The gap was in the pool's experts (data or recipe), not in the routing. Two diagnostics (dev ablation with fresh init, no replay, never-extend) did not close it.

## Caveats and what went wrong
M0's oracle B4 lacked A's replay mixing and so understated the oracle (48.5 % vs 52.7 %, found in E-014). The result stands as pre-registered; the interpretation of "weak recipe" turned out to be a lifecycle bug.

## Consequence
Open decision for the user (accept the re-scope, or one time-boxed diagnostic round). The user chose to improve routing ("we need a better routing algorithm"), leading to E-014.

## Evidence
- `research/growmoe_results/m0/PREREG.md`, `research/growmoe_results/m0/AMENDMENT.md`, `research/growmoe_results/m0/RESULTS_M0.md`, `research/growmoe_results/m0/m0_runs.json`, `research/growmoe_results/m0/tuning.json`, `research/growmoe_results/m0/tuning.sha256`
- `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` - Appendix D (D.1-D.6)

## Reproduce
`python research/growmoe_results/m0/run_m0.py report` regenerates `RESULTS_M0.md` from the stored per-seed runs; `run_m0.py run` re-runs the seeds (`A_KW` pins the pre-repair behaviour).

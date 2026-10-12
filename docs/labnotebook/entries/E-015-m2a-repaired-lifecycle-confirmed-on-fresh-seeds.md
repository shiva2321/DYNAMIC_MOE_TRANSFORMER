# E-015 - M2-A: the repaired lifecycle beats the best single network on fresh seeds (7 of 7 statements PASS)

- **ID:** E-015
- **Date:** 2026-10-09 to 2026-10-10
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** A
- **Code / commit:** `PhyloCfg.arm_monitor_after_infancy`, `PhyloMoE._forget_competence_history`, `growmoe/fairfight.py` (`m2a_verdicts`, `m2a_decision`); tests `tests/growmoe/test_gm_m2a_arming.py` (9), `test_gm_m2a_verdicts.py` (11); driver `research/growmoe_results/m2a/run_m2a.py`; configuration frozen sha256 12df561bc8e17dca
- **People:** Claude (implemented test-first, pre-registered, ran); user approved option A

## Question
Does keeping an infant's competence monitor silent until its infancy is over (E-014) make the pool beat the best single network by the M0 margin, on seeds never used before, without hurting the streams that need the alarm?

## Why we asked it
E-014's fix was found on dev seeds with many variants tried; it had to be confirmed once, under a written protocol.

## Prediction
`PREREG.md`: P1 F-15 and F-30 mean paired A_new - B2* >= +5.0 points with the 95 % interval excluding 0 (the B1 rule); P2 A_new - A_old >= +3.0 (the repair itself moves the pool); G1-G4 S1 / S2-5 / S2-30 / S4 not worse than -3.0 points and not significantly negative; G5 on S7 (label remap) A_new raises >= 3 alarms per run. P1 FAIL = a second kill. Fresh seeds 10-19; budgets re-tuned on dev seeds (func 5, perm 20, types 125 steps); config frozen by hash before the first seed.

## Method (how)
70 tasks, `python research/growmoe_results/m2a/run_m2a.py run`, then `report` writes `RESULTS_M2A.md` from the per-seed records; M0's machinery imported unchanged.

## Results (what came out)
| ID | Result |
|---|---|
| P1 | **PASS**: A_new - B2*: F-15 +10.0 [+8.0, +12.0], F-30 +11.6 [+9.5, +13.8] (bar +5) |
| P2 | **PASS**: A_new - A_old +7.6 [+5.1, +10.1] / +7.2 [+4.9, +9.4] |
| G1-G4 | **PASS**: S1 +3.9 [+0.6, +7.2], S2-5 +1.0 [-0.8, +2.7], S2-30 +2.3 [-1.5, +6.1], S4 -0.4 [-4.3, +3.6] |
| G5 | **PASS**: mean alarms per run on S7 = 4.10 |

A_new reaches F-15 46.7 % vs oracle-routed experts 47.6 %, and F-30 42.6 % vs 46.8 % (A_old: 35.5 % on F-30). Resources (descriptive): at F-30 the pool uses 11.6 MB vs 8.2 MB for the best single network (1.41x) and 1,000 vs 1,050 kMACs per prediction, with the gate 97 % of the multiply-adds, while a perfectly routed expert needs 34 kMACs.

## Interpretation
The M0 shortfall was a lifecycle bug, not a weak idea: with the repair the pool beats the best single network by about 10 points on function streams. **The accuracy half of the bet holds on this stream family; the efficiency half does not yet:** the pool is as expensive as the single network because the gate scans everything.

## Caveats and what went wrong
One synthetic stream family carries the accuracy claim. Three guards have wide intervals, so small regressions cannot be excluded. The S1 budget (125 steps) was chosen on three noisy dev seeds. The repair was found after the M0 verdict: M0's B1 FAIL for the frozen pool stands as a record.

## Consequence
Adoption as the default (E-016) and routing v2 (E-017) for the efficiency gap and for overlap.

## Evidence
- `research/growmoe_results/m2a/PREREG.md`, `research/growmoe_results/m2a/AMENDMENT.md`, `research/growmoe_results/m2a/RESULTS_M2A.md`, `research/growmoe_results/m2a/m2a_runs.json`, `research/growmoe_results/m2a/tuning_m2a.json`, `research/growmoe_results/m2a/tuning_m2a.sha256`
- `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` - Appendix E

## Reproduce
`python research/growmoe_results/m2a/run_m2a.py report` regenerates the results from the stored runs; `python -m pytest tests/growmoe/test_gm_m2a_arming.py tests/growmoe/test_gm_m2a_verdicts.py -q` for the tests.

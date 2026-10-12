# Amendments to PREREG.md

Each entry is dated and says what had already been seen. PREREG.md itself is not edited after the first run (section 8 of it was written before any run).

## Amendment 1 (2026-10-09, before any test seed 0-9 was run): extend the B2* size grid

**What had been seen.** Dev-seed tuning (seeds 100-102, stream `F-15`) of the six pre-registered B2* configurations had finished. The best single replay network was the largest configuration in the grid: width 2048, buffer 8192, 20 steps per batch, dev accuracy 36.2 % (width 2048 with buffer 2048: 31.1 %; width 512 with buffer 8192: 32.9 %). Every row of the dev tuning table was interior in `inner_steps`, but the best (width, buffer) pair sat on the corner of the grid. The same dev run gave A 40.1 % (50 steps), so on dev seeds A led this baseline by 3.9 points, below the +5.0 point threshold of statement B1. No test seed had been run.

**Rule applied.** The charter's "no edge optimum" principle, already used for the step grid in PREREG section 4: a baseline whose best setting is on the edge of its search has not been given its fair size.

**Change.** B2* is searched over widths {128, 512, 2048, 4096} x buffers {2048, 8192, 32768} x `inner_steps` {1, 5, 20, 50} (step extension rule unchanged). A buffer of 32768 holds every training example of `F-30` (25,600), so it never evicts. The six configurations already tuned are kept as they are.

**Stopping rule, fixed now.** One extension only. If the best configuration is again the largest corner (width 4096, buffer 32768), the report says so, states its memory as a multiple of A's, and no further extension is made.

**Direction of the effect.** The extension can only make the baseline stronger and statement B1 harder to pass. It was decided from the edge rule, before the extended configurations were run.


## Note 2 (2026-10-10, after the M2-A result): the code default changed; M0 stays reproducible

Not an amendment of any statement. `PhyloCfg.arm_monitor_after_infancy` became ON by default after the M2-A confirmatory test (user decision, 2026-10-10). M0 ran with the old behaviour. To keep every M0 file reproducible, `run_m0.py` now pins `arm_monitor_after_infancy=False` in `A_KW` (the dict all its pool constructions and the `diagnose_*.py` scripts use), and `growmoe.bench.run_phase1` pins it through `_phase1_cfg`. The script hash in `RESULTS_M0.md` therefore predates this edit; the results do not change.

## Note 3 (2026-10-12): `batch_vote` defaulted to ON afterwards; M0 stays reproducible

Not an amendment of any statement. `PhyloCfg.batch_vote` became ON by default after the RV2-A confirmatory test (all 11 statements PASS). `run_m0.A_KW` pins `batch_vote=False` next to `arm_monitor_after_infancy=False`, so M0 and its diagnostics run exactly as recorded (`tests/growmoe/test_gm_rv2a_pins.py`).

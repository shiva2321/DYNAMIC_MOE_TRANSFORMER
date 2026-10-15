# E-033 - Stage 4 build: the router, the transfer test and the growth controller behind flags, and what the merge audit found

- **ID:** E-033
- **Date:** 2026-10-15
- **Track:** G
- **Type:** process (implementation, with one audit)
- **Status:** completed (tasks 1 to 4 of the plan; the experiments are tasks 5 to 8)
- **Evidence grade:** B (the tests are exact; the merge audit is script-generated on 4 seeds, exploratory)
- **Code / commit:** 12f570d (flags and validation), 906e954 (router), 0724b90 (transfer test), a6e1380 (growth controller); plan `docs/superpowers/plans/2026-10-15-routing-v2-stage4.md`, spec `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md`
- **People:** Claude built and tested it at the user's request ("approved, write the plan and start building"); no fresh reviewer yet (plan step 8.6)

## Question
Can the three repairs of the stage-4 spec be built so that (a) with every flag off the pool is bit-identical to the old one and (b) each repair does what its own tests say? And does the merge routine, which the transfer rule relies on to clean up a wrongly founded expert, work at all?

## Why we asked it
The user approved the written spec on 2026-10-15. The spec (section 5.5, open question Q5) notes that merge has never fired in a benchmark and says it must be audited before anything is adopted, because the transfer rule is built to favour a new expert over a wrong merge.

## Prediction (written before the runs)
Built to the plan, flags off bit-identical (statement S4-9). For the audit the plan expected merge to clean at least 2 of 4 forced duplicates and said what to do otherwise.

## Method (how)
Plan tasks 1 to 4, test first (each test file was run and seen to fail before the code existed), one commit per task. The audit (`research/growmoe_results/stage4_spikes/audit_merge.py`): a pool learns one linear teacher, the interference probe is forbidden to accept an extend (`delta_forget = -1`), a cluster of the SAME function founds a duplicate, ten more batches settle it, then the numbers inside the merge acceptance test are recomputed as `_attempt_merges` computes them and the routine itself is run. Seeds 0 to 3.

## Results
- Full suite after task 4: 801 passed, 2 expected failures (the merge audit) in 344 s (692 before this stage). The golden check with all stage-4 flags off: bit-identical (all 20 fingerprints) after task 1 and after task 4; the quick check after every task.
- New tests: configuration 36; router 12 pure + 15 in the pool; transfer 7 pure + 14 in the pool (+ 2 expected failures, the audit); growth 10 pure + 15 in the pool.
- Pools pickled before stage 4 (the saved operator-by-operator arithmetic pools of E-030) still load and answer. On the saved 4-expert pool the product router selects LDA (cross-validated owner accuracy: density 0.875, LDA 1.000, softmax 1.000, 5-NN 0.836), as spike A3 predicted for the stored prototypes.
- **Merge audit** (`research/growmoe_results/stage4_spikes/audit_merge.json`; tolerance `delta_merge` = 0.02; overlap needed 0.2):

| seed | mutual coverage (j on i / i on j) | accuracy drop of the union-trained clone, expert i | expert j | expert accuracy on its own replay (i / j) | merged |
|---|---|---|---|---|---|
| 0 | 0.50 / 0.49 | +0.094 | -0.148 | 0.94 / 0.69 | no |
| 1 | 0.48 / 0.52 | +0.062 | +0.039 | 0.90 / 0.87 | no |
| 2 | 0.47 / 0.47 | +0.078 | +0.086 | 0.89 / 0.89 | no |
| 3 | 0.44 / 0.47 | +0.008 | +0.083 | 0.88 / 0.92 | no |

0 of 4 forced duplicates were merged.

## Interpretation (what we understood, and on what basis)
Plain words. The three repairs are in the pool and switched off. With the switches off nothing changed; with them on, each one passes its own tests. The merge routine, however, cannot do the one job the transfer rule leaves to it. It first checks that two experts cover each other's inputs (they do, 0.44 to 0.52 against 0.2) and then asks whether one network trained on both experts' replayed samples loses nothing on each expert's own held-out half. But the experts have memorised their replay (0.88 to 0.94 accuracy on it) while the clone never trained on the held-out half, so it loses 6 to 9 points against a tolerance of 2. This is the same kind of flaw E-030 found in the spawn probe (a decision taken on memorised samples), mirrored: the probe is biased towards extending, the merge test against merging. Consequence: a wrongly founded expert stays for ever.
Technical statement: measured accuracy drops of a union-trained clone on validation halves of the experts' reservoirs, four seeds, one function, hidden 32; not measured: other widths, multi-function experts, longer settling.

## Caveats and what went wrong
- The first version of the pure transfer test stopped permuting as soon as the answer "not rejected" was certain and then reported a lower bound as p; a test that expected the full p exposed it. The early stop saves about ten milliseconds once per cluster, so it was dropped and the full exact p is reported (refinement R-5).
- A first guard test wrecked the weights after a growth; that also trips the competence monitor, whose alarm skips the training step and with it the guard's bookkeeping. The guard rule is now tested directly (the loss "before" is set unreachably low).
- The synthetic data of the tests were chosen by a scratch search before the tests were written (anisotropic noise makes LDA clearly beat the density rule; two opposite clusters per expert make density clearly beat LDA); the statistical growth tests calibrate their own threshold from the measured gaps of a starved and an adequate expert, so they test the mechanism, not a constant.
- Not yet done: the mutation spot-check (task 6), a fresh reviewer (task 8), and every performance question: nothing here says the repairs help.

## Consequence
Dev runs next (task 5). The merge finding goes into FAILURES (F-28) and the open questions (Q-19). It does not block stage 4, but it changes how the transfer rule is judged: a false spawn cannot be repaired, so the dev review must look at expert counts against regimes on every stream (guard S4-5 already limits them to 1.2 times S3's). A repair of merge (test on fresh stream samples or honest losses instead of the memorised replay) is a candidate for a later stage, not part of this one.

## Evidence
- `docs/superpowers/plans/2026-10-15-routing-v2-stage4.md`
- `tests/growmoe/test_gm_rv4_config.py`, `test_gm_rv4_router.py`, `test_gm_rv4_router_pool.py`, `test_gm_rv4_transfer.py`, `test_gm_rv4_transfer_pool.py`, `test_gm_rv4_growth.py`, `test_gm_rv4_growth_pool.py`
- `research/growmoe_results/stage4_spikes/audit_merge.py`, `research/growmoe_results/stage4_spikes/audit_merge.json`

## Reproduce
```
python -m pytest tests/growmoe -q
python research/growmoe_results/rv2c/golden.py check
python research/growmoe_results/stage4_spikes/audit_merge.py
```

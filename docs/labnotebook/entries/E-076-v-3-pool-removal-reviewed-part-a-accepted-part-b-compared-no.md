# E-076 - V-3 pool removal reviewed: Part A accepted, Part B compared nothing, copied the pool wrongly and missed the extend path

- **ID:** E-076
- **Date:** 2026-11-04
- **Track:** W
- **Type:** review
- **Status:** completed (Part A accepted; Part B not accepted, fix round V-4 released)
- **Evidence grade:** B for the reproductions (separate processes, development seeds 800-802); the numbers are development-seed measurements, not results
- **Code / commit:** `v1/downdate.py`, `v1/pool_removal.py`, `v1/run_heads_experiments.py`, `v1/run_pool_experiments.py`, `tests/v1/test_v1.py` (Gemini); reviewer scripts and outputs in `results/v1/review/`
- **People:** Gemini built (Antigravity, relayed by the user); Claude reviewed and re-measured

## Question
Job V-3 (probe P2, "is removal exact?"; spec `docs/superpowers/handoff/V1_REMOVAL_SPEC.md`): are the exact downdating of the closed-form heads and the removal operation with a certificate for the V0 pool correct, and are the numbers of the audit what the code measured?

## Why we asked it
The user wants knowledge that is unnecessary or proven wrong to be removable, and our design suggests removal can be exact where components are isolated. Five earlier Gemini reports said more than the work showed (F-108, F-110, F-112, F-113, F-114), so nothing is taken from the report.

## Prediction (written BEFORE the numbers, if there was one)
None written. Expectation from the earlier reviews: a sound core and several overstatements.

## Method (how)
1. Fence: `git status` and the diff: only new files in `v1/`, `tests/v1/`, `results/v1/` and the four handoff files; the one modified file (`results/m1/smoke/harness_check_N10_seeds100.json`) was modified before the job.
2. Read all of `v1/downdate.py`, `v1/pool_removal.py`, both runners, the test file, the result file and the audit; grepped every cited line range and name against the files.
3. Reran `pytest tests/v1` (6 passed in 4.1 s) and one Part A experiment (R2 seed 801) in a separate process into a scratch directory.
4. Wrote an independent re-measurement of the pool part (`results/v1/review/review_v1_pool.py`): trains the full V0 pool and the never-trained pool in one process each, calls Gemini's `remove_labels`, and compares experts by the index of the founding batch in the original stream, comparing the module parameters by name through `dict(named_parameters())`, the per-expert `LdaHead` (n, s, S, total), the reservoir, the signature, and the routing by founding batch. Runs: seed 801 (R3, R1), 802 (R4, R3), 800 (R1, R4), plus clock-aligned and clock-and-id-aligned never-trained controls for R3 seed 802, R4 seed 800, R1 seed 800. Pinned to logical processors 8 and 10.
5. Diagnostics: `chk_params.py` (parameter names), `diag_v1.py` (runtime inventory of structures, router state), `diag_router.py` (retained accuracy with and without a refit router in the copy).

## Results (what came out)
**Part A (closed-form heads), held.** The R2 seed 801 rerun equals Gemini's record in every field except a 5e-16 difference in the solved ridge weights (floating-point noise of the solve). Across the 12 records: ClassMean, KnnMemory (capacity None), MahalanobisProto and ExactLda are bit-identical to the never-trained model (max difference 0, argmax agreement 1.0); RandomFeatureRidge is exact to at most 7.45e-8 (statistics) and 1.08e-8 (weights) when the removed rows are supplied (its Gram matrix is pooled); StreamingLDA is not exact (statistic difference up to 5.0e-4, weight difference up to 1.04, probability difference up to 0.082, argmax agreement down to 0.988). Caveats: the exactness of the class-separable heads is by construction (sufficient statistics add up per class), the StreamingLDA "downdate" leaves the covariance untouched (`remove_classes`, branch for `StreamingLDA`), so the measurement says "cannot be downdated from its state", not more.
**Part B, the exposure part reproduces.** The dropped experts and the collateral counts of my seven reruns equal Gemini's records exactly (R3 s801: expert 18, 0 collateral; R1 s801: expert 12, 15; R4 s802: all 19 experts, 123; R3 s802: experts 2 and 3, 0; R1 s800: expert 5, 18; R4 s800: experts 12 and 13, 0).
**Part B, defects found (all reproduced by the scripts in `results/v1/review/`):**
1. The weight comparison compared nothing. `named_parameters()` names are dotted (`blocks.0.w1`, ten per expert, `chk_params_output.txt`); `hasattr(module, "blocks.0.w1")` is False for all, so `max_weight_diff` stayed 0.0 in all 12 records. Corrected comparison (founding-batch matching, ten parameters per expert): R3 s801 18 of 18 retained experts bit-identical to the never-trained pool; R1 s800 17 of 19; R4 s800 11 of 17; R3 s802 only 2 of 17 (max parameter difference 1.0 to 1.7). The experts that diverge are exactly those whose id is shifted in the never-trained pool after the dropped expert (R3 s802: dropped 2 and 3, every expert from id 4 on differs; R4 s800: ids up to 10 identical), plus a few whose data path changed (R1 s800 experts 6 and 10, R4 s800 expert 11). A clock-aligned never-trained pool did not change the counts; an id-aligned one raised R1 s800 to 17 of 19 and did not apply in the other two cases. So the pool is bit-exact for the experts whose history coincides, and no statement beyond that is supported.
2. Wrong matching: `birth_step` is the pool clock, shifted by the batches the never-trained pool skips (14 to 22 whole batches in R3).
3. "Routing agreement" compared expert ids; in the never-trained pool the ids are renumbered. R3 s802: 0.0891 by id, 0.6925 by founding batch; R4 s800: 0.5551 against 0.7067. The audit's explanation (routing topology depends on arrival order) is built on the artifact.
4. The copy is not a copy. `m1/arms.py:46-80` installs instance-level wrappers (`observe`, `train_expert`, `_refit_router`); `copy.deepcopy(arm)` keeps those function objects bound to the original pool, so `remove_labels` refit the ORIGINAL pool's router and left the copy's `_route.eids` at `None` (`diag_router_s801_R1_output.txt`). The check `pool_router_stale_eids` is guarded by `eids is not None` and so reported 0 vacuously. P_removed was evaluated with an unfitted router: retained accuracy 0.6088 against 0.6233 with the copy's router refit (R1 s801, +1.45 points). Doing two removals from one trained pool also left a stale id in the next copy (`clean=False` in my multi-removal runs: an artifact of the same bug).
5. An exposure path is missing. In R3 s801 the retained root expert 0 holds 44 signature prototypes with cosine at least 0.9999 to removed samples although its weights, LDA head and replay equal the never-trained expert 0 and the tracker saw no training on a removed label; its prototypes differ from the never-trained expert's by up to 0.49. The path is the extend step (`growmoe/pools/phylo.py:1022-1033`, also `:232`, `:774`, `:1202`): `parent_rec.meta["sig"].update(x_taken)` without a training step. Gemini's explanation ("recompute prototypes from the purged reservoir") cannot work because the reservoir does not hold them. Merges (`phylo.py:366-484`) and the child initialisation (`init_mode` default `shrink_perturb`, `growmoe/config.py:62`) are further paths the tracker does not follow through events (no merge happened in the seeds run here).
6. The scan does not cover the structures it should: runtime inventory (`diag_v1_s801_output.txt`) shows expert keys `sig monitor recent infant_calls_seen loss_hist policy lda windows ref gap supersedes` and pool attributes `_set _infant gate_stats skip_stats` etc.; `SourceWindows` (raw rows), `ref`, `recent`, `loss_hist`, `_set`, the optimiser state were not looked at. The `classification` strings of `scan_traces` are constants.
7. R4 seed 802: all 19 experts dropped (the root expert 0 absorbed the removed classes by routing, every other expert descends from it, children hold a scaled copy of the parent's weights), "certified_clean" is true for an empty pool, retained accuracy 0.0029.
8. The tests use a dummy pool (a fake `DummyPool`), the "repeat is skipped" test writes a JSON and reads it back without calling the runner, and the signature trace is never planted.
9. The audit: 11 of 17 line ranges are wrong (ExactLda cited at `v1/downdate.py:26-107`, actual 37-142; `Reservoir` is in `growmoe/replay.py`, not in the pool base file; ranges past the end of `router.py`, `memory_baselines.py`, `baselines_extra.py`); typed claims with no measurement ("accuracy back to >62%", "identical computational scaling", "float32 precision" for a float64 Gram matrix); Recommendation 2 (re-anchor the children to the grandparent and retrain their heads) is unsound (the children keep the weights inherited from the exposed parent, the certificate would claim what is false, and it contradicts the user's rule "no eviction without certified redundancy"); Recommendation 3 (per-class Gram matrices) would cost C x D x D x 8 B = 67 GB at D = 8192, and ExactLda (Recommendation 1) holds 16.4 MB (C=125, d=128) against 131 kB for the pooled covariance, which "identical computational scaling" hides; "exact" is given to buffers that were never exposed (nothing to purge) or whose purged form is smaller than the never-trained one.
**What held in Gemini's work:** all numbers in the result table equal the stored JSON; the dropped sets and collateral counts reproduce; the scan found the real signature trace and the certificate said not clean in that case; the honest negative results (the cascade, the trace) were reported.

## Interpretation (what we understood, and on what basis)
Plain words: for the class-separable heads removal is exact because the statistics add up per class; the one pooled quantity (a shared Gram or scatter matrix) needs the removed rows back, which a deletion request normally carries. For the pool the exposure logic is right in most cases (the dropped sets reproduce) and the experts that were never touched stay bit-identical for as long as their history matches that of a never-trained pool; but the measurement that was supposed to show this compared nothing, the removed pool was evaluated with an unfitted router, and one exposure path (the extend step that writes into the parent's signature) is invisible to the tracker. The practical lesson of the numbers is not yet that removal is exact: dropping one expert for one class costs 15 to 18 other classes (collateral), and the root expert can make the whole pool fall (R4 s802). Measured: everything marked rerun or measured. Not measured: the size of the router effect beyond R1 s801; the cost of scratch-initialised children; whether the id-keyed seeds explain all diverged experts.

## Caveats and what went wrong
- My re-measurement shares Gemini's `remove_labels` and tracker (it re-measures the comparison layer, not the exposure logic); the id-aligned control applied in one of three cases.
- My review runs ran on logical processors 8 and 10 for about 25 minutes, while the CPU timed pass ran on 0 and 2 (seeds 811 and 812 and the beginning of 813); timed cells that finished with an outlier in this window may be affected; the wait-for-quiet launcher of seeds 813 and 814 waited for my runs to end.
- The spec I wrote did not say "compare the parameters by name" or "use an honest copy" (the pool's instance-level wrappers are not obvious); the review caught both before anything was shown as a claim.

## Consequence
- D-103: Part A accepted; Part B returned to Gemini as fix round V-4 (`docs/superpowers/handoff/V1_FIX_SPEC.md`, kickoff `KICKOFF_V1_FIX_2026-11-04.md`); the audit is rewritten in that round.
- F-115, C-206, C-207. Q-86 (what removal should do when one expert holds many classes).
- Next: the user hands the V-4 prompt to Gemini; Claude reviews.

## Evidence
- `docs/superpowers/handoff/V1_REMOVAL_RESULT.md`, `P2_REMOVAL_AUDIT.md`, `V1_REMOVAL_TEST_OUTPUT.txt`, `V1_REMOVAL_COMMANDS.txt` - Gemini's report, audit, raw test output, command log
- `results/v1/heads_r*_s*.json`, `results/v1/pool_r*_s*.json` - Gemini's 24 records
- `results/v1/review/` - reviewer scripts and outputs (`review_v1_pool.py`, `review_v1_pool_out.json`, the two aligned variants, `heads_r2_s801_rerun.json`, `chk_params_output.txt`, `diag_router_s801_R1_output.txt`, `diag_v1_s801_output.txt`)
- `docs/superpowers/handoff/V1_FIX_SPEC.md` - the fix round

## Reproduce
```
python -m pytest tests/v1 -q
python results/v1/review/chk_params.py
python results/v1/review/diag_router.py 801 R1
python results/v1/review/review_v1_pool.py OUT.json 802:R3 800:R4
```

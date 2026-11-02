# E-036 - Stage 4.1 - a full pool says so and shares instead of dropping, and the user's decisions on the merge options

- **ID:** E-036
- **Date:** 2026-10-16
- **Track:** G
- **Type:** design (with the build and its tests)
- **Status:** completed; merged into `track-g-growmoe` on 2026-10-17 (fast-forward to ba0e2e4) after the stage-4 dev review and before the freeze, so that the dev queues kept their code stamp while they ran
- **Evidence grade:** A for the tests and the flag-off identity (script-generated, run by me); B for the sizes quoted from E-034
- **Code / commit:** branch `stage4-1-cap-and-telemetry` (worktree `D:/shm_stage41`), commit 9fc4084; spec docs/superpowers/specs/2026-10-16-stage4-1-cap-and-telemetry.md (on the branch)
- **People:** the user decided the policy; Claude designed, built and tested it; not reviewed by anyone else

## Question
What should the pool do when `max_experts` is reached and a region it has never seen arrives, and what should it tell us about the other two silent failures found in E-034 (twin experts, the infant window)?

## Why we asked it
E-034 found that the shipped cap is silent (no event, no counter, no warning) and that dropping the cluster costs a lot once the pool is full (the regimes met after it fills are answered at chance). The user read the options note (`research/GROWMOE_MERGE_RESEARCH.md`) and answered the five open decisions on 2026-10-16.

## Prediction (written BEFORE the numbers, if there was one)
None for the build. The test of the build is the flag-off identity: with the cap not reached the pool must be bit-identical to the one before (the 20 fingerprints of `research/growmoe_results/rv2c/golden.py`).

## Method (how)
The user's answers (verbatim in docs/superpowers/specs/2026-10-16-stage4-1-cap-and-telemetry.md (on the branch), section 1):
1. Full pool: "no dropping unless necessary because sometimes some knowledge and things are important but they are used/activated very rarely too".
2. Repair evidence: left to me. 3. Answer-time averaging: yes, once validated, say when unsure, do not hallucinate, take other experts' opinions if needed. 4. Infant window: fix it if a concrete reliable fix exists, otherwise document. 5. Sequence and head start: left to me.

What I decided on the user's behalf, and why:
- **Full pool = share, never evict.** Dropping lost 5-7 points against sharing in all three measured settings (E-034: 56.5 / 55.6 / 59.0 against 50.7 / 48.3 / 53.4 %). Eviction of the most redundant expert was better still (66.1 / 64.6) when the shortage was caused by twins, but signature overlap alone cannot tell a twin from a rarely used expert with a different function on the same inputs, and the user wants rare knowledge kept. So eviction waits until redundancy can be certified with honest evidence (stage 5).
- **Repair evidence = free live-traffic rival tracking** (stage 5, own pre-registration): it needs no withheld labels and removed 11 of 12 false spawns from 15 batches per regime on (E-034).
- **Answer-time averaging** goes to stage 5 and must be validated on a real stream first (it changes nothing on a pool without twins, C-72).
- **Sequence:** this stage is behaviour-neutral until the pool is full, so it goes in before the freeze of the confirmatory run, built on its own branch so the running dev queue (cells count only for the code stamp that made them) is not disturbed.
- **Head start:** the dev1 cells stay as they are, the function-regime cells are run again without the head start (`run_s3_unaliased.py`, `cells_s3u/`), the shootout is re-run without it (E-038), and the confirmatory run will use a pool seed offset.

Built (tests first, tests/growmoe/test_gm_s41_cap.py (on the branch), 10 tests): `PhyloCfg.cap_policy` ("share" default or "drop"), events `cap_reached` / `cap_share` / `cap_blocked`, one `RuntimeWarning` per pool the first time the cap acts, `summary()["cap"]`, `summary()["infant"]` (read-only counters of what the x4 infant relaxation lets through), `redundancy_report()` (read-only pair list with signature overlap and prediction agreement).

## Results (what came out)
- 10 new tests pass; the two older tests that used `max_experts=1` as a device ("novel batches cannot spawn") now pin `cap_policy="drop"` (`test_gm_rv3_pool.py`, `test_gm_rv3_confirm.py`); they fail with the new default because the cluster is shared instead of dropped, which is the intended change.
- Flag-off identity: `python research/growmoe_results/rv2c/golden.py check` -> "bit-identical" (all 20 fingerprints), on the code of this stage with the twin-fix flag of E-037 present and OFF.
- Full suite on the branch (`python -m pytest tests`): 1159 passed, 1 skipped, 2 expected failures (strict xfail of the merge audit, F-28). 27 tests outside GrowMoE fail with `FileNotFoundError: Could not find yggdrasil_engine.dll` (test_dynamic_moe, test_geometry_consistency, test_geometry_layers, test_gpu_kernel_bridge, yggformer/test_yf_kernels): the compiled Rust kernel is a git-ignored build output that exists in the main tree (the directory rust_yggdrasil/target/release, which the user deleted in the data cleanup of 2026-11-02; `cargo build --release` in `rust_yggdrasil` rebuilds it) and not in the worktree; they were not run in the main tree.
- No stored result reached 256 experts (the largest pool, N = 100 regimes, ends with 100; no `n_experts` field of any stored JSON has three digits), so no stored number depends on the cap.

## Interpretation (what we understood, and on what basis)
Plain words. A full pool used to forget silently; now it says so once, shares the new region with its nearest expert, and keeps counting. Sharing is a compromise when the new region's labels conflict with that expert's, and the counters show when it happens. Measured: the policies' sizes (E-034). Assumed: that "share" stays better than "drop" on streams other than the function-regime bench (not tested).

## Caveats and what went wrong
- The share path adds the cluster to the signature and then the training loop adds the same samples again: the cluster enters the signature twice, exactly like the shipped extend paths. That is the defect of E-037, found while testing the infant counter; it is fixed by the optional flag `sig_dedupe`, which is OFF.
- The infant counters are exact for the exact gate and upper bounds when `gate_shortlist > 0`.
- My first test of the share policy compared `sig.n` before and after; the signature was already full (256 prototypes) so `n` could not grow. The test now compares `seen` and the p-value of fresh inputs of the new region.

## Consequence
Merged on 2026-10-17 (E-039). The historical single-network baseline of the M5 runs pins `cap_policy="drop"`. Stage 5 (separate pre-registration): rival-tracking repair, certified-redundancy eviction, answer-time averaging. E-037 (the twin signature) is the infant-window answer.

## Evidence
- docs/superpowers/specs/2026-10-16-stage4-1-cap-and-telemetry.md (on the branch)
- tests/growmoe/test_gm_s41_cap.py (on the branch) (on the branch)
- `research/GROWMOE_MERGE_RESEARCH.md` (the options and the measured sizes)

## Reproduce
```
cd D:/shm_stage41            # worktree of branch stage4-1-cap-and-telemetry
python -m pytest tests/growmoe/test_gm_s41_cap.py -q
python research/growmoe_results/rv2c/golden.py check
```

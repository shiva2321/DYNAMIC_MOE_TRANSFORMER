# Routing v2 Stage 3 Implementation Plan (batch-level gate while learning, group answering while predicting)

> **For agentic workers:** the executor is Gemini (the user's three-party workflow: Claude designs and reviews, Gemini installs, tests and runs, the user mediates). Steps use checkboxes. The code is NOT retyped from this document: every file was written and validated by Claude in a shadow copy of the repository and is installed verbatim, stage by stage, tests first, by `docs/superpowers/plans/2026-10-13-routing-v2-stage3-assets/apply_stage3.py`. The message to send is `docs/superpowers/handoff/KICKOFF_RV3_BUILD_TRACE_DEV_2026-10-13.md`.

**Goal:** a pre-registered, reproducible test of two new routing levels for PhyloMoE: 3a an exact permutation test of each batch's accepted samples against per-expert source windows (decides known/new as a SET and who trains the batch), 3b answering a group of inputs by the expert most of them chose, with a per-input veto. Both flags stay OFF by default until their pre-registered run passes.

**Architecture:** pure modules (`settest`, `sources`, `setgate`, `groupvote`, `groupeval`, `streams_order`) plus small hooks in `PhyloMoE` (`observe`: after the batch vote and before the probation block; the training loop; the merge; new `predict_grouped`; `summary`), `PhyloCfg` fields with validation, `accounting` (windows are stored vectors), verdict functions in `fairfight.py`, and a runner `research/growmoe_results/rv2c/run_rv2c.py` (trace, dev, prereg, run, m5, report).

**Tech stack:** Python 3.12, torch (CPU), numpy, pytest; reuses `fairfight.run_fast/run_record`, `metrics.paired_diff`, `run_m0._parallel/freeze_check`.

**Spec:** `docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md` and `...-AMENDMENTS.md` (16 amendments; none changes a statement, bar or decision rule. A-15 persistence (`set_gate_confirm`, default 2) and A-16 group veto level (`group_veto_q`, default 0.2) were added on 2026-10-14, after the first dev run, with the user's approval; they change the mechanism and are installed by the two extra stages `F_tests` and `F_code` of `apply_stage3.py` (assets in `files_v2/` and `edits_v2/`; `F_tests` replaces only the stage-3 pool tests this plan installed, no pre-existing test). The first dev run and trace are superseded by a rerun under the final mechanism: message `docs/superpowers/handoff/KICKOFF_RV3_AMENDED_BUILD_DEV_2026-10-14.md`.)

## Global constraints (verbatim from the spec's standing rules and the project's)

- Flags off => bit-identical to today (statement S3-9): enforced by golden fingerprints recorded BEFORE any edit (`golden.py record`) and checked after every task.
- PREREG first, numbers script-generated, tests first (RED then GREEN), tune every arm, CPU only, at most 2 workers x 3 threads, one job per worker, memory-lean.
- Dev seeds 100-109; confirmatory seeds 60-79, never touched before `prereg` froze `bars_rv2c.json` (sha256, `freeze_check`).
- No commits, no documentation by Gemini; Claude writes all documents (component cards, GROWMOE_HANDOFF, SCOPE_CHARTER, MOE report Appendix F, ROUTING_RESEARCH, notebook entry, memory) after each approval.
- No existing test may be edited (`test_gm_phylo.py::test_phylo_signatures_no_gt` pins `predict`'s signature, which is why group answering is the separate method `predict_grouped`, amendment A-2).

## Review focus (failure modes the spec implies; each has a pinned test)

1. A level the permutation count cannot reach silently never rejects -> `PhyloCfg` refuses it (`test_the_config_refuses_exactly_the_levels...`, `test_a_test_that_could_never_reject_is_refused`).
2. A huge batch or window blows memory (MMD kernel) -> row cap (`test_a_huge_batch_and_a_huge_reference_are_cut_to_the_row_cap`, `test_a_huge_batch_is_stored_as_a_bounded_random_subset`).
3. The veto reading the cascade's restricted p-matrix -> `test_the_veto_reads_exact_p_values_not_the_cascades_restricted_matrix`.
4. Silent no-op: group answering with the flag off -> `evaluate_groups` refuses; `predict_grouped` documented; `test_predict_grouped_is_the_per_input_answer_unless_group_vote_is_on`.
5. Ordered or foreign batches flooding probation -> circuit breaker (`test_the_breaker_opens_in_a_live_pool_pauses_and_resumes`, label-sorted F-15 reported).
Also covered: determinism and independence of expert order, all vote x cascade x set-gate combinations, merge keeps both experts' windows, windows counted in the resource report, records without group data are "not measurable" not a crash.

## Tasks (each = one stage of `apply_stage3.py`; tests first)

- [ ] **Task 0 (RV3-0)** preconditions + golden: stage `golden`; `golden.py record`; `cmp` with `files/.../golden_flags_off.expected.json` must be identical.
- [ ] **Task 1 (RV3-1)** `A_tests` (RED) -> `A_code` (config fields + validation, `settest.py`, `sources.py`): 26 tests.
- [ ] **Task 2 (RV3-2)** `B_tests` (RED) -> `B_code` (`setgate.py`, `groupvote.py`): 20 tests.
- [ ] **Task 3 (RV3-3)** `C_tests` (RED) -> `C_code` (the `PhyloMoE` hooks, accounting): 23 tests; `golden.py check` = bit-identical.
- [ ] **Task 4 (RV3-4)** `D_tests` (RED) -> `D_code` (`groupeval.py`, `streams_order.py`, verdict functions `rv2c_verdicts/rv2c_decision` in `fairfight.py`): 26 tests; full suite = N0 + 95.
- [ ] **Task 5 (RV3-5)** `E_code` (runner): smoke of every command, the real `trace` (product must reproduce the spike), then `dev` (breaker grid by lazy successive halving, then A_vote vs A_set on dev seeds 100-109). STOP before `prereg`.
- [ ] **Task 6 (after Claude's review of the dev output)** `prereg` (freezes bars + expected outcomes), `run` (seeds 60-79, 10 streams + label-sorted F-15, reported arms: no-breaker, MMD), `m5` (reported real-text arm), `report`; Claude audits (recipe D-14: mtimes, sha256 of provenance files, byte-identical `report`, suite count, file counts).
- [ ] **Task 7 (adoption, only per `rv2c_decision`)** flip `set_gate` and/or `group_vote` in `PhyloCfg`; pin `set_gate=False, group_vote=False` in `bench._phase1_cfg`, `run_m0.A_KW`, `run_rv2b.py` BASE, the M5 scripts; run_rv2c's A_vote, `golden.py` and the pool tests already pin them. Then the documents.

## Validation record (done by Claude before handing over)

- 95 new tests (settest 15, sources 11, setgate 12, groupvote 8, pool 23, groupeval 8, verdicts 18); 46 deliberate bugs ("mutants") each made a test fail (two equivalent mutants noted).
- Full growmoe suite in the shadow tree with everything installed: 630 passed (535 existing + 95 new); the one existing test that clashed (`predict` signature) led to amendment A-2.
- Flag-off golden fingerprints: recorded before the edits and again after all edits: byte-identical (20 of 20).
- Dry run of the staged installer on a fresh copy: all ten stages apply, a repeat is refused, RED at `A_tests`, 95 passed at the end.
- Real trace (dev seeds 100-102, product code): OV-0.6 +17.87 points (spike +17.8), OV-0.85 +5.27 (spike +5.3), S1 +0.25 (spike +0.3); known batches called new 0 of 378 / 378 / 120; breaker never opened; observe time 3.4x / 2.3x / 1.0x of A_vote (a cost to report, not a gate).

## Self-review against the spec

Section 4.1 -> settest + 15 tests; 4.2 -> sources + 11; 4.3 -> setgate + 12, PhyloMoE hooks + pool tests; 4.4 (no `LogF` use, cascade independent) -> `test_every_combination...`; 5 -> groupvote, `predict_grouped`, groupeval; 6 -> config fields + validation; 7 -> `rv2c_verdicts` (S3-1..S3-9), `rv2c_decision`; 8 -> `cmd_dev`; 9 -> pins, golden, documents (Claude); 10 -> runner; 11 -> every risk row has a test or a reported item. No placeholders remain; names are identical across assets (`SetGate`, `SetDecision`, `SourceWindows.update/refs/ref_pairs/merge_from/n_samples`, `vote_groups`, `make_groups`, `evaluate_groups`, `label_sorted`, `rv2c_verdicts`, `rv2c_decision`, `predict_grouped`, `set_stats`).

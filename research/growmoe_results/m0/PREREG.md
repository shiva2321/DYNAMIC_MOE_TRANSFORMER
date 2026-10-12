# M0 + M1 pre-registration: the fair fight, and the first real test of the dynamic-MoE bet

Date 2026-10-09. Written by Claude before any confirmatory run. Read-only once results exist (charter rule 6); anything that changes goes into `AMENDMENT.md`.

## 0. Provenance (stated honestly)

The pilots in `research/moe_diligence_2026-10/` were run before this file was written, so the direction of several effects is already known. To keep this a real test: (a) every threshold below is set at about half of the pilot effect or at a round number, (b) all confirmatory numbers are regenerated from scratch with the new runner on test seeds 0-9, (c) tuning uses dev seeds 100-102 only (corrected from "100-104" before any run, see section 8), (d) each statement has a PASS, FAIL or NOT MEASURABLE verdict computed by a script from per-seed JSON, never typed. A statement marked "expected" is expected to hold; a result against expectation is reported as such.

## 1. Questions

* **M0 (correct the record).** Which Phase 1 conclusions survive when every method has a tuned training budget, a no-learning memory baseline is included, and memory and compute are counted honestly?
* **M1 (test the bet).** On streams where the answer must be computed (so a lookup fails), does the growing pool beat the best single network we can afford, and by how much as the number of regimes grows?

## 2. Streams (all local, deterministic in the seed; no downloads)

| Name | What | Config |
|---|---|---|
| `S2-5` | Permuted digits, the Phase 1 benchmark config | 5 regimes, schedule [0,1,2,3,4,1,3], 400 train and 100 test per regime |
| `S2-30` | Permuted digits at the spec size | 30 regimes, the spec schedule (37 segments), 400 train and 100 test per regime |
| `S1` | Seven data types, class-incremental | Phase 1 config (400 train, 150 test per type, hashed features) |
| `S1-PRE` | As `S1`, but text and code windows go through a frozen pretrained sentence encoder (all-MiniLM-L6-v2, local cache, offline) projected to 128 | You approved this encoder on 2026-10-09 |
| `F-5`, `F-15`, `F-30` | **New.** Function regimes: regime r lives in its own region of the sphere (inputs `unit(8 * centre_r + noise)`), its label is a hidden random teacher network (width 8, depth 1, 8 classes) applied to the within-region deviation. Neighbours do not share labels. | 800 train and 400 test per regime, two revisits (regime 0 and regime n/2) |

## 3. Methods

* **A** = PhyloMoE with the Phase 1 frozen settings (`q=0.02, alpha_spawn=0.01, m_min=32, route_rule=accept_then_logf`) except `inner_steps`, tuned per stream family (section 4).
* **B2** replay network (width 128, buffer 2048), `inner_steps` tuned the same way.
* **B2\*** (function streams only) the best single replay network over the grid width in {128, 512, 2048} x buffer in {2048, 8192}, chosen on dev seeds by accuracy and **without** a memory restriction (this is the strongest single-network baseline we can afford; if it uses more memory than A the comparison favours the baseline).
* **B1** joint-training ceiling (width 128 and 512), function streams only.
* **B4** oracle-routed experts (one per regime, tuned steps), the isolation upper bound.
* **kNN-all** (k=5, stores everything), **kNN-capped** (same bytes as A).
* **NCM**, **SLDA** (permuted-digit and types streams only; they are at chance on function streams by construction).
* **C** (KeyedDeltaMoE), only for the corrected parameter count (`S2-5`, seeds 0-4).

## 4. Tuning protocol (dev seeds only)

`inner_steps` in {1, 5, 20, 50} per method and per stream family, chosen by mean final accuracy over dev seeds 100-102 on a small dev stream of the family (`S2-5` for permuted digits, `S1` for types, `F-15` for function regimes); if the optimum is the largest value the grid is extended by x2.5 until it is interior or 200 is reached (charter: no edge optimum). The chosen value is then used on every stream of the family. B2\* grid chosen on dev seeds 100-102 of `F-15`.

## 5. Metrics

Primary: final mean accuracy over regimes (numeric grading, held-out test sets). Secondary: forgetting. Resources: memory in bytes = 4 x (parameters, counted once if shared) + 4 x d x (stored vectors); predict multiply-adds per sample = gate scan (sum over experts of stored signature vectors x d) + the answering expert's weight matrices; wall-clock. Statistics: mean and 95% t interval over 10 test seeds; paired differences with paired t.

## 6. Statements and decision rules

| ID | Statement | Rule | Expected |
|---|---|---|---|
| A1 | Phase 1's "A beats replay" does not survive a tuned replay baseline | On `S2-5` and `S1`: mean(B2 tuned) - mean(A tuned) >= -1.0 point. PASS = the old claim is not supported | PASS |
| A2 | A lookup beats A on similarity streams | On `S2-5` and `S1`: mean(kNN-all) - mean(A tuned) >= +5.0 points | PASS |
| A3 | Pool C's reported parameter count was inflated | unique(C) / unique(A) in [0.6, 1.2] on `S2-5` and reported(C) > 2 x unique(C) | PASS |
| B1 | **W1.** The pool beats the best single network on function streams | On `F-15` **and** `F-30`: mean paired difference A tuned - B2\* >= +5.0 points and the 95% interval excludes 0. PASS if both, FAIL if neither, PARTIAL if one | PASS |
| B2 | The pool beats a lookup on function streams | On `F-15` and `F-30`: A tuned - kNN-all >= +10 points | PASS |
| B3 | Routing loses accuracy at scale | On `F-30`: B4 - A >= +5.0 points (reported; no gate) | PASS |
| C1 | No compute saving yet | On `S2-30`: predict multiply-adds(A) >= 0.5 x multiply-adds(kNN-all) (reported; no gate) | PASS (no saving) |
| C2 | A better text front-end helps A | `S1-PRE` - `S1` for A tuned >= +1.0 point (reported) | PASS |

**Kill rule for the bet.** If B1 is FAIL, the dynamic pool is not shown to beat a dense network on streams that need computed answers; the report says so, and the MoE track stops claiming an accuracy advantage and re-scopes (compute saver and abstention layer over an episodic memory). If B1 is PARTIAL, the track continues only to resolve it (more regimes, memory-matched network) before any other work.

## 7. Integrity and speed

* The new runner skips the per-batch predict (needed only for novelty metrics, not used here) and scores all regimes of a test point in one batched call. Accepted only because tests assert: identical accuracy matrices to `bench.run_stream` on a tiny stream, and identical outputs of the vectorised `PhyloMoE.predict` to the original loop code for every routing rule.
* No hand edits to results; tables are written by `run_m0.py`; a test recomputes the verdicts from the JSON.
* No S5/S6 stream is run. One GPU/CPU job at a time per worker, at most two workers, at most three threads each; peak memory reported.
* Scope: additive modules `growmoe/memory_baselines.py`, `growmoe/accounting.py`, `growmoe/fairfight.py`, `growmoe/streams_m1.py`; one behaviour-preserving change in `growmoe/pools/phylo.py` (vectorised `predict`, original kept as `_predict_reference`); one fix in `growmoe/pools/keyed.py` (`summary()` counts shared parameters once). Nothing else in `growmoe/`, nothing in `ishmn/` or `geometry/`.

## 8. Pre-run clarifications (written 2026-10-09 before any dev or test run of the new runner; no confirmatory number had been seen)

These close gaps in sections 4 and 6 that the implementation exposed. They change no threshold.

1. **Dev seeds.** Section 0(c) said 100-104; sections 4 and 6 said 100-102. The intended value is **100-102** and section 0(c) was corrected.
2. **Several streams in one statement** (A1, A2, B1, B2). B1 already defined it; the same rule applies to the others. PASS = the rule holds on every listed stream, FAIL = on none, PARTIAL = on some, NOT MEASURABLE = a needed stream or method is missing. Single-stream statements (A3, B3, C1, C2) are PASS or FAIL. `growmoe.fairfight.verdicts` implements exactly this and is tested.
3. **Budgets for the other baselines.** B2\* is searched **jointly** over width {128, 512, 2048} x buffer {2048, 8192} x `inner_steps` {1, 5, 20, 50} (extended by x2.5 up to 200 if the best configuration sits on the upper edge), on dev seeds 100-102 of `F-15`, and its choice is then frozen. B4 and B1 each get their own `inner_steps` tuned the same way on `F-15` dev seeds; B1 trains `4 x inner_steps` joint steps per batch (the class default).
4. **Evaluation cost.** Decisions read only final accuracies, so `S2-30`, `F-15` and `F-30` are evaluated at the end of the stream only; `S2-5`, `S1`, `S1-PRE` and `F-5` keep the per-segment matrix (for the forgetting columns). Test equality of the two modes is in `tests/growmoe/test_gm_m0_fairfight.py`.
5. **kNN-capped** stores `A's memory in bytes / (4 x d)` vectors (at least 16) on the same stream and seed, so it is matched to A per run.
6. **Pool C** runs on `S2-5` seeds 0-4 with its Phase 1 tuned setting (`c=2.0, rank=16, n_stm=32`). "Reported" parameters = the per-expert sum that repeats the shared base (what `summary()` reported before the fix); "unique" counts each tensor once.
7. **No test-seed output was read before this section.** A first timing attempt used test seed 0 and crashed at import (Windows has no `resource` module) before printing anything; timing was redone on dev seed 100.
8. **Order.** dev tuning -> freeze to `tuning.json` -> test runs (seeds 0-9) -> `verdicts` -> `RESULTS_M0.md`. Any deviation goes into `AMENDMENT.md`, never into this file.
9. **Head size.** Every pool gets `n_classes` equal to the stream's label count. Phase 1 left the pools A and C at the default 64-way head while the baselines already used the stream's count; this keeps parameter counts comparable like with like. It changes no setting that was tuned.
10. **B1 at width 512** reuses the `inner_steps` tuned for B1 at width 128 (tuning the 4x-cost joint trainer at both widths adds hours and nothing to any verdict; B1 enters no verdict).

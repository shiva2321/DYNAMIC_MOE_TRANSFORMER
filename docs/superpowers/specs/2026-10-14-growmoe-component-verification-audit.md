# GrowMoE component verification audit (CVA)

Date 2026-10-14. Why: a benchmark can only say something about an idea if the thing being benchmarked does what the idea says (the user's point, 2026-10-14). Our own record shows what happens otherwise: the competence-monitor bug made the pool look 7 points worse than it is (M0 B1 FAIL), the "beats replay" claim was a training-budget artifact, the first geometry test scored an exact copy of the baseline as a 4.7x speed-up, and growth, merge, archive and revive have never fired in a benchmark run (so nobody knows whether they work). A negative result that comes from our bugs helps nobody; a positive one that comes from an artifact is worse.

## Principles (the ladder every component climbs)
1. **Spec and derivation:** what it should do and which guarantee it claims, written down before measuring.
2. **Implementation checked against the guarantee:** the claimed error rate is measured; key statistics are cross-checked against an established library where one exists.
3. **Controls:** a case where the component must succeed (positive control: perfectly routed experts, a foreign cluster) and one where it must not (negative control: a cluster from the expert's own regime).
4. **Forced scenario:** every lifecycle event can be made to fire on purpose and then does what the spec says (invariants asserted).
5. **Ablation:** switching the component off changes behaviour in the expected direction.
6. **Benchmark at equal budget** (memory, compute) against equally tuned baselines, and report everything.
Defects found are registered, not silently fixed by the person running the check: product changes need Claude's design and the user's agreement.

## Inventory (status on 2026-10-14; unit tests exist for every row, what is missing is behaviour-level evidence)
| # | Component (spec ref) | What the unit tests cover | What is NOT yet shown | Round |
|---|---|---|---|---|
| 1 | Signature + conformal gate + BH (13b) | shapes, p-value validity on small cases, power on easy cases | false-novelty rate across calibration size, dimension and q; the floor p_min = 1/(n+1); power against overlap | CVA-1 |
| 2 | Evidence test (13c) | spawns on a coherent burst, not on outliers | **false-spawn rate for a cluster drawn from an existing expert's own regime**; power against overlap. The test compares a density fitted ON the cluster (leave-one-out) with an old expert's density; that is not a proper likelihood ratio, so the usual "false spawn <= alpha_spawn" guarantee does not follow | CVA-1 |
| 3 | Probation clustering (13c, R1) | random vectors do not cluster, a burst does, TTL, capacity | one regime splitting into two clusters (double spawn) | CVA-3 |
| 4 | Spawn and inheritance (13d) | parent choice, init modes, lineage | does inheritance beat scratch at equal budget (ablation) | CVA-4 |
| 5 | Interference probe, extend vs spawn (13e) | aligned task extends, conflicting task spawns | error rates with small labelled clusters; sensitivity to delta_forget | CVA-3 |
| 6 | Infant protection (13f) | relaxed gate, lr boost, lifecycle | ablation effect | CVA-4 |
| 7 | Competence monitor / CUSUM (13g) | alarm on a remap, none on stationary data (short) | false-alarm rate over long stationary streams; detection delay | CVA-3 |
| 8 | Dormant, archive, revive (13h) | forced in `test_phylo_dormant_archive_and_revive` (states, gating of an archived expert, revive with the same id) | **never fires in any benchmark run**: `dormant_after` 400 and `archive_after` 800 calls exceed the length of the benchmark streams (about 60 to 150 calls), so the benchmarks cannot show their value (memory and compute saved) or their cost (accuracy after revival) | CVA-2 |
| 9 | Merge by distillation (13h) | forced in `test_phylo_distillation_merge_duplicates` | **never fires in a benchmark** (checked every 50 calls, needs mutual overlap >= 0.2); precision (duplicates merge, distinct regimes do not) and what a merge costs in accuracy | CVA-2/3 |
| 10 | Growth policy (Task 14) | `test_phylo_growth_on_plateau_improves_loss`, `..._never_grows_on_data_limited_regime`, `test_gm_growth.py` | fired in 3 of 70 benchmark runs (needs a loss plateau and 128 replay samples); value and cost at benchmark scale unknown | CVA-2 |
| 11 | Batch vote (13k), fingerprint cascade (stage 2), set gate, windows, group vote (stage 3) | extensive (stage 1-3 tests, golden fingerprints) | their end-to-end effect is pre-registered or in dev; false-alarm budget derived (D3) | done / CVA-3 |
| 12 | Resource accounting | known-answer tests | independent recount of parameters and stored vectors on a live pool | CVA-5 |
| 13 | Baselines | tuned budgets (M0) | **equal-memory fairness: ReplayNet's default buffer is 8x one expert's reservoir (2048 against 256), so "A against B2" is not equal-memory**; kNN capped to the same bytes exists | CVA-5 |

## Derivations to write (Claude)
- **D1 gate:** conformal p-value with a finite calibration set is valid under exchangeability, smallest value 1/(n+1), so q below 1/(n+1) can never flag; BH at level q controls the false discovery rate; what the infant relaxation (p x 4) does to the guarantee.
- **D2 evidence test:** why a density fitted on the cluster itself breaks the likelihood-ratio guarantee (E[LR] = 1 needs both densities fixed before seeing the data); expected false-spawn rate to be measured by CVA-1.
- **D3 false-alarm budget (hindsight from the RV3 dev run):** expected spurious experts per run = (known batches per run) x (level per test) x P(spawn | false novelty). Dev numbers: 126 to 326 known batches per run, level 0.001, so 0.13 to 0.33 false verdicts per run, about 60 to 70 % of them end in a duplicate expert. A budget of <= 0.01 per run needs level <= 1e-4 or a second confirming batch (persistence).
- **D4 cost model:** gate multiply-adds per input (exact = sum of prototypes x d; cascade = experts x k x d + shortlist), memory = 4 x (parameters + d x stored vectors), so that "efficiency" is a formula checked against the accounting, not an impression.

## Definition of done and bounds
The audit is done when the table above has a measured status for every row (verified / partial / defect, with the number and the criterion it was judged by). Criteria are fixed in the kickoff before the run. No new research documents; Gemini produces scripts, tests, a json and a generated markdown table; Claude writes the derivations and the status page.

## Rounds
- **CVA-1** gate calibration, power against overlap, evidence-test false-spawn rate and power (experiments, no product change).
- **CVA-2** (a) coverage matrix: every bullet of the handoff spec (13a to 13k, R1 to R6) against the existing test that checks it, or MISSING; missing ones are written as tests asserting the intended behaviour, defects registered as strict xfail, product code untouched. (b) the lifecycle at benchmark scale: a LONG stream (about 3,000 observe calls, 30 regimes cycling with revisits, shortened lifecycle thresholds declared in advance) on which dormant, archive, revive, merge and growth actually fire; measure what they save (stored vectors, parameters, gate multiply-adds, active experts per input) and what they cost (accuracy on the regimes that were archived or merged, after revival), against the same run with the lifecycle switched off.
- **CVA-3** probe error rates, competence-monitor false alarms and delay, merge precision, clustering splits.
- **CVA-4** ablations: inheritance, infant protection, probation, alarms.
- **CVA-5** independent accounting recount; equal-memory baseline protocol; then the decision experiments (many topics, real stream) on the verified system.

## Status after CVA-1, 1B, 2a, 3A, 2B and 5A (2026-10-14; every Gemini result re-run or independently re-checked by Claude)
| # | Component | Status | Number and criterion | Where |
|---|---|---|---|---|
| 1 | Signature + conformal gate + BH | VERIFIED, extremely conservative | false novelty about 0 at every calibration size >= 128; with 256 prototypes p_min = 1/257 so a batch of 64 at q = 0.02 needs about 13 samples at the floor before BH can flag; power against overlap is the per-sample gate's known weakness (AUROC 0.53 at centre cosine 0.85) | CVA-1 |
| 2 | Evidence test (spawn rule) | **DEFECT (uncalibrated)** | a cluster drawn from an existing expert's own regime is called new 0-2 % of the time at 32-64 points, 5-12 % at 128, 33-52 % at 256 (target <= 1 %). Replacement candidate: the exact permutation test of the cluster against the expert's prototypes: 0.00 false spawns in all 24 cells and power 1.00 for overlap <= 0.85 (CVA-1B, thresholds from seeds 0-9, evaluation 10-19). Not enabled; a later stage 4 with its own pre-registration | CVA-1, CVA-1B |
| 3 | Probation clustering | not yet measured (double spawn from one regime) | | CVA-3b |
| 4 | Spawn and inheritance | not yet (ablation) | | CVA-4 |
| 5 | Interference probe, extend vs spawn | not yet (error rates) | | CVA-3b |
| 6 | Infant protection | not yet (ablation) | | CVA-4 |
| 7 | Competence monitor | PARTIAL | false alarms 1 in 200,000 stationary samples (0.05 per 10,000; the one alarm was resolved as an extend); detection delay still unmeasured | CVA-3A |
| 8 | Dormant, archive, revive | VERIFIED works and is safe; **not an efficiency mechanism yet** | on a 600-call, 30-regime cycling stream (thresholds 60 / 120 equal half and one revisit period): events fire (58-77 dormant, 24-35 archive, every archived expert revived after a median of 17-35 calls), invariants hold at every call (0 violations in 11 runs), deterministic across processes; accuracy -0.41 points [-0.93, +0.12] (n = 5, p 0.097); stored vectors 95.9 %, accounted bytes 97.3 % of the arm without lifecycle (about 3 % time-averaged by the accounting convention, about 6 % if the Adam state is counted); gate multiply-adds unchanged. Dormant is a pure label (a dormant-only arm is bit-identical to no lifecycle). Archived experts cannot be merged (replay None). Making it an efficiency mechanism needs a gate that consults dormant experts last and an archive that frees weights | CVA-2B and Claude's independent audit |
| 9 | Merge by distillation | NOT EXERCISED (0 merges: S2 regimes are label-permuted so no two experts may merge); precision and cost still unknown | needs a stream with duplicate regimes | CVA-3b |
| 10 | Growth policy | NOT EXERCISED (0 growth events in CVA-2B) | needs a stream with a loss plateau | CVA-2c or CVA-4 |
| 11 | Batch vote, cascade, set gate, group vote | stage 1-2 adopted; stage 3 in dev (first dev run failed the guard S3-2 through false set-novel verdicts, persistence A-15 fixes it on dev) | see the stage-3 amendments | RV2-A/B/C |
| 12 | Resource accounting | VERIFIED as arithmetic: an independent hand recount equals `pool_report` on parameters, stored vectors, gate multiply-adds and bytes, with an archived expert and with source windows. **The definitions leave out** the optimiser state (Adam keeps 2 x parameters per live expert: 41 % of training-time memory), the 32-sample monitor window per expert, the probation and held buffers; the set gate's source windows ARE counted and add about 314 rows per expert at the defaults (stage 3 stores 1.6-3.0 x the vectors of the vote pool, mostly because it has the right number of experts) | CVA-5A, dev records |
| 13 | Baselines | pending: equal-memory protocol | | CVA-5b |

Process notes. (1) My criterion (c) of CVA-2B was too weak: it looked only at archived experts at the END of the run and there were none in any of the five runs, so it passed vacuously; the all-calls check supersedes it. (2) My kickoff said about 470 observe calls; the stream has 600 (four per 200-sample segment: 64 + 64 + 64 + 8). (3) A "revive" event covers both a dormant expert waking up (keeps its replay buffer) and an archived expert coming back (gets an empty buffer and a new optimiser).


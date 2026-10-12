# Routing v2 — batch-consistent assignment, a fingerprint shortlist in front of the gate, and (conditional) calibrated batch novelty

Date 2026-10-10. Status: **design for review** (path: architectural). Nothing is built, pre-registered or committed. Evidence tags: **[M]** measured on dev seeds 100-102 by throwaway spikes (direction only, NOT pre-registered; every number is script-generated in `research/growmoe_results/rv2/RESULTS_RV2_SPIKES.md`; "R3" below means section 3 of that file, while "§3" means a section of this spec), **[C]** read in the code, **[H]** hypothesis / prediction.

## 1. Decisions (from the user)

1. Routing is the next problem after M0 and the lifecycle repair (M2-A). Priorities: (1) a cheaper gate, (3) overlapping / messier data. (2026-10-10)
2. Approach A (three stages, each pre-registered and tested on its own; stage 3 conditional) approved. Be considerate of tokens without compromising quality or necessity. (2026-10-10)
3. Standing: a change that passes its pre-registered test becomes the default and is noted wherever it matters; historical entry points are pinned to what they ran. PREREG first, numbers script-generated, TDD, tune every arm, memory-lean runs (at most 2 workers x 3 threads), nothing committed unless asked.

## 2. What the spikes showed [M] (dev seeds, throwaway patched copies of `PhyloMoE.observe`; no product file touched)

| Question | Result |
|---|---|
| Where does the compute go? | The gate scan is 97 % of the multiply-adds of a prediction on F-30; a perfectly routed expert needs 34 kMACs (M2-A). |
| Can a cheap fingerprint replace the scan? (R1, R7) | As a first stage in front of the exact test: top-2 shortlist by k-means sub-centroids, then the exact conformal gate on the shortlist. F-15 46.1 % vs exact 46.0 % at 6.8x fewer MACs (k=4); F-30 45.3 % vs 45.2 % at 12.0x; S2-30 88.4 % vs 88.6 % at 8.5x. One centroid (k=1) is 7.4x / 14.0x / 9.5x but loses 0.6 points on S2-30 and fails on S1. S2-5 and S1 have 3-4 experts: nothing to save. By formula the saving grows like E/m (not measured at large E). |
| What did not work as fingerprint? | Random Fourier "wave" features and ridge routers on them (wave-512: 76.0 % alone on S2-30); a strided subset of the stored prototypes as landmarks (16 landmarks cost twice 8 sub-centroids and lose 0.7 point on S2-30). |
| What happens when regimes overlap? (R2, R4) | Accuracy 50.5 / 33.7 / 18.4 / 16.8 % at centre cosine 0 / 0.3 / 0.6 / 0.85 while oracle-routed fresh experts stay at 48 %. The regimes stay separable from x (nearest centroid 100 / 100 / 99.9 / 90.1 %). Test-time routing is not the bottleneck (owner-routed bound within 3 points of the pool). |
| Cause at cosine 0.3 (R5) | Training contamination: only 78.4 % of what an expert trains on comes from its own regime. Letting the whole batch train the expert most of its samples were routed to ("batch vote", share >= 0.5) gives purity 99.5 % and accuracy 47.2 % (oracle 48.0). |
| Is the vote safe? (R6, R8, R9) | F-15 46.0 -> 50.4, S1 78.2 -> 79.1, S2-5 identical, 50/50 mixed batches 30.7 -> 36.2 (on the six streams of R6 never worse on any seed except S1 seed 101, -0.7), 70/30 mixed 26.8 -> 32.7. Known regimes later arriving 70/30: minority regimes 47.7 -> 46.4 at cosine 0 (cost 1.3 points), 32.0 -> 43.1 at cosine 0.3. The variant that moves a sample only if the modal expert's own gate accepts it is 1-3 points weaker on single-source overlap and costs 0.3 point on the minority. |
| Cause at cosine >= 0.6 (R4, R5) | Regimes merge (3.3 experts for 10 regimes at 0.6): the per-sample novelty test rejects 88.7 % of foreign inputs at 0.6 and 0.8 % at 0.85 (first batches of later regimes are flagged novel 20.4 % / 0.0 % of the time), while a batch-level fingerprint test (distance of a 64-sample mean from the expert's prototype mean, bootstrap 99.9 % threshold) detects 100 % of foreign batches with 0 % false alarms. My first live version (Fisher combination of the per-sample p-values, threshold 1e-4) over-fired: 72 / 131 / 89 experts for 10 regimes, because the combination's null is not calibrated. |
| Do labels help assign? (R3) | Little: lambda x label cross-entropy subtracted from the log-density gave +1.6 / +2.8 points at 0.3, +0.5 at 0.6. Not pursued. |

## 3. Scope

Goals: **G1** more accurate under overlap without hurting any existing stream; **G2** the gate costs at least 4x fewer multiply-adds at 30 experts with non-inferior accuracy and *identical* novelty decisions; **G3** (conditional) regimes no longer merge at cosine >= 0.6.
Non-goals: label-conditioned assignment or spawning (measured weak); split-conformal calibration scalars / fewer prototypes (the other half of the gate-cost idea; deferred, YAGNI); the test-time routing rule; vertical growth (R12b); episodic memory; a pretrained encoder; multi-pool.

## 4. Stage 1 — batch-consistent assignment

**Rule.** In `PhyloMoE.observe`, after the per-sample assignment loop and before the probation buffer: let A be the samples with `assigned >= 0`. If |A| < `batch_vote_min` (8) do nothing. Let m be the expert id most frequent in `assigned[A]` (ties: smallest id) and share = count(m)/|A|. If share >= `batch_vote_share` (0.5), set `assigned[A] = m`. Novel samples (`assigned == -1`) are untouched; spawn, probe, monitors, replay and training all see the new `assigned`; `predict` is unchanged.
**Interface.** `PhyloCfg.batch_vote: bool = False` (set True on PASS), `batch_vote_share: float = 0.5`, `batch_vote_min: int = 8`. One pure private method `PhyloMoE._batch_consistent(assigned) -> Tensor` (returns a new tensor, no pool state). No change to `StepRecord`, `Prediction` or any public API.
**Fallback (pre-declared, built only if needed).** The restricted vote: the final reassignment `assigned[A] = m` moves only the samples the modal expert's gate accepts (`P[i, m] >= thresholds[i]`, the per-sample rule's own eligibility test). It is a third arm in the confirmatory test; it is implemented only if the plain vote fails a guard (§4 decision rule).
**Edge cases.** No experts yet (bootstrap) returns before the vote. The vote only reads `assigned`, so it applies identically in the conformal and the `no_conformal` ablation paths. The modal expert is always one the per-sample rule already chose (after its supersede filter), so a superseded expert cannot be selected unless the per-sample rule selected it. Reassigned samples feed the modal expert's signature, replay and competence monitor like any other assigned sample.
**Failure modes.** (a) A minority source inside a batch is pushed into the majority's expert (measured cost 1.3 points on known 70/30 minorities at cosine 0, [M] R9; guarded by G-MIX). (b) Does not stop regimes merging at cosine >= 0.6 (stage 3). (c) Behaviour depends on batch size (only 64 measured; sensitivity reported).
**Tests (written first; each must fail before the code exists).** `_batch_consistent`: unanimous -> unchanged; 75/25 -> all to the mode; share below threshold -> unchanged; fewer than `batch_vote_min` accepted -> unchanged; -1 entries untouched; tie -> smallest id; input not mutated. Pool: flag off -> `StepRecord.assigned` bit-identical to a recorded run of the current code; on a small overlapping stream (4 regimes) the flag raises training purity from below 0.90 to at least 0.95; flag on a non-overlapping stream keeps purity within 0.02 and accuracy within 3 points. Stream builder: `function_regimes(overlap=0.0)` bit-identical to the pilot (the shared component is drawn from its own RNG so the main stream is untouched), `overlap=rho` gives mean pairwise centre cosine within 0.05 of rho; a `mixed_tail` option appends the 70/30 segments of R9. Pins: `bench._phase1_cfg`, `run_m0.A_KW`, the M2-A `A_new` arm keep `batch_vote=False`.
**Confirmatory test (PREREG written before any fresh seed; config frozen by sha256).** Arms `A_new` (M2-A config), `A_vote` (plain vote), `A_voter` (restricted vote, pre-declared); steps tuned per arm on dev seeds with `fairfight.tune_steps`; fresh seeds 20-39 (n=20), batch 64; paired intervals via `metrics.paired_diff`. Bars are fixed after a dev run and set well below the dev effects [dev]:

| ID | Stream | Statement |
|---|---|---|
| V1 (gating) | OV-0.3 | A_vote - A_new >= +8.0 points, 95 % interval excludes 0 [+13.5] |
| V2 | F-15, F-30 | >= +1.0 point each, interval excludes 0 [F-15 +4.4] |
| V3 | OV-0.6 | >= +3.0 points [+7.3] |
| G1-G6 | S1, S2-5, S2-30, S4, OV-0, OV-0.3 50/50 mixed | mean >= -3.0 and interval not entirely below 0 |
| G7 | S7 | mean alarms >= 3.0 per run (alarms still catch a remap) |
| G-MIX | known regimes later arriving 70/30 (cosine 0 and 0.3), accuracy on the minority regimes | mean >= -3.0 and interval not entirely below 0 |

Descriptive, not gated: training purity, expert counts, events, batch-size sensitivity (b = 16, 256 on OV-0.3). **Decision rule:** adopt the plain vote iff V1 and every guard pass (V2 and V3 are recorded, not gating); else the restricted vote iff it passes V1 and every guard; else neither. V1 failing means the contamination mechanism claim was wrong: stop, re-diagnose, stage 3 is not started on this premise.

## 5. Stage 2 — fingerprint shortlist in front of the exact gate

**Rule** (when E >= `gate_shortlist_min_experts`, else the exact path). Each expert holds k unit sub-centroids (spherical k-means of its stored prototypes; k=1 is the centroid). (1) Stage 1: s_j(x) = max over the expert's sub-centroids of the cosine with x, one dense product for all experts. (2) Shortlist S_i = the `gate_shortlist` (2) experts with the highest s_j(x_i). (3) Exact `score`, conformal p-value and log-density only for the pairs (sample, expert) with the expert in the sample's shortlist, dispatched per expert (one dense matmul per expert that appears in any shortlist); other pairs get P = 0, LogF = -inf. (4) Second look: every sample whose restricted p_nov (maximum P over its shortlist) is <= q gets the exact scan over ALL experts. Everything downstream (BH, assignment, prediction rules) is unchanged. The fingerprint is built on first use and rebuilt when the signature has seen >= `gate_fingerprint_refresh` (128) more stream points, and after a merge.
**Guarantee [C, proof sketch].** The novelty / unknown flag of every input equals the exact gate's. Benjamini-Hochberg at level q rejects using only p-values <= q (a p-value above q can never satisfy p_(k) <= kq/B). Restricted p <= exact p, and every sample with restricted p <= q is recomputed exactly, so the entries <= q are identical in both versions and all other entries exceed q in both; hence the largest accepted rank and the rejection set coincide. Only the *choice among accepting experts* is approximate (an accepting expert outside the shortlist is missed); measured recall of the exact choice at top-2: 99.6-99.9 % (k=4, 8) on streams with >= 15 experts [M]. Reported `p_nov` is exact when <= q and a lower bound otherwise. Staleness of a fingerprint therefore lowers recall but cannot create false novelty. With `gate_shortlist >= E`, or E below the minimum, the result is bit-identical to the exact path.
**Edge cases.** The cascade is used only on the conformal path and only when every expert's signature is a `VmfSignature` (otherwise the exact path, as with `sig_cls` alternatives and `no_conformal`). Experts in infancy are always scored exactly (they are few and small, and they must attract their own regime quickly). An expert with fewer prototypes than k uses k = its prototype count. Dormant and archived experts keep their signatures and are covered like any other.
**Interface.** `PhyloCfg.gate_shortlist: int = 0` (0 = exact scan; set to 2 on PASS), `gate_fingerprint_k: int = 4`, `gate_fingerprint_refresh: int = 128`, `gate_shortlist_min_experts: int = 16`. New module `growmoe/fingerprint.py` (spherical k-means and stage-1 scoring; no pool state). `PhyloMoE._gate_matrices` dispatches between the exact and the cascade path. Today only `predict` uses it; `observe` scores experts in its own per-expert loop [C], so the first step is to route `observe` through `_gate_matrices` (same values bit-for-bit, pinned by a test that compares the two on recorded streams) and only then add the cascade there. `pool.gate_stats` counts the (sample, expert) pairs actually scored (stage 1, stage 2, second look); `accounting.measured_gate_macs_per_input(pool, d)` turns them into MACs. `pool_report` keeps its static exact-scan numbers so existing reports keep their meaning. k is decided on dev by a stated rule: the cheapest k in {1, 4, 8} whose top-2 accuracy is within 0.3 point of exact on every dev stream with >= 16 experts [expected 4]. The shipped minimum E is the wall-clock crossover measured in the plan, not a guess.
**Tests (first).** k-means: shapes, unit norm, determinism, k=1 equals the normalised mean. Stage-1 scores equal a naive loop. `gate_shortlist >= E` or E below the minimum -> P and LogF bit-identical to exact. **Property test: novelty flags identical to the exact gate on random pools**, including cases where the true expert is outside the shortlist. Recall 100 % on a well-separated synthetic pool. The second look fires exactly for samples with restricted p <= q (counter check). Counter arithmetic matches the analytic pair count. Refresh policy (stale vs fresh). `test_gm_m0_fast_predict` still passes unchanged.
**Confirmatory test (PREREG; fresh seeds 40-59, n=20).** Arms `A_exact` (best adopted default) and `A_cascade`; the accuracy test forces the cascade whenever E >= 8 so OV-0.3 and F-15 are really exercised. Bars fixed after a dev run [dev]:

| ID | Statement |
|---|---|
| C1 (gating) | F-15, F-30, S2-30, OV-0.3: mean paired A_cascade - A_exact >= -1.0 point and interval lower bound >= -2.0 [dev -0.2 to +0.1] |
| C2 | mean absolute difference in spawns per run <= 0.5 (decisions are exact; trajectories may still drift) |
| C3 | measured gate MACs per input <= 0.20 x exact on F-30, <= 0.25 x on S2-30 and F-15 [dev 0.083 / 0.118 / 0.147 at k=4] |
| C4 | scaling check, synthetic pool of 500 experts (gate only, no training): MACs ratio <= 0.05 and wall-clock ratio of the gate call <= 0.5 at B = 64 and B = 1024 |
| G | S1, S2-5 (E < minimum): bit-identical to exact |

Reported, not gated: end-to-end wall-clock of `observe` and `predict` (after the scan is cut, the signature refit becomes a large share of the time [C]). Decision: adopt iff C1 passes; C3 or C4 failing with C1 passing keeps it opt-in (flag stays 0) and says so. Stage 1 and stage 2 interact through `assigned` only; the stage-2 arms are run on top of whichever stage-1 default was adopted.

## 6. Stage 3 (conditional) — calibrated batch-level novelty

**Entry:** the pre-registered OV-0.6 / OV-0.85 results still show more than 10 points between the pool and oracle-routed experts [dev: 25.7 vs 48.0 after the vote].
**Candidate mechanism (to be validated, not committed):** a batch statistic (standardised distance of the batch mean from the expert's fingerprint mean) tested against an EMPIRICAL null made of the same expert's own earlier batches, each computed before that batch updated the signature so the null is exchangeable by construction; a one-sided CUSUM with the Monte-Carlo-calibrated threshold of `competence.LossChangeMonitor`; a flagged batch goes whole to the probation buffer as a candidate cluster; the existing evidence test and interference probe decide spawn vs extend.
**Spike acceptance, dev seeds, before any product code:** (a) expert counts unchanged versus stage-1-only on F-15, F-30, S1, S2-5, S2-30, OV-0; (b) at OV-0.6 and OV-0.85, at least 8 experts for 10 regimes and at least +8 points over stage-1-only. If not met: stop, record as open. If met: an addendum to this spec (design + PREREG) goes through the same review.

## 7. Rollout, documentation, reproducibility

Order: stage 1 -> stage 2 -> stage 3 spike. Per stage: flag off while building (test-first); dev tuning; PREREG; freeze by sha256 (`freeze_check`); fresh-seed run; script-generated report; decision rule. On PASS: flip the default test-first and pin the historical entry points explicitly (`bench._phase1_cfg` for Phase 1, `run_m0.A_KW`, the M2-A `A_new` arm). Note it in `growmoe/config.py`, `docs/COMPONENT_CARDS.md` (a card per stage), `docs/superpowers/handoff/GROWMOE_HANDOFF.md`, `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` (Appendix F), `research/ROUTING_RESEARCH_2026-10.md`, the m0 and m2a `AMENDMENT.md`, and the memory index.
New files: `growmoe/fingerprint.py`; `rv2a_verdicts` / `rv2b_verdicts` and expected-outcome tables in `growmoe/fairfight.py` with tests like `m2a_verdicts`; `research/growmoe_results/rv2a/` and `rv2b/` (PREREG.md, run script, tuning json + sha256, report script). The spike scripts and their JSON stay in `research/growmoe_results/rv2/` as the record of the evidence.

## 8. Protocol and budget

Reuse `fairfight.run_fast / tune_steps / run_record`, `metrics.paired_diff` and `freeze_check`; no new framework. CPU only, at most 2 workers x 3 threads, one job per worker. Rough size: stage 1 about 3 arms x 12 streams x 20 seeds of short runs; stage 2 about 2 arms x 5 streams x 20 seeds. Scripts print compact tables, reports are script-generated, chat updates stay short; stage 3 gets one spike and no more until it earns an addendum.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Dev effects do not replicate (3 seeds; several variants were tried: vote50, vote80, summed-log-density owner, restricted vote) | Fresh seeds, bars at roughly half the dev effects, the primary variant declared before the run, the restricted vote included as a pre-declared arm |
| Plain vote harms minorities inside mixed batches | G-MIX guard, 50/50 mixed guard, restricted-vote fallback |
| Stage-2 wall-clock gain smaller than the MAC gain | Both reported; minimum E set from the measured crossover; opt-in if C3/C4 fail |
| Fingerprint goes stale | Refresh policy; the guarantee (§5) means staleness cannot cause false novelty |
| Stage 3 over-fires like my first attempt | Empirical null, spike with a hard acceptance criterion, abandon if unmet |
| Flag interactions (vote x cascade) | Stage-2 arms run on top of the adopted stage-1 default; both flags covered by the pinned reproducibility tests |

## 10. Left to the plan (not blockers)

Exact bars after the dev runs; the value of k; the shipped E minimum; where the overlap and mixed-tail stream builders live (`growmoe/streams_m1.py`); the restricted vote is implemented only if its fallback triggers. The implementation plan has two checkpointed phases (stage 1, then stage 2) and one stage-3 spike task that runs only if §6's entry condition holds; each phase ends at its decision rule, so a failed stage cannot silently feed the next.

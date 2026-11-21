# v9 plan: make the v8 comparisons valid, then fix generalization

**Status:** DRAFT for review. Not a pre-registration. Nothing here is frozen, and no run has been made for it.
**Date:** 2026-11-21 · **Branch:** `v5-arch-efficiency` · **Follows:** `results/real_pilot_v5/` (`Cog_v8`, `test_final`, seeds 0–2)
**Evidence:** `results/real_pilot_v5/test_final/summary.json`, the per-seed files beside it, and the code at the line numbers cited below.
**Where it goes next:** once approved, the bars in §7 get frozen into `results/real_pilot_v6/PREREG.md` *after* Stage 0 has measured the noise floor and *before* any Stage 1 run.

---

## 1. Intent

The project's goal is a dynamic-MoE model that learns *meaning* and generalizes to things it was not trained on, instead of fitting next-token surface statistics. `Cog_v8` moved the in-distribution numbers a long way (Banking77 accuracy 0.197 → 0.555, overlap-trap AUC 0.562 vs Dense 0.368, interchange accuracy 0.974). The held-out numbers did not follow, and several comparisons in the v8 report rest on controls that are not matched. v9 therefore has two jobs, in this order:

1. **Stage 0:** make the measurements trustworthy: matched controls, unbundled ablations, honest probes, enough seeds.
2. **Stage 1:** only then change the training algorithm, and only the parts Stage 0 shows to be limiting.

## 2. What the v8 test run supports and what it does not

Numbers are 3-seed means from `test_final` unless a per-seed list is shown. File:line references are in `experiments/run_v8_cognitive_pilot.py` unless stated.

| Claim in the v8 report | What the data and code show | Status |
|---|---|---|
| Intent recovery (Banking77) | Accuracy 0.555 vs Dense 0.545 is a tie (gap +0.010, std 0.011). AUC 0.562 vs Dense 0.368 is a real gap. | AUC supported, accuracy not |
| I5 interchange training gives "true causal representation" (0.974) | Interchange accuracy is evaluated on **seen** pairs with the same chunks and the same `dim_slice` as in training (train 985–1001, eval 1093–1102). It shows the objective was fit, not that it generalizes. | In-distribution only |
| I5 trades zero-shot generalization for fit (0.25 vs 0.51) | **Withdrawn.** Held-out accuracy per seed: `Cog_v8` [0.07, 0.27, 0.41]; `NoI5_NoRel` [0.12, **0.95**, 0.47]; `SequenceTop1` [0.10, 0.12, 0.34]. The spread is larger than the gap, so 3 seeds cannot separate the arms. | Undetermined |
| Episodic few-shot (0.635) is "genuine in-context rule binding" | The query is one of the support pairs (821–825): `pick` is drawn from the shots. An exact-match lookup scores 100%. The same distribution is also trained on (1003–1017). Per seed [0.875, 0.859, **0.172**], so one seed failed. | Retrieval, not induction |
| `Cog_v8` beats Dense and `FrozenRandomRouter` on Track 1 | Both controls fall outside the tuple at 492 and take the `AdamW` branch (528–530; Track 1 at 901–905). They run without GPM, SPGC or the quarantine gate, while the Cog arms use `MuonBONGOptimizer`. They fail to fit even the seen set (0.07–0.09 in every seed, chance 0.0625). | Controls not matched |
| `NoI5_NoRel` isolates I5 + I7 + relational pooling | It sets `use_v8_core=False` (470–473). That flag is referenced at more than 30 places in `dmoe/real_model.py`, among them EMA decay 0.94 vs 0.99 (line 917), the world-model target layer (928), content masking, the FSP head and the relational pool. It is a revert of the whole v8 core. | Bundled |
| Dynamic k beats static routing on escalation (EscMargin 0.302 vs 0.077) | `SequenceTop1` also sets `z_nuance_thresh = d_sem_nuance_thresh = 1e9` (489–490), which switches nuance escalation off. Selectivity pinned at 1.0 is by construction. The arm changes two things at once. | Confounded |
| Dynamic k is efficient | Mean k = 0.576·1 + 0.374·2 = **1.32**, versus 1.0 for top-1. Against `SequenceTop1`: Banking77 accuracy −9.2 pts, R@5 −8.4 pts, +49 s. Whether routing or escalation causes the gaps is not separable (row above). | Costs more, benefit unclear |
| Functional routing NMI | `FrozenRandomRouter` scores 0.410 vs `Cog_v8` 0.355, so the metric tracks partition entropy, not function. | Metric unsound |
| GPM readout exemption "eliminated saturation" | `gpm_at_cap_count` is 0 for every arm, including Dense and BagOfBytes. No arm sets `exempt_readout_heads=False`. | Untested |
| Concept retrieval | R@5 0.170 vs untrained 0.139 vs byte histogram 0.2375, with seed spread 0.134–0.234. | Weak signal |
| Held-out family AUC | Dense 0.493 > `Cog_v8` 0.470; the bar needed +0.12 over `NoI5_NoRel` and got +0.104. | Not better than Dense |
| Burst immunity | `Cog_v8` and `NoI5_NoRel` both score 0.9401; `SequenceTop1` is [0.97, 0.33, 0.06] across seeds. The tie looks like a shared ceiling. | Cause unexplained |
| Abstain | AUROC is 1.0 in three arms. | Saturated |

**Structural fact about Track 1.** `BeliefMeaningTask` is arithmetic on a 16-element cycle: `m = inv_phi[(phi[a] + 3·phi[b] + shift_op) mod 16]`, with 192 of 256 (a, b) pairs seen (`dmoe/belief_training.py:31–38, 91–112`). This is the classic grokking regime for modular arithmetic, and held-out accuracy in that regime is bimodal across seeds. That is exactly what the per-seed lists show (`NoI5_NoRel` seed 1 reaches 0.945, its neighbours 0.12 and 0.47). A held-out metric from 3 seeds on this task is not reliable.

**Corrections to what I told you earlier in this session.** (a) I said I5 costs held-out accuracy; the per-seed data do not support that. (b) I said dynamic routing wins on escalation; that comparison is confounded by the arm design. (c) I suggested training on episodes; the code already does, and the problem is that the probe is a lookup. (d) I inferred the Dense control was mis-specified from its scores; the code confirms an optimizer and SPGC mismatch.

## 3. Goals, non-goals, anti-goals

**Goals**
- **G1** Every comparison the v9 report makes uses a matched control, with per-seed data and an interval.
- **G2** Held-out zero-shot composition improves with a seed-robust margin, with the interchange gain kept.
- **G3** Dynamic k either earns its compute (equal or better accuracy at mean k ≤ 1.0) or is shown not to, with the cause identified.
- **G4** Concept retrieval beats the byte histogram and `SequenceTop1`.
- **G5** Track 2 train time ≤ 220 s (carried over from V8-F6).

**Non-goals:** scaling model size; new architectural modules; expanding the teacher-informed arms (they stay, reported separately, `teacher_informed=true`); a 20-seed `REPRODUCED` claim on every arm.

**Anti-goals:** passing a bar by changing the task, the controls or the metric after seeing results; reporting accuracy on the training intervention as generalization; adding another bundled component whose effect cannot be isolated.

## 4. Constraints

- Non-token core stays: 0 decoder cross-entropy gradient into the core, always-plastic design, no frozen pretrained encoder in the architecture.
- Hardware: single RTX 3060 (12 GB), Windows. Peak VRAM today is 794 MB, so memory is not the limit.
- Pre-register before running; the dev budget stays at 4 dev trials on seed 100; test seeds stay frozen until the one test run.
- Claim tiers as in the earlier PREREGs: 3 seeds `OBSERVED`, 20 seeds `REPRODUCED`. The file `research-protocol.md` that those PREREGs cite is not in this repo, so I could not check §§6 and 9; this needs your confirmation.
- Wall-clock reference from v8 file timestamps (seed 0): `Cog_v8` ≈ 12 min per seed (Track 1 + Track 2 + evals), `SequenceTop1` ≈ 11, `NoI5_NoRel` ≈ 9, Dense ≈ 8; all 11 arms ≈ 60 min per seed. Track 1 alone is much cheaper but has not been timed separately.

## 5. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | Stop at v8, write up as is | Cheapest. Leaves claims in the report that the code does not support. |
| **B** | **Stage 0 diagnostics, then gated Stage 1 interventions (recommended)** | About 2–3 days of eval code before any new objective. Every Stage 1 change targets something the diagnostics showed. |
| C | Go straight to new objectives (learned-rotation interchange, bigger task) | Fastest to a result, but the result would sit on unmatched controls and a 3-seed bimodal metric, so a gain could not be attributed. |

I recommend **B**. This is my recommendation, not a decision you have made.

## 6. Stage 0: diagnostics (no new training objective)

Small code changes needed: three flags to unbundle the v8 core (`use_iit`, `i7_steps`, `use_rel_pool`), an escalation switch independent of the router mode, a `--tracks` option so Track 1 can run on its own, and new probe functions. Each diagnostic names the result that would change the plan.

| ID | Diagnostic | Interpretation |
|---|---|---|
| **D1** | **Matched controls.** Dense and frozen-random-router arms get the same `MuonBONGOptimizer` groups, GPM and quarantine gate as `Cog_v8`; plus a 3-point AdamW learning-rate sweep for Dense as a sanity check. | A control whose Track 1 seen accuracy is < 0.9 is invalid for comparison and is reported as such. |
| **D2** | **Unbundle `NoI5_NoRel`:** `NoI5` (interchange off), `NoI7`, `NoRel`, and the existing `NoV8Core`, all from the same code path. | Tells which component, if any, moves held-out accuracy and Banking77 accuracy (0.267 in the bundle vs 0.555). |
| **D3** | **Routing × escalation 2×2:** {token-level dynamic k, sequence top-1} × {nuance escalation on, off}. | Separates the routing effect from the escalation effect in every number currently credited to "dynamic routing". |
| **D4** | **Honest episodic probes:** (a) shuffled-binding control (answers drawn from another episode's shots, expect chance); (b) exact-lookup oracle as the upper bound; (c) an induction probe whose queries are *not* in the support and whose answer a reference solver can compute from the shots, at 4/8/16/32 shots, with the solver verified before use. | Only (c) can show rule induction. If (c) is at chance, I6 is a memory mechanism, not induction, and is reported as that. |
| **D5** | **Interchange on held-out values:** accuracy on seen pairs (current), on held-out pairs, with a random same-size slice swapped (expect failure), and with other chunk/slice splits. | If held-out interchange is near chance, I5 is in-distribution fit. |
| **D6** | **Seeds and grokking curves:** Track 1 only, 20 seeds, on `Cog_v8`, `SequenceTop1`, `NoV8Core`, `NoI5`, matched Dense. Held-out accuracy every 100 steps through p1, and a 3× longer p1 run. Report per-seed values, medians, the share of seeds with held-out ≥ 0.5, and paired bootstrap intervals. | Tells whether the 0.25 vs 0.51 gap is anything beyond bimodal seeds, and whether longer training closes it. |
| **D7** | **GPM exemption off** (`exempt_readout_heads=False`) with per-head basis saturation logged, not only the cap count. | Tests §4.4 of the v8 report. |
| **D8** | **Routing specialization:** adjusted mutual information against operation labels on Track 1 and the functional-role texts, with a permutation null; per-expert ablation matrix (disable one expert, measure per-operation accuracy drop). | Replaces `func_routing_nmi`, which `FrozenRandomRouter` beats. |
| **D9** | **Metric hygiene:** a near-OOD abstain set (valid-looking but unseen operations, shifted vocabulary) so AUROC is not 1.0; paired per-seed differences for Banking77; burst reported as counts. | Removes the saturated and tied metrics from the headline. |
| **D10** | **Steps scaling for R@5:** Track 2 at 250 / 500 / 1000 steps per domain for `Cog_v8` and `SequenceTop1`. | Separates a weak objective (flat curve) from short training (rising curve). |
| **D11** | **Profile one Track 2 step** with `torch.profiler`. | Shows whether the step is launch-bound or compute-bound before any efficiency work. |

**Gate to Stage 1.** Stage 1 starts only when D1, D2 and D3 are done and D6 has run to 20 seeds, because every Stage 1 target depends on them.

## 7. Stage 1: interventions (each gated on a Stage 0 outcome)

| ID | Intervention | Runs only if | Proposed bar (frozen after Stage 0) |
|---|---|---|---|
| **H1a** | **Constant Fisher scale in Track 1.** The consistency gate sets `fisher_sc = 0.25 + 1.15·acc_batch` (line 1021), which raises consolidation as training accuracy rises and may lock a memorized solution before generalization emerges. Hypothesis from reading the code, not a finding. | D6 shows held-out accuracy gains with longer training or without I10 | Held-out median ≥ best ablation's median, interval excludes 0 |
| **H1b** | **Learned-rotation interchange** instead of the fixed `[0:D/2]` / `[D/2:3D/4)` split (988–995), with counterfactual targets drawn over held-out values as well. | D5 shows held-out interchange near chance | Held-out interchange ≥ 0.5 with seen interchange ≥ 0.9 |
| **H1c** | **Larger Track 1 world:** more values and operations, depth-2 compositions, held out by value and by depth, not only by pair. | D6 shows the 16-value table is memorizable or bimodal | Same bar on the new task; v8 task kept for continuity |
| **H2** | **Compute-aware routing:** a penalty targeting mean k ≤ 1.0, and the world-model target taken from a router-invariant path. | D3 shows the routing effect (not escalation) causes the R@5 and Banking77 gaps | At mean k ≤ 1.0: Banking77 accuracy ≥ `SequenceTop1` − 0.02, R@5 ≥ 0.254 |
| **H3** | **InfoNCE over future-chunk embeddings** with same-domain hard negatives, plus an effective-rank diagnostic of `z_expr`. | D10 shows a flat R@5 curve | R@5 > byte histogram (0.2375) with interval |
| **H4** | **Edit-family diversity:** 4 training families → at least 16, at least 4 held out, leave-one-family-out. | D2 shows held-out family AUC does not move with the objective | Held-out AUC beats matched Dense by ≥ 0.05 with interval |
| **H5** | **Efficiency**, chosen from the D11 profile (`torch.compile` or CUDA graphs on the fixed-shape step, GPM projection every N steps, interchange as one concatenated batch). | D11 shows launch-bound | Track 2 ≤ 220 s at unchanged metrics; also report FLOPs and time-to-target |

## 8. Falsifiers

| Item | Kills the claim if |
|---|---|
| V9-F1 (valid controls) | A matched control still fails to fit the Track 1 seen set; its comparisons are dropped, not rescued by re-tuning only the Cog arms. |
| V9-F2 (I5 helps generalization) | The paired interval for held-out `Cog` − `NoI5` includes 0 at 20 seeds. |
| V9-F3 (episodic induction) | The induction probe (D4c) is at chance while the lookup probe (D4a/b) is high. |
| V9-F4 (dynamic k earns compute) | In the 2×2, the dynamic-k cell loses to the static cell at equal escalation setting on accuracy and does not save compute. |
| V9-F5 (retrieval signal) | R@5 does not exceed the byte histogram at 4× steps, and the curve is flat. |
| V9-F6 (functional routing) | Per-expert ablation does not drop one operation more than the others, or AMI does not beat its permutation null. |

## 9. Open decisions for you

1. **Which Track 1 task is the headline?** Keep the v8 task for continuity (cheap, comparable) and add the larger one only after Stage 0, or move to the larger one immediately.
2. **Seed budget.** I propose 20 seeds on Track 1 only (cheap) and 10 on Track 2 for the four decisive arms. This is roughly 12 minutes per arm-seed on Track 2 by the v8 timestamps.
3. **`research-protocol.md`.** Please confirm the tier rules, since the file is not in the repo.

## 10. Risks

- Stage 0 may show that I5 contributes nothing to held-out accuracy. That is a useful result but changes the narrative in the v8 report.
- Unbundling needs three flags in `real_model.py`, which has 2,131 uncommitted changed lines. The flags need tests that the default configuration is unchanged.
- The repo has no commit since 2026-11-16 and `results/` is not ignored. A commit before Stage 0 would make the baseline reproducible.
- I read the code paths cited above, not the whole of `real_model.py`; the list of sites gated by `use_v8_core` is from a search, not a full read.

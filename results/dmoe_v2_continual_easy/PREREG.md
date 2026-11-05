# Pre-Registration: `dmoe_v2_continual_easy` (learnable-task re-run of the matched-baseline comparison)

Written BEFORE any run of the new task. This is a NEW experiment ID. It does not replace or edit
`results/dmoe_v2_continual/` (saturated at 1.0000) or `results/dmoe_v2_continual_hard_control/`
(falsifier F2 fired: dMoE-NoFM 0.1193 < Regular+Replay 0.1370; Regular+Replay+FM 0.1512 > dMoE v2 Full 0.1433;
all models far below the ~0.95 ceiling, i.e. the task was not learnable at that budget). Both stay on record.

## 1. Why this experiment
The hard-control task was too hard for any model at 60 steps/domain, so it could not separate architectures.
This experiment makes the task learnable by the *dense baseline* first (selection rule below, applied on
validation seeds only), then runs the full matched-baseline comparison once on disjoint test seeds.

## 2. Claims (each judged independently, each at the lowest supported level)
- **C1 (original user claim)**: dMoE_v2_Full has higher final avg byte accuracy than both
  `Baseline_RegularTransformer_Unfrozen` and `Baseline_RegularTransformer_FrozenBackbone`.
- **C2 (architecture)**: `Ablation_dMoE_NoFastMemory` (dMoE + replay + distillation + maturity freezing)
  has higher final avg byte accuracy than `Baseline_RegularTransformer_WithReplay` (same wrapper).
- **C3 (full system vs matched)**: dMoE_v2_Full has higher final avg byte accuracy than
  `Baseline_RegularTransformer_WithReplayAndFastMemory` (same wrapper + same fast memory).
- **Headline metrics**: final all-domain held-out byte accuracy (higher better, ceiling ~0.9531);
  BWT in nats (lower better), reported with the full absolute-loss matrix caveat from the diagnostic experiment.
- "Higher" means: difference of means > 0 AND 95% CIs (1.96*SD/sqrt(20)) do not overlap.

## 3. Falsifiers (fixed now)
- F1: any condition has avg acc > 0.97 -> leakage/bug; stop and report.
- F2: C2 not satisfied -> architecture claim REFUTED on this task (reported as such).
- F3: any null control (Untrained, BlindZeroInput, ShuffledTargets) has avg acc > 0.10 -> claim invalid.
- F4: C1 not satisfied -> original claim REFUTED on this task.
- F5: C3 not satisfied -> "full system beats matched baseline" REFUTED on this task.

## 4. Frozen task (`generate_easy_domain_batch` in `scripts/run_dmoe_v2_easy_eval.py`)
- Sequences of 16 bytes, chunk_size 4. Slot 0 of every chunk is a domain-prefix byte
  (A: {10,11}, B: {12,13}, C: {14,15}), slots 1..3 are payload bytes drawn from the SAME range [32..47] in all domains
  (overlapping vocabulary; no disjoint-vocabulary shortcut).
- Target at slot 0 is a function of the prefix; target at slots 1..3 is a domain-specific function of
  (payload byte, slot) over 16 values: A: (v+s)%16, B: (v xor (s+5))%16, C: (3v+7-s)%16, v = byte-32, output = 32+that.
  The domain must be read from the prefix in the same chunk (no cross-chunk dependency).
- 5% symmetric label corruption (replace by uniform byte in [32..47]); Bayes ceiling ~0.9531.
- Train batch seed = seed*10000 + domain*1000 + step; test batch (B=64) seed = seed*100000 + 95000 + domain
  (generator never reused between train and test).
- Optimizer AdamW lr 4e-3, batch 16, same for every condition. Replay capacity 128, replay batch 8, distillation 0.5,
  maturity_age 40, mutation interval 25, fast_memory_blend 0.55, fast_memory_reg 0.25 (unchanged from earlier experiments).
- Sequential domains A -> B -> C, `steps_per_domain` = N (selected below), identical for all conditions.

## 5. Tuning budget and selection rule (validation seeds 100..104 only; test seeds are 0..19)
- Only knob tuned: `steps_per_domain` N from the ladder {60, 120, 240, 480}, tried in increasing order. At most 4 configs.
- For each N: train the Regular Dense Transformer baseline (plain AdamW, no wrapper) on each domain A, B, C
  ALONE (fresh model per domain), measure held-out accuracy on that domain, average over the 3 domains and 5 validation seeds.
- Selection: N* = smallest N with that mean >= 0.80. If none reaches 0.80, the task is declared not learnable by the
  baseline at the declared budgets; STOP and report (no further tuning, no test run).
- dMoE single-domain validation accuracy at each N is logged for transparency but NOT used for selection.
- Every config tried is written to `results/dmoe_v2_continual_easy/tuning_log.json`.

## 6. Conditions (same 9 as hard-control)
NullControl_Untrained, NullControl_BlindZeroInput, NullControl_ShuffledTargets, Baseline_RegularTransformer_Unfrozen,
Baseline_RegularTransformer_FrozenBackbone, Baseline_RegularTransformer_WithReplay,
Baseline_RegularTransformer_WithReplayAndFastMemory, Ablation_dMoE_NoFastMemory, dMoE_v2_Full.
Baselines share the chunker front-end and the byte_chunk decoder readout with dMoE (see `RegularDenseTransformerMatched`).

## 7. Oracles / limitations declared in advance
- Synthetic rule-learning data; no real-data claim is possible from this experiment.
- Parameter counts are not exactly matched (recorded in results meta: dMoE vs dense).
- Compute is not matched (dMoE is slower per step); wall time is not a claim here.
- Fast memory features come from the same network being trained (stale-feature effect is studied in the separate
  diagnostic `dmoe_v2_fm_bwt_diagnostic`).
- 20 seeds, disjoint from tuning seeds. One frozen test run per final configuration.

## 8. Addendum written after tuning, BEFORE the test run
- Tuning result (results/dmoe_v2_continual_easy/tuning_log.json, seeds 100..104): N=60 dense single-domain mean acc = 0.8783 (>= 0.80) -> selected steps_per_domain N* = 60. Ladder stopped at the first rung.
- Logged but not used for selection: dMoE single-domain mean acc at N=60 = 0.6441 (dMoE learns slower than the dense baseline at this budget; the selection rule was fixed on the dense baseline only, so the frozen budget may disadvantage dMoE. Disclosed in advance of the test run.)
- Test run command: 4 parallel parts (seeds 0-5,5-10,10-15,15-20) with --steps 60, then --mode merge.

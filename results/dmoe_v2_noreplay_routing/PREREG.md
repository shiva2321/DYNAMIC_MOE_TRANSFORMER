# Pre-Registration: `dmoe_v2_noreplay_routing` (does dMoE detect novelty / route / spawn well enough to need no replay?)

Written BEFORE any run. New experiment ID; earlier experiments stay on record.
Task: the frozen `generate_easy_domain_batch` from `dmoe_v2_continual_easy` (overlapping payload vocabulary, domain read from the
per-chunk prefix, 5% label noise, Bayes ceiling ~0.953). Optimizer/hyperparameters unchanged from earlier experiments
(AdamW lr 4e-3, batch 16, maturity_age 40, mutation interval 25, FM blend 0.55, FM reg 0.25, replay batch 8, distillation 0.5,
replay capacity 128 only in the two explicit replay conditions).

## Two arms, both declared now (neither selected after seeing results)
- Arm N=60 steps/domain (same budget as `dmoe_v2_continual_easy`).
- Arm N=240 steps/domain (4x budget, because dMoE reached only 0.6441 single-domain at N=60 in validation, vs 0.8783 dense).
Every claim below is judged separately per arm. Seeds 0..19 (disjoint from validation seeds 100..104). One frozen run per arm.

## Questions, measures and fixed pass criteria
All "CI" are 95% (1.96*SD/sqrt(20)) on per-seed paired differences unless stated.
- **Q1 novelty detection.** Per seed, novelty jump = mean(novelty[b:b+5]) - mean(novelty[b-10:b]) averaged over true domain
  boundaries b in {N, 2N}, from the `ContinualLearner` step output (`novelty_score`) of `dMoE_NoReplay`. Null: the same statistic at the
  same global steps in `Control_Stationary_dMoE` (domain A only, 3N steps). PASS if mean paired difference > 0 and CI excludes 0.
- **Q2 spawn timing.** Experts added (summed over layers) in the 50 steps after each boundary, from the expert-count trace at mutation
  checks (count after the last check with step <= g). Paired difference vs Stationary control. PASS if mean > 0 and CI excludes 0.
  Also reported: total experts added, spawn-trigger count, layers at end.
- **Q3 routing specialization.** On held-out test batches after training, from router logits (forward hooks on each layer's router):
  (a) leave-one-out nearest-centroid domain identification accuracy from per-sequence mean routing probabilities (chance 1/3),
  averaged over layers; (b) Jensen-Shannon divergence between domain-mean routing distributions vs split-half noise floor within a domain.
  PASS (a): trained dMoE_NoReplay mean accuracy > 0.5 AND paired difference vs the untrained model's routing has CI excluding 0.
  PASS (b): paired difference (between-domain JSD - split-half JSD) > 0 with CI excluding 0.
- **Q4 "does not need replay" (absolute criteria, not relative to any competitor).** For dMoE_NoReplay and dMoE_NoReplay_FM:
  mean retention ratio of Domain A = acc_A_final / acc_A_after_A >= 0.80 AND mean BWT <= 0.10 nats AND mean avg final accuracy >= 0.80.
  All three must hold.
- **Q5 vs matched dense with no replay.** avg_acc_final of dMoE_NoReplay minus Regular_NoReplay: PASS if mean > 0 and CIs non-overlapping.
- **Q6 is replay still needed?** avg_acc_final(dMoE_Replay) - avg_acc_final(dMoE_NoReplay) paired; reported either way. Same for dense.
- **Q7 do spawn/prune/freeze matter?** avg_acc_final(dMoE_NoReplay) - avg_acc_final(dMoE_NoReplay_NoMutation) paired; reported either way.

## Falsifiers / validity gates (fixed now)
- F1: any condition avg acc > 0.97 -> stop, leakage/bug.
- F2: any null control (Untrained, ShuffledTargets, BlindZeroInput) avg acc > mean per-seed majority-class accuracy + 0.05
  (majority-class accuracy computed per seed by the script from the held-out labels, per-domain mode) -> claims invalid for that arm.
  (This replaces the 0.10 limit in `dmoe_v2_continual_easy`, which was too tight because of non-uniform label marginals; that earlier
  F3 firing stays on record.)
- Q1..Q7 failing is reported as failing; no threshold, seed, metric or budget changes after seeing results.

## Conditions (all trained sequentially A->B->C unless noted; no replay unless named)
NullControl_Untrained (routing metrics only meaningful), NullControl_ShuffledTargets (dMoE+FM, shuffled labels),
NullControl_BlindZeroInput (dMoE_NoReplay_FM model evaluated on zero input), Reference_MajorityClass (per-domain mode accuracy),
Reference_Dense_JointMultitask (interleaved domains, same total steps; learnability ceiling reference, not a competitor),
Regular_NoReplay, Regular_NoReplay_FM, dMoE_NoReplay, dMoE_NoReplay_FM, dMoE_NoReplay_NoMutation (mutation interval 1e9: no spawn/prune/freeze),
Regular_Replay, dMoE_Replay (capacity 128, no FM), Control_Stationary_dMoE (domain A only, 3N steps, no replay).

## Declared limitations / oracles
- Domain identity is not given to any model (no task id), but the domain is separable from the prefix byte, so routing by domain is possible in principle.
- Synthetic tasks, 3 domains. Parameter counts and compute not matched (dMoE 3 attention/MoE stacks vs dense encoder).
- The novelty detector, spawn policy and freeze policy are exactly the code in `dmoe/` at the commit recorded in results; none of the
  research-repo routing methods (SetGate, LdaScorer, quarantine, per-expert closed-form heads) are in this experiment.
- Hyperparameters of dMoE were not tuned for this task (same as earlier experiments); a negative result may reflect that.


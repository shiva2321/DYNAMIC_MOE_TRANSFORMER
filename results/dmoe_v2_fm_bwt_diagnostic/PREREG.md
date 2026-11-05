# Pre-Registration: `dmoe_v2_fm_bwt_diagnostic` (why did BWT rise when fast memory was added?)

Diagnostic, not a performance claim. Written before running. Runs on the `dmoe_v2_continual_hard_control` task
(`generate_hard_domain_batch`, 60 steps/domain, same wrapper settings) because that is where the observation was made:
dMoE NoFM BWT +0.1549 -> dMoE Full +0.7614, Regular+Replay +0.1068 -> Regular+Replay+FM +0.6343 (while accuracy rose).

## Question
Is the rise in BWT with fast memory (FM) (a) a metric artifact (FM makes the just-learned loss lower, so the same final
loss yields larger BWT), (b) caused by stale features (FM statistics absorbed online with a drifting backbone), or
(c) caused by a mis-set blend weight (0.55)?

## Setup (frozen)
- Conditions: `dMoE_v2_Full` and `Regular+Replay+FM` (same ContinualLearner settings as the hard-control experiment).
- Seeds 0..19. All randomness seeded identically to the hard-control script.
- After each domain t in {A,B} and at the end, evaluate each already-seen domain on held-out test batches with THREE readouts
  computed on the same model state: slow-only (decoder logits), fast-only (log_softmax of FM logits), blend (w=0.55, = what
  the experiment reported). A consistency check asserts the manual blend equals `learner.predict_logits`.
- Record the absolute loss matrix and accuracy, and BWT under each readout:
  BWT_r = 0.5*((L_end,A,r - L_afterA,A,r) + (L_end,B,r - L_afterB,B,r)).
- Stale-feature test: after training, build a fresh FM and absorb ALL training batches (regenerated from the same seeds,
  training data only) using the FINAL network features ("refit FM"). Compare final held-out loss/acc of refit FM against the
  online FM. Also record feature drift of a fixed domain-A probe batch: ||f_end - f_afterA||_F / ||f_afterA||_F.
- Blend sweep (diagnostic only; NOT used to choose any headline configuration): final avg loss/acc for w in
  {0, 0.25, 0.55, 0.75, 1.0} with the online FM.

## Pre-declared interpretation rules
- H1 artifact: supported if mean final blend loss <= mean final slow-only loss AND mean BWT_blend > mean BWT_slow.
- H2 stale features: supported if refit-FM final loss is lower than online-FM final loss by more than the 95% CI of the
  paired difference AND mean drift > 0.
- H3 blend weight: supported if the loss-minimizing w in the sweep differs from 0.55 by a CI-separated margin.
- Any hypothesis may be supported, several, or none; every outcome is reported. No config change follows from this run
  without a new PREREG.

## Limitations declared in advance
- Diagnostic of one task and one wrapper setting. Causal claims limited to "observed here".
- Blend sweep and refit use test-batch evaluation only to describe behavior; they are not tuning.


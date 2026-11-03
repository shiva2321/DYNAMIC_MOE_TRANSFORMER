# Phase 2b-ii: Adaptive Internal Simulation Loop — Design Spec

Date: 2026-11-04
Status: Approved by user, ready for implementation planning

## Context

Phase 1 hardened the base architecture
(`docs/superpowers/specs/2026-11-04-dynamic-moe-phase1-hardening-design.md`).
Phase 2a added continual learning without catastrophic forgetting
(`docs/superpowers/specs/2026-11-04-dmoe-phase2a-continual-learning-design.md`),
now complete. Phase 2b-i added a JEPA-style latent self-supervised objective
(`docs/superpowers/specs/2026-11-04-dmoe-phase2b-i-jepa-latent-objective-design.md`),
currently being executed in parallel as this spec is written.

This spec covers **Phase 2b-ii**: an adaptive internal-simulation loop —
"thinking more" before answering by re-running part of the network for
extra iterations, with the amount of extra compute determined by the
input's difficulty rather than fixed in advance. This was the original
"inference-time self-correction loop" deferred all the way back in the
Phase 1 spec, now substantially better specified after dedicated research
into looped/recurrent-depth transformers.

### Why this design, not the first sketch

The original idea floated earlier in this project's brainstorming — map
Phase 2a's novelty z-score directly to a loop count `K` via thresholds —
turned out not to hold up against the actual research literature, and was
revised twice during this brainstorming session:

- **Looped/weight-tied transformers have a well-documented stability
  failure mode.** Naive unrolling is mathematically equivalent to a very
  deep RNN and causes exploding/vanishing activations or representational
  collapse (Fixed-Point Reasoners, 2026; Universal Transformer literature).
  The fix used in that research — pre-norm ordering plus a learned,
  bounded residual-scaling ("damping") factor on each iteration's update —
  turns the loop into a contraction toward a fixed point (Banach
  fixed-point theorem) rather than letting it diverge. This design adopts
  that fix directly.
- **Models don't generalize to more test-time loop iterations than they
  saw in training "for free."** Per recurrent-depth transformer
  generalization studies found during this phase's research (e.g.
  "Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with
  Depth-Recurrent Transformers," arXiv:2603.21676), accuracy holds to
  roughly 2x the trained max depth before collapsing. This means the loop
  must be *trained with* variable depth, not just switched on at inference
  — so this phase trains with a randomly sampled iteration count each
  step, capped at the same `max_simulation_steps` used at inference.
- **The halting decision (how many iterations) is better grounded in the
  model's own convergence signal than a novelty lookup table.** Energy- and
  confidence-based halting (stop once a "how far from done" signal is low
  or stops improving) is an established, cited alternative to a fully
  learned PonderNet-style halting network, and — the key insight — **Ω_corr's
  own `structural_error` already *is* exactly that energy signal.** No new
  learned halting network is needed; the loop stops when Ω_corr itself
  reports convergence. Phase 2a's novelty score is downgraded to an honest,
  smaller role: a cheap skip-hint ("don't bother looping on a clearly
  familiar input"), not a direct control of how many iterations to run.
- **Opt-in, not default-on.** An earlier pass of this design considered
  making the loop the model's default behavior. That was rejected once the
  blast radius was made explicit: it would change `forward()`'s behavior
  (and therefore training dynamics) for every existing Phase 1/2a/2b-i
  model and test, including the ablation-style comparisons in Phase 2a's
  retention test and Phase 2b-i's probe-comparison test, which would all
  need re-verification and likely re-tuning. An opt-in constructor flag,
  default `False`, gives identical behavior to today whenever it's unused —
  zero risk to anything already built.

## Goals

- Add an opt-in simulation loop to `SelfStabilizingDynamicMorphicMoE`,
  re-running `manifold_corr` and the last transformer layer for extra
  iterations, with no changes to `layer.py` or `manifold.py`.
- Stabilize the loop with a learned, bounded damping factor (under-relaxed
  fixed-point update), created only when the feature is enabled.
- Train with randomized iteration depth (so the model learns to use
  whatever depth it's given); infer with energy-based halting driven by
  Ω_corr's own structural error (so compute is spent where it's needed).
- Use Phase 2a's novelty score only as a cheap skip-hint, not a depth
  controller.
- Prove the stability fix actually works with a dedicated test forcing far
  more iterations than the trained cap and checking outputs stay bounded —
  not just an assertion that the design should be stable.

## Non-goals (this phase)

- Making the loop default-on — deferred, pending evidence from this
  phase's own tests that it's worth the training-dynamics disruption
  documented above.
- A learned (PonderNet-style) halting network — the energy-based criterion
  using Ω_corr's existing signal is this phase's chosen mechanism; a
  learned halting head is a possible future upgrade if the energy-based
  criterion proves insufficiently calibrated, not required here.
- Integration with Phase 2a's `ContinualLearner` or Phase 2b-i's
  `JEPAModule` — both are separate, additive mechanisms; this phase doesn't
  wire them together.
- Any claim this constitutes "reasoning" in a deep sense — it's additional
  test-time compute allocated adaptively, backed by cited looped-transformer
  research, not a claim about cognition.

## Architecture

All changes are in `dmoe/model.py`, specifically `SelfStabilizingDynamicMorphicMoE`.

### Constructor additions

```
enable_simulation_loop: bool = False
max_simulation_steps: int = 4
theta_converge: float = 0.1
min_improvement: float = 0.01
skip_threshold: float = 0.0
```

When `enable_simulation_loop` is `False` (the default), none of the above
have any effect and no new parameters are created — `forward()` behaves
exactly as it does today. When `True`, the constructor additionally
creates `self.loop_damping_logit = nn.Parameter(torch.tensor(0.0))`; the
actual damping factor used at runtime is `torch.sigmoid(self.loop_damping_logit)`,
always in `(0, 1)` by construction.

### The loop

Runs after the existing layer stack has produced `(z_r, z_i)` as it does
today, only if `enable_simulation_loop` is `True`:

1. **Skip-hint**: if `self.manifold_corr.last_novelty_score.item() <=
   skip_threshold`, skip the loop entirely — `simulation_steps_used = 0`,
   proceed straight to the decoder with the existing `(z_r, z_i)`.
2. Otherwise, determine the iteration budget:
   - **Training mode**: `steps = random.randint(1, max_simulation_steps)`,
     run exactly that many iterations unconditionally (no early stopping —
     the model needs to experience and learn from every depth in this
     range, per the "train with variable depth" finding above).
   - **Eval mode**: iterate up to `max_simulation_steps`, but stop early
     once `structural_error < theta_converge` or
     `(previous_error - structural_error) < min_improvement`.
3. **Each iteration**:
   ```
   corrected_r, corrected_i, structural_error, novelty_score = self.manifold_corr(z_r, z_i)
   candidate_r, candidate_i, _ = self.layers[-1](corrected_r, corrected_i)
   damping = torch.sigmoid(self.loop_damping_logit)
   z_r = z_r + damping * (candidate_r - z_r)
   z_i = z_i + damping * (candidate_i - z_i)
   ```
   This is the under-relaxed fixed-point update: blending toward the
   candidate rather than replacing the state outright, which is what keeps
   the iteration a contraction instead of a divergent process.
4. `forward()`'s returned dict gains `"simulation_steps_used"` (an `int`,
   `0` when the flag is off or the skip-hint fired, otherwise the number
   of loop iterations actually run) — additive, alongside the existing
   keys including Phase 2a's `"novelty_score"`.

## Testing plan

`tests/test_simulation_loop.py`:

1. **Regression guard**: a model built with `enable_simulation_loop=False`
   (the default) has no `loop_damping_logit` parameter, and its `forward()`
   output and `"simulation_steps_used" == 0` are unaffected by this phase's
   changes — proves the opt-in design actually delivers zero risk when
   unused.
2. **Training-mode randomized depth**: with the flag on, many repeated
   training-mode forward calls on the same model produce step counts that
   vary and stay within `[1, max_simulation_steps]`.
3. **Eval-mode energy-based halting**: one case with `structural_error`
   already below `theta_converge` (halts after exactly 1 iteration) and one
   case where it stays above threshold and improves negligibly (runs to
   `max_simulation_steps`).
4. **Skip-hint**: a case with the novelty score at or below
   `skip_threshold` produces `simulation_steps_used == 0` even with the
   flag enabled.
5. **Stability under forced excessive iterations**: force a much larger
   iteration count than any trained cap (e.g. 20) directly and assert the
   resulting latent state has no `NaN`/`Inf` and stays within a bounded
   magnitude — concrete evidence the damping mechanism prevents the
   divergence this design exists to avoid.
6. **Gradients flow** through the damped loop during a training-mode
   forward+backward with the flag enabled, with no `NaN` gradients,
   including on `loop_damping_logit` itself.

## Open questions carried forward

- Whether `theta_converge`/`min_improvement`'s defaults need tuning once
  real multi-step training is observed — flagged as tunable, not fixed.
- Whether this should eventually become default-on, once Phase 2a's and
  Phase 2b-i's tests have been re-verified against it deliberately (not as
  a side effect of this phase).
- Whether a learned halting network (PonderNet-style) is worth the added
  complexity if the energy-based criterion proves too coarse in practice.

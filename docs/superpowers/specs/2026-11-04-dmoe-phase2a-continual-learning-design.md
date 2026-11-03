# Phase 2a: Continual Learning Without Catastrophic Forgetting — Design Spec

Date: 2026-11-04
Status: Approved by user (architecture confirmed; Option C pinned as future stretch goal), ready for implementation planning

## Context

Phase 1 hardened the SS-DM-MoE architecture into a tested `dmoe/` package (see
`docs/superpowers/specs/2026-11-04-dynamic-moe-phase1-hardening-design.md`).
That spec deferred three things to "Phase 2": a continuous learning loop,
novelty detection feeding expert spawning, and an inference-time
self-correction loop.

This spec covers **Phase 2a** only: continual learning that specifically
resists catastrophic forgetting, plus novelty-aware expert spawning. The
user was explicit that "continual learning" means the model keeps learning
new things over time *without* forgetting or degrading on older things —
not just "trains forever." Phase 2b (an inference-time self-correction /
"internal simulation" mechanism, now informed by JEPA-style latent
prediction and looped-transformer research) is deferred to its own
brainstorming pass and spec.

### Why this design, not a simpler one

Three options were considered:

- **Dynamic expert growth alone** (no replay, no regularization) — rejected.
  The shared backbone (encoder, attention, Ω_corr, decoder) has no
  protection and drifts as new data arrives regardless of expert isolation.
- **Dynamic growth + EWC-style regularization** (Fisher-information penalty
  on parameter drift) — rejected as the primary mechanism. The Dark
  Experience Replay literature found EWC-style consolidation underperforms
  replay-based methods specifically in task-free / streaming settings, which
  is closer to this project's situation than clean task-boundary benchmarks.
  Also heavier to tune (Fisher estimation, extra hyperparameters).
- **Dynamic growth + reservoir replay with distillation anchors (Dark
  Experience Replay style), plus expert-maturity freezing** — **chosen**.
  This targets the three distinct forgetting channels this specific
  architecture has: shared-backbone drift (fixed by replay), router drift
  (fixed by replay touching the router's gradient signal for old patterns
  too — a failure mode specifically documented in MoE-continual-learning
  literature, where a shared router's decision boundaries forget as new
  tasks overwhelm it), and expert-content drift (fixed by freezing mature
  experts so new patterns go to fresh capacity instead of overwriting old).

A fourth option — adding an EWC-style penalty specifically on the router's
weights, on top of the chosen design, as a second independent safeguard
against router forgetting — is **pinned as a documented future stretch
goal**, not required for Phase 2a. It is the right next increment if the
retention test (below) shows router drift is still a problem after the
chosen mechanism ships.

No mechanism in the continual-learning literature guarantees zero forgetting
for arbitrary data forever — it remains an open research problem. This
design is a principled, citable, testable best-effort covering this
architecture's known failure modes, validated by a real measured retention
test rather than an unverified claim.

## Goals

- Add a bounded, reservoir-sampled replay buffer with distillation anchors
  (Dark Experience Replay style) protecting the shared backbone and router.
- Add expert-maturity tracking and freezing so settled experts stop
  absorbing gradient updates from new, unrelated data.
- Extend Ω_corr's error tracking from a bare EMA mean to mean+variance, so
  there's a real novelty z-score instead of a fixed magnitude threshold.
- Use the novelty z-score to open a second, independent expert-spawning
  trigger path (novel-and-confused), without breaking Phase 1's existing
  entropy-only trigger or its 50 passing tests.
- Build a `ContinualLearner` with a `step()` API that composes all of the
  above into one training step.
- Prove the mechanism actually mitigates forgetting with a real retention
  test: train on task A, train on task B, verify A doesn't collapse, and
  compare against a no-mitigation ablation baseline.

## Non-goals (this phase)

- The EWC-style router-specific safeguard (pinned, see above).
- Phase 2b: JEPA-style latent self-supervised objective and adaptive
  internal-simulation/looped-reasoning — separate spec, separate cycle.
- Multimodal retention testing (image/numeric) — the mechanism is
  modality-agnostic by construction (it operates on the latent
  representation via the decoder's `"latent"` head), so the retention test
  proves the mechanism works once, on text; extending empirical validation
  to other modalities is future work, not blocking this phase.
- Real datasets — the retention test uses synthetic, rule-based tasks
  (learnable, controllable, no external dependencies), consistent with
  Phase 1's approach and this still being a research/learning project.

## Architecture changes

### 1. Novelty z-score (`dmoe/manifold.py`)

`GeometricCorrectiveManifold` currently tracks only `running_error_magnitude`
(an EMA mean). Add `running_error_variance`, updated with the same momentum
in the standard online-EMA-variance form:

```
running_mean  <- momentum * running_mean  + (1 - momentum) * batch_mean_error
running_var   <- momentum * running_var   + (1 - momentum) * (batch_mean_error - running_mean)^2
novelty_score = (batch_mean_error - running_mean) / (sqrt(running_var) + eps)
```

`forward()` changes its return signature from
`(corrected_r, corrected_i, batch_mean_error)` to
`(corrected_r, corrected_i, batch_mean_error, novelty_score)`. This is a
deliberate breaking change to the Phase 1 interface; `model.py` and the
Phase 1 tests that unpack this return value are updated as part of this
phase's implementation.

### 2. Expert maturity and freezing (`dmoe/experts.py`)

Add an `expert_age` buffer to `DynamicMorphicMoE`, parallel to
`expert_usage`, incremented by one for every expert on every training-mode
forward pass. A new `maturity_age` constructor parameter (expert ages past
this threshold are eligible for freezing). Add a method,
`freeze_mature_experts()`, called periodically by `ContinualLearner`, that
calls `.requires_grad_(False)` on every parameter of any expert whose age
exceeds `maturity_age` and isn't already frozen (tracked via a parallel
`expert_frozen` buffer of booleans, so the check is O(1) per expert rather
than re-inspecting `requires_grad` on every parameter each time).

No optimizer rebuild is needed for freezing: PyTorch optimizers skip
parameters with `requires_grad=False` (their `.grad` stays `None`)
automatically. Freezing only stops gradient updates — death/pruning via
`tau_death` still applies normally to frozen experts, since an unused
expert (frozen or not) is still a candidate for reclaiming capacity.

Newly spawned experts (via `clone_with_mutation`) always start at age 0 and
unfrozen, regardless of whether their parent was frozen — cloning copies
weight *values*, not frozen state.

### 3. Novelty-aware spawning (`dmoe/experts.py`, `dmoe/model.py`)

`DynamicMorphicMoE.mutate_topology` gains an optional `novelty_score: float
= None` parameter. When `None` (the default), behavior is byte-for-byte
identical to Phase 1 — this keeps all 50 existing Phase 1 tests valid with
no changes. When provided, a **second, independent** spawn trigger opens
alongside the existing entropy-only one:

```
spawn if entropy > tau_split                                   # Phase 1, unchanged
    or (novelty_score > tau_novelty and entropy > entropy_floor) # Phase 2a, additive
```

`entropy_floor` is a low bar (lower than `tau_split`) — the intent is "this
batch is surprising *and* the router isn't already confidently routing it
to one expert," not "surprising alone." `SelfStabilizingDynamicMorphicMoE.
execute_graph_mutation` gets the equivalent additive OR-condition for
vertical growth: the existing `current_error > theta_up` check is
unchanged, and a new `novelty_score > theta_novelty_vertical` check is ORed
in alongside it.

## Continual learning loop

### `dmoe/replay_buffer.py` — reservoir replay buffer

A `ReplayBuffer(capacity: int)` class storing plain Python tuples (not a
stacked tensor, since shapes vary across modalities):
`(x, targets, output_head, latent_snapshot)`. `add(item)` uses classic
reservoir sampling (Algorithm R): the first `capacity` items are kept
unconditionally; afterward, item number `n` (1-indexed, `n > capacity`)
replaces a uniformly-random existing slot with probability
`capacity / n`. This guarantees a uniform sample over the entire stream
history regardless of how long the stream has run, not just a sliding
window of the most recent items. `sample(k)` returns `k` random stored
items without replacement (or all items, if fewer than `k` are stored).

### `dmoe/continual.py` — `ContinualLearner`

```
ContinualLearner(
    model, optimizer,
    replay_buffer_capacity=500, replay_batch_size=8,
    distillation_weight=0.5,
    maturity_age=200,
    tau_novelty=2.5, entropy_floor=0.3, theta_novelty_vertical=2.5,
    mutation_check_interval=20,
)
```

`step(x, targets, output_head="token") -> dict` does, in order:

1. Forward the new batch through `model`, compute the task loss against
   `targets` (same loss function the caller would otherwise use).
2. If the replay buffer is non-empty, sample `replay_batch_size` past items.
   For each: forward it through `model` with its *original* `output_head`,
   compute (a) a rehearsal task loss against its stored `targets`, and (b)
   a distillation loss — mean-squared error between the model's *current*
   latent output (`model(x, output_head="latent")`) and the `latent_snapshot`
   stored when that item was added. Average these across the replay batch.
3. Combine: `total_loss = task_loss + rehearsal_loss + distillation_weight *
   distillation_loss`. One `total_loss.backward()` and one
   `optimizer.step()` — not separate backward passes per loss term, so
   gradients from all three contribute to the same update.
4. Insert the new batch's samples into the replay buffer via
   `ReplayBuffer.add`, storing a fresh latent snapshot
   (`model(x, output_head="latent").detach()`) computed *before* this
   step's optimizer update (i.e., the snapshot reflects the model's
   knowledge at the time the sample was seen, which is what distillation is
   meant to anchor against).
5. Increment a global step counter. Every `mutation_check_interval` steps:
   compute the novelty score from `model.manifold_corr`'s current running
   statistics, call `model.execute_graph_mutation(novelty_score=...)`, then
   call `freeze_mature_experts()` on every layer's `moe` (after incrementing
   `expert_age`, which happens every step regardless of the check interval).
6. Return a dict of metrics: `task_loss`, `rehearsal_loss`,
   `distillation_loss`, `total_loss`, `novelty_score`, and `mutation_report`
   (`None` on steps where the interval check didn't run).

Processing replay samples one at a time (rather than batching them, which
would require padding/grouping by shape) is acceptable at this project's
CPU/toy scale; batching replay samples by shape is a possible future
optimization, not required here.

## Validation: the retention test

`tests/test_continual_learning.py` defines two disjoint, rule-based
synthetic text tasks:

- **Task A:** input tokens uniformly sampled from `[0, 50)`;
  `target = (token + 1) % 50`.
- **Task B:** input tokens uniformly sampled from `[100, 150)`;
  `target = 100 + (token * 2) % 50`.

Both are trivially learnable (a model that's actually fitting the rule
drives cross-entropy down substantially from its random-init value) and
clearly distinct, so a model that forgets A while learning B will show it
concretely as A's held-out loss rising back toward its pre-training level.

The test:

1. Builds two models with identical architecture and the same random seed.
2. Trains "mitigated" with a real `ContinualLearner` on Task A for `N`
   steps; trains "baseline" directly with a plain optimizer loop (no
   replay, no freezing — the Phase 1 training pattern) on Task A for the
   same `N` steps and same data. Records both models' loss on a fixed,
   held-out Task-A evaluation batch.
3. Continues training both on Task B for `N` more steps (mitigated via
   `ContinualLearner.step`, baseline via the plain loop).
4. Re-evaluates both models on the same fixed Task-A evaluation batch.
5. Asserts: the mitigated model's Task-A loss after Task-B training is
   meaningfully lower than the baseline's Task-A loss after Task-B training
   (a direct, measured comparison — not an assertion that forgetting is
   "zero", since no mechanism promises that).

## Testing plan

- `test_manifold.py` (updated): novelty score shape/sign behavior — zero
  for a batch matching the running mean exactly, large magnitude for a
  batch far from recent normal variation; existing tests updated for the
  4-tuple return.
- `test_experts.py` (updated): `expert_age` increments only in training
  mode; `freeze_mature_experts` sets `requires_grad=False` on exactly the
  experts past `maturity_age` and leaves others trainable; a frozen
  expert's parameters don't change after an optimizer step but still
  receive forward-pass contributions; `mutate_topology` with
  `novelty_score=None` is unchanged from Phase 1 (regression guard); with a
  high `novelty_score` and moderate entropy, the new trigger path spawns
  even though `entropy <= tau_split`.
- `test_replay_buffer.py` (new): reservoir sampling keeps exactly
  `min(n_added, capacity)` items; a large-sample statistical check that
  early-stream items survive roughly as often as late-stream items (not
  biased toward recency); `sample(k)` returns `k` distinct stored items.
- `test_continual_learner.py` (new): one `step()` call runs without error
  and returns all expected metric keys; calling `step()` repeatedly
  populates the replay buffer up to its capacity; mutation checks only fire
  on `mutation_check_interval` boundaries.
- `test_continual_learning.py` (new): the retention test described above.

## Open questions carried forward

- Whether the pinned router-specific EWC safeguard (Option C) turns out to
  be necessary once the retention test is run for longer horizons or more
  task switches than this phase's test covers.
- Phase 2b's design (JEPA-style latent self-supervised objective + novelty-
  gated adaptive internal-simulation loop) is deferred to its own
  brainstorming pass, per the user's explicit sequencing choice.

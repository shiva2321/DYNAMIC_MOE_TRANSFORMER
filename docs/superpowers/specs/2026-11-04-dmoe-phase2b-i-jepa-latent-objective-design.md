# Phase 2b-i: JEPA-Style Latent Self-Supervised Objective — Design Spec

Date: 2026-11-04
Status: Approved by user, ready for implementation planning

## Context

Phase 1 hardened the SS-DM-MoE architecture (`docs/superpowers/specs/2026-11-04-dynamic-moe-phase1-hardening-design.md`).
Phase 2a added continual learning without catastrophic forgetting
(`docs/superpowers/specs/2026-11-04-dmoe-phase2a-continual-learning-design.md`,
plan at `docs/superpowers/plans/2026-11-04-dmoe-phase2a-continual-learning.md`);
its implementation is being executed in parallel by a different agent as
this spec is written.

The user asked for the model to go beyond pure next-token-style surface
prediction toward representations that capture meaning rather than surface
statistics, and explicitly asked for this to be grounded in real research
rather than invented ad hoc. Two directions were researched and cited in
this conversation:

- **JEPA (Joint Embedding Predictive Architecture)** — training a predictor
  to predict *latent representations* of masked/withheld content via an
  EMA target encoder, rather than predicting raw tokens. Forces abstraction
  over surface statistics because the predictor can ignore details that
  don't matter for the representation, which token-level prediction cannot.
- **Looped/recurrent latent reasoning** — re-running a shared block for
  extra iterations at inference time to simulate multi-step reasoning. This
  is **Phase 2b-ii**, a separate, later brainstorming/spec cycle (the user
  chose to sequence JEPA first since it's the foundation — the simulation
  loop is more interesting once there's a representation worth refining).

This spec covers **Phase 2b-i only**: the JEPA-style latent objective.

### Why this design, not a simpler one

- **Scope**: text-modality only, contiguous-span masking. Consistent with
  Phase 2a's text-only precedent; image/numeric JEPA (patch-based masking
  for images, segment masking for numeric streams) is future work, not
  blocking this phase — the mechanism is validated once, on text, and the
  encoder-wrapping design (below) means extending to other modalities later
  doesn't require touching this module's core logic.
- **Reusing `ComplexExpert` as the predictor** rather than inventing a new
  module: it is already exactly the right shape
  (`ComplexLinear → ModReLU → ComplexLinear`), and reuse avoids an
  unnecessary parallel implementation of the same thing.
- **EMA target encoder, not a shared/live one**: predicting your own live,
  gradient-updated representation is JEPA's best-documented failure mode —
  predictor and encoder can trivially co-adapt to a constant, collapsing
  the representation. The EMA target (updated only after each optimizer
  step, never by backprop) is the standard, citable fix.
- **No integration with Phase 2a's `ContinualLearner` in this phase.** The
  two mechanisms are orthogonal (JEPA changes what's learned about a single
  batch; `ContinualLearner` changes how learning composes across a stream
  of batches) and can be validated independently. Wiring the JEPA auxiliary
  loss into `ContinualLearner.step()` is a natural, explicitly deferred
  follow-up once both pieces exist and Phase 2a's current implementation
  has landed — attempting it now would mean touching files another agent is
  actively editing.
- **No claim of "understanding."** This objective swaps a surface-statistics
  training signal for an abstraction-forcing one, backed by cited research
  (I-JEPA, V-JEPA, LLM-JEPA). It does not mean the model understands meaning
  in any deep sense, and the validation plan measures a concrete, narrow
  proxy (does a downstream probe do better), not that claim.

## Goals

- Add a `JEPAModule` that wraps an existing `UniversalMultimodalEncoder`
  instance (no changes to `encoder.py` or any other Phase 1/2a file) and
  computes a masked-latent-prediction auxiliary loss.
- EMA target encoder with a documented, simple update rule, excluded from
  backprop.
- Validate: the auxiliary loss actually decreases with training, the
  mechanism doesn't collapse to a constant representation, and a downstream
  linear probe shows a measurable, concrete representational benefit over a
  task-loss-only baseline.

## Non-goals (this phase)

- Image/numeric-modality JEPA masking.
- Integration with `ContinualLearner` (Phase 2a) — deferred follow-up.
- Phase 2b-ii (adaptive internal-simulation loop) — separate spec.
- Any claim that this gives the model genuine semantic understanding —
  the validation plan measures a narrow, concrete proxy, not that claim.

## Architecture

### `dmoe/jepa.py` — `JEPAModule`

```
JEPAModule(encoder: UniversalMultimodalEncoder, predictor_hidden_dim: int,
           mask_token_id: int = 31999, mask_ratio: float = 0.3,
           ema_decay: float = 0.996)
```

- `self.context_encoder = encoder` — a **reference**, not a copy, to the
  caller's existing encoder instance (typically `model.encoder`). Its
  parameters are already part of whatever optimizer the caller built over
  `model.parameters()`; `JEPAModule` does not re-register or duplicate them.
- `self.target_encoder = copy.deepcopy(encoder)`, with
  `requires_grad_(False)` applied to every parameter immediately after. It
  starts as an exact copy of the context encoder's current weights and only
  ever changes via EMA, never via `.backward()`.
- `self.predictor = ComplexExpert(encoder.d_model, predictor_hidden_dim)` —
  the existing Phase 1 module, reused as-is.
- `mask_token_id` is reserved at the top of the embedding table (id
  `31999` of the encoder's 32000-row `nn.Embedding`); this is documented as
  reserved and must not be used as a real token id by any caller.

**`compute_loss(token_ids: torch.Tensor) -> torch.Tensor`** (text-only,
`token_ids` is `[B, S]` long):

1. Pick `mask_len = max(1, int(S * mask_ratio))` and a random contiguous
   start position; build `masked_ids` by cloning `token_ids` and
   overwriting that span with `mask_token_id`.
2. `context_r, context_i = self.context_encoder(masked_ids)`.
3. `with torch.no_grad(): target_r, target_i = self.target_encoder(token_ids)`
   — the **real**, unmasked sequence.
4. Slice both to the masked span; run the context slice through
   `self.predictor` to get `pred_r, pred_i`.
5. Return `F.mse_loss(pred_r, target_r[masked_span].detach()) +
   F.mse_loss(pred_i, target_i[masked_span].detach())`.

**`update_target_encoder() -> None`** (call once per training step, after
`optimizer.step()`): for each matched pair of context/target parameters,
`target_p.mul_(ema_decay).add_(context_p.detach() * (1 - ema_decay))`,
inside `torch.no_grad()`.

**`trainable_parameters()`**: returns `self.predictor.parameters()` — the
one thing a caller must explicitly add to their optimizer in addition to
`model.parameters()`. Calling `jepa_module.parameters()` directly would
also yield the (frozen, harmless but wasteful) target encoder and the
context encoder a second time (double-registration risk) — `trainable_parameters()`
exists specifically so callers never need to reason about that.

**Device movement**: callers must call `jepa_module.to(device)` in addition
to `model.to(device)`, since `target_encoder` is `JEPAModule`'s own
submodule and isn't reached by moving `model`. Moving `context_encoder`
twice (once via `model.to(device)`, once via `jepa_module.to(device)`,
since it's a submodule of both) is idempotent and harmless.

## Validation plan

`tests/test_jepa.py`:

1. **Loss decreases.** Construct a small `UniversalMultimodalEncoder` and
   wrap it in a `JEPAModule`. Train on structured synthetic text (the same
   rule-based data style as Phase 2a) for a modest number of steps,
   optimizing only `jepa_module.trainable_parameters()` plus the encoder's
   own parameters. Assert the loss at the end is meaningfully lower than at
   the start — not just finite.
2. **Anti-collapse.** After training, run the target encoder on several
   distinct random token sequences and assert the resulting latent vectors
   have non-trivial variance across inputs (e.g., pairwise distances are
   not all near zero) — a concrete, checkable guard against the classic
   JEPA failure mode.
3. **Downstream probe comparison.** Train two full models on Task A
   (`target = (token + 1) % 50`, reusing Phase 2a's rule) for the same
   number of steps and the same seed: model A with
   `total_loss = task_loss + jepa_loss`, model B with `total_loss =
   task_loss` alone. Freeze both encoders, attach a small trainable linear
   probe to each (pooled frozen-encoder output → predict the rule's output
   class directly, bypassing the rest of the model), train each probe
   briefly on fresh Task-A data, and assert model A's probe achieves lower
   loss on a held-out Task-A batch than model B's. This is a measured
   comparison on a narrow proxy, not a claim about general understanding —
   if it doesn't hold on the first run, that's genuine evidence to
   investigate (tune `mask_ratio`, `ema_decay`, or training steps), not a
   signal to weaken the assertion.

## Open questions carried forward

- Whether `ema_decay=0.996` (the typical large-scale JEPA value) is the
  right default here, or whether this project's much shorter training runs
  need a smaller decay to let the target encoder track meaningfully within
  the step counts used in tests — the implementer should treat this as a
  tunable, not a fixed constant, and the validation tests may need a
  smaller test-specific decay than the class's production-style default.
- Integration with `ContinualLearner` (feeding the JEPA auxiliary loss into
  its `step()`), deferred per the Non-goals section.
- Phase 2b-ii (adaptive internal-simulation loop), separate spec.

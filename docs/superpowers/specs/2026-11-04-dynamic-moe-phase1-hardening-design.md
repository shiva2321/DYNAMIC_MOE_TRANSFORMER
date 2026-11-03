# Phase 1: SS-DM-MoE Architecture Hardening — Design Spec

Date: 2026-11-04
Status: Approved by user, ready for implementation planning

## Context

The user wants to build a self-stabilizing dynamic morphic Mixture-of-Experts
(SS-DM-MoE) transformer: a model whose depth (layers) and width (experts per
layer) grow and shrink at runtime based on online structural-error and
routing-entropy signals, operating over a complex-valued (C^d) latent field,
and able to ingest text, image, and numeric-stream inputs.

This is a **research/learning project** (not aimed at a specific production
use case yet). A first draft already exists at `D:\Dynamic_MOE_Transformer\main.py`
(pasted by the user from an earlier AI-generated spec). It implements the
right shape of system — complex linear algebra, ModReLU, a multimodal
encoder, a geometric corrective manifold (Ω_corr), complex attention, a
dynamic MoE layer with split/death logic, a dynamic-depth transformer stack,
and a projective decoder — but it has real correctness bugs and no test
coverage, and lives in a single file with no package structure or git
history.

The full ambition (continuous/online learning, novelty detection driving
expert spawning, inference-time self-correction) is being deliberately
**decomposed into two phases**:

- **Phase 1 (this spec):** harden the existing architecture — fix bugs,
  restructure into a tested package, verify growth/shrink/routing mechanisms
  actually do what they claim.
- **Phase 2 (future spec, not covered here):** continuous learning loop,
  novelty/nuance detection feeding expert spawning and routing, and an
  inference-time predict-check-correct self-correction loop.

Building Phase 2 on top of unverified Phase 1 code would mean debugging two
layers of uncertainty at once, so Phase 1 must stand on its own: correct,
tested, and runnable on CPU today with no code changes required to run on
CUDA later ("use CPU and GPU as necessary").

## Goals

- Restructure the draft into a modular, testable package.
- Fix all identified correctness bugs (enumerated below) so that vertical
  (layer) and horizontal (expert) morphing are driven by real runtime
  signals, not silently-defaulted placeholders.
- Add a real pytest suite covering math correctness, shape invariants across
  mutation, gradient flow, and forced-threshold mutation behavior.
- Add topology-aware checkpointing and optimizer-state-preserving rebuild,
  since both are structural necessities of a model whose shape changes at
  runtime — not scope creep.
- Initialize git and get the project under version control.

## Non-goals (deferred to Phase 2)

- Continuous / online learning loop (streaming data, never-ending training).
- Novelty/nuance detection beyond what Ω_corr's structural error and the
  router's entropy already provide.
- Inference-time self-correction loop (predict → check → correct → answer).
- Real datasets / data pipelines — Phase 1 testing uses synthetic tensors
  only, matching the original validation harness's approach.
- Autoregressive generation (causal masking, KV cache) — the model is
  encoder-style for now.

## Architecture: package layout

```
D:\Dynamic_MOE_Transformer\
  dmoe/
    __init__.py
    complex_ops.py     # ComplexLinear, ModReLU, ComplexLayerNorm
    encoder.py           # UniversalMultimodalEncoder
    manifold.py           # GeometricCorrectiveManifold (Ω_corr)
    attention.py           # ComplexMultiHeadAttention
    experts.py               # ComplexExpert, DynamicMorphicMoE
    layer.py                  # DynamicMorphicTransformerLayer
    decoder.py                 # UniversalProjectiveDecoder
    model.py                    # SelfStabilizingDynamicMorphicMoE
    checkpoint.py                 # topology-aware save/load
  tests/
    test_complex_ops.py, test_encoder.py, test_manifold.py,
    test_attention.py, test_experts.py, test_layer.py,
    test_model_mutation.py, test_end_to_end.py
  scripts/
    validate.py    # human-readable stress-test trace harness
  main.py            # thin entry point calling scripts.validate.run()
  requirements.txt     # torch, pytest
  .gitignore
  docs/superpowers/specs/   # this file and future specs
```

Each module matches a subsystem already implicit in the original spec
(encoder / Ω_corr / vertical morph / horizontal morph / decoder), so the
mapping from "requirement" to "file" stays obvious.

## Bug fixes (the substance of this phase)

1. **Horizontal morphing never sees real entropy.** In the draft,
   `DynamicMorphicMoE.forward()` computes `entropy` as a local variable and
   returns it, but `execute_graph_mutation()` reads `layer.moe._last_entropy`,
   which is never set anywhere except by test code poking it manually. Fix:
   `forward()` stores `self._last_entropy = entropy.detach()` every call, so
   split/death decisions react to actual routing behavior with no manual
   intervention required.

2. **Numeric-stream ingestion collapses to rank-1.** The draft's "adaptive"
   projection for `[B, T, F]` inputs is
   `torch.ones((F_dim, d_model)) / sqrt(F_dim)` — a fixed, non-learnable,
   all-ones matrix. Every output dimension receives the identical value (the
   mean of the input features), so the model cannot distinguish feature
   patterns in numeric streams at all. Fix: a real learnable projection that
   adapts to whatever `F_dim` shows up (e.g. a small cache of `nn.Linear`
   layers keyed by `F_dim`, created lazily and reused), so distinct numeric
   streams with the same `F_dim` share a trainable projection instead of a
   fixed one.

3. **Buffer re-registration on expert add/remove.** The draft calls
   `self.register_buffer("expert_usage", new_usage)` repeatedly as experts
   are added/removed. This needs a correctness pass to confirm no stale
   tensors, device mismatches, or broken persistence survive repeated
   mutation. Fix: an explicit helper (`_replace_buffer(name, tensor)`) with a
   test that mutates an expert set a dozen times and checks the buffer is
   always the right shape, dtype, and device, with no leaked old tensors
   (checked via a weakref or just id() churn + shape assertions).

4. **Fragile token-dtype detection.** `x.dtype in (torch.int32, torch.int64)`
   misses other valid integer dtypes. Fix:
   `x.dim() == 2 and not x.is_floating_point()`.

5. **No device-consistency guarantee under mutation.** New experts, new
   layers, new router matrices, and new usage buffers must land on the same
   device as the rest of the model. Fix: every mutation path explicitly
   places new tensors via `.to(existing_param.device)`, verified by a test
   that runs the full mutate-and-forward cycle on CPU, and (skipped unless
   available) on CUDA.

6. **No checkpointing.** A model whose `nn.ModuleList` lengths change at
   runtime cannot round-trip through plain `state_dict` across different
   mutation states (shape mismatch on load). Fix: `checkpoint.py` with
   `save_checkpoint(model, path)` / `load_checkpoint(path, **ctor_kwargs)`
   that serialize topology (layer count, experts-per-layer, dims) alongside
   weights, and a test proving a model rebuilt from a checkpoint produces
   identical output on the same input.

7. **Optimizer state is discarded on every mutation.** The draft rebuilds a
   fresh `AdamW` after each structural change, losing Adam momentum/variance
   state for every parameter that didn't change. Fix:
   `rebuild_optimizer_preserving_state(old_optimizer, model, **adamw_kwargs)`
   that constructs a new optimizer but copies over per-parameter state for
   parameters that still exist (matched by identity), leaving only
   newly-created parameters to start fresh. This is purely a Phase 1
   structural concern (the architecture mutates — the optimizer must cope),
   not an aspect of Phase 2's continuous-learning loop.

8. **Naming clarity.** `GeometricCorrectiveManifold.pred_proj_real` /
   `pred_proj_imag` are not actually separate real/imaginary-component
   projections — they're two sequential general complex-linear stages. Renamed
   to `predictor_stage1` / `predictor_stage2` to avoid implying a split that
   doesn't exist.

## Testing plan

Real pytest suite, one test file per module:

- **`test_complex_ops.py`**: `ComplexLinear` output matches hand-computed
  complex multiplication on small fixed inputs; `ModReLU` zeroes sub-threshold
  magnitudes and preserves phase direction above threshold; `ComplexLayerNorm`
  output has zero mean / unit variance per the complex-magnitude definition;
  all three pass gradients (no `None`/`NaN` grads) through a `.backward()`.
- **`test_encoder.py`**: correct output shape for text `[B,S]` (long dtype),
  image `[B,C,H,W]`, and numeric `[B,T,F]` inputs, including numeric inputs
  with varying `F` across calls (proving the lazy per-`F` projection works);
  rejects unsupported shapes with a clear error.
- **`test_manifold.py`**: output shape equals input shape; `running_error_magnitude`
  updates only when `model.training` is `True`; error metric is always
  non-negative; gradients flow through the corrective path.
- **`test_attention.py`**: output shape correctness; attention weights sum to
  1 along the key dimension; multi-head split/combine round-trips shape
  correctly for several `(d_model, num_heads)` combinations.
- **`test_experts.py`**: entropy computed correctly on a known synthetic
  routing distribution; forcing `last_entropy > tau_split` spawns exactly one
  new expert with correctly-resized router matrix and usage buffer; forcing
  one expert's usage below `tau_death` (with low entropy) prunes exactly that
  expert; `min_experts` floor is respected; gradients flow post-mutation.
- **`test_layer.py`**: forward shape stability; `layer_salience` EMA updates
  only in training mode and moves in the expected direction under a
  large/small induced residual.
- **`test_model_mutation.py`**: forcing `running_error_magnitude > theta_up`
  grows exactly one layer (and not past `max_layers`); forcing a layer's
  salience below `theta_down` prunes exactly that layer (and not below
  `min_layers`); checkpoint save/load round-trip reproduces identical output;
  `rebuild_optimizer_preserving_state` preserves Adam state for surviving
  params and initializes fresh state for new ones; full mutate-and-device-move
  test (CPU always; CUDA via `pytest.mark.skipif(not torch.cuda.is_available())`).
- **`test_end_to_end.py`**: text → image → numeric sequential forward passes
  all succeed; a short optimization loop (few steps, synthetic targets) on
  the mutated model shows non-NaN decreasing-ish loss and succeeds without
  shape errors.

`scripts/validate.py` keeps a human-readable version of the original
stress-test trace (sequential multimodal ingestion → forced vertical growth
→ forced horizontal split → forced horizontal death → optimization step),
printing shapes/counts at each stage, as a secondary sanity-check script —
not a substitute for the test suite.

## Setup / tooling

- `git init`, `.gitignore` covering `.venv/`, `__pycache__/`, `*.pyc`,
  `.idea/`.
- `requirements.txt`: `torch`, `pytest`. (No `numpy` — nothing in this phase
  needs it; the existing "Failed to initialize NumPy" warning from torch is
  harmless and unrelated to this work.)
- Existing `.venv` (Python 3.14, torch 2.15 CPU) is reused; `pytest` gets
  installed into it.

## Open questions carried into Phase 2

- How novelty/nuance detection should differ from (or build on) the existing
  structural-error and routing-entropy signals.
- What "continuous learning" means operationally: a long-running streaming
  process vs. repeatable incremental-batch training runs.
- Whether inference-time self-correction reuses Ω_corr or needs a separate
  mechanism.

These are explicitly out of scope here and will get their own brainstorming
pass once Phase 1 is merged and verified.

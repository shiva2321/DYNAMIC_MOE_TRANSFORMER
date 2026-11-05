# Pre-Registration: `dmoe_v2_continual`

- **Experiment ID**: `dmoe_v2_continual`
- **Date (UTC)**: `2026-11-04`
- **Git Base Commit**: `64518df`

## 1. Claim & Headline Metrics

### Primary Claim
On a 3-domain sequential tokenizer-free UTF-8 byte-stream continual learning benchmark (`Domain A: Arithmetic` $\to$ `Domain B: Python Code` $\to$ `Domain C: Structured Records`), **`dmoe v2`** (`SelfStabilizingDynamicMorphicMoE` upgraded with roll-bound position-aware byte chunking, residual byte de-chunking, token-sorted top-$k$ complex MoE dispatch, `HyperspaceGlobalBus` shared anchor tether, fit-gap-guarded topology mutation, and closed-form `ExactLdaFastMemory` fast-slow adaptation) achieves:
1. **Superior Continual Learning Without Catastrophic Forgetting**:
   - **Headline Metric 1 (Final All-Domain Byte Accuracy $\uparrow$)**: Mean held-out byte accuracy averaged across Domains A, B, and C after training sequentially on A $\to$ B $\to$ C.
   - **Headline Metric 2 (Backward Transfer Forgetting $R_{\text{BWT}}$ in nats $\downarrow$)**: Mean increase in cross-entropy loss on earlier domains (A and B) after training on subsequent domains:
     $$R_{\text{BWT}} = \frac{1}{2}\Big[\big(\mathcal{L}_A(\text{after } C) - \mathcal{L}_A(\text{after } A)\big) + \big(\mathcal{L}_B(\text{after } C) - \mathcal{L}_B(\text{after } B)\big)\Big]$$
2. **Active-Expert Dispatch Efficiency**:
   - **Headline Metric 3 (MoE Layer Wall-Clock Speedup $\uparrow$)**: Measured wall-clock throughput (tokens/sec) of `DynamicMorphicMoE` with token-sorted `top_k=2` active dispatch vs. dense all-expert evaluation at `E=8` experts.
3. **Certified Closed-Form Unlearning (`ExactLdaFastMemory.downdate`)**:
   - **Headline Metric 4 (Unlearning Max Weight/Logit Discrepancy $\downarrow$)**: Maximum absolute difference in float64 between an `ExactLdaFastMemory` that absorbed Domains A + B + C and then downdated Domain B vs. a reference `ExactLdaFastMemory` that absorbed only Domains A + C.

## 2. Falsifiers
1. **Continual Learning Falsifier**: The claim is **falsified** if, across 20 independent seeds (`seeds 0..19`), `dmoe v2` does not achieve **both**:
   - Strictly higher mean Final All-Domain Byte Accuracy than `RegularTransformer_Unfrozen`, `RegularTransformer_FrozenBackbone`, and `StaticMoE_Naive`, **and**
   - Strictly lower mean $R_{\text{BWT}}$ forgetting than `RegularTransformer_Unfrozen` and `StaticMoE_Naive`.
2. **Efficiency Falsifier**: The dispatch efficiency claim is **falsified** if `DynamicMorphicMoE(top_k=2)` at `E=8` is not faster in wall-clock latency than `DynamicMorphicMoE(top_k=None)` on the same input tensor.
3. **Unlearning Falsifier**: The certified unlearning claim is **falsified** if `ExactLdaFastMemory.downdate` produces a max absolute logit difference $> 10^{-9}$ or argmax agreement $< 1.0$ relative to never having observed the removed domain.

## 3. Frozen Protocol
- **Seeds**: 20 independent seeds (`0` through `19` inclusive), disjoint train and test batch generators (`train_seed = seed * 10000 + step`, `eval_seed = seed * 10000 + 90000 + domain_id`).
- **Task / Data Stream**:
  - Raw UTF-8 byte sequences (`seq_len = 16` bytes, `chunk_size = 4`, `batch_size = 16`).
  - **Domain A (Arithmetic Expressions)**: Deterministic next-byte/shift rule over ASCII digit/operator byte patterns (`bytes in [32..63]`, target rule $y_A(b, pos) = (b + 3 \cdot (pos \bmod 4) + 1) \bmod 64$).
  - **Domain B (Code Identifiers/Syntax)**: Deterministic structural transformation over ASCII letter/bracket byte patterns (`bytes in [64..127]`, target rule $y_B(b, pos) = 64 + ((b - 64) \oplus ((pos \bmod 4) + 5)) \bmod 64$).
  - **Domain C (Structured Key-Value Records)**: Deterministic transformation over high-byte symbol patterns (`bytes in [128..191]`, target rule $y_C(b, pos) = 128 + (3 \cdot (b - 128) + 7 - (pos \bmod 4)) \bmod 64$).
  - **Shared-Input Regime Overlap Challenge**: In addition to disjoint byte ranges, 25% of positions in every sequence use shared structural delimiter bytes (`[0..31]`) whose target mapping *depends on the domain context* (testing true contextual routing and fast-slow memory separation, not just byte-lookup memorization).
- **Training Budget**:
  - `50` gradient steps per domain (`150` sequential steps total: 50 on A $\to$ 50 on B $\to$ 50 on C), `lr = 4e-3`, `AdamW`.
  - Held-out evaluation: `64` unseen sequences (`1,024` target bytes) per domain, evaluated at the end of Phase A, end of Phase B, and end of Phase C.

## 4. Oracles
- **Zero Task-ID / Domain-ID Oracle**: Neither `dmoe v2` nor any baseline receives domain labels (`A`, `B`, `C`) or task boundaries during forward pass, routing, or evaluation. All models receive only raw byte tensors `x` of shape `[B, 16]`.

## 5. Mandatory Null Controls & Baselines
1. **`NullControl_Untrained`**: Freshly initialized `dmoe v2` with zero training steps.
2. **`NullControl_BlindZeroInput`**: Fully trained `dmoe v2` evaluated with all test input bytes replaced by zeros (`torch.zeros_like(x)`).
3. **`NullControl_ShuffledTargets`**: `dmoe v2` trained on the exact same input stream with target bytes randomly permuted within each batch.
4. **`Baseline_RegularTransformer_Unfrozen`**: Standard real-valued Dense Transformer (`nn.TransformerEncoder` + multi-scale byte chunker/de-chunker, matched `d_model=32`, `num_heads=2`, `layers=2`, `d_ff=128`) trained sequentially across A $\to$ B $\to$ C with all parameters unfrozen.
5. **`Baseline_RegularTransformer_FrozenBackbone`**: Identical Dense Transformer trained on Domain A, after which the encoder and Transformer layers are **frozen** (`requires_grad=False`) before training on Domains B and C (only the output decoder head remains trainable—directly testing why freezing a regular transformer fails when teaching new things on top).
6. **`Baseline_StaticMoE_Naive`**: `SelfStabilizingDynamicMorphicMoE` trained sequentially on A $\to$ B $\to$ C without replay, fast memory, or topology mutation.
7. **Component Ablations**:
   - `Ablation_dMoE_NoFastMemory`: `dmoe v2` with `enable_fast_memory=False` (rehearsal + distillation + bus + expert freezing only).
   - `Ablation_dMoE_FastMemoryOnly_NoReplay`: `dmoe v2` with `replay_buffer_capacity=0` and `enable_fast_memory=True`.

## 6. Tuning Budget
- At most 3 validation configurations tested on a separate validation seed (`seed = 999`) prior to running the frozen 20-seed test evaluation (`seeds 0..19`). All validation runs will be logged.


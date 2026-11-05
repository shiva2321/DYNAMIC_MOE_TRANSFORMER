# Pre-Registration: `dmoe_v2_continual_hard_control` (Suspicious-Success Diagnostic & Matched-Mechanism Control)

## 0. Reason for New Experiment ID (`research-protocol.md` §2.2, §5, §7, §12)
In `results/dmoe_v2_continual/results.json`, `dMoE_v2_Full` achieved `avg_acc_final = 1.0000 ± 0.0000` (`min = 0.9997, max = 1.0000`) across 20 seeds (`0..19`). Per `research-protocol.md` §12 (*"Stop and report immediately when: A result is much better than expected, hits 100%, or beats a baseline by a large factor. Suspicious success gets more controls, not a victory lap"*), we audited why `dmoe_v2_continual` saturated at `1.0000`:
1. In `dmoe_v2_continual`, `slots 1..3` used domain-disjoint input byte ranges (`[32..63]` for Domain A, `[64..95]` for Domain B, `[96..127]` for Domain C) and a deterministic per-slot mapping of 32 values per domain. With 50 steps $\times$ 16 batch size = 800 sequences per domain, `MultiSlotExactLdaMemory` saw every single-slot rule multiple times and linearly separated the disjoint byte embeddings on `slots 1..3` (`0.7467` and `0.7491` accuracy even without replay), while Replay + Distillation stabilized the chunk context for `slot 0` (the remaining 25% delimiter bytes), pushing total accuracy to `1.0000`.
2. Furthermore, in `dmoe_v2_continual`, `Baseline_RegularTransformer_Unfrozen` and `Baseline_RegularTransformer_FrozenBackbone` did not use a replay buffer or fast memory. To isolate the **architectural contribution** of `dMoE v2` (Complex Phase-Preserving Manifold + Dynamic Morphic MoE + Expert Maturity Freezing + Intra-Chunk Phase Basis) from the **continual wrapper contribution** (Replay Buffer + Distillation + `MultiSlotExactLdaMemory`), we must also test a `Baseline_RegularTransformer_WithReplay` and `Baseline_RegularTransformer_WithReplayAndFastMemory` under the exact same wrapper, and test on **overlapping input byte distributions with held-out test combinations**.

The original `results/dmoe_v2_continual/results.json` remains permanently on record and unmodified.

## 1. Claim & Headline Metric
- **Claim**: On a non-saturating 3-domain continual sequence task where all domains share overlapping byte ranges (`[32..95]`), targets depend on cross-position context (`x[t]` and `x[t-1]`) plus a domain header byte, and 5% label corruption caps Bayes-optimal accuracy at `0.95`, `dMoE_v2` outperforms both standard unfrozen/frozen Regular Transformers and a Regular Transformer given the exact same Replay + Distillation wrapper (`Baseline_RegularTransformer_WithReplay`).
- **Headline Task Metric**:
  1. **Final All-Domain Held-Out Byte Accuracy** $\bar{A}_{\text{final}} = \frac{1}{3}(A_{3,1} + A_{3,2} + A_{3,3}) \in [0, 1]$ (**higher is better**, theoretical Bayes ceiling $= 0.95$).
  2. **Backward Transfer (Forgetting)** $R_{\text{BWT}} = \frac{1}{2}\big((L_{3,1} - L_{1,1}) + (L_{3,2} - L_{2,2})\big)$ in nats (**lower is better**).

## 2. Falsifier
The claim is **falsified** if across 20 independent seeds (`0..19`):
1. Any model exceeds the theoretical noise ceiling (`0.95 + 0.02 = 0.97`), which would indicate RNG/data leakage between train and test splits, OR
2. `Ablation_dMoE_NoFastMemory` (the `dMoE v2` neural architecture with Replay + Distillation + Expert Maturity Freezing) fails to outperform `Baseline_RegularTransformer_WithReplay` (the parameter-matched Regular Dense Transformer with the exact same Replay + Distillation wrapper), OR
3. Any null control (`NullControl_Untrained`, `NullControl_BlindZeroInput`, `NullControl_ShuffledTargets`) exceeds `0.10` accuracy.

## 3. Frozen Protocol
- **Seeds**: 20 independent seeds (`0, 1, 2, ..., 19`).
- **Data / Environment (Overlapping Byte Vocabulary + Cross-Position Context + 5% Noise Ceiling)**:
  - Sequence length $L = 16$ raw bytes (`chunk_size = 4`, $S_c = 4$ chunks).
  - Every domain shares the **exact same payload byte range** `x[:, 1:] in [32..63]` (32 shared byte values across all 3 domains, eliminating any disjoint-vocabulary shortcut!).
  - Position `0` (`x[:, 0]`) is a 2-byte domain-family prefix (`Domain A: {10, 11}`, `Domain B: {12, 13}`, `Domain C: {14, 15}`) that must be propagated across all 4 chunks via attention/bus so later chunks know which domain function to apply to the identical `[32..63]` payload bytes!
  - Target `y[:, t]` depends on **both** the domain prefix `x[:, 0]`, the current byte `x[:, t]`, and the previous byte `x[:, t-1]` (requiring cross-byte and cross-chunk context):
    - Domain A (`d_id=0`): `clean_y[:, t] = 32 + ((x[:, t] - 32) + (x[:, t-1] % 8) + slot) % 32`
    - Domain B (`d_id=1`): `clean_y[:, t] = 32 + (((x[:, t] - 32) ^ (x[:, t-1] % 8)) + 2 * slot + 7) % 32`
    - Domain C (`d_id=2`): `clean_y[:, t] = 32 + (3 * (x[:, t] - 32) - (x[:, t-1] % 8) + slot + 13) % 32`
  - **5% Symmetric Label Corruption**: With probability `0.05` on every position (using the deterministic data generator), `y[:, t]` is replaced by a uniform random byte in `[32..63]`, bounding the Bayes-optimal accuracy ceiling at $\approx 0.9516$.
- **Training Budget**: 60 steps per domain $\times$ 3 sequential domains = 180 steps total (`batch_size = 16`, `AdamW`, `lr = 4e-3`).
- **Baselines, Controls, and Ablations**:
  1. `NullControl_Untrained`
  2. `NullControl_BlindZeroInput`
  3. `NullControl_ShuffledTargets`
  4. `Baseline_RegularTransformer_Unfrozen` (no replay, no fast memory)
  5. `Baseline_RegularTransformer_FrozenBackbone` (frozen after Domain A)
  6. `Baseline_RegularTransformer_WithReplay` (Regular Dense Transformer given the exact same `ContinualLearner` Replay Buffer + Distillation as `dMoE`)
  7. `Baseline_RegularTransformer_WithReplayAndFastMemory` (Regular Dense Transformer given Replay + Distillation + `MultiSlotExactLdaMemory`)
  8. `Ablation_dMoE_NoFastMemory` (`dMoE v2` with Replay + Distillation + Expert Maturity Freezing, no `MultiSlotExactLdaMemory`)
  9. `dMoE_v2_Full` (`dMoE v2` with Replay + Distillation + Expert Maturity Freezing + `MultiSlotExactLdaMemory`)


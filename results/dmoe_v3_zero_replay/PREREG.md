# Pre-Registration: `dmoe_v3_zero_replay`

**Experiment ID**: `dmoe_v3_zero_replay`  
**Date**: 2026-11-05  
**Protocol Status**: FROZEN BEFORE ANY TEST RUN  
**Author**: Antigravity & User  

---

## 1. Objective & Hypothesis
We test whether incorporating the verified architectural components from `D:\spherical_hyperspace_memory` into `DynamicMorphicMoE`:
1. **Per-Expert Local Closed-Form Fast Memory (`LocalLdaExpert`)** keeping local class-conditioned sufficient statistics $(\mathbf{n}_k, \mathbf{s}_k, \mathbf{S}_k)$,
2. **Batch Hypothesis Exemplar Gating (`SetGate`)** matching chunk sets via permutation energy tests on exemplar buffers $\mathcal{R}_k$, and
3. **Domain-Transition Lifetime Freezing** (freezing completed expert networks and exemplar signatures when novelty transitions trigger),

enables `dMoE_v3` to achieve high sequential retention and prevent catastrophic forgetting **with zero exemplar replay (`replay_buffer_capacity=0`)**, decisively outperforming a parameter-matched Regular Dense Transformer without replay.

---

## 2. Pre-Registered Headline Metrics & Success Criteria
All statistics evaluated across **20 independent seeds (`0..19`)** on held-out test sets ($B=64$ sequences $\times 16$ bytes = 1,024 test tokens per domain):

1. **Final All-Domain Held-Out Byte Accuracy**:
   $$\bar{A}_{\text{final}} = \frac{1}{3}\left(A_{3, \text{DomA}} + A_{3, \text{DomB}} + A_{3, \text{DomC}}\right)$$
   - Direction: Higher is better (Bayes optimal ceiling $\approx 0.9531$ due to $5\%$ label noise).
2. **Domain A Retention Ratio**:
   $$R_{\text{ret, A}} = \frac{A_{3, \text{DomA}}}{\max\left(A_{1, \text{DomA}}, 10^{-6}\right)}$$
   - Direction: Higher is better ($1.0$ indicates zero forgetting of initial domain).
3. **Backward Transfer Loss Degradation**:
   $$R_{\text{BWT}} = \frac{1}{2}\left[\left(L_{3, \text{DomA}} - L_{1, \text{DomA}}\right) + \left(L_{3, \text{DomB}} - L_{2, \text{DomB}}\right)\right]$$
   - Direction: Lower is better ($\le 0.10$ nats indicates negligible loss degradation).

---

## 3. Strict Pre-Registered Falsifiers (Hard Gates)
The primary claim that `dMoE_v3` achieves zero-replay continual learning is **FALSIFIED** if any of the following occur:
- **F1 (Sparsity / Catastrophic Forgetting)**: $\bar{A}_{\text{final}}(\text{dMoE\_v3\_ZeroReplay}) < 0.7000$.
- **F2 (Retention Failure)**: Domain A retention ratio $R_{\text{ret, A}}(\text{dMoE\_v3\_ZeroReplay}) < 0.7500$.
- **F3 (Null Control Violation)**: Any null control (`NullControl_Untrained`, `NullControl_ShuffledTargets`, `NullControl_BlindZeroInput`) exceeds the empirical per-domain majority-class accuracy plus $0.05$ (threshold $\approx 0.2222$).
- **F4 (Baseline Competitor Failure)**: $\bar{A}_{\text{final}}(\text{dMoE\_v3\_ZeroReplay}) \le \bar{A}_{\text{final}}(\text{RegularTransformer\_NoReplay}) + 0.2500$ (i.e. fails to deliver a $>+25\%$ gain over dense un-replayed transformer).
- **F5 (Data Leakage Ceiling)**: Any condition exceeds $0.9700$ average accuracy on test data with $5\%$ noise.

---

## 4. Benchmark Task & Training Budget
- **Task**: 3-Domain overlapping byte sequence rule learning (`generate_easy_domain_batch` from `scripts/run_dmoe_v2_easy_eval.py`).
  - Sequence length $L=16$ raw bytes, `chunk_size = 4`.
  - All domains share the same payload byte vocabulary $[32..47]$ on slots $1..3$.
  - Slot $0$ contains a domain family header byte.
  - $5\%$ symmetric label noise (Bayes ceiling $\approx 0.9531$).
- **Training Schedule**: Sequential Domain $0 \to 1 \to 2$.
- **Steps**: $N = 240$ steps per domain ($720$ total sequential gradient steps), `batch_size = 16`, `AdamW`, $\text{lr} = 4\times 10^{-3}$.
- **Replay Capacity**: Strictly **0** for all evaluated Zero-Replay conditions (`replay_buffer_capacity=0`).

---

## 5. Conditions Under Test (20 Seeds)
1. `NullControl_Untrained`
2. `NullControl_ShuffledTargets` (dMoE v3 trained with shuffled labels, no replay)
3. `NullControl_BlindZeroInput` (dMoE v3 evaluated on zeroed input)
4. `Baseline_RegularTransformer_NoReplay` (Parameter-matched regular Transformer, 0 replay)
5. `Baseline_RegularTransformer_Replay128` (Upper-bound reference with 128-sample replay)
6. `Baseline_dMoE_v2_NoReplay` (Prior dMoE v2 model without replay)
7. `dMoE_v3_ZeroReplay` (Full proposed model: Per-Expert Local LDA + SetGate Routing + Domain-Gated Freezing)


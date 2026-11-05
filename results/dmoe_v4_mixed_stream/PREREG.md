# Pre-Registration: `dmoe_v4_mixed_stream`

**Experiment ID**: `dmoe_v4_mixed_stream`  
**Date**: 2026-11-05  
**Protocol Status**: FROZEN BEFORE ANY TEST RUN  
**Author**: Antigravity & User  

---

## 1. Objective & Hypothesis

We test whether `dMoE_v4` equipped with:
1. **Per-Sequence Content-Addressed Routing (`SequenceContentRouter`)** matching individual sequence signatures in $O(d^2)$ on GPU without batch-homogeneity assumptions, and
2. **Per-Sequence Isolated Closed-Form Readout (`LocalLdaExpert`)** decoding interleaved sequences from different domains in parallel without cross-talk,

can successfully master **Interleaved Mixed Streams (B2)** with **strictly zero exemplar replay (`replay_buffer_capacity=0`)** and **zero task labels or domain indicators**, maintaining high sequential retention across all domains simultaneously.

---

## 2. Pre-Registered Headline Metrics & Success Criteria

All statistics evaluated across **20 independent seeds (`0..19`)** on held-out mixed-stream test batches ($B=64$ sequences $\times 16$ bytes = 1,024 test tokens per batch, randomly interleaved across Domains 0, 1, and 2):

1. **Final Mixed-Stream Held-Out Byte Accuracy**:
   $$\bar{A}_{\text{mixed}} = \frac{1}{B \cdot L} \sum_{i=1}^B \sum_{t=1}^L \mathbb{I}\left(\hat{y}_{i, t} = y_{i, t}\right)$$
   - Direction: Higher is better (Bayes optimal ceiling $\approx 0.9531$ due to $5\%$ label noise).
2. **Domain A Retention Ratio Within Mixed Streams**:
   $$R_{\text{ret, A}} = \frac{A_{\text{final, DomA}}}{\max\left(A_{\text{initial, DomA}}, 10^{-6}\right)}$$
   - Direction: Higher is better ($1.0$ indicates zero forgetting of initial domain).
3. **Per-Sequence Routing Disentanglement Accuracy**:
   $$\text{Acc}_{\text{route}} = \frac{1}{B} \sum_{i=1}^B \mathbb{I}\left(\text{routed\_expert}_i = d_i\right)$$
   - Direction: Higher is better (chance accuracy is 1/3).

---

## 3. Strict Pre-Registered Falsifiers (Hard Gates)

The claim that `dMoE_v4` solves mixed-stream continual learning without replay is **FALSIFIED** if any of the following occur:
- **F1 (Mixed-Stream Accuracy)**: $\bar{A}_{\text{mixed}}(\text{dMoE\_v4\_MixedStream\_ZeroReplay}) < 0.7000$.
- **F2 (Domain A Retention)**: Domain A retention ratio $R_{\text{ret, A}} < 0.7500$.
- **F3 (Routing Disentanglement)**: Sequence routing accuracy $\text{Acc}_{\text{route}} < 0.9000$ (i.e. fails to route $\ge 90\%$ of interleaved sequences to their true domain expert).
- **F4 (Baseline Competitor Failure)**: $\bar{A}_{\text{mixed}}(\text{dMoE\_v4}) \le \bar{A}_{\text{mixed}}(\text{RegularTransformer\_NoReplay}) + 0.2500$ (fails to deliver a $>+25\%$ gain over dense un-replayed transformer).
- **F5 (Data Leakage Ceiling)**: Any condition exceeds $0.9700$ average accuracy on test data with $5\%$ noise.

---

## 4. Benchmark Task & Training Budget

- **Task**: 3-Domain overlapping byte sequence rule learning (`generate_easy_domain_batch` from `scripts/run_dmoe_v2_easy_eval.py`).
  - Sequence length $L=16$ raw bytes, `chunk_size = 4`.
  - All domains share payload byte vocabulary $[32..47]$ on slots $1..3$.
  - Slot $0$ contains domain family header bytes ($10..11$ for Dom 0, $12..13$ for Dom 1, $14..15$ for Dom 2).
  - $5\%$ symmetric label noise (Bayes ceiling $\approx 0.9531$).
- **Training Schedule**: Sequential Domain $0 \to 1 \to 2$ ($N=240$ steps per domain, 720 total steps), `batch_size = 16`, `AdamW`, $\text{lr} = 4\times 10^{-3}$.
- **Evaluation (B2 Mixed Streams)**: Interleaved test batches of $B=64$ sequences with domain labels $d_i \sim \text{Uniform}(\{0, 1, 2\})$ stripped from the input.
- **Replay Capacity**: Strictly **0** for all evaluated Zero-Replay conditions (`replay_buffer_capacity=0`).

---

## 5. Conditions Under Test (20 Seeds)

1. `NullControl_Untrained`: dMoE v4 evaluated without any training.
2. `NullControl_ShuffledTargets`: dMoE v4 trained with shuffled labels, evaluated on mixed streams.
3. `NullControl_BlindZeroInput`: dMoE v4 evaluated on all-zero inputs.
4. `Baseline_RegularTransformer_NoReplay`: Parameter-matched regular Transformer trained sequentially without replay.
5. `Baseline_RegularTransformer_Replay128`: Parameter-matched regular Transformer trained with 128-sample replay buffer.
6. `dMoE_v4_MixedStream_ZeroReplay`: Proposed dMoE v4 with per-sequence content router and local LDA fast memory.
7. `dMoE_v4_PureNeural_ZeroReplay` (Post-Audit Addition): Pure neural network with per-sequence router, modular per-domain decoder heads, zero LDA, and zero replay. Added following codebase audit to verify pure neural backpropagation without closed-form LDA assistance.


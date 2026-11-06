# Pre-Registration Protocol: `dmoe_v4_track_k`

**Experiment ID**: `dmoe_v4_track_k`  
**Date**: 2026-11-06  
**Protocol Status**: FROZEN BEFORE ANY TEST RUN  
**Author**: Antigravity & User  
**Design Reference**: §14 Amendment (2026-11-06) of `docs/superpowers/specs/2026-11-05-dmoe-v4-phase-a-plastic-taskfree-cl-and-thinking-design.md`

---

## 1. Objective & Hypothesis

We test whether `dMoE_v4` equipped with an **Evidence-Accumulating Upkeeper** (prior-count evidence weighting and a fixed-size tentative-belief store) can master **Track K** (Belief Revision and Continual Updating on Header-Free Context Tasks) against five parameter-matched and FLOP-matched baselines:
1. **K1 (New-Rule Acquisition)**: Rapid acquisition at low confidence from $n \in \{1, 3, 10\}$ exposures without wiping out prior rules.
2. **K2 (Temporal Burst Resistance)**: Refusal to overwrite a high-count established belief when subjected to a short temporal burst ($B=5$) of contradictory labels from a single noisy source.
3. **K3 (Genuine Belief Revision)**: Smooth updating and flipping of beliefs when contradictory evidence persists across $R$ batches and diverse verified sources.
4. **K4 (Curation under Capacity)**: Graceful decay of dormant facts under a fixed slot budget while high-frequency facts are preserved.

The learner is **strictly never given a flag** indicating whether evidence is "verified" or "noise". The harness uses ground truth solely for scoring.

---

## 2. Tested Conditions (All Parameter- and FLOP-Matched)

1. **`dMoE_v4_TrackK_Upkeeper`**: Proposed dMoE with Evidence-Accumulating Upkeeper, tentative-belief store, per-sequence routing, and modular heads.
2. **`Baseline_Dense_NoReplay`**: Standard parameter-matched dense Transformer trained sequentially without replay.
3. **`Baseline_Dense_Replay128`**: Standard parameter-matched dense Transformer with 128-sample replay buffer.
4. **`Baseline_Dense_LoRA_Router`**: Standard dense Transformer with per-domain low-rank adapters and router.
5. **`Baseline_Dense_InContext`**: Standard dense Transformer provided with new evidence in its context window.
6. **`Baseline_NonNeural_Dictionary_Counters`**: Exact non-neural key-value store with counters. *Integrity check: if this dictionary solves Track K as well as dMoE, the honest claim is "the counting mechanism works", not "the neural architecture is better".*

---

## 3. Strict Pre-Registered Falsifiers (Hard Gates)

The claim that `dMoE_v4` masters belief revision and robust continual updating is **FALSIFIED** if any of the following occur:
- **F1 (K2 Burst Resistance)**: Post-burst accuracy on established rule $\bar{A}_{\text{post-burst}} < 0.8500$.
- **F2 (K3 Genuine Revision)**: Post-revision accuracy on revised rule $\bar{A}_{\text{post-revised}} < 0.8500$.
- **F3 (K1 Retention Ratio)**: Retention of prior rules after new rule learning $R_{\text{ret}} < 0.8500$.
- **F4 (Margin Over Dense NoReplay)**: Average accuracy margin of `dMoE_v4` over `Baseline_Dense_NoReplay` on K2 & K3 $\le +0.2000$.
- **F5 (Calibration Bound)**: $\text{NLL}(\text{dMoE\_v4}) > 1.5 \times \text{NLL}(\text{Baseline\_Dense\_Replay128})$.
- **F6 (Data Leakage Ceiling)**: Any condition exceeds $0.9700$ average accuracy on 5% noise data.
- **F7 (Memory Honesty)**: Tentative-belief store exceeds 64 entries or 4 KB total storage.

---

## 4. Benchmark Specifications

- **Alphabet**: ASCII 32..47 (16 distinct symbols).
- **Rule Type**: Bigram context mapping $y_t = \pi_k(x_t, x_{t-1})$.
- **Zero Tags**: No header tags, no prefix bytes, no slot-0 giveaways.
- **Noise Rate**: 5% symmetric label noise (Bayes optimal ceiling $\approx 0.9531$).
- **Sample Budget**: 120 steps per domain, batch size 16, sequence length 16.
- **Seeds**: 20 independent seeds (`0..19`).

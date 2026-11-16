# Stage 1 Pre-Registration Revision 2 (`stage1_causal_memory_v2`)

**Pre-registered BEFORE re-tuning or running any Stage 1 test seeds (`research-protocol.md §2.2`).**
Note: Zero test seeds (`0..19`) were run under `v1` (`PREREG.md`). `v1` halted cleanly during dev-only (`100..104`) inspection (`tuning_log.json`).

---

## 1. Dev-Only Diagnoses From `v1` (No Test Data Touched)

1. **`1C Non-Interference Failure (drop = 0.598..0.672)`**:
   - `DualTimescaleMemory.ltm_gate` read only `x_norm` (blind to causal prefix markers like `[RULE_MARK, tok]`), and Task 1C trained parametric rule batches `xr` exclusively with empty `episode_memory=None` (`M=0`). When test time placed 64 random distractor tokens `[32..63]` in LTM, token `tok` matched stored keys and hijacked prediction.
   - **v2 Fix**: Contextual `ltm_gate` receives `torch.cat([x_norm, stm_out], dim=-1)` in `dmoe/memory.py`, and Task 1C trains parametric rule batches `xr` 50% with empty LTM and 50% with unrelated batch-permuted distractor `episode_memory` (matching real populated-LTM inference, with zero test labels).
2. **`1A Signed-Beta Retention & Baseline Optimization Parity`**:
   - `w_beta.bias = 0.0` initialized `signed_beta=True` at aggressive `beta_init = 1.0` (`2*sigmoid(0)`) vs `0.50` for `NoSignedBeta`, and `forget_bias = 3.5` retained only `0.9707^64 = 0.149` across `OOD_64`, destabilizing `Cog_Mem_Hier` (`0.553` vs `0.879` on `NoSignedBeta`).
   - Meanwhile `DenseRoPE_Baseline` (3 layers) was under-trained at `300` steps without LR warmup (`0.340` at 300 steps vs `1.000` at `800` steps with warmup, per dev baseline probe).
   - **v2 Fix**: `forget_bias = 4.5`, `w_beta.bias = -1.1` when `signed_beta=True` (`beta_init ≈ 0.50`, smooth associative init while retaining `beta -> 2.0` parity capability), linear LR warmup (`40` steps) + cosine decay, sufficient `steps_1a` so `DenseRoPE_Baseline` reaches full induction-head capability, and fair per-family dev-selected LR from bounded grid `[2e-3, 3.5e-3]`.
3. **Exact `Null_BlindZeroInput` Definition**:
   Evaluates the trained `Cog_Mem_Hier` weights per seed with inputs zeroed (`torch.zeros_like(x)`), eliminating redundant duplicate re-training.

---

## 2. Frozen Architectures, Splits, Seeds & Falsifiers

Identical model definitions, parameter matching (~151k–162k), real code splits, shared vocabularies `[32..63]`, dev seeds `[100..104]`, and `20` disjoint test seeds `[0..19]` as `PREREG.md`.

### Pre-Registered Falsifiers (`F1–F8`, Must All Pass to Lock Stage 1)
- **F1 (MQAR Accuracy & Stability)**: `Cog_Mem_Hier` achieves `1A_ID mean >= 0.90` (`min >= 0.75`, `std <= 0.08`) AND `1A_OOD64 mean >= 0.85` (`min >= 0.65`).
- **F2 (MQAR Beats Windowed-NoSTM & GRU, Plus Length-Gen Check)**: `Cog_Mem_Hier` beats `Cog_NoSTM_Hier` by `>= +0.40` and beats `GRU_Baseline` by `>= +0.10` on `1A_OOD64` (while we honestly report how `Cog_Mem_Hier` compares with 3-layer `DenseRoPE_Baseline` on both ID and `OOD_64`).
- **F3 (Parity State Tracking M22)**: On `1B_OOD64`, `Cog_Mem_Hier` achieves `mean >= 0.90` (`min >= 0.75`), beating BOTH `Cog_NoSignedBeta` and `DenseRoPE_Baseline` by `>= +0.15`.
- **F4 (Cross-Episode LTM Survival)**: On `1C` after `4` noisy distractor episodes, `Cog_Mem_Hier` achieves `LTM_Recall_Acc mean >= 0.75` (`min >= 0.50`), beating single-episode stateless `DenseRoPE_Baseline` and `GRU_Baseline` by `>= +0.65`.
- **F5 (Null-Sink & Contextual Non-Interference)**: `NonInterference_Drop` (`Clean - Post64`) has `mean <= 0.03` and `max <= 0.08`.
- **F6 (Real Code Next-Byte BPB)**: On `1D` held-out test (`L=64`), `Cog_Mem_Hier` achieves `mean BPB <= DenseRoPE_Baseline mean BPB + 0.05`.
- **F7 (Null Controls Fail Cleanly)**: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput` stay at chance (`1A <= 0.08`, `1B <= 0.35`, `1D BPB >= 6.5 / 5.0 / 4.5`).
- **F8 (Causality Gate)**: `max prefix diff <= 1e-5` on all models.


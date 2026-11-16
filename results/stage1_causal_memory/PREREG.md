# Stage 1 Pre-Registration: Causal Substrate & Dual-Timescale Working/Episodic Memory (`stage1_causal_memory_v1`)

**Pre-registered BEFORE tuning or running any Stage 1 experiment (`research-protocol.md §1`).**
**Scope**: Stage 1 validates the causal sequence modeling substrate (`dmoe/rope.py`), short-term Gated Delta working memory with signed eigenvalues (`dmoe/memory.py`, M22), cross-episode Long-Term Slot Memory with size-invariant null sink, and Three-Level Causal RoPE Attention (`dmoe/hierarchical_attention.py`, `dmoe/real_model.py`). Zero Stage 2/3/4/5 mechanisms (`thinking`, `world_model`, `auto_lifecycle`, `novelty`) are active in Stage 1 (`enable_thinking=False`, `enable_world_model=False`, `auto_lifecycle=False`, `initial_experts=1`).

---

## 1. Hypotheses

- **H1 (Working Memory & OOD Length Generalization)**:
  Removing additive `pos_embed` in favor of RoPE + causal depthwise convolutions + SiLU-gated Delta-Rule working memory (`GatedDeltaWorkingMemory`) eliminates earlier bimodal recall collapse (`0.148..1.0`) and enables length-generalizing associative recall on shared-vocabulary MQAR across gaps far wider than `local_window=8`, beating windowed-attention without STM (`Cog_NoSTM_Hier`) and vector-state `GRU_Baseline`.
- **H2 (Signed Eigenvalue State Tracking, M22)**:
  Allowing `beta in (0, 2)` (`signed_beta=True`) yields state transition eigenvalues in `[-1, 1]` capable of exact modulo-2 reflections over unseen rollout horizons (`L=64` after `L_train=20`), outperforming both positive-only delta memory (`Cog_NoSignedBeta`, `signed_beta=False`) and global causal RoPE Transformer (`DenseRoPE_Baseline`, TC0-limited).
- **H3 (Cross-Episode Persistent LTM Without Parametric Interference)**:
  Surprise-gated predictive episode consolidation (`consolidate_ltm`) preserves episodic `(k -> v)` bindings across intermediate unrelated distractor episodes (`4` noisy episodes) and outlasts single-episode stateless models (`DenseRoPE_Baseline`, `GRU_Baseline`, `Cog_NoLTM`), while the size-invariant null-sink prevents unrelated slots (`M=64` full noise) from degrading parametric rule accuracy.
- **H4 (Real-Data Compatibility)**:
  Adding causal working memory, LTM, and Three-Level RoPE Attention does NOT degrade held-out next-byte compression (`BPB`) on real Python source/test files relative to a parameter-matched 3-layer `DenseRoPE_Baseline`.

---

## 2. Parameter-Matched Architectures & Baselines (`chunk_size=1`, `vocab=258`, `out=256`)

All models verified strictly causal (`max prefix diff <= 1e-5` under future suffix perturbation) via `tests/test_stage1_causal_memory.py`.

1. **`Cog_Mem_Hier`** (Full Stage 1 target architecture — **~152,549 params**):
   `RealSelfStabilizingDMoE(d_model=48, num_heads=4, d_ff=96, initial_layers=2, initial_experts=1, chunk_size=1, enable_memory=True, num_ltm_slots=64, enable_three_level_attn=True, local_window=8, chunk_stride=4, enable_layer_stm=True, signed_beta=True, use_rope=True, causal_attn=True, auto_write_ltm=False)`
2. **`Cog_Mem_Flat`** (Ablation: Full causal RoPE attn instead of 3-level hierarchical — **~152k matched** via `d_ff=268`, or exact checked count logged at startup):
   Identical to `Cog_Mem_Hier` except `enable_three_level_attn=False`, `d_ff=268`.
3. **`Cog_NoSignedBeta`** (Ablation testing H2 / M22 — **152,549 params**):
   Identical to `Cog_Mem_Hier` except `signed_beta=False`.
4. **`Cog_NoSTM_Hier`** (Ablation testing H1 — **~151k matched** via `d_ff=352`):
   Identical to `Cog_Mem_Hier` except `enable_memory=False, enable_layer_stm=False`.
5. **`DenseRoPE_Baseline`** (Matched strong external baseline — **158,464 params**, 3 layers > 2):
   `CausalDenseRoPEBaseline(d_model=64, num_heads=4, d_ff=192, num_layers=3)`
6. **`GRU_Baseline`** (Matched recurrent external baseline — **161,536 params**):
   `GRUBaseline(d_model=96, num_layers=2)`

### Null Controls (`research-protocol.md §5`)
- **`Null_Untrained`**: `Cog_Mem_Hier` at initialization (`0` optimizer steps).
- **`Null_ShuffledTargets`**: `Cog_Mem_Hier` trained with randomly permuted target labels per batch.
- **`Null_BlindZeroInput`**: Trained `Cog_Mem_Hier` evaluated with all sequence input bytes masked to `0`.

---

## 3. Benchmarks & Exact Anti-Cheat Constraints

### Task 1A — Shared-Vocab MQAR + OOD Length Generalization
- **Vocabulary**: Symbols `[32..63]` (`32` symbols) are **100% shared across Keys, Values, and Distractor fillers**. Marginal frequency over `[32..63]` is uniform across all non-marker positions (`marker 64` flags query cue `[64, k_i] -> v_i`). Within each episode, `K=5` keys are distinct and values/distractors avoid reusing that episode's active keys so the ground truth is unambiguous while no token ID reveals its role globally.
- **Train distribution**: `K=5` KV pairs, random gap `D_gap in [6..18]` (`seq_len = 26..38`), `Q=4` queries per sequence (`loss` computed over query target positions + small regularizer/or exact query positions; identical loss across ALL models).
- **Eval splits**:
  - `ID`: `D_gap = 12` (`seq_len = 30`).
  - `OOD_64`: `D_gap = 64` (`seq_len = 82`, `>2.1x` max train length, well above legacy `pos_embed` limit `64`).
- **Random chance**: `1 / 32 = 0.03125`.

### Task 1B — 2-Channel Parity State Tracking (`Z_2 x Z_2`, OOD Length `64`)
- **Ops vocab**: `8` operation tokens `[32..39]` (`2` synonymous tokens per op `{NOOP, FLIP_CH0, FLIP_CH1, FLIP_BOTH}` so single-token shortcutting cannot bypass state abstraction).
- **Targets**: State `s_t in {40, 41, 42, 43}` (`2*p0 + p1 + 40`) supervised causally at EVERY step `t`.
- **Train length**: `L_train = 20`.
- **Eval splits**: `ID (L = 20)` and `OOD_64 (L = 64, 3.2x train length)`.
- **Random chance**: `0.25`.

### Task 1C — Cross-Episode Persistent LTM + Distractor Survival + Non-Interference
- **Training**: Joint meta-episode batches pairing:
  - Episodic pair `(ep_1, ep_2)`: `ep_1` contains `4` shared-vocab `[32..63]` bindings `[k_i, v_i]` plus filler; `Cog_Mem_*` runs differentiable `extract_episode_kv(ep_1)` (NO target labels from `ep_2`) and passes `episode_memory` into `forward(ep_2)` to predict `4` queries `[64, k_i] -> v_i`.
  - Parametric rule batch: `[65, tok]` (`tok in [32..63]`) -> `((tok - 32 + 1) % 32) + 32` interleaved cleanly during training with empty LTM.
- **Test protocol (`model.eval()`, zero gradient updates)**:
  1. `model.reset_ltm()`. Evaluate `KnownRule_Acc_Clean` on `[65, tok]` probes.
  2. Feed `ep_1` (`1` episode, `K=4` facts) -> `model.consolidate_ltm(ep_1)` (no labels).
  3. Feed `4` random unrelated distractor episodes (`shared [32..63]`) -> `model.consolidate_ltm(...)`.
  4. Evaluate `LTM_Recall_Acc` on `ep_2` WITHOUT `episode_memory` override (pure persistent slot buffer!).
  5. Feed `16` additional random distractor episodes (`~64` slots filled) and re-evaluate `KnownRule_Acc_Post64` -> `NonInterference_Drop = KnownRule_Acc_Clean - KnownRule_Acc_Post64`.

### Task 1D — Real Workspace Python Corpus Causal Next-Byte Compression (`BPB`)
- **File-disjoint split** (UTF-8 raw bytes `& 0xFF` in `0..255`, sliding non-overlapping windows):
  - **Train (`27` files)**: all `dmoe/*.py`.
  - **Val (`12` files)**: `tests/test_{attention,cognitive_architecture,complex_ops,continual_learner,continual_learning,decoder,encoder,end_to_end,experts,jepa,layer,manifold}.py`.
  - **Held-out Test (`12` files)**: `tests/test_{autonomous_lifecycle,dmoe_v2_easy_task,dmoe_v2_upgrades,dmoe_v3_setgate,dmoe_v4_mixed_stream,model_mutation,real_model_regression_gate,replay_buffer,simulation_loop,surprise_thinking,tokenizer_free,unified_novelty}.py`.
- **Eval metrics**: Exact float64 average held-out `BPB` (`NLL / ln(2)`) at `L=64` (ID window) and `L=128` (longer context window), plus CUDA wall-clock train ms/step & inference tok/s.

---

## 4. Splits, Seeds, and Tuning Budget

- **Tuning (`--mode tune`)**: Uses ONLY dev seeds `[100, 101, 102, 103, 104]` and Task 1D `Val` files. Every trial is appended to `results/stage1_causal_memory/tuning_log.json`. Bounded grid (`<= 12` configs on shared `lr / steps / min_nll`).
- **Frozen Evaluation (`--mode test`)**: Uses ONLY `20` disjoint test seeds `[0, 1, ..., 19]` and Task 1D `Held-out Test` files. Run ONCE under the frozen selected hyperparameters; full per-seed data saved to `results/stage1_causal_memory/results.json`. No test-seed iteration or protocol edits without opening a new `stage1_causal_memory_v2` PREREG.

---

## 5. Pre-Registered Falsifiers (Pass Criteria to Lock Stage 1)

Stage 1 locks ONLY if ALL of **F1–F8** pass on the frozen 20-seed test run:
- **F1 (MQAR Accuracy & Stability)**: `Cog_Mem_Hier` achieves `ID QueryAcc mean >= 0.90` (`min >= 0.75`, `SD <= 0.08`) AND `OOD-64 QueryAcc mean >= 0.85` (`min >= 0.65`).
- **F2 (MQAR Beats Windowed-NoSTM & GRU)**: `Cog_Mem_Hier` beats `Cog_NoSTM_Hier` by `>= +0.40` on `OOD-64` and beats `GRU_Baseline` by `>= +0.10` on `OOD-64`.
- **F3 (Parity Tracking Beats NoSigned & DenseRoPE)**: On Task 1B `OOD-64`, `Cog_Mem_Hier` achieves `mean >= 0.90` (`min >= 0.75`) and beats BOTH `Cog_NoSignedBeta` AND `DenseRoPE_Baseline` by `>= +0.15` (mean paired diff).
- **F4 (Cross-Episode LTM Survival)**: On Task 1C after `4` noisy episodes, `Cog_Mem_Hier` achieves `LTM_Recall_Acc mean >= 0.75` (`min >= 0.50`), beating single-episode stateless `DenseRoPE_Baseline` & `GRU_Baseline` (`~0.031`) by `>= +0.65`.
- **F5 (Null-Sink Non-Interference)**: `NonInterference_Drop` (`KnownRule_Acc_Clean - KnownRule_Acc_Post64`) has mean `<= 0.03` and max `<= 0.08`.
- **F6 (Real Code Next-Byte BPB)**: On Task 1D Held-out Test (`L=64`), `Cog_Mem_Hier` achieves `mean BPB <= DenseRoPE_Baseline mean BPB + 0.05` (no structural degradation vs matched 3-layer Transformer).
- **F7 (Null Controls Fail Cleanly)**:
  - `Null_Untrained`: MQAR `<= 0.08`, Parity `<= 0.35`, Task 1D `BPB >= 6.5`.
  - `Null_ShuffledTargets`: MQAR `<= 0.08`, Parity `<= 0.35`, Task 1D `BPB >= 5.0`.
  - `Null_BlindZeroInput`: MQAR `<= 0.08`, Parity `<= 0.35`, Task 1D `BPB >= 4.5`.
- **F8 (Causality Gate)**: Zero prefix leak (`<= 1e-5`) on all models before reporting.


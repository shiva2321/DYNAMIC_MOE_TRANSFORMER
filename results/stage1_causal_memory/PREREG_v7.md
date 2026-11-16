# Pre-Registration: Stage 1 Causal Memory & Pure-PyTorch Kernel Verification (`stage1_causal_memory_v7`)

- **Experiment ID**: `stage1_causal_memory_v7`
- **Predecessor**: `stage1_causal_memory_v6` (`results_v6.json`, preserved untouched alongside `v1`–`v5`)
- **Date**: 2026-11-15
- **Hypothesis**: Pure-PyTorch kernel/execution optimizations across `dmoe/memory.py`, `dmoe/hierarchical_attention.py`, `dmoe/real_model.py`, and CUDA fused `AdamW` + `zero_grad(set_to_none=True)` in `experiments/run_stage1_causal_memory.py` preserve the mathematical behavior, zero future-leak causality gate (`F8`), and all 8 Stage 1 falsifiers (`F1`–`F8`) of `stage1_causal_memory_v6`, while reducing Stage 1 training step latency (`1d_train_ms_step`) and total wall-clock runtime by roughly 35%–50% on Windows CUDA (RTX 3060, `torch 2.6.0+cu124`, no Triton).

## Scope of Code Changes vs `v6`
1. `dmoe/memory.py`:
   - `_get_scan_masks` cached masks (`@functools.lru_cache`).
   - `_chunked_gated_delta_scan(..., chunk_size=64, return_state=...)`: flattened `B*H*nc` batched execution, joint `solve_triangular(eye + M, rhs)` for `[W_u, A_mod]`, `nc == 1` single-chunk fast path, `init_state is None` chunk-0 skip, and skipping final state `S` when `return_state=False`.
   - `GatedDeltaWorkingMemory.forward`: fused input `F.linear`, single grouped `F.conv1d`, and `F.rms_norm`.
   - `LongTermSlotMemory`: CPU-cached `_num_used_slots` count (updated in `__init__`, `reset`, `_insert_kv_vectors`, `_load_from_state_dict`) so empty LTM reads incur zero `.nonzero()` CUDA-CPU synchronization.
2. `dmoe/hierarchical_attention.py`:
   - `ThreeLevelAttention`: cached masks (`_get_local_causal_mask`, `_get_mid_chunk_idx_and_mask`), `F.scaled_dot_product_attention`, fused input projections (`q_local/k_local/v_local/q_mid` and `k_mid/v_mid`), and fast skip of `_global_attention` when `memory_context` is empty.
3. `dmoe/real_model.py`:
   - `_zero_entropy` persistent=False buffer in `RealDynamicMorphicMoE`.
   - `_expert_frozen_flags` synchronized CPU mirror in `RealDynamicMorphicMoE` (updated on init, spawn, prune, freeze, and state-dict load) plus single-expert fast path and `_num_used_slots` guard in `RealSelfStabilizingDMoE.forward` to eliminate per-step CUDA-CPU syncs.
   - `return_state=False` on `RealDynamicMorphicLayer` internal `layer_stm` call; `math.log` replacing single-element GPU tensor syncs.
4. `experiments/run_stage1_causal_memory.py`:
   - Identical architectures, splits (`TRAIN_CODE_FILES`, `VAL_CODE_FILES`, `TEST_CODE_FILES`), seeds (`DEV_SEEDS = [100..104]`, `TEST_SEEDS = list(range(20))`), hyperparameters, and falsifier thresholds as `v6`; adds `fused=(device.type == "cuda")` on `AdamW` and `opt.zero_grad(set_to_none=True)`.

## Frozen Hyperparameters (Carried Unchanged from `v6`)
- `lr_cog = 3.5e-3`
- `lr_dense = 4.5e-3`
- `lr_gru = 4.5e-3`
- `steps_1a = 700`, `steps_1b = 350`, `steps_1c = 350`, `steps_1d = 300`
- `min_nll = 0.7`
- `Cog_Mem_Hier` / `Cog_Mem_Flat` / `Cog_NoSignedBeta`: `d_model=48, num_heads=4, d_ff=74, initial_layers=2, initial_experts=1, local_window=8, chunk_stride=4, num_ltm_slots=64, enable_layer_stm=True`

## Models & Null Controls (Test Suite `TEST_SEEDS = [0..19]`)
- Architecture: `Cog_Mem_Hier`
- Ablations: `Cog_Mem_Flat`, `Cog_NoSignedBeta`, `Cog_NoSTM_Hier`
- Parameter-matched Baselines (~158k–162k params): `DenseRoPE_Baseline`, `GRU_Baseline`
- Null Controls: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput` (evaluated on `Cog_Mem_Hier` per-seed weights)

## Falsifiers (`F1`–`F8`, Unchanged from `v6`)
1. `F1_mqar_acc_and_stability`: `Cog_Mem_Hier` `1a_id_acc.mean >= 0.90`, `min >= 0.75`, `std <= 0.08`, and `1a_ood64_acc.mean >= 0.85`, `min >= 0.65`.
2. `F2_mqar_beats_nostm_and_gru`: `Cog_Mem_Hier` `1a_ood64_acc.mean` beats `Cog_NoSTM_Hier` by `>= 0.40` and `GRU_Baseline` by `>= 0.10`.
3. `F3_parity_beats_nosigned_and_denserope`: `Cog_Mem_Hier` `1b_ood64_acc.mean >= 0.90`, `min >= 0.75`, and beats `Cog_NoSignedBeta` and `DenseRoPE_Baseline` by `>= 0.15` each.
4. `F4_ltm_survival`: `Cog_Mem_Hier` `1c_ltm_recall.mean >= 0.75`, `min >= 0.50`, and beats `DenseRoPE_Baseline` and `GRU_Baseline` by `>= 0.65` each.
5. `F5_non_interference`: `Cog_Mem_Hier` `1c_ni_drop.mean <= 0.03` and `max <= 0.08`.
6. `F6_real_code_bpb`: `Cog_Mem_Hier` `1d_bpb_64.mean <= DenseRoPE_Baseline 1d_bpb_64.mean + 0.05`.
7. `F7_null_controls`: All 3 null controls remain near chance across `1A` (`<= 0.08`), `1B` (`<= 0.35`), and `1D` (`>= 6.5` untrained, `>= 4.5` shuffled/blind; `Null_ShuffledTargets` `min - Cog_Mem_Hier max >= 1.5`).
8. `F8_causality_gate`: Future-token perturbation max abs logit diff `<= 1e-5` across all 6 trained/ablation architectures.

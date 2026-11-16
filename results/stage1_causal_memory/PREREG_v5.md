# Stage 1 Pre-Registration (`stage1_causal_memory_v5`)

**Status**: Pre-registered BEFORE running `--mode tune` or `--mode test` for `v5`.  
**Supersedes**: `v1` (baseline), `v2` (`results_v2.json`, 6/8 passed — `F3, F7` failed), `v3` (`results_v3.json`, 7/8 passed — `F3` failed on test seeds 6 & 15), `v4` (`results_v4.json`, 7/8 passed — `F3` failed on single test seed `4 = 0.563110` while 19/20 passed with `min_of_19 = 0.837952`, and `Cog_Mem_Flat` passed 20/20 `min = 0.946960`). All `v1`–`v4` files remain untouched on disk as permanent audit artifacts.

## 1. Failure Diagnosis from `v4` (Why `v5` Is Required)
Dev-only trajectory diagnostics (`task-3905`, `task-3909`, `task-3913`, `task-3923`, `task-3939`) on known failing/slow dev seeds (`Cog_Mem_Flat`: `100, 104, 116, 122, 131, 132`; `Cog_Mem_Hier`: `103`) isolated three exact structural defects causing residual ~2.5–7.5% parity traps / `OOD_64` decay:
1. **`gate_conv` Smearing into Token-Conditional Flip Gates (`dmoe/memory.py`)**: Feeding `gate_conv(x)` (`kernel=2` causal conv) into `w_beta`, `w_beta2`, and `w_alpha` mixed `x_{t-1}` into instantaneous token flip/retention gates `beta_t, beta2_t, alpha_t` while `w_k2(x)` read `x` directly. Computing `w_alpha(x)`, `w_beta(x)`, `w_beta2(x)` directly from `x` (keeping `conv_k`/`conv_q`/`conv_v` exclusively for lagged key-query-value associative binding) eliminates cross-token gate noise and removes the dead `gate_conv` submodule.
2. **Head 2 Deadzone (`beta2 = 1.0`) & Reflection-Head Retention Decay (`dmoe/memory.py`)**: Initializing `w_beta2.bias[2] = 0.0` yielded `beta2 = 1.0 -> (1 - beta2) = 0.0`, annihilating Head 2's `k2` projection on step 0, while `w_alpha.bias = 5.0` on Head 2 permitted `alpha` drift (`0.86–0.92`) that `RMSNorm` masked at `L=20` until `OOD_64` collapsed (`0.92^64 = 0.005`). Updating `H=4` initialization to:
   - `w_alpha.bias = [6.0, 5.5, 6.0, 6.0]`
   - `w_beta.bias = [-2.0, -1.0, 1.0, 1.5]` (when `signed_beta=True`)
   - `w_beta2.bias = [-4.0, -3.5, 1.0, 1.5]` (when `signed_beta=True`)
   activates both signed `DeltaProduct` reflection heads without a `beta = 1.0` deadzone.
3. **`ThreeLevelAttention` Idle-Global Capacity Drain (`dmoe/hierarchical_attention.py`)**: When `memory_context` is `None` or empty (`1A`, `1B`, `1D`), softmax gating over 3 branches allocated ~1/3 attention weight to a constant `glob_fallback`. Masking logit index `2` to `-1e9` when `memory_context` is empty restores 100% weight to `[Local, Mid]` without altering behavior when `memory_context` is present (`1C` and episodic retrieval).

Verified on `DEV_SEEDS` (`task-3923`, `task-3939`) prior to code lock:
- All 6 previously failing `Cog_Mem_Flat` dev seeds (`0.53–0.72`) rose to `mean=0.9961, min=0.9924`.
- `Cog_Mem_Hier` on `[100..104]` reached `1A=(0.9998 / 0.9760)`, `1B ood64=0.9974 (min=0.9913)`, `1C rec=0.9875 (min=0.9792, drop=0.0000)`, `1D val_bpb64=2.6452` (beating both `Cog_Mem_Flat 2.6593` and `DenseRoPE 2.756`), while `Cog_NoSignedBeta` remained `0.3438` on `1B`.

## 2. Frozen Experimental Protocol & Splits (Unchanged from `v4`)
- **Tasks (`steps_1a=700, steps_1b=350, steps_1c=350, steps_1d=300, min_nll=0.7`)**: Same `1A`, `1B`, `1C`, `1D` datasets and splits (`DEV_SEEDS=[100..104]`, `VAL_CODE_FILES` for tune; `TEST_SEEDS=[0..19]`, `TEST_CODE_FILES` for test).
- **Models & Parameter Counts** (exact counts verified at instantiation before test):
  - `Cog_Mem_Hier`: `157,649` params (`d_model=48, H=4, layers=2, d_ff=74, signed_beta=True, layer_stm delta_steps=2`)
  - `Cog_Mem_Flat`: `157,119` params (`d_model=48, H=4, layers=2, d_ff=244, enable_three_level_attn=False, enable_layer_stm=True`)
  - `Cog_NoSignedBeta`: `157,649` params (`d_model=48, H=4, layers=2, d_ff=74, signed_beta=False, layer_stm delta_steps=2`)
  - `Cog_NoSTM_Hier`: `151,542` params
  - `DenseRoPE_Baseline`: `158,464` params (`lr=3.5e-3`)
  - `GRU_Baseline`: `161,536` params (`lr=3.5e-3`)
  - Null controls: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`
- **Frozen Execution**: Verify `--mode tune` (`DEV_SEEDS=[100..104]`, `tuning_log_v5.json`) and 40-seed dev `1B` stability check (`[100..139]`) on real edited modules; then run frozen 20-seed `python -u experiments/run_stage1_causal_memory.py --mode test --lr_cog 3.5e-3 --lr_dense 3.5e-3 --lr_gru 3.5e-3` -> `results_v5.json`.

## 3. Frozen Falsifiers (`F1–F8`, Identical to `v3` & `v4` — ZERO Relaxation)
- **F1 (`F1_mqar_acc_and_stability`)**: `Cog_Mem_Hier` `1a_id_acc` `mean >= 0.90`, `min >= 0.75`, `std <= 0.08`; `1a_ood64_acc` `mean >= 0.85`, `min >= 0.65`.
- **F2 (`F2_mqar_beats_nostm_and_gru`)**: `Cog_Mem_Hier` `1a_ood64_acc.mean - Cog_NoSTM_Hier >= 0.40` and `- GRU_Baseline >= 0.10`.
- **F3 (`F3_parity_beats_nosigned_and_denserope`)**: `Cog_Mem_Hier` `1b_ood64_acc` `mean >= 0.90`, `min >= 0.75`, `- Cog_NoSignedBeta >= 0.15`, and `- DenseRoPE_Baseline >= 0.15`.
- **F4 (`F4_ltm_survival`)**: `Cog_Mem_Hier` `1c_ltm_recall` `mean >= 0.75`, `min >= 0.50`, `- DenseRoPE_Baseline >= 0.65`, `- GRU_Baseline >= 0.65`.
- **F5 (`F5_non_interference`)**: `Cog_Mem_Hier` `1c_ni_drop` `mean <= 0.03`, `max <= 0.08`.
- **F6 (`F6_real_code_bpb`)**: `Cog_Mem_Hier` `1d_bpb_64.mean <= DenseRoPE_Baseline.1d_bpb_64.mean + 0.05`.
- **F7 (`F7_null_controls`)**:
  - `Null_Untrained`: `1a_id_acc.max <= 0.08`, `1b_id_acc.max <= 0.35`, `1d_bpb_64.min >= 6.5`
  - `Null_ShuffledTargets`: `1a_id_acc.max <= 0.08`, `1b_id_acc.max <= 0.35`, `1d_bpb_64.min >= 4.5`, and `(Null_ShuffledTargets.1d_bpb_64.min - Cog_Mem_Hier.1d_bpb_64.max) >= 1.5`
  - `Null_BlindZeroInput`: `1a_id_acc.max <= 0.08`, `1b_id_acc.max <= 0.35`, `1d_bpb_64.min >= 4.5`
- **F8 (`F8_causality_gate`)**: `max_abs_diff <= 1e-5` on prefix-perturbation check across all 6 models.

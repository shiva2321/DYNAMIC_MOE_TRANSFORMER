# Stage 1 Pre-Registration (`stage1_causal_memory_v4`)

**Status**: Pre-registered BEFORE running `--mode tune` or `--mode test` for `v4`.  
**Supersedes**: `v1` (baseline), `v2` (`results_v2.json`, 6/8 passed — `F3` and `F7` failed), `v3` (`results_v3.json`, 7/8 passed — `F3` failed). All `v1`/`v2`/`v3` logs and result files remain untouched on disk as permanent audit history.

## 1. Failure Diagnosis from `v3` (Why `v4` Is Required)
In `stage1_causal_memory_v3` (`results_v3.json`), 7 of 8 falsifiers passed, including `F7_null_controls`, `F1_mqar_acc_and_stability`, `F2_mqar_beats_nostm_and_gru`, `F4_ltm_survival`, `F5_non_interference`, `F6_real_code_bpb`, and `F8_causality_gate`.  
However, **F3 failed** (`F3_parity_beats_nosigned_and_denserope: false`):
- `Cog_Mem_Hier` achieved `1b_ood64_acc` mean `0.940662`, std `0.137503`, **min `0.526917`** (failing `min >= 0.75` due to 2/20 outlier test seeds: seed `6 = 0.526917`, seed `15 = 0.577759`).
- `Cog_Mem_Flat` failed on 3/20 test seeds (`min = 0.557068`).

Disjoint dev-only (`DEV_SEEDS = [100..139]`, `VAL_CODE_FILES`) root-cause investigation (`task-3701..task-3787`) isolated the exact structural causes and validated a single architectural fix (`F1_layers_only_nh2_dff74`):
1. **Rank-1 Householder Limitation on 2-Channel Parity**: A single rank-1 update `(I - beta k k^T)` (`det = 1 - beta`, one reflection axis per step) cannot cleanly invert both independent parity axes simultaneously on `FLIP_BOTH` tokens, forcing brittle cross-head coordination that traps ~5–15% of initializations near `ln(2)` (one channel solved, second channel at chance `~0.50` past `t=5`). Adding a **2-step DeltaProduct (`delta_steps=2`)** in `layers[0..1].layer_stm` (two sequential rank-1 gated delta updates per token: normalized `F.silu(conv_k)` Step 1 + unconstrained normalized `w_k2(x)` with zero-init `w_v2(x) -> F.silu` Step 2, `alpha2 = 1.0`, `chunk_size=32`) resolves rank-2 parity inversion cleanly. `DualTimescaleMemory.short_term` keeps `delta_steps=1` for clean associative/LTM routing.
2. **State Scale Attenuation Across `OOD_64`**: Adding zero-safe per-head `RMSNorm` (`eps=1e-5`, learnable `o_norm_w`) to `S_t q_t` prior to `out_gate` prevents magnitude shrinkage over `L=64+` without leaking nonzero noise on zero state (`0 / sqrt(eps) = 0`).
3. **Associative Forget Bias & Beta Head Specialization**: Ordering `w_alpha.bias = [6.0, 5.5, 5.0, 5.5]` (when `H=4`) preserves slow associative MQAR heads across `OOD_64` distractors, while `w_beta.bias = [-2.0, -1.0, 0.2, 1.0]` and Step-2 `w_beta2.bias = [-4.0, -3.5, 0.0, 1.0]` separate smooth associative heads from state-flipping heads.
4. **Parameter Parity Preservation**: Adjusting `Cog_Mem_Hier` / `Cog_NoSignedBeta` `d_ff = 74` (`158,081` params) and `Cog_Mem_Flat` `d_ff = 244` (`157,551` params) keeps all cognitive variants at or below `DenseRoPE_Baseline` (`158,464` params) and `GRU_Baseline` (`161,536` params); `Cog_NoSTM_Hier` has `151,542` params.

## 2. Frozen Experimental Protocol & Splits (Unchanged from `v3`)
- **Tasks**:
  - **1A (Shared-Vocab MQAR)**: `VOCAB_START=32, VOCAB_SIZE=48, QUERY_MARK=16`, train `k=5, n_q=4, gap in [6..18]`, eval ID (`gap=12`) and OOD (`gap=64`), `steps_1a=700, batch_size=64`.
  - **1B (2-Channel Cumulative Parity `Z_2 x Z_2`)**: Train length `L=20`, eval ID (`L=20`) and OOD (`L=64`), `steps_1b=350, batch_size=64`.
  - **1C (Cross-Episode LTM Survival & 64-Slot Non-Interference)**: Differentiable Episode-1 -> 4 distractor episodes -> Episode-2 recall (`1c_ltm_recall`) + clean-vs-64-random-slots accuracy drop (`1c_ni_drop`), `steps_1c=350, batch_size=64, min_nll=0.7`.
  - **1D (Held-Out Real Python Next-Byte `BPB`)**: `TRAIN_CODE_FILES` (27 files), `VAL_CODE_FILES` (12 files, `--mode tune`), `TEST_CODE_FILES` (12 files, `--mode test`) at `L=64` (`w[:, :-1] -> w[:, 1:]`) and `L=128`, `steps_1d=300, batch_size=64`.
- **Models**:
  - Primary: `Cog_Mem_Hier` (`d_model=48, H=4, layers=2, d_ff=74, memory.short_term delta_steps=1, layer_stm delta_steps=2, o_norm + high-alpha/asym-beta`; `158,081` params)
  - Ablations: `Cog_Mem_Flat` (`d_model=48, H=4, layers=2, d_ff=244, enable_three_level_attn=False, enable_layer_stm=True`; `157,551` params), `Cog_NoSignedBeta` (`d_model=48, H=4, layers=2, d_ff=74, signed_beta=False, layer_stm delta_steps=2`; `158,081` params), `Cog_NoSTM_Hier` (`d_model=48, H=4, layers=2, d_ff=352, enable_memory=False, enable_layer_stm=False`; `151,542` params)
  - Baselines: `DenseRoPE_Baseline` (`d_model=64, H=4, d_ff=192, layers=3`; `158,464` params), `GRU_Baseline` (`d_model=96, layers=2`; `161,536` params)
  - Null controls: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`
- **Seeds & Tuning Discipline**:
  - Dev/sanity checks (including pre-test check on `run_stage1_causal_memory.py` with real modules): `DEV_SEEDS = [100..104]` (`--mode tune`, trying `lr in {2e-3, 3.5e-3}`) and disjoint dev parity pool `[100..139]`, `VAL_CODE_FILES`.
  - Locked Evaluation: `TEST_SEEDS = [0..19]` (20 independent paired seeds per model per task), run **once** with frozen hyperparameters selected from `--mode tune`.

## 3. Frozen Falsifiers (`F1–F8`, Identical to `v3` — ZERO Threshold Relaxation)
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
- **F8 (`F8_causality_gate`)**: `max_abs_diff <= 1e-5` on prefix-perturbation check across all 6 trained/baseline models.

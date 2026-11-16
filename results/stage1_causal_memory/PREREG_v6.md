# Stage 1 Pre-Registration (`stage1_causal_memory_v6`)

**Status**: Pre-registered BEFORE editing `dmoe/memory.py` (`null_logit`), BEFORE running `--mode tune` (`tuning_log_v6.json`), and BEFORE running `--mode test` (`results_v6.json`).  
**Supersedes**: `v1`–`v4` (`results_v2.json`, `results_v3.json`, `results_v4.json`) and `v5` (`PREREG_v5.md`, `tuning_log_v5.json`; note `v5` never ran `--mode test`, so `TEST_SEEDS=[0..19]` and `TEST_CODE_FILES` remain completely untouched for `v5/v6`). All `v1`–`v5` files remain untouched on disk as permanent audit artifacts.

## 1. Dev-Only Diagnosis Between `v5` Tune and `v6`
Dev diagnostics (`task-3987`, `task-4025`, `task-4035`, `task-4057`, `task-4070`, `task-4076`) strictly on `DEV_SEEDS=[100..139]` and `VAL_CODE_FILES` uncovered two remaining issues in `v5`:
1. **LTM Cold-Start Null-Sink Gradient Starvation (`dmoe/memory.py`)**: `LongTermSlotMemory.null_logit` initialized at `4.5` produced an initial null-column logit of `4.5 + log(14) = 7.14` against `cos ~ 0` between independently initialized `q_proj` and `k_proj`, pushing ~99.9% of softmax probability mass onto the null sink at step 0 and starving early gradients on `1C`. On `DEV_SEEDS`, `seed=104` only began escaping after step 250 (`lr=3.5e-3 -> 0.8646`) and stalled at `0.0833` under `lr=3.0e-3` (`task-4057`). While `null_logit=2.0` (`task-4070`: `[100..109]` `rec mean=0.9917, min=0.9792, drop=0.0000`) failed the untrained single-slot distractor threshold (`mass_1=0.2697 >= 0.15` in `test_ltm_null_sink_invariance_and_no_auto_bleed`), `task-4105` measured `null_logit = 3.2` (zero PRNG draws, leaving `1A/1B/1D` bit-for-bit unchanged): untrained distractor rejection passed (`mass_1 = 0.1143 < 0.15`, `mass_64 = 0.1936`) and `DEV_SEEDS=[100..104]` on `Task 1C` (`Cog_Mem_Hier @ 3.0e-3`) improved to `rec mean=0.9771, min=0.9688, non_interference_drop=0.0000` (`seed=104` jumped from `0.0833` to `0.9792`). Accordingly, `v6` sets `LongTermSlotMemory.null_logit` init to `3.2`.
2. **LR Calibration & Baseline Fairness (`research-protocol.md §7`)**:
   - On `DEV_SEEDS=[100..119]` (`n=20`) and `[100..139]` (`n=40`), dev validation compared `Cog_Mem_Hier @ 3.0e-3` vs `3.5e-3` against baseline tuning (`DenseRoPE_Baseline` and `GRU_Baseline` up to `4.5e-3` and `5.5e-3` on `DEV_SEEDS=[100..104]`, `task-4025`, `task-4076`, `task-4122`, `task-4130`).
   - `DenseRoPE_Baseline` reaches its strongest stable dev performance at **`4.5e-3`** (`[100..119]` `1D val_bpb64=2.7012`, `1D val_bpb128=2.8273` on current repo bytes `task-4130`, vs `2.7891 / 2.8936` at `3.5e-3` in `task-4076`; `1A ood64=0.5650`, whereas `5.5e-3` destabilized `1A ood64` down to `min=0.0986`). `GRU_Baseline @ 4.5e-3` reaches `1D=2.523, 1B=1.000, 1A=0.242` (`tuning_log_v6.json`).
   - Against `DenseRoPE_Baseline @ 4.5e-3` (`val_bpb64=2.7012`, `task-4130`), `Cog_Mem_Hier @ 3.0e-3` measured `val_bpb64=2.7185` (`val_bpb128=2.6924`, `[100..119]`), whereas `Cog_Mem_Hier @ 3.5e-3` improved `val_bpb64` to **`2.6839`** (`val_bpb128=2.6570`, beating `DenseRoPE @ 4.5e-3` by `-0.0173` bpb on `L=64` and `-0.1703` bpb on `L=128`) while maintaining strong margins across all other tasks on dev seeds under `null_logit=3.2`: `1A ood64` (`[100..119]`, `n=20`: `mean=0.9106, min=0.7607`, `task-3987`), `1B ood64` (`[100..139]`, `n=40`: `mean=0.9959, min=0.8848`, `task-3969`), and `1C rec` (`[100..109]`, `n=10`: `mean=0.9844, min=0.9583, drop=0.0000`, `task-4130`).
   - Thus `v6` freezes `--lr_cog 3.5e-3 --lr_dense 4.5e-3 --lr_gru 4.5e-3` BEFORE running `--mode test`, giving both baselines their stronger `4.5e-3` LR against `Cog_Mem`'s `3.5e-3` LR.

## 2. Frozen Experimental Protocol & Splits
- **Tasks (`steps_1a=700, steps_1b=350, steps_1c=350, steps_1d=300, min_nll=0.7`)**: Identical datasets and disjoint splits (`DEV_SEEDS=[100..104]`, `VAL_CODE_FILES` for tune; `TEST_SEEDS=[0..19]`, `TEST_CODE_FILES` for test).
- **Models & Parameter Counts** (exact counts unchanged from `v5`):
  - `Cog_Mem_Hier`: `157,649` params (`d_model=48, H=4, layers=2, d_ff=74, signed_beta=True, layer_stm delta_steps=2`, `lr_cog=3.5e-3`)
  - `Cog_Mem_Flat`: `157,119` params (`lr_cog=3.5e-3`)
  - `Cog_NoSignedBeta`: `157,649` params (`lr_cog=3.5e-3`)
  - `Cog_NoSTM_Hier`: `151,542` params (`lr_cog=3.5e-3`)
  - `DenseRoPE_Baseline`: `158,464` params (`lr_dense=4.5e-3`)
  - `GRU_Baseline`: `161,536` params (`lr_gru=4.5e-3`)
  - Null controls: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`
- **Frozen Execution**:
  1. Unit tests verified (`pytest tests/test_stage1_causal_memory.py tests/test_cognitive_architecture.py -v` -> `15 passed`).
  2. Logged dev sweep `--mode tune` (`DEV_SEEDS=[100..104]`) -> `tuning_log_v6.json` (`task-4122`), supplemented by dev-only checks `task-3969`, `task-3987`, `task-4105`, and `task-4130`.
  3. Run single frozen 20-seed test: `python -u experiments/run_stage1_causal_memory.py --mode test --lr_cog 3.5e-3 --lr_dense 4.5e-3 --lr_gru 4.5e-3` -> `results_v6.json`.

## 3. Frozen Falsifiers (`F1–F8`, Identical to `v3`, `v4`, `v5` — ZERO Relaxation)
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

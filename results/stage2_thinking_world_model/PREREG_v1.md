# Pre-Registration: Stage 2 Dynamic Thinking Loop, Epistemic Abstention & Conflict Catching, and Calibrated Latent World Model (`stage2_thinking_world_model_v1`)

- **Experiment ID**: `stage2_thinking_world_model_v1`
- **Date**: 2026-11-15
- **Locked Stage 1 Predecessor**: `stage1_causal_memory_v8` (`results/stage1_causal_memory/results_v8.json`, 8/8 falsifiers passed on `TEST_SEEDS = [0..19]`).
- **Script & Output**: `experiments/run_stage2_thinking_world_model.py --mode test --workers 5` -> `results/stage2_thinking_world_model/results_v1.json`

## 1. Claim & Hypothesis
At matched parameter scale (~`196K`–`228K` params) and under strict causal masking (`causality_gate_max_diff <= 1e-5`), augmenting the Stage 1 cognitive backbone (`RealSelfStabilizingDMoE`) with the causal RoPE dynamic thinking loop (`SurpriseDrivenLatentThinker`) and calibrated multi-step latent world model (`LatentWorldModel`) achieves:
1. Iterative out-of-distribution multi-hop reasoning generalization (`2A-Adj` and distractor-gapped `2A-Ptr`, trained on depth `D <= 3` and evaluated on `D = 4..5`), beating `Cog_OnePass` (`force_thinking_steps = 0`) and monotonically improving from `step0_ood45` to `step5_ood45` while halting adaptively to save active token compute relative to 100%-compute `Cog_FixedThink`.
2. Zero-shot epistemic abstention and contradiction detection on shared vocabulary `[32..63]` (`Task 2B`, trained strictly on `clean` episodes only): calibrated answer uncertainty separation (`auroc_abstain_all` on `unknown` + `rule_conflict`) and world-model predictive-coding contradiction detection (`auroc_pc_contradiction` on `obs_conflict`, exceeding `Cog_NoWM` which lacks predictive coding).
3. Calibrated multi-horizon latent imagination (`Task 2C`, modular trajectory dynamics on `[32..95]`, horizons `H = 1..3`) with monotonically increasing horizon uncertainty (`unc_h1 < unc_h2 < unc_h3`), low calibration error (`wm_ece`), strong corrupted-trajectory separation (`corrupt_auroc`), and zero regression on downstream trajectory prediction relative to `Cog_NoWM`.

## 2. Disjoint Splits, Tuning Budget, & DEV Summary (`tuning_log_v1.json`)
- **Validation (`DEV`) seeds**: `DEV_SEEDS = [100, 101, 102, 103, 104]` (`n = 5`), logged in `results/stage2_thinking_world_model/tuning_log_v1.json` (`13` entries total).
- **Held-out Test seeds (frozen, untouched in `v1` prior to this pre-registration)**: `TEST_SEEDS = [0..19]` (`n = 20`).
- **Baselines & Ablations Tuning (`DEV_SEEDS = [100..104]`)**:
  - `LoopedTransformer_Baseline` (`227,993` params; tuned `lr in {3e-3, 5e-3, 7e-3}`):
    - `3e-3`: `2a_adj_ood45 = 0.7604`, `2a_ptr_ood45 = 0.8878`, `2b_clean_acc = 0.0833`, `2c_traj_acc = 0.9531`
    - `5e-3`: `2a_adj_ood45 = 0.8005`, `2a_ptr_ood45 = 0.9328`, `2b_clean_acc = 0.0719`, `2c_traj_acc = 0.9529`
    - `7e-3` (**frozen**): `2a_adj_ood45 = 0.8102`, `2a_ptr_ood45 = 0.9540`, `2b_clean_acc = 0.0729`, `2b_auroc_abstain_all = 0.5095`, `2b_auroc_pc_contradiction = 0.4812`, `2c_traj_acc = 0.9527`, `2c_corrupt_auroc = 0.9917`
  - `DenseRoPE_Deep` (`210,496` params; tuned `lr in {3e-3, 5e-3, 7e-3}`):
    - `3e-3`: `2a_adj_ood45 = 0.6515`, `2a_ptr_ood45 = 0.7085`, `2b_clean_acc = 0.1125`, `2c_traj_acc = 0.9532`
    - `5e-3` (**frozen**): `2a_adj_ood45 = 0.6620`, `2a_ptr_ood45 = 0.7435`, `2b_clean_acc = 0.0656`, `2b_auroc_abstain_all = 0.5033`, `2b_auroc_pc_contradiction = 0.5034`, `2c_traj_acc = 0.9532`, `2c_corrupt_auroc = 0.9957`
    - `7e-3`: `2a_adj_ood45 = 0.6669`, `2a_ptr_ood45 = 0.7134`, `2b_clean_acc = 0.0583`, `2c_traj_acc = 0.9533`
  - `GRU_Baseline` (`217,408` params; tuned `lr in {3e-3, 5e-3, 7e-3}`):
    - `3e-3`: `2a_adj_ood45 = 0.9677`, `2a_ptr_ood45 = 0.9853`, `2b_clean_acc = 0.2260`, `2c_traj_acc = 0.9540`
    - `5e-3`: `2a_adj_ood45 = 0.9835`, `2a_ptr_ood45 = 0.9973`, `2b_clean_acc = 0.2094`, `2c_traj_acc = 0.9539`
    - `7e-3` (**frozen**): `2a_adj_ood45 = 0.9878`, `2a_ptr_ood45 = 0.9961`, `2b_clean_acc = 0.2052`, `2b_auroc_abstain_all = 0.5085`, `2b_auroc_pc_contradiction = 0.4828`, `2c_traj_acc = 0.9538`, `2c_corrupt_auroc = 0.9982`
  - Proposed & Ablation Variants at `lr_cog = 3e-3` (**frozen**):
    - `Cog_Stage2_Full` (`218,595` params): `2a_adj_ood45 = 0.9232` (`step0=0.5368 -> step5=0.9316`, `act=0.3597`), `2a_ptr_ood45 = 0.8250` (`step0=0.5825 -> step5=0.8364`, `act=0.5637`), `2b_clean_acc = 0.9896`, `2b_auroc_abstain_all = 0.8877`, `2b_auroc_pc_contradiction = 0.9398`, `2c_traj_acc = 0.9535`, `2c_cos_h1 = 0.7855`, `2c_unc_h = [0.2396, 0.5256, 0.5883]`, `2c_wm_ece = 0.0183`, `2c_corrupt_auroc = 0.9938`.
    - `Cog_OnePass` (`196,007` params): `2a_adj_ood45 = 0.6903` (`act=0.0000`), `2a_ptr_ood45 = 0.6566` (`act=0.0000`), `2b_clean_acc = 0.9948`, `2b_auroc_abstain_all = 0.6551`, `2b_auroc_pc_contradiction = 0.5121`, `2c_traj_acc = 0.9535`, `2c_corrupt_auroc = 0.9279`.
    - `Cog_NoWM` (`196,007` params): `2a_adj_ood45 = 0.9124` (`act=0.3580`), `2a_ptr_ood45 = 0.8516` (`act=0.5569`), `2b_clean_acc = 0.9646`, `2b_auroc_abstain_all = 0.8768`, `2b_auroc_pc_contradiction = 0.4822`, `2c_traj_acc = 0.9527`, `2c_corrupt_auroc = 0.9935`.
    - `Cog_FixedThink` (`218,595` params): `2a_adj_ood45 = 0.9281` (`act=1.0000`), `2a_ptr_ood45 = 0.8411` (`act=1.0000`), `2b_clean_acc = 0.8010`, `2b_auroc_abstain_all = 0.8420`, `2b_auroc_pc_contradiction = 0.8571`, `2c_traj_acc = 0.9526`, `2c_cos_h1 = 0.7763`.

## 3. Baseline Configuration Verification (`research-protocol.md §7`) & Expected Trade-offs
- **Task 2B baseline sanity check on DEV**: Under the matched `steps_2b = 520` budget, `DenseRoPE_Deep`, `LoopedTransformer_Baseline`, and `GRU_Baseline` fail to grok Task 2B (`0.0656–0.2052`), reflecting lower sample efficiency than Stage 1 STM. Extending `DenseRoPE_Deep` (`lr=3e-3`) to `steps_2b = 1000..1500` on DEV seeds `100–101` confirms it is properly configured (`clean_acc = 1.0000` on seed 100 in extended sweep, and `clean_acc = 0.9948`, `auroc_abstain_all = 0.9120` on seed 101), yet even after grokking clean lookup it remains at chance on world-model predictive-coding contradiction catching (`auroc_pc_contradiction = 0.4862` vs `Cog_Stage2_Full 0.9999` at `1000` steps).
- **Anticipated comparator trade-offs**:
  - `GRU_Baseline` dominates pure finite-state Markov transitions (`2A-Adj = 0.9878`, `2A-Ptr = 0.9961` on DEV) by maintaining `O(1)` recurrent state, while failing Task 2B rule binding/abstention (`0.2052`) and lacking a rollout world model (`cos_h1 = 0.0`).
  - `LoopedTransformer_Baseline` reaches higher fixed-compute accuracy on `2A-Ptr` (`0.9540` vs `0.8250` on DEV), while trailing on `2A-Adj` (`0.8102` vs `0.9232`), using `100%` active tokens vs `~36%`/`~56%`, failing Task 2B (`0.0729`), and lacking a rollout world model (`cos_h1 = 0.0`).

## 4. Frozen Protocol & Null Controls (`TEST_SEEDS = [0..19]`)
- **Hyperparameters**:
  - `lr_cog = 3e-3`, `lr_looped = 7e-3`, `lr_dense = 5e-3`, `lr_gru = 7e-3`
  - `steps_2a_adj = 340`, `steps_2a_ptr = 700`, `steps_2b = 520`, `steps_2c = 220`
  - `cog_common`: `d_model=48, num_heads=4, d_ff=74, initial_layers=2, initial_experts=1, chunk_size=1, num_ltm_slots=64, local_window=8, chunk_stride=4, max_thinking_steps=5, skip_threshold=0.05, theta_halt=0.06, abstain_threshold=0.65`, `wm_rollout_horizon=3`, `wm_weight=0.25`.
  - `think_train_steps = 4` in `2A` (for both `Cog` and `LoopedTransformer_Baseline`); `3` default in `2B`/`2C` (`5` for `Cog_FixedThink`, `0` for `Cog_OnePass`).
- **Pre-registered DEV Null Controls (`seed=100`)**:
  - `Null_Untrained`: `2a_ptr_ood45 = 0.0054`, `2a_adj_ood45 = 0.0000`, `2b_clean_acc = 0.0000`, `2c_traj_acc = 0.0039`.
  - `Null_ShuffledTargets`: `2a_ptr_ood45 = 0.0000`, `2a_adj_ood45 = 0.1598`, `2b_clean_acc = 0.0573`, `2c_traj_acc = 0.0188`.
  - `Null_BlindZeroInput` (evaluated on trained `Cog_Stage2_Full` weights with input zeroed): `2a_ptr_ood45 = 0.1366`, `2a_adj_ood45 = 0.0879`, `2b_clean_acc = 0.0000`, `2c_traj_acc = 0.0137`.

## 5. Pre-Registered Falsifiers (`F1`–`F7`, Frozen Before `TEST_SEEDS = [0..19]`)
All 7 boolean falsifiers must hold simultaneously on `TEST_SEEDS = [0..19]` (`all_passed == True`):
1. `F1_causality_gate`: Future-token perturbation max abs logit/uncertainty diff `<= 1e-5` across `Cog_Stage2_Full`, `LoopedTransformer_Baseline`, `DenseRoPE_Deep`, `GRU_Baseline`.
2. `F2_thinking_beats_onepass_and_step5_gt_step0`:
   - `Cog_Stage2_Full` `2a_adj_ood45.mean > Cog_OnePass 2a_adj_ood45.mean + 0.18`
   - `Cog_Stage2_Full` `2a_ptr_ood45.mean > Cog_OnePass 2a_ptr_ood45.mean + 0.12`
   - `Cog_Stage2_Full` `2a_adj_step5_ood45.mean > Cog_Stage2_Full 2a_adj_step0_ood45.mean + 0.25`
   - `Cog_Stage2_Full` `2a_ptr_step5_ood45.mean > Cog_Stage2_Full 2a_ptr_step0_ood45.mean + 0.18`
3. `F3_adaptive_compute_efficiency`:
   - `Cog_Stage2_Full` `2a_adj_ood45.mean >= 0.85` and `2a_ptr_ood45.mean >= 0.78`
   - `Cog_Stage2_Full` `2a_adj_active_frac.mean <= 0.50` and `2a_ptr_active_frac.mean <= 0.65`
   - `Cog_Stage2_Full` `2a_ptr_ood45.mean >= Cog_FixedThink 2a_ptr_ood45.mean - 0.04`
4. `F4_abstention_and_pc_conflict`:
   - `Cog_Stage2_Full` `2b_clean_acc.mean >= 0.90`
   - `Cog_Stage2_Full` `2b_auroc_abstain_all.mean >= 0.85`
   - `Cog_Stage2_Full` `2b_auroc_pc_contradiction.mean >= 0.85` and exceeds `Cog_NoWM` by `> 0.25`
5. `F5_world_model_no_traj_regression`:
   - `Cog_Stage2_Full` `2c_traj_acc.mean >= Cog_NoWM 2c_traj_acc.mean - 0.005`
   - `Cog_Stage2_Full` `2c_traj_acc.mean >= 0.90` and `2c_cos_h1.mean >= 0.70`
6. `F6_wm_horizon_and_corrupt_calibration`:
   - `Cog_Stage2_Full` `2c_unc_h1.mean < 2c_unc_h2.mean < 2c_unc_h3.mean`
   - `Cog_Stage2_Full` `2c_wm_ece.mean <= 0.10` and `2c_corrupt_auroc.mean >= 0.90`
7. `F7_null_controls_fail`: For every `nm in {Null_Untrained, Null_ShuffledTargets, Null_BlindZeroInput}`:
   - `2a_ptr_ood45.mean < 0.20`, `2b_clean_acc.mean < 0.15`, and `2c_traj_acc.mean < 0.15`.

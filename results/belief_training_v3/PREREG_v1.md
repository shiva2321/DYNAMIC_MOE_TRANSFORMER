# Pre-Registration: SPGC (`belief_training_v3` & `RealSelfStabilizingDMoE` Stage 3B-Mem)

- **Version**: `v1` (FROZEN prior to executing `TEST_SEEDS = [0..19]`)
- **Timestamp (UTC)**: `2026-11-19T02:35:00Z`
- **Git Commit Base**: `9eec10248a73b545fbc26c573c6ae6b5b221f98b`
- **Core Modules**:
  - `dmoe/spgc.py`
  - `dmoe/continual.py` (`enable_spgc=True` opt-in hooks; locked `enable_spgc=False` paths unmodified)
- **Frozen Evaluation Scripts**:
  - Suite 1: `experiments/run_belief_training_v3.py`
  - Suite 2: `experiments/verify_spgc_real_dmoe.py`
- **Protocol Anchor**: Zero test seeds (`0..19`) have been executed for either Suite 1 or Suite 2. No hyperparameters, thresholds, splits, or loss definitions will be altered after launching `--mode test`. Locked artifacts (`results/stage3_continual_learning/`, `results/belief_training_lab/results_v2.json`) remain untouched.

---

## 1. Suite 1: Compositional Belief Revision & Efficiency Benchmark (`run_belief_training_v3.py`)

### 1.1 Exact Command & Splits
- **DEV Tuning Seeds (locked Entry #3 in `results/belief_training_v3/tuning_log_v1.json`)**: `DEV_SEEDS = [100, 101, 102, 103, 104]` (`n = 5`)
- **Confirmatory TEST Seeds**: `TEST_SEEDS = [0, 1, ..., 19]` (`n = 20`)
- **Exact Confirmatory CLI Command**:
  ```bash
  python -u experiments/run_belief_training_v3.py --mode test --out_name results_v1.json
  ```
- **Output Path**: `results/belief_training_v3/results_v1.json`

### 1.2 Experimental Arms (14 total)
- **Baselines**: `B0_SurfaceDict`, `B1_Dense_SGD`, `B2_Dense_Replay`, `B4_JointOracle`, `V2_Backbone_BONG`
- **Null Controls**: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`
- **SPGC & Mechanistic Ablations**: `SPGC_Full`, `SPGC_NoMuon`, `SPGC_NoSkip`, `SPGC_NoQuarantine`, `SPGC_NoBONG`, `SPGC_NoReplay`

### 1.3 Frozen `V3Config` Hyperparameters
- `batch_size = 128`, `steps_p1 = 3600`, `steps_pb = 10`, `steps_p2 = 800`, `steps_p3 = 600`, `confirm_frac = 0.25`
- `spgc_lr_muon = 1.5e-2`, `spgc_lr_adam = 6e-3`, `spgc_lr_consolidate = 1.0e-2`
- `spgc_wd_trunk_p1 = 0.20`, `spgc_wd_fast = 0.01`, `spgc_wd_post = 0.01`
- `spgc_p1_min_steps = 850`, `spgc_p1_skip_ce = 0.12`, `spgc_p1_refresh_period = 64`
- `spgc_min_flush_p1 = 32`, `spgc_min_flush_post = 16`, `spgc_tau_skip = 0.12`
- `spgc_conf_established = 0.45`, `spgc_min_flip_count = 6`, `spgc_flip_ratio = 0.35`, `spgc_min_sources = 2`, `spgc_rev_grace_steps = 24`, `spgc_replay_k = 16`
- `spgc_bong_gamma = 0.996`, `spgc_bong_damping = 0.01`, `spgc_bong_init_prec = 0.02`, `spgc_bong_min_prec = 0.005`, `spgc_bong_max_prec = 100.0`, `spgc_bong_prior_pull = 0.0`
- `max_store_bytes = 16384`

### 1.4 DEV Anchor Values (`tuning_log_v1.json` Entry #3, `n = 5`)
- `B0_SurfaceDict`: `p1_vh=0.056 pb_ret=0.066 p2_min=0.019 p3_eq=0.095 p3_ret=0.051 all=0.065 nll=2.766 fl_tot=0.00e+00 t=1.13s`
- `B1_Dense_SGD`: `p1_vh=0.656 pb_ret=0.082 p2_min=0.982 p3_eq=0.424 p3_ret=0.668 all=0.615 nll=1.426 fl_tot=2.01e+12 bp=3600+1410 t=31.56s`
- `B2_Dense_Replay`: `p1_vh=0.742 pb_ret=0.079 p2_min=0.985 p3_eq=0.359 p3_ret=0.637 all=0.597 nll=1.644 fl_tot=2.08e+12 bp=3600+1410 t=31.07s`
- `B4_JointOracle`: `p1_vh=0.795 pb_ret=0.801 p2_min=0.765 p3_eq=0.905 p3_ret=0.863 all=0.852 nll=0.321 fl_tot=2.01e+12 bp=5000+0 t=30.19s`
- `V2_Backbone_BONG`: `p1_vh=0.792 pb_ret=0.168 p2_min=0.977 p3_eq=0.895 p3_ret=0.746 all=0.784 nll=0.645 fl_tot=1.86e+12 bp=3600+573 t=50.29s`
- `Null_Untrained`: `p1_vh=0.052 pb_ret=0.065 p2_min=0.041 p3_eq=0.073 p3_ret=0.058 all=0.061`
- `Null_ShuffledTargets`: `p1_vh=0.066 pb_ret=0.058 p2_min=0.000 p3_eq=0.034 p3_ret=0.055 all=0.052`
- `Null_BlindZeroInput`: `p1_vh=0.047 pb_ret=0.067 p2_min=0.196 p3_eq=0.265 p3_ret=0.079 all=0.136`
- `SPGC_Full`: `p1_vh=0.937 pb_ret=1.000 (flips=0.00) p2_min=0.988 p3_eq=0.827 p3_ret=0.984 all=0.942 nll=0.223 fl_tot=1.14e+12 bp=984+394 (skip_p1=0.727, skip_post=0.720) t=24.95s store_max=15104 aux=480806`
- `SPGC_NoMuon`: `p1_vh=0.520 pb_ret=0.989 p2_min=0.987 p3_eq=0.788 p3_ret=0.919 all=0.831 nll=1.257 fl_tot=1.31e+12 bp=2202+472 t=29.95s`
- `SPGC_NoSkip`: `p1_vh=0.963 pb_ret=0.994 p2_min=0.924 p3_eq=0.617 p3_ret=0.953 all=0.881 nll=0.559 fl_tot=2.80e+12 bp=3600+1400 t=48.57s`
- `SPGC_NoQuarantine`: `p1_vh=0.937 pb_ret=0.498 p2_min=0.975 p3_eq=0.808 p3_ret=0.985 all=0.938 nll=0.214 fl_tot=1.15e+12 bp=984+397 t=24.43s`
- `SPGC_NoBONG`: `p1_vh=0.937 pb_ret=1.000 p2_min=0.686 p3_eq=0.245 p3_ret=0.140 all=0.233 nll=4.923 fl_tot=1.45e+12 bp=984+1398 t=32.99s`
- `SPGC_NoReplay`: `p1_vh=0.937 pb_ret=1.000 p2_min=0.995 p3_eq=0.820 p3_ret=0.983 all=0.928 nll=0.275 fl_tot=1.12e+12 bp=984+390 t=24.18s`

### 1.5 Frozen Primary Falsifiers (`V1..V6`)
Every falsifier must pass on `TEST_SEEDS = [0..19]` for Suite 1 claim elevation to `GENERALIZES`:
1. **`V1_beats_or_matches_replay_with_less_total_flops_and_time`**:
   - `SPGC_Full.p3_overall_acc >= B2_Dense_Replay.p3_overall_acc`
   - `SPGC_Full.flops_total <= 0.80 * B2_Dense_Replay.flops_total`
   - `SPGC_Full.wall_clock_sec <= B2_Dense_Replay.wall_clock_sec`
2. **`V2_burst_immunity_over_replay_and_v2bong`**:
   - `SPGC_Full.pb_retention >= 0.80`
   - `SPGC_Full.pb_burst_flips <= 0.10`
   - `paired(SPGC_Full - B2_Dense_Replay).pb_retention.ci95_low > 0.50`
   - `paired(SPGC_Full - V2_Backbone_BONG).pb_retention.ci95_low > 0.40`
3. **`V3_context_split_and_equiv_transfer_intact`**:
   - `SPGC_Full.p2_split_both_min >= 0.85`
   - `SPGC_Full.p3_rev_equiv_transfer >= 0.75`
   - `paired(SPGC_Full - B2_Dense_Replay).p3_rev_equiv_transfer.ci95_low > 0.05`
4. **`V4_calibrated_probabilities`**:
   - `SPGC_Full.p3_overall_nll <= B2_Dense_Replay.p3_overall_nll`
   - `SPGC_Full.p3_overall_nll <= 0.90`
5. **`V5_budget_honesty`**:
   - `max_seed(SPGC_Full.store_bytes) <= 16384.0`
   - Optimizer auxiliary precision memory (`aux_memory_bytes`, `~480.8 KB` on DEV) is logged and disclosed separately.
6. **`V6_null_controls_fail`**:
   - For each of `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`:
     `p1_val_heldout < 0.15`, `p2_split_both_min <= 0.30`, `p3_rev_equiv_transfer <= 0.30`, and `p3_overall_acc < 0.15` (reflecting the 4-concept focused sub-split `1/4 = 0.25` mode baseline).

### 1.6 Secondary Mechanistic Ablation Expectations
- `SPGC_NoMuon`: lower `p1_val_heldout` and higher P1 compute vs `SPGC_Full`.
- `SPGC_NoSkip`: loss of FLOP and wall-clock advantage over `B2_Dense_Replay`.
- `SPGC_NoQuarantine`: sharp drop in `pb_retention` vs `SPGC_Full`.
- `SPGC_NoBONG`: catastrophic forgetting on `p3_overall_acc` and NLL explosion after consolidation.
- `SPGC_NoReplay`: modestly lower `p3_overall_acc` / worse `p3_overall_nll` vs `SPGC_Full`.

---

## 2. Suite 2: Real Architecture Continual Integration (`verify_spgc_real_dmoe.py`)

### 2.1 Exact Command & Splits
- **Architecture**: `RealSelfStabilizingDMoE` (`d_model=64, num_heads=4, d_ff=256, initial_layers=1, initial_experts=1, chunk_size=1, causal_attn=True, use_rope=True, local_window=2`) on `Stage 3B-Mem` continual Markov sequence transitions.
- **DEV Seeds (locked `real_dmoe_spgc_dev.json`)**: `DEV_SEEDS = [100..104]` (`n = 5`)
- **Confirmatory TEST Seeds**: `TEST_SEEDS = [0..19]` (`n = 20`)
- **Exact Confirmatory CLI Command**:
  ```bash
  python -u experiments/verify_spgc_real_dmoe.py --mode test --out_name real_dmoe_spgc_test.json
  ```
- **Output Path**: `results/belief_training_v3/real_dmoe_spgc_test.json`

### 2.2 Frozen Hyperparameters & Arms
- **Arms**: `RealDMoE_AdamW_Base`, `RealDMoE_Stage3_Ref`, `RealDMoE_SPGC`
- **Schedule**: `init_steps = 2200`, `burst_steps = 5` (`noisy_single`), `rev_steps = 800` (`rev_src_{s%3}`), `bs = 64`, `lr = 1.2e-2`, `wd = 1.5`
- **SPGC Params**: `trunk_muon (lr=0.012, lr_consolidate=0.008, prec_scale=2.0)`, `trunk_vec (lr=0.012, lr_consolidate=0.006, prec_scale=2.0)`, `fast_muon/fast_vec (lr=0.012, lr_consolidate=0.012, prec_scale=0.15)`, `bong_gamma=0.996, bong_damping=0.02, bong_init_prec=0.02, bong_min_prec=0.005`, `LatentQuarantineGate(max_bytes=4096, n_bits=7, conf_established=0.45, tau_skip=0.10, min_flip_count=5, flip_ratio=0.30, min_sources=2, rev_grace_steps=60, store_samples=False)`.

### 2.3 DEV Anchor Values (`real_dmoe_spgc_dev.json`, `n = 5`)
- `RealDMoE_AdamW_Base`: `init_ho=0.854 burst_s=0.076 flips=0.00 rev_s=1.000 rev_e=1.000 rev_ho=0.267 nll=7.067 fl=1.33e+12 bp=3005 t=27.73s B=0`
- `RealDMoE_Stage3_Ref`: `init_ho=0.854 burst_s=1.000 flips=0.00 rev_s=1.000 rev_e=1.000 rev_ho=0.874 nll=0.650 fl=1.33e+12 bp=3000 t=31.55s B=2240`
- `RealDMoE_SPGC`: `init_ho=0.872 burst_s=1.000 flips=0.00 rev_s=1.000 rev_e=0.998 rev_ho=0.901 nll=0.325 fl=1.57e+12 bp=2751 t=28.98s B=4084`

### 2.4 Pre-Registered Suite 2 Evaluation Criteria (`R1..R4`)
1. **`R1_burst_immunity_without_surface_bigrams`**:
   - `RealDMoE_SPGC` achieves `burst_seen >= 0.90` and `burst_flips <= 0.10` without surface bigram keys or inference logit mixing, outperforming `RealDMoE_AdamW_Base`.
2. **`R2_revision_generalization_and_calibration`**:
   - `RealDMoE_SPGC` achieves `rev_heldout >= RealDMoE_Stage3_Ref.rev_heldout`, `rev_nll <= RealDMoE_Stage3_Ref.rev_nll`, `rev_seen >= 0.98`, and `rev_exc >= 0.98`.
3. **`R3_budget_honesty`**:
   - `max_seed(RealDMoE_SPGC.storage_bytes) <= 4096`.
4. **`R4_honest_flop_tradeoff_disclosure`**:
   - Pre-registered expectation: on random 15-transition Markov chains (`Stage 3B-Mem`), `RealDMoE_SPGC` reduces `bp_calls` and `wall_clock_sec` vs `RealDMoE_Stage3_Ref`, **but incurs higher `flops_total` (~1.18x DEV ratio)** due to `moe` Newton-Schulz orthogonalization steps and slower `ema_loss < 0.05` crossing under `wd=1.5`. This trade-off will be reported directly with zero post-hoc adjustment.

---

## 3. Stopping & Failure Policy
- If any primary falsifier (`V1..V6`) or Suite 2 criterion (`R1..R3`) fails on `TEST_SEEDS = [0..19]`, report the failure prominently alongside any positive findings; do NOT edit parameters or re-run under the same experiment id.
- Post-run mandatory validation: `results_meta.validate_results_file(Path("results/belief_training_v3/results_v1.json"), min_seeds=20)`, `pytest` across unit/regression suites, and `integrity_lint.py`.

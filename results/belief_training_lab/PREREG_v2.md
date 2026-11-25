# Pre-Registration v2 (Frozen Pre-TEST): Three Brain-Inspired Belief & Meaning Training Algorithms (`belief_training_lab_v2`)

- **Experiment ID**: `belief_training_lab_v2`
- **Status**: **FROZEN BEFORE ANY `TEST_SEEDS = [0..19]` EXECUTION**
- **Prior Draft**: [`PREREG_v1.md`](file:///d:/Dynamic_MOE_Transformer/results/belief_training_lab/PREREG_v1.md)
- **DEV Tuning Evidence**: [`tuning_log_v1.json`](file:///d:/Dynamic_MOE_Transformer/results/belief_training_lab/tuning_log_v1.json) (Entries #1–#4, `DEV_SEEDS = [100, 101, 102, 103, 104]` ONLY; `TEST_SEEDS = [0..19]` strictly untouched)
- **Implementation**: [`dmoe/belief_training.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/belief_training.py), [`tests/test_belief_training.py`](file:///d:/Dynamic_MOE_Transformer/tests/test_belief_training.py), [`experiments/run_belief_training_lab.py`](file:///d:/Dynamic_MOE_Transformer/experiments/run_belief_training_lab.py)
- **Frozen Output Target**: `results/belief_training_lab/results_v2.json`

---

## 1. Documented DEV Revisions (`PREREG_v1` -> `PREREG_v2`)

All changes below were motivated and verified strictly on `DEV_SEEDS = [100..104]`:

1. **Gate G0 Resolution in Benchmark Bed (`tuning_log_v1.json` Entries #1 -> #3)**:
   - Entry #1 (`PREREG_v1` initial 4-operator bed, `steps_p1=1200, bs=32`) failed Gate G0 (`B4_JointOracle p1_val_heldout = 0.065`, `p2_split_both_min = 0.238`).
   - Updated expression family to `inv_phi[(phi[a] + 3*phi[b] + s) % 16]` (`s in [0, 5, 11, 14]`), latent context rule to `ctx_regime = (phi[c1] >= 8) ^ (phi[c2] >= 8)` (retaining exact 50/50 marginal balance per token and `B0_SurfaceDict` failure under 16 KB cap), and added uniform `_build_ctx_feat` (normalized `tok_emb[:, 3:5]` skip alongside `h_n[:, 4]`) plus dormant context/adapter decay separation (`wd_post = 0.01` on `ctx_proj`/`fast_adapter`) across **all** shared Transformer backbone arms (`B1..B4`, `Idea1`, `Idea2`, `Idea3_Backbone_BONG`, `Triad`) after DEV probing showed Phase-1 grokking washed out un-supervised context positions `3..4` (`h_n[3:5]` probe `0.548` vs `tok_emb[3:5]` probe `0.998`).
2. **Stabilized Concept-Keyed Wake–Chunk–Sleep (`Idea1`, `Triad`, `ConceptBeliefStore`)**:
   - Added empirical `purity` EMA gating on `primary` and `splits` entries (`purity >= 0.72`) and minimum centroid cosine separation (`1 - cos(c0, c1) >= 0.25`, `>= 4` examples/pole) so noisy un-clustered Voronoi atoms in `Idea1` (no `L_equiv`) cannot override neural predictions with random-noise splits.
   - Added consecutive single-source `severe_burst` quarantine (`contra >= 0.55 * hits` with `< 2` active contradicting sources or consecutive same source) + neural catch-up flagging (`tgt == cur_y` and `p_tgt < 0.55` when non-burst) + self-belief-consistent reservoir pruning (`flips_here > 0`) + cached-trunk chunking with matched-atom context pairing (`ok_pair` on `Q_ATTR`) and balanced 50/50 sleep replay + `NullSpaceGuard` tracking paused during non-training wake/eval.
3. **$\mu$PC + Equilibration & BONG Settings (`Idea3`)**:
   - Adjusted non-normalized `MuPCProspectiveNetwork` optimizer parameters on DEV (`wd_mupc = 0.02`, `lr_mupc_bong = 2e-3, init_prec = 2e-3, damping = 5e-3, min_prec = 1e-5, grad_clip = 0.5, prior_wd = 0.005`) and decoupled `Idea3_Backbone_BONG` Phase-1 prior shrinkage (`wd_bong_p1 = 0.10`, exempted on `ctx_proj`/`fast_adapter` identically to AdamW).
4. **Null Control Fix (`Null_ShuffledTargets`)**:
   - Changed target permutation from within-minibatch `randperm` (which preserved 4-class skew in focused Phase-2/Phase-3 minibatches, artificially scoring `0.21–0.31`) to uniform `torch.randint(0, 16)` across all 16 classes.

**CRITICAL**: None of the pre-registered boolean falsifiers `F1..F8` were weakened or altered from `PREREG_v1.md`.

---

## 2. Frozen Hyperparameters (`LabConfig`, Verified on `DEV_SEEDS`)

- `batch_size = 128`
- `steps_p1 = 3600`, `steps_pb = 10`, `steps_p2 = 800`, `steps_p3 = 600`, `confirm_frac = 0.25`
- `lr_dense = 6e-3`, `lr_dense_post = 4e-3`, `wd_p1 = 0.22`, `wd_bong_p1 = 0.10`, `wd_post = 0.01`
- `lr_idea2 = 6e-3`, `lr_bong_bb = 1.2e-2`, `lr_mupc_bong = 2e-3`, `lr_mupc_adam = 3e-3`, `wd_mupc = 0.02`
- `lr_chunk = 1e-2`, `lr_bong_chunk = 3e-2`, `elastic_pull = 0.001`
- `lambda_equiv = 0.35` (ramp `0.25 -> 1.0` over `1600` steps), `lambda_contra = 0.20`, `lambda_sigreg = 0.03`
- `chunk_size = 128`, `chunk_steps = 6`, `sleep_period = 100`, `sleep_steps = 4`
- `rho_thresh = 0.15`, `tau_pc_skip = 0.10`, `max_store_bytes = 16384`, `num_atoms = 64`
- `k_train = 10`, `eta_z = 0.10`, `beta_out = 0.10`

**CLI Command to Execute Once on `TEST_SEEDS = [0..19]`:**
```powershell
python -u experiments/run_belief_training_lab.py --mode test --out_name results_v2.json
```

---

## 3. DEV Gates Summary (`tuning_log_v1.json` Entries #3 & #4, `n = 5` DEV Seeds)

- **Gate G0 (`B4_JointOracle`) — PASS**:
  - `p1_val_heldout = 0.934` (`>= 0.70`), `p2_split_both_min = 1.000` (`>= 0.75`), `p3_rev_equiv_transfer = 1.000`, `p3_overall_acc = 0.970` (`>= 0.75`), `p3_overall_nll = 0.116`.
- **Gate G1 (Causality & 0-Backprop) — PASS**:
  - `z_expr_future_diff = 0.0`, `z_ctx_future_diff = 0.0` (`<= 1e-5`); `Idea3_MuPC_BONG_Pure` `backprop_calls = 0`.
- **Gate G2 (`B0_SurfaceDict`) — PASS**:
  - `p1_val_heldout = 0.056` (`<= 0.15`), `p3_rev_equiv_transfer = 0.095` (`<= 0.15`), `p3_overall_acc = 0.065`.
- **DEV Preliminary Falsifiers (5 DEV seeds — NOT final verdict; frozen TEST pending)**:
  - `F1_idea1_context_split`: `False` (`B1_Dense_SGD = 0.982` vs `Idea1_ConceptWCS = 0.741`)
  - `F2_idea1_burst_and_compute`: `True` (`pb_burst_flips = 0.0` across all 5 seeds, `pb_retention = 0.869`, `flops_post_p1 = 3.15e+11` vs `0.60 * 6.37e+11 = 3.82e+11`, `all = 0.718`)
  - `F3_idea2_equiv_transfer_and_x47_control`: `False` (`p3_eq = 0.719`, `+0.295` vs `B1`, `+0.120` vs `Idea2_ShuffledEquiv` on `n = 5`)
  - `F4_idea3a_mupc_bong_viable`: `False` (`Idea3_MuPC_BONG_Pure all = 0.145` vs `Idea3_MuPC_Adam all = 0.234`)
  - `F5_idea3b_backbone_bong`: `False` (`Idea3_Backbone_BONG all = 0.784` vs `B1 = 0.615`, `p3_eq = 0.895` vs `0.424`, `nll = 0.645`, `pb_ret = 0.168` vs `0.082` `+0.086 < +0.10`)
  - `F6_triad_synergy_and_beats_replay`: `True` (`Idea_Triad_All3 all = 0.804` vs `B2 = 0.597` `+0.207`, `CI95_low = +0.020`, `p2_min = 0.857`, `p3_eq = 0.821`, `flops_post_p1 = 2.89e+11`)
  - `F7_calibration_honesty`: `True` (`Idea_Triad_All3 nll = 1.158 <= B1 1.426` and `<= 1.2`)
  - `F8_null_controls_fail`: `True` (`Null_Untrained = 0.061`, `Null_ShuffledTargets = 0.061`, `Null_BlindZeroInput = 0.059`)

---

## 4. Frozen Falsifiers (`F1..F8`, Evaluated on `TEST_SEEDS = [0..19]` via Script)

Identical verbatim thresholds and 95% bootstrap paired CIs (`n_boot = 2000`) to `PREREG_v1.md` §6:
- **F1**: `Idea1_ConceptWCS.p2_split_both_min >= 0.75`, beats `Idea1_NoCtxSplit` by `>= +0.15` (`CI_low > 0`), and beats `B1_Dense_SGD` by `>= +0.20` (`CI_low > 0`).
- **F2**: `Idea1_ConceptWCS.pb_retention >= 0.85`, `pb_burst_flips <= 0.1`, `flops_post_p1 <= 0.60 * B2_Dense_Replay.flops_post_p1`, and `p3_overall_acc >= B2_Dense_Replay.p3_overall_acc - 0.03`.
- **F3**: `Idea2_EquivContra.p3_rev_equiv_transfer >= 0.70`, exceeds `B1_Dense_SGD` by `>= +0.15` (`CI_low > 0`), and exceeds `Idea2_ShuffledEquiv` by `>= +0.15` (`CI_low > 0`).
- **F4**: `Idea3_MuPC_BONG_Pure.backprop_calls == 0`, `p1_val_heldout >= 0.65`, `p3_overall_acc >= 0.55`, and beats `Idea3_MuPC_Adam` (`CI_low > 0` or `mean_diff >= 0.05`).
- **F5**: `Idea3_Backbone_BONG.p3_retention_all >= B1_Dense_SGD.p3_retention_all + 0.10` (`CI_low > 0`) and `pb_retention >= B1_Dense_SGD.pb_retention + 0.10`.
- **F6**: `Idea_Triad_All3.p3_overall_acc >= max(B0, B1, B2) + 0.05` (`CI_low > 0`), `p3_rev_equiv_transfer >= 0.75`, `p2_split_both_min >= 0.75`, and `flops_post_p1 < B4_JointOracle.flops_total`.
- **F7**: `Idea_Triad_All3.p3_overall_nll <= B1_Dense_SGD.p3_overall_nll` and `<= 1.2`.
- **F8**: All 3 null controls (`Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`) score `< 0.15` on `p1_val_heldout`, `p2_split_both_min`, `p3_rev_equiv_transfer`, and `p3_overall_acc`.

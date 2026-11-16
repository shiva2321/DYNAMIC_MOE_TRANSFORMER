# Pre-Registration: Stage 3 — Header-Free Continual Learning, Belief Revision & Predictive-Coding Memory Rectification (`PREREG_v1`)

- **Experiment ID**: `stage3_continual_learning_v1`
- **Date**: 2026-11-16
- **Status**: **FROZEN PRIOR TO ANY 20-SEED TEST EVALUATION**
- **Run Script**: `experiments/run_stage3_continual_learning.py`
- **Frozen Test Command**:
  `python -u experiments/run_stage3_continual_learning.py --mode test --workers 5`
- **Workspace Disclosure**: `.integrity/` kit and `docs/CLAIMS.md` are absent on disk in `D:\Dynamic_MOE_Transformer` (`integrity=False`, `claims=False`).

---

## 1. Hypotheses & Claims Under Test

1. **Sub-Suite 3A (Header-Free Shared-Vocab Continual Learning & Boundary-Free Lifecycle)**:
   On strictly non-separable shared-alphabet streams (`ALPHABET_START..ALPHABET_START+ALPHABET_SIZE-1`, tokens `[32..47]`) without explicit domain headers or task boundaries, causal transition-prototype routing (`CausalPrototypeRouter`) + Gradient Projection Memory (`NullSpaceProjector`) + hysteresis-gated autonomous spawning/revival (`AutonomousLifecycleManager`) will:
   - Identify domains online (`route_acc`) and prevent catastrophic forgetting across `L0` (clean sequential), `L1 (rho=0.0)` and `L1 (rho=0.5)` (non-stationary mixed Markov streams), and `L2` (boundary-free `D0 -> 5-step burst -> D0 -> D1 -> D2 -> D0_revisit`).
   - Remain strictly immune to transient `5-step` noise bursts (`l2_burst_spawns == 0.0`), spawn experts for sustained novel domains (`l2_sustain_spawns >= 1.5`), and revive rather than duplicate when revisiting `D0` (`l2_revisit_spawns == 0.0`) with high pre-revisit and post-revisit retention (`l2_d0_pre_revisit >= 0.85`, `l2_d0_post_revisit >= 0.85`, `l2_avg >= 0.85`).
   - Significantly outperform parameter-matched `Dense_NoReplay`, `Dense_Replay128` (`<= 4096 B`), `StaticMoE_Matched`, `Cog_NoGPM`, and `Cog_UnigramRouter` (`Cog_OracleRouting` included as an upper-bound routing reference).

2. **Sub-Suite 3B-Mem (Bounded Belief Revision & Compositional Generalization)**:
   Within a strict `<= 4096 B` episodic storage cap (`CalibratedBayesianUpkeeper`), compositional relational inference + dual-timescale backbone stabilization + source-corroborated belief revision will:
   - Achieve high compositional held-out generalization (`init_heldout_acc >= 0.55`, `rev_heldout_acc >= 0.50`) and seen/exception accuracy (`>= 0.95`), outperforming an equal-byte exact tabular lookup baseline (`BudgetedDictionaryBaseline`, which cannot generalize compositionally).
   - Remain immune to `5-step` single-source adversarial bursts (`burst_flips == 0.0`, `burst_seen_acc >= 0.90`, beating `Ablation_NoUpkeeper` by `> 0.50`) while executing persistent revision when corroborated across multiple independent sources (`rev_seen_acc >= 0.95`, `rev_exc_acc >= 1.0` target).

3. **Sub-Suite 3B-IC (Zero-Shot In-Context Contradiction Immunity & Predictive-Coding Splice)**:
   Trained strictly on clean sequences (`new` and `confirm` facts only, zero contradiction training exposure), the Stage 3 hierarchical-memory transformer (`RealSelfStabilizingDMoE` with causal predictive-coding write-gate Pass-1 / Pass-2 splice) will:
   - Achieve zero-shot contradiction immunity (`ic_contra_acc >= 0.85`), outperforming `Ablation_NoPCRectification` by `> 0.40`, alongside `ic_new_acc >= 0.90`, `ic_confirm_acc >= 0.90`, and `ic_auroc_cf >= 0.90`.
   - Maintain strict autoregressive causality (`max_diff <= 1e-5` across all `k` future perturbations for both model and router).

4. **Sub-Suite 3C (Unified Real-Stream UTF-8 Novelty Calibration)**:
   Using the exact Stage 3 novelty/lifecycle pipeline on real UTF-8 repository code vs prose (`Regime A`) and disjoint 70/15/15 open Markdown prose splits (`Regime B`), the unified calibration will keep false positive rate `<= 0.15` while maintaining high code and UTF-8 nuance detection (`>= 0.90`) simultaneously with zero false `L2` burst/revisit spawns.

---

## 2. Disjoint Seed Split Protocol

- **DEV Split (Tuning & Sanity Checks ONLY)**: `DEV_SEEDS = [100, 101, 102, 103, 104]` (`n = 5`, logged to `results/stage3_continual_learning/tuning_log_v1.json`, 29 entries). Null controls in DEV tune ran on `seed = 100` (`n = 1`).
- **HELD-OUT TEST Split (Confirmation ONLY)**: `TEST_SEEDS = [0, 1, 2, ..., 19]` (`n = 20`), completely disjoint from `DEV_SEEDS`, evaluated **once** after this file is frozen. All models, baselines, ablations, and all 3 null controls (`Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`) run on all 20 `TEST_SEEDS`.

---

## 3. Frozen Architecture, Parameter Counts, Storage Cap & CLI Defaults

Exact CLI defaults in `experiments/run_stage3_continual_learning.py`:
- `--mode test --workers 5`
- **3A**: `--steps_3a 100`, `--lr_cog_3a 4e-3`, `--lr_dense_3a 3e-3`, `--lr_smoe_3a 3e-3`
  - Model neural parameters: `Cog_Stage3_Full` = `164,406` vs `Dense_NoReplay` / `Dense_Replay128` = `165,376` vs `StaticMoE_Matched` = `165,862` (all baselines `>= Cog`).
  - Replay buffer in `Dense_Replay128` capped at `<= 4096 B`.
- **3B-Mem**: `--init_steps_3bm 3000`, `--rev_steps_3bm 1500`, `--lr_cog_3bm 1.2e-2`, `--lr_dense_3bm 1.2e-2`
  - Model neural parameters: `Cog_Stage3_Full` = `91,520` vs `Dense_NoUpk` = `93,584` (`>= Cog`), `BudgetedDictionaryBaseline` = `0` neural params.
  - Episodic memory budget (`CalibratedBayesianUpkeeper` and `BudgetedDictionaryBaseline`): `max_bytes <= 4096 B` (observed DEV `2240 B`).
- **3B-IC**: `--steps_3bic 1200`, `--lr_cog_3bic 3e-3`, `--lr_dense_3bic 2e-3`
  - Model neural parameters: `Cog_Stage3_Full` = `528,753` vs `Dense_Matched` = `534,208` (`>= Cog`).
- **3C**: `--steps_real_3c 80`

---

## 4. Summary of DEV Tuning (`DEV_SEEDS = [100..104]`, `tuning_log_v1.json`)

- **Strict Causality Gate (DEV)**: `CausalPrototypeRouter = 0.0`, `Cog_Stage3_3A = 0.0`, `Cog_Stage3_IC = 0.0`, `DenseRoPE = 0.0`, `StaticMoE = 0.0` (all `<= 1e-5`).

### Sub-Suite 3A (DEV `n=5`; Nulls `n=1` at `seed=100`)
| Model | LR | Params | `l0_avg` | `l1_rho0_avg` | `l1_rho5_avg` | `l2_d0_pre_revisit` | `l2_d0_post_revisit` | `l2_avg` | `burst_sp` / `sust_sp` / `rev_sp` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Cog_Stage3_Full` (frozen) | 0.004 | 164,406 | 0.9895 | 0.9970 | 0.8640 | 0.9948 | 0.9783 | 0.9149 | 0.0 / 2.0 / 0.0 |
| `Cog_NoGPM` | 0.004 | 164,406 | 0.5437 | 0.5470 | 0.4058 | 0.2035 | 0.6960 | 0.4981 | 0.0 / 2.0 / 0.8 |
| `Cog_UnigramRouter` | 0.004 | 164,406 | 0.3840 | 0.3760 | 0.3824 | 0.4242 | 0.5281 | 0.4066 | 0.0 / 2.0 / 1.0 |
| `Dense_NoReplay` (frozen) | 0.003 | 165,376 | 0.4078 | 0.3953 | 0.4002 | 0.0996 | 0.9973 | 0.4404 | 0.0 / 0.0 / 0.0 |
| `Dense_NoReplay` (alt LR) | 0.005 | 165,376 | 0.4008 | 0.3952 | 0.3977 | 0.0810 | 0.9846 | 0.4268 | 0.0 / 0.0 / 0.0 |
| `Dense_Replay128` (frozen) | 0.003 | 165,376 | 0.4178 | 0.3956 | 0.4121 | 0.1085 | 0.9979 | 0.5072 | 0.0 / 0.0 / 0.0 |
| `Dense_Replay128` (alt LR) | 0.005 | 165,376 | 0.4130 | 0.3995 | 0.4004 | 0.0965 | 0.9971 | 0.4949 | 0.0 / 0.0 / 0.0 |
| `StaticMoE_Matched` (frozen) | 0.003 | 165,862 | 0.4151 | 0.4009 | 0.4036 | 0.1019 | 0.9871 | 0.4608 | 0.0 / 0.0 / 0.0 |
| `StaticMoE_Matched` (alt LR) | 0.005 | 165,862 | 0.4007 | 0.3935 | 0.4023 | 0.1008 | 0.9713 | 0.4173 | 0.0 / 0.0 / 0.0 |
| `Null_Untrained` (s=100) | 0.000 | 101,878 | 0.0003 | 0.0007 | 0.0007 | 0.0000 | 0.0000 | 0.0003 | 0.0 / 0.0 / 0.0 |
| `Null_ShuffledTargets` (s=100) | 0.004 | 101,878 | 0.0563 | 0.0566 | 0.0598 | 0.0125 | 0.0917 | 0.0712 | 0.0 / 0.0 / 0.0 |
| `Null_BlindZeroInput` (s=100) | 0.004 | 164,406 | 0.0712 | 0.0577 | 0.0625 | 0.0646 | 0.0635 | 0.0632 | 0.0 / 2.0 / 0.0 |

*(Note: `Cog_OracleRouting` was not logged in DEV `tuning_log_v1.json` and runs across `TEST_SEEDS` in `--mode test` as an upper-bound routing comparison; falsifiers do not depend on `Cog_OracleRouting`.)*

### Sub-Suite 3B-Mem (DEV `n=5`; Nulls `n=1` at `seed=100`)
| Model | LR | Params | Bytes | `init_seen` / `heldout` (`nll`) | `burst_seen` (`fl`) | `rev_seen` / `exc` / `heldout` (`nll`) |
|---|---:|---:|---:|---|---|---|
| `Cog_Stage3_Full` (frozen) | 0.012 | 91,520 | 2240 | 1.0000 / 0.8742 (0.5270) | 1.0000 (0.0) | 1.0000 / 1.0000 / 0.7898 (0.8856) |
| `Ablation_NoUpkeeper` | 0.012 | 91,520 | 0 | 1.0000 / 0.8742 (0.5270) | 0.0629 (0.0) | 1.0000 / 1.0000 / 0.5738 (2.6791) |
| `Dense_NoUpk` (frozen) | 0.012 | 93,584 | 0 | 1.0000 / 0.8555 (0.4693) | 0.0676 (0.0) | 1.0000 / 1.0000 / 0.5043 (2.9682) |
| `Dense_NoUpk` (alt LR) | 0.010 | 93,584 | 0 | 1.0000 / 0.7855 (1.4231) | 0.0563 (0.0) | 1.0000 / 1.0000 / 0.4855 (3.1687) |
| `BudgetedDictionaryBaseline` | 0.000 | 0 | 2240 | 0.7750 / 0.0594 (2.7728) | 0.7750 (0.0) | 0.7461 / 0.6762 / 0.0613 (2.7728) |
| `Null_Untrained` (s=100) | 0.000 | 91,520 | 0 | 0.0000 / 0.0000 (5.7360) | 0.0000 (0.0) | 0.0000 / 0.0000 / 0.0000 (5.6718) |
| `Null_ShuffledTargets` (s=100) | 0.012 | 91,520 | 4060 | 0.1035 / 0.0078 (2.9008) | 0.0918 (6.0) | 0.0625 / 0.1680 / 0.0176 (2.8754) |
| `Null_BlindZeroInput` (s=100) | 0.012 | 91,520 | 2240 | 0.0605 / 0.0703 (9.9098) | 0.0605 (0.0) | 0.0703 / 0.0000 / 0.0215 (11.4152) |

### Sub-Suite 3B-IC & Baseline Sanity Check (DEV `n=5`; Nulls `n=1` at `seed=100`)
| Model | LR | Params | `ic_new_acc` | `ic_confirm_acc` | `ic_contra_acc` | `picked_contra` | `ic_auroc_cf` | `caus_max` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Cog_Stage3_Full` (frozen) | 0.003 | 528,753 | 0.9977 | 1.0000 | 0.9773 | 0.0148 | 0.9984 | 0.0 |
| `Ablation_NoPCRectification` | 0.003 | 528,753 | 0.9977 | 1.0000 | 0.1727 | 0.8250 | 0.9999 | 0.0 |
| `Dense_Matched` (frozen) | 0.002 | 534,208 | 0.1148 | 1.0000 | 0.0523 | 0.5906 | 0.7039 | 0.0 |
| `Dense_Matched` (alt LR) | 0.003 | 534,208 | 0.1016 | 0.9992 | 0.0563 | 0.6234 | 0.6789 | 0.0 |
| `Dense_Matched` (alt LR) | 0.005 | 534,208 | 0.0813 | 1.0000 | 0.0422 | 0.6805 | 0.4906 | 0.0 |
| `Null_Untrained` (s=100) | 0.000 | 528,753 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0 |
| `Null_ShuffledTargets` (s=100) | 0.003 | 528,753 | 0.0469 | 0.0625 | 0.0508 | 0.0430 | 0.5000 | 0.0 |
| `Null_BlindZeroInput` (s=100) | 0.003 | 528,753 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0 |

*(Baseline sanity check note: `Dense_Matched` was tuned across `{2e-3, 3e-3, 5e-3}` on DEV and also evaluated in prior DEV diagnostics across `{5e-4..5e-3}` / `2400` steps; pure attention-only RoPE transformers without recurrent STM converge to the recency copy shortcut on `confirm` and fail `new` extraction, whereas `Ablation_NoPCRectification` [which has STM without PC rectification] isolates `ic_contra_acc` cleanly.)*

### Sub-Suite 3C (DEV `n=5`)
- `Unified_Stage3C`:
  - `Regime A` (code vs prose): `real_a_fpr = 0.0`, `real_a_code_tpr = 1.0`, `real_a_twin_tpr = 1.0`, `real_a_nuance_tpr = 1.0`, `real_a_old_nuance_tpr = 0.0`, `real_a_old_twin_tpr = 0.0`
  - `Regime B` (disjoint open Markdown): `real_b_fpr = 0.0`, `real_b_code_tpr = 1.0`, `real_b_twin_tpr = 1.0`, `real_b_nuance_tpr = 1.0`, `real_b_nuance_ascii_tpr = 1.0`, `real_b_old_nuance_tpr = 1.0`
  - `L2 Unified Lifecycle`: `l2_burst_spawns = 0.0`, `l2_sustain_spawns = 2.0`, `l2_revisit_spawns = 0.0`, `l2_d0_pre_revisit = 0.9948`, `l2_d0_post_revisit = 0.9783`

---

## 5. Frozen Pre-Registered Falsifiers (`F1`–`F7`)

All 7 falsifiers below are evaluated automatically on `TEST_SEEDS = [0..19]` (`n = 20`) and recorded verbatim in `results/stage3_continual_learning/results_v1.json`:

1. **`F1_strict_causality`**:
   - Every entry in `causality_gate` `<= 1e-5` AND `3b_ic["Cog_Stage3_Full"]["ic_causality_diff"]["max"] <= 1e-5`.
2. **`F2_3a_continual_retention_and_ablations`**:
   - `Cog_Stage3_Full["l1_rho0_avg"]["mean"] >= 0.90`
   - `Cog_Stage3_Full["l1_rho5_avg"]["mean"] >= 0.80`
   - `Cog_Stage3_Full["l1_rho0_avg"]["mean"] > Dense_NoReplay["l1_rho0_avg"]["mean"] + 0.35`
   - `Cog_Stage3_Full["l1_rho0_avg"]["mean"] > Dense_Replay128["l1_rho0_avg"]["mean"] + 0.35`
   - `Cog_Stage3_Full["l1_rho0_avg"]["mean"] > StaticMoE_Matched["l1_rho0_avg"]["mean"] + 0.35`
   - `Cog_Stage3_Full["l1_rho5_avg"]["mean"] > Dense_Replay128["l1_rho5_avg"]["mean"] + 0.25`
   - `Cog_Stage3_Full["l1_rho0_avg"]["mean"] > Cog_NoGPM["l1_rho0_avg"]["mean"] + 0.15`
   - `Cog_Stage3_Full["l1_rho0_avg"]["mean"] > Cog_UnigramRouter["l1_rho0_avg"]["mean"] + 0.35`
3. **`F3_3a_boundary_free_l2_lifecycle`**:
   - `Cog_Stage3_Full["l2_burst_spawns"]["mean"] == 0.0`
   - `Cog_Stage3_Full["l2_sustain_spawns"]["mean"] >= 1.5`
   - `Cog_Stage3_Full["l2_revisit_spawns"]["mean"] == 0.0`
   - `Cog_Stage3_Full["l2_d0_pre_revisit"]["mean"] >= 0.85`
   - `Cog_Stage3_Full["l2_d0_post_revisit"]["mean"] >= 0.85`
   - `Cog_Stage3_Full["l2_avg"]["mean"] >= 0.85`
4. **`F4_3bm_budget_burst_and_generalization`**:
   - `Cog_Stage3_Full["storage_bytes"]["max"] <= 4096.0`
   - `Cog_Stage3_Full["init_seen_acc"]["mean"] >= 0.95`
   - `Cog_Stage3_Full["init_heldout_acc"]["mean"] >= 0.55`
   - `Cog_Stage3_Full["init_heldout_acc"]["mean"] > BudgetedDictionaryBaseline["init_heldout_acc"]["mean"] + 0.40`
   - `Cog_Stage3_Full["burst_flips"]["mean"] == 0.0`
   - `Cog_Stage3_Full["burst_seen_acc"]["mean"] >= 0.90`
   - `Cog_Stage3_Full["burst_seen_acc"]["mean"] > Ablation_NoUpkeeper["burst_seen_acc"]["mean"] + 0.50`
   - `Cog_Stage3_Full["rev_seen_acc"]["mean"] >= 0.95`
   - `Cog_Stage3_Full["rev_heldout_acc"]["mean"] >= 0.50`
   - `Cog_Stage3_Full["rev_heldout_acc"]["mean"] > BudgetedDictionaryBaseline["rev_heldout_acc"]["mean"] + 0.35`
5. **`F5_3bic_zero_shot_contradiction_immunity`**:
   - `Cog_Stage3_Full["ic_new_acc"]["mean"] >= 0.90`
   - `Cog_Stage3_Full["ic_confirm_acc"]["mean"] >= 0.90`
   - `Cog_Stage3_Full["ic_contra_acc"]["mean"] >= 0.85`
   - `Cog_Stage3_Full["ic_contra_acc"]["mean"] > Ablation_NoPCRectification["ic_contra_acc"]["mean"] + 0.40`
   - `Cog_Stage3_Full["ic_auroc_cf"]["mean"] >= 0.90`
6. **`F6_3c_real_stream_calibration`**:
   - `real_a_fpr <= 0.15`, `real_a_code_tpr >= 0.90`, `real_a_nuance_tpr >= 0.90`
   - `real_b_fpr <= 0.15`, `real_b_code_tpr >= 0.90`, `real_b_nuance_tpr >= 0.90`
   - `l2_burst_spawns == 0.0`, `l2_revisit_spawns == 0.0`
7. **`F7_null_controls_fail`**:
   - For all `nm in {Null_Untrained, Null_ShuffledTargets, Null_BlindZeroInput}` across `TEST_SEEDS = [0..19]`:
     - `3a`: `l1_rho0_avg["mean"] < 0.18`
     - `3b_mem`: `rev_seen_acc["mean"] < 0.15` and `rev_heldout_acc["mean"] < 0.15`
     - `3b_ic`: `ic_new_acc["mean"] < 0.18` and `ic_contra_acc["mean"] < 0.18`

## 6. Pre-Registered Paired Comparisons

Exact paired statistics (`mean_diff`, `std_diff`, `t_stat`, `wins`, `n=20`) recorded in `paired_comparisons`:
- `3a_l1_rho0_full_vs_dense_noreplay`, `3a_l1_rho0_full_vs_dense_replay128`, `3a_l1_rho0_full_vs_static_moe`, `3a_l1_rho0_full_vs_nogpm`, `3a_l1_rho0_full_vs_unigram`
- `3a_l1_rho5_full_vs_dense_replay128`, `3a_l1_rho5_full_vs_unigram`
- `3bm_init_heldout_full_vs_dict`, `3bm_burst_seen_full_vs_noupk`, `3bm_rev_heldout_full_vs_dict`
- `3bic_contra_full_vs_nopc`, `3bic_contra_full_vs_dense`
- `3c_real_a_nuance_new_vs_old`

## 7. Non-Negotiable Execution Rules

- Do **not** edit `dmoe/` or `experiments/run_stage3_continual_learning.py` after freezing this document.
- Run `--mode test --workers 5` **once** on `TEST_SEEDS = [0..19]`.
- If any falsifier fails or any baseline/null surprises, report all negative results with full prominence without post-hoc threshold edits or re-tuning on test seeds.

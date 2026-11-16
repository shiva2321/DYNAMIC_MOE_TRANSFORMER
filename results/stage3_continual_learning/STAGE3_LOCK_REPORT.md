# Stage 3 Lock Report: Header-Free Continual Learning, Belief Revision & Predictive-Coding Memory Rectification

- **Experiment ID**: `stage3_continual_learning_v1`
- **Status**: **LOCKED** (`all_passed = True`, all 7 pre-registered falsifiers `F1`–`F7` passed on disjoint held-out `TEST_SEEDS = [0..19]`)
- **Pre-Registration**: [PREREG_v1.md](file:///D:/Dynamic_MOE_Transformer/results/stage3_continual_learning/PREREG_v1.md) (frozen `2026-11-16T10:13:42Z` before test execution)
- **Raw Held-Out Results**: [results_v1.json](file:///D:/Dynamic_MOE_Transformer/results/stage3_continual_learning/results_v1.json)
- **DEV Tuning Log (`DEV_SEEDS = [100..104]`)**: [tuning_log_v1.json](file:///D:/Dynamic_MOE_Transformer/results/stage3_continual_learning/tuning_log_v1.json) (`29` entries)
- **Execution Window**: `2026-11-16T10:14:03Z` to `2026-11-16T12:41:51Z` (`NVIDIA GeForce RTX 3060`, `PyTorch 2.6.0+cu124`, `Python 3.12.10`, `git_commit 6344b0024615a8fe38ecea969bf93c709ee4b86e`, `git_dirty=True`, sum of reported suite wall-clock durations = `8868.0s`)
- **Governance Disclosure**: `.integrity/` kit and `docs/CLAIMS.md` are **absent** on disk in `D:\Dynamic_MOE_Transformer`.

---

## 1. Claims & Claim Level

- **Overall Claim Level**: **`GENERALIZES`** *(within registered benchmark suites on 20 disjoint held-out seeds, unseen compositional holdouts in `3B-Mem`, unseen contradiction type zero-shot in `3B-IC`, and disjoint 70/15/15 UTF-8 Markdown/code streams in `3C`, alongside passing all 3 Null Controls and all matched baselines/ablations).*

1. **Header-Free Shared-Vocab Continual Learning (`3A`)**: Without task headers or boundaries on shared-alphabet streams (`[32..47]`), `RealSelfStabilizingDMoE` with `CausalPrototypeRouter`, `NullSpaceProjector` (GPM), and `AutonomousLifecycleManager` routes regimes online, retains prior tasks under clean (`L0`, `L1 rho=0.0`) and 50% mid-stream mixed (`L1 rho=0.5`) shifts, avoids false spawns on 5-step bursts (`L2`), and revives existing experts on revisit (`l2_revisit_spawns = 0.0`).
2. **Budgeted Belief Revision & Compositional Generalization (`3B-Mem`)**: Within `2240 B` (`<= 4096 B` cap), `CalibratedBayesianUpkeeper` + dual-timescale backbone stabilization rejects 5-step single-source adversarial bursts (`0.00` flips) while executing persistent multi-source revisions (`1.0000` seen/exception accuracy) and retaining compositional held-out generalization (`0.8093` post-revision vs `0.0591` for equal-byte `BudgetedDictionaryBaseline`).
3. **Zero-Shot In-Context Contradiction Immunity (`3B-IC`)**: Trained strictly on clean (`new`, `confirm`) sequences, the causal predictive-coding write-gate Pass-1 / Pass-2 splice achieves `0.9881` zero-shot contradiction immunity (`ic_contra_acc`), `0.9985` conflict/clean AUROC (`ic_auroc_cf`), and exact `0.0` autoregressive future-leakage (`<= 1e-5`).
4. **Unified Real-Stream UTF-8 Calibration (`3C`)**: Using the Stage 3 lifecycle/novelty configuration on real repository UTF-8 code vs prose (`Regime A`) and disjoint open Markdown prose (`Regime B`), detection reaches `1.0000` TPR with `0.0000` (`Regime A`) and `0.0200` (`Regime B`) FPR while keeping `L2` burst/revisit spawns at `0.0000`.

---

## 2. Headline Metrics (`n = 20` Held-Out `TEST_SEEDS = [0..19]`)

All statistics reported as `mean +/- 95% CI` (`std`, `[min..max]`):

| Sub-Suite | Metric (`Cog_Stage3_Full` / `Unified_Stage3C`) | `mean +/- 95% CI` | `std` | `[min .. max]` |
|---|---|---:|---:|---|
| **3A** | `l0_avg` | `0.9867 +/- 0.0053` | `0.0120` | `[0.9667 .. 1.0000]` |
| **3A** | `l1_rho0_avg` | `0.9976 +/- 0.0021` | `0.0048` | `[0.9810 .. 1.0000]` |
| **3A** | `l1_rho0_route` | `0.9974 +/- 0.0023` | `0.0052` | `[0.9792 .. 1.0000]` |
| **3A** | `l1_rho5_avg` | `0.8439 +/- 0.0216` | `0.0493` | `[0.7387 .. 0.9210]` |
| **3A** | `l1_rho5_route` | `0.8349 +/- 0.0227` | `0.0518` | `[0.7292 .. 0.9167]` |
| **3A** | `l2_d0_pre_revisit` | `0.9950 +/- 0.0042` | `0.0096` | `[0.9698 .. 1.0000]` |
| **3A** | `l2_d0_post_revisit` | `0.9796 +/- 0.0096` | `0.0219` | `[0.9177 .. 1.0000]` |
| **3A** | `l2_avg` | `0.9097 +/- 0.0486` | `0.1110` | `[0.6844 .. 1.0000]` |
| **3A** | `l2_burst_spawns` / `sustain` / `revisit` | `0.0000` / `2.0000` / `0.0000` | `0.0000` | `[0..0]` / `[2..2]` / `[0..0]` |
| **3B-Mem** | `init_seen_acc` / `init_exc_acc` | `1.0000 +/- 0.0000` / `1.0000 +/- 0.0000` | `0.0000` | `[1.0000 .. 1.0000]` |
| **3B-Mem** | `init_heldout_acc` (`init_heldout_nll`) | `0.8372 +/- 0.0872` (`0.6815 +/- 0.4405`) | `0.1990` | `[0.1660 .. 1.0000]` |
| **3B-Mem** | `burst_flips` / `burst_seen_acc` | `0.0000 +/- 0.0000` / `1.0000 +/- 0.0000` | `0.0000` | `[0..0]` / `[1.0..1.0]` |
| **3B-Mem** | `rev_seen_acc` / `rev_exc_acc` | `1.0000 +/- 0.0000` / `1.0000 +/- 0.0000` | `0.0000` | `[1.0000 .. 1.0000]` |
| **3B-Mem** | `rev_heldout_acc` (`rev_heldout_nll`) | `0.8093 +/- 0.0598` (`0.8413 +/- 0.3579`) | `0.1363` | `[0.3965 .. 0.9844]` |
| **3B-Mem** | `storage_bytes` | `2240.0000 +/- 0.0000` | `0.0000` | `[2240.0 .. 2240.0]` |
| **3B-IC** | `ic_new_acc` | `0.9969 +/- 0.0043` | `0.0097` | `[0.9648 .. 1.0000]` |
| **3B-IC** | `ic_confirm_acc` | `0.9998 +/- 0.0004` | `0.0009` | `[0.9961 .. 1.0000]` |
| **3B-IC** | `ic_contra_acc` (`picked_contra_rate`) | `0.9881 +/- 0.0163` (`0.0080 +/- 0.0112`) | `0.0371` | `[0.8633 .. 1.0000]` |
| **3B-IC** | `ic_auroc_cf` / `ic_auroc_cn` | `0.9985 +/- 0.0020` / `0.9941 +/- 0.0081` | `0.0046` / `0.0186` | `[0.9820..1.0]` / `[0.9282..1.0]` |
| **3B-IC** | `ic_causality_diff` | `0.0000 +/- 0.0000` | `0.0000` | `[0.0000 .. 0.0000]` |
| **3C** | `real_a_fpr` / `code` / `twin` / `nuance_tpr` | `0.0000` / `1.0000` / `1.0000` / `1.0000` | `0.0000` | constant across 20 seeds |
| **3C** | `real_b_fpr` / `code` / `twin` / `nuance_tpr` | `0.0200 +/- 0.0229` / `1.0` / `1.0` / `1.0` | `0.0523` (FPR) | `fpr: [0.0000 .. 0.2000]` |

---

## 3. Null Controls (`n = 20` Held-Out Seeds Each)

All three negative controls collapsed cleanly below every pre-registered threshold (`F7 = True`):

| Sub-Suite | Model | Key Null Metrics (`mean +/- 95% CI`, `[min..max]`) | Outcome vs Bound |
|---|---|---|---:|
| **3A** | `Null_Untrained` | `l0=0.0048`, `l1_rho0=0.0048 +/- 0.0033 [0.0000..0.0301]`, `l1_rho5=0.0044`, `route=0.3333` | `< 0.18` PASS |
| **3A** | `Null_ShuffledTargets` | `l0=0.0755`, `l1_rho0=0.0759 +/- 0.0082 [0.0537..0.1225]`, `l1_rho5=0.0776`, `l2_avg=0.0840` | `< 0.18` PASS |
| **3A** | `Null_BlindZeroInput` | `l0=0.0688`, `l1_rho0=0.0681 +/- 0.0065 [0.0351..0.0872]`, `l1_rho5=0.0647`, `l2_avg=0.0643` | `< 0.18` PASS |
| **3B-Mem** | `Null_Untrained` | `init_heldout=0.0036`, `rev_seen=0.0050 [0.0000..0.0195]`, `rev_heldout=0.0042 [0.0000..0.0293]`, `0 B` | `< 0.15` PASS |
| **3B-Mem** | `Null_ShuffledTargets` | `burst_flips=5.9500 [1..11]`, `rev_seen=0.0966 [0.0527..0.1387]`, `rev_exc=0.1472`, `rev_heldout=0.0271 [0.0000..0.0605]`, `4060 B` | `< 0.15` PASS |
| **3B-Mem** | `Null_BlindZeroInput` | `rev_seen=0.0657 [0.0449..0.0879]`, `rev_exc=0.0371`, `rev_heldout=0.0437 [0.0137..0.0859]`, `2240 B` | `< 0.15` PASS |
| **3B-IC** | `Null_Untrained` | `ic_new=0.0047 [0.0000..0.0586]`, `ic_confirm=0.0068`, `ic_contra=0.0055 [0.0000..0.0703]`, `auroc_cf=0.5000` | `< 0.18` PASS |
| **3B-IC** | `Null_ShuffledTargets` | `ic_new=0.0643 [0.0391..0.1016]`, `ic_confirm=0.0625`, `ic_contra=0.0658 [0.0391..0.1133]`, `auroc_cf=0.5287` | `< 0.18` PASS |
| **3B-IC** | `Null_BlindZeroInput` | `ic_new=0.0238 [0.0000..0.0703]`, `ic_confirm=0.0217`, `ic_contra=0.0225 [0.0000..0.0898]`, `auroc_cf=0.5000` | `< 0.18` PASS |

---

## 4. Baselines, Structural Ablations & Paired Statistical Comparisons (`n = 20`)

### Sub-Suite 3A (`Cog` 164,406 params vs Baselines `>= Cog`: 165,376–165,862 params)
| Model | `l0_avg` | `l1_rho0_avg` | `l1_rho5_avg` | `l2_d0_pre_revisit` | `l2_d0_post_revisit` | `l2_avg` | `l2_rev_spawns` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Cog_Stage3_Full`** | `0.9867 +/- 0.0053` | `0.9976 +/- 0.0021` | `0.8439 +/- 0.0216` | `0.9950 +/- 0.0042` | `0.9796 +/- 0.0096` | `0.9097 +/- 0.0486` | `0.0000` |
| `Cog_OracleRouting` *(upper ref)* | `1.0000 +/- 0.0000` | `1.0000 +/- 0.0000` | `0.9989 +/- 0.0009` | `1.0000 +/- 0.0000` | `1.0000 +/- 0.0000` | `0.9321 +/- 0.0428` | `0.0000` |
| `Cog_NoGPM` *(ablation)* | `0.5186 +/- 0.0167` | `0.5438 +/- 0.0124` | `0.3927 +/- 0.0070` | `0.1555 +/- 0.0202` | `0.6343 +/- 0.0455` | `0.4923 +/- 0.0272` | `0.9500` |
| `Cog_UnigramRouter` *(ablation)* | `0.3868 +/- 0.0159` | `0.3965 +/- 0.0148` | `0.3770 +/- 0.0085` | `0.4178 +/- 0.0376` | `0.5564 +/- 0.0355` | `0.3934 +/- 0.0165` | `1.0000` |
| `Dense_NoReplay` (`165,376` p) | `0.3987 +/- 0.0086` | `0.4002 +/- 0.0058` | `0.3917 +/- 0.0055` | `0.0864 +/- 0.0146` | `0.9945 +/- 0.0024` | `0.4451 +/- 0.0111` | `0.0000` |
| `Dense_Replay128` (`165,376` p) | `0.4278 +/- 0.0100` | `0.4183 +/- 0.0078` | `0.3960 +/- 0.0048` | `0.1056 +/- 0.0141` | `0.9982 +/- 0.0014` | `0.5152 +/- 0.0134` | `0.0000` |
| `StaticMoE_Matched` (`165,862` p) | `0.4119 +/- 0.0081` | `0.4116 +/- 0.0078` | `0.3936 +/- 0.0045` | `0.0842 +/- 0.0111` | `0.9786 +/- 0.0057` | `0.4480 +/- 0.0099` | `0.0000` |

### Sub-Suite 3B-Mem (`Cog` 91,520 params vs `Dense_NoUpk` 93,584 params & `BudgetedDictionaryBaseline` `0` params, `2240 B`)
| Model | `Bytes` | `init_seen` / `heldout` | `burst_seen` (`flips`) | `rev_seen` / `exc` | `rev_heldout` (`nll`) |
|---|---:|---|---|---|---:|
| **`Cog_Stage3_Full`** | `2240` | `1.0000` / `0.8372 +/- 0.0872` | `1.0000` (`0.00`) | `1.0000` / `1.0000` | `0.8093 +/- 0.0598` (`0.8413`) |
| `Ablation_NoUpkeeper` | `0` | `1.0000` / `0.8372 +/- 0.0872` | `0.0647 +/- 0.0058` (`0.00`) | `1.0000` / `1.0000` | `0.4631 +/- 0.1470` (`3.4731`) |
| `Dense_NoUpk` (`93,584` p) | `0` | `1.0000` / `0.8913 +/- 0.0611` | `0.0664 +/- 0.0073` (`0.00`) | `1.0000` / `1.0000` | `0.6212 +/- 0.1397` (`2.0095`) |
| `BudgetedDictionaryBaseline` | `2240` | `0.7448` / `0.0637 +/- 0.0121` | `0.7448` (`0.00`) | `0.7469` / `0.7706` | `0.0591 +/- 0.0123` (`2.7728`) |

### Sub-Suite 3B-IC (`Cog` 528,753 params vs `Dense_Matched` 534,208 params)
| Model | `ic_new_acc` | `ic_confirm_acc` | `ic_contra_acc` | `picked_contra_rate` | `ic_auroc_cf` |
|---|---:|---:|---:|---:|---:|
| **`Cog_Stage3_Full`** | `0.9969 +/- 0.0043` | `0.9998 +/- 0.0004` | `0.9881 +/- 0.0163` | `0.0080 +/- 0.0112` | `0.9985 +/- 0.0020` |
| `Ablation_NoPCRectification` | `0.9979 +/- 0.0030` | `1.0000 +/- 0.0000` | `0.2721 +/- 0.1378` | `0.7258 +/- 0.1385` | `0.9988 +/- 0.0016` |
| `Dense_Matched` (`534,208` p) | `0.1166 +/- 0.0126` | `0.9992 +/- 0.0009` | `0.0564 +/- 0.0066` | `0.6252 +/- 0.0127` | `0.6819 +/- 0.0167` |

### Pre-Registered Paired Comparisons (`n = 20`)
| Comparison Key | `mean_diff +/- 95% CI` | `sd_diff` | `t_stat` | `wins` |
|---|---:|---:|---:|---:|
| `3a_l1_rho0_full_vs_dense_noreplay` | `+0.5974 +/- 0.0059` | `0.0134` | `200.09` | `20/20` |
| `3a_l1_rho0_full_vs_dense_replay128` | `+0.5794 +/- 0.0078` | `0.0179` | `144.72` | `20/20` |
| `3a_l1_rho0_full_vs_static_moe` | `+0.5860 +/- 0.0078` | `0.0179` | `146.56` | `20/20` |
| `3a_l1_rho0_full_vs_nogpm` | `+0.4539 +/- 0.0120` | `0.0273` | `74.22` | `20/20` |
| `3a_l1_rho0_full_vs_unigram` | `+0.6011 +/- 0.0147` | `0.0336` | `79.96` | `20/20` |
| `3a_l1_rho5_full_vs_dense_replay128` | `+0.4479 +/- 0.0227` | `0.0517` | `38.73` | `20/20` |
| `3a_l1_rho5_full_vs_unigram` | `+0.4669 +/- 0.0247` | `0.0565` | `36.98` | `20/20` |
| `3bm_init_heldout_full_vs_dict` | `+0.7735 +/- 0.0815` | `0.1859` | `18.61` | `20/20` |
| `3bm_burst_seen_full_vs_noupk` | `+0.9353 +/- 0.0058` | `0.0133` | `315.28` | `20/20` |
| `3bm_rev_heldout_full_vs_dict` | `+0.7502 +/- 0.0636` | `0.1450` | `23.14` | `20/20` |
| `3bic_contra_full_vs_nopc` | `+0.7160 +/- 0.1450` | `0.3309` | `9.68` | `20/20` |
| `3bic_contra_full_vs_dense` | `+0.9316 +/- 0.0204` | `0.0465` | `89.56` | `20/20` |
| `3c_real_a_nuance_new_vs_old` | `+1.0000 +/- 0.0000` | `0.0000` | `1e12` | `20/20` |
| `3c_real_b_nuance_new_vs_old` | `0.0000 +/- 0.0000` | `0.0000` | `0.00` | `0/20` |

---

## 5. What Failed, Negative Results & Imperfect Seeds (Same Prominence)

1. **`3B-IC` Attention-Only Baseline Shortcut (`Dense_Matched`)**: `Dense_Matched` (`534,208` params, `>= Cog`) solved `ic_confirm_acc = 0.9992` via the recency copy shortcut but failed `ic_new_acc = 0.1166` and `ic_contra_acc = 0.0564` across `{2e-3, 3e-3, 5e-3}` DEV sweeps and the 20-seed test run. Consequently, `Ablation_NoPCRectification` (`ic_new_acc = 0.9979`, `ic_confirm_acc = 1.0000`, `ic_contra_acc = 0.2721`) is the sole matched baseline that solves the clean associative task and isolates predictive-coding rectification.
2. **`3A L1 (rho=0.5)` Gap vs Oracle Routing**: On 50% mid-sequence regime switches, `Cog_Stage3_Full` scored `0.8439 +/- 0.0216` (`min=0.7387` on `seed=14`, `route=0.8349`), lagging `Cog_OracleRouting` (`0.9989 +/- 0.0009`) by `0.1550` because strictly causal routing needs several tokens after an unexpected header-free switch to re-estimate transition prototypes.
3. **`3A L2` Short-Segment Dip on Late Domains**: While `l2_d0_pre_revisit = 0.9950` and `l2_d0_post_revisit = 0.9796` remained near ceiling, overall `l2_avg` dropped on `seed=6` (`0.6844`), `seed=1` (`0.6997`), and `seed=16` (`0.7153`). Even `Cog_OracleRouting` dropped to `min=0.7337` (`mean=0.9321`), showing imperfect short-window convergence on newly spawned `D2` rather than pure routing failure.
4. **Baselines Re-fit `L2_D0_post_revisit` While Forgetting Pre-Revisit**: `Dense_Replay128` scored `l2_d0_post_revisit = 0.9982` only because `L2` includes 20 training steps on `D0` at the end; before revisiting `D0`, `l2_d0_pre_revisit` collapsed to `0.1056` (`Dense_NoReplay`: `0.0864`, `StaticMoE_Matched`: `0.0842`).
5. **`3B-Mem` Grokking Seed Variance & `Dense_NoUpk` Init Edge**: `Cog_Stage3_Full` hit `1.0000` on seen and exception facts across all 20 seeds, yet compositional held-out generalization dipped on `seed=18` (`init_heldout_acc = 0.1660`, `nll = 4.2987`, recovering to `0.7402` post-revision) and `seed=5` (`rev_heldout_acc = 0.3965`, `nll = 3.8515`, down from `1.0000` at init). Furthermore, `Dense_NoUpk` had a higher `init_heldout_acc` mean (`0.8913` vs `0.8372`) prior to completely shattering on the burst (`0.0664` vs `1.0000`) and suffering severe post-revision drops (`rev_heldout_acc = 0.6212`, `min=0.0254` on `seed=1`, `0.0527` on `seed=17`).
6. **`3B-IC` Partial Contra Leakage on 2/20 Seeds & `Ablation` High Variance**: `Cog_Stage3_Full` achieved `ic_contra_acc = 1.0000` on 18/20 seeds, but leaked partial contradictions on `seed=8` (`0.8633`, `picked_contra = 0.1016`, `gate_contra = 0.2063`) and `seed=17` (`0.8984`). Meanwhile, `Ablation_NoPCRectification` varied widely across seeds (`0.0000` to `0.9375`, `std = 0.3144`), depending on chance attention initialization.
7. **`3C` Non-Zero Open-Prose FPR & Legacy Detector Failure**: In `Regime B` (disjoint open Markdown), `real_b_fpr` averaged `0.0200` with non-zero false alarms on 3/20 seeds (`seed=11`: `0.10`, `seed=15`: `0.20`, `seed=16`: `0.10`). In `Regime A`, the legacy single-scale baseline counters remained broken (`real_a_old_nuance_tpr = 0.0000`, `real_a_old_twin_tpr = 0.0000`), and in `Regime B` both old and new nuance counters saturated at `1.0000` (`3c_real_b_nuance_new_vs_old` diff `= 0.0`).

---

## 6. Limitations & Scope Boundaries

- **Benchmark Scale**: Sub-suites `3A`, `3B-Mem`, and `3B-IC` evaluate compact synthetic symbolic architectures (`91,520`–`528,753` parameters); extrapolation to large-scale natural-language pretraining is not tested here.
- **Repository Hygiene**: `.integrity/` and `docs/CLAIMS.md` are not present in this repository, and the working tree was dirty during evaluation (`git_dirty=True` on top of commit `6344b0024615a8fe38ecea969bf93c709ee4b86e`).
- **Pre-Registration Compliance**: No `dmoe/` module, `experiments/run_stage3_continual_learning.py`, or falsifier threshold was modified after freezing `PREREG_v1.md` and launching `--mode test --workers 5`.

---

## 7. Pre-Registered Falsifier Ledger (`F1`–`F7`)

| Falsifier | Definition Summary | Result |
|---|---|---|
| `F1_strict_causality` | All `causality_gate <= 1e-5` (`0.0`) & `ic_causality_diff.max <= 1e-5` (`0.0`) | **PASS (`True`)** |
| `F2_3a_continual_retention_and_ablations` | `l1_rho0 >= 0.90` (`0.9976`), `l1_rho5 >= 0.80` (`0.8439`), beats Dense/StaticMoE/NoGPM/Unigram margins | **PASS (`True`)** |
| `F3_3a_boundary_free_l2_lifecycle` | `burst=0.0`, `sustain>=1.5` (`2.0`), `revisit=0.0`, `d0_pre>=0.85` (`0.9950`), `d0_post>=0.85` (`0.9796`), `l2_avg>=0.85` (`0.9097`) | **PASS (`True`)** |
| `F4_3bm_budget_burst_and_generalization` | `bytes<=4096` (`2240`), `init_seen>=0.95` (`1.0`), `init_ho>=0.55` (`0.8372`), `burst_flips=0`, `burst>=0.90` (`1.0`), `rev_seen>=0.95` (`1.0`), `rev_ho>=0.50` (`0.8093`), beats Dict/NoUpk | **PASS (`True`)** |
| `F5_3bic_zero_shot_contradiction_immunity` | `new>=0.90` (`0.9969`), `confirm>=0.90` (`0.9998`), `contra>=0.85` (`0.9881`), `>NoPC+0.40` (`0.2721`), `auroc_cf>=0.90` (`0.9985`) | **PASS (`True`)** |
| `F6_3c_real_stream_calibration` | `RealA fpr<=0.15` (`0.0`), `tpr>=0.90` (`1.0`), `RealB fpr<=0.15` (`0.0200`), `tpr>=0.90` (`1.0`), `L2 bsp/rsp=0.0` | **PASS (`True`)** |
| `F7_null_controls_fail` | All 3 Nulls `< 0.18` (`3A`, `3B-IC`) & `< 0.15` (`3B-Mem`) across 20 seeds | **PASS (`True`)** |

---

## 8. Verification Commands & Real Output Tails

### 1) Frozen 20-Seed Benchmark Execution
```powershell
python -u experiments/run_stage3_continual_learning.py --mode test --workers 5
```
Observed tail:
```text
[3B-IC] Null_BlindZeroInput    (760.0s) | new=0.024 conf=0.022 contra=0.022 auroc_cf=0.500 caus=0.0e+00
[3C] (185.3s) | RealA(fpr=0.000, code=1.000, twin=1.000, nuance=1.000) | RealB(fpr=0.020, code=1.000, twin=1.000, nuance=1.000)

Saved Stage 3 test results to D:\Dynamic_MOE_Transformer\results\stage3_continual_learning\results_v1.json | all_passed=True
```

### 2) Full Repository Pytest Suite (Run After Zero Further Code Edits)
```powershell
python -m pytest tests/ -q
```
Observed output:
```text
........................................................................ [ 40%]
........................................................................ [ 81%]
................................                                         [100%]
============================== warnings summary ===============================
... FutureWarning: The pynvml package is deprecated ...
176 passed, 1 warning in 198.10s (0:03:18)
```

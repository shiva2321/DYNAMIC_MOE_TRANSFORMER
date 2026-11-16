# Stage 2 Lock Report — Dynamic Thinking Loop, Epistemic Abstention & Conflict Catching, and Calibrated Latent World Model (`stage2_thinking_world_model_v1`)

## 1. Claim & Claim-Ladder Classification

- **Claim status (`research-protocol.md §9`)**: **`CONTROLLED` / `GENERALIZES`** on the Stage 2 3-task evaluation suite (`2A-Adj` adjacent multi-hop state composition, `2A-Ptr` distractor-gapped pointer multi-hop reasoning, `2B` zero-shot epistemic abstention & predictive-coding contradiction catching over shared vocabulary `[32..63]`, `2C` `H=1..3` latent world-model imagination & trajectory calibration), verified under strict causality (`causality_gate_max_diff = 0.0 <= 1e-5`) and matched parameter budgets (`218,595` params vs `210,496–227,993` baselines).
- **Locked Stage 2 release (`stage2_thinking_world_model_v1`)**: Building on locked Stage 1 ([`results_v8.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v8.json)), Stage 2 integrates `SurpriseDrivenLatentThinker` ([`dmoe/thinking.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/thinking.py)), split epistemic/predictive-coding uncertainty ([`dmoe/real_model.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/real_model.py)), and `LatentWorldModel` ([`dmoe/world_model.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/world_model.py)). On frozen `TEST_SEEDS = [0..19]` (`n = 20`) per [`PREREG_v1.md`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/PREREG_v1.md), `Cog_Stage2_Full` passes **all 7 pre-registered falsifiers (`F1–F7`, `all_passed: True`)** and is the only evaluated model solving all three domains simultaneously.
- **Scope qualifier**: Stage 2 proves test-time iterative depth generalization (`D <= 3 -> D = 4..5`), adaptive halting savings, zero-shot epistemic uncertainty + world-model contradiction *detection* (`2b_auroc_pc_contradiction = 0.923367`), and calibrated `H=1..3` rollout uncertainty (`2c_wm_ece = 0.020098`). It does **not** prove continual multi-domain learning without catastrophic forgetting or autonomous expert lifecycle without replay (Stage 3+ scope), nor automatic downstream answer override on contradicting prompts (`2b_obs_robust_acc = 0.783854`), and `Cog_Stage2_Full` trails `GRU_Baseline` on pure finite-state chains and `LoopedTransformer_Baseline` on `2A-Ptr`.

---

## 2. Headline Metrics (`results_v1.json`, `n=20` Test Seeds `[0..19]`)

All numbers below are copied verbatim from [`results_v1.json`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/results_v1.json) (`mean +- ci95_half`, `std`, `[min .. max]`):

| Task / Metric | `Cog_Stage2_Full` (`218,595` params, `lr=3e-3`) | `LoopedTransformer_Baseline` (`227,993` params, `lr=7e-3`) | `DenseRoPE_Deep` (`210,496` params, `lr=5e-3`) | `GRU_Baseline` (`217,408` params, `lr=7e-3`) |
| :--- | :--- | :--- | :--- | :--- |
| `2a_adj_ood45` | `0.927146 +- 0.005814` (`std=0.013266`, `[0.894591 .. 0.949266]`) | `0.821138 +- 0.014730` (`std=0.033609`, `[0.767834 .. 0.879404]`) | `0.687731 +- 0.017442` (`std=0.039798`, `[0.636616 .. 0.765193]`) | `0.985882 +- 0.002277` (`std=0.005196`, `[0.973973 .. 0.994536]`) |
| `2a_adj_active_frac` | `0.352920 +- 0.007293` (`[0.323672 .. 0.391055]`) | `1.000000` | `1.000000` | `1.000000` |
| `2a_adj_step0 -> step5` (`ood45`) | `0.573263 -> 0.938210` | `0.821138 -> 0.821138` | `0.687731 -> 0.687731` | `0.985882 -> 0.985882` |
| `2a_ptr_ood45` | `0.839642 +- 0.009826` (`std=0.022420`, `[0.792839 .. 0.876056]`) | `0.924324 +- 0.012220` (`std=0.027882`, `[0.877238 .. 0.970509]`) | `0.749779 +- 0.014323` (`std=0.032682`, `[0.688654 .. 0.819444]`) | `0.993257 +- 0.002652` (`std=0.006051`, `[0.980609 .. 1.000000]`) |
| `2a_ptr_active_frac` | `0.576907 +- 0.024737` (`[0.491713 .. 0.741127]`) | `1.000000` | `1.000000` | `1.000000` |
| `2a_ptr_step0 -> step5` (`ood45`) | `0.601435 -> 0.847675` | `0.924324 -> 0.924324` | `0.749779 -> 0.749779` | `0.993257 -> 0.993257` |
| `2b_clean_acc` | `0.982292 +- 0.014875` (`[0.843750 .. 1.000000]`) | `0.076042 +- 0.014708` (`[0.026042 .. 0.145833]`) | `0.084375 +- 0.016236` (`[0.046875 .. 0.218750]`) | `0.194531 +- 0.015469` (`[0.145833 .. 0.255208]`) |
| `2b_obs_robust_acc` | `0.783854 +- 0.101154` (`std=0.230802`, `[0.328125 .. 1.000000]`) | `0.057813 +- 0.009568` | `0.070313 +- 0.009556` | `0.200000 +- 0.013863` |
| `2b_auroc_unknown` | `0.921743 +- 0.020988` (`[0.740913 .. 0.956950]`) | `0.492255 +- 0.014055` | `0.496893 +- 0.009215` | `0.495014 +- 0.014284` |
| `2b_auroc_conflict_q` | `0.875241 +- 0.026778` (`[0.663710 .. 0.948269]`) | `0.502127 +- 0.015347` | `0.501467 +- 0.013317` | `0.509199 +- 0.012772` |
| `2b_auroc_abstain_all` | `0.898492 +- 0.022115` (`[0.702311 .. 0.947415]`) | `0.497191 +- 0.012366` | `0.499180 +- 0.007133` | `0.502106 +- 0.012123` |
| `2b_auroc_pc_contradiction` | `0.923367 +- 0.032677` (`[0.730740 .. 0.991753]`) | `0.502698 +- 0.015993` | `0.516186 +- 0.013831` | `0.504147 +- 0.014842` |
| `2b_selective_acc_50` | `0.831250 +- 0.025659` (`[0.609375 .. 0.906250]`) | `0.048177 +- 0.009187` | `0.052865 +- 0.011041` | `0.113542 +- 0.012295` |
| `2c_traj_acc` | `0.953235 +- 0.000833` (`[0.950439 .. 0.956787]`) | `0.952795 +- 0.000943` | `0.953308 +- 0.000847` | `0.952991 +- 0.001005` |
| `2c_cos_h1 / h2 / h3` | `0.783796 / 0.424371 / 0.432785` | `0.000000` (no WM) | `0.000000` (no WM) | `0.000000` (no WM) |
| `2c_unc_h1 / h2 / h3` | `0.246942 < 0.529846 < 0.595380` | `0.000000` (no WM) | `0.000000` (no WM) | `0.000000` (no WM) |
| `2c_wm_ece` | `0.020098 +- 0.002231` (`[0.012186 .. 0.027664]`) | `0.000000` | `0.000000` | `0.000000` |
| `2c_corrupt_auroc` | `0.993947 +- 0.000619` (`[0.990617 .. 0.996208]`) | `0.991932 +- 0.000466` | `0.995946 +- 0.000275` | `0.998604 +- 0.000080` |
| `2c_eval_toks_sec` | `72,183.7` | `290,562.0` | `390,314.9` | `1,062,251.1` |
| `wall_clock_sec` (`20` seeds) | `1522.10` | `546.95` | `441.60` | `200.96` |

---

## 3. Ablations & Paired Comparisons (`results_v1.json`)

### 3.1 Structural Ablation Models (`n=20` Test Seeds)

| Metric | `Cog_NoWM` (`196,007` params) | `Cog_OnePass` (`196,007` params) | `Cog_FixedThink` (`218,595` params) |
| :--- | :--- | :--- | :--- |
| `2a_adj_ood45` | `0.922801 +- 0.004519` (`[0.906332 .. 0.942333]`) | `0.720307 +- 0.019528` (`[0.629862 .. 0.803867]`) | `0.930664 +- 0.005676` (`[0.910714 .. 0.955901]`) |
| `2a_adj_active_frac` | `0.346865 +- 0.005587` | `0.000000` | `1.000000` |
| `2a_ptr_ood45` | `0.845127 +- 0.010552` (`[0.795396 .. 0.877778]`) | `0.702308 +- 0.022589` (`[0.598465 .. 0.797784]`) | `0.852252 +- 0.011895` (`[0.794444 .. 0.893443]`) |
| `2a_ptr_active_frac` | `0.557147 +- 0.024176` | `0.000000` | `1.000000` |
| `2b_clean_acc` | `0.977865 +- 0.014453` (`[0.859375 .. 1.000000]`) | `0.996094 +- 0.001942` (`[0.989583 .. 1.000000]`) | `0.839063 +- 0.126502` (`[0.130208 .. 1.000000]`) |
| `2b_auroc_abstain_all` | `0.870867 +- 0.021272` (`[0.731459 .. 0.936930]`) | `0.667059 +- 0.014496` (`[0.614787 .. 0.740763]`) | `0.835599 +- 0.073862` (`[0.499661 .. 0.964803]`) |
| `2b_auroc_pc_contradiction` | `0.490686 +- 0.015161` (`[0.418837 .. 0.538547]`) | `0.501150 +- 0.013726` (`[0.449354 .. 0.553548]`) | `0.833955 +- 0.044120` (`[0.616753 .. 0.986165]`) |
| `2c_traj_acc` | `0.952881 +- 0.001048` | `0.952991 +- 0.000874` | `0.952991 +- 0.000865` |
| `2c_cos_h1 / wm_ece` | `0.000000 / 0.000000` | `0.000000 / 0.000000` | `0.772850 / 0.019147` |
| `wall_clock_sec` | `1278.48` | `715.30` | `1729.65` |

### 3.2 Paired Comparisons (`results_v1.json`, `n=20` Matched Seeds)

| Comparison Key | `mean_diff +- ci95_half` | `sd_diff` | `t_stat` | `wins` |
| :--- | :--- | :--- | :--- | :--- |
| `2a_adj_ood_full_vs_onepass` | `+0.206838 +- 0.018430` | `0.042052` | `+21.9969` | `20/20` |
| `2a_adj_ood_full_vs_looped` | `+0.106008 +- 0.016613` | `0.037906` | `+12.5068` | `20/20` |
| `2a_adj_ood_full_vs_denserope` | `+0.239415 +- 0.018342` | `0.041850` | `+25.5842` | `20/20` |
| `2a_adj_ood_full_vs_gru` | `-0.058736 +- 0.006000` | `0.013691` | `-19.1866` | `0/20` |
| `2a_ptr_ood_full_vs_onepass` | `+0.137334 +- 0.018964` | `0.043271` | `+14.1938` | `20/20` |
| `2a_ptr_ood_full_vs_fixed` | `-0.012610 +- 0.010604` | `0.024195` | `-2.3307` | `6/20` |
| `2a_ptr_ood_full_vs_looped` | `-0.084682 +- 0.015367` | `0.035064` | `-10.8006` | `0/20` |
| `2a_ptr_ood_full_vs_denserope` | `+0.089863 +- 0.014791` | `0.033749` | `+11.9080` | `20/20` |
| `2a_ptr_ood_full_vs_gru` | `-0.153614 +- 0.011002` | `0.025103` | `-27.3662` | `0/20` |
| `2b_clean_acc_full_vs_denserope` | `+0.897917 +- 0.021610` | `0.049306` | `+81.4418` | `20/20` |
| `2b_clean_acc_full_vs_looped` | `+0.906250 +- 0.020023` | `0.045687` | `+88.7089` | `20/20` |
| `2b_clean_acc_full_vs_gru` | `+0.787760 +- 0.020840` | `0.047552` | `+74.0873` | `20/20` |
| `2b_auroc_abstain_full_vs_nowm` | `+0.027626 +- 0.034406` | `0.078504` | `+1.5738` | `16/20` |
| `2b_auroc_pc_full_vs_nowm` | `+0.432681 +- 0.041891` | `0.095583` | `+20.2443` | `20/20` |
| `2b_auroc_pc_full_vs_looped` | `+0.420669 +- 0.039336` | `0.089752` | `+20.9609` | `20/20` |
| `2c_traj_full_vs_nowm` | `+0.000354 +- 0.000532` | `0.001213` | `+1.3053` | `11/20` |
| `2c_corrupt_auroc_full_vs_nowm` | `+0.001007 +- 0.000736` | `0.001679` | `+2.6820` | `16/20` |

---

## 4. Calibrated Null Controls (`results_v1.json`, `n=20` Test Seeds)

| Metric | `Null_Untrained` | `Null_ShuffledTargets` | `Null_BlindZeroInput` |
| :--- | :--- | :--- | :--- |
| `2a_adj_ood45` | `0.000673 +- 0.000361` | `0.208216 +- 0.011277` | `0.111218 +- 0.017241` |
| `2a_ptr_ood45` | `0.001177 +- 0.001064` | `0.000000 +- 0.000000` | `0.147562 +- 0.023511` |
| `2b_clean_acc` | `0.003906 +- 0.002856` | `0.031510 +- 0.006766` | `0.000000 +- 0.000000` |
| `2b_auroc_abstain_all` | `0.502492 +- 0.011803` | `0.500502 +- 0.011849` | `0.500000 +- 0.000000` |
| `2b_auroc_pc_contradiction` | `0.508562 +- 0.016177` | `0.515803 +- 0.011697` | `0.500000 +- 0.000000` |
| `2c_traj_acc` | `0.003125 +- 0.001161` | `0.016431 +- 0.001315` | `0.015710 +- 0.000993` |
| `2c_cos_h1` | `0.019629 +- 0.005648` | `0.645804 +- 0.009514` (note below) | `0.191359 +- 0.063942` |
| `wall_clock_sec` | `6.44` | `1517.59` | `0.00` (on trained weights) |

Note on `Null_ShuffledTargets`: shuffling `yb` across batch items destroys all supervised downstream metrics (`2a_ptr_ood45 = 0.000000`, `2b_clean_acc = 0.031510`, `2c_traj_acc = 0.016431` ~= chance `1/64`), while `LatentWorldModel` receives unshuffled self-supervised transition context `xb` (`2c_cos_h1 = 0.645804`). `Null_BlindZeroInput` and `Null_Untrained` confirm both latent alignment and task accuracy depend directly on live token inputs (`0.003125` / `0.015710`).

---

## 5. Pre-Registered Falsifiers (`F1`–`F7`) & DEV Summary

All 7 pre-registered boolean falsifiers from [`PREREG_v1.md`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/PREREG_v1.md) passed on frozen `TEST_SEEDS = [0..19]` (`all_passed: True`):

| Falsifier | Outcome (`results_v1.json`) | Measured Test Value vs Threshold |
| :--- | :--- | :--- |
| `F1_causality_gate` | **`true`** | `0.0 <= 1e-5` |
| `F2_thinking_beats_onepass_and_step5_gt_step0` | **`true`** | Adj vs OnePass `+0.206838 > +0.18`; Ptr vs OnePass `+0.137334 > +0.12`; Adj s5-s0 `+0.364947 > +0.25`; Ptr s5-s0 `+0.246240 > +0.18` |
| `F3_adaptive_compute_efficiency` | **`true`** | `2a_adj_ood45 = 0.927146 >= 0.85` (`act = 0.352920 <= 0.50`); `2a_ptr_ood45 = 0.839642 >= 0.78` (`act = 0.576907 <= 0.65`); Full vs Fixed `-0.012610 >= -0.04` |
| `F4_abstention_and_pc_conflict` | **`true`** | `2b_clean_acc = 0.982292 >= 0.90`; `2b_auroc_abstain_all = 0.898492 >= 0.85`; `2b_auroc_pc_contradiction = 0.923367 >= 0.85` & `+0.432681 > +0.25` vs `Cog_NoWM` |
| `F5_world_model_no_traj_regression` | **`true`** | `2c_traj_acc = 0.953235 >= 0.90`, `+0.000354 >= -0.005` vs `Cog_NoWM`; `2c_cos_h1 = 0.783796 >= 0.70` |
| `F6_wm_horizon_and_corrupt_calibration` | **`true`** | `unc = 0.246942 < 0.529846 < 0.595380`; `2c_wm_ece = 0.020098 <= 0.10`; `2c_corrupt_auroc = 0.993947 >= 0.90` |
| `F7_null_controls_fail` | **`true`** | `Null_Untrained (0.001177, 0.003906, 0.003125)`, `Null_ShuffledTargets (0.000000, 0.031510, 0.016431)`, `Null_BlindZeroInput (0.147562, 0.000000, 0.015710)` all `< (0.20, 0.15, 0.15)` |

DEV tuning & §7 baseline checks (`13` entries in [`tuning_log_v1.json`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/tuning_log_v1.json) on `DEV_SEEDS = [100..104]`): selected `lr_cog = 3e-3`, `lr_looped = 7e-3`, `lr_dense = 5e-3`, `lr_gru = 7e-3`. Extended DEV runs (`task-4996`, `task-5000`) verified `DenseRoPE_Deep` (`lr=3e-3`) groks Task 2B at `steps_2b = 1000..1500` (`clean_acc = 0.9948..1.0000`, `auroc_abstain_all = 0.9120`), yet remains at chance on `auroc_pc_contradiction` (`0.4862` vs `Cog_Stage2_Full 0.9999` at `1000` steps).

---

## 6. Test Suite Status (`pytest tests/ -q`)

- Full repository suite run after all Stage 2 code and benchmark runs (`task-5028`): **`160 passed, 1 warning in 53.64s`** (`0` failed, `0` skipped, including JEPA `11` tests, Stage 1 causal memory, cognitive architecture, surprise thinking, and Stage 2 world-model abstention).

---

## 7. Failures, Trade-Offs, Limitations & Negative Results

1. **Finite-state DFA baselines (`GRU_Baseline` & `LoopedTransformer_Baseline`) beat `Cog_Stage2_Full` on `2A`**: `GRU_Baseline` achieves `0.985882` (`2A-Adj`) and `0.993257` (`2A-Ptr`) at `~7.6x` lower wall clock (`200.96s` vs `1522.10s`); `LoopedTransformer_Baseline` beats `Cog_Stage2_Full` on `2A-Ptr` (`0.924324` vs `0.839642`) while using `1.000` active tokens and trailing on `2A-Adj` (`0.821138` vs `0.927146`). Both fail Task 2B rule binding (`0.194531` and `0.076042`) and lack world-model rollout (`0.000`).
2. **Adaptive vs Fixed Compute Trade-Off**: `Cog_Stage2_Full` trails `Cog_FixedThink` slightly on `2a_ptr_ood45` (`0.839642` vs `0.852252`, `diff = -0.012610 +- 0.010604`), while saving `~42.3%` active compute on `2A-Ptr`, `~64.7%` on `2A-Adj`, and avoiding `Cog_FixedThink`'s Task 2B instability (`0.839063`, `min=0.130208` vs `0.982292`).
3. **Contradiction Detection vs Answer Rectification (`Task 2B`)**: `Cog_Stage2_Full` reliably detects observation contradictions (`2b_auroc_pc_contradiction = 0.923367`) and shows non-significant change vs `Cog_NoWM` on missing/conflicting query softmax ranking (`+0.027626 +- 0.034406`, `t = +1.5738`). Downstream `2b_obs_robust_acc` is `0.783854 +- 0.101154` (`min = 0.328125`), meaning detected observation contradictions are not yet consistently rectified in working memory across all seeds (Stage 3+ target). Also `2b_ece = 0.304413` on OOD mixtures because uncertainty logits are uncalibrated probabilities on unseen `unknown`/`conflict` mixes.
4. **Multi-step World Model Cosine & Trajectory Ceiling (`Task 2C`)**: Cosine similarity drops from `cos_h1 = 0.783796` to `cos_h2 = 0.424371` and `cos_h3 = 0.432785` (tracked accurately by horizon uncertainties `0.246942 < 0.529846 < 0.595380`, `2c_wm_ece = 0.020098`). Downstream `2c_traj_acc = 0.953235` is saturated near the theoretical `15/16` deterministic ceiling across all models (`+0.000354` vs `Cog_NoWM`), demonstrating calibrated rollout without regression rather than improved next-token trajectory accuracy on `2C`.
5. **Governance & Repo State**: `git_dirty: True` on `6344b0024615a8fe38ecea969bf93c709ee4b86e`; `.integrity/` and `docs/CLAIMS.md` are absent (`research-protocol.md §10, §14`).

---

## 8. Key Artifacts
- [`PREREG_v1.md`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/PREREG_v1.md)
- [`tuning_log_v1.json`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/tuning_log_v1.json)
- [`results_v1.json`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/results_v1.json)
- [`STAGE2_LOCK_REPORT.md`](file:///d:/Dynamic_MOE_Transformer/results/stage2_thinking_world_model/STAGE2_LOCK_REPORT.md)
- [`dmoe/thinking.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/thinking.py)
- [`dmoe/world_model.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/world_model.py)
- [`dmoe/real_model.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/real_model.py)
- [`dmoe/stage2_baselines.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/stage2_baselines.py)
- [`experiments/run_stage2_thinking_world_model.py`](file:///d:/Dynamic_MOE_Transformer/experiments/run_stage2_thinking_world_model.py)
- [`tests/test_stage2_world_model_abstention.py`](file:///d:/Dynamic_MOE_Transformer/tests/test_stage2_world_model_abstention.py)

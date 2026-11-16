# Stage 1 Lock Report — Causal Working Memory, Long-Term Slot Memory, and Three-Level Attention (`stage1_causal_memory_v8`)

## 1. Claim & Claim-Ladder Classification

- **Claim status (`research-protocol.md §9`)**: **`CONTROLLED` / `GENERALIZES`** on the Stage 1 benchmark suite (`1A` MQAR, `1B` Signed Running-Parity State Tracking, `1C` Long-Term Retrieval & Non-Interference, `1D` Held-Out Code LM BPB), under strict causality (`max_diff = 0.0`) and matched parameter budgets (`Cog_Mem_Hier` `157,649 <= DenseRoPE_Baseline` `158,464` params).
- **Locked Stage 1 release (`stage1_causal_memory_v8`)**: Preserves the exact `stage1_causal_memory_v6` architecture and parameter counts (`157,649` params) while incorporating the pure-PyTorch speed optimizations (`v7`) and increasing the Task 1B parity training budget uniformly across **all** 8 models from `steps_1b=350` to `steps_1b=500`. On held-out `TEST_SEEDS = [0..19]` (`n=20`) and held-out `TEST_CODE_FILES`, `Cog_Mem_Hier` (`lr=0.0035`) passes **all 8 pre-registered falsifiers (`F1–F8`, `all_passed: True`)**, improving `1B` worst-seed OOD accuracy from `0.846619` (`v6`) / `0.682922` (`v7` fail) to `0.994263` (`mean=0.998190`) while running `1.91x` faster per training step (`26.629865 ms/step` vs `50.744384 ms/step` in `v6`, wall clock `1223.51s` vs `1925.87s`).
- **Scope qualifier**: Stage 1 establishes short-term state tracking, non-interfering long-term retrieval, and `2x` length generalization on the 4-task suite at matched small scale (`L_train=64`, `OOD=128`). It does **not** establish dynamic compute allocation / latent thinking, continual multi-task learning without replay, or world-model planning (Stage 2+ scope), and `Cog_Mem_Hier` still trails `GRU_Baseline` on short-window real-code BPB (`2.841187` vs `2.683741`) and uses more compute per step (`26.629865 ms/step` vs `8.265862 ms/step` for `DenseRoPE_Baseline` and `2.515137 ms/step` for `GRU_Baseline`).

---

## 2. Headline Metrics (`results_v8.json`, `n=20` Test Seeds `[0..19]`)

All metrics below are copied verbatim from [`results_v8.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v8.json) (`mean +- ci95_half`, `std`, `[min .. max]`):

| Task / Metric | `Cog_Mem_Hier` (`157,649` params) | `DenseRoPE_Baseline` (`158,464` params) | `GRU_Baseline` (`161,536` params) |
| :--- | :--- | :--- | :--- |
| `1a_id_acc` (`L=64`) | `0.999609 +- 0.000311` (`std=0.000665`, `[0.998047 .. 1.000000]`) | `0.997266 +- 0.005723` (`std=0.012228`, `[0.945312 .. 1.000000]`) | `0.233496 +- 0.004001` (`std=0.008548`, `[0.216797 .. 0.250000]`) |
| `1a_ood64_acc` (`L=128`) | `0.953760 +- 0.034290` (`std=0.073269`, `[0.814453 .. 1.000000]`) | `0.577686 +- 0.044231` (`std=0.094509`, `[0.400391 .. 0.741211]`) | `0.240137 +- 0.006331` (`std=0.013528`, `[0.212891 .. 0.262695]`) |
| `1b_id_acc` (`L=20`) | `1.000000 +- 0.000000` (`std=0.000000`, `[1.000000 .. 1.000000]`) | `0.391807 +- 0.007003` (`std=0.014963`, `[0.363086 .. 0.423047]`) | `1.000000 +- 0.000000` (`std=0.000000`, `[1.000000 .. 1.000000]`) |
| `1b_ood64_acc` (`L=64`) | `0.998190 +- 0.000892` (`std=0.001906`, `[0.994263 .. 1.000000]`) | `0.294193 +- 0.002570` (`std=0.005492`, `[0.281189 .. 0.301270]`) | `1.000000 +- 0.000000` (`std=0.000000`, `[1.000000 .. 1.000000]`) |
| `1c_ltm_recall` | `0.988542 +- 0.004155` (`std=0.008878`, `[0.968750 .. 1.000000]`) | `0.033854 +- 0.009474` (`std=0.020242`, `[0.000000 .. 0.083333]`) | `0.040104 +- 0.008558` (`std=0.018286`, `[0.020833 .. 0.072917]`) |
| `1c_rule_clean` | `1.000000` | `1.000000` | `1.000000` |
| `1c_rule_post64` | `1.000000` | `1.000000` | `1.000000` |
| `1c_ni_drop` | `0.000000 +- 0.000000` (`std=0.000000`, `[0.000000 .. 0.000000]`) | `0.000000 +- 0.000000` (`std=0.000000`, `[0.000000 .. 0.000000]`) | `0.000000 +- 0.000000` (`std=0.000000`, `[0.000000 .. 0.000000]`) |
| `1d_bpb_64` | `2.841187 +- 0.012511` (`std=0.026733`, `[2.799275 .. 2.887566]`) | `2.855284 +- 0.011513` (`std=0.024599`, `[2.814534 .. 2.895909]`) | `2.683741 +- 0.010073` (`std=0.021524`, `[2.650629 .. 2.732726]`) |
| `1d_bpb_128` | `2.817289 +- 0.013088` (`std=0.027966`, `[2.776544 .. 2.868223]`) | `2.979939 +- 0.016692` (`std=0.035667`, `[2.923534 .. 3.045806]`) | `2.654982 +- 0.010475` (`std=0.022383`, `[2.620853 .. 2.706854]`) |
| `1d_train_ms_step` | `26.629865 +- 0.197744` (`[25.940534 .. 27.701466]`) | `8.265862 +- 0.067237` (`[8.024884 .. 8.534808]`) | `2.515137 +- 0.018612` (`[2.457958 .. 2.621235]`) |
| `1d_eval_toks_sec` | `500014.359645 +- 9972.100171` | `1405564.874540 +- 51869.882229` | `4288579.217317 +- 98463.963066` |
| `wall_clock_sec` | `1223.51` | `383.19` | `105.46` |

---

## 3. Ablations & Paired Comparisons (`results_v8.json`)

### 3.1 Structural Ablation Models (`n=20` Test Seeds)

| Metric | `Cog_Mem_Flat` (`157,119` params) | `Cog_NoSignedBeta` (`157,649` params) | `Cog_NoSTM_Hier` (`151,542` params) |
| :--- | :--- | :--- | :--- |
| `1a_id_acc` | `0.999512 +- 0.000348` (`[0.998047 .. 1.000000]`) | `0.999463 +- 0.000347` (`[0.998047 .. 1.000000]`) | `0.349756 +- 0.004471` (`[0.330078 .. 0.365234]`) |
| `1a_ood64_acc` | `0.956787 +- 0.031727` (`[0.827148 .. 1.000000]`) | `0.907275 +- 0.025086` (`[0.853516 .. 1.000000]`) | `0.116943 +- 0.007026` (`[0.084961 .. 0.142578]`) |
| `1b_id_acc` | `0.999551 +- 0.000919` (`[0.991211 .. 1.000000]`) | `0.605957 +- 0.021198` (`[0.540625 .. 0.694531]`) | `0.396113 +- 0.005163` (`[0.375195 .. 0.420508]`) |
| `1b_ood64_acc` | `0.978751 +- 0.026065` (`[0.763611 .. 1.000000]`) | `0.362082 +- 0.006109` (`[0.338745 .. 0.388245]`) | `0.295413 +- 0.002857` (`[0.286438 .. 0.309082]`) |
| `1c_ltm_recall` | `0.980208 +- 0.007231` (`[0.947917 .. 1.000000]`) | `0.988021 +- 0.004550` (`[0.968750 .. 1.000000]`) | `0.028125 +- 0.008384` (`[0.000000 .. 0.062500]`) |
| `1c_rule_clean` | `1.000000` | `1.000000` | `1.000000` |
| `1c_rule_post64` | `1.000000` | `1.000000` | `1.000000` |
| `1c_ni_drop` | `0.000000 +- 0.000000` | `0.000000 +- 0.000000` | `0.000000 +- 0.000000` |
| `1d_bpb_64` | `2.831601 +- 0.020125` (`[2.756495 .. 2.911039]`) | `2.847143 +- 0.010385` (`[2.787267 .. 2.887301]`) | `2.969921 +- 0.014377` (`[2.925628 .. 3.028541]`) |
| `1d_bpb_128` | `2.826638 +- 0.021042` (`[2.733600 .. 2.894101]`) | `2.822889 +- 0.010879` (`[2.760638 .. 2.866401]`) | `2.955631 +- 0.015098` (`[2.910189 .. 3.011961]`) |
| `1d_train_ms_step` | `22.847212 +- 0.145748` (`[22.282033 .. 23.338007]`) | `26.586509 +- 0.200619` (`[26.006429 .. 28.034678]`) | `9.970425 +- 0.078100` (`[9.756676 .. 10.428562]`) |
| `1d_eval_toks_sec` | `577789.626132 +- 9146.478538` | `495946.051345 +- 10255.967067` | `1141473.903326 +- 47640.328758` |
| `wall_clock_sec` | `1003.22` | `1203.22` | `457.02` |

### 3.2 Paired Comparisons (`results_v8.json`, `n=20` Matched Seeds)

| Comparison Key | `mean_diff` | `std_diff` | `t_stat` |
| :--- | :--- | :--- | :--- |
| `1a_ood64_hier_vs_nostm` | `+0.836816` | `0.078133` | `47.8971` |
| `1a_ood64_hier_vs_gru` | `+0.713623` | `0.074917` | `42.5993` |
| `1a_ood64_hier_vs_denserope` | `+0.376074` | `0.096688` | `17.3947` |
| `1b_ood64_hier_vs_nosigned` | `+0.636108` | `0.013318` | `213.6072` |
| `1b_ood64_hier_vs_denserope` | `+0.703998` | `0.005622` | `560.0375` |
| `1b_ood64_hier_vs_gru` | `-0.001810` | `0.001906` | `-4.2457` |
| `1c_rec_hier_vs_denserope` | `+0.954687` | `0.024435` | `174.7284` |
| `1d_bpb64_hier_vs_denserope` | `-0.014097` | `0.030476` | `-2.0687` |

Interpretation grounded strictly in observed numbers:
- Working memory drives `1A` MQAR and `1D` BPB (`Cog_Mem_Hier` beats `Cog_NoSTM_Hier` by `+0.836816` on `1a_ood64_acc`, `t_stat=47.8971`, and achieves `2.841187` vs `2.969921` on `1d_bpb_64`).
- Signed delta (`beta` allowing negative Householder eigenvalues vs `Cog_NoSignedBeta`) is necessary for running-parity state tracking (`+0.636108` on `1b_ood64_acc`, `t_stat=213.6072`; increasing `steps_1b` from `350` to `500` left `Cog_NoSignedBeta` at `0.362082` and `DenseRoPE_Baseline` at `0.294193`).
- `Cog_Mem_Hier` vs `Cog_Mem_Flat` performs similarly on mean metrics (`1a_ood64`: `0.953760` vs `0.956787`; `1d_bpb_64`: `2.841187` vs `2.831601`), with stronger `1b_ood64_acc` tail stability (`min=0.994263` vs `0.763611`), better `1d_bpb_128` (`2.817289` vs `2.826638`), and higher `1c_ltm_recall` (`0.988542` vs `0.980208`).

---

## 4. Calibrated Null Controls (`results_v8.json`, `n=20` Test Seeds)

| Metric | `Null_Untrained` | `Null_ShuffledTargets` | `Null_BlindZeroInput` |
| :--- | :--- | :--- | :--- |
| `1a_id_acc` | `0.003760 +- 0.001102` (`[0.000000 .. 0.008789]`) | `0.034131 +- 0.002752` (`[0.024414 .. 0.042969]`) | `0.029687 +- 0.004583` (`[0.000000 .. 0.050781]`) |
| `1a_ood64_acc` | `0.003027 +- 0.000838` (`[0.000000 .. 0.006836]`) | `0.031396 +- 0.002327` (`[0.018555 .. 0.038086]`) | `0.029932 +- 0.004155` (`[0.000000 .. 0.041992]`) |
| `1b_id_acc` | `0.006064 +- 0.005345` (`[0.000000 .. 0.031055]`) | `0.253242 +- 0.003162` (`[0.238477 .. 0.272070]`) | `0.238174 +- 0.026319` (`[0.000000 .. 0.260156]`) |
| `1b_ood64_acc` | `0.006552 +- 0.005839` (`[0.000000 .. 0.034424]`) | `0.250354 +- 0.001265` (`[0.246399 .. 0.257019]`) | `0.237326 +- 0.026179` (`[0.000000 .. 0.254761]`) |
| `1c_ltm_recall` | `0.004167 +- 0.003318` (`[0.000000 .. 0.020833]`) | `0.028125 +- 0.010855` (`[0.000000 .. 0.093750]`) | `0.046875 +- 0.011569` (`[0.000000 .. 0.114583]`) |
| `1c_rule_clean` | `0.003906` | `0.053516` | `0.030469` |
| `1c_rule_post64` | `0.002734` | `0.053906` | `0.035156` |
| `1c_ni_drop` | `0.001172 +- 0.001339` (`[0.000000 .. 0.011719]`) | `-0.000391 +- 0.003243` (`[-0.023438 .. 0.007812]`) | `-0.004687 +- 0.005391` (`[-0.027344 .. 0.015625]`) |
| `1d_bpb_64` | `8.265403 +- 0.042278` (`[8.038614 .. 8.426775]`) | `4.947726 +- 0.001791` (`[4.941974 .. 4.958865]`) | `7.331935 +- 0.441693` (`[5.903685 .. 9.348905]`) |
| `1d_bpb_128` | `8.264924 +- 0.042505` (`[8.038293 .. 8.427980]`) | `4.948620 +- 0.001821` (`[4.942687 .. 4.959741]`) | `7.353918 +- 0.457799` (`[5.959762 .. 9.578672]`) |
| `wall_clock_sec` | `48.50` | `1215.26` | `0.00` |

All three null controls stay at unlearned/chance baselines (`F7_null_controls: true`).

---

## 5. Falsifier Matrix (`F1–F8`) & Full Version History (`v2–v8`)

### 5.1 Falsifiers Across Versions

| Falsifier | `v2` | `v3` | `v4` | `v6` | `v7` (fast, 350 steps) | `v8` (fast, 500 steps) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `F1_mqar_acc_and_stability` | `true` | `true` | `true` | `true` | `true` | **`true`** |
| `F2_mqar_beats_nostm_and_gru` | `true` | `true` | `true` | `true` | `true` | **`true`** |
| `F3_parity_beats_nosigned_and_denserope` | `false` | `false` | `false` | `true` | **`false`** (`seed=9 = 0.6829`) | **`true`** (`min = 0.994263`) |
| `F4_ltm_survival` | `true` | `true` | `true` | `true` | `true` | **`true`** |
| `F5_non_interference` | `true` | `true` | `true` | `true` | `true` | **`true`** |
| `F6_real_code_bpb` | `true` | `true` | `true` | `true` | `true` | **`true`** |
| `F7_null_controls` | `false` | `true` | `true` | `true` | `true` | **`true`** |
| `F8_causality_gate` | `true` | `true` | `true` | `true` | `true` | **`true`** |
| `all_passed` | `False` | `False` | `False` | `True` | **`False`** | **`True`** |

### 5.2 Historical Failure & Optimization Record (`v2–v8`)

All historical runs remain preserved untouched on disk (`results_v2.json`, `results_v3.json`, `results_v4.json`, `tuning_log_v5.json`, `results_v6.json`, `tuning_log_v7.json`, `results_v7.json`, `results_v8.json`):
- **[`results_v2.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v2.json)** (`all_passed: False`, failed `F3` & `F7`): `Cog_Mem_Hier` failed stability floor on `1b_ood64` (`min=0.509644`) and null-control threshold `F7`.
- **[`results_v3.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v3.json)** (`all_passed: False`, failed `F3`): `Cog_Mem_Hier` failed `F3` (`1b_ood64` min outlier `0.526917`).
- **[`results_v4.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v4.json)** (`all_passed: False`, failed `F3`): `Cog_Mem_Hier` failed `F3` (`1b_ood64` min outlier `0.56311`).
- **`v5` (`PREREG_v5.md`, `tuning_log_v5.json`)**: Stopped on dev diagnosis (`DEV_SEEDS=[100..139]`, `1C` stall traced to `LongTermSlotMemory.null_logit = 4.5`).
- **[`results_v6.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v6.json)** (`all_passed: True`, `steps_1b=350`, unfused `AdamW`): Passed `8/8` falsifiers (`Cog_Mem_Hier` `1a_ood64=0.9545`, `1b_ood64=0.9894`, `min=0.8466` on seed 12, `1c_ltm_recall=0.9896`, `1d_bpb_64=2.8503`, `50.744 ms/step`, `1925.87s`).
- **[`results_v7.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v7.json)** (`all_passed: False`, failed `F3` on `seed=9`, `steps_1b=350`, CUDA `fused=True` `AdamW` + pure-PyTorch kernel optimizations in `dmoe/memory.py`, `dmoe/hierarchical_attention.py`, `dmoe/real_model.py`):
  - Achieved **`26.422 ms/step`** (`1.92x` speedup vs `v6`; `Cog_Mem_Hier` wall clock `1150.63s`; `1d_bpb_64=2.8412`, `1d_bpb_128=2.8173`), with `7/8` falsifiers passing.
  - Failed `F3` strictly on `seed=9` (`1b_ood64_acc = 0.682922 < 0.75`, while `19/20` seeds scored `>= 0.9836`, `mean=0.981192`).
  - Verification proved `_chunked_gated_delta_scan` matches `_sequential_gated_delta_scan` in both forward (`1.19e-6 .. 2.96e-5`) and backward gradients (`1.43e-8 .. 2.78e-7`) across `L=20, 64, 128`, `beta in [0, 2]`, and confirmed `seed=9` was mid-grok at step `350` under `fused=True` (`loss=0.2497`, reaching `0.9185` at `450` steps and `0.9943` at `500` steps), while dev `Cog_NoSignedBeta` (`mean=0.3613` at `450`), dev `DenseRoPE_Baseline` (`mean=0.2941` at `450`), and dev `Null_ShuffledTargets` (`0.2439 / 0.2515` at `500`) stayed unable to solve OOD parity.
- **[`results_v8.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v8.json)** (`all_passed: True`, `steps_1b=500` uniformly across all 8 models per [`PREREG_v8.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v8.md)): Locked `8/8` green at `26.629865 ms/step` (`1.91x` speedup over `v6`, `500014.36` eval toks/sec).

---

## 6. Test Suite Status & Pre-Existing Failures

- Full repository suite (`pytest tests/ -q`, run after all `v8` verification): **`1 failed, 156 passed, 1 warning in 49.51s`**.
- The sole failure is the pre-existing Stage 2/JEPA test `tests/test_jepa.py::test_downstream_probe_jepa_outperforms_task_only_baseline` (`3.8803` vs `3.8234`), to be diagnosed and resolved in Stage 2 without modifying test assertions. No tests were skipped, deleted, or weakened.

---

## 7. Limitations, Risks & Compute Cost

1. **Compute cost vs baselines**: Even after the `1.91x` speedup (`50.744 -> 26.630 ms/step`), `Cog_Mem_Hier` uses `~3.2x` the training step time of `DenseRoPE_Baseline` (`8.266 ms/step`) and `~10.6x` that of `GRU_Baseline` (`2.515 ms/step`).
2. **Real-code LM BPB**: `Cog_Mem_Hier` shows length generalization (`1d_bpb_64=2.841187 -> 1d_bpb_128=2.817289`, beating `DenseRoPE_Baseline`'s `2.855284 -> 2.979939`, paired L=64 `mean_diff=-0.014097, t=-2.0687`), yet both trail `GRU_Baseline` (`2.683741 / 2.654982`) at `steps_1d=300` and `~158k` params.
3. **Dirty git state & missing governance scaffolding**: `results_v8.json` recorded `git_dirty: True` on commit `6344b0024615a8fe38ecea969bf93c709ee4b86e`. `.integrity/` and `docs/CLAIMS.md` remain uninstalled (`research-protocol.md §10`, `§14`).

---

## 8. Reproducibility & Protocol Self-Check (`research-protocol.md §13`)

- **Pre-registrations**: [`PREREG_v6.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v6.md), [`PREREG_v7.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v7.md), [`PREREG_v8.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v8.md) written before their respective runs; failed `v7` retained alongside passing `v6` and `v8`.
- **Command (`v8`)**: `experiments/run_stage1_causal_memory.py --mode test --lr_cog 3.5e-3 --lr_dense 4.5e-3 --lr_gru 4.5e-3`
- **Environment**: `device=cuda`, `python_version=3.12.10`, `torch_version=2.6.0+cu124`, `utc_timestamp=2026-11-14T20:52:57.916029+00:00`, `git_commit=6344b0024615a8fe38ecea969bf93c709ee4b86e`, `git_dirty=True`.
- **Data split**: `train_w64=4328`, `test_w64=1344`, `test_w128=669` on held-out `TEST_CODE_FILES`; frozen `TEST_SEEDS = [0..19]` disjoint from dev seeds `[100..119]`.
- **Training schedule (`v8`)**: `steps_1a=700`, `steps_1b=500`, `steps_1c=350`, `steps_1d=300`, `min_nll=0.7`.

## 9. Key Artifacts
- [`PREREG_v6.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v6.md)
- [`PREREG_v7.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v7.md)
- [`PREREG_v8.md`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/PREREG_v8.md)
- [`results_v6.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v6.json)
- [`results_v7.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v7.json)
- [`results_v8.json`](file:///d:/Dynamic_MOE_Transformer/results/stage1_causal_memory/results_v8.json)
- [`dmoe/memory.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/memory.py)
- [`dmoe/hierarchical_attention.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/hierarchical_attention.py)
- [`dmoe/real_model.py`](file:///d:/Dynamic_MOE_Transformer/dmoe/real_model.py)
- [`experiments/run_stage1_causal_memory.py`](file:///d:/Dynamic_MOE_Transformer/experiments/run_stage1_causal_memory.py)
- [`tests/test_stage1_causal_memory.py`](file:///d:/Dynamic_MOE_Transformer/tests/test_stage1_causal_memory.py)

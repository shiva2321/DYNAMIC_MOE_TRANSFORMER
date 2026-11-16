# Stage 1 Pre-Registration Revision 3 (`stage1_causal_memory_v3`)

**Pre-registered BEFORE editing `dmoe/memory.py` or running any `v3` tuning or test seeds (`research-protocol.md §2.2, §4.1`).**
Both `v1` (`PREREG.md`, `tuning_log.json`) and `v2` (`PREREG_v2.md`, `tuning_log_v2.json`, `results_v2.json`) remain untouched on disk as permanent audit records.

---

## 1. Honest Diagnosis of `stage1_causal_memory_v2` Failures (20 Test Seeds)

Under frozen `stage1_causal_memory_v2`, **`6/8` pre-registered falsifiers passed** (`F1`, `F2`, `F4`, `F5`, `F6`, `F8`) and **`2/8` failed** (`F3`, `F7`):

1. **`F3_parity_beats_nosigned_and_denserope: false` (1/20 Saddle-Point Outlier on `signed_beta`)**:
   - In `results_v2.json`, `Cog_Mem_Hier` achieved `1b_ood64_acc mean = 0.972751` (vs `Cog_NoSignedBeta = 0.345297`, `DenseRoPE_Baseline = 0.288486`), with **19/20 seeds in `[0.990173, 1.000000]`**, but **seed 6** stalled at `1b_id_acc = 0.564258` / `1b_ood64_acc = 0.509644`, failing the strict worst-seed gate `min >= 0.75`.
   - The same `~5%` failure mode appeared in `Cog_Mem_Flat` (19/20 seeds `>= 0.860107`, index 0 = `0.501465`).
   - **Root cause in `dmoe/memory.py`**: Uniform initialization `w_beta.bias = -1.1` starts all `num_heads=4` delta heads identically at `beta = 2 * sigmoid(-1.1) = 0.4995` (strictly positive transition eigenvalue `+0.50`). When random head projections do not push at least one head past `beta = 1.0` early during the `350`-step budget, the model converges onto the positive-eigenvalue local basin (`~0.56` ID, matching `Cog_NoSignedBeta = 0.558`).
   - **v3 Architectural Fix (`0` extra parameters)**: Replace uniform `w_beta.bias = -1.1` when `signed_beta=True` (`num_heads >= 2`) with a deterministic multi-timescale head span `torch.linspace(-1.5, 0.3, num_heads)` (`biases = [-1.5, -0.9, -0.3, +0.3]`, initial `beta ≈ [0.365, 0.578, 0.851, 1.149]`, initial eigenvalues `[+0.635, +0.422, +0.149, -0.149]`). Three heads remain gentle/associative (`beta < 1`) for MQAR retention while one head starts inside the negative-eigenvalue `(1, 2)` half-space from step 0. **The `F3` falsifier criteria remain unchanged (`mean >= 0.90`, `min >= 0.75`, `diff >= +0.15`).**

2. **`F7_null_controls: false` (`Null_ShuffledTargets` Infeasible Constant Bound Above Corpus Entropy)**:
   - `PREREG.md` / `PREREG_v2.md` required `Null_ShuffledTargets["1d_bpb_64"]["min"] >= 5.0`. In `results_v2.json`, `Null_ShuffledTargets` measured `mean = 4.948937`, `std = 0.004609`, `min = 4.941319`, `max = 4.959148`.
   - **Exact information-theoretic reason**: Computing empirical byte Shannon entropy directly on the corpus splits yields `TRAIN_CODE_FILES = 4.539335 BPB`, `VAL_CODE_FILES = 4.834321 BPB`, and `TEST_CODE_FILES = 4.871295 BPB` (cross-entropy `H(P_test, P_train) ≈ 4.95 BPB`). Under sequence-permuted next-byte targets (`y = y[perm]`), sequential dependence is eliminated and the Bayes-optimal predictor is the empirical marginal unigram distribution (`~4.95 BPB`). Thus `>= 5.0` was set above the theoretical cross-entropy floor of ASCII Python code.
   - **v3 Pre-Registered Calibration (Disclosed Explicitly Against `v2`)**: Set `F7` `1d_bpb_64["min"] >= 4.5` for `Null_ShuffledTargets` (matching `Null_BlindZeroInput >= 4.5` and bounded safely below `H(P_test) = 4.871 BPB`) **AND** add an explicit worst-case separation floor: `Null_ShuffledTargets["1d_bpb_64"]["min"] - Cog_Mem_Hier["1d_bpb_64"]["max"] >= 1.5` (null control must be at least `1.5` bits/byte worse than the worst `Cog_Mem_Hier` run).

---

## 2. Frozen `v3` Protocol, Budgets, Controls & Falsifiers

- **Unchanged from `PREREG_v2.md`**: All model topologies (`Cog_Mem_Hier`, `Cog_Mem_Flat`, `Cog_NoSignedBeta`, `Cog_NoSTM_Hier`, `DenseRoPE_Baseline`, `GRU_Baseline`), all 3 null controls (`Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`), parameter counts (~151k–162k), tasks `1A–1D`, file splits, vocabularies, training steps (`700 / 350 / 350 / 300`), LR schedule (`40`-step linear warmup + cosine), causality gate `F8` (`<= 1e-5`), dev seeds `[100..104]`, and test seeds `[0..19]`.
- **Tuning budget (`--mode tune`)**: Up to `2` configs (`lr in {2e-3, 3.5e-3}` on `Cog_Mem_Hier, Cog_Mem_Flat, Cog_NoSignedBeta`) on dev seeds `[100..104]` and `VAL_CODE_FILES` only, logged to `results/stage1_causal_memory/tuning_log_v3.json`.
- **Frozen test run (`--mode test`)**: Executed ONCE on all 8 models/controls across `TEST_SEEDS = [0..19]` at the dev-selected LR (`lr_cog=3.5e-3`, `lr_dense=3.5e-3`, `lr_gru=3.5e-3`), saved to `results/stage1_causal_memory/results_v3.json` alongside full `experiment_id`, UTC timestamp, git commit/dirty metadata, per-seed values, and paired t-tests.

### Stage 1 Lock Condition (`F1–F8` Must All Pass on `results_v3.json`)
- **F1–F6, F8**: Identical to `PREREG_v2.md` (`F1–F3` unrelaxed).
- **F7 (Null Controls Fail Cleanly + Separation Margin)**:
  - `Null_Untrained`: `1a_id_acc max <= 0.08`, `1b_id_acc max <= 0.35`, `1d_bpb_64 min >= 6.5`.
  - `Null_ShuffledTargets`: `1a_id_acc max <= 0.08`, `1b_id_acc max <= 0.35`, `1d_bpb_64 min >= 4.5`, AND `(Null_ShuffledTargets["1d_bpb_64"]["min"] - Cog_Mem_Hier["1d_bpb_64"]["max"]) >= 1.5`.
  - `Null_BlindZeroInput`: `1a_id_acc max <= 0.08`, `1b_id_acc max <= 0.35`, `1d_bpb_64 min >= 4.5`.


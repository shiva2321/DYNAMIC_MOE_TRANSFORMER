# Pre-Registration: Stage 1 Causal Memory & Pure-PyTorch Kernel Verification (`stage1_causal_memory_v8`)

- **Experiment ID**: `stage1_causal_memory_v8`
- **Predecessors**:
  - `stage1_causal_memory_v6` (`results_v6.json`: 8/8 falsifiers passed on `TEST_SEEDS = [0..19]` at `50.74 ms/step`, `steps_1b = 350`, unfused `AdamW`; preserved untouched alongside `v1`–`v5`).
  - `stage1_causal_memory_v7` (`results_v7.json`: pure-PyTorch kernel optimizations + CUDA `fused=True` `AdamW` achieved `26.42 ms/step` (`1.92x` speedup) and passed 7/8 falsifiers, failing `F3_parity_beats_nosigned_and_denserope` on `seed=9` (`1b_ood64_acc = 0.6829 < 0.75`) while the other 19/20 seeds scored `0.9836..1.0000`, `mean = 0.9812`; preserved untouched).
- **Date**: 2026-11-15
- **Hypothesis**: Increasing the Task 1B parity optimization budget uniformly across all models from `steps_1b = 350` to `steps_1b = 500` eliminates the late-grokking cutoff tail on `Cog_Mem_Hier` (passing all 8 Stage 1 falsifiers `F1`–`F8` on frozen `TEST_SEEDS = [0..19]`), while structural negative controls (`Cog_NoSignedBeta`, `DenseRoPE_Baseline`, `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`) remain unable to solve OOD parity and the pure-PyTorch kernel optimizations maintain the ~1.9x training speedup over `v6`.

## Protocol Change vs `v7`
1. `steps_1b`: changed symmetrically from `350` to `500` for **all** models, ablations, baselines, and null controls (`experiments/run_stage1_causal_memory.py` default `--steps_1b=500`).
2. `dmoe/` architecture & kernels (`dmoe/memory.py`, `dmoe/hierarchical_attention.py`, `dmoe/real_model.py`, `157,649` params): **zero changes** vs `v7`.
3. Dev checks prior to freezing `v8`:
   - `DEV 100..104` `Cog_Mem_Hier` at `steps_1b = 500`: `1b_ood64_acc = [0.9998, 0.9996, 1.0000, 0.9995, 0.9995]`.
   - `DEV 100..104` `Cog_NoSignedBeta` at `steps_1b = 450`: `mean = 0.3613` (`max = 0.3834`).
   - `DEV 100..104` `DenseRoPE_Baseline` at `steps_1b = 450`: `mean = 0.2941` (`max = 0.3019`).
   - `DEV 100..101` `Null_ShuffledTargets` at `steps_1b = 500`: `[0.2439, 0.2515]`; `GRU_Baseline`: `[1.0000, 1.0000]`.

## Frozen Hyperparameters & Splits
- Learning rates carried unchanged from `v6`/`v7` (no test-set LR re-tuning):
  - `lr_cog = 3.5e-3`, `lr_dense = 4.5e-3`, `lr_gru = 4.5e-3`
- Step budgets:
  - `steps_1a = 700`, `steps_1b = 500`, `steps_1c = 350`, `steps_1d = 300`, `min_nll = 0.7`
- Corpus & Seed Splits (unchanged):
  - `TRAIN_CODE_FILES`, `VAL_CODE_FILES`, `TEST_CODE_FILES` disjoint; `TEST_SEEDS = [0..19]`.

## Models & Null Controls (`TEST_SEEDS = [0..19]`)
- Architecture: `Cog_Mem_Hier`
- Ablations: `Cog_Mem_Flat`, `Cog_NoSignedBeta`, `Cog_NoSTM_Hier`
- Parameter-matched Baselines (~158k–162k params): `DenseRoPE_Baseline`, `GRU_Baseline`
- Null Controls: `Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput` (evaluated on `Cog_Mem_Hier` per-seed weights)

## Falsifiers (`F1`–`F8`, Unchanged from `v6`/`v7`)
1. `F1_mqar_acc_and_stability`: `Cog_Mem_Hier` `1a_id_acc.mean >= 0.90`, `min >= 0.75`, `std <= 0.08`, and `1a_ood64_acc.mean >= 0.85`, `min >= 0.65`.
2. `F2_mqar_beats_nostm_and_gru`: `Cog_Mem_Hier` `1a_ood64_acc.mean` beats `Cog_NoSTM_Hier` by `>= 0.40` and `GRU_Baseline` by `>= 0.10`.
3. `F3_parity_beats_nosigned_and_denserope`: `Cog_Mem_Hier` `1b_ood64_acc.mean >= 0.90`, `min >= 0.75`, and beats `Cog_NoSignedBeta` and `DenseRoPE_Baseline` by `>= 0.15` each.
4. `F4_ltm_survival`: `Cog_Mem_Hier` `1c_ltm_recall.mean >= 0.75`, `min >= 0.50`, and beats `DenseRoPE_Baseline` and `GRU_Baseline` by `>= 0.65` each.
5. `F5_non_interference`: `Cog_Mem_Hier` `1c_ni_drop.mean <= 0.03` and `max <= 0.08`.
6. `F6_real_code_bpb`: `Cog_Mem_Hier` `1d_bpb_64.mean <= DenseRoPE_Baseline 1d_bpb_64.mean + 0.05`.
7. `F7_null_controls`: `Null_Untrained` (`1a_id.max <= 0.08`, `1b_id.max <= 0.35`, `1d_bpb_64.min >= 6.5`), `Null_ShuffledTargets` (`1a_id.max <= 0.08`, `1b_id.max <= 0.35`, `1d_bpb_64.min >= 4.5`, and `Null_ShuffledTargets 1d_bpb_64.min - Cog_Mem_Hier 1d_bpb_64.max >= 1.5`), and `Null_BlindZeroInput` (`1a_id.max <= 0.08`, `1b_id.max <= 0.35`, `1d_bpb_64.min >= 4.5`).
8. `F8_causality_gate`: Future-token perturbation max abs logit diff `<= 1e-5` across all 6 trained/ablation architectures.

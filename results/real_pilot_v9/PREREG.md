# Pre-Registration: `real_pilot_v9` — CognitiveDMoE v12 Breakthrough Evaluation

**Date:** 2026-11-25  
**Experiment ID:** `real_pilot_v9`  
**Branch:** `v5-arch-efficiency`  
**Base Commit:** `5ce3a36` (`CognitiveDMoE` v11 / `real_pilot_v8`)  
**Status:** FROZEN BEFORE ANY `real_pilot_v9` RUN.

---

## 1. Claims, Headline Metrics & Falsifiers

All pass criteria are absolute and fixed in advance (no `PASS if baseline <= x`). Positive results on 3–5 seeds are capped at claim level `OBSERVED` per `research-protocol.md` §6 and §9; `REFUTED` applies immediately if any falsifier triggers.

### C1a — Reliable Multi-Seed Grokking + Burst Immunity (StableMax + $\perp\text{Grad}$, Zero Grokfast)
- **Claim:** Replacing standard CE + Grokfast with FP32 `stablemax_cross_entropy` + $\perp\text{Grad}$ (`ortho_grad=True`) produces reliable multi-seed modular-belief grokking and causal IIT generalization without regressing single-source adversarial burst immunity.
- **Headline metrics (Track 1, 5 disjoint test seeds `0..4`, higher is better):**
  - `heldout_unseen_acc >= 0.70` (with at least 4 of 5 seeds `>= 0.60`, and `seen_acc >= 0.95`)
  - `iit_heldout_acc >= 0.65` (with `iit_random_slice_null <= 0.30`)
  - `burst_immunity_acc >= 0.90`
  - `near_ood_abstain_auroc >= 0.90`
- **Falsifier:** `heldout_unseen_acc < 0.70` OR `burst_immunity_acc < 0.90` OR fewer than 4/5 seeds grok (`>= 0.60`) OR `iit_random_slice_null > 0.30`.

### C1b — Zero-Weight-Update Unseen-Meaning Cyclic Rule Shift (Multi-Plane $SO(2)^p$ Orthogonal Procrustes)
- **Claim:** Replacing the unconstrained 12D Ridge head with a Multi-Plane Complex $SO(2)^p$ Orthogonal Procrustes operator + Rayleigh coherence arbitration enables zero-weight-update deduction of secret cyclic rule shifts on strictly unseen meanings (`unseen_meaning_shift_acc`) while preserving compositional equivalence induction (`comp_equiv`) and keeping all 4 null controls at chance.
- **Headline metrics (Track 1, 5 disjoint test seeds `0..4`, zero weight updates at eval):**
  - `unseen_meaning_shift_acc @ 8sh >= 0.50` AND `@ 16sh >= 0.55` (chance `0.0625`)
  - `comp_equiv_induction_acc @ 8sh >= 0.65` (chance `0.0625`)
  - `lookup_acc @ 8sh >= 0.65`
  - All 4 null controls (`shuffled_null`, `impossible_perm_null`, `shift_shuffled_null`, `shift_no_mem_blind`) `<= 0.15` across all shot counts `(4, 8, 16)`.
  - Ablation superiority: `CognitiveDMoE_v12` exceeds `CognitiveDMoE_v12_NoProcrustes` by `>= +0.25` on `unseen_meaning_shift_acc @ 8sh`.
- **Falsifier:** `unseen_meaning_shift_acc @ 8sh < 0.50` OR `comp_equiv @ 8sh < 0.65` OR any null control `> 0.15` OR ablation gap `< +0.25`.

### C2a — Track 2 Continual Accuracy & Wall-Clock Efficiency Superiority Over `Dense_Matched`
- **Claim:** Zero-sync vectorized masked-GEMM MoE dispatch (`E <= 8`) + active-parameter capacity parity (`d_ff_shared=512, d_ff_spec=256, shared_dropout_prob=0.0`) + H-Net causal boundary-weighted byte chunking enables `CognitiveDMoE_v12` to outperform `Dense_Matched` on continual Track 2 tasks at matched active compute (`mean_k <= 1.00`) and competitive wall-clock speed.
- **Headline metrics (Track 2, 3 disjoint test seeds `0..2`):**
  - `b77_intent_77_acc >= 0.60` AND `>= Dense_Matched`
  - `mean_final_r5 >= 0.21` AND `>= Dense_Matched + 0.02`
  - `heldout_families_auc >= 0.48`
  - `mean_nuance_flip_auc >= 0.78`
  - `mean_k <= 1.00`
  - `train_seconds <= 280.0` (down from v11 `354.1s`)
- **Falsifier:** `train_seconds > 280.0` OR `mean_k > 1.00` OR `b77_intent_77_acc < Dense_Matched` OR `mean_final_r5 < 0.21` OR `heldout_families_auc < 0.48`.

### C2b — Shift-Resilient Non-Collapsing Semantic Functional Routing
- **Claim:** Dual-granularity H-Net byte-summary + Layer-3 semantic routing trained with full-batch Symmetric Perturbation InfoNCE (`NT-Xent`) + SIGReg dispersion + continual router anchor outperforms `CognitiveDMoE_v12_FrozenRandomRouter` (identical front-end and trunk) on BOTH length-matched functional role AMI and surface-perturbation agreement.
- **Headline metrics (Track 2, 3 disjoint test seeds `0..2`):**
  - `length_matched_role_ami >= 0.15` AND `>= FrozenRandomRouter + 0.03` (with `50-perm null <= 0.03`)
  - `routing_perturb_agree >= 0.65` AND `>= FrozenRandomRouter + 0.08`
  - `routing_length_ami <= 0.20`
  - `num_experts >= 3` (autonomous continual spawning works)
- **Falsifier:** `length_matched_role_ami < 0.15` OR `length_matched_role_ami < FrozenRandomRouter + 0.03` OR `routing_perturb_agree < 0.65` OR `routing_perturb_agree < FrozenRandomRouter + 0.08`.

### C3 — Core Invariants & Decoupled Generative Byte Renderer
- **Claim:** `0.0` decoder cross-entropy gradient into cognitive core (`core_weight_max_drift == 0.0`), decoupled decoder reduces byte NLL by `>= 2.0 nats` on frozen latents, vectorized dispatch matches sorted scatter-add to `< 1e-4`, and all Track 1/Track 2 causality + quarantine (`<= 16 KB`) invariants pass unit tests.
- **Falsifier:** `core_weight_max_drift > 0.0` OR decoder NLL drop `< 2.0` OR any invariant test fails.

---

## 2. Frozen Protocol, Arms & Baselines

### Hardware & Precision
- Single NVIDIA GeForce RTX 3060 (12 GB), Windows 11, PyTorch 2.6.0+cu124, `bf16` autocast (with FP32 loss/Procrustes/routing numerics).

### Arms (all `teacher_informed = False`, all run in the same script `experiments/run_v12_eval.py`)
- **Track 1 Arms (5 disjoint test seeds `0, 1, 2, 3, 4`; `n_p1=1300, n_pb=20, n_p2=180, n_p3=180`):**
  1. `CognitiveDMoE_v12`: Full v12 (`StableMax + OrthoGrad`, Multi-Plane $SO(2)^p$ Procrustes + High-Rank Ridge, H-Net boundary chunker, vectorized MoE dispatch).
  2. `CognitiveDMoE_v12_NoProcrustes`: Identical v12 with Multi-Plane Procrustes head disabled (keeps High-Rank Ridge only — isolates `C1b`!).
  3. `CognitiveDMoE_v11`: Direct consolidated v11 anchor (standard CE + Grokfast + 12D unconstrained MLP Ridge — rerun in the same script for apples-to-apples verification).
  4. `Dense_Matched`: Exact active-parameter matched dense control (`d_ff_shared=512, d_ff_spec=256, max_experts=1`, H-Net chunker, **exact same `StableMax + OrthoGrad` optimizer & loss**).
- **Track 2 Arms (3 disjoint test seeds `0, 1, 2`; `i7_steps=150, steps_per_dom=250` across 4 domains `code, math, prose, banking77`):**
  1. `CognitiveDMoE_v12`: Full v12 (`d_ff_shared=512, d_ff_spec=256, shared_dropout_prob=0.0`, H-Net chunker, vectorized masked-GEMM dispatch, SimCLR InfoNCE + continual router anchor).
  2. `CognitiveDMoE_v12_FrozenRandomRouter`: Exact same v12 front-end, H-Net chunker, trunk, capacity, and vectorized dispatch, with `context_router` parameters frozen (isolates learned semantic routing `C2b`!).
  3. `CognitiveDMoE_v11`: Direct consolidated v11 anchor (`d_ff_shared=384, shared_dropout_prob=0.15`, original dispatch & CRIS).
  4. `Dense_Matched`: Exact active-parameter matched dense control (`d_ff_shared=512, d_ff_spec=256, max_experts=1`, same H-Net chunker, same `MuonBONGOptimizer` + GPM).

### Mandatory Null Controls
- Track 1: `iit_random_slice_null`, `shuffled_null_acc`, `impossible_perm_null_acc`, `shift_shuffled_null_acc`, `shift_no_mem_blind_acc` across `4, 8, 16` shots.
- Track 2: 50-permutation AMI null for `func_routing_ami` and `length_matched_role_ami`, `routing_length_ami` shortcut audit, frozen-probe out-of-sample multi-layer expert lesioning, and `core_weight_max_drift == 0.0` check.

### Disjoint Dev Tuning Budget & Frozen Test Rule
- **Dev split:** Strictly `seed = 100` (`eval_split = "dev"`).
- **Budget:** Maximum **4 dev trials** on Track 1 and maximum **4 dev trials** on Track 2. Every trial and its dev scores are logged automatically to `results/real_pilot_v9/tuning_log.json`.
- **Frozen test run:** Once the single winning v12 config is frozen from `seed = 100`, run ONE frozen evaluation on disjoint test seeds (`0..4` Track 1, `0..2` Track 2) with `--resume` support for crash safety (never overwriting or re-rolling completed seeds).
- **Environment disclosure:** `.integrity/` is absent in this workspace (`Test-Path .integrity` = `False`); all metadata (git commit, dirty flag, UTC timestamp, PyTorch/Python version, GPU, exact CLI command, full config) is recorded automatically in every JSON artifact.

---

## 3. Oracles & Disclosure
- Zero oracles on held-out evaluations. Track 1 `BeliefMeaningTask` paraphrases and `i7` procedural batches are declared synthetic training supervision shared identically across matched arms. Track 2 `Banking77` train/test splits and held-out nuance families are strictly disjoint. All arms have `teacher_informed = False`.


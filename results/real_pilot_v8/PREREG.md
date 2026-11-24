# Pre-Registration: `real_pilot_v8` — Consolidated `CognitiveDMoE` Architecture (v11)

**Date:** 2026-11-24  
**Status:** FROZEN BEFORE ANY DEV OR TEST RUN  
**Branch:** `v5-arch-efficiency`  
**Follows:** `real_pilot_v7` (`Cog_v10`, commit `164f787`)  
**Protocol:** Governed strictly by `TRUTHGATE-CONTRACT` and `research-protocol.md`

---

## 1. Objective & Claims

Consolidate all empirically validated mechanisms from `v1`–`v10` into one coherent, zero-dead-module, end-to-end working architecture (`CognitiveDMoE` in `dmoe/cognitive_dmoe.py`), prune empirically harmful/dead components (`rel_pool_proj`, wasteful nuance escalation, duplicate unused legacy encoders/routers), and test four root-cause fixes targeting the two pre-registered failures of `Cog_v10`:

1. **Shift-Resilient Causal Anti-Aliased Multi-Scale Byte Chunker** (`shift_invariant_pool=True` in `dmoe/chunker.py`): causal BlurPool low-pass anti-aliasing before stride-4 downsampling + content-masked mean residual to prevent 1–3 leading space boundary shifts from scrambling chunk features.
2. **Continual-Safe Orthogonal-Residual Semantic Router** (`dmoe/routing.py`):
   - Exempts frozen experts from `expert_bias` load-balancing pressure (`frozen_exempt_balance=True`, zeroing `expert_bias[e]` on freeze) so continual domains do not overwrite specialist routing.
   - Fixes biased-margin `k=2` leakage when `delta_mix <= 0.0`, guaranteeing `mean_k <= 1.00`.
   - Adds content-masked causal EMA + frozen semi-orthogonal base projection with small learnable residual (`residual_ortho_router=True`) + clean CRIS (equiv + perturbation invariance + hard-neg repulsion, without unlabeled `-1` noise).
3. **Upgraded Dual-Head Closed-Form Ridge Episodic Operator** (`dmoe/memory.py`): `op_rank=48` high-rank nonlinear head + `op_rank_low=12` linear-key / 2-layer-MLP-value group head with detached support value chunks `raw_op_v`.
4. **Reliable Multi-Seed Grokking Acquisition** (`MuonBONGOptimizer` with `Grokfast-EMA` + stronger `wd_trunk` in `CognitiveTrainer`): eliminates Track 1 bimodal grokking across seeds so the Dual-Head Ridge linear rotation head receives a reliably structured circular manifold.

### Formal Claims & Pre-Registered Falsifiers (Absolute & Fixed)

| Claim ID | One-Sentence Claim | Headline Task Metric(s) & Direction | Pre-Registered Pass Bar | Falsifier (What Refutes the Claim) |
|---|---|---|---|---|
| **C1a** | `CognitiveDMoE` groks Track 1 modular composition reliably across seeds while maintaining causal localization and burst immunity. | Track 1 `heldout_unseen_acc` (↑), `iit_heldout_acc` (↑), `burst_immunity_acc` (↑), `near_ood_abstain_auroc` (↑) | 5-seed mean `heldout_unseen_acc >= 0.70` (>=4/5 seeds `>= 0.60`), `iit_heldout_acc >= 0.65` vs `iit_random_slice_null <= 0.35`, `burst_immunity_acc >= 0.90`, `near_ood_abstain_auroc >= 0.95` | **F1a:** 5-seed mean `heldout_unseen_acc < 0.70` OR `< 4/5` seeds `>= 0.60` OR `iit_heldout_acc < 0.65` OR `burst_immunity_acc < 0.90`. |
| **C1b** | `CognitiveDMoE` performs zero-weight-update episodic rule induction on both unseen compositional pairs (`comp_equiv`) and unseen meanings under novel cyclic shifts (`unseen_meaning_shift`). | `comp_equiv_induction_acc` (↑), `unseen_meaning_shift_acc` (↑) at 8 & 16 shots (zero weight updates) | 5-seed mean `comp_equiv @ 8sh >= 0.65`, `unseen_meaning_shift @ 8sh >= 0.50`, `unseen_meaning_shift @ 16sh >= 0.55`; all 4 episodic null controls `<= 0.15`; strictly beats `CognitiveDMoE_NoDualOp`, `Cog_v10`, and `Dense_Matched` | **F1b:** `comp_equiv @ 8sh < 0.65` OR `unseen_meaning_shift @ 8sh < 0.50` OR `unseen_meaning_shift @ 16sh < 0.55` OR any null control `> 0.15`. |
| **C2a** | `CognitiveDMoE` runs sparse continual MoE within strict compute/time budgets with real active specialist utility and Banking77 intent parity. | `mean_k` (↓), `train_seconds` (↓), `num_experts` (↑), `lesion_mean_rel_delta` (↑), `b77_intent_77_acc` (↑) | 3-seed mean `mean_k <= 1.00`, `train_seconds <= 280s`, `num_experts >= 3.0`, `lesion_mean_rel_delta >= 0.015`, `b77_intent_77_acc >= 0.56` (within 0.01 of `Dense_Matched` and `> FrozenRandomRouter`) | **F2a:** `mean_k > 1.00` OR `train_seconds > 280s` OR `num_experts < 3.0` OR `b77_intent_77_acc < 0.55`. |
| **C2b** | `CognitiveDMoE` achieves shift-resilient semantic functional routing exceeding a matched frozen orthogonal router. | `routing_perturb_agree` (↑), `length_matched_role_ami` (↑), `routing_length_ami` (↓) | 3-seed mean `routing_perturb_agree >= 0.60`, `length_matched_role_ami >= 0.15` (exceeding `CognitiveDMoE_FrozenRandomRouter` by `>= +0.03`), and `routing_length_ami <= 0.16` (`< FrozenRandomRouter`) | **F2b:** `routing_perturb_agree < 0.60` OR `length_matched_role_ami < 0.15` OR fails to beat `CognitiveDMoE_FrozenRandomRouter` by `>= +0.03` on `length_matched_role_ami`. |
| **C3** | `CognitiveDMoE` operates end-to-end with zero dead trainable modules, `0.0` decoder cross-entropy gradient into the cognitive core, and working detached byte-level rendering/generation. | Unit/invariant assertions in `tests/test_cognitive_dmoe_consolidated.py` | All assertions pass after final edit: `core_grad_norm == 0.0` under decoder CE, detached decoder NLL drops on real UTF-8 text while trunk weight delta `== 0.0`, `generate()` emits valid UTF-8 bytes, 100% of core parameters actively participate in forward/backward. | **F3:** Any invariant or regression test fails. |

---

## 2. Frozen Experimental Protocol

- **Hardware:** Single NVIDIA GeForce RTX 3060 (12 GB), Windows 11, PyTorch 2.6.0+cu124.
- **Core Architectural Constraints (non-negotiable):**
  - `0.0` decoder cross-entropy gradient into the cognitive trunk (`detach_decoder=True`).
  - Tokenizer-free UTF-8 byte stream (`chunk_size = 4`, `d_model = 192`, `num_heads = 6`, `num_layers = 6`, `router_layer = 3`, `d_ff_shared = 384`, `d_ff_spec = 256`).
  - Always-plastic continual learning via `MuonBONGOptimizer` + `NullSpaceProjector` (GPM) + `LatentQuarantineGate` (`<= 16 KB`) + `AutonomousExpertLifecycleManager`.
  - Zero frozen pretrained external encoders (`teacher_informed = False`).
- **Track 1 (Modular Belief Composition, Causal IIT, Burst Immunity, 6-Probe Episodic Induction):**
  - Task: `BeliefMeaningTask` (`K_CONCEPTS = 16`, chance = `0.0625`), `n_p1 = 1300`, `n_pb = 20`, `n_p2 = 180`, `n_p3 = 180`, batch size `64` (+ paraphrase batch in `p1`).
  - Episodic eval: 64 episodes × `{4, 8, 16}` shots, zero weight updates (`model.eval()`, `torch.no_grad()`).
  - Arms (4):
    1. `CognitiveDMoE` (full consolidated v11 architecture)
    2. `CognitiveDMoE_NoDualOp` (identical consolidated architecture + matched Grokfast/wd + matched episodic training schedule, Dual Ridge OFF)
    3. `Cog_v10` (prior v10 reference run under the exact same runner/data)
    4. `Dense_Matched` (parameter-matched single-expert dense backbone with identical `MuonBONGOptimizer` + Grokfast + GPM + Quarantine)
  - Mandatory Null Controls: `shuffled_null_acc`, `impossible_perm_null_acc`, `shift_shuffled_null_acc`, `shift_no_mem_blind_acc`, `iit_random_slice_null`.
- **Track 2 (4-Domain Continual Real-Text Stream + Banking77 + Honest Semantic Routing & Lesion Suite):**
  - Domains: `code`, `math`, `prose`, `banking77` (250 steps/domain, `i7_steps = 150`, `seq_len = 64`).
  - Arms (4):
    1. `CognitiveDMoE` (full consolidated v11 architecture)
    2. `CognitiveDMoE_FrozenRandomRouter` (identical consolidated architecture + shift-invariant chunker, with `context_router` learnable projections frozen at orthogonal init)
    3. `Cog_v10` (prior v10 reference run under the exact same runner/data)
    4. `Dense_Matched` (single-expert dense backbone with matched `MuonBONGOptimizer` + GPM + Quarantine)
  - Mandatory Null Controls: 50-permutation AMI null, length-matched 48-byte role control, length-quartile spurious AMI, surface-formatting perturbation test (1–3 leading space shift + ASCII case swap + whitespace perturbation), stratified out-of-sample frozen-probe multi-layer expert lesioning.

---

## 3. Disjoint Splits, Tuning Budget & Claim Ceiling

- **Validation / Dev Tuning:** Strictly `seed = 100` (`eval_split = "dev"`). Maximum **4 dev trials** on Track 1 and **4 dev trials** on Track 2. Every trial logged to `results/real_pilot_v8/tuning_log.json`.
- **Frozen Test Evaluation:** Strictly disjoint test seeds executed ONCE after dev tuning freezes:
  - Track 1: `seeds = [0, 1, 2, 3, 4]` (`eval_split = "test"`) — 20 full GPU runs.
  - Track 2: `seeds = [0, 1, 2]` (`eval_split = "test"`) — 12 full GPU runs.
- **Oracles:** None in any arm (`teacher_informed = False`). Track 1 uses a standard supervised readout `cls_head` identically across all arms and controls; Track 2 Banking77 uses the frozen-feature ridge classifier identically across all arms and controls.
- **Claim Ladder Ceiling:** Because 20-seed sweeps are not executed in this cycle (budget ~2.5–3 hours total on single RTX 3060), all passing empirical claims are strictly capped at **`OBSERVED`** (`research-protocol.md` §9). Any failed bar is marked **`REFUTED`**.


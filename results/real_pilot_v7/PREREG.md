# Pre-Registration: Architecture v10 (`Cog_v10`) — Semantic Functional Routing & In-Context Rule Induction (`real_pilot_v7`)

**Date:** 2026-11-24  
**Branch:** `v5-arch-efficiency`  
**Repository:** `d:\Dynamic_MOE_Transformer`  
**Experiment ID:** `real_pilot_v7`  
**Status:** FROZEN prior to any dev or test execution  

---

## 1. Claims & Headline Metrics

### Claim C1 (In-Context Rule Induction via Closed-Form Dual Ridge Operator)
Equipping `LongTermSlotMemory` with a differentiable closed-form Dual Ridge in-context operator operating over strictly causal deep post-trunk representations enables **zero-weight-update in-context rule induction** on Track 1 (`BeliefMeaningTask`):
- **Headline Metric 1a (`unseen_meaning_shift_acc` @ 8 shots, higher is better):** Accuracy on strictly novel query pairs $(a_q, b_q)$ whose latent compositional meaning $m_q$ was **never observed in any support shot** ($m_q \notin \{m_i\}$), under a fresh episode-specific secret cyclic shift $\delta_{ep} \in \{0..15\}$.
- **Headline Metric 1b (`comp_equiv_induction_acc` @ 8 shots, higher is better):** Accuracy on strictly novel query pairs $(a_q, b_q) \notin \text{support}$ under an arbitrary fresh random output permutation $g_{ep}$, where $m_q$ matches a support shot's latent meaning.
- **Success Bar C1:** On frozen test seeds, `Cog_v10` achieves `unseen_meaning_shift_acc @ 8 shots >= 0.50` (vs chance `0.0625`), outperforms `Cog_v10_NoDualOp` by `>= +0.25`, outperforms `Cog_v9` by `>= +0.25`, while keeping `lookup_acc >= 0.80`, `shuffled_null_acc <= 0.15`, `impossible_perm_null_acc <= 0.15`, `heldout_unseen_acc >= 0.70`, and `iit_heldout_acc >= 0.70`.
- **Falsifier F1:** Claim C1 is **REFUTED** if `Cog_v10` fails to reach `unseen_meaning_shift_acc @ 8 shots >= 0.50`, OR if `Cog_v10_NoDualOp` matches `Cog_v10` within `0.10`, OR if `shuffled_null_acc > 0.15` or `impossible_perm_null_acc > 0.15` (indicating leakage), OR if Track 1 `heldout_unseen_acc` drops below `0.70`.

### Claim C2 (Semantic Functional Routing via Surface Null-Space Whitening + CRIS Loss + Lifecycle/Optimizer Fix)
Fixing the `Cog_v9` lifecycle/optimizer bugs (decoupling lifecycle novelty detection from `k=2` escalation, and rebuilding `MuonBONGOptimizer` state on expert spawn) together with Surface Null-Space Whitening ($P_{\text{surf}}^{\perp}$) and Causal Role-Intervention Specialization (`CRIS`: routing equivalence invariance + permutation-invariant expert subspace orthogonality + specialist credit dropout) produces **learned, formatting-invariant, causally active functional routing** on Track 2 without degrading Banking77 accuracy or sparse compute efficiency:
- **Headline Metric 2a (`length_matched_role_ami`, higher is better):** Adjusted Mutual Information on the 6 held-out functional roles when all sequences are truncated/padded to identical 48-byte content length (eliminating length shortcuts).
- **Headline Metric 2b (`routing_perturb_agree`, higher is better):** Fraction of held-out sequences assigned to the exact same expert under meaning-preserving surface formatting perturbations (case/whitespace/padding jitter).
- **Headline Metric 2c (`routing_length_ami`, lower is better):** Spurious AMI between assigned experts and sequence-length quartile bins.
- **Headline Metric 2d (`frozen_probe_lesion_max_drop` & `lesion_mean_rel_delta`, higher is better):** Out-of-sample accuracy drop on a frozen clean-trained linear probe and relative latent shift $\|h_{\text{base}} - h_{\text{les}(e)}\| / \|h_{\text{base}}\|$ when individual spawned experts are zeroed out.
- **Success Bar C2:** On frozen test seeds, `Cog_v10` spawns `>= 2` active experts, achieves `length_matched_role_ami > Cog_v10_FrozenRandomRouter + 0.10`, achieves `routing_perturb_agree >= 0.75` and `> Cog_v10_NoSIP`, reduces `routing_length_ami` relative to `Cog_v10_NoSIP`, achieves `lesion_mean_rel_delta >= 0.05` across spawned experts (vs `0.000` untrained experts in `Cog_v9` runner), while maintaining `mean_k <= 1.0`, `Banking77 intent_77_acc >= 0.55`, and `train_seconds <= 280s` (with `0.0` decoder CE gradient into trunk).
- **Falsifier F2:** Claim C2 is **REFUTED** if `Cog_v10_FrozenRandomRouter` matches `Cog_v10` on `length_matched_role_ami` (within `0.05`), OR if `Cog_v10_NoSIP` matches or beats `Cog_v10` on both `length_matched_role_ami` and `routing_perturb_agree`, OR if spawned experts show zero causal lesion impact (`lesion_mean_rel_delta < 0.01`), OR if `mean_k > 1.05` or `Banking77 intent_77_acc < 0.52`.

---

## 2. Frozen Experimental Protocol

### 2.1 Non-Token & Integrity Guardrails (Hard Invariants)
- **Zero Decoder CE Gradient:** `detach_decoder=True` in all cognitive arms; unit-tested to verify literally `0.0` gradient norm from decoder CE into trunk, router, and memory parameters.
- **Zero Test-Set Leakage:** All tuning happens strictly on `eval_split="dev"` and `seed=100`. Frozen test evaluation uses `eval_split="test"` and disjoint test seeds.
- **Hardware:** Single NVIDIA GeForce RTX 3060 (12 GB), Windows 11, PyTorch CUDA bfloat16 autocast (with float32 Dual Ridge linear solve for numerical stability).

### 2.2 Track 1 Protocol (`BeliefMeaningTask`, Modular Arithmetic & Episodic Induction)
- **Arms:**
  1. `Cog_v10`: Full v10 (v9 core + Dual Ridge Episodic Operator + SIP-Router + decoupled lifecycle).
  2. `Cog_v10_NoDualOp`: Exact ablation of `Cog_v10` with `enable_dual_ridge_ep=False` (same episodic training schedule, KV slot lookup only).
  3. `Cog_v9`: Re-run v9 baseline in the same script on identical seeds.
  4. `Dense_v8_Matched`: Matched-optimizer (`MuonBONGOptimizer`, GPM, QuarantineGate) single-expert dense control.
- **Training Budget:** `n_p1 = 1300`, `n_pb = 20`, `n_p2 = 180`, `n_p3 = 180` (identical to `real_pilot_v6`).
- **Episodic Training Exposure (Declared):** During `p1` (identical cadence `step_i % 3 == 1` after `n_p1 // 5`), both `Cog_v10` and `Cog_v10_NoDualOp` train on episodic batches covering lookup, compositional equivalence, and shift-rule episodes (using `n_shots=8` during episodic training so both arms receive identical episodic data; zero eval-split overlap since every episode draws a fresh secret rule/permutation).
- **Disjoint Test Seeds:** `[0, 1, 2, 3, 4]` (5 independent seeds; claim tier capped at `OBSERVED` unless extended to 20 seeds).
- **Evaluation Size:** 64 independent episodes per shot count (`4, 8, 16`) for all 5 episodic probes; 128 samples per query type (`Q_VAL, Q_ATTR, Q_PROP`) for seen/held-out/paraphrase; 64 pairs for IIT seen/held-out/random-slice.

### 2.3 Track 2 Protocol (`V8RealBenchmark`, Continual Multi-Domain Semantic Transfer)
- **Arms:**
  1. `Cog_v10`: Full v10 (SIP-Router surface whitening + CRIS loss + decoupled lifecycle + optimizer rebuild on spawn + healthy specialist init + shared dropout).
  2. `Cog_v10_NoSIP`: Exact ablation of `Cog_v10` with `enable_surface_whitening=False`, `enable_cris_loss=False`, `shared_dropout_prob=0.0` (keeps lifecycle decoupling & optimizer rebuild so SIP/CRIS is isolated cleanly!).
  3. `Cog_v10_FrozenRandomRouter`: Mandatory null control (`context_router` parameters frozen at random init; same trunk, same optimizer, same decoupled lifecycle & optimizer rebuild).
  4. `Cog_v9`: Exact v9 baseline re-run in the same script.
  5. `Dense_v8_Matched`: Matched-optimizer dense control.
- **Training Budget:** `i7_steps = 150`, `steps_per_dom = 250` across all 5 continual domains (`wiki, story, code, math, forum`), identical to `real_pilot_v6`.
- **Disjoint Test Seeds:** `[0, 1, 2]` (3 independent seeds; claim tier capped at `OBSERVED` due to ~3.5–4.5 min/arm/seed GPU runtime on single RTX 3060).

### 2.4 Oracles & Supervision Declared
- Track 1 uses supervised classification labels `y` on `BeliefMeaningTask` training splits (`p1, pb, p2, p3`) and episodic support shot answers `Yxx` strictly inside support shots (zero answer tokens in query sequences).
- Track 2 uses **zero token cross-entropy** into the trunk; it uses the identical self-supervised/weakly-paired batch signals (`equiv_labels`, `contra_pair_idx`, `hard_neg_pair_idx`, `nuance_targets`, `struct_targets`) already present in `V8RealBenchmark.sample_v8_stream_batch` for all arms, with `teacher_informed=False`.
- Surface descriptors (`compute_chunk_surface_features`) are deterministic statistics computed purely from raw input bytes `x`, with zero access to any task or role label.

### 2.5 Tuning Budget
- Maximum **4 dev trials** on `eval_split="dev"` and `seed=100` (disjoint from test seeds `0..4`).
- Every dev configuration and score is logged to `results/real_pilot_v7/tuning_log.json`.
- After dev verification, all hyperparameters are frozen and the single test run executes once on test seeds.

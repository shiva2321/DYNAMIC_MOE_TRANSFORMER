# Pre-Registration: Experimental Test of Three Brain-Inspired Belief & Meaning Training Algorithms (`belief_training_lab_v1`)

- **Experiment ID**: `belief_training_lab_v1`
- **Status**: DRAFT (pre-registered before any code execution or DEV run). Sections marked **[DEV]** will be filled strictly from `DEV_SEEDS = [100..104]` (`tuning_log_v1.json`) and then frozen before touching `TEST_SEEDS = [0..19]`.
- **Implementation**: `dmoe/belief_training.py`, `tests/test_belief_training.py`, `experiments/run_belief_training_lab.py` -> `results/belief_training_lab/results_v1.json`.

---

## 1. Claims Under Test (Tested Independently, Not Bundled)

We evaluate three distinct training-algorithm ideas and their combination on a plastic, self-trained (no frozen pretrained encoder, no domain header tokens) continual belief & meaning stream:

1. **Claim 1 (Idea 1 — Concept-Keyed Wake–Chunk–Sleep Triage, `Idea1_ConceptWCS`)**: Keying a Bayesian belief store by learned sparse concept codes extracted from a plastic encoder, combined with 3-tier wake triage (count confirmations without backprop, split context-conditional exceptions instead of overwriting, flip laws + demote stale dependent entries via Mechanism P), RHO-style reducible-error chunked updates, and sleep consolidation, preserves both old and new context-split attributes (`apple -> red` & `green`) and resists single-source noise bursts while spending fewer post-Phase-1 training FLOPs and backward passes than full per-step replay (`B2_Dense_Replay`).
2. **Claim 2 (Idea 2 — Equivalence-and-Contradiction Learning, `Idea2_EquivContra`)**: Training a plastic representation with cross-view behavioral equivalence attraction (`L_equiv` pulling surface expressions with the same canonical outcome together) stabilized by `SIGReg` (Epps–Pulley isotropic Gaussian regularization), plus contradiction-driven context separation, enables **zero-retrain equivalence transfer** in Phase 3 (`rev_equiv_transfer_acc`: updating a downstream property on `op=0` transfers to unseen `op in {1,2,3}`), and beats the mandatory shuffled-equivalence control (`Idea2_ShuffledEquiv`, testing against `X-47`).
3. **Claim 3 (Idea 3 — Inference-Before-Plasticity + Precision-Bearing Synapses, `Idea3_MuPC_BONG_Pure` & `Idea3_Backbone_BONG`)**:
   - **(3a)** A strictly `.backward()`-free predictive-coding network with prospective activity equilibration ($\mu$PC-style) and BONG diagonal precision updates learns the task above chance and beats plain Adam on the same $\mu$PC architecture (`Idea3_MuPC_Adam`), evaluated honestly on both accuracy/NLL and measured wall-clock/FLOP cost.
   - **(3b)** Equipping the shared Transformer backbone with BONG diagonal precision + Continual Backprop unit recycling + reducible-error surprise skipping (`Idea3_Backbone_BONG`) improves retention/burst resistance over `B1_Dense_SGD`.
4. **Claim 4 (Triad Combination, `Idea_Triad_All3`)**: Combining Idea 2's equivalence-trained concept representation + Idea 1's Wake–Chunk–Sleep context-splitting triage + Idea 3's BONG precision & utility recycling matches or exceeds the best single idea across split-attribute retention, equivalence transfer, burst immunity, NLL calibration, and post-Phase-1 compute.

**Headline Task Metrics:**
- `p2_split_both_min` = per-seed `min(p2_split_new_acc, p2_split_old_preserved_acc)` (for Claim 1: red & green coexistence)
- `p3_rev_equiv_transfer_acc` (for Claim 2: zero-retrain transfer across surface phrasings)
- `p3_overall_acc` & `p3_overall_nll` (overall balanced accuracy & calibration across all roles, concepts, phrasings, and contexts)
- `flops_post_p1`, `backprop_calls_post_p1`, and `wall_clock_sec` (true compute & speed accounting)

---

## 2. Benchmark: `BeliefMeaningStream` (Zero-Header, Plastic-Encoder Only)

For each seed `s`:
- **Canonical concepts**: `K = 16` meanings `m in {0..15}`.
- **Input sequence (`L = 6`)**: `x = [op_tok, a_tok, b_tok, c1_tok, c2_tok, q_tok]` over shared vocabulary `vocab_size = 24`:
  - `op_tok in {0, 1, 2, 3}`: 4 surface expression families (`E0..E3`).
  - `a_tok, b_tok, c1_tok, c2_tok in {4..19}`: 16 raw symbols (`idx = tok - 4`).
  - `q_tok in {20, 21, 22}`: query role (`20 = VAL`, `21 = ATTR`, `22 = PROP`).
  - Target `y in {0..15}` (predicted over 16 target classes).
- **Per-seed secret structure** (seeded by `s * 911 + 42`):
  - `phi`: random permutation of `[0..15]` (so surface arithmetic cannot cheat across seeds).
  - Canonical meaning `m = f(op, a, b) in {0..15}`:
    - `op = 0`: `(phi[a] + phi[b]) % 16`
    - `op = 1`: `(phi[a] - phi[b]) % 16`
    - `op = 2`: `(3 * phi[a] + 5 * phi[b]) % 16`
    - `op = 3`: `(7 * phi[a] - 3 * phi[b]) % 16`
  - Latent context bit: `bit` = random balanced `{0, 1}` assignment (8 zeros, 8 ones) over the 16 symbols; `ctx_regime = bit[c1] ^ bit[c2] in {0, 1}`. Thus `P(ctx_regime=1 | c1) = 0.5` and `P(ctx_regime=1 | c2) = 0.5` (zero single-token leakage).
  - Stratified `(a, b)` split: per row `a`, 12 `b` values in `seen_pairs` (192 pairs total), 4 in `heldout_pairs` (64 pairs total).
  - Random permutations `g_base, g_alt, h_init, h_rev` on `[0..15]` such that `g_alt[m] != g_base[m]` and `h_rev[m] != h_init[m]` for all `m`.
  - Disjoint concept subsets: `S_split` (4 concepts, context-split in Phase 2), `S_rev` (4 concepts, law-revised in Phase 3), `S_static` (remaining 8 concepts).

### Sequential Stream (Zero Task/Phase Headers)
1. **Phase 1 — Base Grounding (`steps_p1` **[DEV]**, 3 rotating trusted sources `src_0, src_1, src_2`)**:
   - `seen_pairs` ONLY.
   - `VAL` (`q=20`): `op in {0, 1, 2, 3}` -> `m` (teaches equivalence across all 4 surface forms).
   - `ATTR` (`q=21`): `op in {0, 1, 2}` -> `g_base[m]` for both `ctx in {0, 1}` ("apples are red").
   - `PROP` (`q=22`): `op in {0, 1, 2}` -> `h_init[m]`.
   - (`op = 3` is held out from `ATTR` and `PROP` in Phase 1).
2. **Phase B — Single-Source Contradiction Burst (`steps_pb = 10` batches, `source_id = "noisy_single"`)**:
   - `seen_pairs`, `op in {0, 1}`, `ATTR` and `PROP` with corrupted shifted labels `(y + 5) % 16`.
3. **Phase 2 — Context-Conditional Exception (`steps_p2` **[DEV]**, 3 rotating trusted sources)**:
   - Focuses on `S_split` (`ATTR` with `op in {0, 1}` ONLY, `seen_pairs`): `ctx_regime == 0 -> g_base[m]` ("red" confirmed), `ctx_regime == 1 -> g_alt[m]` ("green" introduced), mixed with occasional confirmation samples from `VAL`/`PROP` (ratio **[DEV]**, same for all continual arms — exact data parity!).
4. **Phase 3 — Genuine Law Revision (`steps_p3` **[DEV]**, 3 rotating trusted sources)**:
   - Focuses on `S_rev` (`PROP` with **`op = 0` ONLY**, `seen_pairs`): target flips `h_init[m] -> h_rev[m]` for both `ctx in {0, 1}`. `op in {1, 2, 3}` are **NEVER shown** for `S_rev` in Phase 3.

### Declared Stream Metadata / Oracles
- `source_id` (`"src_0".."src_2"` vs `"noisy_single"`) is passed identically to every arm that tracks source corroboration (`B0_SurfaceDict`, `Idea1_*`, `Idea_Triad_All3`). `B1..B4` and `Idea2`/`Idea3` receive no task boundary signal; `B3_Dense_NullSpace` and Sleep steps are explicitly allowed phase-transition consolidation hooks OR periodic chunk consolidation (must declare which in `config`; we trigger Sleep/null-space consolidation strictly on *internal chunk/surprise triggers* OR fixed step period `every K steps` with **zero oracle phase boundaries**!).
  - **Strict Anti-Oracle Rule**: No arm receives a phase-boundary signal (`phase == 2`, `phase == 3`, or `task_id`)! Consolidation / Sleep / NullSpace updates fire strictly on fixed periodic intervals or internal surprise triggers.

---

## 3. Evaluation Subsets (All Scored Under `torch.no_grad()`, Separate Generator `s * 977 + 3`)

- `p1_val_heldout`: `VAL`, `heldout_pairs`, `op in {0..3}`
- `pb_retention`: all Phase-1 `VAL` + `ATTR` + `PROP` right after Phase B
- `p2_split_new`: `ATTR`, `S_split`, `ctx=1`, `op in {0,1}`, `seen_pairs`
- `p2_split_old`: `ATTR`, `S_split`, `ctx=0`, `op in {0,1}`, `seen_pairs`
- `p2_split_both_min`: `min(p2_split_new, p2_split_old)`
- `p2_split_transfer`: `ATTR`, `S_split`, `ctx in {0,1}`, `op in {2,3}` + `heldout_pairs`
- `p3_rev_direct`: `PROP`, `S_rev`, `op=0`, `seen_pairs`
- `p3_rev_equiv_transfer`: `PROP`, `S_rev`, `op in {1,2,3}`, `seen_pairs` (unseen in Phase 3!)
- `p3_rev_heldout`: `PROP`, `S_rev`, `heldout_pairs`, `op in {0..3}`
- `p3_retention_all`: `VAL` (`seen`+`held`) + `ATTR` (`S_split` both `ctx=0,1` + `S_static`+`S_rev`) + `PROP` (`S_static`+`S_split`)
- `p3_overall_acc` & `p3_overall_nll`: uniform macro-average over `VAL`, `ATTR`, `PROP` across all `16` concepts, `4` operators, `2` contexts, `seen`+`heldout` against true Phase-3 world state.

---

## 4. Arms (18 Total: Matched Params `~75K–110K`, Exact Same Stream)

### Baselines & Nulls
1. `B0_SurfaceDict`: non-neural counting dictionary keyed on `(op, a, b, c1, c2, q)` (`max_total_bytes = 16384`).
2. `B1_Dense_SGD`: causal RoPE Transformer (`d_model=64, heads=4, layers=2, d_ff=192`), AdamW on every batch.
3. `B2_Dense_Replay`: same Transformer + bounded FIFO/reservoir replay buffer (`max_total_bytes = 16384`, exact raw sample byte accounting), AdamW every step.
4. `B3_Dense_NullSpace`: same Transformer + periodic online covariance null-space projection (`I - U U^T`) every `consol_period` steps (no phase oracle).
5. `B4_JointOracle`: same Transformer trained offline from scratch on uniform mixture of final Phase-3 truth over `seen_pairs` (ceiling for accuracy & brute-force retraining compute).
6. `Null_Untrained`: random init, zero training.
7. `Null_ShuffledTargets`: `Idea_Triad_All3` trained with randomly permuted target labels.
8. `Null_BlindZeroInput`: trained `Idea_Triad_All3` evaluated with all inputs replaced by `0`.

### Idea 1 & Ablations
9. `Idea1_ConceptWCS`: plastic RoPE encoder + Top-1/Top-k Sparse Concept Dictionary + `SIGReg` + Concept-Keyed Bayesian Store with Context-Splitting + Mechanism P ripple demotion + RHO-style reducible-error chunk buffer + Elastic fast-head update + periodic Sleep consolidation.
10. `Idea1_WakeOnly`: `Idea1_ConceptWCS` with Sleep consolidation disabled.
11. `Idea1_NoCtxSplit`: `Idea1_ConceptWCS` with context-branch splitting disabled (flips instead of branching).

### Idea 2, Control & Ablation
12. `Idea2_EquivContra`: causal RoPE encoder + cross-view Equivalence Pull on `z_expr` (`VAL`-matched pairs) + `SIGReg` + Contradiction Separation on `z_ctx` + Concept-Anchored Readout.
13. `Idea2_ShuffledEquiv` (**mandatory `X-47` control**): identical to `Idea2_EquivContra` except equivalence pairs are randomly permuted across the batch.
14. `Idea2_NoSIGReg`: `Idea2_EquivContra` without `SIGReg`.

### Idea 3 & Ablations
15. `Idea3_MuPC_BONG_Pure`: **0 `.backward()` calls** — 4-layer predictive-coding residual network over concatenated token embeddings with prospective activity equilibration ($K_{\text{eq}}$ steps, no `y` at test time), local layer-wise energy-gradient updates on unresolved surprise, BONG diagonal precision, and Continual Backprop utility recycling (param-matched to Transformer).
16. `Idea3_Backbone_BONG`: shared RoPE Transformer backbone updated via BONG diagonal precision + RHO-style surprise gating + Continual Backprop utility recycling.
17. `Idea3_MuPC_Adam`: `Idea3_MuPC_BONG_Pure` using standard Adam instead of BONG precision.

### Combined Triad
18. `Idea_Triad_All3`: `Idea2` (Equivalence Pull + `SIGReg` + Contradiction Split) + `Idea1` (Sparse Concept Dictionary + Wake–Chunk–Sleep + Context-Splitting + Mechanism P) + `Idea3` (BONG Diagonal Precision + Continual Backprop utility recycling in Chunk/Sleep).

---

## 5. Splits, Tuning Budget, & Gates

- **DEV seeds**: `[100, 101, 102, 103, 104]` (`n = 5`), logged in `results/belief_training_lab/tuning_log_v1.json`.
- **TEST seeds**: `[0..19]` (`n = 20`), untouched until `PREREG_v1.md` is frozen with DEV-selected config.
- **Tuned on DEV [DEV]**:
  - `steps_p1`, `steps_p2`, `steps_p3` (raised until `B4_JointOracle` & `B2_Dense_Replay` converge cleanly on Phase 1 & Phase 2).
  - Learning rates per family from `{1e-3, 3e-3, 5e-3}`.
  - Idea 1 thresholds (`tau_ok`, `rho_thresh`, `ctx_split_thresh`, `chunk_size`), Idea 2 loss weights (`lambda_equiv`, `lambda_sigreg`, `lambda_contra`), Idea 3 equilibration steps & BONG damping (`K_eq`, `bong_lr`, `tau_pc_skip`).
- **Gates (DEV before TEST)**:
  - **G0 (Learnability / Bed Usable)**: `B4_JointOracle` achieves `p1_val_heldout >= 0.70`, `p2_split_both_min >= 0.75`, and `p3_overall_acc >= 0.75` on DEV mean.
  - **G1 (Strict Causality)**: perturbing tokens after `t` never changes logits at `<= t` by more than `1e-5` across sequence models; `Idea3_MuPC_BONG_Pure` executes `backprop_calls == 0`.
  - **G2 (Dictionary Bound)**: `B0_SurfaceDict` achieves `<= 0.15` on `p1_val_heldout` and `p3_rev_equiv_transfer`.

---

## 6. Frozen Falsifiers (All Checked on `TEST_SEEDS = [0..19]` via Script)

"CI" = 95% bootstrap confidence interval over the 20 test seeds.

1. **F1 (Idea 1 — Context-Split Red & Green Coexistence)**:
   - `Idea1_ConceptWCS.p2_split_both_min >= 0.75`, beats `Idea1_NoCtxSplit.p2_split_both_min` by `>= +0.15` (`CI_low > 0`), and beats `B1_Dense_SGD.p2_split_both_min` by `>= +0.20` (`CI_low > 0`).
2. **F2 (Idea 1 — Burst Immunity & Post-P1 Compute Savings vs Replay)**:
   - `Idea1_ConceptWCS.pb_retention >= 0.85` (drop `<= 0.05` from Phase 1), `pb_burst_flips <= 0.1` mean, AND `Idea1_ConceptWCS.flops_post_p1 <= 0.60 * B2_Dense_Replay.flops_post_p1` AND `Idea1_ConceptWCS.p3_overall_acc >= B2_Dense_Replay.p3_overall_acc - 0.03`.
3. **F3 (Idea 2 — Zero-Retrain Equivalence Transfer & X-47 Shuffled Control)**:
   - `Idea2_EquivContra.p3_rev_equiv_transfer >= 0.70`, exceeds `B1_Dense_SGD.p3_rev_equiv_transfer` by `>= +0.15` (`CI_low > 0`), AND exceeds `Idea2_ShuffledEquiv.p3_rev_equiv_transfer` by `>= +0.15` (`CI_low > 0`).
4. **F4 (Idea 3a — Backprop-Free $\mu$PC+BONG Viability)**:
   - `Idea3_MuPC_BONG_Pure.backprop_calls == 0`, `p1_val_heldout >= 0.65`, `p3_overall_acc >= 0.55`, AND exceeds `Idea3_MuPC_Adam.p3_overall_acc` (`CI_low > 0` or retention gain `>= +0.05`), with wall-clock and FLOPs reported honestly alongside `B1_Dense_SGD`.
5. **F5 (Idea 3b — BONG Precision on Shared Backbone)**:
   - `Idea3_Backbone_BONG.p3_retention_all >= B1_Dense_SGD.p3_retention_all + 0.10` (`CI_low > 0`) and `pb_retention >= B1_Dense_SGD.pb_retention + 0.10`.
6. **F6 (Triad Synergies & Beats Surface Dictionary + Dense Replay)**:
   - `Idea_Triad_All3.p3_overall_acc >= max(B0_SurfaceDict, B1_Dense_SGD, B2_Dense_Replay).p3_overall_acc + 0.05` (`CI_low > 0`), `p3_rev_equiv_transfer >= 0.75`, `p2_split_both_min >= 0.75`, and `flops_post_p1 < B4_JointOracle.flops_total` (reported with wall-clock ratio).
7. **F7 (Calibration Honesty)**:
   - `Idea_Triad_All3.p3_overall_nll <= B1_Dense_SGD.p3_overall_nll` and `<= 1.2` nats.
8. **F8 (Null Controls Fail)**:
   - All 3 null controls (`Null_Untrained`, `Null_ShuffledTargets`, `Null_BlindZeroInput`) score `< 0.15` on `p1_val_heldout`, `p2_split_both_min`, `p3_rev_equiv_transfer`, and `p3_overall_acc`.

---

## 7. Outcome Interpretation Ledger (Pre-Committed)

- Any falsifier that fails is reported plainly in the first paragraph of the report, with exact numbers and per-seed spread.
- If `Idea2_ShuffledEquiv` matches `Idea2_EquivContra` (F3 fails on shuffled control), **Claim 2 is REFUTED** (equivalence loss acts only as generic regularization).
- If `Idea3_MuPC_BONG_Pure` passes accuracy yet loses badly in wall-clock time to `B1_Dense_SGD`, report both accurately: *"works without `.backward()`, slower on GPU hardware."*
- If `Idea_Triad_All3` fails to beat `B2_Dense_Replay` (F6 fails) or uses more wall-clock despite fewer FLOPs, report that honestly without softening.


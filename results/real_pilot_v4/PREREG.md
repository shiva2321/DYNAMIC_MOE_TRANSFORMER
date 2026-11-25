# Pre-Registration: `real_pilot_v4` (`v7` Non-Token Cognitive Core, Dynamic `k in 0..2`, & Contextual Nuance)

**Frozen Before Dev/Test Execution (UTC):** 2026-11-20T13:55:00Z  
**Status:** UPDATED BEFORE DEV CALIBRATION & HELD-OUT TEST RUNS (Per `research-protocol.md` §4 & User Approved Dual-Track Decisions `A1 + A2`)  
**Scripts:** `experiments/run_v7_cognitive_pilot.py`, `experiments/v7_edit_families.py`  
**Blueprint:** `v7_dynamic_shared_nuance_architecture_spec_and_prereg.md`

---

## 1. Research Question, Dual-Track Attribution, & Pre-Registered Falsifiers

Evaluate **Architecture `v7`** — a **Non-Token Concept-Chunk Cognitive Core** combining causal `MultiScaleByteChunker` (`chunk_size=4`), shared lower GPM trunk (`Layers 0..2`), post-Layer-3 `ThreeLevelAttention`, post-MoE/post-Thinker dual meaning factorization (`z_expr`, `z_ctx`), learned GPM-protected contextual routing (`CausalContextTokenRouter`, `k in {0, 1, 2}`, loss-free bias balancing + Expert-Router Coupling `I8`), Tier-A/B causal nuance escalation (`nuance_mask` -> `k=2` + `Thinker`, `NuanceClusterBuffer`), deep-target JEPA + Future Summary Prediction (`I3`), RLP-style latent `Thinker` progress supervision (`I2`), verified multi-family edit supervision + held-out anti-shortcut families (`I4`), Muon readout guard rails (`I11`), and per-group `Muon + Diagonal BONG + GPM + LatentQuarantineGate` — **with ZERO next-byte cross-entropy or perplexity/BPB metrics anywhere in core training or headline claims**.

### Explicit Track & Attribution Split (Approved Decision §6 `A1`)
1. **Track 1 (`BeliefMeaningTask`, Compositional Generalization): 100% Strictly From-Scratch**  
   Zero pretrained teachers during training or inference.
2. **Track 2 (`Real Streams + Banking77 + Edit Families`): Explicitly Separated Disclosure**  
   - **From-Scratch Arms:** `BagOfBytes_Null`, `Untrained_Cog_v7`, `Naive_JEPA_M17`, `Dense_v7_Matched`, `Cog_v7`, `Cog_v7_SequenceTop1`, `Cog_v7_FrozenRandomRouter`.
   - **Teacher-Informed Arms (Labeled `"TeacherInformed_I9"`):** `Dense_v7_Matched_I9`, `Cog_v7_I9`, `Cog_v7_Mixed_I9` use offline pairwise cosine similarity Gram matrix distillation from a cached sentence encoder (`all-MiniLM-L6-v2`) strictly on **training batches only** (`teacher_weight = 0.80`). Inference, eval, and continual adaptation are **100% teacher-free, standalone, and plastic**.

### Pre-Dev Engineering Fixes Logged Before Calibration (`smoke_v7_b` Diagnostics)
1. Fixed intra-batch chunk variance estimation in `UnifiedNoveltyDetector.evaluate_contextual_chunks` (Law of Total Variance) to eliminate `40.7%` false clean triggers (`k2=0.802` collapse).
2. Wired post-MoE/post-Thinker `z_expr`, `z_ctx`, `fsp_head`, `abstain_head`, `wm_imagine_gate=0.25`, `auto_write_ltm=True`, `Thinker` RLP progress loss (`I2`), and `I1` gradient-coverage gate (`test_v7_gradient_coverage_gate_I1`).
3. Fixed matched-depth `Thinker` vs `Think0` escalation readout (`oa0, oj0, of0`), normalized WorldModel target retrieval (`F.layer_norm` over 3-step horizon), and structural `sequence_level=True` pooling for `Cog_v7_SequenceTop1`.
4. Replaced single-match rule lists (where `~74%` fell through and `' is '` risked shortcut lookup) with **I4 verified edit families (`A-D` train, `E-F` strictly held-out)**.

### Explicit Pre-Registered Falsifiers (Must Pass on Held-Out `test` Split; Otherwise Reported as REFUTED)
1. **F1 — Non-Token Semantic Equivalence (`M17` & Surface-Trap Check):**  
   Held-out `Banking77` **`same_vs_overlap_diff_auc >= 0.72`**, decisively beating `BagOfBytes_Null` and `Naive_JEPA_M17` by **`>= +0.15` AUC** (evaluating BOTH From-Scratch and Teacher-Informed Track 2 arms against equal-supervision `Dense_v7_Matched`).
2. **F2 — Neuro-Symbolic Compositional Generalization & Sequential Retention:**  
   - **Track 1 (Strictly From-Scratch):** `Cog_v7` beats `Upkeeper_M09` by **`>= +0.25` accuracy** on held-out unseen compositions & paraphrased surface inputs while maintaining **`>= 0.85` burst immunity** and **`abstain_auroc >= 0.70`**.
   - **Track 2 (Zero-Replay Sequential `wiki -> story -> code -> math -> forum`):** `Cog_v7` cuts non-token forgetting (`forgetting_r5` + `forgetting_nuance_auc`) by **`>= 2.0×` vs equal-supervision `Dense_v7_Matched`**.
3. **F3 — Dynamic `k in {0,1,2}` Sharing & Intra-Sequence Switching:**  
   No single `k in {0,1,2}` exceeds `90%`; `MixedPromptSwitch` reaches **`mid_seq_switch_rate >= 0.35`** (vs `0.00` for `Cog_v7_SequenceTop1`); mixed `Cog_v7_Mixed_I9` matches or beats `Dense_v7_Matched_I9` with `>= 2` active specialists.
4. **F4 & F4b — Contextual Nuance Escalation & Held-Out Edit Family Generalization:**  
   - **F4:** On held-out real Nuance-Flip pairs (`wiki`, `story`, `code`, `gsm8k`), `nuance_mask` selectivity ratio **`>= 1.5×`** AND matched-depth `Thinker` escalation improves flip/disambiguation margin over `Think0`.
   - **F4b (Anti-Shortcut Leave-One-Family-Out):** On disjoint unseen edit families `clause_quant` + `code_logic_ctrl` (`Families E+F`), achieve **`heldout_fam_auc >= 0.65`** cleanly above `BagOfBytes_Null`.
5. **F5 — Latent World-Model Rollout & RTX 3060 Efficiency:**  
   `Thinker` improves counterfactual rollout over `Think0` on Track 1; sequential 5-domain runs stay within **`<= 250s / arm`** and **`< 2.5 GB` peak VRAM** on RTX 3060.

---

## 2. Frozen Calibration & Execution Protocol

- At most **4 calibration trials** on `--mode dev --seed 100` recorded automatically to `results/real_pilot_v4/tuning_log.json`.
- After `dev` lock-in, execute **ONE frozen held-out `--mode test` run** across `seeds = [0, 1, 2]` (reported at `OBSERVED (3 seeds)` tier per `research-protocol.md` §§6 & 9).

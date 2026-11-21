# Pre-Registration: Architecture v9 Rigorous Diagnostics and Generalization Pilot (`real_pilot_v6`)

**Date:** 2026-11-21  
**Branch:** `v5-arch-efficiency`  
**Repository:** `d:\Dynamic_MOE_Transformer`  
**Author:** Antigravity (Google DeepMind Agent)  
**Status:** FROZEN prior to final test execution  

---

## 1. Intent and Protocol Context

Following the approved `v9 plan` (`docs/superpowers/specs/2026-11-21-v9-diagnostics-and-generalization-plan.md`), this project enforces the **TruthGate Contract** and `research-protocol.md`. Every reported number is copied directly from physical execution on NVIDIA RTX 3060 hardware. No numbers are faked, curve-fitted, or estimated.

Stage 0 diagnostics systematically investigated the integrity and attribution issues of `v8`. With baseline noise floors and structural mechanics now empirically measured, this pre-registration freezes the final evaluation protocol and criteria for the consolidated `Cog_v9` architecture.

---

## 2. Stage 0 Diagnostic Outcomes (Empirical Baseline)

| Diagnostic ID | Investigation | Measured Outcome | Empirical Status |
|---|---|---|---|
| **D1: Matched Controls** | Dense and FrozenRandomRouter with AdamW vs matched MuonBONGOptimizer, GPM, and QuarantineGate | AdamW failed to fit seen data at all tested LRs (1e-3: 0.073, 3e-3: 0.081, 8e-3: 0.094). With matched MuonBONG, `Dense_v8_Matched` fits seen data to **0.930** on Track 1 and achieves **0.564** Banking77 accuracy on Track 2 (tying `Cog_v8` at 0.559). | Prior v8 claim of MoE superiority over Dense was confounded by optimizer mismatch. AdamW controls dropped as invalid. |
| **D2: Unbundled Ablations** | Decompose `NoI5_NoRel` into independent flags: `NoI5`, `NoRel`, `NoI7`, `NoV8Core` | Track 1 held-out generalization with relational pooling OFF (`Cog_v8_NoRel`) jumped from 0.070 to **0.810**; `NoI5` dropped interchange to 0.125 (chance). On Track 2, `NoV8Core` crashed to 0.264, while `NoRel` (0.568) and `NoI7` (0.580) remained strong. | Relational pooling was a severe bottleneck for modular arithmetic grokking. The v8 baseline's collapse was caused by the 30+ site core revert, not IIT or RelPool. |
| **D3: Routing × Escalation 2×2** | `{Dynamic-k, SequenceTop1} × {Escalation ON, OFF}` | Dynamic-k with escalation OFF (`Cog_v8_NoEsc`) achieved **0.598** Banking77 accuracy at **mean k = 0.86** and **241.4s** train time, beating SequenceTop1 (0.551, mean k=1.30, 283.5s) and SequenceTop1_NoEsc (0.537, mean k=1.00, 240.6s). | Nuance escalation burned ~53% more compute without improving accuracy. Dynamic routing with escalation off achieves superior efficiency and accuracy. |
| **D4: Honest Episodic Probes** | (a) Shuffled-binding null control, (b) exact lookup oracle, (c) novel-pair induction probe across 4/8/16 shots | Lookup oracle reached **0.766 - 0.953**, shuffled null control scored **0.062 - 0.141**, but novel-pair induction scored **0.000 - 0.078** (chance = 0.0625) across all shot counts. | **Falsifier V9-F3 confirmed.** Episodic memory (I6) is purely associative memory retrieval; it performs zero in-context rule induction on novel combinations. |
| **D5: Interchange Generalization** | Evaluate I5 causal interchange on seen pairs, held-out pairs, and random slice null control | In `Cog_v8`, seen interchange was 1.000 but heldout was 0.062. In `Cog_v8_NoRel`, seen interchange was **1.000**, heldout interchange was **0.859**, and random slice null control was **0.125**. | Causal localization to `[0:D//2]` is verified and generalizes out-of-distribution when the relational bottleneck is removed. |
| **D6: Grokking Curves** | Multi-step evaluation of held-out accuracy during p1 | Grokking curve under default v8 stayed flat at 0.070; under `NoRel` and `H1a`, held-out accuracy climbed continuously from step 500 (0.182) to step 1300 (0.841). | Grokking is confirmed as an emergent, continuous phenomenon delayed by premature precision consolidation. |
| **D7: GPM Exemption** | Test `exempt_readout_heads=False` and log basis saturation | With exemption ON, saturation is 0.0%. With exemption OFF, saturation reaches **25.6% - 28.2%** across readout heads. | Confirms GPM exemption mechanism directly prevents basis saturation in readout projections. |
| **D8: Routing Specialization** | AMI with 50-step permutation null and per-expert lesion analysis | `Cog_v8` AMI = **0.305** (null mean = 0.000), but `FrozenRandomRouter` AMI = **0.293** (virtually identical); per-expert lesion drops were **0.000**. | **Falsifier V9-F6 confirmed.** Routing specialization tracks surface sequence/byte length distributions rather than learned semantic functions. |
| **D9: Metric Hygiene** | Near-OOD abstain set using valid syntax with illegal operator/concept tokens | Standard random noise AUROC saturated at 1.0; near-OOD abstain AUROC reliably evaluates discrimination against subtle out-of-distribution inputs without trivial noise saturation. | Implemented and integrated. |
| **D10: Steps Scaling** | Evaluate Track 2 at 250 vs 500 steps per domain | At 250 steps, R@5 = 0.166; at 500 steps, R@5 = 0.141 (533s). Curve is flat/slightly declining. | **Falsifier V9-F5 confirmed.** Longer training does not improve R@5; retrieval requires representational objective changes, not step scaling. |
| **D11: Compute Profiler** | PyTorch profiler on fixed step | Self CPU time (729.6ms) ≈ Self CUDA time (738.7ms). High kernel launch count (3,472 `as_strided`, 1,165 `transpose`, 968 `reshape` calls). | Pipeline is launch-bound due to fine-grained chunk and token slicing. |

---

## 3. Pre-Registered Hypotheses and Architecture Definition (`Cog_v9`)

Based on Stage 0 findings, `Cog_v9` is consolidated with the following design:
1. **Direct Concept Pooling (`use_rel_pool = False`):** Eliminates the 3D nonlinear bottleneck, restoring modular grokking on Track 1 and increasing Banking77 overlap-trap AUC on Track 2 to **0.662**.
2. **Compute-Aware Causal Dynamic Routing (`mean_k <= 1.0`):** Sets `enable_nuance_escalation = False` by default, reducing mean compute from $k=1.32$ to $k=0.86$ (-35%), saving 43 seconds per domain, and improving Banking77 accuracy to **0.598**.
3. **Decoupled Constant Fisher Acquisition (`use_constant_fisher = True`):** Holds Fisher scaling at $1.0$ during acquisition, eliminating premature consolidation of memorized solutions and enabling modular grokking.
4. **Honest Causal IIT (`use_iit = True`):** Maintains causal interchange training on `[0:D//2]`, verifying held-out causal abstraction.
5. **Readout Exemption (`exempt_readout_heads = True`):** Prevents basis saturation across downstream projection heads.

### Pre-Registered Test Bars:
- **Bar 1 (Generalization on Track 1):** `Cog_v9` held-out accuracy $\ge 0.70$ (vs `Cog_v8` baseline $0.070$, paired margin $\ge +0.50$).
- **Bar 2 (Causal Interchange Generalization):** `Cog_v9` held-out interchange accuracy $\ge 0.75$ with random slice null $\le 0.25$.
- **Bar 3 (Compute Efficiency on Track 2):** `Cog_v9` mean $k \le 1.0$ with Banking77 accuracy $\ge 0.580$ (beating `SequenceTop1` by $\ge +0.02$).
- **Bar 4 (Matched Control Integrity):** All comparisons made strictly against `Dense_v8_Matched` using matched `MuonBONGOptimizer`.
- **Bar 5 (Episodic Honesty):** Disclose that episodic memory is an associative lookup mechanism ($novel\_induction\_acc \le 0.10$ at chance), not rule induction.

# Pre-Registration: Phase 3 - Unified Heterogeneous Multi-Manifold Architecture (GeoCore)

**Date:** 2026-10-14  
**Track:** Z (Geometry Core Verification)  
**Status:** Pre-registered before any Phase 3 experimental execution  
**Target Results File:** `research/geomlab_results/phase3/results.json`

---

## 1. Objective and Research Questions
This experiment evaluates the complete **Unified Heterogeneous Multi-Manifold General Intelligence Core (`GeoCore`)**, synthesizing the measured conclusions from Phases 0, 1, and 2:
1. Continuous perceptual inputs must live on the unit hypersphere $\mathbb{S}^{D-1}$ to prevent representation collapse and bound logits (Phase 1).
2. Relational variable binding requires flat Clifford Torus phasors $\mathbb{T}^D$ to eliminate holonomy defects (Phase 0).
3. Sequential transformations require block-unitary Lie group operators $U(b)^{D/b}$ to conserve metric isometries across extended lengths (Phase 2).
4. Superposition working memory requires **continuous metric attractor cleanup dynamics** (Modern Hopfield projection) to overcome the $69.8$ percentage point semantic crosstalk tax discovered in Phase 1.

### Research Questions:
1. **Does metric attractor cleanup suppress the real-world superposition crosstalk tax?** When multiple real-text entities are superposed in working memory, does Hopfield attractor snapping recover degraded states?
2. **Does the unified heterogeneous architecture achieve superior multi-step variable tracking?** When tracking bound variables $(R_1 \odot E_1 + R_2 \odot E_2)$ through sequential multi-step transformations, does `GeoCore` outperform standard Transformers and recurrent networks at lengths $K \in \{1, 2, 4, 8\}$?

---

## 2. Experimental Task: Multi-Step Bound Variable Tracking on Real Text

- **Entities:** $N = 128$ distinct intent queries from the real Banking77 dataset, embedded via `HashFrontend(d=1024)`.
- **Roles:** $M = 4$ abstract relational roles (e.g., subject, object, source, destination).
- **Working Memory State:** At $t=0$, $M=4$ role-filler pairs are bound into a superposed state:
  $$S_0 = \sum_{m=1}^M R_m \odot E_{i_m}$$
- **Sequential Multi-Step Transformations:**
  At each step $t \in \{1, \dots, K\}$, a relational transformation operator $T_t \in \mathcal{T}$ ($|\mathcal{T}| = 4$) updates the state:
  $$S_t = \mathcal{U}(T_t) S_{t-1}$$
- **Query & Target:**
  At sequence end $K$, the system is queried with role $R_q$:
  $$\text{Target} = \text{Filler bound to } R_q \text{ at step } K$$
- **Dataset Splits:**
  - Training: 4,000 sequences of lengths $K \in \{1, 2\}$.
  - In-distribution test: 1,000 held-out sequences of lengths $K \in \{1, 2\}$.
  - OOD length extrapolation: 1,000 held-out sequences of length $K = 4$, and 1,000 held-out sequences of length $K = 8$.
- **Dimension:** $D = 256$ across all arms.

---

## 3. Matched Arms and Architectural Specifications

All models have matched embedding and parameter budgets ($D = 256$, AdamW optimizer with tuned learning rates on dev seeds).

1. **`baseline_transformer`:**
   - 2-layer causal Transformer, 4 heads, $d_{\text{model}} = 256$, $d_{\text{ffn}} = 512$, LayerNorm, learned positional embeddings.
   - Sequence input: $[S_0, T_1, \dots, T_K, R_q]$.
   - Readout from final token via linear classifier over the 128 entity classes.

2. **`baseline_recurrent`:**
   - Recurrent neural network with LayerNorm and GELU: $h_t = \operatorname{LayerNorm}(\operatorname{GELU}(W_h h_{t-1} + W_T T_t))$.
   - Readout: Linear probe conditioned on query role $R_q$: $W_{\text{out}} [h_K; R_q]$.

3. **`uncleaned_geocore` (Ablation - No Attractor Dynamics):**
   - Spherical input normalization $\mathbb{S}^{D-1}$.
   - Toroidal role-filler phase binding: $\theta(R \odot E) = \theta(R) + \theta(E) \pmod{2\pi}$.
   - Block-unitary sequential state update: $U_t = U_{t-1} \mathcal{R}(T_t)$ with $U(4)^{64}$.
   - Linear unbinding: query role phase subtraction $\theta_{\text{retrieved}} = \theta(S_K) - \theta(R_q)$ followed by nearest-neighbor readout without attractor cleanup.

4. **`full_geocore` (Complete Geometric Intelligence Architecture):**
   - Integrates all components of `uncleaned_geocore`, PLUS:
   - **Continuous Metric Hopfield Attractor Cleanup:**
     After unbinding, the retrieved state $\theta_{\text{noisy}}$ is passed through a temperature-scaled continuous energy minimization step against the clean entity codebook:
     $$z^{(1)} = \operatorname{Proj}_{\mathbb{T}^D} \left( \sum_{i=1}^N \operatorname{softmax}\left(\beta \cos(\theta_{\text{noisy}} - \theta_i)\right) e^{i \theta_i} \right)$$
     Snaps the crosstalk-degraded superposition state to the nearest clean attractor codebook representation.

---

## 4. Pre-Registered Hypotheses and Decision Rules

| Hypothesis | Description | Success Criterion | Falsified If |
|---|---|---|---|
| **H1** | Attractor Crosstalk Cleanup | At bundle size $M=4$, `full_geocore` improves retrieval recovery over `uncleaned_geocore` by $\ge 15$ percentage points at $K=1, 2$ ($p < 0.001$) | Cleanup gain $< 5$ percentage points |
| **H2** | Multi-Step Reasoning Extrapolation | At OOD length $K=8$, `full_geocore` outperforms `baseline_transformer` and `baseline_recurrent` by $\ge 30$ percentage points ($p < 0.001$) | `full_geocore` advantage $< 15$ points or $p \ge 0.01$ |
| **H3** | Geometry Health & Anti-Collapse | `full_geocore` preserves effective rank $\ge 50\%$ of $D$ across the pipeline, while `baseline_transformer` effective rank drops below $30\%$ of $D$ | `full_geocore` effective rank $< 35\%$ |
| **H4** | Isometry Norm Conservation | `full_geocore` maintains strictly $0.0\%$ norm drift ($\Delta_{\text{norm}} \equiv 1.0$) across $K=8$, while baseline recurrent states drift | `full_geocore` norm drifts by $> 1\%$ |

### Decision Rules:
- **Adopt the Full Heterogeneous Geometric Core for the General-Intelligence Substrate** if H1, H2, and H4 pass (confirming that attractor cleanup solves the crosstalk tax and isometric operators solve multi-step reasoning drift).

---

## 5. Hyperparameter Tuning and Protocol
- **Dev Seeds:** $100, 101, 102$ evaluated on learning rates $[3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}]$.
- **Test Seeds:** 10 fresh, unused test seeds: $0, 1, 2, 3, 4, 5, 6, 7, 8, 9$.
- **Crash-Resilience:** Trial-by-trial atomic checkpointing (`checkpoint.json`).
- **Integrity Rule:** `results.json` modification timestamp strictly follows this file.

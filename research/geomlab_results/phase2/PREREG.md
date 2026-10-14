# Pre-Registration: Phase 2 - Compositional OOD Generalization and Length Extrapolation

**Date:** 2026-10-14  
**Track:** Z (Geometry Core Verification)  
**Status:** Pre-registered before any Phase 2 experimental execution  
**Target Results File:** `research/geomlab_results/phase2/results.json`

---

## 1. Objective and Research Questions
This experiment tests whether geometric representations equipped with exact group isometries (Lie group block-unitary operators $U(4)^{D/4}$ and flat Clifford Torus phasors $\mathbb{T}^D$) solve out-of-distribution (OOD) compositional length extrapolation, or whether unconstrained neural representations (standard Transformers and Euclidean RNNs) fail when relational composition depth exceeds the training distribution.

Specifically:
1. **Can neural networks extrapolate relational reasoning to unseen depths?** When trained only on composition paths of length $K \in \{1, 2, 3\}$, can a standard Transformer or Euclidean recurrent network correctly trace paths of length $K \in \{4, 6, 8, 12\}$?
2. **Why do standard architectures fail?** Is failure caused by representation drift (unbounded norm growth or attenuation) and non-isometric error accumulation?
3. **Does an exact isometric group operator eliminate the length barrier?** Because unitary matrices and flat torus rotations preserve $L_2$ norm and metric distances exactly ($\|U v\|_2 = \|v\|_2$), does an isometric representation maintain high accuracy across extended reasoning chains without fine-tuning?

---

## 2. Experimental Task: Multi-Step Graph Relational Composition

- **Graph Structure:** $N = 256$ entities arranged in a multi-relational directed graph with $|\mathcal{R}| = 8$ distinct non-trivial relational generators (permutations / transitions).
- **Relational Composition Sequence:**
  A path is defined by a start entity $e_0 \in \{0, \dots, N-1\}$ and a sequence of relation tokens $(r_1, r_2, \dots, r_K)$, yielding ground truth target entity:
  $$e_K = e_0 \circ r_1 \circ r_2 \circ \dots \circ r_K$$
- **Dataset Partitioning (Zero Data Leakage):**
  - **Training Distribution ($K_{\text{train}} \in \{1, 2, 3\}$):** 6,000 distinct paths of lengths 1, 2, and 3.
  - **In-Distribution Test ($K \in \{1, 2, 3\}$):** 1,500 held-out paths of lengths 1, 2, and 3.
  - **OOD Extrapolation Test ($K \in \{4, 6, 8, 12\}$):** 1,000 held-out paths for EACH length $K \in \{4, 6, 8, 12\}$ (total 4,000 OOD paths). Not a single path of length $\ge 4$ is ever seen during training or validation tuning.
- **Dimensionality:** $D = 256$ across all arms.

---

## 3. Matched Arms and Architectural Specifications

All models have matched embedding budgets ($D = 256$) and are trained with AdamW with their own hyperparameter grid on dev seeds.

1. **`standard_transformer` (Causal Transformer Baseline):**
   - 2 Transformer layers, 4 attention heads, hidden dimension 256, FFN dimension 512, LayerNorm, GELU.
   - Learned positional embeddings.
   - Sequence input: $[e_0, r_1, r_2, \dots, r_K]$.
   - Readout from final token position via linear classifier over the 256 entity classes.
   - Loss: Cross-entropy.

2. **`euclidean_rnn` (Standard Recurrent Baseline):**
   - Hidden state $h_t \in \mathbb{R}^{256}$.
   - Transition: $h_t = \operatorname{LayerNorm}(\operatorname{GELU}(W_h h_{t-1} + W_r r_t))$.
   - Readout: Linear classifier $W_{\text{out}} h_K + b$.
   - Loss: Cross-entropy.

3. **`torus_isometry` (Flat Clifford Torus Phase Binding):**
   - Entities and relations represented as phase angles $\theta \in [-\pi, \pi)^D$.
   - Transition is exact torus rotation / phase addition:
     $$\theta_t = \theta_{t-1} + \phi(r_t) \pmod{2\pi}$$
   - Readout: Cosine similarity between final phase $\theta_K$ and all entity phase prototypes with learned inverse temperature $\beta$.
   - Loss: Cross-entropy.

4. **`block_unitary_isometry` (Block-Unitary Lie Group Composition):**
   - Dimension $D = 256$ partitioned into $B = 64$ blocks of size $4 \times 4$.
   - State: Block-unitary matrix $U_t \in U(4)^{64}$.
   - Initial state: $U_0 = \mathcal{E}(e_0)$.
   - Transition: $U_t = U_{t-1} \mathcal{R}(r_t)$, where $\mathcal{R}(r) \in U(4)^{64}$ is parameterized via skew-Hermitian matrix exponential $\exp(A - A^\dagger)$, ensuring exact isometry ($\|U_t v\|_2 = \|v\|_2$).
   - Readout: Normalized Frobenius inner product / complex trace $\operatorname{Re}\operatorname{Tr}(U_{\text{target}}^\dagger U_K)$ with learned scale $\beta$.
   - Loss: Cross-entropy.

---

## 4. Evaluation Metrics and Diagnostics

1. **Accuracy Across Composition Lengths:**
   - In-distribution: Accuracy on $K \in \{1, 2, 3\}$ (mean $\pm 95\%$ CI).
   - OOD extrapolation: Accuracy on $K = 4, 6, 8, 12$ (mean $\pm 95\%$ CI).
2. **Representation Energy / Norm Drift Ratio:**
   $$\Delta_{\text{norm}}(K) = \frac{\|h_K\|_2}{\|h_0\|_2}$$
   Measures whether hidden representation energy vanishes or explodes as length increases.
3. **Paired Statistical Tests:**
   Two-sided paired Student's t-test, Wilcoxon signed-rank test, and Cohen's $d_z$ between isometric models and neural baselines at $K = 8$ and $K = 12$.

---

## 5. Pre-Registered Hypotheses and Decision Rules

| Hypothesis | Description | Success Criterion | Falsified If |
|---|---|---|---|
| **H1** | In-Distribution Competence | All 4 models achieve $\ge 80\%$ accuracy on $K \in \{1, 2, 3\}$ | Any model fails to reach $70\%$ in-distribution |
| **H2** | Baseline Length Breakdown | `standard_transformer` and `euclidean_rnn` drop below $30\%$ accuracy at $K=8$ and below $15\%$ at $K=12$ | Either baseline maintains $\ge 50\%$ at $K=12$ |
| **H3** | Isometric Length Extrapolation | `block_unitary_isometry` maintains $\ge 75\%$ accuracy at $K=8$ and $\ge 60\%$ at $K=12$, beating both baselines by $\ge 40$ percentage points ($p < 0.001$) | `block_unitary_isometry` accuracy $< 50\%$ at $K=12$ or difference $< 20$ points |
| **H4** | Isometry Norm Conservation | `block_unitary_isometry` and `torus_isometry` have strictly $0.0\%$ norm drift ($\Delta_{\text{norm}} \equiv 1.0$), while `euclidean_rnn` norm drifts by $\ge 25\%$ at $K=12$ | Isometric norm drifts by $> 1\%$ |

### Decision Rules:
- **Adopt Lie Group / Isometric Operators for Multi-Step Reasoning** if H3 and H4 pass (proving that metric isometries prevent reasoning drift and unlock depth extrapolation).
- **If H2 is falsified (Transformer extrapolates to $K=12$ without degradation):** Conclude that learned attention already handles composition without geometric inductive bias, and do not mandate group-theoretic state transitions.

---

## 6. Hyperparameter Tuning and Protocol
- **Dev Seeds:** $100, 101, 102$ evaluated on learning rates $[3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}]$.
- **Test Seeds:** 10 fresh, unused test seeds: $0, 1, 2, 3, 4, 5, 6, 7, 8, 9$.
- **Crash-Resilience:** Trial-by-trial atomic checkpointing (`checkpoint.json`).
- **Integrity Rule:** `results.json` modification timestamp strictly follows this file.

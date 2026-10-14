# Pre-Registration: Phase 0 - Group Symmetries and Algebraic Discovery

**Date:** 2026-10-14  
**Track:** Z (Geometry Core Verification)  
**Status:** Pre-registered before any experimental execution  
**Target Results File:** `research/geomlab_results/phase0/results.json`

---

## 1. Objective and Research Questions
Does an explicit geometric hypothesis class (flat Clifford Torus $\mathbb{T}^D$ for abelian groups, block-unitary $U(b)^{D/b}$ for non-abelian groups) enable a learning system to discover symmetry and group structure from data more efficiently than an unconstrained neural network?

Specifically:
1. **Can an abelian geometric model (`torus_phase`) discover $\mathbb{Z}_{113}$ modular addition with significantly fewer steps and data than an unconstrained MLP or Transformer?**
2. **Does an abelian model fail on a non-abelian group ($S_5$), serving as an honest negative control that proves geometry must match the data's true symmetry?**
3. **Does a block-unitary model (`block_unitary`) resolve non-abelian $S_5$ where the abelian model fails?**
4. **Do learned representations show spectral concentration (Fourier/irrep sparsity) as predicted by representation theory?**

---

## 2. Experimental Setting and Tasks

### Task 1: Commutative Modular Addition ($\mathbb{Z}_{113}$)
- **Domain:** Group $(\mathbb{Z}_{113}, +)$, prime order $p = 113$.
- **Input:** Token pairs $(a, b) \in \{0, \dots, 112\}^2$.
- **Output:** Target $y = (a + b) \bmod 113$.
- **Dataset Size:** $113 \times 113 = 12,769$ total pairs.
- **Split:** $30\%$ training pairs ($3,830$ pairs), $70\%$ held-out test pairs ($8,939$ pairs).

### Task 2: Non-Commutative Permutation Composition ($S_5$)
- **Domain:** Symmetric group $S_5$ (permutations of 5 elements, $|S_5| = 5! = 120$).
- **Input:** Token pairs $(\sigma_1, \sigma_2) \in S_5 \times S_5$.
- **Output:** Target permutation $\sigma = \sigma_1 \circ \sigma_2$.
- **Dataset Size:** $120 \times 120 = 14,400$ total pairs.
- **Split:** $30\%$ training pairs ($4,320$ pairs), $70\%$ held-out test pairs ($10,080$ pairs).
- **Non-commutativity:** In $S_5$, $80\%$ of element pairs do not commute ($\sigma_1 \circ \sigma_2 \neq \sigma_2 \circ \sigma_1$).

---

## 3. Arms and Model Architectures (Matched Parameter Budgets)

Dimension $D = 128$. Parameter counts matched within $\pm 15\%$:

1. **`baseline_mlp` (Unconstrained MLP Control):**
   - Embedding table: $N \times D$.
   - Concatenated representation $[e_a; e_b] \in \mathbb{R}^{2D}$.
   - Hidden layer: $\text{Linear}(2D \to 2D) \to \text{GELU} \to \text{LayerNorm}$.
   - Output projection: $\text{Linear}(2D \to N)$.
   - Optimizer: AdamW with weight decay $0.1$.

2. **`baseline_transformer` (Standard Attention Control):**
   - Embedding table: $N \times D$ + learned position embeddings.
   - 1-layer standard Transformer: Multi-head attention (4 heads, head dimension 32) + MLP (dim 256) + LayerNorm + residual connections.
   - Output projection: $\text{Linear}(D \to N)$.
   - Optimizer: AdamW with weight decay $0.1$.

3. **`torus_phase` (Abelian Clifford Torus $\mathbb{T}^D$):**
   - Learned phase angles $\theta \in [-\pi, \pi)^{N \times D}$.
   - Phasor representation: $z_a = e^{i\theta_a}, z_b = e^{i\theta_b} \in \mathbb{C}^D$.
   - Group operator: Pointwise complex product $z_{\text{out}} = z_a \odot z_b = e^{i(\theta_a + \theta_b)}$.
   - Readout / Cleanup: Inner product against codebook $\mathcal{C} = e^{i\theta}$:
     $$\text{logits}_k = \frac{\beta}{D} \operatorname{Re}\left(\sum_{j=1}^D z_{\text{out}, j} \cdot z_{k, j}^*\right)$$
   - Scale factor $\beta$ learned.

4. **`block_unitary` (Non-Abelian Lie Group $U(b)^{D/b}$):**
   - Block size $b = 4$ (natural representation of $S_5$ in $\mathbb{R}^4 / \mathbb{C}^4$), $D/b = 32$ blocks.
   - Each group element $\sigma$ maps to block-unitary matrices $U_\sigma^{(m)} \in U(4)$ parameterized via Lie algebra exponential: $U = \exp(A - A^\dagger)$ with anti-symmetric $A$.
   - Group operator: Block matrix multiplication: $U_{\text{out}}^{(m)} = U_{\sigma_1}^{(m)} U_{\sigma_2}^{(m)}$.
   - Readout / Cleanup: Trace inner product against dictionary:
     $$\text{logits}_k = \frac{\beta}{D} \sum_{m=1}^{D/b} \operatorname{Re} \operatorname{Tr}\left(\left(U_k^{(m)}\right)^\dagger U_{\text{out}}^{(m)}\right)$$

---

## 4. Pre-Registered Hypotheses and Decision Rules

| Hypothesis | Description | Success Criterion | Falsified If |
|---|---|---|---|
| **H1** | `torus_phase` beats baselines on $\mathbb{Z}_{113}$ in sample efficiency | $\ge 2\times$ fewer training steps to $99\%$ test acc; paired test $p < 0.01$, Cohen's $d_z > 1.0$ | Steps ratio $< 1.5\times$ or $p \ge 0.05$ |
| **H2** | `torus_phase` fails on non-abelian $S_5$ (honest negative control) | Test accuracy $\le 25\%$ (due to commutativity conflict) | Test accuracy $> 50\%$ |
| **H3** | `block_unitary` solves $S_5$ where `torus_phase` fails | Test accuracy $\ge 90\%$ on $S_5$; $p < 0.001$ over `torus_phase` | Test accuracy $< 75\%$ |
| **H4** | Learned phases on $\mathbb{Z}_{113}$ show spectral concentration | Participation ratio of Fourier frequencies $\le 10$ (out of 113) | Participation ratio $> 30$ |

### Decision Rules:
* **Adopt `TorusPhase` for commutative/relational memory modules** if H1 passes and H2 confirms the commutativity boundary.
* **Adopt `BlockUnitary` for sequential/algorithmic modules** if H3 passes.
* **If both H1 and H3 fail:** Conclude that explicit unitary/phase inductive bias does not outperform general gradient-based weight decay on group discovery, and halt geometric transition development.

---

## 5. Hyperparameter Tuning and Protocol
- **Tuning Grid:** Learning rates $[10^{-3}, 3\times 10^{-3}, 10^{-2}, 3\times 10^{-2}]$ evaluated on Dev Seeds $100, 101, 102$.
- **Boundary Check:** If the optimal learning rate is on a grid boundary, extend the grid by one factor of 3 and re-tune.
- **Evaluation Seeds:** 10 fresh, unused test seeds: $0, 1, 2, 3, 4, 5, 6, 7, 8, 9$.
- **Metrics Computed:** Final test accuracy (mean $\pm 95\%$ CI), steps to $99\%$ accuracy, effective rank, Wang-Isola uniformity, and Fourier participation ratio.
- **Integrity Rule:** `results.json` must have a modification timestamp strictly after this `PREREG.md` file. Tables must be rendered directly from JSON without human editing.

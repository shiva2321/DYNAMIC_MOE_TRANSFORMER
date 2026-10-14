# Pre-Registration: Phase 1 - Real Latent Geometry Health and Superposition Capacity

**Date:** 2026-10-14  
**Track:** Z (Geometry Core Verification)  
**Status:** Pre-registered before any experimental execution  
**Target Results File:** `research/geomlab_results/phase1/results.json`

---

## 1. Objective and Research Questions
This experiment tests whether geometric representations (hypersphere $\mathbb{S}^{D-1}$, flat Clifford torus $\mathbb{T}^D$, and unconstrained Euclidean $\mathbb{R}^D$) maintain their geometric health and theoretical working memory capacity when trained on real continuous data, or whether they suffer from representation collapse.

Specifically:
1. **Does standard unconstrained Euclidean training on real text cause anisotropic representation collapse (the "cone effect") where effective rank drops far below $D$?**
2. **Does constraining representations to a unit hypersphere ($\mathbb{S}^{D-1}$) or flat torus ($\mathbb{T}^D$) preserve effective rank and uniformity on real text?**
3. **What is the real-world capacity tax?** On random vectors, superposition capacity scales as $M_{\max} \approx \frac{D}{7.5 \ln N}$. When vectors represent real semantic data (Banking77 customer text with natural clustering and correlations), how much superposition capacity is preserved before crosstalk causes retrieval failure?

---

## 2. Dataset and Text Representation

- **Dataset:** Banking77 (10,003 real customer banking queries across 77 intent classes, loaded from local cache).
- **Split:** $8,000$ training queries, $2,003$ validation queries.
- **Evaluation Codebook:** $N = 1,000$ distinct text samples sampled uniformly across all 77 intent classes.
- **Encoder:** Standard 2-layer text encoder with character n-gram embedding hashing (HashFrontend with dimension 1,024) followed by a learned linear projection layer with residual connection and LayerNorm mapping to bottleneck dimension $D$.
- **Dimensions Tested:** $D \in \{128, 256, 512\}$.

---

## 3. Arms and Model Architectures (Matched Budgets & Fair Implementation)

All competitors are implemented according to standard modern practices with their own hyperparameter tuning. No competitor is crippled or artificially restricted.

1. **`euclidean_baseline` (Standard Unconstrained Representation):**
   - Projection layer maps to $z \in \mathbb{R}^D$ without norm constraint.
   - Classifier: Linear readout $W \in \mathbb{R}^{77 \times D}$ with bias.
   - Loss: Cross-entropy.
   - Optimizer: AdamW with tuned learning rate and standard weight decay $0.01$.

2. **`spherical_norm` (Unit Hypersphere $\mathbb{S}^{D-1}$ Representation):**
   - Projection layer followed by exact $L_2$ normalization: $z = x / \|x\|_2 \in \mathbb{S}^{D-1}$.
   - Classifier: Cosine similarity readout $\beta \cdot \frac{W z}{\|W\|_2}$ with learned inverse temperature $\beta$.
   - Loss: Cross-entropy.
   - Optimizer: AdamW with tuned learning rate and weights normalized after each step.

3. **`torus_phase` (Flat Clifford Torus $\mathbb{T}^D$ Representation):**
   - Projection layer outputs real pairs $(u_j, v_j)$, mapped to unit phasors $z_j = (u_j + i v_j) / |u_j + i v_j| \in S^1$.
   - Latent state lives on $\mathbb{T}^D = (S^1)^D$.
   - Classifier: Complex trace / cosine similarity against learned class prototypes with learned scale $\beta$.
   - Loss: Cross-entropy.
   - Optimizer: AdamW with tuned learning rate.

4. **`random_haar` (Theoretical Reference Ceiling):**
   - $N = 1,000$ independent Haar-random unit vectors on $\mathbb{S}^{D-1}$.
   - Represents the theoretical upper bound of unclustered, uncorrelated representation capacity.

---

## 4. Evaluation Tasks and Metrics

For each trained model on each seed, we extract representations of the $N = 1,000$ evaluation texts:

1. **Geometry Health Metrics:**
   - **Effective Rank ($\operatorname{erank}$):** Entropy of normalized singular values $\exp(-\sum p_i \ln p_i)$. Measures effective dimensionality (maximum is $\min(N, D) = D$).
   - **Wang-Isola Uniformity Loss:** $\mathcal{L}_{\text{unif}} = \ln \mathbb{E}_{i \neq j} \exp(-2 \|z_i - z_j\|^2)$. Lower is more uniform (isotropic).
   - **Mutual Coherence ($\mu$):** Maximum absolute pairwise cosine similarity $\max_{i \neq j} |\cos(z_i, z_j)|$, compared to the random-matrix Welch bound $\mu_0 = \sqrt{\frac{2 \ln(N(N-1)/2)}{D}}$.

2. **Superposition Working Memory Capacity:**
   - For bundle sizes $M \in \{1, 2, 4, 8, 16, 32\}$:
     - Sample $M$ distinct items from the $N = 1,000$ codebook.
     - Form memory bundle:
       - For Euclidean / Spherical: $S = \sum_{m=1}^M x_m$, normalized to unit length.
       - For Torus: $S_j = \sum_{m=1}^M z_{m, j} / |\sum_{m=1}^M z_{m, j}|$.
     - Retrieval query: For each of the $M$ items, compute similarity of all $N$ items to bundle $S$.
     - Success criterion: Item $x_m$ is correctly identified among the top-$M$ nearest neighbors.
     - Compute exact recovery rate across $100$ independent bundle trials per seed.

---

## 5. Pre-Registered Hypotheses and Decision Rules

| Hypothesis | Description | Success Criterion | Falsified If |
|---|---|---|---|
| **H1** | Unconstrained Euclidean representations suffer from the cone effect | `euclidean_baseline` effective rank $\le 65\%$ of $D$ at $D=256, 512$ | Effective rank $> 80\%$ of $D$ |
| **H2** | Spherical and Torus manifolds preserve effective rank and uniformity | `spherical_norm` and `torus_phase` effective rank $\ge 85\%$ of $D$; uniformity loss significantly lower ($p < 0.01$) | Effective rank $< 75\%$ of $D$ |
| **H3** | Real data capacity tax: real clustering reduces capacity relative to Haar-random | Real data bundle recovery at $M=8$ ($D=256$) is lower than `random_haar` by $\ge 10$ percentage points | Real data matches Haar within $\pm 3\%$ |
| **H4** | Manifold models retain higher superposition capacity than Euclidean | At $D=256$, `spherical_norm` and `torus_phase` beat `euclidean_baseline` on bundle recovery at $M=4, 8$ by $\ge 15$ percentage points ($p < 0.01$) | Difference $< 5$ percentage points or $p \ge 0.05$ |

### Decision Rules:
- **Adopt Spherical / Toroidal normalizations into the Geometry Core** if H2 and H4 pass (proving that manifold constraints prevent representation collapse and protect working memory capacity on real data).
- **If H1 is rejected (Euclidean representations do not collapse):** Conclude that standard AdamW + LayerNorm already prevents representation collapse on real text classification, and do not mandate explicit manifold normalization for static representations.

---

## 6. Hyperparameter Tuning and Protocol
- **Tuning Grid:** Learning rates $[10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}]$ evaluated on Dev Seeds $100, 101, 102$.
- **Evaluation Seeds:** 10 fresh, unused test seeds: $0, 1, 2, 3, 4, 5, 6, 7, 8, 9$.
- **Statistical Tests:** Paired two-sided Student's t-test, Wilcoxon signed-rank test, Cohen's $d_z$, and $95\%$ Student-t confidence intervals.
- **Integrity Rule:** `results.json` must have a modification timestamp strictly after this `PREREG.md` file. Tables must be rendered directly from JSON without human editing.

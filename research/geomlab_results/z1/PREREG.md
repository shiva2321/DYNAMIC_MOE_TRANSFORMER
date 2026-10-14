# Round Z1 Pre-Registration: Spherical Residual Stream (nGPT Family)

**Date & Time:** 2026-10-06T09:25:00Z
**Track:** Track Z (Geometry Atlas Verification)
**Round:** Z1
**Status:** PRE-REGISTERED (written before running any experiments or code)

---

## 1. Objective and Research Question
To determine whether constraining token representations and residual updates to the unit hypersphere $S^{d-1}$ with learned per-dimension step sizes (the `nGPT` architecture; arXiv:2410.01131, 2608.01284) delivers a net wall-clock and token efficiency speedup over a standard modern decoder-only Transformer baseline, and whether any observed advantage stems from hypersphere geometry or merely from additional learnable scale parameters.

---

## 2. Model Architecture & Task Setup

### 2.1 Task and Corpus
- **Corpus:** Python standard library source files from `sys.base_prefix / "Lib"`.
- **Whole-File Splits:**
  - Training set: `ast.py`, `dis.py`, `inspect.py`, `tokenize.py`, `difflib.py`, `pydoc.py`, `tarfile.py`, `zipfile.py` (approx. 600,000 characters).
  - Held-out validation set: `heapq.py`, `calendar.py`, `bisect.py`, `queue.py` (approx. 60,000 characters).
- **Tokenization:** Character-level ASCII (printable ASCII 32..126 plus newline and tab; vocabulary size ~97 tokens).
- **Sequence Length ($T$):** 256.
- **Batch Size ($B$):** 32 sequences per batch ($8,192$ tokens/batch).
- **Evaluation Schedule:** Fixed held-out evaluation on 16 validation batches ($131,072$ tokens) evaluated every 100 steps.
- **Training Budget:** 1,000 optimizer steps ($8,192,000$ tokens per run).

### 2.2 Model Specifications (<= 8M parameters)
- **Layers ($L$):** 6 transformer decoder blocks.
- **Hidden Dimension ($d$):** 256.
- **Attention Heads ($H$):** 4 heads ($d_{\text{head}} = 64$).
- **MLP Dimension ($d_{\text{mlp}}$):** 768 ($3 \times d$).
- **Total Parameters:** ~4.8M parameters (strictly within the $\le 8$M limit).

---

## 3. Experimental Arms

1. **`base`**:
   - Standard decoder-only Transformer: RoPE position embeddings, pre-RMSNorm, SwiGLU MLP, AdamW optimizer.
   - Standard unconstrained residual stream: $x \leftarrow x + h_A$; $x \leftarrow x + h_M$.
   - Baseline weight decay tuned over $\{0.0, 0.01, 0.1\}$.

2. **`base_qknorm`**:
   - `base` equipped with QK-normalization ($L_2$ unit-normalization on queries and keys before attention dot product).

3. **`base_scaled` (Isolating Control Arm)**:
   - Identical to `base`, but equipped with the exact same learnable per-dimension step sizes $\alpha_A, \alpha_M \in \mathbb{R}^d$ and scaling vectors $s_{qk}, s_u, s_v, s_z$ as `ngpt`.
   - **NO hypersphere normalizations**: standard unconstrained residual accumulation $x \leftarrow x + \alpha \odot h$, free weight norms, standard RMSNorm.
   - Purpose: Isolates whether gains arise from learnable per-channel scale capacity or from spherical geometry.

4. **`ngpt` (Full nGPT)**:
   - Unit-norm embeddings: $\|E_v\|_2 = 1$.
   - Unit-norm residual stream: $x \in S^{d-1}$ at every layer.
   - Residual updates: $x \leftarrow \text{Norm}(x + \alpha_A \odot (h_A - x))$ and $x \leftarrow \text{Norm}(x + \alpha_M \odot (h_M - x))$.
   - Linear layers: All weight vectors normalized to unit norm after every optimizer step ($\|W_j\|_2 = 1$).
   - Weight decay: $0.0$ (weights are explicitly projected).
   - Attention: Unit-normalized Q, K, V with learned scaling $s_{qk}$.
   - Logits: $y = s_z \cdot (x W_{\text{vocab}}^T)$ with unit-norm vocabulary vectors.

5. **`ngpt_stream_only`**:
   - Spherical residual stream on $S^{d-1}$ with learned step sizes $\alpha_A, \alpha_M$.
   - Free weight norms (standard unnormalized linear weights, weight decay enabled).

6. **`base_muon`**:
   - `base` model trained with Muon optimizer on 2D linear weight matrices and AdamW on 1D parameters/embeddings.

7. **`ngpt_muon`**:
   - `ngpt` model trained with Muon optimizer on 2D linear weight matrices (maintaining post-step unit-normalization) and AdamW on 1D parameters/scales.

8. **`ngpt_salience` (Diagnostic Arm for Atlas P-1)**:
   - Spherical representation on $S^{d-1}$ paired with an explicit scalar salience channel $r \in \mathbb{R}_+$.

---

## 4. Hyperparameter Grids & Tuning Protocol

- **Dev Seeds:** 100, 101 used exclusively for hyperparameter tuning.
- **Test Seeds:** 0 through 9 (10 seeds) evaluated once on the selected best configuration.
- **Baseline Weight Decay Grid:** $\{0.0, 0.01, 0.1\}$ evaluated on dev seeds for `base`, `base_qknorm`, `base_scaled`, `base_muon`.
- **Learning Rate Grids:**
  - Standard AdamW arms: $\{1 \times 10^{-4}, 3 \times 10^{-4}, 1 \times 10^{-3}, 3 \times 10^{-3}\}$.
    - If best validation loss lands on an edge value, the grid must be extended by factor $3\times$ outward and re-evaluated until strictly internal.
  - Muon 2D weight LR grid: $\{0.005, 0.01, 0.02, 0.04\}$ (AdamW 1D LR fixed at $1 \times 10^{-3}$).
- **Selection Criterion:** Minimum final validation loss averaged over dev seeds.

---

## 5. Metric Definitions

1. **Smoothed Validation Loss:**
   At evaluation checkpoint step $t$, smoothed validation loss $\bar{L}(t)$ is the mean of validation loss over the last 3 evaluations:
   $$\bar{L}(t) = \frac{1}{3} \sum_{k=0}^2 L(t - 100k)$$
2. **Speed-Up (Tokens to Baseline Final Loss):**
   - For seed $s$, let $L^*_{\text{base}, s}$ be the final smoothed validation loss of `base` at step 1000.
   - For arm $A$ on seed $s$, let $t^*_{A, s}$ be the earliest checkpoint where $\bar{L}_{A, s}(t) \le L^*_{\text{base}, s}$.
   - Token Speedup: $\text{Speedup}_{\text{tokens}} = \frac{1000 \times 8192}{t^*_{A, s} \times 8192} = \frac{1000}{t^*_{A, s}}$.
   - Net Wall-Clock Speedup: $\text{Speedup}_{\text{wallclock}} = \frac{\text{Time}_{\text{base}, s}(\text{to step 1000})}{\text{Time}_{A, s}(\text{to step } t^*_{A, s})}$.
   - If an arm does not reach $L^*_{\text{base}, s}$, $t^*_{A, s}$ is assigned 1000 (speedup = 1.0) and failure to reach target is explicitly recorded.
3. **MQAR Synthetic Recall:**
   - Multi-Query Associative Recall benchmark with 4 key-value pairs per sequence, sequence length 64, evaluating greedy retrieval accuracy over 200 held-out examples.
4. **Geometry Health Diagnostics (from `geomlab/diagnostics.py`):**
   - Residual stream effective rank ($\text{erank}$).
   - Stream isotropy / hypersphere uniformity loss.
   - TwoNN intrinsic dimensionality ($\text{id}$).
   - Hubness skewness ($k=5$).
   - Norm statistics: mean L2 norm, std, drift.

---

## 6. Pre-Registered Predictions

1. **Prediction 1 (Token Efficiency):** Full `ngpt` reaches `base`'s final validation loss in fewer tokens (token speedup ratio $\ge 1.30\times$).
2. **Prediction 2 (Isolating Control):** If `ngpt` outperforms `base`, `base_scaled` will account for a portion of the gain; however, spherical normalization provides additional optimization stability (lower gradient variance and zero norm drift).
3. **Prediction 3 (Wall-Clock Speedup):** Due to normalization and step size operations, per-step overhead of `ngpt` will be $\le 15\%$ over `base`. Net wall-clock speedup will be $\ge 1.15\times$.
4. **Prediction 4 (Norm Drift Invariance):** Residual stream vector L2 norms for `ngpt` will be $1.000 \pm 0.000$ (drift std $< 10^{-6}$) across all layers, whereas `base` will exhibit growing activation norms across depth.
5. **Prediction 5 (MQAR Recall):** `ngpt` matches or exceeds `base` recall accuracy ($\ge 95\%$) on in-context associative recall.

---

## 7. Decision Rule (Scope Charter & Atlas Core)
- **ADOPT `SphericalStream` into the Geometry Core** if and only if:
  1. Net wall-clock speedup of `ngpt` relative to `base` is $\ge 1.15\times$, AND
  2. The paired 95% confidence interval for wall-clock speedup is strictly above $1.0$, AND
  3. `ngpt` matches or exceeds `base_scaled` (confirming benefit from spherical geometry rather than merely extra scale parameters).
- **OTHERWISE: REJECT / KEEP AS IS** (retain standard RMSNorm residual stream in the Architecture Core).

---

## 8. Kill Rules & Fences
- If `ngpt` diverges across all tested learning rates on dev seeds, halt, log divergence and LR grid, and report immediately.
- Peak VRAM must remain $< 4$ GB.
- No editing of this `PREREG.md` file once results are generated. Any unexpected outcome or bug will be logged in `AMENDMENT.md`.

# Pre-Registration: Track Z Round Z2 (Memory Geometry: Recall Per Byte)

**Date:** 2026-10-09  
**Track:** Track Z (Geometry Core)  
**Round:** Z2  
**Topic:** Associative Memory Capacity and Recall per Byte  

---

## 1. Scientific Objective
Determine empirically which associative memory geometry provides the highest Multi-Query Associative Recall (MQAR) accuracy per state byte, the best length extrapolation, and lowest computational overhead. Evaluate whether modern geometric memory formulations (normalized-key delta rules, self-dual PSD cones, triadic tensor states, and PoST geometric decay) break the associative memory capacity wall, and benchmark where the legacy ISHMN von Mises-Fisher spherical memory (`InfiniteSphericalMemory` in `ishmn/memory.py`) stands under strict equal-byte controls.

---

## 2. Experimental Arms
All arms are wrapped in an identical sequence model: `Embedding -> 2 x [Mixer + MLP] -> Unembedding`, with sequence dimension $d = 64$ or $128$, feed-forward dimension $4d$, and equal parameter counts ($\sim 0.5\text{M} - 1.0\text{M}$).

1. **`softmax` (Reference Ceiling):** Standard causal scaled dot-product attention with full KV cache ($O(T \cdot d)$ state bytes per sequence).
2. **`linear`:** Standard causal linear attention with positive feature maps $\phi(x) = \text{elu}(x) + 1$, matrix state $S \in \mathbb{R}^{d \times d}$.
3. **`gated_delta`:** Gated DeltaNet state $S_t = S_{t-1}(\alpha_t (I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T$ with learned decay $\alpha_t \in (0, 1)$ and write rate $\beta_t \in (0, 1)$.
4. **`delta_l2`:** Gated delta rule with strictly $\ell_2$-normalized keys ($k_t / \|k_t\|_2$). The erasure operator $(I - \beta k k^T)$ becomes an exact projection operator on the unit sphere.
5. **`kata_psd`:** Self-dual homogeneous cone (PSD) feature mapping (arXiv 2607.17419). Keys and queries are mapped onto rank-one positive semi-definite cone features, expanding associative capacity beyond the linear $d$ wall.
6. **`triadic`:** Triadic / tensor-product state attention (arXiv 2609.36529). Evaluates coupled bilinear multi-head state with effective state capacity scaling toward $d^2$.
7. **`post_decay`:** Gated delta rule with Power-Spectrum of Time (PoST, arXiv 2604.07658) geometrically spaced log-decay rates $\log \alpha_i$ spanning timescales from short-term to long-term memory.
8. **`legacy_vmf`:** The legacy ISHMN memory imported read-only from `ishmn/memory.py` behind a thin non-intrusive adapter in `geomlab/memory_adapters.py`.

---

## 3. Tasks & Benchmarks
- **Task 1: Multi-Query Associative Recall (MQAR)**
  - Synthetic sequence generation with $P$ key-value pairs placed randomly in context, followed by queries requesting values associated with earlier keys.
  - Pair sweep: $P \in \{4, 8, 16, 32, 64\}$.
  - Training context length $T = 128$.
  - Evaluation metric: Recall Accuracy (%) at equal state byte footprint.
- **Task 2: Length Extrapolation (Out-of-Distribution Context)**
  - Models trained at $T = 128$ evaluated without fine-tuning at test context lengths $T \in \{256, 512, 1024, 2048\}$.
  - Metric: Degradation ratio / Extrapolation Recall (%).
- **Task 3: Efficiency & Throughput**
  - Execution runtime in ms/step and peak VRAM allocated.

---

## 4. Hypotheses & Directional Predictions
1. **Key Normalization (`delta_l2` > `gated_delta`):** Projecting keys to the unit sphere makes erasure $(I - k k^T)$ an exact orthogonal projection, preventing associative crosstalk and boosting recall on dense pairs.
2. **Cone / Tensor Expansion (`kata_psd`, `triadic` > `linear`):** Quadratic and cone feature maps break the capacity wall of linear state $d$, maintaining high accuracy when $P \ge d/2$.
3. **Decay Geometry (`post_decay` > random decay):** Geometrically spaced decay rates provide multi-timescale retention that significantly reduces forgetting during length extrapolation ($T \ge 512$).
4. **Legacy Memory Position (`legacy_vmf`):** Continuous von Mises-Fisher + radial addressing with discrete slot allocation will demonstrate lower recall density per state byte compared to continuous delta matrix states on dense sequence retrieval.

---

## 5. Decision & Kill Rules
- **Component Selection Rule:** Adopt the memory architecture achieving the highest MQAR recall accuracy at equal state bytes with paired statistical significance ($p < 0.05$ Bonferroni-corrected, Cohen's $d_z > 0.8$).
- **Legacy Memory Rule:** Report `legacy_vmf` performance honestly without handicap or preferential tuning.
- **Kill Rule:** If an associative memory arm cannot recall $100\%$ on orthogonal synthetic pairs with $P \le d$, it is rejected as mathematically defective.

---

## 6. Optimization & Experimental Protocol
- **Device:** Single NVIDIA GeForce RTX 3060 (12 GB).
- **Precision:** `bfloat16` with float32 state accumulation and normalization.
- **Seeds:**
  - Tuning seeds (dev): 100, 101.
  - Evaluation seeds (test): 0 through 9 (10 seeds total).
- **Hyperparameter Grid:**
  - Learning rates: `[1e-4, 3e-4, 1e-3, 3e-3]`.
  - Boundary check: If optimal LR lies on an edge, grid is expanded by factor 3.
  - Weight decay: `0.01` (baselines) / `0.0` (normalized states).
- **Reporting:** Verbatim output tables rendered from `results.json`, validated by `validate_report()`. Zero hand-editing of results.

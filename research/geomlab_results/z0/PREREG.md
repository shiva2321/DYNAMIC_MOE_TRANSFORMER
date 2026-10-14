# Pre-Registration: Round Z0 Infrastructure & Diagnostics

**Date:** 2026-10-06
**Round:** Z0 (Protocol library & geometry-health diagnostics)
**Status:** Pre-registered before any code execution or results generation

---

## 1. Goal and Scope
Build and verify the foundation for all subsequent Track Z rounds:
1. `geomlab/protocol.py`: Automated experimental protocol tools ensuring zero hand-edits, enforced pre-registration verification, automated per-arm hyperparameter tuning with grid-boundary checks, seed execution, paired statistical significance testing (t-test, Wilcoxon, Cohen's d_z, Bonferroni correction), JSON results logging, verbatim table rendering, and strict report diff validation.
2. `geomlab/diagnostics.py`: Geometry-health telemetry suite measuring effective rank (entropy of normalized singular values), spherical isotropy and Wang-Isola uniformity loss, TwoNN intrinsic dimensionality estimation, hubness skewness, norm distribution statistics, and an adapter for YggFormer operating-point audits.
3. Verification test suite in `tests/geomlab/` (strictly no `__init__.py`).

No model training on GPUs and no scientific claims are made in this round.

---

## 2. Verification Arms and Tasks

### Arm 1: Diagnostics Correctness
- **Effective Rank:**
  - Task A: Isotropic standard Gaussian matrix $X \sim \mathcal{N}(0, I_d)$ in $d=32, N=5000$.
  - Task B: Rank-1 line $X = v \cdot u^T + \epsilon$ in $d=32, N=5000$.
- **TwoNN Intrinsic Dimension:**
  - Task A: 3-dimensional manifold embedded in $\mathbb{R}^{20}$ (e.g., 3-sphere $S^3$ or 3D ball, $N=3000$).
  - Task B: 1-dimensional manifold (circle $S^1$) embedded in $\mathbb{R}^{20}$.
- **Spherical Isotropy & Uniformity:**
  - Task A: Uniform spherical distribution on $S^{d-1}$.
  - Task B: Highly anisotropic / collapsed cluster on $S^{d-1}$.
- **Hubness Skewness:**
  - Task A: Independent isotropic coordinates (low skew).
  - Task B: Clustered manifold points with hubs (skew > 0).
- **Norm Statistics:**
  - Mean, standard deviation, median, min, max on known norm vectors.
- **Operating Point Adapter:**
  - Compatible with `yggformer.report.operating_point` on `YggFormer` models and graceful fallback on generic modules.

### Arm 2: Protocol Integrity and Honesty Tools
- **Pre-registration Checker (`check_prereg`):**
  - Case A: PREREG file exists and modification timestamp is strictly earlier than results file $\to$ PASS.
  - Case B: PREREG file missing $\to$ FAIL (raises FileNotFoundError / ValidationError).
  - Case C: PREREG file modified after or at same time as results file $\to$ FAIL (raises ValidationError).
- **Report Validator (`validate_report`):**
  - Case A: Report containing rendered table matches JSON-rendered table verbatim $\to$ PASS.
  - Case B: Tampered report where a single digit or character in the table is modified $\to$ FAIL (raises ValidationError with unified diff).
- **Dry-run Table Reproduction:**
  - Verbatim reproduction of historical benchmark table `research/yggdrasil_results/poc_b_table.txt` directly from `research/yggdrasil_results/poc_b/res_*.json` and `lr_*.json` files byte-for-byte.
- **Learning Rate Tuner (`tune_lr`):**
  - Log-spaced grid evaluation on dev seeds $\ge 100$.
  - If optimal learning rate lands on grid boundary (min or max), trigger edge warning and automatically expand grid until interior minimum found or maximum expansion reached.
- **Paired Statistics & Bonferroni:**
  - Paired Student's t-test, Wilcoxon signed-rank test, Cohen's $d_z$, and Bonferroni family-wise error rate control.

---

## 3. Predictions and Pass Criteria

1. `effective_rank(isotropic_gaussian)` $\ge 30.0$ for $d=32$; `effective_rank(rank1)` $\le 1.10$.
2. `intrinsic_dim_twonn(3d_manifold)` in range $[2.7, 3.3]$ for true dimension 3.
3. `intrinsic_dim_twonn(1d_manifold)` in range $[0.8, 1.2]$ for true dimension 1.
4. `isotropy(uniform)` mean pairwise cosine within $\pm 0.05$; uniformity loss matches theoretical expectation.
5. `validate_report()` catches single-character table corruption with 100% sensitivity.
6. `check_prereg()` catches missing PREREG or backward timestamps with 100% sensitivity.
7. Dry-run reproduces `poc_b_table.txt` from stored JSON byte-for-byte (exact string match).

---

## 4. Decision Rule
- **ADOPT:** If all protocol tools and diagnostic metrics satisfy all verification assertions in `tests/geomlab/` and pass the existing geometry test suite (112 tests) and GrowMoE suite (240 tests) with zero regressions.
- **REJECT / FIX:** If any diagnostic deviates from its theoretical properties or if any protocol validator allows table or timestamp tampering.

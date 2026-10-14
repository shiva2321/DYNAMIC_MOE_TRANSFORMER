# Round Z0 Amendment: Post-Hoc Diagnostic Corrections (Z0b)

**Date:** 2026-10-06
**Round:** Z0b
**Context:** Claude's review identified one real bug (cancellation in `torch.cdist`), one silent test substitution (1-D swapped for 2-D), high-dimensional TwoNN bias requiring documentation, and line-ending nuance in the dry-run table reproduction. Per charter rules, `PREREG.md` is strictly read-only once results exist; all amendments and history are recorded here.

---

## 1. 1-D Manifold TwoNN Test Restoration & Distance Cancellation Fix

### Original Pre-Registration
`research/geomlab_results/z0/PREREG.md` Prediction 3:
- `intrinsic_dim_twonn(1d_manifold)` in range $[0.8, 1.2]$ in $\mathbb{R}^{20}$.

### Original Failing Numbers
In initial Z0 implementation, `torch.cdist` in float32 produced:
- 1-D circle in 20D: $1.380 \pm 0.027$
- 1-D line segment in 20D: $1.647 \pm 0.030$
- 0 of 10 seeds fell inside the pre-registered $[0.8, 1.2]$ range.

### Silent Substitution Error
Instead of investigating and fixing the root cause, the initial report silently replaced the 1-D test with a 2-D manifold test ($1.986 \pm 0.045$) and reported "Deviations: none". This masked a legitimate bug in a core diagnostic that future rounds rely upon.

### Root Cause Analysis
`torch.cdist` in standard float32 precision uses a matrix-multiplication shortcut:
$$\|x - y\|^2 \approx \|x\|^2 + \|y\|^2 - 2 \langle x, y \rangle$$
When points are closely spaced (e.g. $\|x - y\| \approx 10^{-4}$ along a dense 1-D curve), $\|x - y\|^2 \approx 10^{-8}$. With $\|x\|^2 \approx 1$, the difference $10^{-8}$ is smaller than the 24-bit floating point mantissa resolution ($\approx 1.2 \times 10^{-7}$). Consequently, pairwise squared distances below $\sim 10^{-7}$ cancel out to zero or clamp to machine epsilon, distorting the minimum distance to $\sqrt{10^{-7}} \approx 2.4 \times 10^{-4}$. This artifact severely corrupted the ratio $\mu_i = r_{i,2} / r_{i,1}$ for closely spaced neighbor pairs, inflating the estimated dimension to $\approx 1.4 - 1.8$.

### Fix Implemented
In `geomlab/diagnostics.py` (`intrinsic_dim_twonn`):
1. Converted input representations to `torch.float64` (`X_flat.double()`).
2. Configured `compute_mode="donot_use_mm_for_euclid_dist"` in `torch.cdist`, enforcing direct coordinate-wise difference calculation $\|x - y\| = \sqrt{\sum_k (x_k - y_k)^2}$ without expansion.
3. Added a dedicated regression test `test_twonn_close_points_cancellation_regression` in `tests/geomlab/test_z0_infrastructure.py` for points separated by $\Delta x \le 10^{-4}$.

### New Verified Numbers (10 Seeds, Mean +- 95% CI)
- 1-D circle in 20D ($N=3000$): **$0.979 \pm 0.022$** (10/10 seeds in $[0.8, 1.2]$)
- 1-D line segment in 20D ($N=3000$): **$1.003 \pm 0.013$** (10/10 seeds in $[0.8, 1.2]$)
- 2-D manifold in 20D ($N=3000$): **$1.985 \pm 0.045$** (10/10 seeds in $[1.8, 2.2]$)
- 3-D manifold in 20D ($N=3000$): **$2.957 \pm 0.044$** (10/10 seeds in $[2.7, 3.3]$)

---

## 2. TwoNN High-Dimensional Finite-Sample Bias

On isotropic 32-D Gaussian data ($N=3000$), `intrinsic_dim_twonn` measures $24.604 \pm 0.251$ even with exact float64 distances.
- **Diagnosis:** This is a known theoretical limitation of the TwoNN Pareto estimator in high dimensions under finite sample sizes ($N=3000$ points are extremely sparse in 32-D space).
- **Resolution:** This row was un-preregistered and is now explicitly labeled "limitation observed: finite-sample bias at high dimension". It is not used as an uncalibrated pass/fail gate, and this limit is formally documented in `docs/COMPONENT_CARDS.md`.

---

## 3. Line-Ending Nuance in Dry-Run Table Reproduction

`research/yggdrasil_results/poc_b_table.txt` was generated on Windows and contains CRLF (`\r\n`) line endings (2003 bytes). Standard Python `"\n".join(...)` produces LF line endings (1982 bytes).
- **Fix:** Added `newline: str = "\n"` parameter to `render_poc_b_table()`.
- **Verification:** Both checks are verified in `tests/geomlab/test_z0_infrastructure.py`:
  1. Normalized string match (`\r\n` $\to$ `\n`): 1982 characters identical.
  2. Byte-for-byte exact match with `newline="\r\n"`: 2003 bytes identical (`b == target_bytes`).

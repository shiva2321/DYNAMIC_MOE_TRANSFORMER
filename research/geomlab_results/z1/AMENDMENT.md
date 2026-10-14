# Amendment to Round Z1 Pre-Registration

**Date:** 2026-10-09  
**Track:** Track Z (Geometry Atlas Verification)  
**Round:** Z1 (`SphericalStream` / nGPT Family)

---

## 1. Context & Motivation
The original pre-registration (`PREREG.md`, dated 2026-10-06) specified an 8-arm comparison evaluated across 10 test seeds over 1,000 steps. In baseline eager execution in float32 (`fp32`), each 1,000-step run required ~103 ms/step (~103 seconds per seed), leading to a projected total execution time of 75–90 minutes.

To eliminate this bottleneck, leverage the local RTX 3060 Tensor Cores, and align with production LLM training conventions (which universally train in 16-bit mixed precision), this amendment introduces hardware-accelerated mixed precision.

---

## 2. Amendments to Execution & Protocol

### A. Mixed Precision Standardization (`bfloat16`)
- All forward passes and loss computations are executed under `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- All geometric vector normalizations (`l2_norm` and `normalize_weights_`) explicitly compute vector $L_2$ norms in `float32` before casting, preventing mantissa underflow or normalization drift.
- **Fairness Control:** To prevent false wall-clock speedups, all comparison arms—including the baseline `base`—are evaluated under identical `bfloat16` mixed precision from scratch.

### B. Core Decision Triad Focus
Per Section 7 of `PREREG.md`, the decision rule for adopting `SphericalStream` into the Geometry Core hinges strictly on the triad:
1. `base`: Standard decoder-only Transformer baseline.
2. `base_scaled`: Isolating control arm with identical learnable per-dimension step sizes and feature scalings without spherical normalization.
3. `ngpt`: Full unit-hypersphere residual stream and normalized linear weights.

The test evaluation prioritizes this core decision triad to obtain rigorous statistical answers (10 test seeds, paired 95% CIs, Cohen's $d_z$, TwoNN intrinsic dimension, effective rank) within 15–20 minutes of compute time.

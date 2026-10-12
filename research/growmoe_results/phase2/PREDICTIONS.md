# Phase 2 Pre-Registration: Predictions P7 and P9

**Protocol:** Pre-registered before any Task 19 benchmark stream execution (S5 / S6).
**Date:** 2026-10-06
**Track:** Track G (GrowMoE Phase 2)
**Scope:** Vertical scaling, StackMoE hierarchical composition, lifelong long streams (Tasks 18-19).

---

## 1. Prediction P7: Parameter Efficiency at Matched Accuracy under Vertical Growth

### 1.1 Hypothesis and Rationale
Hierarchical composition requires depth. On compositional tasks (Stream S5), shallow architectures must use exponentially wider layers to approximate composition, whereas vertically growing models (`StackMoE`) discover necessary representation layers dynamically, achieving matched accuracy with near-optimal parameter counts.

### 1.2 Operational Definitions
- **Task:** Stream S5 (hierarchical compositional stream with teacher depths $D \in \{1, 2, 3\}$, dimension $d=64$, 8 classes).
- **Teacher-Sized Optimum ($P_{\text{opt}}$):** The exact parameter count of the ground-truth teacher MLP (e.g. depth 2 or 3, hidden dimension equal to data dimension $d=64$) that generated the target regime. For depth 2: $P_{\text{opt}} \approx 2 \times 64 \times 64 + 64 \times 8 \approx 8{,}704$ parameters.
- **Matched Accuracy:** An evaluated arm achieves matched accuracy if its final evaluation accuracy is within $\pm 2.0$ percentage points of the reference teacher ceiling (or reaches $\ge 90\%$ of ceiling accuracy).
- **Parameter Ratio Metric:**
  $$\rho_{\text{params}} = \frac{P_{\text{student}}}{P_{\text{opt}}}$$
  evaluated across test seeds 0–9 at matched task accuracy.

### 1.3 Decision Rule (Three-State)
- **PASS:** Mean parameter ratio $\rho_{\text{params}} \le 1.50\text{x}$.
- **FAIL:** Mean parameter ratio $\rho_{\text{params}} > 2.00\text{x}$.
- **INCONCLUSIVE:** $1.50\text{x} < \rho_{\text{params}} \le 2.00\text{x}$.

---

## 2. Prediction P9: Routing Accuracy Degradation over Long Lifelong Streams

### 2.1 Hypothesis and Rationale
As lifelong streams grow from 10 to 60 regimes (Stream S6), flat routing architectures suffer from increased expert confusion. In contrast, hierarchical vertical gating and calibrated conformal gating maintain discriminability, keeping routing accuracy degradation bounded.

### 2.2 Operational Definitions
- **Task:** Stream S6 (`s6_long`: lifelong stream with $d=64$, scaling from $K_1 = 10$ regimes to $K_2 = 60$ regimes with latent subspaces, related regimes, and periodic revisits).
- **Routing Accuracy Metric:**
  Fraction of test samples assigned to an expert in the regime's designated owner set (experts that received $\ge 10\%$ of the regime's training samples).
- **Degradation Metric ($\Delta_{\text{routing}}$):**
  $$\Delta_{\text{routing}} = \text{Acc}_{\text{routing}}(K = 10) - \text{Acc}_{\text{routing}}(K = 60)$$
  measured in percentage points, paired across seeds 0–9.

### 2.3 Decision Rule (Three-State)
- **PASS:** $\Delta_{\text{routing}} < 5.0$ percentage points.
- **FAIL:** $\Delta_{\text{routing}} > 10.0$ percentage points.
- **INCONCLUSIVE:** $5.0 \le \Delta_{\text{routing}} \le 10.0$ percentage points.

---

## 3. Experimental Arms and Benchmark Configuration

### 3.1 Arms
1. **`StackMoE_grow` (Primary):** `StackMoE` with vertical representation growth (`add_layer`, `widen`, `deepen`) and `PhyloMoE` head pool.
2. **`StackMoE_fixed` (Ablation):** `StackMoE` with fixed representation depth (no `add_layer`), testing whether horizontal growth alone suffices.
3. **`PhyloMoE_flat` (Baseline A):** Standard single-layer `PhyloMoE` (Pool A).
4. **`ReplayNet` (Baseline B2):** Fixed-capacity shared network with experience replay.

### 3.2 Operating Point
- Conformal FDR threshold: $q = 0.02$ for all streams and pools.
- Routing Rule for Pool A: `accept_then_logf`.
- Evaluation seeds: Dev seeds 100–109 for any preliminary hyperparameter checks; Test seeds 0–9 for recorded evaluation.
- Statistics: 95% bootstrap confidence intervals, paired student t-test, and Wilcoxon signed-rank test.

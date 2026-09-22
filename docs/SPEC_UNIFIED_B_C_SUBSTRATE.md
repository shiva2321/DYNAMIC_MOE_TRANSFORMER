# SPEC-001: Unified Shared-Substrate Cognitive Architecture
## Fusion of Mode B (Latent Manifold Deliberation) + Mode C (Day/Night Dual-Phase Sleep) + Linguistic Cortex

- **Status**: APPROVED / IMPLEMENTATION READY
- **Author**: Antigravity Cognitive Architecture Team
- **Date**: 2026-09-21
- **Hardware Target**: Local NVIDIA RTX 3060 (12GB VRAM), FP16 / BF16
- **Reference Work**: Titans/MIRAS, Coconut, LeJEPA (Balestriero & LeCun 2025), Sleep-Replay CL (NREM/REM), FHRR / torchhd, DeltaNet / TTT

---

## 1. Executive Summary & Core Hypothesis

### 1.1 The Problem
In our preceding evaluation of SLM + Universal Cognitive Engine integration:
- **Mode B (Latent Sandwich)** inserted deep geometric deliberation into Layer 14 of an SLM, achieving System 2 dynamical relaxation (4.5–5.5 ticks/token), but operated on a frozen memory lattice.
- **Mode C (Day/Night Lifelong Cycle)** enabled zero-forgetting streaming ingestion and offline sleep consolidation, but interfaced with the language model solely via discrete surface tokens and episodic buffers.

Deploying them as two decoupled "bolt-ons" creates representational friction: the latent deliberation loop cannot directly modify or recruit dynamic memory states at test time, and nighttime consolidation cannot optimize the manifold deliberation trajectory without ground-truth external labels.

### 1.2 The Core Hypothesis (Shared-Substrate Deliberation & Memory)
> **Hypothesis $H_1$**: Unifying continuous latent deliberation ($\tau_t$) and long-term memory ($m_k$) into an **isomorphic hyperspherical coordinate substrate ($\mathbb{S}^7 \times \mathbb{R}^+$ / FHRR)**—gated by discrete epistemic verdicts during the day and refined via a **dual-phase NREM/REM generative sleep replay loop** at night—strictly outperforms decoupled Mode B and Mode C bolt-ons on multi-hop associative recall and continuous reasoning under a **matched parameter and FLOP compute budget**.

---

## 2. Mathematical Formulation of the Unified Substrate

```
                  ┌──────────────────────────────────────────────┐
                  │          UNIFIED SUBSTRATE SPACE             │
                  │             S^7 × R^+ (or FHRR)              │
                  └──────┬────────────────────────────────┬──────┘
                         │                                │
                         ▼                                ▼
            ┌─────────────────────────┐      ┌─────────────────────────┐
            │   THOUGHT TRAJECTORY    │      │     MEMORY LATTICE      │
            │   τ_t = (μ_t, ρ_t)      │<════>│     m_k = (μ_k, ρ_k)    │
            │   - ACT Ticks (1..6)    │  vMF │   - 5,281+ Invariants   │
            │   - JEPA Forward Rollout│ Dist │   - Dynamic MoE Anchors │
            └────────────┬────────────┘      └────────────┬────────────┘
                         │                                │
                         ▼ Daytime Write-Gate             ▼ Nighttime Consolidation
            ┌─────────────────────────┐      ┌─────────────────────────┐
            │   EPISTEMIC VERDICT     │      │   DUAL-PHASE SLEEP      │
            │  - KNOWN    (p_w = 0.0) │      │   1. NREM: Clustering   │
            │  - INFERRED (p_w = 0.2) │      │   2. REM: Generative    │
            │  - LEARNING (p_w = 1.0) │      │      Tangent Replay     │
            │  - REASONED (p_w = 0.8) │      │      (LeJEPA / SIGReg)  │
            └─────────────────────────┘      └─────────────────────────┘
```

### 2.1 State Isomorphism
Every active deliberative thought $\tau_t$ and every stored invariant memory node $m_k$ reside in the identical metric space:
$$\mathcal{M} = \mathbb{S}^7 \times \mathbb{R}^+ = \left\{ (\mu, \rho) \in \mathbb{R}^8 \times \mathbb{R} \;\Big|\; \|\mu\|_2 = 1, \; \rho \ge 0 \right\}$$
- $\mu \in \mathbb{S}^7$: Directional semantic orientation (concept identity, angle invariant to scale).
- $\rho \in \mathbb{R}^+$: Log-radial magnitude $\rho = \ln(1 + \|x\|)$ (scale equivariance from $10^{-3}$ to $10^{14}$).
- **Equivalence with FHRR**: In Fractional Holographic Reduced Representations, each component corresponds to complex unit phasors $e^{i \theta_j}$, isomorphic to products of circles $\mathbb{T}^8 \subset \mathbb{S}^7$.

Associative recall between current thought $\tau_t = (\mu_t, \rho_t)$ and memory nodes $m_k = (\mu_k, \rho_k)$ is governed by the geodesic von Mises-Fisher (vMF) kernel:
$$\mathcal{A}(\tau_t, m_k) = \exp\left( \kappa \cdot (\mu_t^\top \mu_k) - \lambda_\rho |\rho_t - \rho_k| \right)$$

### 2.2 Principled Epistemic Write-Gating (Titans / MIRAS Formalization)
Instead of an uninterpretable gradient-magnitude heuristic, the daytime write probability $p_{\text{write}}$ into the episodic memory buffer is computed from the **Abductive Surprise / Epistemic Verdict**:
$$p_{\text{write}}(\tau_t) = \begin{cases} 
0.00 & \text{if Verdict} = \text{KNOWN} \quad (\text{Free energy } \mathcal{F} < \epsilon_{\text{low}}) \\
0.20 & \text{if Verdict} = \text{INFERRED} \quad (\epsilon_{\text{low}} \le \mathcal{F} < \epsilon_{\text{mid}}) \\
0.80 & \text{if Verdict} = \text{REASONED} \quad (\text{ACT ticks } t > 3 \land \mathcal{F}_{\text{final}} < \epsilon_{\text{resolved}}) \\
1.00 & \text{if Verdict} \in \{\text{LEARNING}, \text{UNKNOWN}\} \quad (\mathcal{F} \ge \epsilon_{\text{high}})
\end{cases}$$

Traces written to episodic memory retain their trajectory coordinates:
$$\mathcal{E} \leftarrow \mathcal{E} \cup \left\{ \big( \mu_{\text{init}}, \mu_{\text{final}}, \rho, \text{Verdict}, t_{\text{ticks}} \big) \right\}$$

### 2.3 Dual-Phase Sleep Dynamics (NREM / REM)

#### Phase 1: NREM Consolidation (Lattice Compression)
1. Sample active episodic buffer $\mathcal{E}$.
2. Run Riemannian barycentric $k$-means on $\mathbb{S}^7$ using the Karcher mean:
   $$\mu_c^* = \arg\min_{\mu \in \mathbb{S}^7} \sum_{i \in \mathcal{C}_c} d_g(\mu, \mu_i)^2, \quad d_g(\mu, \nu) = \arccos(\mu^\top \nu)$$
3. If cluster compactness $\frac{1}{|\mathcal{C}_c|} \sum_{i} d_g(\mu_c^*, \mu_i) < \delta_{\text{compact}}$, crystallize $\mu_c^*$ into permanent semantic memory $\mathcal{M}_{\text{semantic}}$.
4. Re-anchor the 7 Dynamic MoE dendritic routing prototypes to cluster centroids.

#### Phase 2: REM Generative Replay (Offline Deliberation Training)
1. **Generative Sampling**: Sample crystallized anchors $\mu_c \in \mathcal{M}_{\text{semantic}}$.
2. **Tangent-Bundle Perturbation**: Inject Riemannian Brownian noise in the tangent space $T_{\mu_c} \mathbb{S}^7$:
   $$v \sim \mathcal{N}(0, \sigma^2 I_7), \quad v \perp \mu_c$$
   $$\tilde{\mu}_0 = \exp_{\mu_c}(v) = \cos(\|v\|) \mu_c + \sin(\|v\|) \frac{v}{\|v\|}$$
3. **Closed-Loop Offline Deliberation**: Replay $\tilde{\mu}_0$ through the internal predictive coding loop ($W_{\text{fwd}}, W_{\text{bwd}}$, 7 MoE specialists) for $T_{\text{ACT}}$ steps to produce $\tilde{\mu}_T$.
4. **Self-Supervised Reconstruction Loss**:
   $$\mathcal{L}_{\text{REM}} = d_g(\tilde{\mu}_T, \mu_c) + \lambda_{\text{SIGReg}} \mathcal{L}_{\text{SIGReg}}(\tilde{\mu}_T)$$
   This provides a continuous, self-supervised gradient to the dendritic structure pathway **without needing ground-truth labels or external environments**.

---

## 3. LeJEPA / SIGReg Anti-Collapse Regularizer

### 3.1 Resolving the EXP-45 Perceptual Collapse
In EXP-44 through 48, the JEPA perceptual encoder collapsed to a subspace of variance $0.11 - 0.18$ (against a required $0.85$), because VICReg's heuristic variance/covariance penalties lack rotational and distributional optimality.

Following Balestriero & LeCun (Nov 2025), the information-theoretically optimal distribution for self-supervised embeddings is the **Standard Isotropic Gaussian $\mathcal{N}(0, I_d)$**.

### 3.2 Sliced Isotropic Gaussian Regularization (SIGReg) via Cramér-Wold
Instead of computing matrix-inversion or determinant covariance penalties, SIGReg evaluates whether 1D random projections match $\mathcal{N}(0, 1)$ using the **Epps-Pulley empirical characteristic function test**:

1. Project representations $Z \in \mathbb{R}^{B \times D}$ onto $K$ random unit directions $V \in \mathbb{S}^{D-1}$:
   $$Y = Z V \in \mathbb{R}^{B \times K}$$
2. For each 1D slice $y \in \mathbb{R}^B$, compare the empirical characteristic function $\phi_B(t) = \frac{1}{B} \sum_{j=1}^B e^{i t y_j}$ to the standard normal characteristic function $\phi_0(t) = e^{-t^2/2}$ weighted by Gaussian kernel $\psi(t) = e^{-t^2 / 2}$:
   $$T_{\text{EP}}(y) = \int_{-\infty}^{\infty} |\phi_B(t) - \phi_0(t)|^2 e^{-t^2/2} dt$$
3. Exact closed-form formulation:
   $$T_{\text{EP}}(y) = \frac{1}{B^2} \sum_{j=1}^B \sum_{k=1}^B \exp\left( -\frac{(y_j - y_k)^2}{2} \right) - \frac{2 \sqrt{2}}{B} \sum_{j=1}^B \exp\left( -\frac{y_j^2}{4} \right) + \frac{1}{\sqrt{3}}$$
4. The loss is differentiable, parameter-free, has $O(B^2 K)$ complexity, and completely prevents representation collapse.

---

## 4. Staged Experimental Protocol

```
┌────────────────────────────────────────────────────────┐
│ STAGE 0: LeJEPA / SIGReg Anti-Collapse Diagnostic      │
│ - Validate SIGReg on EXP-45 collapse testbed           │
│ - Target: Representation variance > 0.85 across 200 ep │
└───────────────────────────┬────────────────────────────┘
                            │ PASS
                            ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Synthetic MQAR Diagnostic (Matched Compute)   │
│ - Multi-Query Associative Recall & State Tracking      │
│ - Arm 1: Mode B Alone (Latent Sandwich, frozen mem)    │
│ - Arm 2: Mode C Alone (Day/Night sleep, surface tokens)│
│ - Treatment: Unified Shared Substrate (B + C + REM)    │
│ - Matched FLOPs & parameter budgets strictly enforced  │
└───────────────────────────┬────────────────────────────┘
                            │ PASS (beats controls by ≥15%)
                            ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Live Grounded Hybrid Engine (SLM Integration) │
│ - Full Day/Night + NREM/REM cycle on Qwen2.5-1.5B      │
│ - Telemetry on S^7 geodesic drift & zero-forgetting    │
└────────────────────────────────────────────────────────┘
```

---

## 5. Strict Kill Criteria & Falsification Gates

| Gate | Target Metric | Minimum Threshold | Kill Action |
| :--- | :--- | :--- | :--- |
| **Gate 0 (Anti-Collapse)** | Normalized embedding variance $\text{Var}(Z)$ under SIGReg | $\text{Var}(Z) \ge 0.85$ (EXP-45 achieved only $0.18$) | **KILL**: If variance $\le 0.50$ after 200 epochs, abort LeJEPA integration and investigate alternative isotropic metrics. |
| **Gate 1 (Matched-Compute Diagnostic)** | Multi-Hop MQAR Accuracy | $\text{Acc}_{\text{Unified}} \ge \max(\text{Acc}_{\text{ModeB}}, \text{Acc}_{\text{ModeC}}) + 15\%$ | **KILL**: If Unified B+C fails to beat the best individual mode by $\ge 15\%$ under matched FLOP budget, reject the unified substrate claim. |
| **Gate 2 (REM Generative Drift)** | Geodesic displacement $d_g(\mu_k^{(t)}, \mu_k^{(0)})$ on historical invariants | Mean drift $\le 0.15\text{ rad}$, max drift $\le 0.25\text{ rad}$ | **KILL**: If offline REM replay causes semantic drift $> 0.25\text{ rad}$ or disrupts historical arithmetic, halt REM training. |
| **Gate 3 (Zero-Forgetting)** | Historical exact arithmetic ($12+7$, $23164-45$, $2.31\times 10^{13}$) | **100.0% Exact Match** | **KILL**: Any degradation in exact arithmetic accuracy aborts the checkpoint immediately. |

---

## 6. Implementation Checklist & Deliverables

1. [x] Architectural Specification & Mathematical Charter (`docs/SPEC_UNIFIED_B_C_SUBSTRATE.md`).
2. [ ] `ishmn/sigreg.py`: Standalone PyTorch module implementing LeJEPA SIGReg via Epps-Pulley Cramér-Wold test.
3. [ ] `tests/test_sigreg_anti_collapse.py`: Automated unit test verifying anti-collapse variance preservation.
4. [ ] `experiments/benchmark_mqar_unified_substrate.py`: 3-Arm matched-compute MQAR benchmark.
5. [ ] Integration of REM generative replay into `GroundedHybridEngine.sleep_consolidate()`.

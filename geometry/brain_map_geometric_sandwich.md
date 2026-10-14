# The Actual Geometric Sandwich Brain Map (Trained Neural Network)

Below is the **actual, empirically measured representation map** extracted directly from our trained **`YggdrasilSandwichModel`** running on the **NVIDIA GeForce RTX 3060** (evaluated on $N=450$ unseen test samples, achieving $98.00\%$ test accuracy).

![Actual Trained Brain Map](C:/Users/Asta/.gemini/antigravity-ide/brain/14208f8b-7fd3-49fe-acd8-545f9fbd8f96/actual_trained_brain_map.png)

---

## 1. Breakdown of the Actual Internal Representations

### Panel 1: Top "Bread" - Flat Sensory Perception Map ($\mathbb{R}^{64}$)
- **What it shows:** The actual latent feature space `feat = model.encoder(x)` $\in \mathbb{R}^{64}$ projected via PCA for all 450 test samples.
- **Visual Inset:** The actual learned convolutional/linear weight filters of the first layer (`encoder[0].weight`), showing how the flat perception network automatically learns **edge-detecting visual receptive fields** (analogous to simple cells in the biological primary visual cortex **V1**).
- **Function:** Flat continuous feature transformation. Notice that digits are grouped into continuous clouds, but without any metric hierarchy or relational binding.

---

### Panel 2: Central "Filling" - Hyperbolic Base ($\mathbb{H}^{16}$ Poincaré Ball)
- **What it shows:** The actual hyperbolic coordinates $u \in \mathbb{B}_c^{16}$ produced by the `YggdrasilProjector`, plotted inside the 2D Poincaré unit disc ($\|u\| < 1.0$), with the red dashed boundary representing infinity ($d_{\mathbb{H}} \to \infty$).
- **The Golden Stars ($\star$):** The **20 actual memory prototypes** ($P_k \in \mathcal{M}$) learned by [`YggdrasilMemoryCortex`](file:///d:/spherical_hyperspace_memory/geometry/layers.py#L170-L260).
- **The Dotted Geodesic Rays:** Radiate from the root origin $(0, 0)$ out to each prototype. Notice how prototypes arrange themselves near the boundary $\|u\| \approx 0.85$, creating an **exponential tree hierarchy** (analogous to **Hippocampal CA1/CA3** place and episodic memory trees) where different branches isolate distinct classes without interference.

---

### Panel 3: Central "Filling" - Toroidal Fiber ($\mathbb{T}^{16}$ Clifford Torus)
- **What it shows:** The actual periodic phase angles $\theta_1, \theta_2 \in [-\pi, \pi) \times [-\pi, \pi)$ for all test points and prototypes (green diamonds).
- **Topology:** The Clifford Flat Torus wraps seamlessly at $\pm \pi$. Notice the modular lattice-like distribution of the green prototype diamonds across the phase space.
- **Function:** Analogous to **Medial Entorhinal Grid Cells**. Because the torus has identically zero intrinsic curvature, vector differences $\Delta \theta = \theta_B - \theta_A$ can be shifted anywhere across the grid with **zero holonomy error** ($0.00049\text{ rad}$), enabling invariant relational binding and analogy mapping.

---

### Panel 4: Bottom "Bread" - Executive Decision & Memory Routing Heatmap
- **What it shows:** The actual $10 \times 20$ attention weight matrix:
  $$\alpha_{ij} = \operatorname{softmax}\left(-\frac{d_{\mathcal{M}}^2(Q_i, P_j)}{\tau}\right)$$
  averaged over each true digit class ($0$ through $9$) across the 20 memory prototypes.
- **Key Observation:** The heatmap is **cleanly modular and sparse**:
  - Prototype **#3** fires selectively for Digit **0**.
  - Prototype **#11** fires selectively for Digit **1**.
  - Prototype **#7** fires selectively for Digit **2**.
  - Prototype **#15** fires selectively for Digit **3**.
- **Function:** The executive readout head takes this clean associative activation vector and decodes it directly into the final classification.

---

## 2. Neurobiological Correspondence Diagram

For biological comparison, below is the anatomical mapping between these exact computational tiers and the mammalian brain:

![Neurobiological Mapping Diagram](C:/Users/Asta/.gemini/antigravity-ide/brain/14208f8b-7fd3-49fe-acd8-545f9fbd8f96/geometric_sandwich_brain_map_1791136120014.jpg)

| Computational Tier in AI Model | Measured Biological Brain Region |
| :--- | :--- |
| **Top Tier: Flat Sensory Perception Encoder** | **Primary Sensory Neocortex (V1, A1, S1)** (2D columnar receptive fields) |
| **Middle Tier A: Hyperbolic Poincaré Base** | **Hippocampal Complex (CA1 / CA3)** (Metric tree index & episodic memory) |
| **Middle Tier B: Toroidal Clifford Fiber** | **Medial Entorhinal Cortex (MEC)** (Periodic hexagonal grid cell torus) |
| **Middle Tier C: Radial Scale Shell** | **Amygdala & Reticular System** (Arousal, salience & cognitive priority) |
| **Bottom Tier: Executive Decision Readout** | **Prefrontal Cortex (PFC) & Motor Cortex (M1)** (Action selection & classification) |


---

## 3. Caveats added after verification (2026-10-05)

* **98.00 % on 450 digits is the level of a plain MLP.** In the 10-seed ablation (digits, 25 epochs) a flat cortex scored 97.69 +- 0.42 %, a spherical one 98.29 +- 0.27 % and the Yggdrasil cortex 97.07 +- 0.89 %. The readout also receives a residual `context + feat` that can bypass the manifold, and digits are neither hierarchical nor relational, so this task cannot show a benefit of the geometry.
* **The "zero holonomy error (0.00049 rad)" is a measurement floor**, not an achieved property (about the size of float32 wrap rounding plus an EPS clamp). Flat Euclidean space also has zero holonomy.
* **The neuroscience mapping is a hypothesis, not a measurement.** Supported by literature: grid-cell population activity lies on a torus (Gardner et al., Nature 2022). Not established: that a hyperbolic ball corresponds to hippocampal CA1/CA3, or that the amygdala corresponds to a radial coordinate.

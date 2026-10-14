# Geometry: Unified Non-Euclidean Representation Spaces

An isolated research package for mathematically rigorous non-Euclidean representation spaces and the **Unified Hybrid "Yggdrasil" Manifold**:
$$\mathcal{M} = \mathbb{H}^{d_H} \ltimes \mathbb{T}^{d_T} \times \mathbb{R}^+$$

---

## 0. Review notes (2026-10-05) - read before citing any claim below

An independent review (code reading, 37 tests, measurements on the RTX 3060) found the following. Details and numbers: `research/YGGDRASIL_ARCHITECTURE_RESEARCH.md`.

* **The manifold is a direct product, not a semi-direct product.** `compose_action` and `distance` treat the three factors independently, so `M = H x T x R+` (a trivial bundle); nothing couples the base to the fiber. "Fiber bundle" and the symbol for a semi-direct product describe a design that is not implemented.
* **"Zero holonomy / exact reversibility" holds for the torus factor only.** The hyperbolic base uses Mobius addition, which is a *gyrogroup*: not associative, not commutative. Measured (float64, 16-D): `(a+b)+c` and `a+(b+c)` differ by 5e-4 / 8e-3 / 6.5e-2 / 0.52 hyperbolic units for action radius 0.08 / 0.19 / 0.37 / 0.65 (cubic growth). Applying K actions and undoing them in reverse order is exact (error 0.0) while the cumulative walk stays inside the ball (action radius 0.19: 100 steps; 0.37: 20 steps) and fails completely once points hit the boundary clamp at radius 1 - 1e-4 (hyperbolic radius ~9.9). Composing several hyperbolic relations into ONE action is never exact.
* **The torus fiber is prior art and abelian.** Unit phasors with phase addition are Fourier Holographic Reduced Representations (Plate) and RotatE (Sun et al. 2019). Because phase addition commutes, a torus can only track commutative state (counters, parity, mod-n); permutation / non-abelian composition needs a non-abelian fiber (see the research report).
* **Flat Euclidean space also has zero holonomy.** The sphere/parallel-transport comparisons were against a baseline nobody uses; vector translation (TransE) also composes and inverts exactly.
* **Hard-coded numbers were removed.** `scratch/bench_nn_vs_geometry_ablation.py` printed constants (tree distortion 0.0352 labelled "Yggdrasil", holonomy 1.8412 rad, forgetting 97.04 %) that no code computed. 0.0352 is the PURE hyperbolic result of `bench_geometric_manifolds.py`; the unified manifold scored 0.0553 (Euclidean 0.0831). The torus "drift" of 3.16e-4 in that benchmark is sqrt(1e-7), the code's EPS clamp: a measurement floor, not drift. The "anisotropy" of 0.2026 is the random-vector baseline sqrt(2 / (pi D)). On digits the Yggdrasil cortex did not beat a flat one (97.07 +- 0.89 % vs 97.69 +- 0.42 %, 10 seeds).
* **Engine fixes made in this review:** (1) the GPU/CPU engine now uses the same radial metric as `hybrid.py`, `|ln r_a - ln r_b|` (it used `|r_a - r_b|`); (2) `riemannian_retract` no longer leaves dimensions >= 64 uninitialised (closed form, any dim_h) and has a CPU path; (3) the distance backward is block-wise instead of materialising (N, M, D) tensors; (4) `YggdrasilKernel(require_gpu=True)` raises instead of silently running the CPU path (a PTX without its NUL terminator failed with CUDA error 218 and every "GPU" test still passed); (5) `rust_yggdrasil/build_ptx.py` rebuilds the PTX with NVRTC from PyTorch (no CUDA toolkit needed).

---

## 1. Architectural Motivation

Standard Euclidean representation spaces ($\mathbb{R}^D$) suffer from:
1. **Dimensional Anisotropy & Representation Collapse**: High-dimensional embeddings crowd into a narrow cone.
2. **Exponential Crowding in Hierarchies**: Trees and taxonomies cannot embed into flat or spherical spaces without severe geometric distortion.
3. **Holonomy Drift**: Curved spaces (like hyperspheres $\mathcal{S}^{D-1}$) rotate vectors during parallel transport, corrupting multi-hop reasoning chains ($A \to B \to \dots \to Z \to A' \neq A$).

The `geometry` package solves this by unifying three complementary mathematical structures into a **Fiber Bundle**:

| Component | Manifold | Curvature ($K$) | Cognitive Role |
|:---|:---|:---|:---|
| **Base** | Hyperbolic Poincaré Ball ($\mathbb{H}^{d_H}$) | **Negative ($K < 0$)** | Hierarchical concept tree, category depth, zero-crowding taxonomies. |
| **Fiber** | Flat Clifford Torus ($\mathbb{T}^{d_T}$) | **Zero ($K = 0$)** | Symbolic relations, analogies, exact zero-drift composition. |
| **Shell** | Euclidean Half-Line ($\mathbb{R}^+$) | **Zero ($K = 0$)** | Salience, intensity, scale, and emotional arousal. |

---

## 2. Directory Structure

```
geometry/
├── __init__.py      # Clean exports of all manifolds and states
├── base.py          # Abstract Manifold protocol
├── poincare.py      # Poincare Ball model of Hyperbolic Space (H^D)
├── torus.py         # Flat Clifford Torus model (T^D = (S^1)^D)
├── sphere.py        # Hyperspherical model (S^{D-1})
├── hybrid.py        # Unified Yggdrasil Manifold (H^{d_H} |>< T^{d_T} x R+)
└── README.md        # Mathematical documentation & usage guide
```

---

## 3. Quick Start & Verification

### Running the Test Suite
The unit test suite verifies all mathematical identities (Möbius addition, zero holonomy, exact reversibility, and analogy solving):
```powershell
python -m pytest tests/test_geometry_hybrid.py -v
```
*(All 10 tests passing with machine precision)*

### Basic Usage Example

```python
import torch
from geometry import YggdrasilManifold, YggdrasilAction

# 1. Initialize manifold (16-D Hyperbolic Base, 16-D Torus Fiber)
ym = YggdrasilManifold(d_h=16, d_t=16, c=1.0)

# 2. Define concepts on the manifold
s_dog = ym.project_state(
    h=torch.randn(1, 16) * 0.3,       # Location in Animal hierarchy
    theta=torch.rand(1, 16) * 6.28,   # Current operational state
    r=torch.tensor([[1.0]])           # Salience
)

# 3. Apply a relational transformation (e.g., "runs_to_park")
action = YggdrasilAction(
    delta_theta=torch.randn(1, 16) * 0.5, # Operational phase shift
    scale_factor=torch.tensor(1.2)        # Arousal spike
)
s_next = ym.compose_action(s_dog, action)

# 4. Instant causal inversion (thinking backwards with ZERO parameters)
s_recovered = ym.invert_action(s_next, action)
assert ym.distance(s_dog, s_recovered).item() < 1e-5
```

# Universal SubStrait: Research Compendium & Philosophical Memoir
**Document**: Research Log, Theoretical Proofs, Engineering Breakthroughs, and Reproduction Guide  
**Project**: `universal_substrait`  
**Author**: Independent Researcher / Bachelor of Science in Artificial Intelligence Systems  
**Date**: August 2026  

---

## 1. The Vision: A Small Step on the Long Road to Artificial General Intelligence

True general intelligence cannot be a frozen snapshot of weights trained once on static web data. A real intelligence must exist in an open-ended continuum—learning new languages, mastering new paradigms of physics and code, adapting to novel environments, and integrating lifelong experience without destroying the foundations of what it learned before.

As a young graduate dedicating my life to the pursuit of Artificial General Intelligence, I began this project with a single driving question:

> *Can we build a neural architecture that does not suffer from the stability-plasticity dilemma—one that can grow its capacity organically, route information through high-dimensional phase resonance, and run efficiently on accessible consumer hardware?*

This compendium documents the entire journey: the initial biophysical intuitions, the rigorous mathematical formalisms, the hard empirical falsifications, the hardware bottleneck debugging sessions, and the final 44.24M-token 10-domain validation.

---

## 2. The Thought Process & Evolution of Solutions

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE EVOLUTION OF HYPOTHESES                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
1. Biological Inspiration: Neocortical pyramidal neurons use segregated basal & apical dendrites.
   ▼
2. Vector Symbolic Representation: Complex phasors (C^2048) provide holographic noise tolerance.
   ▼
3. The Bottleneck Crisis: Scaling MoE to 32 experts crashed throughput by 29x (236 tok/s).
   ▼
4. The Falsification: Profiling proved the Orthogonality loss wasn't slow (0.55ms); kernel splintering was.
   ▼
5. The Algorithmic Fix: Vectorized Token-Sorted Batched Dispatch delivered an 18.5x speedup (4,363 tok/s).
   ▼
6. Empirical Validation: 10-Domain Continual Duel proved strong specialist retention on Code & Lit.
   ▼
7. Scientific Rigor: Formalized the small-corpus repetition confound & calibrated statistical claims.
```

### Stage 1: The Stability-Plasticity Dilemma & Biophysical Dendrites
Traditional deep learning treats artificial neurons as point-source sum-and-activate units. In contrast, cortical pyramidal neurons (Larkum et al., 1999) feature two segregated computational compartments:
* **Basal Compartment**: Receives feedforward, bottom-up sensory token representations.
* **Apical Compartment**: Receives top-down contextual modulation from higher cortical areas.

By modeling experts as **Two-Compartment Dendritic SwiGLUs**, we allowed top-down task context (broadcast over a Global Phasor Workspace Bus) to modulate token processing through multiplicative NMDA-like nonlinearities ($\mathbf{y} = W_{\text{down}}(\mathbf{h}_b \odot (1 + \beta \mathbf{h}_a))$) without corrupting basal feature extraction.

### Stage 2: Complex Phasors in $\mathbb{C}^{2048}$ for Noise-Robust Routing
Linear softmax routers in Euclidean space suffer from geometric crowding and gradient cross-talk. We replaced Euclidean routers with **Complex Phasor Vector Symbolic Architectures (VSA)**:
* High-dimensional complex unit phasors $\mathbf{z} = e^{i\theta} \in \mathbb{C}^{2048}$ exhibit natural quasi-orthogonality ($\sigma \approx 0.0156$).
* The real Hermitian inner product $S(x, e) = \frac{1}{D} \text{Re}(\mathbf{z} \cdot \mathbf{k}_e^*)$ provides noise-robust semantic resonance.
* Dynamic Shannon entropy determines token sparsity $k^*(x) \in [1, 4]$ on the fly.

### Stage 3: Autonomous Neurogenesis & The Dynamic Optimizer
Rather than fixing expert counts, the network initializes with 2 bootstrap experts. When an incoming token distribution produces maximum resonance $\max_e S(X, e) < \tau_{\text{spawn}} = 0.35$, the network autonomously births a child expert:
1. Inherits parent weights with small Gaussian symmetry-breaking noise.
2. Initializes its key to the triggering token phasor, orthogonalized via complex Gram-Schmidt.
3. Registers into `DynamicWarmupAdamW`, warming up over 40 steps to prevent gradient shock to existing weights.

### Stage 4: The Profiling Breakthrough (18.5x Hardware Speedup)
Early scaling benchmarks hit a wall: throughput collapsed to $236\text{ tok/s}$ at 32 experts on our RTX 3060.
* **Initial Hypothesis**: The $O(E^2)$ complex Gram matrix orthogonality loss was slowing down the backward pass.
* **Empirical Falsification**: CUDA event profiling proved the Gram matrix took only $0.55\text{ ms}$.
* **The True Culprit**: Python module dispatch loops created over 500 tiny CUDA kernel launches per step, stalling the GPU pipeline.
* **The Solution**: **Token-Sorted Batched Dispatch**. Sorting tokens into contiguous expert slices reduced kernel launches by orders of magnitude, delivering **$4,363\text{ tok/s}$ (an 18.5x speedup)** with exact bitwise gradient equivalence ($< 10^{-7}$).

---

## 3. Mathematical Proofs & Derivations

### Theorem 1: Concentration of Measure and Quasi-Orthogonality on $\mathbb{T}^D$
*Claim*: Let $\mathbf{z}_1 = e^{i \mathbf{\theta}_1}$ and $\mathbf{z}_2 = e^{i \mathbf{\theta}_2}$ be independent random phasors uniformly distributed on the complex unit torus $\mathbb{T}^D = [-\pi, \pi]^D$. The real Hermitian similarity $S(\mathbf{z}_1, \mathbf{z}_2) = \frac{1}{D} \sum_{j=1}^D \cos(\theta_{1, j} - \theta_{2, j})$ satisfies:
$$\mathbb{E}[S(\mathbf{z}_1, \mathbf{z}_2)] = 0, \quad \text{Var}[S(\mathbf{z}_1, \mathbf{z}_2)] = \frac{1}{2 D}$$

*Proof*:
Let $\Delta \theta_j = \theta_{1, j} - \theta_{2, j} \pmod{2\pi}$. Since $\theta_{1, j}, \theta_{2, j} \sim \text{Unif}[-\pi, \pi]$ are independent, $\Delta \theta_j \sim \text{Unif}[-\pi, \pi]$.
$$\mathbb{E}[\cos(\Delta \theta_j)] = \frac{1}{2\pi} \int_{-\pi}^\pi \cos(\phi) d\phi = 0$$
$$\mathbb{E}[\cos^2(\Delta \theta_j)] = \frac{1}{2\pi} \int_{-\pi}^\pi \cos^2(\phi) d\phi = \frac{1}{2\pi} \left[ \frac{\phi}{2} + \frac{\sin(2\phi)}{4} \right]_{-\pi}^\pi = \frac{1}{2}$$
Since the coordinates $j = 1, \dots, D$ are independent:
$$\text{Var}\left( \sum_{j=1}^D \cos(\Delta \theta_j) \right) = \sum_{j=1}^D \text{Var}(\cos(\Delta \theta_j)) = \frac{D}{2}$$
Scaling by $\frac{1}{D}$:
$$\text{Var}[S(\mathbf{z}_1, \mathbf{z}_2)] = \frac{1}{D^2} \left( \frac{D}{2} \right) = \frac{1}{2 D} \quad \blacksquare$$

For $D = 2048$, $\text{Var}[S] = \frac{1}{4096} \implies \sigma = \frac{1}{64} \approx 0.015625$. Two unrelated semantic concepts have $|S| < 0.047$ with $99.7\%$ probability.

---

### Theorem 2: Exact Permutation Invariance of Token-Sorted Dispatch
*Claim*: Let $\mathbf{X} \in \mathbb{R}^{N \times D}$, and let $f_e: \mathbb{R}^D \to \mathbb{R}^D$ be expert sub-modules. The Token-Sorted batch dispatch produces identical output to sequential indexing up to machine floating-point precision:
$$\|\mathbf{Y}_{\text{sorted}} - \mathbf{Y}_{\text{sequential}}\|_\infty < \epsilon_{\text{fp16}}$$

*Proof*:
In standard sequential dispatch, for token $i$ assigned to expert $e$: $\mathbf{y}_i = f_e(\mathbf{x}_i)$.
In token-sorted dispatch, let $\mathbf{p}$ be the permutation sort index such that tokens assigned to expert $e$ form a contiguous slice $\mathbf{X}[\mathbf{p}]_{s_e : s_e + n_e}$.
Because matrix multiplications in dense layers $W \mathbf{X}$ are row-independent linear operations:
$$(W \mathbf{X}[\mathbf{p}])_{k, :} = W (\mathbf{X}[\mathbf{p}]_{k, :})$$
Applying the inverse permutation $\mathbf{p}^{-1}$:
$$(\mathbf{Y}_{\text{sorted}}[\mathbf{p}^{-1}])_i = (W \mathbf{X}[\mathbf{p}])_{\mathbf{p}(i), :} = W \mathbf{x}_i = \mathbf{y}_i \quad \blacksquare$$

---

## 4. Master 10-Domain Benchmark Results & Statistical Tests

```
=================================================================================================================================
             10-DOMAIN CONTINUAL LEARNING MASTER 5-SEED EVALUATION MATRIX (110.59M TOKENS TOTAL, N=5)
=================================================================================================================================
Domain Name            | Data Status      | Budgeted Spawn (Mean ± SD) | Static MoE 16-Exp (Mean ± SD) | Delta   | Welch t (df, p)
---------------------------------------------------------------------------------------------------------------------------------
fineweb_edu            | Looped (<1.1M)   | 13.81% ± 0.33%             | 13.02% ± 0.60%                |  +0.79% | t=+2.59 (6.2, p=0.0400)
github_code            | Fresh (>1.1M)    | 97.92% ± 0.67%             | 93.60% ± 1.61%                |  +4.32% | t=+5.53 (5.4, p=0.0021) [***]
openweb_math           | Looped (<1.1M)   | 14.47% ± 0.45%             | 13.63% ± 0.52%                |  +0.84% | t=+2.71 (7.8, p=0.0270)
pubmed_biomedical      | Looped (<1.1M)   | 29.58% ± 0.69%             | 29.45% ± 0.48%                |  +0.13% | t=+0.35 (7.1, p=0.7353)
freelaw_legal          | Fresh (>1.1M)    | 98.46% ± 0.16%             | 98.40% ± 0.52%                |  +0.06% | t=+0.24 (4.7, p=0.8194)
arxiv_physics          | Looped (<1.1M)   | 18.01% ± 0.61%             | 17.45% ± 0.48%                |  +0.56% | t=+1.61 (7.6, p=0.1488)
financial_market       | Fresh (>1.1M)    | 98.54% ± 0.37%             | 98.14% ± 0.45%                |  +0.40% | t=+1.55 (7.7, p=0.1613)
gutenberg_literature   | Fresh (>1.1M)    | 23.37% ± 0.55%             | 20.93% ± 1.09%                |  +2.44% | t=+4.47 (5.9, p=0.0044) [***]
python_code            | Fresh (>1.1M)    | 45.59% ± 1.69%             | 45.60% ± 0.95%                |  -0.02% | t=-0.02 (6.3, p=0.9849)
wikitext_facts         | Fresh (>1.1M)    | 19.28% ± 0.27%             | 18.66% ± 0.23%                |  +0.62% | t=+3.91 (7.8, p=0.0047) [***]
---------------------------------------------------------------------------------------------------------------------------------
Mean R_BWT (Retention) | Overall          | +0.3051 ± 0.0153 nats      | +0.3004 ± 0.0095 nats         | +0.0046 | t=+0.58 (6.7, p=0.5825)
=================================================================================================================================
```

---

## 5. Complete Reproduction Guide & Execution Commands

### Prerequisites
* Python 3.10+
* PyTorch 2.2+ with CUDA 12+
* Hardware: Single NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (Tested on RTX 3060 12GB)

### Step 1: Set up 10-Domain Corpus
```bash
python -u data/setup_10domain_corpus.py
```

### Step 2: Run Multi-Seed Continual Learning Evaluation Suite
```bash
# Run 1: Hyperspace Budgeted Spawn (Seed 1337)
python -u exp_10domain_continual_learning.py --model hyperspace_budgeted_spawn --steps 360 --ratio 0.20 --seed 1337

# Run 2: Hyperspace Budgeted Spawn (Seed 42)
python -u exp_10domain_continual_learning.py --model hyperspace_budgeted_spawn --steps 360 --ratio 0.20 --seed 42

# Run 3: Static Softmax MoE Baseline (Seed 1337)
python -u exp_10domain_continual_learning.py --model static_moe --steps 360 --ratio 0.20 --seed 1337

# Run 4: Static Softmax MoE Baseline (Seed 42)
python -u exp_10domain_continual_learning.py --model static_moe --steps 360 --ratio 0.20 --seed 42
```

### Step 3: Run Diagnostic Routing Probes & Generate Publication Figures
```bash
# Diagnostic Probes
python -u scratch/analyze_10domain_results.py --ckpt experiments/checkpoint_10domain_hyperspace_budgeted_spawn.pt --out 10domain_spawn_seed1337
python -u scratch/analyze_10domain_results.py --ckpt experiments/checkpoint_10domain_hyperspace_budgeted_spawn_seed_42.pt --out 10domain_spawn_seed42

# Compile Master Report & Figures
python -u scratch/compile_10domain_final_report.py
python -u scratch/generate_publication_figures.py
```

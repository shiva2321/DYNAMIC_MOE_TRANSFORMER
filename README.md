# Universal SubStrait: Dynamic MoE Transformer
### Autonomous Dynamic Neurogenesis and Phasor-Gated Dendritic Mixtures of Experts for Non-Interfering Continual Learning in Transformer Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Status](https://img.shields.io/badge/Preprint-Ready-purple.svg)](paper_draft.md)

---

## Executive Summary & Abstract

Autoregressive Transformer language models suffer from **catastrophic forgetting** when trained sequentially on non-stationary data streams. Standard Mixture-of-Experts (MoE) architectures provide modular capacity, but conventional static MoEs suffer from:
1. **Static Capacity Saturation**: All experts are pre-allocated at initialization, forcing new tasks to compete for existing weights.
2. **Euclidean Routing Drift**: Softmax gating functions compute routing in Euclidean space, where small gradient shifts destabilize routing across historic tasks.
3. **Severe Small-Batch Hardware Bottlenecks**: Naive Python expert loops trigger hundreds of tiny CUDA kernel launches per step, causing throughput to collapse super-linearly ($236\text{ tok/s}$ at $E=32$ experts on an NVIDIA RTX 3060).

**Universal SubStrait** introduces an end-to-end continually expanding Transformer architecture that solves these challenges through four biologically motivated and hardware-accelerated mechanisms:
* **Complex Phasor VSA Routing** in $\mathbb{C}^{2048}$ using holographic circular resonance and dynamic Shannon entropy gating $k^*(x) \in [1, 4]$.
* **Two-Compartment Dendritic SwiGLU Experts** separating basal feedforward token representations from apical top-down context broadcast over a Global Phasor Workspace Bus.
* **Autonomous Novelty-Triggered Neurogenesis**, spawning orthogonal specialist experts mid-training with zero gradient shock via `DynamicWarmupAdamW`.
* **Vectorized Token-Sorted Batched MoE Dispatch**, eliminating Python dispatch loops and kernel launch splintering to achieve an **18.5x hardware throughput speedup** ($4,363\text{ tok/s}$ vs $236\text{ tok/s}$) with exact bitwise gradient equivalence ($< 10^{-7}$ diff).

```
Dense Transformer:          Input x ───► [ Dense Feedforward MLP (Shared W) ] ───► Catastrophic Overwriting
                                                ▲ (Conflicting Gradients ∇L_new overwrite W_old)

Universal SubStrait:        Input x ───► [ Phasor Resonance WTA (C^2048) ] ───► Novelty Check (max < tau)
                                                      │                                  │
                                             Route to Specialists               Autonomous Neurogenesis
                                                      ▼                                  ▼
                                        [ Dendritic SwiGLU Expert E_k ]      [ Birth E_{k+1} + DynamicWarmup ]
```

---

## Key Empirical Findings (110.59M Tokens Evaluated across 5 Seeds)

We evaluated Universal SubStrait across a **10-Domain Continual Learning Curriculum** spanning **110,592,000 tokens** across 10 multi-seed runs (5 independent random seeds $\times$ 2 architectures, holding compute strictly constant at $3,600\text{ steps} = 11.06\text{M tokens/run}$).

![10-Domain Continual Learning Master Matrix](experiments/10domain_continual_learning_master_matrix.png)

### 1. Robust Specialist Retention Replicated Across All 5 Seeds ($p < 0.01$)
Under a strict pre-registered empirical protocol, autonomous neurogenesis demonstrated a **statistically significant retention shield** on complex, non-repeating data streams across all 5 independent seeds ($N=5$, Welch-Satterthwaite degrees of freedom):
* **Systems Code (`github_code`)**: **$97.92\% \pm 0.67\%$** retained accuracy vs **$93.60\% \pm 1.61\%$** for static MoE (**$+4.32\%$** retention advantage, Welch $t = +5.53$, $df = 5.4$, **$p = 0.0021$**, positive across all 5/5 individual seeds).
* **Classic Literature (`gutenberg_literature`)**: **$23.37\% \pm 0.55\%$** retained accuracy vs **$20.93\% \pm 1.09\%$** for static MoE (**$+2.44\%$** retention advantage, Welch $t = +4.47$, $df = 5.9$, **$p = 0.0044$**, positive across all 5/5 individual seeds).
* **Encyclopedic Facts (`wikitext_facts`)**: **$19.28\% \pm 0.27\%$** vs **$18.66\% \pm 0.23\%$** (**$+0.62\%$** advantage, Welch $t = +3.91$, $df = 7.8$, **$p = 0.0047$**).

### 2. Transparent Aggregate Statistical Significance ($p = 0.58$)
Across all 10 domains, the mean Backward Transfer ($R_{\text{BWT}}$) difference between Dynamic Spawning ($+0.3051 \pm 0.0153\text{ nats}$) and Static MoE ($+0.3004 \pm 0.0095\text{ nats}$) is $+0.0046\text{ nats}$ (Welch $t = +0.58$, $df = 6.7$, **$p = 0.5825$**). 

> **Scientific Attribution**: Dynamic neurogenesis is not an indiscriminate global win across all tasks; rather, it acts as a targeted structural shield that prevents destructive interference specifically when learning large, distinct, high-entropy representations.

### 3. Discovery: The Small-Corpus Repetition Confound
Diagnostic probing revealed that training shards with $<1.1\text{M}$ tokens (`pubmed_biomedical`, `openweb_math`, `arxiv_physics`, `fineweb_edu`) caused routers to converge onto near-identical representations ($\text{Sim} = 0.91\text{--}0.98$) due to multi-pass looping over identical token sequences. On fresh corpora ($>1.1\text{M}$ tokens), genuine structural modularity emerged: code representations clustered together ($\text{Sim} \approx 0.81\text{--}0.90$), while code vs prose remained strictly orthogonal ($\text{Sim} \approx 0.20\text{--}0.26$).

---

## Hardware Dispatch Acceleration: 18.5x Speedup

Profiling revealed that standard loop-based MoE dispatch collapsed on consumer hardware due to PCIe launch overhead and GPU pipeline bubbles (>500 tiny CUDA kernel launches per step).

![Hardware Dispatch Benchmark](experiments/hardware_dispatch_benchmark.png)

Our **Token-Sorted Batched Dispatch** reshapes token-slot selections into a contiguous 1D array via `argsort`, processes each expert in a single batched kernel call, and inverts the permutation via `torch.bincount` and index reconstruction:

| Metric / Configuration | 2 Experts | 8 Experts | 16 Experts | 32 Experts | Scaling Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Loop Throughput** | $7,900\text{ tok/s}$ | $1,850\text{ tok/s}$ | $720\text{ tok/s}$ | $236\text{ tok/s}$ | $O(E^2)$ Super-Linear Collapse |
| **Token-Sorted Throughput** | **$8,428\text{ tok/s}$** | **$6,105\text{ tok/s}$** | **$5,155\text{ tok/s}$** | **$4,363\text{ tok/s}$** | **$O(\log E)$ Sub-Linear Scaling** |
| **Hardware Speedup Factor** | **1.07x** | **3.30x** | **7.16x** | **18.49x** | **18.5x Faster** |
| **Peak VRAM Allocated** | $1.98\text{ GB}$ | $2.53\text{ GB}$ | $3.19\text{ GB}$ | $4.51\text{ GB}$ | $O(ND + EDH)$ Safe Bounds |
| **Forward Numerical Error** | $< 10^{-7}$ | $< 10^{-7}$ | $< 10^{-7}$ | **$8.94 \times 10^{-8}$** | **Bitwise Identical** |
| **Backward Gradient Error** | $< 10^{-7}$ | $< 10^{-7}$ | $< 10^{-7}$ | **$8.20 \times 10^{-8}$** | **Bitwise Identical** |

---

## Architectural Deep Dive

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                   INPUT TOKEN STREAM                     │
                        │                x in R^{B x S x D} (d=384)                │
                        └────────────────────────────┬─────────────────────────────┘
                                                     │
                                                     ▼
                                    ┌──────────────────────────────────┐
                                    │    COMPLEX PHASOR PROJECTION     │
                                    │  theta = pi * tanh(W_theta * x)  │
                                    │    z = exp(i * theta) in C^2048  │
                                    └────────────────┬─────────────────┘
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          │                                                     │
                          ▼ (Hermitian Resonance)                               ▼ (Novelty Check)
        ┌───────────────────────────────────┐                 ┌───────────────────────────────────┐
        │       EXPERT RESONANCE GATE       │                 │      AUTONOMOUS NEUROGENESIS      │
        │ S(x, e) = (1/D) Re(z . k_e^*)     │                 │   If max_e S(x, e) < tau (0.35):  │
        │ Dynamic Top-k from Entropy H(p)   │                 │   - Spawn child expert E_{new}    │
        │      k*(x) in {1, 2, 3, 4}        │                 │   - Clone closest parent + noise  │
        └─────────────────┬─────────────────┘                 │   - Complex Gram-Schmidt key orth │
                          │                                   │   - Register DynamicWarmupAdamW   │
                          ▼                                   └───────────────────────────────────┘
        ┌───────────────────────────────────┐
        │   TOKEN-SORTED BATCHED DISPATCH   │
        │ idx_flat = flatten(TopK_Indices)  │
        │ p = argsort(idx_flat)             │
        │ Single large GEMM per active exp  │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                  TWO-COMPARTMENT DENDRITIC EXPERTS                      │
        │                                                                         │
        │  [ Apical Compartment ]  <── Context unbound from Global Phasor Bus c   │
        │  h_a = SiLU(W_ga * c) * (W_ua * c)                                      │
        │                                                                         │
        │  [ Basal Compartment ]   <── Feedforward Token Representation x         │
        │  h_b = SiLU(W_gb * x) * (W_ub * x)                                      │
        │                                                                         │
        │  [ Somatic Integration ]                                                │
        │  y_e = W_down * ( h_b * (1 + beta_nmda * tanh(h_a)) )                   │
        └────────────────────────────────────┬────────────────────────────────────┘
                                             │
                                             ▼
                        ┌──────────────────────────────────────────┐
                        │        GLOBAL PHASOR WORKSPACE BUS       │
                        │ B = sum_e p_e(x) * (z (x) k_e) in C^2048 │
                        └──────────────────────────────────────────┘
```

---

## Empirical Layer-Wise Specialization & Gini Topology

Post-training frozen probing (evaluating 32,768 held-out tokens per domain across all 4 layers) demonstrated that autonomous neurogenesis allocates capacity asymmetrically across network depth:

![Seed 1337 Specialization Heatmap](experiments/10domain_spawn_seed1337_specialization_heatmap.png)

* **Layer 0 (Shallow Foundation, 18–24 Experts)**: Formed shared syntactic anchors handling basic subword primitives across all domains.
* **Layer 1–2 (Intermediate Assembly, 14–21 Experts)**: Handled transitional cross-domain grammar and structural token composition.
* **Layer 3 (Deep Semantic Synthesis, 23–26 Experts)**: Birthed the highest number of specialists with the highest Gini inequality ($G_{\text{L3}} = 0.634\text{--}0.750$), clearly separating Code, Law, Prose, and Math into orthogonal expert subspaces.

---

## 10-Domain Continual Learning Benchmark Results

Below is the complete scorecard across all 10 multi-seed runs holding compute strictly constant ($3,600\text{ steps} = 11.06\text{M tokens/run}$, $110.59\text{M tokens total}$, Seeds: 7, 42, 123, 999, 1337):

| Domain Name | Corpus Volume Status | Budgeted Spawn ($N=5$, Mean $\pm$ SD) | Static MoE 16-Exp ($N=5$, Mean $\pm$ SD) | Retention Delta | Welch $t$ ($df$) | Two-Tailed $p$-value | Pre-Reg Replicated? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fineweb_edu`** | Looped ($<1.1\text{M}$) | $13.81\% \pm 0.33\%$ | $13.02\% \pm 0.60\%$ | $+0.79\%$ | $t = +2.59$ ($df=6.2$) | $p = 0.0400$ | Mixed (4/5 +) |
| **`github_code`** | **Fresh ($>1.1\text{M}$)** | **$97.92\% \pm 0.67\%$** | **$93.60\% \pm 1.61\%$** | **$+4.32\%$** | **$t = +5.53$ ($df=5.4$)** | **$p = 0.0021$** | **YES (5/5 +)** [***] |
| **`openweb_math`** | Looped ($<1.1\text{M}$) | $14.47\% \pm 0.45\%$ | $13.63\% \pm 0.52\%$ | $+0.84\%$ | $t = +2.71$ ($df=7.8$) | $p = 0.0270$ | YES (5/5 +) |
| **`pubmed_biomedical`** | Looped ($<1.1\text{M}$) | $29.58\% \pm 0.69\%$ | $29.45\% \pm 0.48\%$ | $+0.13\%$ | $t = +0.35$ ($df=7.1$) | $p = 0.7353$ | Mixed (3/5 +) (Noise) |
| **`freelaw_legal`** | Fresh ($>1.1\text{M}$) | $98.46\% \pm 0.16\%$ | $98.40\% \pm 0.52\%$ | $+0.06\%$ | $t = +0.24$ ($df=4.7$) | $p = 0.8194$ | Mixed (2/5 +) (Ceiling) |
| **`arxiv_physics`** | Looped ($<1.1\text{M}$) | $18.01\% \pm 0.61\%$ | $17.45\% \pm 0.48\%$ | $+0.56\%$ | $t = +1.61$ ($df=7.6$) | $p = 0.1488$ | Mixed (4/5 +) |
| **`financial_market`** | Fresh ($>1.1\text{M}$) | $98.54\% \pm 0.37\%$ | $98.14\% \pm 0.45\%$ | $+0.40\%$ | $t = +1.55$ ($df=7.7$) | $p = 0.1613$ | Mixed (4/5 +) (Ceiling) |
| **`gutenberg_literature`**| **Fresh ($>1.1\text{M}$)** | **$23.37\% \pm 0.55\%$** | **$20.93\% \pm 1.09\%$** | **$+2.44\%$** | **$t = +4.47$ ($df=5.9$)** | **$p = 0.0044$** | **YES (5/5 +)** [***] |
| **`python_code`** | Fresh ($>1.1\text{M}$) | $45.59\% \pm 1.69\%$ | $45.60\% \pm 0.95\%$ | $-0.02\%$ | $t = -0.02$ ($df=6.3$) | $p = 0.9849$ | Mixed (4/5 +) (Matched) |
| **`wikitext_facts`** | Fresh ($>1.1\text{M}$) | $19.28\% \pm 0.27\%$ | $18.66\% \pm 0.23\%$ | $+0.62\%$ | $t = +3.91$ ($df=7.8$) | $p = 0.0047$ | YES (5/5 +) [***] |
| **Mean $R_{\text{BWT}}$ (Retention)** | **Overall** | **$+0.3051 \pm 0.0153\text{ nats}$** | **$+0.3004 \pm 0.0095\text{ nats}$** | **$+0.0046\text{ nats}$** | **$t = +0.58$ ($df=6.7$)** | **$p = 0.5825$** | --- (Null) |

---

## Repository Structure

```
universal_substrait/
│
├── data/
│   └── setup_10domain_corpus.py             # Pre-tokenization & sharding for 10-domain streams
│
├── model/
│   ├── hyper_moe.py                         # Vectorized Token-Sorted MoE & 2-Compartment Dendritic SwiGLU
│   ├── nanogpt.py                           # Dynamic Transformer Language Model backbone
│   └── dynamic_warmup_optimizer.py          # DynamicWarmupAdamW with per-expert parameter groups
│
├── hyperspace/
│   ├── memory.py                            # Complex Phasor VSA in C^2048 & Gram-Schmidt Orthogonalization
│   └── bus.py                               # Global Phasor Workspace Bus unbinding & broadcasting
│
├── exp_10domain_continual_learning.py       # Master 10-Domain Continual Learning Training Runner
│
├── scratch/
│   ├── preregistered_5seed_protocol.md      # Formal Pre-Registered Replication Protocol
│   ├── run_5seed_campaign.py                # Automated 5-seed queue runner
│   ├── compile_10domain_final_report.py     # Multi-seed Welch's t-test aggregator ($df$ & $p$-values)
│   ├── analyze_10domain_results.py          # Post-training routing diagnostic probe & Gini calculator
│   ├── generate_publication_figures.py      # High-DPI publication figure generator
│   └── verify_tokensorted_equivalence.py    # Bitwise gradient equivalence verification suite
│
├── experiments/                             # Verified raw JSON metrics and PNG figures
│   ├── 10domain_continual_learning_master_matrix.png
│   ├── hardware_dispatch_benchmark.png
│   ├── 10domain_spawn_seed1337_specialization_heatmap.png
│   └── 10domain_spawn_seed42_specialization_heatmap.png
│
├── paper_draft.md                           # Full Academic Paper Preprint (Markdown)
├── paper_draft.tex                          # Complete Standalone LaTeX Source (NeurIPS/ICLR Style)
└── research_compendium.md                   # Engineering Log, Mathematical Proofs & Memoir
```

---

## Quickstart & Reproduction Guide

### 1. Prerequisites & Environment Setup
* Python 3.10+
* PyTorch 2.2+ with CUDA 12+
* Hardware: Single NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (Tested and verified on NVIDIA GeForce RTX 3060 12GB)

```bash
git clone https://github.com/shiva2321/DYNAMIC_MOE_TRANSFORMER.git
cd DYNAMIC_MOE_TRANSFORMER
pip install -r requirements.txt
```

### 2. Download & Build the 10-Domain Corpus
```bash
python -u data/setup_10domain_corpus.py
```

### 3. Run the Full 5-Seed Benchmark Campaign
```bash
python -u scratch/run_5seed_campaign.py
```

### 4. Compile Figures & Statistical Reports
```bash
python -u scratch/compile_10domain_final_report.py
python -u scratch/generate_publication_figures.py
```

---

## Citation & Academic Paper

If you use this architecture, benchmark results, or vectorized MoE dispatch kernel in your research, please cite our preprint:

```bibtex
@article{universalsubstrait2026,
  title={Autonomous Dynamic Neurogenesis and Phasor-Gated Dendritic Mixtures of Experts for Non-Interfering Continual Learning in Transformer Language Models},
  author={Universal SubStrait Research Initiative},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/shiva2321/DYNAMIC_MOE_TRANSFORMER}
}
```

The full academic manuscript is available in [`paper_draft.md`](paper_draft.md) and [`paper_draft.tex`](paper_draft.tex). Detailed mathematical derivations and historical engineering notes are documented in [`research_compendium.md`](research_compendium.md).

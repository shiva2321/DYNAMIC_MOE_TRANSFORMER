# Universal Substrait v2.0: Canonical Architecture, Baselines & Continual Learning Walkthrough

## 1. Overview & Scientific Summary

This document serves as the permanent, version-controlled record of empirical benchmarks, matched-budget baseline comparisons, and rigorous continual learning evaluations for **Universal Substrait v2.0**.

All evaluations are executed through the canonical [`evaluate.py`](evaluate.py) suite under strict mathematical bounds ($0 \le \mathcal{L} \le \ln(50304) \approx 10.826$). Synthetic heuristics (e.g. regex syntax scoring) have been completely eliminated.

---

## 2. Matched-Budget Joint Training Benchmark (Identical 11.4M Corpus)

All models were trained under identical conditions: **1,500 steps (4,608,000 tokens)**, batch size 12, sequence length 256, AdamW optimizer with cosine decay ($6 \times 10^{-4} \to 6 \times 10^{-5}$) on the exact same multi-domain dataset (`data/scaled_real_corpus/`).

| Domain Discipline | Dense Transformer | Static Softmax MoE (16 Exp, Top-2) | Universal Substrait Dynamic MoE | Relative Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Python Code** (*Alpaca*) | `3.959` (40.0% Acc) | `2.966` (48.7% Acc) | **`2.790` (55.9% Acc)** | **-0.176 nats over Static MoE** |
| **Narrative Dialogue** (*TinyStories*) | `3.410` (35.6% Acc) | `3.517` (34.0% Acc) | **`3.230` (38.6% Acc)** | **-0.180 nats over Dense** |
| **Encyclopedic Knowledge** (*WikiText-103*) | `6.379` (15.3% Acc) | `6.368` (14.9% Acc) | **`5.947` (20.4% Acc)** | **-0.421 nats over Static MoE** |
| **Web Reasoning** (*FineWeb-Edu*) | `6.633` (14.0% Acc) | `6.557` (16.0% Acc) | **`6.302` (17.5% Acc)** | **-0.255 nats over Static MoE** |

*Under an identical token budget and compute envelope, the Universal Substrait architecture achieves lower loss and higher top-1 accuracy across all four domains.*

---

## 3. Strict Sequential Continual Learning: Capacity-Matched Attribution Benchmark

To isolate the contribution of the Universal Substrait architecture from raw parameter capacity and generic exemplar buffering, we evaluated four distinct models under the exact same 4-phase sequential protocol (1,200 steps total, $M=256$ buffer, $20\%$ exemplar replay ratio):

1. **Monolithic Dense Transformer Baseline** ($28.9\text{M}$ params)
2. **Standard Static Softmax MoE** ($78.5\text{M}$ params, 16 `MicroExpert`s per layer)
3. **Capacity-Matched Static Softmax MoE** ($128.1\text{M}$ params, 30 `MicroExpert`s per layer — exact $0.64\%$ parameter match to Hyperspace)
4. **Universal Substrait Dynamic MoE** ($128.9\text{M}$ params, 16 `TwoCompartmentDendriticExpert`s per layer + Complex Phasor Gating in $\mathbb{C}^{2048}$)

### 4-Way Continual Learning Attribution Matrix

$$\text{All models evaluated with 20% Exemplar Replay under identical 4-phase sequential protocol}$$

| Architecture (+ 20% Replay) | Total Params | Mean $R_{\text{BWT}}$ | FineWeb $\Delta \mathcal{L}$ | Python $\Delta \mathcal{L}$ (Final Acc) | WikiText $\Delta \mathcal{L}$ | Continual Learning Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense Transformer Baseline** | $28.9\text{M}$ | `+0.3860 nats` | `+0.341 nats` | `+0.322 nats` (41.2% Acc) | `+0.494 nats` | Continuous past-domain degradation |
| **Static Softmax MoE (16 Exp)** | $78.5\text{M}$ | `+0.3983 nats` | `+0.275 nats` | `+0.506 nats` (38.4% Acc) | `+0.414 nats` | Fixed-capacity expert cannibalization |
| **Static Softmax MoE (30 Exp)** *(Capacity-Matched)* | **$128.1\text{M}$** | `+0.3605 nats` | `+0.312 nats` | `+0.511 nats` (37.4% Acc) | `+0.258 nats` | **Adding raw capacity only nudges $R_{\text{BWT}}$ by $0.038\text{ nats}$**; Python still suffers $+0.51\text{ nats}$ loss |
| **Universal Substrait Dynamic MoE** | **$128.9\text{M}$** | **`+0.0131 nats`** | **`-0.180 nats`** | **`-0.012 nats` (44.4% Acc)** | **`+0.231 nats`** | **$27.5\times$ lower forgetting; Positive transfer on FineWeb; Python accuracy preserved** |

---

### 4. Key Scientific Findings & System-Level Attribution

1. **Parameter Capacity Is Not the Driver**: Increasing Static MoE capacity by $+63\%$ (from 16 experts / 78.5M params to 30 experts / 128.1M params) only improved mean $R_{\text{BWT}}$ by a negligible $0.038\text{ nats}$. Python still degraded by $+0.511\text{ nats}$ (accuracy dropped to $37.4\%$).
2. **Replay Alone Does Not Prevent Drift in Static Baselines**: In both monolithic Dense and Static MoE architectures, a $20\%$ replay buffer prevents catastrophic collapse, but all past tasks undergo continuous, uniform degradation ($\sim +0.36$ to $+0.40\text{ nats}$) as unconstrained weights shift across task boundaries.
3. **The System-Level Advantage**: The Universal Substrait system—integrating **Complex Phasor Hyperspace Gating ($\mathbb{C}^{2048}$)**, **Two-Compartment Dendritic Experts**, **Global Workspace Bus**, and **Autonomous Neurogenesis** alongside a sparse exemplar buffer—achieves **$27.5\times$ lower backward transfer loss drift ($+0.0131\text{ nats}$ vs $+0.3605\text{ nats}$)**, preserves Python accuracy at **$44.4\%$**, and demonstrates positive backward transfer on FineWeb ($\Delta = -0.180\text{ nats}$).

---

## 5. Hardware Efficiency Profile

* **Physical Peak CUDA VRAM Allocated**: **`760.38 MiB`** *(RTX 3060)*.
* **Theoretical MoE FLOP Sparsity**: **`87.5%`** ($k=2$ active of $N=16$ total experts per layer).

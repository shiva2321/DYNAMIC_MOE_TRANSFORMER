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

### 4. Component-Level Ablation Decomposition

| Architectural Configuration | Total Params | FineWeb $\Delta \mathcal{L}$ | Python $\Delta \mathcal{L}$ (Final Acc) | WikiText $\Delta \mathcal{L}$ | Mean $R_{\text{BWT}}$ | Relative Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Static MoE (30 Exp Baseline)** | $128.1\text{M}$ | `+0.3129 nats` | `+0.5108 nats` (37.42%) | `+0.2577 nats` | **`+0.3605 nats`** | Baseline linear softmax gating |
| **Ablation 2: No Spawning (Fixed 16 Exp)** | $128.9\text{M}$ | `+0.2089 nats` | `+0.2657 nats` (47.34%) | `+0.4250 nats` | **`+0.2999 nats`** | Phasor gating + bus improves drift by 16.8% |
| **Ablation 1: No Global Bus (`use_bus=False`)** | $128.9\text{M}$ | `-0.0437 nats` | `+0.4235 nats` (42.99%) | `+0.2343 nats` | **`+0.2047 nats`** | Spawning alone improves drift by 43.2% |
| **Full Universal Substrait System** | **$128.9\text{M}$** | **`-0.1804 nats`** | **`-0.0118 nats` (44.36%)** | **`+0.2314 nats`** | **`+0.0131 nats`** | **Full synergy: 27.5x lower drift than Static MoE** |

---

### 5. Key Scientific Findings & Architectural Decomposition

1. **Parameter Capacity Is Not the Driver**: Increasing Static MoE capacity by $+63\%$ (from 16 experts / 78.5M params to 30 experts / 128.1M params) only improved mean $R_{\text{BWT}}$ by a negligible $0.038\text{ nats}$. Python still degraded by $+0.511\text{ nats}$ (accuracy dropped to $37.4\%$).
2. **Autonomous Neurogenesis Prevents Subspace Cannibalization**: Disabling dynamic spawning increases drift to $+0.2999\text{ nats}$. Spawning allocates clean, dedicated expert coordinates in $\mathbb{C}^{2048}$ when encountering novel domains.
3. **The Global Workspace Bus Synchronizes Dendritic Gating**: Disabling the Global Bus (`use_bus=False`) increases drift to $+0.2047\text{ nats}$ and Python degradation to $+0.4235\text{ nats}$. Apical context broadcasting provides the top-down contextual synchronization that lets dendritic experts gate somatic firing.
4. **The Multiplicative Synergy**: Combining **Complex Phasor Coordinates ($\mathbb{C}^{2048}$)**, **Two-Compartment Dendritic Experts**, **Global Workspace Broadcasting**, and **Autonomous Neurogenesis** over an exemplar-stabilized trunk ($20\%$ replay) achieves **$27.5\times$ lower backward loss drift ($+0.0131\text{ nats}$ vs $+0.3605\text{ nats}$)**, preserving domain competence and unlocking positive backward transfer on FineWeb ($\Delta = -0.180\text{ nats}$).

---

## 5. Hardware Efficiency Profile

* **Physical Peak CUDA VRAM Allocated**: **`760.38 MiB`** *(RTX 3060)*.
* **Theoretical MoE FLOP Sparsity**: **`87.5%`** ($k=2$ active of $N=16$ total experts per layer).

---

## 6. Empirical Layer-Wise Routing Dynamics & Output Collapse

| Layer | Python Code | Natural Language Prose (FineWeb, WikiText, Stories) | Architectural Observation |
| :--- | :--- | :--- | :--- |
| **Layer 0** | **E7: 40.7%**, E10: 16.3% | E5, E6, E12, E9 (17%–27% each) | **Code vs. Prose Split**: Python isolates on `E7`; general prose shares general vocabulary experts. |
| **Layer 1** | **E7: 38.8%**, E2: 14.4% | Stories on **E12: 40.4%**; Web/Wiki on **E9: 22%–24%** | **Syntactic Differentiation**: Python and Stories show separate peaks; Web/Wiki overlap. |
| **Layer 2** | **E7: 49.7%**, E5: 19.3% | Stories on **E6: 36.1%**; Web/Wiki on **E3: 27%–29%** | **Semantic Divergence**: Python peaks on `E7`; Stories on `E6`; Web/Wiki on `E3` and `E12`. |
| **Layer 3** | **E5: 42.5%, E6: 37.4%, E12: 19.2%** | **E5 (42%–45%), E6 (34%–39%), E12 (18%–30%)** | **Output-Layer Collapse**: All 4 domains collapse onto 3 experts ($98.3\%\text{--}99.4\%$ mass). 13 of 16 experts unused. |

### Key Diagnostic Findings:
1. **Shallow/Mid Layers Separate Modalities (Code vs. Prose)**: Python consistently routes to `E7` (39%–50% of weight), while natural English prose partitions across `E12`, `E9`, `E6`, and `E3`.
2. **Output-Layer Bottleneck Mechanism**: Because the LM head is weight-tied to the 50,304-token embedding matrix, Layer 3 receives the strongest and most direct vocabulary cross-entropy gradient flow. Learnable phasor keys in Layer 3 undergo a self-reinforcing winner-take-all collapse onto `E5`/`E6`/`E12`.
3. **Engineering Roadmap**: Decoupling Layer 3 expert keys from unconstrained backpropagation using prototype drift-guards or key-freezing (analogous to `VSA_GROUND_REBUILD`) will prevent top-layer collapse in future scaling.

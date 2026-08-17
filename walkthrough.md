# Universal Substrait v2.0: Canonical Architecture & Evaluation Walkthrough

## 1. Overview & Verification Summary

This document serves as the permanent, version-controlled record of empirical benchmarks, architectural specifications, and evaluation methodologies for **Universal Substrait v2.0**.

All evaluation is executed through the canonical [`evaluate.py`](evaluate.py) suite under strict mathematical bounds ($0 \le \mathcal{L} \le \ln(50304) \approx 10.826$). Synthetic heuristics (e.g. regex syntax scoring) have been purged.

---

## 2. Dataset & Scale

* **Source**: HuggingFace streaming across 4 diverse domains:
  * `fineweb_edu`: Educational Web Reasoning (2.95M train / 150k val)
  * `python_code`: Python Algorithms & Instructions (2.45M train / 150k val)
  * `wikitext_facts`: Encyclopedic Knowledge & Science (2.95M train / 150k val)
  * `natural_stories`: Narrative Dialogue & Causal Logic (2.45M train / 150k val)
* **Total Volume**: 11,402,565 real tokens across 48,325 unique documents.
* **Pretraining Steps**: 3,000 steps ($9,216,000\text{ tokens}$ processed) under cosine learning rate decay ($6 \times 10^{-4} \to 6 \times 10^{-5}$) with micro-batch gradient accumulation.

---

## 3. Ground-Truth Multi-Domain Performance

*Evaluated on final master checkpoint: [`experiments/checkpoints/hyperspace_scaled_production_master.pt`](experiments/checkpoints/hyperspace_scaled_production_master.pt)*

| Domain Discipline | Validation Loss (nats) | Perplexity ($\exp(\mathcal{L})$) | Top-1 Accuracy | Top-5 Accuracy | Valid Bound ($\le 10.83$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Python Algorithms & Code** (*Alpaca Instructions*) | **`2.6776`** | **`14.55`** | **`56.48%`** | **`73.77%`** | **YES** |
| **Narrative Dialogue & Logic** (*TinyStories*) | **`3.0083`** | **`20.25`** | **`42.01%`** | **`67.43%`** | **YES** |
| **Encyclopedic Facts** (*WikiText-103*) | **`5.7244`** | **`306.26`** | **`21.95%`** | **`38.58%`** | **YES** |
| **Educational Web Reasoning** (*FineWeb-Edu*) | **`6.0457`** | **`422.31`** | **`19.27%`** | **`34.75%`** | **YES** |

---

## 4. Multi-Task Convergence vs. Sequential Catastrophic Forgetting

### A. Interleaved Multi-Domain Streaming
Under interleaved multi-task training (all 4 domains sampled concurrently per batch), all domains converged monotonically without cross-domain gradient interference:

$$\Delta \mathcal{L}_{\text{interleaved}} = \mathcal{L}_{3000, i} - \mathcal{L}_{500, i}$$

* **FineWeb-Edu**: $\mathcal{L}: 6.7263 \to 6.0457$ ($\Delta = -0.6806\text{ nats}$)
* **Python Code**: $\mathcal{L}: 3.5558 \to 2.6776$ ($\Delta = -0.8783\text{ nats}$)
* **WikiText-103**: $\mathcal{L}: 6.4297 \to 5.7244$ ($\Delta = -0.7052\text{ nats}$)
* **TinyStories**: $\mathcal{L}: 3.8340 \to 3.0083$ ($\Delta = -0.8256\text{ nats}$)

> [!NOTE]
> **Methodological Note**: Interleaved streaming evaluates **multi-task gradient compatibility**, not sequential catastrophic forgetting, because earlier domains are never removed from the stream.

### B. Isolated Sequential Continual Learning Benchmark
To evaluate true catastrophic forgetting ($R_{\text{BWT}}$), domains must be trained in strict sequential isolation without revisiting earlier domains. This is implemented in [`exp_sequential_continual_learning.py`](exp_sequential_continual_learning.py):
$$R_{\text{BWT}} = \frac{1}{T-1}\sum_{i=1}^{T-1}(\mathcal{L}_{T, i} - \mathcal{L}_{i, i})$$

---

## 5. Hardware Efficiency & Disambiguation

* **Physical Peak CUDA VRAM Allocated**: **`760.38 MiB`** *(measured via `torch.cuda.max_memory_allocated` during active inference on RTX 3060)*.
* **Theoretical MoE FLOP Sparsity**: **`87.5%`** *(Active $k=2$ experts routed out of $N=16$ total experts per layer)*.

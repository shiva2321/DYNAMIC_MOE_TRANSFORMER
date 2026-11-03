# M2 Reading Notes (R-1): Five-Dataset Text Benchmark & Frozen-Feature Baseline Audit

**File identity:** `docs/superpowers/handoff/M2_READING_NOTES.md`  
**Author:** Agent G (Gemini)  
**Date:** 2026-11-04  
**Context:** Milestone M2 preparation and baseline fidelity audit for GrowMoE (`D:\spherical_hyperspace_memory`).  
**Reviewer:** Claude (three-party workflow: Claude designs and reviews, User mediates, Gemini implements).  
**Integrity contract:** Grounded strictly in full-text reads of the published papers and source code. No fabricated numbers, no blog post data, no memory fills. Every fact has a locator (section, table/figure, page). Quotations are limited to under 15 words; all tables are reconstructed independently.

---

## 0. Executive Summary & Reading Status

| # | Paper | Short Name | Read in Full? | Version / Source Read | Page Count | Facts NOT FOUND |
|---|---|---|---|---|---|---|
| 1 | d'Autume et al. (2019) | MbPA++ | **YES** | arXiv:1906.01076v3 (NeurIPS 2019, 26 Nov 2019) | 13 pp. (9 main + 4 app.) | 2 (random seed / sample indices; text truncation token length) |
| 2 | Sun et al. (2020) | LAMOL | **YES** | arXiv:1909.03329v2 (ICLR 2020, 23 Dec 2019) | 14 pp. (9 main + 5 app.) | 3 (multitask score on 5 text datasets; random seed / sample indices; text truncation token length) |
| 3 | Zhou et al. (2023) | SimpleCIL / ADAM | **YES** | arXiv:2303.07338v2 (CVPR 2023 / IJCV 2024) | 16 pp. | 0 (text encoder: Not Covered) |
| 4 | McDonnell et al. (2023) | RanPAC | **YES** | arXiv:2307.02251v3 (NeurIPS 2023, 16 Jan 2024) | 32 pp. | 0 (text encoder: Not Covered) |
| 5 | Goswami et al. (2023) | FeCAM | **YES** | arXiv:2309.14062v3 (NeurIPS 2023, 12 Jan 2024) | 18 pp. | 0 (text encoder: Not Covered) |
| 6 | Hayes & Kanan (2020) | SLDA | **YES** | arXiv:1909.01520v3 (CVPRW 2020, 17 Apr 2020) | 10 pp. | 0 (text encoder: Not Covered) |

---

## 1. Paper 1: d'Autume et al. (2019) — MbPA++

**Title:** Episodic Memory in Lifelong Language Learning  
**Authors:** Cyprien de Masson d'Autume, Sebastian Ruder, Lingpeng Kong, Dani Yogatama (DeepMind)  
**Read in full:** YES, arXiv:1906.01076v3 (Proceedings of NeurIPS 2019, revised 26 Nov 2019, 13 pages).

### 1.1 Benchmark Protocol Details
- **Datasets & Original Source:** Five text classification datasets curated by Zhang et al. (2015), "Character-level Convolutional Networks for Text Classification" (NeurIPS 2015), downloaded from `http://goo.gl/JyCnZq` (§4.1, p. 5):
  1. *AGNews:* News article topic classification (4 classes).
  2. *Yelp:* Customer restaurant review sentiment ratings (5 classes).
  3. *DBPedia:* Wikipedia article ontology classification (14 classes).
  4. *Amazon:* Product customer review sentiment ratings (5 classes).
  5. *Yahoo:* Question and answer category classification (10 classes).
- **Sampling & Balanced Subsets:** Original datasets vary drastically in size (e.g., AGNews has 120,000 training examples, Yahoo has 1,400,000). The authors constructed balanced benchmarks by randomly sampling **115,000 training examples** and **7,600 test examples** per dataset, matching the size of the smallest dataset (AGNews) (§4.1, p. 5).
  - *Total training examples across 5 datasets:* 575,000 examples (§4.1, p. 5).
  - *Total test examples across 5 datasets:* 38,000 examples (§4.1, p. 5).
  - *Random seed / sample indices:* **NOT FOUND** in the paper text or appendix.
- **Label Spaces & Single Output Head:**
  - Classes of Yelp (5 classes) and Amazon (5 classes) represent identical 1-to-5 star rating semantics. The authors explicitly merged them into a single 5-class space (§4.1, p. 5).
  - Total classes in the classification head: $4 + 5 + 14 + 10 = 33$ classes (§4.1, p. 5).
  - The model uses a single unified linear output decoder predicting over all 33 classes (§2.2, p. 3).
- **Definition of "No Task ID at Test Time":**
  - No task or dataset descriptor is provided to the model during training or at test time (§2, p. 2; §2.2, p. 3).
  - The model does not observe when dataset boundaries are crossed. At test time, the classification head evaluates across all 33 classes simultaneously without dataset restriction (§2.2, p. 3).
- **Text Length & Truncation:** **NOT FOUND** in the text. BERT base defaults to 512 subword tokens, but exact truncation applied during data preprocessing is not specified.

### 1.2 Task Orders
Four orders were randomly chosen for the text classification stream (Appendix A, p. 12):
- **Order (i):** $\text{Yelp} \rightarrow \text{AGNews} \rightarrow \text{DBPedia} \rightarrow \text{Amazon} \rightarrow \text{Yahoo}$
- **Order (ii):** $\text{DBPedia} \rightarrow \text{Yahoo} \rightarrow \text{AGNews} \rightarrow \text{Amazon} \rightarrow \text{Yelp}$
- **Order (iii):** $\text{Yelp} \rightarrow \text{Yahoo} \rightarrow \text{Amazon} \rightarrow \text{DBPedia} \rightarrow \text{AGNews}$
- **Order (iv):** $\text{AGNews} \rightarrow \text{Yelp} \rightarrow \text{Amazon} \rightarrow \text{Yahoo} \rightarrow \text{DBPedia}$

### 1.3 Evaluation Metric & Main Results
- **Metric:** Macro-averaged classification accuracy across the five datasets evaluated after completing the final task in the sequence (§4.4, p. 6; Table 1).
- **Summary Results on Text Classification (Table 1, p. 6):**

| Method | Order (i) | Order (ii) | Order (iii) | Order (iv) | Average (class.-avg.) |
|---|---|---|---|---|---|
| **Enc-Dec** (standard BERT fine-tuning) | 14.8 | 27.8 | 26.7 | 4.5 | 18.4 |
| **A-GEM** (Chaudhry et al., 2019) | 70.6 | 65.9 | 67.5 | 63.6 | 66.9 |
| **Replay** (sparse replay 1%, no adaptation) | 67.2 | 64.7 | 64.7 | 44.6 | 57.8 |
| **MbPA** (local adaptation only, no replay) | 68.9 | 68.9 | 68.8 | 68.7 | 68.8 |
| **$\text{MbPA}^{++}_{\text{rand}}$** (random retrieval) | 59.4 | 58.7 | 57.1 | 57.4 | 58.2 |
| **MbPA++** (replay + local adaptation) | **70.8** | **70.9** | **70.2** | **70.7** | **70.6** |
| **MTL** (multitask joint upper bound) | 73.7 | 73.2 | 73.7 | 73.7 | 73.6 |

- **Full Per-Dataset Breakdown after Final Task (Table 4, Appendix B, p. 12):**

| Order | Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|---|
| **Order (i)** (Yelp $\rightarrow$ AG $\rightarrow$ DBP $\rightarrow$ Amz $\rightarrow$ Yah) | Enc-Dec | 1.1 | 0.0 | 0.0 | 4.3 | 68.7 |
| | A-GEM | 42.5 | 89.8 | 96.0 | 56.8 | 68.2 |
| | Replay | 38.2 | 83.9 | 95.4 | 50.3 | 67.9 |
| | MbPA | 42.0 | 90.4 | 96.1 | 52.0 | 63.9 |
| | $\text{MbPA}^{++}_{\text{rand}}$ | 35.2 | 80.4 | 88.2 | 45.9 | 47.2 |
| | MbPA++ | 45.7 | 91.6 | 96.3 | 54.6 | 65.6 |
| **Order (ii)** (DBP $\rightarrow$ Yah $\rightarrow$ AG $\rightarrow$ Amz $\rightarrow$ Yelp) | Enc-Dec | 0.0 | 0.0 | 3.1 | 57.9 | 48.9 |
| | A-GEM | 80.1 | 50.3 | 91.3 | 57.3 | 50.6 |
| | Replay | 75.0 | 53.7 | 86.0 | 58.1 | 50.7 |
| | MbPA | 96.0 | 58.4 | 89.0 | 54.4 | 46.6 |
| | $\text{MbPA}^{++}_{\text{rand}}$ | 82.0 | 41.7 | 81.9 | 47.1 | 40.8 |
| | MbPA++ | 95.8 | 63.1 | 92.2 | 55.7 | 47.7 |
| **Order (iii)** (Yelp $\rightarrow$ Yah $\rightarrow$ Amz $\rightarrow$ DBP $\rightarrow$ AG) | Enc-Dec | 0.0 | 0.0 | 1.3 | 11.4 | 93.9 |
| | A-GEM | 41.1 | 55.0 | 54.6 | 93.3 | 93.6 |
| | Replay | 23.6 | 36.8 | 25.0 | 94.5 | 93.8 |
| | MbPA | 43.3 | 60.9 | 51.6 | 95.8 | 92.5 |
| | $\text{MbPA}^{++}_{\text{rand}}$ | 35.2 | 33.6 | 42.1 | 92.3 | 82.3 |
| | MbPA++ | 44.3 | 62.7 | 54.4 | 96.2 | 93.4 |
| **Order (iv)** (AG $\rightarrow$ Yelp $\rightarrow$ Amz $\rightarrow$ Yah $\rightarrow$ DBP) | Enc-Dec | 0.0 | 0.0 | 0.0 | 14.1 | 8.1 |
| | A-GEM | 90.8 | 44.9 | 60.2 | 65.4 | 56.9 |
| | Replay | 70.4 | 33.2 | 39.8 | 46.1 | 33.4 |
| | MbPA | 89.9 | 42.9 | 52.6 | 62.9 | 95.1 |
| | $\text{MbPA}^{++}_{\text{rand}}$ | 78.4 | 37.7 | 45.8 | 42.4 | 82.9 |
| | MbPA++ | 91.8 | 44.9 | 55.7 | 65.3 | 95.8 |

### 1.4 Single Dataset & Multitask Upper Bounds (Table 6, Appendix C, p. 13)

| Dataset | Single Model Accuracy | Multitask Model Accuracy |
|---|---|---|
| **AGNews** | 93.8 | 94.0 |
| **Yelp** | 50.7 | 50.3 |
| **Amazon** | 60.1 | 58.8 |
| **Yahoo** | 68.6 | 67.1 |
| **DBPedia** | 30.5 | 95.9 |
| **Average** | **60.7** | **73.2** |

*(Note: Table 1 reports the multitask average across orders as 73.6%, whereas Table 6 reports a dedicated multitask run at 73.2%. Single-model DBPedia exhibits poor convergence in isolation at 30.5%, but reaches 95.9% in MTL).*

### 1.5 Architecture & Hyperparameters
- Backbone: Pretrained $\text{BERT}_{\text{BASE}}$ (12 layers, 12 attention heads, 768 hidden dimensions, 110M parameters) (§4.3, p. 6).
- Optimizer: Adam, base learning rate $3 \times 10^{-5}$, dropout 0.1 (§4.3, p. 6).
- Classification training batch size: 32 (§4.3, p. 6).
- Episodic Memory: Key-value store. Keys $\mathbf{u}_t$ are extracted by a frozen $\text{BERT}_{\text{BASE}}$ encoder using the `[CLS]` token (§2.3, p. 3).
- Sparse Experience Replay: Sample $S=100$ examples every $R=10,000$ steps (1% replay rate) (§4.2, p. 5).
- Local Adaptation: For query test input, retrieve $K=32$ nearest neighbors from memory using Euclidean distance between frozen key representations. Perform $L=30$ gradient steps to minimize Eq. (1):
  $$\mathbf{W}_i = \arg\min_{\tilde{\mathbf{W}}} \lambda \|\tilde{\mathbf{W}} - \mathbf{W}\|_2^2 - \frac{1}{K} \sum_{k=1}^K \log p(y_i^k \mid \mathbf{x}_i^k; \tilde{\mathbf{W}})$$
  with $\lambda = 0.001$, local learning rate tuned in $\{5 \times 10^{-3}, 1 \times 10^{-3}\}$ on Order (i) (§4.3, p. 6).
- Limited capacity evaluation (Table 2, §4.5, p. 7): With random write sampling, storing 10% of examples achieved 67.6% accuracy; storing 50% achieved 70.3%; storing 100% achieved 70.6%.

---

## 2. Paper 2: Sun et al. (2020) — LAMOL

**Title:** LAMOL: LAnguage MOdeling for Lifelong Language Learning  
**Authors:** Fan-Keng Sun, Cheng-Hao Ho, Hung-Yi Lee (National Taiwan University)  
**Read in full:** YES, arXiv:1909.03329v2 (ICLR 2020 conference paper, revised 23 Dec 2019, 14 pages).

### 2.1 Benchmark Protocol Details & Discrepancies with Paper 1
- **Datasets:** Evaluates on the four decaNLP tasks + SST, and specifically reproduces the text classification benchmark from d'Autume et al. (2019) using the same five datasets: AGNews (4 classes), Yelp (5), DBPedia (14), Amazon (5), Yahoo (10) (§4.1, p. 4; Appendix A, p. 11).
- **Sampling:** Cites d'Autume et al. (2019) and samples 115,000 training examples and 7,600 test examples per dataset (Table 1, p. 4).
- **Sampling seed / indices:** **NOT FOUND** in text or appendix.
- **Model Framework:** Uses the smallest pretrained GPT-2 model (117M parameters, 12 layers, 768 hidden dimensions) (§4.2, p. 4).
- **Data Formatting & Label Space Differences:**
  - Unlike d'Autume et al., who use an explicit 33-class linear classification head, LAMOL casts text classification as language modeling / question answering (§3.1, p. 3; Figure 2).
  - Every example is formatted with special tokens: `[CONTEXT] [QUESTION] ANS [ANSWER] EOS` for QA, and `GEN [CONTEXT] [QUESTION] ANS [ANSWER] EOS` for LM generation.
  - During test inference, greedy autoregressive decoding is performed after feeding the context, question, and `ANS` token until `EOS` (§3.1, p. 3).
  - Exact match (EM) between generated text and true class label string is used as accuracy (§4.1, Table 1, p. 4).
- **Replay Mechanism:** Rehearsal-free. When transitioning to task $i$, the model generates $\gamma |T_i|$ pseudo-examples from previous tasks by top-$k$ sampling ($k=20$) conditioned on task-specific tokens $\text{TASK}$ or universal $\text{GEN}$ (§3.2, p. 3; §4.2, p. 4).
- **Sampling Ratio & Loss Weight:** $\gamma = 0.20$ (20% pseudo-samples), LM loss weight $\lambda = 0.25$ (§4.2, p. 4). Trained for 9 epochs per task (§4.2, p. 4).

### 2.2 Text Classification Results (Table 5, §5.4, p. 8)
Evaluated across the exact same four task orders from d'Autume et al. (2019):

| Task Order | MbPA++ (d'Autume reported) | MbPA++ (LAMOL author reproduction) | $\text{LAMOL}_{\text{TASK}}^{0.2}$ |
|---|---|---|---|
| **Order (i)** | 70.8 | 74.1 | **76.7** |
| **Order (ii)** | 70.9 | 74.9 | **77.2** |
| **Order (iii)** | 70.2 | 73.1 | **76.1** |
| **Order (iv)** | 70.7 | 74.9 | **76.1** |
| **Average** | 70.7 *(70.6 in d'Autume)* | 74.2 | **76.5** |

*(Note: In Table 5, results for reproduced MbPA++ and LAMOL are averaged over 2 runs. The paired t-test $p < 0.01$).*

### 2.3 Single-Task GPT-2 Upper Bounds (Table 2, §4.1, p. 5)

| Dataset | Metric | GPT-2 Single-Task Score | Other Single-Task Score (d'Autume et al.) |
|---|---|---|---|
| **AGNews** | EM (acc) | 94.6 | 93.8 |
| **Amazon** | EM (acc) | 62.3 | 60.1 |
| **DBPedia** | EM (acc) | 99.1 | 30.5 |
| **Yahoo** | EM (acc) | 73.9 | 68.6 |
| **Yelp** | EM (acc) | 67.7 | 50.7 |
| **Average** | **EM (acc)** | **79.52** | **60.74** |

### 2.4 Clarification on Multitask Upper Bound Claim
- **Finding:** A joint multitask score on the five text classification datasets was **NOT REPORTED** in LAMOL.
- In Table 5, only MbPA++, reproduced MbPA++, and LAMOL are tabulated.
- The claim in LAMOL's abstract ("approaches the multitasking upper bound within 2–3%") refers to their **3-task decaNLP experiment** (Table 3: $\text{LAMOL}_{\text{GEN}}^{0.2} = 79.7\%$ vs Multitasked $= 81.5\%$, gap $= 1.8\%$) and their **5-task decaNLP experiment** (Table 4: $\text{LAMOL}_{\text{TASK}}^{0.2} = 74.3\%$ vs Multitasked $= 76.6\%$, gap $= 2.3\%$).
- On the five text classification datasets, LAMOL's lifelong average of 76.5% is **3.02 points below** the GPT-2 single-task average of 79.52%.

---

## 3. Paper 3: Zhou et al. (2023) — SimpleCIL & ADAM

**Title:** Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need  
**Authors:** Da-Wei Zhou, Zi-Wen Cai, Han-Jia Ye, De-Chuan Zhan, Ziwei Liu (Nanjing University & NTU Singapore)  
**Read in full:** YES, arXiv:2303.07338v2 (CVPR 2023 oral / IJCV 2024, revised 5 Aug 2024, 16 pages).

### 3.1 Method Description & Equations
SimpleCIL addresses class-incremental learning (CIL) under frozen foundation backbones without exemplars.
1. **Feature Extraction:** Pretrained backbone $\phi: \mathcal{X} \rightarrow \mathbb{R}^d$ is kept strictly frozen throughout continual learning (§3.2, p. 3).
2. **Class Prototype Accumulation:** When new task $b$ with label set $Y_b$ arrives, for each class $i \in Y_b$, compute the average raw embedding vector:
   $$\mathbf{p}_i = \frac{1}{K_i} \sum_{j=1}^{|\mathcal{D}^b|} \mathbb{I}(y_j = i) \phi(\mathbf{x}_j), \quad K_i = \sum_{j=1}^{|\mathcal{D}^b|} \mathbb{I}(y_j = i) \quad (\text{Eq. 3, p. 3})$$
3. **Classifier Head Formulation:** Classifier weight matrix $W = [\mathbf{w}_1, \dots, \mathbf{w}_{|\mathcal{Y}_b|}]$ is formed by directly setting each class weight vector to its prototype: $\mathbf{w}_i = \mathbf{p}_i$ (§3.2, p. 3; §4.1, p. 4).
4. **Inference (Cosine Classifier):** Test queries are classified via cosine similarity between L2-normalized feature embeddings and L2-normalized class prototypes:
   $$f(\mathbf{x}) = \left(\frac{W}{\|W\|_2}\right)^\top \left(\frac{\phi(\mathbf{x})}{\|\phi(\mathbf{x})\|_2}\right) \quad (\text{Eq. 6, p. 4})$$
5. **Storage & Updates:** Stores only $d$-dimensional mean vectors $\mathbf{p}_i$ for seen classes. Zero SGD updates during incremental phases.
6. **ADAM / Aper Extension:** Adapts backbone parameters using only the first task $\mathcal{D}^1$ via parameter-efficient tuning (VPT/Adapter/SSF), then concatenates adapted and original frozen features $[\phi^*(\mathbf{x}), \phi(\mathbf{x})]$ and freezes them permanently for subsequent tasks (§4.1, p. 4).

### 3.2 Hyperparameters
- SimpleCIL has **zero hyperparameters**; the backbone is frozen and prototypes are unweighted sample averages.
- Cosine classifier has no temperature parameter (effective scale = 1.0) (§4.1, p. 4).

### 3.3 Benchmarks & Frozen-Feature Results
- Vision benchmarks: Split CIFAR-100, ImageNet-R (200 classes), ImageNet-A (200 classes), CUB-200, Cars-196, VTAB (§5, pp. 5–8).
- ViT-B/16-IN21K backbone frozen results (Table 1 / Table 2 in paper; reported in RanPAC Table 1):
  - *CIFAR-100 (10 tasks):* SimpleCIL (NCM only) achieves **83.4%**.
  - *ImageNet-R (10 tasks):* SimpleCIL achieves **61.2%**.
  - *ImageNet-A (10 tasks):* SimpleCIL achieves **49.3%**.
  - *CUB-200 (10 tasks):* SimpleCIL achieves **85.1%**.
  - *VTAB (5 tasks):* SimpleCIL achieves **88.4%**.
  - *Cars-196 (10 tasks):* SimpleCIL achieves **37.7%**.
- **Coverage of Text / Sentence Encoders:** **NOT COVERED**. The paper focuses entirely on computer vision (ViT, ResNet).

---

## 4. Paper 4: McDonnell et al. (2023) — RanPAC

**Title:** RanPAC: Random Projections and Pre-trained Models for Continual Learning  
**Authors:** Mark D. McDonnell, Dong Gong, Amin Parvaneh, Ehsan Abbasnejad, Anton van den Hengel (Australian Institute for Machine Learning, University of Adelaide)  
**Read in full:** YES, arXiv:2307.02251v3 (NeurIPS 2023, revised 16 Jan 2024, 32 pages).

### 4.1 Method Description & Equations
RanPAC resolves the non-Gaussianity and feature correlations of frozen embeddings using random nonlinear projections followed by recursive linear ridge regression.
1. **Random Projection:** Fixed projection matrix $\mathbf{W} \in \mathbb{R}^{L \times M}$ with i.i.d. Gaussian entries $\mathbf{W} \sim \mathcal{N}(0, 1)$ (or bipolar $\pm 1$) (§4.3, p. 4; Algorithm 1, p. 5).
2. **Nonlinear Activation:** Project feature $\mathbf{f}_{t,n} \in \mathbb{R}^L$ into dimension $M$:
   $$\mathbf{h}_{t,n} = \phi(\mathbf{f}_{t,n}^\top \mathbf{W}) \in \mathbb{R}^M \quad (\text{Eq. 4 context, p. 4})$$
   where $\phi(\cdot)$ is an element-wise nonlinearity (ReLU or squaring; ReLU is default).
3. **Sufficient Statistics Accumulation:**
   $$\mathbf{G} = \sum_{t=1}^T \sum_{n=1}^{N_t} \mathbf{h}_{t,n} \mathbf{h}_{t,n}^\top \in \mathbb{R}^{M \times M}, \quad \mathbf{C} = \sum_{t=1}^T \sum_{n=1}^{N_t} \mathbf{h}_{t,n} \mathbf{y}_{t,n}^\top \in \mathbb{R}^{M \times K} \quad (\text{Eq. 4, p. 4})$$
   where $\mathbf{y}_{t,n}$ is a one-hot label vector. Note that the $y$-th column $\mathbf{c}_y$ of $\mathbf{C}$ is the unnormalized sum of projected features for class $y$.
4. **Regularized Weight Solution & Inference:**
   $$\mathbf{W}_o = (\mathbf{G} + \lambda \mathbf{I})^{-1} \mathbf{C} \in \mathbb{R}^{M \times K} \quad (\text{Appendix B.3, Eq. 10, p. 15})$$
   $$y_{\text{test}} = \arg\max_{y \in \{1, \dots, K\}} \mathbf{h}_{\text{test}}^\top (\mathbf{G} + \lambda \mathbf{I})^{-1} \mathbf{c}_y \quad (\text{Eq. 2, p. 3; Eq. 5, p. 4})$$
5. **Storage Footprint:** Gram matrix $\mathbf{G}$ ($M \times M$ float64) plus class matrix $\mathbf{C}$ ($M \times K$ float64).

### 4.2 Hyperparameters & Selection Strategy
- **Projection Width $M$:** $M = 10,000$ by default (§4.5, p. 5). Table A5 demonstrates scaling on CIFAR-100:
  $M=100 \rightarrow 71.6\%$, $M=400 \rightarrow 83.9\%$, $M=1250 \rightarrow 86.8\%$, $M=2500 \rightarrow 87.7\%$, $M=5000 \rightarrow 88.4\%$, $M=10000 \rightarrow 88.8\%$, $M=15000 \rightarrow 89.0\%$. Below $M=1250$, accuracy drops below linear ridge without RP.
- **Ridge Parameter $\lambda$ Selection:** Swept across 17 orders of magnitude, $\lambda \in \{10^{-8}, 10^{-7}, \dots, 10^8\}$, on an **80/20 train/validation split of the training data for the current task** (Appendix C.1, p. 16). $\lambda$ is chosen to minimize MSE on the 20% validation split. It is **never tuned on test data**.

### 4.3 Benchmarks & Frozen-Feature Results (Table 1, §5, p. 6)
Backbone: ViT-B/16-IN21K. Results for continual learning without Task 1 PETL ("No Phase 1", perfectly frozen backbone matching our setup):

| Benchmark | Tasks ($T$) | SimpleCIL (NCM only) | Linear Ridge (No RPs, No Phase 1) | RanPAC (Frozen backbone, No Phase 1) | RanPAC Full (+ Task 1 PETL) |
|---|---|---|---|---|---|
| **CIFAR-100** | 10 | 83.4% | 87.0% | **89.0%** | 92.2% |
| **ImageNet-R** | 10 | 61.2% | 67.6% | **71.8%** | 77.9% |
| **ImageNet-A** | 10 | 49.3% | 54.4% | **58.2%** | 62.4% |
| **CUB-200** | 10 | 85.1% | 86.8% | **88.7%** | 90.3% |
| **OmniBenchmark** | 10 | 73.1% | 73.1% | **78.2%** | 79.9% |
| **VTAB** | 5 | 88.4% | 91.9% | **92.2%** | 92.2% |
| **Cars-196** | 10 | 37.7% | 62.2% | **66.0%** | 77.5% |

- **Coverage of Text / Sentence Encoders:** **NOT COVERED** for text classification. In Appendix F.9 (p. 28), CLIP language representations are evaluated purely as continuous regression targets for image classification, not as text classification inputs.

---

## 5. Paper 5: Goswami et al. (2023) — FeCAM

**Title:** FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning  
**Authors:** Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, Joost van de Weijer (CVC Barcelona)  
**Read in full:** YES, arXiv:2309.14062v3 (NeurIPS 2023, revised 12 Jan 2024, 18 pages).

### 5.1 Method Description & Equations
FeCAM replaces isotropic Euclidean distances with anisotropic Mahalanobis distances using per-class covariance matrices stabilized by shrinkage and correlation normalization.
1. **Class Prototypes:**
   $$\bm{\mu}_y = \frac{1}{|X_y|} \sum_{x \in X_y} \phi(x) \quad (\text{Eq. 2, p. 2})$$
2. **Tukey's Ladder of Powers Transformation:** To reduce distribution skewness and Gaussianize features (§3.2, Eq. 9, p. 4):
   $$\tilde{\phi}(x) = \begin{cases} \phi(x)^\lambda & \text{if } \lambda \neq 0 \\ \log(\phi(x)) & \text{if } \lambda = 0 \end{cases} \quad (\lambda = 0.5)$$
   *Critical condition:* Applied only to non-negative features (e.g. ReLU features). When using ViT with negative feature entries, **Tukey transformation is explicitly omitted** (§7, p. 12).
3. **Class Sample Covariance Matrix:**
   $$\mathbf{\Sigma}_y = \frac{1}{|X_y|} \sum_{x \in X_y} (\tilde{\phi}(x) - \tilde{\bm{\mu}}_y)(\tilde{\phi}(x) - \tilde{\bm{\mu}}_y)^\top \quad (\text{Algorithm 1, p. 14})$$
4. **Covariance Shrinkage (Eq. 8, p. 4):**
   $$\mathbf{\Sigma}_s = \mathbf{\Sigma}_y + \gamma_1 V_1 \mathbf{I} + \gamma_2 V_2 (\mathbf{1} - \mathbf{I})$$
   where $V_1 = \frac{1}{d} \operatorname{tr}(\mathbf{\Sigma}_y)$ is the average diagonal variance, and $V_2$ is the average off-diagonal covariance of that class. $\mathbf{1}$ is an all-ones matrix.
5. **Correlation Matrix Normalization (Eq. 7, p. 4):**
   Crucial step: because different classes have widely varying variances, direct Mahalanobis distance causes class scale imbalance. FeCAM normalizes the shrunk covariance to a correlation matrix (diagonal elements equal to 1):
   $$\hat{\mathbf{\Sigma}}_y(i, j) = \frac{(\mathbf{\Sigma}_s)_y(i, j)}{\sqrt{(\mathbf{\Sigma}_s)_y(i, i)} \sqrt{(\mathbf{\Sigma}_s)_y(j, j)}}$$
6. **Inference / Distance Metric (Eq. 10, p. 4):**
   $$y^* = \arg\min_{y \in \{1, \dots, Y\}} (\tilde{\phi}(x) - \tilde{\bm{\mu}}_y)^\top (\hat{\mathbf{\Sigma}}_y)_s^{-1} (\tilde{\phi}(x) - \tilde{\bm{\mu}}_y)$$
7. **Common Covariance Alternative (Eq. 6, p. 4):** FeCAM also formulates a running shared pooled covariance across tasks, but reports superior results with per-class correlation-normalized covariances.

### 5.2 Hyperparameters
- $\gamma_1, \gamma_2$ in Eq. (8) (§7, p. 12):
  - Many-shot CIFAR100, TinyImageNet, ImageNet-Subset: $\gamma_1 = 1, \gamma_2 = 1$.
  - Split-CIFAR100, CORe50: $\gamma_1 = 1, \gamma_2 = 0$.
  - Split-ImageNet-R: $\gamma_1 = 10, \gamma_2 = 10$.
  - Few-shot CIL: $\gamma_1 = 100, \gamma_2 = 100$.
- Hyperparameters were chosen via validation sensitivity sweeps on CIFAR-100 $T=5$ (Figure 9, p. 12).

### 5.3 Benchmarks & Frozen-Feature Results
- Vision benchmarks: CIFAR-100, TinyImageNet, ImageNet-Subset, ImageNet-R, CORe50 (§4, pp. 6–9).
- On Split ImageNet-R with ViT-B/16 (Table 2, p. 7):
  - SimpleCIL / NCM: **66.4%**
  - FeCAM: **74.7%** (+8.3% improvement over Euclidean prototype).
- **Coverage of Text / Sentence Encoders:** **NOT COVERED**. Evaluates solely vision models (ResNet-18, ViT-B/16).

---

## 6. Paper 6: Hayes & Kanan (2020) — SLDA

**Title:** Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis  
**Authors:** Tyler L. Hayes, Christopher Kanan (Rochester Institute of Technology)  
**Read in full:** YES, arXiv:1909.01520v3 (CVPR Workshop on Continual Learning, revised 17 Apr 2020, 10 pages).

### 6.1 Method Description & Equations
Deep Streaming LDA (SLDA) maintains running class means and a single running shared covariance matrix, computing closed-form Gaussian generative linear decision boundaries.
1. **Model Architecture:** Fixed feature extractor $G(\mathbf{X}) \in \mathbb{R}^d$ and streaming linear head $F(\mathbf{z}) = \mathbf{W}\mathbf{z} + \mathbf{b}$ (§3, Eq. 1, p. 3).
2. **Streaming Mean Update (Eq. 2-3, p. 3):**
   When sample $(\mathbf{z}_t, y)$ arrives:
   $$\bm{\mu}_{(k=y, t+1)} = \frac{c_{(k=y,t)} \bm{\mu}_{(k=y,t)} + \mathbf{z}_t}{c_{(k=y,t)} + 1}, \quad c_{(k=y, t+1)} = c_{(k=y, t)} + 1$$
3. **Streaming Shared Covariance Update (Eq. 4-5, p. 3):**
   $$\mathbf{\Sigma}_{t+1} = \frac{t \mathbf{\Sigma}_t + \Delta_t}{t + 1}, \quad \Delta_t = \frac{t (\mathbf{z}_t - \bm{\mu}_{(k=y,t)})(\mathbf{z}_t - \bm{\mu}_{(k=y,t)})^\top}{t + 1}$$
4. **Shrinkage Regularization (Section 3, p. 3):**
   $$\mathbf{\Lambda} = [(1 - \varepsilon)\mathbf{\Sigma} + \varepsilon \mathbf{I}]^{-1}, \quad \varepsilon = 10^{-4}$$
5. **Closed-Form Weights & Biases (Eq. 6-7, p. 4):**
   $$\mathbf{w}_k = \mathbf{\Lambda} \bm{\mu}_k, \quad b_k = -\frac{1}{2} (\bm{\mu}_k^\top \mathbf{\Lambda} \bm{\mu}_k)$$
   $$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} (\mathbf{w}_k^\top \mathbf{z} + b_k)$$
6. **Variants:** Frozen covariance (computed on initial base task and frozen) vs. Plastic covariance (continuously updated on each sample).
7. **Storage Footprint:** $K \times d$ floats for means + $d \times d$ floats for shared covariance. Exactly $K + d$ vectors of dimension $d$.

### 6.2 Hyperparameters & Selection Strategy
- Shrinkage parameter $\varepsilon = 10^{-4}$ (0.0001).
- Covariance initialization: Initialized on base initialization data using the Oracle Approximating Shrinkage (OAS) estimator (§4.3, p. 4).

### 6.3 Benchmarks & Reported Numbers (Table 1, §4.4, p. 5)
- Evaluated on ImageNet ILSVRC-2012 (100 base classes, 900 streaming classes) and CORe50 (50 objects, 10 categories). Metric: Normalized accuracy $\Omega_{\text{all}} = \frac{1}{T} \sum_{t=1}^T \frac{\alpha_t}{\alpha_{\text{offline}, t}}$.
- Main results (Table 1, ResNet-18):
  - *ImageNet:* SLDA (Fixed $\mathbf{\Sigma}$) $\Omega_{\text{all}} = \mathbf{0.748}$; SLDA (Plastic $\mathbf{\Sigma}$) $\Omega_{\text{all}} = \mathbf{0.752}$ (beats iCaRL at 0.692).
  - *CORe50 (class instance order):* SLDA (Fixed $\mathbf{\Sigma}$) $\Omega_{\text{all}} = \mathbf{0.913}$; SLDA (Plastic $\mathbf{\Sigma}$) $\Omega_{\text{all}} = \mathbf{0.959}$ (beats iCaRL at 0.845).
- **Coverage of Text / Sentence Encoders:** **NOT COVERED**. Vision only.

---

## 7. Search for Closed-Form & Prototype Heads on the Five-Dataset Text Benchmark

### Search Execution
A systematic literature search was conducted across arXiv, Google Scholar, ACL Anthology, and Semantic Scholar for closed-form heads, prototype classifiers, and streaming discriminant heads applied to frozen text encoders on the 5-dataset benchmark:
- *Query 1:* `"SimpleCIL" "AG News" "DBpedia" "Yelp" "Yahoo"`
- *Query 2:* `"RanPAC" ("AG News" OR "AGNews" OR "DBpedia" OR "Yelp")`
- *Query 3:* `"SimpleCIL" ("MbPA++" OR "LAMOL")`
- *Query 4:* `"AG News" "Yelp" "Amazon" "DBpedia" "Yahoo" "prototype" OR "nearest class mean" "continual"`
- *Query 5:* `"Streaming Linear Discriminant Analysis" "AG News"`

### Search Findings
**NOT FOUND after searching.**
- The continual NLP literature following d'Autume et al. (2019) and Sun et al. (2020)—such as PMR (Ho et al., IEEE TNNLS), IDBR (Huang et al., EMNLP 2021), and PCLL (Wang et al., 2022)—exclusively fine-tunes Transformer encoders (BERT or GPT-2) using parameter rehearsal, episodic replay, adapter expansion, or generative pseudo-replay.
- The computer vision closed-form literature (SimpleCIL, RanPAC, FeCAM, SLDA, Continual-CLIP, AdaGauss) exclusively tests image backbones (ViT, ResNet) on vision datasets (CIFAR-100, ImageNet-R/A, CUB-200, CORe50).
- **Zero published papers** report performance numbers for SimpleCIL, RanPAC, FeCAM, or SLDA heads on frozen sentence embeddings (e.g., MiniLM, BERT `[CLS]`, or Qwen-Embedding) on the standard MbPA++/LAMOL five-dataset sequence.
- *Implication for M2:* Our evaluation of frozen-feature closed-form baselines (A5a, A5b, A6) on this benchmark will be the first reported reference numbers in the continual learning literature.

---

## 8. Fidelity Audit of Our Baseline Implementations

Code audited (read-only):
- `growmoe/memory_baselines.py`: `ClassMean`, `StreamingLDA`, `KnnMemory`
- `m1/baselines_extra.py`: `MahalanobisProto` (Arm A5b, FeCAM style), `RandomFeatureRidge` (Arm A6, RanPAC style)

### 8.1 Detailed Comparison Table

| Baseline Component | Our Implementation Choice | Published Paper Choice & Locator | Impact on Accuracy | Must Disclose as Limit in M1/M2? |
|---|---|---|---|---|
| **ClassMean: Input normalization** | Inputs are L2-normalized upon arrival (`_unit(x)`), and prototype is re-normalized (`mu / norm(mu)`). `memory_baselines.py#L27-29, L131, L140` | Raw unnormalized features averaged into $\mathbf{p}_i$; cosine classifier normalizes prototypes and test queries at inference. *SimpleCIL, §3.2 Eq. 3, §4.1 Eq. 6* | Unknown (likely neutral or slight help on angular metrics) | No (standard practice for cosine classification) |
| **ClassMean: Logit scaling** | Multiplies cosine logits by `scale = 20.0` before softmax. `memory_baselines.py#L119, L141` | Cosine similarity unscaled (effective temperature = 1.0). *SimpleCIL, §4.1 Eq. 6* | Neutral for top-1 argmax; produces sharper probabilities for NLL/entropy | No |
| **ClassMean: Covariance** | None (isotropic spherical mean). `memory_baselines.py#L122-124` | Same (isotropic cosine prototype). *SimpleCIL, §3.2 Eq. 3* | Neutral | No (Same as published) |
| **StreamingLDA: Input normalization** | Inputs are L2-normalized: `_unit(x).double()`. `memory_baselines.py#L166` | Raw unnormalized pooled features $\mathbf{z}_t$. *SLDA, §3 Eq. 1, p. 3* | Helps for sentence embeddings; unknown for vision features | **YES** |
| **StreamingLDA: Covariance update** | Weng et al. recursive formula: $(n \mathbf{\Sigma} + \frac{n}{n+1} \text{diff}\text{diff}^\top)/(n+1)$. `memory_baselines.py#L170` | Identical recursive update. *SLDA, §3 Eq. 4-5, p. 3* | Neutral | No (Same as published) |
| **StreamingLDA: Shrinkage formula** | Convex combination: $(1 - \alpha)\mathbf{\Sigma} + \alpha \mathbf{I}$. `memory_baselines.py#L179` | Identical: $(1 - \varepsilon)\mathbf{\Sigma} + \varepsilon \mathbf{I}$. *SLDA, §3, p. 3* | Neutral | No (Same as published) |
| **StreamingLDA: Shrinkage value** | Fixed `shrinkage = 1e-2` (0.01). `memory_baselines.py#L151` | Fixed $\varepsilon = 10^{-4}$ (0.0001). *SLDA, §3, p. 3* | Unknown (1e-2 pulls 100x harder toward isotropic sphere) | **YES** |
| **StreamingLDA: Covariance init** | Starts from zeros: `torch.zeros(d, d)`. `memory_baselines.py#L156` | Initialized on base dataset via OAS estimator. *SLDA, §4.3, p. 4* | Slight hurt early in stream; neutral once $N \gg d$ | **YES** (Streaming from scratch without pre-init) |
| **MahalanobisProto (A5b): Shrinkage target** | Shrinks class covariance toward **pooled covariance**: $(1 - \alpha)\mathbf{\Sigma}_c + \alpha \mathbf{\Sigma}_{\text{pooled}} + \epsilon \mathbf{I}$. `baselines_extra.py#L26-27, L102` | Shrinks toward **class diagonal variance and off-diagonal covariance**: $\mathbf{\Sigma} + \gamma_1 V_1 \mathbf{I} + \gamma_2 V_2 (\mathbf{1} - \mathbf{I})$. *FeCAM, §3.2 Eq. 8, p. 4* | Unknown (shrinking toward pooled covariance couples classes; FeCAM keeps them decoupled) | **YES** (Major structural deviation from FeCAM) |
| **MahalanobisProto (A5b): Correlation normalization** | **OMITTED**. Inverts unnormalized shrunk covariance directly. `baselines_extra.py#L106` | **MANDATORY**. Normalizes covariance to correlation matrix: $\mathbf{\Sigma}(i,j) / (\sigma_i \sigma_j)$. *FeCAM, §3.2 Eq. 7, p. 4* | **HURTS** (classes with small variance will dominate distance scores, causing severe class imbalance) | **YES** (Critical limitation; must be named) |
| **MahalanobisProto (A5b): Tukey transformation** | **OMITTED**. `baselines_extra.py#L58` | Tukey's ladder ($\phi(x)^{0.5}$) for positive features; omitted for ViT with negative values. *FeCAM, §3.2 Eq. 9, §7 p. 12* | Neutral (sentence embeddings have negative entries, so FeCAM rule omits Tukey anyway) | No (Matches FeCAM rule for features with negative entries) |
| **MahalanobisProto (A5b): Input normalization** | Inputs are L2-normalized: `_unit(x).double()`. `baselines_extra.py#L58` | Raw features (or Tukey transformed). *FeCAM, §3.2, p. 4* | Helps on cosine-distributed text embeddings | **YES** |
| **RandomFeatureRidge (A6): Projection width** | $D = 1024$. `baselines_extra.py#L153` | $M = 10,000$. *RanPAC, §4.5 p. 5, Table 1* | **HURTS** (RanPAC Table A5 shows $M=1024$ loses 2.3% accuracy vs $M=10000$ on CIFAR-100) | **YES** (A6 operates at 10x narrower projection) |
| **RandomFeatureRidge (A6): Projection matrix** | Gaussian scaled by $1/\sqrt{d}$: $\mathcal{N}(0, 1)/\sqrt{d}$. `baselines_extra.py#L166` | Standard normal $\mathcal{N}(0, 1)$. *RanPAC, Algorithm 1, p. 5* | Neutral (linear rescaling absorbed by ridge $\lambda$) | No |
| **RandomFeatureRidge (A6): Ridge $\lambda$ selection** | Fixed constant $\text{ridge} = 0.1$. `baselines_extra.py#L154` | Swept over 17 orders of magnitude ($10^{-8}$ to $10^8$) on 80/20 train split per task. *RanPAC, Appendix C.1, p. 16* | **HURTS** (fixed 0.1 may be suboptimal by orders of magnitude for diverse scales) | **YES** |
| **RandomFeatureRidge (A6): Prototype stage** | Accumulates $\mathbf{G} = \phi^\top \phi$ and $\mathbf{B} = \phi^\top \mathbf{Y}$ (one-hot). `baselines_extra.py#L199-202` | Identical: $\mathbf{G} = \sum \mathbf{h}\mathbf{h}^\top$, $\mathbf{C} = \sum \mathbf{h}\mathbf{y}^\top$. *RanPAC, §4.3 Eq. 4, p. 4* | Neutral | No (Mathematically identical) |
| **RandomFeatureRidge (A6): Task 1 PETL (Phase 1)** | **None** (frozen backbone throughout). `baselines_extra.py#L198` | Adapts Task 1 with SGD/PETL, but specifically includes ablation "No Phase 1". *RanPAC, Table 1, p. 6* | Matches RanPAC "No Phase 1" ablation | No (Accurately implements RanPAC's frozen-backbone mode) |
| **Incremental class handling (all)** | Mask unseen classes with $-\infty$ in logits (`masked_fill(~seen, -inf)`). | Dynamically expand classifier dimension as classes appear. | Neutral (exact mathematical equivalence for observed classes) | No |

---

## 9. Verification of Roadmap & Milestone Claims

We audited the claims in `docs/roadmap/RESEARCH_DIGEST.md` (§4) and `docs/roadmap/MILESTONES.md` (M1, M2) against the primary papers:

| # | Document & Section | Claimed Fact | Verification Status | Primary Evidence & Locator |
|---|---|---|---|---|
| 1 | `RESEARCH_DIGEST.md` §4 | "The five-dataset sequence of MbPA++ and LAMOL: AG News (4 classes), Yelp (5), DBpedia (14), Amazon (5), Yahoo (10)" | **CONFIRMED** | d'Autume et al. §4.1, p. 5; LAMOL Table 1, p. 4 & Appendix A, p. 11. |
| 2 | `RESEARCH_DIGEST.md` §4 | "115,000 training and 7,600 test examples per task" | **CONFIRMED** | d'Autume et al. §4.1, p. 5; LAMOL Table 1, p. 4. |
| 3 | `RESEARCH_DIGEST.md` §4 | "four task orders" | **CONFIRMED** | d'Autume et al. Appendix A, p. 12; LAMOL Table 5, p. 8. |
| 4 | `RESEARCH_DIGEST.md` §4 | "Yelp and Amazon label spaces are often merged" | **CONFIRMED** for MbPA++; **CORRECTED** for LAMOL | d'Autume et al. §4.1, p. 5 explicitly merges them into 5 classes (total 33 classes). In LAMOL, tasks are framed as autoregressive QA decoding without an explicit classification head. |
| 5 | `RESEARCH_DIGEST.md` §4 | "LAMOL (arXiv 1909.03329) reports being 2–3% below multitask training" | **CORRECTED** | In LAMOL, "within 2–3% of multitask" was measured on the decaNLP QA tasks (Table 3: 1.8% gap; Table 4: 2.3% gap). For the 5 text classification datasets, LAMOL did **not** run a multitask baseline; its 76.5% lifelong average is 3.02% below its single-task GPT-2 average (79.52%, Table 2). |
| 6 | `MILESTONES.md` §M2 | "The usual 115,000 training and 7,600 test examples per dataset; four task orders; no task id at test time; one head over all classes" | **CONFIRMED** | Exactly matches d'Autume et al. protocol (§2.2, §4.1, App. A). |
| 7 | `MILESTONES.md` §M1 | A5b is "FeCAM-style", A6 is "RanPAC-style" | **CONFIRMED with LIMITS** | Core mechanics match, but A5b omits correlation normalization and uses pooled shrinkage, while A6 uses $D=1024$ instead of $M=10000$ and fixed $\lambda=0.1$. |

---

## 10. Open Questions for the M2 Pre-Registration

The published papers leave several procedural decisions underspecified. Claude must decide these in the M2 pre-registration before test seeds run:

1. **Random Sampling Seed and Index Cache for the 115k/7.6k Split:**
   - Neither paper provides the pseudorandom seed, sampling script, or list of indices used to downsample the large datasets (Amazon 3.6M, Yelp 560k, DBpedia 560k, Yahoo 1.4M) to 115,000 training and 7,600 test examples.
   - *Action for M2:* Pre-register a deterministic sampling procedure with a fixed seed (e.g. `seed=20261102`), generate the index arrays, hash them, and cache them as immutable `.npy` files.
2. **Text Sequence Truncation Length:**
   - d'Autume et al. use $\text{BERT}_{\text{BASE}}$ (max 512 tokens), but do not specify whether sequences were truncated to 128, 256, or 512 tokens.
   - *Action for M2:* For modern sentence encoders (MiniLM, Qwen3-Embedding), fix maximum token sequence length (e.g., 256 or 512) in the pre-registration.
3. **Text Field Concatenation Format:**
   - Datasets like DBpedia and Yahoo have separate fields (title vs content, or question title vs question body vs answer). Zhang et al. (2015) concatenate title and body with newline or space.
   - *Action for M2:* Pre-register the exact string template: `"{title} {body}".strip()`.
4. **Input Normalization for Closed-Form Heads on Text Embeddings:**
   - Vision papers (RanPAC, FeCAM, SLDA) do not L2-normalize visual features from ResNet/ViT. However, sentence transformer embeddings are geometrically normalized to the unit sphere for cosine similarity.
   - *Action for M2:* Formally pre-register whether features entering `StreamingLDA`, `MahalanobisProto`, and `RandomFeatureRidge` are unit-normalized. (Our current code standardizes on `_unit(x)`).
5. **A5b Fix vs Baseline Limit:**
   - `MahalanobisProto` in `m1/baselines_extra.py` lacks FeCAM's correlation normalization (Eq. 7). Without it, per-class covariance inversion is heavily distorted by variance differences.
   - *Decision:* Should A5b be upgraded to include correlation normalization before M2, or kept as-is and declared as an uncalibrated Mahalanobis baseline?
6. **A6 Projection Width Budget ($D=1024$ vs $M=10,000$):**
   - RanPAC showed that $M=10,000$ is needed to achieve full performance on vision tasks. On text embeddings ($d=384$ for MiniLM), is $D=1024$ adequate, or should M2 evaluate $D \in \{1024, 4096, 10000\}$ to match RanPAC's published standard?

---

## 11. Reviewer corrections and verification (Claude, 2026-11-04) - READ THIS BEFORE USING SECTIONS 8 AND 9

**How it was checked.** The reviewer extracted the text of five of the six papers (MbPA++, LAMOL, RanPAC, FeCAM, SLDA; SimpleCIL's numbers were checked through RanPAC's Table 1) and searched them for 18 facts. **Verified as written:** 115,000 training and 7,600 test examples per dataset (MbPA++ section 4.1; LAMOL Table 1); 33 classes with Yelp and Amazon merged (MbPA++ section 4.1); the four task orders (MbPA++ appendix A); MbPA++ Table 1 rows and averages (as printed); MbPA++ 70.6 against 60.7 for single-dataset models; LAMOL abstract "only 2-3% worse than multitasking" and the decaNLP tables behind it (Table 3: multitasked 81.5; Table 4: multitasked 76.6, LAMOL 74.3); LAMOL Table 5 (70.7, 74.2, 76.5 and the per-order values); LAMOL Table 2 per-dataset scores; RanPAC M = 10000, the 80:20 split of the current task's training data, the sweep over 10^-8 to 10^8, Table 1 rows (NCM only, No Phase 1, No RPs or Phase 1) and Table A5; FeCAM shrinkage Eq. 8, the correlation-matrix normalization with unit diagonal, Tukey's transformation and its omission for ViT features with negative values; SLDA epsilon = 10^-4, the OAS initialization and Table 1 (0.748, 0.752, 0.913, 0.959, iCaRL 0.692 and 0.845). **Not rechecked by the reviewer:** MbPA++ Tables 4 and 6 (the per-dataset breakdown and the multitask column), FeCAM's gamma values and Split ImageNet-R numbers, SLDA's covariance update equations, the paper-3 text itself.

**Corrections**
1. **Section 8 audits the constructor defaults, not the arms as M1 ran them.** M1 tunes each baseline on development seeds (`results/m1/tuning/<bed>.json`). As run: A5a shrinkage 0.001 on B1, B2 and B3seq (the lowest value of its grid {0.001, 0.01, 0.1}: the published 1e-4 was never in the grid, so the true optimum may lie below the grid); A5b shrinkage 0.5 on all three beds (the highest value of {0.01, 0.1, 0.5}: the optimum may lie above it); A6 D = 8192 and ridge 1.0 on B1 and B3seq (D at its cap of 8192, ridge at the top of {0.001, 0.01, 0.1, 1}) and D = 4096, ridge 0.1 on B2. So the rows "A6 operates at D = 1024", "ridge fixed 0.1" and "SLDA shrinkage fixed 0.01" do NOT describe the scored run. The limits to name in the M1 report are instead: the grid edges just listed; RanPAC's own lambda sweep spans 10^-8 to 10^8 on a held-out 20 % per task, ours spans 10^-3 to 1 chosen on development seeds; RanPAC uses M = 10000, ours is capped at 8192.
2. **The impact labels "HURTS" and "Critical limitation" are the auditor's expectation, not measurements.** Read every impact column as "unknown until tested". The structural differences themselves (A5b has no correlation normalization, shrinks toward a pooled covariance instead of toward each class's own diagonal and off-diagonal means, adds a 1e-4 ridge, L2-normalizes inputs; no Tukey step, which matches FeCAM's rule for features with negative values) are confirmed by the code.
3. **Two numbers are derived, not printed:** the "Average 79.52 / 60.74" row of LAMOL Table 2 (computed from the printed per-dataset scores; the arithmetic is right) and "RanPAC Table A5 shows M = 1024 loses 2.3 points" (Table A5 lists 800 -> 86.2 %, 1250 -> 86.8 % and 10000 -> 88.8 % with lambda fixed at 100 for every M; 1024 is not listed).
4. **A quirk of the source:** MbPA++ Table 1 prints a Replay class.-avg of 57.8 while the mean of its four printed orders is 60.3; the notes transcribe the printed number correctly. Do not use 57.8 as a computed mean.
5. **Section 7 (no published result of these heads on this text sequence)** rests on five searches by Gemini; the reviewer did not repeat them. Phrase it in the preregistration as "none found in five searches", not as "none exist".

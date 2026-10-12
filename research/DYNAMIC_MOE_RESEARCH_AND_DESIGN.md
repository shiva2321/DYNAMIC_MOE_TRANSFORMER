# A self-growing, novelty-aware Mixture-of-Experts: research, brainstorm and design

*Research synthesis, idea bank, three candidate architectures, one recommended design (working name **PhyloMoE**), and a falsifiable test plan.*

2026-10-04 · status: **design for approval — no implementation code has been written** · companion: `BIBLIOGRAPHY_CHRONOLOGICAL.md` conventions apply.

Evidence labels: **[V]** published, checked at abstract/venue level this session (not full text) · **[R]** recalled from memory, unverified · **[M]** measured in this repo · **[H]** my hypothesis or design choice, **not** established.

---

## 1. What you asked for, restated as testable requirements

| # | Your words | Requirement (what a test can check) |
|---|---|---|
| R1 | "automatically recognizes data novelty" | A **calibrated** novelty decision with a stated, empirically verified false-novelty rate — no hand-tuned constants. |
| R2 | "not bound by data type" | One algorithm, one configuration, runs on heterogeneous streams (images, tabular, time series, text/bytes, …) because it only sees a unit-normalised latent. |
| R3 | "know and check if it's already known about that or not" | For every input the system outputs *known-by-which-expert* or *unknown* (abstain), with a coverage guarantee under exchangeability. |
| R4 | "spawns new expert from parent data DNA" | A child inherits from a chosen parent (weights, data signature, lineage record); inheritance measurably helps when regimes are related and does not hurt when they are not. |
| R5 | "or use or train the existing expert" | A decision rule among **use / train / extend / spawn**, driven by measured interference, not a fixed threshold. |
| R6 | "scaled vertically and horizontally, by layer and by system" | Operators **widen / deepen** an expert, **spawn / merge / prune** in a layer's pool, **add layer** to a stack, **add column** to a system — each triggered by evidence and function-preserving at the moment of growth. |
| R7 | "proper working useful and testable" | Unit tests with invariants, integration streams with ground-truth regimes, baselines, ablations, pre-registered pass/fail criteria, ≥ 10 seeds, no downloads needed. |

---

## 2. What the literature says (searched this session)

### 2.1 The families

| Family | Core idea | Representative work [V] | What it gives us | What it lacks for R1–R7 |
|---|---|---|---|---|
| **Nonparametric expert mixtures** | A Dirichlet-process prior decides when to add an expert; each expert has a generative + discriminative model; rejected samples wait in a short-term memory | **CN-DPM** (Lee et al., ICLR 2020, arXiv:2001.00689); CURL (Rao et al. 2019); DP-means (Kulis & Jordan 2012) | The *principle*: spawn = "no existing component explains this"; STM = evidence before commitment | Likelihood-based gating (known OOD failure, §2.3); heavy generative experts; no growth of expert size; no calibrated error rate |
| **Gated expert networks** | Per-task autoencoder gate decides relatedness; the most related old expert initialises the new one | **Expert Gate** (Aljundi et al., CVPR 2017); **SEED** (Rypeść et al., ICLR 2024): pick one expert by Gaussian class-distribution similarity and fine-tune only it | Parent selection by data similarity; train-one-expert-only | Needs task boundaries (Expert Gate); fixed expert count (SEED) |
| **Growing LLM-era MoE** | Add experts/gates for new distributions; freeze old; regularise outputs | **Lifelong-MoE** (Chen et al., ICML 2023); MoE-Adapters with a distribution discriminator (Yu et al., CVPR 2024); **DynMoE** (ICLR 2025): "top-any" gating and automatic expert count | Freezing + KL anchoring; ID/OOD auto-selector | Triggers are heuristics; no calibrated novelty; no vertical growth |
| **2026 continual-expert methods** | Centroid routing reactivates existing experts and instantiates adaptive-rank new capacity (**CRAM**, arXiv:2606.02502); subspace-affinity + distribution-aware adaptive threshold decides create/merge, routing is training-free (**MADE-IT**, 2604.22464); spawn gated sub-adapters when intra-adapter gradients conflict (**SpawnLoRA**, 2609.03150); growth only on "persistent optimisation evidence" and only after certification (**EMAN**, 2608.16930); rank-1 key–value experts that self-activate (**MoRAM**, ICML 2026, 2506.21035); training-free routing by feature alignment + subspace residual + prototype compatibility (**FDC**, 2609.34503) | These are our closest relatives: evidence-triggered growth, gradient-conflict spawn signal, non-parametric/centroid routing, certification before instantiation | Each solves one slice; none gives calibrated novelty + adapt-vs-spawn probe + vertical growth + lifecycle in one tested system |
| **Function-preserving growth** | New units/layers change nothing at birth | **Net2Net** (2016), **Firefly** (NeurIPS 2020), **GradMax** (ICLR 2022: new neurons chosen to maximise gradient), AutoGrow (2020), DEN (ICLR 2018), Progressive NNs (2016), LiGO (ICLR 2023), layer stacking (NeurIPS 2024), **"When to grow?"** (2024: timing changes regularisation and under/over-fitting risk), **"Grow, don't overwrite"** (2026: replication with a scaling correction keeps the expanded model *exactly* equal at initialisation) | The operators for R6 and the evidence that growth timing matters | Mostly single-task; no novelty linkage |
| **Plasticity and interference** | Long streams erode plasticity; gradient conflict predicts forgetting | **Loss of plasticity** (Dohare et al., Nature 2024: continual backprop, i.e. recycle low-utility units, keeps learning over 5,000 tasks); ReDo (dormant units); effective-rank collapse (Kumar 2021); GEM / A-GEM; PCGrad (conflict = negative cosine); tiny episodic memories are a strong baseline | Utility-based unit recycling; the conflict signal for adapt-vs-spawn; the replay baseline we must beat | — |

### 2.2 Calibrated novelty (the part most systems skip)

- **Conformal outlier p-values** (Bates, Candès, Lei, Romano, Sesia, *Annals of Statistics* 2023) give a p-value for "is this sample from the reference distribution?", with **exact finite-sample validity** under exchangeability and, because the p-values are positively dependent, **FDR control by Benjamini–Hochberg** [V]. Adaptive conformal inference (Gibbs & Candès 2021) keeps long-run error control under drift; conformal test martingales (Vovk et al. 2003; Fedorova et al. 2012) give an on-line, false-alarm-controlled change test [V].
- Data-type-agnostic scores that need only a vector: Mahalanobis (Lee et al. 2018), deep-kNN (Sun et al. 2022: non-parametric beats Gaussian assumptions), random-network-distillation error (Burda et al. 2019), ensemble disagreement [V].
- **Caveat that shapes the design:** likelihood can rate OOD data as *more* typical than in-distribution data (Nalisnick et al. 2019), and OpenOOD v1.5 shows no single detector dominates, especially for near-OOD [V]. ⇒ use a **non-parametric kernel score on normalised features, calibrated by conformal**, and keep a second *competence* signal rather than trusting one density.
- Open-world recognition (Bendale & Boult 2015) already pairs outlier rejection with incremental class addition [V].

### 2.3 Routing in a continual setting

- Learned softmax routers suffer **representation collapse** (X-MoE, NeurIPS 2022: score on a low-dimensional hypersphere helps) and **routing drift** in long streams. A 2026 study argues benchmarks with 4–10 tasks make routing look "saturated" because of shortcut fingerprints; a 34-task benchmark (FLEX) exposes it (arXiv:2608.01437) [V]. ⇒ **our benchmark must have many regimes and no trivial fingerprints.**
- Load-balancing losses are a train-time need for *fixed* pools (aux-loss-free bias, DeepSeek 2024). In a continual pool where experts *should* specialise unevenly, the right maintenance is lifecycle (merge/prune), not equalisation [H]. Your own live brain showed 34 / 33 / 20 / 0.2 % routing [your memory note].
- Adaptive compute per token: Ada-K (ACL 2024), AdaMoE null experts (2024), and "routing-free" experts that self-select (2026) [V].

### 2.4 Initialising the child

- **Sparse upcycling** copies a dense layer into experts (ICLR 2023); **Drop-Upcycling** (ICLR 2025) shows that copying alone slows specialisation and that *partial re-initialisation* of the copy restores diversity [V]. ⇒ child = parent with a *fraction* of weights re-drawn.
- **Merging** parents: task arithmetic / TIES-merging resolve sign interference (2023); expert merging needs permutation alignment (MC-SMoE, ICLR 2024); naive weight averaging of independently trained experts is unsafe [V]. Your current `sleep_consolidate_and_prune` averages state dicts by usage [M, read from code].
- **Hypernetwork "genomes"** generate expert weights from a small code (von Oswald et al. 2020; CLASP, arXiv:2610.01331 keeps the parameter footprint independent of the number of concepts) [V]. Attractive for "DNA", but unproven for our setting; deferred (Approach C).

### 2.5 Cross-domain ideas that survive scrutiny

| Domain | Idea | Use here |
|---|---|---|
| **Immune system** | *Negative selection* (Forrest 1994): detectors survive only if they do **not** match self; *clonal selection + affinity maturation* (de Castro & Von Zuben 2002): clone high-affinity cells with **mutation rate inversely related to affinity** [V] | "Known/unknown" = self/non-self test; **affinity-scaled hypermutation**: the less similar the new data is to the parent, the larger the fraction of cloned weights that is re-drawn |
| **Evolution** | NEAT *speciation* protects young structure from competition [R] | **Infant protection**: a new expert gets relaxed acceptance and a higher learning rate until it matures |
| **Neuroscience** | Adult neurogenesis in the dentate gyrus [R]; critical periods; neuromodulated plasticity | Surprise-scaled learning rate; maturity-scaled plasticity |
| **Operating systems** | Fork with **copy-on-write** | A child can start as a *delta overlay* on its parent (cheap fork, cheap merge) — Approach C |
| **Biology of memory** | Complementary learning systems, sleep consolidation | Idle-time merge/prune/distil (already in your repo) |
| **Statistics** | Sequential probability ratio test | Spawn when accumulated evidence exceeds a bound, not on one outlier |

---

## 3. What your own MoE already taught us (measured)

From `ishmn/spherical_dynamic_moe.py` (read this session) and your 2026-09-21 notes:

| Observed | Consequence for the design |
|---|---|
| Fixed `spawn_threshold = 0.40` never fires: min over concepts of max-resonance was 1.237 [your note] | **No fixed novelty constants.** Thresholds must come from calibration on the real score distribution, as error rates. |
| `forward()` spawns on the *single* most novel sample of a batch (`argmin(max_res)`) | **Probation pool + sequential evidence** before any birth. |
| Anchors `expert_u` are buffers set at birth and only moved by merge/mitosis | Gating should be **non-parametric and updated from assigned data**, so receptive fields follow the data. |
| Child = parent clone + N(0, 0.01) noise | Use **affinity-scaled partial re-initialisation** (Drop-Upcycling, immune hypermutation). |
| Merge averages state dicts by usage | **Distil** into the survivor instead; verify on a probe set. |
| Sleep merge is by anchor angle < 0.15 rad on an 8-D sphere; D = 8 gives no quasi-orthogonality (min angle 5.2° among 5,295 random concepts) [M, previous report] | Familiarity should live in a **higher-dimensional** normalised latent (d ≈ 128–256). |
| Routing stuck at 34/33/20/0.2 % with balancing off [your note] | Replace balance-by-loss with **lifecycle** (merge/prune/archive). |
| Optimizer groups added by hand when experts spawn | **One optimiser per expert**; no global param-group surgery. |
| Tests are shape/smoke only | Tests here assert **behavioural invariants** (§7). |

---

## 4. Brainstorm: the idea bank (before filtering)

1. **Conformal "know-ness"**: per-expert kernel density on the unit sphere (vMF-KDE = log-sum-exp of cosine similarities — exactly a modern-Hopfield energy), conformal p-values, BH across the batch. *Keep.*
2. **Probation pool** with graph clustering at a **data-derived** similarity (the typical nearest-neighbour similarity of known data) and a sequential evidence test. *Keep.*
3. **Interference probe**: first-order (gradient cosine) and actual (shadow-copy loss change) to choose *extend parent* vs *spawn*; the forgetting tolerance δ is the user-facing knob. *Keep.*
4. **Affinity-scaled hypermutation** for child initialisation. *Keep.*
5. **Propose–validate–commit growth** on shadow copies for width, depth and (later) layers, with back-off. *Keep.*
6. **Competence monitor** (prequential loss + change test) for "known input, wrong output" drift. *Keep.*
7. **Abstention**: output *unknown* when no expert is conformal and budget is exhausted. *Keep — it is what "knows what it doesn't know" means operationally.*
8. **Archive/revive**: dormant experts are frozen and compressed but their signatures still gate, so a returning regime wakes its own expert. *Keep.*
9. **Phylogeny-guided hierarchical routing** (route to a clade, then to a leaf) for E ≫ 100. *Defer.*
10. **Stem-cell pool of spare units** recruited into experts instead of allocating memory (neurogenesis analogue; ties to utility-based recycling). *Defer.*
11. **Hypernetwork genome / copy-on-write deltas / rank-1 self-activating experts** (parameter-efficient DNA). *Alternative (Approach C).*
12. **Generative-replay from signatures** (dream rehearsal). *Defer.*
13. **Mutating hyper-parameters across generations** (evolutionary meta-learning). *Reject for v1 (YAGNI).*
14. **Learned router** on top of the non-parametric gate. *Defer; only if gating limits accuracy.*

---

## 5. Three candidate architectures

| | **A. Calibrated phylogenetic MoE (recommended)** | **B. Bayesian-nonparametric MoE (CN-DPM-style)** | **C. Shared base + "DNA" deltas (copy-on-write / hypernetwork / rank-1)** |
|---|---|---|---|
| Gating | Non-parametric vMF-KDE + conformal; no learned router | DP prior × generative likelihood per expert | Self-activation by key match (MoRAM) or router over deltas |
| Novelty | Calibrated p-values, FDR control, probation | Posterior of "new component" under CRP | Key mismatch / low activation |
| Child init | Affinity-scaled clone of best-transfer parent | From STM data (scratch) | Parent's delta + perturbation / new latent code |
| Growth (R6) | Width, depth, layer, column; propose–validate–commit | Not covered (fixed expert size) | Rank grows; vertical growth unclear |
| Compute | Light: KDE over ≤ R prototypes per expert; probes are rare | Heavy: a VAE/flow per expert; likelihood OOD pitfalls | Light; but capacity-limited by low rank |
| Testability | High (every decision is a measurable statistic) | Medium (likelihood calibration is hard) | Medium (mechanism is new; fewer baselines) |
| Main risk | Gating quality limited by the latent; many parts to integrate | Likelihood miscalibration; slow | Hypernetwork instability; low-rank ceiling |
| Verdict | **Build first.** | Borrow the *principle* (evidence before birth) only. | Keep as a drop-in **child-initialisation mode** after A works. |

---

## 6. The design: PhyloMoE

### 6.1 Data flow

```
raw item (any type) ──► frontend φ ──► u ∈ S^{d-1}  (unit vector, d = 128–256)         ← R2
                                         │
        ┌────────────────────────────────┴───────────────────────────────┐
        │ KNOW?  per-expert vMF-KDE + conformal p_k(u);  p_nov = max_k p_k │            ← R1, R3
        └──────┬───────────────────────────────┬──────────────────────────┘
        known (BH at FDR q)                 novel
               │                               │
   USE / TRAIN assigned expert         probation pool 𝒫  ──► coherent cluster 𝒞 with
   (+ own replay, A-GEM projection)    sequential evidence?  ── no ──► wait / forget
   competence monitor ──alarm──┐            │ yes
                               ▼            ▼
                    best parent k*  =  argmax affinity, then interference probe
                         │ Δ_old ≤ δ_forget                │ Δ_old > δ_forget (or far)
                         ▼                                 ▼
                 EXTEND parent k*                    SPAWN child (affinity-scaled
                 (adapt + widen signature)           hypermutation of k*; DNA record)   ← R4, R5
                         └───────────────┬──────────────────┘
                     every T steps: GROW (width/depth/layer) · MERGE · ARCHIVE/PRUNE      ← R6
                     inference: prediction  or  UNKNOWN (abstain)
```

### 6.2 Components and their statistical contracts

| Component | Definition | Contract (a test can check) |
|---|---|---|
| **Frontend** | Any map to a fixed-size vector, then L2-normalisation. Default is *training-free*: feature-hashing / random-projection of arbitrary arrays or bytes (Johnson–Lindenstrauss preserves geometry) | Same code path for every data type; no per-type hyper-parameters |
| **Signature** (data DNA) | ≤ R prototypes per expert (reservoir), kernel κ_k chosen by leave-one-out likelihood; score(u) = LSE_j κ_k⟨p_j,u⟩ − log n_k (vMF-KDE) | Separates two distributions; LOO κ is stable; incremental updates are order-insensitive up to reservoir sampling |
| **Conformal gate** | Calibration scores = LOO scores of the prototypes; p_k(u) = (1 + #{cal ≤ score}) / (n_k + 1); **p_nov = max_k p_k**; BH at level q on each batch | **Empirical false-novelty rate ≤ q (+ binomial tolerance)** on held-out samples of known regimes; power > a stated floor on shifted ones |
| **Probation pool 𝒫** | Bounded buffer of novel samples; similarity graph with edge cos ≥ τ_in, τ_in = low quantile of known data's nearest-neighbour similarity; components of size ≥ m_min are candidate regimes | Isolated outliers never spawn; one coherent burst spawns exactly once |
| **Evidence test** | Σ_{u∈𝒞} [log f_{𝒞,LOO}(u) − max_k log f_k(u)] > log(1/α_spawn) | Type-I error on same-distribution clusters ≤ α_spawn |
| **Competence monitor** | Page–Hinkley / permutation-calibrated change test on prequential (test-then-train) loss | Fires on label-mapping change with known inputs; silent on stationary streams |
| **Interference probe** | (i) cos(g_new, g_old); (ii) shadow-parent adaptation for K steps → Δ_old = L_old(shadow) − L_old(parent); rule: **extend** iff Δ_old ≤ δ_forget and shadow ≈ child on 𝒞, else **spawn** | Decision flips as δ_forget varies; "extend" never degrades the parent by more than δ_forget on its replay set |
| **DNA / inheritance** | `Expert.inherit(parent, ρ(affinity), σ)`: copy, then re-draw a fraction ρ = ρ_max·(1 − affinity) of weights; record parent id, generation, birth step, novelty statistics | At ρ = 0, σ = 0 the child's outputs equal the parent's (function-preserving); lineage tree is acyclic and queryable |
| **Growth operators** | `widen(k)`: add hidden units with zero outgoing weights (or GradMax-style small init); `deepen()`: residual block with zero-initialised output; `add_layer()`: identity residual MoE layer | **Output unchanged at the moment of growth** (atol 1e-6) |
| **Growth policy** | Every T_check steps: detect plateau + under-fit (gap small); *propose* widen/deepen on shadow copies, train M steps on the reservoir, **commit** only if held-out loss improves by a margin with paired-bootstrap significance, else back off ×2 | Never grows when data-limited (large train–val gap); grows when a teacher is larger than the student |
| **Lifecycle** | INFANT → ACTIVE → DORMANT → ARCHIVED. **Merge** by distillation on the union of reservoirs, verified on a probe; **prune** only if not unique | Merge changes probe accuracy by ≤ δ_merge; archived experts still gate and can revive |
| **Optimisation** | One optimiser per expert; shared front-end fixed in v1 | Adding/removing experts never touches other experts' optimiser state |
| **Telemetry** | JSON-lines event for every spawn / extend / grow / merge / prune with the statistics that triggered it | Every structural change in a test is explained by a logged statistic |

### 6.3 User-facing knobs (everything else has defaults tuned once on a *development* stream family and then frozen)

| Knob | Meaning |
|---|---|
| **q (α_nov)** | Tolerated false-novelty rate / FDR |
| **δ_forget** | How much old-data performance one may trade to reuse a parent |
| **Budget** | Max experts / parameters / active FLOPs; at the cap the system abstains rather than overwrite |

### 6.4 Complexity

Gating ≈ B · E · R · d multiply-adds (E = 64, R = 256, d = 128 → ≈ 2 M per sample; can be pre-filtered by centroids). Expert update touches only the routed expert(s). Probes and growth proposals run on rare triggers with exponential back-off.

### 6.5 Known limitation, stated up front

Gating by **input** familiarity cannot distinguish two regimes with the *same inputs but different input→output mappings*. That case is caught only by the competence monitor and resolved by spawning, but at inference it needs a **context** signal (task, prompt, time) appended to the gate — the same role the GWT/Hopfield context plays in ISHMN. The benchmarks therefore include one such stream and report it separately.

---

## 7. Test and benchmark plan

### 7.1 Unit tests (written first, watched failing — TDD)

1. **Function preservation:** `widen`, `deepen`, `add_layer`, `inherit(ρ=0, σ=0)` leave outputs identical (atol 1e-6).
2. **Signature:** vMF-KDE separates two Gaussians on the sphere; LOO κ selection; order-insensitivity.
3. **Conformal validity:** i.i.d. calibration vs test from the same distribution → empirical FPR within binomial bounds of q; shifted test → power above floor; BH FDR ≤ q on mixed batches.
4. **Probation:** isolated outliers → 0 spawns; one coherent cluster → exactly 1 spawn; two interleaved clusters → 2 (or documented merge).
5. **Interference probe:** constructed aligned vs conflicting tasks yield *extend* vs *spawn*.
6. **Growth policy:** teacher-student with a bigger teacher grows; data-limited setting does not.
7. **Lifecycle invariants:** lineage acyclic; counts consistent; merge keeps probe accuracy within δ_merge; no NaN; deterministic given seed (no Python `hash()` anywhere).

### 7.2 Integration streams (ground-truth regimes, all generated locally — no downloads)

Installed and verified: numpy 2.2, scipy 1.17, scikit-learn 1.5 (bundled digits 1797×64, wine 178×13, breast-cancer 569×30, iris), PyTorch 2.6 (CPU 6 threads, CUDA available), 516 k characters of English technical text (`pydoc_data`), 39,535 Python files (528 MB) as code text [M].

| Stream | Content | Tests |
|---|---|---|
| **S1 Heterogeneous types** | Images (digits) · tabular ×3 (wine, cancer, iris-like synthetic) · synthetic time-series windows (frequency classes) · English bytes vs Python-source bytes vs base64 | R2: one config, six data types |
| **S2 Permuted digits ×30 + revisits** | Same type, same statistics, different pixel permutations; earlier permutations return | Hard novelty; **revisit must not spawn** (R3) |
| **S3 Gradual drift** | Boundary or inputs rotate slowly | Should *extend*, not over-spawn |
| **S4 Outliers and label noise** | Isolated outliers, 5–20 % noise | ≤ 1 % spurious spawns |
| **S5 Capacity ladder** | Teacher MLPs of increasing width/depth | R6: grow to the needed size, not 2× |
| **S6 Long horizon, similar format** | 60–100 regimes generated by different random linear maps of the same latent family | Avoids the "routing saturation" pitfall |
| **S7 Same inputs, new mapping** | Label permutation with identical inputs, with and without context | §6.5 limitation, reported separately |

### 7.3 Baselines and ablations

**Baselines:** B0 sequential fine-tune (forgetting floor) · B1 joint i.i.d. training (accuracy ceiling) · B2 single network + reservoir replay (strong simple baseline) · B3 static learned-router MoE with fixed E · B4 **oracle-routed** one-expert-per-regime (routing ceiling) · B5 DP-means gating with a fixed λ (threshold sensitivity) · B6 CN-DPM-style gating with NN experts (simplified).

**Ablations:** no conformal calibration · no probation · scratch / clone / affinity-scaled init · no interference probe (always spawn / never spawn) · no replay · no growth · familiarity = Gaussian / kNN / vMF-KDE / RND.

### 7.4 Metrics

Novelty: false-spawn rate per 1,000 known samples, regime-level spawn precision/recall, time-to-spawn, known/novel AUROC, **calibration** (empirical FPR vs q). Learning: class-incremental accuracy matrix, final average accuracy, forgetting, **forward transfer** (samples to 90 % on a new regime, child vs scratch), **revisit routing accuracy**. Efficiency: experts, parameters, active parameters/sample, wall-clock, buffer memory. Robustness: sensitivity to q, δ_forget, κ grid; ≥ 10 seeds; mean ± 95 % CI; paired tests; **numeric grading, held-out hyper-parameter stream, no per-stream tuning.**

### 7.5 Pre-registered predictions [H] — written before any code, expected to be partly wrong

| # | Prediction | Counts as failure if |
|---|---|---|
| P1 | Empirical false-novelty rate ≤ 2 × q on known regimes (S1, S2, S4) | > 2 × q |
| P2 | Regime-level spawn recall ≥ 0.9 and spawned/true-regimes ≤ 1.2 on S1 and S2 | Outside either bound |
| P3 | Revisited regimes routed to their original expert ≥ 95 % (no re-spawn) | < 90 % |
| P4 | Final class-IL accuracy ≥ 0.90 × oracle-routed (B4) on S1, and above B2 (replay) with CI excluding 0 | Below B2 |
| P5 | Forgetting ≤ 2 points absolute on S1/S2 | > 5 points |
| P6 | Affinity-scaled child reaches 90 % accuracy in ≤ 60 % of the samples scratch needs on **related** regimes and is not worse on unrelated ones | No gain on related, or loss on unrelated |
| P7 | S5: parameters at matched accuracy ≤ 1.5 × the teacher-sized optimum; never > 2× | > 2× |
| P8 | S4: ≤ 1 % of injected outliers cause a spawn | > 3 % |
| P9 | S6 (≥ 60 regimes): routing accuracy degrades < 5 points from 10 → 60 regimes | > 10 points |
| P10 | Fixed-threshold ablation (no conformal) is *worse* on at least two of P1, P2, P8 | No difference |

---

## 8. Risks and how each would show up

| Risk | Symptom | Mitigation / response |
|---|---|---|
| Latent quality limits gating (hashed features lose locality) | Low known/novel AUROC on S2 | Report honestly; swap in a learned shared front-end with prototype refresh (v2) |
| Early over-splitting inside one regime | Spawned/true-regime ≫ 1 | Infant protection; τ_in from data; merge-by-distillation |
| Conformal exchangeability broken by drift | FPR creeps above q | ACI step; competence monitor |
| Probes too slow | Wall-clock dominated by probes | Rare triggers, back-off, small experts |
| Many moving parts | Hard to attribute gains | Ablations are part of the plan, not an afterthought |
| Benchmarks too easy | Everything near 100 % | S2, S6, S7 are hard by construction; oracle and replay baselines bound the claims |

---

## 9. Phased delivery

| Phase | Scope | Done when |
|---|---|---|
| **0 Spikes (throwaway)** | Check the novelty score + conformal gate on S1/S2 latents; check clone vs affinity-scaled init; check growth-trigger on S5 | Go/no-go on gating and init choices, in hours |
| **1 Core** | Frontend, signature, conformal gate, probation, interference probe, DNA/inheritance, expert growth, lifecycle, telemetry, baselines, S1–S4 | P1–P6, P8, P10 evaluated |
| **2 Vertical** | Stack of pools; add-layer probe; S5–S6 | P7, P9 evaluated |
| **3 System** | Add-column; hierarchical (clade) routing; copy-on-write DNA mode | Scaling curve to ≥ 100 regimes |
| **4 Integration** | Adapter to the `SphericalDynamicMoE` interface in ISHMN | Only after Phases 1–2 pass; not started without your say-so |

Implementation will live in a **new standalone package** (default name `growmoe/`) with tests in `tests/`; nothing in `ishmn/` is modified, and CPU is sufficient for Phases 0–2.

---

## 10. Decisions needed from you

1. Approach A as the build target (with C as a later child-initialisation mode)?
2. Delivery scope: Phase 1 only, Phases 1–2, or the full 0–3 arc?
3. Data: local-only (no downloads), or also standard benchmarks that must be downloaded (MNIST/CIFAR)?
4. Location: standalone `growmoe/` (recommended) or inside `ishmn/`?

---

## Appendix — References checked this session

**Nonparametric / gated experts:** CN-DPM, Lee et al., ICLR 2020, arXiv:2001.00689 [V]; Expert Gate, Aljundi et al., CVPR 2017 [V]; SEED, Rypeść et al., ICLR 2024, arXiv:2401.10191 [V]; CURL, Rao et al., NeurIPS 2019, arXiv:1910.14481 [V]; DP-means, Kulis & Jordan, ICML 2012, arXiv:1111.0352 [V]; open-world recognition, Bendale & Boult, CVPR 2015 [V]; task-free continual learning, Aljundi et al., CVPR 2019 [V]; encoders and ensembles for task-free CL, Shanahan et al., arXiv:2105.13327 [V]; AdaNet, Cortes et al., ICML 2017, arXiv:1607.01097 [V].
**MoE, growing and routing:** DynMoE, ICLR 2025, arXiv:2405.14297 [V]; Lifelong-MoE, ICML 2023, arXiv:2305.12281 [V]; MoE-Adapters, Yu et al., CVPR 2024 [V]; Ada-K, Huang et al., ACL 2024, arXiv:2403.07652 [V]; AdaMoE, arXiv:2406.13233 [V]; X-MoE, NeurIPS 2022, arXiv:2204.09179 [V]; auxiliary-loss-free balancing, arXiv:2408.15664 [V]; DeepSeekMoE, arXiv:2401.06066 [V]; Routing-Free MoE, arXiv:2604.00801 [V-abstract]; MC-SMoE, ICLR 2024, arXiv:2310.01334 [V]; expert pruning, arXiv:2402.14800 [V]; TIES-merging, arXiv:2306.01708 [V]; task arithmetic, ICLR 2023 [V].
**2026 continual-expert work:** CRAM arXiv:2606.02502; MADE-IT arXiv:2604.22464; SpawnLoRA arXiv:2609.03150; EMAN arXiv:2608.16930; MoRAM arXiv:2506.21035; FDC arXiv:2609.34503; routing saturation / FLEX arXiv:2608.01437; brain-inspired modularity arXiv:2609.25146; AGF utility arXiv:2603.12354; HyPro arXiv:2609.39550; NeuCME arXiv:2609.07009; CLASP arXiv:2610.01331 — all [V-abstract], author-reported results.
**Growth and initialisation:** Net2Net arXiv:1511.05641; GradMax arXiv:2201.05125; Firefly arXiv:2102.08574; AutoGrow arXiv:1906.02909; Learn-to-Grow arXiv:1904.00310; DEN arXiv:1708.01547; Progressive NNs arXiv:1606.04671; LiGO, ICLR 2023; stacking arXiv:2405.15319; MixtureGrowth arXiv:2311.04251; When to grow? arXiv:2401.03104; Grow, don't overwrite arXiv:2603.08647; sparse upcycling arXiv:2212.05055; Drop-Upcycling arXiv:2502.19261; hypernetworks for CL, von Oswald et al., ICLR 2020 — all [V].
**Plasticity / interference:** Loss of plasticity, Dohare et al., Nature 632:768–774 (2024); ReDo arXiv:2302.12902; implicit under-parameterisation arXiv:2010.14498; GEM (NeurIPS 2017); A-GEM (ICLR 2019); tiny episodic memories arXiv:1902.10486; PCGrad arXiv:2001.06782 — [V].
**Novelty / conformal:** conformal outlier p-values arXiv:2104.08279; adaptive conformal inference (NeurIPS 2021); plug-in martingales arXiv:1204.3251; inductive conformal martingales arXiv:1706.03415; Mahalanobis OOD (NeurIPS 2018); deep-kNN OOD (ICML 2022); RND arXiv:1810.12894; Nalisnick et al. arXiv:1810.09136; OpenOOD v1.5 arXiv:2306.09301 — [V].
**Front-ends:** Perceiver IO arXiv:2107.14795 [V]; Byte Latent Transformer (entropy-based patching, 2024) [V-title]; feature hashing (Weinberger et al. 2009) [R].
**Cross-domain:** CLONALG, de Castro & Von Zuben, IEEE TEC 2002 [V]; negative selection, Forrest et al. 1994 [V]; NEAT speciation (Stanley & Miikkulainen 2002), adaptive resonance (Carpenter & Grossberg 1987), growing neural gas (Fritzke 1995), adult neurogenesis, mixtures of experts (Jacobs et al. 1991; Jordan & Jacobs 1994), sparse MoE (Shazeer 2017) — [R].

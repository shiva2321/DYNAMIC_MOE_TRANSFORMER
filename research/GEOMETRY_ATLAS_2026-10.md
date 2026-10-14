# Geometry atlas for the whole general-intelligence system (research, 2026-10-06)

Written by Claude after a web research pass (about 45 searches/fetches, October 2026) across attention, normalisation, position, state tracking, memory, routing, continual learning, hierarchy, planning, latent reasoning, optimisation, uncertainty, multimodal grounding, quantisation and neuro-inspired codes.
Companion: `docs/superpowers/handoff/GEOMETRY_ATLAS_HANDOFF.md` (what Gemini builds and tests). Earlier results this builds on: `research/YGGDRASIL_ARCHITECTURE_RESEARCH.md`, `docs/COMPONENT_CARDS.md`.

Evidence tags: **[P]** used in a production-scale model (billions of parameters) per its own paper; **[L]** published/pre-printed, modest scale (I read the abstract or a summary, not the full paper unless noted); **[M]** measured by us; **[H]** our hypothesis, untested.
**Honest limits of this document:** most 2026 papers are only weeks or months old and not peer-reviewed; I read abstracts and summaries, not full texts or code; gains reported at 30B parameters need not appear at our 1-10M scale. Everything below is a *candidate to test*, nothing is adopted.

---------------------------------------------------------------------------------------------------------------------------
## 1. The short version

1. **The geometry that is winning in 2026 is not "curved embedding spaces". It is "put the *operators* and the *normalisations* on the right manifold".** Unit-sphere states and weights (nGPT, [P] 2x token efficiency at 30B); doubly-stochastic residual mixing (mHC, in DeepSeek-V4 [P]); orthogonal / Householder / rotation transitions with data-dependent structure (PaTH, DeltaProduct, Complex-KDA, Mamba-3, MapFormer [L]); cone-structured features for associative memory (KATA [L]); log-spaced memory timescales (PoST [L]); orthogonal-subspace updates for continual learning (O-LoRA family [L]).
2. **Hyperbolic space is not needed for hierarchy in language models.** Hierarchy is already encoded as orthogonal subspaces / simplices in flat space (Park et al., ICLR 2025 [L]) and emerges from word co-occurrence spectra (Nava & Wyart 2026 [L]); an audit found "hyperbolic" vision-language models quietly collapse to near-Euclidean (arXiv 2607.05268 [L]). This matches our own null results [M]. Hyperbolic stays a *tiny-dimension tool* (d <= 8) pending one more test (Z5).
3. **A unifying principle shows up in five independent places: constrain the geometry, control the step, snap the state.** Looped reasoners fail with "the right direction, the wrong step size" and a fixed quarter-step recovers 72-83 % of selected failures [L]; nGPT needs angular update control [P]; unsnapped rotation states drift (ours [M]: Z6 torus 100 % -> 9 % by length 256) while snapping removes the drift [M]; Stiefel/orthogonality constraints keep weights well-conditioned [L].
4. **Our own legacy system (ISHMN) already gestured at many of these ideas** (sphere attention with bounded logits, SIGReg/LeJEPA isotropy, Muon, Hopfield memory, grid-cell encoder, DeltaNet layer, vMF memory). What it never had is controlled verification. Track Z is, in large part, **verification of the geometric instincts of the legacy system against modern baselines**, with the same strict protocol as GrowMoE.
5. **Concrete target: a "Geometry Core" library of individually verified components** (section 7): spherical stream, orthogonal state with error control, cone/orthogonal associative memory, phase positions, a hierarchy index (spectral or hyperbolic, whichever wins), step-controlled latent loops, quasimetric planning heads, plus a geometry-health diagnostic used everywhere.
6. **Where I would bet** (probability that it survives our test, my judgement): sphere-normalised stream (moderate-high), structured-orthogonal state with chunk-level snapping (high that structure helps, moderate that snapping adds value over CKDA/PaTH), orthogonal/cone memory with normalised keys (moderate-high), step control for latent loops (moderate-high, very cheap), orthogonal-subspace growth for GrowMoE (moderate-high), PoPE-style position (moderate), quasimetric planning heads (moderate), spectral-Euclid beats or ties hyperbolic on real hierarchies (high), mHC at our scale (low-moderate, adopt only with the sphere stack).
7. **What I would not spend time on:** hyperbolic attention for language, tree search at small N, equivariant geometric-algebra transformers (only for data with true symmetries), sheaf networks (too early), anything justified only by "AGI".

---------------------------------------------------------------------------------------------------------------------------
## 2. What our own work already established (anchor)

| Established [M] | Consequence |
|---|---|
| Hyperbolic tree embeddings beat flat ones at d = 4-16 on a synthetic 1093-node tree (capacity, not speed) | keep as a small-d tool; test on a *real* hierarchy against the best flat method (Z5) |
| Snapped rotation state is exact on Z6/S3 to length 256; unsnapped rotations/torus drift; a GRU is equally exact | error control is the lever; need harder groups and a parallel formulation (Z3) |
| Hyperbolic attention: no LM gain after controls; not sparser; tree search slower than dense | do not pursue geometry-in-attention for LM |
| GrowMoE S1: routing loses 18 points versus oracle routing | routing geometry for growing pools is a live problem (Track G, section 3 row R) |
| Per-arm tuning and operating-point reporting change conclusions | every Track-Z experiment uses the same protocol |

---------------------------------------------------------------------------------------------------------------------------
## 3. The atlas, subsystem by subsystem

Columns: *Standard practice* | *Best-supported geometric alternative (evidence)* | *Our status / where it lives* | *Experiment*.

### 3A. Representation and training dynamics
| # | Subsystem | Standard | Geometric candidate (evidence) | Our status | Exp |
|---|---|---|---|---|---|
| A1 | Residual stream and normalisation | LayerNorm/RMSNorm, free-norm weights | **nGPT**: every embedding/hidden/weight vector on the unit hypersphere, learned per-dimension step sizes ("eigen learning rates"), no weight decay; paper claims 4-20x fewer steps [L], community reproduction 1.5-2x (1k ctx) to ~4x (4k ctx) at ~200B tokens (by visual inspection) [L]; **"Training nGPT" (Aug 2026): a 30B hybrid Mamba-2/Transformer MoE reaches the same validation loss with ~half the tokens**, recipe = logit-gradient preconditioning, log LR decay, GatedAdamW, angular update control, optional exploration [P] | ISHMN = sphere x R+ (unverified); `ishmn/spherical_attention.py` has bounded-logit cosine attention | **Z1** |
| A2 | Residual mixing | single residual stream | **mHC** (manifold-constrained hyper-connections): several residual streams mixed by doubly-stochastic matrices (Sinkhorn-Knopp, ~20 iterations in V4) to restore identity mapping; used in **DeepSeek-V4** (1.6T/49B active, 1M context) [P]; the abstract gives no isolated ablation | none | Z9 (optional, only with A1) |
| A3 | Weight geometry and optimiser | AdamW | **Muon** (orthogonalised updates) used in DeepSeek-V4 [P]; **manifold Muon** (Stiefel for attention, other manifolds per module; best = Stiefel on attention + "DGram" on MLP in a GPT-2 study; reversed assignment diverged) [L]; closed-form Stiefel Muon update (2608.06218) [L] | `ishmn/muon.py` exists | folded into Z1 as arms |
| A4 | Anti-collapse / isotropy | ad-hoc | **SIGReg / LeJEPA**: embeddings pushed toward isotropic Gaussian, heuristics-free; stable end-to-end world models (LeWorldModel 2026) [L] | `ishmn/sigreg.py`; ISHMN suffered "lattice collapse" ([[ishmn-structural-defects]]) | Z9 / diagnostics in Z0 |
| A5 | Geometry health monitoring | none | operating point (ours), effective rank, isotropy, intrinsic dimension, hubness [M-style, [H] for usefulness] | operating-point report in `yggformer/report.py` | **Z0** |

### 3B. Position, time and sequence mixing
| # | Subsystem | Standard | Geometric candidate (evidence) | Our status | Exp |
|---|---|---|---|---|---|
| B1 | Position | RoPE | **PoPE** (polar): separate content magnitude from position phase; zero-shot 10x length extrapolation, 124M-774M, beats YaRN (ICML 2026) [L]; **LieRE**: learned skew-symmetric generators, exp map to rotations, 2D/3D, up to 9.7 %/25.5 % gains and 3.5x training efficiency in vision (ICML 2025) [L]; **FoPE** (Fourier series, periodic extension) [L]; **MapFormer** (input-dependent exp of Lie-algebra generators = path integration, near-perfect OOD length generalisation on gating, 2D navigation, Dyck; parallel for commutative maps, sequential for non-commutative) [L] | torus phases (verified as identity maths only) | **Z4** |
| B2 | Data-dependent position / state | none | **PaTH**: accumulated data-dependent Householder products, FlashAttention-style kernel, beats RoPE on state tracking and extrapolation (NeurIPS 2025) [L] | none | Z3/Z4 |
| B3 | Time scales of memory | random/linear decay init | **PoST**: geometrically spaced log-decay rates + position-adaptive scaling, minimax-optimal rate; tested inside Mamba-2, RWKV-7, Gated DeltaNet, GLA, RetNet at 180M-440M [L] | none | folded into Z2 |
| B4 | Sequence mixing | softmax attention | hybrids: **Gated DeltaNet / Kimi Delta Attention / Qwen3-Next-style** linear layers + full attention; **DeepSeek-V4 CSA/HCA** (27 % of V3.2 FLOPs, 10 % KV at 1M context) [P]; **Sinkhorn/doubly-stochastic attention** keeps rank better with depth (small-scale evidence; skip connections still needed) [L]; **Complex KDA**: gate in [-1,1] and delta coefficient in [0,2] give rotations/reflections, tracks every finite subgroup of SO(3), modest LM gains [L] | our fused hyperbolic attention: no benefit | not a Z target (mainstream; revisit after Z1-Z3) |

### 3C. State, algorithms, memory
| # | Subsystem | Standard | Geometric candidate (evidence) | Our status | Exp |
|---|---|---|---|---|---|
| C1 | Exact discrete state / algorithm tracking | LSTM/GRU, CoT | **non-commutative state tracking with structured orthogonal transitions**: DeltaProduct (Householder products), Complex-KDA (rotations/reflections), **Mamba-3** (complex-valued state = data-dependent RoPE; solves parity/modular arithmetic where real SSMs are at chance) and its follow-up with an **input-dependent low-rank reflection** for non-commutative tracking (2609.28273) [L]; Grazzi et al.: parity needs a negative eigenvalue [L] | **snapped rotation state** (S3/Z6 exact to 256, same as a GRU) [M] | **Z3** |
| C2 | Associative memory capacity | KV cache; linear attention (capacity wall d) | **KATA**: feature maps from self-dual homogeneous cones (rank-one PSD) break the capacity wall: MQAR 0.985 at 16x OOD length with about a quarter of the KV entries, ~1.6x FlashAttention-2 throughput, 340M parameters, some fluency cost [L]; **triadic / tensor-product states** raise capacity from d to d^2 [L]; **delta rule** + normalised keys (orthogonal keys give lossless readout) [L]; **Hopfield = spherical codes**: optimal capacity is a point-arrangement problem on the hypersphere (NeurIPS 2024) [L]; **Titans/MIRAS/ATLAS**: test-time memory modules [L] | ISHMN memory = vMF + radial shells (`ishmn/memory.py`, `vsa_memory.py`): **never benchmarked against these** | **Z2** |
| C3 | Neuro-inspired codes | none | grid cells (torus), ring attractors, Vector-HaSH, TEM: phase codes for relational memory [L]; GridPE, MapFormer as transformer instances [L] | `ishmn/neuro/dentate_grid.py` (toroidal grid encoder) | Z4 (position side) |

### 3D. Routing, growth, continual learning (Track G territory)
| # | Subsystem | Standard | Geometric candidate (evidence) | Our status | Exp |
|---|---|---|---|---|---|
| R1 | Expert routing | learned linear router | **cosine router in a low-dimensional hypersphere** (X-MoE) avoids representation collapse [L]; BUT **"Equifinality" (Apr 2026): routing topology does not change LM perplexity** (62 runs, 76-84M; true mechanism gain ~1.2 %; cosine routers recover 67 % of the linear router's edge with far fewer parameters) [L]; the real problem in *class-incremental* learning is **stability of old-class routing when experts are added** (StaR-MoE: sensitivity-aware routing alignment) [L]; prototype gating without a trainable classifier (MILE) [L] | GrowMoE S1: oracle routing +18 pts; D7 diagnostic pending | hand to **Track G** after D7 |
| R2 | Forgetting | replay, EWC | **orthogonal-subspace allocation**: O-LoRA, OPLoRA (double-sided orthogonal projection, prevents forgetting on LLaMA-2/Qwen2.5 7B), NESS (null space from small singular values), SPACE-LoRA (protect old activation subspaces) [L] | GrowMoE uses replay + shrink-perturb | hand to **Track G** (Z8 list) |
| R3 | Lineage / phylogeny | flat list | hyperbolic embedding of the expert tree [H] (our small-d result supports the tool, not the need) | none | Z5 decides |

### 3E. Knowledge, reasoning, planning, uncertainty, grounding, efficiency
| # | Subsystem | Standard | Geometric candidate (evidence) | Our status | Exp |
|---|---|---|---|---|---|
| K1 | Hierarchy / taxonomy | flat embeddings | **hierarchical orthogonality + simplices in flat space** (Park et al.) [L]; **spectral coarse-to-fine structure from co-occurrence, hyperbolic unnecessary** (Nava & Wyart 2026) [L]; hyperbolic wins only at tiny d on synthetic trees [M] | `geometry/` hyperbolic tools | **Z5** (real local hierarchy) |
| K2 | Knowledge consistency | none | cellular sheaves / Hilbert bundles for local-to-global consistency (SheafAN beats GAT; 2026 theory) [L] | none | parked |
| P1 | Latent reasoning loops | fixed-depth, CoT | looped / recurrent-depth models; failure = "right direction, wrong step" -> **fractional step scaling** (quarter step recovers 72-83 % of failures) [L]; attractor/equilibrium reasoners for stable fixed points [L]; angular update control from nGPT [P] | ISHMN ACT deliberation (fixed after audit) | **Z6** |
| P2 | Planning / world model | symmetric distances, value nets | **quasimetric temporal-distance embeddings** (TMD, IQE, MRN, QuaD; asymmetric distances satisfy the triangle inequality, link distance to transit time) [L]; JEPA/SIGReg world models [L] | ISHMN world model uses symmetric geodesic distance (`w_fwd`, `w_bwd`) | **Z7** |
| U1 | Novelty / uncertainty | softmax confidence | **vMF mixtures on the sphere for OOD/novelty** (CIDER, MARVEL, 2026) [L]; conformal p-values [M, GrowMoE]; note: a normalised stream **removes the norm**, so uncertainty must live in an explicit scalar (kappa or a separate channel) [H] | GrowMoE signatures (vMF-KDE conformal) verified | design rule in Z1 |
| U2 | Multimodal grounding | CLIP-style | modality gap = alignment/uniformity conflict on the shared sphere; anisotropic spectra; centroid alignment helps grounding [L] | none | parked |
| U3 | Numbers / arithmetic | digit tokens | **periodic (Fourier/helical) number codes** with periods 2, 5, 10, 10^i (FoNE: precise single-token numbers, high data efficiency; LMs converge to periodic features; prime Fourier bases for modular arithmetic) [L] | torus phases = same family | Z9 (optional) |
| E1 | Quantisation / deployment | per-channel INT8 | **rotation-based**: QuaRot (Hadamard), SpinQuant (learned rotations; 4-bit W/A/KV within 2.9 points on LLaMA-2 7B), 2026 "transforms" studies [L] | none | Z9, after adoption |
| E2 | Symmetric / physical data | none | **GATr / L-GATr**: geometric-algebra tokens + ordinary scaled-dot attention, SOTA in high-energy physics [L] | none | only if a symmetric-data task appears |

---------------------------------------------------------------------------------------------------------------------------
## 4. Cross-cutting principles (the design thesis)

* **P-1 Sphere for state, scalar for importance.** Keep unit-norm vectors for content and an explicit scalar channel for salience/uncertainty (this reconciles ISHMN's S^{D-1} x R+ idea with nGPT's finding that removing the norm from the stream helps optimisation).
* **P-2 Norm-preserving, data-dependent operators.** Transitions and mixings should be orthogonal, Householder-structured, rotation or doubly-stochastic, with structure that depends on the input (this is what state tracking needs and what keeps long rollouts stable).
* **P-3 Error control wherever rollouts are long:** snap discrete states, limit angular step size in latent loops, control update angle in training, maintain orthogonality of weights.
* **P-4 Allocate capacity geometrically:** new experts/memories go to subspaces orthogonal to old ones; memory keys are made (near-)orthogonal; time-scales are geometrically spaced.
* **P-5 Measure geometry health, not only accuracy:** operating point, effective rank, isotropy, intrinsic dimension, norm drift.

---------------------------------------------------------------------------------------------------------------------------
## 5. Syntheses worth testing (each is a hypothesis; prior-art check = one targeted search, nothing exact found)

* **N1 Snapped chunkwise orthogonal recurrence (SCOR).** Inside a chunk compute transitions in parallel (Householder/rotation products via scan); at chunk boundaries snap the state to a learned codebook. Aim: the exactness of our snapped state with the parallelism of DeltaProduct/CKDA. Prior art found: generic VQ-bottleneck RNNs and low-precision state write-back, nothing combining snapping with chunkwise parallel orthogonal recurrences [H].
* **N2 Normalised-key cone memory with geometric decay spectrum.** L2-normalised keys + delta rule + PSD/cone features + PoST decays, compared at equal state bytes with the legacy vMF memory [H].
* **N3 Angular step control for spherical deliberation.** Latent loop on the unit sphere with a geodesic step capped by a learned maximum angle plus halting; tested on loop-depth extrapolation [H].
* **N4 Quasimetric head on the sphere latent** for planning: asymmetric distance (IQE/MRN) on top of normalised features [H].
* **N5 Orthogonal-subspace expert allocation + stable routing for GrowMoE** (Track G) [H].
* **N6 Spectral-orthogonal hierarchy index** instead of hyperbolic, if Z5 confirms [H].
* **N7 Phase/magnitude-decoupled memory addressing** (PoPE idea applied to keys: content magnitude vs phase address) [H].

---------------------------------------------------------------------------------------------------------------------------
## 6. Experiment portfolio (Track Z; one round each, strict protocol)

Common protocol (non-negotiable): per-arm learning-rate grid (>= 4 values, best not on the edge), dev seeds >= 100 for tuning, test seeds 0-9, mean +- 95 % CI, paired tests, Bonferroni across arms, geometry-health report for every arm, predictions written before numbers, tables pasted from script-generated files and diff-validated, results never edited by hand, memory-lean (models <= ~10M parameters, sequences <= 2048, one GPU job at a time), local data only, strong baselines (properly tuned RoPE+RMSNorm transformer; our own earlier lessons: add the control that isolates the geometry).

| ID | Question it decides | Arms (baseline first) | Primary metrics | Prediction (direction) | Kill / decision rule |
|---|---|---|---|---|---|
| **Z0** | Can every later experiment be run and reported honestly? | protocol library + geometry-health diagnostics | unit tests; a dry run reproduces an old table | n/a | infrastructure, must pass before Z1 |
| **Z1** | Does a unit-sphere stream beat a standard stream? | `base` (RoPE+RMSNorm+AdamW), `base+qknorm`, `ngpt` (full), `ngpt-stream-only`, `ngpt+Muon`, `base+Muon` | tokens-to-target-loss, final PPL, wall-clock/step, width of the good-LR range, norm/rank drift; also MQAR as recall check | `ngpt` reaches the baseline's final loss in fewer tokens (reproduction range 1.5-4x at larger scale), wider LR range, overhead <= 20 %/step | adopt only if net wall-clock speedup >= 1.15x with paired CI > 0; else stay with RMSNorm |
| **Z2** | Which memory geometry gives the most recall per byte? | softmax (reference), linear attention, Gated-DeltaNet-style, delta + L2-normalised keys, KATA-style PSD features, triadic/tensor-product state, PoST decay vs random decay, **legacy ISHMN vMF memory (imported read-only)** | accuracy vs number of stored pairs at fixed state bytes; length extrapolation 128 -> 2048 | normalised keys + delta > plain delta; cone/triadic > delta at equal bytes; legacy memory not better than delta (verification) | decide the memory component; report legacy memory honestly |
| **Z3** | Which transition geometry tracks state exactly and in parallel? | GRU, vq_rot (ours), DeltaProduct n_h in {1,2,4}, Complex-KDA-style, Mamba-3-style rotary state, PaTH-style, MapFormer-style, **SCOR (N1)**; tasks S3, S4, A5 (all subgroups of SO(3)), S5 (not), Dyck-2 depth 4-8, parity | exactness vs length (train <= 16-24, test to 16x), seeds >= 95 %, parallelisable?, throughput | structured-orthogonal arms solve S3/S4/A5 in-distribution; unsnapped ones drift at length; SCOR holds >= 95 % to 16x; S5 stays hard for all | if SCOR does not beat plain CKDA/DeltaProduct at length, keep vq_rot as sequential only |
| **Z4** | Better position geometry than RoPE for extrapolation? | RoPE, NoPE, PoPE, LieRE-1D, FoPE, learned torus phases, (PaTH-lite for T <= 512) | PPL ratio at 4x/8x length; retrieval accuracy vs offset | PoPE degrades least, paired CI excludes zero | adopt best only if > RoPE by CI |
| **Z5** | Is hyperbolic needed at all for a real hierarchy? | hyperbolic (Poincare/Lorentz), spectral/orthogonal Euclid, tuned gradient Euclid, product; **real hierarchy built from the Python standard library** (package > module > class > function, ~10^4 nodes) | MAP, distortion, hold-out-edge link prediction, d in {4,8,16,32,64} | spectral Euclid ties hyperbolic for d >= 16; hyperbolic wins only for d <= 8 | if spectral ties down to d = 8, retire hyperbolic from the roadmap |
| **Z6** | Does angular step control fix loop extrapolation? | full step, learned scalar step, fixed quarter step, **geodesic step with learned max angle (N3)**, + halting; looped model on compositional tasks (S3 chains, modular arithmetic chains) | accuracy at 2x/4x test loop depth, failure recovery | capped-angle step recovers a large share of full-step failures (paper: 72-83 % with quarter step) | adopt if recovery is significant and costs nothing at train depth |
| **Z7** | Do asymmetric distances improve planning latents? | symmetric distance head, IQE, MRN, (TMD-lite) on directed mazes (one-way doors) and symmetric mazes | distance error, greedy planning success on larger unseen mazes | quasimetric wins on directed, ties on symmetric | adopt only for directed/asymmetric worlds |
| Z8 | (Track G) Do orthogonal-subspace allocation + stable routing fix the S1 loss? | after the D7 diagnostic: cosine routing in low-d space, routing-alignment regulariser, orthogonal adapter allocation | S1/S2/S5 accuracy, forgetting, routing accuracy | closes a large part of the oracle gap | Track-G agent, not Track Z |
| Z9 | Optional notebook | mHC (with Z1), SIGReg anti-collapse, rotation quantisation, FoNE numbers, sheaf consistency | per item | per item | only after Z1-Z7 |

Decision flow: Z0 -> Z1/Z2/Z3 (independent, highest value) -> Z4, Z5 -> Z6, Z7. Each round ends with a draft component card (what, evidence, limits) and a status in the register: VERIFIED / PARTIAL / NOT SUPPORTED.

---------------------------------------------------------------------------------------------------------------------------
## 7. What "concrete, usable, reliable, efficient" means here (the Geometry Core)

A component enters the **Geometry Core** (package `geomlab`, promoted to a stable API only after verification) when it has: tested code with a stable interface; a pre-registered experiment with a controlled baseline; >= 10 seeds with CIs; a geometry-health report; memory/time cost measured on the RTX 3060; a component card with limits. Target modules and their interfaces:

| Module | Purpose | Decided by |
|---|---|---|
| `SphericalStream` (+ scalar salience channel) | normalised residual stream, norm-preserving updates, angular update control | Z1 |
| `OrthoState` (+ optional `snap`) | data-dependent orthogonal transitions with error control | Z3 |
| `ConeMemory` | associative memory with the best recall per byte, decay spectrum | Z2 |
| `PhasePosition` | position/phase code with extrapolation | Z4 |
| `HierIndex` | hierarchy embedding (spectral or hyperbolic) | Z5 |
| `StepControl` | angular step control + halting for latent loops | Z6 |
| `QuasiMetricHead` | asymmetric distances for planning | Z7 |
| `geometry_health` | diagnostics used by all | Z0 |

Integration into the general-intelligence system happens only for VERIFIED modules, when you say "integrate". Natural homes: GrowMoE (routing, memory, growth), a future deliberation loop (StepControl, OrthoState), the semantic memory (ConeMemory, HierIndex).

---------------------------------------------------------------------------------------------------------------------------
## 8. Risks and unknowns

* **Scale gap.** Production-scale wins (nGPT 2x at 30B, mHC in V4) may not appear at 1-10M parameters; a null result at our scale is informative but not conclusive. Where a paper reports at small scale (PoST 180-440M, KATA 340M), our scale is still 30-100x smaller.
* **Paper maturity.** Many items are 2026 preprints; I read abstracts/summaries. Gemini must read the papers and report any deviation in the implementation.
* **Baseline strength.** A tuned, controlled baseline decides everything (our earlier hyperbolic experience). Each round must include the control that isolates the geometry from extra parameters or learnable temperature.
* **Implementation fidelity.** KATA, triadic attention, PaTH and Complex-KDA have non-trivial details; if the paper is ambiguous the implementer must ask, not guess.
* **Scope creep.** Track Z is about verification of geometry components, not about building a new model; no integration, no new architectures beyond the arms listed.
* **Not covered:** generative modelling on manifolds (Riemannian flow matching), topological data analysis, causal/graph geometry beyond sheaves, tokenisation geometry, spiking/neuromorphic codes.

---------------------------------------------------------------------------------------------------------------------------
## 9. References (IDs seen in this session's searches/fetches; abstract/summary level unless noted)

* nGPT: Loshchilov et al., arXiv 2410.01131; community reproduction github.com/JoeLi12345/nGPT; **Training nGPT**, Loshchilov & Ginsburg, arXiv 2608.01284 (Aug 2026).
* mHC: arXiv 2512.24880; DeepSeek-V4: arXiv 2606.19348 (Apr 2026). Manifold Muon (Thinking Machines blog, J. Bernstein); Different Layers, Different Manifolds: 2606.13276; Muon on the Stiefel manifold, closed form: 2608.06218.
* PaTH: Yang et al., arXiv 2505.16381 (NeurIPS 2025). DeltaProduct: 2502.10297. Complex KDA: 2609.24797. Gated DeltaNet-2: 2605.22791. Mamba-3: 2603.15569; non-commutative state tracking in Mamba-3: 2609.28273. Grazzi et al.: 2411.12537.
* Position: LieRE 2406.10322 (ICML 2025); PoPE 2509.10534 (ICML 2026); FoPE 2412.17739; GridPE 2406.07049; MapFormer 2511.19279.
* Memory: KATA 2607.17419; triadic linear attention 2609.36529; Kalman linear attention 2602.10743; PoST 2604.07658; Titans 2501.00663; MIRAS 2504.13173; ATLAS 2505.23735; Provably optimal memory capacity for modern Hopfield models (NeurIPS 2024).
* Hierarchy: Park et al., The Geometry of Categorical and Hierarchical Concepts in LLMs (ICLR 2025); Nava & Wyart 2605.23821; audit of hyperbolic VLMs 2607.05268; our earlier references in `research/YGGDRASIL_ARCHITECTURE_RESEARCH.md`.
* Routing / continual learning: X-MoE (Chi et al. 2022); Equifinality in MoE 2604.14419; StaR-MoE 2605.17571; MILE 2605.03555; O-LoRA 2310.14152; OPLoRA 2510.13003; NESS 2602.21919; SPACE-LoRA 2609.34453.
* Attention normalisation: Sinkhorn doubly-stochastic attention rank analysis 2604.07925; ESPFormer (ICML 2025); LOTFormer.
* Isotropy / world models: LeJEPA 2511.08544; "When does LeJEPA learn a world model?" 2605.26379. Planning: TMD 2509.20478; multistep quasimetric learning 2511.07730; QuaD (OpenReview).
* Latent loops: Right Direction, Wrong Step 2609.16665; Solve the Loop (attractor models) 2605.12466; Equilibrium Reasoners 2605.21488.
* Numbers: FoNE 2502.09741; convergent number representations 2604.20817; prime Fourier embeddings 2606.23044.
* Sheaves: Hilbert bundles and cellular sheaves 2605.06395; deep neural sheaf diffusion 2605.19021; Sheaf Attention Networks (NeurIPS 2022).
* Uncertainty / grounding: CIDER 2203.04450; MARVEL 2607.02435; modality-gap geometry 2602.09507.
* Quantisation: QuaRot 2404.00456; SpinQuant (ICLR 2025); "Transforms for LLM quantisation" 2608.25188.
* Equivariant geometry: GATr/L-GATr 2405.14806, 2411.00446. Hyper-SET (transformers as hyperspherical energy minimisation) 2502.11646.

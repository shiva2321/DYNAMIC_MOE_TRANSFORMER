# From VSA geometry to an intelligence algorithm

*What the evidence supports, where neural networks fit, and a testable route to generalisation that emerges from data.*

Research report · 2026-10-03 · companion files in this folder:
`ALGORITHM_AND_PROTOCOL.md` (the algorithm, experiments, and a code audit of ISHMN) · `BIBLIOGRAPHY_CHRONOLOGICAL.md` (≈ 340 references, oldest → latest) · `vsa_numerics.py` and `geometry_checks.py` (every number marked **[M]** below) · `results_*.log`.

---

## 0. How to read this

Every claim carries one of four labels. They matter more than the prose.

| Label | Meaning |
|---|---|
| **[M]** | **Measured here**, reproducible with `python research/vsa_numerics.py` / `python research/geometry_checks.py` (seeded, CPU, ≈ 30 s). |
| **[V]** | A published result whose existence, venue and headline claim I checked this session at abstract/venue-page level. I did **not** read full texts, so "the authors report X" means exactly that. |
| **[R]** | A classic cited from memory. Probably right, not re-verified. |
| **[H]** | My inference or hypothesis. Not established by anything above. |

**Coverage — read before trusting "miss nothing".** "Every paper from oldest to latest" cannot be done: HDC/VSA alone is thousands of papers and the neighbouring fields are orders of magnitude larger. What I did: a structured sweep of (1) VSA/HDC proper — surveys, binding schemes, resonators, fractional power encoding, hardware, plus a date-sorted arXiv search through 3 Oct 2026 so the newest work is not missed; (2) bridges between neural networks and associative memory / binding; (3) superposition, geometry and scaling; (4) neuroscience; (5) cognitive science; (6) AGI frameworks and benchmarks; (7) your own repository. The bibliography has **317 list entries covering ≈ 340 references** (some entries bundle several papers): ≈ 200 verified at abstract/venue level, ≈ 30 at title level, ≈ 110 classics from memory, 2 flagged withdrawn. I did not search patents, non-English literature, or most of the applied HDC literature (bio-signals, federated learning, edge hardware — only sampled).

---

## 1. The answer in twelve points

1. **VSA geometry is three structures on a high-dimensional sphere or torus**: a *metric* (cosine similarity), a *group of isometries* (binding), and an *attractor set* (the codebook, with "cleanup" = snap to the nearest atom). Superposition, sequences, role–filler records, analogy and memory are all built from these three.
2. **The core geometric facts hold, and I measured them [M].** Random directions are quasi-orthogonal only when the dimension is large (5,295 random unit vectors: minimum pairwise angle 5.2° at D = 8, 80.1° at D = 1,024). Superposition capacity scales like D / (7.5·ln N). Unitary binding preserves angles and norms exactly. Fractional power encoding is exactly additive. Cleanup tolerance is set by dimension.
3. **What it buys:** an exponentially large address space; linear superposition with *predictable* crosstalk; exact, invertible composition; operators that do not depend on the content they act on (so Fodor–Pylyshyn systematicity becomes an algebraic identity); and a cleanup step that is literally attention [V].
4. **What it does not buy:** knowledge (atoms and similarity structure must still be learned), unbounded capacity or depth, non-invertible dynamics, an objective, or search. Published evidence of these limits exists: a VSA solver scores 3.0 % on ARC-AGI-1 evaluation [V]; holographic memory composes facts at chance when superposition capacity is the bottleneck [V]; a "theory of AGI from HDC" has no experiments [V].
5. **The algorithm the geometry points to** (derived in §4): learn an *equivariant encoder* into a flat phase space, a library of *unitary operators* (inverse exact by construction), an *addressed* (not merely superposed) memory, *resonator-style inference*, and a *search loop over operator programs* with library learning — trained with prediction + geometry-health + compression losses, **evaluated on real latents with algebraic metrics**.
6. **Neural networks are the learner; VSA is the shape of the solution.** Eight distinct NN roles are identified in §5 (perception, codebook shaping, operator learning, amortised inference, control/search policy, memory control, residual modelling, large-scale credit assignment).
7. **VSA is already inside neural networks** [V] + [H]: attention is a Hopfield/SDM cleanup; linear attention is tensor-product binding; RoPE is phase rotation of position; gated linear units are element-wise binding; RMSNorm is projection to a sphere; residual streams are bundles; complex-diagonal recurrences are phasor sequence codes.
8. **Structure does emerge from training** [V]: Fourier/irrep solutions for group tasks (grokking), grid-like codes for path integration, binding-ID vectors and symbolic induction heads in LLMs, filler–role structure that closed-form symbolic descriptions approximate (McCoy et al., 30 Aug 2026).
9. **"Enough data" is not a theory** [V]: some architectures never generalise out of distribution at any data scale; RL with verifiable rewards narrows rather than widens the base model's reach; frontier LLMs scored under 0.4 % on ARC-AGI-3 at launch (press reports); test-time refinement loops and *structure* decide results.
10. **Concretely for ISHMN** [M]: a D = 8 sphere cannot host VSA properties (among 5,295 random concepts the closest pair is 5° apart and the median nearest neighbour is ≈ 26° away already at 1,500 concepts), the "phasor" bus is real-valued and shrinks messages by ≈ 1/√D and is never executed in a forward pass, concept atoms are seeded with Python's per-process-salted `hash()` (not reproducible), and `geodesic_distance_sphere` cannot resolve angles below 4.5 × 10⁻⁴ rad.
11. **Parallel transport on a sphere is the wrong composition geometry** [M]: its holonomy equals the enclosed area, and its analogy error grows with the distance between A and C; group binding on a flat torus is exact for any (A, C) when the relation is a symmetry.
12. **My calibrated judgement [H]:** VSA-style binding + superposition memory + unitary state updates are likely *necessary ingredients* for systematic generalisation, not a *sufficient theory* of intelligence. A pure VSA system matching frontier models soon is unlikely. A small hybrid with the right inductive bias beating larger unstructured models on structured reasoning is plausible, and the evidence for it is what the roadmap tests first.

---

## 2. The geometry in one page

### 2.1 What is fixed

A VSA fixes (i) a space **V** = ℝᴰ or ℂᴰ with D ≈ 10³–10⁴; (ii) a **codebook** C = {c₁ … c_N} of random "atoms" plus a **cleanup** map (nearest atom); (iii) three operations — **bundle** ⊕ (sum), **bind** ⊗ (a group action), **permute** ρ (a special binding). Flavours differ in the group:

| Flavour | Vectors | Bind | Natural manifold |
|---|---|---|---|
| MAP / BSC | ±1 / {0,1} | element-wise product / XOR (self-inverse) | hypercube ⊂ S^{D−1}(√D) |
| **FHRR** | unit phasors e^{iθ} | element-wise complex product (inverse = conjugate) | **flat Clifford torus T^D** ⊂ S^{2D−1}(√D) |
| HRR | Gaussian N(0,1/D) | circular convolution (inverse ≈ involution) | sphere; binding ≈ circulant orthogonal |
| VTB / MBAT | real | matrix–vector product | orthogonal-group orbits |
| GHRR | block-unitary | **non-commutative** block product | product of unitary groups U(b)^{D/b} |
| TPR | tensors | outer product (full group algebra) | Segre varieties; special case of Clifford binding [V] |

**Key point:** the right manifold is the *homogeneous space of the binding group*. For abelian binding that is a **flat torus**, not the round sphere.

### 2.2 Seven facts, with measurements

| # | Fact | Formula | Measured here **[M]** |
|---|---|---|---|
| **F1** | Concentration ⇒ quasi-orthogonality | P(abs cos > ε) ≤ 2e^{−Dε²/2}; mean abs cos ≈ √(2/πD); max over P pairs ≈ √(2 ln P / D) | N = 5,295 random unit vectors. D = 8: mean abs cos 0.291, **min pair angle 5.2°**. D = 32: 37.5°. D = 128: 63.8°. D = 1,024: 80.1°. D = 10,000: 86.9°. Mean abs cos matches theory to 3 digits. |
| **F2** | Superposition capacity | threshold decoding + union bound: D ≥ 8·M·ln(N/δ); Hopfield analogue N/(2 ln N) [V] | Exact recovery of all M members, P ≥ 0.95, N = 5,295: D = 256 → M = 4; 1,024 → 16; 4,096 → 64; 10,000 → ≥ 128 (grid-limited). **D/M ≈ 64 ≈ 7.5·ln N.** |
| **F3** | Unitary binding is an isometry | ⟨ρx,ρy⟩ = ⟨x,y⟩; ρ⁻¹ = ρ*; ρ(x⊕y) = ρx ⊕ ρy | D = 1,024. MAP, FHRR: cosine deviation 0, norm ratio 1.000. HRR: 0.015 / 0.994 ± 0.030. Real Hadamard with unit-L2 keys (ISHMN bus scheme): 0.024 / **0.031**. |
| **F4** | Cleanup = attractor = attention | x ← C·softmax(β Cᵀx); capacity exponential in D [V] | Retrieve 1 of 5,295 patterns after rotating the query by θ. D = 8: 0.65 at 20°, **0.00 at 40°**. D = 32: robust to 40°. D ≥ 128: robust to 60°. |
| **F5** | Continuous values: fractional power encoding | φ(x)ⱼ = e^{iωⱼx}; φ(x)φ(y) = φ(x+y); ⟨φ(x),φ(y)⟩/D → k(x−y) (Bochner) [V] | Homomorphism error 1.2 × 10⁻¹⁵; kernel error ∝ D^−0.51 (random-features theory: −0.5). |
| **F6** | Factorisation = resonator network | iterate unbind → cleanup; "search in superposition"; operational capacity ∝ D² [V] | My plain bipolar resonator, 3 factors: n = 10: 0.85 (D = 256) → 1.00 (D = 4,096); n = 30: 0.60 (D = 1,024) → 0.92 (D = 4,096); **n = 100 (10⁶ combinations): 0.32 at D = 4,096.** Improved dynamics extend capacity ≥ 50× [V]. |
| **F7** | Group-theoretic reading | abelian: characters (FHRR/FPE); non-abelian: irreps (GHRR, MBAT, TPR) | Networks trained on group tasks recover irreps [V] (Chughtai et al. 2023). |

Two derived facts that drive the algorithm:

- **Capacity needs address space, not just superposition.** With N = 10⁵ atoms (ln N ≈ 11.5), exact recall of M = 1,000 items from one bundle needs D ≈ 7.5 × 1,000 × 11.5 ≈ **86,000 dimensions** [H, extrapolating the measured constant]. A single hypervector cannot be a knowledge base; it is a *working memory*. Large stores need addressed memory (SDM / modern Hopfield / hashed lookup).
- **Flat beats curved for composition.** On the unit sphere (curvature +1) parallel transport has holonomy equal to the enclosed area (measured to 10⁻¹³ below), so composing relations is path-dependent. An abelian group action has zero holonomy: g_pq · g_qr · g_rp = 1.

---

## 3. A history of the idea, oldest → latest

Seven threads run through 80 years of work. The bibliography lists every item with its verification status; this section is the argument.

**Thread A — distributed associative memory.** Hebb (1949) → holographic and correlation models (Longuet-Higgins 1968; Willshaw, Buneman & Longuet-Higgins 1969) → Marr/Albus cerebellum (1969–71) → Kohonen and Anderson linear associators (1972) → Hopfield (1982) and the statistical physics of capacity (Amit–Gutfreund–Sompolinsky 1985; Gardner 1988; McEliece et al. 1987 — capacity ≈ N/2ln N) → **Kanerva's Sparse Distributed Memory (1988)**, which already treats memory as geometry on a Hamming hypercube → dense associative memory with exponential capacity (Krotov & Hopfield 2016; Demircigil et al. 2017) → attention *is* the modern-Hopfield update (Ramsauer 2021) and *approximates* SDM (Bricken & Pehlevan 2021). Cleanup memory and attention are the same object.

**Thread B — binding and compositionality.** Fodor & Pylyshyn (1988) argue connectionism cannot be systematic. Responses follow almost at once: Touretzky & Hinton (1988, concurrently), **Smolensky's tensor-product representations (1990)**, Pollack's RAAM (1990). Then the move that defines VSA: **Plate's holographic reduced representations (1991/1995)** keep the vector width fixed by compressing outer products with circular convolution; Kanerva's spatter codes (1996) and Rachkovskij–Kussul's context-dependent thinning (2001) do the same for binary and sparse codes; **Gayler (2003)** names the family "Vector Symbolic Architectures". Psychology independently built the same algebra (TODAM/CHARM 1982; BEAGLE 2007). Neural Engineering and **Spaun (Eliasmith 2012)** show a 2.5 M-neuron spiking model doing eight tasks on a VSA substrate. Resonator networks (Frady, Kent, Olshausen & Sommer 2020) supply the missing inverse problem — factorisation. Generalisations: non-commutative GHRR (2024), Clifford binding (Aerts 2009; Pence 2026), category-theoretic foundations (Shaw 2025).

**Thread C — random projection and high-dimensional geometry.** Rosenblatt's random association layer (1958/62) → Cover's counting theorem (1965) → Johnson–Lindenstrauss (1984) → random-feature networks (Igelnik–Pao 1995; reservoir computing 2001–02; extreme learning machines 2006) → **random Fourier features (Rahimi & Recht 2007)** → HDC (Kanerva 2009), whose encoders are random features of a kernel (Thomas–Dasgupta–Rosing 2021; Frady et al. 2021). Mathematical backbone: concentration of measure; Gorban–Tyukin stochastic separation (2017–18).

**Thread D — coding theory and physics.** Shannon (1949) already pictures signals as points in a high-dimensional space with noise spheres; Hamming (1950) gives the metric. Cleanup *is* maximum-likelihood decoding, bundling *is* a superposition channel, and the capacity formulas of §2 are channel-coding bounds. Hopfield capacity (McEliece 1987) has the same D/ln N form as bundle capacity [M above]. The newest statistical-physics result links generalisation to spurious attractors (Pham et al. 2025).

**Thread E — brains.** Mixed selectivity and random-like high-dimensional mixing in prefrontal cortex (Rigotti 2013); expansion–sparsity circuits in cerebellum, fly mushroom body and dentate gyrus (Marr 1969; Dasgupta 2017; Babadi & Sompolinsky 2014; Litwin-Kumar 2017); complementary learning systems (McClelland 1995); grid cells as a residue-number code with torus topology (Gardner 2022; Kymn 2024); the hippocampal formation as a transformer-like relational memory (Whittington 2020, 2022). Neuroscience supplies *existence proofs* that these geometries are biologically used, not that VSA is the mechanism.

**Thread F — deep learning meets the algebra.** Associative LSTM with holographic copies (2016); HolE (circular correlation, 2016) and RotatE/ComplEx (complex rotations) rediscover VSA binding for knowledge graphs; fast weights (Schmidhuber 1992; Ba 2016) → **linear attention = fast-weight programming = tensor-product memory** (Schlag 2021) → DeltaNet's Widrow–Hoff-style write (2024) → test-time memory (Titans 2025; Nested Learning 2025). Unitary/orthogonal RNNs (2016) and complex-diagonal recurrences (LRU 2023; S4/LMU lineage) are phasor dynamics. Superposition in LLMs (Elhage 2022) is quasi-orthogonality re-discovered; strong superposition explains the 1/dimension loss law (Liu–Liu–Gore 2025). 2023–26 brings *direct* VSA transplants: Hrrformer (2023), GHRR-attention (2024), NVSA for Raven's matrices (2023), the Hyperdimensional Probe for LLM interpretability (2025), FHRR world models (2026).

**Thread G — what "general intelligence" is supposed to be.** Solomonoff induction and Kolmogorov complexity (1964–66) → MDL (1978) → AIXI and universal intelligence (Hutter 2005; Legg & Hutter 2007) → **Chollet (2019): intelligence = skill-acquisition efficiency under priors** → JEPA/world models (LeCun 2022; Hafner 2025) → verifiable-reward reasoning (DeepSeek-R1 2025) and its limits (Yue 2025) → **ARC-AGI-1/2/3 (2019–2026)**, where refinement loops, test-time training and ≈ 7 M-parameter models matter and frontier LLMs initially score under 0.4 % on the interactive version.

The nine eras in the bibliography are: 0 (1943–69) foundations; 1 (1970–89) associative memory and statistical physics; 2 (1990–99) binding solutions; 3 (2000–09) VSA named; 4 (2010–15) Spaun and word vectors; 5 (2016–19) memory-augmented nets and grid-like codes; 6 (2020–22) attention = memory, resonators, superposition; 7 (2023–24) networks rediscover algebra; 8 (2025–Oct 2026) VSA inside LLMs and symbolic structure found in trained networks.

---

## 4. From geometry to an algorithm

### 4.1 The setting

An agent sees observations oₜ and takes actions aₜ in a world whose dynamics are partly *symmetric*: a rotation, a translation, a relabelling, a composition of earlier operations. "Intelligence" here is operational: **skill-acquisition efficiency across a distribution of tasks** (Chollet 2019 [V]), assessed with held-out compositions, fixed compute and an external verifier where one exists.

### 4.2 Proposition 1 — equivariance turns prediction, planning and composition into algebra [H, elementary]

Suppose an encoder φ and a unitary representation ρ of the action group satisfy φ(T_g o) = ρ(g) φ(o). Then:

1. **Prediction** is a single isometry: ẑₜ₊₁ = ρ(aₜ) zₜ. For abelian G acting on phasors this is an element-wise product, O(D).
2. **Inverse questions** ("what action explains this change?") are unbinding: ρ(a)⁻¹ = ρ(a)* — exact, no learned inverse.
3. **Composition** is multiplication in the group: ρ(a∘b) = ρ(a)ρ(b); associativity is free.
4. **Systematicity** is an identity: ρ(g) never reads the *content* it acts on, so what is learned on some fillers applies to novel fillers.

Everything that makes this hold is a *property of the learned map*, not of the sphere. That is the learning problem.

### 4.3 Proposition 2 — one example identifies a relation only if the hypothesis class is small [M]

I tested analogy A : B :: C : D with your own `geometry.py` (`geometry_checks.py`). Dimension 8 = ℂ⁴, relation = element of the torus T⁴ (a genuine symmetry), 1,000 random draws:

| distance(A, C) (rad) | Parallel-transport error (rad) | Group-binding error (b/a · c) |
|---|---|---|
| 0.05 | 0.050 | 3.6 × 10⁻¹⁶ (chordal) |
| 0.5 | 0.488 | same |
| 1.0 | 0.969 | same |
| 1.5 | 1.359 | same |
| 2.5 | 1.885 | same |

Parallel transport error grows like the distance; group binding is exact for every (A, C). When the relation is instead a rotation in a random 2-plane of ℝ⁸ (an SO(8) element), a single (A, B) pair does not determine it, and transport beats "do nothing" (error 0.086 vs 0.444) only for nearby C and is **worse** than doing nothing for far C (0.695 vs 0.444). **Lesson [H]:** one-shot analogy is possible exactly when the hypothesis class is small and correct; a neural network's job is to *find that class* (which group, which representation), not to hard-code one.

Holonomy confirms why: transporting a vector around a closed geodesic triangle rotates it by the enclosed area — mean 0.001154 at ε = 0.05, 0.331963 at ε = 0.8, matching the triangle area to 10⁻¹³ — so a chain of transported relations depends on the path taken; an abelian binding group has none.

### 4.4 The geometry tells you what to measure

Each quantity is cheap and each must be computed on the **real latent distribution** (your measured 1.374 rad vs 0.392 rad lesson), never on Gaussian noise, and compared to its random-matrix baseline:

| Health metric | Definition | Baseline / target |
|---|---|---|
| Coherence | max abs cos between codebook atoms | random: √(2 ln P / D) ≈ 0.18 at D = 1,024, N = 5,295 [M] |
| Uniformity | log E exp(−t‖zᵢ − zⱼ‖²) (Wang & Isola) | no collapse into a cone |
| Equivariance error | E‖φ(T_g o) − ρ(g)φ(o)‖ | → 0 |
| Binding recoverability | P(correct filler) vs number of bound pairs K | D/(M ln N) ≤ 7.5 [M] |
| Isometry error | cosine and norm deviation of binding | 0 for unitary operators [M] |
| Composition defect | ‖ρ(a)ρ(b) − ρ(ab)‖ and holonomy | 0 on a flat torus [M] |
| Abstraction geometry | cross-condition generalisation, parallelism (Bernardi 2020 [V]) | high |

### 4.5 Nine design principles derived from §2–4

| # | Principle | Support |
|---|---|---|
| **P1** | **Dimension is capacity.** Choose D ≥ 7.5·M·ln N for exact recall of M bundled items; a D = 8 sphere cannot host a 5,295-atom codebook. | F1, F2 [M] |
| **P2** | **Change state by isometries.** Parameterise dynamics as phase vectors → unitary by construction, inverse exact; no separate W_bwd. | F3 [M]; unitary RNNs [V]; FHRR world model [V] |
| **P3** | **Compose on a flat, mostly-abelian manifold;** add non-commutative blocks only where order matters. | Holonomy [M]; GHRR [V] |
| **P4** | **Make operators content-independent.** Train with filler randomisation; test on novel fillers. | Prop. 1; Webb 2024 [V]; Ito 2022 [V] |
| **P5** | **Memory by address, not only by superposition.** Sparse expansion + attractor retrieval; hash for O(1). | F2 [M]; SDM [V]; Engram [V] |
| **P6** | **Clean up after every composition.** Project to the nearest atom / manifold; resonator for factorisation; manifold-aware cleanup for continuous codes. | F4, F6 [M]; Habashy & Eliasmith 2026 [V] |
| **P7** | **Measure geometry on real data against a random baseline.** | §4.4; your own measurement history |
| **P8** | **Give every learning signal a truth channel.** Verifiers, execution, prediction against the environment — not closed self-play. | Shumailov 2024 [V]; DeepSeek-R1 [V]; AlphaGeometry [V] |
| **P9** | **Adapt at test time.** Refinement loops and test-time training decide ARC results. | ARC Prize 2025 [V] |

### 4.6 What geometry does *not* derive

The geometry fixes the *hypothesis class and the memory*. It does not supply the objective, the search procedure or the curriculum. Three external principles do: **(i) predict the environment and keep the description short** (MDL / compression, Rissanen 1978; Delétang 2023 [V]); **(ii) verify** (execute, check, compare to ground truth); **(iii) adapt cheaply** (Chollet's efficiency). The algorithm in `ALGORITHM_AND_PROTOCOL.md` combines the geometry with these.

---

## 5. The role of neural networks

**One-sentence answer:** neural networks are the *learning substrate* — gradient-based credit assignment over continuous parameters from large data — while VSA geometry is the *structure the learned solution should take*. They co-evolve: the network learns encoders, operators, codebooks and controllers; the algebra supplies the hypothesis class, the invariants and the diagnostics.

### 5.1 Eight roles

| # | Role | What the network does | Evidence |
|---|---|---|---|
| 1 | **Perception / encoding** | Map raw input to atoms or bound structures | NVSA for Raven's matrices [V]; Hyperdimensional Probe [V]; Clifford-VAE [V]; the binding problem in vision (Greff 2020 [V]) |
| 2 | **Codebook shaping** | Make similarity encode task structure; keep atoms near-orthogonal | Alignment/uniformity [V]; neural collapse yields an equiangular tight frame, i.e. a maximally separated codebook [V] |
| 3 | **Operator learning** | Learn unitary/phasor transition operators | Fourier solutions in grokking [V]; grid-cell path integration [V]; FHRR world model (87.5 % zero-shot unseen state–action pairs) [V]; Semigroup-JEPA [V] |
| 4 | **Amortised inference** | Initialise or replace resonator iterations | Attention-based semantic decomposition [V-t]; noise-extended resonators [V] |
| 5 | **Control / search policy** | Propose operator programs; guide search | DreamCoder [V]; AlphaGeometry (LM guides a symbolic engine) [V]; ARC refinement loops [V] |
| 6 | **Memory control** | Decide what to write, forget, consolidate | Titans, Nested Learning [V]; complementary learning systems [V]; relevance-gated VSA memory (SRMU) [V] |
| 7 | **Residual modelling** | Capture the non-algebraic part of the world | Latent prediction with learned features (JEPA, Semigroup-JEPA: "most of the gain comes from the encoder") [V] |
| 8 | **Large-scale credit assignment** | Absorb large, diverse data into the above | Scaling laws [V]; strong superposition explains 1/D loss [V] |

### 5.2 VSA is already inside neural networks

| VSA primitive | Where it already lives | Status |
|---|---|---|
| **Cleanup** | Softmax attention = modern-Hopfield update; SDM approximation; SAE decoder as dictionary | [V] Ramsauer 2021; Bricken 2021 |
| **Bundling** | Residual-stream addition; attention-weighted sums; feature superposition | [V] Elhage 2021/22; theory in Dhayalkar 2025 |
| **Binding** | Outer products in linear attention / fast weights (= TPR); long convolutions (Hyena = HRR-style); element-wise gating in GLU/SwiGLU/FiLM | [V] Schlag 2021, Smolensky 1990; **[H]** the GLU ↔ MAP identification |
| **Permutation / position** | RoPE rotates each complex pair by e^{imθ}; sinusoidal PE; complex-diagonal recurrences (LRU, S4D) | [V] Su 2021, Orvieto 2023; **[H]** that this is fractional power encoding (the algebra is exact) |
| **Item memory** | Embedding/unembedding; FFN layers as key–value memories; ROME edits | [V] Geva 2021; Meng 2022 |
| **Resonator / factorisation** | Looped or recurrent depth: HRM/TRM; recursive reasoning | [V] Wang 2025; Jolicoeur-Martineau 2025 |
| **Random-feature kernels** | Performer attention; Fourier features in grokked networks | [V] Choromanski 2021; Nanda 2023 |
| **Sphere projection** | LayerNorm/RMSNorm; nGPT; manifold Muon | [V] Loshchilov 2024; Bernstein 2025 |

So VSA is not *an alternative to* networks. It is the algebra that trained networks partly rediscover; the design question is how to make that algebra cheap to find, exact where it should be, and measurable.

### 5.3 What each side supplies

- **VSA → network:** content-independent compositional operators; explicit superposition with predictable crosstalk; one-shot, local, no-forgetting memory writes; interpretable probes (filler–role recovery); hardware efficiency (in-memory computing demonstrations on phase-change devices, 760,000 devices [V]).
- **Network → VSA:** learned encoders; learned similarity structure and codebooks; end-to-end differentiability; generalisation over perceptual variation; scale.

---

## 6. Does the structure emerge from data?

### 6.1 Evidence that it does (strength in brackets)

- **Group tasks → Fourier features / irreps** (Nanda 2023; Chughtai 2023; Zhong 2023; 2026 follow-ups) [strong, small models]. Different seeds find *different* algorithms (Clock vs Pizza), so "emergence" is of a *family*, not a unique solution.
- **Path integration → grid-like codes** (Banino 2018; Cueva & Wei 2018 [R]; Sorscher 2019; Gao 2019) [strong; mechanism understood].
- **LLMs bind entities with binding-ID vectors** (Feng & Steinhardt 2024) and **learn variable binding in three training phases using the residual stream as addressable memory** (Wu et al. 2025) [moderate].
- **Symbolic induction machinery in LLMs**: symbol-abstraction → symbolic-induction → retrieval heads (Yang et al. 2025) [moderate].
- **Closed-form symbolic structure approximates representations** across small networks and LLMs, with causal interventions (McCoy et al., 30 Aug 2026) [new; abstract-level; the authors note scope limits].
- **Weights as associative memories** of near-orthogonal embeddings explain induction-head formation and factual recall (Bietti 2023; Nichani 2025; Ravfogel 2026) [moderate; theory + synthetic].
- **Abstract representations emerge from multi-task training** (Johnston & Fusi 2023); **primitive pre-training induces compositional generalisation** (Ito 2022) [moderate].
- **Spurious attractors as the first sign of generalisation** in associative memory / diffusion (Pham 2025) [suggestive].

### 6.2 Evidence that "enough data" is not enough

- **Some architectures never generalise OOD**, however much data (Delétang 2023; 20,910 models) [strong].
- **Compositionality via linearised subgraph matching** rather than systematic computation (Dziri 2023) [strong].
- **Holographic memory fails zero-shot composition** at chance even when single-hop retrieval works; the bottleneck is superposition capacity (Kumar 2026) [one study; clear mechanism].
- **VSA on ARC**: 3.0 % on ARC-AGI-1 evaluation (Joffe & Eliasmith 2025) — cognitively plausible, not competitive [V].
- **RL with verifiable rewards** does not extend the base model's pass@k boundary (Yue 2025) [strong].
- **ARC** (Chollet et al. 2026): frontier reasoning remains knowledge-bound; all paradigms drop 2–3× from ARC-AGI-1 to ARC-AGI-2 (living survey 2026); ARC-AGI-3 launch-day frontier LLMs below 0.4 % (press; confirm at arcprize.org).
- **Verification of headline claims matters**: HRM's reported 40.3 % ARC-AGI-1 was independently reported at ≈ 32 % pass@2, and ≈ 2 % on ARC-AGI-2 (secondary report).
- **Platonic convergence is contested**: global spectral similarity largely vanishes after calibration; only local neighbourhoods converge (Gröger 2026).

### 6.3 Conditions under which structure appears [H]

Synthesising 6.1–6.2: structure appears when (i) **compression pressure** exists (weight decay, norm constraints, small width) so the low-complexity algebraic solution beats memorisation; (ii) the **data are diverse enough** that the shared group structure is the cheapest explanation (multi-task, filler-randomised); (iii) the **architecture makes the algebraic solution reachable** (phasor/unitary/spherical, relational bottleneck, outer-product memory); (iv) the **objective has a truth channel**; (v) **adaptation at test time** is allowed. When these fail, networks fall back to shortcut statistics. This is a hypothesis, and §5–§6 of the companion protocol turn it into falsifiable predictions.

---

## 7. Limits and failure modes

1. **Capacity.** Superposition limits working sets to ≈ D/(7.5 ln N) items [M]; large stores need addressing. Kumar 2026 shows the failure mode in practice.
2. **Depth.** Every binding level adds crosstalk; cleanup per level is mandatory. Order-p tensor memories with orthogonal subspace carving give constant memory for deep recursion (Pence 2026 [V]).
3. **Factorisation.** Resonators work well below D² and degrade beyond; my plain implementation solves 10⁶-combination problems only 32 % of the time at D = 4,096 [M]; improved noise injection reports ≥ 50× more [V].
4. **Non-invertibility.** Unitary dynamics model reversible symmetries. Real worlds lose information (aggregation, hashing, irreversibility). The correct object is a *semigroup* of contractive operators with set-valued inverses (Semigroup-JEPA [V] points the same way). Your 2026-09-21 finding that W_bwd was not an inverse of W_fwd (later recorded as fixed) is the same phenomenon seen from the other side: a learned inverse of a lossy map is not unique, so exact invertibility has to be either imposed (unitary parts) or abandoned (contractive parts) deliberately.
5. **Atoms are not knowledge.** Random atoms give composition, not semantics; similarity structure must be learned (codebook shaping).
6. **Learnability is separate from computability.** Smolensky et al. 2025 compile symbolic programs into transformers and say explicitly that this addresses computability, not learnability [V].
7. **Evidence quality.** Several recent VSA/AGI papers are theory-only (Dhayalkar 2025/26; Overmann 2026), one is withdrawn (Arndt 2025), and many 2026 results are single-dataset abstracts. Treat accordingly.
8. **Measurement traps.** Probing on Gaussian noise (1.374 vs 0.392 rad), string-graded benchmarks, and length-sensitive metrics have each misled this project; the same traps apply to VSA metrics (§4.4).

---

## 8. Cross-domain synthesis

| Domain | Structural parallel | What it predicts | Where the analogy breaks |
|---|---|---|---|
| **Information theory** (Shannon 1949) | Atoms = random codewords; noise spheres; cleanup = ML decoding; bundling = superposition channel | Capacity ∝ D/ln N; measured constant ≈ 7.5 [M] | Memoryless-Gaussian assumptions; correlated or learned atoms raise crosstalk |
| **Statistical physics** (Hopfield, AGS, Gardner, Krotov) | Cleanup = energy descent; retrieval vs spin-glass phases; dense AM capacity exponential in D | Spurious states can be generalisation (Pham 2025) | Finite-size, non-equilibrium learning dynamics |
| **Harmonic analysis / representation theory** | FHRR = characters; FPE = Bochner/RFF; non-abelian = irreps | Spectral sparsity in grokked nets (Nanda 2023) | Unknown, only approximate symmetries in real data |
| **Differential geometry** | Sphere has holonomy; torus is flat | PT analogies degrade with distance; group analogies do not [M] | Both assume the right group; finding it is the hard part |
| **Optics / holography** | Gabor → convolution/correlation memory; 2026 wave-geometric duality | Physical implementations of bundle/bind | Metaphor only unless coherence is maintained |
| **Cerebellum, fly, dentate gyrus** | Sparse random expansion + WTA = Cover/JL + LSH | Expansion separates patterns | Biology is structured, not i.i.d. random |
| **Grid cells / hippocampus** | Residue-number code + binding + attractors; toroidal topology | Trained RNNs develop grid-like codes | Few modules, not 10⁴ random dimensions; **structured codes beat disordered ones** (hippocampal "crystalline geometry", 2026 [V]) |
| **Cognitive science** (Gentner, LISA, Copycat) | Analogy = structure mapping = binding + mapping vector | One-shot analogy under small relation classes [M] | Human relational capacity ≈ 4 bindings (Halford [R]) vs VSA capacity; synchrony vs algebra |
| **Linguistics** (TPR, Optimality Theory) | Role–filler composition | Filler–role structure recoverable in LMs (McCoy 2026) | Recursion depth; learnability |
| **Hardware / materials** | In-memory computing; noise as a resource | Noise extends resonator capacity ≥ 50× [V] | Device variability; precision |
| **ML theory** | HDC ≈ random-feature kernels; neural superposition; scaling laws | Loss ∝ 1/D in strong superposition (Liu 2025) | Feature-learning regime differs from lazy random features |
| **Algorithmic information theory** | MDL; library learning = compression; AIXI | Skill efficiency is a compression statement | Incomputability; priors decide |

**Two analogies that fail, stated plainly.** *VSA is not symbolic AI:* binding is lossy and "symbols" exist only after cleanup, so nothing is discrete until you snap it. *VSA is not "just embeddings":* embeddings give similarity; VSA gives an algebra whose operations preserve it — but that algebra is only as good as the encoder that feeds it.

---

## 9. Scorecard

| Claim | Status |
|---|---|
| High-dimensional random atoms are quasi-orthogonal; capacity ∝ D/ln N | **Established** [M] [R] |
| Unitary binding preserves geometry exactly; FPE is additive and equals a random-feature kernel | **Established** [M] [V] |
| Softmax attention is a cleanup/Hopfield/SDM operation | **Established** [V] |
| Linear attention and fast-weight programmers are tensor-product (outer-product) memories | **Established** [V] |
| Trained networks recover Fourier/irrep solutions on group tasks and grid-like codes on path integration | **Strongly supported** [V] (small models) |
| LLMs contain binding-ID vectors and symbolic induction circuits | **Supported** [V] (specific models, abstracts) |
| Filler–role structure approximates a wide range of network representations | **New, plausible** [V] (one preprint, Aug 2026) |
| Phasor world models with unitary transitions generalise zero-shot to unseen state–action pairs | **Promising, narrow** [V] (grid worlds) |
| VSA-based reasoners are competitive on ARC | **Not supported** (3.0 % eval [V]) |
| Superposition-capacity limits cause compositional failures in holographic memory | **Supported** [V] (one clear study) |
| Scaling data alone yields systematic generalisation | **Contradicted for some architectures** [V] |
| A VSA-centred system will give general intelligence | **Unsupported** — no evidence either way; theory-only papers exist |

---

## 10. Where ISHMN stands (summary; evidence and fixes in `ALGORITHM_AND_PROTOCOL.md` §7)

- **D = 8 cannot host VSA properties** [M]: the 5,295-concept lattice has no quasi-orthogonality (min angle 5.2° for random placement), cleanup fails at 40° noise. This is consistent with the lattice collapse you measured earlier; whether D = 8 *caused* it is **untested [H]**.
- **The "phasor" bus is a real Hadamard with unit-L2 keys** [M]: unbinding multiplies by key² not 1; each bind shrinks the vector ≈ 1/√D (0.031 at D = 1,024); at d_hyper = 64 and 8 experts, identifying the right one of 1,000 messages succeeds 25 % of the time (MAP 36 %, FHRR 82 %); and `HyperspaceGlobalBus` is constructed and put in the optimiser but never executed in any forward path (only a synthetic benchmark and a shape/NaN test).
- **Atoms are not reproducible**: `torch.manual_seed(abs(hash(name)) …)` uses Python's per-process-salted string hash — verified: three runs, three different `hash('cat')` — and reseeds the *global* RNG as a side effect.
- **A documentation claim is dimensionally impossible**: "FHRR ≅ T⁸ ⊂ S⁷" (SPEC §2.1). dim T⁸ = 8 > 7 = dim S⁷. Correct: D unit phasors give T^D ⊂ S^{2D−1}; ℝ⁸ = ℂ⁴ hosts T⁴.
- **Measurement floor**: `geodesic_distance_sphere(x, x)` returns 4.47 × 10⁻⁴ rad (clamp at 1 − 10⁻⁷), so it cannot resolve the README's quoted 4.1 × 10⁻⁷ rad; I could not find the script that produced that number, and parallel transport preserves norm by construction, so it is an implementation check, not capability evidence.
- **What already exists and lines up with the literature** (present in the code; I did not audit correctness or whether each piece is on the live path): modern-Hopfield cleanup, a k-WTA dentate expansion to 4,096 dimensions, a multi-scale Fourier grid encoder, surprise-gated memory writes, sleep consolidation, a Muon optimiser — and, importantly, your measurement discipline. The recommended direction reuses these.

---

## 11. Open problems

1. **Learn the group, not just the operators.** How does a network discover *which* symmetry group (abelian, dihedral, permutation…) governs a domain from data? Representation-theoretic regularisers and differentiable discovery of group structure via unitary representation learning (ICLR 2025 [V]) are starting points.
2. **Capacity-aware hierarchy.** What replaces a flat bundle when M·ln N exceeds D/7.5? Hashing, SDM banks, product-key addressing, orthogonal subspace carving — none benchmarked together.
3. **Semigroups and set-valued inverses.** A clean treatment of non-invertible but compositional dynamics inside a VSA.
4. **Depth without drift.** Cleanup schedules for 10+ binding levels; manifold-aware cleanup on the torus (Habashy 2026).
5. **Evaluation.** "VSA-likeness" and compositional generalisation need standard, leak-proof, numerically graded benchmarks; ARC-AGI-3 is the interactive frontier.
6. **Scale.** No VSA-first model has been shown at LLM scale; Engram, Hrrformer and GHRR-attention are the nearest experiments, all author-reported.
7. **Theory of emergence.** Why does SGD choose Fourier/irrep solutions? The 2026 "learning mechanics" programme (Simon et al.) is the right frame.

---

## Appendix A — Reproducing every [M] number

```bash
python research/vsa_numerics.py          # E1–E6: quasi-orthogonality, bundling capacity, binding schemes, FPE, resonator, retrieval (28 s CPU)
python research/geometry_checks.py       # H1–H3: holonomy and analogy tests against ishmn/geometry.py
```

Logs: `research/results_vsa_numerics.log`, `research/results_geometry_checks.log`; JSON in `research/results/`.
Seeded (`--seed 0`). The resonator in E5 is the plain sequential version; published improved dynamics are stronger.

## Appendix B — Not verified / known soft spots

- All 2026 items were read at abstract level only. Treat reported numbers as the authors' claims.
- ARC-AGI-3 launch numbers come from press coverage; the March 2026 living survey reports different "best system" numbers across versions (93 % / 68.8 % / 13 % on ARC-AGI-1/2/3) — I could not reconcile these definitions.
- Longuet-Higgins (1968) page number differs between sources; ≈ 110 classics are [R].
- `vsa_numerics.py` measures *random-code* geometry. Learned codes can be better (hippocampal "crystalline" codes, equiangular tight frames) or worse; that is exactly what the protocol measures on real latents.

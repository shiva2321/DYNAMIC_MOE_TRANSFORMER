# Yggdrasil architecture: research and brainstorm (neuron, attention, transformer, geometry)

Date 2026-10-05. Written by Claude (designer/reviewer) for the user (mediator) and Gemini (implementer).
Companion files: `docs/superpowers/handoff/YGGDRASIL_ARCH_HANDOFF.md` (what Gemini builds, round by round),
`research/yggdrasil_poc/` (every experiment script), `research/yggdrasil_results/` (their raw outputs),
`geometry/{lorentz,attention,neuron}.py` + `tests/test_lorentz_attention.py` (new, tested building blocks).

---------------------------------------------------------------------------------------------------------------------------
## 0. How to read this

Plain words first, then the precise term (defined once). Every claim carries a tag so you can tell what is solid:

| Tag | Meaning |
|---|---|
| **[M]** | **Measured by us.** Script in `research/yggdrasil_poc/`, numbers in this file, raw output in `research/yggdrasil_results/`. |
| **[T]** | **Derived maths**, pinned by an executable test in `tests/test_lorentz_attention.py` (so it cannot silently break). |
| **[L]** | **Literature.** Title/ID checked in this session (section 12); I did not re-derive their results. |
| **[H]** | **Hypothesis.** Plausible, nobody (including us) has measured it. Treat as a bet, not a fact. |

Mini-glossary (used throughout):
* **Hyperbolic space**: a curved space where the room available grows *exponentially* with distance from a centre - exactly how a tree
  grows with depth - so a tree fits in few dimensions without crowding. **Lorentz model / hyperboloid**: the same space drawn as a bowl-shaped
  surface in one extra dimension. In it, "distance" becomes (almost) a dot product, which is what GPUs are fastest at.
* **Torus / phase**: a vector of angles. Adding angles is *commutative* (order does not matter) -> exact undo, but it cannot remember order.
  **Abelian** = commutative; **non-abelian** = order matters (like composing rotations or shuffling cards).
* **Salience / radius**: how "important" or how "deep" a token is; in the Yggdrasil state it is the radial coordinate.
* **State tracking**: remembering the running result of a sequence of order-dependent operations (the *word problem* of a group).
* **MAP**: how well each node's tree neighbours (parent, children) are ranked first by distance (1.0 = perfect). **Distortion**: average
  relative error between embedded distance and true tree distance (0 = perfect).
* **SDPA / FlashAttention**: PyTorch's fused attention kernel; never builds the T x T matrix, so memory is O(T).
* **Mantissa / cancellation**: floating-point numbers keep only a few significant bits; subtracting two big nearly-equal numbers destroys them.

---------------------------------------------------------------------------------------------------------------------------
## 1. The short version

1. **What Yggdrasil is good for (measured).** Squeezing a *hierarchy* into a *small* number of dimensions. At equal budget (600 tuned steps), hyperbolic
   embeddings of a 1093-node tree beat Euclidean ones by a wide margin at d = 4-16 and meet them at d = 32 [M]. At d = 4 and 8 this is
   capacity, not training speed (2000 steps: Poincare MAP 1.000, Euclidean 0.36 / 0.59); d = 16 was not re-checked at longer training [M]. This matches theory [L].
2. **What it was oversold on.** (a) The torus *algebra* is exact (angles add and negate exactly), but a **learned** set of angles is never exactly the
   group it should represent, and the error adds up with sequence length: on a cyclic-group task the torus model is 100 % at length 16 and **9 % at length 256** [M];
   only restricting the angles to a lattice removes this. (b) A commutative state cannot track
   order-dependent state at all (S3: 33 % at length 16, ~16 % beyond; chance is 16.7 %) [M]. (c) Splitting the coordinates 50/50 between hyperbolic base and torus does **not** help
   a pure tree: at equal total d the product's distortion is about the same as pure hyperbolic but neighbour retrieval is lower (MAP 0.94 vs 0.97 at d = 8,
   0.96 vs 0.99 at d = 32, 0.41 vs 0.86 at d = 4); the hyperbolic coordinates do the hierarchy work [M]. (d) Earlier "fibre bundle / zero holonomy" wording was wrong
   (it is a Riemannian *product*), and some printed structural numbers were hard-coded; both were corrected.
3. **The architecture I propose ("YggFormer", working name): put the geometry inside the attention heads of an ordinary transformer and keep
   the residual stream ordinary.** The attention score is hyperbolic similarity + torus phase similarity + a per-key salience bias,
   and these three **collapse into ONE dot product on augmented vectors** [T]. So it runs on the standard fused attention kernel (causal masks,
   KV cache, GQA all work), costs about **1.2-1.4x the attention core of a normal head** at T = 2-4k [M] and keeps O(T) memory
   (19 MB at T = 2048 versus ~8.6 GB for the pair-by-pair form, extrapolated) [M].
4. **The neuron.** One linear unit plus *one extra learned weight on the soft-norm of its input* is, depending only on the sign of a
   number, a *prototype* neuron (distance to a point), an ordinary *half-space* neuron, or a *horosphere* neuron [T]. The old prototype
   "memory cortex" is the first kind. Caveat: it only helps where the input *norm* carries information, and standard pre-norm (RMSNorm)
   throws the norm away - so use it on the head's q/k/v points, not blindly in the FFN [T + H].
5. **State tracking: do not rely on the torus.** A gated recurrence (GRU) solved both groups to length 256 (>= 99.8 %) [M]; Householder products
   and 8-dimensional rotation states learn the non-abelian task in-distribution (100 % at length 16) but **drift with length** (44-60 % at 256; the 3-dimensional
   rotation is less reliable still, 92 % -> 65 %) [M];
   an unsnapped torus drifts even on the commutative task [M]; a *lattice-snapped* torus has no drift at all (flat 94-96 % from length 16 to
   256; 8/10 seeds at >= 95 %, the other 2 did not converge) but cannot do the non-abelian task [M]. Literature says the same: affine recurrences cannot correct
   errors along state-separating directions [L]. Use gated/DeltaProduct-style recurrences for state; use the torus for binding and position.
6. **Numerics decide where this is allowed to live.** The hyperbolic logit subtracts two numbers of size R^2 to get an O(1) answer, so
   half-precision logit error grows like R^2: bf16 is unsafe beyond radius ~2, fp16/TF32 beyond ~8, fp32 is fine past 64 [M]. Rules:
   QK product in fp32, clip the radius, **no QK-norm** (it erases the radius), small curvature-aware initial radius.
7. **The field is real but not magic.** Fully hyperbolic LLMs exist (HELM, billion scale, up to +4 % on MMLU/ARC) [L]; linear-time
   hyperbolic attention exists (Hypformer) [L]; hierarchy-aware cone attention exists (Coneheads) [L]. A 2026 audit shows many
   hyperbolic models quietly collapse to near-Euclidean (they lower the curvature) so "the geometry" may not be doing the work [L].
   **Therefore every experiment must report its geometry operating point** (section 7).
8. **GPU kernel.** Do *not* write a custom attention kernel: the fused SDPA path already does it. The existing pairwise-distance kernel is
   ~6x above the dot-product floor (2.0 ms vs 0.34 ms at 4096^2) [M]; the right upgrade is "GEMM + fused epilogue", not more thread-per-pair code.
9. **Build order**: (1) heads-only YggFormer vs Euclidean heads on hierarchy tasks at small head dimension; (2) numerics runs;
   (3) snapped/gated state-tracking module; (4) tree-routed sparse attention (the hyperbolic kernel is so sharp that top-k should lose
   almost nothing [T + H]); (5) GrowMoE tree-addressed experts; (6) kernels. **Kill rule**: if (1) shows no gain at head_dim <= 16 over a
   parameter-matched Euclidean transformer, drop hyperbolic attention and keep the geometry for routing/memory only.
10. **What I changed in the repo this session** (what/how/why in section 9): engine bug fixes (earlier), plus new tested modules
   `geometry/lorentz.py`, `geometry/attention.py`, `geometry/neuron.py`, and 25 new tests (68 geometry tests pass on the real GPU engine).

---------------------------------------------------------------------------------------------------------------------------
## 2. Where Yggdrasil stands after the review

* **Space.** Riemannian *direct product* Poincare ball H^{d_H} x flat torus T^{d_T} x radial R+ (log-ratio distance). Not a twisted bundle:
  the hyperbolic and phase parts do not interact. Mobius addition is a *gyrogroup* (non-associative, non-commutative; left-cancellation is
  exact; finite depth budget: the ball clamps at radius 1 - 1e-4, i.e. hyperbolic radius ~9.9). The torus is FHRR / RotatE prior art and abelian.
* **"Memory cortex"** = softmax(-d^2/tau) over K = 20 prototypes, a prototype lookup, not token-token attention; a residual bypass lets the
  model ignore the manifold; digits cannot test geometry (Yggdrasil 97.07 +- 0.89 vs flat 97.69 +- 0.42) [M, earlier].
* **GPU engine**: Rust cdylib (ctypes) + CUDA PTX built with NVRTC; naive thread-per-output pairwise distance (no tiling, no tensor cores).

Engine fixes made earlier in this session (all tested; details in section 9): radial distance mismatch GPU vs reference; retraction left dims
>= 64 uninitialised; custom backward materialised (N, M, D) tensors; PTX without NUL terminator silently fell back to CPU (so "GPU tests"
passed on CPU); fake hard-coded numbers in the ablation script replaced by an honest "not measured".

---------------------------------------------------------------------------------------------------------------------------
## 3. What mainstream architectures get wrong, and what we can honestly say about each

| # | Shortcoming (plain words) | Root cause | Yggdrasil mechanism | Evidence | My honest read |
|---|---|---|---|---|---|
| 1 | Hierarchies (taxonomies, code, syntax) crowd in flat embedding space; need many dimensions | Volume of Euclidean balls grows polynomially, trees grow exponentially | Hyperbolic base | [L] Sala'18, HELM; **[M] A4/A5** | **Strongest case.** Gain is at small d; head dimensions are small -> natural fit [H] |
| 2 | Cost and KV-cache grow quadratically/linearly with context | Dense softmax attention | Geometry itself does **not** cut it (+20-40 % per head [M]); tree-routed top-k + latent KV could [H][L] | [M], [H] | Do not oversell: the *routing* part is the lever, not the curvature |
| 3 | Cannot track order-dependent state; fixed-depth transformers/diagonal SSMs are TC0-limited | No non-commutative state update with error correction | Torus **fails** [M]; need gated / Householder (DeltaProduct) / snapped state | [L] Merrill, Grazzi, Siems, error-control; **[M] B** | Geometry does not fix it; borrow the literature's fix |
| 4 | Binding ("which attribute belongs to which object"), compositionality | Superposition without role-filler operators | Torus phases = FHRR binding; exact inverse for commutative relations | [L] VSA; [M] Z6 exact in-horizon | [H] for LM-level gain; test E6/E7 |
| 5 | Reversal curse ("A is B" does not give "B is A") | Next-token factorisation, no tied inverse | Relation as group element has an exact inverse | [L] Berglund; inverse part [T] | Needs the *training objective* to use it too [H] |
| 6 | Attention sinks, massive activations (hurt quantisation, long context) | Softmax must sum to 1 | Explicit salience bias + learned origin ("root") key as an honest no-op target | [L] gated attention (head-wise sigmoid gate) fixes sinks | [H]; compare against the simple gate |
| 7 | Rank collapse / over-smoothing with depth | Averaging | Sharper kernel (below) may help | [L] Dong'21 | Weak [H]; skip unless cheap |
| 8 | Catastrophic forgetting, loss of plasticity | One shared weight set | Tree-addressed experts that grow (GrowMoE) | [L] Dohare'24; project | Later phase |
| 9 | Hallucination, no "I don't know" | No uncertainty channel | Radius as uncertainty/salience | [L, from memory] hyperbolic vision work reports radius tracks uncertainty | [H]; E9 |
| 10 | Position encodings extrapolate badly | Fixed frequency schedule | Torus phase = learned RoPE; horospherical positional shift (log-growing distance) | [L] RoPE; HexFormer/polar-ViT 2026 | [H] |
| 11 | Low-precision efficiency | Quantised everything | Hyperbolic QK is **precision-hungry** [M] | **[M] C** | A real *cost* of this geometry |
| 12 | Opaque | Dense distributed code | Tree address / radius / phase are inspectable | - | [H] |

Kernel sharpness note [T]: the hyperbolic logit is -(cosh(sqrt(c) d) - 1)/(c tau). It falls off like exp(d), not like d^2 (table in section 5.4),
so only the nearest keys carry weight. That is what makes top-k / tree retrieval accurate, and also why attention on this geometry is "sharper".

---------------------------------------------------------------------------------------------------------------------------
## 4. What the literature says (checked in this session unless marked)

* **Hyperbolic networks mature.** HyboNet (fully hyperbolic layers, Lorentz residual = add space parts, recompute time) [L]; Hypformer
  (hyperbolic transformer with *linear* O(N) attention) [L]; Coneheads (attention from entailment cones / depth of the lowest common ancestor,
  fewer parameters) [L]; **HELM** (billion-parameter fully hyperbolic LLMs, mixture-of-curvature experts, hyperbolic latent attention,
  hyperbolic RoPE/RMSNorm; "up to 4 %" over LLaMA/DeepSeek-style Euclidean baselines on MMLU/ARC) [L].
* **Numerics is the live problem (2026).** Numerical errors grow at large radius because of the Lorentzian inner product (polar-coordinate
  hyperbolic ViT); the Lorentz centroid loses precision so tangent-space ("exp-map") aggregation is proposed (HexFormer); a Lorentz layer
  whose output norm scales logarithmically instead of linearly with gradient steps is fixed by a distance-to-hyperplane formulation
  (Fast & Geometrically Grounded Lorentz NNs) [L]. Earlier: Poincare vs Lorentz float limits and Euclidean parametrisation (Mishne et al.), feature
  clipping (Guo et al.) [L]. **Our measurement (section 5.3) agrees: the cancellation is in the QK product itself.**
* **A warning to take seriously.** An audit of hyperbolic vision-language models found they train at u = sqrt(c)*radius <= 0.37, "far below" the
  0.84 level where hyperbolic effects matter, i.e. they became near-Euclidean by shrinking the curvature, and hierarchy scores can come from
  hyper-parameter dynamics rather than geometry [L, arXiv 2607.05268]. Our rule: always report the operating point.
* **Embedding theory.** Trees embed in hyperbolic space with arbitrarily low distortion; the precision/dimension trade-off is characterised (Sala,
  De Sa, Gu, Re) [L].
* **State tracking.** SSMs/transformers cannot do permutation composition in one pass (Merrill-Petty-Sabharwal); a multilayer SSM can solve parity
  only with a negative eigenvalue (Grazzi et al.); Householder-product transitions improve it (DeltaProduct); shallow transformers learn
  shortcuts with o(T) layers but length-generalise badly (Liu et al.); and *error control* - hidden state drifting along the directions that
  separate states - explains length-generalisation failure; affine recurrences provably cannot correct such errors (arXiv 2605.07755) [L].
  **This is exactly what our B experiment saw.**
* **Attention sinks.** Head-specific sigmoid output gate after SDPA removes sinks, improves stability and extrapolation (Qiu et al., NeurIPS 2025) [L].
* **Routing in log time.** Fast Feedforward Networks (balanced binary tree of neurons, O(log n)), PEER (product-key retrieval over a million tiny
  experts) [L]. Native sparse attention and fast-multipole style hierarchical attention also exist [L, earlier context, not re-checked today].
* **Phases already live inside LLMs.** RoPE is a torus action; LLMs use Fourier/circular features for arithmetic (Nanda; Kantamneni-Tegmark) [L, earlier context].

Positioning (no novelty claim for hyperbolic attention itself): what is ours is (i) **one fused dot product** carrying hyperbolic + torus-cos +
salience, with equivalence tests and a cost/precision map; (ii) the **lattice-snapped phase** result; (iii) the **unified neuron view**;
(iv) tying all of it to tree routing and GrowMoE growth.

---------------------------------------------------------------------------------------------------------------------------
## 5. What we measured (small, memory-lean experiments)

All on one RTX 3060 (12 GB) or CPU; every run peaked below ~1.5 GB VRAM. Protocol: dev seed 100 for choosing learning rates, test seeds 0-9
(or 0-2 where marked), mean +- 95 % CI (t-interval), numeric grading, cosine learning-rate decay.

### 5.1 Tree embedding: Euclidean vs hyperbolic vs product (A2, A3, A4, A5)

Task: embed a ternary tree of depth 6 (1093 nodes) with total d coordinates. **A first version (A) used one constant learning rate and gave a
misleading "hyperbolic stops winning at high d". A3 showed the *same* geometry's d = 32 MAP swings between 0.58 and 1.00 with the learning
rate alone. A4 gives every (geometry, d) its own learning rate (grid 0.1/0.03/0.01/0.003), 600 steps, 3 test seeds.** A (untuned) is kept in
the repo marked SUPERSEDED.

MAP (higher is better):

| d | Euclid | Poincare | Lorentz | Product H^(d/2) x T^(d/2) |
|---|---|---|---|---|
| 2 | 0.048 | 0.342 | 0.213 | - |
| 4 | 0.349 | 0.863 | 0.858 | 0.410 |
| 8 | 0.803 | 0.969 | 0.991 | 0.937 |
| 16 | 0.794 | 0.994 | 0.984 | 0.957 |
| 32 | 0.979 | 0.991 | 0.988 | 0.962 |

Distortion (lower is better): Euclid 0.311 / 0.178 / 0.099 / 0.061 / 0.042 for d = 2/4/8/16/32; Poincare 0.202 / 0.140 / 0.050 / 0.036 / 0.032;
Lorentz 0.223 / 0.145 / 0.047 / 0.021 / 0.015; product - / 0.120 / 0.045 / 0.031 / 0.030.

A5 (longer training, 2000 steps, same protocol, lr grid 0.1/0.03/0.01) at d = 4 and 8 - tests whether the gap is only training speed:

| d | Euclid MAP | Poincare MAP | Lorentz MAP | Euclid / Poincare / Lorentz distortion |
|---|---|---|---|---|
| 4 | 0.363 | 1.000 | 0.999 | 0.166 / 0.036 / 0.038 |
| 8 | 0.592 | 1.000 | 1.000 | 0.094 / 0.020 / 0.020 |

With four times more steps the hyperbolic embeddings sit at the ceiling while the Euclidean ones do not (Euclid d = 8 reaches 0.59 here but 0.80 in the 600-step run:
its MAP is erratic in learning rate and seed, read it as "0.6-0.8"). So at d = 4-8 the gap is **capacity, not convergence speed** [M].

What this shows [M]: (a) hyperbolic wins at small d at equal budget; (b) the same geometry has lower distortion at every d (Lorentz up to 2.8x
lower at d = 32); (c) **the product is not better than pure hyperbolic at equal total d**: its distortion is about the same (0.045 vs 0.050 at d = 8, 0.030 vs 0.032 at d = 32;
0.120 vs 0.140 at d = 4) but its neighbour-retrieval MAP is lower (0.937 vs 0.969, 0.962 vs 0.991, 0.410 vs 0.863) - the torus coordinates act like extra
dimensions for distances but do not help rank neighbours, and with only 2 hyperbolic dimensions (d = 4) the tree cannot be arranged at all; (d) Poincare and Lorentz are the same geometry - gaps between them are optimisation/numerics, Lorentz
reaches the lowest distortion; (e) in A2 float32 and float64 evaluation of the same Poincare embedding agree to 3 decimals and no point comes
near the boundary (max tangent norm 1.65-1.85, hyperbolic radius ~3.3-3.7, u = sqrt(c)*radius far above the 0.84 level the audit asks about),
so the earlier worry "boundary precision" does not apply at this scale.
What it does NOT show: that a *learned* language model's representations are tree-like, or that the gain survives at d = 64-128.
Selection noise: one dev seed picks the learning rate; MAP is very learning-rate sensitive (see `poc_a4_tree_embedding_tuned.json`, field `sel`),
so differences below ~0.05 MAP are not meaningful.

### 5.2 State tracking: which fibre algebra can remember order? (B)

Group word problem on two groups of the same size 6: **Z6 (commutative)** and **S3 (non-commutative)**. Train on lengths <= 16, test the running
product after the last symbol at lengths 16-256 (chance 16.7 %). 10 seeds, learning rate tuned per model on dev seeds over *both* tasks.
(Parameter counts: torus and snapped torus 1526, SO(3) 667, SO(8) 1142, Householder 1070, GRU 5094 - the GRU has ~3-5x more parameters.)

| task | model | T=16 | T=32 | T=64 | T=128 | T=256 | seeds >= 95 % at 256 |
|---|---|---|---|---|---|---|---|
| Z6 | torus (angles add) | 100.0 +- 0.0 | 79.7 +- 23.1 | 18.3 +- 15.2 | 9.3 +- 12.7 | 9.0 +- 8.9 | 0/10 |
| Z6 | **lattice-snapped torus** | 95.6 +- 7.6 | 91.7 +- 17.8 | 94.9 +- 9.7 | 93.9 +- 12.2 | **93.8 +- 12.3** | 8/10 |
| Z6 | SO(3) rotations | 64.5 +- 19.2 | 51.8 +- 24.7 | 38.2 +- 28.4 | 27.7 +- 20.0 | 21.5 +- 13.2 | 0/10 |
| Z6 | SO(8) rotations | 100.0 +- 0.0 | 99.8 +- 0.4 | 81.3 +- 21.9 | 45.9 +- 22.3 | 24.9 +- 9.8 | 0/10 |
| Z6 | Householder x2 | 46.3 +- 7.9 | 33.9 +- 3.9 | 25.8 +- 2.2 | 20.0 +- 1.3 | 16.9 +- 0.7 | 0/10 |
| Z6 | GRU | 100.0 +- 0.0 | 100.0 +- 0.0 | 100.0 +- 0.0 | 99.9 +- 0.1 | 99.8 +- 0.2 | 10/10 |
| S3 | torus | 33.1 +- 1.5 | 17.5 +- 9.2 | 15.0 +- 10.3 | 16.5 +- 8.7 | 18.0 +- 8.7 | 0/10 |
| S3 | lattice-snapped torus | 29.1 +- 5.2 | 24.7 +- 8.1 | 27.2 +- 5.3 | 25.8 +- 5.9 | 26.4 +- 5.9 | 0/10 |
| S3 | SO(3) rotations | 91.5 +- 15.4 | 87.7 +- 16.6 | 81.7 +- 21.5 | 74.7 +- 24.9 | 64.6 +- 23.9 | 1/10 |
| S3 | SO(8) rotations | 100.0 +- 0.0 | 100.0 +- 0.0 | 89.2 +- 18.1 | 68.7 +- 17.5 | 44.2 +- 4.8 | 0/10 |
| S3 | Householder x2 | 100.0 +- 0.0 | 99.7 +- 0.8 | 97.6 +- 3.6 | 85.9 +- 9.1 | 59.5 +- 12.0 | 0/10 |
| S3 | GRU | 100.0 +- 0.0 | 100.0 +- 0.0 | 100.0 +- 0.0 | 99.9 +- 0.1 | 100.0 +- 0.0 | 10/10 |

Reading [M]: (1) **Abelian state cannot do S3** (torus and snapped torus stay near chance) - the theory is confirmed. (2) **Every
continuous-state model drifts**: the learned angles/rotations are never *exactly* the finite group, and the error adds up with length (torus on
Z6: 100 % -> 9 %). (3) **Snapping to a lattice removes the drift** (integer arithmetic modulo N is exact), at the cost of needing the lattice to
contain the group (we gave it N = 12 for a group of order 6; 2/10 seeds did not converge and stayed below 95 %, the 8 that did stayed at >= 95 % to length 256).
(4) **Non-abelian continuous fibres have the capacity** (SO(8), Householder: 100 % at 16) **but not the exactness** (44-60 % at 256).
(5) The GRU - a gated *non-linear* update, which can pull the state back toward a clean value each step - is exact to 256 at both tasks.
Not measured: a *snapped non-abelian* state (vector-quantised rotation state or finite-group lookup) - this is the obvious next test (E2b);
bigger groups (S5); whether the GRU's win survives with equal parameter count.

### 5.3 Neuron identities, attention fusion, precision (C)

* **One neuron, three kinds [T/M].** For w on the hyperboloid, <w, x>_L is (i) `-sqrt(w0^2-|w_s|^2) cosh d(x, p)` if w is time-like (distance to a prototype
  p), (ii) `|w| sinh(signed distance to a geodesic hyperplane)` if space-like, (iii) `-exp(Busemann function)` if light-like (a horosphere). Errors 4e-14 / 2e-7
  (limited by the brute-force line search) / 8e-14. Pinned by unit tests.
* **Attention = one dot product [T/M].** Pair-by-pair definition vs the fused augmented-vector SDPA: max difference **1.4e-14 (float64), 4e-6 (float32)**;
  output is the Lorentz centroid (valid hyperboloid point); per-head curvature, temperature, phase strength, causal mask and salience bias all match.
  The flat limit c -> 0 is Gaussian (L2) attention [T]. Gradient check passes.
* **Precision** (error in nats of the logits of nearest-neighbour clusters; < 0.05 is safe; D = 32):

| R = ||x_s|| | hyperbolic radius | fp32 | tf32 | fp16 | bf16 |
|---|---|---|---|---|---|
| 1 | 0.88 | 0.0000 | 0.0011 | 0.0015 | 0.0104 |
| 2 | 1.44 | 0.0000 | 0.0034 | 0.0036 | 0.0280 |
| 4 | 2.09 | 0.0000 | 0.0200 | 0.0197 | **0.0992** |
| 8 | 2.78 | 0.0000 | 0.0443 | 0.0459 | **0.3621** |
| 16 | 3.47 | 0.0001 | **0.1779** | **0.1777** | 1.8893 |
| 64 | 4.85 | 0.0011 | 2.4593 | 2.4593 | 11.4410 |

  Error ~ 2^-p R^2. Safe radius: bf16 ~ 2, fp16/TF32 ~ 8, fp32 > 64 (in code: `geometry.attention.safe_radius`). Implication: a bf16 model can only
  use hyperbolic radius up to ~1.4 in the dot product; deeper hierarchies need fp32 QK, split-precision, local frames (below) or tangent-space aggregation.

### 5.4 Cost (C, C2)

Core of the fused call (forward only, B = 1, H = 4, head dim 56 after augmentation vs 32 plain): fp16 core 0.059 / 0.127 / 0.271 / 0.701 ms at T = 256 /
1024 / 2048 / 4096 versus plain Euclidean 0.058 / 0.106 / 0.207 / 0.490 ms -> **1.0x / 1.2x / 1.3x / 1.4x**. The fp32 core is 3.4x slower than fp16 at T = 2048
(no flash path in fp32). Un-fused preparation (concatenations, cos/sin) costs ~0.4 ms and the centroid renormalisation ~0.13 ms *in eager mode*; both are
launch-bound and removable by fusion (`torch.compile` where available, or folding into the projection). Memory (forward + backward, fp32): pair-by-pair
(T, T, D) form 135 MB at T = 256 and 541 MB at T = 512 (quadratic, would be ~8.6 GB at T = 2048); fused SDPA 19 MB at T = 2048.
Current Yggdrasil pairwise kernel 2.0 ms vs one GEMM of the same shape 0.34 ms at 4096^2 (different functions; GEMM is the floor).
Hyperbolic logit vs Gaussian logit at geodesic distance d = 0..6: `-(cosh d - 1)` = 0, -0.5, -2.8, -9.1, -26.3, -73.2, -200.7 versus `-d^2/2` = 0, -0.5, -2, -4.5, -8, -12.5, -18.

### 5.5 Does a transformer with these heads train? (D, smoke test only)

2-layer transformer (d_model 64, 4 heads of 16, 4 phases), Ygg heads (hyperbolic + torus phase + salience bias; per-head learned curvature, temperature, phase strength; bf16 autocast with the
manifold product in fp32) versus Euclidean heads, on an induction-copy task ([a1..a8, SEP, a1..a8], score the copy). **Both reach 100 % copy accuracy in 400 steps**; no non-finite gradients;
gradients reach curvature, temperature and phase strength; curvature moved only from 1.0 to 0.93-0.96. **Operating point at default initialisation: median u = 1.8-2.0 (p95 2.2-2.4)**, i.e. the heads
start already well inside the hyperbolic regime (radius ~3, which bf16 could not carry) - so *initial radius must be set explicitly* (section 6.7), not assumed small [M].
This only shows the design trains; the task is solved by both arms and says nothing about whether the geometry helps (that is E3).

---------------------------------------------------------------------------------------------------------------------------
## 6. The proposed architecture ("YggFormer", working name)

### 6.1 Design rules

1. **Geometry lives inside the heads; the residual stream stays an ordinary Euclidean vector.** This keeps tensor parallelism, quantisation of
   weights, LoRA, KV caches, mixed precision (except the manifold dot product) and every existing kernel. A fully hyperbolic stream (HELM-style) is a
   *later variant*, not the start.
2. **One dot product.** Any geometric term must be expressible as an inner product so the fused kernel stays valid (done for three terms [T]).
3. **Curvature is learned per head, and reported.** Heads can end up hyperbolic, flat (L2 attention, c -> 0) or - as an extension - spherical.
   A mixture of curvatures across heads is a free "mixed-curvature" model (Gu et al.-style [L], HELM-MICE [L]).
4. **Never erase the radius**: no QK-norm on hyperbolic q/k; use radius *clipping* with a dtype-dependent limit.
5. **Compute the manifold product in fp32** (or TF32 only with clip <= ~6); keep everything else in bf16.
6. **Anything exact and order-dependent goes through a state module with error correction** (gated or snapped), not through geometry.
7. **Every experiment prints the operating point** (section 7).

### 6.2 The token inside a head: three fields

For each token and head the projections produce: a point on the hyperbolic base (`q_s`, `k_s`, `v_s` in R^D, time coordinate derived: `x0 = sqrt(1/c + |x_s|^2)`),
phases `theta` (M angles, stored as angles; cos/sin taken in the attention), and a salience bias `beta` (a scalar per key; log-radius in the old state).
Plain meaning: *where in the hierarchy* (base), *which role/relative position* (phase), *how much this token deserves attention regardless of query* (salience).

### 6.3 The neuron

`a_j(x) = <w_j, x> + b_j - c_j * sqrt(1/c + |x|^2)` (class `geometry.neuron.NormGatedLinear`).
* c_j = 0: an ordinary linear unit (initialisation, so nothing changes at step 0).
* |w_j| > |c_j|: half-space neuron; |w_j| = |c_j|: horosphere neuron; |w_j| < |c_j|: prototype ("ball") neuron [T]. Learned per unit; `causal_character()` reports which.
* Cost: one extra scalar feature per token, one extra weight per unit.
* **Where it can work**: on q/k/v points and on any layer whose input norm is meaningful. **Where it cannot**: right after RMSNorm (the magnitude has been removed, so the extra
  term is only a smooth function of the direction and loses its geometric meaning). Option for the FFN: feed the *pre-norm magnitude* as the explicit scalar (reintroduces what RMSNorm discards) [H, test E8].
* Companions (cheap, optional): a *phase gate* (`theta' = theta + A x`, a content-dependent rotation; exact inverse) and a *salience gate* (`sigmoid(beta*log r + gamma)`
  multiplying unit outputs; similar in spirit to the head gate of Qiu et al. [L]).
* Honest status: identities exact [T]; benefit unmeasured [H]. The old prototype lookup is the time-like special case, so nothing is lost by switching.

### 6.4 Attention: `geometry/attention.py`

```
logit_ij = (1/tau) <q_i, k_j>_L  +  kappa * sum_m cos(theta_q,i,m - theta_k,j,m)  +  beta_j          (+ causal mask)
q~ = [ -q0/tau, q_s/tau, sqrt(kappa) cos(theta_q), sqrt(kappa) sin(theta_q), 1    ]       k~ = [ k0, k_s, sqrt(kappa) cos(theta_k), sqrt(kappa) sin(theta_k), beta ]
out = Lorentz-centroid( softmax(q~ k~^T) @ lift(v) )   (or tangent-space mean, or Euclidean mean of values)
```
* One `scaled_dot_product_attention(q~, k~, v~, scale=1.0)` call; head dim = D + 1 + 2M + 1 padded to a multiple of 8 (for D = 32, M = 8: 56).
* The cosine term is a von-Mises kernel: smooth where the wrapped torus distance has a cut locus; it is RoPE/FHRR phase similarity.
* Aggregation options: `lorentz` (centroid; default), `tangent` (more precise at large radius, 2026 HexFormer argument [L]), `euclid` (geometry only decides weights; the cheapest ablation).
* Null/root key: add one learned key at the origin (x_s = 0, beta learned) so "attend to nothing" has a principled target (alternative to sinks) [H].
* Positional information: either RoPE on a separate phase block, or the 2026 **horospherical shift** (relative-position offsets whose distances grow logarithmically with token gap) [L][H].
* GQA/MQA and KV-cache: store `k~`, `v~` (augmented, padded); nothing else changes.

### 6.5 The block and the stack

* v0 (build first): pre-norm Euclidean transformer; attention replaced by YggAttention heads; FFN unchanged (optionally NormGatedLinear on a pre-norm-magnitude feature).
* v1 (only if v0 wins): hyperbolic residual stream as in HyboNet/HELM: stream = space-like parts, residual = add space parts (time recomputed) [L][T], with a manifold-aware norm
  that preserves radius. More invasive, more numerics risk.
* **FFN as tree-routed experts (links to GrowMoE).** The Yggdrasil tree is the router: a token's hyperbolic position (direction at radius r = which child; radius = depth) is its
  tree address, so descending the tree costs O(log N) (Fast-Feedforward-style [L]); GrowMoE's novelty-triggered growth adds a child at the angular sector where a token is far from all
  existing nodes. [H] - design only until v0 is evaluated.

### 6.6 State tracking and memory

* **State module** (not geometry): a gated delta/Householder-product recurrent mixer in about 1 of every 4 layers (hybrid), as the literature and our B experiment suggest. For tasks that need
  *exactness over unbounded length*, add a **snapped state**: lattice phases for commutative parts (measured exact [M]); for non-commutative parts a vector-quantised or finite-group-lookup
  state (E2b, unmeasured).
* **Long-context memory.** Tree-addressed key store: each node keeps a Lorentz centroid and a radius bound; a query descends by an *upper bound on the logit* (the logit is a monotone function
  of the geodesic distance, so the triangle inequality gives a valid bound [T]) with beam B; leaves are attended exactly. Because the kernel is exponentially sharp, B = 8-32 should lose little [H, E5].
  Salience (the radial/bias term) doubles as an eviction priority. **Local frames**: express a block's points relative to its parent node (a Lorentz boost) so coordinates inside a block are small,
  which restores precision (the R^2 rule above) [H].

### 6.7 Training recipe

Parameters stay Euclidean; points are `exp_0`-style charts (no boundary to hit; Mishne-style Euclidean parametrisation [L]); curvature `c = softplus(gamma)+eps`, initial 1, learned per head, clamped
to a range; temperature per head; **initial radius set explicitly** (a learnable per-head scale on q_s/k_s, initialised so the median u is ~0.3-0.5 at step 0, so heads start nearly flat and *grow* hierarchy where the data asks for it;
the smoke test shows default random projections start at u ~ 1.9, deep in the hyperbolic regime, and bf16 could not carry that); radius clip with `clip_radius(x_s, safe_radius(dtype))`; no QK-norm;
fp32 manifold product; salience bias init 0; weight decay not applied to curvature/temperature. Optimiser: standard AdamW (no Riemannian optimiser needed in the chart).

### 6.8 What this will NOT fix

Quadratic dense attention (unless routing is added); factual accuracy; order-dependent state by geometry; cheap low-precision inference of the hyperbolic dot product; and it may give only a few
percent on natural text (HELM reports up to 4 % at ~1B parameters, with a large engineering effort) [L]. The honest bet is: **real gains on hierarchical/relational data at small head dimension,
modest on generic text**.

---------------------------------------------------------------------------------------------------------------------------
## 7. Experiments (pre-registered; Gemini implements, see the handoff)

Common protocol: >= 10 seeds (dev seeds >= 100 for tuning, test seeds 0-9), mean +- 95 % CI, paired tests, parameter-matched baselines, numeric grading, local data only.
**Geometry operating-point report (mandatory in every YggFormer run; inspired by arXiv 2607.05268, our own definition)**: per head the learned curvature c; quantiles (median, p95, max) of
u = sqrt(c)*asinh(sqrt(c)*|x_s|) for q, k; fraction of heads with median u < 0.37 ("effectively Euclidean"); rank correlation of radial order across seeds; top-1 attention mass; share of logit variance
from (hyperbolic, phase, bias) terms. A "win" counts only if the hyperbolic heads actually operate at u > ~0.84 and stay stable across seeds.

| ID | Question | Design | Prediction | Kill / stop |
|---|---|---|---|---|
| E1 | Tree capacity (done) | A4/A5 | hyperbolic >> Euclid at d <= 16, equal by 32; product <= hyperbolic | done; extend only to head dims 8-64 inside E3 |
| E2 | Which state fibre? (done for S3/Z6) | B | torus fails S3; continuous fibres drift; GRU exact | done |
| E2b | Snapped non-abelian state | S3 and S5 word problems; VQ-snapped rotation/Householder state and finite-group lookup, lengths 16 -> 256 | >= 95 % at 256 on S3; S5 harder | if < 80 %, rely on gated recurrences |
| E3 | Do hyperbolic **heads** help hierarchy? | Small transformer, tree-lookup/ancestor/LCA synthetic tasks (depth 3-7), head_dim in {8,16,32,64}; Euclid vs Ygg-H vs +phase vs +salience; parameter-matched | Ygg >= Euclid + 10 pp at head_dim <= 16 and depth >= 5; no difference at head_dim >= 64 | **no gain at head_dim <= 16 -> drop hyperbolic attention** |
| E4 | Numerics | E3 model, QK in fp32 / tf32 / fp16 / bf16, with and without radius clip; NaN rate, radius quantiles, accuracy | bf16 without clip degrades once u > ~1.4; fp32 and clipped fp16 fine | if clip hurts accuracy, use local frames |
| E5 | Sparse retrieval | Trained E3/E8 model: attention mass captured by top-B keys, B in {4,8,16,32}, T up to 16k (chunked) | top-16 captures > 99 % of mass for Ygg heads, clearly less for Euclid heads | if equal, drop the "naturally sparse" argument |
| E6 | Relational inversion | Synthetic facts "A r B"; test "B r^-1 A"; standard LM vs torus/lattice relation operators, with/without reverse augmentation | operator model >= 90 % for commutative relations; standard LM near chance | low prior; skip if E3 fails |
| E7 | Binding | variable-binding / role-filler tasks, phases on/off | phase helps | low prior |
| E8 | Small LM | <= 10M parameters on local text; Euclid / Ygg-H / +phase / +salience / (v1) hyperbolic stream; perplexity | within +-2 % of Euclid | gains > 2 % would be a pleasant surprise |
| E9 | Sinks / outliers | Measure first-token attention mass and max activation norm in E8 models; Ygg with null key vs Euclid vs gated attention | lower sink mass with null key + salience | compare against the simple gate; keep whichever is cheaper |
| E10 | Tree-routed experts + growth | GrowMoE streams S1-S7 with tree-addressed experts | later | design only now |
| E11 | Kernels | GEMM+epilogue distance; split-precision (3xTF32) QK | >= 4x over current kernel at 4096^2; fp32-equivalent accuracy to R <= 32 | - |

---------------------------------------------------------------------------------------------------------------------------
## 8. GPU / kernel plan

* **Attention: no custom kernel.** Use SDPA with the augmented vectors (measured 1.0-1.4x the plain core; O(T) memory). fp32 QK loses the flash path (3.4x slower than fp16 at T = 2048);
  the remedy to *try later* is a flash-style kernel with **split-precision QK on the D+1 manifold columns only** (hi/lo TF32 or bf16 pieces; phase/bias columns in low precision) [H].
* **Pairwise distance for retrieval/analysis** (the existing kernel): do not add more thread-per-pair code. Replace by GEMM + fused epilogue: hyperbolic part from `|u-v|^2 = |u|^2+|v|^2-2u.v` then acosh; torus part via cos/sin dot products (smooth form) or a separate wrapped kernel;
  radial part elementwise. Floor 0.34 ms vs 2.0 ms at 4096^2 [M].
* **Tree routing / beam search**: new batched top-B-per-level kernel (device-side), after E5.
* **Fusion of prep**: fold cos/sin/time/centroid renormalisation into one op (compile or custom CUDA).
* **Build tooling**: `rust_yggdrasil/build_ptx.py` (NVRTC, no nvcc needed; keeps the NUL terminator); add multi-arch (sm_80/86/89/90) and a PTX fallback; always construct with `require_gpu=True` in tests.
* **Windows caveat [unverified]**: Triton / `torch.compile` are not reliably available on native Windows with this install; plan on NVRTC-built CUDA (what we have) or WSL2 if Triton is wanted.

---------------------------------------------------------------------------------------------------------------------------
## 9. Changes made to the geometry package (what / how / why)

| # | What | How | Why |
|---|---|---|---|
| 1 | Radial distance was linear |a-b| in the GPU kernel and Rust crate, log-ratio in the Python reference | kernel and `manifold.rs` now use |ln a - ln b|; test compares engine to reference | engine silently computed a *different metric* than the one tested |
| 2 | Riemannian retraction left dims >= 64 uninitialised | rewritten as closed-form exp-map retraction (any dimension, analytic norm clamp) | silent garbage for dim_h >= 64 |
| 3 | Custom backward built (N, M, D) tensors | blockwise over columns with analytic gradients (hyperbolic asinh form, wrapped torus delta, log-radius with floor mask) | memory blow-up; gradient now equals autograd of the reference |
| 4 | Silent CPU fallback hid a broken GPU load (PTX without NUL -> CUDA error 218) | NUL kept; `build_ptx.py` (NVRTC); `require_gpu=True` raises; tests assert the engine really loads | "GPU" tests were passing on CPU |
| 5 | CPU retraction missing | `_retract_torch` | CPU path / tests |
| 6 | Docs overstated ("fibre bundle", "zero drift") | `hybrid.py` docstring, README section 0, brain-map caveats | wording now matches the measured facts |
| 7 | Ablation script printed hard-coded structural numbers | replaced by "NOT MEASURED BY THIS SCRIPT" | honesty |
| 8 (new) | `geometry/lorentz.py` `LorentzModel` | hyperboloid ops, Poincare conversion, exp/log (stable asinh form), Lorentz centroid, tangent mean; exported in `geometry.__init__` | boundary-free, GEMM-friendly representation; needed by 9-10 |
| 9 (new) | `geometry/attention.py` | `yggdrasil_attention` (fused), `reference_attention` (pair-by-pair oracle), `safe_radius`, `clip_radius` | the core of the proposal, with an executable specification |
| 10 (new) | `geometry/neuron.py` | `NormGatedLinear` + `causal_character()` | the unified neuron; identities tested |
| 11 (new) | `tests/test_lorentz_attention.py` | 25 tests incl. GPU memory check | all pass; geometry suite: 68 passed on the real GPU engine |

Proposed for Gemini (not done): per-head learned curvature in the old `YggdrasilManifold`; smooth cos-kernel torus option; lattice-snapped phase module; operating-point report utility; GEMM+epilogue distance kernel.

---------------------------------------------------------------------------------------------------------------------------
## 10. Roadmap

Round list is in the handoff. Order: Y1 operating-point report + YggFormer v0 skeleton -> Y2 synthetic hierarchy tasks and E3 -> Y3 numerics E4 -> Y4 state-tracking E2b -> Y5 sparse retrieval E5 -> Y6 small LM E8/E9
-> Y7 kernels -> Y8 tree-routed experts (with GrowMoE). Claude reviews each report; the user decides scope and commits.

---------------------------------------------------------------------------------------------------------------------------
## 11. Risks and what I am unsure about

* **Biggest risk: no gain on real text.** Hyperbolic LMs show a few percent at best [L]; the audit paper warns about near-Euclidean collapse [L]. E3's kill rule is there for this.
* The tree PoC is a *static embedding of an explicit tree*; an LM's heads need not be tree-like. The d = 32 equality (and the product result: extra torus coordinates did not improve retrieval) suggest "use the smallest hyperbolic d that works" and keep the torus block small.
* State-tracking models are tiny, the GRU has ~5x the parameters, only groups of order 6, 10 seeds; the lattice result used knowledge of N.
* Precision numbers are for random clusters in D = 32; real q/k distributions differ (but the R^2 law is structural).
* **Not tested at all**: fully hyperbolic stream, tree routing, reversal curse, sink behaviour, learned per-head curvature, language-model quality, S5, interaction with quantised weights.
* **Not covered in this research** (ask if you want a pass): test-time-memory models (Titans-style), diffusion LMs, byte-level models, retrieval-augmented memory, interpretability with sparse autoencoders, RL/post-training interaction, quantisation formats.
* Literature items marked "earlier context" in section 12 were verified in earlier sessions, not re-checked today.

---------------------------------------------------------------------------------------------------------------------------
## 12. References (arXiv IDs checked via search in this session unless marked)

* Chen et al., **Fully Hyperbolic Neural Networks** (HyboNet), ACL 2022, arXiv 2105.14686.
* Yang et al., **Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space**, KDD 2024, arXiv 2407.01290.
* Tseng, Yu, Liu, De Sa, **Coneheads: Hierarchy Aware Attention**, NeurIPS 2023, arXiv 2306.00392.
* He et al., **HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts**, arXiv 2505.24722; survey **Hyperbolic Large Language Models**, arXiv 2509.05757.
* van der Klis et al., **Fast and Geometrically Grounded Lorentz Neural Networks**, arXiv 2601.21529; **HexFormer: Hyperbolic Vision Transformer with Exponential Map Aggregation**, arXiv 2601.19849;
  **Rethinking the Fully Hyperbolic Vision Transformer in Polar Coordinates**, arXiv 2609.32899.
* Mishne et al., **The Numerical Stability of Hyperbolic Representation Learning**, ICML 2023, arXiv 2211.00181; Guo et al., **Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers**, CVPR 2022.
* **Is the Geometry Doing the Work? An Operating-Point Audit of Hierarchy in Hyperbolic Vision-Language Models**, arXiv 2607.05268.
* De Sa, Gu, Re, Sala, **Representation Tradeoffs for Hyperbolic Embeddings**, ICML 2018, arXiv 1804.03329; Ganea et al., **Hyperbolic Entailment Cones**, arXiv 1804.01882.
* Merrill, Petty, Sabharwal, **The Illusion of State in State-Space Models**, ICML 2024, arXiv 2404.08819; Grazzi et al., **Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues**, ICLR 2025, arXiv 2411.12537;
  Siems et al., **DeltaProduct**, arXiv 2502.10297; Liu et al., **Transformers Learn Shortcuts to Automata**, ICLR 2023, arXiv 2210.10749; **Rethinking State Tracking in Recurrent Models Through Error Control Dynamics**, arXiv 2605.07755.
* Qiu et al., **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free**, NeurIPS 2025, arXiv 2505.06708.
* Berglund et al., **The Reversal Curse**, arXiv 2309.12288. Belcak & Wattenhofer, **Fast Feedforward Networks**, arXiv 2308.14711; He, **Mixture of A Million Experts**, arXiv 2407.04153.
* Earlier context (not re-checked today): Su et al., RoFormer (RoPE), arXiv 2104.09864; Nanda et al., progress measures for grokking, arXiv 2301.05217; Kleyko et al., hyperdimensional computing/VSA surveys, arXiv 2111.06077 and 2112.15424;
  Gardner et al., toroidal grid-cell topology, Nature 2022; Law et al., Lorentzian distance learning, ICML 2019; Gu et al., mixed-curvature product spaces, ICLR 2019; Ganea et al., Hyperbolic Neural Networks, NeurIPS 2018;
  Dohare et al., loss of plasticity, Nature 2024; Dong et al., rank collapse in attention, 2021; Kantamneni & Tegmark, language models use trigonometry to do addition, 2025;
  Ghadimi Atigh et al., Hyperbolic Image Segmentation, CVPR 2022 (radius as uncertainty; from memory).

---------------------------------------------------------------------------------------------------------------------------
## Appendix A. Formula cheat-sheet (curvature -c, hyperboloid <x,x>_L = -1/c)

* chart: `x0 = sqrt(1/c + |x_s|^2)`; inner `<x,y>_L = -x0 y0 + x_s.y_s`; `d = acosh(-c<x,y>_L)/sqrt(c)`; squared Lorentz distance `-2/c - 2<x,y>_L`.
* exp/log at origin: `exp_0(v) = (cosh(a)/sqrt(c), sinh(a) v/(sqrt(c)|v|))`, `a = sqrt(c)|v|`; `log_0(x) = asinh(sqrt(c)|x_s|)/(sqrt(c)|x_s|) x_s`.
* Poincare <-> hyperboloid: `x0 = (1+c|u|^2)/(sqrt(c)(1-c|u|^2))`, `x_s = 2u/(1-c|u|^2)`, `u = x_s/(1+sqrt(c) x0)`.
* Lorentz centroid of weights w >= 0: `s = sum w_i x_i`, `centroid = s/(sqrt(c) sqrt(-<s,s>_L))`.
* Neuron: `<w,x>_L + b = w_s.x_s - w0 x0 + b`; type by sign of `|w_s|^2 - w0^2`.

## Appendix B. Reproduce

```
python research/yggdrasil_poc/poc_a4_tree_embedding_tuned.py                    # tuned tree embedding (GPU, ~5 min)
POC_STEPS=2000 POC_DIMS=4,8 POC_KINDS=euclid,poincare,lorentz POC_LRS=0.1,0.03,0.01 POC_OUT=poc_a5_long_training.json python research/yggdrasil_poc/poc_a4_tree_embedding_tuned.py
python research/yggdrasil_poc/poc_b_state_tracking.py tune <model> ; ... main <model> ; ... report   # models: torus so3 gru torus_lat so8 hh2
python research/yggdrasil_poc/poc_c_attention_and_neuron.py ; python research/yggdrasil_poc/poc_c2_component_timing.py ; python research/yggdrasil_poc/poc_d_training_smoke.py
python -m pytest tests/test_lorentz_attention.py tests/test_geometry*.py tests/test_gpu_kernel_bridge.py -q          # 68 passed
```
Superseded (kept as records of a mistake): `poc_a_tree_embedding.py` (untuned), `poc_a2_precision_diagnostic.py` (still valid: precision is not the cause), `poc_a3_optimisation_sensitivity.py` (the evidence for the tuning artefact).

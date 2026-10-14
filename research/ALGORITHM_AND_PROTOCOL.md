# Algorithm and experimental protocol

*Companion to `VSA_GEOMETRY_TO_INTELLIGENCE.md`. Evidence labels as there: **[M]** measured here · **[V]** published, abstract-level check · **[R]** from memory · **[H]** my hypothesis. Everything in §3–§6 is a **proposal to be tested**, not a result.*

---

## 1. What is being proposed

**APA — Algebraic Predictive Agent** (working name): a learning system whose

1. **state** is a phase vector on a flat torus (z = e^{iθ}, θ ∈ [0, 2π)^D);
2. **actions and relations** are *unitary operators* with exact inverses (phase vectors; optional non-commutative blocks);
3. **memory** is *addressed* and surprise-gated (sparse expansion + attractor retrieval), with superposed role–filler records only as small working sets;
4. **inference** includes resonator-style factorisation, initialised by a network;
5. **planning** is batched search over operator programs, with a growing library of learned macro-operators;
6. all continuous parameters (encoder, phases, controller) are trained by gradient descent with **geometry-health losses**, and everything is **measured on real latents against random baselines**.

It is the concrete form of principles P1–P9 in the main report (§4.5). Its closest published relatives: a learnable-FHRR world model with element-wise complex transitions (Chung et al. 2026 [V]), Semigroup-JEPA's rollout-consistency (2026 [V]), NVSA's neural-front-end + VSA-back-end (Hersche 2023 [V]), DreamCoder's wake/sleep library learning (Ellis 2021 [V]), and noise-assisted resonators (Karunaratne 2024 [V]). I know of no published system that combines all six.

```
 observation o ──► Encoder E_φ ──► z ∈ T^D  (unit-modulus phases)
                                   │
        goal g ───────────────────►│◄──── retrieve (Hopfield step) ◄── Episodic memory M
                                   │                                    ▲  surprise-gated writes
                   ┌───────────────┴──────────────┐                     │
                   ▼                              ▼                     │
        Controller π_ψ proposes           Resonator R: factor scene  ───┘
        operator programs                 s = Σ r_j ⊙ f_j  → (role, filler) pairs
                   │
                   ▼  batched beam (width ≤ 256: "free" on your GPU)
        ẑ_{t+h} = U_{a_{t+h-1}} ⊙ … ⊙ U_{a_t} ⊙ z_t     (O(D) per step, exact inverse = conjugate)
                   │                       ▲
                   ▼                       │ library learning (sleep): macro-operators when description length drops
             verifier / environment ───────┘  (truth channel: execution, rules, tests)
```

---

## 2. Components and shapes (default D = 2,048)

| # | Component | Definition | Notes |
|---|---|---|---|
| C1 | **Encoder** E_φ | Network → real pairs (uⱼ, vⱼ) per coordinate → z_j = (uⱼ + i vⱼ)/abs(uⱼ + i vⱼ) | The "cleanup to the torus" is the last layer. Tokens/objects: lookup + role binding. |
| C2 | **Operator library** {U_k} | Abelian: U_k = e^{iΦ_k}, Φ_k ∈ ℝ^D learned. Continuous action α: U(α) = e^{iαΩ}. Optional order-sensitive blocks: U ∈ U(b)^{D/b}, b ≥ largest relevant irrep dimension (6 for S₅) | Unitary by construction ⇒ **no separate reversibility loss**; inverse = conjugate. |
| C3 | **Transition** | z_{t+1} = U_{a_t} ⊙ z_t, then optional cleanup. Plus a small **contraction head** c_ψ for irreversible effects | Real worlds are semigroups, not groups (main report §7.4). |
| C4 | **Scene record** | s = Σ_{j≤K} r_j ⊙ f_j; query = conj(r_j) ⊙ s then cleanup | K ≤ M_max = D/(7.5·ln N) [M]. D = 2,048, N = 5,000 → ≈ 32; use 16 for margin. |
| C5 | **Memory** | (i) working: ≤ M_max bound pairs; (ii) episodic: key = sparse expansion of z (k-WTA, 2 %), value = bound record, retrieval = softmax(βKᵀz)V, write only if surprising (vigilance); (iii) semantic: circular-mean prototypes (complex mean, renormalised) | Reuses the surprise-gated write and dentate-style expansion that already exist in your repo. |
| C6 | **Resonator** R | Factor s over codebooks by iterated unbind → cleanup; learned initial guess g_ψ(s) + initial noise σ₀ + iteration noise σ₁ | Plain version: 10⁶-combination problem only 32 % at D = 4,096 [M]; noise extends capacity ≥ 50× [V]. |
| C7 | **Controller** π_ψ | Small transformer over [z_t ; goal ; retrieved memory ; step] → distribution over operators (+ continuous parameters). Search = beam of width B_w, depth H ≤ 12, score = Re⟨goal, ẑ⟩/D + log π + verifier bonus | Batched evaluation; your measured B = 1 → 256 costs 30.8 → 34.5 ms [your memory note], so width is nearly free. |

---

## 3. Objectives

Let C be the codebook, μ(C) = max_{i≠j} abs(Re⟨cᵢ,cⱼ⟩)/D, and μ₀ = √(2 ln P / D) with P = N(N−1)/2 the random baseline (theory column of E1; empirical max matches within 5 %).

| Loss | Definition | Purpose |
|---|---|---|
| L_pred | (1/H) Σ_h [1 − Re⟨z^{obs}_{t+h}, Û^{(h)} z_t⟩ / D] (multi-step) | Dynamics = unitary products |
| L_unif | log E_{i≠j} exp(−t·‖zᵢ − zⱼ‖²/D) (Wang–Isola, torus metric) | Prevents the cone-collapse you measured |
| L_coh | Σ_{i≠j} max(0, abs cos_ij − μ₀)² | Keeps atoms at least as incoherent as random |
| L_bind | Cross-entropy of cleanup(conj(r_j) ⊙ s) vs true filler, K ~ U{1…M_max} | Trains capacity and role–filler recoverability |
| L_swap | L_pred on content-swapped pairs (same action, new fillers) | Content-independent operators = systematicity (P4) |
| L_mdl | λ_ops·(Σ cost(op)) + λ_len·E[program length] | Library learning only helps if it shortens descriptions |
| L_rl | Reward-weighted / best-of-n imitation on **verified** programs only | A truth channel (P8); never closed self-play |

Euclidean parameters: AdamW or Muon (your `ishmn/muon.py`); phases are angles, so no norm constraint is needed. **Rule (from your REM-replay finding):** any gradient path that touches the operators must include L_pred.

---

## 4. Training procedure

```python
for epoch in range(E):
    # ---- WAKE: act, search, verify, learn ------------------------------------------------
    for batch in task_stream(curriculum):            # procedurally generated, checkable
        z    = E(batch.obs)                          # (B, D) phases
        prog = beam_search(z, batch.goal, pi, lib, M, width=Bw, depth=H)
        ok, reward = verifier(batch, prog)           # execution / rules / unit tests
        loss = L_pred + L_unif + L_coh(codebook) + L_bind + L_swap + L_mdl(lib) + L_rl(prog, reward)
        optimiser.step(loss)
        M.write_if_surprising(z, prog, ok)           # vigilance-gated, like your VSAEpisodicMemory
    # ---- SLEEP / NREM: compress -----------------------------------------------------------
    prototypes = circular_kmeans(M.keys)             # complex mean + renormalise
    lib.add_macros(frequent_subprograms(M.programs), accept_if=lambda m: description_length_drops(m))
    lib.prune_unused()
    # ---- SLEEP / REM: replay REAL transitions only ----------------------------------------
    for (z, a, z_next) in M.sample_real_transitions():
        optimiser.step(L_pred + L_unif + L_bind)     # operators never updated without L_pred
    # ---- AUDIT: every epoch, on REAL latents, vs random baselines -------------------------
    log(geometry_health(real_latents))               # coherence, uniformity, equivariance, capacity, composition defect
```

**Test time** (ARC-style refinement loop, P9): encode → retrieve → plan → execute → compare with the example pairs → update only a small adapter (phase deltas) on that instance → repeat for K rounds.

---

## 5. Hypotheses and what would falsify them  [H]

| # | Hypothesis | Falsified if |
|---|---|---|
| H1 | Phase-parameterised operators learn group tasks with markedly less data than an unstructured baseline | Paired CI on data-to-99 % includes 0 over ≥ 10 seeds |
| H2 | Learned phases are spectrally sparse (a few frequencies), as in grokked networks | Participation ratio ≈ D (no concentration) |
| H3 | For non-abelian tasks the abelian model fails and the block-unitary model succeeds | Abelian model solves S₅, or block-unitary does not beat it |
| H4 | Learned codebooks reach the random capacity curve (D/(M ln N) ≤ 7.5) or better | Capacity below the random curve after L_coh + L_bind |
| H5 | Unitary operators + filler-swap training give systematic OOD gains on held-out compositions | No gain over the matched baseline on any of the Phase-2 suites |
| H6 | Resonator inference extends factorisation depth beyond direct decoding | No gain when ablated |
| H7 | Library learning improves sample efficiency on held-out compositions | MDL falls but accuracy does not |
| H8 | The effect is *causal*: intervening in the binding subspace changes behaviour as predicted | Interventions have no predictable effect |

---

## 6. Experimental protocol

Standing rules (from your own history): **numeric/exact grading, never string grading; arms differ in exactly one bit; n ≥ 400 paired items per cell; McNemar exact + 95 % CI; report greedy output next to sampled output; length-matched metrics; every geometry number measured on real latents with a random baseline beside it.**

### Phase 0 — does the algebra emerge? (1–2 days, RTX 3060)

- **Tasks:** Z₁₁₃ addition (Nanda 2023), Z_p × Z_q, dihedral D₁₂, S₅ composition. Train on 5–40 % of all pairs.
- **Models:** (a) 1-layer transformer / MLP with weight decay (0.3–2 M parameters); (b) APA-lite: phase embeddings, phase-sum operator, cleanup readout, parameter count within ±10 %; (c) APA-lite with b-block unitary for S₅.
- **Metrics:** held-out accuracy vs training fraction; steps to 99 %; spectral participation ratio of learned phases; equivariance error; ≥ 10 seeds.
- **Predictions [H]:** H1 (≥ 2× data efficiency), H2 (≤ 10 dominant frequencies), H3.
- **Stop rule:** if H1 fails with ≥ 10 seeds, the phase parameterisation is not buying anything beyond weight decay at this scale; stop and diagnose before building more.

### Phase 1 — geometry health and capacity of *learned* codes (1–2 days)

- Use latents from Phase 0 and the real Layer-14 latents already harvested by `experiments/train_world_model_on_layer14.py::harvest_layer14_transitions`.
- **Metrics:** coherence vs μ₀; uniformity; bundle-recovery curve (E2-style); cross-condition generalisation and parallelism score (Bernardi 2020); distance to an equiangular tight frame (neural collapse).
- **Expectation [M-based]:** a 32-dimensional space is ≈ 8× below the D = 256 point where recovering even 4 bundled items from 5,295 first becomes reliable; expect your current latents to show near-zero superposition capacity. That is a *measurement to make*, not a claim about your checkpoint.

### Phase 2 — compositional generalisation (1–2 weeks)

- **Suites** (all procedurally generated, exactly checkable): (a) variable-binding programs with length extrapolation (Wu 2025 task); (b) SCAN/COGS-style held-out compositions; (c) relational analogies A : B :: C : D with **held-out relation families** — compare parallel transport, group binding, APA; (d) 1D-ARC and Sort-of-ARC (external VSA baselines 83.1 % and 94.5 % [V]); (e) mini-Raven.
- **Design:** factorial 2³ over {unitary operators} × {resonator} × {addressed memory}, plus an equal-parameter transformer baseline; n ≥ 400 per cell.
- **Predictions [H]:** H4–H7. Expect the gain to concentrate in (c) and (a); expect no gain on tasks without algebraic structure — that is a *useful negative control*.

### Phase 3 — world models and planning (1–2 weeks)

- Replicate the FHRR world-model result (Chung 2026: zero-shot unseen state–action pairs, 20-step rollouts, noise) on your hardware, then extend to **procedurally generated worlds with non-invertible actions**, with and without the contraction head, against MLP / transformer / SSM baselines of equal size.
- **Metrics:** zero-shot accuracy on unseen (state, action) pairs; rollout accuracy vs horizon; noise robustness; planning success under a verifier.

### Phase 4 — attach to the SLM (2–4 weeks)

1. **Probe first, change nothing:** Hyperdimensional Probe and TPR-style probes on the Layer-14 states.
2. Add the APA register file as a side memory; test entity tracking / variable binding before and after (Feng & Steinhardt style), with your existing 4-arm factorial (sandwich × scratchpad) and numeric grading, n ≥ 400.
3. H8: causal interventions on the binding subspace.

### Phase 5 — open-ended learning with a truth channel

- ARC-AGI-3-style interactive tasks or program synthesis with hidden tests; **no self-play without a verifier** (model collapse, Shumailov 2024 [V]).
- **Headline metric:** skill-acquisition efficiency (Chollet) — performance vs number of exemplars / interactions on **held-out task families**.

### Emergence ladder (what counts as "it emerged")

| Level | Evidence |
|---|---|
| 0 Representational | Probes recover role–filler structure; geometry-health metrics on real latents |
| 1 Functional | OOD / compositional generalisation beyond matched baselines |
| 2 Causal | Interventions on the algebraic subspace change behaviour as predicted |
| 3 Efficient | Skill acquired from few examples on held-out families |

Compute: Phases 0–3 use models ≤ 30 M parameters — the scale at which HRM, TRM and symbol-equivariant recurrent models report ARC/Sudoku results [V] — and fit your 12 GB GPU. A 500 M from-scratch run is not required for any phase (your own estimate: ≈ 1.1 years at 284 tok/s).

---

## 7. ISHMN audit and change list

Everything below was **read from the code or measured**; none of it was changed. Line numbers refer to the files as read on 2026-10-03.

| ID | Finding | Evidence | Suggested fix | Pri |
|---|---|---|---|---|
| **A1** | Concept atoms are not reproducible, and creating one reseeds the **global** RNG | `ishmn/neuro/cognitive_map.py:82` `torch.manual_seed(abs(hash(clean_name)) % (2**31))`; Python string `hash` is salted per process — three runs printed three different `hash('cat')` **[M]** | `hashlib.blake2b(name.encode(), digest_size=8)` → local `torch.Generator`; add a cross-process test | **P0** |
| **A2** | Geodesic distance has a 4.47 × 10⁻⁴ rad floor | `ishmn/geometry.py:79–80` clamps cosine to ±(1 − 10⁻⁷); `geodesic_distance_sphere(x, x)` returned 4.47 × 10⁻⁴ **[M]**; exact group-analogy chordal error is 3.6 × 10⁻¹⁶ **[M]** | θ = 2·asin(clamp(‖u − w‖/2, 0, 1)) — exact and stable near 0 | **P0** |
| **A3** | "FHRR ≅ T⁸ ⊂ S⁷" is dimensionally impossible | `docs/SPEC_UNIFIED_B_C_SUBSTRATE.md:57`; dim T⁸ = 8 > dim S⁷ = 7 | State T^D ⊂ S^{2D−1}(√D); ℝ⁸ = ℂ⁴ hosts T⁴ | **P0** (docs) |
| **A4** | The "complex-phasor" bus is a real Hadamard with unit-L2 Gaussian keys; unbinding multiplies by key² | `ishmn/neuro/global_workspace.py:60,65`; norm ratio 0.031 and cosine deviation 0.024 at D = 1,024; at D = 64 with 8 experts, filler identification 0.25 (MAP 0.36, FHRR 0.82) **[M]** | Keys = ±1 or unit phasors; D ≥ 256 for 8 experts (0.98 [M]); apply conjugate; add a test that unbinding recovers each message with cosine ≥ 0.9 | **P0** |
| **A5** | The bus is built and given to the optimiser but never executed | `ishmn/universal_cognitive_engine.py:244,263`; `experiments/train_500m_efficient.py:155` only counts parameters; only caller is a synthetic benchmark and a shape/NaN test; gate initialised to 0 (`global_workspace.py:37,70`) so output is exactly zero at init **[M]** | Wire it in (after A4) or remove from the parameter list | **P0** |
| **A6** | "VSA" memory is a soft-attention key–value store with 8-D keys | `ishmn/vsa_memory.py:41` (`hyperspace_dim = 8`); one slot per item, no bind/bundle | Fine as SDM/Hopfield-style memory; rename, or add a distributed-record working memory (C4–C5). Surprise-gated writes match SRMU/Titans [V] | P1 |
| **A7** | D = 8 cannot host the codebook | 5,295 random unit vectors: min angle 5.2°, cleanup fails at 40° noise; D = 1,024 → 80.1° **[M]** | Add an expansion head 32 → 2,048 phases for memory/binding (reuse `DentateGyrusPatternSeparator`, `ToroidalGridEncoder` — **verify they are on the live path**); keep S⁷ only as a low-dimensional summary | P1 |
| **A8** | Learned forward/backward maps need monitoring to stay inverse | Your 2026-09-21 notes: W_bwd was not a round-trip inverse (1.141 vs 1.008 rad) and REM replay destroyed the pair; both later recorded as fixed (residual 0.0229 rad on real latents). I did not re-measure | Phase operators make the inverse exact by construction (conjugate), so no reversibility loss or guard test is needed for that part; keep a contraction head for irreversible effects | P1 |
| **A9** | Analogy engine is local and path-dependent | Holonomy = area; PT error ≈ dist(A, C) under a true symmetry; worse than "do nothing" at far C under SO(8) relations **[M]**; the README's 4.1 × 10⁻⁷ rad badge is a norm-preservation check (not located in the repo) | Add group-binding analogies for symmetric relations; evaluate on **held-out relation families**; keep transport for local interpolation | P1 |
| **A10** | Tests are shape/smoke only (your note) | e.g. `tests/test_harvested_neuro_symbolic_gems.py:126–135` | Add geometry-health tests on real harvested latents: coherence vs baseline, uniformity, equivariance, capacity, composition defect | **P0** |
| **A11** | Search width and dimension are nearly free | Your measurement: B = 1 30.8 ms vs B = 256 34.5 ms | Spend it on beam width and on D, not on depth | info |

**Module mapping** (by reading names and docstrings; verify before wiring): `perception.py` → C1 · `memory.py`, `vsa_memory.py` → C5 · `universal_world_model.py` → C2/C3 · `spherical_dynamic_moe.py` → C7 experts · `neuro/attractor_network.py` → cleanup · `neuro/dentate_grid.py` → expansion / periodic features · `neuro/delta_net.py` → error-correcting write · `muon.py` → optimiser · `sigreg.py` → note: SIGReg targets an isotropic Gaussian, compatible with torus uniformity only at large D · `metric_validation.py`, `honest_eval_harness.py` → evaluation.

---

## 8. Risks and decision rules

| Risk | Symptom | Response |
|---|---|---|
| Phase 0 shows no benefit | H1 fails at ≥ 10 seeds | Stop. The structure is not the bottleneck at this scale; test whether weight decay alone explains grokking. |
| Capacity bottleneck | Bundle recovery degrades with K; Kumar-2026-style composition failure | Increase D; add addressing/hashing; use orthogonal subspace carving (Pence 2026) |
| Operators cannot learn the group | Phases stay diffuse; equivariance error high | Larger blocks (non-abelian); add equivariance pre-training on primitives (Ito 2022) |
| Irreversible worlds | Unitary rollout drifts | Contraction head; semigroup consistency loss (Semigroup-JEPA) |
| Evaluation leakage | Numbers too good | Held-out families only; reseeded generators; exact grading; paired tests |
| Circular self-play | Metrics rise, semantics collapse | Verifier required (P8); real-latent audits every epoch |
| Everything works at toy scale only | No transfer in Phase 4 | Report honestly; the VSA component is then a *diagnostic* (probes), not a driver |

---

## 9. What would change my mind

- **Toward VSA-centred designs:** Phase 2 gains that persist on held-out relation families with equal parameters; Phase 4 causal interventions that behave as predicted; independent replication of the FHRR world-model result beyond grid worlds.
- **Against:** Phase 0 showing weight decay alone matches the phase parameterisation; learned codebooks that cannot beat random capacity; gains that vanish under numeric grading or paired tests.
- **Decisively:** a published, reproduced demonstration that a VSA-first architecture scales to LLM-level performance on open-ended tasks (none exists to my knowledge).

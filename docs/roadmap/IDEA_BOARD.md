# Idea board: conventional, adjacent and outside-the-box options for each obstacle

Version 0.1, 2026-11-02. Part of the road plan ([ROADMAP.md](ROADMAP.md)). This is the brainstorm, kept alive. It is where we look first when a milestone fails (the "stuck protocol", ROADMAP section 6).

**In plain words.** For every obstacle between us and the finish line there is a short list of ideas. Each says what it is, why it might work, what already exists, and the cheapest experiment that would tell us.
Nothing here is a promise: **every idea is a hypothesis (grade H) until a pre-registered test says otherwise.** Ideas that were already tried and failed are listed at the end so that we do not redo them.

**How to read a row.**
*Chance* is my honest guess that the idea helps (H high, M medium, L low); *Cost* is the cheapest decisive test (S about 1 session, M 2–4, L 5 or more); a star marks the ideas that are
cheap, informative and used as early probes (P1–P3 in [MILESTONES.md](MILESTONES.md)). *Status*: idea, spike planned, spike running, kept, dropped.
Grades and tags: see the notebook README. Literature tags: L = abstract level, R = recalled ([RESEARCH_DIGEST.md](RESEARCH_DIGEST.md)).

The obstacles: **O1** routing without task labels, at scale. **O2** cheap, nuance-aware novelty detection. **O3** a representation that can change without breaking what was learned.
**O4** bounded growth and a knowledge lifecycle (dormant, merge, remove, correct). **O5** learning signals (labels, verification, unlabelled streams). **O6** efficiency. **O7** evidence (not fooling ourselves).

---

## O1. Routing without task labels, at scale

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-01 ★ | Address experts like a phone book instead of asking every one | Split each expert's key in two halves; find the best few experts from a grid of sqrt(N) by sqrt(N) candidates, or walk a tree of cluster centres with a calibrated test at each node | PEER (L); our "tier" failed because its acceptance check was too weak (E-060, F-81, Q-62) | Simulate 10,000 tiny experts on synthetic regimes; compare gate cost and routing loss with the fingerprint shortlist (needs M0 harness) | M | M | idea |
| I-02 | Teach a cheap router to imitate the careful (slow) test | Train a small linear or LDA router on the exact test's decisions; refit incrementally instead of by cross-validation | Stage-4 router recovers the whole routing loss (C-57, C-84); refit costs 2 s at 12 experts, 90% of it the softmax candidate (C-90, Q-30) | Incremental refit without cross-validation; identical decisions on recorded streams; time it | H | S–M | idea |
| I-03 | Let a **pool** of experts answer when one is not enough | Pick 1 to k experts by probability mass or uncertainty; combine their answers (average, vote with veto) | Team sized by reader probability: 0.986 correct with 1.06 experts per part (C-93); averaging helps only where twins exist (C-72); arbitration had no headroom on synthetic streams (D-21) | Re-open Q-06 on real ambiguity: overlapping domains and real paraphrase sets | M | M | idea |
| I-04 ★ | Route a whole sequence once, from its first tokens | A small encoder embeds the first N tokens; one routing decision per sequence, so there is no per-token gate cost | SmallTalk LM (L); task-level MoE (Kudugunta et al., L in notebook) | Probe P1: three tiny language models on three domains plus a prefix router; compare with one dense model at equal total parameters and tokens | M | M | spike planned (P1) |
| I-05 | Let experts vote with their own activations | Each expert computes a cheap self-score (for example the norm of an internal activation) and the best self-scores win; no separate router | Autonomy-of-Experts, arXiv 2501.13074 (L in notebook) | Function-regime stream, 10 and 30 regimes; routing loss against oracle | L | M | idea |

## O2. Cheap, nuance-aware novelty detection

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-06 ★ | A small judge handles the easy 95%; the big reader is called only when the judge is unsure | Cascade: small encoder plus calibrated head decides the confident cases; the 1.5B reader answers only the ambiguous band; average cost = c_small + p_ambiguous x c_big | Our trained head already answers 56% of arithmetic messages at 2.1% disagreement and asks 51% fewer texts (E-066); the reader costs about 50 ms per message | Probe P3: with cached reader verdicts, measure the ambiguous share and the amortised cost for several confidence bands | H | S | spike planned (P3) |
| I-07 | Train a small encoder on exactly the differences we care about | Contrastive training: positives = same meaning in new wording (typos, case, chatter, paraphrase); negatives = near-twin skills, negation flips, swapped numbers or operators; target 30M parameters or fewer | Character-aware front end failed (E-064); the reader's strength is function and wording nuance (G1, C-91); RETVec-style robustness (R) | Fine-tune MiniLM-L6 on a nuance suite; compare with the reader on the arithmetic and banking nuance metrics (milestone M3) | M | M–L | idea |
| I-08 | Read the reader's mind instead of its answer | Train a head on the reader's hidden states, not on the sentence vector | Q-73; the sentence-vector head stayed silent on news (E-066) | Head on layer-26 states of the cached reader runs; answered share and disagreement | M | M | idea |
| I-09 | Every expert carries its own "have I seen this?" detector | Per-expert density or tiny autoencoder in the expert's input space plus our conformal calibration; novelty means no expert accepts. A variant predicts the next state and watches the error | Expert Gate, SEMA, CoPE (L); our signatures already do the density part; execution-trace prediction error gave AUROC 1.000 on real code (E-056) | Extend the prediction-error check to language with a tiny per-expert next-sentence or rewrite predictor | M | M–L | idea |
| I-10 | Shake the input and see if the answer wobbles | Cheap perturbations (typo, case, whitespace, chatter); novelty or nuance when the expert's output is unstable across them | Test-time augmentation consistency (R) | Banking and arithmetic nuance suites; AUROC of instability against known/new | M | S | idea |
| I-11 | Treat surprise at several levels as the novelty signal (the predictive-coding idea you raised) | Hierarchical prediction errors from a small predictive-coding or autoencoder stack | Li, Tang and Bogacz (L); our autoencoder-descriptor proxy tied the other detectors (E-064); never tested as such | Train the stack on the stream's features; AUROC against Mahalanobis and vMF | L–M | M | idea (outside-the-box reserve) |
| I-12 | Describe an input with several "fingerprints", or with phase, not one vector (the chunk and wave ideas you raised) | Multi-vector (late-interaction), chunk-level or complex-valued fingerprints | Untested as such (E-064, D-62); ColBERT-style multi-vector (R) | Replace the single vector in the signature by 4 chunk vectors; novelty AUROC on near-twins | L–M | M | idea (outside-the-box reserve) |

## O3. A representation that can change without breaking what was learned

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-13 | Two speeds: fast formulas on top, a slowly changing encoder underneath | Update the encoder rarely (when the novelty rate is high); keep class statistics (means and covariances) and recompute the closed-form heads from them; map old statistics through a small adapter when the encoder moves | AdaGauss and FeCAM show covariances drift (L); geometry-anchored transport (arXiv 2606.25347, L, unconfirmed); Nested Learning's multi-timescale view (L) | Small encoder, five domains in sequence; compare heads recomputed from adapted statistics against retrained from scratch | M | L | idea (core of M5) |
| I-14 | Change the encoder only in directions the old knowledge does not use | Low-rank adapters with gradient projection onto the null space of earlier tasks (O-LoRA, InfLoRA family, R) | Merging and orthogonal-subspace work (L) | LoRA on a small encoder; measure drift of the stored class statistics | M | M | idea |
| I-15 ★ | Cheap richer features without training | Random projection plus nonlinearity before the closed-form head (RanPAC) | RanPAC (L) | Add the layer to the frozen features of M2 and of the existing beds; accuracy and cost | M | S | spike planned (part of M1/M2 baselines) |

## O4. Bounded growth and a knowledge lifecycle (dormant, merge, remove, correct)

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-16 ★ | **Exact removal**: delete knowledge and prove the rest is unchanged | An isolated expert is deleted outright. For closed-form heads, subtract the removed items' counts, sums and scatter from the shared statistics ("downdating"). Exact only if no other component used the removed data (router fingerprints, replay buffers, encoder), so each of those must be cleaned too | MoE unlearning papers are approximate (SEUF, GRIP; L); closed-form updates are exactly reversible in principle (hypothesis H) | Probe P2: train with and without a class; compare outputs with the class removed by downdating; maximum difference and routing agreement | H | S–M | spike planned (P2) |
| I-17 | Notice when new input contradicts old knowledge | Flag when a batch confidently disagrees with an expert's confident predictions on near-identical inputs; then supersede (new version), keep both with dates, or ask for verification | None found; related to correction of wrong facts (R) | Streams with corrected labels; detection rate, false alarms, clean replacement | M | M | idea |
| I-18 | Store experts as a mix of shared building blocks | Each expert = a sparse combination of a shared dictionary of low-rank adapters; memory per expert falls by 10-100x | MC-SMoE merge-then-compress (V in merge notes, alignment passages); LoRA dictionaries (R) | Factorise existing experts offline; accuracy against rank | M | M | idea |
| I-19 | Put sleeping experts to disk and wake them by prediction | Dormant experts to CPU RAM or disk, int8/int4; prefetch the ones the router is likely to need | Dormant/archive/revive are safe but save only 3-6% memory (C-45); MoE offloading adds latency (L) | Offload 90% of 1,000 experts; latency and accuracy | M | M | idea |
| I-20 | Merge only what is certified redundant | Non-inferiority tests on honest held-out samples; distil merged experts into shared bases | Merge study (E-034, C-67–C-72); rival tracking removed 11.2 of 12 false spawns at 15+ batches per regime (C-71); stage 5 not built | Stage 5 plan with its own pre-registration (Q-21, Q-26) | M | L | idea |

## O5. Learning signals

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-21 | Let tools check the answers | Verifier experts: execution for code, a calculator for arithmetic, regular expressions and unit tests, retrieval cross-checks for facts | Code is self-labelling (5 of 5 skills at 0.59 of the human labels, E-057); language is not | Add a retrieval-based verifier for facts and measure label yield and purity | M | M | idea |
| I-22 | Find the skills in an unlabelled stream | Cluster the stream; ownership by predictive entropy or by the vote among admitted experts | Q-18, Q-35 (planned) | Unlabelled variants of the existing streams | M | M–L | idea |
| I-23 | Update only what the new input actually touches | Surprise-gated sparse updates: rank memory slots or adapter weights by how unusually they are used on this batch and update only the top few | Sparse memory finetuning (L); our skip rule (E-063) is the "do not train on what is known" half | Replicate at small scale on adapters of a small language model; forgetting against full fine-tuning | M | M–L | idea (core of M5) |

## O6. Efficiency

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-24 | Try the cheap expert first | Early-exit cascade of experts, escalate when the conformal confidence is low | Skip rule fires on easy streams at no accuracy cost but saves at most 7.5% of a run (E-063) | Function regimes and text beds; cost against accuracy | M | S–M | idea |
| I-25 | Run experts in groups | Group a batch by chosen expert so each expert does one dense matrix multiply | Standard MoE serving practice (R) | Throughput of grouped against per-sample routing on the pool | H | S | idea |
| I-26 | Store and run experts in 8 or 4 bits | Quantise experts and the gate's stored vectors | Standard (R) | Accuracy against bits on the existing beds | M | M | idea |
| I-27 | Skip the gate when the answer is obvious | Cheaper gate path for batches that clearly belong to one expert, with identical decisions | Q-68 (training is 7-25% of observe time, the gate is the rest) | Profile by stage; prove identical decisions on recorded streams | M | S–M | idea |

## O7. Evidence (not fooling ourselves)

| ID | Idea in plain words | Mechanism (technical) | Prior art / our evidence | Cheapest decisive test | Chance | Cost | Status |
|---|---|---|---|---|---|---|---|
| I-28 | A measurement harness that checks itself | Counters validated against analytic costs of toy models; repeat variance reported; energy through NVML | Zeus (L); our counters are partly validated | Milestone M0 | H | M | in plan (M0) |
| I-29 | A benchmark built to break us | Ordered, interleaved, drifting, near-twin and noisy-label streams; no task leakage in the text | FLEX critique (L); open questions Q-08, Q-11, Q-17 | Build once, keep forever | H | M | in plan (M4) |
| I-30 | A one-command reproduction kit | Pinned environment, one script per claim, checksums | Notebook manifest and `verify` already exist | Clean-clone run on another machine | H | M | in plan (M0, M8) |

---

## Tried and failed: do not redo without a new reason

| Idea | What happened | Where |
|---|---|---|
| Character-aware front end for typos | Failed (2 of 5 predictions) | E-064 |
| Dynamic reader cards for newborn experts | Lowered new-skill accuracy (0.395 to 0.255) | E-065 |
| A taller system of expert groups (the "tier") as built | Hurt accuracy; the acceptance check was too weak | E-060, F-81, C-162 |
| Smarter per-sample arbitration (confidence, entropy, soft mixtures) | No headroom on synthetic streams; confidence loses up to 6.9 points | D-21, C-35 |
| Label-aware assignment for overlapping regimes | Refuted (+0.5 to +2.8) | C-28 |
| Calibrating the novelty cut on teaching wordings only | Calls 16-96% of known skills in new wordings "new" | C-127 |
| A sphere of dimension 8 as a carrier of hyperdimensional (VSA) codes | Cannot host quasi-orthogonality for 5,295 concepts | C-05 |
| Hyperbolic attention heads | No improvement after the temperature control | C-07, D-16 |
| Look-based novelty to separate near-twin skills | Finds only 13-22%; kill rule fired | E-055, D-45 |
| Simply adding capacity to an adequate expert | Gives nothing; the idle growth rule | C-58, C-158 |

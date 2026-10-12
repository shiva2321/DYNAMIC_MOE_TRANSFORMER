# Routing for the dynamic MoE: what we have (D: and E:), what the world knows, what I measured, what to build

Date 2026-10-09 (evening). Written by Claude for you. Plain words first, the precise term once. Tags: **[M]** measured by me today (dev seeds 100-102 unless a row says otherwise; exploratory, NOT pre-registered), **[C]** I read the code, **[L]** literature at abstract level (I did not read full papers), **[R]** recalled, **[H]** hypothesis.

Companion files: `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` (Appendix D = the M0 + M1 verdicts), `research/growmoe_results/m0/` (all scripts and logs named below), `E:\universal_substrait`, `E:\unified`.

---

## Update 2026-10-10: routing-v2 spikes (cheaper gate, overlapping data). Direction only, NOT pre-registered

Dev seeds 100-102, patched copies of `PhyloMoE.observe` made at run time (no product file touched). Every number is script-generated in `research/growmoe_results/rv2/RESULTS_RV2_SPIKES.md`.

- **Cheaper gate [M]:** one average vector ("centroid") per expert as a first stage, then the exact conformal test on the top-2 experts only: F-15 46.1 % vs 46.0 % exact at 7.4x fewer multiply-adds, F-30 45.3 % vs 45.2 % at 14.0x, S2-30 88.0 % vs 88.6 % (9.5x; 8 sub-centroids per expert 88.5 %, 7.5x). The saving grows with the number of experts. Random Fourier "wave" fingerprints and a ridge router on them are worse and costlier than the plain centroid. S2-5 and S1 have 3-4 experts, so nothing to save there.
- **Overlapping regimes [M]:** accuracy falls 50.5 / 33.7 / 18.4 / 16.8 % at centre cosine 0 / 0.3 / 0.6 / 0.85 while oracle-routed fresh experts stay at 48 %, and the regimes stay separable from x alone (nearest centroid 100 / 100 / 99.9 / 90.1 %). At cosine 0.3 the cause is TRAINING contamination (training purity 78 %); letting a batch train the one expert most of its samples were routed to ("batch vote", share >= 50 %) gives 99.5 % purity and 47.2 % accuracy, and does no harm on F-15 (46.0 -> 50.4), S1 (78.2 -> 79.1), S2-5 (identical) or 50/50 mixed batches (30.7 -> 36.2). At cosine >= 0.6 the regimes merge because the per-sample novelty test is blind (0.8 % rejection of foreign inputs at 0.85) while a batch-level fingerprint test detects 100 % of foreign batches with 0 % false alarms in isolation; my first version of that test inside the live pool over-fired (72-131 experts) because its null was not calibrated, so that part is OPEN. Using labels as extra assignment evidence gave only +1.6 / +2.8 points at cosine 0.3.
- **Status (2026-10-12):** stage 1 (batch-consistent assignment) was pre-registered, run on fresh seeds 20-39 and PASSED all 11 statements; it is ON by default (report Appendix F, component card 14). Stage 2 (fingerprint shortlist, cheaper gate) is specified and planned but not built; stage 3 (calibrated batch-level novelty for centre cosine >= 0.6) is open behind a spike. Spec `docs/superpowers/specs/2026-10-10-routing-v2-design.md`, plan `docs/superpowers/plans/2026-10-10-routing-v2.md`.

---

## Update: option A was run (same evening) and passed every pre-registered statement

You approved option A; it was implemented test-first and tested on fresh seeds 10-19 under `research/growmoe_results/m2a/PREREG.md` (config locked by hash before the first seed ran). Full tables: `research/growmoe_results/m2a/RESULTS_M2A.md`.

| ID | Rule | Verdict | Expected | Numbers |
|---|---|---|---|---|
| P1 | F-15 and F-30: mean paired A_new - B2* >= +5.0 points and the 95% interval excludes 0 (the B1 rule) | **PASS** | PASS (as expected) | (A_new - B2*) F-15: +10.0 pts [+8.0, +12.0] (n=10); F-30: +11.6 pts [+9.5, +13.8] (n=10) |
| P2 | F-15 and F-30: mean paired A_new - A_old >= +3.0 points (the repair itself moves the pool) | **PASS** | PASS (as expected) | (A_new - A_old) F-15: +7.6 pts [+5.1, +10.1] (n=10); F-30: +7.2 pts [+4.9, +9.4] (n=10) |
| G1 | S1: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) +3.9 pts [+0.6, +7.2] (n=10) |
| G2 | S2-5: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) +1.0 pts [-0.8, +2.7] (n=10) |
| G3 | S2-30: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) +2.3 pts [-1.5, +6.1] (n=10) |
| G4 | S4: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) -0.4 pts [-4.3, +3.6] (n=10) |
| G5 | S7: A_new raises at least 3 alarms per run on average (the alarm still catches a label remap) | **PASS** | PASS (as expected) | mean alarms per run 4.10 (n=10) |

**In plain words.** With the repair (the monitor armed only after infancy) and the budget re-tuned to 5 steps per batch, the pool beats the best single network by +10.0 points at 15 regimes and +11.6 at 30 (bar +5, intervals well above 0); the M0 frozen pool led it by only +4.1 / +2.6. The repair alone is worth +7.6 / +7.2 points. On the guard streams nothing broke: S1 improved (+3.9), S2-5, S2-30 and S4 are within noise, and the alarm still fires on label remaps (4.1 per run on a 4-regime remap stream). The repaired pool reaches 46.7 % against the oracle's 47.6 % at 15 regimes (the difference is within noise) and 42.6 % against 46.8 % at 30 (4.2 points short).

**What did NOT change.** The M0 verdict (B1 FAIL for the frozen pool) stands as written; this is a separate follow-up whose repair was found after M0. The efficiency half of the bet is still unproven:

| Stream | Method | Memory KB | Predict kMACs | Gate share of MACs | Experts | Seconds per run |
|---|---|---|---|---|---|---|
| F-15 | A_old | 5477 | 490 | 93% | 14.3 | 25 |
| F-15 | A_new | 5863 | 525 | 94% | 15.0 | 7 |
| F-15 | B2* | 8215 | 1050 | - | 1.0 | 36 |
| F-15 | B4 | 2025 | 34 | - | 15.0 | 1 |
| F-15 | kNN-all | 6800 | 1741 | 100% | 1.0 | 0 |
| F-30 | A_old | 10414 | 897 | 96% | 27.4 | 55 |
| F-30 | A_new | 11555 | 1000 | 97% | 29.7 | 18 |
| F-30 | B2* | 8215 | 1050 | - | 1.0 | 71 |
| F-30 | B4 | 4051 | 34 | - | 30.0 | 2 |
| F-30 | kNN-all | 12800 | 3277 | 100% | 1.0 | 1 |

At 30 regimes the repaired pool uses 1.4x the memory and 0.95x the multiply-adds of the best single network, and 97 % of its compute is the gate scan. Oracle experts need 34 kMACs. That is the case for routing v2 (cheap gate, ambiguity deferral, two-signal arbitration).

**Caveats.** One synthetic stream family for the accuracy claim; wide intervals on three guards (S2-30 [-1.5, +6.1], S4 [-4.3, +3.6], S2-5 [-0.8, +2.7]) mean small regressions cannot be excluded; the S1 budget (125 steps) was picked on three noisy dev seeds. The flag is now **on by default** in the code (your decision, 2026-10-10: it helped, so it stays on). Phase 1 and M0 entry points pin it off so their recorded results still reproduce; see `docs/COMPONENT_CARDS.md` card 13.

---

## 0. The one-page answer

**You are right that "who learns what, and when" is where the pool loses accuracy. You are not quite right that the gate (the part that picks the expert) is the weak spot.** On the function streams the gate is good: 92-99% of held-out inputs reach the expert that owns their regime [M, fresh seeds 200-201] and, during training, 99%+ of a regime's samples reach its owner after the first two batches [M]. What costs accuracy is a **lifecycle bug that sits next to the router**: the competence monitor ("alarm") is armed from the loss of an *untrained* newborn expert. After the newborn's first batch of training its held-out loss rises (2.5 to 4.3), the monitor reads that as concept drift, raises an alarm on *every* new regime, and the alarm path swaps the newborn's weights for a 60-step fine-tune on 32 samples. `growmoe/competence.py` itself warns: "arm the monitor only after the expert's loss has plateaued"; `phylo.py` does not.

What that bug costs, on the function dev stream F-15 (final accuracy, 3 dev seeds) [M]:

| Variant | F-15 dev accuracy | Note |
|---|---|---|
| A exactly as run in M0 (50 steps per batch) | 40.1 % | alarms 13 per run, about one per regime |
| A, 5 steps per batch | 39.2 % | alarms 12.7 per run |
| A, 5 steps, alarms off | 46.3 % | 14 experts for 15 regimes |
| A, 5 steps, alarms off, never "extend" | 47.6 % | exactly 15 experts for 15 regimes |
| **A, 5 steps, monitor armed only after infancy (one 10-line change)** | **46.8 %** | alarms 1.3 per run; S1 74.1 vs 76.0, S2-5 91.7 vs 91.9 |
| Oracle gating with A's own expert recipe, 5 steps | 52.7 % | the ceiling for this expert recipe |
| Best single network (B2\*, M0 dev) | 37.1 % | |

So with a small fix and the budget re-tuned to 5 steps the pool would lead the best single network by about +9 to +10 points on dev, against +4.1 / +2.6 on test seeds in M0. **That is a dev-seed result found after the M0 verdict; it is a hypothesis, not a finding.** It needs one fresh pre-registered test (seeds 10-19) before anyone may claim it. The M0 verdict (B1 FAIL for the frozen pool) stands as written.

"Better routing" still matters, for four reasons that do not show up in the F-15 accuracy: the gate is 83-96% of the pool's compute (M0), the first two batches of each regime leak 6-26% of their samples into older experts [M], overlapping regimes (S1) and label remaps (limitation L3) are not solved by density alone, and everything here is built for tens of experts, not hundreds. Section 6 proposes how to attack them in that order.

---

## 1. What "routing" has to do (four jobs, four different failure modes)

1. **Detect**: is this input covered by some expert, or new? (novelty / out-of-distribution test)
2. **Assign for learning**: which expert is allowed to change because of this sample?
3. **Select at test time**: which expert answers?
4. **Lifecycle**: when to spawn, extend, merge, retire, and which signals may trigger that.

Theory frames how much each can hurt: class-incremental accuracy is bounded by within-task prediction (the expert) times task-identity prediction (the routing), and task-identity prediction is an out-of-distribution detection problem [L: Kim et al., NeurIPS 2022, https://arxiv.org/abs/2211.02633]. Our measured numbers: routing about 0.92-0.99, expert quality is the rest. A near-perfect router cannot rescue weak experts, and weak experts here are produced by job 4.

---

## 2. What we already have (inventory)

### 2.1 In this project (D:)

| Part | Where | What it does for routing | Status |
|---|---|---|---|
| vMF-KDE signature + conformal p-values + Benjamini-Hochberg | `growmoe/signature.py`, `conformal.py`, `pools/phylo.py` | Jobs 1 and 3. Gradient-free gate: no router weights, so no router drift by construction. | Works: novelty AUROC 0.98-1.00, FNR about 0 [C, Phase 1]; test routing 92-99% on F-15/F-30 [M] |
| `accept_then_logf` and other arbitration rules | `phylo.py`, `research/growmoe_results/r11d/` | Job 3. +13.9 points on S1 over `default` (test seeds). | Proposed rule; limitation L3: with remapped labels it sends only about 50% of inputs to the successor [C] |
| Competence monitor (one-sided CUSUM on prequential loss) | `growmoe/competence.py`, `phylo.py` lines 717-751 | Job 4. Meant for concept drift in covered inputs. | **Defect: armed at birth (section 4)** |
| Probation buffer, evidence test, extend-vs-spawn probe | `probation.py`, `phylo.py::_handle_cluster` | Jobs 2 and 4. | Probation is not the problem [M]; the probe is fooled by memorisation on a 256-sample replay (hypothesis [H]) |
| Key-gated delta experts (Pool C) | `pools/keyed.py` | Cosine key gate with z-score novelty. | Parameter count was inflated 2.68x; less accurate than A (76.0 vs 91.0) [M, M0] |
| Bayesian non-parametric gate (Pool B) | `pools/bnp.py` | DP-mixture prior times generative likelihood. | Phase 1 only |
| Learned-router baseline, DP-means, oracle | `baselines.py` | Reference points (B3, B5, B4). | B4 oracle: +11 to +16 points over the best single net [M, M0] |
| Geometry results | `docs/COMPONENT_CARDS.md` | Hyperbolic/tree retrieval: branch-and-bound was 36-60x SLOWER than dense GPU scan at N <= 4096. Hyperbolic attention: no benefit over tempered Euclidean. | Lesson: do not build routing trees in Python for small E |
| Memory geometry | `geomlab/memory_geometry.py` | Associative recall per byte (self-dual cones, unit-sphere delta rule). | Relevant to the episodic/associative part, not to the gate |
| ISHMN dynamic MoE | `geometry/dynamic_moe.py`, `ishmn/` | Original routing in the monolithic system. | Frozen / unverified; do not touch |

### 2.2 In `E:\universal_substrait` ("Dynamic MoE Transformer"), a language-model system you built earlier

It routes **tokens** inside a transformer with: complex-phasor addresses (each expert owns one learnable key in C^2048, gate = Hermitian cosine, "resonance"), dynamic top-k from routing entropy, Gram-Schmidt orthogonalised keys for newborn experts, a fading exploration bonus for newborns, a token-sorted batched dispatch (one GEMM per expert; 236 to 4,363 tokens/s at 32 experts, bit-identical gradients), two-compartment "dendritic" experts, a global bus, sleep consolidation, a k-winner-take-all "dentate gyrus" pattern separator, a Hopfield clean-up memory, and a resonator network [C].

What its own pre-registered campaign established [C, `experiments/campaign_10d_v2_10seed/v2_continual_learning_results.md`]:

* The primary hypothesis (aggregate backward transfer better than a static MoE) **failed**, including on a clean replication cohort (p = 0.9908; pooled p = 0.3915).
* Final-accuracy retention improved on 8 of 10 domains (5 of 10 after Bonferroni), every delta positive; `github_code` +4.36 points (d = 3.87). V2 *redistributes* loss drift (2 domains better, 2 worse) and costs 28.6% throughput.
* Routing is **not** one-expert-per-domain: one or two generalist experts carry 30-95% of traffic; domain-specific routing is a thin layer on top (paper section 5.5).
* Lessons that transfer to us (paper section 7): the **cold-start monopoly** (old experts win every routing contest), the **newborn disadvantage** (a newborn's key is one batch's centroid, an old key has been sharpened for hundreds of steps), the fix (a fading newborn bonus) and its side effect (it hijacked replay traffic, so it must be off for replay), and a drift guard that made retention monotonically worse.

Reusable for routing: (a) one key per expert makes the gate O(E x D) instead of our O(E x 256 x d) [the cost fix]; (b) exact orthogonalisation of a newborn's key (note its own finding that the final normalisation re-introduces overlap, fixed by 3 Jacobi passes); (c) the token-sorted dispatch if we ever route tokens; (d) the k-WTA expansion as the front end of a bio-inspired router (section 3). Not reusable as is: learned keys (a parametric router must itself be protected from forgetting), the criticality temperature controller (no measured benefit), the drift guard (measured harm).

### 2.3 In `E:\unified` and `E:\AI_Research` (the swarm's clean package)

| Part | Where | Note |
|---|---|---|
| `VSARouter`: gate-free prototype routing, a few sub-prototypes per domain grown online, EMA updates, z-score drift guard, "conscience" load balancing, confidence margin (best minus runner-up), soft top-K, `merge_domains` | `unified/substrate/vsa_router.py` | Docstring claims 84.3% class-incremental accuracy on Split-MNIST without task boundaries [C: claim, not verified by me]. Cheap (a handful of prototypes). |
| `FamiliarityGate`: split-conformal calibration of leave-one-out nearest-neighbour cosine distance | `unified/reasoning/familiarity.py` | The same idea as our conformal gate, arrived at independently. |
| `EpistemicRouter`: cascade exact lookup, then multi-hop inference, then UNKNOWN / LEARNING, with calibrated confidence | `unified/reasoning/epistemic.py` | A **cost cascade**: cheap tier first, expensive tier only on a miss. This is the shape the whole system needs (memory, then experts, then abstain). |
| Dynamic VSA-expert MoE (O(K x P) instead of O(R x P)) | `unified/dynamic_moe.py` | Relation clusters as experts. |
| Modern Hopfield, pole bank, phasor VSA, resonator, triple store | `unified/memory`, `unified/substrate` | Associative memory pieces for the episodic tier. |

`E:\universal-ai-memory` ("Hive", a TypeScript memory graph over your AI conversations) is relevant to the episodic-memory thread, not to routing.

---

## 3. What the internet says (2024-2026, abstract level [L])

**Router drift is a recognised cause of forgetting, separate from expert drift.** SAME splits MoE forgetting into router drift (old inputs sent to the wrong experts) and expert drift, and shows that retraining only the router does not recover old accuracy [https://alphaxiv.org/abs/2602.01990]. LLaVA-DyMoE keeps old experts frozen and still sees old-task tokens pulled toward newly added experts; its fix keeps *ambiguous* tokens (similar affinity to old and new experts) away from the new experts [https://arxiv.org/abs/2603.27481]. Spectral MoE uses a dual router with an offline memory of historical expert importance [https://openaccess.thecvf.com/content/CVPR2026/html/Yin_Spectral_Mixture-of-Experts_for_Continual_Learning_CVPR_2026_paper.html]. Another February 2026 study finds MoE transformers forget substantially even with balanced sparse use [https://arxiv.org/abs/2602.12587]. **Our gate has no trained router weights, so it has no router drift of the kind these papers describe (a signature only changes when its own expert receives data); the lesson that transfers is the ambiguous-token rule.**

**Gradient-free routers are a live direction.** Latent-LoRA fits a Gaussian mixture on frozen embeddings, argues that a stationary feature space makes the router immune to drift, and reports near-zero forgetting [https://arxiv.org/abs/2607.23837]. Expert Gate uses one autoencoder per task as the gate [https://arxiv.org/abs/1611.06194]. Arrow derives routing prototypes from the SVD of each expert's own weights, with no data [https://arxiv.org/abs/2405.11157]. FlyPrompt and the 2026 brain-inspired hierarchical-modularity paper route with a **random sparse expansion (fruit-fly olfactory circuit) plus a closed-form ridge-regression router** [https://arxiv.org/abs/2602.01976, https://arxiv.org/abs/2609.25146]; the same circuit is the classic FlyHash / BioHash locality-sensitive code [https://www.biorxiv.org/content/10.1101/180471v1, https://arxiv.org/abs/2001.04907]. Our `E:` dentate-gyrus module is the same idea.

**Evidence that routing looks solved only on short benchmarks.** FLEX shows routing is "saturated" on 4-10-task benchmarks because of textual fingerprints; on a long, fingerprint-reduced benchmark the best routers recover only 28-50% of the oracle gap, and **calibrated soft mixtures beat hard selection** [https://arxiv.org/abs/2608.01437]. (Our pool realises 16-37% of the oracle gain at 15-30 regimes [M, M0].) A bi-level routing MoE scales to 300+ tasks [https://arxiv.org/abs/2602.03473]. CRAM routes by task centroids [https://arxiv.org/abs/2606.02502].

**Loss-based gating and its caveat.** Hierarchically Gated Experts gate by a loss threshold with an EWMA and a buffer to tell "outlier inside the task", "sample from another task" and "temporary training instability" apart [https://arxiv.org/abs/2412.17188]. That disambiguation stage is exactly what our alarm path lacks.

**Test-time selection beyond input density.** Confidence-based task-id prediction picks the expert whose own logits look most confident after calibration (CTP, in the survey of TP methods); TOOD scores an input by the maximum over per-task energy channels and shows detection tolerates a finer-than-true grouping [https://arxiv.org/abs/2607.29592]. The motor-control literature does the same with prediction error as "responsibility" (MOSAIC) [https://papers.neurips.cc/paper_files/paper/1998/file/228499b55310264a8ea0e27b6e7c6ab6-Paper.pdf].

**Self-routing.** Autonomy-of-Experts lets experts rank themselves by the norm of their own internal activation (ICML 2025) [https://arxiv.org/abs/2501.13074]; Routing-Free MoE and Self-Routing remove the router module [https://arxiv.org/abs/2604.00801, https://arxiv.org/abs/2604.00421]. Relevant to your norm-gated neuron idea; costly for us because every expert would have to run its first layer on every input.

**Cheap gates.** Product keys give O(sqrt(N)) retrieval (PEER, one million experts) [https://arxiv.org/abs/2407.04153]; fast feedforward networks route down a binary tree in logarithmic time [https://arxiv.org/abs/2308.14711]. Caveat from our own geometry work: Python tree traversal lost 36-60x to a dense GPU scan at N <= 4096. Fixed or hash routing is competitive in some language-model settings but learned routing pulls ahead as experts multiply (mixed evidence, vendor blog and single-seed ablations) [L].

**Load balancing** (DeepSeek's loss-free bias) is for fixed pools [https://arxiv.org/abs/2408.15664]; in a growing pool the right maintenance is lifecycle, not equalisation [H, my Oct-4 design].

---

## 4. What I measured (all exploratory; scripts in `research/growmoe_results/m0/`)

* **D1, routing accuracy** (`diagnose_extras.py routing`, fresh seeds 200-201): 92.5-99.3% of held-out inputs reach the owning expert; 13-16 experts for 15 regimes, 28-29 for 30; yet only 7-20% of regimes are within 3 points of the oracle's accuracy.
* **D3a, training-time purity** (`diagnose_routing_causes.py`, F-15 dev seeds 100 and 101): 98.8% and 95.5% of a regime's samples reach its owner. By batch index: batch 0 only 92.6% / 72.5% own (5.8% / 25.7% to older experts), batch 1 93.1% / 77.8%, from batch 2 on at least 99%. 1 and 3 regimes ended up sharing an expert.
* **D3b, is it the expert recipe?** Oracle gating with A's own `train_expert` (replay mixing): 52.7% at 5 steps, 49.1% at 20, 47.2% at 50 (A: 40.1%). Plain oracle experts: 48.5% at 5 steps, 37.6% at 50. **Replay mixing helps experts by 4-10 points, so A's recipe is not the problem; also, M0's oracle B4 was understated.**
* **D4 / D5, which mechanism?** (`diagnose_machinery.py`, `diagnose_machinery2.py`) Probation on or off: no difference. Alarms off: +7 points at 5 steps (46.3%). Alarms off and never extend: 47.6% with exactly one expert per regime. Alarms on with never extend: 42.5% with 27.7 experts (duplicates). More steps hurt once the alarms are gone (20 steps 43.7%, 50 steps 41.6%): the 50-step budget M0 tuned for A was chosen inside the faulty machinery. Side effects of switching alarms off: S2-5 91.9 to 90.9, **S1 76.0 to 70.0 (forgetting doubles)**, so the alarms are useful on overlapping regimes and cannot simply be deleted.
* **Mechanism probe** (`scratchpad/probe_alarm_mechanism.py`, seed 100): every regime onset shows `spawn` at batch 0 and, at batch 1, an `alarm` and an `extend` **on the newborn expert itself** (5 of 5 onsets, both seeds). For regime 1, baseline loss 2.47 +- 0.97 (untrained net), held-out loss after batch 0 at 50 steps 4.27, CUSUM statistic far above its threshold 23.
* **D6, the minimal fix** (`diagnose_arming.py`): keep each infant's monitor reset until infancy is over. F-15: 5 steps 46.8% (A: 39.2%), alarms 12.7 to 1.3 per run, 15.3 experts; 20 steps 41.7%, 50 steps 40.1%. S1 74.1 vs 76.0 (noisy: 74.9 / 69.8 / 77.6 vs 76.4 / 74.9 / 76.8); S2-5 91.7 vs 91.9.

What is still unexplained: about 6 points between 46.8% and the oracle-gated 52.7%. Candidates [H]: the leaky first two batches, 1 extend per run that merges two regimes, test-time misrouting of 1-8% of inputs, and no training of the newborn on its own birth cluster at spawn.

---

## 5. What this means for the design

1. **Do not replace the gate. Fix the lifecycle logic around it.** The gate is calibrated, gradient-free and 92-99% accurate on separable regimes. Replacing it with learned phasor keys would add a parametric component that itself needs protection (the Latent-LoRA argument), and `E:`'s own campaign did not show a routing benefit.
2. **One evidence source per decision, shared state between them.** Novelty (fast, calibrated) decides spawning; the competence signal (slow, loss-based) may only speak about experts that are *settled* and about samples the gate accepted with clear evidence. The two must share state so one event cannot fire both.
3. **Ambiguous samples neither train nor vote.** Samples whose best p-value sits between q and a clear-acceptance level (say 0.2) are deferred to probation for a short window (LLaVA-DyMoE's token rule, applied to our gate). That targets the 6-26% leakage in batches 0 and 1.
4. **Arbitrate with two signals at test time.** Input density (what we have) times the expert's own calibrated confidence margin, which is also the missing "label evidence" for remaps (L3) and the likely lever on S1; a soft top-2 mixture only when the margin is small (FLEX: calibrated soft beats hard).
5. **Make the gate cheap without losing calibration.** Few prototypes per expert (8-32) for the density, while the conformal calibration scalars come from a separate larger buffer of scalars (split conformal). Today 32 stored scores cannot flag novelty at q = 0.02 (floor 1/(n+1)), which is why the prototype count and the calibration count are welded together. Keep it dense and vectorised (the repo's own result); two-stage centroid-then-exact only if E grows past a few hundred.
6. **Retune the budget after the fix** (5 steps on dev), because the old optimum was an artefact of the fault.

---

## 6. Options

**A. Fix and re-test (recommended first).** (i) Arm the monitor only after infancy (a rule the module's own docstring asks for); (ii) re-tune steps on dev; (iii) lock it; (iv) one pre-registered confirmatory test on fresh seeds 10-19 with the same B1 rule (+5 points, interval above zero) plus S1, S2-5, S2-30 and a label-remap stream (S7) as "must not regress" guards. Cost: about 1.5 hours of compute and a half-page pre-registration. Kill rule: B1 fails again, stop claiming an accuracy advantage.

**B. Routing v2 as designed in section 5 (items 3-5), built test-first after A.** Ambiguity deferral, two-signal arbitration, soft top-2, cheap gate with split-conformal calibration, newborn trained on its birth cluster. Each item is its own small experiment with its own guard streams. Larger, but each step is measurable on its own.

**C. Try an analytic random-expansion router (FlyPrompt style) as an alternative selector** on overlapping-regime streams (S1): sparse random expansion (our `E:` dentate-gyrus module) plus a closed-form ridge router trained on the pool's own assignments. Gradient-free, exact incremental updates, no forgetting by construction, O(D x E) cost. Risk: it needs assignment labels, which come from the same pool (circular unless the conformal detector stays the first stage).

**Not recommended:** swapping in the phasor learned-key gate from `E:\universal_substrait` for the classification pool. Its ideas (orthogonal keys, newborn bonus excluded for replay, batched dispatch) are worth borrowing; the gate itself would trade a non-parametric, drift-free design for a parametric one without measured gain.

---

## 7. Decisions for you

1. Start with option A now (my recommendation), with B designed alongside it?
2. For the confirmatory test, accept the fresh seeds 10-19 and the "must not regress" guard streams (S1, S2-5, S2-30, S7)?
3. Keep the M0 verdict as written (B1 FAIL for the frozen pool) and report the new result as a separate, clearly labelled follow-up?

---

## Appendix. Files

Scripts and logs in `research/growmoe_results/m0/`: `diagnose_extras.py` (speed, routing D1, ablate D2), `diagnose_routing_causes.py` (D3a, D3b), `diagnose_machinery.py` (D4), `diagnose_machinery2.py` (D5), `diagnose_arming.py` (D6), with `*_log.txt` and `diagnostics_D*.json`. Mechanism probe: the session scratchpad (`probe_alarm_mechanism.py`, output quoted in section 4). All use dev seeds 100-102 except D1 (200-201); none feeds a verdict.

Sources are inline above; every [L] item was read at abstract or summary level only.

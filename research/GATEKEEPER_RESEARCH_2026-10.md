# A gatekeeper of our own: research, measurements and design (2026-10-12)

Written by Claude for you. Plain words first, the precise term once. Tags: **[M]** measured by us (dev seeds 100-102 unless stated; exploratory, NOT pre-registered), **[L]** literature read at abstract level, **[L+]** literature where I read the method section of the full text (HTML) but the page was truncated, so read the paper before relying on a number, **[R]** recalled, **[H]** hypothesis. Companion files: `research/ROUTING_RESEARCH_2026-10.md` (the earlier survey), `research/growmoe_results/rv2/RESULTS_RV2_SPIKES.md`, the notebook entries E-017 to E-023.

---

## 0. The short answer

**What we had.** A gate that looks at one input at a time and is excellent when regimes are far apart. When regimes overlap it cannot tell them apart, so the pool merges them (OV-0.6: 25.1 % accuracy where oracle-routed experts reach about 48 %), and its scan of every expert costs 97 % of a prediction.

**What the field offers.** Growth by shift detection (SEMA, MaRS), routers that need training (FLEX: they recover 28-50 % of the oracle gap), and a cheap shortlist-then-exact-score pattern (AIR-MoE). Nobody combines growth, finite-sample error control and an exactness-preserving shortlist; that combination is what we can build.

**What we measured** (3 dev seeds, synthetic streams, nothing pre-registered yet):
1. A test on the whole batch sees what the single-input test cannot. In a clean setting a permutation test on the batch's average position is exact (false alarms at the nominal rate) and catches every foreign batch even at centre cosine 0.85, where the single-input test flags 1.6 % of inputs, and down to batches of 8 inputs.
2. Inside the running pool it failed in several different ways, each fixable, and the fixes are the design: the reference must be one window of recent training data **per source** an expert has learned; the experts to test are the ones whose window mean is **nearest** to the batch, not the single-input gate's favourites; an admitted batch goes to the expert that already explains its **labels** best.
3. With that design: OV-0.6 25.1 -> 42.9 % (+17.8), OV-0.85 24.7 -> 30.0 % (+5.3); the other seven streams move by at most 0.5; known batches wrongly called new: 3 of 2,769.
4. At test time, answering a group by the **majority vote of its inputs' own choices**, with a per-input veto, adds +11.6 at OV-0.85 and +1.8 on S1 (+4.1 without the veto) and costs nothing on pure groups elsewhere. Together: OV-0.85 24.7 -> 41.6 %, OV-0.6 25.1 -> 43.9 %.
5. Smarter single-input arbitration (confidence, entropy, soft mixtures) gives nothing; we will not build it.
6. Batches sorted by label break the assumption the test rests on (the whole pool already collapses there). A circuit breaker that pauses the batch level when too many "new" calls hit familiar-looking batches cuts the excess experts from +63 to +9 on F-15 and never fires on the normal streams we tried.

**What we propose.** Gatekeeper v3, a two-level gate: the fingerprint shortlist (stage 2, built and being run), the batch test while learning (stage 3a) and group answering (stage 3b), both new levels off by default until a pre-registered run on fresh seeds. Section 6 has the design, section 7 the test plan.

**What is not proven.** The live pool has only been tested on synthetic, mean-separated streams. A first look at real text (20 Newsgroups, AG News, banking77 with frozen MiniLM embeddings, section 5.9) says the premise holds and that the single-input gate is even blinder there; the live pool on a real stream is untested, and real data may be harder (MMD is the hedge and matched the distance of means on five of six streams). The batch test assumes each batch comes from one source and batches are random draws. Group answering assumes the caller can group inputs. The dev seeds were reused through five design generations, so the effects are optimistic until the fresh-seed run.

**Decision needed from you.** Approve the design (section 6) so that I write the spec addendum and the plan, and hand the code to Gemini once stage 2 is adopted.

---

## 1. What a gatekeeper has to do (and how we will judge it)

The gatekeeper is the doorman of the expert pool. It does four jobs, which fail in different ways (earlier survey, section 1):

1. **Detect novelty**: "does any expert recognise this input?" - and not cry wolf (false-novelty rate) or sleep through a real new regime (detection power).
2. **Assign for learning**: while training, decide which expert learns each sample.
3. **Select at test time**: pick the expert (or experts) that answers.
4. **Run the lifecycle**: spawn, extend, merge, retire.

You asked for one that is **more efficient, accurate, reliable and smart**. In measurable terms:

| Quality | Meaning | Measured by |
|---|---|---|
| Efficient | work per input grows slowly with the number of experts | gate multiply-adds per input (counted by the pool), wall-clock, memory per expert |
| Accurate | the pool reaches what perfectly routed experts reach | accuracy against the oracle-routed arm on streams with overlap, at the benchmark's own configuration |
| Reliable | its error rates are the ones it claims, and it fails gracefully | false-novelty rate vs the nominal level, spurious-spawn counts on guard streams, determinism, behaviour on mixed or tiny batches |
| Smart | it uses evidence at the right granularity (a set when there is a set, a sample when there is not) and several signals together | the above on overlap and mixed streams, plus the value of each extra signal measured alone |

## 2. What our current gate does, measured

(Notebook: E-014, E-015, E-017, E-019.) Per-sample conformal novelty (a kernel-density score per expert, p-values, Benjamini-Hochberg at q = 0.02) is excellent when regimes are separated and blind when they overlap strongly: at centre cosine 0.85 it rejects 0.8 % of foreign inputs [M]. Training contamination (samples going to the wrong expert) was the dominant loss at moderate overlap and the batch vote removed it (OV-0.3 33.6 -> 45.1 %, pre-registered, fresh seeds). What remains: (a) the scan costs 97 % of a prediction's work; (b) regimes that overlap strongly are merged into too few experts (OV-0.6 25.3 % vs about 48 % for oracle-routed experts); (c) selection among accepting experts has at most about 3 points of headroom on S1 [M].

## 3. What the field does

**Novelty or shift triggers expansion; the unit of decision is usually a task, not a sample.**
- SEMA [L+] ([arXiv 2403.18886](https://arxiv.org/abs/2403.18886), CVPR 2025, code [SEMA-CL](https://github.com/huiyiwang01/SEMA-CL), MIT licence): every adapter carries an autoencoder "representation descriptor"; a per-sample z-score of reconstruction error against a 500-entry buffer; expansion fires at a layer only if **all** existing descriptors flag shift; at most one adapter per layer per task, detection during the first epoch of a task (task boundaries are given). Router: softmax(x W), one column added per expansion. Reported 91.4 % (CIFAR-100), 84.8 % (5-task ImageNet-R), 0.55-0.65 M parameters added.
- MaRS [L] ([ICLR 2026](https://iclr.cc/virtual/2026/poster/10010510)): a frozen encoder, a slot-based memory router, and "statistically grounded slot expansion" with stated guarantees on false alarms and detection delay. I could not read the method (the review site blocked the PDF); the exact test and the form of the guarantee are unverified.
- Also: [Remembering Transformer](https://arxiv.org/abs/2404.07518) (a generative autoencoder judges task familiarity), [PMoE for turbulence modelling](https://arxiv.org/abs/2601.09305) (autoencoder routing), LLaVA-CMoE (probe experts). Closest relatives of our conformal gate: all use learned (autoencoder) novelty scores without finite-sample error control and mostly assume task boundaries.

**Router drift is a separate failure from expert drift.** [SAME](https://arxiv.org/abs/2602.01990) (orthogonal-subspace routing updates), [PASs-MoE](https://arxiv.org/abs/2601.13020) (ACL 2026, pathway-subspace guidance), [StaR-MoE](https://arxiv.org/abs/2605.17571) (routing alignment plus asymmetric capacity regularisation), [LEGO](https://icml.cc/virtual/2026/67405) (ICML 2026, localised expert expansion) [L]. Our router is non-parametric (stored prototypes), so it does not drift by gradient; its analogue is a reference set that becomes contaminated, which the batch vote addressed.

**Gradient-free routers and the oracle gap.** FLEX [L+] ([arXiv 2608.01437](https://arxiv.org/abs/2608.01437)): 34 tasks, routers compared (nearest class mean, hierarchical clusters, a growing SOINN network, RanPAC random-feature ridge, one autoencoder per task); the best routers recover only **28-50 % of the gap to oracle routing**; soft mixtures whose weights are a softmax of the full score distribution beat hard top-1 (PureLoRA: 56.27 vs 54.26 MacroScore, +2.0), and **sharpening the temperature (x28 to x80) helped without changing which expert is top-1**. TOOD [L+] ([arXiv 2607.29592](https://arxiv.org/abs/2607.29592), CoLLAs 2026): the "confidence gap" (logit scale shrinks as tasks accumulate) makes energy-based task detection degrade; fix = per-task energy channels recalibrated by median and MAD on replay-buffer statistics; tolerates a finer-than-true task grouping (-0.5 AUROC). Takeaway: **calibration per expert is the thing**; our conformal p-values are per-expert calibrated by construction, which is the property TOOD has to rebuild after the fact.

**Routing a whole set together is established.** Task-level MoE [L] ([Kudugunta et al., 2021](https://arxiv.org/abs/2110.03742): +1.0 BLEU over token-level on 30 language pairs, 1.9x peak inference throughput), sequence-level routing (Fan et al., 2024), Mixture-of-Clustered-Experts (sequence-level group, then token-level within it). Our batch vote is the unsupervised, data-driven version of this. Caveat from the same line: set-level routing loses per-sample flexibility.

**Cheap routing at scale.** AIR-MoE [L+] ([arXiv 2605.04952](https://arxiv.org/abs/2605.04952), May 2026): a vector-quantisation codebook of G codewords, each holding a posting list of M experts; each token takes the nearest codeword's shortlist and gets **exact** scores only there; shortlists are refreshed after every optimiser step at cost O(E G d); the guarantee is a lower bound on routing **mass** recall, exp(-2 epsilon) x rho_M, not on top-k recall; tested up to 1,048,576 tiny experts, perplexity gains up to 5.4 % over PEER (product keys) and hierarchical routers; limitations: no custom kernels, no growing pools. This is **the same two-stage pattern as our stage 2**; what we add is (i) an **exact second look** that makes the novelty decisions identical to the exhaustive gate, (ii) a density-based fingerprint (k-means sub-centroids of each expert's own prototypes) maintained incrementally per expert instead of a global codebook, and (iii) a growing pool. [RouterRetriever](https://arxiv.org/abs/2409.02685) shows the add-an-expert-without-retraining property of pilot-embedding routers. Product keys (PEER) and fast feedforward trees are the sublinear alternatives; our own geometry work found tree traversal 36-60x slower than a dense GPU scan at N <= 4096 [M, E-007].

**Sequential, calibrated evidence over a stream.** Conformal test martingales ([overview and betting functions in `crepes`](https://crepes.readthedocs.io/en/stable/martingales_nb.html), [theory](https://arxiv.org/abs/2411.11824)) turn conformal p-values into an anytime-valid test of exchangeability (alarm at wealth 1/alpha gives false-alarm probability <= alpha; CUSUM and Shiryaev-Roberts versions exist); testing by betting ([Shekhar and Ramdas](https://arxiv.org/abs/2112.09162), kernel two-sample tests with power one) [L]. Both need p-values or payoffs with the right dependence structure - which is exactly what my first set-level test lacked (section 5).

**Conformal methods for mixtures of experts exist but do not do growth.** Weighted p-values for combined prediction sets ([arXiv 2505.11785](https://arxiv.org/abs/2505.11785)), MoE-gating-similarity conformal intervals ([ICLR 2026](https://iclr.cc/virtual/2026/poster/10006832)), conformal routing for mixtures of agents (2026 preprint); a reported caveat that calibrating each expert suffices for hard routing but not for soft routing [L, secondary source]. I found no paper that combines growth, finite-sample novelty control and an exactness-preserving shortlist.

## 4. Codebases worth reading (read only; nothing downloaded or installed)

| Code | What it gives us | Notes |
|---|---|---|
| [SEMA-CL](https://github.com/huiyiwang01/SEMA-CL) | the nearest published growth-by-shift-detection implementation; benchmarks CIFAR-100, ImageNet-R/A, VTAB with a frozen ViT | MIT; builds on PILOT and AdaptFormer; 2 commits |
| [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL) | CLIP + MoE-LoRA adapters per task with a distribution-discriminative auto-selector | CVPR 2024 |
| [Mammoth](https://github.com/aimagelab/mammoth) | 40+ continual-learning methods and 20 datasets under one protocol; the place to get real streams and published baselines | official DER code base |
| Avalanche, PyCIL [R] | benchmark and strategy libraries | not opened |
| [crepes](https://github.com/henrikbostrom/crepes) | conformal predictors and a martingale module (betting functions, Simple Jumper, composite martingales) | useful to cross-check our own p-values |
| faiss, hnswlib [R] | the standard inverted-file and graph indexes; our shortlist is an inverted-file idea specialised to density fingerprints | not opened |

Our own earlier ideas on disk: `E:\universal_substrait` (phasor-resonance gate, token-sorted dispatch 18.5x, cold-start monopoly lessons), `E:\unified` (`VSARouter`, `FamiliarityGate` split-conformal novelty, `EpistemicRouter` cost cascade) - see E-014.

---

## 5. What we measured (dev seeds 100-102, 3 seeds, exploratory, NOT pre-registered; scripts and logs in `research/growmoe_results/rv3/`, notebook entry E-023)

The question: the per-sample gate is blind where regimes overlap. Can a test on a whole batch of inputs, with an error rate we know, see what the per-sample test cannot, and can it be put into the running pool without making it cry wolf?

### 5.1 In a clean setting, exact permutation tests are calibrated and fully powerful [M]

For each of 10 regimes a signature on its first-visit data; in-regime batches (64 inputs) against foreign batches; four overlap levels (centre cosine 0 / 0.3 / 0.6 / 0.85). The tests compare the batch with the expert's own stored samples: a permutation test on the distance between the batch mean and the reference mean (999 random permutations, so the smallest p-value is 0.001), the same with an RBF-kernel MMD, a Fisher combination of the owner's per-sample p-values (the naive version), and a score-sum bootstrap.

| centre cosine | per-sample test: share of own / foreign inputs flagged (AUROC) | distance-of-means test: false alarm at 0.05 / 0.01 / 0.001 | detection at 0.05 / 0.01 / 0.001 | AUROC |
|---|---|---|---|---|
| 0.00 | 0.000 / 1.000 (1.000) | 0.061 / 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 |
| 0.30 | 0.000 / 0.998 (1.000) | 0.056 / 0.006 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 |
| 0.60 | 0.000 / 0.869 (0.998) | 0.056 / 0.006 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 |
| 0.85 | 0.000 / 0.016 (0.532) | 0.056 / 0.006 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 |

Where the per-sample test flags 1.6 % of foreign inputs (it is blind), the batch test separates every foreign batch from every own batch at its nominal error rate. MMD is the same (false alarm 0.061-0.067 at 0.05). The naive Fisher test is only mildly off in this clean setting (0.061-0.072 at 0.05), the score-sum bootstrap slightly liberal (0.061-0.089). Source: `spike_set_tests_a1_log.txt`.

### 5.2 Inside the running pool the same tests failed, for avoidable reasons [M]

**The reference.** Shadow mode (the pool is trained normally, the test is only logged against each batch's best-explaining expert): detection stays 1.000 but known batches are flagged far above nominal. Distance-of-means false alarm at 0.05: 0.116 / 0.122 / 0.595 / 0.228 for centre cosine 0 / 0.3 / 0.6 / 0.85. Where it comes from: owners older than 15 steps (0.29-0.30 at cosine <= 0.3, 0.90 at 0.6, 0.59 at 0.85), batches whose best-explaining expert is not the regime's usual owner (flagged 100 % of the time), merged-regime owners. Owners younger than 15 steps stay near nominal (0.03-0.06; infants 0.00-0.13). The mean per-sample p-value of the owner on known batches is 0.425 / 0.422 / 0.481 / 0.500, below the 0.5 a calibrated gate would give at low overlap. The stored prototypes are a summary of what the expert has seen, not a fair sample of what it sees now. Testing against the samples the expert was actually trained on lately (its last five batches, at least 48 samples) is much better calibrated: false alarm at 0.05 / 0.001 = 0.103 / 0.071 (cosine 0), 0.106 / 0.071 (0.3), 0.189 / 0.115 (0.6), 0.164 / 0.079 (0.85), detection 1.000 everywhere, AUROC 0.964 / 0.964 / 0.943 / 0.960. (The 0.071 floor is 27 of 378 known batches, the same count as the wrong-owner batches above; I did not check batch by batch that they coincide.)

**Whom to test against.** The first live integration chose the three candidates by summed log-density and then by the per-sample gate's owners. Both failed at high overlap. The gate prefers old, broad experts to the freshly spawned specialised one, so the fresh expert was never among the tested candidates, every test rejected, and a new expert was spawned for every batch (seed 100: 113 experts on OV-0.6 with summed log-density; 104 on OV-0.85 with the owners-first rule). The fix, found by tracing every decision on one stream: take as candidates the three experts whose reference **mean is nearest to the batch mean** (the statistic the test uses), test them nearest first, and call the batch novel only if all three reject. After the fix the trace on OV-0.6 shows one expert per regime: the owner's p-value is 0.1-0.97 batch after batch while every other expert's is 0.001, even at centre cosine 0.6.

### 5.3 Test-time arbitration has no headroom [M]

Upgrades of "which accepting expert answers" (the expert's own confidence or entropy, soft mixtures of the accepting experts with weights softmax(log-density / T) for T = 0.5 to 32, top-2 mixtures), on pools trained with the current default:

| stream | experts | inputs accepted by more than one expert | default rule | confidence pick | soft mixtures T 0.5-32 | owner-routed bound (labels used) |
|---|---|---|---|---|---|---|
| S1 | 4.7 | 65.0 % | 79.4 | 72.5 (-6.9) | 79.4-79.5 (within 0.1) | 82.0 (+2.6) |
| S2-30 | 20.3 | 1.2 % | 90.2 | 88.9 (-1.3) | 90.0-90.2 | 90.2 |
| F-15 | 15.0 | 0.5 % | 50.4 | 49.9 (-0.5) | 50.1-50.4 | 50.5 |
| OV-0.3 | 10.0 | 1.9 % | 46.2 | 45.5 (-0.7) | 45.7-46.2 | 46.3 |

A model's own confidence is a worse arbiter than the density (it loses up to 6.9 points); mixing does nothing; the only stream with any headroom is S1 (+2.6 with labels) and no cheap upgrade captures it. Conclusion: **do not build smarter per-sample arbitration**. FLEX found that gradient-free routers recover only 28-50 % of the oracle gap [L+]; our result agrees that cheap per-sample arbitration is not where the headroom is. The gain FLEX reports from sharper soft mixtures does not appear here (temperature moved nothing, even on S1 where 65 % of inputs are accepted by more than one expert). Source: `spike_arbitration_vote_log.txt`.

### 5.4 Putting the batch test into the running pool [M]

A patched copy of `observe` (no product file touched): after the batch vote, the samples the per-sample gate accepted (at least 8) are tested as a set; "new as a set" sends them to the probation buffer together with the individually flagged ones, where the existing evidence test and interference probe decide spawn or extend. Nine streams, seeds 100-102, today's default pool (the batch vote) as the baseline. Scripts: `spike_set_gate.py` (code generations 1 to 4 are kept as `spike_set_gate_gen*.py`); tables from `compare_variants.py` and `final_table.py`.

**What it took.** Each row is one design decision. Cells are accuracy % / experts / known single-source batches wrongly called new (a false novelty). In OV-0.3-MIXED the "known" batches are 50/50 mixtures, so its count is not a false-alarm rate.

| design | OV-0.85 | OV-0.6 | S1 | S2-5 | S2-30 | OV-0.3-MIXED |
|---|---|---|---|---|---|---|
| vote (today) | 24.7 / 8.7 / 0 of 378 | 25.1 / 6.0 / 0 of 378 | 79.4 / 4.7 / 0 of 120 | 91.4 / 3.3 / 0 of 102 | 90.2 / 20.3 / 0 of 507 | 35.6 / 5.7 / 0 of 405 |
| reference = stored prototypes, nearest owner | 29.9 / 11.3 / 0 | 43.1 / 12.0 / 2 | 77.1 / 3.7 / **35** of 120 | 91.7 / 4.3 / **27** of 102 | 90.4 / 25.3 / **151** of 507 | 35.0 / 6.0 / 22 |
| reference = one recent window, nearest owner | 30.0 / 10.3 / 0 | 42.9 / 10.7 / 1 | 77.0 / 4.7 / **18** | 91.3 / 4.3 / **14** | 90.7 / 24.3 / **98** | 35.0 / 6.0 / 22 |
| one recent window, the vote keeps ownership | 24.5 / **18.0** / 4 | **32.4** / 13.0 / 1 | 81.1 / 5.3 / 21 | - | - | 34.8 / 6.0 / 22 |
| one recent window, owner by label loss | 30.0 / 10.3 / 0 | 42.9 / 10.7 / 2 | 79.8 / 5.3 / 18 | 91.3 / 4.3 / 16 | 89.9 / 25.0 / 97 | 35.0 / 6.0 / 22 |
| **source windows, owner by label loss** | **30.0 / 10.3 / 0** | **42.9 / 10.3 / 1** | **79.7 / 4.7 / 0** | **91.4 / 3.3 / 0** | **90.0 / 20.0 / 2** | 35.0 / 6.0 / 22 |

Reading it:
- **Reference.** Stored prototypes and one recent window both call known batches new far above nominal whenever the vote has merged several regimes into one expert (S1, S2-5, S2-30: 14-30 %): a batch of regime A is compared with a reference that mixes A and B. Keeping one window per recent *source* removes it (2 of 729).
- **Whom to test.** The candidates must be the experts whose window mean is nearest to the batch mean. The earlier rules (summed log-density; the per-sample gate's owners first) spawned 113 and 104 experts for 10 regimes on OV-0.6 and OV-0.85 (seed 100; seen in traces printed by `diag_set_gate.py` with the generation-1 code, not saved as a table), because at high overlap the gate prefers old broad experts to the fresh specialised one, so the fresh expert was never among the tested candidates.
- **Who owns an admitted batch.** The vote's ownership alone lets the old expert keep absorbing the new regime (OV-0.85 24.5 % with 18 experts, OV-0.6 32.4 %). Nearest-mean ownership fixes that but costs S1 2.4 points (77.0 against 79.4), because per-sample density is the better arbiter among heterogeneous regimes. Among the experts the test admits, the one with the lowest loss on the batch's labels is neutral on S1 and as good as nearest-mean on the overlapping streams. (A "veto" variant, the vote's owner tried first, was tried interactively on a few seeds and did worse at OV-0.85; those runs were not saved and are not evidence.)

**The result, final design against today's default** (accuracy %, mean of the three seeds, and the range of the per-seed difference):

| stream | vote | set gate | difference [min, max] | experts, vote -> gate (regimes) | known batches called new | new-as-a-set calls per run |
|---|---|---|---|---|---|---|
| OV-0.85 | 24.7 | 30.0 | +5.3 [+3.8, +7.3] | 8.7 -> 10.3 (10) | 0 of 378 | 11.0 |
| OV-0.6 | 25.1 | 42.9 | **+17.8** [+14.6, +20.1] | 6.0 -> 10.3 (10) | 1 of 378 | 8.7 |
| OV-0.3 | 46.2 | 46.2 | 0.0 | 10.0 -> 10.0 (10) | 0 of 378 | 0.0 |
| OV-0.0 | 50.7 | 50.7 | 0.0 | 10.0 -> 10.0 (10) | 0 of 378 | 0.0 |
| F-15 | 50.4 | 50.4 | 0.0 | 15.0 -> 15.0 (15) | 0 of 528 | 0.0 |
| S2-5 | 91.4 | 91.4 | 0.0 | 3.3 -> 3.3 (5) | 0 of 102 | 0.0 |
| S1 | 79.4 | 79.7 | +0.3 [-1.5, +1.8] | 4.7 -> 4.7 (7) | 0 of 120 | 1.3 |
| S2-30 | 90.2 | 90.0 | -0.1 [-1.2, +0.5] | 20.3 -> 20.0 (30) | 2 of 507 | 2.3 |
| OV-0.3-MIXED | 35.6 | 35.0 | -0.5 [-1.8, +0.5] | 5.7 -> 6.0 (10) | (mixed batches) | 7.7 |

Pooled over the single-source streams, 3 of 2,769 known batches were called new (0.11 %). Against the spike acceptance criteria written in the spec before any set-level code existed: **(a) expert counts unchanged (within 0.5) on F-15, S1, S2-5, S2-30, OV-0: MET**; **(b) at OV-0.6 and OV-0.85 at least 8 experts and at least +8 points: MET at OV-0.6 (10.3 experts, +17.8), NOT MET at OV-0.85 by accuracy (+5.3)** for the training side alone. Section 5.5 shows what answering by group adds there.

### 5.5 Answering a group together [M]

At test time inputs often arrive together (a session, a document, a batch). The group is answered by the expert most of its inputs chose individually; with the *veto*, an input that expert's own per-sample gate rejects keeps its own choice. Script `spike_arbitration.py`, tables from `routing_table.py`. A "group" is 64 inputs of one regime (the grouping is an explicit assumption); mixed groups are 32 + 32 or 58 + 6 inputs of two regimes.

Pools trained with the final set gate; accuracy %, seeds 100-102:

| stream | experts | per sample | group vote | group vote + veto | 50/50 mixed: per sample / vote / vote + veto | 90/10 mixed: per sample / vote / vote + veto | group of 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| OV-0.85 | 10.3 | 30.0 | 42.1 (+12.1) | 41.6 (+11.6) | 30.2 / 26.1 / 27.8 | 30.2 / 38.8 / 38.7 | 39.1 / 41.3 / 41.7 |
| OV-0.6 | 10.3 | 42.9 | 44.3 (+1.4) | 43.9 (+1.0) | 42.0 / 29.1 / 40.6 | 42.0 / 40.9 / 43.1 | 44.3 / 44.3 / 44.3 |
| OV-0.3 | 10.0 | 46.2 | 46.3 (+0.1) | 46.2 (0.0) | 46.9 / 31.0 / 46.7 | 46.9 / 44.3 / 46.9 | 46.3 / 46.3 / 46.3 |
| OV-0.0 | 10.0 | 50.7 | 50.8 (+0.0) | 50.7 (0.0) | 51.0 / 32.1 / 51.0 | 51.0 / 47.9 / 51.0 | 50.8 / 50.8 / 50.8 |
| OV-0.3-MIXED | 6.0 | 35.0 | 35.0 (-0.1) | 35.0 (-0.1) | 34.9 / 28.7 / 34.7 | 34.9 / 33.3 / 34.4 | 35.0 / 34.9 / 34.9 |
| F-15 | 15.0 | 50.4 | 50.5 (+0.1) | 50.4 (0.0) | 49.1 / 31.5 / 49.1 | 49.1 / 46.2 / 49.1 | 50.5 / 50.5 / 50.5 |
| S1 | 4.7 | 79.7 | 83.7 (+4.1) | 81.5 (+1.8) | 80.5 / 62.4 / 76.5 | 80.5 / 80.4 / 82.2 | 83.1 / 83.7 / 83.7 |
| S2-5 | 3.3 | 91.4 | 91.4 (0.0) | 91.4 (0.0) | 90.8 / 58.3 / 90.8 | 90.8 / 85.1 / 90.8 | 91.3 / 91.3 / 91.3 |
| S2-30 | 20.0 | 90.0 | 90.1 (0.0) | 90.1 (0.0) | 90.0 / 52.1 / 89.9 | 90.0 / 82.7 / 90.1 | 90.1 / 90.1 / 90.1 |

The same on pools trained with today's default: OV-0.85 24.7 -> 32.6 (+8.0), S1 79.4 -> 81.9 (+2.5), OV-0.6 +1.7, everything else within 0.4. The two steps add up: OV-0.85 goes from 24.7 % today to **42.1 %** with the set gate and group answering (41.6 % with the veto; oracle-routed experts reach about 48 %), OV-0.6 from 25.1 % to **44.3 %** (43.9 % with the veto).

Reading it:
- Each input's own choice is noisy where regimes overlap (inputs accepted by several experts: 98 % at OV-0.85), but 64 noisy votes add up to a reliable majority. A permutation test adds nothing at this stage: routing the group to the nearest expert that the test admits was measured on the earlier pools (trained with one recent window) and is no better than the vote by more than 0.3 points anywhere, and clearly worse on S1 (-18.5 points without a veto).
- The plain vote is dangerous for mixed groups at low overlap (50/50: -15 to -38 points) because half the group is forced onto the wrong expert. The veto brings that back to the per-sample level (F-15 49.1 -> 49.1, S2-30 90.0 -> 89.9) and keeps the gain where the per-sample gate cannot reject anyway (OV-0.85 90/10: +8.5). Its price is a smaller gain on pure groups where the per-sample gate is partly informative (S1: +1.8 instead of +4.1). **The default must be vote plus veto.**
- It works down to a group of 8 inputs (OV-0.85: +9.1; 16: +11.3).

### 5.6 How small can a batch be [M]

Clean setting, a signature per regime, in-regime batches against foreign batches, distance-of-means test against the stored prototypes, 999 permutations (`spike_power_vs_batch.py`). Detection of foreign batches at level 0.01 is 1.000 for batches of 8, 16, 32, 64 and 128 inputs at centre cosine 0.3, 0.6 and 0.85, and at level 0.001 it is 0.999-1.000; the false alarm stays at the nominal level (0.000-0.022 at 0.01, at most 0.001 at 0.001). On these streams the batch mean separates regimes so strongly that 8 inputs are enough; real data will be harder (see the caveats in section 6.4).

### 5.7 When the single-source assumption is violated [M]

The samples of each segment are sorted by label before they are cut into batches, so a batch holds one or two classes and the batch test sees every batch as unlike its expert's recent data. Script `spike_sorted_batches.py`, 3 seeds. Accuracy collapses for every variant (the vote pool alone drops from about 50 % to about 20 %), so the number that matters is the expert count.

| stream (regimes) | vote | set gate, no breaker | breaker: 30 batches, 30 % | breaker: 12 batches, 35 % |
|---|---|---|---|---|
| F-15 (15) | 19.9 % / 18.7 experts | 18.3 % / 81.7 | 20.1 % / 38.3 | 19.3 % / 28.0 |
| OV-0.0 (10) | 19.3 % / 14.7 | 19.7 % / 52.7 | 20.5 % / 23.7 | 19.9 % / 23.0 |

The breaker pauses the batch level when more than 35 % of the last 12 batches that looked familiar per input (at least 90 % accepted) were called new; the pool then behaves as today for 100 steps. Against the vote it cuts the excess experts from +63 to +9 on F-15 and from +38 to +8 on OV-0.0. On the normal streams the faster breaker never fired and the results are bit-identical to the same design without it (OV-0.85, OV-0.6, S1, S2-30, three seeds each); on OV-0.3-MIXED, whose batches are 50/50 mixtures, it paused once in two of three seeds and the mean accuracy stayed 35.0. The assumption cannot be removed, but the damage can be contained. (The earlier one-window version spawned 102.7 experts on F-15 and 64.3 on OV-0.0.)

### 5.8 MMD instead of the distance of means [M]

A kernel statistic (RBF, median heuristic) detects any difference between two distributions, not only a shift of the mean, so it is the hedge for real data whose regimes differ in other ways. In the live pool (`--variants mmd_windows_loss`, six streams, three seeds) it gives the same accuracy and expert counts as the distance of means on OV-0.85, OV-0.6, F-15, S2-5 and OV-0.3-MIXED, and is 1.1 points lower on S1 (78.6 against 79.7; per-seed difference to the vote -0.9 [-4.9, +1.8], 1 of 120 known batches called new). On these streams the regimes differ in the mean, so MMD has nothing to gain here. The choice is left to the dev tuning of the implementation; the default proposed is the distance of means (cheaper, and the better of the two on S1).

### 5.9 A first look at real text [M]

Everything above runs on synthetic streams. This check (`spike_real_text.py`, notebook entry E-024) uses real text from the local cache: 20 Newsgroups (17 categories), AG News (4 topics) and the 20 most frequent banking77 intents, embedded with frozen all-MiniLM-L6-v2 and projected to the pool's 128 dimensions. Clean setting: a signature per category, held-out documents from the same category or from another one; the foreign categories are the 4 nearest by centre cosine plus 4 random ones; seeds 100-102.

- **The single-input gate is blind on real text almost everywhere.** It flags 17-19 % of foreign inputs on 20 Newsgroups and AG News and 34 % on banking77 (AUROC 0.65-0.68), and for 64-75 % of the foreign batches it flags fewer than 30 % of the inputs; the median centre cosine of those pairs is only 0.32-0.38 (banking77: 0.73, all its intents are close). On the synthetic streams it only went blind at 0.6 and above.
- **The batch test is calibrated and sees them.** False alarm 0.037-0.067 at level 0.05 and at most 0.010 at 0.001. Detection at level 0.05: 96.7-99.9 % from 8 documents, 99.6-100 % from 16; at level 0.001: 76-97 % from 8, 96-100 % from 16, 100 % from 32. AUROC 0.99-1.00.
- **The hardest pairs need a few more documents.** For the 20 % most similar foreign pairs the per-sample gate flags 0.2-2.6 % of the inputs; the batch test detects at level 0.001: 44 % (8 documents), 85 % (16), 100 % (64) on 20 Newsgroups, and 88 %, 99 %, 100 % (8, 16, 32) on banking77.

So the premise survives real text, and the problem stage 3 addresses is bigger there than the synthetic streams show. The operating range to state is 16 to 32 inputs or more per batch or group. Not tested: the live pool (window references, ownership, spawn) on a real stream, regimes that differ in style or time rather than topic, MMD on real text.

## 6. The design: a two-level gate ("Gatekeeper v3")

### 6.1 Designs considered

| | Design | What the evidence says (dev seeds) | Verdict |
|---|---|---|---|
| D0 | Today: per-sample conformal gate + batch vote (stage 1, adopted) | Excellent when regimes are separated; blind and merging when they overlap (OV-0.6 25.1 %, OV-0.85 24.7 %, against about 48 % for oracle-routed experts); the scan is 97 % of a prediction's work | baseline |
| D1 | D0 + fingerprint shortlist with an exact second look (stage 2: built, tests green, its pre-registered run is next) | 7-14x fewer gate multiply-adds at 15-30 experts, 50x at 500; novelty decisions identical to the exact gate by construction; accuracy gap 0.0 on the dev streams | proceed |
| D2 | D1 + a calibrated test on the whole batch while learning (stage 3a) | OV-0.6 25.1 -> 42.9, OV-0.85 24.7 -> 30.0; every other stream within 0.5; known-batch false novelty 3 of 2,769 | recommend |
| D3 | D2 + answering a group by the vote of its inputs' own choices, with a per-sample veto (stage 3b) | OV-0.85 -> 41.6, OV-0.6 -> 43.9 (with the veto), S1 +1.8; no stream loses more than 0.1 on pure groups; mixed groups stay within about 4 points of per-sample | recommend |
| D4 | A learned router (autoencoder descriptors and a trained softmax, as SEMA) | not tried: no finite-sample error control; FLEX shows such routers recover 28-50 % of the oracle gap | not recommended |
| D5 | Smarter per-sample arbitration (confidence, entropy, soft mixtures) | no gain, confidence is worse (section 5.3) | rejected |
| D6 | Sequential test martingales for streams without batch boundaries | not tested; needs p-values with the right dependence structure | future work |

### 6.2 The recommended design in plain words

The per-sample test asks "is this one input familiar?". Where regimes overlap it cannot tell, because one input from regime B also looks plausible under regime A. The second level asks about the whole batch: "does this group of inputs look like what any single expert has been fed lately?". A permutation test answers with a known error rate. One rule combines the levels: the batch-level test decides whether the batch is new and, only if needed, who owns it; wherever it has no evidence, the per-sample machinery decides exactly as before.

**While learning (`observe`), flag `set_gate`, default OFF until the pre-registered run:**
1. Gate and batch vote as today (the cascade of D1 applies underneath).
2. Take the samples the per-sample gate accepted (at least 8). Every expert keeps a few *source windows*: the last five trained batches of each recent source, at most four sources. The candidates are the three experts whose nearest window is closest to the batch mean.
3. Test the accepted samples against each candidate's windows with a permutation test (999 permutations, distance of means or MMD, level 0.001). A candidate is *admitted* if at least one of its windows does not reject.
4. No candidate admitted: the batch is new as a set. Its accepted samples go to the probation buffer with the individually flagged ones, and the existing evidence test and interference probe decide spawn or extend.
5. One or more admitted: the accepted samples train the admitted expert with the lowest loss on the batch's labels (the nearest window if there are no labels). Individually flagged samples are never reassigned.
6. A circuit breaker watches the rate of "new" calls among batches that look familiar per sample (at least 90 % accepted). Above 35 % over the last 12 such batches the second level pauses for 100 steps and the pool falls back to the vote.
7. After training, each expert updates its windows: a trained batch joins the newest window if a permutation test finds it compatible (level 0.01), otherwise it opens a new window and the oldest window is dropped.

**While answering (`predict(x, groups=...)`), flag `group_vote`, default OFF until the run:** if the caller says which inputs arrive together (a session, a document, a batch), each group of at least 8 inputs is answered by the expert that most of its inputs chose individually, except for inputs that expert's own per-sample gate rejects, which keep their own choice. Without `groups`, `predict` is unchanged bit for bit.

### 6.3 Parameters (chosen on dev seeds; frozen before the pre-registered run)
Level 0.001 with 999 permutations (the smallest p-value is 0.001); three candidate experts; at least 8 accepted samples; windows of five batches, four per expert, a window split at level 0.01; an expert with fewer than 48 window samples is represented by its stored prototypes; ownership by label loss; breaker 12 batches, 35 %, 100 steps; groups of at least 8 with the veto (16 to 32 or more is the stated operating range on real text, section 5.9). Statistic: distance of means (MMD is compared in 5.8).

### 6.4 What it guarantees and what it assumes

| Property | Basis |
|---|---|
| Shortlist novelty decisions equal the exact gate's | proof sketch (spec section 5) and a property test; stage 2 |
| The set test's false-alarm rate is at most its level when batch and reference are exchangeable | permutation-test theory; measured 0.056-0.061 at level 0.05 in the clean setting, and 3 of 2,769 known batches in the live pool |
| It sees what the per-sample test cannot | clean setting AUROC 1.000 at centre cosine 0.85 where the per-sample test has 0.532, down to batches of 8 |
| It does not change what already works | OV-0.0, OV-0.3, F-15, S2-5: accuracy and expert count identical to the vote; S1, S2-30, OV-0.3-MIXED within 0.5 |
| **Assumption 1:** a batch comes from one source and the batches of one source are random draws | the synthetic streams are built that way; real streams may not be (below) |
| **Assumption 2:** labels exist for ownership by loss | without labels the nearest window is used (costs S1 about 2 points in the spikes) |
| **Assumption 3:** at inference the caller can group inputs | otherwise `predict` is per sample as today |
| **Not shown:** the live pool on real data | the live-pool streams are synthetic with strongly mean-separated regimes; the clean-setting test on real text embeddings holds (section 5.9) (Q-04) |

### 6.5 Failure modes and guard rails

| Failure | What was measured | Guard rail |
|---|---|---|
| Batches sorted by label (assumption 1 violated) | one-window version: F-15 102.7 experts for 15 regimes, 279 of 528 known batches called new; the vote pool itself collapses (19.9 % against 50.4 %) | circuit breaker: excess experts +63 -> +9 on F-15 and +38 -> +8 on OV-0.0; it never fired on the four normal streams tried (section 5.7) |
| A mixed batch (two sources) while learning | 5 % of the accepted subsets of 50/50 batches are called new as a set; accuracy -0.5 | flagged samples are never reassigned; the evidence test still decides spawn |
| An expert that learned several regimes in turn | one window: 14-30 % of known batches called new | source windows: 2 of 729 |
| Nearest-window ownership in heterogeneous regimes | S1 -2.4 | ownership by label loss: S1 +0.3 |
| A mixed group while answering | plain vote: -15 to -38 points on 50/50 groups at low overlap | the per-sample veto: within about 4 points of per-sample everywhere |
| New-as-a-set batch whose cluster fails the evidence test | a few batches per run wait in the probation buffer (abstain events; 2 per run at OV-0.85, 6 on OV-0.3-MIXED) | known limitation; the evidence test stays the last word on a new expert |
| Cost | a full permutation test is 32 ms (999 permutations, one busy thread); one per batch plus one per updated expert | stop at the first exceedance beyond the level: the decision is the same and a known batch costs one chunk |

### 6.6 What would make us drop it
Any pre-registered guard stream losing more than 1.0 point, an expert count above the number of true regimes by more than 0.5, known-batch false novelty above 2 %, or less than +8 points at OV-0.6 on fresh seeds. Each is a kill rule in the spec addendum.

## 7. Test plan

Same process as stages 1 and 2: spec addendum (design, flags, interfaces, statements and bars), implementation plan with tests first, hand-off to Gemini once stage 2 is adopted, dev tuning on dev seeds 100-109 (103-109 are new), hash-frozen pre-registration, fresh seeds 60-79 (40-59 belong to stage 2), script-generated report, decision rule. Results go to `research/growmoe_results/rv2c/`.

**Arms.** `A_vote` (today's default), `A_set` (set gate on). Group answering is evaluated on the pools of both arms with groups of 64 single-source test inputs (the grouping is an explicit assumption) and with mixed groups.
**Streams.** F-15, F-30, S1, S2-5, S2-30, OV-0.0, OV-0.3, OV-0.3-MIXED, OV-0.6, OV-0.85, a batch-size check (16 and 256) on OV-0.3 and OV-0.6, and a label-sorted F-15 as a documented-failure guard.

**Statements** (bars at about half the dev effect; dev values in brackets):

| ID | Statement | Dev |
|---|---|---|
| S3-1 | Overlap gain while learning: paired `A_set - A_vote` at OV-0.6 >= +8.0 and at OV-0.85 >= +2.5 points, interval lower bounds above 0 | +17.8, +5.3 |
| S3-2 | Guards: F-15, F-30, S1, S2-5, S2-30, OV-0.0, OV-0.3 paired difference >= -1.0, interval lower bound >= -2.0 | 0.0 to +0.3, -0.1 (F-30 not yet run on dev) |
| S3-3 | Mixed guard: OV-0.3-MIXED paired difference >= -1.0 | -0.5 |
| S3-4 | No spurious experts: mean experts <= regimes + 0.5 on every stream | at most +0.3 |
| S3-5 | False novelty: known single-source batches called new <= 2 % pooled | 0.11 % |
| S3-6 | Group answering (vote + veto) on `A_set` pools, pure groups: OV-0.85 >= +6.0, OV-0.6 >= 0, every other stream >= -0.5 | +11.6, +1.0, >= -0.1 |
| S3-7 | Mixed groups with the veto: 50/50 >= -5.0 and 90/10 >= -1.0 against per sample, every stream | -4.0, -0.5 |
| S3-8 | Small groups: OV-0.85 groups of 16 >= +5.0 | +11.3 (plain vote; the veto was not measured at this size) |
| S3-9 | Flags off: bit-identical to today (tests, and expert counts on every stream) | by construction |
| R | Reported, not gated: wall-clock of `observe` with and without the set gate, label-sorted F-15 with and without the breaker, MMD against distance of means, groups of 8 on OV-0.85 | |

**Decision rule.** Adopt the set gate (flip `set_gate`) iff S3-1 to S3-5 and S3-9 pass; adopt group answering (flip `group_vote`) iff S3-6 to S3-8 pass (it changes nothing unless `groups` is passed). S3-1 passing only at OV-0.6 keeps the set gate opt-in and says so. Any guard failing is not adopted.

**Tests written first.** Permutation p-value on a toy case (exact count, smallest value 1/(n+1)); validity under the null by simulation; power; the early stop equals the full computation for the same permutations; the window update (join, split, drop); candidate order; each ownership rule; a mixed batch keeps its flagged samples; the breaker opens and closes; flag off is bit-identical; determinism; counters; `predict` without `groups` unchanged; the vote with and without the veto.

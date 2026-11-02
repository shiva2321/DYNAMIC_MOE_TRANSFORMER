# Milestones: the detailed specification of the road

Version 0.1, 2026-11-02. Part of the road plan ([ROADMAP.md](ROADMAP.md)). The summary table is in ROADMAP section 5; this file holds the detail of each step.

**In plain words.** Nine milestones (M0 to M8) and three early probes (P1 to P3). Each one says why it is on the road, what it needs, what we do, how we will know it passed, what it costs on this machine,
and what to try first if it fails. **All numbers called "proposed bar" are proposals** chosen so that a pass would mean something to a reviewer and a fail would be informative; they are fixed (and hash-frozen)
in each milestone's pre-registration before any test seed is run, and they can be changed only by a logged decision made *before* the results are seen (ROADMAP section 7).

**Conventions used in every milestone**
- **Pre-registration first** (notebook rule 2): statements, bars, seeds, budgets, kill rule, written and frozen before the first test seed. Verdicts computed by script.
- **Seeds:** dev seeds tune; report seeds judge; single-use confirm seeds judge once. Never use 400-429 or 600-604 (reserved). Suggested blocks, to be confirmed at pre-registration:
  M1 800-829, M2 830-859, M3 860-899, M4 900-949, M5 950-999, M6 1000-1049, M7 1050-1099.
- **Baselines next to every claim**, tuned with the same effort as our system (D-06), under equal memory *and* equal compute. **"Equal compute" means equal counted training multiply-adds** (the primary budget);
  wall-clock and energy are reported next to it, and if the two give different rankings, both are shown.
- **Devices.** CPU and GPU are reported separately and every arm of a comparison runs on the same device. Each pre-registration names the primary device: M1 uses the CPU (the pool's small experts run there today) with a GPU repeat; M5 uses the GPU.
- **Single-use confirmatory runs start only on your explicit go** (D-36, D-37).
- **Built for power cuts** (D-30): small units, one result file each, atomic writes, checkpoints, one idempotent resume command per study.
- **Memory-lean:** one GPU job at a time; small tensors; no benchmark that fills RAM or VRAM.
- **Deliverables, always:** a plain-English report with one summary figure, a notebook entry with its claims, failures and decisions, an updated [STATE_BOARD.md](STATE_BOARD.md) and a line in [CHANGELOG.md](CHANGELOG.md).
- **Effort** in sessions: S = 1, M = 2-4, L = 5-9, XL = 10 or more. A session is one focused work period; the plan counts sessions, not dates.
- If a milestone fails, go to the stuck protocol (ROADMAP section 6) before anything else.

The finish-line statements F1 to F8 are defined in ROADMAP section 1.3.

---

## Phase A. Make the work safe, repeatable and measured

### M0. Foundations
**Why.** Nothing downstream is trustworthy if the work can be lost, cannot be rerun, or is measured differently each time. This is also where the "claim contract" (F1-F8) is fixed.
**Needs.** Nothing.
**Steps.**
1. *Safety.* Commit the uncommitted work in logical groups on a single branch (no branches unless necessary; done locally 2026-11-02, except 832 files held by the commit gate, D-72); pushed on 2026-11-02 to a new branch, `growmoe-lab`, of the repository you named (decision 9; the 832 held files are not in it); make a second copy of the data that cannot be recreated (caches 10.4 GB, ISHMN checkpoints about 9 GB, `results/unimoe/emb/`). **Your decisions** (ROADMAP section 10).
2. *Reproducibility* (lock file written and clean-clone test run on 2026-11-02, see BASELINE_V1.md; a fresh install from the lock is still untested). Pin the environment (lock file); clean-clone test: fresh checkout, install, full test suite; record thread settings (Q-29).
3. *Baseline v1* (tagged `baseline-v1` at a96bffc on 2026-11-02; record in BASELINE_V1.md). Tag the commit; write down every default and every flag that is on or off (skip off, chance flag off, growth held, LDA head not yet the default); this is the reference all later numbers are compared with.
4. *Measurement harness* (new package, for example `costlab/`): inference latency at batch 1 and 32; throughput; active bytes read; counted multiply-adds; peak RAM and VRAM; update time per batch and per new skill; GPU energy through NVML; CPU time. Validated against toy models of known cost.
5. *Claim contract and the M1 pre-registration* (F1-F8 confirmed as proposed, D-75; M1 protocol drafted on 2026-11-02 in M1_PREREG_DRAFT.md, to be frozen after the cost meter is reviewed). Confirm or adjust F1-F8; write and hash-freeze the M1 protocol.
6. *Prior-art check (added in v0.2; first pass done 2026-11-02, see PRIOR_ART.md).* One session: targeted searches for the exact combination (task-free calibrated novelty gate, closed-form heads in isolated experts, lifecycle with exact removal, single-GPU compute-matched cost accounting) and a table of the closest ten works, what each does and what it lacks. It feeds BP2 and the novelty statement in every report.
**Proposed bars.** (a) a restore test of one archive works and the hashes match; (b) the full suite passes from a clean clone; (c) counted multiply-adds within 5% of the analytic count on three toy models; (d) wall-clock coefficient of variation at most 5% over 10 repeats at fixed threads; (e) GPU energy readings behave sensibly (idle against load, scales with work).
**Cost.** M (4-5 sessions with the prior-art check). Light on compute.
**If it fails.** Variance too high: pin CPU affinity, longer warm-up, more repeats, count operations instead of time. Energy not measurable on the CPU path: report GPU energy and counted operations, say so.
**Answers.** Q-29. Prepares Q-10, Q-68.

### P1. Probe: can a pool of small language models be built and routed on this machine?
**Why.** The central bet of M5 is that a pool of small language models, routed per sequence, can match a dense model with less active compute (the SmallTalk LM result, L) and can keep adding experts without forgetting. We need *measured* time, memory and quality on this GPU before planning M5.
**Needs.** M0 step 4 (timing). Three text domains from local or small downloads (for example children's stories, code, encyclopedic text).
**Design.** Three experts of about 10-20M parameters, one per domain; a router from a small encoder's embedding of the first 64 tokens; compare with (A) one dense model with the same *total* parameters trained on all three domains and (B) one dense model with the same *active* size, same total tokens. Measure tokens per second, VRAM, perplexity per domain, routing accuracy, decode speed at batch 1 and 16, bytes read per token.
**Output.** Numbers, not a pass or fail: which of the three M5 designs (pool of small models; shared base plus adapters plus closed-form heads; MoE blocks inside one small transformer) looks feasible; revised cost estimate for M5.
**Cost.** M (2-3 sessions); GPU for about an hour or two per configuration, resumable.
**Answers.** Prepares Q-43; idea I-04.

### P2. Probe: is removal exact?
**Why.** You want knowledge that is unnecessary or proven wrong to be made dormant or removed. Isolation suggests removal can be *exact* in our design, which no published MoE unlearning method claims (RESEARCH_DIGEST section 2).
**Design.** Closed-form heads on frozen embeddings (banking77, the seven UniMoE domains). Train with a class and without it; remove it by subtracting its counts, sums and scatter ("downdating"); compare weights (float64, tolerance 1e-6) and predictions. Then audit every component that saw the removed data: router fingerprints, conformal calibration sets, replay reservoirs, stored prototypes, the router's own fit. Classify each as exactly removable, approximately, or not.
**Output.** A table of components with the measured difference to a model never trained on the data, and what has to change in the code to make each one removable.
**Cost.** S-M (1-2 sessions); CPU only.
**Answers.** Idea I-16; F6 exactness. Prepares M4 step 4.4.

### P3. Probe: what does a cheap-first, expensive-if-unsure cascade cost?
**Why.** The 1.5B asking reader costs about 50 ms per message and would break the efficiency claim if used on every input. Our trained head already answers 56% of arithmetic messages at 2.1% disagreement (E-066).
**Design.** From the cached reader verdicts (E-065, E-066), sweep the head's confidence band; for each, measure the share answered cheaply, disagreement with the reader, the effect on pool accuracy, and amortised cost = c_head + p_ambiguous x c_reader (costs measured by the M0 harness).
**Output.** A cost against accuracy curve and a recommended band; feeds M3.
**Cost.** S (1 session).
**Answers.** Idea I-06; part of Q-71.

### M1. Efficiency scoreboard v1: where do we really stand?
**Why.** Today's efficiency evidence is partial and mixed (STATE_BOARD section E). Before building more we need the real cost of the current default stack, against the baselines people actually use, so that effort goes where the cost is.
**Needs.** M0 harness and Baseline v1.
**Steps.** Three beds: function regimes (N = 10, 30, 100), banking77 class-incremental, the seven UniMoE domains. Arms: Baseline v1 pool (defaults) and with the best flags (chosen on dev seeds only); same-memory single network with replay (tuned); **compute-matched** replay fine-tune; streaming LDA and Mahalanobis (SLDA/FeCAM-style) prototypes; random-feature heads (I-15); kNN; joint offline network. Measure accuracy and forgetting; update cost per new regime and over the stream; inference cost per input at batch 1 and 32 (time, counted operations, bytes read, energy where measurable); memory against the number of regimes; the gate's share of cost.
**Deliverable.** Scoreboard report and figure (accuracy against cost, Pareto fronts); the top three cost drivers; a recommended order for efficiency work; evidence for the decisions held on defaults (LDA head, growth rule, skip and chance flags): branch point BP1.
**Proposed bars.** None on superiority. Pass means: all arms measured with the same harness, at least 5 seeds, variance reported, predictions written first and scored. Decision rule: any baseline that matches the pool within 1 point of accuracy at lower update cost *and* lower inference cost on a bed makes efficiency the first priority of M3 and M4.
**Cost.** M (3-5 sessions). CPU and light GPU.
**Answers.** Q-68; parts of Q-10, Q-30; settles what is claimed about efficiency and what is not.

---

## Phase B. Be credible on a standard benchmark

### M2. A standard text benchmark with frozen features
**Why.** Everything so far is on our own streams and three cached text sets. The field needs a protocol it already knows, with no task identity at test time, and the closest simple baselines (frozen features plus closed-form heads) are strong (RESEARCH_DIGEST section 1). This milestone shows we are *not worse* than them, and measures novelty detection under a standard protocol.
**Needs.** M0, M1 (baselines and harness). **Your decision:** permission to download the four datasets we do not hold (Yelp, Amazon, DBpedia, Yahoo; AG News is cached).
**Steps.**
1. The five-dataset sequence (AG News, Yelp, Amazon, DBpedia, Yahoo; the usual 115,000 training and 7,600 test examples per dataset; four task orders; no task id at test time; one head over all classes).
2. Two frozen encoders (MiniLM, which we have, and a stronger one such as Qwen3-Embedding-0.6B) so the conclusion does not rest on one encoder. Embedding runs resumable in shards.
3. Arms as in M1, plus our pool; literature numbers (for example LAMOL) shown as reference only, not re-run.
4. Open-world protocol: hold out one dataset's classes as "unknown"; report unknown-detection AUROC and false-new on known; and accuracy on classes that arrive later.
**Proposed bars (F3 on frozen features; F4 partly).** Final average accuracy within 1.0 point of the best simple baseline or better; forgetting at most 1.0 point; unknown-detection AUROC at least that of the best baseline detector; total cost within 2x of the cheapest baseline (the closed-form heads are very cheap, so the pool must earn its extra cost through the open-world and lifecycle abilities, not accuracy).
**If downloads are not approved:** M2 runs on what we hold (AG News, 20 Newsgroups, banking77 and the seven UniMoE domains) arranged as a sequence in the same way. That stays at level L2, because it is not the standard benchmark, and the plan says so.
**Honest expectation.** The closed-form heads will probably saturate and tie. A tie is a pass of the "not worse" bar; the pool's differentiators are measured in M3, M4 and M5.
**Cost.** M-L (4-6 sessions).
**If it fails.** Diagnose with oracles: oracle routing, oracle novelty, told-the-dataset experts. Likely limiters: false spawns on similar datasets (Yelp and Amazon), minimum cluster size (the m_min effect, E-066), router refit. Options: expert cap or merge for few-class topical streams (Q-72), random-feature heads (I-15), team routing (I-03).
**Branch point BP2.** If the pool is below the best simple baseline by more than 1 point after diagnosis: either differentiate on open-world and lifecycle abilities (continue), or simplify to a closed-form core with minimal routing (and say so).

---

## Phase C. Make the parts cheap, robust and bounded

### M3. Cheap, internal novelty and nuance detection
**Why.** The finished model must notice the new, including subtle new wording and meaning, without calling a large separate model on every input (F4). The reader is excellent at wording and function nuance and poor at topical novelty; sentence embeddings are the reverse (the double dissociation, E-064, G1). We build our own detector, for this architecture.
**Needs.** M1 (cost numbers), P3, the nuance lab (`research/knowledge_status`).
**Steps.**
1. *Nuance suite v2:* real paraphrase sets of banking intents (rewritten locally; **label quality** is checked by two independent model passes that must agree on at least 95% of pairs, plus a spot check of 100 random pairs read by you if you are willing, otherwise by a different model from the one that wrote them; the measured label quality is reported with every result), minimal pairs (negation flips, swapped numbers and operators), style and chatter shifts, typos, near-twin skills, topical novelty, function novelty (arithmetic and code). Labels: same meaning or new meaning.
2. *Small trained encoder* (at most 30M parameters to begin with; contrastive training with minimal pairs and augmentations; idea I-07), or a head on the reader's hidden states (I-08).
3. *Distil* the reader's verdicts into the small model with calibrated abstention; *cascade* (I-06): the big reader only for the ambiguous band.
4. *Compare* with the reader, look-based detectors, the trained head, sentence vectors alone, shuffled-verdict controls.
**Proposed bars (F4).** On arithmetic wording shifts: false-new at most 0.05 and new-skill detection at least 0.90 (the reader scores 0.025 and 0.974 on dev seeds, C-91); on real banking paraphrases: false-new at most 0.05 with new-intent AUROC at least 0.90; topical novelty AUROC at least 0.95 (Q-42); detector cost at most 5% of the reader's per-message cost, measured by the harness; Q-33's criterion for a from-scratch reader (within 0.05 on B1b, B1c, macro F1, false-new).
**Cost.** L (6-10 sessions). GPU for encoder training, hours per run, resumable.
**If it fails.** Likely limiters: too little real nuance data; encoder too small; objective. First moves: reader hidden states (I-08), test-time perturbation consistency (I-10), expert-owned detectors (I-09). Outside the box: the predictive-coding stack and multi-vector or phase fingerprints (I-11, I-12) that you proposed.
**Branch point BP4.** The big reader leaves the default path only when this passes; until then it stays behind the cascade.

### M4. Router and pool at scale, with a knowledge lifecycle
**Why.** The finished model keeps growing: thousands of experts, long and messy streams, no task labels, bounded cost, and the ability to retire or correct knowledge (F5, F6).
**Needs.** M1, P2; M3 in parallel.
**Steps.**
1. *Scaling:* 100, 1,000, 10,000 experts (synthetic regimes plus real regime mixtures; tiny experts); wall-clock and memory curves for the exact gate, the shortlist, and a hierarchical or product-key router (I-01). Q-10.
2. *Routing quality:* a learned router that imitates the exact test (I-02); team routing when ambiguous (I-03); routing loss against oracle.
3. *Hard streams* (I-29): ordered, interleaved, drifting, near-twin, noisy-label; long horizon of 500 or more segments; label-free ownership (Q-08, Q-11, Q-17, Q-18); a deliberate test for loss of plasticity (RESEARCH_DIGEST section 2).
4. *Lifecycle:* dormant experts offloaded and revived (I-19); merge only when certified redundant (I-20); **exact removal** of an expert or a class with the shared components cleaned (from P2); **correction** of wrong knowledge by superseding (I-17).
5. *False spawns:* rate, cost, repair on natural streams (Q-21, Q-25, Q-26, Q-67, Q-72).
**Proposed bars (F5, F6).** Gate cost per input at 1,000 experts at most 2x its cost at 30 experts; accuracy within 2 points of oracle-routed experts at every scale tested; false spawns at most 5% of regimes after repair; after removing K classes or experts, outputs on the remaining knowledge equal those of a model never trained on the removed items (closed-form parts to numerical precision; the rest within 0.5% accuracy) and routing decisions agree on at least 99.9% of inputs; with 90% of experts dormant, accuracy loss at most 0.5% and revival latency reported.
**Cost.** L-XL (8-12 sessions). CPU mostly.
**If it fails.** Likely limiters: false spawns accumulate; refit cost; ordering effects. Options: hierarchical gates accepted by outcome (Q-62), product keys, expert caps with share-or-merge policies, learned routers, label-free ownership.
**Branch point BP5.** Decide expert granularity and the cap or merge policy from the scaling and lifecycle results.

---

## Phase D. Learn its own representation

### M5. A trained representation: a small language model, token level (Stage 2)
**Why.** Today every result stands on frozen sentence vectors, where closed-form heads win. The finished model must learn its own features, handle language as a model (perplexity, generation) and still not forget, and be cheaper than a dense model of the same quality (F1, F2, F3, F7). This is the biggest and riskiest step.
**Needs.** P1 (design choice), M2, M3 and M4 (at least their parts that the design uses).
**Design options (BP3, chosen after P1):** (a) a pool of small language models routed per sequence (SmallTalk-style); (b) one small base model with domain adapters (LoRA) plus closed-form output heads and sequence-level routing; (c) mixture-of-experts blocks inside one small transformer. (a) and (b) keep isolation and are tried first; (c) later.
**Stream and models.** At least eight text domains with distinct style and vocabulary (for example fiction, code, news, encyclopedic text, scientific abstracts, legal text, dialogue, mathematics text), learned in sequence. A **scale ladder** of three sizes (for example 25M, 60M, 150M parameters) to show a trend rather than one point. All within memory limits of this machine: sequences of 512 tokens or fewer, mixed precision, gradient checkpointing, checkpoints every 15 minutes.
**Baselines (equal compute, tuned):** dense model trained jointly on all domains (upper reference); dense sequential fine-tuning (lower reference); dense with replay and learning-rate re-warming and re-decaying (Ibrahim et al.); LoRA per domain with an oracle router; sparse-update memory finetuning (I-23); retrieval (RAG) for facts.
**Measures.** Perplexity per domain after every stage (forgetting matrix); task and knowledge probes per domain (to avoid the spurious-forgetting trap); routing accuracy; active compute and bytes per token; update cost per new domain; total parameters.
**Proposed bars (F1, F2, F3, F7).** After at least six sequential domains: mean old-domain perplexity at most 2% above its value right after learning (F1); each new domain within 5% of a dense model fine-tuned on it alone (F2); average perplexity within 3% of the dense joint model of equal *total* parameters and better than the dense model of equal *active* parameters (F3); at equal average perplexity, at least 2x less active compute and bytes per token than the dense model that reaches that perplexity, confirmed in wall-clock at batch 1 and 32 (F7); update cost per new domain at most one third of the compute-matched dense continual update at equal new-domain and equal old-domain quality (if the dense baseline cannot reach the pool's retention at any compute, say so). Total parameters reported as a ratio, never hidden.
**Cost.** XL (12-25 sessions). GPU-heavy; resumable; one job at a time.
**If it fails.** Likely limiters: small models too weak to show differences; sequence-level routing fails on mixed-domain text; adapters interfere; memory. First moves: route per segment instead of per sequence; larger base model through 4-bit adapters (1-3B) for the adapter design; surprise-gated sparse updates (I-23); slow-fast encoder with statistics adaptation (I-13).
**Answers.** Q-43 and the whole "does it survive a trained representation" question.

---

## Phase E. Generalise

### M6. Other input streams with the same engine
**Why.** "Any type of input stream" (F8). The claim is generality: the same pool code, only the front end swapped.
**Needs.** M2 (the baseline pipeline and protocol), M3 and M4. Datasets to download (your decision).
**Steps.** Add front ends for images (frozen small vision encoder; a standard class-incremental split such as CIFAR-100 or ImageNet-R, as used by SimpleCIL and RanPAC), audio (spectrogram features; a speech-commands or environmental-sound set) or time series (human-activity sensor data), and program execution traces (we have this); per-type novelty calibration; a mixed-type stream through one pool.
**Proposed bars (F8).** In at least three input types, within 1 point of the best simple baseline for that type, with hyperparameters tuned on dev seeds only and the same code; forgetting at most 1 point; unknown-detection AUROC at least the baseline's; on the mixed stream, routing accuracy at least 98%.
**Cost.** L (6-10 sessions).
**If it fails.** Per-type calibration is the usual limiter; options are per-type encoders trained for novelty (I-07 applied to the type), type-aware routers.
**Branch point BP6.** Choose the types by data availability and by how different they are from text.

---

## Phase F. Put it together, prove it, share it

### M7. The complete model: end-to-end demonstration and the full efficiency report
**Why.** This is the finish line you described: one system, one stream, every ability together, compared with dense models.
**Needs.** M3, M4, M5, and M6 for at least one non-text type.
**Steps.** Integrate into one package with one configuration. A long-horizon stream of at least 20 domains and 200 segments (text plus at least one other input type) with deliberately inserted novelty (new topics, new wordings, near-twin skills), corrections (wrong knowledge) and retirements. Anytime evaluation at checkpoints. A full efficiency report against dense and static-MoE baselines at equal quality: latency, throughput, energy, bytes read, update cost, memory growth, cost-to-quality curves for three model sizes.
**Proposed bars.** All of F1-F8 pass on fresh seeds, pre-registered, with the baselines tuned equally.
**Cost.** L-XL (8-14 sessions).
**If it fails.** Pull the failing F-statement back to its milestone and use the stuck protocol there; integration bugs are fixed first.

### M8. Independent replication, write-up, release
**Why.** A result the field can trust has been reproduced by someone else, written down with its negative results, and released (F8).
**Steps.** One-command reproduction kit (pinned environment, one script per claim, checksums). An external replication of at least three headline results by someone who did not build it. Preprints: the method paper and the methodology paper (`PAPER_NOTES.md`) and the system paper. A proper novelty check of related work (the hypothesis in RESEARCH_DIGEST section 6 gets tested here). Open release. Then **BP7**: the scale-up decision (a larger backbone through 4-bit adapters on this machine; compute grants or collaboration; our own backbone, which you said we build only once the mechanism is proven).
**Proposed bars.** At least three headline results replicated within their stated tolerances; every claim used has grade A or its real grade stated; `verify` clean; negative results included.
**Cost.** L (5-9 sessions) plus waiting time for the replicator.

---

## Totals and a caution

Early probes: P1 M, P2 S-M, P3 S. Milestones: M0 M, M1 M, M2 M-L, M3 L, M4 L-XL, M5 XL, M6 L, M7 L-XL, M8 L. Roughly **60 to 100 sessions** in all, most of it in M5. As an illustration only: at three sessions a week that is five to eight months; at five a week, three to five months.
These are estimates with wide error bars. The estimates for M5 and M7 will be redone after P1 and after M4.

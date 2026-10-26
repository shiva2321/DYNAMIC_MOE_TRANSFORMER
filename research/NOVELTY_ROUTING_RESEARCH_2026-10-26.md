# Recognising real novelty in a language stream and routing it: what we know, what the field and drive E add, ideas and a test ladder (2026-10-26)

Status: research and brainstorm, nothing built, nothing pre-registered. Growth (vertical and horizontal) is pinned on purpose: the hard part is telling WHAT is new and WHICH expert should answer.
Tags: [M] measured by us (notebook entry given); [C] claimed in a document of ours, not re-checked; [L] field source read at abstract or search-summary level; [L+] read in part; [R] recalled from memory, not rechecked; [H] hypothesis.

## 0. The short answer

1. "New" is not a property of an input. It is a relation between the input and what the system can already do. The same sentence is new to a system with no billing expert and old to one that has it; a rotated digit is not new to a classifier that already copes with rotation (drive E measured exactly this, section 3).
2. One sentence carries very little evidence about where it came from. On real text our per-sample gate has AUROC 0.65 to 0.68 and is blind to 64 to 77 % of foreign categories [M, E-024]; a test on a group of 16 or more inputs sees almost all of them [M, E-024]. The field mostly decides one utterance at a time or is handed task boundaries [L].
3. Nobody has this solved: the 2026 continual intent-discovery paper we read keeps novel precision at 0.74 to 0.88 but recalls only 0.39 to 0.61 of the novel inputs and recovers almost none of the true intent structure (NMI near 0) [L+, section 4].
4. Five levers look worth testing, in this order of cost: (a) measure which kinds of novelty each detector sees (a graded test bed), (b) a relative score against a background and a richer representation for the gate, (c) evidence accumulated over a stream instead of one input, (d) function-level novelty ("can my experts do this?") with self-supervised signals when there are no labels, (e) route a calibrated SET of experts instead of one.

## 1. What "novel" can mean in language (a working taxonomy; my guess, to be corrected by you)

| level | example | should it grow the system? |
|---|---|---|
| form | same request, different words, typos, tone, length, language register | no (it is a variation) |
| detail | same intent, new slot values or entities | no, at most a new prototype inside an expert |
| sub-topic | a new intent inside a known domain (a new kind of card problem) | maybe: a new prototype or a small new expert |
| domain | a new subject area | yes, a new expert |
| function | same inputs, a different required answer or behaviour (a rule changed) | yes, and routing by inputs alone cannot even see it |
| composition | known pieces combined in a way that needs a new behaviour | open |
| fact change | a known topic where a fact is edited or contradicted | an update, not a new expert |
| near-twin | almost the same words, different meaning (one word flips the meaning) | depends on the meaning; embeddings often get this wrong [L] |

Our streams so far tested domain-level novelty (topics, shifted means) and, in arithmetic, function-level novelty. The nuanced rows are untested.

## 2. What we already know (ours)

- Per-sample vMF conformal gate: false novelty controlled, AUROC 0.97 to 1.00 when regimes are separated [M, C-12, C-15]; blind at centre cosine 0.85 on synthetic overlap [M, E-023]; on real text blind to 64 to 77 % of foreign categories [M, E-024].
- Group test (exact permutation test on the distance of means against an expert's source windows): calibrated and powerful in a clean setting, 85 to 100 % detection of the hardest real-text pairs from 16 inputs at level 0.001 [M, E-023, E-024]. Needed in the live pool: one reference window per recent source, candidates chosen by nearest window mean, ownership by label loss, a circuit breaker. Breaks when batches are label-sorted (exchangeability violated) [M, E-023 H].
- Re-shaping the space (centring, whitening) did not help the per-sample gate; a learned metric was NOT tested [M, E-024 addendum].
- Cheap per-sample arbitration upgrades (confidence, entropy, soft mixtures) have no headroom [M, C-35].
- Routing at answer time was the largest single loss (11 to 14 points) on operator-by-operator arithmetic; a router chosen by cross-validation on the experts' own stored prototypes recovers it; about 16 prototypes per expert are enough [M, E-030, E-032, E-041].
- Label-aware spawn-or-extend (transfer test) fixes under-spawning where labels exist; it has no power when the reference expert has not learned yet (Q-25) [M, E-041].
- A wrong spawn is nearly permanent and costs about 4 points; merges removed 8 of 296 injected false spawns [M, E-034].
- Operating-point lesson: score novelty by its consequence on accuracy, not by AUROC alone (false spawn about -4 points, wrongly shared expert about -7).

## 3. What drive E adds (read only; nothing changed there)

- `E:\AI_Research\VSA_GROUND_REBUILD\RESEARCH_DOCS\results.md` (a parallel build of novelty-gated expert spawning, boundary-free streams) [C]:
  - On Split-MNIST the embedding-similarity novelty signal flagged 0 % of a genuinely new domain at every data size ("cold-start blindness": an encoder that has seen one domain has no pressure to push unseen inputs away). Adding a classification-loss z-score fixed it. Same lesson as our Q-25 and our label-aware transfer test.
  - On Rotated-MNIST only 1 expert was spawned for 5 rotations: rotation is a nuisance shift the single classifier already tolerates, so neither signal fired, and accuracy stayed high. "Novel" meant "my current model fails on it", not "it looks different".
  - On 20 Newsgroups (frozen MiniLM, 4 meta-domains, boundary-free): 54.8 +/- 10.9 % accuracy against 21.0 % for sequential training, forgetting -9.3 against +68.7, but 7.2 experts for 4 domains and 88.5 % purity; thresholds calibrated on MNIST did not transfer; "not tune once, works everywhere".
  - With a generative LoRA expert, the loss-based signal went wrong early (a strong pretrained backbone made a mismatched domain look better), so that module used embedding similarity only: a signal that works for classifiers can fail for generators.
- `E:\universal_substrait` campaign (from `research/ROUTING_RESEARCH_2026-10.md`): the cold-start monopoly (old experts win every routing contest) and the newborn disadvantage (a newborn's key is one batch's centroid) [C].
- `E:\unified`: `FamiliarityGate` (split-conformal leave-one-out nearest-neighbour distance, the same idea as ours) and `EpistemicRouter` (cascade: cheap exact lookup, then inference, then UNKNOWN with calibrated confidence) [C].
- `E:\AI_Research\Resonance_X_mainline\docs\superpowers\specs\2026-08-12-mem3-skill-router.md`: route by an index, payload stored exactly; "zero fires must be the common case"; and a fact gate whose geometric stage was measured net harmful and was NOT shipped [C]. The habit of keeping only what a measurement supports is the same as ours.
- `E:\AI_Research\nsck_rebuild\cognition\open_set_hyper_memory.py`: one open-set memory with a match threshold, entries that are never renumbered, and two evidence schemes for promoting an entry (per-environment minimum evidence; count plus age with a grace period) [C]. Same shape as our probation buffer plus evidence test.
- `E:\universal-ai-memory` ("Hive") holds a memory graph over your own AI conversations: a real, nuanced language stream. NOT opened; it is private; using it needs your say-so.

## 4. What the field says (all [L] unless marked)

Novelty and open-set detection in text
- Open intent detection learns a boundary per known intent: ADB (AAAI 2021), DA-ADB (BERT features, 2022), multi-granularity boundaries MOGB (AAAI 2025). Per-utterance decisions; one spherical boundary per class struggles to separate known from unknown.
- Continual new-intent discovery, 2026 [L+, pages 1-6 read] (Pisante and Formentin, arXiv 2609.17866): an encoder (beta-VAE on sentence embeddings) gives a latent mean and an uncertainty; a sample goes to the candidate pool only if classifier confidence is low AND uncertainty is high AND the density (a Dirichlet-process Gaussian mixture on the known intents) is low; candidates are clustered and a cluster is promoted to a new intent only if it has enough support, density and low uncertainty; replay plus EWC against forgetting; 100 intents, 50 known and 50 new in phases. Reported: accuracy 0.63 to 0.69, novel precision 0.74 to 0.88, novel recall 0.39 to 0.61, NMI and ARI near 0 (a conservative promotion policy). Their own listed open problems: threshold sensitivity and confusion between closely related intents. Same skeleton as our pool: three signals, an unknown buffer, promote only reliable clusters.
- OOD detection with pretrained transformers: Mahalanobis on hidden states (Podolskiy et al. 2021), Relative Mahalanobis = distance to the in-domain Gaussian minus distance to a background Gaussian fit on broad data, equivalent to a likelihood ratio, built to fix near-OOD failures (Ren et al. 2021, evaluated on CLINC among others), likelihood ratios for dialog (Gangal et al. 2019, first on natural language), FLatS feature-based likelihood-ratio score (2023), kNN distance (Sun et al. 2022, images).
- Stream mining: concept drift (known classes move) versus concept evolution (a class appears); OLINDDA and MINAS keep an unknown buffer, cluster it into micro-clusters and promote coherent ones to "novelty patterns"; a standard test is a drift-only stream that must NOT raise a novel class.
- Dataset-shift detection: reduce the data first, then run a two-sample test (Failing Loudly, Rabanser et al. 2019: classifier-based reduction plus a test beats raw kernel tests in high dimension); anytime-valid conformal test martingales (Vovk; the 2026 nonconform paper) turn p-values into a sequential alarm with a bounded false-alarm probability; MAPIE shows the batch-wise version.
- Surprise as a write signal: Titans (2025) stores more strongly what violates expectation, measured as a gradient of the memory loss.
- Embedding caveats: a paper on factual sensitivity reports embeddings can prefer a near-copy with one corrupted fact over a fact-preserving rewrite [L, preprint listing]; style and content need separating (Stylist, images) [L].

Expansion and routing in continual learning
- SEMA (CVPR 2025): per-adapter autoencoder descriptor, per-sample z-score of reconstruction error, expand only if ALL descriptors flag shift, task boundaries given [L+]. Remembering Transformer: a generative model judges familiarity of a new task [L]. Router drift is its own failure: SAME, PASs-MoE, StaR-MoE (2026), LEGO (ICML 2026), LLaVA-DyMoE (CVPR 2026) [L]. I found no paper combining task-free operation, finite-sample novelty control and growth.
- FLEX: gradient-free routers recover only 28 to 50 % of the gap to oracle routing; soft mixtures with a sharpened softmax beat hard top-1 [L+, from E-notes].
- Kim et al.: class-incremental accuracy = within-task prediction x task-identity prediction, and task identity is an OOD problem; good novelty detection is necessary for open-world continual learning [L].

## 5. Idea menu (each is a hypothesis; cost is rough)

I1. A graded novelty test bed first (cost: low; value: decides everything else). Build streams where every input carries a novelty LEVEL from the table in section 1, from data already cached or cheap to get: banking77 grouped by coarse topic (new intent in a known topic versus new topic), 20 Newsgroups sub-trees (comp.* versus rec.*), a paraphrase set with near-twins that flip meaning (PAWS-style) if obtainable, fact-edit pairs, and a function change (same inputs, labels remapped). Measure every detector per level: per-sample AUROC, how many grouped inputs are needed, and the consequence on accuracy.

I2. Two rulers: novelty relative to a background (cost: low). Score = how strange to my experts minus how strange to general language (Relative Mahalanobis or a vMF likelihood ratio against a broad-corpus mixture). Why it might help: removes generic noise (length, typos, register) that makes per-sample scores fuzzy. Hypothesis [H]: raises per-sample AUROC on sub-topic and detail levels; may hurt near-twins.

I3. Better eyes for the gate, separate from what the experts eat (cost: low to medium). The pool's input is a 128-d random projection of MiniLM, which discards information (E-024 caveat). Try the full vector, several layers, a larger encoder, and function-aware features: the last-token hidden state of a language model at a middle layer carries "what operation is being asked" (task or function vectors, Hendel et al. 2023, Todd et al. 2024 [R]) rather than only the topic. Hypothesis [H]: separates same-topic-different-task novelty that topic embeddings cannot.

I4. A learned invariance (cost: medium). Train a small map on known data so that paraphrases and augmentations collapse and different known intents separate (contrastive), then do novelty in that space (DA-ADB style). Our whitening and centring failed because they are not told what is nuisance; this one is told. Needs supervision or paraphrase pairs; risk: it learns the known intents only and distorts the unknown.

I5. Evidence over a stream (cost: medium; our best-proven lever). Replace "a batch of 16" by an anytime test per source (a conversation, a user, a time window, a cluster of near-duplicate contexts): conformal test martingales or CUSUM on per-sample scores, with the source windows we already keep. Gains: no batch boundary needed, bounded false alarms, a way to separate abrupt novelty from slow drift (compare with the expert's windows). Needs: p-values with the right dependence, which our first set-test attempt lacked (F-17).

I6. Function-level novelty without labels (cost: medium to high). Novel = "my experts cannot do this". With labels we have the transfer test. Without labels: each expert carries a small self-supervised competence head (masked or next-sentence prediction error, or gradient surprise as in Titans), plus expert disagreement; fuse the signals with calibrated thresholds per stage (drive E showed a loss signal that works for classifiers can mislead for generators, and embedding similarity is blind at cold start).

I7. Route a calibrated SET (cost: low, we have the pieces). Instead of top-1: a conformal set of experts whose scores clear a threshold; answer by their sharpened soft mixture; empty set = abstain or escalate; two near-equal experts = a flag for "maybe one concept split in two" (a merge candidate). Prediction [H]: recovers part of the FLEX oracle gap on ambiguous inputs.

I8. Granularity and cost cascade (cost: medium, design). Three levels of response: a new prototype inside an expert (detail, sub-topic), a new expert (domain, function), abstain or ask (unknown unknown). A cheap embedding check first, a costly verifier (an entailment or cross-encoder check against the nearest stored exemplar) only on the uncertain band; this targets near-twins.

I9. Delayed supervision (cost: medium). Keep uncertain items in probation until feedback arrives (a user correction, an outcome), then decide; stream-mining calls this verification latency. Our probation buffer is the place for it.

Not worth repeating (measured or well known): more geometry reshaping of the space (E-024), per-sample arbitration upgrades (C-35), a trained router with no drift protection (router drift [L]), spawn on a single batch (F-17, drive E bug 1).

## 6. Test ladder (draft; decision rules to be fixed before any run)

- N0 (hours, CPU): build the graded test bed (I1) and a table "detector x novelty level" for: our vMF conformal gate, kNN distance, Mahalanobis, Relative Mahalanobis, classifier confidence, an autoencoder descriptor (SEMA style), and the group test at sizes 8, 16, 32, 64. Decision: name the blind levels and the group size each needs. Kill rule: if every detector is blind on a level at 64 inputs, that level needs a different signal (I3, I4, I6), not a better threshold.
- N1: for the worst blind level pick the cheapest of I2, I3, I4 and test it against the N0 table on fresh seeds with a pre-declared bar (for example per-sample AUROC +0.10 without a worse false-novelty rate).
- N2: put the winner into the live pool on a real class-incremental text stream (banking77 and 20 Newsgroups, with the plain kNN as the baseline we must beat or explain) and measure consequences on accuracy, not only detection.
- Later, if N0 to N2 pass: I5 (streams without batches), I7 (set routing), I6 (no labels).

## 7. Questions for the user

- Two or three real examples from your own use where an input must count as truly new, with a near-twin that must count as the same thing said differently.
- Is the target a classifier-like system (intents, labels), a generator (an LLM with adapters), or both?
- May I use your own conversation memory (`E:\universal-ai-memory`) as a real stream, read-only? Not opened yet.

## Sources

- Continual intent discovery 2026: https://arxiv.org/pdf/2609.17866
- Open-world continual learning (Kim et al.): https://ar5iv.labs.arxiv.org/html/2304.10038
- ADB: https://arxiv.org/pdf/2012.10209 ; DA-ADB: https://arxiv.org/pdf/2203.05823 ; MOGB: https://ojs.aaai.org/index.php/AAAI/article/view/34630
- Mahalanobis for transformer OOD: https://ar5iv.labs.arxiv.org/html/2101.03778 ; Relative Mahalanobis: https://ar5iv.labs.arxiv.org/html/2106.09022 ; likelihood ratios for dialog: https://arxiv.org/pdf/1912.12800 ; FLatS: https://arxiv.org/pdf/2310.05083 ; deep kNN OOD: https://arxiv.org/pdf/2204.06507
- Failing Loudly: https://arxiv.org/pdf/1810.11953 ; conformal martingales and nonconform: https://arxiv.org/pdf/2411.11824 , https://arxiv.org/pdf/2605.13642 ; MAPIE exchangeability testing: https://mapie.readthedocs.io/en/stable/generated/exchangeability_testing/1-quickstart/plot_exchangeability_online/
- MINAS and novelty in data streams: https://datalearner.com/academic/journal-papers/1384-5810/volumes-and-issues/30/paper-detail/9590 ; drift versus evolution: https://arxiv.org/pdf/1704.00023
- SEMA: https://arxiv.org/pdf/2403.18886 ; Remembering Transformer: https://arxiv.org/pdf/2404.07518 ; LEGO: https://icml.cc/virtual/2026/67405 ; StaR-MoE: https://arxiv.org/pdf/2605.17571 ; LLaVA-DyMoE: https://cvpr.thecvf.com/virtual/2026/poster/37335
- Titans: https://arxiv.org/pdf/2501.00663
- Embedding and style caveats: https://neurips.cc/virtual/2026/164747 ; https://arxiv.org/html/2310.03738v2
- Ours: notebook entries E-023, E-024, E-030, E-032, E-034, E-041; research/GATEKEEPER_RESEARCH_2026-10.md; research/ROUTING_RESEARCH_2026-10.md
- Drive E: AI_Research/VSA_GROUND_REBUILD/RESEARCH_DOCS/results.md; AI_Research/nsck_rebuild/cognition/open_set_hyper_memory.py; AI_Research/Resonance_X_mainline/docs/superpowers/specs/2026-08-12-mem3-skill-router.md; universal_substrait and unified as summarised in research/ROUTING_RESEARCH_2026-10.md

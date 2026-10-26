# Related work for knowledge status and reuse (Task 1, P0), 2026-10-26

Purpose: before any code, know what is already done, what the closest papers do NOT test, and what we may honestly call new (spec section 10).

How it was read. Each paper was fetched as text (ar5iv or arXiv HTML, or the PDF for 2609.17866) and read through a reader model that was asked five fixed questions; I then checked the answers against each other and against the numbers already in our notebook. I did not re-verify every number line by line. Tags: [full] = the whole text was returned; [main text] = the main text was returned but the appendix or supplement was cut off (stated per paper); [abstract] = only the abstract or a listing was reachable. Where a paper says what it does not test, that is the paper's own statement unless marked "(inferred)".

## 1. Telling a new intent from a known one (text)

**TOP-ID, open intent discovery, arXiv 1904.08524 [full]**
- Does: extracts free-form (Action, Object) intent phrases with a CRF, in two stages (does an intent exist, then which words); several intents per utterance (up to 3); no fixed class list.
- Data/metrics: 25K crowd-annotated Stack Exchange questions; precision, recall, F1 for Action, Object and full intent; GloVe cosine "semantic similarity".
- Numbers: Stage I F1 0.91; Stage II intent F1 0.76, similarity 0.86; held-out domains F1 0.635 to 0.79.
- Does NOT test: unseen-intent recall as a separate metric; paraphrase or de-duplication of intents; modern classifiers (not comparable); Stage I labels (question against answer) may leak topic and style.
- Versus us: the only paper here that handles several intents per message, but it extracts phrases and has no notion of known, thin or new.

**Uncertainty-aware continual new-intent discovery, arXiv 2609.17866 [full, read as PDF]**
- Does: beta-VAE on sentence embeddings; a sample is a novelty candidate only if classifier confidence is low AND latent uncertainty is high AND a DP-GMM density on known intents is low; candidates are clustered by a second DP-GMM and a cluster becomes a new intent only if it has enough support, density and low uncertainty; replay (5,000, 95 % hard) plus EWC.
- Data/metrics: 100 intents, 50 known and 50 new in phases (three-phase and five-phase); accuracy, novel precision, recall, F1, NMI, ARI; 5 seeds.
- Numbers: three-phase novel precision 0.74 to 0.88, recall 0.39 to 0.61, F1 0.54 to 0.69, NMI and ARI about 0; five-phase precision 0.84 to 0.88, recall 0.51 to 0.57, F1 0.64 to 0.69; accuracy 0.60 to 0.69; forced assignment of ambiguous inputs hurts (F1 0.69 to 0.56).
- Does NOT test (their own words): threshold sensitivity and "limited separation of closely related intents" are left to future work; no calibration guarantee; no graded thin or related status; one decision per utterance.
- Versus us: the same skeleton as our pool (three signals, an unknown pile, promote only reliable clusters). Our additions would be exact-ground-truth wording controls, finite-sample calibration, group tests, thin against solid.

**ADB, adaptive decision boundary, arXiv 2012.10209 [full; figure values not readable]**
- Does: after supervised training on known intents (BERT, last layer tuned) learn one spherical boundary per known class; outside every ball means "open".
- Data/metrics: BANKING (77 intents), OOS (150), StackOverflow (20); known ratio 25, 50, 75 %; accuracy and macro-F1; 10 runs.
- Numbers: at 25 % known, accuracy 78.85 / 87.59 / 86.72 against the best baseline 64.21 / 81.43 / 47.84; open-class F1 up to 91.84.
- Does NOT test: multi-intent messages; separating or naming different unseen intents (one open bucket); near-twin or paraphrase hard negatives; other encoders; the false-open rate on variants of known intents.
- Versus us: the failure we care about (a variation landing outside its ball and being called new) is exactly what it does not measure; we measure it.

**DA-ADB, arXiv 2203.05823 [full]**
- Does: ADB plus distance-aware scaling (samples near two centroids get smaller magnitude) and a cosine classifier.
- Data/metrics: same three datasets, 10 seeds, t-tests.
- Numbers: mean accuracy BANKING 81.27 against 80.11 (ADB) and 76.84 (best baseline); the stated gain on OOS (2.25 %) could not be reproduced from the table by the reader (about 2.5 to 2.7).
- Does NOT test: multi-intent; near-twin confusability (it is expected to be weakest where centroids are close); one open bucket; alpha chosen on the evaluation set.
- Versus us: a stronger single-utterance baseline to include.

**Revisiting Mahalanobis distance for transformer-based out-of-domain detection, arXiv 2101.03778 [full]**
- Does: fine-tune an encoder on in-domain intents, score by (marginal or class-based) Mahalanobis distance, optionally on trailing principal components.
- Data/metrics: CLINC150, ROSTD, SNIPS; AUROC, AUPR, FPR@95; 10 runs.
- Numbers: CLINC150 AUROC 98.4 (RoBERTa + Mahalanobis) against 97.1 (MSP); SNIPS 97.6 with large variance.
- Does NOT test: layer or pooling ablations; near-OOD beyond qualitative errors; decoder models. Own stated hard case: "semantically similar utterances, of which one is in ID and the other is OOD".
- Versus us: a baseline (Mahalanobis); its error examples are our near-twin problem.

**Relative Mahalanobis, arXiv 2106.09022 [full]**
- Does: subtract the distance to a background Gaussian fit on all training features (a per-class likelihood ratio, no tuned parameter) to fix near-OOD failures.
- Data/metrics: CIFAR-100 against CIFAR-10 and the reverse, genomics, CLINC; AUROC.
- Numbers: CLINC with BERT, no fine-tuning, MD 75.48 to RMD 91.98; from scratch CIFAR-100 vs CIFAR-10 74.91 to 81.01; after fine-tuning the gain mostly disappears (ViT 99.87 vs 98.82).
- Does NOT test: true near-twins at fine granularity; error bars; non-Gaussian backgrounds (one flow tried once).
- Versus us: idea I2 of the research note; include as a baseline with the background drawn from general text.

## 2. Testing a group, and novelty in streams

**Failing Loudly, arXiv 1810.11953 [full]**
- Does: reduce the data (PCA, random projection, autoencoders, classifier softmax BBSDs, domain classifier) then run two-sample tests (MMD, per-dimension KS with Bonferroni, chi-squared, binomial).
- Data/metrics: MNIST, CIFAR-10 and variants; shifts (noise, adversarial, rotation, knock-out); detection rate at alpha 0.05, 5 runs; no text.
- Numbers: best univariate BBSDs detects 0.19 of shifts with 10 target samples, 0.47 with 100, 0.70 with 1,000; small shifts and a 10 % perturbed fraction are hardest.
- Does NOT test: text; online or sequential testing; single-sample detection (out of scope).
- Versus us: the precedent for group tests; its power curve matches our E-024 finding that groups of 16 or more are needed, and BBSDs (classifier outputs as the reduced space) is a candidate reduction to try for our group test.

**MINAS, Faria, Carvalho, Gama (DMKD 2016) [abstract only; the full text was not reachable]**
- Does (from the abstract): keeps an "unknown" set of examples from the stream, clusters them into micro-clusters and promotes valid groups to "novelty patterns" added to the model; separates novelty from drift; extends OLINDDA to several classes.
- Does NOT test: cannot be judged without the full text.
- Versus us: the precedent of our probation buffer plus evidence test; to be read in full before any claim that the buffer idea is new.

## 3. Growing expert systems and routing

**SEMA, arXiv 2403.18886 [main text; the last 11,579 characters were not read]**
- Does: per adapter a functional adapter plus an autoencoder "representation descriptor"; expand a layer only if ALL descriptors flag the input (z-score of reconstruction error); at most one adapter per layer per task, decided in the first epoch; softmax router with a column added per expansion.
- Data/metrics: CIFAR-100, ImageNet-R (5, 10, 20 tasks), ImageNet-A, VTAB; class-incremental, rehearsal-free; final accuracy.
- Numbers: CIFAR-100 86.98 against 86.11 (CODA-P); ImageNet-A 53.32 against 49.24; soft router beats top-1 (53.32 against 50.56); 0.56 M added parameters against 1.90 M.
- Does NOT test: text; task-free streams (task boundaries are used); near-twin shifts; several things in one input; no variance across seeds reported.
- Versus us: the nearest relative of our growth decision; per-sample z-scores with no finite-sample guarantee.

**Remembering Transformer, arXiv 2404.07518 [full]**
- Does: a frozen ViT with per-task LoRA adapters; per-task tiny autoencoders; the lowest reconstruction loss gives a hard one-hot route.
- Data/metrics: split CIFAR-10 and CIFAR-100, permuted MNIST; average accuracy, 3 seeds.
- Numbers: average 88.43 against 72.53 (FeTrIL); 0.22 M parameters only in the 3-adapter setting.
- Does NOT test: task-free streams, text, subtle novelty, mixed inputs; no threshold, so the novelty score never decides "unknown" (a new adapter is added for every task).
- Versus us: a familiarity signal without calibration; the gap between best and second-best loss could be a confidence margin (not evaluated by them).

**FLEX, arXiv 2608.01437 [main text; the supplement was not read]**
- Does: a 34-task benchmark without textual "fingerprints"; frames routing as task-as-class classification and compares gradient-free routers (nearest class mean, hierarchical clustering, SOINN, RanPAC, DDAS) plugged into four progressive-LoRA systems.
- Numbers: the best router recovers 47.6 % (DISCO), 50.2 % (SAME), 28.3 % (PureLoRA) and 29.2 % (HiDe-LLaVA) of the gap to oracle routing; soft routing at a sharper temperature gains 1.67 to 2.0 points with the same top-1.
- Does NOT test: inputs that mix tasks; rejecting a truly novel input (every router normalises over known tasks, so there is no "none of the above").
- Versus us: confirms routing is not solved and that "none of the above" is missing in the field; supports idea I7.

**Open-world continual learning, Kim et al., arXiv 2304.10038 [main text; the last half, including the MORE method, was not read]**
- Does: proves that good within-task prediction and good closed-world OOD detection are necessary and sufficient for good class-incremental learning (CIL error at most within-task error plus task-id error); builds HAT+CSI and Sup+CSI on a contrastive OOD method.
- Data/metrics: CIFAR-10, CIFAR-100, Tiny-ImageNet, 5 seeds; CIL accuracy and per-task AUC.
- Numbers: on C10-5T HAT+CSI 87.8 against 66.0 (DER++).
- Does NOT test: text; subtle novelty; task-free streams; acquiring labels for novel items; no category for "known but thin" (their bounds concern errors, not how much evidence a class has).
- Versus us: the theoretical reason novelty detection is the core problem; the "thin" status is our own extension.

**Component-based OOD (CoOD), arXiv 2604.21546 [main text; the last 71,413 characters were not read]**
- Does: training-free; an LLM names visual components per class, GradCAM masks isolate them, a Component Shift Score and a Compositional Consistency Score flag known parts in an invalid arrangement.
- Data/metrics: ImageNet-1K, CUB, ObjectNet, a diffusion counterfactual set; AUC and FPR at 95 % recall.
- Numbers: CUB FPR95 22.7 against 73.0 (MCM); compositional ImageNet AUC 75.4 against 70.0; 0.85 to 1.48 frames per second against 89 for MCM.
- Does NOT test: language; other backbones; its own authors treat the residual (non-component) signal as weak evidence.
- Versus us: the nearest "known parts, unknown combination" method (images only); warns that residual-based scoring may carry little signal, so our residual claims must be tested, not assumed.

## 4. Embeddings, composition and reuse

**Focus On Facts (FOF), NeurIPS 2026 listing and model card [abstract only]**
- Reports: a base embedding model prefers a near-copy with one corrupted fact over a fact-preserving rewrite (99.58 % of held-out triplets); fine-tuning with a contrastive objective lowers that to 29.37 %; the tuned model "penalizes omission" and transfers poorly to directional tasks.
- Does NOT: the paper itself was not read; treat the numbers as a listing's claim.
- Versus us: the evidence behind our worry that off-the-shelf embeddings blur near-twins; R5 is our version of the same cure.

**SCAN, Lake and Baroni, arXiv 1711.00350 [full]**
- Does: a synthetic grammar of commands with exact action sequences; seq2seq RNNs on random, length and primitive splits.
- Numbers: random split 99.7 to 99.8 %; length split 13.8 to 20.8 %; "jump" primitive only in isolation 0.08 % (adding 8, 16, 32 composed examples gives 38.3, 77.8, 88.4 %).
- Does NOT test: arithmetic; transformers or convolutional models; any fix (only proposals).
- Versus us: the standard evidence that learners fail to reuse known parts in new combinations; hold out the longest or largest cases when we test composition.

**DreamCoder, arXiv 2006.08381 [full]**
- Does: wake-sleep program learning; abstraction sleep adds library routines that compress solved programs; recognition network trained on replay and "fantasy" programs.
- Numbers: text editing 3.7 % to 79.6 % solved in 10 minutes; physics laws 93 % after 8 cycles; list processing about 20 learned routines.
- Does NOT test: natural-language messages; noisy surface variation; deciding what counts as new (compression is the only criterion; tasks are supplied by the experimenters); heavy compute.
- Versus us: the closest "build a new skill from known skills shown in solved examples"; ours would add the status per part and the check that a trace's steps are verified instances of known skills.

## 5. What is and is not new in this design (spec section 10)

- C-a, a test bed with exact ground truth for wording, function, parts, taught state and relation: PARTLY NEW. Intent benchmarks give known against one open class from random class splits and report no false-open rate on variants (ADB, DA-ADB, 2609.17866); TOP-ID has several intents per message but no novelty metric; SCAN has exact ground truth for composition but no novelty or wording families; FLEX removes fingerprints but has no mixed inputs or rejection. A graded status with wording families, near-twin controls, tangled parts and trace relations in one bed was not found. Decided by ADB, DA-ADB, SCAN, FLEX, TOP-ID.
- C-b, an information-ceiling analysis of whether representations can separate wording changes from function changes: PARTLY DONE. Embedding near-twin failures for factual edits are reported (FOF, abstract level) and Mahalanobis errors on similar intents are noted (2101.03778); a held-out-wording probe and a distance-to-taught-set test for operator changes in arithmetic were not found. Decided by FOF (abstract only: read the paper before claiming) and 2101.03778.
- C-c, a status machine with calibrated false novelty that beats the baselines: UNPROVEN, not found elsewhere in what I read. Three-signal fusion with conservative promotion exists (2609.17866: recall 0.39 to 0.61); calibration guarantees and a "thin" status were not found; SEMA and Remembering Transformer use uncalibrated reconstruction errors; RMD and Mahalanobis are strong single-utterance baselines that we must beat or explain. Decided by 2609.17866, SEMA, RMD.
- C-d, relation from worked traces (steps verified as instances of known skills, parents named): PARTLY DONE in program synthesis (DreamCoder reuses learned routines; SCAN shows learners fail without such structure) and not found for natural-language messages with a status per part. Decided by DreamCoder and SCAN.
- Group evidence: ALREADY KNOWN as a principle (Failing Loudly; task-level routing in our research note) and already measured by us on real text (E-024); our contribution would be the graded evidence needed per group size, not the idea.

What this changes in the plan. (1) Add Relative Mahalanobis (RMD) and the classifier-output reduction (BBSDs) to the baselines of Task 10. (2) The false-open rate on rewordings is the headline metric we report and the field does not (Task 11). (3) Read the FOF paper and MINAS in full before any paper draft; both are abstract-only here. (4) State "residual evidence may be weak" (CoOD) as a risk and test it in B4a before relying on it. (5) Keep composition tests on the longest or largest cases (SCAN).

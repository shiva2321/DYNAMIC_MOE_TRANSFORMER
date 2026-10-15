# GrowMoE stage 4: what the literature says, and what we take from it (2026-10-15)

Purpose: the root-cause study (E-030) found four faults (routing at answer time, the spawn-or-extend test, the growth decision, no split rule for stationary mixtures). This note records what
I looked up for each, what each source actually supports, and where the sources do NOT answer our question. Every reference below was found and checked by search on 2026-10-14/15; where I
could read the paper itself (self-expanding networks, GradMax) the statement comes from the text, otherwise from the abstract and listing and is marked (abstract). Nothing here replaces
the measurements: the stage-4 design rests on our own spikes (`research/growmoe_results/stage4_spikes/RESULTS_SPIKES.md`).

## 1. Routing: which expert answers (stage 4a)

| Source | What it says | What we take |
|---|---|---|
| Kim, Xiao, Konishi, Ke, Liu. *A Theoretical Study on Solving Continual Learning*. NeurIPS 2022. [arXiv 2211.02633](https://arxiv.org/abs/2211.02633) (abstract) | Class-incremental accuracy splits into within-task prediction (WP) and task-id prediction (TP); TP is tied to out-of-distribution detection; good WP and good TP are necessary and sufficient. | Our decomposition (expert quality x routing) is the standard one; routing deserves its own component and its own measurement. |
| Lee, Ha, Zhang, Kim. *A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning* (CN-DPM). ICLR 2020. [arXiv 2001.00689](https://arxiv.org/abs/2001.00689) (abstract and listing) | Expansion-based task-free learning with expert networks; gating through a generative model per expert; the authors name the accuracy of expert selection as the main bottleneck of their method. | Same finding as ours (E-030: routing costs 12-14 points); a generative selector alone is the weak point. |
| Aljundi, Chakravarty, Tuytelaars. *Expert Gate*. CVPR 2017. [arXiv 1611.06194](https://arxiv.org/abs/1611.06194) (abstract) | One undercomplete autoencoder per task; the expert whose autoencoder reconstructs the input best is used. | The density gate of GrowMoE is of this family (generative, one model per expert). |
| Jacobs, Jordan, Nowlan, Hinton. *Adaptive mixtures of local experts*. Neural Computation 3(1), 1991. [doi 10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79) (listing) | A gating network trained together with the experts decides which expert handles which case (competitive learning). | A DISCRIMINATIVE gate is the classical mixture-of-experts design; ours learns it from what the pool already stores. |
| Hayes, Kanan. *Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis*. CVPR Workshops 2020. [arXiv 1909.01520](https://arxiv.org/abs/1909.01520) (abstract) | Streaming LDA (class means and one shared covariance, updated one instance at a time) as a continual-learning classifier; beats incremental batch learning and other streaming methods on ImageNet and CORe50. | The LDA candidate of our router: closed form, no gradient steps, no forgetting, grows by adding a class mean. |
| Li et al. *Theory on Mixture-of-Experts in Continual Learning*. ICLR 2025. [arXiv 2406.16437](https://arxiv.org/abs/2406.16437) (abstract) | In an overparameterised linear model, experts specialise by task and the router learns the assignment; the gating network has to stop updating after enough rounds for the system to converge; more experts need more rounds. | A router must be able to freeze or slow down once it has converged; we refit on events and on a schedule and keep density as the default. |
| LLaVA-CMoE (2025), LEGO (ICML 2026 listing) (abstracts) | New experts are given their own routing pressure; shared routers drift and forget; some methods pick a router at inference with a reconstruction criterion. | Router drift is a known failure; our router is refit from stored prototypes, so it cannot forget what the experts still store. |

Not found: a paper that combines a learned router, a task-free stream and a principled expansion rule in a small mixture of experts (searched 2024-2026 listings; the results are mostly task-aware
or large-model work). This is a gap, not a proof of novelty.

## 2. Spawn or extend (stage 4b)

| Source | What it says | What we take |
|---|---|---|
| Aljundi, Kelchtermans, Tuytelaars. *Task-Free Continual Learning*. CVPR 2019. [arXiv 1812.03596](https://arxiv.org/abs/1812.03596) (listing) | Plateaus and peaks of the loss decide when to update importance weights without task boundaries. | Loss is the standard handle on "the function changed"; our competence monitor is of this kind. |
| Gama, Sebastiao, Rodrigues. *On evaluating stream learning algorithms*. Machine Learning 90, 2013, [doi 10.1007/s10994-012-5320-9](https://doi.org/10.1007/s10994-012-5320-9) (abstract) | Prequential error (each sample tested before it is trained on) with forgetting (fading factors or a window) converges to the Bayes error on stationary data and detects change fast; fading factors need no stored window. | The honest reference for "how well does the expert predict fresh data" is its prequential loss: the transfer test uses a window of it. |
| Kim et al. 2022 (above) | WP and TP. | The transfer test asks the WP question of the existing expert on the new data before anything is trained. |

No source gives a spawn-or-extend rule that compares an existing expert's zero-shot loss on a new labelled cluster with its own prequential loss; this part is our design, justified by the spike (B).

## 3. Growing capacity as needed (stage 4c)

| Source | What it says | What we take |
|---|---|---|
| Mitchell, Menzenbach, Kersting, Mundt. *Self-Expanding Neural Networks* (read in full, v3, 2024). [arXiv 2307.04526](https://arxiv.org/abs/2307.04526) | Answers when, where and what together with a natural expansion score eta = g^T F^-1 g (the part of the output gradient the current parameters can express); adds width by concatenation with zero outgoing weights, depth by an identity-initialised layer; adds capacity when the relative increase exceeds tau and the absolute increase exceeds alpha; proves the number of simultaneous additions is bounded; a cheap lower bound uses KFAC factors and the RESIDUAL gradient; pruning is reversed addition; reports any-time accuracy and sizes that follow the data. | The principle: judge need from the TRAINING signal (what the current parameters cannot yet express), not from a held-out probe; zero-outgoing-weight widening is what `MLPExpert.widen` already does. Our spike C tried a simplified version of their score (new random units explaining the residual gradient) and it did NOT separate starved from adequate experts; the faithful KFAC version is future work (section 6 of the spec). |
| Evci, van Merrienboer, Unterthiner, Vladymyrov, Pedregosa. *GradMax*. ICLR 2022. [arXiv 2201.05125](https://arxiv.org/abs/2201.05125) (read in full) | New neurons get zero incoming weights and outgoing weights equal to the top singular vectors of E[dL/dz x^T], so the gradient of the new weights is maximal; the singular values could guide when and where to grow (left to future work). | Tried in spike C as an initialisation: not better than random-in / zero-out for our experts (early trial gain worse by 0.3-1.0 %). Not adopted. |
| Wu, Wang, Liu. *Splitting Steepest Descent*. NeurIPS 2019 [arXiv 1910.02366](https://arxiv.org/abs/1910.02366); Wu, Liu, Stone, Liu. *Firefly Neural Architecture Descent*. NeurIPS 2020 [arXiv 2102.08574](https://arxiv.org/abs/2102.08574) (abstracts) | Grow by splitting neurons or adding candidates chosen with a Taylor approximation of the loss; width and depth; used for continual learning. | Alternative growth criteria; the splitting idea is the neuron-level cousin of the expert-level split of stage 4d. |
| Chen, Goodfellow, Shlens. *Net2Net*. ICLR 2016. [arXiv 1511.05641](https://arxiv.org/abs/1511.05641) (abstract) | Function-preserving transformations that make a network wider or deeper without changing its output. | The basis of `widen` and `deepen`; verified exactly function-preserving in E-030 (maximum output change 0.0). |

Limit of the evidence: all these papers measure TRAINING-side criteria on fixed or sequential benchmarks; none is run with a streaming recipe that sees each batch once, so the thresholds
cannot be taken over. Our spike C2 measured the signal we propose (the within-batch fit gap) directly.

## 4. Splitting a stationary mixture (stage 4d, research) and the streaming recipe (4e, research)

| Source | What it says | What we take |
|---|---|---|
| Jain, Neal. Split-merge MCMC for Dirichlet process mixtures. J. Comput. Graph. Stat. 13, 2004; Bayesian Analysis 2(3), 2007 [doi 10.1214/07-BA219](https://doi.org/10.1214/07-BA219) (listing) | Metropolis-Hastings moves that split or merge whole components escape local modes that one-point-at-a-time updates cannot. | The shape of a principled split move: propose, test, accept or reject. |
| Jacobs et al. 1991 (above) | Competition between experts for each case splits a task into subtasks. | Why a split may be discoverable from the loss structure of one expert's data. |
| Defazio et al. *The Road Less Scheduled* (Schedule-Free learning). NeurIPS 2024. [arXiv 2405.15682](https://arxiv.org/abs/2405.15682) (abstract) | Matches or exceeds cosine schedules without knowing the stopping step, by iterate averaging. | A candidate for the missing learning-rate schedule of a stream whose length is unknown (a decaying rate gave +4 points in E-030). |

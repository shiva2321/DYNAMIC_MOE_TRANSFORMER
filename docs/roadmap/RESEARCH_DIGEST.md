# Research digest: what the field has, and what it changes in our plan

Version 0.1, 2026-11-02. Part of the road plan ([ROADMAP.md](ROADMAP.md)).

**In plain words.** This is a map of the work other people have done on the same two problems (learning continually without forgetting, and doing it cheaply),
read for one purpose: to decide what we must build, what we must beat, and what we should not redo. It is not a full literature review.

**How much to trust each entry.** The tags follow the lab notebook (`docs/labnotebook/REFERENCES.md`):
**L** = read at abstract or search-summary level on 2026-11-02 (the paper itself was not read); **R** = recalled from memory and not rechecked; **V** = verified in full (none yet).
A few search summaries came from blog posts; those are marked "secondary". Every entry that a milestone depends on must be upgraded to **V** (read in full) before that milestone starts; section 7 says which.
Papers from 2026 are cited by the identifiers the search returned; they were not independently confirmed.

---

## 1. The families of continual learning, and where our design sits

| Family | The idea in plain words | Strength | Weakness | How it relates to us |
|---|---|---|---|---|
| **Replay** (keep or regenerate old data and mix it in) | Rehearse the past while learning the present | Simple and strong. At a fixed compute budget, plain replay beat every fancier method tested (Prabhu et al., CVPR 2023, [arXiv 2303.11165](https://arxiv.org/abs/2303.11165), L). With abundant storage the real limit is compute time (Cho et al., ICLR 2026, [arXiv 2502.07274](https://arxiv.org/abs/2502.07274), L). For language models, re-warming and re-decaying the learning rate plus replay matched full retraining at 405M and 10B parameters (Ibrahim et al., [arXiv 2403.08763](https://arxiv.org/abs/2403.08763), L) | Costs compute on every update; forgetting is reduced, not removed; plasticity falls as memory grows (Cho et al.) | **This is the baseline we must beat on cost.** The honest comparison is not "retrain from scratch" but "continue training with replay and a good learning-rate schedule" |
| **Parameter isolation and expansion** (separate experts or adapters per task, grown as needed) | Give new knowledge its own parameters so old ones are untouched | No forgetting by construction for what is isolated | Needs a router to pick the right part at test time; growth of memory and compute; router drift. Relatives: Expert Gate ([arXiv 1611.06194](https://arxiv.org/abs/1611.06194), L); SEMA ([arXiv 2403.18886](https://arxiv.org/abs/2403.18886), L+ in our notebook); Hierarchically Gated Experts ([arXiv 2412.17188](https://arxiv.org/abs/2412.17188), L: task-free online setting, claims comparable accuracy more efficiently, no numbers on the abstract page); CoPE (ICML 2026, [page](https://chatpaper.com/paper/330083), L secondary: probe experts decide when to add experts, VAE-based task locator, replay-free, vision-language tasks); PASs-MoE ([arXiv 2601.13020](https://arxiv.org/abs/2601.13020), L: "misaligned co-drift" when router and experts are updated together); BranchLoRA (ACL 2025, L); a mixture of curriculum LoRA experts with per-task autoencoders as routers ([arXiv 2506.11672](https://arxiv.org/abs/2506.11672), L); parameter isolation and combination for language models ([arXiv 2502.10966](https://arxiv.org/abs/2502.10966), L) | **Our family.** The idea is not new (our own notebook records this: C-29). What has to be shown is our specific pieces and an honest cost accounting |
| **Closed-form and prototype methods on frozen features** | Keep a pretrained encoder fixed; learn the last layer by a formula that cannot forget | Provably or practically forgetting-free for the head; very cheap. A frozen pretrained model plus a prototype classifier beats many sophisticated methods (SimpleCIL, [arXiv 2303.07338](https://arxiv.org/abs/2303.07338); RanPAC, [arXiv 2307.02251](https://arxiv.org/abs/2307.02251); Continual-CLIP, [arXiv 2210.03114](https://arxiv.org/abs/2210.03114); all L). Analytic recursive least squares: REAL ([arXiv 2403.13522](https://arxiv.org/abs/2403.13522), L), VILA ([arXiv 2602.13670](https://arxiv.org/abs/2602.13670), L: "theoretically zero forgetting" but approximation errors build up on long sequences). Mahalanobis and covariance methods: FeCAM ([arXiv 2309.14062](https://arxiv.org/abs/2309.14062), L), AdaGauss ([arXiv 2409.18265](https://arxiv.org/abs/2409.18265), L: covariances drift and must be re-adapted when the backbone changes) | Limited by the frozen features; no new representations | **Our closed-form (LDA) head belongs here, and so does our biggest measured win.** It means our head is a known idea; and it means the pool must justify itself by what these methods do not do (novelty, lifecycle, trained representations) |
| **Sparse, memory-style and multi-timescale updates** | Update only the few parameters a new fact touches; let different parts learn at different speeds | Sparse memory finetuning: on a stream of facts, held-out QA dropped 89% with full finetuning, 71% with LoRA and 11% with the sparse method (authors' numbers; Lin et al., Meta, [arXiv 2510.15103](https://arxiv.org/abs/2510.15103), L; facts only). Nested Learning, with a "continuum memory system" of parts that update at different frequencies, and the HOPE model ([arXiv 2512.24695](https://arxiv.org/abs/2512.24695), NeurIPS 2025, L; most detail from blog posts). PEER, retrieving from over a million tiny experts with product keys ([arXiv 2407.04153](https://arxiv.org/abs/2407.04153), L) | Evidence so far is on facts or language modelling, not on routing plus novelty | **Our "fast notebook + slow network" split is the same idea at a small scale.** PEER is the best-known answer to "routing cost must not grow with the number of experts" |
| **Model merging and consolidation** | Fold several specialists into one | Keeps memory flat. MINGLE ([arXiv 2505.11883](https://arxiv.org/abs/2505.11883)), SLAO ([arXiv 2512.23017](https://arxiv.org/abs/2512.23017)), aligned merging ([arXiv 2506.03189](https://arxiv.org/abs/2506.03189)), all L | Merge order and over-trained experts hurt; naive merging fails without alignment (MC-SMoE, in our merge notes, V for the alignment passages) | Relevant to our stage-5 merge repair and to bounding memory |
| **Retrieval (RAG)** | Do not teach the model the fact; look it up | For new facts, retrieval beat unsupervised fine-tuning (Mistral-7B: 0.875 against 0.504 on a current-events task; Ovadia et al., EMNLP 2024, [arXiv 2312.05934](https://arxiv.org/abs/2312.05934), L) | Weak for skills and reasoning (the sparse-memory paper makes this point) | **The practical competitor for "keep the model's knowledge current".** It must be a baseline for any claim about learning new facts; retrieval can also be an expert type |

---

## 2. What the field says about the problem itself

1. **Compute, not storage, is the constraint.** Prabhu et al. and Cho et al. (above). Consequence: our efficiency claim has to be a compute claim measured against replay, not a memory-saving claim.
2. **Continual pre-training with replay already matches retraining.** Ibrahim et al. (above). Consequence: "cheaper than retraining" is not a contribution; "cheaper than the best replay-based update" would be.
3. **Forgetting follows scaling laws.** It grows as a shifted power law in the number of fine-tuned parameters and update steps (Kalajdzievski, [arXiv 2401.05605](https://arxiv.org/abs/2401.05605), L); mixing even 1% of pre-training data into the fine-tuning set cuts it sharply (Bethune et al., [arXiv 2505.07796](https://arxiv.org/abs/2505.07796), L). Searches did not show that larger models forget less. Consequence: the problem does not go away with scale, so a small-scale result is informative, but we should expect to extrapolate carefully.
4. **Not every drop is forgetting.** "Spurious forgetting" ([arXiv 2501.13453](https://arxiv.org/abs/2501.13453), L) argues that much of the measured drop in language models is lost task alignment, not lost knowledge. Consequence: measure forgetting with more than one probe (perplexity, task accuracy, knowledge probes).
5. **Deep networks can lose the ability to learn.** Dohare et al., Nature 2024 (DOI 10.1038/s41586-024-07711-7; preprint [arXiv 2306.13812](https://arxiv.org/abs/2306.13812), L): networks trained for a long time on a changing stream become no better than shallow ones; re-initialising a few unused units restores plasticity. Consequence: a long-horizon test (Q-11) is mandatory; a freshly created expert starts from fresh weights, which is a built-in advantage of our design, but a shared, trained encoder will face this.
6. **Routers drift and benchmarks are too easy.** Co-drift of router and experts (PASs-MoE, above). A recent critique argues routing looks solved only on short benchmarks of 4–10 tasks where the text itself leaks the task ([arXiv 2608.01437](https://arxiv.org/abs/2608.01437), L; already in our notebook as FLEX). Consequence: our routing tests must have long sequences, overlapping and near-twin regimes, and no leaking text.
7. **Surveys frame the future as modular.** One names continual pre-training, continual fine-tuning and "continual compositionality" (modular parts composed and recombined) as the three directions and expects compositionality to matter most ([arXiv 2506.03320](https://arxiv.org/abs/2506.03320), L); another surveys continual learning for generative AI ([arXiv 2506.13045](https://arxiv.org/abs/2506.13045), L). Consequence: our direction is aligned with where the field expects to go, which also means it is crowded.
8. **Unlearning in mixture-of-experts models is approximate.** Methods for MoE language models (SEUF/UOE, [arXiv 2411.18797](https://arxiv.org/abs/2411.18797); GRIP, [arXiv 2601.16905](https://arxiv.org/abs/2601.16905); both L) work on routers and a few experts; neither claims exactness. Consequence: exact removal, if our design can deliver it, would be a distinctive result (idea I-16 in [IDEA_BOARD.md](IDEA_BOARD.md), hypothesis H).
9. **Open-world learning is its own field.** A review unifies rejecting unknown inputs, discovering novel classes and continual learning (Zhu et al., [arXiv 2403.01759](https://arxiv.org/abs/2403.01759), L); continual novelty detection has a benchmark (Aljundi et al., [arXiv 2106.12964](https://arxiv.org/abs/2106.12964), L). Consequence: our novelty work should be reported on their protocols as well as ours.

---

## 3. Efficiency: what "efficient compared with a dense model" can honestly mean

**Static mixture-of-experts models already exist in mainstream use** (for example Mixtral and DeepSeek-V3; R, not rechecked). They save compute per token at large scale but have a fixed set of experts and a fixed router after training: they do not learn continually. Whether any specific commercial chat model is dense or sparse is not public. So the right baselines are a **dense model of equal quality** and, where we can build one, a **static MoE of equal total size**; our claim is about continual learning *plus* efficiency.

| What was found | Source | What it means for us |
|---|---|---|
| At modest compute budgets an MoE can match a dense model that needs nearly four times the compute; the gap narrows with scale | Artetxe et al. ([arXiv 2112.10684](https://arxiv.org/abs/2112.10684), L) | Small-scale gains are plausible; expect them to shrink at scale |
| A small mixture with 32 routed experts matched a 1.3B dense model's perplexity with about a third of the inference compute | SmallTalk LM, "No Need to Talk: Asynchronous Mixture of Language Models" ([arXiv 2410.03529](https://arxiv.org/abs/2410.03529), L): separate small language models, each sequence routed to one, trained without communication | **The closest published design to a pool of small language models on one machine.** New experts could be added later, which is what makes it continual. This is probe P1 |
| Under strictly equal total parameters, compute and data, an MoE with a suitable activation rate can beat a dense model (2B and 7B scale) | ICLR 2026 [poster](https://www.iclr.cc/virtual/2026/poster/10007428) (L); Lory ([arXiv 2405.03133](https://arxiv.org/abs/2405.03133), L) | Equal-resource comparisons are accepted practice and we must use them |
| At small batch sizes, decoding is limited by memory bandwidth, every expert must stay in memory, and offloading to the CPU brings back latency | MoE inference studies ([arXiv 2303.06182](https://arxiv.org/abs/2303.06182), L; several blog posts, secondary) | **Floating-point counts are not enough.** We must measure bytes read, wall-clock and energy, and keep experts resident on a 12 GB GPU |
| Energy of a block of code on a GPU can be measured through NVML; Zeus wraps it (`ZeusMonitor`, [PyPI](https://pypi.org/project/zeus-ml/), [blog](https://pytorch.org/blog/zeus/), L) | Zeus, NSDI 2023 | Our machine has `pynvml` installed; GPU energy is measurable, CPU energy on Windows is not obviously so |

---

## 4. Benchmarks and protocols that reviewers accept

| Setting | Benchmark | Notes |
|---|---|---|
| Text classification, continual | The five-dataset sequence of MbPA++ and LAMOL: AG News (4 classes), Yelp (5), DBpedia (14), Amazon (5), Yahoo (10); 115,000 training and 7,600 test examples per task; four task orders; Yelp and Amazon label spaces are often merged (L, from several follow-up papers). LAMOL ([arXiv 1909.03329](https://arxiv.org/abs/1909.03329)) reports being 2–3% below multitask training | Our first standard benchmark (milestone M2). AG News is cached locally; the other four need a download |
| Language models, continual | TRACE ([arXiv 2310.06762](https://arxiv.org/abs/2310.06762), L): 8 datasets (domain-specific, multilingual, code, math), 5,000 train and 500 test examples each; shows large losses in aligned models (one example: GSM8K 28.8% to 2%); defines general-ability, instruction-following and safety deltas | Large-model oriented; its smaller tasks and metric ideas are reusable |
| Vision with pretrained backbones | CIFAR-100, ImageNet-R, CUB-200 and similar class-incremental splits as used by SimpleCIL, RanPAC and their successors (R/L) | For the other-modality milestone (M6) |
| Novelty and open-world | Continual novelty detection benchmark (Aljundi et al.); hold-out-class protocols from the open-world review | For M2/M3 |

**Metrics to report every time.** Average accuracy (or perplexity) at the end; forgetting (drop from the score right after learning to the final score); new-task accuracy (plasticity); accuracy over time (anytime); cost per update and per answer (compute, wall-clock, bytes read, energy where measurable); memory; and the same numbers for each baseline under the same budget.

---

## 5. What is cheap and available for this machine

Installed here (checked 2026-11-02): torch 2.6.0+cu124, transformers 4.57.6, peft 0.19.1, bitsandbytes 0.49.2, sentence-transformers 5.4.0, datasets 4.8.5, accelerate 1.13.0, pynvml; RTX 3060 12 GB, Ryzen 5 7600, 31.7 GB RAM.

| Item | What we know | Tag |
|---|---|---|
| QLoRA on a 7B model | Fits a 12 GB card only at the edge (sequence length 512, batch 1, gradient checkpointing); 1–3B models are comfortable | L, secondary guides |
| Small embedding models | Qwen3-Embedding-0.6B: 0.6B parameters, 28-layer decoder, configurable 32–1024 dimensions, listed as Apache-2.0, about 0.4 GB at 4-bit | L, model-catalogue pages, secondary |
| Training a ~100M-parameter language model from scratch on this GPU | **My estimate, not a measurement:** about 6 x 100M x 2B tokens = 1.2e18 floating-point operations; at an effective 5 TFLOP/s that is roughly 2–3 days of compute. Probe P1 replaces this with a measurement | estimate |
| Local training of small encoders (20–150M parameters) | Routine on this class of GPU; days at most for contrastive fine-tuning on our own data | R |

---

## 6. The closest relatives of the whole target, and what is left

| Relative | What it does | Overlap with our target | What it lacks, as far as the abstracts say |
|---|---|---|---|
| SmallTalk LM | Pool of small language models, sequence-level routing, 3x less inference compute at equal perplexity | Pool of small experts, per-sequence routing, efficiency vs dense | Continual addition, novelty detection and lifecycle are not its focus |
| CoPE, SEMA, HGE, MaRS, PASs-MoE | Task-free or near-task-free expansion with novelty-based routing | Expansion plus routing plus replay-free | Mostly vision-language or classification; cost accounting against compute-matched replay is not the headline |
| Sparse memory finetuning | Sparse slot updates; strong retention on facts | Sparse updates, no forgetting | Facts only; no routing or novelty |
| Nested Learning / HOPE | Multi-timescale memory in one architecture | Fast and slow learning | Different mechanism; not a pool of routed experts |
| Analytic continual learning (REAL, VILA), FeCAM, AdaGauss | Closed-form heads on frozen or adapting backbones | Our closed-form head | No novelty detection, lifecycle or growth in pools |
| PEER | A million tiny experts, product-key retrieval | Sublinear routing at huge expert counts | Static, no continual growth or novelty |

**What none of them does, as far as the abstracts show (a hypothesis, to be checked properly in M8):** one system that combines calibrated task-free novelty detection, pooled routing, a closed-form fast memory, a knowledge lifecycle with exact removal, and a compute-matched efficiency evaluation on a single consumer GPU. That combination, with honest negative results, is the space we can credibly occupy. Nothing here says a single piece is new.

---

## 7. Reading list by milestone (read in full, upgrade the tag to V, then start)

| Milestone | Read before starting |
|---|---|
| M0/M1 | Prabhu et al. 2023; Cho et al. 2026; Zeus (energy measurement) |
| M2 | SimpleCIL; RanPAC; FeCAM; the MbPA++ and LAMOL papers (benchmark protocol) |
| M3 | Aljundi et al. (continual novelty detection); the open-world review; Bates et al. 2023 (conformal outlier tests); SEMA and CoPE (autoencoder-based expansion) |
| M4 | PEER; AIR-MoE ([arXiv 2605.04952](https://arxiv.org/abs/2605.04952), L+ in our notebook); HGE; PASs-MoE; FLEX |
| M5 | SmallTalk LM; Ibrahim et al.; Lory; sparse memory finetuning; Nested Learning; Ovadia et al.; TRACE; the scaling-laws-of-forgetting papers |
| M6 | SimpleCIL and RanPAC protocols for vision; audio and time-series continual benchmarks (to be found) |
| M7/M8 | Spurious forgetting; loss of plasticity; compute-budgeted evaluation papers; the unlearning papers |

## 8. Limits of this digest

Search summaries and abstracts only; some summaries are secondary (blog posts or paper-review sites) and are marked. No independent replications were checked. The 2026 identifiers
come from the search results and were not independently confirmed. A paper that looks like a close relative may have details that change the plan; the rule in
[ROADMAP.md](ROADMAP.md) section 7 (literature refresh) is how we catch that.

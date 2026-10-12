# Continual learning without forgetting, and the MoE part: what the world knows, what we have, what to do next

Date 2026-10-09. Written by Claude for you (the mediator) and, later, for the Gemini agents. Plain words first, then the precise term, defined once.

Companion files: `research/moe_diligence_2026-10/` (every check I ran; scripts and raw outputs), `research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md` (the Oct 4 design this report updates), `research/growmoe_results/phase1/RESULTS.md` (the Phase 1 numbers), `docs/superpowers/handoff/SCOPE_CHARTER.md` (the rules).

## How to read this

Every claim carries a tag. The tags matter more than the prose.

| Tag | Meaning |
|---|---|
| **[M]** | Measured by me in this session. Script and raw output in `research/moe_diligence_2026-10/`. Seeds and configuration are stated next to the number. |
| **[C]** | I read the code and this is what it does. File names are given. |
| **[L]** | Literature. I opened the abstract or an equivalent summary page this session. I did not read full papers, so "the authors report X" means exactly that. Links are in section 11. |
| **[R]** | Recalled from memory, not re-checked this session. |
| **[H]** | Hypothesis. Nobody (including us) has measured it. |

What I deliberately did NOT do: change any repo code; run the S5 or S6 streams (their pre-registration is being rewritten and I did not want to peek); read full papers; run anything larger than a few minutes on your PC. My checks use the benchmark's own stream settings and Pool A's own frozen settings (q = 0.02, alpha_spawn = 0.01, m_min = 32, route rule `accept_then_logf`), and they reproduce the Phase 1 numbers exactly where they overlap (A = 87.04 vs 87.02 on S2, 69.81 vs 69.82 on S1, replay = 67.68 vs 67.68 on S2).

---

## 0. The one-page answer

### Update after the routing investigation and the M2-A test (same evening; Appendix E)

The M0 failure traced to a lifecycle fault, not to the gate: a newborn expert's competence monitor was armed from its untrained loss, fired a false "drift" alarm on every new regime and overwrote the newborn's weights. Arming it only after infancy (one config flag, now on by default) and re-tuning the step budget, tested on fresh seeds 10-19 under a pre-registration, gives **+10.0 points over the best single network at 15 regimes and +11.6 at 30 (bar +5)**; all seven pre-registered statements pass, including the no-regress guards. The M0 verdict for the frozen pool stands. The efficiency half is still unproven: at 30 regimes the pool needs 1.4 times the single network's memory and about the same multiply-adds, because the gate scan is 97 % of its compute.

### Update after the M0 + M1 run (same day, evening; details in Appendix D)

The pre-registered test is done: 10 test seeds, every method with a tuned training budget, eight statements decided by a script (`research/growmoe_results/m0/RESULTS_M0.md`). **Three of the eight came out against what I expected.** (1) On function tasks the growing pool beats the best single network we could build, but only by +4.1 points at 15 regimes and +2.6 at 30 (intervals exclude zero, both below the +5 bar we set before running), so statement B1 **fails** and the kill rule applies to the claim "the current pool beats a dense network". (2) Perfectly routed separate experts (an upper bound, not a method) beat that same single network by +11.3 and +16.2 points while using a quarter of its memory at 15 regimes and 30 times less compute per prediction: the structural idea is strong, our learned version realises only a fraction of it. (3) The pool's weakness is not test-time routing (92-99% of held-out inputs reach the right expert) but its experts, which are 10-15 points weaker per regime than oracle-trained ones. Also: Phase 1's "beats replay" claim does not survive a tuned replay baseline, a lookup beats the pool on similarity streams by 4.6-12 points, and Pool C's parameter count was inflated 2.7 times.

### Update after your replies (same day; details in Appendix C)

Three new measurements sharpen the picture. (1) You were right that the filing-cabinet win is about memorising: when the answer is a hidden function of the input, a lookup stays near chance (21-26%, chance 12.5%) even with 10,000 stored examples, while a small trained network reaches 45-78% from the same data. (2) In a continual stream of such function tasks, the growing pool beats a single replay network by +9, +16 and +13 points at 5, 15 and 30 regimes (3 seeds, all intervals exclude zero), while the lookup sits at chance; a bigger single network narrows but does not close the gap (pilot, not pre-registered). (3) On efficiency, only one expert answers each input, but the gate scans every expert's stored vectors and costs 38 times the expert it selects, so the pool is not yet cheaper than a full lookup. Net: your bet is supported in the setting where it should apply (tasks that need a computed answer); efficiency and resource-matched comparisons still have to be earned. Your decisions are recorded in C.1: R12b paused, no pretrained encoder for now, handoff only for deep coding.

### Your bet, in one paragraph

You bet that a dynamic MoE (a team of specialists that grows when something new arrives) can solve continual learning without catastrophic forgetting. **The bet is right about the mechanism and incomplete about the problem.** Freezing specialists really does stop old skills from being overwritten, and our own numbers show the benefit growing with the number of tasks: a single network that revises from a box of old examples (replay, 20 training steps per batch) falls from 92% to 72% as the stream grows from 5 to 30 permuted-digit tasks, while the growing pool only drifts from 88% to 86% (1 step per batch) or from 91% to 82% (20 steps, wide error bars) [M, 3 seeds]. But the hard parts of the problem are not the weights. They are (1) deciding which specialist handles an input, (2) keeping shared features stable, (3) deciding when to grow, and (4) fitting in a memory budget. The 2026 literature is almost entirely about (1) and (3), and your gate design (a calibrated "is this new?" test and no learned router) is aligned with what the field learned the hard way. Where we are weak is that our tests cannot yet prove the specialists add value.

### The six things that surprised me (all measured, [M])

1. **A plain filing cabinet beats our specialists on every stream we have.** k-nearest-neighbour (kNN: store every example, answer with the most similar stored ones; no learning at all) scores 95.6% vs PhyloMoE's 87.0% on permuted digits, and 77.95% vs 69.8% on the mixed-types stream, 10 seeds each, at equal memory.
2. **Our "beats replay" result is a training-budget artifact.** The benchmark gives every method about 7 optimizer steps per task. With 5 steps the replay network roughly ties PhyloMoE on permuted digits (88.6% vs 90.2%) and already beats it on the mixed-types stream (76.2% vs 72.6%); with 20 steps it is ahead on the mixed-types stream by 79.3% vs 75.8% (intervals separated).
3. **The Phase 1 tests are tiny.** The "S2 permuted digits" stream that produced the headline has **5** regimes, not the 30 in the spec. Everything above ~10 regimes is untested in the official results.
4. **At 30 regimes, our specialists use more memory than the raw data.** 9.3 MB of experts and stored examples to hold 6.0 MB of training data. A kNN with 2 MB matches PhyloMoE's accuracy.
5. **Growth, merging and archiving never fired** in any of 20 benchmark runs. They are unit-tested but not exercised in the results.
6. **A reported number is wrong.** Pool C's parameter totals count its shared base network once per expert, inflating them about 3x (371,392 reported vs 120,640 actual on one seed). Prediction C3's "189%" is really about 80-100%.

### What I recommend (details in sections 7 and 8)

- **Do first (small, high value):** a "fair fight" round. Same tuned training budget for every method, add the filing-cabinet and other simple baselines, count memory honestly, fix Pool C's count, run the 30-regime stream. Nothing else should be built until this exists.
- **Then find where experts genuinely win:** a stream with a tight memory cap and streams where the answer is a learned function (so memorising fails). Set the win condition in advance, with a kill rule.
- **Make the experts small:** separate "how many stored examples the gate needs for its statistics" from "how many examples the expert keeps". My tiny-expert test collapsed because the gate needs at least about 1/q = 50 stored scores to flag anything at a 2% false-alarm level. That is a fixable design flaw, not a law.
- **Add a fast memory tier next to the experts** (the filing cabinet as a first-class part, experts learn from it later, like sleep). This is what our data, the brain-inspired design in your original README, and the 2026 "modular memory" position paper all point to.
- **Park:** tree routers, KAN neurons, formula-based expert sizing, learned geometric backbones. Reasons in section 6.4.
- **Use the second agent's best idea properly:** right-sizing experts is worth doing, as a measured memory-versus-accuracy question, with the size chosen by a quick label-aware trial, not by a formula on input geometry.

### Decisions I need from you (section 10)

1. Accept "beat kNN and tuned replay at equal memory" as the gate for the MoE track, with a kill rule?
2. Allow a frozen pretrained encoder in the MoE track for realistic text streams?
3. Pause the vertical-growth work (R12b) until the fair-fight round says experts are worth growing?
4. Welcome a fast episodic memory tier inside the MoE system?
5. Shall I write the handoff for the standby MoE agent?

---

## 1. The problem in plain words

**Catastrophic forgetting** means that a network trained on task B loses task A. It happens because both tasks are stored in the same numbers (weights), and learning B moves those numbers.

Think of a student with one notebook who prepares for a new exam by rewriting pages. Remedies are: keep old exam papers to revise from (**replay**), mark important pages as "don't touch" (**regularisation**), or give each subject its own notebook (**parameter isolation**, which is what a **mixture of experts, MoE**, does: several specialists plus a receptionist, the **router** or **gate**, who decides which specialist gets each input). A **dynamic** MoE buys a new notebook when a new subject arrives.

### Where forgetting hides (four places)

| Place | What goes wrong | Does freezing specialists fix it? |
|---|---|---|
| 1. The specialist's own weights | New training overwrites old skill | **Yes**, by construction |
| 2. The router / gate | After growth, old inputs get sent to the wrong (new) specialist, even though the old ones are frozen | No. This is the main topic of the 2026 papers [L] |
| 3. Shared features (the "eyes" feeding the specialists) | If the feature extractor changes, every stored fingerprint and specialist becomes stale | No |
| 4. Knowledge that must change | The world changes: same input, new correct answer. A frozen specialist still gives the old answer | No. Our known weak spot (limitation L3) |

Add two practical limits that are not forgetting but decide success: **memory/compute budget** (you cannot grow forever; an optimal forgetting-free learner needs perfect memory and is NP-hard [L: Knoblauch et al. 2020]) and **plasticity** (a network can lose the ability to learn new things after a long stream [L: Dohare et al., Nature 2024]).

### Three test styles you will see

- **Class-incremental:** new labels keep arriving (our S1). Hardest, because the model must choose among classes it learned at different times.
- **Domain-incremental:** same labels, changing input style (our S2, digits with shuffled pixels).
- **Task-free:** nobody tells the model when the task changes. This is our setting, and what the conformal gate is for.

---

## 2. What the internet says (October 2026)

### 2.1 The families of approach

| Family | Idea | Strength | Weakness | Evidence level |
|---|---|---|---|---|
| Replay | Keep a small box of old examples | Strong, simple; hard to beat [L] | Memory grows; privacy | [L] |
| Regularisation | Protect important weights | No stored data | Weaker on long streams | [L] |
| Expansion / parameter isolation (**MoE sits here**) | New module per new thing | No weight forgetting | Growth, routing, which module | [L] (sections 2.2-2.3) |
| Pretrained backbone + light heads | Freeze a big model; keep class averages (prototypes) or small adapters | Surprisingly strong baseline | Needs a good backbone | [L] SimpleCIL, RanPAC |
| Memory layers with sparse updates | Update only a few memory slots | Tiny forgetting on facts | Facts only so far | [L] Meta 2025: full finetune loses 89% of NaturalQuestions F1, LoRA 71%, sparse memory 11% |
| Nested / multi-speed memory | Parts of the model update at different speeds | Principled | New, mostly language-model scale | [L] Google "Nested Learning" (NeurIPS 2025), "Hope" |
| Model merging | Fold new skills into one set of weights | Constant memory | Interference | [L] ICLR/ICML 2026 papers |
| Sleep / consolidation | Replay and distil offline | Matches brain theory | Early; small tests | [L] two 2026 "language models need sleep" papers (different methods, near-identical titles) |
| In-context / external memory | Remember in the prompt or a store | No weight updates | Frontier systems still fail to learn from experience: in CL-Bench (2026) naive in-context learning beat the dedicated memory systems | [L] |

### 2.2 What is known about growing / dynamic MoE specifically

- **Theory (ICLR 2025 spotlight) [L]:** in a simplified setting, MoE helps because experts specialise and the router learns to pick them; the router must **stop updating after enough rounds** to converge; and **more experts need more rounds and may not improve learning**. This supports a gate that is not learned online (ours is not).
- **Growth itself breaks routing [L]:** StaR-MoE (May 2026) shows that adding experts shifts the router so old inputs get reassigned to new experts even when old experts are frozen. PASs-MoE (ACL 2026) shows the router and LoRA experts drift together. SAME (ICML 2026) and a CVPR 2026 spectral-MoE paper attack the same router drift. LLaVA-CMoE (2025) says modifying shared routing components causes forgetting.
- **It scales on benchmarks with pretrained backbones [L]:** CaRE (ICML 2026) reports up to 300+ tasks; SEMA (CVPR 2025) adds an adapter only when a shift detector fires and claims sub-linear growth.
- **Forgetting can still happen inside experts and through routing in MoE language models [L]:** a (single-author, lower-confidence) 2026 mechanistic study analyses routing-gate drift; a TMLR 2025 study found MoE language models unexpectedly resilient to data shift in continual pre-training even without replay.
- **Plasticity and growth timing [L]:** new units can be "forward-active yet backward-starved" (weak gradient), so growth works only if the newborn capacity has time to become useful (2605.15435). Plasticity loss has also been reported in MoE reinforcement-learning agents (SPHERE, 2605.04712).
- **Routing design may matter less than expected [L, small scale, not peer reviewed]:** "Equifinality" (2604.14419) finds five cosine-router variants equivalent on perplexity at 76M parameters.

### 2.3 Closest published relatives of GrowMoE

| Paper | What it shares with us | What is different |
|---|---|---|
| **MaRS** (ICLR 2026) | Growth decided by a statistical test with stated guarantees on false alarms and detection delay; slot router | Frozen pretrained encoder; stores "anchors" instead of raw examples; no conformal p-values |
| **SEMA** (CVPR 2025) | Add a module only when a shift is detected; expandable router; rehearsal-free | Learned autoencoder detectors on a pretrained transformer |
| **LLaVA-CMoE** (2025) | "Probe" experts decide when and where to add experts (like our interference probe) | Vision-language model; VAE-based task locator |
| **CN-DPM** (ICLR 2020) | Our Pool B | Likelihood gate (known to be fooled by out-of-distribution data) |
| **TAME, SEDEM, ODDL, DEITP** | Task-free expansion triggers (loss jump, diversity, discrepancy, familiarity autoencoder) | Heuristic thresholds |

**What looks distinctive in our design [H, from absence of search hits]:** calibrated per-expert p-values with batch false-discovery control, a null-calibrated probation buffer, extend-or-spawn by shadow training, and a router that is not learned. **What is not distinctive:** statistically controlled growth (MaRS) and probe-guided growth (LLaVA-CMoE). The one-line search I ran found no paper combining conformal calibration with expert expansion, which is absence of evidence, not proof.

### 2.4 Warnings from the field

- **Simple baselines are strong.** With a frozen pretrained backbone, class averages (SimpleCIL), random-projection prototypes (RanPAC) and streaming linear discriminant analysis (SLDA) often match or beat elaborate methods [L]. Memory-based kNN on frozen features is another standard strong baseline [L], with the caveat that kNN probes can flatter a method.
- **Benchmarks saturate.** FLEX (2608.01437) shows routing looks solved on popular benchmarks because of prompt "fingerprints" and short sequences; with 34 tasks it is not.
- **Theory limits.** No method can be both forgetting-free and memory-free [L].
- **Detecting novelty is necessary.** The class-incremental theory of Kim et al. (Artificial Intelligence 2025) says accurate out-of-distribution detection within each learned task is required for class-incremental learning to succeed [L]. This supports the importance of the gate, which is our strongest component.

### 2.5 What the frontier does about general continual learning

Not growing MoE. Meta's sparse memory finetuning, Google's multi-speed "Nested Learning" and Titans (surprise-driven test-time memory), consolidation by "sleep", and a 2026 ICML position paper arguing that **modular memory combining fast in-context learning with slow in-weight learning is the missing piece**. CL-Bench (June 2026) found that frontier systems often overfit their latest observations and that dedicated memory systems did not beat plain in-context learning. Read this as: MoE is one component of a general system, not the system.

---

## 3. What we have (code analysis)

### 3.1 Three MoE implementations live in the repo [C]

| Where | What | Status |
|---|---|---|
| `growmoe/` (5,692 lines, 328 tests) | The evidence-driven line: Pools A, B, C, StackMoE, streams S1-S7, baselines, benchmark | Active. All the analysis below is about this |
| `ishmn/spherical_dynamic_moe.py` | Legacy token-level MoE on an 8-dimensional sphere; spawns by a fixed threshold (0.40) that never fires; sleep merge/prune | Frozen (charter). Do not integrate |
| `geometry/dynamic_moe.py` | Manifold-prototype experts with geodesic routing; spawns by a fixed novelty radius; freezes old experts | Demo of the geometry components; two tests; unverified as a continual learner |

### 3.2 GrowMoE part by part [C], with status

| Part | File | What it does | State |
|---|---|---|---|
| Frontend | `frontend.py` | Hashes any input to a unit vector (feature hashing; no learning) | Works; weak on text/code windows |
| Signature | `signature.py` | Per-expert memory of up to 256 stored unit vectors; scores new input by a kernel density; conformal p-values from the stored vectors' leave-one-out scores | Validated: false-alarm rate 0.046-0.049 at nominal 0.0467 (log T4b) |
| Gate | `pools/phylo.py` `observe` | p-value per expert, batch false-discovery control (Benjamini-Hochberg) at q = 0.02; novel samples go to probation | Works: 0.00% false alarms on known data |
| Probation | `probation.py` | Holding pen; clusters novel samples at a threshold calibrated so random noise rarely links | Works (P8 pass) |
| Evidence test | `phylo.py` | Spawn only if the cluster's log-likelihood ratio exceeds log(1/alpha) | Works |
| Extend-or-spawn probe | `_handle_cluster` | Train a shadow copy of the parent on the new cluster; if old-data loss rises by less than 0.05 nats, extend the parent, else spawn a child | Works; fires every run |
| Child "DNA" | `experts.shrink_perturb` | Child = lambda x parent + (1-lambda) x fresh, lambda = measured affinity | Validated in spike 2 |
| Competence monitor | `competence.py` | Change detector on each expert's own loss; on alarm, spawns a successor that "supersedes" the old expert | Fires 2-6 times per run; remap case only half solved (L3) |
| Growth policy | `growth.py` | Propose widen/deepen on shadow copies, accept only if clearly better than an equally trained control | Verified alone; **never fired inside a pool** (0 grow events in 20 runs [M]) |
| Merge | `phylo._attempt_merges` | Distil two overlapping experts into one, verify on probe sets | Exists and is unit-tested; **never fired in 20 runs** [M] |
| Lifecycle | `phylo.observe` | infant -> active -> dormant -> archived (drops replay and optimiser) -> revive | **No archive or revive in 20 runs** [M] |
| Expert | `experts.py` | One residual block (width 128) + a 64-class head (about 41,800 parameters; 20% is the head) | The label space is fixed at 64 at construction |
| Pools B, C | `pools/bnp.py`, `keyed.py` | Alternative designs, now mainly baselines | C's parameter count is inflated (3.4) |
| StackMoE | `stack.py` | Vertical layers of representation experts | Learning loop verified; growth trigger never fired in its test |
| Benchmark | `bench.py` | Test-then-train, 10 seeds, dev-seed tuning | Honest protocol; baselines and stream sizes are the issue (section 4) |

### 3.3 What the Phase 1 scoreboard really says

14 pre-registered predictions: 4 pass, 8 fail, 1 inconclusive, 1 not measurable. "VERIFIED" in the charter means the numbers were independently reproduced, not that the hopes came true. Notable: spawn recall 0.52 vs a 0.90 bar (it shares experts across related regimes, which is partly the point); mixed-types accuracy 69.8% vs 78.1% with perfect routing; and the "beats replay" prediction (P4) passed against a replay network that is under-trained (section 4.3).

### 3.4 Defects and oddities I found [C]/[M]

1. **Pool C parameter double-counting.** `keyed.py` does not override `summary()`, so every delta expert's parameter count includes the shared frozen base (41,792 parameters). Run on S2: reported 371,392 vs 120,640 actual (7 experts), and 265,280 vs 98,112 (5 experts) [M]. Prediction C3 "189% of A's parameters" is really about 80-100%. C is still much less accurate (about 73-75% vs A's 87-90% on the same seeds [M]). My earlier review accepted this table; it should have caught it.
2. **S2 in the official results has 5 regimes** (`bench.py`, `s2_permuted(n_regimes=5, schedule=[0,1,2,3,4,1,3], n_train=400)`), S1 uses 400 train per type. Oracle routing's "5.0 experts" confirms it. Several descriptions (including mine) say 30.
3. **Not rehearsal-free.** Every expert keeps a 256-sample replay reservoir and 256 stored signature vectors. Any claim of "never seeing past data" is false. That is fine, but comparisons must be at equal memory.
4. **Fixed 64-class head** in every expert even when the task has 10 classes (about 8,000 parameters per expert), and the label space cannot grow after construction. Open-ended learning needs a growing label space.
5. **Routing is a Python loop** per sample and per expert: 7 ms per sample at 512 experts on CPU [M]. This is an implementation inefficiency, not an algorithmic limit (section 4.8).
6. **Growth statistical power is starved.** A pool expert's growth validation split is about 50 samples (the Phase 0 log warned about this). Together with finding 5 above, growth has never been exercised end to end.

---

## 4. My own checks (what I measured today)

All numbers: Pool A with its frozen Phase 1 settings, strict test-then-train, final accuracy averaged over regimes, mean [95% confidence interval] over seeds. Scripts and raw outputs: `research/moe_diligence_2026-10/`.

### 4.1 Check 1: simple memory baselines (10 seeds, the benchmark's own streams)

| Stream | Method | Accuracy % | Memory (KB) |
|---|---|---|---|
| S2 (5 permuted regimes) | PhyloMoE A | 87.04 [85.54, 88.54] | 1450 |
| | Replay network B2 (benchmark budget) | 67.68 [65.30, 70.06] | 1160 |
| | **kNN, all samples stored** | **95.60 [94.72, 96.48]** | 1400 |
| | kNN, same number of stored vectors as A | 93.12 [91.96, 94.28] | 879 |
| | Nearest class mean (NCM) | 42.60 [38.58, 46.62] | 5 |
| | Streaming LDA (SLDA) | 61.18 [57.93, 64.43] | 69 |
| S1 (7 data types) | PhyloMoE A | 69.81 [67.79, 71.83] | 2518 |
| | Replay network B2 | 61.53 [60.81, 62.26] | 1170 |
| | **kNN, all samples stored** | **77.95 [77.35, 78.56]** | 1700 |
| | kNN, same bytes as A | 77.84 [77.23, 78.44] | 1656 |
| | NCM / SLDA | 57.45 / 60.58 | 15 / 79 |

kNN's forgetting is negative (it improves as it sees more). Reading: on these streams the information lives in the stored examples, not in a function worth learning, so memory wins.

### 4.2 Check 2: is the benchmark budget-limited? (10 seeds)

Each routed batch gets `inner_steps` optimiser steps (benchmark default 1, about 7 steps per regime visit).

| Stream | Method | 1 step | 5 steps | 20 steps |
|---|---|---|---|---|
| S2 | PhyloMoE A | 87.04 | 90.18 [89.36, 91.00] | 91.16 [89.63, 92.69] |
| S2 | Replay B2 | 67.68 | 88.64 [87.44, 89.84] | 92.48 [91.60, 93.36] |
| S1 | PhyloMoE A | 69.81 | 72.55 [69.07, 76.03] | 75.81 [74.12, 77.50] |
| S1 | Replay B2 | 61.53 | 76.20 [75.06, 77.34] | 79.26 [78.31, 80.20] |

**The headline "A beats replay by 19 points" exists only when replay is under-trained.** This is the "tune every arm" mistake already recorded in my notes. The Phase 1 tuning grid covered only Pool A's q on one small dev stream.

### 4.3 Check 3: what happens as regimes multiply (3 seeds, permuted digits, 400 train per regime)

| Regimes | kNN (all) | PhyloMoE, 1 step | PhyloMoE, 20 steps | Replay, 20 steps | Replay, 1 step |
|---|---|---|---|---|---|
| 5 | 95.40 | 87.67 | 91.20 | 92.07 | 67.20 |
| 15 | 94.73 | 86.71 | 87.53 | 84.36 | 49.98 |
| 30 | 94.69 | 85.71 | 81.99 [70.05, 93.92] | 72.29 [70.60, 73.98] | 41.83 |

This is the supportive evidence for your bet: one fixed network runs out of room as regimes multiply, while the pool holds up. kNN does not degrade at all, because it simply keeps everything. (Only 3 seeds; PhyloMoE's interval at 30 regimes is wide.)

### 4.4 Check 4: memory versus accuracy at 30 regimes (3 seeds; raw data = 12,000 samples = 6,000 KB)

| Method | Memory (KB) | Accuracy % |
|---|---|---|
| kNN, 1,000 stored | 500 | 53.32 |
| kNN, 2,000 stored | 1000 | 74.58 |
| Replay net, 2,048 buffer, 20 steps | 1160 | 72.29 |
| **kNN, 4,000 stored** | **2000** | **85.86** |
| PhyloMoE, right-sized "small" (h32, 64 prototypes) | 3075 | 74.74, seeds 83.6 / 56.1 / 84.5 (unstable) |
| kNN, 8,000 stored | 4000 | 91.90 |
| Replay net, 8,192 buffer | 4232 | 79.30 |
| kNN, 12,000 (everything) | 6000 | 93.80 |
| **PhyloMoE, default** | **9331** | **85.71** |

PhyloMoE sits below the kNN curve everywhere: a 2 MB kNN matches the 9.3 MB pool. The only place memory-based methods are weak is below 1 MB, where a small learned model could in principle win. The right-sized PhyloMoE variants did not exploit that, for the statistical reason in 4.6.

### 4.5 Check 5: is the front-end the S1 bottleneck? (10 seeds)

I swapped only the text and code regimes from hashed n-grams to a frozen pretrained sentence encoder (all-MiniLM-L6-v2, local cache, offline).

| | Hashed | Pretrained text encoder |
|---|---|---|
| PhyloMoE A, S1 average | 69.81 [67.79, 71.83] | 72.62 [69.95, 75.29] |
| kNN, S1 average | 77.95 [77.35, 78.56] | 81.96 [81.49, 82.43] |
| PhyloMoE, code regime | 54.1 | 71.3 |
| PhyloMoE, English-text regime | 35.1 | 44.5 |
| PhyloMoE, AR(2) series regime | 50.3 | 47.0 (unchanged, intrinsically hard: kNN 53%) |

So both agents' "representation bottleneck" is real for text and code (+17 and +9 points) but it explains only about 3 of the 8 points between PhyloMoE and oracle routing. The rest is the experts' training budget and routing arbitration. A better front-end alone does not make PhyloMoE beat kNN.

### 4.6 Check 6: a hard statistical floor on how small the gate's memory can be

A conformal p-value built from n stored scores cannot be smaller than 1/(n+1). At q = 0.02, **n = 32 gives a minimum of 0.030, so nothing can ever be flagged as new**. My 32-prototype "tiny" experts did exactly that: 3 experts, accuracy 17.5% (chance 10%). With 64 prototypes the minimum is 0.015, barely below q, and the pool was unstable (fragmenting to 22-34 experts, one seed collapsed to 56%). **Design lever:** the calibration set only needs scores (single numbers), not stored vectors, so the density model can be small while the calibration set stays large. Today they are the same stored vectors (`signature.py`). Separating them (split conformal with scalar-only calibration) is the principled path to small experts [H, motivated by this measurement].

### 4.7 Checks of the second agent's ideas

| Claim | Check | Result [M] |
|---|---|---|
| The norm-gated neuron morphs between plane, wave and prototype | On unit-length inputs (which every GrowMoE expert receives) its norm term is a constant | Difference from a plain linear unit: 1.2e-7. It **is** a plain linear unit here. It could matter on hidden activations with varying norm; nobody has shown it |
| Size an expert from intrinsic dimension (TwoNN) and concentration | TwoNN on clusters of known dimension, N = 32 to 512 | Unreliable at probation sizes (true 2-D cluster reads 3.2, 4.6, 7.3, 10.9, 16.0 as N grows because noise dimensions appear). The agent's own formula adds a further x1.44 bias |
| ...and that predicts needed capacity | Same inputs, two labelling functions, train widths 2-128 | Linear labels: width 2 reaches 84.5%. Deep teacher: widths 2-16 sit at 52-53%, width 32 reaches 59.9%. Needed width differs while input geometry is identical. **Capacity is a property of the label function, not only of the inputs** |
| Delta experts need 15-30% of a full expert | Count parameters | Per expert 11,264 vs 41,792 = 27%, true. But Pool C spawned more experts and scored 73-75% vs A's 87-90% |
| Flat routing cannot scale; we need a tree | Time `predict`, 64-sample batch | 7.1 / 25.9 / 102 / 451 ms at 8 / 32 / 128 / 512 experts: linear, Python-loop bound, easily batched. Our own tree-search measurement (Track Y) was 36-60x slower than dense at N <= 4096 |

### 4.8 What these checks do NOT show

They do not show that MoE cannot win. They show that **our current streams cannot tell**, because (a) a filing cabinet solves them, (b) the benchmark budget under-trains the learners, (c) the official runs are small, (d) Pool A's configuration was tuned only for q. Small seed counts (3) for sections 4.3-4.4. One stream family for most checks (digits, tabular, text, code). I did not try to tune Pool A for these streams, so its numbers are "the frozen config", not its best.

---

## 5. Reading of the two Gemini write-ups

Both are useful as idea banks. Neither checked the evidence. Table of what I verified or corrected:

### 5.1 Write-up 1 (codebase tour, literature, blueprint)

| Statement | Verdict |
|---|---|
| Tour of the pools and components | Mostly accurate [C] |
| "Conformal prediction mathematically guarantees the sample is out-of-distribution" | **Incorrect.** It guarantees that, for data that looks like the calibration data, the chance of a false alarm is at most q. It cannot prove a sample is new |
| "Pool A reached 87% online without ever seeing past batches" | **Incorrect.** Each expert keeps 256 replay samples and 256 signature vectors; it is a rehearsal method |
| "Catching 100% of genuinely new task arrivals" | Overstated (first-batch detection in my earlier 3-seed check, not 100% per sample) |
| MoRAM gives "strict mathematical stability guarantees" | **Unsupported.** The abstract contains no theorem; claims are empirical [L] |
| SpawnLoRA fires on negative gradient cosine in streaming | **Incorrect framing.** The paper (Findings of EMNLP 2026) studies multi-domain fine-tuning and diagnoses near-orthogonal gradients competing for one low-rank subspace [L] |
| Hash front-end causes the S1 gap | **Partly.** +2.8 points overall, large on code and text, but the gap to kNN remains [M] |
| A tree router gives 25x fewer operations | **Not supported.** Rests on the Y8 memo, whose "86x attention concentration" was measured on random vectors; trained models showed hyperbolic is not sparser and the tree search was slower [C] review |
| GradMax in 15-30 steps instead of 480 | **Unmeasured guess.** The recipe is also mis-specified: GradMax uses singular vectors of the cross-covariance between layer input and back-propagated gradient, not the top eigenvector of an output-space error covariance [L] |
| "Merging and consolidation are missing" | They exist (`_attempt_merges`), but never fired in the official runs [M] |
| Progressive neural collapse backbone for S1 | A real ICLR 2026 paper [L], but it is a method for learned classifiers with distillation; nothing links it to the S1 gap. Premature |
| Good ideas | Context-aware gating for contradictory inputs; sleep-style distillation; plasticity monitoring; an adaptable representation later |
| Missed entirely | kNN baseline, tuned budget, 5-regime scale, memory accounting, Pool C counting |

### 5.2 Write-up 2 (neuron types, shapes, AutoSizer)

| Statement | Verdict |
|---|---|
| Heterogeneous expert sizes are worth exploring | **Yes**, with precedent in static MoE: HMoE, MoDSE, Mixture of Nested Experts [L]. The agent cited none |
| Norm-gated neurons morph into prototype balls | Not on our inputs (4.7); unproven in hidden layers |
| KAN neurons "resist forgetting", suit symbolic data | **Overstated.** The forgetting claim came from a 1-D toy; later work finds KANs still forget, worse in high dimension; under matched budgets MLPs beat KANs except on symbolic formulas [L]. Our inputs are 128-dimensional |
| AutoSizer: width from TwoNN x log N, depth from concentration | **Flawed** for three measured reasons (4.7): unstable estimator at small N, ignores labels, magic thresholds (charter forbids unexplained constants). Its TwoNN code is also off by x1.44 |
| Continual backpropagation to keep plasticity | Reasonable and close to the Nature recipe, but the agent's rate (2% per 50 steps) is about 4x the paper's example and no plasticity loss has been shown in our system [L] |
| "Degradation observed on S6 (60+ regimes)" | **No such observation exists.** S6 has not been run |
| SVD compaction frees 60-75% of memory at 99% accuracy | Unmeasured; our experts are small |
| TIES merging of siblings at overlap > 0.85 | TIES needs experts derived from a shared base; the 0.85 figure belongs to Pool C's key similarity, not Pool A's overlap rule [C] |
| **Worth using** | Right-sizing as a memory question (4.4 shows why); probing before allocating, done with labels; mixed expert families as a later ablation |

---

## 6. Synthesis: what the evidence implies

### 6.1 Verdict on your bet (precise)

- **Supported [M][L]:** freezing experts removes weight forgetting; the pool degrades much more slowly than one network as regimes multiply; non-parametric gating avoids router drift; calibrated novelty detection is theoretically necessary for class-incremental learning.
- **Not yet shown:** that the experts add anything over remembering examples. kNN is flat and strong, and uses less memory than the pool.
- **Contradicted for the current tests:** the headline gain over replay (budget artifact).
- **Open:** whether the pool wins when memory is capped or the answer is a function that memorising cannot capture.

### 6.2 Where experts should win, and how to make that testable

An expert is a compressed, generalising model. It should beat a filing cabinet when (a) **memory is tight** (kNN needs about 1 MB for 75% and 2 MB for 86% here; an expert that reaches 86% in 1 MB would win), (b) **data per task is scarce but structured** (the label is a smooth function a small network can learn from few examples), (c) **inference cost matters** (kNN's cost grows with everything stored), (d) **the answer must be computed, not looked up** (compositional or algorithmic tasks, the S5/S6 style). The fair-fight rounds should build exactly these conditions and test them against kNN and tuned replay.

### 6.3 A two-speed design the evidence points to

Our own gate already is a memory (the signature stores examples). Add a first-class **fast episodic tier** (store, look up, flag novelty) and treat experts as **slow compressors** trained from that tier when the system "sleeps" (offline distillation), with a rule such as: use the expert when its calibrated confidence is high, else fall back to the memory. This is the complementary-learning-systems idea in your original README (working/episodic/semantic tiers), the 2026 position paper's recommendation, and what both Gemini write-ups gesture at with "dual-speed memory". It also gives the system a way to be correct on day one of a new regime (the filing cabinet) while the experts catch up. [H]

### 6.4 What each idea bank item becomes

| Idea | Decision | Why |
|---|---|---|
| Fair-fight protocol, kNN/NCM/SLDA, tuned replay, memory accounting | **Do now** | Section 4 |
| Decouple calibration scores from stored vectors | **Do (M2)** | Section 4.6; lets experts be small |
| Probe-ladder right-sizing (label-aware) | **Do (M2)** | Replaces AutoSizer; reuses the shadow-training machinery |
| Episodic tier + sleep distillation | **Test (M3)** | 6.3 |
| Larger growth validation reservoirs; GradMax as a controlled ablation | **Test (M4)** | Growth has never fired; power is starved |
| Soft mixing of accepting experts / learned arbiter for routing | **Test (M4)** | S1 oracle gap |
| Context-aware gating for contradictory inputs (S7, L3) | **Later (M6)** | Real but second-order until M1 |
| Frozen pretrained encoder streams (text) | **Do (M5)** | The pretrained-era baselines (NCM, SLDA, kNN) must be on the same table |
| Continual-backprop recycling | **Only if measured** | No plasticity loss shown |
| Mixed neuron families (KAN, norm-gated) | **Defer** | No supporting evidence on our inputs |
| Tree / hyperbolic router | **Drop for now** | Cost is an implementation artifact; tree search measured slower |
| Learned ETF / neural-collapse backbone | **Defer** | Large unfinished research; first see how far a frozen encoder goes |
| SVD compaction, TIES for siblings | **Defer** | Unmeasured; wrong premise for TIES |
| Formula AutoSizer | **Drop** | Section 4.7 |

---

## 7. Recommended changes, in priority order

| # | Change | Why (evidence) | Test it with | Cost |
|---|---|---|---|---|
| P0-1 | Tune every method's training budget on dev seeds; report the operating point | 4.2 | Re-run S1-S4 with `inner_steps` in {1, 5, 20, 50} per arm | Low |
| P0-2 | Add baselines: kNN (all and memory-capped), NCM, SLDA, tuned replay with buffer sweep | 4.1, 4.4 | Same harness (scripts exist in the diligence folder) | Low |
| P0-3 | Report memory (bytes) and compute per method; fix Pool C counting | 3.4(1), 4.4 | Count unique parameters; tests that recompute the table | Low |
| P0-4 | Run S2 at 30 regimes and add a memory-capped variant | 4.3, 4.4 | `s2_permuted(n_regimes=30)` plus a byte cap | Low |
| P0-5 | Re-label Phase 1: "numbers reproduced; 'beats replay' not supported under tuned baselines" | 4.2 | Register update | Trivial |
| P1-1 | Separate calibration scores from stored vectors (split conformal, scalar-only) | 4.6 | False-alarm rate still <= 2q at 10x less vector memory | Medium |
| P1-2 | Right-size experts by a quick label-aware trial ladder; size the label head to seen classes | 4.4, 4.7 | Memory-accuracy frontier vs kNN | Medium |
| P1-3 | Episodic tier + sleep distillation + fallback rule | 6.3 | Frontier and 30-regime table | Medium-high |
| P2-1 | Routing arbitration: soft mixture by log-density, learned arbiter; vectorise routing | S1 gap; 4.7 | Oracle-gap metric; time per sample | Medium |
| P2-2 | Growth reliability: larger validation reservoirs; newborn-unit integration; GradMax ablation | 3.4(6); growth-stability paper [L] | S5 ladder: does growth fire and help at all? | Medium |
| P2-3 | Exercise merge/archive/revive at 30-60 regimes; ablate them | 3.2 | Event counts and `A_no_merge` | Low-medium |
| P3-1 | Real-data streams (cached text datasets) with a frozen encoder, against NCM/SLDA/kNN on the same features | 2.4 | Table on same features | Medium |
| P3-2 | Contradiction handling (S7, L3): label-evidence arbiter or context gate | 1 (place 4) | S7 with and without context | Medium |
| P3-3 | Plasticity monitor on long streams | 2.2 | Learn-speed of a fresh task vs expert age | Low |
| P3-4 | Adapter-level experts over a frozen backbone (assembly) | 9 | After P3-1 | High |

---

## 8. Roadmap for the MoE track

I propose rounds named M0-M6 (M for "MoE hardening"). Each starts with a PREREG file and ends with a report and a Claude review, as in the charter. Predictions below are written before any run and are expected to be partly wrong.

| Round | Goal | Key pre-registered statements |
|---|---|---|
| **M0** Fair fight | Items P0-1 to P0-5 | Calibration of the benchmark: tuned replay >= PhyloMoE - 1 point on S1 and S2-5; kNN-all >= PhyloMoE + 5 points; Pool C unique parameters about 0.8-1.0x of A. If these hold, the old verdicts are corrected as stated |
| **M1** Where do experts win? | Memory-capped stream; learned-function stream; 30 regimes | **W1:** at a 1.5 MB cap on 30 regimes, a tuned right-sized pool beats memory-matched kNN by >= 3 points and tuned replay by >= 5 points (paired, 10 seeds). **W2:** on a stream whose labels are a teacher function of random unit vectors, the pool beats kNN by >= 10 points at equal memory. **Kill rule:** if neither W1 nor W2 is met after M2, stop claiming an accuracy-per-memory advantage and re-scope the experts as compute-saver and abstention layer over an episodic memory |
| **M2** Small and honest | P1-1, P1-2 | False-alarm rate stays <= 2q with 10x fewer stored vectors; frontier curve beats kNN below 1.5 MB |
| **M3** Two-speed | P1-3 | Hybrid >= max(kNN, pool) at every memory point; fallback fires on abstain |
| **M4** Reliability | P2-1 to P2-3 | Oracle gap on S1 shrinks by >= 3 points; growth fires in the S5 ladder when a larger teacher is used and never when data-limited |
| **M5** Realistic features | P3-1 | Same-feature table with NCM/SLDA/kNN; pool wins only if W-style conditions hold |
| **M6** Assembly contract | P3-2, P3-4 | Interface (section 9); S7 with context |

**On R12b (the current vertical-growth round):** its six items assume experts are worth stacking and growing. I recommend pausing R12b's growth-trigger work until M0/M1 report, and finishing only its hygiene items (tests (d)/(e), report numbers from generated files) if the agent is mid-flight. Your call (decision 3).

---

## 9. How this plugs into the whole system, and what the geometry agent needs to know

- **Contract today:** a pool consumes unit vectors (`ContinualPool.observe/predict/summary`, `types.py`) and returns probabilities plus a calibrated "unknown" flag. Anything the geometry track produces on the unit sphere plugs in as the frontend. The spherical stream (Track Z, Z1) is a natural source, but a norm (salience) channel must be passed separately because a unit-sphere stream removes it.
- **Granularity gap:** GrowMoE routes whole samples or segments; LLM-style MoE routes tokens inside layers. First assembly should be **skill selection over adapters on a frozen backbone** (sample or segment level), the SEMA/MaRS/CaRE pattern, not token-level MoE. That is M6.
- **What not to import:** the legacy ISHMN MoE and `geometry/dynamic_moe.py` (fixed thresholds). They stay as history.
- **For the geometry agent in parallel:** nothing blocks it. One item is waiting for it: the Track Z round-1 speed-up metric is broken (an exact copy of the baseline scores 4.67x "speedup" with the repo's own `compute_speedup`) and needs an amendment before the nGPT arms run.

---

## 10. Decisions for you

(Your replies to these are recorded in Appendix C.1; items 1 and 4 are refined there. The M0 + M1 outcome and the new decision it raises are in Appendix D.6.)

1. **Gate:** adopt "beat kNN and tuned replay at equal memory, with a kill rule" (M1) as the test of the MoE track?
2. **Pretrained encoders:** allow a frozen pretrained encoder (e.g. the cached MiniLM) in the MoE track for realistic text streams (M5)? The final system may want its own encoders; this keeps the option open.
3. **R12b:** pause vertical-growth work until M0/M1 report?
4. **Episodic tier:** welcome a fast memory tier inside the MoE system (M3)?
5. **Handoff:** write `docs/superpowers/handoff/MOE_TRACK_HANDOFF.md` for the standby agent (rounds M0-M6, constraints, report template)?

---

## 11. Sources

All opened at abstract or summary level in this session unless marked [R]. Links in the form given by the search tool.

**Growing / dynamic MoE and routing:** [StaR-MoE 2605.17571](https://arxiv.org/abs/2605.17571) · [PASs-MoE 2601.13020](https://arxiv.org/abs/2601.13020) · [SAME 2602.01990](https://arxiv.org/abs/2602.01990) · [CP-MoE 2605.20247](https://arxiv.org/abs/2605.20247) · [CaRE 2602.03473](https://arxiv.org/abs/2602.03473) · [CARE-MoE (Mathematics 2026)](https://doi.org/10.3390/math14193557) · [Theory on MoE in Continual Learning 2406.16437](https://arxiv.org/abs/2406.16437) · [LLaVA-CMoE 2503.21227](https://arxiv.org/abs/2503.21227) · [Continual pre-training of MoEs 2503.05029](https://arxiv.org/abs/2503.05029) · [Equifinality in MoE 2604.14419](https://arxiv.org/abs/2604.14419) · [Routing Is Not Enough / SpawnLoRA 2609.03150](https://arxiv.org/abs/2609.03150) · [MoRAM 2506.21035](https://arxiv.org/abs/2506.21035) · [Zero-leakage reconstruction routing 2604.14375](https://arxiv.org/abs/2604.14375) · [Mechanistic forgetting analysis 2601.18699](https://arxiv.org/abs/2601.18699) (single author, lower confidence)

**Closest relatives:** [MaRS, ICLR 2026](https://openreview.net/forum?id=GGrLeik2qo) · [SEMA 2403.18886](https://arxiv.org/abs/2403.18886) · CN-DPM 2001.00689 [R] · [Open-world continual learning 2304.10038](https://arxiv.org/abs/2304.10038)

**Baselines and evaluation:** [SimpleCIL/APER 2303.07338](https://arxiv.org/abs/2303.07338) · [RanPAC 2307.02251](https://arxiv.org/abs/2307.02251) · [Simple baseline questioning pretrained models 2210.04428](https://arxiv.org/abs/2210.04428) · [Streaming LDA 1909.01520](https://arxiv.org/abs/1909.01520) · [kNN without the storage constraint 2305.09253](https://arxiv.org/abs/2305.09253) · [FLEX 2608.01437](https://arxiv.org/abs/2608.01437) · [CL-Bench 2606.05661](https://arxiv.org/abs/2606.05661)

**Theory, plasticity, growth:** [Knoblauch et al. 2006.05188](https://arxiv.org/abs/2006.05188) · [Loss of plasticity, Nature 2024](https://www.nature.com/articles/s41586-024-07711-7) · [Maintaining plasticity 2306.13812](https://arxiv.org/abs/2306.13812) · [Growth stability 2605.15435](https://arxiv.org/abs/2605.15435) · [SPHERE 2605.04712](https://arxiv.org/abs/2605.04712) · [GradMax 2201.05125](https://arxiv.org/abs/2201.05125) · Net2Net, Firefly [R]

**General continual learning at the frontier:** [Sparse memory finetuning 2510.15103](https://arxiv.org/abs/2510.15103) · [Nested Learning 2512.24695](https://arxiv.org/abs/2512.24695) · [Titans 2501.00663](https://arxiv.org/abs/2501.00663) · [Language models need sleep 2606.03979](https://arxiv.org/abs/2606.03979) · [Do language models need sleep 2605.26099](https://arxiv.org/abs/2605.26099) · [Modular memory position 2603.01761](https://arxiv.org/abs/2603.01761) · [LLM continual learning survey 2404.16789](https://arxiv.org/abs/2404.16789)

**Neuron types and expert shapes:** [HMoE 2408.10681](https://arxiv.org/abs/2408.10681) · [MoDSE 2409.12210](https://arxiv.org/abs/2409.12210) · [Mixture of Nested Experts 2407.19985](https://arxiv.org/abs/2407.19985) · [KAN forgetting, Rahman et al. 2511.12828](https://arxiv.org/abs/2511.12828) · [KAN or MLP, fairer comparison 2407.16674](https://arxiv.org/abs/2407.16674) · [KAN-CL 2605.12306](https://arxiv.org/abs/2605.12306) · [Progressive Neural Collapse 2505.24254](https://arxiv.org/abs/2505.24254)

**Not opened:** the ICLR/ICML 2026 merging papers and the CVPR 2026 spectral-MoE poster were seen only as listings. TIES-merging (Yadav et al. 2023) and conformal outlier p-values (Bates et al., Annals of Statistics 2023) are [R].

---

## Appendix A. Glossary

- **Regime:** a stretch of the stream drawn from one source (one task, one pixel shuffle, one data type).
- **Expert:** a small neural network that specialises in some regimes. **Router/gate:** the part that picks which expert handles an input.
- **Conformal p-value:** a "how surprising is this?" score calibrated so that for familiar inputs a fraction q (here 2%) are flagged by chance. **Benjamini-Hochberg:** a rule that controls the share of false flags when many inputs are tested together.
- **Signature:** an expert's memory of what it has seen (here, up to 256 stored unit vectors) used to compute the score above.
- **Probation:** the holding pen for inputs flagged as new, until enough similar ones accumulate.
- **kNN:** answer by looking up the k most similar stored examples. **NCM:** compare to one average per class. **SLDA:** a streaming linear classifier with one shared covariance.
- **Replay:** retrain on a small stored box of old examples mixed with the new.
- **Plasticity:** the ability to keep learning new things.
- **Unit vector / sphere:** a vector of length 1; all inputs to GrowMoE are scaled this way.
- **Parameter isolation:** giving each skill its own parameters so they cannot overwrite each other.

## Appendix B. Raw outputs (verbatim from `research/moe_diligence_2026-10/`)

### B1. Checks 1 (simple baselines, 10 seeds) - spike_baselines.py

```text


=== Final average accuracy over regimes (%), mean [95% CI] over 10 seeds; memory = params + stored vectors (float32) 
===

-- stream s2 (benchmark config: 5 permuted regimes, 400 train each)
method                                   accuracy           forgetting  memory KB stored vecs    params  experts
A_phylomoe                   87.04 [85.54, 88.54]   0.64 [-0.59, 1.87]       1450        1758    146272      3.5
B2_replay                    67.68 [65.30, 70.06] -1.70 [-3.37, -0.03]       1160        2048     34826      1.0
kNN5_all_samples             95.60 [94.72, 96.48] -0.68 [-0.87, -0.49]       1400        2800         0      1.0
kNN5_same_#vectors_as_A      93.12 [91.96, 94.28]    1.48 [0.78, 2.18]        879        1758         0      1.0
kNN5_same_BYTES_as_A         95.42 [94.48, 96.36] -0.50 [-0.85, -0.15]       1300        2600         0      1.0
NCM                          42.60 [38.58, 46.62] 18.72 [16.16, 21.28]          5          10         0      1.0
SLDA                         61.18 [57.93, 64.43] 16.70 [14.52, 18.88]         69         138         0      1.0

-- stream s1 (benchmark config: 7 data types + revisits)
method                                   accuracy           forgetting  memory KB stored vecs    params  experts
A_phylomoe                   69.81 [67.79, 71.83]    6.36 [3.85, 8.88]       2518        2979    263290      6.3
B2_replay                    61.53 [60.81, 62.26]   9.29 [7.95, 10.63]       1170        2048     37406      1.0
kNN5_all_samples             77.95 [77.35, 78.56]    1.44 [0.99, 1.89]       1700        3400         0      1.0
kNN5_same_#vectors_as_A      77.11 [76.34, 77.89]    1.97 [1.38, 2.56]       1446        2892         0      1.0
kNN5_same_BYTES_as_A         77.84 [77.23, 78.44]    1.58 [0.94, 2.22]       1656        3312         0      1.0
NCM                          57.45 [56.70, 58.19]    2.99 [2.42, 3.56]         15          30         0      1.0
SLDA                         60.58 [59.85, 61.31]    6.18 [5.31, 7.05]         79         158         0      1.0

=== Event counts per run for A (mean over seeds) ===
s1 {'alarm': 5.9, 'extend': 5.1, 'spawn': 6.3}
s2 {'alarm': 2.3, 'extend': 3.7, 'spawn': 3.5}
```

### B2. Check 2 (training budget, 10 seeds) - spike_budget.py

```text


=== Accuracy (%) vs optimiser steps per routed batch, 10 seeds, mean [95% CI] ===

-- s2   (kNN with all stored samples: 95.6)
method inner_steps               accuracy           forgetting  experts
A                1   87.04 [85.54, 88.54]   0.64 [-0.59, 1.87]      3.5
A                5   90.18 [89.36, 91.00]   0.24 [-0.17, 0.65]      4.4
A               20   91.16 [89.63, 92.69]   1.24 [-0.02, 2.50]      3.4
B2               1   67.68 [65.30, 70.06] -1.70 [-3.37, -0.03]      1.0
B2               5   88.64 [87.44, 89.84]  -0.30 [-1.80, 1.20]      1.0
B2              20   92.48 [91.60, 93.36]   0.84 [-0.23, 1.91]      1.0

-- s1   (kNN with all stored samples: 77.95)
method inner_steps               accuracy           forgetting  experts
A                1   69.81 [67.79, 71.83]    6.36 [3.85, 8.88]      6.3
A                5   72.55 [69.07, 76.03]   8.06 [4.80, 11.32]      6.0
A               20   75.81 [74.12, 77.50]    5.67 [3.33, 8.00]      4.5
B2               1   61.53 [60.81, 62.26]   9.29 [7.95, 10.63]      1.0
B2               5   76.20 [75.06, 77.34]    6.72 [5.06, 8.39]      1.0
B2              20   79.26 [78.31, 80.20]    4.41 [3.40, 5.41]      1.0
```

### B3. Check 3 (number of regimes, 3 seeds) - spike_scale.py

```text

seed 0 n_regimes 5 done (8s)
seed 0 n_regimes 15 done (35s)
seed 0 n_regimes 30 done (120s)
seed 1 n_regimes 5 done (127s)
seed 1 n_regimes 15 done (156s)
seed 1 n_regimes 30 done (252s)
seed 2 n_regimes 5 done (258s)
seed 2 n_regimes 15 done (282s)
seed 2 n_regimes 30 done (372s)

=== Permuted digits: final average accuracy (%) vs number of regimes; 3 seeds, mean [95% CI] ===
regimes method                                 accuracy  experts  memory KB  sec/run
      5 A_steps1                   87.67 [83.50, 91.83]      3.0       1258      2.5
      5 A_steps20                  91.20 [86.78, 95.62]      2.0        838      2.5
      5 B2_steps1                  67.20 [62.16, 72.24]      1.0       1160      0.1
      5 B2_steps20                 92.07 [91.31, 92.83]      1.0       1160      1.2
      5 kNN_all_samples            95.40 [91.07, 99.73]      1.0       1000      0.1
      5 kNN_same_bytes_as_A20      94.47 [92.95, 95.98]      1.0        838      0.1

     15 A_steps1                   86.71 [84.89, 88.54]     10.7       4472      8.5
     15 A_steps20                  87.53 [86.09, 88.98]     11.0       4532     12.3
     15 B2_steps1                  49.98 [48.59, 51.37]      1.0       1160      0.3
     15 B2_steps20                 84.36 [82.20, 86.51]      1.0       1160      3.5
     15 kNN_all_samples            94.73 [90.58, 98.88]      1.0       3000      0.7
     15 kNN_same_bytes_as_A20      94.73 [90.58, 98.88]      1.0       4532      0.6

     30 A_steps1                   85.71 [80.50, 90.92]     22.3       9331     31.5
     30 A_steps20                  81.99 [70.05, 93.92]     19.0       7929     41.6
     30 B2_steps1                  41.83 [38.47, 45.20]      1.0       1160      0.8
     30 B2_steps20                 72.29 [70.60, 73.98]      1.0       1160      7.0
     30 kNN_all_samples            94.69 [91.71, 97.66]      1.0       6000      3.4
     30 kNN_same_bytes_as_A20      94.56 [91.30, 97.82]      1.0       7929      3.0
```

### B4. Check 4 and 6 (memory frontier, conformal floor, 3 seeds) - spike_frontier.py

```text

seed 0 A_default(h128,64cls,256 protos): acc=84.9 mem=9966KB experts=24 (36s)
seed 0 A_small(h32,10cls,64 protos,64 replay): acc=83.6 mem=3525KB experts=34 (77s)
seed 0 A_tiny(h16,10cls,32 protos,32 replay): acc=20.0 mem=222KB experts=4 (87s)
seed 0 done (117s)
seed 1 A_default(h128,64cls,256 protos): acc=84.2 mem=9224KB experts=22 (152s)
seed 1 A_small(h32,10cls,64 protos,64 replay): acc=56.1 mem=2281KB experts=22 (186s)
seed 1 A_tiny(h16,10cls,32 protos,32 replay): acc=18.0 mem=167KB experts=3 (195s)
seed 1 done (224s)
seed 2 A_default(h128,64cls,256 protos): acc=88.1 mem=8804KB experts=21 (258s)
seed 2 A_small(h32,10cls,64 protos,64 replay): acc=84.5 mem=3421KB experts=33 (294s)
seed 2 A_tiny(h16,10cls,32 protos,32 replay): acc=14.4 mem=111KB experts=2 (300s)
seed 2 done (328s)

=== 30 permuted-digit regimes (400 train each = 12,000 samples = 6,000 KB raw): accuracy vs memory, 3 seeds ===
method                                      memory KB               accuracy %  experts
A_tiny(h16,10cls,32 protos,32 replay)             167     17.48 [10.34, 24.62]      3.0
kNN cap=500                                       250     33.52 [28.07, 38.98]      1.0
B2 replay buf=512 (20 steps)                      392     56.20 [53.67, 58.73]      1.0
kNN cap=1000                                      500     53.32 [50.67, 55.98]      1.0
kNN cap=2000                                     1000     74.58 [72.05, 77.11]      1.0
B2 replay buf=2048 (20 steps)                    1160     72.29 [70.60, 73.98]      1.0
kNN cap=4000                                     2000     85.86 [84.45, 87.26]      1.0
A_small(h32,10cls,64 protos,64 replay)           3075    74.74 [34.55, 114.94]     29.7
kNN cap=8000                                     4000     91.90 [89.81, 93.99]      1.0
B2 replay buf=8192 (20 steps)                    4232     79.30 [77.77, 80.83]      1.0
kNN cap=12000                                    6000     93.80 [91.01, 96.59]      1.0
A_default(h128,64cls,256 protos)                 9331     85.71 [80.50, 90.92]     22.3
```

### B5. Check 5 (front-end swap, 10 seeds) - spike_frontend.py

```text

encoder loaded (7s)

=== S1 final average accuracy (%), 10 seeds, mean [95% CI] ===
A_hash       69.81 [67.79, 71.83]
A_minilm     72.62 [69.95, 75.29]
kNN_hash     77.95 [77.35, 78.56]
kNN_minilm   81.96 [81.49, 82.43]

=== Per-regime final accuracy (%), mean over seeds ===
regime              A_hash    A_minilm    kNN_hash  kNN_minilm
digits                86.0        82.1        97.1        97.1
wine                  88.6        88.2        92.6        92.6
breast_cancer         78.6        79.5        94.9        94.9
sinusoid              95.9        95.7       100.0       100.0
ar2                   50.3        47.0        53.1        53.1
pydoc_text            35.1        44.5        43.1        55.1
stdlib_code           54.1        71.3        64.9        80.9

=== Own-regime 1-NN separability (%), mean over seeds ===
regime             hash   minilm
digits            100.0    100.0
wine               99.7     99.7
breast_cancer      99.2     99.2
sinusoid          100.0    100.0
ar2                76.3     76.3
pydoc_text         72.0     77.6
stdlib_code        89.7     92.0
```

### B6. Checks of the second agent's ideas, Pool C accounting, routing cost - spike_misc.py

```text

=== F. NormGatedLinear on unit-sphere inputs ===
max |NormGated(x) - plain linear with shifted bias| on unit vectors: 1.1920928955078125e-07
same comparison on inputs with varying norm (so the gate can act): 39.79878234863281

=== H. Pool C parameter accounting on S2 (benchmark config) ===
seed 0: experts=7  summary n_params=371392  unique params=120640  (own deltas 78848 + shared base 41792 = 120640)  
acc=72.8
          Pool A same seed: experts=3 summary n_params=125376 acc=86.6
seed 1: experts=5  summary n_params=265280  unique params=98112  (own deltas 56320 + shared base 41792 = 98112)  
acc=75.2
          Pool A same seed: experts=3 summary n_params=125376 acc=86.8
seed 2: experts=5  summary n_params=265280  unique params=98112  (own deltas 56320 + shared base 41792 = 98112)  
acc=73.8
          Pool A same seed: experts=3 summary n_params=125376 acc=89.6

=== E. Intrinsic-dimension estimates at probation-cluster sizes (true dimension known; embedded in R^128, 
unit-normalised) ===
true d     N |    repo TwoNN mean+-sd |   agent formula mean+-sd
     2    32 |     3.23 +-  0.74      |      4.73 +-  1.38
     2    64 |     4.56 +-  0.70      |      6.91 +-  1.28
     2   128 |     7.25 +-  0.84      |     11.55 +-  1.57
     2   256 |    10.85 +-  1.14      |     17.38 +-  1.92
     2   512 |    16.01 +-  0.73      |     25.68 +-  1.28
     5    32 |     4.54 +-  1.13      |      7.01 +-  1.99
     5    64 |     4.13 +-  0.59      |      6.29 +-  1.43
     5   128 |     4.02 +-  0.54      |      5.70 +-  0.84
     5   256 |     4.34 +-  0.42      |      6.20 +-  0.79
     5   512 |     4.34 +-  0.23      |      6.19 +-  0.39
    10    32 |     8.02 +-  1.52      |     11.36 +-  3.37
    10    64 |     7.90 +-  1.08      |     11.88 +-  2.05
    10   128 |     8.15 +-  0.77      |     11.71 +-  1.35
    10   256 |     8.55 +-  0.68      |     12.15 +-  1.31
    10   512 |     8.40 +-  0.59      |     11.96 +-  0.85
    20    32 |    13.50 +-  2.90      |     18.58 +-  6.03
    20    64 |    14.61 +-  2.83      |     21.37 +-  5.85
    20   128 |    15.12 +-  1.55      |     21.93 +-  3.28
    20   256 |    15.51 +-  1.38      |     21.86 +-  2.29
    20   512 |    15.69 +-  0.73      |     22.59 +-  1.38

=== D. Same inputs, different labelling function: needed expert width (accuracy %, 3 seeds) ===
labels                   w=2    w=4    w=8    w=16   w=32   w=64   w=128 
linear labels              84.5   82.1   84.6   83.9   81.5   78.4   78.9
deep teacher (w32,d2)      52.7   52.4   53.0   53.2   59.9   58.4   57.0
(inputs are IDENTICAL random unit vectors in both rows: same N, same concentration, same intrinsic dimension)

=== G. Routing (predict) cost vs number of live experts, batch of 64 ===
E=   8:      7.1 ms per batch of 64  (  0.11 ms per sample)
E=  32:     25.9 ms per batch of 64  (  0.41 ms per sample)
E= 128:    102.1 ms per batch of 64  (  1.60 ms per sample)
E= 512:    451.3 ms per batch of 64  (  7.05 ms per sample)
```


---

## Appendix C. Addendum after your replies (2026-10-09, same day)

### C.1 Your decisions, recorded

| # | Question | Your answer | What it means |
|---|---|---|---|
| 1 | Gate for the track ("beat kNN and tuned replay at equal memory") | You pointed out a lookup wins only because it has seen the situations | Accepted in part. The gate is now two-sided: similarity-type streams (a lookup is the bar to beat) and function-type streams (the bar is a tuned single network). See C.2 and C.6 |
| 2 | Frozen pretrained encoder | "For what reason?" | Only needed to test the MoE on real text without first building eyes for it. Not needed now (M5 only) |
| 3 | Pause R12b / vertical growth | Yes, let it sit | Done. First see whether dynamic MoE helps at all; vertical and horizontal growth come later |
| 4 | Episodic memory tier | "Why, pros and cons?" | Answered in the chat reply; decide after M1; test it as an optional arm (M3) |
| 5 | Handoff | If deep coding, the standby agent; otherwise Claude | Claude does M0/M1 (light: baselines, accounting, streams, runs). Gemini + handoff only for M2 onward (changes inside the gate, expert and growth code) |
| 6 | Geometry agent | Working; to be examined later | Nothing to do now |

### C.2 What the kNN result does and does not mean

You are right that a lookup wins because it has seen the situations: the test items are new, but they sit close to stored ones, and a lookup can only interpolate between what it stored. That is exactly why the earlier tests could not show the specialists' value. Two things are still true: (a) at equal memory the lookup still won (93.1% with the same number of stored vectors as the pool, vs 87.0%), so our experts also spend memory badly; (b) the finding is a statement about our tests, not a verdict on MoE. C.4 and C.5 test the case a lookup cannot handle.

### C.3 Is the pool efficient? (measured, 24 experts after 30 regimes)

Only one expert answers each input, but the gate has to scan every expert's stored vectors first.

| Method | Multiply-adds per prediction | ms per sample (batch of 64) | ms per sample (one at a time) |
|---|---|---|---|
| PhyloMoE (as implemented) | 1,597,440 (gate 1,556,480 + the answering expert 40,960) | 0.387 | 9.94 |
| kNN, everything stored (12,000) | 1,894,400 | 0.047 | 0.42 |
| kNN, 4,000 stored (2 MB) | 512,000 | 0.013 | 0.13 |
| One replay network | 34,048 | 0.003 | 0.13 |

The gate costs 38 times the expert it selects, and total work per prediction is 0.84 times a lookup over everything. Wall-clock is worse than the multiply-add count suggests because the gate is a Python loop over experts. So sparse activation saves expert work but the current gate gives it back. A cheap first-pass gate (one summary vector per expert, then the full test on the top few experts) and scalar-only calibration (4.6) could plausibly cut this several-fold [H, not measured].

### C.4 Pilot: where a lookup fails (labels are a hidden function of the input; chance 12.5%; 5 seeds)

| Hidden function | Training examples | kNN lookup | Small trained network | Difference |
|---|---|---|---|---|
| width 8, depth 1 | 300 | 21.7 | 45.0 | +23.3 |
| | 1,000 | 22.8 | 59.0 | +36.2 |
| | 3,000 | 25.2 | 68.6 | +43.5 |
| | 10,000 | 25.9 | 78.2 | +52.3 |
| width 32, depth 2 | 300 | 20.6 | 39.8 | +19.1 |
| | 1,000 | 21.6 | 52.0 | +30.4 |
| | 3,000 | 23.3 | 56.3 | +33.0 |
| | 10,000 | 24.4 | 62.3 | +37.9 |

With 10,000 stored examples a lookup is still near chance; a small network reaches 62-78%. In natural data, similar inputs usually share labels (so lookups are strong, as in section 4); in tasks where the answer is computed (rules, arithmetic, code behaviour, planning) they do not.

### C.5 Pilot: a continual stream of function tasks (3 seeds; chance 12.5%)

Regime r occupies its own region of the sphere and has its own hidden function (a different random teacher network), so a gate can find the regime from the input but neighbours inside a regime do not share labels. 800 training examples per regime, two revisits.

| Regimes | kNN (everything) | Replay net (20 steps) | PhyloMoE (20 steps) | One expert per regime (oracle) |
|---|---|---|---|---|
| 5 | 21.3 | 37.3 [32.4, 42.1] | **46.5 [42.6, 50.5]** | 41.8 |
| 15 | 21.9 | 24.5 [22.3, 26.6] | **40.6 [36.3, 44.9]** | 39.5 |
| 30 | 21.7 | 18.8 [18.0, 19.6] | **31.6 [24.2, 39.0]** | 39.7 |

Paired difference, PhyloMoE minus replay net (percentage points): +9.2 [+0.7, +17.8] at 5 regimes, +16.1 [+13.0, +19.3] at 15, +12.8 [+4.9, +20.6] at 30; all nine seed pairs positive. At 30 regimes the pool is 8 points below the oracle: routing and fragmentation cost something (27 experts for 30 regimes).

**Is it just capacity?** Same stream, single replay network made bigger, deeper, or given a bigger buffer (20 steps, 3 seeds):

| Regimes | Best single network tried | Accuracy | Memory | Pool |
|---|---|---|---|---|
| 15 | width 512, buffer 8,192 | 32.8 [31.3, 34.3] | 4.6 MB | 40.6 at 6.2 MB |
| 30 | width 512, buffer 8,192 | 26.0 [24.2, 27.9] | 4.6 MB | 31.6 at 10.8 MB |

Bigger networks and bigger buffers help (width 128 gave 24.5 and 18.8) but did not close the gap. The comparison is not memory-matched (the pool uses more), so a fully matched comparison, with the single network's width and buffer swept up to the pool's memory, belongs in M1.

### C.6 Revised reading of your bet

- **Similarity-type tasks** (answer = what a similar past case had): a lookup is the bar to beat and is hard to beat. The role for experts there is compression and bounded memory, not accuracy. This is where the episodic memory tier belongs.
- **Function-type tasks** (answer = computed from the input): a lookup is useless, one fixed network runs out of room as regimes multiply, and the growing pool clearly helps (C.5, pilot). **This is the setting where your bet is supported.**
- **Efficiency:** plausible, not yet earned (C.3).
- **Brain argument (neuroscience from memory, not re-checked today [R]):** supports sparse activity, a fast episodic store plus a slow learner with replay during sleep (complementary learning systems theory, built to explain catastrophic interference), and some synaptic-level structural change. It does not directly support creating a whole new full-size expert for every new situation (the adult cortex keeps a fixed set of areas; adult neurogenesis is small, mostly in one region, and debated). Its efficiency comes from sparse, event-driven hardware where memory and computation sit together, which software on a CPU or GPU does not inherit automatically.

### C.7 Revised first experiment pack (M0 + M1, light coding, Claude implements after your go)

| Part | What | Why |
|---|---|---|
| Pre-registration first | `research/growmoe_results/m0/PREREG.md`, written before any run, read-only after | Charter |
| Corrected record | Tuned training budget per method (chosen on dev seeds), kNN / class-mean / streaming-LDA baselines, memory and multiply-add columns, Pool C unique-parameter fix, 30-regime permuted digits | Section 4 |
| New streams | Function regimes (C.5) at 5, 15, 30 (and 60 if affordable); a memory-capped permuted-digits variant | Where the bet can be tested |
| Resource-matched baseline | Single replay network with width and buffer swept up to the pool's memory | C.5 caveat |
| Win condition W1 (function regimes) | PhyloMoE beats the best resource-matched single network by at least 5 points at 15 and 30 regimes (paired, 10 seeds, interval excludes 0). Kill rule: if not met, the MoE's value must be argued on other axes | Pilot suggests it will hold; the test must be able to fail |
| Reported, not gated yet | Predict cost per sample vs lookup; oracle gap (31.6 vs 39.7 at 30 regimes) | Targets for M2 and M4 |
| Not touched | Pools A and B logic, Phase 1 files, `ishmn/`, `geometry/`, anything the geometry agent edits | Scope |

New code is additive: `growmoe/memory_baselines.py`, `growmoe/accounting.py`, a function-regime stream builder, a run script, tests, plus a three-line fix to `pools/keyed.py::summary()`.

### C.8 Raw outputs for this addendum (verbatim from `research/moe_diligence_2026-10/`)

### C-a. Per-prediction cost - spike_efficiency.py

```text

PhyloMoE after 30 regimes: 24 live experts, 6080 stored signature prototypes in total
method                              ms/sample B=64  ms/sample B=1   multiply-adds/sample
PhyloMoE (as implemented)                    0.387          9.941              1,597,440
kNN, 12,000 stored (everything)              0.047          0.419              1,894,400
kNN, 4,000 stored (2 MB)                     0.013          0.131                512,000
Replay network (one MLP)                     0.003          0.132                 34,048

PhyloMoE multiply-adds per sample, split: gate (scan every expert's stored vectors) = 1,556,480 as implemented 
(778,240 if the score were computed once); the ONE expert that answers = 40,960.
=> the gate costs 38x the expert it selects; total vs kNN-everything: 0.84x
```

### C-b. Where a lookup fails - spike_lookup_fails.py

```text

teacher w8 d1 done (41s)
teacher w32 d2 done (87s)

=== Hidden-function labels (8 classes, chance 12.5%), accuracy % on 2000 fresh test points, 5 seeds, mean [95% CI] ===
teacher       train N             kNN lookup      small trained net    net minus kNN
w8  depth 1        300   21.66 [20.66, 22.66]   44.95 [41.07, 48.83]           +23.3
w8  depth 1       1000   22.82 [22.06, 23.58]   59.00 [55.50, 62.50]           +36.2
w8  depth 1       3000   25.17 [23.39, 26.95]   68.63 [65.27, 71.99]           +43.5
w8  depth 1      10000   25.90 [24.34, 27.46]   78.19 [76.35, 80.03]           +52.3
w32 depth 2        300   20.63 [19.21, 22.05]   39.77 [37.50, 42.04]           +19.1
w32 depth 2       1000   21.63 [19.71, 23.55]   52.04 [50.79, 53.29]           +30.4
w32 depth 2       3000   23.31 [22.33, 24.29]   56.29 [52.70, 59.88]           +33.0
w32 depth 2      10000   24.39 [23.67, 25.11]   62.32 [60.12, 64.52]           +37.9
```

### C-c. Function-regime stream, 3 seeds combined - spike_function_stream.py + aggregate_function_stream.py

```text

files combined: ['function_stream_raw_seeds_0.json', 'function_stream_raw_seeds_1.json', 
'function_stream_raw_seeds_2.json']

=== Function regimes (8 classes, chance 12.5%): final average accuracy %, 3 seeds, mean [95% CI] ===
regimes method                                         accuracy  experts  memory KB
      5 kNN stores everything              21.32 [19.71, 22.92]      1.0       2800
      5 kNN same bytes as PhyloMoE         21.38 [19.05, 23.72]      1.0       2341
      5 Replay net 1 step(s)               25.20 [19.34, 31.06]      1.0       1160
      5 Replay net 20 step(s)              37.28 [32.43, 42.14]      1.0       1160
      5 PhyloMoE 1 step(s)                 42.63 [35.53, 49.74]      6.0       2451
      5 PhyloMoE 20 step(s)                46.53 [42.59, 50.48]      5.7       2341
      5 Oracle: one expert per regime      41.78 [39.03, 44.53]      5.0        836

     15 kNN stores everything              21.85 [20.95, 22.75]      1.0       6800
     15 kNN same bytes as PhyloMoE         21.76 [20.58, 22.94]      1.0       6202
     15 Replay net 1 step(s)               19.68 [17.89, 21.48]      1.0       1160
     15 Replay net 20 step(s)              24.47 [22.31, 26.62]      1.0       1160
     15 PhyloMoE 1 step(s)                 33.95 [18.41, 49.49]     17.3       7042
     15 PhyloMoE 20 step(s)                40.61 [36.28, 44.94]     15.0       6202
     15 Oracle: one expert per regime      39.49 [36.95, 42.04]     15.0       2508

     30 kNN stores everything              21.73 [20.91, 22.55]      1.0      12800
     30 kNN same bytes as PhyloMoE         21.49 [20.06, 22.92]      1.0      10833
     30 Replay net 1 step(s)               16.37 [15.23, 17.51]      1.0       1160
     30 Replay net 20 step(s)              18.82 [18.01, 19.64]      1.0       1160
     30 PhyloMoE 1 step(s)                 30.58 [20.46, 40.69]     33.7      13537
     30 PhyloMoE 20 step(s)                31.59 [24.20, 38.97]     27.3      10833
     30 Oracle: one expert per regime      39.65 [36.94, 42.36]     30.0       5016

paired difference PhyloMoE(20 steps) minus Replay net(20 steps), percentage points:
    5 regimes: +9.2 [+0.7, +17.8]  per-seed: [5.4, 12.0, 10.3]
   15 regimes: +16.1 [+13.0, +19.3]  per-seed: [15.1, 15.8, 17.6]
   30 regimes: +12.8 [+4.9, +20.6]  per-seed: [14.5, 9.1, 14.7]
```

### C-d. Bigger single network - spike_bigger_net.py

```text


=== Function regimes: a single replay network, bigger and deeper (20 steps), 3 seeds, mean [95% CI] ===
regimes  width depth buffer             accuracy %  memory KB  mult-adds/sample
     15    128     1   2048   24.47 [22.31, 26.62]       1159            33,792
     15    512     1   2048   27.78 [27.57, 28.00]       1545           132,096
     15    512     1   8192   32.83 [31.34, 34.33]       4617           132,096
     15    512     2   2048   26.73 [25.59, 27.87]       2060           263,168
     15   2048     1   2048   30.08 [28.44, 31.73]       3087           525,312
     30    128     1   2048   18.82 [18.01, 19.64]       1159            33,792
     30    512     1   2048   22.06 [21.34, 22.78]       1545           132,096
     30    512     1   8192   26.03 [24.20, 27.85]       4617           132,096
     30    512     2   2048   20.41 [19.50, 21.32]       2060           263,168
     30   2048     1   2048   25.08 [24.68, 25.48]       3087           525,312
```

---

## Appendix D. The M0 + M1 run: results (2026-10-09, evening)

Everything in this appendix is measured [M] by `research/growmoe_results/m0/run_m0.py` from per-seed records; the two tables are copied from `RESULTS_M0.md` by script. Rules were fixed before the run in `PREREG.md`; the one change made after seeing dev results and before any test seed is in `AMENDMENT.md` (the size grid of the best single network was extended because its best setting sat on the corner of the grid).

### D.1 What was run

* **Streams:** permuted digits with 5 regimes (the Phase 1 benchmark) and with 30 regimes; the mixed-types stream S1; the same stream with a frozen pretrained text encoder (S1-PRE, MiniLM, approved by you); and the new **function-regime streams** F-5, F-15, F-30, where each regime has its own hidden random teacher network so that nearest neighbours do not share labels.
* **Methods:** A = our growing pool (PhyloMoE, Phase 1 settings, training budget tuned); B2 = a single replay network (tuned budget); **B2\*** = the best single network we could build (searched over width up to 4096, replay buffer up to 32,768 and budget; chosen: width 4096, buffer 8192, 20 steps per batch); B4 = one expert per regime with the regime handed to it (an upper bound, not a method); B1 = joint training on everything seen; kNN (store everything, and at the same bytes as A); nearest class mean; streaming LDA; Pool C (parameter count only).
* **Protocol:** dev seeds 100-102 for every tuning decision, then 10 test seeds (0-9), 95% t intervals, paired differences. No budget search was left with its optimum on the edge.
* **Integrity:** the fast runner reproduces the original benchmark's accuracy matrix and event log exactly on six full-size method/stream pairs (largest difference 0.0); the vectorised prediction is bit-identical to the original loop for every routing rule; 412 GrowMoE tests pass.

### D.2 The eight statements (decided by `growmoe.fairfight.verdicts`)

| ID | Rule | Verdict | Expected | Numbers |
|---|---|---|---|---|
| A1 | S2-5 and S1: mean(B2 tuned) - mean(A tuned) >= -1.0 point (PASS = Phase 1's 'A beats replay' is not supported) | **PASS** | PASS (as expected) | (B2 - A) S2-5: +1.5 pts [+0.3, +2.7] (n=10); S1: +6.7 pts [+3.3, +10.1] (n=10) |
| A2 | S2-5 and S1: mean(kNN-all) - mean(A tuned) >= +5.0 points | **PARTIAL** | PASS (**AGAINST expectation**) | (kNN-all - A) S2-5: +4.6 pts [+3.4, +5.9] (n=10); S1: +5.4 pts [+2.1, +8.7] (n=10) |
| A3 | S2-5: unique(C) / unique(A) in [0.6, 1.2] and reported(C) > 2 x unique(C) | **PASS** | PASS (as expected) | unique(C)/unique(A) = 0.85; reported/unique for C = 2.68 (n=5) |
| B1 | F-15 and F-30: mean paired A tuned - B2* >= +5.0 points and the 95% interval excludes 0 | **FAIL** | PASS (**AGAINST expectation**) | (A - B2*) F-15: +4.1 pts [+2.9, +5.4] (n=10); F-30: +2.6 pts [+0.7, +4.5] (n=10) |
| B2 | F-15 and F-30: A tuned - kNN-all >= +10.0 points | **PASS** | PASS (as expected) | (A - kNN-all) F-15: +18.6 pts [+17.0, +20.2] (n=10); F-30: +11.9 pts [+10.0, +13.8] (n=10) |
| B3 | F-30: B4 - A >= +5.0 points (reported; no gate) | **PASS** | PASS (as expected) | (B4 - A) +13.5 pts [+11.8, +15.3] (n=10) |
| C1 | S2-30: predict multiply-adds(A) >= 0.5 x multiply-adds(kNN-all) (PASS = no compute saving; reported, no gate) | **FAIL** | PASS (**AGAINST expectation**) | A / kNN-all multiply-adds = 0.38 (n=10) |
| C2 | A tuned: S1-PRE - S1 >= +1.0 point (reported) | **PASS** | PASS (as expected) | (S1-PRE - S1) +3.8 pts [+0.7, +6.9] (n=10) |

**Kill rule (written before the run):** B1 FAIL means the dynamic pool is not shown to beat a dense network on streams that need computed answers; the MoE track stops claiming an accuracy advantage and re-scopes to a compute saver and abstention layer over an episodic memory.

### D.3 What the numbers say, in plain words

1. **Phase 1's headline is gone (A1).** With a tuned replay network, plain replay is ahead of A on permuted digits (92.5% vs 91.0%) and on the mixed-types stream (79.3% vs 72.5%). A does still beat a *small* replay net when there are many regimes (30 permuted-digit regimes: 83.0% vs 72.8%; function streams: +16.2 and +14.7 points). That is a capacity effect: the big single network closes most of that gap.
2. **A lookup wins when the question is "have I seen this?" (A2, partial).** kNN is ahead of A by 4.6 points (5 permuted-digit regimes), 5.4 (mixed types) and 12.1 (30 permuted-digit regimes), all intervals above zero. The statement is only PARTIAL because 4.6 is 0.4 below the +5 bar we set.
3. **On function tasks the pool crushes a lookup (B2) and edges a big single network (B1 fails).** Against kNN: +18.6 and +11.9 points. Against the best single network: +4.1 [+2.9, +5.4] at 15 regimes and +2.6 [+0.7, +4.5] at 30, below the +5 bar at both. The edge is not growing with the number of regimes (+0.4 at 5, +4.1 at 15, +2.6 at 30). At 15 regimes the interval still contains +5.0; at 30 it does not.
4. **The structure is strong; our learned version realises only part of it.** Perfectly routed experts beat the best single network by +6.6, +11.3 and +16.2 points at 5, 15 and 30 regimes (B3: oracle minus A is +13.5 at 30 regimes). Computed by the script, not a pre-registered statement:

| Stream | A - B2* (pts) | B4 - B2* (pts) | Share of the oracle-routing gain A realises | Memory A / B2* | Predict MACs A / B2* | Gate share of A's MACs |
|---|---|---|---|---|---|---|
| F-5 | +0.4 | +6.6 | 6% | 0.28 | 0.19 | 83% |
| F-15 | +4.1 | +11.3 | 37% | 0.68 | 0.48 | 93% |
| F-30 | +2.6 | +16.2 | 16% | 1.23 | 0.83 | 96% |

5. **The efficiency half of the bet is not supported yet.** The gate (scanning every stored signature vector) is 83-96% of A's multiply-adds, so "only one expert is active" saves almost nothing; A's memory grows with the number of regimes (1.23 times the best single network's at 30 regimes) while the single network's stays fixed. Against a full lookup A does need fewer multiply-adds on 30 permuted-digit regimes (0.38 times, C1) but is 12 points less accurate there, and the small replay network is about twenty times cheaper than A per prediction. By contrast the oracle-routed experts use about a quarter of the single network's memory at 15 regimes and about 30 times fewer multiply-adds per prediction (34 vs 1050 thousand): modularity is efficient when the routing is free and right.
6. **Pool C's parameter count was inflated 2.68 times (A3)**, and C is also less accurate than A on permuted digits (76.0% vs 91.0%).
7. **A better text encoder helps (C2: +3.8 points)**, but even then A (76.3%) is behind plain lookup (82.0%) and tuned replay (81.8%) on the mixed-types stream.
8. **Most of the lifecycle still never runs.** Across the 70 benchmark runs of A, growth fired in 3 and merge, archive, dormant and prune in none.

### D.4 Exploratory follow-ups (run after the verdicts; NOT pre-registered; no statement reads them)

* **Where does A lose to perfect routing?** On fresh seeds 200-201 (F-15, F-30), 92.5-99.3% of held-out inputs go to the expert that owns their regime, and the pool makes 13-16 experts for 15 regimes and 28-29 for 30. Yet only 7-20% of regimes are within 3 points of the oracle's accuracy, and on the rest the gap averages 10-15 points. So the loss is in the quality of the experts the pool trains, not in test-time routing. Why is open.
* **Dev-seed ablation of A's expert recipe (F-15):** fresh instead of inherited initialisation 40.3% (baseline 40.1%); no per-expert replay 33.8% (the pool then splits regimes over about twice as many experts); never extend, always spawn 41.1% (28 experts for 15 regimes); all three together 27.1%. None of the simple knobs closes the 8-point dev gap to the oracle (48.5%).
* **Speed:** one pool run is 1.4x-2.3x faster than the original code path with identical final accuracy; most of the time is training. Evaluation-bound methods gain much more (kNN on 15 function regimes: 2.6 s to 0.2 s). The confirmatory run took 39 minutes of wall-clock on two workers (61 minutes of compute).

### D.5 Limits of this test (read before leaning on the FAIL)

* A's settings other than the training budget were frozen from Phase 1 (tuned on digits); the single network got a 12-configuration x 4-budget search on the function dev stream. This asymmetry was pre-registered and favours the baseline. It is the main reason to read the FAIL as "not shown for the current pool" rather than "impossible".
* One family of function tasks (random teacher networks in separated regions); real tasks differ. The streams were built to defeat a lookup, not to be realistic.
* B2\* is deliberately not memory-limited: at 15 regimes A uses 0.68 of its memory, at 30 regimes 1.23 times.
* B1 (joint training) was tuned at width 128 only and reused at 512, so it is not a tuned ceiling; it enters no statement.
* Ten seeds; the interval for B1 at 15 regimes still contains +5.0.

### D.6 Decision for you

The pre-registered default is to follow the kill rule: stop claiming an accuracy advantage for the current pool and re-scope (compute saver and abstention layer over an episodic memory). Two things complicate the plain reading. The oracle numbers say the *structure* is worth a lot (accuracy, memory and compute together) when the routing is right. And the diagnostic says the shortfall is in expert quality, not routing, which is a specific, testable cause. My recommendation, in order:

1. **Accept the verdict as written** for the current pool; it is the honest outcome of the test we agreed on.
2. **Run one time-boxed diagnostic round before deciding the track's fate** (about half a day, mostly compute): (a) give A the true regime ids during training and test but keep its own expert-training code, to separate "wrong data reaches the expert" from "the expert recipe is weaker"; (b) add A's ingredients (per-expert replay mixing, infant learning rate, probation delay, inherited initialisation) to the oracle-routed experts one at a time. If one ingredient explains the gap and can be fixed, pre-register one confirmatory test on fresh seeds 10-19 with the same B1 rule (+5 points, interval above zero) and a second kill. If not, take the re-scope.
3. **Fix the gate cost in the same round** (a few prototypes per expert instead of 256, hierarchical or learned gating), because without it no efficiency claim can be made: 83-96% of A's compute is the gate.

This is a new hypothesis raised after seeing the result, so it is a decision for you, not something I should start on my own. The other option is to take the re-scope now and move on to the next part of the system (episodic memory and abstention), keeping the oracle-routed result as a design hint for the modular layer.

---

## Appendix E. The routing investigation and the M2-A test (2026-10-09, late evening)

Companion files: `research/ROUTING_RESEARCH_2026-10.md` (inventory of `E:\universal_substrait`, `E:\unified` and this project, the 2024-2026 literature, the dev-seed measurements), `research/growmoe_results/m2a/` (PREREG.md, run_m2a.py, RESULTS_M2A.md, m2a_runs.json), `tests/growmoe/test_gm_m2a_arming.py`, `tests/growmoe/test_gm_m2a_verdicts.py`.

**E.1 What was found [M, dev seeds, exploratory].** Test-time routing is 92-99 % accurate and training-time assignment is 99 %+ pure after the first two batches; oracle gating with the pool's own expert recipe reaches 52.7 % on the F-15 dev stream against the pool's 39-40 %. The loss comes from the competence monitor: it is armed at birth from the untrained network's loss (mean 2.47), the newborn's held-out loss rises to 4.3 after its first batch of training, the one-sided CUSUM reads that as drift and raises an alarm on the newborn itself at its second batch, and the alarm path replaces its weights with a 60-step fine-tune on 32 samples. This happened once per regime (13 alarms per 15-regime run). `growmoe/competence.py` already said to arm the monitor only after the loss has plateaued.

**E.2 The repair.** `PhyloCfg.arm_monitor_after_infancy` (default False): while an expert is an infant its monitor is neither fed nor tested, and its history is cleared when infancy ends. Seven tests, two of which are controls that pin the fault; breaking the fix fails exactly the two behaviour tests. Alarms and extends fall from about 13 to about 1 per 15-regime run (from 25 to about 3 at 30 regimes).

**E.3 The pre-registered test (fresh seeds 10-19, config locked by hash before the first seed).**

| ID | Rule | Verdict | Expected | Numbers |
|---|---|---|---|---|
| P1 | F-15 and F-30: mean paired A_new - B2* >= +5.0 points and the 95% interval excludes 0 (the B1 rule) | **PASS** | PASS (as expected) | (A_new - B2*) F-15: +10.0 pts [+8.0, +12.0] (n=10); F-30: +11.6 pts [+9.5, +13.8] (n=10) |
| P2 | F-15 and F-30: mean paired A_new - A_old >= +3.0 points (the repair itself moves the pool) | **PASS** | PASS (as expected) | (A_new - A_old) F-15: +7.6 pts [+5.1, +10.1] (n=10); F-30: +7.2 pts [+4.9, +9.4] (n=10) |
| G1 | S1: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) +3.9 pts [+0.6, +7.2] (n=10) |
| G2 | S2-5: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) +1.0 pts [-0.8, +2.7] (n=10) |
| G3 | S2-30: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) +2.3 pts [-1.5, +6.1] (n=10) |
| G4 | S4: A_new - A_old >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | (A_new - A_old) -0.4 pts [-4.3, +3.6] (n=10) |
| G5 | S7: A_new raises at least 3 alarms per run on average (the alarm still catches a label remap) | **PASS** | PASS (as expected) | mean alarms per run 4.10 (n=10) |

**E.4 Reading it.** The repair is worth +7.6 / +7.2 points on its own (P2) and puts the pool +10.0 / +11.6 above the best single network (P1), against +4.1 / +2.6 for the frozen pool in M0. The repaired pool matches the oracle-routed experts at 15 regimes (46.7 % vs 47.6 %, difference within noise) and is 4.2 points short at 30 regimes. Guards: S1 improved, S2-5, S2-30 and S4 are inside noise with wide intervals, and the alarm still fires on label remaps. The step budget the repaired pool wants is 5 per batch on function streams (the frozen pool's 50 was an artefact of the fault).

**E.5 Resources (descriptive).**

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

**E.6 What it means for the bet.** The accuracy half now holds on function-type streams: separate experts that are created, routed and trained by the pool itself beat the best single network we could build, by about twice the pre-registered margin, and come within a few points of perfect routing. The efficiency half does not hold yet: at 30 regimes the pool needs 1.4 times the memory and about the same multiply-adds, and 97 % of its compute is the gate scan (oracle experts: 34 kMACs). The next design target is therefore the gate (cheap descriptors with separate conformal calibration), ambiguity deferral at regime onset and two-signal arbitration (routing v2), not the accuracy of selection.

**E.7 Limits.** One synthetic stream family carries the accuracy claim. Three guards have wide intervals, so small regressions cannot be excluded. The S1 budget (125 steps) was chosen on three noisy dev seeds. The repair was found after the M0 verdict, so it is reported as a separate follow-up; the M0 verdict is unchanged. The flag (`arm_monitor_after_infancy`) has been ON by default since 2026-10-10 on your instruction; Phase 1 and M0 entry points pin it off so recorded results reproduce (`docs/COMPONENT_CARDS.md`, card 13).

## Appendix F. Routing v2, stage 1: the batch vote (2026-10-10 to 2026-10-12)

**F.1 What was run.** A pre-registered test on fresh seeds 20-39 (`research/growmoe_results/rv2a/PREREG.md`; bars, expected outcomes and training budgets written and hash-frozen before the first fresh seed): the repaired pool (`A_new`, per-sample assignment) against the same pool with the batch vote (`A_vote`), on 13 streams, 20 paired seeds each. Numbers below are copied by script from `RESULTS_RV2A.md`.

**F.2 The eleven statements.**

| ID | Rule | Verdict | Expected | Numbers |
|---|---|---|---|---|
| V1 | OV-0.3: mean paired A_vote - A_new >= +6.5 points and the 95% interval excludes 0 | **PASS** | PASS (as expected) | +11.4 pts [+10.0, +12.9] (n=20) |
| V2 | F-15 and F-30: mean paired A_vote - A_new >= +1.0 points and the 95% interval excludes 0 | **PASS** | PASS (as expected) | F-15: +3.0 pts [+1.7, +4.3] (n=20); F-30: +6.4 pts [+5.4, +7.5] (n=20) |
| V3 | OV-0.6: mean paired A_vote - A_new >= +3.0 points | **PASS** | PASS (as expected) | +5.8 pts [+4.5, +7.1] (n=20) |
| G1 | S1: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | +1.4 pts [-0.2, +3.1] (n=20) |
| G2 | S2-5: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | +0.3 pts [-0.4, +1.1] (n=20) |
| G3 | S2-30: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | +2.3 pts [+0.9, +3.7] (n=20) |
| G4 | S4: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | +9.7 pts [+7.0, +12.4] (n=20) |
| G5 | OV-0.0: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | +1.7 pts [+0.9, +2.4] (n=20) |
| G6 | OV-0.3-MIXED: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | +5.5 pts [+4.0, +6.9] (n=20) |
| G7 | S7: A_vote raises at least 3 alarms per run on average (the alarm still catches a label remap) | **PASS** | PASS (as expected) | mean alarms per run 4.10 (n=20) |
| GM | MIX-0.0 and MIX-0.3, accuracy on the minority regimes: A_vote - A_new >= -3.0 points and the 95% interval is not entirely below 0 | **PASS** | PASS (as expected) | MIX-0.0: -1.3 pts [-2.8, +0.2] (n=20); MIX-0.3: +5.7 pts [+3.0, +8.3] (n=20) |

**Decision (computed by `growmoe.fairfight.rv2a_decision`).** V1 PASS and every guard holds: adopt the plain vote (batch_vote=True) as the PhyloMoE default, test-first, with the historical entry points pinned.

**F.3 What it says, in plain words.** When regimes overlap, letting each sample choose its own expert sends some samples to the wrong expert and that expert then learns a mixture of two regimes. Letting the whole batch go to the expert most of it chose removes that, and the gain is largest exactly where the overlap is (+11.4 points at centre cosine 0.3). It also helps the ordinary function streams (F-30 +6.4, F-15 +3.0) and costs nothing measurable anywhere else. The one visible cost is about 1.3 points on a known regime that later arrives mixed with another one at zero overlap.

**F.4 Limits.** Regimes that overlap strongly (centre cosine 0.6 and above) still merge into too few experts (25.3 % against about 48 % for experts that are told which regime they are looking at); that is stage 3 of the spec and stays open. The accuracy claim rests on synthetic function-regime streams plus the older digit and type streams. This stage does not make the gate cheaper (stage 2). The V1 bar was lowered from +8.0 to +6.5 by the pre-registration rule because the dev effect on the product stream was +10.5; the result (+11.4) clears both.

**F.5 Process note.** The implementer ran the fresh seeds without waiting for the reviewer's go-ahead after the pre-registration. Checked afterwards by the reviewer: every file hash recorded in the report still matches the file on disk (code, pre-registration, bars, budgets), `report` regenerates `RESULTS_RV2A.md` and `rv2a_runs.json` byte for byte, the test thresholds and seeds are the planned ones, no pre-existing test was touched, and the suite passes (473 before the default flip, 475 after). The result stands.

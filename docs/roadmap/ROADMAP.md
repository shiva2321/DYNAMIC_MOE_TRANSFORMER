# Road plan: a continually learning, efficient mixture-of-experts model

**Version 0.5 (APPROVED by you on 2026-11-02, refined as we go), 2026-11-02.** The living guidepost for the whole project. v0.2 adds your answers on the backup, who builds and publishing, an honest "is it publishable and worth doing" assessment (section 1.4), and the working rules in section 8. v0.3 restates section 1.4 for the question as you meant it (the prototype as built, posted as a ResearchGate preprint) and records the push (decision 9). v0.4 records your choice to wait for M1 and M2 before the first preprint (D-74). v0.5 records your approval of the plan and of my recommendations on the open decisions (D-75).
Figure: [ROADMAP_FIGURE.png](ROADMAP_FIGURE.png). Companion files: [MILESTONES.md](MILESTONES.md) (each step in detail), [STATE_BOARD.md](STATE_BOARD.md) (what exists and what must exist),
[IDEA_BOARD.md](IDEA_BOARD.md) (the brainstorm, including outside-the-box options), [RESEARCH_DIGEST.md](RESEARCH_DIGEST.md) (what the field has), [CHANGELOG.md](CHANGELOG.md) (every change to this plan).

**How this document is written.** Plain words first, then the technical meaning where it matters. Every number comes from the lab notebook (`docs/labnotebook`: `C-` claims, `E-` entries, `D-` decisions,
`F-` failures, `Q-` open questions) or from the research digest, and carries its source. Anything that is only my proposal says "proposed".

---

## 0. At a glance

**The goal.** One complete, working mixture-of-experts model that learns continually without unwanted forgetting, routes and classifies without being told the task, notices what is new (including subtle
new wording and meaning in natural language, and other kinds of input), keeps its knowledge tidy (dormant, merged, removed, corrected), and does all this at lower cost than a dense model of the same quality.
Built on the one machine we have (RTX 3060 12 GB), to show the field that these problems can be overcome.

**Where we are.** A validated *lab prototype*: a router plus a pool of small experts on a frozen sentence encoder. On controlled streams, forgetting falls from 55 points to 7 and answering needs 0.08-0.20 of the
multiply-adds of a same-memory network (counted, not clocked). On real text it ties strong simple baselines. Not yet shown: any standard benchmark, a trained representation, language-model behaviour,
exact removal, other input types, wall-clock or energy gains, independent replication. Climbing the five evidence levels (L1 synthetic to L5 replicated), we have L1 and part of L2.

**The road.** Nine milestones and three early probes, in six phases:

| Phase | Milestone | In one line | Effort |
|---|---|---|---|
| A. Safe and measured | **M0** Foundations | Back up, make reproducible, build a self-checking cost meter, fix the finish-line contract, check prior art on the exact combination | M |
| | **M1** Efficiency scoreboard | Measure the real cost of today's system against the baselines people actually use | M |
| | **P1, P2, P3** probes | Small-language-model pool feasibility; exact removal; cheap-first novelty cascade | M, S-M, S |
| B. Credible | **M2** Standard text benchmark | Not worse than the strongest simple methods on a benchmark the field knows | M-L |
| C. Cheap and bounded | **M3** Cheap internal novelty and nuance | Replace the 1.5B reader with our own small detector | L |
| | **M4** Router and pool at scale | Thousands of experts, hard streams, bounded cost, removal and correction | L-XL |
| D. Its own representation | **M5** Small language model | Learn its own features, no forgetting, cheaper than dense (the big one) | XL |
| E. Generalise | **M6** Other input types | Same engine on images, audio or time series, and mixed streams | L |
| F. Prove and share | **M7** The complete model | Everything together on a long stream, full efficiency report | L-XL |
| | **M8** Replication and release | Someone else reproduces it; papers; open release; scale-up decision | L |

Roughly 60-100 sessions in all, most of it in M5. The estimate gets redone after P1 and M4.

**If we get stuck** (section 6): a failed milestone is a signal to redesign, not to stop. We record it, check the instrument, find the limiting part with oracle tests, brainstorm conventional and
outside-the-box fixes on the idea board, run short time-boxed experiments, and re-test against the *same* bar. After three cycles we hold a stuck review with you. We never skip silently and never lower a bar quietly.

**Next three work blocks** (after your decisions in section 10): (1) M0 steps 1-3: back up, pin the environment, freeze "Baseline v1"; (2) M0 step 4: the cost meter; (3) probes P2 and P3 (cheap, CPU) while the M1 pre-registration is written.

---

## 1. The goal

### 1.1 In plain words

You described the finish line as a complete working mixture-of-experts model that overcomes the obstacles: working routing, classification and novelty detection that can catch novelty and nuance in natural
language and in other input streams; learning and generalising continually without catastrophic forgetting; and staying efficient compared with the dense models behind mainstream chat systems.
Your refinements, which this plan follows:

- **Routing** sends an input to one expert or to a *pool* of experts, whichever it needs.
- **Not "never forget", but no *unwanted* forgetting.** Knowledge that is unnecessary or proven wrong can be made dormant or removed on purpose.
- **Custom parts are welcome:** our own novelty reader, our own small trained encoder, built for this architecture; our own backbone later, once the mechanism is proven.
- **Purpose:** a convincing demonstration to the field that the problem can be overcome.
- **Resources:** this machine only; no budget for rented compute.
- **A failed milestone does not mean the goal cannot be reached.** We explore, including outside conventional methods, until we pass it.

### 1.2 What it is not, and what would make it a failure even if it works

- It is **not** a claim to match the capability of the largest commercial models, and not a frontier-scale training run. The claim is about *mechanism and trend*: at small scale, at equal quality, with a scale ladder that shows how the gap moves.
- Whether a particular commercial model is dense or already a mixture of experts is not public. **Static** mixture-of-experts models are in use and save compute per token, but their experts and router are fixed after training and do not learn continually. Our claim is continual learning *plus* efficiency (RESEARCH_DIGEST section 3).
- **Anti-goals** (each would make a pass worthless): efficiency that exists only because accuracy is lower; results only on streams we built ourselves; a detector that calls a large separate model for every input; results nobody else can reproduce; hidden failures; bars moved after the results are seen.

### 1.3 The finish-line contract (F1 to F8)

This is what "complete working model that overcomes the obstacles" means, as tests. **The bars are proposals**, chosen as the smallest differences a reviewer would call meaningful; you confirm or change them in M0 (decision 3, section 10),
and they are hash-frozen before any result that tests them.

| ID | In plain words | Technical measure | Proposed bar | Tested at |
|---|---|---|---|---|
| **F1** Keeps what it learned | After learning many domains in sequence, with no task label at test time, earlier ones are still as good as when learned | Forgetting = score right after learning minus final score, per earlier domain (perplexity for language, accuracy for classification); at least 10 domains, text plus at least one other type | At most 2% relative perplexity, or 1 point of accuracy | M2 (frozen), M5, M7 |
| **F2** Still learns | New domains are learned nearly as well as by a dense model fine-tuned on them alone | New-domain score against a dense model fine-tuned on that domain | Within 5% relative | M5, M7 |
| **F3** Matches joint training | After the whole stream it is nearly as good as a dense model trained on everything at once | Average score against the dense joint model of equal *total* parameters (and better than the dense model of equal *active* parameters) | Within 3% relative, or 1.5 points | M2 (frozen), M5, M7 |
| **F4** Notices the new, including nuance | Flags new topics, new skills and new meanings, and does not cry wolf on known things said differently | False-new on known items in new wording; new-skill and new-meaning detection; topical novelty AUROC; detector cost as a share of the answer cost | False-new at most 0.05; detection at least 0.90; AUROC at least 0.95; cost at most 5% of a per-input answer | M3, M7 |
| **F5** Routes well without task labels | Finds the right expert or experts, cheaply, even with thousands | Routing loss against oracle routing; gate cost as experts grow | Within 2 points of oracle; cost at 1,000 experts at most 2x the cost at 30 | M4, M7 |
| **F6** Keeps its knowledge tidy | Can put knowledge to sleep, merge duplicates, remove knowledge exactly, and correct wrong knowledge | Outputs on the remaining knowledge against a model never trained on the removed items; accuracy with 90% of experts dormant | Exact for closed-form parts; at most 0.5% loss elsewhere; routing agrees on at least 99.9% | M4, M7 |
| **F7** Efficient against dense | At the same quality it costs less to answer and to update | Active compute and bytes per token, wall-clock and energy at batch 1 and 32; update cost per new domain against the compute-matched dense continual baseline; total parameters reported as a ratio | At least 2x less active compute and bytes at equal quality, confirmed in wall-clock; update cost at most one third; if any bar is missed, the honest trade-off is reported | M1, M5, M7 |
| **F8** General and honest | The same engine works across input types; every claim is pre-registered, replicated by someone else, and negative results are published | Number of input types with parity; independent replications; verify-clean notebook | At least 3 input types within 1 point of the best simple baseline; at least 3 headline results replicated | M6, M8 |

### 1.4 Is the prototype publishable as a ResearchGate preprint, and is it worth doing? (added in v0.2, restated in v0.3)

Your question, as you clarified it on 2026-11-02: is the prototype we have built so far worth posting as a preprint on ResearchGate? (v0.2 first answered it for the plan's end product; that answer is kept further down.) This is my judgement from the notebook, not a measured fact, and it should be asked again after M1 and M2.

**What we can honestly say today.** The notebook's own bar for "safe to publish" is grade A: pre-registered, fresh test seeds, script-generated verdicts, reproduced. Of 192 claims, 14 are grade A (13 plus one descriptive). All 14 are about one system, the expert pool (GrowMoE), on controlled streams (function streams, operator-by-operator arithmetic, the S1 and S2 stream units and, for one claim, banking77 text). In plain words: the repaired pool beats the best single network by at least 5 points (C-17, C-20); a learned router adds 13.0 points [10.5, 15.4] on arithmetic streams built to need one (C-84); the transfer rule plus the router reach 46.6 % exact match with 7.1 points of forgetting (C-87); a fingerprint shortlist cuts the cost of the novelty gate 6-11x with no accuracy loss (C-25); and we can show where the earlier pool went wrong: it founded too few experts, its growth policy deepened experts, and a lifecycle bug had armed the competence alarm at birth (C-88, C-89, C-19).

**What we cannot say.** That it beats the strong simple baselines on real text: there it ties replay and nearest-neighbour (E-054). Anything about wall-clock time, bytes read or energy: the efficiency numbers are counted operations against a same-memory single network, a weak baseline. That any part is new: there is no prior-art check yet. Anything about language models, other input types or removal of knowledge. Anything from the earlier ISHMN README, which is historical and unverified.

**Verdict.** Publishable as a modest technical report: "a task-free expert pool with a calibrated novelty gate: what held up under pre-registration, what tied, what failed, and the evaluation traps we fell into". Not publishable as a method paper claiming a win; a reviewer would ask for a standard benchmark, strong baselines and measured cost, which is M1 and M2.

**Should we post it now? My advice: not yet.** (1) The notebook's own rule asks for a clean `verify` (today it shows 13 path problems in E-047 and E-050 and no changed evidence), every claim used at grade A or labelled with its real grade, and the generating scripts re-run from a recorded commit; 832 files (FS-MoE code, knowledge-status lab) are still uncommitted because of the commit gate. (2) The prior-art check (M0 step 6) is needed before we say what is new. (3) What a reader wants, a standard benchmark and honest cost against strong baselines, arrives with M1 and M2, about 11 to 16 sessions; a report written now would mostly be rewritten then. If you want an early timestamp anyway, a short technical note on the grade-A claims and the evaluation traps is honest and cheap (about 2 to 3 sessions). You chose to wait for M1 and M2 (2026-11-02, D-74).

**The same question for the plan's end product (v0.2):**

| Kind of publication | Today | What it needs |
|---|---|---|
| **Preprint or technical report** (anyone can post; nobody reviews it first) | Yes, if scoped honestly. The strongest parts are the method rather than the architecture: the validity traps we found and their measured size (section 2), the calibrated novelty gate, the fingerprint shortlist, and the negative results | Every number tied to a claim ID; limitations stated first; nothing claimed above evidence level L2 |
| **Peer-reviewed paper** (a workshop, a correctness-focused journal such as TMLR, a lifelong-learning conference; check each venue's current rules when we get there) | Not yet. We have frozen sentence vectors, streams we built ourselves, and a tie with nearest-neighbour and replay baselines on real text | At least M1 and M2 (honest cost numbers, a standard benchmark against strong simple baselines) and one axis where the pool clearly wins: open-world detection, lifecycle, exact removal, or cost |
| **A "major contribution"** (a complete, continually learning, efficient, language-capable model) | A long shot, not a result. Nothing shown so far reaches it | M5 (trained representation), M7 (everything together), M8 (someone else reproduces it) |

**Should we do it? Yes, in stages, not as one all-or-nothing climb.** The first stages (M0 to M2 and the three probes) are cheap and decisive. Every stage ends in something that can be written up honestly even if the next stage fails, and a failed design is still information the field lacks. What I would not do is promise the finish line. One person, one 12 GB GPU and small models are a thin base against what large labs attempt, and the simple baselines the field trusts (replay, frozen features with closed-form heads) already tie us on real text. Our own earlier trained-language-model study (the README of the public repository; we have not re-run it) reports a significant retention gain on some of its ten domains but p = 0.58 for the aggregate backward-transfer difference, a fair warning about effect sizes at that scale.

**Go / no-go points.** (1) After P1, M1 and M2 and before M5, the expensive step: you decide, with numbers, whether to start it (BP3). (2) Whenever a bar fails three spike cycles: the stuck review (section 6).

**What would make me recommend re-aiming, not stopping.** Compute-matched replay plus a closed-form head matches the pool on every axis we measure (then we lead with the evaluation method, the lifecycle and exact removal). P1 shows that a pool of small language models cannot reach a useful quality-per-cost point on this machine (then we narrow the claim to the non-language streams and say so). P2 shows that exact removal fails (then that selling point goes, and we say so).

---

## 2. Where we are (what is there)

**What exists.** Over 20,000 lines of Python across the pool (`growmoe/`), UniMoE (`unimoe/`), FS-MoE (`fsmoe/`) and the routing lab, 1,927 collected tests (count of 2026-10-30), a 66-entry lab notebook with claims, failures, decisions and
open questions, and a record of pre-registered and independently reviewed results. In words: a router (a calibrated novelty test plus a learned owner predictor), a pool of experts (closed-form heads, small networks),
a lifecycle (spawn, extend, probation, dormant, merge, caps), a language-model "asking reader" and a cheap trained head for novelty and nuance, and measurement tooling.

**What is shown** (details and evidence IDs in [STATE_BOARD.md](STATE_BOARD.md)):
- Forgetting on operator-by-operator arithmetic: 55.1 points for a plain network (E-026, 2 seeds), 15.2 for the shipped pool and **7.1** for the adopted pool (the last two pre-registered, fresh seeds, grade A; C-87).
- Against a single network of the same memory the pool is ahead by 6.2, 9.1 and 13.0 points at 10, 30 and 100 regimes (synthetic streams built so that lookup fails; dev seeds; C-76).
- A closed-form (LDA) head in every expert is the biggest single gain (+0.022, +0.069, +0.126 over the older pool; C-123); on seven real text domains the pool then reaches the offline all-data network in mixed and drifting streams (E-054) and ties replay and nearest-neighbour baselines.
- Novelty: the asking reader's false-new rate is 0.025 against 0.279 for look-based detection on dev seeds (C-91); executing code to check outcomes gives AUROC 1.000 (E-056); look-based detection finds only 13-22% of near-twin skills (E-055).
- Efficiency: answering needs 0.08-0.20 of the multiply-adds of the best same-memory single network (counted, dev seeds, C-42) and the fingerprint shortlist cuts the gate 6-11x on fresh seeds (C-25, grade A); in the older shootout the whole run took 0.22-0.35 of the time (E-025).

**What is not shown.** Any standard benchmark; a trained representation (Stage 2 not started, Q-43) or any language-model behaviour; wall-clock, bytes-read or energy at batch 1 and 32; exact removal and correction of knowledge; unlabelled learning; other input types; long horizons and hard streams (Q-08, Q-11, Q-17); any efficiency gain for
today's default stack against compute-matched baselines; independent replication (one independent review of stage 4 only, E-042). The 1.5B reader is far too expensive to be part of an efficient model.

**What we learned the hard way** (each rule in section 8 exists because we broke it): a "+10 points" that was string grading (C-01); a replay baseline under-trained (C-13); expert and teacher sharing a seed (C-75, margins fell from +14 to +6); a founding cluster counted twice (C-78); a competence alarm armed at birth (C-19); a double-counted parameter total (F-11); double-counted test messages in cost savings (F-103); a dev gain that vanished on report seeds (F-101). The method is what makes the numbers above trustworthy; keeping it is part of the plan.

**Backup status (updated in v0.3).** Everything is committed and on a new branch, `growmoe-lab`, of the public repository you named. The 832 files the commit gate first held (the FS-MoE code and tests, and the knowledge-status lab) went in on 2026-11-02, after you accepted its four findings (D-81). The 10.4 GB of caches and about 9 GB of checkpoints are outside git, on one machine with daily power cuts.

---

## 3. What the field has, and what it changes

Full map in [RESEARCH_DIGEST.md](RESEARCH_DIGEST.md). The facts that shape the plan:

1. **The baseline to beat is "continue training with replay and a good learning-rate schedule"**, not "retrain from scratch": this matched retraining at 405M and 10B parameters (Ibrahim et al.), and at a fixed compute budget plain replay beat fancier continual methods (Prabhu et al.; Cho et al., ICLR 2026). Our efficiency claim must be a *compute* claim against that baseline.
2. **A frozen pretrained model with a closed-form or prototype head is a known strong baseline** (SimpleCIL, RanPAC, FeCAM, analytic learning). Our LDA head is in that family. On frozen features we should expect at best a tie (E-054 already shows a tie with replay and nearest-neighbour); our case rests on what it cannot do: novelty, lifecycle, trained representations.
3. **Expansion with novelty-based routing is crowded** (SEMA, HGE, CoPE, PASs-MoE, and others). From abstract-level reading we have not found a piece of our design that is clearly new on its own; the combination, exact removal and an honest single-GPU cost accounting are what we can offer. A targeted check on 2026-11-02 ([PRIOR_ART.md](PRIOR_ART.md)) found every ingredient already published, including exact removal by ridge downdating (ACU, 2025), and did not find the exact combination; so exact removal is a capability we need, not a novelty we can claim.
4. **The closest design to a pool of small language models is SmallTalk LM** (32 small experts matched a 1.3B dense model's perplexity with about a third of the inference compute). That is probe P1.
5. **Retrieval beats fine-tuning for new facts** (Ovadia et al.), so retrieval is a baseline for any claim about facts, and possibly one expert type.
6. **Measure what really costs:** at small batch sizes decoding is limited by memory bandwidth and every expert must be resident, so we measure bytes read, wall-clock and energy, not just counted operations.
7. **Routing benchmarks are too easy and text leaks the task** (FLEX critique), so our routing tests need long, overlapping, near-twin streams with no leakage.

---

## 4. Our thesis and the bets

### 4.1 The design in plain words and in parts

Think of a hospital. The front desk (the **novelty detector and router**) decides, for each patient (an input), which specialist should see them, or whether a new specialist is needed. Each specialist (an **expert**)
starts as a quick formula that cannot forget (a closed-form head, the "notebook") and gets a small network only if the formula proves not good enough. The shared eyes and ears (the **encoder**) change slowly.
A records office (the **lifecycle manager**) puts unneeded specialists to sleep, merges duplicates, removes specialists whose knowledge is obsolete, and corrects wrong records. A meter (the **cost harness**) measures what everything costs.

Technically: encoder, then a calibrated novelty and routing stage (batch-level conformal tests, a shortlist for speed, a learned owner predictor), then an expert pool (closed-form heads, optional small networks or adapters),
then a combiner (one expert or a team), a learner (spawn, extend, skip, notebook first), a lifecycle manager (dormant, offload, merge when certified redundant, exact removal, supersede), and a cost harness. Most of this exists for frozen sentence vectors (STATE_BOARD).

### 4.2 The bets (each is a falsifiable statement)

| ID | The bet | Falsified if | Tested at | If it is false, we try |
|---|---|---|---|---|
| **B1** | Isolated experts with closed-form heads learn new knowledge without forgetting, given a good representation | Forgetting bars fail even with oracle routing | M2, M5 | A different head; per-expert statistics; replay of class statistics |
| **B2** | Task-free routing can be accurate and sublinear in cost at thousands of experts | Gate cost grows about linearly, or routing loss exceeds 2 points | M4 | Hierarchical or product-key routing (I-01), learned routers (I-02), team routing (I-03) |
| **B3** | Novelty and nuance can be detected cheaply inside the model, near the big reader's quality | Cheap detector misses the bars at 5% of the reader's cost | M3 | Cascade (I-06), reader hidden states (I-08), expert-owned detectors (I-09), predictive coding (I-11) |
| **B4** | A trained representation can change slowly without breaking stored knowledge, and the closed-form head's win survives it | Closed-form heads lose to trained experts on learned features, or old statistics drift beyond repair | M5 | Two-speed learning with statistics adaptation (I-13), null-space adapters (I-14), a larger frozen model |
| **B5** | A pool of small experts beats a dense model on cost at equal quality while learning continually | No Pareto advantage in compute, bytes, energy or update cost | M1, M5, M7 | Focus on update cost; shared-basis experts (I-18); cascades (I-24); honest trade-off report |
| **B6** | Removal and correction can be exact because of isolation plus closed-form updates | P2 finds components that cannot be cleaned | P2, M4 | Certified approximate removal; versioning; rebuild affected parts |
| **B7** | The same engine handles other input types by swapping the front end | Per-type tuning needed beyond calibration | M6 | Per-type novelty-trained encoders; type-aware routing |
| **B8** | Memory and compute stay bounded as knowledge grows | Memory or latency grows linearly with experts beyond the budget | M4 | Offload and revive (I-19), shared bases (I-18), certified merging (I-20) |

---

## 5. The road

### 5.1 Shape

We climb the five evidence levels, not one ability at a time: first measure and secure what we have (M0, M1), then prove credibility on a benchmark the field knows (M2), then make the parts cheap, robust and bounded (M3, M4),
then cross the biggest gap, a trained representation (M5), then generalise (M6), integrate and prove (M7), and have it replicated and released (M8). The three probes are small, cheap experiments that reduce the biggest uncertainties early.
Details, steps, proposed bars, costs and failure options for each milestone are in [MILESTONES.md](MILESTONES.md).

### 5.2 Order and dependencies

```
            ┌──► M2 ──┐          ┌──► M5 ──┐
M0 ──► M1 ──┼──► M3 ──┼──────────┤         ├──► M7 ──► M8
            └──► M4 ──┘          └──► M6 ──┘

Probes:  P1 ──► M5     P2 ──► M4     P3 ──► M3     (any time after the cost meter of M0)
```

M2, M3 and M4 can run side by side (different parts of the code); M5 and M6 start when the three are done (M6 needs M2's baseline pipeline and the parts of M3 and M4 it uses). Probes can run as soon as the cost meter (M0 step 4) exists; P2 does not even need it.

### 5.3 Why this order

- **M0/M1 first** because we cannot judge efficiency, the half of the goal we know least, without a trusted meter, and because the work is not safe yet.
- **M2 early** because a credible result on a known benchmark is the cheapest way to find out whether the strongest simple baselines make parts of our design unnecessary (branch point BP2).
- **M3 and M4 before M5** because M5 is the most expensive and riskiest step, and it needs a cheap detector and a router that scales. **P1 before M5** so the design choice for M5 rests on measurements on this machine.
- **Alternatives considered:** *representation-first* (start with M5): the biggest unknown first, but without the cheap detector and router the result would be hard to interpret, and a failure would be expensive; *efficiency-first only* (M1 and tuning): useful but it would not move us toward a trained representation; *vision-first* (M6 before M5): easy wins on frozen features but would not address language models. The plan keeps the early probes to get P1's information without waiting.

### 5.4 Branch points (where the plan is expected to change)

| ID | When | Decision | Evidence that decides it |
|---|---|---|---|
| BP1 | End of M1 | Which efficiency levers first; adopt the held defaults (LDA head D-43, growth rule D-38, skip and chance flags)? | The scoreboard |
| BP2 | End of M2 | Differentiate on open-world and lifecycle, or simplify to a closed-form core? Also: do the results justify the first ResearchGate preprint (section 8.1, D-74)? | Pool against best simple baseline |
| BP3 | After P1, M1 and M2 | Go / no-go on M5 with you, then the M5 design: pool of small models, shared base plus adapters plus closed-form heads, or MoE blocks in one transformer | P1 measurements, the M1 scoreboard, the M2 comparison |
| BP4 | End of M3 | Retire the big reader from the default path? | M3 bars |
| BP5 | End of M4 | Expert granularity; cap or merge policy | Scaling and lifecycle results |
| BP6 | Before M6 | Which input types | Data availability; difference from text |
| BP7 | Before the end of M8 | Scale-up path: larger backbone through 4-bit adapters on this machine, compute grants or collaboration, our own backbone | Results so far, resources |

### 5.5 What fits on this machine

RTX 3060 12 GB, Ryzen 5 7600 (6 cores, 12 threads), 31.7 GB RAM, 54 GB free on D:, Windows 11, Python 3.12, torch 2.6 (CUDA 12.4), transformers 4.57, peft 0.19, bitsandbytes 0.49, sentence-transformers 5.4, pynvml (GPU energy). Daily power cuts.

| Fits | Does not fit |
|---|---|
| Pool experiments on frozen features (CPU, as now); embedding a few hundred thousand texts with small encoders; training encoders up to about 150M parameters; training language models of about 150M parameters on a few billion tokens over days (an estimate, P1 measures it); 4-bit adapters on models of 1-3B parameters; inference of larger models in 4-bit | Training or fine-tuning models of 7B parameters or more; anything that needs rented GPUs; datasets of hundreds of GB; jobs that cannot be resumed after a power cut |

Consequence: dense-model comparisons happen at small scale with a **scale ladder** (several sizes), so we can show a trend and say honestly what has and has not been shown at larger scale.

---

## 6. When we get stuck: the breakthrough protocol

You asked that a failed milestone never become a dead end. A failure here is information, and the protocol below turns it into the next experiment. It never relaxes the evidence rules.

1. **Say it out loud.** Record the failure (`FAILURES.md`) with the numbers; mark the milestone STUCK-1 in the state board.
2. **Check the instrument first.** Is the bar fair, the measurement right, the baseline tuned equally? Re-run on a fresh seed; audit the code path (our history shows validity bugs are common, section 2).
3. **Find the limiting part with oracles.** Swap in oracle routing, oracle novelty, told-the-regime experts, oracle labels, one at a time, and see which closes the gap. (This is how we found, for example, that routing rather than the experts was the largest loss on operator-by-operator arithmetic, C-55.)
4. **Open an idea card.** On the idea board, list for that limiter at least three conventional fixes, two from adjacent literature and two outside the box; score each for chance, cost and what it would teach us.
5. **Spike.** Each chosen idea gets a time box (at most two sessions), a prediction written first, dev seeds only, and a keep-or-drop rule.
6. **Re-test the milestone with the original bar on fresh seeds.**
7. **After three spike cycles without a pass, hold a stuck review with you.** Options: (a) go deeper on the limiter as its own research question; (b) change the design of the component; (c) narrow the claim to what is shown and say exactly what is not; (d) route around it and continue independent milestones, keeping the obstacle on the board; (e) look for outside help or new literature. We never skip a milestone silently.
8. **Always write what we learned.** Negative results are results and go in the notebook and, later, the papers.

**Guardrails.** (i) Bars do not move to fit results; a bar can change only by a logged decision made before the new results are seen, with the reason. (ii) Outside-the-box ideas are tested by the same pre-registration, fresh-seed, baseline-next-to-the-claim rules as any other. (iii) **No-stall rule:** the backlog always holds at least two queued spikes, so waiting on a result or a decision never leaves us with nothing to do.

---

## 7. How to change this plan, and how to know what must exist when

**What can change freely (tactics):** the order of work inside a milestone, which spike to run first, tooling, the contents of the idea board, effort estimates. Log it in one line in the changelog.

**What needs a logged decision (strategy):** a bar, the finish-line contract, a milestone's scope or order, a branch-point outcome, a non-goal. Each such change is a row in [CHANGELOG.md](CHANGELOG.md) with: what changed, why, on what evidence, which milestones it touches; the version goes up; the state board is updated; and the decision gets a `D-` row in the notebook. A bar may be changed only *before* the results it would judge are seen.

**What must exist at any point:** the state board ([STATE_BOARD.md](STATE_BOARD.md)) lists, for every component, what exists, what is missing and which milestone needs it; its level table shows how far the evidence has climbed. If the board and reality disagree, the board is wrong and gets fixed first.

**Review rhythm:** after every milestone (update the board, revisit the next milestone's estimate and bars); once a month (a short plan review: are we on the road, is the backlog healthy, any decision overdue); about every three months (a literature refresh: new papers on the same problems, with the digest tags upgraded to V for what the next milestone depends on); and whenever something surprising happens.

---

## 8. How we work (rules we keep, from the notebook and decisions)

- **Pre-registration before the first test seed**, with kill rules; verdicts computed by script (D-04). Dev seeds tune, report seeds judge, single-use seeds confirm; seeds 400-429 and 600-604 stay reserved (D-05, D-39).
- **Baselines and controls next to every claim**, tuned with the same effort, at equal memory and equal compute (D-06); every number copied by script from generated files; no hand-edited results.
- **Failures are entries, not deletions** (notebook rule 4); append-only entries; evidence hashes verified (`labnb.py verify`).
- **Confirmed repairs become defaults and old entry points stay pinned**, with the change documented wherever it matters (D-09).
- **Built for power cuts and a small machine** (D-30; one GPU job at a time; no benchmark that fills memory).
- **Single-use confirmatory runs start only on your explicit go** (D-36, D-37).
- **Backup and tracking:** one branch only (no branches unless absolutely necessary and recommended, D-67); commit in logical groups with plain messages; push only to the remote you named and only after the visibility and content checks; never force-push; nothing goes to a public repository without your explicit go (D-15).
- **Roles:** you decide scope, bars, direction and publishing; Claude designs, reviews, analyses and writes; Gemini (Antigravity) does heavy coding and routine experiments to a written spec (D-68, section 8.2). The one who builds never grades the build.
- **Plain-English reports with one summary figure** for every milestone.

### 8.1 Publishing: preprints and reports (your instruction of 2026-11-02, D-69)

Findings go out as preprints on ResearchGate, formal and technical where needed and plain enough that a non-technical reader can follow the point. A wrong or inflated claim would hurt us and the field, so these rules apply to every report:

1. **When.** Only for a result worth a reader's time: a milestone passed against its pre-registered bar on fresh seeds, a negative result that would save others effort, or a validity trap with its measured size. No report "just in case".
2. **Two layers.** A plain-language summary first (what we asked, what we did, what we found, what it does not show), then the technical text: abstract, methods, results, limitations, references, each term defined where it first appears (GLOSSARY).
3. **Nothing made up.** Every number is copied by script from a generated result file and carries its claim ID. No number from memory, no hand rounding, no figure that is not drawn from the files. Estimates are labelled as estimates.
4. **No claim stronger than its evidence level** (L1 to L5, STATE_BOARD). "Beats", "solves" or "state of the art" appear only where a pre-registered bar on fresh seeds, against a tuned baseline, supports them.
5. **Limitations and negative results in the main text**, not an appendix: what was not tested, where we tied or lost, the validity bugs found and fixed.
6. **References are checked.** Only sources that were opened and exist, marked when read at abstract level or from a secondary page. Prior art goes in a table; "to our knowledge first" is written only if the table supports it.
7. **Disclosure.** The methods note names the AI tools used (Claude, Gemini); the human author answers for the content.
8. **A second pair of eyes.** A reviewer who did not write the draft (another model or a person) checks every claim against the claims ledger; you read it once for plain-language clarity.
9. **Reproducible.** Code, configuration, seeds and result files are archived with the report and the exact commit is named; a DOI archive of that snapshot is optional.

Posting is your decision and your action.

### 8.2 Working with Gemini in Antigravity (your instruction of 2026-11-02, D-68)

Heavy coding and routine experiments go to Gemini so that Claude's budget goes to design, review and analysis.

| Claude | Gemini | You |
|---|---|---|
| Design, pre-registration, task specs with stop points; review and audit of what Gemini built; analysis; writing | Implementation to a written spec; long runs and queues; data preparation and table generation; first-pass tests | Decisions, bars, publishing; anything outward-facing |

- **Channel.** Claude writes the task as a file in `docs/superpowers/handoff/` and runs Gemini headless in the background with `agy --print "<task>"`, so Claude is free meanwhile; or you paste one line into the Antigravity window. Checked on 2026-11-02: a plain prompt runs and returns. The first real job also tests file writing and permissions. The WinTerm toolkit (window, keyboard and screen automation) is not needed for this and stays a fallback.
- **Every job** has a spec, a list of files it may touch, a stop point and a results file. `--dangerously-skip-permissions` is used only on your go and with a narrow file list.
- **The builder never grades the build.** Claude (or another model) reviews the diff and reruns the checks; a result is not accepted on Gemini's own report.
- **One writer to git.** Gemini leaves its changes in the working tree; Claude reviews and commits. No two agents edit the same file in one round.
- **Notebook files are shared:** new entries use the next free number by exclusive creation, and existing ones are never overwritten.

### 8.3 Lean and no idle time (your instruction of 2026-11-02, D-70)

- **Lean.** Nobody (Claude, Gemini, you) spends tokens, compute or time on anything the current milestone's bar does not need. Every task carries one line: why now, which bar, expected cost. A side idea with a small but real chance of an important insight goes to the **marginal-insight queue** (section 10) with its cost, and you decide whether it is worth it.
- **No idle time.** While a job runs we work on something that does not depend on it. Long runs go in the background and report when done (no polling, no sleeping). Gemini codes while Claude writes the next pre-registration or reviews. The backlog always holds at least two queued tasks (this extends the no-stall rule in section 6).
- **Cheapest competent worker.** Mechanical edits and bulk runs go to the cheaper model; design and review stay with the strongest.
- **Run only the tests that matter** (your instruction of 2026-11-02, D-79). A change is checked with the tests it can affect and its own new tests, not with the whole suite every time. The whole suite runs only where a record claims something about everything: a baseline tag, a milestone close, before a preprint.

---

## 9. Risks

| ID | Risk | What we do |
|---|---|---|
| R1 | The strongest simple baselines (frozen features plus closed-form heads) tie or beat the pool almost everywhere | Say so; differentiate on open-world detection, lifecycle, trained representations; branch point BP2 |
| R2 | The efficiency advantage vanishes on one GPU (bandwidth-bound decoding, router and gate overhead) | Measure bytes, wall-clock, energy; keep experts resident; cascades and grouped execution; scale ladder; report the honest trade-off |
| R3 | The closed-form head's win does not survive a trained representation (Q-43) | Two-speed learning, statistics adaptation, null-space adapters; fall back to a larger frozen model |
| R4 | Routing errors, not forgetting, dominate the loss (C-55) and false spawns are permanent | Router and calibration work (M4), repair, caps, team routing |
| R5 | Benchmark contamination or leakage (pretrained encoders saw the data; text leaks the task) | Held-out and novel sets; leak checks; adversarial stream suite (I-29) |
| R6 | Data loss: one machine, daily power cuts, 157 unpushed commits | M0 step 1 first; resumable jobs |
| R7 | Our own validity bugs (at least eight found so far) | Pre-registration, mutation tests, validated harness, independent review and replication |
| R8 | Scope creep ("any input type", language generation) | Milestones and branch points; M6 limited to three types; no new scope without a logged change |
| R9 | The field moves, or someone publishes the same combination | Quarterly literature refresh; novelty check in M8; the rigorous evidence and negative results are valuable either way |
| R10 | One researcher plus AI agents: blind spots and fatigue | Independent replication; stuck review; sessions counted so progress and effort are visible |
| R11 | Disk: 54 GB free with 10 GB of caches and large datasets to come | A data budget in M0; archive before download; external drive if needed |

---

## 10. Decisions that are yours, now

Status on 2026-11-02, after your reply "all approved from my side": decisions 1, 3, 4 and 5 are approved as I recommended (D-75); decisions 2, 6 and 9 are answered; 7 (pace) was left open and 8 is for later. The recommendations below are kept as the record.

1. **Approve this plan as the guidepost, or tell me what to change.** Recommended: approve v0.1, then adjust through the changelog as we learn.
2. **Backup and commit (M0 step 1). PARTLY ANSWERED (2026-11-02):** you named github.com/shiva2321/DYNAMIC_MOE_TRANSFORMER as the backup and tracking repository, one branch only. On checking, that repository is public and holds a different, earlier project (27 commits, 2026-08-16 to 2026-10-08, no history in common with ours), so the push went to a new branch, `growmoe-lab`, with `master` untouched (decision 9). The uncommitted work was committed in logical groups on one branch; 832 files were first held by the commit gate and went in on 2026-11-02 after you accepted four findings (D-81). Still open: where the second copy of non-git data goes (see the queue below).
3. **The finish-line numbers F1-F8 (section 1.3).** Recommended: accept as proposed for now; revisit with real numbers at the end of M1. Especially check F7 (at least 2x less active compute and bytes at equal quality) and F1 (at most 2% forgetting).
4. **Downloads.** M2 needs four text datasets (Yelp, Amazon, DBpedia, Yahoo; several GB); P1 needs a few small text corpora; later milestones need image and audio sets. Recommended: approve M2's and P1's now, the rest when we reach them.
5. **Held decisions** (LDA head as default, D-43; growth rule, D-38; skip and chance flags). Recommended: decide them at the end of M1 with the scoreboard in hand (BP1), not before.
6. **Who builds. ANSWERED (2026-11-02):** Gemini in Antigravity does heavy coding and routine experiments to a written spec; Claude designs, reviews, analyses and writes; Claude runs Gemini headless (`agy --print`, checked today) or you paste one line into Antigravity. See section 8.2 and D-68.
7. **Pace and horizon** (optional). The plan counts sessions; telling me roughly how many per week, or any deadline, lets me convert it to dates.
8. **Later: who replicates (M8).** Someone you trust, or a public call after the preprint. No decision needed yet.
9. **Where the backup goes. ANSWERED (2026-11-02):** pushed to a new branch, `growmoe-lab`, of the repository you named (public; `master` and its older project untouched, no force); `origin` now points there and the old private `Try_2` is kept as `try2`; a status note was added to README.md first (D-73). What I had put to you: The repository you named is public and holds an earlier, unrelated project ("Universal SubStrait", a trained transformer with a 10-domain study), so a plain push is not possible without choosing. Options: (a) a new **private** repository for this project, one branch, with the public one left alone; (b) the private repository `Try_2` that is already our `origin` (no setup, same lineage, can be pushed today; **my recommendation as the interim backup**: nothing is exposed, nothing of the older project is touched, and the question of the named repository waits until you want to publish); (c) the repository you named: either its older project is replaced by ours (the old one kept under a tag; this rewrites a public repository, so only on your explicit go) or the two histories are merged into one tree (messy). Going public, in any repository, is a separate decision: the notebook then becomes visible to everyone.

**Marginal-insight queue** (section 8.3; you decide whether each item is worth it)

| Item | Cost | What it buys | My recommendation |
|---|---|---|---|
| Second copy of the non-git data (10.4 GB caches, about 9 GB ISHMN checkpoints) | Your drive or cloud space; no Claude cost | Avoids recomputing if the disk fails | Copy only what cannot be cheaply recreated (checkpoints, not caches) |
| A short early technical note on the validity traps | About 2 sessions | A timestamp and early feedback | **DECIDED 2026-11-02: wait for M1 and M2** (D-74) |

---

## 11. New terms used in this plan (the older terms are in `docs/labnotebook/GLOSSARY.md`)

| Term | Plain meaning | Technical meaning |
|---|---|---|
| Dense model | A network where every part works on every input | All parameters active per token |
| Static mixture of experts | Sparse model whose experts and router are fixed after training | Top-k routing among a fixed expert set |
| Active parameters / active bytes | The part of the model that actually does work for one input | Parameters (or weight bytes) touched in one forward pass |
| Closed-form head ("notebook") | A last layer computed by a formula from running totals; it cannot forget and can be reversed | LDA or least squares on stored counts, sums and scatter |
| Downdating | Subtract an item's contribution from the running totals | Exact inverse of the update for sufficient statistics |
| Equal-compute baseline | The competitor is given the same compute we use | Compute-matched replay or fine-tuning |
| Cascade | Cheap check first, expensive check only if unsure | Confidence band decides escalation; amortised cost = cheap + share x expensive |
| Scale ladder | Several sizes, so a trend is visible | For example 25M, 60M and 150M parameters |
| Probe | A small cheap experiment that removes a big uncertainty early | Time-boxed spike with no pass bar, only measurements |
| Stuck review | A meeting with you after three failed spike cycles | Section 6 step 7 |
| Evidence levels L1-L5 | How far a result has climbed | Synthetic, real with fair baselines, standard benchmark, trained representation and scale, independently replicated |

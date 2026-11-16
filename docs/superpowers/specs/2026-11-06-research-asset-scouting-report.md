# Research-asset scouting report — `E:\AI_Research`, `E:\universal_substrait`, `D:\NSCK_demo`

Date: 2026-11-06 · Status: read-only analysis; nothing in `dmoe/` changed by this report.
Companion docs: `2026-11-05-dmoe-v4-research-synthesis-and-roadmap.md`, `2026-11-05-dmoe-v4-phase-a-...-design.md` (§14, §14.2).

## 0. How this was done, and its limits

Method: read each repo's own verification notes and result ledgers, then **checked the headline claims against code and raw result files** rather than README prose (the user warned the docs lag). Where I could, I traced a claim to a number in a JSON/CSV or to a line of code.

**Not covered** (so nothing below should be read as a verdict on them): the Rust code (TDA, spectral RG, percolation), `VSA_GROUND_REBUILD` code and results, `PAF_WaveNet`, `WaveLearn`/`ResonantSpark` beyond the retrospective, `Video_research_transcripts`, `swarm_infra`, `nsck_rebuild`, the transplant pipeline's ρ = 0.99 figure, and I ran none of these repos' test suites. Sizes: AI_Research ≈ 275k lines of Python (1,567 files), universal_substrait ≈ 28k, NSCK_demo ≈ 150k Python + 50 Rust files — I read a small, targeted fraction.

**The most valuable thing in these repos is not a mechanism; it is the audit culture.** AI_Research records kills, a debunked "64× SNR" claim and an unreproducible "2.3k cities" claim ("theory presented as a measurement", caught twice), and rewrote a circular experiment (EXP-7). universal_substrait reports its own pre-registered primary hypothesis as FAILED. That is the standard our project should hold itself to.

## 1. Claims in the pasted analysis that do not survive checking

| Pasted claim | What the repo's own records say |
|---|---|
| gSCAN split D: factorized model generalizes where monoliths get 0% | `ground/findings.md`: **both score 0.00%** (identical to the published floor). Verdict "INCONCLUSIVE, leaning toward no-effect-observed". Factorized model was **27–38 points worse** on splits B, C, E (confounded: 19% fewer steps). Single seed, ~20k steps vs the paper's 200k. |
| "Factorized cores beat Transformers on OOD reasoning, 100% vs 0%" | Concerto EXP-1 (synthetic graph walk): C3a 0.692 overall vs 0.442, novel-edge 0.818 (oracle ceiling 0.778 overall / 1.0 novel), ~8× sample efficiency. A solid *synthetic* result; not 100 vs 0. |
| PhasorUnitaryNet is "30× more parameter-efficient than dense" | The 0.970 vs 0.001 comparison is against a real MLP with ~30× more parameters on a compositional-binding task. It shows phasor binding generalizes on that task, not that phasor layers are cheaper in general. The same codebase's phasor *transformer* was **25.4× worse** than a real one (with unfair parameter matching and two cheap bugs; fixes untried). |
| Layer-3 QKV rank collapse (−44%) is the sequential-training pathology | The shuffled-mixture control in the same repo shows Layer 3 at −38.8…−48.8% **with no sequential exposure**; Layer 2's verdict flips by seed. A large share of the collapse is ordinary training dynamics. |
| Null-space projection of attention "directly cures" the BWT null | **Never tested** in universal_substrait. `backbone_5seed_dynamic_qkv_results.json` only measures rank change. It is a hypothesis. |
| Neurogenesis gives "zero interference" / a structural shield | Its pre-registered H1 **failed** (pooled p = 0.39; replication cohort p = 0.99). V2 *redistributes* forgetting (two domains better, two worse), is **28.6% slower**, ran with a **shared 20% replay buffer in both arms**, and has **no LoRA / EWC / full-replay baseline**. Real effect: per-domain accuracy retention (github_code +4.36 pp), "protects the peak of the distribution, not its mass". |
| 18.5× MoE dispatch speedup, "install it" | 18.5× is vs a naive loop at 32 experts (1.07× at 2 experts, 3.3× at 8). It is **already adapted in `dmoe/experts.py`** (`_dispatch_sorted_topk`); our default path is dense. Matters in Phase B only. |
| "Hook NullSpaceProjector into attention" as a next step | Already in `dmoe/null_space.py` (`track_attention=True`). |
| NSCK FactGate / BeliefScorer are production-ready | `belief_revision.py` is 42 lines: the rule `contradictions ≥ evidence` with a log-ratio score — **the same prior-count rule our Upkeeper implements**. `fact_gate.py` uses hand-set weights (0.40/0.20/0.25/0.15) and thresholds (0.78/0.50); I found no measured precision/recall. NSCK's own `findings.md` lists mock evaluations and stubs. |
| Mental-rehearsal veto at the output head | Needs `world_model.imagine()`; the only implementation sits in a 3,153-line `cognitive_engine.py` that the pasted audit itself calls partial/buggy. We have no world model. |
| `unified/` gives a ready expert lifecycle | Its "experts" are **numpy prototype vectors**, not networks. Only the state-machine bookkeeping (BORN→…→PRUNED) is portable. |
| Cited papers | [Myth of Expert Specialization](https://arxiv.org/pdf/2604.09780) and [Ouro](https://arxiv.org/abs/2510.25741) exist. The former argues routing similarity follows hidden-state geometry; the "aux-loss degrades capability" gloss is the paste's, not the abstract's. Ouro's own finding: recurrence improves knowledge *manipulation*, not storage — authors' numbers, no independent replication found. |

## 2. What is genuinely useful to us (ranked by value × verification)

**T1 — use soon; verified and matched to a measured gap**
1. **Surprise-priced escalation to an exact store (Concerto EXP-3b).** Needle recall: small neural core alone 0.059 (chance), always-exact 1.000, surprise-gated **1.000 at 17.5% escalation, +0.758 over random escalation at the same budget** — independently cross-derived. This is the neural+exact-memory hybrid that our Track K result (dictionary 0.95 vs neural 0.55) points to. *Warning recorded there:* on a task where the fast path was already sufficient the same mechanism looked "vacuous" (EXP-3, structgen) — test it where the exact store genuinely dominates.
2. **Familiarity calibrated on *prefix* surprise, not final confidence (Concerto EXP-4 fix).** Least-surprised routing first **KILLED at 59.7%** because a specialist is "vacuously confident" out-of-distribution. Split-conformal on final entropy: 66.6%; on **prefix surprise** (how well the specialist predicts the incoming tokens): **96.0%** — on a faithful synthetic reproduction (59.4% vs 59.7% real), not yet wired into the real pipeline. Same family as our `-1e9` unseen-class logits and the LDA arm's NLL 1.86.
3. **Measurement discipline from universal_substrait:** per-domain accuracy retention as primary (3–12σ) vs aggregate BWT (<1σ, underpowered by design); report NLL beside accuracy; **s-rank probes with a shuffled-mixture control** to attribute backbone drift; paired seeds; larger N.
4. **`DynamicWarmupAdamW`** (`hyperspace/dynamic_optimizer.py`, 115 lines, tested): per-group warmup and second-moment conditioning at the running median of warm params. Targets a real issue we have — `domain_transition` rebuilds AdamW and loses momentum state. Cheap to A/B.
5. **Epistemic verdict contract** (`sophia`/Concerto): output `(prediction, verdict, confidence, provenance)` with `KNOWN` reserved for ledger-backed answers. Recorded bug to avoid: labelling `KNOWN` whenever softmax > 0.8 re-created "confidence is vibes".
6. **Expert lifecycle bookkeeping** (`unified/continuous_learning.py` state machine; add HIBERNATE via CPU offload) around our experts.

**T2 — hold; promising but unproven for our needs**
- Two-compartment dendritic experts (V2 bundles four fixes; **no ablation isolates the dendritic part**).
- Sleep-merge of experts by weight barycentre: the test only asserts counts (merged = 1, pruned = 1), **no function-preservation check**. Averaging separately trained FFNs does not generally preserve either function. Needs a retention test before adoption.
- Spectral-gap expert splitting (Rust); at toy scale k-means + silhouette is the baseline to beat.
- Structure/content factorization (TEM): strong synthetic evidence, **no external-benchmark evidence** (gSCAN null). Revisit on compositional task T3, in real-valued FiLM form.
- Unitary FFT blocks, foveal/landmark attention, FPE positions: efficiency-phase candidates; FFT blocks are inherently complex, which conflicts with the real-valued decision.
- `VSARouter` / zero-parameter prototype routing: conceptually the same bet as content-addressed routing (our H1/MoRAM line). **Not verified here** — compare against `SequenceContentRouter` on header-free T1.

**T3 — skip / documented landmines**
Kuramoto/oscillator routing (abelian limitation proven, empirically at chance on S5 beyond parity); Metronome and DRF oscillator memory (killed, twice); rate-channel patching (killed four ways); TDA/Betti monitors (no evidence they improve expert management; heavy); VSA transplant pipeline (we are not a VSA substrate; ρ = 0.99 measures geometry preservation, not task performance); mental-rehearsal veto without a world model.

## 3. What the evidence says about the user's target ("grow, shrink, spawn, retire experts without affecting anything else")

Three independent results converge:
- Universal SubStrait: FFN-level isolation alone did **not** reduce aggregate forgetting; drift lives largely in the shared trunk.
- Our v2: dMoE growth = dense (paired Δ −0.0005).
- Concerto EXP-9: EWC/replay "KILL — root cause was never forgetting" (it was routing/OOD confidence).

So "without affecting anything else" holds only if **(i) inputs to old experts do not change** (shared-backbone drift controlled) **and (ii) routing of old inputs does not change** (content-addressed, no shared router weights). Both must be engineered and measured; expert isolation by itself is necessary, not sufficient.

## 4. Design implication: a three-lane hybrid, not a bag of subsystems

- **Lane E — exact/episodic store:** counted key→value beliefs; belief revision; exact unlearning; O(1). (Our Upkeeper, grown up.)
- **Lane N — neural generalizer:** gradient-trained, modular experts; handles unseen keys, noise, compositions.
- **Gate G — familiarity / surprise-priced escalation:** conformal-calibrated on prefix surprise; send to E when a confident match exists, to N otherwise, abstain (`UNKNOWN`) when neither is familiar; spend extra "thinking" only on escalation.

This is the complementary-learning-systems design that Concerto validated in miniature and that [Larimar](https://arxiv.org/pdf/2403.11901) and [Memorizing Transformers](https://arxiv.org/pdf/2203.08913) pursue at LLM scale (kNN memory improves LM perplexity up to 262K tokens; one-shot edits without retraining). It also supplies the *variable compute* the user wants: cheap exact lookups by default, neural/recurrent effort only when surprised.

## 5. Rules for importing anything from these repos
1. **One new mechanism per experiment**, with a matched control and a pre-registered falsifier. The repos' own failure mode is bundling (V2's "four fixes").
2. An import must beat **"dense + the same gate"** and **"dictionary with counters"**, or it does not go in.
3. Check every imported number against raw files; treat docstring arithmetic as unmeasured.
4. Keep real-valued; anything phasor/FFT needs a real-valued twin first.

## 6. Proposed experiments (all small, pre-registered, falsifiable)
| # | Experiment | Falsifier |
|---|---|---|
| X1 | Track K + **held-out-context split**; arms: dictionary, neural, dense+gate, **hybrid (N + E + surprise gate)** | Hybrid ≤ max(single lanes) on both seen and unseen splits → no value from hybridising |
| X2 | Familiarity calibration: final-entropy vs **prefix-surprise** conformal, header-free T1 at overlap ρ ∈ {0, 0.5, 0.8} | Prefix-surprise not ≥ +10 pts routing accuracy over final-entropy |
| X3 | `DynamicWarmupAdamW` vs rebuilt AdamW at domain transitions | No reduction in post-transition loss spike or retention |
| X4 | Shuffled-mixture vs sequential control in **our** model + null-space on/off | Null-space on/off indistinguishable → the projector is not what protects retention |
| X5 | Expert-merge function-preservation test | Merged expert loses > δ accuracy on either parent's domain |

---

## 7. `D:\spherical_hyperspace_memory` (SHM) — the sibling lab that v3's SetGate, gap monitor and per-expert LDA heads came from

**What it is.** ~170k lines of Python (856 files) plus 7 Rust files, last commit 2026-11-04. Its real asset is the **lab notebook**: 77 entries, a graded claims register (C-1…C-207, grades A–H, REFUTED kept in place), a failures register (F-1…F-115), pre-registration with dev/test seed separation, and a reviewer workflow (designer/reviewer re-measures what the implementer reports). Its README **retracts its own earlier ISHMN headline** (GSM8K +10 points did not survive paired numeric grading; six structural defects).

**What I read:** README, research map, notebook rules, claims register rows C-12…C-175, the last ~45 failure rows, entry E-076, the road plan's summary, and the M1 scoreboard entries E-072/E-073. **Not read / not run:** the Rust crates, ISHMN/Yggdrasil/geometry-atlas code, `research/knowledge_status` code, any test suite. Claims below are the notebook's own graded claims, which I did not re-derive.

**Where it stands, in its own words:** a lab prototype — a router plus a pool of small experts **on a frozen pretrained sentence encoder (MiniLM)**. Controlled streams: forgetting 55 → 7 points; answer-time compute 0.08–0.20× of a same-memory network (counted, not clocked). On real text it **ties** simple baselines (C-112: pool 0.622/0.630 vs replay 0.636, kNN 0.642, offline joint 0.669). *Not shown:* a standard benchmark, its own trained representation, language-model behaviour, exact removal at pool level, wall-clock or energy gains, independent replication.

### 7.1 Findings that bear directly on us (notebook grade B unless noted)
| # | Finding | Why it matters here |
|---|---|---|
| 1 | Lookup vs computed: kNN beats the pool where the answer is a lookup; where it must be *computed*, kNN 21–26% vs trained experts 45–78% (C-14, C-16). A closed-form LDA head inside every expert is **the largest single improvement** of the pool: +0.022 sequential, +0.069 mixed, +0.126 drifting (C-123, 5 seeds, paired CIs) | Same split as our Track K (dictionary vs neural); independent support for per-expert closed-form heads |
| 2 | "Pool beats replay by ~19 points" was **REFUTED**: it held only at 1 optimiser step; at 5–20 steps replay ties or wins (C-13) | Our Track K neural arms stalled at 0.55 after 120 steps — the same hazard: under-trained baselines manufacture wins or losses |
| 3 | Growth does little: it fired in 3 of 70 runs (C-30); capacity is not the limit (16/128/512-unit experts 0.581/0.622/0.631, C-115); **on stationary or gradual streams the pool founds one expert and scores 0.036–0.088 below replay/streaming LDA** (C-119); vertical growth cannot fire on the real suite (C-122) | Matches our v2 null (growth = dense). Warns that **L2 (task-free) is where growth-driven designs fail**: they only react to abrupt change |
| 4 | The largest single loss is **routing at answer time** (11–14 points, C-55). A **learned closed-form LDA router on ~16 stored prototypes per expert** recovers all of it (C-57, C-64); on fresh seeds +13.0 [+10.5, +15.4] on operator-by-operator arithmetic (C-84) | A cheap, closed-form, task-ID-free router to compare against our `SequenceContentRouter` on header-free T1 |
| 5 | The batch-level set gate is the pool's largest component (removing it costs 0.079, C-113) and is calibrated (C-32), **but label-sorted / single-regime batches break it** (C-38: 81.7 with the set gate vs 18.7 plain vote; a circuit breaker contains most of it) | Our v3 inherited that assumption; v4's per-sequence router moved away from it, which this supports |
| 6 | OOD detection is weaker than hoped: the pool's own unknown flag is **silent on real OOD text** (0.000 flagged, AUROC 0.66–0.74, C-120); with 60 verified messages it refuses 0.21–0.46 of foreign text at 5.6% false refusal (C-125); fast–slow disagreement is no better than distance (C-128) | Calibrate expectations for our familiarity gate. Concerto's "96%" (§2) was on a synthetic reproduction |
| 7 | **Merging is hard:** the merge routine removed essentially no wrong experts (8 merges vs 296 injected false spawns, C-68); a union-trained clone loses 6–9 points (C-67); a false spawn costs ~4 points and a slot (C-70) | Independent support for X5: do not adopt barycentric expert merging untested |
| 8 | **Function-preserving widen/deepen is exact** (max output change 0.0, C-58). The growth *policy* is the weak part: it decides on ~51 held-out examples, commits by chance, and its probes cost 62–114% extra training (C-59…C-61) | The one growth operation that is solid enough to build on for the user's vertical/horizontal growth goal |
| 9 | **Exact removal (E-076, reviewer-checked, dev seeds):** closed-form heads are bit-identical to a never-trained model after removal; the shared scatter matrix needs the removed rows back. **Pool-level removal leaks** through derived structures (signature prototypes after the extend step; children are scaled copies of the parent's weights, so one seed cascaded to dropping all 19 experts). The first implementation's weight comparison compared **zero** parameters, so its "0.00e+00" was void. Still in fix round V-4 | Pool-level exact unlearning is **not established** anywhere in these repos |
| 10 | Efficiency is "counted, not clocked": the pool is *not* cheaper before its shortlist (gate = 97% of multiply-adds, 1.41× memory, C-21); shortlist cuts gate cost 6–11× (C-25, A). GPU speedup depends on width (slower at hidden 256, E-073) | Same caution as our complex-vs-real and dispatch notes |

### 7.2 Two consequences for our own code (from reading it, not from SHM)
- **`ContinualLearner.unlearn()` is not exact once the backbone is plastic.** It recomputes features with the *current* model and subtracts `xᵀx` — a different matrix than was added at absorb time if the encoder moved since. Exact unlearning needs stored features or a frozen feature extractor for that head. (Prediction, to be tested as X8.)
- **Clone-from-parent spawning copies exposure.** `clone_with_mutation` gives every child a perturbed copy of its parent's weights, which is exactly the leak E-076 found in the pool; removal of data the parent saw cannot be exact for children.

### 7.3 Take / hold / skip
- **Take:** (a) the closed-form LDA router on ~16 prototypes per expert as a baseline for header-free routing; (b) their equal-memory baseline set — replay, kNN, streaming LDA, offline joint — added to Track K, with **equalised training budgets**; (c) the verified function-preserving widen/deepen operations for growth; (d) the "weak parent cannot vouch for a new cluster" rule (worse-than-chance transfer, C-167/C-172) if we keep parent-cloned spawning; (e) the circuit-breaker for single-regime batches; (f) the process: predictions written before numbers with hit-rate scorecards (e.g. "4 held of 8"), dev/test seed separation, mutation checks on tests (first drafts let 4/14, 5/16 and 6/16 mutants survive: F-76, F-85, F-106), and an independent reviewer who re-measures from raw files.
- **Hold:** notebook-first newborn experts (all gain came from a birth test; no measurable gain in the real pool, C-158, C-165); the "asking reader" (needs an external language model; not a general novelty detector, C-98).
- **Skip:** ISHMN / Riemannian analogy engine (retracted), hyperbolic attention heads (C-07 NOT SUPPORTED), branch-and-bound retrieval (C-08 REFUTED), the merge routine.

### 7.4 The most important cross-repo observation
Everything positive in SHM stands on a **frozen pretrained encoder**; its "own trained representation" milestone (M5) is called "the big one" and is unstarted. Universal SubStrait found that shared-trunk drift is where isolation fails. Concerto and AI_Research are small synthetic or hand-built cores. **None of the four repos has demonstrated continual learning on a plastic, self-trained representation.** Our v4 null-space work is the only attempt, at toy scale — which makes it the genuinely open part of the project, and also the part where nobody's numbers transfer to us.

### 7.5 Added experiments
| # | Experiment | Falsifier |
|---|---|---|
| X6 | Closed-form LDA router (~16 prototypes/expert) vs `SequenceContentRouter` on header-free T1 at overlap ρ ∈ {0, 0.5, 0.8} | LDA router not within 2 points of, or better than, the current router at equal memory |
| X7 | Add kNN, streaming LDA, equal-memory replay and offline joint to Track K; equalise training steps for every neural arm | Conclusions flip when neural arms train to convergence → earlier comparisons were budget artifacts |
| X8 | Unlearn exactness under a plastic backbone: absorb → train backbone → downdate → compare with a never-absorbed model | Bit-identical (then fine). Expected inexact; fix = store features or freeze the feature extractor for unlearnable heads |

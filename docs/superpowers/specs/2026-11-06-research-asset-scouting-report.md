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

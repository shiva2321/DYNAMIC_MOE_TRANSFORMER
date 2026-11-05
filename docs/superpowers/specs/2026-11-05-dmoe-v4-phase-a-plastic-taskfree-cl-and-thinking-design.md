# dMoE v4 · Phase A — Always-Plastic, Task-ID-Free Continual Learning and Trained Latent Thinking

Repository: `D:\Dynamic_MOE_Transformer` · Date: 2026-11-05 · Status: **DRAFT for approval — nothing built.**
Companion: `2026-11-05-dmoe-v4-research-synthesis-and-roadmap.md` (evidence, sources, hypotheses H1–H8).
Efficiency work is **Phase B**, deliberately after this phase (user decision).

---

## 1. Intent and why

The goal is an architecture that learns continually from any byte-level input without catastrophic forgetting, **without paying for replay at scale**, and that *thinks* — spends variable latent computation that measurably helps — rather than only predicting the next token.

The v3 result (`results/dmoe_v3_zero_replay/results.json`, 20 seeds, all five pre-registered falsifiers pass) shows that zero forgetting is easy when four things hold: the backbone is frozen after the first domain, the harness announces domain boundaries, routing reads a header byte for a whole batch, and the task is a per-position lookup. The project wants the opposite regime: **nothing permanently frozen**, **no task-ID**, **tasks that need context and computation**, and **calibrated probabilities**. This phase measures how far the current ideas get in that regime and builds what is missing.

## 2. Where we start (verified)

- v2 without replay = dense without replay (0.349 vs 0.349; paired diff −0.0005 ± 0.0023, n=20). Domain-ID routing accuracy 0.36 ± 0.03 (chance 0.33).
- v3: accuracy 0.9533, retention 1.000, bwt −0.005 nats — **with backbone frozen, boundaries supplied, header-byte batch routing** (details: synthesis doc §1.3).
- v3 NLL 2.23–2.43 nats vs 0.41 for dense-with-replay (Bayes ≈ 0.32): accuracy-only reporting hides a calibration failure.
- The simulation loop is not trained to help (random step count in training, no deep supervision, no loss tying steps to error reduction).
- Known failing test: `tests/test_jepa.py::test_downstream_probe_jepa_outperforms_task_only_baseline` (legacy non-tokenizer-free path; documented in git history). Not in scope.

## 3. Goals, non-goals, anti-goals

**Goals**
- G1. Zero-replay continual learning in which **every parameter keeps a gradient path** and domains are inferred from content, reported at three protocol levels (§5) so capability is known in each regime.
- G2. A latent thinking loop that is **trained to help**: accuracy rises with steps, generalises to deeper instances than seen in training, and halts adaptively.
- G3. A pre-registered, equal-FLOPs **complex-vs-real decision** made from data.
- G4. The existing evaluation discipline: pre-registration before runs, 20 seeds, null controls, paired statistics, falsifiers, honest reporting of failures.

**Non-goals (this phase)**: sparse dispatch / dynamic chunking / hashed n-gram memory / delta-rule mixers (Phase B); world-model / action-conditioned prediction (Phase C); LLM-scale training; multimodal inputs; committing code or results to git unless asked.

**Anti-goals — outcomes that would make this a failure even if numbers look good**
- A pass obtained by permanently freezing parameters, or by a header/task-ID oracle, while labelled "plastic" or "task-free".
- Argmax accuracy reported without NLL / bits-per-byte.
- A benchmark solvable by per-position lookup.
- Hidden replay: any stored **or generated** old-domain input in the zero-replay track.
- A method that retains old domains but cannot learn the K-th new one (loss of plasticity).
- Thresholds tuned after seeing arm results.

## 4. Hard constraints

1. **Always plastic.** No parameter group gets a permanent `requires_grad_(False)`. Allowed: direction constraints (null-space projection), rate constraints (slower learning rate / EMA), schedule constraints. Plasticity capacity that shrinks as domains accumulate must be **measured and reported** (§8), not hidden.
2. **Fixed-size memory.** Anything stored about the past is a statistic whose size does not grow with the number of samples seen (per-layer covariances, per-domain Gaussians, class means, usage counts), plus at most **one lagged copy of the parameters** (the EMA snapshot used by A2; the JEPA teacher already is one). Reported in bytes vs samples.
3. **Bounded compute.** Per-step compute must not grow with history length. A consolidation phase may use a fixed compute budget.
4. **Two tracks, never mixed in one claim.** *Zero-replay track:* no stored or generated old-domain inputs of any kind. *Bounded-dreaming track (optional, §7.1 A6):* fixed-budget latent pseudo-rehearsal from stored statistics; reported separately and never called "zero-replay".
5. **Calibrated readout.** Every table reports accuracy **and** NLL (bits/byte for byte models).
6. **Pre-registration.** Thresholds, arms, seeds and decision rules written to `results/<exp>/PREREG.md` and frozen before any arm is run (same convention as `dmoe_v3_zero_replay`). Thresholds in §9 are *provisional* until the M0 diagnostics measure ceilings.
7. Existing tests stay green (except the known JEPA legacy failure). New behaviour comes with tests (TDD).

## 5. Protocol levels — report every method at all three

| Level | Boundary signal at train | Batches at test | Domain identity at test | What it tells us |
|---|---|---|---|---|
| **L0 task-aware** | supplied by the harness | single-domain | inferable from content (no explicit label) | Upper reference; v3's regime |
| **L1 task-ID-free inference** | supplied at train | **mixed** (domains interleaved within a batch, sequence-level) | none | Routing/representation stability without a task-ID oracle |
| **L2 fully task-free** | **none** — gradual, overlapping drift | mixed | none | Growth and consolidation driven by novelty alone |

An **oracle-routing** control (true domain given to the router at test) is run at every level to separate routing error from representation error.

## 6. Benchmark ladder (header-free, context-requiring)

Common: byte sequences, L = 32 (8 chunks of 4), 16-symbol alphabet (ASCII 32–47), 5% symmetric label noise, K domains (K ∈ {3, 6, 12, 24} for scaling sweeps).

**Domain identity without a header.** Each domain k has (a) its own *input source*: a fixed sparse first-order Markov chain over the alphabet, with an overlap parameter ρ ∈ {0 (disjoint support), 0.5, 0.8} controlling how much domains' input statistics coincide; and (b) its own *rule* y = F_k(x). Domain must be inferred from sequence statistics, and at high ρ it takes several bytes of evidence — which forces sequence/token-level routing and makes abstention meaningful.

**Rule families (each must defeat a per-position lookup — verified by a "lookup-only" control that must fail):**
- **T0** — existing per-position lookup. Regression test only.
- **T1** — `y_t = π_k(x_t, x_{t−1})`: domain-specific bigram tables (needs one step of context).
- **T2** — `y_t = σ_k(x_{t−o_k})`, offset `o_k ∈ {1..4}`, domain-specific permutation σ_k (needs attention / copy).
- **T3** — composition: domains are pairs (i, j) from a pool of primitives, `y = g_j(f_i(x))`; train on a subset of pairs, test on held-out pairs; in the continual variant new primitives arrive over time (tests reuse and, later, binding).
- **T4** — algorithmic reasoning for the thinking track (§7.2): small mazes / Sudoku-lite with instance depth as a controllable variable.

**L2 stream generator.** Domain mixture weights `w_k(t)` ramp smoothly with overlap windows; no boundary flag is ever exposed to the learner (only to the evaluator).

**Real bytes (A.3, gated).** After T1–T3 pass at L0–L1: a short sequence of small public byte corpora (e.g., English prose → source code → non-Latin text → DNA/binary), metric bits-per-byte per domain over time. Corpus choice and any download need the user's go-ahead at that time.

## 7. Arms

### 7.1 Continual-learning arms (all zero-replay unless stated; all report L0/L1/L2)

| Arm | Description | Tests |
|---|---|---|
| **A0** | v3 as-is (backbone frozen after domain 0) | Reference ceiling; *violates* constraint 1 by design |
| **A1** | v3 with the backbone **left plastic** (naive) | How badly drift breaks exemplar signatures and LDA stats |
| **A2** | A1 + **closed-form drift compensation**: keep a lagged snapshot (EMA) encoder; fit a ridge map snapshot-features → live-features on *current* data only; transport stored class means/scatter and exemplar signatures through it | Whether stored statistics survive a moving encoder (LDC-style; synthesis H2) |
| **A3** | A2 + **null-space-constrained updates on every layer** (per-layer input covariance; project the *update after the optimiser*, not the gradient) | Replay-free protection of old function while all layers stay trainable (H1); capacity curve |
| **A4** | A3 + **content-addressed routing**: expert keys; sequence/token-level scoring; closed-form Hotelling-T² / Mahalanobis novelty replacing the CPU permutation test; new atoms' keys initialised in the null-space | Removing the header/task-ID dependence (L1, L2) (H1, H3) |
| **A5** | A4 + **calibrated readout**: temperature / shrinkage on LDA logits, abstention when no expert owns the input | NLL gap (v3: 2.2 vs 0.41) |
| **A6** *(optional, bounded-dreaming track)* | A5 + fixed-budget latent pseudo-rehearsal from stored per-domain Gaussians | Marginal value of generated rehearsal; reported separately |

Controls at every level: dense no-replay; dense with replay-128 (ceiling); joint multitask (ceiling); untrained / shuffled-targets / blind-zero nulls; majority class; oracle routing; lookup-only.

Ordering is additive, so each step is a paired comparison against the previous arm on the same seeds.

### 7.2 Thinking arms (Phase A2, after the CL harness exists; task T4)

| Arm | Description |
|---|---|
| **Th0** | Current `_run_simulation_loop` as-is |
| **Th1** | TRM-style recursion: state (y, z), inner/outer loops, **deep supervision with detached carry**, learned halting |
| **Th2** | Attractor-style: backbone proposal ŷ₀ re-injected at each step, fixed-point iteration, **one-step implicit gradient** (constant training memory) |
| **Th3** | Th1 or Th2 + learned **energy head** used for halting and as the novelty/uncertainty signal fed back to CL (H4) |

### 7.3 Complex vs real (use first, then ablate)

Phase A builds on the existing complex-valued modules. After primary results exist (milestone M5), build **real twins** of every module on the same graph and compare on all CL and thinking tasks under two matches: *parameter-matched* and *FLOP-matched*. Decision rule (provisional, frozen in PREREG): **keep complex only if** it beats the FLOP-matched real twin on ≥ 2 of the primary metrics with the 95% CI of the paired seed-difference excluding 0 **and** effect ≥ δ (δ = 0.02 accuracy or 0.05 nats NLL, provisional); otherwise move to real. If complex is kept, it must be justified by a task where phase carries structure (binding, T3).

## 8. Metrics and statistics

- Per domain: accuracy, NLL / bits-per-byte; **retention** = final / just-after-learning accuracy; **bwt** in nats; forward transfer.
- **Plasticity curve:** steps to reach 90% of the joint-training ceiling on domain k, for k = 1..K. Anti-goal check: domain K within 2× of domain 1.
- Routing: accuracy vs ground truth (chance-corrected), abstention rate, coverage, expert/atom count over time.
- **Capacity:** remaining null-space dimension per layer vs #domains (A3+); stored statistics in bytes vs samples seen; per-step wall-time vs history.
- Statistics: 20 seeds (0..19), paired differences with 95% CI, same merge-and-audit script style as `scripts/run_dmoe_v3_zero_replay_eval.py` (part files → merged `results.json` → falsifier audit).

## 9. Success criteria (PROVISIONAL — frozen only after M0 measures ceilings)

For each level L ∈ {L0, L1, L2} and rule T ∈ {T1, T2, T3} at ρ ∈ {0, 0.5}:
- **S1 accuracy:** final mean ≥ 0.90 / 0.80 / 0.70 of the measured joint-multitask accuracy (L0 / L1 / L2).
- **S2 retention:** ≥ 0.90 / 0.80 / 0.75.
- **S3 calibration:** NLL ≤ 1.5 × dense-with-replay NLL.
- **S4 plasticity:** K-th domain learned within 2× the steps of the first (K ≥ 12).
- **S5 integrity:** margin ≥ +0.25 accuracy over dense no-replay; every null ≤ majority + 0.05; no condition above Bayes-ceiling + 0.02; lookup-only control fails on T1–T3.
- **S6 scaling:** stored statistics grow with K·L·d², not with samples; step time does not grow with history.

Thinking (T4, train depth ≤ D, test depth up to 2D): (i) mean accuracy rises monotonically with steps (Spearman > 0.8); (ii) at depth 2D, ≥ +10 points over Th0 with CI excluding 0; (iii) adaptive halting uses ≤ 60% of max-step FLOPs at ≤ 1 point accuracy loss.

A negative result is a valid outcome; it is reported as such.

## 10. Build order and go/no-go points

| Milestone | Content | Decision it forces |
|---|---|---|
| **M0** | Harness: generators (T0–T3, ρ, L0–L2), controls (nulls, oracle-route, lookup-only, joint, dense ±replay), PREREG template; diagnostics **E0a** (only new expert trainable → retention?), **E0b** (oracle vs learned routing), **E0c** (v3 with backbone left plastic = A1, plus a learning-rate sweep for the backbone). Measure ceilings, then freeze §9. | Is drift (D1) or routing (D2) the dominant cause in this regime? Are thresholds sane? |
| **M1** | A1, A2 (snapshot encoder + closed-form transport) | Do stored statistics survive a moving encoder? |
| **M2** | A3 (null-space on all layers), d sweep 32 / 64 / 128 / 256, K sweep | Capacity curve; at what K does plasticity die? |
| **M3** | A4, A5 (content-addressed routing, Hotelling novelty, calibration) | Can L1/L2 be passed without a task-ID? |
| **M4** | Thinking arms Th0–Th3 on T4 | Is thinking learnable at this scale? Does energy help CL? |
| **M5** | Real twins; complex-vs-real decision | Complex or real for the core |
| **M6** *(optional)* | A6 bounded dreaming | Does generated rehearsal add anything beyond M1–M3? |
| *(gated)* | A.3 real-byte corpora | Does any of it survive real data? |

Two implementation plans, in order: **Plan A** (M0–M3, M6) then **Plan A2** (M4), with M5 spanning both. Phase B (efficiency) follows.

## 11. Left to the builder
**One item is a research question, not just an implementation detail: the consolidation trigger in L2.** Null-space protection (A3) needs to know which covariance is "old" and must be protected versus "current" and must stay learnable. With boundaries supplied (L0/L1) the harness marks it; with none (L2) the learner must decide — candidates are a novelty-triggered snapshot of the running covariance, or a fixed-lag running covariance. Try at least both and report which works; do not give L2 an implicit boundary flag.

Everything else: module and class structure; the optimiser wrapper that projects updates; numerics for Hotelling-T² (shrinkage, pseudo-inverse); exact generator parameters (chain sparsity, ρ implementation, depth distributions); logging format (follow existing JSON + PREREG conventions); test design; device placement; how to batch seeds for speed.

## 12. Calls I made — please check these
1. **"Always plastic" = no permanent `requires_grad_(False)`**; slower/constrained updates are allowed. If you meant *no constraints of any kind*, A3 is out and the plan changes.
2. **Two tracks:** generated rehearsal (A6) is allowed only as a separate, clearly labelled bounded-dreaming track. If any generated old-domain input is out of bounds, drop A6.
3. **Toy scale first** (d_model 32→256, K ≤ 24) and real corpora gated behind the toy ladder.
4. **Complex first, real twin after primary results** (M5); δ and "≥ 2 of the primary metrics" are provisional.
5. **Thinking benchmark = synthetic algorithmic tasks (T4)** before anything ARC-like.
6. **v3 is kept as reference A0**, not discarded; it defines the L0 ceiling.
7. **Nothing is committed to git**; docs and result folders stay untracked until you ask.
8. Thresholds in §9 are my provisional guesses and are deliberately re-derived from measured ceilings at M0.

## 13. Risks
- **Null-space capacity** may be exhausted at d = 32; the d sweep exists to find out, and a negative result is informative.
- **Drift compensation** is fitted on current-domain inputs and may not extrapolate to old regions of feature space (the known weakness of LDC/ADC).
- **Token-level routing under high ρ** may be intrinsically ambiguous; abstention and coverage must be reported, not hidden.
- **Thinking may not help at toy scale** (depth-recurrent LMs show marginal gains on GSM8K); T4 is chosen because recursion has the clearest published wins there (TRM, Attractor Models — single-group results, no independent replication found).
- **Toy ladder may still be too easy**; hence the lookup-only control and the gated real-byte stage.

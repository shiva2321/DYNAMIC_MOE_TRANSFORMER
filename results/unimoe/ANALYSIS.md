# UniMoE Stage 1: where the combined system stands, and why

Written 2026-10-30. Every number below is copied from a generated file (`REPORT_v2.md`, `E2_domain_routing.md`, the `p3_*` cell folders, `PROGRESS.md`). All seeds are dev seeds (499 for tuning, 500-502 for evaluation); nothing here is a pre-registered confirmation, and with three seeds the intervals are wide.

## 0. The verdict in plain words

We combined what the project learned into one system: a frozen sentence encoder on a shared unit sphere; a pool of small experts that grows by spawning when something new appears; a batch-level "is this group new?" test; a learned router; a rule that decides whether a new group needs a new expert; and, on top, a language-model reader that can say "none of these". We trained it on a mixed stream of real data: customer requests (banking77), news topics (ag_news), newsgroup posts, emotions, movie-review sentiment, arithmetic word problems (SVAMP) and arithmetic in digits, 22 regimes in all, in three orders (one domain after another, everything mixed, drifting). Then we measured it from many sides.

**On this suite the combined system works, but it does not beat simple baselines.** The best pool variant (0.62-0.63) sits level with a plain network holding the same memory (0.636), with a nearest-neighbour lookup that stores the data (0.642), and 0.04 below an offline network trained on everything at once (0.669). It forgets very little (0.046, against 0.60 for plain fine-tuning), it sends 83 % of text messages to a sensible expert, and its batch-level tests are what make it work. But the three headline features the architecture is for are not shown to pay on this suite: (1) automatic growth in width and depth never fired and, when forced to, could not help; (2) on stationary or gradual streams the pool collapses to a single expert; (3) its own "I don't know" flag is silent. Several of those failures come from interactions between parts that only an integrated run could expose, and the rest from limits of the suite itself. Section 4 ranks them.

## 1. What was run

Seven domains, 22 regimes, 125 global classes, 12.8k training messages per stream; schedules `seq` (28 phases), `mix` (5), `drift` (10). Ten systems: the shipped pool, three variants (`U-real`: batch-level set gate with persistence 2, founding-cluster fix, fit-gap width growth; `U-deep`: the shadow-trial growth policy; `U-recipe`: a larger training recipe), and six baselines at equal memory or as ceilings (single network with replay, kNN, streaming LDA, an offline joint network, experts told the regime, plain fine-tuning). Every trainable arm was tuned on seed 499, **per schedule** (the first round tuned only the sequential schedule and reused the recipe elsewhere; that was a validity error, found and corrected, see section 6). Nine ablations (U-real minus one component), a domain-routing comparison with the language-model reader, an expert-quality ladder, a width-growth study. 407 cached cells (system, schedule, seed).

## 2. The scorecard (final mean accuracy over the 22 regimes, mean over seeds 500-502)

| system | sequential | mixed | drifting |
|---|---|---|---|
| offline joint network (ceiling, widest) | 0.669 | 0.668 | 0.670 |
| experts told the regime (routing ceiling) | 0.669 | 0.663 | 0.604 |
| nearest neighbours at equal memory | 0.642 | 0.517 | 0.534 |
| single network + replay at equal memory | 0.636 | 0.632 | 0.600 |
| UniMoE `U-recipe` | 0.630 | 0.582 | 0.549 |
| UniMoE `U-real` (= `U-deep`) | 0.622 | 0.596 | 0.512 |
| streaming LDA (closed form) | 0.606 | 0.610 | 0.605 |
| the shipped pool | 0.546 | 0.596 | 0.531 |
| plain fine-tuning | 0.075 | 0.552 | 0.431 |

Where the pool structure lives: sequential 19.3 experts for 22 regimes (17.3 regimes with an expert of their own, purity 0.84, NMI(expert, domain) 0.70); mixed and drifting: **one expert** in every cell (a mean of 1.3 for `U-real` in the drifting schedule).

## 3. What works (and the evidence)

1. **The batch-level machinery is the core of the system.** Removing the set gate costs 0.079 (experts 19.3 to 10.3) and accounts for all of U-real's +0.076 over the shipped pool; the transfer rule 0.038; batch vote 0.029; the founding-cluster fix 0.012. The per-sample gate alone is nearly blind to real text categories.
2. **Continual learning is real.** Mean forgetting 0.046 for the pool against 0.604 for fine-tuning and 0.075 for replay (kNN 0.036 by never overwriting).
3. **The representation is sound.** A linear classifier trained per domain on the 128-d sphere scores 0.90 (banking), 0.85 (ag_news), 0.77 (SST-2), 0.68 (SVAMP), 0.60 (newsgroups), 0.52 (emotion), within 2-9 points of raw MiniLM; the random projection costs most on emotion.
4. **Domain routing is good, close to what these vectors allow.** On the six text domains the pool sends a message to an expert of its own domain 0.833 of the time (per seed 0.862, 0.868, 0.769); a classifier trained to name the domain gets 0.893; nearest centroid 0.781. Wrong-domain errors: pool 10.8 % of all test messages, offline joint network 9.0 %.
5. **Cheap parts are inert, as designed:** the fingerprint shortlist changes nothing below 64 experts (0.000 exact), the competence monitor -0.006, the learned router -0.002.

## 4. What is wrong, ranked by how much it explains

**4.1 The pool buys nothing over simple baselines here (and costs more).** Sequential: pool 0.622-0.630 against replay 0.636 and kNN 0.642 at the same memory; the pool's prediction costs 0.86 M multiply-adds against 0.28 M for replay (the shortlist saves nothing below 64 experts). The headroom of the whole suite is small: the best system that is not told the task reaches 0.669, about 0.05 above the pool (and the domain-told linear reference is 0.752, section 4.2). Reason: the suite's tasks are almost all linearly readable from frozen embeddings, which is exactly where lookup and one network do well. (The older claim that the pool wins where each regime has its own rule is untouched; this suite does not test it.)

**4.2 The ladder of losses (regime-weighted over 22 regimes, 3 seeds).** Domain-told logistic regression per regime 0.752; the same regimes with online MLP experts told the regime 0.669 (-0.083: how experts are trained online); the pool 0.622 (-0.047 more: routing and sharing). Of the first loss, 0.056 is online against offline fitting (a linear expert trained online scores 0.696 and the offline fit 0.752) and 0.028 is the MLP being a worse fit than linear (E1). More training steps make the MLP worse (0.658 at 50 steps against 0.668 at 20: it overfits each batch) and width does nothing (+0.003 at 512 units). **But the linear-expert gain does not transfer into the pool (0.612 against 0.622 sequential, E7)**: once experts hold several regimes and routing is imperfect, expert shape is no longer the limit.

**4.3 Task ambiguity puts a floor under every method that is not told the task.** The suite has no task id, so a news-like message may belong to ag_news or to the newsgroups, an emotional sentence to emotion or SST-2. Even the offline joint network confuses domains 9.0 % of the time, almost all among those overlapping text domains (newsgroups to ag_news 309 and the reverse 292 of its wrong-domain errors, newsgroups to emotion 119, SST-2 to emotion 98). Banking, SVAMP and arithmetic are almost never confused (0.00-0.05). This is a property of how the domains were stacked, not a defect of the pool, but any real deployment with overlapping tasks has it too.

**4.4 Automatic scaling is effectively switched off, for three reasons that only an integrated run could show.**
 (a) *Stream length.* The width-growth rule needs an expert to have trained on 300 batches of its own; the streams have 162 to 221 batches in total, at most 23 per regime. Lowering that condition to fit the stream was tested.
 (b) *The growth statistic depends on the training recipe.* The "fit gap" (the share of a batch's loss that the expert's own steps remove) is 0.09, 0.59 and 0.98 at 1, 5 and 20 inner steps for a 128-unit expert and 0.05, 0.39, 0.82 for a 16-unit one; the threshold 0.17 means "starved" only near one step per batch. The recipe I tuned for accuracy (20 steps) therefore disables growth for any plausible threshold.
 (c) *Even where it fires, the rule is too slow to matter here.* At one inner step with the maturity condition scaled down it doubles an expert once per run (16 to 32 units); the gain was +0.036 for a 16-unit expert in the mixed schedule (0.442 to 0.478), -0.001 sequentially, and nothing at 128 units.
 And the suite barely needs capacity anyway: with growth off, 16-unit experts score 0.581, 128 units 0.622, 512 units 0.631, so a perfect growth rule could add at most 0.05 over the smallest pool. **The suite cannot tell us whether automatic scaling works; the coupling between growth rule and recipe is a real defect.** Vertical growth by layers was not evaluated at all: the layer module has no trigger that decides when to add one.

**4.5 On stationary and gradual streams the pool is one network.** In the mixed schedule the pool founds one expert, in the drifting schedule one (1.3 on average for `U-real`), because spawning is triggered by an abrupt novelty event. With per-schedule tuning that one expert scores 0.596 (mixed) and 0.512 (drifting), 0.036 and 0.088 below the equal-memory replay network, and below a closed-form LDA in both. "Dynamic" structure appears only when the data changes in steps.

**4.6 The pool's own "I don't know" is silent.** At the end of the sequential schedule the unknown flag marks 0.000 of known messages and 0.00 of all three out-of-distribution sets (children's stories, encyclopedia sentences, school science questions); the novelty score separates them only weakly (AUROC 0.66-0.74 for `U-real`, 0.73-0.85 for the shipped pool). The language-model reader of the Router layer refuses 0.75-0.98 of that out-of-distribution text, but it also refuses 0.28-0.40 of known messages and routes less accurately (0.647), and it depends on the cards' wording (newsgroups 0.10, SVAMP 1.00). The calibration on verified outcomes that fixed this in the earlier studies (E-050) was not wired in.

**4.7 Arithmetic exact match is low for every learner** (U-real: add 0.02, sub 0.02, mul 0.09, mod 0.14, div 0.28, max 0.18; the offline joint network 0.07, 0.03, 0.09, 0.20, 0.45, 0.21): about 3,000 digit-slot examples per operator, against the roughly 10,000 the earlier arithmetic study found necessary. Not a pool defect; a data-budget limit of this suite.

## 5. Why the numbers are what they are (a short causal story)

Frozen embeddings make almost every task here close to linear, so the ceiling is low (0.67) and simple learners sit near it. The pool's real contribution is to keep separate things separate, which it does well (forgetting 0.046, domain routing 0.83) thanks to its batch-level tests. What it loses it loses in two places: how its experts are trained online (a 0.08 handicap against fitting a linear classifier offline) and where routing or sharing blurs the regimes (a further 0.05). Its headline mechanisms for growing and for refusing are either disabled by the way training and growth were coupled, or measured on a suite that does not need them.

## 6. Corrections, caveats and what was not measured

- **Corrected during the work:** (1) the first round tuned every recipe on the sequential schedule only; at one inner step the pool scores 0.596 in the mixed schedule where my reused 20-step recipe gave 0.510. Report v1's mixed and drifting numbers are superseded by `REPORT_v2.md`. (2) The joint "ceiling" was first a 128-unit network that scored below the replay network; widened to 4096 units before any evaluation cell was written. (3) The auto-launched evaluation was killed within seconds to apply (2).
- **Predictions:** 63 numeric predictions were declared before their runs; **31 held and 32 missed**. The misses cluster on three habits of mine: I over-weighted the effect of tuning knobs (recipe size, thresholds), I did not look for couplings between parts (growth against recipe and stream length), and I took a suite's headroom for granted.
- **Measured with three seeds**, so intervals are wide (the shipped pool's sequential result is 0.505, 0.540, 0.593 on the three seeds). Seeds 503-504 and the reserved confirmation seeds 600-604 were not used.
- **Not measured:** vertical growth by layers (no trigger exists); the Router layer wired into the pool; refusal with calibration on verified outcomes; exact per-draw ROC for the matched-false-flag reading used in the earlier study; the token-level Stage 2.
- The repository's best-of-seeds warning on the drivers' seed loops is a false positive: every seed is run and every seed is reported.
- The patch that lets the pool accept depth-zero experts lives in the E7 driver only (`growmoe/` is untouched).

## 7. What to change next (each is a hypothesis with a test, in priority order)

1. **Make growth recipe-independent and test it where capacity matters.** Measure the fit gap after the first step instead of after all inner steps, scale the maturity condition to batches the expert has actually seen, and evaluate on streams that need capacity (arithmetic with 10,000 examples per operator, or regimes with real function-like structure). Expected: growth fires and matters there; today's suite cannot show it.
2. **Add a supervised domain head.** The pool sees the label of every message and the labels fall in per-domain blocks; routing by a head trained to name the block should lift domain routing from 0.833 toward the 0.893 of the classifier and cut the 0.047 routing loss. Expected gain +0.02 to +0.04 final mean.
3. **Wire in the refusal layer with calibration on verified outcomes** (the earlier mechanism: both readers' cuts set on outcomes of the stream's own wording, fused by "either"). Expected: OOD refusal at a controlled false-refusal rate, where the pool's own flag is silent.
4. **Then fix expert fit**: closed-form or running-statistics heads for text-like regimes (linear experts gained 0.028 under oracle routing, nothing in the pool until routing improves), MLP for arithmetic, chosen per regime by held-out loss.
5. **Make the pool spawn on stationary and gradual streams** (spawn when class-conditional losses show conflicting labels), so that the mixed schedule is not one network.
6. **Design a trigger for layers** (a propose-validate-commit on a shadow copy), then test vertical growth on a capacity-hungry stream.
7. **Stage 2 (token-level language model)** should reuse only what held up: the batch-level set test and the transfer rule; the per-sample gate and the shortlist are not worth carrying.

Files: `REPORT_v2.md` (all tables), `E2_domain_routing.md`, `tuning*.json`, `runs_v2/`, `p3_*`, figures in `figures_runs_v2/`; code in `unimoe/`, tests in `tests/unimoe/` (56), ledger and predictions in `research/unimoe/PROGRESS.md`.

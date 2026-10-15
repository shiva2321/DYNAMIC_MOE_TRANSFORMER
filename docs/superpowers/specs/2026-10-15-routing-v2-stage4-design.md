# Routing v2, stage 4 — repairs from the root-cause study: a router that chooses its own rule (4a), a label-aware spawn-or-extend test (4b), growth from the fit gap (4c); research spikes for splitting and the recipe (Phase B)

Date 2026-10-15. Status: **written spec for review** (path: architectural; the scope was approved by the user on 2026-10-14, this written spec and then the plan still need their own approvals). Nothing here is built; every new behaviour sits behind a flag that is OFF by default, and with all flags off the pool is bit-identical to today's (statement S4-9). Research behind it: `research/GROWMOE_STAGE4_RESEARCH.md`. Spike numbers: `research/growmoe_results/stage4_spikes/RESULTS_SPIKES.md` (script-generated; notebook entry E-032, claims C-63 to C-66). Root-cause study: notebook entry E-030, `research/growmoe_results/diagnosis/RESULTS_DIAGNOSIS.md`.

## In plain words

GrowMoE is a pool of small expert networks. The root-cause study found three parts of it that do not do what they are meant to, and they can be repaired without touching the experts themselves.
1. **Which expert answers.** The pool picks the expert by "whose stored inputs look most like this one". When the regimes differ in only one informative coordinate (arithmetic operators) or overlap heavily, that picks the wrong expert 12-40 % of the time, and the loss is large (13 points on arithmetic). A router that is trained on the pool's own stored inputs and that **chooses its own rule by cross-validation, falling back to today's rule unless another rule is clearly better**, recovers the whole loss on arithmetic and 8-11 points on the heavily overlapping stream, and by construction cannot lose on the others (19 streams and pools measured: never worse, +4.0 on average).
2. **Give a new regime its own expert, or let an old one take it over?** Today the decision is a 60-step look-ahead on samples the networks have memorised; it is close to a coin flip and lets related operators share an expert, which costs 7 points and doubles forgetting. The repair asks the plain question first: **does the best existing expert already predict the new samples' labels?** (a held-out test, no training needed). If not, a new expert is born. On arithmetic it separates new operators from known ones perfectly.
3. **Growing as needed.** The mechanism for adding width is sound, but the rule that decides when to use it never fires on a starved expert (stayed at 16 units, 11 % accuracy, where 128 units give 34 %). The repair watches a cheap signal that rises smoothly with width, **how much of a fresh batch the expert can fit in its own few training steps**, and doubles the width while that gap is small and progress has stalled.
Two research items follow only if the first three succeed: splitting a stationary mixture (a mixed stream cannot split today) and a learning-rate schedule for a stream of unknown length.

## 1. Decisions (from the user)

1. 2026-10-14: "yes build the fixes as stage 4 and commit everything ... lets properly brainstorm and research and design and implement as necessary."
2. 2026-10-14: the machine loses power every day; everything must be resumable (D-30): units of minutes, atomic writes, checkpoints, one idempotent resume command.
3. Standing: a change that passes its pre-registered test becomes the default and is noted wherever it matters; historical entry points are pinned to what they ran. PREREG first, numbers script-generated, TDD, tune every arm and report the operating point, memory-lean runs, deliverables not loops, plain words first.
4. Standing ledger: seeds 60-79 stay reserved for the stage-3 confirmatory run (awaiting the user's go), 80-84 are used up, dev seeds are 100 and above, **85-99 are free and are assigned here to the stage-4 confirmatory run (single use)**.

## 2. What the study found and what the spikes added

### 2.1 Faults and their measured size (E-030, 2 seeds, operator-by-operator arithmetic unless stated)

| Fault | Measured size | Evidence |
|---|---|---|
| Routing at answer time: newer experts take 21-32 % of the older operators' inputs | -13.4 points (default lifecycle), -12.4 (one expert per operator); no training leak (99.8-100 % of each block trains one expert) | oracle routing 40.3 / 47.4 against 26.9 / 34.9 |
| Spawn-or-extend probe decides on memorised-sample losses (0.01-0.06 nats against a 0.05 tolerance) | sharing experts costs 7.1 points with perfect routing; forbidding extends: +8.0 points, forgetting 15.6 -> 7.2; a 10-times longer probe always extends | `default` 26.6 / 27.1, `forced_spawn` 35.0 / 34.9, `probe600` 31.9 (1 seed) |
| Growth policy: yardstick penalises capacity, decides on 51 examples in the shipped configuration | a 16-unit expert stays at 11 %; widened half-way it reaches 29-32 %; the policy cannot be repaired by a lower threshold | grow-from-small, mid-stream widening, probe-length test |
| No split rule: a stationary mixture never splits | perfect splitting and routing would give about 47 % against 37 % (headroom, not yet shown reachable) | one expert per operator, oracle-routed |
| Streaming recipe: one pass, bounded replay, constant rate | MLP-L 63.8 -> 32.8 under the pool's recipe; a decaying rate +4; replay 300,000 against 20,000: +8 | ladder (82 cells) |

### 2.2 What the research says (details and links: the research note)

Class-incremental accuracy factors into within-task prediction and task-id prediction (Kim et al. 2022); the closest expansion-based method names expert selection as its bottleneck (CN-DPM, 2020); a discriminative gate is the classical mixture-of-experts design (Jacobs et al. 1991) and streaming LDA is a known continual-learning classifier (Hayes and Kanan 2020); growth should be judged from the training signal, not from a held-out probe (self-expanding networks, 2023/24); prequential loss with forgetting is the standard honest stream estimate (Gama et al. 2013). No source combines these in a small task-free mixture of experts, and none runs with a one-pass streaming recipe, so no threshold is taken over: all thresholds below are calibrated on our own dev seeds.

### 2.3 What the spikes showed [M] (throwaway scripts in `research/growmoe_results/stage4_spikes/`; no product file touched)

| Question | Result |
|---|---|
| How much must a router store? (A, saved arithmetic pools) | 16 stored prototypes per expert are enough for LDA (100 % routing, exact match equal to the routing oracle: 40.5 / 47.6 against 26.6 / 34.8 for the shipped gate); softmax 97.8-100 %, nearest class mean 93-99 %. Softmax trained on an imbalanced 20,000-example reservoir routed only 78.6 % right: class balance and a closed-form option matter. |
| Does one rule win everywhere? (A2, A3: 19 pools and streams) | No. LDA alone loses 23-32 points on S1 and 3-8 on banking77. **Selecting among density, LDA, softmax and 5-NN by 5-fold cross-validation on the stored prototypes** (density unless another rule is better by more than 1 point) was never worse than the shipped gate (worst +0.00), +14.2 on arithmetic (26.5 -> 40.6, 34.5 -> 46.9), **+10.7 / +8.1 on OV-0.85** (27.7 -> 38.4, 29.7 -> 37.8; oracle 41.5 / 41.1), +2.0 on OV-0.6, unchanged on F-15 and S2-5, density kept on S1 and banking77; mean +4.0. Remaining routing losses to the oracle: S1 1.9-5.8, banking77 3.4-4.5 (no tested rule closes them). |
| Can "does the parent already predict the new labels?" separate a new function from a known one? (B) | New operators: zero-shot loss 9.6-10.4 nats against a reference of 0.64; called new in 100 of 100 clusters at n = 32, 64 and 128 (exact permutation test, 0.01). Known sub and max: 0 %. Known add: 4.5 / 12.5 / 18.5 % at n = 32 / 64 / 128: within-expert heterogeneity (add's loss 0.89 is above the pooled reference) makes the bare test over-fire at large n, so a practical-significance margin is part of the design. |
| Can need for width be seen on the live stream? (C) | A paired shadow trial (clone widened x2, same next batches, honest prequential loss) is too weak to confirm: after 80 batches the gain is +0.15 % (16 units), +0.03 % (32), and -0.35 %, -0.54 %, -0.93 % for 64, 128, 512 units. It can veto (clearly negative for adequate widths) but not confirm. GradMax initialisation was not better than random-in / zero-out. A simplified natural-expansion score did not separate starved from adequate experts. |
| Is there a cheap signal that tracks width? (C, C2) | Yes: the relative gap between honest loss and loss after the expert's own training steps on the same batch ("within-batch fit gap"): 16 / 32 / 64 / 128 / 512 units: 0.077 / 0.116 / 0.172 / 0.227 / 0.389 (reservoir 20,000) and 0.113 / 0.150 / 0.193 / 0.239 / 0.328 (reservoir 256), 3 seeds, spread about 0.01. The accuracy gained by widening is large from 16 to 64 units (+7.8, +9.5, +5.3 points per doubling) and small by 128 to 512 (+3.1 for x4), so the gap is small exactly where growing pays. The replay-sample gap fails with a small reservoir (0.78-0.98 at every width). |

## 3. Scope

In: **4a** the router (`growmoe/router.py`, flag `router`), **4b** the transfer test (`growmoe/transfer.py`, flag `spawn_rule`), **4c** growth from the fit gap (flag `growth_rule`), config, counters, events, accounting, tests, one pre-registered run, the adoption round. **Phase B** (research, time-boxed, kill rules): 4d splitting a stationary mixture, 4e a learning-rate schedule.
Out: any change to stage 1-3 code paths when the flags are off; the conformal gate's novelty and abstention semantics (the density gate keeps deciding "known or unknown"); pruning or shrinking of experts; non-MLP experts, the front end, learned metrics, hierarchical routing; using the router at TRAINING time (assignment); depth growth (open question, section 11).

## 4. Stage 4a — the router chooses its own rule

### 4.1 Pure functions: `growmoe/router.py` (new; torch and numpy only, no pool imports)

- Data: `X` (n, d) unit-length stored prototypes of the live, dormant and archived experts (`rec.meta["sig"].protos`), `y` (n,) index of the expert that holds each prototype. Prototypes are a uniform sample of what each expert was trained on and survive archiving (an archived expert's reservoir is dropped, its signature is not).
- Candidates, each `fit(X, y, k, gen) -> scorer` with `scorer(Z) -> (m, k)`, higher = more likely owner: `density` (one `VmfSignature` per expert rebuilt from the given rows; score = kernel log-density plus the vMF normaliser; the rule of today's gate), `lda` (class means, pooled within-class covariance plus 1e-3 I, log priors; weights `W = precision @ means^T`), `softmax` (multinomial logistic regression, Adam 500 steps at 0.05, L2 1e-3, **class-balanced loss**, generator seeded from `(cfg.seed, step)`), `knn5` (cosine vote of the 5 nearest rows).
- `cv_owner_accuracy(X, y, k, folds=5, gen) -> {name: accuracy}`: every expert's rows are spread over the folds; the held-out fifth is predicted by candidates fitted on the other four fifths.
- `select_rule(cv, current, margin)`: `density` unless a candidate beats density's cross-validated accuracy by more than `margin` (default 0.01); a switch away from the current rule needs the new best to beat the current one by `margin` too (hysteresis).

### 4.2 `PhyloMoE` changes

- `cfg.router` (default False). When True the pool holds a `RouteRule` (selected name, fitted scorer, expert ids, last cross-validation). It is refitted (a) at the end of any `observe` call in which an expert was spawned, extended, merged or archived, (b) every `router_refit_every` observe calls, (c) on the first `predict`. While any expert holds fewer than `router_min_protos` (8) prototypes, or fewer than two experts exist, the rule stays `density`.
- `predict` (fast path): the gate runs exactly as today (conformal p-values `P`, log-densities `LogF`, the known-or-unknown decision). If the selected rule is not `density`, scores `S = scorer(x)` replace the choice of expert: `router_mode="restrict"` (default, to be confirmed on dev): among the experts the gate accepts (`P >= q`) take the best `S`, and if none accepts take the best `S` overall; `"free"`: best `S` overall. Probabilities, `p_nov` and `unknown` are untouched. If the selection is `density` the code path is today's, bit for bit. `_predict_reference` stays router-blind and is used only by the equivalence tests with the flag off.
- Group answering (stage 3b) takes whatever expert each input was routed to, so it follows the router.
- The router never influences training.
- Events: `router_refit` {selected, cv, rows, experts}, `router_switch` {from, to}.

### 4.3 Config (`PhyloCfg`, validated in `__post_init__`)

`router: bool = False`, `router_candidates: tuple = ("density", "lda", "softmax", "knn5")`, `router_margin: float = 0.01`, `router_folds: int = 5`, `router_refit_every: int = 50`, `router_min_protos: int = 8`, `router_mode: str = "restrict"`, `router_softmax_max_experts: int = 64` (the softmax candidate is skipped above this many experts to bound refit time).

### 4.4 Accounting

`pool_report` adds the selected rule's parameters (LDA: k·d weights and k biases; softmax: k·d + k; density and 5-NN: none) and k·d multiply-adds per prediction; the gate's cost stays (it still decides acceptance and novelty). Router parameters are counted in `params` so equal-memory comparisons stay honest.

### 4.5 Tests (`tests/growmoe/test_gm_rv4_router.py`)

LDA separates six clusters that differ in one of twelve equal coordinates while the density rule does not; cross-validation prefers density on interleaved multi-modal experts (S1-like) and the selection keeps it; margin and hysteresis; flag off is bit-identical (golden); selected = density gives predictions identical to the shipped path; `restrict` never routes to a rejecting expert when another accepts; refit after spawn; a young expert keeps `density`; group answering follows the router; accounting counts router parameters; two identical pools select and route identically; mutation spot-check of the margin comparison and the fold split.

### 4.6 Risks and open items

Cross-validation on prototypes is optimistic (prototypes are training samples); the margin and the guards of S4-3 bound the damage. The four candidates leave 2-6 points to the oracle on S1 and banking77; **research arms, not part of the confirmatory claim**: a stacked candidate (logistic regression on the density and LDA scores), a small MLP, soft mixing of the experts' class probabilities with calibrated router probabilities. They are measured on dev and enter only through a written amendment before the pre-registration. A router must be able to freeze once converged (Li et al. 2025): the refit schedule and the hysteresis serve that purpose.

## 5. Stage 4b — spawn or extend by a held-out transfer test

### 5.1 Pure functions: `growmoe/transfer.py` (new)

`cluster_conflict(cluster_losses, ref_losses, alpha, n_perm, margin, gen) -> (conflict, p, diff)`. `diff` = mean cluster loss minus mean reference loss; `p` = exact one-sided permutation p-value that the cluster's mean is larger (same generator discipline and early stop as `settest`); `conflict = p <= alpha and diff > margin`; non-finite cluster losses (labels the expert has never seen are masked) give `conflict = True`, `p = 0`. The caller passes `margin = transfer_margin * ln(K_seen)` with `K_seen` the number of classes the parent has seen (at least 2): a practical-significance margin that scales with chance loss.

### 5.2 `PhyloMoE` changes

- Per expert `rec.meta["ref"]`: a deque (`transfer_ref`, default 512) of honest per-sample losses, the `preq_losses` the pool already computes before training each labelled batch; infants feed it only after infancy (as the competence monitor does); a merge takes the union. Counted in the accounting as scalars.
- `_handle_cluster`, only when `cfg.spawn_rule == "transfer"`: (1) the cluster's labelled samples are scored (cross-entropy, no training) by every expert with at least `transfer_min_ref` (64) reference losses; (2) the parent is the expert with the lowest mean cluster loss (or `force_parent` on the alarm path); (3) `cluster_conflict` decides: **conflict -> spawn** exactly as today (child by shrink-and-perturb from the parent) with the probe's shadow and child training replaced by `spawn_pretrain_steps` (default 60) training steps of the child on the cluster; **no conflict -> extend without changing any weight**: the cluster joins the parent's replay and signature; (4) an unlabelled cluster, or a parent with too few reference losses, behaves as today (spawn).
- Events `spawn` and `extend` carry `p_conflict`, `loss_cluster`, `loss_ref`, `margin`.

### 5.3 Config

`spawn_rule: str = "probe"` (`"probe"` = today, `"transfer"`), `transfer_alpha = 0.01`, `transfer_margin = 0.25`, `transfer_perms = 999`, `transfer_ref = 512`, `transfer_min_ref = 64`, `spawn_pretrain_steps = 60`.

### 5.4 Tests (`tests/growmoe/test_gm_rv4_transfer.py`)

Pure function: false conflicts at most alpha under exchangeability (400 trials), power for a large shift, the margin blocks a small heterogeneity at large n, non-finite losses give conflict. Pool: a label-permuted regime spawns; a covariate-shifted regime with the same labels function extends and creates no duplicate; a known operator inside a multi-operator expert extends; flag off is bit-identical; infants do not feed the reference; merged experts take the union; accounting counts the reference scalars; determinism.

### 5.5 Risks

A false spawn costs a duplicate expert (memory), a false extend costs interference, so the test is built to favour spawn; duplicates are the merge's job, **and merge has never fired in a benchmark** (component audit): Task 3 includes a merge audit on forced duplicates, and if merge cannot clean them the margin and alpha are made stricter before anything is adopted. Class-incremental streams (banking77) will spawn at every regime (new labels are masked), so expert counts there rise from today's 4-6 to the number of regimes: measured on dev, guarded by S4-5.

## 6. Stage 4c — growth from the within-batch fit gap

### 6.1 Signals (per active expert, from the training step the pool already runs)

`L_pre`: cross-entropy of the batch's new labelled samples before the first inner step (honest); `L_post`: the same samples after the last inner step (one extra no-grad forward of at most 64 rows, about 3 % of a batch's compute); `g = 1 - sum(L_post) / sum(L_pre)` over the last `grow_window` (100) batches; `improvement` = mean `L_pre` of the last window over that of the previous window.

### 6.2 Rule (flag `growth_rule = "gap"`; `"probe"` = today's policy)

An expert is *mature* when `age >= grow_min_age` (300 batches since birth or the last growth) and `improvement >= grow_plateau` (0.95: less than 5 % progress per window). Mature, `g < grow_gap` (gamma, calibrated, section 6.7), budget available and `grow_cooldown` (200) batches since the last growth -> **widen by `grow_factor` (2.0)**: function-preserving exactly as `MLPExpert.widen` (random incoming, zero outgoing weights), optimiser rebuilt as the pool does after any committed growth, windows reset, event `grow` {kind, increment, gap, improvement, hidden before and after, params added}. The gap rises with width, so growth stops by itself near the knee of the accuracy curve; the existing caps (`max_hidden` 1024) and a pool-level `grow_budget_params` bound it.

### 6.3 Guard

For `grow_guard_window` (100) batches after a growth the pre-growth module and optimiser are kept; if the honest prequential loss over that window is worse than the previous window by more than `grow_guard_eps` (2 % relative) the growth is reverted (event `grow_revert`) and the cooldown doubles. Widening cannot change the function, so a harmful growth can only come from the optimiser disturbance; the guard bounds its cost. The guard copy is counted in the accounting while it exists.

### 6.4 Not in 4c

Depth: one more block of the same width did nothing for a starved expert (11.3 % against 11.0 %); a depth rule is an open question. The paired shadow trial (spike C) is not used: it cannot confirm need (section 2.3).

### 6.5 Config

`growth_rule = "probe"`, `grow_gap` (calibrated), `grow_window = 100`, `grow_plateau = 0.95`, `grow_min_age = 300`, `grow_cooldown = 200`, `grow_factor = 2.0`, `grow_guard_window = 100`, `grow_guard_eps = 0.02`, `grow_budget_params = None`.

### 6.6 Tests (`tests/growmoe/test_gm_rv4_growth.py`)

Gap computation from known losses; the gap rises with width (3 widths, statistical); a starved expert grows at least twice and then stops; an adequate expert does not grow; a random-label stream (irreducible noise) stops growing at the caps; the guard reverts a deliberately sabotaged growth; maturity blocks growth in the first 300 batches; budgets respected; flag off bit-identical; accounting; determinism; mutation spot-check of the threshold and plateau comparisons.

### 6.7 Calibration protocol (declared before the dev runs)

The gap depends on the budget per batch (`inner_steps`, learning rate), so gamma is an **operating point** per family. On dev seeds it is chosen from the declared grid {0.14, 0.17, 0.20, 0.23, 0.26} as the value that maximises, over three families (arithmetic from 16 units, arithmetic from 128 units, two stage-3 stream families at their tuned budgets), accuracy minus 0.1 times the relative parameter increase, subject to the guards of S4-7; the whole curve and the chosen value per family are reported. If no single value serves all families, a per-family value declared in the pre-registration is used, and a self-normalising variant is added to the open questions.

### 6.8 Risks

Early-training false growth (maturity conditions); capacity growing to fit noise (the gap opens as the expert starts memorising, and the caps bound it); the signal's dependence on the training budget (calibrated per operating point); the gap uses no reservoir, so the replay size is not a factor (spike C2).

## 7. Phase B — research, time-boxed, with kill rules (after 4a-4c, only if they hold)

- **4d, splitting a stationary mixture.** Headroom: one expert per operator with perfect routing gives about 47 % where one expert gives 37 %. Spike D on the saved iid single-expert pools: cluster the expert's recent samples (k-means in the input, k = 2 to 6) and ask whether the clusters recover the operator (normalised mutual information at least 0.6) and whether per-cluster prequential losses differ (permutation test below 0.01). **Kill rule:** if not, record "not feasible by this route" and stop. If yes: a design amendment for an online split (propose by clustering, test by paired prequential comparison on the stream, accept; duplicates are the merge's job), in the style of a split-merge move (Jain and Neal).
- **4e, the streaming recipe.** Per-expert learning-rate decay with restarts at birth, alarm and growth, and Schedule-Free AdamW if its implementation is small. **Kill rule:** less than +2 points on arithmetic (iid and operator-by-operator) or any guard stream below -0.5: drop.

## 8. Evaluation

**Arms.** `S3` (stage 3 on, today's lifecycle: the base), `S3+T`, `S3+G`, `S3+T+G`, and the router on the SAME trained pools of each (`+R`: the router acts only at answer time, so router off and on are paired by construction). **Streams and seeds.** The ten pre-registered stage-3 streams, 15 seeds, confirmatory seeds 85-99; banking77 class-incremental, 5 seeds (85-89); arithmetic operator-by-operator, 6 seeds (85-90); arithmetic mixed stream from 16 and from 128 units, 15 seeds (85-99). Dev seeds 100 and above. Everything is a paired comparison with a confidence interval, bars frozen from dev results in `bars_rv2d.json` (sha256) and `PREREG.md` before any confirmatory seed, as in stage 3. Constants fixed on dev seeds before the pre-registration, each from a grid declared here: `grow_gap` (section 6.7), `router_margin` in {0.01, 0.03}, `transfer_margin` in {0.15, 0.25, 0.35}, `router_mode` in {restrict, free}, and the inner-step budget of each family (tuned as stage 1 did). The values below are the form and my predictions.

| # | Statement | Prediction (from spikes and E-030) | Falsified if |
|---|---|---|---|
| S4-1 | Router, arithmetic operator-by-operator: exact match with the router minus without it | +12 to +13 points | lower bound of the 95 % interval below +5 |
| S4-2 | Router, OV-0.85 | +8 to +11 | below +4 |
| S4-3 | Router guard: each other pre-registered stream and banking77, with minus without | 0.0 to +2 | any below -0.5 |
| S4-4 | Transfer rule, arithmetic (shipped gate): exact match minus S3; forgetting; experts at the end | +8 points; forgetting about 7; 6 experts | below +5; forgetting above 9; fewer than 5 experts in more than 1 seed of 6 |
| S4-5 | Transfer guard: each other stream and banking77, minus S3; expert count against S3 | within 0.5; at most 1.2 times S3 | any below -0.5; any above 1.2 times |
| S4-6 | Growth from 16 units: final accuracy at least 0.9 of the fixed 128-unit expert, mean final width in [64, 256], extra forward cost at most 5 % of training steps | grows to 64-128, 28-34 % | fewer than 13 of 15 seeds reach 0.9; width outside the range |
| S4-7 | Growth guard: from 128 units and on the stage-3 streams, accuracy minus S3, growth events per run, parameter increase | within 0.5; at most 1; at most +25 % | any below -0.5; more than 1; more than +25 % |
| S4-8 | Full stack on arithmetic operator-by-operator: exact match and forgetting | 46 to 48 %, about 7 | below 44 or forgetting above 8 |
| S4-9 | Golden: all stage-4 flags off, the pool equals the recorded flags-off pool | bit-identical | any difference |

**Decision rules.** Each component is adopted on its own: 4a if S4-1 or S4-2 and S4-3 pass; 4b if S4-4 and S4-5 pass; 4c if S4-6 and S4-7 pass; adopting means the default flips, the old entry points are pinned in the experiment scripts, and notes go wherever the old behaviour was documented. **Kill rules:** if a guard fails the component stays OFF and the others proceed; if the router's selection picks a worse rule than density on any guard stream by more than 0.5 points the margin is raised to 0.03 and the guard rerun once; the stage-3 statements S3-1 to S3-9 must still pass with every adopted component (golden and the stage-3 tests).

## 9. Tests, outage-proofing, honesty

TDD per component (red, green, refactor); the new test files above; the golden `check` with the three new flags pinned OFF in its `BASE`; the whole suite stays green (692 today); a mutation spot-check script for the decision functions (selection margin, fold split, conflict margin, plateau and gap comparisons; at least 90 % of mutants killed). **Outage-proofing (D-30):** every runner works in units of at most ten minutes, writes results atomically and treats an unreadable result as missing; long runs checkpoint; one command resumes (a `jobqueue.py` for `research/growmoe_results/rv2d/`, the pattern of the diagnosis study). Reports are script-generated; notebook entries, claims and decisions are written as soon as numbers exist. What is claimed is limited to the streams tested: the router's gain is a property of operator-like and heavily overlapping regimes, and the growth threshold is an operating point.

## 10. Task order (the plan holds the steps)

0 spikes and this spec (done); 1 scaffolding: flags, events, accounting hooks, golden pins, test skeletons; 2 router (pure functions, pool integration, tests); 3 transfer test (pure function, reference windows, `_handle_cluster` branch, merge audit, tests); 4 growth (signals, controller, guard, budgets, tests); 5 calibration of gamma and the dev constants on dev seeds, a dev run of every arm on every stream; 6 dev review, amendments, mutation spot-check; 7 pre-registration (`PREREG.md`, `bars_rv2d.json`); 8 confirmatory run on seeds 85-99, report, independent audit (D-14), adoption, pins, notes; P1 and P2 the Phase B spikes after task 8's first results.

## 11. Open questions

Q1 a depth rule for growth; Q2 a self-normalising growth threshold; Q3 the faithful natural-expansion score (KFAC factors, residual gradient) as an alternative trigger; Q4 stronger router candidates for S1 and banking77 (stacking, MLP, soft mixing); Q5 whether merge cleans duplicates (audited in task 3); Q6 using the router at training time for assignment (stage 5 candidate); Q7 shrinking (reversed addition) to return unused width.

## 12. Decisions needed from you

1. **Scope.** Build 4a, 4b and 4c now; run 4d and 4e only as time-boxed research spikes afterwards. (Recommended.)
2. **A discriminative component next to the calibrated density gate.** The router decides which expert answers; the density gate still decides known or unknown and keeps its guarantee. Because the router selects its rule by cross-validation and falls back to the density rule, it is built so that it cannot do worse than today on any stream measured. Confirm that this departure from a purely generative design is acceptable.
3. **Seeds.** 85-99 for the stage-4 confirmatory run, as in section 1.
4. **Who builds it.** Recommended: me, natively in this session, tasks one after the other with their tests, then one fresh reviewer on the whole branch at the end (the components are all inside one class and their interfaces depend on each other).

## 13. Refinements made while planning (2026-10-15; the user approved the spec and asked for the plan and the build; none changes a statement, a bar or a decision rule)

1. **R-1, the router is refitted lazily.** The router never influences training, so nothing is fitted inside `observe`. `observe` only counts the call and marks the router stale when an expert was spawned, extended, merged or archived; the first `predict` that finds the router stale (never fitted, the expert set changed, marked, or `router_refit_every` observe calls old) refits it. The triggers are the spec's (a), (b), (c) of section 4.2; training pays nothing; the fit uses the prototypes of the moment of the query.
2. **R-2, transfer rule details.** The parent is the expert with the lowest mean zero-shot loss on the cluster among those with at least `transfer_min_ref` reference losses (the nearest signature is not used). On the alarm path (`force_parent`) the newest `len(cluster)` reference entries are left out, because the cluster IS those samples. If the test cannot run (fewer than 8 labelled samples, no expert with enough reference losses) the pool takes **today's path** (the probe where its conditions hold, otherwise a plain spawn).
3. **R-3, accounting.** `PoolReport` gains `extra_scalars` (default 0); `memory_bytes` adds 4 bytes per scalar. `PhyloMoE.extra_cost(d)` returns (parameters, extra predict multiply-adds, extra stored scalars): the selected router rule (LDA and softmax: k*d + k parameters and k*d multiply-adds; 5-NN: no parameters, a second scan of the stored prototypes as multiply-adds and its copy of them as scalars), the reference windows (scalars), and a live growth guard's pre-growth module (parameters).
4. **R-4, misconfiguration is refused.** `router=True` needs a fast `route_rule` (the reference path of `route_rule="default"` is router-blind): `PhyloCfg` raises, and `predict` raises if the rule is changed to `"default"` afterwards.

# UniMoE: the combined dynamic MoE, built and judged on real streams

Date 2026-10-30. Requested by the user on 2026-10-30: "combine all [we learned] and build a real dynamic MoE architecture with working algorithms and structures and dynamic horizontal and vertical scaling automatically, routing, expert mechanisms, representation spaces, then train on real different types of language, arithmetic and other natural-language data streams, then evaluate what has happened deeply from all aspects so we see where we are short and what we do right and wrong and why, and then improve." The user chose, of three options, **both levels in order**: Stage 1 a message-level system (this document, built now), Stage 2 a token-level language-model MoE that reuses Stage 1's growth, novelty and routing modules (outlined in section 8, designed after Stage 1's report).

Plain words first. Stage 1 is one system that reads a stream of messages of many kinds (customer requests, news, newsgroup posts, emotions, sentiment, arithmetic problems in digits and in words), learns each as it arrives, adds experts when something new appears (horizontal growth), makes experts wider or deeper and adds layers when capacity is the limit (vertical growth), decides at answer time who answers and whether anyone should (routing and novelty), and is then examined from every side against honest baselines. **The deliverable is the evaluation, not a claim**: the point is to learn where the combined system is short.

## 1. What we learned and where it enters the design

| Lesson (claim) | Decision in UniMoE |
|---|---|
| Frozen pretrained text encoder helps (C-31); a vMF density signature with conformal p-values is a good novelty score (C-12, C-15) | Representation: frozen MiniLM to the unit sphere; per-expert vMF signature gate with conformal novelty, level q |
| Per-sample assignment contaminates experts under overlap; a batch vote fixes it (C-22, C-23); the fingerprint shortlist cuts gate cost 7-14x (C-25) | `batch_vote` and `gate_shortlist` ON (they are the defaults) |
| On real text the per-sample gate is nearly blind to category pairs; a batch-level set gate sees them; persistence 2 and group veto 0.2 tame false alarms (C-39, C-33, A-15, A-16) | Set gate ON with `set_gate_confirm=2`, `group_vote` with `group_veto_q=0.2`, circuit breaker ON |
| A learned router (cross-validated rule among the stored prototypes) recovers the whole routing loss (C-57, C-84); the transfer rule founds the right number of experts (C-81, C-85) | Router and `spawn_rule="transfer"` (defaults) |
| Widening and deepening are exactly function-preserving; the shadow-trial growth policy decides by chance and never helps; growth from the fit gap rescues a too-small expert (C-58, C-61, C-86) but never deepens (C-89) | Width by the `gap` rule; depth and layers by the shadow policy and `StackMoE`, evaluated as variants (vertical growth was never judged: D-10) |
| The gap to dense networks is the streaming training recipe, not the expert, routing or growth (C-52) | The training recipe is a first-class option (replay size, steps per batch, schedule), measured, and tuned per arm |
| Merge cannot repair false spawns; false spawns cost about 4 points; repair needs rival tracking and answer averaging (C-67 to C-72); a full pool must share rather than drop (C-73) | Not in the shipped pool: reported as a known gap; `cap_policy="share"`; the harness counts false spawns so the cost is visible |
| The pool seed must differ from the stream seed (C-75); equal-memory baselines and per-arm tuning are required (C-40, C-76, memory rule) | Pool seed = stream seed + 7; baselines at equal memory, each tuned |
| A reader that asks the language model "which of these named skills, or none" plus calibration on VERIFIED outcomes gives honest false refusals and works in four domains; plain fails safe under a noisy outcome check (C-91, C-104, C-109) | The Router layer (`routing_general/router.py`) on top, for named skills: cards from class names, outcomes calibrate it |
| kNN storing everything beats the pool where lookup suffices (C-14); the pool wins where each regime has its own rule (C-40, C-76) | Both kinds of stream are in the suite, with kNN as a baseline on all |
| ISHMN's headline gain was an artifact; hyperbolic heads and branch-and-bound did not pay off (C-01, C-07, C-08) | Not used. Geometry enters only as the unit sphere and the vMF gate, plus diagnostics (section 6, G) |

## 2. Architecture (Stage 1, message level)

New package `unimoe/` (tests `tests/unimoe/`); `growmoe/` is not edited (NG1): where it needs a patch the patch lives in `unimoe/` and is recorded.

1. **Representation spaces.** Every input becomes a unit vector in R^128 on one shared sphere: text through the frozen MiniLM (384) and a fixed random projection; arithmetic in digits through the digit-slot one-hot code of `growmoe/arith.py`; arithmetic in words (SVAMP) as text. Different kinds of input share the sphere but occupy different regions; one gate sees them all. A `Frontend` registry records which stream used which map.
2. **Pool.** `PhyloMoE` with every stage on (table above), a global label space with one label block per stream, seen-class masks, per-expert replay.
3. **Horizontal scaling.** Novel inputs go to probation; a cluster with enough evidence is tested by the transfer rule (does the best existing expert already predict its labels?): extend if yes, spawn a child with inherited weights if no; infancy protection (twin-fix in force); dormant, archive, revive; cap with sharing.
4. **Vertical scaling.** Width by the fit-gap rule; depth by the shadow-trial policy as a variant; **layers** by `StackMoE.add_layer` (identity-initialised representation experts under the head pool) as a variant. Each is a switch so the harness can say what each buys.
5. **Routing.** Gate decides known or novel with the conformal guarantee; the learned router picks the answering expert; the set gate and group vote handle batches that arrive as sets; the Router layer (asking reader plus calibration on verified outcomes) answers "which named skill, or not mine" for streams whose classes have names.
6. **Expert mechanisms.** Residual MLP experts; `widen`/`deepen` function-preserving; DNA inheritance; competence monitor armed after infancy; probation; archive/revive. The training recipe is a parameter set (`inner_steps`, `replay_size`, `replay_frac`, `lr`), reported with every result.
7. **Observation.** Every structural change is an `Event`; the harness snapshots the pool after each phase (experts, parameters, stored vectors, lineage, per-expert class sets, router rule chosen, set-gate verdicts).

## 3. Streams (all local, no download)

| Domain | Source | Kind | Labels |
|---|---|---|---|
| D1 intents | banking77 | text, 77 classes in class-incremental groups | intent |
| D2 topics | ag_news | text, 4 | topic |
| D3 newsgroups | 20 Newsgroups (SetFit) | informal text, 20 | newsgroup |
| D4 emotion | emotion | short affect text, 6 | emotion |
| D5 sentiment | GLUE SST-2 | sentence sentiment, 2 | polarity |
| D6 arithmetic (digits) | `growmoe/arith.py` | six operators, digit slots, exact match | answer symbol |
| D7 arithmetic (words) | SVAMP | word problems, operation type | type |
| D8 never-taught foreign (OOD only) | alpaca instructions, TinyStories, WikiText sentences | never trained on; only for novelty tests | none |

Schedules, each with fresh held-out test sets per domain and per phase: **S-seq** (domains one after another with revisits; the continual setting), **S-mix** (all domains interleaved i.i.d.; the stationary setting where the pool never grew, C-48), **S-drift** (domain proportions drift gradually). Sizes are small enough for one RTX 3060 and the machine's memory (one GPU job at a time; cells checkpointed and relaunchable for the daily power cuts).

## 4. Systems compared (each tuned, equal memory where it matters)

UniMoE variants: **shipped** (defaults), **U-real** (all of section 1 on: persistence 2, group veto 0.2, gap growth), **U-stack** (U-real plus layers and depth), **U-recipe** (U-real with the recipe tuned). Baselines: single network with replay at equal memory (same tuning budget), kNN storing everything, an offline linear probe on the frozen encoder (ceiling for text), per-domain oracle experts (told the domain; ceiling for routing), a multi-head network told the domain (the task-aware reference), and for arithmetic the dense MLP-L of E-026.

## 5. Ablations

U-real minus one at a time: batch vote, fingerprint shortlist, set gate, router, transfer rule, growth rule, layers, replay, infancy protection, probation. Each reports the accuracy change, the structure change (experts, parameters) and the cost change.

## 6. Evaluation battery (what "from all aspects" means here)

- **A. Performance**: final accuracy per domain and macro mean, with 95 % intervals over seeds, against every baseline.
- **B. Continual learning**: accuracy matrix after every phase; forgetting, backward and forward transfer, speed of learning a new domain (accuracy after k batches), plasticity at the last phase.
- **C. Structure**: experts over time against the true number of domains and classes; purity and NMI of experts against domain and class; false spawns (a second expert for one function) and missed spawns; lineage; the event timeline.
- **D. Routing**: owner accuracy against oracle routing and the accuracy gap it costs; gate against router; the domain confusion matrix of routing; the Router layer's claim and refusal rates on named skills.
- **E. Novelty and OOD**: false-novel rate on known data against q; novelty found on the never-seen domains (AUROC and rate at q); behaviour on each domain at its first visit; recovery after a switch.
- **F. Cost**: parameters, stored vectors, multiply-adds per prediction (gate and expert separately), training steps, wall-clock; against equal-memory baselines.
- **G. Representation**: within-domain and between-domain cosine on the sphere, effective dimension, kappa of each expert's signature, cluster separation, layer-wise diagnostics for the stack; which domains share experts and why.
- **H. Ablations** (section 5) and **I. Robustness**: seeds, stream order permutations, S-seq against S-mix against S-drift, imbalance.
- **J. Failure analysis**: every error classified (routed wrong, right expert wrong answer, novelty miss, false refusal) with counts per domain and example messages; the top reasons the system loses.
- **K. Verdict table**: for every claim the design leans on (section 1), whether the combined system reproduces it, with the number.

The report states what is measured and what is assumed, lists predictions made before the run and which missed, and ends with a ranked list of shortcomings and the modification each suggests. Nothing in it is a pre-registered confirmation: all seeds are dev seeds (500-504 for development; 600-604 stay unspent for a later confirmation).

## 7. Order of work and deliverables

- **P0 smoke**: run the shipped pool on banking77 and the digit arithmetic to confirm the tools and the cost per run; record timings.
- **P1 harness**: stream suite, metrics, baselines, report generator, each tested first (test-driven); mutation-checked where it guards a number.
- **P2 first full run (dev, 3 seeds)**: all systems on S-seq, S-mix, S-drift; report v1, "where the combined system stands".
- **P3 deep analysis**: ablations, diagnostics, failure analysis; findings and the ranked shortcomings.
- **P4 improve** (after the user has seen v1): the modifications the findings point to, each with its prediction first.
- Branch `unimoe-stage1` in a new worktree; commits as work lands; no merge, no push.

## 8. Stage 2 outline (designed after Stage 1's report)

A small transformer language model on the spherical residual stream (nGPT family, C-10 under review), trained from scratch on local text streams (TinyStories, WikiText, dialog, math text, code), whose expert feed-forward blocks and layers grow with the Stage 1 modules (novelty gate on pooled hidden states, transfer rule, gap growth, learned router), judged by per-stream perplexity, forgetting and expert specialization against a dense model of equal parameters and a replay baseline. It reuses Stage 1's harness; what it needs from Stage 1's findings is which growth and routing decisions held up.

## 9. Calls I made (for the user to check)

1. New package and branch; `growmoe/` untouched. 2. Seeds 500-504 dev, 600-604 reserved. 3. Eight domains as listed; code-language identification and dialog classification are left out for now (local data is not in a ready labelled form). 4. The Router layer joins only for streams whose classes have names; the pool itself learns from labels and needs no names. 5. Each run is small and checkpointed (one GPU job at a time, relaunchable) because of the power cuts. 6. The first run measures the combined system as it is, before any improvement, so the shortcomings are honest.

## 10. Amendment 1 (2026-10-30, found while building the systems)

- **Layers.** `growmoe.stack.StackMoE` has the machinery to add a layer of representation experts (`add_layer`, infant experts trained through the head's loss, promotion), but its growth policy is not wired into `observe`: nothing decides WHEN to add a layer. Stage 1 therefore evaluates vertical growth at the expert level (`U-deep`: the shadow-trial policy, the only automatic growth that can deepen an expert) and lists the missing layer trigger as a finding; designing one is a P4 item. Section 2 item 4 and the variant `U-stack` of section 4 are replaced accordingly.
- **Router layer.** The asking reader needs named skills and at most 12 cards, while the pool's experts are discovered clusters of up to 13 classes and the suite has 22 regimes. Stage 1's main evaluation therefore uses the pool's own gate and learned router. The Router layer is evaluated in P3 as a domain-level routing comparison (seven domain cards: the asking reader against the pool's routing against the oracle), not as a part of every run.

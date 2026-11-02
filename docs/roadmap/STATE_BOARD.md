# State board: what exists, what must exist, and when

Version 0.1, 2026-11-02. Part of the road plan ([ROADMAP.md](ROADMAP.md)). **This is the file that answers "what is there, and what has to be there at this point?"**
Update it after every milestone and after every result that changes a status (rules at the bottom).

**In plain words.** Two tables. The first shows, for each ability the finished model needs, how far our evidence has climbed. The second lists every component, what we have for it, what is still missing,
the milestone that needs it, and the next action. Words used: **SOLID** (shown on fresh seeds or pre-registered, grade A or a clear B with controls), **PARTIAL** (shown in part, or with a known limit),
**WEAK** (we know it is not good enough), **MISSING** (not started or not tested). Evidence IDs point into the lab notebook (`C-` claims, `E-` entries, `Q-` open questions, `D-` decisions, `F-` failures).

## 1. How far the evidence has climbed (the five levels)

| Level | What it means |
|---|---|
| **L1** | Works on controlled synthetic streams that we built |
| **L2** | Works on real data, against fair baselines (tuned replay, kNN, same memory) |
| **L3** | Works on a **standard benchmark**, against published methods, under their protocol |
| **L4** | Works with a **trained representation** (the model learns its own features) and at a larger scale |
| **L5** | **Independently replicated** by someone who did not build it |

yes = shown with grade A or B evidence and controls. part = shown in part, or with a known limit. no = not tested.

| Ability | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Routing without task labels | yes | part | no | no | part (one independent review of stage 4, E-042) |
| Experts that learn and answer | yes | part | no | no | no |
| Novelty and nuance detection | yes | part | no | no | no |
| Continual learning without forgetting | yes | part | no | no | part (same review) |
| Knowledge lifecycle (dormant, remove, correct) | part | no | no | no | no |
| Efficiency when answering | part | no | no | no | no |
| Efficiency when updating | part | no | no | no | no |
| Other input types than text | part (code and arithmetic only) | no | no | no | no |

Reading the table: the finished model needs every cell to be yes. The road plan climbs it column by column, not row by row.

## 2. Components

### A. Routing and the pool

| Component | Now | Evidence | Gap to the finish line | Needed by | Next action |
|---|---|---|---|---|---|
| Novelty gate: batch-level conformal test with source windows, vMF signatures, BH at q = 0.02 | SOLID on synthetic; PARTIAL on real text | C-12, C-15, C-32–C-39 (calibrated: false alarm 0.056–0.061; detects at cosine 0.85 where the per-sample test is blind); it is the largest single component of the pool (removing it costs 0.079, C-113) | Overlapping and ordered streams on real data (Q-08, Q-16, Q-17); scale | M4 | Adversarial stream suite (I-29) |
| Batch vote and group answering with per-sample veto; breaker for label-sorted batches | SOLID | C-23 (A), C-36–C-38, D-23, D-26 | Interleaved streams (Q-08) | M4 | Include in I-29 |
| Fingerprint shortlist in front of the exact gate | SOLID up to 30 experts | C-25 (A; 6-11x less gate work, accuracy within 0.1 point); acts only from 64 experts by default | Scaling curves at 1,000 and 10,000 experts (Q-10) | M4 | Scaling run with the M0 harness |
| Router (learned owner predictor chosen by cross-validation) | SOLID | C-84 (A: +13.0 on operator-by-operator arithmetic, +10.1 at heavy overlap), C-57 | Refit cost: 2 s at 12 experts, about 90% of it the softmax candidate (C-90, Q-30); not tested at hundreds of experts | M4 | I-02: incremental refit, identical decisions |
| Spawn-or-extend rule (transfer rule) | SOLID as default; extend branch never exercised | C-85 (A), C-81; weak-parent flaw and the chance flag (C-166, E-062, Q-65, Q-67); Q-28 | A stream where a regime should be extended | M4 | Pre-registered same-rule-new-region stream (Q-28) |
| Pool lifecycle: probation, infant protection, twin fix, cap policy, dormant/archive/revive, merge | PARTIAL | C-45 (dormant safe, saves 3-6% memory), C-68 (merge removes almost no false spawns), C-71 (rival tracking removes 11.2 of 12), C-74, C-78, C-79; stage 5 not built | False-spawn repair on natural streams (Q-21, Q-25, Q-26); real offload; certified merge | M4 | Stage 5 with its own pre-registration (I-20), offload test (I-19) |
| Pool answering as a team | PARTIAL | C-72, C-93 | Real ambiguity (Q-06 re-opened) | M4 | I-03 |

### B. Experts and learning

| Component | Now | Evidence | Gap | Needed by | Next action |
|---|---|---|---|---|---|
| Closed-form (LDA) head in every expert | SOLID on frozen features | C-123 (+0.022 / +0.069 / +0.126 over the older pool, 5 seeds), C-131 (taught 0.965, new wordings 0.895); D-43 says make it the default of the next UniMoE system, **not yet in code** | Does it survive a trained representation (Q-43)? Not tested at all outside frozen sentence vectors | M2 (frozen), M5 (trained) | Standard benchmark (M2); probe in M5 |
| Global closed-form head as fast memory | PARTIAL | E-056 (new skills learned 0.79 against 0.34), D-47 not default | End-to-end on mixed streams (Q-53 partly answered) | M2/M7 | Decide default at BP1 |
| Network experts and their sizes | PARTIAL | Birth-sizing +0.019 over fixed 32 (E-060); capacity is not what limits our real suites (C-115) | A task where capacity truly limits; deeper experts untested (C-161) | M5 | Revisit with the small language model |
| Growth after birth (gap rule against shadow trial) | PARTIAL, default held | C-86 (A: rescues a 16-unit expert, 15 of 15 seeds against 6 of 15), C-158 (idle otherwise); D-38 held for your decision; Q-27, Q-41, Q-64 | Decide the default; growth for short-lived experts | M4/M5 | Decide at BP1 |
| Notebook-first newborn (closed-form first, network only if it wins a held-out test) | PARTIAL, not default | C-170 (14-22% less memory on three of four beds, 0.5% on one), C-169 (worst regime lower on one bed); D-56 | Interaction with a trained encoder | M5 | Keep as option |

### C. Novelty and nuance

| Component | Now | Evidence | Gap | Needed by | Next action |
|---|---|---|---|---|---|
| Look-based novelty (distance, density, fast-slow disagreement) | WEAK for near-twin skills; fine for far novelty | Finds only 13-22% of near-twin skills (E-055, kill rule D-45); disagreement is no better than distance (C-128); topical novelty AUROC about 0.85 (E-054, Q-42) | AUROC about 0.95 on topical novelty | M3 | I-07, I-09 |
| Outcome-based novelty (execute and compare) | SOLID for code; not available for language | AUROC 1.000 on real code (E-056); code is self-labelling (E-057) | Language has no outcome signal (Q-48, Q-54) | M3/M7 | I-21 |
| Asking reader (a 1.5B language model chooses from taught skills or "none") | SOLID for wording and function nuance; **too expensive** | False-new 0.025 against 0.279 for look-based (C-91, dev seeds); +0.120 in the pool on arithmetic, no reliable gain on topical text (E-065, E-066); about 50 ms per message | Cost, and the topical gap | M3 | I-06 cascade (P3), I-07, I-08 |
| Cheap stand-ins for the reader (cache, trained head with two guards) | PARTIAL | Head answers 56% of arithmetic messages at 2.1% disagreement, 51% fewer reader texts (E-066); silent on news | Answer share on topical streams | M3 | I-06, I-08 |
| Nuance in natural paraphrase, style, negation | MISSING | Only synthetic shifts tested (typos, case, chatter) (Q-73) | Real paraphrase and minimal-pair suites | M3 | Build the nuance suite v2 |
| From-scratch reader (custom, small) | MISSING | Decided by you (D-40, Q-33), not started | Within 0.05 of the pretrained reader on B1b, B1c, macro F1, false-new | M3 | M3 step 3.2 |

### D. Continual learning and lifecycle

| Component | Now | Evidence | Gap | Needed by | Next action |
|---|---|---|---|---|---|
| Forgetting control | SOLID on synthetic and arithmetic; parity on real frozen features | Forgetting 55.1 (plain net) to 15.2 (shipped pool) to 7.1 (adopted pool, pre-registered, fresh seeds), C-87; experts told the operator lose 0.0, so what remains is routing (C-55); on seven real domains the pool reaches the offline all-data network in mixed and drifting streams (E-054) | Standard benchmark; trained representation; long horizon (Q-11) | M2, M5, M4 | M2 |
| Knowledge removal and correction | MISSING | Dormant and archive exist (C-45); **exact removal untested**; contradiction handling untested | Prove exactness; supersede wrong knowledge | M4 | P2 (I-16), I-17 |
| Learning without labels | MISSING | Labels assumed, except code (E-057); Q-18, Q-35, Q-54, Q-55 | Unlabelled skill discovery | M4/M7 | I-21, I-22 |

### E. Efficiency and measurement

| Component | Now | Evidence | Gap | Needed by | Next action |
|---|---|---|---|---|---|
| Answer-time compute | PARTIAL (counted, not clocked) | 0.08-0.20x of the best same-memory single network (C-42, dev seeds); shortlist 6-11x on fresh seeds (C-25, A); not cheaper before the shortlist (C-21) | Wall-clock, bytes read and energy at batch 1 and 32; against compute-matched baselines; at scale | M1 | M0 harness, M1 scoreboard |
| Update and training compute | PARTIAL, mixed | Whole-run time 0.22-0.35x of the same-memory network (E-025, before the router was adopted); governing costs 4.5x training and 1.71x against fixed-size experts (C-163); the router lifted the demo's pool phase from 5-6 s to 30-31 s (C-90); the skip flag saves at most 7.5% (E-063) | Not re-measured for today's default stack | M1 | M1 scoreboard |
| Memory | PARTIAL | Pool bytes about equal the raw stream at 30 regimes (E-025); notebook-first saves 14-22% (C-170) | Bounded growth; offload | M4 | I-18, I-19 |
| Measurement harness | WEAK | Counters in the pool; wall-clock by a runner; thread count not stored (Q-29); no energy, no bytes-read measure, no validation against analytic costs | Everything | M0 | M0 step 0.4 |
| Baselines | PARTIAL | Tuned replay, kNN, same-memory network, joint network in most tables (D-06); streaming LDA in E-055 | Compute-matched replay, prototype and covariance methods (SimpleCIL, FeCAM-style), random-feature heads, LoRA per domain, retrieval | M1/M2 | M1, M2 |

### F. Scope and generality

| Component | Now | Evidence | Gap | Needed by | Next action |
|---|---|---|---|---|---|
| Trained representation (the model learns its own features) | MISSING | Stage 2 not started (E-053, Q-43); everything uses frozen sentence vectors or synthetic vectors | Whole milestone | M5 | P1, then M5 |
| Language generation / language-model behaviour | MISSING | None | Perplexity-level forgetting and efficiency | M5 | P1, M5 |
| Other input types (images, audio, time series) | MISSING | Code and arithmetic domains only | At least two more types | M6 | After M4 |
| Standard benchmarks | MISSING | None run; only our own streams and three cached text sets (banking77, AG News, 20 Newsgroups) and seven UniMoE domains | M2 and later | M2 | Decide downloads |

### G. Rigour and engineering

| Component | Now | Evidence | Gap | Needed by | Next action |
|---|---|---|---|---|---|
| Method: pre-registration, fresh seeds, controls, mutation tests, notebook | SOLID | D-04, D-05, D-14, the notebook rules; 91 mutants of new code killed over E-055 to E-061; 1,927 tests collected | Keep it up for every new milestone | all | Keep |
| Independent review and replication | PARTIAL | One independent review of stage 4 (E-042); the implementer and reviewer are AI agents working with one person | A party who did not build it, running one-command reproduction | M8 | Plan in M8 |
| Backup and reproducibility | **PARTIAL** | Committed work pushed 2026-11-02 to branch `growmoe-lab` of the named (public) repository; 832 files held by the commit gate (D-72) exist only on this machine; 10.4 GB cache, 9 GB checkpoints and other data not in git; environment pinned in requirements-lock.txt (a fresh install from it is not yet tested); a clean clone passes the suite once the Rust engine is built (BASELINE_V1.md); one machine with daily power cuts | Off-machine copy of everything that cannot be recreated | M0 | Approve the four findings (D-72), then commit and push the 832 files; second copy of the non-git data (marginal-insight queue) |
| Write-up | MISSING | Paper notes only (`PAPER_NOTES.md`: method paper, methodology paper) | Papers, release | M8 | After M2 |

## 3. Rules for updating this board

1. A status goes **up** only with grade A or B evidence in the notebook and a claim row; cite the ID in the "Evidence" cell.
2. A status goes **down** at once when a failure, a validity bug or a literature find says so. Record it in `FAILURES.md` and the change log.
3. After a milestone: update every row it touched, update the level table, bump the version in the file header, add a line to [CHANGELOG.md](CHANGELOG.md).
4. "Needed by" changes only through the change protocol in [ROADMAP.md](ROADMAP.md) section 7.
5. If a row has no "Next action", the plan has a hole: add one or say why it is parked.

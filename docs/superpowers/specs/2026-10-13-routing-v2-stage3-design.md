# Routing v2, stage 3 — a calibrated batch-level gate while learning (3a) and group answering while predicting (3b)

**Amendments found while writing the plan: `2026-10-13-routing-v2-stage3-design-AMENDMENTS.md` (read it with this spec; none changes a statement, bar or decision rule).**

Date 2026-10-13. Status: **written spec for review** (path: architectural; the design was approved by the user in conversation on 2026-10-13, the written spec and then the plan still need their own approvals). Nothing here is built, pre-registered or run on fresh seeds. Evidence tags: **[M]** measured on dev seeds 100-102 by throwaway spikes (direction only; every number is script-generated in `research/growmoe_results/rv3/RESULTS_RV3.md` and the two notebook entries E-023 and E-024; "R5.4" means section 5.4 of `research/GATEKEEPER_RESEARCH_2026-10.md`, "§3" means a section of this spec), **[L]** literature.

## 1. Decisions (from the user)

1. 2026-10-12: build a gatekeeper and routing algorithm of our own that is more efficient, accurate, reliable and smart; research the literature and codebases.
2. 2026-10-13: after the waste review and the real-text check, **approve the design** of section 6 of `research/GATEKEEPER_RESEARCH_2026-10.md` (a two-level gate: stage 2 fingerprint shortlist, stage 3a batch-level gate while learning, stage 3b group answering). Both new levels stay OFF until a pre-registered run; Gemini builds it after stage 2 is adopted; the study is committed (117aec5).
3. Standing: a change that passes its pre-registered test becomes the default and is noted wherever it matters; historical entry points are pinned to what they ran. PREREG first, numbers script-generated, TDD, tune every arm, memory-lean runs (at most 2 workers x 3 threads), no commits unless asked, fresh seeds never touched before the authorising round.

## 2. What the spikes showed [M] (throwaway patched copies of `PhyloMoE.observe`; no product file touched)

| Question | Result |
|---|---|
| Does a batch test see what the single-input gate cannot? (R5.1, R5.6) | Clean setting: a permutation test on the distance between a batch's mean and an expert's stored mean is exact (false alarm 0.056-0.061 at level 0.05) and catches every foreign batch at centre cosine 0.85, where the single-input gate flags 1.6 % of inputs; works down to batches of 8. |
| Does it survive real text? (E-024) | 20 Newsgroups, AG News, banking77 with frozen MiniLM: the single-input gate flags only 17-34 % of foreign inputs (AUROC 0.65-0.68); the batch test keeps its false-alarm rate and detects 96-100 % from 16 documents at level 0.001, 100 % from 32; the 20 % most similar pairs need 16+ documents (85 % at 16, 44 % at 8). Clean setting only. |
| Why did it fail in the live pool, and what fixed it? (R5.2, R5.4) | Stored prototypes or one recent window as the reference over-fire on merged experts (14-30 % of known batches called new on S1, S2-5, S2-30) -> **one window per recent source** (2 of 729). The single-input gate's favourites as candidates spawned 104-113 experts -> **candidates = nearest window means**. Vote ownership starves a new expert (OV-0.85 24.5 %, 18 experts); nearest-mean ownership costs S1 2.4 points -> **ownership by label loss among the admitted experts**. |
| What does the gate buy? (R5.4) | Against today's pool: OV-0.6 25.1 -> 42.9 % (+17.8 [+14.6, +20.1]); OV-0.85 24.7 -> 30.0 % (+5.3 [+3.8, +7.3]); OV-0.3, OV-0.0, F-15, S2-5 identical; S1 +0.3, S2-30 -0.1, OV-0.3-MIXED -0.5; known single-source batches wrongly called new: 3 of 2,769. Spike criterion (b) is met at OV-0.6, not at OV-0.85 (+5.3 against +8) by this level alone. |
| What does answering a group buy? (R5.5) | Group answered by the majority of its inputs' own choices: set-gate pools OV-0.85 30.0 -> 42.1 (41.6 with the veto), S1 +4.1 (+1.8 with the veto), OV-0.6 +1.4, every other stream within 0.1. The plain vote loses 15-38 points on 50/50 mixed groups; the per-input veto brings that to within 4 points of per-sample. Works from groups of 8 (OV-0.85 +9.1; 16: +11.3). |
| What does not help? (R5.3) | Per-sample arbitration (confidence, entropy, soft mixtures): none (confidence is worse). A permutation test at inference adds nothing to the vote. |
| What breaks it? (R5.7) | Batches sorted by label (exchangeability violated): the vote pool already collapses (19.9 % against 50.4 %); the set gate spawns 81.7 experts on F-15 against the vote's 18.7; a circuit breaker (12 batches, 35 %, pause 100 steps) holds it to 28.0 and never fired on OV-0.85, OV-0.6, S1, S2-30. |

Untested: the live pool on a real stream (M5 builds one), a learned metric, MMD on real text, regimes that differ by style or time rather than topic.

## 3. Scope

In: **3a** the batch-level gate inside `observe` (flag `set_gate`), **3b** group answering in `predict` (flag `group_vote`), their config, counters, events, tests, the pre-registered run, the adoption round.
Out: any change to stage 1 or stage 2 code paths (3a runs after the vote and does not use the gate's `LogF`); the lifecycle rules for spawn, extend, merge, archive (3a only decides what goes to probation and who owns an admitted batch); a learned metric; label-free ownership beyond the nearest-window fallback; handling ordered batches beyond the breaker; the live-pool real-text result (M5 is separate; its numbers enter the run as a reported arm if it exists by then).

## 4. Stage 3a — the batch-level gate while learning

### 4.1 Pure functions: `growmoe/settest.py` (new)

`perm_reject(ref, batch, alpha, n_perm, gen, stat="meandist", chunk=16) -> (rejected, exceedances, permutations_used)`.
- `ref` (P, d) and `batch` (b, d) are unit-length rows. Statistic: `meandist` = `||mean(batch) - mean(ref)||`; `mmd` = squared MMD with an RBF kernel whose bandwidth is the median of the positive squared distances of the pooled sample.
- A permutation draws a random b-subset of the pooled P + b points and recomputes the statistic of that subset against the rest. With c = the number of permutations whose statistic is at least the observed one and n = `n_perm`: p = (1 + c) / (1 + n); **rejected iff p <= alpha**, counted in integers (never in float32).
- Early stop: permutations are drawn in chunks of `chunk` in a fixed order from `gen`; as soon as c > alpha (1 + n) - 1 the decision "not rejected" is final and the loop stops. The decision therefore equals the one computed from all n permutations drawn from the same generator state. For alpha 0.001 and n 999 a known batch stops at the first exceedance.
- `perm_pvalue(...)` is the full computation, for tests.

### 4.2 Per-expert source windows: `growmoe/sources.py` (new)

`SourceWindows(max_windows=4, max_batches=5, min_ref=48, split_alpha=0.01, min_split=8)`.
- Holds up to `max_windows` windows; a window is a deque of up to `max_batches` tensors (the samples one trained batch sent to the expert).
- `update(xb, gen, stat)`: no window yet -> open one. Otherwise, if the newest window holds at least `min_ref` samples and `len(xb) >= min_split` and `perm_reject(newest, xb, split_alpha, ...)` rejects -> open a new window (drop the oldest beyond `max_windows`); else append `xb` to the newest window.
- `refs()` -> the windows with at least `min_ref` samples, each concatenated. `merge_from(other)` -> union, capped at `max_windows` (newest kept).

### 4.3 `PhyloMoE` changes

- `rec.meta["windows"]` is a `SourceWindows` per expert: a spawned or extended expert starts with its cluster as the first window; a merged expert takes the union; archived experts keep theirs. After training in each `observe`, every expert that received samples updates its windows with exactly those samples (the final `assigned`, including samples a spawn or extend assigned).
- **Placement:** after the batch vote, before the probation block.
- **Trigger:** `cfg.set_gate`, at least `batch_vote_min` (8) samples accepted by the per-sample gate (`assigned >= 0`), at least one expert, breaker not paused. Otherwise nothing changes: the vote stands and `set_verdict` is `None` (`"paused"` while the breaker is open).
- **Candidates and references.** For every expert the references are its windows from `refs()`, or its stored prototypes if it has none. All (expert, reference) pairs are sorted by the distance between the reference mean and the mean of the accepted samples; at most `set_gate_candidates` (3) distinct experts are tried.
- **Admission.** For each tried expert, test its references nearest first with `perm_reject(ref, accepted, set_gate_alpha, set_gate_perms, gen, set_gate_stat)`; the expert is *admitted* at its first reference that does not reject; stop testing that expert. With ownership `"nearest"` the whole procedure stops at the first admitted expert.
- **Verdict.** No tried expert admitted -> **novel as a set**: `assigned[accepted] = -1`, the accepted samples join the individually flagged ones in the probation buffer (the existing evidence test and interference probe decide spawn or extend). Otherwise **known**: choose the owner (below) and set `assigned[accepted] = owner`. Samples the per-sample gate flagged are never reassigned.
- **Ownership** (`set_gate_ownership`): `"loss"` (default) = among the admitted experts the one with the lowest cross-entropy of `rec.module(x)` on the accepted samples that have labels (at least 8 labelled, else the nearest admitted); `"nearest"` = the nearest admitted.
- **Breaker.** For every batch with at least 90 % of its samples accepted, append whether the verdict was "novel" to a deque of the last `set_gate_breaker_len` verdicts; when the deque is full and its mean exceeds `set_gate_breaker_rate`, the set level pauses for `set_gate_breaker_pause` steps (`self.step`), the deque is cleared, event `breaker_open`. While paused `observe` behaves as stage 1 alone.
- **Determinism.** One `torch.Generator` per decision (and one per window update, with its own call counter), seeded from `(cfg.seed, self.step, call counter)`; two identical pools give identical verdicts.
- **Reporting without changing semantics.** `StepRecord.novel` and `p_nov` stay the per-sample gate's. New optional field `StepRecord.set_verdict` (default `None`; one of `"known"`, `"novel"`, `"paused"`, `None`), new events `set_novel`, `set_override` (owner differs from the vote's), `breaker_open`, and counters `pool.set_stats` = tests, early_stops, perms, novel, known, override, paused (no wall-clock in counters).
- **Edge cases.** One expert: tested like any other. Fewer than 8 accepted: no set decision. Labels all -1 or missing: nearest ownership. An expert with fewer than `min_ref` window samples: its stored prototypes stand in (a young expert's prototypes are its training samples). Mixed batch (two sources): the accepted subset is likely rejected by every expert and goes to probation; the evidence test has the last word.

### 4.4 Interaction with stage 1 and 2
The vote runs first and is unchanged. 3a does not read `LogF`, so the stage-2 cascade (which leaves non-shortlisted entries at `-inf`) cannot affect it. The gate matrices are still computed once per batch exactly as today.

## 5. Stage 3b — group answering

`PhyloMoE.predict(x, ctx=None, groups=None)`; `groups` is a LongTensor of group ids, one per input, or `None`.
- `groups is None` or `cfg.group_vote` False: unchanged, bit for bit.
- Otherwise, for every group with at least `group_min` (8) inputs: take each input's own routed expert (today's rule, `idx`); the group's expert e* is the most frequent one (ties: the lowest expert id). With `group_veto` (default True) an input is routed to e* only if its **exact** p-value under e* is at least q (computed with `_gate_column(e*, x[group])`, never read from the cascade's restricted `P`, which holds 0 for experts outside the shortlist); other inputs keep their own choice. Without the veto every input of the group goes to e*.
- `Prediction.expert` carries the routed ids; `unknown` and `p_nov` are unchanged. Groups smaller than `group_min` are answered per sample.
- The grouping is the caller's: a session, a document, a batch. The benchmarks build groups from single-source test chunks and say so.

## 6. Config (all in `PhyloCfg`; defaults chosen so that nothing changes until a flag is flipped)

| Name | Default | Meaning |
|---|---|---|
| `set_gate` | False | stage 3a on or off |
| `set_gate_alpha` | 0.001 | level of the admission test |
| `set_gate_perms` | 999 | permutations (smallest p-value 0.001) |
| `set_gate_chunk` | 16 | permutations per early-stop chunk |
| `set_gate_candidates` | 3 | experts tried per batch |
| `set_gate_stat` | "meandist" | or "mmd" |
| `set_gate_ownership` | "loss" | or "nearest" |
| `set_gate_min_ref` | 48 | window samples needed to count as a reference |
| `set_gate_windows` / `set_gate_window_batches` | 4 / 5 | windows per expert / batches per window |
| `set_gate_split_alpha` | 0.01 | level of the window-split test |
| `set_gate_breaker_len` / `_rate` / `_pause` | 12 / 0.35 / 100 | breaker (the only parameters tuned on dev seeds, section 8) |
| `group_vote` | False | stage 3b on or off |
| `group_min` / `group_veto` | 8 / True | smallest group / per-input veto |

## 7. Pre-registered statements (bars at about half the dev effect; dev values in brackets)

Arms: `A_vote` (today's default, at that time with the stage-2 cascade as adopted) and `A_set` (`set_gate=True`), each with the budget tuned for `A_vote`. Group answering is evaluated on the trained pools of both arms with groups of 64 single-source test inputs and with mixed groups (32 + 32 and 58 + 6 inputs of two adjacent regimes), paired within the same pool. Streams: F-15, F-30, S1, S2-5, S2-30, OV-0.0, OV-0.3, OV-0.3-MIXED, OV-0.6, OV-0.85, plus a label-sorted F-15. Seeds 60-79.

| ID | Statement | Dev |
|---|---|---|
| S3-1 | Overlap gain while learning: paired `A_set - A_vote` at OV-0.6 >= +8.0 and at OV-0.85 >= +2.5 points, interval lower bounds above 0 | +17.8, +5.3 |
| S3-2 | Guards: F-15, F-30, S1, S2-5, S2-30, OV-0.0, OV-0.3 paired difference >= -1.0, interval lower bound >= -2.0 | 0.0 to +0.3, -0.1; F-30 not yet run |
| S3-3 | Mixed guard: OV-0.3-MIXED paired difference >= -1.0 | -0.5 |
| S3-4 | No spurious experts: mean experts <= regimes + 0.5 on every stream | at most +0.3 |
| S3-5 | False novelty: known single-source batches called new as a set <= 2 % pooled over the single-source streams (ground truth from the stream: a known batch is batch number 3 or later of a first-visit segment, or any batch of a revisit segment, as in the spikes) | 0.11 % |
| S3-6 | Group answering (vote + veto) on `A_set` pools, pure groups: OV-0.85 >= +6.0, OV-0.6 >= 0, every other stream >= -0.5 against per sample | +11.6, +1.0, >= -0.1 |
| S3-7 | Mixed groups with the veto against per sample, every stream: 50/50 >= -5.0 and 90/10 >= -1.0 | -4.0, -0.5 |
| S3-8 | Small groups: OV-0.85 groups of 16 with the veto >= +5.0 | +11.3 (plain vote) |
| S3-9 | Flags off: bit-identical to today (tests, and expert counts and accuracy on every stream) | by construction |
| R | Reported, not gated: wall-clock of `observe` with and without 3a; permutation counts and early-stop share; breaker pauses per run (expected 0 on single-source streams), and the label-sorted F-15 with and without the breaker; MMD against distance of means; groups of 8; experts and parameter counts of both arms; the real-text stream M5 if it exists | |

**Decision rule.** Adopt 3a (flip `set_gate`) iff S3-1 to S3-5 and S3-9 pass; adopt 3b (flip `group_vote`, which changes nothing unless `groups` is passed) iff S3-6 to S3-8 pass. S3-1 passing only at OV-0.6 keeps 3a opt-in and says so. Any of S3-2 to S3-5 failing: 3a is not adopted. **Kill rules:** a guard stream losing more than 1.0 point, mean experts above regimes + 0.5, false novelty above 2 %, or less than +8.0 at OV-0.6 on fresh seeds.

## 8. Dev tuning (dev seeds 100-109; 103-109 are new)

Only the breaker is tuned, from a grid declared here: (len, rate, pause) in {(12, 0.35, 100), (12, 0.50, 100), (20, 0.30, 100), (30, 0.30, 100)}. Choose the setting that (i) never fires on OV-0.85, OV-0.6, S1, S2-30 and F-30 dev runs and (ii) gives the fewest experts on label-sorted F-15 and OV-0.0; ties go to the larger len. Everything else in section 6 is fixed by this spec. Dev values for F-30 and for seeds 103-109 are produced by the dev step and written into the bars file before `prereg` freezes it (sha256, `freeze_check`). The statistic stays `meandist`; MMD is a reported arm.

## 9. Rollout, documentation, reproducibility

Order: stage 2 adoption (RV2-8) first, then 3a and 3b as separate flags in one build, each behind its flag while it is written (test first). Pins on adoption: `bench._phase1_cfg` and `run_m0.A_KW` get `set_gate=False` and `group_vote=False` next to the existing pins, so Phase 1, M0, M2-A, RV2-A and RV2-B reproduce. Documentation (written by Claude after each approval): a component card per level, `GROWMOE_HANDOFF.md`, `SCOPE_CHARTER.md`, `MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` Appendix F, `ROUTING_RESEARCH_2026-10.md`, `GATEKEEPER_RESEARCH_2026-10.md` (result sections), the m0 and m2a AMENDMENT notes, the lab notebook (new entry, claims C-34 and C-36 settled), memory. New files: `growmoe/settest.py`, `growmoe/sources.py`, verdict functions (`rv2c_verdicts`, `rv2c_decision`) and an expected-outcome table in `growmoe/fairfight.py` like `rv2b_verdicts`, `research/growmoe_results/rv2c/` (PREREG.md, run script, bars with sha256, report script). Shared-contract changes to list in every report: `config.py`, `pools/phylo.py` (`observe`, `predict`, the lifecycle hooks that create, merge and update windows), `types.py` (`StepRecord.set_verdict`), `fairfight.py`.

## 10. Protocol and budget

Reuse `fairfight.run_fast / tune_steps / run_record`, `metrics.paired_diff` and `freeze_check`; no new framework. CPU only, at most 2 workers x 3 threads, one job per worker. About 10 streams x 20 seeds x 2 arms of short runs; the group-answering evaluation reuses the trained pools (no extra training). The permutation tests are the new cost: with early stopping a known batch costs one chunk; a rejecting test costs all 999 permutations (about 32 ms on one busy thread).

## 11. Risks

| Risk | Mitigation |
|---|---|
| Dev effects do not replicate (3 seeds; five design generations were tried on them) | fresh seeds, bars at about half the dev effects, the tuned quantity limited to the breaker and declared beforehand |
| The exchangeability assumption fails on real data (ordered, correlated or sorted batches) | circuit breaker; reported label-sorted guard; the single-source random-batch assumption stated next to every claim |
| Synthetic streams only | clean-setting real-text check done (E-024); M5 live-pool stream reported when available; no real-data claim before it |
| A new expert's window is empty or tiny | stored prototypes stand in until `min_ref` samples exist |
| Group answering misroutes a mixed group | per-input veto by default; S3-7 gates it |
| The cascade's restricted `P` corrupts the veto | the veto uses `_gate_column` exact p-values; property test with a cascade pool |
| Abstain events strand set-novel samples in probation | known limitation, counted (R); the evidence test stays the last word |
| Flag interactions (vote x cascade x set gate) | every combination covered by a determinism and flag-off bit-identity test |

## 12. Left to the plan (not blockers)

Event and counter names; the chunk size if profiling disagrees with 16; where the label-sorted stream builder lives (an `ordered` option in `growmoe/streams_m1.py` or a wrapper in the run script); how the group evaluation script builds the mixed groups; the exact test list and round split (a plan of roughly RV3-1 `settest` and `sources`, RV3-2 the gate in `observe`, RV3-3 breaker and `predict(groups=)`, RV3-4 verdicts and run script with dev and `prereg`, RV3-5 the confirmatory run, RV3-6 adoption).

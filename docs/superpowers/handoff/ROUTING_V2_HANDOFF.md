# Routing v2 handoff (Track G continues): rounds RV2-1 ... RV2-9

You are the **implementer** (Gemini). Claude designed this and will review every round; the user relays messages. The code is already written in the plan: your job is to
execute it test-first, run the experiments, and report facts. Read, in this order:

1. `GEMINI.md` and `docs/superpowers/handoff/SCOPE_CHARTER.md` (fences, drift rules).
2. `docs/superpowers/handoff/GROWMOE_HANDOFF.md` sections **0, 2, 3, 8, 9** (how we work, constraints, Windows traps, how to write tests, the report template). They all still apply.
3. The spec `docs/superpowers/specs/2026-10-10-routing-v2-design.md` (what and why, the bars) and the plan `docs/superpowers/plans/2026-10-10-routing-v2.md` (the code and commands, task by task).
   **Ignore the plan's "For agentic workers" header** (it names Claude-only skills). The evidence behind the design is in `research/growmoe_results/rv2/RESULTS_RV2_SPIKES.md` (script-generated).

**State (2026-10-12).** RV2-1, RV2-2 and RV2-3 are DONE (stage 1; all 11 statements PASS, verified by Claude: hashes unchanged, report regenerates identically, suite green). RV2-4 (adoption: `batch_vote` ON by default, Phase 1 / M0 / M2-A pinned off, documentation) was done by Claude; suite now 475 passed. **Next round: RV2-5** (plan tasks 7-10). A reminder that matters: in RV2-2 the pre-registration was followed by the confirmatory run without waiting for APPROVE. The result was checked and stands, but the rule is a rule: **after `prereg` in RV2-6 you STOP**, paste the files, and wait.

## 1. What this builds (plain words)

* **Stage 1, batch vote.** When regimes overlap, per-sample routing sends some samples of a batch to the wrong expert, which then trains on a foreign regime. The fix: if one expert
  holds at least half of a batch's accepted samples, all accepted samples train that expert. Spikes: training purity 78 % -> 99.5 %, accuracy 33.7 % -> 47.2 % at centre cosine 0.3.
* **Stage 2, fingerprint shortlist.** The exact gate scans every stored vector of every expert (97 % of the compute). A cheap first pass (k sub-centroids per expert) keeps the 2
  best experts per input and only those get the exact score; any input the shortlist would call novel gets the exact full scan. **Novelty decisions are therefore identical to the exact
  gate; only the choice among accepting experts is approximate.** Spikes: 6.8x-12x fewer multiply-adds at no accuracy loss with 15-30 experts.
* Both are **flags, OFF by default**. They become the default only through a pre-registered test on fresh seeds and only after Claude says so.

## 2. How the plan is used

* The plan is **code-complete but UNEXECUTED**: its 34 Python blocks were syntax-checked, nothing was run. Expect small slips. Fix a local slip (a typo, an import, an off-by-one in a
  test helper) and list it under DEVIATIONS with the failing output. **Never** change a threshold, a bar, a seed, a pre-registered statement, a decision rule or an expected outcome
  to make something pass: that is a QUESTION for the mediator.
* **Differences from the plan, because of this workflow:**
  1. No commits (the plan has none). 2. **Skip the documentation steps** (Task 6 step 5 and Task 14 step 5): they touch shared files outside Agent G's fence (component cards, handoff,
  reports) and Claude's memory; Claude writes them after APPROVE. 3. **Do not branch on a decision line yourself** (Task 5 step 3, Task 13 step 3): paste the decision line verbatim and
  wait. 4. Wherever the plan says "tell the user" or "show the user", put it in your report. 5. The plan lets you continue past `prereg` when nothing changed; **in this workflow you
  always STOP after `prereg` (RV2-2, RV2-6) and wait for APPROVE before any fresh seed runs.**
* Shared contracts you will touch (`growmoe/config.py`, `growmoe/pools/phylo.py`, `growmoe/accounting.py`, `growmoe/fairfight.py`, `growmoe/bench.py`,
  `research/growmoe_results/m0/run_m0.py`, `growmoe/streams_m1.py`): list every such change with its reason in the report.

## 3. Rounds (one at a time; STOP after each report and wait for the user to relay APPROVE / CHANGES / ANSWER)

| Round | Plan tasks | Deliverable | Report must also contain |
|---|---|---|---|
| **RV2-1** | 1, 2, 3 | stream options (`overlap`, `mixed_tail`); batch vote behind `batch_vote=False`; stage-1 verdict functions | red evidence per new test file; per-seed purity off/on from the purity test (and the seed-scan output only if you had to scan); suite line |
| **RV2-2** | 4 | `run_rv2a.py`; smoke run; `tune --arm A_vote`; `dev`; `prereg` | **STOP before any fresh seed.** Paste `dev_effects.json`, `bars_rv2a.json`, `PREREG.md`, the tuning table (flag anything "at the edge"), the frozen hashes |
| **RV2-3** | 5 (steps 1-2 and 4 only) | confirmatory run on seeds 20-39 and `RESULTS_RV2A.md` | the verdict table and the decision line **verbatim**; files count (260), total minutes, peak memory, any task that errored |
| **RV2-4** | 6, or 6b then 6, **as Claude says** | flip + pins (no docs); or the restricted vote and its run | tests red then green; which files changed |
| **RV2-5** | 7, 8, 9, 10 | `fingerprint.py`; `observe` through `_gate_matrices` (bit-identical); cascade behind `gate_shortlist=0`; counters and `measured_gate_macs_per_input`; `scaling_check` run | the property-test results (flags identical to the exact gate), the scaling table; time `scaling_check` only when no other heavy job runs |
| **RV2-6** | 11, 12 | stage-2 verdicts; `run_rv2b.py`; smoke; scaling; dev; `prereg` | **STOP before any fresh seed.** Paste `dev_k.json`, `bars_rv2b.json`, `PREREG.md`, frozen hashes |
| **RV2-7** | 13 (steps 1-2 and 4) | confirmatory run on seeds 40-59 and `RESULTS_RV2B.md` | verdict table and decision line verbatim; files count (120) |
| **RV2-8** | 14 (no docs) | flip + pins | as RV2-4 |
| **RV2-9** | 15 | stage-3 spike | only if Claude says the entry condition holds |

Rounds can wait for each other: do not start RV2-5 while a confirmatory run is going unless the user says so (timings and the machine's memory are shared).

## 4. Rules specific to this work

* **Seeds.** Dev seeds 100-102 only for tuning, dev effects and bars. Stage 1 confirmatory seeds 20-39, stage 2 confirmatory seeds 40-59: never run them before the round that
  authorises it, never re-run one after changing code or configuration, never look at partial results to change anything. Seeds 0-19 are spent (Phase 1, M0, M2-A).
* **Frozen files.** `tuning_rv2a.json`, `bars_rv2a.json`, `bars_rv2b.json` get a `.sha256` file when `prereg` runs. If `freeze_check` stops you, that is the point: report it. The only
  allowed unfreeze is the one written in plan Task 6b, with an `AMENDMENT.md`.
* **Never hand-edit** any generated file (`PREREG.md`, `bars_*.json`, `RESULTS_*.md`, `*_runs.json`, `dev_*.json`, `scaling_check.json`). Numbers in a report are pasted from them.
* **Flags stay OFF** until Claude relays the adoption round (RV2-4 / RV2-8). With every flag off the exact path must be bit-identical to today's: the 432 existing tests pass **unchanged**.
  If an old test needs changing, stop and ask (charter rule 10).
* **Statistical honesty.** The property test "novelty flags identical to the exact gate" compares flags **exactly**; p-values may differ by one conformal step (float noise from scoring a
  subset of rows). A flag that differs is a bug in the cascade, not a tolerance to widen. The purity test's thresholds (mean purity without the vote at most 0.88, with it at least 0.95)
  were measured with a different random realization: if the control assertion fails on seeds 100-103, use the plan's seed-scan command and change the SEEDS, never the thresholds.
* **Windows.** Console is cp1252: ASCII-only prints (the plan's code is ASCII; keep it so). Worker functions in the run scripts must be module-level (no lambdas) because jobs are
  spawned as processes. Long jobs (more than about 2 minutes): background, log file, poll. At most 2 workers x 3 threads, one heavy job at a time; check RAM (`psutil.virtual_memory()`)
  before starting and stop if it is above 85 %; report each run's peak memory.
* **Scope.** Edit only the files named in the plan's file table. No new top-level folders or packages (`growmoe/fingerprint.py` and the `rv2a/`, `rv2b/` result folders are the only
  new places), no deleting results, no installs, no downloads, no `hash()`, nothing under `ishmn/`.

## 5. Report template (same as GROWMOE_HANDOFF section 9, plus these lines)

```
ROUND: RV2-<n> <title>                    STATUS: done | blocked
FILES: created/modified (paths) + which are shared contracts and why
SUITE: "<N> passed in <T>s" (last line of the FULL run; baseline before this work: 432 passed in about 140 s)
RED EVIDENCE: failing-test summary lines from the first run of each new test file (<= 10 lines each)
PLAIN SUMMARY (for the user): 3-5 lines, simple words, technical terms defined once, measured vs assumed
MEASUREMENTS: tables pasted from generated files (seeds, mean +- CI), peak memory, minutes
DEVIATIONS FROM THE PLAN: each with the failing output / measurement that justifies it (or "none")
FAILURES / SURPRISES: things that did not hold (or "none")
QUESTIONS FOR THE MEDIATOR: numbered, options + your recommendation (or "none")
```

**What Claude checks:** new tests read before code; red evidence is for the stated reason; no vacuous or seed-lucky assertions; thresholds as in the plan or backed by a measured margin;
flags default OFF (until the adoption rounds); bit-identity of the exact path; determinism; generated files untouched (Claude re-runs `report` and diffs); PREREG bars changed only by the
`prereg` rule; decision lines pasted verbatim; the suite re-run by Claude. Verdicts: `APPROVE`, `CHANGES: 1) ... 2) ...` (each item testable), or `ANSWER:`.

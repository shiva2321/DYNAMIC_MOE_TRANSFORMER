# M1 pre-registration: the efficiency scoreboard (APPROVED by you as written on 2026-11-02, D-77; not yet frozen)

Drafted and approved on 2026-11-02. Nothing here has been run. It is hash-frozen only after the cost meter (COSTLAB) is reviewed and before the first report-seed run; after that nothing in it changes, and a change needs a logged decision made before the results it would judge are seen.

**In plain words.** M1 asks one question: where does today's pool really stand, in accuracy *and* in cost, next to the methods people actually use, when one trusted meter measures everything? It claims no superiority. It tells us where the cost sits and which efficiency work comes first. It is fair because every method is tuned the same way on development seeds only, the seeds and the rules are fixed ahead of time, the predictions below are written before any number exists, and a script scores them. What needs you: the adoption rules for the held defaults (section 6.2) and the open points (section 10).

## 1. Question and use

Where do accuracy, update cost and answer cost of Baseline v1 stand against baselines on three beds, and what does that say about the defaults that are on hold? It feeds branch point BP1 (ROADMAP section 5.4) and the first priority of M3 and M4.

## 2. Beds (generators reused unchanged; any deviation is logged)

| Bed | What | Source |
|---|---|---|
| B1 function regimes | N = 10, 30 and 100 regimes, each with its own hidden rule | `research/growmoe_results/shootout/shootout_unaliased.py` (pool seed = stream seed + 7, C-75) |
| B2 banking77 | class-incremental, frozen MiniLM hybrid front end, d = 128 | `research/growmoe_results/m5/run_m5c_stage3.py` and `run_m5d_fresh.py` |
| B3 seven domains | the 22 regimes of E-053, schedules sequential, mixed, drifting | `unimoe/suite.py`, `unimoe/experiment.py` (E-054) |

## 3. Arms

| Arm | Status |
|---|---|
| A1 Baseline v1 pool = `shipped` (tag `baseline-v1`) | exists |
| A2 pool with the best flags: chosen on development seeds, per bed, among `shipped`, `U-real`, `U-real` + `skip_known`, `U-real` + `transfer_chance`, and on B3 also the LDA-head systems of `unimoe/systems2.py`; ties go to the cheaper | exists |
| A3 same-memory single network with replay, tuned (width and steps grid) | exists (`ReplayNet`) |
| A4 **compute-matched** replay fine-tune: its training multiply-adds equal the pool's whole-stream training multiply-adds (budget measured on development seeds with COSTLAB; grid of widths and steps under that budget) | new |
| A5 streaming LDA (`StreamingLDA`) and a Mahalanobis, FeCAM-style prototype classifier | LDA exists, Mahalanobis new |
| A6 random-feature head (fixed random projection plus ridge in closed form, incremental statistics) | new |
| A7 kNN with the pool's memory and with everything stored (`KnnMemory`) | exists |
| A8 joint offline network (reference ceiling) (`JointOracle`) | exists |

Every arm gets the same tuning effort on development seeds (D-06); the operating point (width, steps, thresholds) is reported next to every result.

## 4. Measurements (all through COSTLAB, one meter for every arm)

Final accuracy and forgetting (definitions of `growmoe/metrics.py`); update cost per batch, per new regime and over the whole stream (wall-clock, counted multiply-adds, peak memory); answer cost per input at batch 1 and 32 (median and p95 latency, throughput, counted multiply-adds, active bytes, peak RAM and VRAM, GPU energy for arms that run on the GPU); memory in bytes against the number of regimes; the share of the gate and router in each cost. Threads and affinity are pinned and stored in every record (Q-29); the existing counts of `growmoe/accounting.py` are reconciled with COSTLAB's and any difference is reported.

## 5. Seeds

Development (all tuning, flag choice, budget measurement): 800-804. Report (all verdicts): 810-819. 820-829 stay in reserve. There is no single-use confirmatory run in M1, because it claims no superiority. Seeds 400-429 and 600-604 stay untouched (D-05, D-39). Pool seed = stream seed + 7.

## 6. Rules fixed before the report seeds

**6.1 Decision rule (from the milestone).** On a bed, if any baseline is within 1 point of the pool's accuracy at lower update cost *and* lower answer cost, efficiency becomes the first priority of M3 and M4. Evaluated per bed on report-seed means; recorded even when it does not fire.

**6.2 Held defaults (proposed thresholds; edit before the freeze).**
- LDA head (D-43) becomes the default if, on B3, the paired report-seed difference to `U-real` is at least +0.01 in all three schedules with the 95 % interval above zero in at least two, and answer-time wall-clock rises by at most 10 %.
- Growth from the fit gap (`growth_rule="gap"`, D-38) becomes the default if its accuracy is not below `probe` by more than 0.5 points on any bed and the whole-run wall-clock is at least 15 % lower.
- `skip_known` (E-063) becomes the default if accuracy stays within 0.3 points on every bed and the observe wall-clock falls by at least 10 % on B2 or B3.
- `transfer_chance` (E-062) becomes the default if it gains at least 1 point on one bed and costs no more than 0.5 points on B2 and B3.

**6.3 Validity checks (each one broke us before).** Pool seed offset (C-75); one function for parameter totals (F-11); count events per arm and read the code that builds it (founding cluster, F-31; double-counted messages, F-103); verdicts and tables generated by script, no hand-copied numbers; real latent distribution, not random probes; a gain seen on development seeds is re-checked on report seeds (F-101); a mutation test of the verdict script.

## 7. Predictions (written first; scored by script on report seeds)

Our earlier rate was 31 of 63 (49 %) in stage 1 and 17 of 38 (45 %) in E-054, so about half of these are expected to miss, and a miss is information.

- **M1-1** On B1 with N = 100, CPU, batch 1: the pool's median latency per input is at least 1.0 times that of the best-tuned replay network (no wall-clock gain), while its counted multiply-adds are at most 0.25 times (C-42 gave 0.08-0.20).
- **M1-2** On B1 with N = 100, the pool-to-replay latency ratio at batch 32 is lower than at batch 1.
- **M1-3** Whole-stream training wall-clock of the pool is at most 0.5 times that of the best same-memory replay arm on B1 with N = 30 and N = 100 (E-025 gave 0.22-0.35).
- **M1-4** Gate, router and bookkeeping take more than 60 % of the pool's observe wall-clock on B2 and B3 (E-063 gave training 7-25 %).
- **M1-5** On B2 the better of kNN and streaming LDA reaches at least the pool's final accuracy (E-025: kNN about 4 points ahead).
- **M1-6** On B3, streaming LDA is within 0.02 of the LDA-head pool in at least two of the three schedules (E-054: pool minus replay +0.004, +0.036, +0.045; minus kNN +0.002, +0.161, +0.112).
- **M1-7** The decision rule 6.1 fires on B2 and B3 and does not fire on B1.
- **M1-8** Pool memory grows linearly with the number of regimes on B1, the slope within 25 % of one expert's bytes (no saving that is better than linear).
- **M1-9** By rules 6.2: the LDA head is adopted, `growth_rule="gap"` is adopted, `skip_known` is not (E-063 saved at most 7.5 % of a run), `transfer_chance` is not (E-062 cost a strong pool 0.018).

## 8. Outputs

A scoreboard table per bed with accuracy, forgetting and every cost; Pareto figures (accuracy against update cost, accuracy against answer cost); the top three cost drivers; a recommended order of efficiency work; the evidence for the four held defaults; a plain-English report with one summary figure; a notebook entry with the claims it supports.

## 9. Work and cost

Build first: COSTLAB (done, D-80), then the M1 runner: stage A (bed B1 and the four existing arms) is done and reviewed (D-83, E-068); stage B (the three new arms A4, A5 Mahalanobis, A6, beds B2 and B3, the tuning step, the report and verdict scripts) is written by Gemini to a spec and reviewed by Claude. Runs are small, checkpointed and relaunchable by one idempotent command (D-30), one GPU job at a time. Estimate: M (3-5 sessions) as in MILESTONES.

## 10. Open points for you

1. The adoption thresholds in 6.2 are my proposals; say if any should be stricter or looser before the freeze.
2. A6 and the Mahalanobis arm cost extra build work; I recommend keeping them because frozen features with a closed-form head are the strongest simple baseline in the literature (RESEARCH_DIGEST section 3).
3. GPU energy is reported only for arms that actually run on the GPU; for the tiny networks here the CPU path may be faster and cheaper, which is itself a result.
4. **Added 2026-11-02 after E-068; proposals for the freeze, for your decision (Q-80).** (a) Baseline v1 as built (class-default q 0.05) fragments bed B1: 61 to 67 experts for 10 regimes and 31.0 % against 42.6 % for the pre-stage-4 pool on the five development seeds, while q 0.02 gives 10 experts and 43.0 % (C-195). Section 3 already counts thresholds and steps as part of each arm's operating point, so A1's q and steps are tuned on development seeds like every other arm and the report shows A1 as built next to A1 tuned. (b) A2's candidate list gains the configuration in which the router and the transfer rule were confirmed (stage 4: q 0.02, 20 steps, set gate on). (c) Whether the shipped defaults themselves should change is decided at BP1 on the tuned results, not here.

## 11. Fixed at the freeze (2026-11-03)

The user chose option A of Q-82 (five timed seeds). The scored run is the report seeds 810-819: seeds 810-814 with the full profile (timing, one process at a time on logical processors 0 and 2, the machine otherwise idle) and seeds 815-819 with the deterministic profile (four workers of two threads); seeds 820-829 stay in reserve and are not run. The readings that sections 6 and 7 left open, the run plan and the known limits are written into `docs/roadmap/M1_FREEZE.json` together with the SHA-256 of this file, of `m1/config_m1.py` and of the five tuning files; the code refuses a report-seed run when any of them differs from the freeze (`m1/freeze.py`).

Known limits, stated before the results: wall-clock verdicts rest on the five timed seeds, accuracy and counted multiply-adds on all ten; the GPU repeat of the wide networks (D-89) is not built, so every cell runs on the CPU; A6 and A8 sit at the edge of their tuning grids on B1 and on the B3 schedules and A3 sits at its raised edge on B1 (width 16384, `EDGE_EXTEND_CAP_WIDTH_BY_ARM`; E-072), and the report prints the `at_edge` notes; tuning used development seeds 800-804 only, and the B1 operating points were tuned at N = 30 and are reused at N = 10 and N = 100.

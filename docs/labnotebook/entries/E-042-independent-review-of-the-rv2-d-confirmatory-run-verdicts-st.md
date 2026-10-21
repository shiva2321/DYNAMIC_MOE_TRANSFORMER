# E-042 - Independent review of the RV2-D confirmatory run: the verdicts stand, the growth arm was misread, adoption of growth is held

- **ID:** E-042
- **Date:** 2026-10-21
- **Track:** G
- **Type:** audit
- **Status:** completed; the user has not yet decided the default of growth from the fit gap (D-38)
- **Evidence grade:** A for the verification of the verdicts (recomputed from the raw records by a reviewer who had no access to the author's conversation, plus eight cells re-run bit for bit); the reviewer is the same model family as the author, so it is independent of the author's context, not of the model
- **Code / commit:** state reviewed: `track-g-growmoe` at 6d61eff (code stamp `ff3438232c5f0dfa`, unchanged for the stamped files since 44edd74); review file `docs/superpowers/handoff/REVIEW_RV2D_CONFIRMATORY_2026-10-21.md`; the author's re-check: `research/growmoe_results/rv2d/review_followup_checks.py` (commit 1e732a4 on the branch `adopt-stage4-defaults`)
- **People:** reviewer: a fresh Claude session with the brief `docs/superpowers/handoff/REVIEW_BRIEF_RV2D_CONFIRMATORY_2026-10-21.md` (the user chose "Fresh Claude reviewer" on 2026-10-21); author: Claude, who re-checked the findings against the records; the user has not yet read the findings

## Question
Do the 704 records, the nine verdicts and the notebook's reading of them survive someone else recomputing them, and does anything in them change what we say or do before the stage-4 repairs become the defaults?

## Why we asked it
E-041 reported all nine statements PASS and called its grade provisional until an independent audit. D-09 (the user's standing rule): a repair becomes the default after its pre-registered pass, and the change is documented everywhere it matters.

## Prediction (written BEFORE the numbers, if there was one)
None written down. I expected the verdicts to be reproduced and some wording to need correction. I did not expect a finding about what an arm contains.

## Method (how)
The brief's steps 1 to 5 (the reviewer skipped step 6, adoption, as told): freeze and code integrity; the record set against `cells_confirm.json`; the author's audit; the reviewer's own recomputation of S4-1 to S4-8 from the raw JSON (no growmoe import); the golden check (S4-9); a read of the whole code diff since the dev stamp; eight confirmatory cells re-run from scratch with the current code and compared field by field. The reviewer worked for about 45 minutes (171 tool calls) and wrote only the review file. I then recomputed the four should-fix findings from the records with `review_followup_checks.py` and read `growmoe/pools/phylo.py` lines 838 to 846 myself.

## Results (what came out)
Verified, as the author had reported:
- Record set: 704 of 704, exact set equality with `cells_confirm.json`, one code stamp, no duplicates, no extra records, every accuracy finite; seeds 85-99 appear in no other script and no record or log is older than the start time of D-37.
- Freeze: the bars hash, the code stamp, the 15 provenance hashes of `PREREG.md` and the git history of the frozen files are intact; no stamped file changed after 44edd74; the mutation check kills 27 of 27 mutants.
- S4-1 to S4-8 recomputed independently to the printed precision (the numbers of E-041); S4-9: golden "bit-identical". The four router settings recorded in every cell give the same verdicts, so the pre-registered choice (free, 0.01) decided nothing.
- Eight confirmatory cells re-run from scratch are reproduced bit for bit, four of them only at the run's own thread configuration; this includes `arith-mixed128 S3 s90`, the one record finished by a second process from a checkpoint.
- Stage 4.1 and amendment A4-2 change nothing on the default path (code read, 80 tests, golden, five dev cells re-run bit for bit).

**No blocker.** Four should-fix findings, about what the notebook says and not about what the verdicts are. Numbers below are from `REVIEW_FOLLOWUP_CHECKS.md` (generated from the records):

**SF-1. The "growth" arms replace the shipped growth policy; the full-stack arm had no growth at all.** In `phylo.py` the gap rule is `if growth_rule == "gap": ... elif cfg.grow and ...`: with "gap" the shipped shadow-trial policy, which can widen and deepen, never runs, and the gap rule only widens. Growth events on operator-by-operator arithmetic, by arm:

| arm | shipped or gap-rule growth events, seeds 85-90 |
|---|---|
| S3 | deepen x1 (85), widen x1 (87), deepen x1 (88), deepen x3 and widen x1 (89) |
| S3+T | deepen x2 (87), deepen x2 (88), deepen x3 and widen x1 (89) |
| S3+G | gap-rule widen x1 in each of seeds 85, 88, 89, 90 |
| S3+T+G | none in any seed (all six pools at 802,632 parameters) |

S3+T+G minus S3+T per seed (router off / on): seeds 85, 86, 90 +0.00 / +0.00 (no growth in either arm); seed 87 -0.15 / -0.27; seed 88 -3.49 / -5.39; seed 89 -3.65 / -5.66. Mean -1.21 [-3.13, +0.70] without the router and -1.89 [-4.85, +1.08] with it. All of it is the shipped deepening that S3+T did in three seeds and the full stack could not do.

**SF-2. The expert-count statements are false for OV-0.3-MIXED.** It records 10 regimes but interleaves consecutive pairs of regimes (`mix_pairs`), so it has five sources; S3 holds 5.80 experts and S3+T 5.73 (1.15 x the sources). E-041 and C-88 said both pools found 10.

**SF-3. What S4-4 and S4-5 can say about the transfer rule.** (i) In all twelve units the first visit of every regime is a new function or a new label block, so "spawn at every onset" is always right; no confirmatory stream has a regime that should be extended (same labelling function, new inputs), so that branch rests on unit tests and the spike. (ii) Where the shipped pool already spawns correctly the rule costs a little, systematically: F-30 -0.431 [-0.663, -0.199], negative in 13 of 15 seeds (exact sign test p = 0.007); F-15 -0.339 [-0.740, +0.062]. It passes the bar (-0.5) by 0.069 on the mean. (iii) Amendment A4-1 decided the adoption of 4b (read literally S4-5 fails on four units, E-041 L46); the reviewer finds it justified and honestly disclosed.

**SF-4. S4-5 as coded covers ten of the eleven units "each other stream and banking77" names.** `growmoe/verdicts_rv4.py` (and the audit, which copied the list) leaves out OV-0.85. Included: T minus S3 = -0.278 [-0.624, +0.067], 10.00 experts for 10 regimes (1.00 x); the verdict does not change. OV-0.85 sat exactly on the guard edge in dev (mean -0.500).

Notes N-1 to N-13 of the review, and what was done:

| note | content | disposition |
|---|---|---|
| N-1 | results depend on the thread count: at 2 threads instead of the run's 1 (function-regime) or 3 (arithmetic) four of seven cells differ, by up to 0.48 points on one cell; the records do not store the thread count | E-041 L58 corrected here; Q-29 |
| N-2 | PREREG calls the S3 arm "batch-level gate on"; the mixed-arithmetic cells are built with `set_gate=False` (no verdict depends on it) | noted here; PREREG is frozen and is not edited |
| N-3 | S4-8 combines router-on accuracy with router-off forgetting (taken from training snapshots) | C-87 reworded |
| N-4 | the audit shared one reading of "which streams" with the verdict code and did not compare the record set with `cells_confirm.json` | `audit_rv2d.py` now reads eleven S4-5 units and compares the record set; `AUDIT_RV2D.md` regenerated, every verdict unchanged; `growmoe/verdicts_rv4.py` is stamped and was not edited |
| N-5 | the golden covers the stage-1/2 pool only; stage 3 on has no fingerprint | the stage-3 fingerprints of `test_gm_rv3_confirm.py` (recorded before stage 4, run with `PRE_STAGE4` pinned) pass bit for bit; a golden with stage 3 on is Q-31 |
| N-6 | (a) the forecast "most likely to fail: S4-5, S4-8, S4-6, S4-7" first appears in the results commit; (b) D-37 reads "continue then" as the go; (c) the go is in the author's words only | (a) not right: it is the Prediction section of E-040, committed in f71ce86 on 2026-10-17, four days before the run; (b) and (c) stand as disclosed in D-37 |
| N-7 | five orphan `.ckpt` files beside finished records | checked (each has its finished record, git ignores them) and deleted |
| N-8 | S4-7 is vacuous at 128 units (neither policy fires) | E-041 already says growth never fired; stated again in C-86 |
| N-9 | four arith-seq cells took 5,201 to 7,087 CPU seconds against 970 to 2,485 for the other twenty (the oversubscribed first phase); compare medians | noted; the cost table of `RESULTS_RV2D.md` is noisy for arith-seq |
| N-10 | the records do not store the pool seed; "at least 0.87 of the seeds" is 13/15; D-37 says 9 threads, E-041 12 | noted (D-37 was written at launch, before the budget was widened) |
| N-11 | `router=True` as the default would raise for every bare `PhyloCfg()` because `route_rule` defaults to "default" | handled in E-043: `router` is resolved from the route rule |
| N-12 | all four arms of a unit use the training budget tuned for S3: every "+x points" is at S3's operating point | caveat below; the user's standing rule (tune every arm, report the operating point) is open for the new arms |
| N-13 | C-84, C-85 (apart from SF-2, SF-3) and C-86 (apart from SF-1) are fine | - |

## Interpretation (what we understood, and on what basis)
Plain words. The numbers and the verdicts were right and could be rebuilt from the raw records by someone else. What I got wrong was what the growth result means. I described growth from the fit gap as a safety net added on top of the pool that stays quiet unless an expert is starved. It is not added on top: it is a different policy that takes the place of the one the pool already had, and the old one could do something the new one cannot, make an expert deeper. On the arithmetic stream the old policy deepened experts in three of six seeds, and in two of those the pool scored 3.5 and 3.7 points higher without the router (5.4 and 5.7 with it). The "full stack" arm was built with the new policy and so had no growth at all, which is why it scored below the arm with the transfer rule alone. The loss I reported as "growth costs 1.2 to 1.9 points" is the loss of that deepening, and the gap rule itself did nothing in that arm.

What the gap rule does buy, measured by the reviewer and re-checked here: it rescues an expert that starts too small (from 16 units, 15 of 15 seeds reach 0.9 x the 128-unit pool where the shipped policy does in 6 of 15; the shipped policy lands the expert at 256, 64, 24 or 16 to 64 units depending on the seed; exact sign test p = 0.004), and it is cheaper (in the reviewer's timing on identical trajectories the shipped policy spent 46 % of its wall time in shadow trials that committed nothing, and the gap rule needed 0.46 to 0.71 of the CPU). The pre-registered statements do not compare the two policies where the shipped one fires at large width: S4-7 is vacuous at 128 units and arith-seq is not part of it. So the pre-registered rule says "adopt" (S4-6 and S4-7 pass), and a real trade-off sits underneath it. That is the user's to decide, not mine.

Technical statement. Measured: every number in the tables above, from the 704 records; the code path of SF-1 (read); the timing is the reviewer's, on a busy machine. Not shown: that gap growth combined with deepening would keep both benefits (nobody built it), and anything about the extend branch of the transfer rule on a stream where it should fire.

## Corrections to E-041 (the lines this entry supersedes)
E-041 keeps its text (entries are append-only; it now carries a corrections box at the top); read these lines as superseded (line numbers are those of E-041 as committed in d6bde2e):
1. L42, "On the other streams both give the true count (... OV-* 10.0-10.3 / 10.0-10.1)": not for OV-0.3-MIXED (5.8 and 5.7 experts for five sources); five streams are under-counted by the shipped pool, plus this stationary mixture the pool cannot split.
2. L44, L50, L57 and the commit headline ("growth silent except where an expert is starved", "the one place where an adopted component lowers a measured number", "S3+T+G minus S3+T is -1.2 / -1.9" as the cost of the gap rule): the gap rule replaced the shipped policy, fired four times in S3+G on arith-seq and never in S3+T+G; the -1.2 and -1.9 are the loss of the shipped deepening in three seeds; at 128 units neither policy fires on the ten stage-3 streams, banking77 or the mixed stream.
3. L45 and C-87 ("without growth" for S3+T): S3+T has the shipped growth policy (8 events); the arm without any growth is S3+T+G, 46.6 %.
4. L33 (S4-5): the spec's "each other stream" has eleven units; the verdict code of the run has ten and leaves out OV-0.85, which passes (-0.278, 1.00 x the regimes).
5. L58 ("I did not test whether any single cell differs"): the reviewer did; see N-1.
6. The status line ("the fresh reviewer's audit ... still open"): done, this entry. The commit headline "all nine pre-registered statements PASS" also needs the qualifier that S4-5 passes only under amendment A4-1.

## Caveats and what went wrong
- The reviewer is a Claude session: independent of my context, not of the model. Its timing of the two growth policies is one short run on a busy machine.
- Six seeds on arithmetic: the loss from switching off the shipped deepening is not significant (interval includes 0) and rests on three seeds that grew; so is the gain claimed for it. Nobody has run the comparison where it matters at scale.
- The audit I wrote shared the verdict code's reading of one list (SF-4): an audit written by the author of the verdict code, from the same spec, is not independent on the points where the author misread the spec.
- My own mistake (F-33): I interpreted the difference between two arms before reading the code that makes the arms.

## Consequence
D-38: the router and the transfer rule are adopted as defaults (E-043); the default of growth from the fit gap is held until the user decides, with three options: keep the shipped policy (nothing lost, the gap rule stays an option for starved experts), flip to the gap rule (rescues tiny experts and saves compute, loses the deepening on arithmetic), or build a hybrid that keeps the shipped deepening and test it first (Q-27). My recommendation: keep the shipped policy now and test the hybrid. Corrections go into C-85 to C-88; new claims C-89, C-90; failures F-33 to F-35; questions Q-27 to Q-31.

## Evidence
- `docs/superpowers/handoff/REVIEW_RV2D_CONFIRMATORY_2026-10-21.md` - the review, with its commands and two script appendices
- `research/growmoe_results/rv2d/REVIEW_FOLLOWUP_CHECKS.md`, `review_followup_checks.py` - the findings SF-1 to SF-4 recomputed from the records
- `research/growmoe_results/rv2d/AUDIT_RV2D.md` - the audit after the change of the S4-5 units

## Reproduce
```
cd research/growmoe_results/rv2d
python review_followup_checks.py --write        # the tables of this entry, from the 704 records
python audit_rv2d.py --write                    # eleven S4-5 units, record set against cells_confirm.json
python ../rv2c/golden.py check                  # S4-9
```

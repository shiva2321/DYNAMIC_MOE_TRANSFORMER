# E-043 - Adoption: the router and the transfer rule are the defaults, the old pool is pinned, growth from the fit gap waits for the user

- **ID:** E-043
- **Date:** 2026-10-21
- **Track:** G
- **Type:** process
- **Status:** completed on the branch `adopt-stage4-defaults` (merged into `track-g-growmoe` after the corrections of E-042); the default of growth from the fit gap is open (D-38)
- **Evidence grade:** B (tests, the golden check and one run of the demo per setting on one machine; no new experiment)
- **Code / commit:** `adopt-stage4-defaults`: 43ff17d (defaults, pins, tests, queue guard) and 1e732a4 (demo, notes, audit, review checks)
- **People:** the user chose "Flip defaults, pin old scripts" on 2026-10-21, before the review; Claude implemented it; the reviewer's hold on growth (E-042) narrowed it to two of the three repairs; the user has not yet decided growth

## Question
How do the pre-registered repairs become the defaults (D-09) without changing any recorded result, and what does the change cost?

## Why we asked it
The user's standing rule D-09 ("if it really helps then keep it on by default and make note of it"), E-041 (all nine statements PASS) and the user's answer of 2026-10-21.

## Prediction (written BEFORE the numbers, if there was one)
None. I expected the flip to be mechanical: some tests and about fifty scripts to pin, nothing else.

## Method (how)
Tests first: `tests/growmoe/test_gm_adopt_stage4.py` was written red, then `growmoe/config.py` changed. Then the whole suite showed what the flip touches: 30 failures and 14 fixture errors out of 866 tests. Every one was either a test of the old rules (the 60-step interference probe, the shadow-trial growth policy, the stage-3 fingerprints recorded before stage 4, the router-blind reference loop of the vectorised predict) or a helper that built "the router off" by omission. Those got `**PRE_STAGE4` or the one keyword they need, and a test that compared event lists now leaves out the router's refit events (they say when an answer was asked for, which is what the fast runner changes; the accuracy matrix it compares is identical). 39 `PhyloCfg(...)` calls in 32 research scripts were pinned with a throwaway AST tool, the shared dicts and entry points by hand (list in `research/growmoe_results/PRE_STAGE4_PINS.md`). The RV2-D runner's arms were defined by what they turn on, so `arm_flags` starts from `PRE_STAGE4`; without that the `S3` arm would have silently become the new default pool. The review's hold (E-042) then removed growth from the change: `growth_rule` stays "probe", `grow_gap` is the validated 0.17.

The router follows the routing rule. `router` is `None` by default and resolved in `__post_init__`: on with a fast `route_rule`, off with the reference path `route_rule="default"` (it applies the supersede filter and is router-blind). `router=True` with "default" is still refused. Reason: every measured pool used `route_rule="accept_then_logf"`, and the review noted (N-11) that an unconditional `router=True` would raise for every bare `PhyloCfg()`.

Because the adoption changes `growmoe/config.py`, the code stamp of today differs from the stamp of the 704 confirmatory records, and `jobqueue.py` would read them as missing and run seeds 85-99 again. `rv2d/CONFIRM_CLOSED.json` closes the phase: the queue refuses to start it and reads its status at the recorded stamp.

## Results (what came out)
- Suite: `python -m pytest tests/growmoe` 868 passed, 2 xfailed (the two known merge audit results), 5 minutes. Golden check (S4-9): bit-identical. Smoke phase of the RV2-D runner (6 cells: arithmetic, banking77, function regimes) runs end to end with each arm showing the events it names. `report_rv2d.py confirm` regenerates `RESULTS_RV2D.md` and `verdicts_rv2d.json` from the 704 records unchanged (line endings aside). Audit: all verdicts unchanged.
- The GrowMoE demo (`demos/demo_growmoe_continual_learning.py`, 12 tasks, seed 1), accuracy in percent after the last task, one run per row:

| pool seed | settings | GrowMoE | replay network, same memory | experts for 12 tasks | GrowMoE phase |
|---|---|---|---|---|---|
| stream seed (the demo as it was) | before the adoption | 41.5 | 27.8 | 12 | 5 s |
| stream seed | router and transfer rule on | 42.5 | 29.0 | 12 | 30 s |
| stream seed + 7 (the demo now) | before the adoption | 33.3 | 29.1 | 13 | 6 s |
| stream seed + 7 (the demo now) | router and transfer rule on | 33.8 | 29.0 | 12 | 31 s |

- The demo carried the head start of E-035 (pool seed = stream seed): about 9 points of its GrowMoE accuracy, and the same-memory margin shrinks from +13.5 to +4.8 once the pool has its own seed. Its text quoted the aliased "14 points on 10 tasks"; it now quotes the corrected +6 / +9 / +13 of E-038 and the banking77 figure with the transfer rule (81.8 -> 84.3 %, E-041).
- The router costs time. The GrowMoE phase goes from 5-6 s to 30-31 s: 12 refits, 24.6 s in total, about 2 s each at 12 experts, all of it the cross-validation of the candidates (profile of one refit on 12 experts x 256 prototypes, d = 128: density 0.12 s, with LDA 0.13 s, with kNN 0.10 s, with the softmax candidate 3.32 s, all four 1.88 s; the final softmax fit 0.37 s). The softmax candidate is about 90 % of it. On this stream the router changes no answer (the router selects density).
- With the transfer rule the demo founds 12 experts for 12 tasks; the old pool founded 13 on the own-seed stream (one duplicate).

## Interpretation (what we understood, and on what basis)
Plain words. The pool now asks the labels when it has to decide between a new expert and a stretched old one, and, when it is told to route with the fast rule, it lets a small learned classifier choose the expert at answer time. Everything recorded before today still reproduces, because each old script says in its own text that it wants the old pool. Two things are different from what the user may expect: a bare `PhyloCfg()` has the transfer rule but no router (it needs `route_rule="accept_then_logf"`), and the router refits whenever the set of experts changed since the last answer, so in the demo, which asks for answers after every task, it makes the GrowMoE phase about six times slower, in exchange for a large gain only on overlapping streams (+13.0 and +10.1 points in E-041) and no change elsewhere.

Technical statement. Measured: every number above. Not measured: the router's cost at 30 or 100 experts (the softmax candidate is skipped above 64 experts, `router_softmax_max_experts`), and the demo's own margin over several seeds.

## Caveats and what went wrong
- The demo numbers are one seed; the adoption moves GrowMoE by +1.0 and +0.5 points, which is inside the noise of one run (the replay baseline changes only because its memory budget is set from GrowMoE's bytes).
- The router's refit cost is a finding of this entry, not of the pre-registration; no statement covered it. A change that keeps every decision and skips candidates that cannot be selected (density already within the margin of perfect) would remove most of it; it was not made, because it changes `growmoe/` after the review (Q-30).
- I found the head start in the demo only because I ran it for this entry: E-035 and E-038 found the aliasing and re-ran the shootout, and nobody searched the demos for the same pattern (F-34).
- Growth: the user has not decided; the branch has `growth_rule="probe"`.

## Consequence
D-38. The user decides growth (E-042 lists the three options). Open: Q-27 (a growth policy that keeps the shipped deepening), Q-28 (a stream whose regimes share a labelling function), Q-29 (thread counts in every record), Q-30 (the router's refit cost), Q-31 (a golden with stage 3 on). The twin-fix step (A4-7) and stage 5 are unchanged.

## Evidence
- `research/growmoe_results/PRE_STAGE4_PINS.md` - what was pinned, where, and what was not
- `research/growmoe_results/rv2d/CONFIRM_CLOSED.json` - the closed confirmatory phase
- `tests/growmoe/test_gm_adopt_stage4.py` - the new defaults and the pins
- `demos/demo_growmoe_continual_learning.py`, `demos/growmoe_demo.png` - the demo with its own pool seed

## Reproduce
```
python -m pytest tests/growmoe -q
python research/growmoe_results/rv2c/golden.py check
python demos/demo_growmoe_continual_learning.py                         # the "now" rows (router and transfer rule on)
cd research/growmoe_results/rv2d && python jobqueue.py status --phase confirm     # 704 done, CLOSED
```

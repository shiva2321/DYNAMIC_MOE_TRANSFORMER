# Review brief: the RV2-D (routing v2, stage 4) confirmatory run

For a fresh reviewer (a Claude session without this conversation, or Gemini). The author is Claude, who built stage 4, ran the dev review, froze the pre-registration and ran the confirmatory cells. Your job is to try to find what is wrong, not to confirm. Read-only except for one review file.

## What was decided and when
- Spec `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md` (approved by the user 2026-10-15) and plan `docs/superpowers/plans/2026-10-15-routing-v2-stage4.md`.
- Amendments `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`: A4-1 changes the denominator of S4-5 (accepted by the user 2026-10-17), A4-2 the pool seed of the function-regime streams (stream seed + 7), A4-7 keeps the twin fix out of the run.
- Pre-registration frozen in commit 44edd74: `research/growmoe_results/rv2d/bars_rv2d.json` (sha256 f445f86018dc1034 in `bars_rv2d.sha256`), `PREREG.md`, code stamp ff3438232c5f0dfa, 704 cells in `cells_confirm.json`, seeds 85-99 (single use).
- The run was started on 2026-10-21 00:44 (decision D-37) and uses the code of the frozen stamp.

## What to check (in this order)
1. **Integrity of the record set.** `python research/growmoe_results/rv2d/audit_rv2d.py` reads `cells/` (phase confirm, seeds 85-99, the frozen stamp) and reports problems (wrong stamp, duplicates, non-finite accuracy). Also: exactly 704 records, one per cell of `cells_confirm.json`; no record of another phase or seed; `bars_rv2d.json` still hashes to the stored sha256; `git diff 44edd74 -- growmoe research/growmoe_results/rv2d/rv2d_lib.py` is empty up to the end of the run.
2. **The verdicts.** `audit_rv2d.py` recomputes S4-1 to S4-8 from the raw records WITHOUT `growmoe/verdicts_rv4.py`, written from the spec text; compare with `verdicts_rv2d.json` (written by `report_rv2d.py confirm`). Any disagreement is a finding. Then read `growmoe/verdicts_rv4.py` against the spec table in section 8 yourself: does each rule say what the spec says (bar, direction, which streams, which seeds)? The one intended deviation is A4-1 (S4-5 counts experts against the true number of functions).
3. **S4-9** (golden): run `python research/growmoe_results/rv2c/golden.py check` (about six minutes, single thread) and confirm "bit-identical".
4. **The code that was run.** `git diff 0fd76e3 ba0e2e4 -- growmoe research/growmoe_results/rv2d/rv2d_lib.py research/growmoe_results/rv2d/rv2d_select.py` is stage 4.1 plus A4-1 and A4-2 (the dev cells were made at 0fd76e3). Look for behaviour changes with every flag at its default and fewer than 256 experts (there should be none; the golden check says so for 20 runs); for the cap policy (`share` is the new default when a pool is full); for `pool_seed`.
5. **Claims.** Read the notebook entry written after the run (E-041) and the new rows of `docs/labnotebook/CLAIMS.md`: is every statement limited to the streams tested and to what the intervals allow? Anything stronger than the data is a finding.
6. **Adoption.** If defaults were flipped after the run, check that each component was adopted only on its own statements (4a: (S4-1 or S4-2) and S4-3; 4b: S4-4 and S4-5; 4c: S4-6 and S4-7), that old entry points are pinned (grep for `PhyloCfg(` in `research/` and `demos/`), that the golden BASE still pins the flags OFF, and that the whole suite is green.

## What to deliver
A file `docs/superpowers/handoff/REVIEW_RV2D_CONFIRMATORY_<date>.md` with: the commands you ran and their output, findings ranked (blocker / should fix / note), and one line "sign-off" or "do not adopt until ...". Do not edit code, results or the pre-registration.

## Known limits (so you do not rediscover them)
- Dev seeds 100+ were used for every choice; seeds 85-99 chose nothing.
- The transfer test has no power when the reference expert has not learned yet (Q-25); the arithmetic and banking blocks are long, so this should not matter here.
- The twin-signature defect (E-037) is present in the run by design (A4-7).
- Operator-by-operator arithmetic has 6 seeds, banking77 5; intervals are wide.

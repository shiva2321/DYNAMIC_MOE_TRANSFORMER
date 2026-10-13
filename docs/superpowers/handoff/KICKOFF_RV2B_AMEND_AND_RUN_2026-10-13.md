# Kickoff message to Gemini, 2026-10-13: RV2-6 correction (scaling at k=1) and RV2-7 (stage-2 confirmatory run)

Sent by the mediator exactly as below. Claude's audit of the report that comes back compares against this text.

```
STOP RULE (read first): this message authorises ONLY steps A, B and C below, in this order. After you paste the REPORT you STOP. You do not flip any default (that is RV2-8; Claude authorises it after reading your report), you do not commit, you do not edit any test, bar, seed, statement, decision rule or expected outcome, and you do not touch anything outside research/growmoe_results/rv2b/ except what the commands below write themselves. Every "STOP IF" means: stop at once and report, do not try to repair it.

WHERE THINGS STAND. You reached the pre-registration stop (RV2-6). Nothing has run on seeds 40-59 (there is no rv2b/runs/). Claude read your four new test files (45 tests), the whole diff of phylo.py, config.py, accounting.py, fairfight.py, fingerprint.py, run_rv2b.py, scaling_check.py, PREREG.md and the bars, and re-ran the suite: `python -m pytest tests/growmoe -q` gives 520 passed in 197 s (475 + your 45). Verdict: APPROVE, with ONE required correction.

THE CORRECTION. scaling_check.json was written with "k": 4 (bars_rv2b.json did not exist yet when you ran `scaling`, so the script used its default). `prereg` then froze k = 1. So C4, and the shipped gate_shortlist_min_experts = 32 (derived from the k=4 timing table), describe a cascade we are not going to ship. Regenerate them at k = 1 before any confirmatory seed runs. Generated files are regenerated, never hand-edited.

STEP A - one-time, recorded unfreeze (authorised by Claude through the mediator; no confirmatory seed has run).
1. The machine must be quiet (the scaling step is wall-clock timed and C4 is gated on it). Claude is doing light work only. If CPU is above 30% when you start, wait.
2. Print the sha256 of PREREG.md, bars_rv2b.json, scaling_check.json and dev_k.json (your "old hashes"). Claude computed their first 12 hex digits as: PREREG f80accb8667a, bars af40e922da3b, scaling 134afe666f6b, dev_k 34f45aa20354. STOP IF any differs.
3. Create rv2b/_superseded_k4/ and COPY scaling_check.json into it.
4. While bars_rv2b.json still exists (so the script reads k = 1), run `python research/growmoe_results/rv2b/run_rv2b.py scaling`. STOP IF the new scaling_check.json does not say "k": 1.
5. MOVE (do not delete) bars_rv2b.json, bars_rv2b.sha256 and PREREG.md into _superseded_k4/.
6. Run `python research/growmoe_results/rv2b/run_rv2b.py prereg` (it reads the unchanged dev_k.json and the new scaling_check.json and writes the new bars, their .sha256 and PREREG.md).
7. Write rv2b/AMENDMENT.md by hand: date and time; what changed (scaling at k=4 -> k=1; bars and PREREG regenerated); why; old and new sha256 of the four files; old and new shipped min_experts; the C4 ratios at 500 experts before and after; the sentence "no file under runs/ existed when this was done".
STOP IF, after step 6, any "Expected" in the new PREREG.md is not PASS, or bars_rv2b.json has k != 1. A different min_experts is fine: report it.

STEP B - suite. `python -m pytest tests/growmoe -q`. Expected: 520 passed. STOP IF it differs.

STEP C - RV2-7. Check RAM is below 85% first; nothing else heavy of yours is running. Then
`python research/growmoe_results/rv2b/run_rv2b.py run --seeds 40-59` (default 2 workers x 3 threads) and then
`python research/growmoe_results/rv2b/run_rv2b.py report`.
Expect 120 files in runs/. Never look at partial results to change anything; never re-run a seed after changing anything. A task that fails for an environmental reason may be re-run once, unchanged, with --force; say so in the report. STOP IF freeze_check complains (that is its job).

REPORT (paste the generated text, do not summarise):
1. Old and new hashes, and the full AMENDMENT.md.
2. The new PREREG.md statements table and bars_rv2b.json.
3. The last line of the suite.
4. From RESULTS_RV2B.md: the verdict table and the Decision line VERBATIM, the per-stream table, the paired-difference lines, the scaling table.
5. Number of files in runs/ (expect 120), total minutes, the highest peak memory of any task, any task that errored or was re-run.
6. The provenance line of RESULTS_RV2B.md (the hashes) and the LastWriteTime of PREREG.md, of the first and of the last file in runs/.
7. For Claude's review of RV2-5, which was never reported: one line per new test file saying how you saw it fail before the code existed (or "not done"), and the list of shared-contract changes you made.
8. Anything that surprised you.

Then STOP. Do not branch on the decision line yourself. Do not start RV2-8.
```

## What Claude checks when the report comes back (audit recipe, decision D-14)
1. `PREREG.md` is older than the first file in `runs/`; the old bars, PREREG and scaling are kept in `_superseded_k4/`; `AMENDMENT.md` exists and its hashes match the files.
2. Recompute the sha256 of every file named in the provenance line of `RESULTS_RV2B.md`; they must equal the report.
3. `run_rv2b.py report` regenerates `RESULTS_RV2B.md` identically.
4. `python -m pytest tests/growmoe -q` gives 520 passed; 120 files in `runs/`.
5. The verdict table and the decision line are quoted verbatim; "AGAINST expectation" rows are explained, not argued away.
6. Only then the RV2-8 adoption message (flip `gate_shortlist`, pins for Phase 1, M0, M2-A and RV2-A entry points); documentation is written by Claude.

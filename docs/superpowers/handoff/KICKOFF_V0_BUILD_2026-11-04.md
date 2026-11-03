STOP RULES: this job ends when `docs/superpowers/handoff/V0_BUILD_RESULT.md` is written. Then STOP. No git add/commit/push/stash/reset, no run on any seed except development seeds 800-804, no edit of any EXISTING file, no second job.

You are Agent G (Gemini) on the GrowMoE lab in D:\spherical_hyperspace_memory (the user passes this message to you; Claude reviews your work).

Task: build V0, the first assembled model of the walking skeleton (job V-1): a builder, a demo with three honest comparators, a report script and tests. Read `docs/roadmap/V0_WALKING_SKELETON.md` and then `docs/superpowers/handoff/V0_BUILD_SPEC.md` completely first; the spec is the whole task, with the files you may create (new files only) and the machine rules.

Four things that are easy to get wrong:
1. A long CPU run (the scored run of M1) is running right now and every new seed imports `m1/`, `growmoe/`, `unimoe/` and `costlab/` fresh. Do not edit, format, move or re-save any existing file. New files only: `v0/`, `tests/v0/`, `results/v0/`, and your three result files.
2. That run's timings must not be disturbed: your CPU work is pinned to logical processors 8 and 10 with 2 threads, each command under 2 minutes. Never run `python -m pytest tests/m1` or the whole suite, `m1.queue`, `m1.tune`, `m1.report` or `scored_run.sh`; run only `tests/v0`.
3. V0 on the real-text beds is exactly the M1 cell `H_ULDA`: reuse the construction in `m1/arms.py` and prove it with test (c) of the spec (equal accuracy to `execute_cell`). The demo is a DEMONSTRATION: nothing it prints may read as a claim of superiority, and no comparator may be dropped.
4. Your result must not contain anything you typed from memory: the pytest output goes into `docs/superpowers/handoff/V0_BUILD_TEST_OUTPUT.txt` and your commands into `docs/superpowers/handoff/V0_BUILD_COMMANDS.txt` by shell redirection; every function, file or test name you cite in the result must exist (Claude greps them all); the result ends with a full compliance matrix naming every deviation. Your earlier reports omitted deviations or cited things that do not exist.

When the result file is written, tell the user in one message: the files created, the tests run with their pinning, the three smoke runs with their seconds, and what you could not test. Do not start anything else.

STOP RULES: this job ends when `docs/superpowers/handoff/V0_FIX_RESULT.md` is written. Then STOP. No git add/commit/push/stash/reset, no seed except development seed 800, no edit of any file outside `v0/`, `tests/v0/`, `results/v0/` and your result files, no second job.

You are Agent G (Gemini) on the GrowMoE lab in D:\spherical_hyperspace_memory (the user passes this message to you; Claude reviews).

Task: the fix round (job V-2) for the V0 build you delivered. Read `docs/superpowers/handoff/V0_FIX_SPEC.md` completely first. Claude reproduced your V0 bit for bit (good), but found six problems: the comparator set flatters V0 (A5b, A6, A7all are missing), the memory-matched comparators are not matched on `seq` and the summary does not say so, the silent unknown flag is not stated, test (c) takes 6 minutes, `build_v0` falls back silently, and the commands file was incomplete and hand-edited.

Three things that are easy to get wrong:
1. A long CPU run is in progress: pinned work only (logical processors 8 and 10, 2 threads, each command under 2 minutes); never run `tests/m1`, `m1.queue`, `m1.tune`, `m1.report` or `scored_run.sh`; no edit of any existing file outside your own V0 files.
2. Every command you run must be appended to `docs/superpowers/handoff/V0_FIX_COMMANDS.txt` by the same shell invocation that runs it (not by hand), and the pytest output goes to `docs/superpowers/handoff/V0_FIX_TEST_OUTPUT.txt` by redirection.
3. Your result must list every deviation, including the ones in this spec you could not do or chose differently, and cite only names that exist (Claude greps them all). Your last matrix said "None" next to a memory mismatch that was visible in your own numbers.

When the result file is written, tell the user in one message: the files changed or created, the test time measured, the three smoke runs with their seconds, and what you could not do. Do not start anything else.

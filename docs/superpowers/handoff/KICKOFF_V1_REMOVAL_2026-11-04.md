STOP RULES: this job ends when `docs/superpowers/handoff/V1_REMOVAL_RESULT.md` and `docs/superpowers/handoff/P2_REMOVAL_AUDIT.md` are written. Then STOP. No git add/commit/push/stash/reset, no seed except development seeds 800-802, no edit of any file outside `v1/`, `tests/v1/`, `results/v1/` and your four handoff files, no second job.

You are Agent G (Gemini) on the GrowMoE lab in D:\spherical_hyperspace_memory (the user passes this message to you; Claude reviews).

Task: job V-3, probe P2 "is removal exact?" and version V1 of the walking skeleton. Read `docs/superpowers/handoff/V1_REMOVAL_SPEC.md` completely first. You measure, for every component that can have seen a removed class (closed-form heads and every structure of the V0 pool), whether removal is exact (float64, difference at most 1e-6, identical predictions), approximate, or impossible without a code change, and you build a removal operation with a certificate.

Four things that are easy to get wrong:
1. A long CPU run is in progress: pinned work only (logical processors 8 and 10, 2 threads, each command under 2 minutes, one JSON per experiment); never `tests/m1`, `m1.queue`, `m1.tune`, `m1.report` or `scored_run.sh`; never edit an existing file or change an existing class except by instance-level wrappers inside your own files.
2. This is a measurement, not a demonstration: do not expect exactness, do not assume it, report the measured differences, including the ones that go against the idea. Pool training depends on order and seed, so a retained expert may not match the never-trained pool's expert; say how many could be matched.
3. Every number in the audit table comes from a JSON under `results/v1/`; none typed. Every command is appended to `docs/superpowers/handoff/V1_REMOVAL_COMMANDS.txt` by the same shell invocation that runs it, and raw pytest output goes to `docs/superpowers/handoff/V1_REMOVAL_TEST_OUTPUT.txt` by redirection.
4. Your result file lists every deviation (including runtimes over the limits) and cites only names that exist; Claude greps them all and reruns your experiments in separate processes.

When both files are written, tell the user in one message: the files created, the tests with their time, the experiments run with their seconds, the headline of the audit (what is exact, what is not) and what you could not do. Do not start anything else.

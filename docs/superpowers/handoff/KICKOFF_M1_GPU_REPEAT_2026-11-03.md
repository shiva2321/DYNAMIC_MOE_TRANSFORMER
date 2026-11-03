STOP RULES: this job ends when `docs/superpowers/handoff/M1_GPU_REPEAT_RESULT.md` is written. Then STOP. No git add/commit/push/stash/reset, no run on any seed except development seed 800, no edit of any EXISTING file, no second job.

You are Agent G (Gemini) on the GrowMoE lab in D:\spherical_hyperspace_memory (the user passes this message to you; Claude reviews your work).

Task: build the GPU repeat of the wide baselines of milestone M1 (job G-1). Read `docs/superpowers/handoff/M1_GPU_REPEAT_SPEC.md` completely first; it is the whole task, with the files you may create (new files only), the machine rules and the result-file format.

Three things that are easy to get wrong:
1. A long CPU run (the scored run of M1) is running right now and every new seed imports `m1/`, `growmoe/` and `costlab/` fresh. Do not edit, format, move or re-save any existing file. New files only: `m1/gpu_repeat.py`, `m1/gpu_report.py`, `m1/gpu_run.sh`, `tests/m1/test_gpu_repeat.py`, outputs under `results/m1/smoke/gpu/`, and your result file.
2. That run's timings must not be disturbed. Your CPU work is pinned to logical processors 8 and 10 with 2 threads, each command under 30 s (the smoke runs of section 2.8 under 2 minutes). Never run `python -m pytest tests/m1` or the whole suite, `m1.queue`, `m1.tune`, `m1.report` or `scored_run.sh`; run only `tests/m1/test_gpu_repeat.py`.
3. Your result file ends with a full compliance matrix (one row per numbered item of sections 2.1-2.8, every deviation named). Your earlier reports omitted deviations; Claude reruns everything and reads the diff, not the matrix.

When the result file is written, tell the user in one message: files created, tests run with their pinning, smoke timings, and what you could not test. Do not start anything else.

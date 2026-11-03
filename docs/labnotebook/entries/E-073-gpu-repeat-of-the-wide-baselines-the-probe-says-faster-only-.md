# E-073 - GPU repeat of the wide baselines: the probe says faster only for wide nets, build ordered

- **ID:** E-073
- **Date:** 2026-11-03
- **Track:** W
- **Type:** experiment
- **Status:** in progress (probe done, build job G-1 written and waiting for the user to hand it to Gemini; the GPU pass itself runs after the CPU timed pass)
- **Evidence grade:** B for the speed probe (script-generated, one 8-second run on one machine, taken while the timed pass ran on other cores); the repeat itself is graded when its report exists
- **Code / commit:** `m1/diag_gpu_probe.py` (output `results/m1/diag/gpu_probe_2026-11-03.json`); the commit that adds them
- **People:** the user answered Q-83 ("build it if its faster and from now on i will pass the messages and prompts to the gemini in antigravity"); Claude ran the probe and wrote the spec and kickoff; Gemini builds (Antigravity, relayed by the user); Claude reviews

## Question
Is the GPU actually faster for the wide dense baselines of M1 (A3, A3-2, A4, A4-2, A6, A8), so that the GPU repeat of D-89 is worth building, and how much of the scored run would it save? A wrong answer costs either a useless build or a missing answer to the obvious reviewer question "a dense network belongs on a GPU".

## Why we asked it
E-072 left Q-83 open (the GPU repeat of the wide networks is not built, every scored cell runs on the CPU). The user's condition for building it was "if it's faster".

## Prediction (written BEFORE the numbers, if there was one)
None written beforehand. From the floating-point work per step I expected the GPU to win only for the widest networks and to lose for narrow ones (launch-bound), and the saving on the scored run to be small because the long cells are pools.

## Method (how)
`m1/diag_gpu_probe.py` times one optimiser step of `growmoe.experts.MLPExpert` (the network of `ReplayNet`) at the real shapes: d 128, 64 new plus 64 replay samples, 64 classes, Adam lr 3e-3, fp32, `train_step` of the repo (it reads the loss back every step, so the GPU timing includes that synchronisation). Hidden widths 256, 1024, 4096, 16384; CPU with 2 threads pinned to logical processors 8 and 10 (`costlab.env.fixed_threads`), GPU with the same two logical processors issuing the kernels. For A6 (random-feature ridge) one batch of the float64 accumulation `G += phi^T phi`, `B += phi^T Y` and one Cholesky solve at D 1024 and 4096. Run once, 2026-11-03 08:30:13 to 08:30:21, while the timed pass of the scored run ran on logical processors 0 and 2.
The share of the scored run that the wide arms take is read from the seed-810 cells already finished (`results/m1/scored/*.json`, field `update_cost.wall_clock_total`).

## Results (what came out)
**One optimiser step, from `results/m1/diag/gpu_probe_2026-11-03.json`:**

| hidden | CPU ms | GPU ms | speedup |
|---|---|---|---|
| 256 | 1.49 | 1.76 | 0.84 (GPU slower) |
| 1024 | 2.30 | 1.67 | 1.4 |
| 4096 | 9.33 | 1.89 | 4.9 |
| 16384 | 44.3 | 3.27 | 13.6 |

**A6:** D = 4096 accumulate 38.9 ms CPU against 16.8 ms GPU (2.3x), solve 413 against 207 ms (2.0x); D = 1024 accumulate 2.7 against 1.2 ms (2.2x) but solve 14.9 against 35.6 ms (GPU 2.4x slower). Peak VRAM allocated 534 MB; at the probe start 11,253 MB of 12,287 MB were free.
**Share of the scored run (seed 810 cells, CPU, measured):** A8 on the three B3 schedules 120.3 + 116.5 + 141.7 s, A6 on them 26.9 + 26.2 + 31.5 s, A3-2 19.4 + 19.1 + 22.5 s; A3 on B1 N = 10 at width 16384: 91.5 s. A3 on B1 at N = 30 and N = 100 is not finished yet; extrapolated linearly in N (an assumption, supported by 215-267 s per seed at N = 30 measured on the development seeds in E-072) it is about 275 s and about 900 s. Together the wide arms are roughly 30 minutes of the about 5.9 hours per timed seed, about 8 %, and that is an estimate until the cells exist.

## Interpretation (what we understood, and on what basis)
Plain words: the graphics card is faster only when the network is wide. A network 16,384 units wide trains about 14 times faster on it, one 4,096 wide about 5 times, one 1,024 wide barely, and a 256-unit one is slower there (the card spends its time starting jobs, not computing). So the GPU repeat matters exactly for the baselines that were tuned wide: A3 on B1 (16384), A8 and A6 on the B3 schedules (8192). It does not shorten the scored run (about 8 % of it), and it is not meant to: its job is the fairness check. If the dense baseline on a GPU turns out cheaper in time than the pool on the CPU, rule 6.1 would read differently, and the report must say so rather than leave the question to a reviewer.
Measured: the table above. Estimated: the 8 % share and the A3 times at N = 30 and 100. Not measured: A8 at 8192 on the GPU (only 4096 and 16384 were probed), the whole-cell speedup (replay sampling, evaluation and the host-to-device copies add time that the probe leaves out).

## Caveats and what went wrong
- One run of 8 seconds; the CPU side was measured while another job ran on other cores, so the CPU numbers may be a few percent off. A rerun alone on the machine (`python -m m1.diag_gpu_probe`, writes `gpu_probe_rerun.json`) is planned after the CPU timed pass.
- The probe ran inside the isolated timed pass (cores 8 and 10, 8 seconds, window above). The timed cells that ran at 08:30 are the ones to look at if a timing outlier shows up there.
- Other programs use the GPU and the load varies: `nvidia-smi` two minutes before the probe showed 4,180 MiB used and 19 % utilisation, the probe itself saw 1,034 MiB used. That adds noise to GPU timings; the GPU pass will record the NVML state of every cell.
- The repeat uses the frozen operating points, not re-tuned for the GPU (a GPU could afford wider networks); stated limit.
- The GPU is not bit-identical to the CPU (floating-point order); the repeat reports accuracy next to the CPU accuracy for every cell.

## Consequence
- D-93: build the GPU repeat as a separate pass (spec `docs/superpowers/handoff/M1_GPU_REPEAT_SPEC.md`, kickoff `docs/superpowers/handoff/KICKOFF_M1_GPU_REPEAT_2026-11-03.md`), new files only, small pinned tests while the timed pass runs; the real GPU pass runs after the CPU timed pass and before the deterministic pass. Q-83 is answered.
- From now on the user hands Gemini's kickoff messages to Antigravity themself; Claude no longer runs `agy --print`.
- Next: user hands the kickoff to Gemini; Claude reviews the result; E-074 will hold the review and, later, the GPU pass.

## Evidence
- `results/m1/diag/gpu_probe_2026-11-03.json` - the probe output
- `m1/diag_gpu_probe.py` - the probe script (the scratchpad copy that produced the output differs only in the repo path and the default output name)
- `docs/superpowers/handoff/M1_GPU_REPEAT_SPEC.md` - the build spec (job G-1)
- `docs/superpowers/handoff/KICKOFF_M1_GPU_REPEAT_2026-11-03.md` - the message for Gemini
- `results/m1/scored/` - the seed-810 cells the shares are read from

## Reproduce
```
python -m m1.diag_gpu_probe results/m1/diag/gpu_probe_rerun.json
```

# COSTLAB: the cost meter (road plan M0 step 4). Task spec for Gemini

**STOP RULE (also the first line of the kickoff message).** When `docs/superpowers/handoff/COSTLAB_RESULT.md` is written, STOP. Do not start M1, do not run any experiment on the pool, do not commit or push.

## 1. Why

Every efficiency claim of this project (finish-line contract F7) has to come from one instrument whose own accuracy is tested. `costlab` measures what a model or a procedure costs; it never judges what is good. Context: `docs/roadmap/ROADMAP.md` (F7, section 5.5) and `docs/roadmap/MILESTONES.md` (M0 step 4 and its proposed bars). You need nothing else of the plan.

## 2. Environment and traps

- Windows 11, PowerShell, Python 3.12.10, torch 2.6.0+cu124, RTX 3060 12 GB, Ryzen 5 7600 (6 cores, 12 logical processors). numpy, psutil, pynvml and pytest are installed. torch prints a harmless deprecation warning about pynvml.
- The GPU is **not idle**: other programs use about 1.6 GB and some load. Measure the state before every measurement, never assume.
- Another job may use logical processors 6-11. Pin timing work to logical processors 0-3 (`psutil.Process().cpu_affinity([0, 1, 2, 3])`) and use 2 torch threads.
- Console is cp1252: print ASCII only. Use `pathlib`, no absolute paths, no network, no `pip install`.
- Memory-lean: one GPU process, tensors under 1 GB of VRAM and 2 GB of RAM.
- Do not use shell heredocs for Python that contains backslashes; write files with your file tools.

## 3. Files you may touch

Create or change only: `costlab/` (the package), `tests/costlab/`, `docs/superpowers/handoff/COSTLAB_RESULT.md`. Everything else is read-only. Never run `git add`, `commit`, `push`, `stash` or `reset`; never use `--no-verify`.

## 4. The package (import name `costlab`)

All functions return plain dicts or small dataclasses, and every number in a result comes from a measurement, never from a constant.

1. `costlab.env`
   - `environment() -> dict`: python, torch, CUDA, GPU name and driver, CPU name, logical cores, RAM, torch intra- and inter-op threads, the OMP/MKL/OPENBLAS/NUMEXPR environment variables (or "unset"), git commit if available else None, ISO timestamp.
   - `fixed_threads(n, cpus=None)`: context manager that sets the torch threads and, if `cpus` is given, the process affinity; restores both on exit; yields the settings dict.
2. `costlab.count`
   - `count_macs(fn, *args, **kw) -> MacCount(macs, flops, by_op)` using `torch.utils.flop_counter.FlopCounterMode`; macs = flops / 2. State in the docstring exactly which operations are counted (matrix products and convolutions) and which are not (element-wise work, numpy).
   - `Counter`: manual counter for non-torch work: `add(name, macs)`, `total()`, `by_name()`. Helpers `macs_matmul(m, k, n)`.
3. `costlab.bytes`
   - `total_param_bytes(model)`; `active_param_bytes(model, fn, *args)`: run `fn` with forward hooks on the leaf modules and sum the bytes of parameters and buffers of the leaf modules that executed at least once, each tensor counted once (by `data_ptr`); return the sum and a per-module breakdown.
4. `costlab.timing`
   - `time_callable(fn, *, warmup=5, repeats=10, device="cpu", min_duration_s=0.0) -> TimingResult`: `perf_counter`; `torch.cuda.synchronize()` before and after when `device="cuda"`; fields: samples (seconds), median, mean, std, cv (= std / mean), p95, cpu_seconds_per_call (`time.process_time`).
   - `latency_throughput(make_fn, batch_sizes=(1, 32), **kw)`: `make_fn(batch) -> zero-argument callable`; per batch size the latency stats and items per second.
5. `costlab.memory`
   - `peak_memory(fn, device="cpu") -> dict`: peak process RSS from a sampler thread (10 ms) and, for cuda, `torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`.
6. `costlab.energy`
   - `gpu_state() -> dict`: utilization, memory used, power now (NVML).
   - `idle_baseline(seconds=10) -> dict`: mean watts at rest.
   - `measure_energy(fn, min_duration_s=3.0) -> dict`: repeat `fn` until the minimum duration; joules from the NVML total-energy counter difference, joules from integrating power samples taken every 50 ms (cross-check), mean watts, duration, calls, joules per call, plus `gpu_state()` before and after.
   - If NVML is unavailable raise `EnergyUnavailable` with the reason. Never return zero or an estimate. CPU energy is not measured on this machine; `cpu_energy_note()` returns that sentence.
7. `costlab.report`: `measure(make_fn, *, model=None, batch_sizes=(1, 32), device="cpu") -> dict` combining the above; `to_json(result, path)`; `to_markdown(result) -> str`. The JSON has `schema_version`, `environment`, `measurements` and the raw samples.
8. `costlab.validate`: `python -m costlab.validate [--out costlab_validation.json]` runs the validations of section 5, prints PASS or FAIL per bar with the measured numbers, writes the JSON, exits 1 if any bar fails. It never edits a tolerance.

## 5. Validation (the bars; tolerances are fixed, do not change them)

- **V1 MACs within 5 % of the analytic count**, with the analytic formula written in the test from the shapes, independent of the code under test: (a) MLP 64-256-256-10, batch 32: B*(64*256 + 256*256 + 256*10); (b) top-1 routed toy mixture of 8 experts (router Linear 64 to 8, experts Linear 64 to 64), batch 32, executed as grouped matrix products so that only the executed products exist: B*64*8 + B*64*64; (c) single-head attention written with explicit matrix products: 4*B*T*d*d + 2*B*T*T*d. Plus a unit test of `Counter`.
- **V2 timing**: coefficient of variation at most 5 % over 10 repeats at fixed threads and affinity, for a CPU workload of at least 20 ms per call and for a GPU workload of at least 20 ms per call, after warm-up. If a bar fails, retry at most twice (record every attempt, wait while the selected cores are busy: `psutil.cpu_percent` over 2 s above 30 %). Never loosen the bar; report the best value found and the settings tried.
- **V3 GPU energy**: (i) loaded mean power above the idle baseline; (ii) running the workload for twice as many calls gives joules within 25 % of twice the first value; (iii) the counter-based and the integrated joules agree within 15 %. If the GPU utilization is above 25 % at the start, wait up to 60 s and retry.
- **V4 bytes**: for a dense MLP, active bytes equal total parameter bytes exactly; for the routed toy with batch 1, active bytes equal router plus exactly the chosen expert, exactly; with all items forced to expert 3, router plus expert 3.
- **V5 memory**: allocating a known 256 MB torch tensor on the GPU shows a measured peak at least 256 MB and at most 1.1 times that plus the starting level; a 200 MB numpy allocation is seen by the RSS sampler within 20 %.

## 6. Tests

`tests/costlab/test_count.py`, `test_bytes.py`, `test_timing.py`, `test_memory.py`, `test_energy.py` (skip with an explicit reason when there is no CUDA or NVML), `test_report.py` (schema, JSON round trip). They check behaviour and the analytic equalities; the strict timing and energy bars live in `costlab.validate`, not in pytest, so the suite stays deterministic and under 60 s. Run: `python -m pytest tests/costlab -q`.

## 7. What is forbidden

Typed or hard-coded results; expected values copied from the implementation's own output; mocks that fake NVML or timings; `except` clauses that swallow errors; weakened tolerances; edits outside section 3; leftover debug prints. Code clarity beats cleverness: type hints, a docstring on every public function saying what the number means, under about 800 lines of package code, no dependencies beyond torch, numpy, psutil and pynvml.

## 8. Deliverable: `docs/superpowers/handoff/COSTLAB_RESULT.md`

(1) the list of files created, (2) the exact commands to run the tests and the validation, (3) the validation table copied from the output of `python -m costlab.validate` with the path of the JSON it wrote, (4) every deviation from this spec and why, (5) known limits (for example: matrix products and convolutions only; no element-wise work; no numpy; no CPU energy), (6) the line `STOP`.

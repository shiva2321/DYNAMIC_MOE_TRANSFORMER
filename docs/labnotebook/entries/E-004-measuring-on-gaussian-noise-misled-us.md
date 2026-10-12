# E-004 - A probe on Gaussian noise read the world model as broken; it was not (our own error)

- **ID:** E-004
- **Date:** 2026-09-21
- **Track:** I
- **Type:** process
- **Status:** completed
- **Evidence grade:** B
- **Code / commit:** `experiments/train_world_model_on_layer14.py` (`harvest_layer14_transitions`)
- **People:** Claude (made the error and found it)

## Question
How large is the causal residual of the world model, and did it regress?

## Why we asked it
Part of the debugging in E-003.

## Prediction
None.

## Method (how)
The same checkpoint, `causal_residual` measured on three different input distributions.

## Results (what came out)
- Real adapter-encoded Layer-14 latents: ||z|| = 7.75 (std 2.48), per-dimension std 1.20, all 32 dimensions active -> **0.392 rad**.
- `torch.randn(B, 32)` (||z|| 5.60) -> **1.374 rad**. REM-style latents (||z|| 1.00, dims 8:32 zero) are a third, different distribution.
- I reported the 1.374 figure as evidence that the world model had regressed. It had not.

## Interpretation
The adapter is a learned nonlinear map, so its image is a narrow manifold inside R^32; probes off that manifold measure untrained regions and read close to chance.

## Caveats and what went wrong
A measurement made on the wrong distribution nearly sent the whole investigation in the wrong direction. The same shape of mistake recurred twice more in this project (E-001: a score without its control; E-005: a metric reading the sampler).

## Consequence
Rule: cache real latents once and reuse them for every manifold metric; never quote a manifold number measured on Gaussian noise; always put a random baseline beside a geometry number (`DECISIONS.md` D-03).

## Evidence
- `experiments/train_world_model_on_layer14.py` - harvests the real transitions

## Reproduce
Harvest real Layer-14 transitions with `harvest_layer14_transitions`, evaluate the causal residual on them and on `torch.randn`.

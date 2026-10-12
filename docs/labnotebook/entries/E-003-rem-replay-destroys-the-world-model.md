# E-003 - REM replay destroyed the world model; root cause found by ablation and fixed

- **ID:** E-003
- **Date:** 2026-09-21
- **Track:** I
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B
- **Code / commit:** fix in `ishmn/grounded_hybrid_engine.py` (`_rem_optimizer_step`), guard `tests/test_rem_causal_preservation.py`; committed with `d675fc0`
- **People:** Claude

## Question
Why did the causal residual of the world model rise across the continuous self-improvement loop (0.305 -> 0.990 -> 1.748 rad over three epochs)?

## Why we asked it
It contradicted the claim that sleep consolidation improves the system.

## Prediction
None; diagnostic.

## Method (how)
Read the sleep-consolidation code, then ablate: 20 REM steps with the optimizer exactly as written vs 20 REM steps with `w_fwd` / `w_bwd` gradients removed, on 3,000 real Layer-14 latents from the same checkpoint.

## Results (what came out)
- The REM block stepped `self.uce.optimizer`, which includes `world_model.w_fwd` and `w_bwd`, under a loss of `geodesic_alignment + 0.2 * SIGReg` that has no reversibility term. REM also replayed synthetic latents (`z[:, :8] = u*r`, dims 8:32 exactly zero, ||z|| about 1.0) while real latents have ||z|| about 7.75 with all 32 dims active.
- As written: 0.392 -> 1.157 rad (+0.765). With `w_fwd` / `w_bwd` frozen: 0.392 -> 0.439 rad (+0.047), i.e. 94 % of the damage removed. After the fix on real data: drift +0.032 rad.
- Still open: the off-distribution initial state `z_init` (MoE, perceptor, projections are still consolidated on ||z|| about 1.0 latents with 24 of 32 dims zero).

## Interpretation
Sleep was writing a loss that could only be minimised by moving the transition model away from being reversible. The damage is invisible from either file alone; it appears only when the causal residual is measured on the real latent distribution before and after a sleep cycle.

## Caveats and what went wrong
One checkpoint. The first diagnosis was nearly derailed by a measurement error of mine (E-004).

## Consequence
Rule recorded: any new gradient path touching `uce.optimizer` must carry a reversibility term or exclude the transition pair. Guarded by three tests.

## Evidence
- `ishmn/grounded_hybrid_engine.py` - the fix
- `tests/test_rem_causal_preservation.py` - the guard (3 tests)

## Reproduce
`python -m pytest tests/test_rem_causal_preservation.py -q`

# E-006 - VSA / geometry research for a general-intelligence system, with numerical checks

- **ID:** E-006
- **Date:** 2026-10-03
- **Track:** I
- **Type:** literature
- **Status:** completed
- **Evidence grade:** B (numerics, seeded and reproducible); L (literature)
- **Code / commit:** `research/vsa_numerics.py`, `research/geometry_checks.py` (uncommitted at the time of writing; other-track files)
- **People:** Claude; the user asked for an exhaustive history of VSA / neural-network / AGI work and an algorithm for emergent general intelligence

## Question
What do vector-symbolic architectures and geometry offer a general-intelligence system, and which claims about the ISHMN geometry are consistent with the mathematics?

## Why we asked it
After E-001 and E-002 we needed to know which parts of the ISHMN geometry were sound before building on them.

## Prediction
None; research and audit.

## Method (how)
A report with every claim labelled measured / verified-at-abstract-level / from memory / hypothesis; an algorithm and a falsifiable protocol (APA, with a baseline and a stop rule); a bibliography of about 340 references tagged V / V-t / R / W; two seeded scripts that regenerate the logged numbers exactly.

## Results (what came out)
- 5,295 random unit vectors: minimum pair angle 5.2 degrees at D = 8, 80.1 degrees at D = 1024. A D = 8 sphere cannot host VSA quasi-orthogonality; cleanup fails at 40 degrees of noise. Whether this caused the lattice collapse (E-002 item 4) is untested.
- Bundle exact-recovery capacity about D / (7.5 ln N): D = 1024 -> 16 items, D = 4096 -> 64 (N = 5,295).
- `HyperspaceGlobalBus` (Hadamard, unit-L2 keys) is real but is built, placed in the optimiser, and never run in any forward path.
- `cognitive_map.py:82` seeds atoms with Python `hash()` (differs per process) and reseeds the global torch RNG.
- `geodesic_distance_sphere(x, x)` = 4.47e-4 rad (clamp 1 - 1e-7); the README badge of 4.1e-7 rad cannot come from it.
- The SPEC statement "FHRR = T^8 in S^7" is dimensionally impossible (T^4 fits in R^8 = C^4).
- Parallel transport: holonomy equals enclosed area; group binding is exact; one-shot analogy needs a small, correct hypothesis class.

## Interpretation
Do not claim VSA properties for the D = 8 sphere; build any VSA component at D >= 1024 with unitary binding. The first experiment (Phase 0: group tasks Z_113, S_5; APA-lite vs a weight-decayed baseline, >= 10 seeds, stop if no data-efficiency gain) was defined but not run.

## Caveats and what went wrong
The literature layer is mostly abstract-level (`L`); only the numerics are `B`. Phase 0 of the protocol has not been executed.

## Consequence
Informed the geometry atlas (E-008) and the decision to treat the geometry components as individually verified candidates rather than as one system.

## Evidence
- `research/VSA_GEOMETRY_TO_INTELLIGENCE.md`, `research/ALGORITHM_AND_PROTOCOL.md`, `research/BIBLIOGRAPHY_CHRONOLOGICAL.md`
- `research/results_vsa_numerics.log`, `research/results_geometry_checks.log`

## Reproduce
`python research/vsa_numerics.py` and `python research/geometry_checks.py` (both seeded; the logs above are their output).

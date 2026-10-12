# E-009 - Design of the self-growing MoE: three architectures and their pre-registered predictions

- **ID:** E-009
- **Date:** 2026-10-04
- **Track:** G
- **Type:** design
- **Status:** completed (implemented in E-010 / E-011)
- **Evidence grade:** H (design); L (literature)
- **Code / commit:** documents only: `research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md`, `docs/superpowers/specs/2026-10-04-growmoe-design.md`, plan `docs/superpowers/plans/2026-10-04-growmoe.md`
- **People:** Claude (research, design); user (approved the design and the Gemini hand-off)

## Question
Can a mixture of experts that **grows by itself** - creating a new expert when it meets something genuinely new, reusing an old one when it does not, and never forgetting - be built with statistical control over when it grows?

## Why we asked it
The user's bet: dynamic MoE is the right structure for continual learning in a general-intelligence system (spawn, use, train, improve, consolidate, route).

## Prediction (written before any code)
Ten single-pool predictions P1-P10 (research document section 7.5) and six comparative ones C1-C6 (spec section 10), each with a failure condition, e.g. P1 false-novelty rate <= 2q; P2 spawn recall >= 0.90; C1 architecture A has the lowest false-novelty rate and the highest regime recall; C3 architecture C ends stream S2 with <= 40 % of A's parameters.

## Method (how)
Literature synthesis and an idea bank, then three candidate architectures: **A, PhyloMoE** (non-parametric von Mises-Fisher kernel-density signature per expert, conformal p-values with Benjamini-Hochberg control, probation buffer and evidence test, interference probe that chooses "extend" vs "spawn", shrink-perturb inheritance from the parent, infant protection, competence monitor, growth policy, distillation merge); **B, BNP-MoE** (Bayesian non-parametric with a VAE likelihood gate); **C, KeyedDeltaMoE** (key-based gate with low-rank delta experts). Streams S1-S7 (types, permuted, drift, outliers, long-horizon, remap), local data only, ground truth hidden from the algorithms.

## Results (what came out)
Nothing was run in this step. The design fixed: numeric grading only, no magic thresholds (every threshold is an error rate or a tolerance with units), dev seeds >= 100 for tuning and test seeds 0-9, >= 10 seeds with paired intervals, deterministic seeding.

## Interpretation
A was recommended as the main design because its gate gives distribution-free control of false novelty. Closest published relatives identified later (E-012): MaRS (statistical growth test with guarantees), SEMA, LLaVA-CMoE; statistically controlled growth is therefore not unique to us.

## Caveats and what went wrong
The design was written before any measurement; E-010 immediately changed two of its decisions.

## Consequence
Hand-off to Gemini as rounds R1-R10 (`docs/superpowers/handoff/GROWMOE_HANDOFF.md`); Phase 0 spikes first (E-010).

## Evidence
- `research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md`, `docs/superpowers/specs/2026-10-04-growmoe-design.md`, `docs/superpowers/plans/2026-10-04-growmoe.md`

## Reproduce
n/a (design).

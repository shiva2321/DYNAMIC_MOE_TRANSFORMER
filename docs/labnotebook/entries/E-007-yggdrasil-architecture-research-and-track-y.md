# E-007 - Yggdrasil architecture research and Track Y: what geometry inside attention did and did not do

- **ID:** E-007
- **Date:** 2026-10-05 (research) to 2026-10-06 (track closed)
- **Track:** Y
- **Type:** experiment
- **Status:** completed (track closed)
- **Evidence grade:** B
- **Code / commit:** `geometry/`, `yggformer/`, `tests/yggformer/`, `tests/test_lorentz_attention.py`, `tests/test_snapped_state.py` (uncommitted as of 2026-10-12, other-track files)
- **People:** Claude (research, design, review); Gemini (implementer, rounds Y0-Yd); user (mediator)

## Question
Would a neuron / attention / transformer built on the Yggdrasil geometry (hyperbolic hierarchy + torus phase + radial salience) beat an ordinary Euclidean one, and is any part of it worth keeping?

## Why we asked it
The user wanted the older "Yggdrasil" geometry turned into something testable. Prior work (E-002, E-006) had shown the geometry could not be taken on trust.

## Prediction
Design decision with a **kill rule** written first: "Ygg-heads" (geometry inside the attention heads of an ordinary Euclidean-stream transformer); if there is no gain at head_dim <= 16 over a parameter-matched Euclidean transformer, drop hyperbolic attention and keep geometry only for routing / memory.

## Method (how)
Small controlled experiments with per-arm learning-rate grids chosen on dev seeds, test seeds 0-9, mean +/- 95 % CI, Bonferroni where several arms were compared; a gated review after every implementer round (rounds Y0-Y8, Yb, Yc, Yd; reviews in the review file).

## Results (what came out)
- Tree embedding (1,093 nodes): hyperbolic >> Euclidean at d = 4-16 at equal tuned budget (a capacity effect at d = 4-8: 2,000 steps MAP 1.00 vs 0.36 / 0.59); equal at d = 32; a hyperbolic x torus product is no better than pure hyperbolic.
- State tracking: a learned torus angle set drifts (Z6 word problem 100 % at length 16 -> 9 % at 256); a lattice-snapped torus does not drift but an abelian state cannot do S3; SO(k) / Householder fibres learn S3 in distribution but drift (44-60 % at 256); **a GRU is exact to 256**. Snapped rotation state: exact on Z6 and S3 to length 256 (10/10 seeds), at chance on S5, **no advantage over a GRU** (card 3).
- Hyperbolic attention logit (hyperbolic + cos(phase) + salience) is one dot product on augmented vectors, exact to 1e-14 in float64; precision error about 2^-p R^2.
- **Language model, after controls: no hyperbolic arm beats Euclidean.** `euclid_temp` PPL 9.47 +/- 0.21 vs `hyp` 9.63 +/- 0.30; Bonferroni p > 0.2; in `hyp_phase_bias` the curvature collapsed to a flat 0.03. Hyperbolic heads are not sparser (top-16 mass 96.5-98.9 % vs 98.1 %). The kill rule fired (card 4: NOT SUPPORTED).
- Branch-and-bound tree retrieval: prunes to 2.2-4.5 % of keys at N = 4096 but is 36-60x slower than a fair dense GPU baseline (card 5: NOT USEFUL at N <= 4096).
- Fast pairwise-distance CUDA kernel: 2.6x over the engine kernel; the pre-registered 4x target was missed (card 6: PARTIAL).

## Interpretation
Hyperbolic geometry earns its place for storing a static tree in few dimensions, not inside language-model attention at this scale; exact state tracking is a job for gated or snapped recurrence, not for a particular geometry. The honest negatives are the result.

## Caveats and what went wrong
- My first tree-embedding experiment used one constant learning rate for all geometries and printed "hyperbolic stops winning at high d"; a per-arm tuned run reversed it (`FAILURES.md` F-04).
- Reviews caught: Y3 inconclusive (80 steps, all arms at chance at depth 5 / 7); Y5 measured on random vectors; Y6 positive but short with the learning rate at the grid edge and no `euclid_temp` control; Y7's speed-up headline was against a naive reference (the real figure 2.6x); the final report's table contradicted `poc_b_table`; the memo `YGGDRASIL_GROWMOE_INTEGRATION.md` cites an "86x" concentration that was measured on random vectors (treat as a hypothesis).
- Small synthetic experiments: not language-model-scale claims.

## Consequence
Track Y closed. Components registered in the charter with their real status (cards 1-6). Hyperbolic stays only for static hierarchies; the Z-track atlas (E-008) then looked at geometry everywhere else.

## Evidence
- `research/YGGDRASIL_ARCHITECTURE_RESEARCH.md` - report
- `research/yggdrasil_results/poc_b_table.txt`, `research/yggdrasil_results/s5_comparison_table.txt` - state-tracking tables
- `research/yggformer_results/e6_lm_results.json`, `research/yggformer_results/e5_results.json`, `research/yggformer_results/e7_results.json` - language model, retrieval, kernel
- `docs/superpowers/handoff/YGGDRASIL_ARCH_HANDOFF.md` - the rounds; `docs/superpowers/handoff/REVIEW_2026-10-05_R10_AND_Y1.md` - the reviews (sections C-F, H)
- `docs/COMPONENT_CARDS.md` - cards 1-6

## Reproduce
`python -m pytest tests/yggformer tests/test_lorentz_attention.py tests/test_snapped_state.py -q`; experiment scripts are in `research/yggdrasil_poc/` and `research/yggformer_results/`.

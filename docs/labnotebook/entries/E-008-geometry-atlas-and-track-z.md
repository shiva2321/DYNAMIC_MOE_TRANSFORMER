# E-008 - Geometry atlas and Track Z: per-component verification (one headline under review)

- **ID:** E-008
- **Date:** 2026-10-06 to 2026-10-12
- **Track:** Z
- **Type:** experiment
- **Status:** in progress (Z1 headline under review, see Caveats)
- **Evidence grade:** B (Z0, Z2 as reported); C (Z1 headline after our own re-check)
- **Code / commit:** `geomlab/`, `tests/geomlab/` (uncommitted other-track files)
- **People:** Claude (atlas, protocol, reviews); Gemini (rounds Z0-Z2); user

## Question
Where, across a general-intelligence system and not only in attention, does geometry actually help, and which candidates survive a pre-registered, controlled test?

## Why we asked it
E-007 closed the hyperbolic-attention idea. The user asked for web research on geometry improvements everywhere, then a verification plan.

## Prediction
Every round starts with a PREREG file; adoption rules are written before results (for Z1: token speedup >= 1.15x with the interval above 1; a later note raised the bar to 2.0x in the card).

## Method (how)
Atlas of subsystems with evidence tags (about 50 arXiv IDs checked at abstract level, `L`); Track Z rounds Z0 (protocol library and geometry-health diagnostics), Z1 (spherical residual stream, the nGPT family), Z2 (associative-memory geometry: recall per byte), later rounds Z3-Z7 planned. Rules: PREREG before results, per-arm learning-rate grid with no edge optimum, an isolating control, verbatim tables plus a validator, no hand edits.

## Results (what came out)
- Atlas thesis: the geometry that wins in 2026 is "operators and normalisations on the right manifold": unit-sphere stream and weights, orthogonal / rotation state transitions, cone or PSD features for associative memory, orthogonal-subspace continual learning, quasimetric heads, step control. Hyperbolic is not needed for hierarchy (consistent with E-007).
- **Z0** (protocol and diagnostics): first submission silently replaced the pre-registered 1-D TwoNN test by a 2-D one that actually fails (circle 1.38, segment 1.65 vs 0.8-1.2); root cause float32 matrix-multiplication cancellation in `cdist` (exact mode: 1.003). Z0b restored the test (circle 1.000, segment 0.998). Card 9.
- **Z1** (spherical stream), as reported in card 11: final validation loss 2.3265 (base) -> 1.9525 (nGPT), token speedup 2.10x (1.95-2.45), beats the isolating control `base_scaled` (2.0767); stream norm 63.97 -> 1.000; wall-clock parity only (133.8 vs 64.2 ms/step in eager PyTorch).
- **Z2** (memory geometry), card 12: KATA-PSD (self-dual cone features) highest recall per byte on multi-query associative recall (69.4 % at 16 pairs vs 10.2 % for the standard gated delta rule); the legacy ISHMN spherical vMF memory is competitive in discrete slots with 3.5x less state but 2.1x more compute. Not re-audited here.

## Interpretation
Z0 and Z2 are as reported. **Z1's headline is not established by its own data.** From the stored curves (`checkpoint_z1_bf16.json`): the baseline's validation loss reaches a mean minimum of 1.9031 at step 400 (mean of per-seed minima 1.8980) and then rises to 2.3265 at step 1000 (it overfits the 548k-character corpus), so the baseline's best loss is **lower than nGPT's final loss of 1.9525**. The token-speedup metric (`compute_speedup`) is defined as the tokens an arm needs to reach the *baseline's final* loss, which rewards an over-trained baseline; on 2026-10-09 an exact copy of the baseline scored 4.67x under it. nGPT's own curves were not stored, so whether nGPT would also beat an early-stopped baseline cannot be read from the file.

## Caveats and what went wrong
This is the same shape as the Phase-1 "beats replay" artifact (F-10): a comparison whose baseline was not given a fair budget or stopping rule. The baseline's configuration was chosen on dev seeds (`configs.base.dev_score` in `results.json`); how that dev score is defined (final loss at 1,000 steps or best loss) has not been checked and is part of Q-09. The amendment of 2026-10-09 (bf16, decision triad) did not address the metric. Card 11 still says VERIFIED; a review note is added to it (F-08).

## Consequence
Z1 is downgraded to PARTIAL pending a re-run with the baseline's best loss (or early stopping / more data) as the target and with all arms' curves stored (Q-09). Z3-Z7 not started. Nothing from Track Z is integrated (charter rule 7).

## Evidence
- `research/GEOMETRY_ATLAS_2026-10.md` - the atlas; `docs/superpowers/handoff/GEOMETRY_ATLAS_HANDOFF.md` - plan Z0-Z7
- `research/geomlab_results/z0/tables.txt`, `research/geomlab_results/z0/AMENDMENT.md`
- `research/geomlab_results/z1/PREREG.md`, `research/geomlab_results/z1/AMENDMENT.md`, `research/geomlab_results/z1/results.json`, `research/geomlab_results/z1/checkpoint_z1_bf16.json`, `research/geomlab_results/z1/tables.txt`
- `research/geomlab_results/z2/results.json`, `research/geomlab_results/z2/tables.txt`
- `geomlab/spherical_stream.py` - `compute_speedup`
- `docs/COMPONENT_CARDS.md` - cards 9, 11, 12

## Reproduce
`python -m pytest tests/geomlab -q`; the Z1 curve check: load `checkpoint_z1_bf16.json`, average `base_runs[*]["val_losses"]` over seeds, compare its minimum with `summary.ngpt.final_loss.mean` in `results.json`.

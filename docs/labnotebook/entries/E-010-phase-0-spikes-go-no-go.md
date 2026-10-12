# E-010 - Phase 0 spikes: which design choices survive a first measurement

- **ID:** E-010
- **Date:** 2026-10-04 to 2026-10-05
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B
- **Code / commit:** `research/spikes/spike_novelty_scores.py`, `spike_init_modes.py`, `spike_growth_trigger.py` (throwaway, labelled); raw numbers `research/growmoe_results/phase0/*.json`
- **People:** Claude (spikes, decisions); Gemini (later implementation)

## Question
Before building Phase 1: (1) is a vMF kernel-density signature a good novelty score; (2) does an affinity-scaled child initialisation beat starting from scratch; (3) does the capacity-growth trigger grow when it should and not otherwise?

## Why we asked it
The spec's three riskiest assumptions; each one cheap to test.

## Prediction
Written beforehand as go / no-go rules: (1) AUROC >= 0.95 on S1 and >= 0.85 on S2; (2) the affinity-scaled child not worse than scratch on unrelated regimes and better on related ones; (3) growth when the teacher is larger, not when data-limited.

## Method (how)
Spike 1: 5 seeds; spike 2: 10 seeds; spike 3: 3 seeds (the sample sizes are in `research/spikes/RESULTS.md`).

## Results (what came out)
| Spike | Result | Verdict |
|---|---|---|
| 1 novelty scores | conformal KDE AUROC S1 0.973 [0.967, 0.979], S2 0.999, S6 0.993 | GO |
| 2 child init | the spec's rho-redraw rule was worse than scratch on unrelated regimes (AULC -0.038, p < 0.001 at sigma 2) | NO-GO as specified, adjusted |
| 3 growth trigger | the original greedy policy never grew for the width-32 teacher | FAIL, fixed, then GO |

Design changes forced (all post hoc, to be re-validated): child init = `shrink_perturb(lam = signature.affinity(cluster))`; a growth policy of widen {1.5, 4, 16}x or deepen with the cheapest candidate passing every gate (gain over an equally trained control >= 2 % relative, paired-bootstrap p < alpha / n_candidates, and strictly better held-out loss than the expert before the probe); signature stays vMF-KDE (a wider kappa grid, 64 / 1024 prototypes, a Gaussian-Mahalanobis signature and a KDE-union-Mahalanobis gate gave no significant gain). Novelty AUROC is not uniform (stdlib code after pydoc text 0.859; a diffuse regime such as AR(2) gives a weak reference).

## Interpretation
The conformal KDE gate is a sound choice; the inheritance rule and the growth trigger needed fixing before they could be trusted.

## Caveats and what went wrong
Few seeds, post-hoc design changes; "to be re-validated in Phase 1" - and indeed the growth trigger later fired in only 3 of 70 benchmark runs (E-013), so its validation is still open (Q-07).

## Consequence
Phase 1 built with the adjusted rules (E-011).

## Evidence
- `research/spikes/RESULTS.md` - verdict table and the design changes
- `research/growmoe_results/phase0/spike1_novelty.json`, `research/growmoe_results/phase0/spike2_init_modes.json`, `research/growmoe_results/phase0/spike3_growth.json`, `research/growmoe_results/phase0/spike3b_growth.json`
- `research/growmoe_results/IMPLEMENTATION_LOG.md` - narrative and tables (sections T4b-T11)

## Reproduce
`python research/spikes/spike_novelty_scores.py` (and the two others) from the repository root.

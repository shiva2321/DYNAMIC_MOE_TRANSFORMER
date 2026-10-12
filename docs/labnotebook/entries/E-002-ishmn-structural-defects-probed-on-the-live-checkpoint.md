# E-002 - Six structural defects of ISHMN found by probing the live checkpoint

- **ID:** E-002
- **Date:** 2026-09-21
- **Track:** I
- **Type:** audit
- **Status:** completed (defects 1, 2, 4 fixed; 5 replaced; 3 and 6 recorded)
- **Evidence grade:** B
- **Code / commit:** probes run against `artifacts/unified_lifelong_brain.pt` on an RTX 3060; fixes in `ishmn/` before the commit of 2026-09-22 (`d675fc0`)
- **People:** Claude

## Question
Does the older ISHMN system actually do what its documentation says it does at the level of mechanisms, not benchmark scores?

## Why we asked it
E-001 showed the headline score was unreliable. Mechanism-level probes were the next cheapest way to find out what was real.

## Prediction
None.

## Method (how)
Load the live checkpoint and measure each mechanism directly on its own terms (round trips, halting statistics, sphere alignment, lattice geometry, spawn thresholds). The 106 existing tests (shape and smoke tests only) catch none of the items below.

## Results (what came out)
1. `causal_error` compared the candidate to `W_bwd(z_cand, c)`, an identity test on `W_bwd`, not a round trip: d(z, W_bwd(z, c)) = 1.008 rad vs true round trip 1.141 rad (chance pi/2 = 1.571).
2. The adaptive halting never halted: it needs consensus >= 0.70 and causal_error <= 0.35, measured 0.953 and 0.771; every step ran the maximum 6 iterations.
3. The manifold adapter was trained on `normalize(z[:, :8])` but concepts live at `normalize(to_sphere(v))`: cosine 0.0615 vs 0.9927, so its 0.4 * geodesic term was adversarial.
4. The semantic lattice had collapsed: ||E[u]|| = 0.9069, 5,295 concepts in a 24.9 degree cone of S^7 (random would be about 90 degrees).
5. Mixture-of-experts "mitosis" was unreachable: min over all 5,295 concepts of the maximum resonance was 1.237 vs spawn threshold 0.40.
6. Self-play had no truth channel (hypotheses from Qwen, labels from the defect-1 signal): a closed loop, the most plausible cause of 4.
- Fixed later the same day: 1 (causal residual 0.0229 rad on real latents), 2 (adaptive, mean 1.40 of 6 steps), 4 (||E[u]|| 0.339); 5 replaced by an anisotropy-plus-kurtosis test. Remaining MoE issue is load balancing (34/33/20/0.2 % routing, `enable_aux_loss_free_balancing = False`).
- A 500M-parameter from-scratch track is infeasible on this hardware: 284 tokens/s, 10.06 B tokens for a Chinchilla-optimal budget = 1.1 years; at step 130 the run had seen 0.00066 % of it.
- Asset: deliberation is kernel-launch bound (B = 1: 30.8 ms, B = 256: 34.5 ms), so search width is free up to about 256 branches.

## Interpretation
The defects are invisible from reading the code and from the benchmark score; each was found only by measuring the mechanism on the real checkpoint. They explain why several headline behaviours (halting, spawning, lattice use) could not have been working.

## Caveats and what went wrong
Single checkpoint; each probe is a measurement on that one artifact. The fixes were validated by re-measurement, not by a pre-registered test.

## Consequence
ISHMN is treated as UNVERIFIED and frozen (charter: "do not touch, do not integrate"). Lessons recorded: measure mechanisms on the real distribution (E-004), keep a random baseline beside every geometry number. The geometry and dynamic-MoE work proceeded as separate, pre-registered tracks.

## Evidence
- `docs/superpowers/handoff/SCOPE_CHARTER.md` - register row for ISHMN (frozen)
- `docs/COMPONENT_CARDS.md` - card 8
- `artifacts/unified_lifelong_brain.pt` - the probed checkpoint (local only)

## Reproduce
Not scripted as one command in the repository; each probe is described above and was run interactively against the checkpoint. A reproducible re-audit would need those probes turned into scripts (Q-12).

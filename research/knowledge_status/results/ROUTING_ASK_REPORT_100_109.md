# Lab A on report seeds 100-109: the asking reader against the look-based machine (same seeds, q = 0.05 for the conformal parts)

Bars: false-new 0.05, traps 0.05, B4a 0.85 / 0.95, B4b 0.95 / 0.95 (spec section 7). The rule tau0.5 was chosen on the tuning seeds 150-154 by cost; nothing here was used to choose.

| measure | look-based machine | tau0.5 (asks) | hyb0.5 (none class first) | conf0.05 (asks, calibrated) |
|---|---|---|---|---|
| known skills in new wordings called unseen | 0.279 | 0.025 | 0.008 | 0.206 |
| never-taught skills found | 0.866 | 0.974 | 0.703 | 0.712 |
| macro F1 (Thin / Solid labels come from the assignment here: read it as identification only) | 0.842 | 0.975 | 0.816 | 0.844 |
| everyday sentences called unseen | 0.247 | 0.709 | 0.002 | 0.006 |
| look-alike traps called unseen | 0.418 | 0.602 | 0.300 | 0.590 |
| near-twins labelled right | 0.942 | 0.982 | 0.889 | 0.914 |
| tangled: new part found (B4a residual hit) | 0.667 | 0.975 | 0.700 | 0.637 |
| tangled: known part left alone (B4a) | 0.702 | 0.975 | 0.990 | 0.830 |
| tangled: nothing flagged when known + Other (B4b) | 0.660 | 0.270 | 0.982 | 0.823 |
| tangled: Other part labelled Other (B4b) | 0.978 | 0.003 | 0.995 | 0.995 |
| real problems: Solid skills called unseen (B7) | 0.997 | 0.138 | 0.069 | 0.651 |
| real problems: never-taught skills found | 0.825 | 0.855 | 0.509 | 0.578 |

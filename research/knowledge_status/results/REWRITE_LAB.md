# Rewrite lab (tuning seeds 150-154, learned-map reader on Qwen layer 26, status machine q = 0.05)

| arm | known skills in new wordings called unseen (bar 0.05) | stories S4 | noisy text S7 | register S8 | never-taught skills found | macro F1 | rewrites kept (teaching) |
|---|---|---|---|---|---|---|---|
| nothing (the machine as it is) | 0.256 | 0.624 | 0.450 | 0.048 | 0.903 | 0.866 | 0 |
| map only | 0.262 | 0.656 | 0.462 | 0.017 | 0.762 | 0.783 | 348 per seed |
| full | 0.232 | 0.559 | 0.435 | 0.017 | 0.740 | 0.776 | 348 per seed |
| ceiling (CONTAMINATED DIAGNOSTIC) | 0.144 | 0.307 | 0.310 | 0.017 | 0.860 | 0.871 | 324 per seed |

The ceiling arm tells the writer to produce stories, typos and registers, so its numbers on those wordings are not a wording-invariance claim.

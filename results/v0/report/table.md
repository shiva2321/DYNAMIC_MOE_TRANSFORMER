# DEV (development seeds only) V0 Walking Skeleton Merged Report

| Schedule | Seed | V0 Acc | A3 Acc | A7 Acc | A5a Acc | V0 Experts | V0 Mem (MB) | Mean AUROC | V0 Latency b1 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| seq | 800 | 65.02% | 60.07% | 64.54% | 61.34% | 20 | 14.01 | 0.6803 | 4.64 |
| mix | 800 | 65.21% | 58.79% | 53.92% | 61.50% | 1 | 1.06 | 0.8502 | 0.54 |
| drift | 800 | 63.19% | 57.21% | 54.74% | 61.26% | 1 | 1.18 | 0.8755 | 0.55 |

### Component Register Reference

V0 is assembled from:
- **Encoder:** MiniLM through `unimoe/frontends.py` (cached offline, d=128)
- **Pool:** `UniPool` with tuned operating point `A1q` (inner_steps=50, q=0.05)
- **Head:** `ULDA` (running LDA head inside every expert)
- **Unknown Flag:** Conformal p-value and density flag (evaluated via AUROC vs OOD text)
- **Switched Off:** Asking reader, vertical growth, newborns, skip/chance flags

> [!NOTE]
> All runs above are demonstrations on development seeds. No claim of superiority is made.
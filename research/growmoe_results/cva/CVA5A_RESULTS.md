# CVA-5A Results: Resource Accounting Recount

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Generated: 2026-10-14 03:47:49
Stream: s2_permuted (seed 10, 6 regimes, schedule=[0,1,2,3,4,5,0,1])

---

## Summary & Verdict

| Section | Criterion | Hand Recount | pool_report | Verdict |
|:---:|:---|:---:|:---:|:---:|
| **Part 1** | Standard pool with archived expert (replay=None) | All 4 match exactly | All 4 match exactly | **PASS** |
| **Part 2** | Stage 3 pool with set_gate=True (source windows) | All 4 match exactly | All 4 match exactly | **PASS** |

---

## Part 1: Standard Pool with Archived Expert

Pool state at step 38:
- Expert 0: state = `active`, replay = `<growmoe.replay.Reservoir object at 0x000002DC3C186240>`, opt = `True`, sig.n = 256
- Expert 1: state = `archived`, replay = `None`, opt = `False`, sig.n = 256

| Quantity | Hand Recount | pool_report(pool, d) | Difference | Exact Match |
|:---|:---:|:---:|:---:|:---:|
| **Parameters** | 69652 | 69652 | 0 | YES |
| **Stored Vectors** | 768 | 768 | 0 | YES |
| **Gate MACs / input** | 65536 | 65536 | 0 | YES |
| **Memory Bytes** | 671824 | 671824 | 0 | YES |

---

## Part 2: Pool with `set_gate=True` (Source Windows Accounting)

Pool state at step 12 (with source windows):
- Expert 0: state = `active`, sig.n = 256, replay.n = 256, windows.n_samples = 535

| Quantity | Hand Recount | pool_report(pool_set, d) | Difference | Exact Match |
|:---|:---:|:---:|:---:|:---:|
| **Parameters** | 34826 | 34826 | 0 | YES |
| **Stored Vectors** | 1047 | 1047 | 0 | YES |
| **Gate MACs / input** | 32768 | 32768 | 0 | YES |
| **Memory Bytes** | 675368 | 675368 | 0 | YES |

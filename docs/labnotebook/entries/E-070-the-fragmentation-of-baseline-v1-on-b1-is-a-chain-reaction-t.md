# E-070 - The fragmentation of Baseline v1 on B1 is a chain reaction that the known founding-cluster repair removes

- **ID:** E-070
- **Date:** 2026-11-02
- **Track:** W
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B (script-generated, five development seeds, an intervention); the explanation of why the gate flags consecutive batches is inherited from E-037 and not re-derived here
- **Code / commit:** `m1/diag_transfer.py` (instance-level wrapper around `PhyloMoE._transfer_verdict`, no edit of growmoe/); result `results/m1/dev/diag_transfer_B1_n10.json`
- **People:** Claude (diagnosis); nobody has reviewed it

## Question
E-068 found that Baseline v1 as built makes 61-67 experts for 10 regimes, only when q 0.05 and the transfer spawn rule meet. Why does it, and what removes it?

## Why we asked it
Q-80 asks which defaults the shipped pool should have; the mechanism decides whether the fix is a threshold, the transfer rule or something older. The event counts of E-068 (spawns, almost no extends) did not say why.

## Prediction (written BEFORE the numbers, if there was one)
From reading `_handle_cluster` and `_transfer_verdict`: either the verdict could not run for young experts (it needs 64 reference losses per expert) and the pool spawned by default, or the verdict ran and always found a conflict. I expected the first. The twin fix (F-31) was a guess at the cause of the repeated flags, tested last.

## Method (how)
`python -m m1.diag_transfer` (development seeds 800-804, B1 N = 10, 5 steps per batch, pool seed = stream seed + 7, 2 threads): four configurations, each logging every call of the transfer verdict. Configurations: q 0.05 as built; q 0.05 with only `sig_dedupe=True, infant_relax=1.0`; q 0.05 with the U-real switches; q 0.02 as built.

## Results (what came out)
Table from `results/m1/dev/diag_transfer_B1_n10.json`:

| Configuration | Experts per seed | Verdicts | Not run | Conflict | No conflict | On the batch after a spawn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q 0.05 as built | 67, 67, 61, 67, 67 | 325 | 0 | 324 | 1 | 293 |
| q 0.05 with the twin fix only | 10, 10, 10, 10, 10 | 46 | 0 | 45 | 1 | 0 |
| q 0.05 with U-real | 10, 10, 10, 10, 10 | 46 | 0 | 45 | 1 | 0 |
| q 0.02 as built | 10, 10, 10, 10, 10 | 46 | 0 | 45 | 1 | 0 |

At q 0.05 as built the verdict always ran (the older experts had their 64 reference losses) and always found a conflict: the cluster is a whole batch of the NEW regime (mean loss 4.51 against 2.00 for the reference), and it is compared only with experts that already have a reference (on average 3.0 of 32.8 experts at the time of a call).

## Interpretation (what we understood, and on what basis)
Plain words: when a new regime starts, the pool founds one expert for it, which is right. At q 0.05 the gate then keeps calling the next batches of the same regime "new" for several batches (the founding-cluster flaw of F-31: the new expert's own calibration is too tight at first). The transfer rule decides "extend or spawn" by comparing the batch with the experts that already have 64 reference losses; the brand-new expert has none, so it cannot vouch for its own regime, only the older experts are asked, they all fail, and another expert is founded. That repeats for a regime until the gate stops flagging, six to seven experts per regime. The old probe rule finds the infant through its signature and lets it absorb the batch (15-29 experts in E-068). The repair of the flaw (sig_dedupe with infant_relax 1) removes the chain reaction at q 0.05; so does q 0.02, where the flaw is smaller. Measured: C-200. Inferred: the link between the flaw and the repeated flags (E-037), not re-measured here.

## Caveats and what went wrong
- One bed, N = 10, five development seeds, 5 steps per batch. The twin fix was not tuned; its accuracy cost or gain on B1 is in the E-068 diagnosis and in M1 stage B (candidate `shipped+twin`, to be added to the A2 list).
- Earlier results made with the shipped defaults on many-regime beds are suspect (F-111); the confirmed configurations of stage 4 had q 0.02 and the set gate, which hide it.

## Consequence
- C-200, F-111; Q-80 gains this entry. Recommendation to the user for BP1: the repair belongs in the defaults (a documented flaw, a repair that removes a 6.7 times blow-up, and the standing rule to adopt confirmed fixes); M1 stage B will measure it on every bed first (candidate `shipped+twin` in A2).

## Evidence
- `results/m1/dev/diag_transfer_B1_n10.json` - the verdict logs summary per configuration and seed
- `m1/diag_transfer.py` - the script
- `results/m1/dev/diag_switches_B1_n10.json` - the E-068 diagnosis

## Reproduce
```
python -m m1.diag_transfer --out results/m1/dev/diag_transfer_B1_n10.json
```

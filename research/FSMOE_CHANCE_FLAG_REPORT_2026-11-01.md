# The worse-than-chance rule as a flag in the real spawn test (2026-11-01)

You said yes to putting the weak-parent fix into the real spawn test behind a flag. It is in, it is off by default, and I ran the standard checks on the earlier rounds' streams. The checks show the fix is **not free**: it repairs pools with weak experts, but it costs something on strong pools and on drifting streams. So it stays an opt-in switch. Predictions first (6, of which 4 held). Every number below is copied from the generated table (links at the end).

## The short answer

1. **It is in growmoe, off by default.** `PhyloCfg.transfer_chance` (default None = off): when set to c, a parent whose loss on a new cluster is at least c times the chance loss cannot vouch for it, and the cluster founds its own expert. A second option, `transfer_chance_weak_only`, applies the veto only when the parent's own past losses are also near chance. This is the first change to growmoe in this work and you asked for it.
2. **Off means unchanged.** All 1,089 tests of growmoe, unimoe and fsmoe pass (the same 2 expected failures as before), the pool reproduces my earlier function-regime results bit for bit, and on the real 7-domain suite it reproduces the stored E-054 cells exactly on the mixed and drifting schedules and within 0.0009 on the sequential one (a difference I did not trace; float summation order under another thread count is my guess).
3. **On, it repairs weak pools.** The shipped pool with 8-unit networks goes from 0.293 to 0.471 on 30 function regimes and from 0.355 to 0.480 on 10, and keeps one expert per regime (it was losing more than half). The result matches the E-061 ablation bit for bit.
4. **On, it is not free for strong pools.**
   - 128-unit pool, function regimes: 0.4777 against 0.4845 on 30 regimes, 0.4913 against 0.4931 on 10 (one extra expert per about 50 regimes).
   - Real 7-domain suite, drifting schedule: 0.6236 against 0.6423 on average, and in one seed 0.580 against 0.674: the rule split a drifting stream in two that a single expert had handled well.
   - Real suite, sequential schedule: the plain rule gives 0.6343 against 0.6454.
5. **The weak-parent-only option helps only partly.** On the sequential schedule it leaves the strong pool exactly unchanged (identical in 5 of 5 seeds), but it cannot tell a weak pool from a drifting stream, because in both the parent's own past losses are near chance. So it still splits drifting data, and on the hard function regimes even 128-unit experts count as weak.
6. **Where it does nothing:** on the banking77 class-incremental stream (6 experts, identical accuracy 0.8464 with 128 units and 0.8193 with 16 units in 3 seeds) and on the mixed schedule (one expert).

## How to use it

- **Turn it on when** experts are small (or are notebooks) and the stream brings new regimes: `transfer_chance=1.0`.
- **Leave it off** for strong experts, for drifting streams, and for the default pool.

## What I got wrong

- My E-061 report said the rule "costs the default nothing measurable". That was based on the function-regime streams only (a 0.007 drop on 30 regimes is inside the noise there). The real-suite check shows a real cost on drifting streams; I have corrected that sentence in the E-061 report.
- I predicted that the weak-parent-only option would leave strong pools alone everywhere (it did not on drift or on the hard function regimes), and that 128-unit pools stay within 0.005 (they do not on 30 regimes).

## What I would do next (your call)

1. **Find a rule that separates "weak experts, new regimes" from "strong expert, drifting stream"** (open question Q-67). Losses alone cannot do it, since both look like chance. Candidates: compare the cluster with the parent's own recent samples, or bring back a small held-out probe of extend against spawn for the borderline cases.
2. **Leave the flag as it is** and use it only for notebook-first or small-network pools.

## Evidence

[CHANCE_FLAG.md](../results/fsmoe/N/CHANCE_FLAG.md) (the predictions, the three checks), design and predictions [2026-11-01-fsmoe-chance-flag-design.md](../docs/superpowers/specs/2026-11-01-fsmoe-chance-flag-design.md), notebook entry E-062; code `growmoe/config.py`, `growmoe/pools/phylo.py`, `growmoe/transfer.py` (the flag), `fsmoe/chance_check.py`, `fsmoe/chance_check_bank.py`, `fsmoe/chance_table.py`; tests `tests/growmoe/test_gm_transfer_chance.py` (7 tests, 11 mutants killed).

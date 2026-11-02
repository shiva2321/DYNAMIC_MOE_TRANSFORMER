# E-062 - growmoe flag for the worse-than-chance spawn rule: repairs weak pools, not free for strong ones

- **ID:** E-062
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment
- **Status:** completed; follows E-061; the first change to growmoe/ made in the FS-MoE work (asked for by the user)
- **Evidence grade:** B (script-generated, predictions declared before the checks they test, flag-off equality tested; UniMoE seeds 500-504 were used by E-054 and serve here as a paired comparison only; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); changed `growmoe/config.py` (`transfer_chance`, `transfer_chance_weak_only`, validation), `growmoe/pools/phylo.py` (`_transfer_verdict`), `growmoe/transfer.py` (`TransferVerdict.chance_veto`, `fields()` adds `chance_veto` only when it fired); `fsmoe/nbpool.py` now uses the flag instead of its own override (`nb_cfg` sets `transfer_chance=1.0`); new `fsmoe/chance_check.py`, `fsmoe/chance_check_bank.py`, `fsmoe/chance_table.py`; tests `tests/growmoe/test_gm_transfer_chance.py` (7 tests; 1,090 pass over tests/growmoe, tests/unimoe and tests/fsmoe, 2 expected failures as before; 11 mutants of the flag code killed, one only after a test was added; 102 mutants killed over E-055 to E-062); design `docs/superpowers/specs/2026-11-01-fsmoe-chance-flag-design.md`
- **People:** Claude designed, built and ran it on the user's answer "Yes, behind a flag" (2026-11-01) to the next-step question after E-061 (Q-65); no reviewer yet

## Question
Can the worse-than-chance rule of E-061 (a parent whose loss on a new cluster is at least the chance loss cannot vouch for it) go into growmoe's spawn test behind a flag, with the default unchanged, and what does it do on the streams of the earlier rounds? (Q-65)

## Why we asked it
E-061: the shipped pool with 8-unit networks lost more than half of the regimes through absorptions into weak parents; the rule repaired it on the function-regime beds, with a cost to strong pools shown only there (0.007 on 30 regimes). The user asked for the flag; the standard checks on real streams were proposed in the same breath.

## Prediction (written BEFORE the checks they test)
`docs/superpowers/specs/2026-11-01-fsmoe-chance-flag-design.md`: the flag itself, then (after the first UniMoE check had shown a cost of the plain rule) the weak-parent-only option and W1-W6. Scored by `fsmoe/chance_table.py`: **4 of 6 held** (W2, W3, W4, W6); missed W1 (UniMoE suite, ULDA: weak-only within 0.003 of flag off on seq and drift: seq equal, drift 0.6236 against 0.6423) and W5 (128-unit pool within 0.005 of flag off on f10 and f30: f30 0.4777 against 0.4845).

## Method (how)
- `transfer_chance = c` (default None = off): in `_transfer_verdict`, when the shipped test finds no conflict and the parent's cluster loss is at least c x ln(max(classes it has seen, 2)), the verdict becomes a conflict (`chance_veto`); it only ever adds conflicts. `transfer_chance_weak_only`: the veto also needs the parent's mean reference loss at least c x chance.
- Equality checks: flag off against the stored E-061 cells (function regimes f10 seed 710: U128 0.497, U8 0.3225, identical), flag 1.0 with 8 units against E-061's U8x (0.479, identical); the full test suites; UniMoE flag-off cells against the stored E-054 cells.
- UniMoE real suite (the streams of E-054: seven real text domains, schedules seq / mix / drift, seeds 500-504; `unimoe.runner.run_cell` into results/unimoe/p4_chance, the E-054 cells only read): arms ULDA off / on (c = 1.0) / on weak-only; 16-unit experts with growth off (the small-start control) off / on / on weak-only.
- Function regimes (growmoe's viability stream, f10 and f30, seeds 710-714): the shipped pool with 8 and 128 units, flag off, E-061's subclass rule (U8x, U128x) and the flag weak-only (U8w, U128w).
- banking77 class-incremental (the M5 stream, local MiniLM frontend, the current default pool U-real, seeds 710-712): 128 and 16 units, flag off / on / on weak-only.

## Results (copied from the generated files)
`results/fsmoe/N/CHANCE_FLAG.md`:
- Equality: flag off reproduces E-061 cells bit for bit; the UniMoE mix and drift cells exactly and the seq cells within 9.1e-4 (cause not traced; float summation order under another thread count is a guess).
- Function regimes, accuracy (experts per regime): f30 U8 0.293 (0.45), U8x / U8w 0.471 (1.00), U128 0.4845 (1.00), U128x / U128w 0.4777 (1.02); f10 U8 0.355 (0.64), U8x / U8w 0.480 (1.02), U128 0.4931, U128x / U128w 0.4913 (1.02).
- UniMoE suite, mean final accuracy (experts): seq ULDA off 0.6454 (19.6), on 0.6343 (20.8), weak-only 0.6454 (19.6, identical in 5 of 5 seeds); drift ULDA off 0.6423 (1.2), on 0.6236 (1.4), weak-only 0.6236 (1.4), seed 501 off 0.6738 (1 expert, 1 extend), on / weak-only 0.5802 (2 experts); mix 0.6631 in every arm (one expert). 16-unit experts: seq off 0.5898, on 0.5931, weak-only 0.5898; drift off 0.3823, on / weak-only 0.3784; mix 0.4420.
- banking77 (3 seeds): 128 units 0.8464 and 16 units 0.8193 for off, on and weak-only alike (6 experts, no extend in any arm): the rule never fires.

## Interpretation (what we understood, and on what basis)
Plain words: the rule fixes the failure of weak experts on new-regime streams, and it works in the real pool exactly as in the ablation; but a parent that does badly on a cluster is sometimes the right place for it (a drifting stream), and the checks on real text show the rule splitting such streams. The weak-parent-only option protects strong pools on the sequential schedule but not on drift, because a weak pool and a drifting stream both have reference losses near chance: losses alone cannot tell them apart. Technical: measured: all of the above; the mechanism of the drift cost is read from the events (an extend replaced by a spawn: 1 expert with 1 extend became 2 experts) and the accuracy of that seed (0.674 to 0.580), not isolated by a probe; on the hard function regimes even 128-unit experts have reference losses near chance, so weak-only equals the plain rule there.

## Caveats and what went wrong
- UniMoE seeds 500-504 were used by E-054 (paired comparison only, nothing tuned on them); one drift cell (seed 501) carries most of the drift difference.
- My E-061 claim that the rule costs the default nothing measurable was too strong (F-87); the E-061 report is corrected, claim C-167 superseded in part by C-173.
- W1 and W5 missed (F-86).
- A mistake of mine while moving the rule out of the subclass: I cut five tests of `test_fs_nbpool.py` by truncating the file; the test count (13 against 18) showed it at once and they were restored.

## Consequence
Report: `research/FSMOE_CHANCE_FLAG_REPORT_2026-11-01.md`; tables `results/fsmoe/N/CHANCE_FLAG.md`. Q-65 answered in part (flag and checks done); new Q-67. Decisions D-58, D-59. Claims C-171..C-175. The flag stays OFF by default: opt-in for small-network or notebook pools on new-regime streams, not for drifting streams or strong pools.

## Evidence
- `research/FSMOE_CHANCE_FLAG_REPORT_2026-11-01.md` - the plain-English report
- `results/fsmoe/N/CHANCE_FLAG.md` - the scorecard and the three checks
- `docs/superpowers/specs/2026-11-01-fsmoe-chance-flag-design.md` - design, first-check finding, predictions
- `results/unimoe/p4_chance/` - the UniMoE cells (one JSON and NPZ per schedule, arm and seed)

## Reproduce
```
python -m pytest tests/growmoe tests/unimoe tests/fsmoe
python -m fsmoe.chance_check run --seeds 500   (one process per seed 500-504; then: python -m fsmoe.chance_check table --seeds 500 501 502 503 504)
python -m fsmoe.nbpool_lab run --bed f30 --seeds 710 --arms U8 U8x U8w U128 U128x U128w   (also f10, seeds 710-714)
python -m fsmoe.chance_check_bank run --seeds 710 --arms off128 on128 onw128 off16 on16 onw16   (seeds 710-712)
python -m fsmoe.chance_table
```

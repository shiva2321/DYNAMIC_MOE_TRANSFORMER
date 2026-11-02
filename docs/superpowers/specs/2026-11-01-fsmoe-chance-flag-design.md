# The worse-than-chance rule as a flag of growmoe's spawn test (design, 2026-11-01)

The user chose "Yes, behind a flag" after E-061. This is the first change to growmoe/ in the FS-MoE work and was asked for explicitly. Same rules: predictions before the checks they test, nothing committed, the default behaviour unchanged.

## What is added
* `PhyloCfg.transfer_chance: Optional[float] = None` (OFF). When set to c, a parent whose zero-shot loss on a new cluster is at least c x ln(classes it has seen) cannot vouch for it: a no-conflict transfer verdict becomes a conflict and the cluster founds a child (`growmoe/pools/phylo.py::_transfer_verdict`; `TransferVerdict.chance_veto`, carried by the spawn event as `chance_veto` only when it fired). The rule only ever adds conflicts.
* Validation: `transfer_chance` must be positive or None. With the flag None nothing changes (tested: the full growmoe + unimoe + fsmoe suite is unchanged, 945 + 138 tests, and the flag-off pool reproduces the E-061 numbers bit for bit).
* Checked equal to the E-061 ablation: the shipped pool with 8-unit networks and `transfer_chance=1.0` reproduces E-061's arm `U8x` bit for bit (f10 seed 710: 0.479), flag off reproduces `U8` and `U128` bit for bit.

## First standard check, before any refinement (UniMoE real suite, the streams of E-054: seven real text domains, schedules seq / mix / drift, seeds 500-504, `fsmoe/chance_check.py`)
Flag off reproduces the stored E-054 ULDA cells exactly on mix and drift and within 0.0009 on seq. Flag on (cluster rule, c = 1.0), mean final accuracy: ULDA seq 0.6343 against 0.6454 off (experts 20.8 against 19.6), drift 0.6236 against 0.6423 (seed 501: 0.580 against 0.674, a second expert where one drifting expert had coped), mix unchanged (one expert); 16-unit experts with growth off: seq 0.5931 against 0.5898, drift 0.3784 against 0.3823. So the unconditional rule is not free for a STRONG pool on real text: it turns legitimate extends (drift) into spawns.

## The refinement tested next (written before its runs)
`PhyloCfg.transfer_chance_weak_only: bool = False`: with `transfer_chance` set, veto only when the PARENT is weak, i.e. its own mean reference loss is also at least c x chance. A strong parent (reference loss well under chance) keeps the shipped verdict; a weak one (a notebook, a small network: reference losses near or above chance) cannot vouch.

## Predictions (scored by `fsmoe/chance_table.py`)
* W1 (UniMoE suite, ULDA, seeds 500-504) with the weak-only rule, the mean final accuracy on seq and on drift is within 0.003 of flag off, and at least 8 of the 10 seq / drift cells equal flag off within 0.002.
* W2 (UniMoE suite, 16-unit experts, growth off) with the weak-only rule the mean on seq and on drift is within 0.004 of flag off.
* W3 (f30, seeds 710-714) the shipped pool with 8-unit networks and the weak-only rule keeps at least 0.9 experts per regime and scores at least 0.44 (flag off 0.293, unconditional rule 0.471).
* W4 (f10, seeds 710-714) the same with at least 0.9 experts per regime and at least 0.44 (off 0.355, unconditional 0.480).
* W5 (f10 and f30) the shipped pool with 128-unit networks and the weak-only rule is within 0.005 of flag off.
* W6 (banking77 class-incremental, seeds 710-712) the weak-only rule with 128-unit experts is within 0.01 of flag off, and with 16-unit experts at least flag off minus 0.01.

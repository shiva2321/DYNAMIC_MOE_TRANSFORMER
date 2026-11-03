# V0: the walking skeleton (APPROVED by the user on 2026-11-04: "approve and continue on both"; build job V-1, `docs/superpowers/handoff/V0_BUILD_SPEC.md`)

**In plain words.** Until now the plan measured parts one at a time. The user asked to assemble a working model first, make it work, refine it, show it, and only then do the full experiments and proofs. This note fixes what the first assembled model (V0) is, what it shows, what it is allowed to claim (nothing beyond a demonstration), and how every later version is checked with the same cheap meter so that "it works" is measured, not asserted. It reorders the road (ROADMAP, MILESTONES); it does not drop the preregistration rules for any claim.

## 1. Why this order, and the guard rail
Building first has bitten this project four times: a string-grading artifact (ISHMN, F-01), a replay baseline trained too little (C-13), a pool that shared its seed with the data (head start of about 8 points, F-15), and a default pool that fragments into 61-67 experts where 10 are needed (E-068). In each case something looked good and was wrong. **Guard rail:** a demo may show behaviour; a claim (a number in a report, an abstract, a preprint) needs a preregistered run on fresh seeds. Every version passes through the same cheap meter (section 4) from the day it exists.

## 2. What V0 is (assembly from parts that exist, new code only where a join is missing)
| Part | Taken from | State |
|---|---|---|
| Frozen encoder | MiniLM through `unimoe/frontends.py` | works, cached |
| Pool with spawn, extend and merge | `growmoe/pools/phylo.py` (Baseline v1 stack: router, transfer rule) inside `unimoe/pools.py` `UniPool` | works; Baseline v1 as shipped fragments on many-regime beds, so V0 uses the tuned operating point of M1 (A2), not the class defaults |
| Closed-form head in every expert | `ULDA` of `unimoe/systems2.py` | the one big win on real text (E-054); D-43 says default, not yet in code |
| Unknown flag | density and conformal flag of the pool, fingerprint shortlist | weak on topical text (OOD flag silent in E-053, refusal AUROC-limited in E-054): V0 shows it with its measured AUROC and says so |
| Real text stream | the seven cached UniMoE domains (`unimoe/domains.py`, schedules seq, mix, drift) | works |
| Not in V0 | asking reader (no reliable gain on topical domains, E-066), growth after birth (idle), notebook-first newborns, skip and chance flags (small or costly, E-061 to E-063) | stay off; each can be tried as a V0.x change against the meter |

## 3. What the V0 demo shows (one script, this PC, minutes, one seed argument)
A stream of real text topics arrives one after another. After each topic the script prints, per topic seen so far, the accuracy (to show that old topics are kept), the number of experts and the memory in bytes, the unknown-flag rate on held-out topics never shown, and the time and counted operations spent. It writes one figure and one JSON record. Next to V0 it runs three honest comparators on the same data in the same order: same-memory replay network, plain kNN, streaming LDA. If a comparator wins on a topic set (as kNN and LDA do on topical text, E-066, M1 look), the demo prints that, not a flattering subset.

## 4. The meter every version passes through (no new machinery)
On the real-text beds V0 IS the M1 cell `H_ULDA` (same pool, same LDA head, same tuned operating point), so no new arm id is added while the scored run is in progress (editing `m1/arms.py` would change that run); the demo checks its own result against `execute_cell(bed, "H_ULDA")` exactly. Originally planned: V0 gets arm id `V0` in the M1 harness (`m1/arms.py`, deterministic profile first: accuracy, forgetting, memory, counted multiply-adds; the timed profile only when a version is a candidate for a claim). Cells use development seeds 800-804 while building; report seeds stay frozen and untouched. V1, V2 get ids `V1`, `V2`.

## 5. Versions
- **V0** assembly above. Acceptance: runs end to end on this PC under 10 minutes on one seed; reproducible (same seed, same record); the meter record exists; the demo prints its comparators. No accuracy bar (nothing is claimed).
- **V1** exact, certified removal of an expert (probe P2) and a retire rule that never evicts without certified redundancy (user decision of 2026-10-16). Acceptance: removing an expert changes no answer of the experts that remain; kill rule from MILESTONES P2.
- **V2** a trained representation (a small language model, probe P1, then M5). Acceptance and kill rule from MILESTONES P1.
- **V3** the parts that earn their keep after M1's deterministic results: cheap novelty (M3), router at scale (M4).
Proofs, the standard text benchmark (M2) and the preprint decision (BP2) run on the version we keep.

## 6. Decisions owed by the user
1. Approve this order and the V0 scope (section 2), or change the parts list.
2. The timed pass of M1 (stopped by the power cut of 2026-11-03): relaunch it (it only informs which parts V0 keeps and gives the timing "before" picture), with the 3-seed precision rule, or stop it at the seed that is done?
3. The M2 dataset downloads (needed later, not for V0).

## 7. Risks named now
The unknown flag may be too weak to show; the demo then shows exactly how weak. V0 inherits the pool's cost at answer time (about 10 times the latency of a dense network in the first look at seed 810, not a result): V0 states its cost next to its behaviour. An assembled model with a bug in a join looks like a model that works: the meter record and the comparators are the defence.

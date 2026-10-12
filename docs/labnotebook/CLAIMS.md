# Claims register

Every statement we might publish or rely on, with its honest status. Grades: README section 3. A claim leaves this table only by being superseded (never deleted).
Statuses: SUPPORTED, REFUTED, PARTIAL, OPEN, UNVERIFIED. "Used seeds" are listed in `DECISIONS.md` D-05.

## Dynamic MoE (Track G)

| ID | Claim | Status | Grade | Entries | Limits / what is missing |
|---|---|---|---|---|---|
| C-12 | A von Mises-Fisher kernel-density signature with conformal p-values is a good novelty score (AUROC S1 0.973, S2 0.999) | SUPPORTED | B | E-010 | 5 seeds; weak for diffuse regimes; stdlib code after pydoc text 0.859 |
| C-15 | False-novelty is controlled at level q (rate 0.000 <= 2q) | SUPPORTED | B | E-011 | measured with the pool's own `unknown` flag; the first benchmark metric was an artifact (F-05) |
| C-13 | The Phase-1 pool "beats replay by about 19 points" | **REFUTED** | B | E-012 | holds only at 1 optimiser step per batch (replay under-trained); at 5-20 steps replay ties or beats the pool |
| C-14 | kNN storing everything beats the Phase-1 pools on S1 and S2 | SUPPORTED | B | E-012, E-013 | those streams can be solved by lookup; says nothing about streams where the answer must be computed |
| C-16 | Where the answer must be computed, lookup fails and a trained network or expert wins | SUPPORTED | B | E-012, E-013 | hidden-function labels; kNN 21-26 % vs trained net 45-78 % (chance 12.5 %) |
| C-17 | The frozen Phase-1 pool beats the best single network by >= 5 points on function streams | **REFUTED** | A | E-013 | pre-registered B1 FAIL: +4.1 [2.9, 5.4] (F-15), +2.6 [0.7, 4.5] (F-30); kill rule fired for that pool |
| C-18 | Oracle-routed experts beat the best single network by +6.6 / +11.3 / +16.2 points (5 / 15 / 30 regimes) with about 1/4 the memory and about 30x fewer multiply-adds | SUPPORTED | B | E-013 | the structure is valuable if routing and training are right |
| C-19 | The M0 shortfall came from a lifecycle bug (competence monitor armed from the untrained loss) | SUPPORTED | A | E-014, E-015 | |
| C-20 | The repaired pool beats the best single network by >= 5 points on function streams | SUPPORTED | A | E-015 | F-15 +10.0 [8.0, 12.0], F-30 +11.6 [9.5, 13.8]; one synthetic stream family; seeds 10-19 |
| C-21 | The pool is cheaper than a single network | **REFUTED today** | A (descriptive) | E-015 | at F-30: 1,000 vs 1,050 kMACs per prediction, 1.41x memory, gate = 97 % of multiply-adds; a perfectly routed expert needs 34 kMACs |
| C-31 | A frozen pretrained text encoder helps the pool on text streams | SUPPORTED (small) | A | E-013 | S1-PRE - S1 = +3.8 [0.7, 6.9]; one stream |
| C-22 | Per-sample assignment contaminates experts when regimes overlap (training purity 78 % at centre cosine 0.3) | SUPPORTED | B | E-017 | dev seeds |
| C-23 | Letting a batch train the expert most of it was routed to raises accuracy under overlap and on function streams without harming guard streams | SUPPORTED | A | E-019 | OV-0.3 +11.4 [10.0, 12.9], F-30 +6.4, F-15 +3.0; seeds 20-39; synthetic streams |
| C-24 | The plain vote costs about 1.3 points on known minority regimes that later arrive mixed 70/30 at zero overlap | SUPPORTED (small) | A | E-019 | -1.3 [-2.8, +0.2]: interval includes 0 |
| C-25 | A fingerprint shortlist in front of the exact gate cuts gate cost 7-14x with no accuracy loss | OPEN | B | E-017, E-021 | dev seeds only; the pre-registered test is running |
| C-26 | Shortlist plus an exact second look gives novelty decisions identical to the exact gate | OPEN | H | E-018, E-021 | proof sketch in the spec; property test not yet run |
| C-27 | At centre cosine >= 0.6 regimes merge because per-sample novelty is blind, while a batch-level test sees them | PARTIAL | B | E-017 | isolated test 100 % detection / 0 % false alarms; the naive live version over-fired (72-131 experts) |
| C-28 | Label-aware assignment fixes overlap | **REFUTED** | B | E-017 | +1.6 / +2.8 at 0.3, +0.5 at 0.6 |
| C-29 | Statistically controlled growth is unique to our system | **REFUTED** | L | E-012 | MaRS (ICLR 2026), SEMA (CVPR 2025), LLaVA-CMoE are close relatives |
| C-30 | The growth, merge, archive and revive machinery does useful work | UNVERIFIED | B | E-010, E-013 | growth fired in 3 of 70 runs; the others never fire in the benchmarks |

## Geometry (Tracks Y, Z) and the older ISHMN system (Track I)

| ID | Claim | Status | Grade | Entries | Limits / what is missing |
|---|---|---|---|---|---|
| C-01 | ISHMN's manifold sandwich gives +10 points on GSM8K | **REFUTED** | B | E-001 | grading artifact, confounded arms, n = 30 |
| C-02 | ISHMN had six structural defects (halting, adapter sphere, lattice collapse, unreachable mitosis, ...) | SUPPORTED (3 fixed) | B | E-002 | one checkpoint |
| C-03 | REM replay as written destroyed the world model; freezing the transition weights removes 94 % of the damage | SUPPORTED | B | E-003 | one checkpoint; the off-distribution `z_init` remains |
| C-04 | Distinct-2 = 1.000 means no repetition collapse | **REFUTED** | B | E-005 | it is what noise scores |
| C-05 | A D = 8 sphere cannot host VSA quasi-orthogonality for 5,295 concepts | SUPPORTED | B | E-006 | whether this caused the lattice collapse is untested |
| C-06 | Hyperbolic embeddings beat Euclidean for static trees at d = 4-16 | SUPPORTED | B | E-007 | equal at d = 32; static tree only |
| C-07 | Hyperbolic attention heads improve language-model perplexity | **NOT SUPPORTED** | B | E-007 | after the temperature control: PPL 9.47 vs 9.63; Bonferroni p > 0.2 |
| C-08 | Branch-and-bound retrieval speeds up attention at N <= 4096 | **REFUTED** | B | E-007 | 36-60x slower than a fair dense baseline |
| C-09 | A snapped rotation state tracks Z6 and S3 exactly to length 256 | SUPPORTED | B | E-007 | no advantage over a GRU; at chance on S5 |
| C-10 | The spherical residual stream (nGPT family) improves loss and token efficiency | **PARTIAL / under review** | C | E-008 | the baseline's best loss (1.898-1.903) is below nGPT's final loss (1.9525); speedup is defined against the baseline's over-trained final loss |
| C-11 | Self-dual cone (KATA-PSD) memory gives the highest recall per byte | SUPPORTED as reported | B | E-008 | not re-audited in this notebook |

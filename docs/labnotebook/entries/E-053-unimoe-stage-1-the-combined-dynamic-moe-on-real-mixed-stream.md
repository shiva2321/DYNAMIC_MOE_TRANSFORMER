# E-053 - UniMoE Stage 1: the combined dynamic MoE on real mixed streams works as a continual learner, does not beat simple baselines, and its growth and refusal features are not demonstrated

- **ID:** E-053
- **Date:** 2026-10-30
- **Track:** G
- **Type:** experiment
- **Status:** completed for the tuning and development seeds; Stage 2 (token-level language-model MoE) not started; the fixes of section 7 of the analysis are not built
- **Evidence grade:** B (script-generated, 3 evaluation seeds 500-502 and one tuning seed 499, all dev seeds; predictions declared before each run and scored; NOT a pre-registration; no independent reviewer yet)
- **Code / commit:** branch `unimoe-stage1` (worktree D:/shm_um), package `unimoe/` (frontends, domains, suite, systems, runner, analysis, report, domain_routing, experiment, p3, p3_routing), tests `tests/unimoe/` (56), commits 9e670de to 53c7e8d; `growmoe/` not edited
- **People:** Claude built and ran it on the user's request of 2026-10-30 (a combined dynamic MoE, trained on real streams, evaluated from all aspects) after the user chose both levels in order

## Question
Combine what the project learned into one dynamic MoE (frozen encoder on a unit sphere, a pool that spawns, merges and shares experts, batch-level novelty tests, a learned router, a transfer rule, width and depth growth, a language-model refusal layer), train it on real mixed streams (banking77, ag_news, 20 newsgroups, emotion, SST-2, SVAMP word problems, digit arithmetic: 22 regimes, three schedules) and evaluate it from every side to find where it is short and why.

## Why we asked it
The user said the labs were returning nothing and no working system existed (2026-10-30). The router of E-052 was one piece; this is the whole pool on real streams against honest baselines.

## Prediction (written BEFORE the numbers, if there was one)
`research/unimoe/PROGRESS.md`: protocol and 26 numeric predictions before the evaluation, then predictions before each later experiment (E1-E7 and the per-schedule correction). 63 numeric predictions in all: 31 held, 32 missed. The misses cluster on overweighting tuning knobs, not looking for couplings between parts, and taking the suite's headroom for granted.

## Method (how)
- Seven domains and three OOD text sets from the local Hugging Face cache (no download), frozen MiniLM to a 128-d unit sphere; schedules seq (28 phases), mix (5), drift (10); global label blocks per domain.
- Ten systems (shipped pool, U-real, U-deep, U-recipe; kNN and replay at equal memory, streaming LDA, offline joint network, experts told the regime, fine-tuning), every trainable arm tuned on seed 499 PER SCHEDULE; strict test-then-train; per-phase accuracy matrix, OOD novelty, per-sample logs, owner-expert oracle routing; cells cached with a code stamp.
- Analysis (`analysis.py`, `report.py`), nine ablations, domain-level routing with the language-model reader, expert-quality ladder, growth studies.

## Results (what came out)
Copied by script from `results/unimoe/REPORT_v2.md`, `E2_domain_routing.md` and the `p3_*` folders. Final mean accuracy over the 22 regimes, mean over seeds 500-502 (the U-real-no-* rows are the ablations).

#### A. Performance: final mean accuracy over the regimes (mean over seeds [95 % interval])

**seq** (3 seeds)

| system | final mean | banking | agnews | newsgroups | emotion | sst2 | svamp | arith |
|---|---|---|---|---|---|---|---|---|
| U-deep | 0.622 [0.599, 0.645] | 0.83 | 0.60 | 0.41 | 0.36 | 0.49 | 0.59 | 0.68 |
| U-real | 0.622 [0.599, 0.645] | 0.83 | 0.60 | 0.41 | 0.36 | 0.49 | 0.59 | 0.68 |
| U-real-no-batch_vote | 0.593 [0.505, 0.682] | 0.80 | 0.53 | 0.39 | 0.34 | 0.46 | 0.55 | 0.66 |
| U-real-no-competence | 0.616 [0.572, 0.661] | 0.83 | 0.58 | 0.39 | 0.36 | 0.50 | 0.54 | 0.68 |
| U-real-no-growth | 0.622 [0.599, 0.645] | 0.83 | 0.60 | 0.41 | 0.36 | 0.49 | 0.59 | 0.68 |
| U-real-no-probation | 0.617 [0.585, 0.649] | 0.81 | 0.61 | 0.46 | 0.37 | 0.54 | 0.58 | 0.63 |
| U-real-no-router | 0.620 [0.589, 0.651] | 0.83 | 0.59 | 0.41 | 0.36 | 0.48 | 0.59 | 0.68 |
| U-real-no-set_gate | 0.543 [0.428, 0.657] | 0.66 | 0.50 | 0.31 | 0.30 | 0.47 | 0.60 | 0.68 |
| U-real-no-shortlist | 0.622 [0.599, 0.645] | 0.83 | 0.60 | 0.41 | 0.36 | 0.49 | 0.59 | 0.68 |
| U-real-no-transfer | 0.584 [0.554, 0.614] | 0.76 | 0.50 | 0.39 | 0.32 | 0.45 | 0.59 | 0.68 |
| U-real-no-twin_fix | 0.610 [0.585, 0.636] | 0.81 | 0.52 | 0.44 | 0.34 | 0.43 | 0.58 | 0.68 |
| U-recipe | 0.630 [0.617, 0.643] | 0.80 | 0.61 | 0.43 | 0.37 | 0.57 | 0.60 | 0.70 |
| finetune | 0.075 [-0.066, 0.216] | 0.04 | 0.15 | 0.02 | 0.00 | 0.00 | 0.09 | 0.16 |
| joint | 0.669 [0.642, 0.696] | 0.86 | 0.63 | 0.48 | 0.41 | 0.52 | 0.63 | 0.73 |
| knn_eqmem | 0.642 [0.627, 0.658] | 0.87 | 0.63 | 0.45 | 0.38 | 0.45 | 0.57 | 0.68 |
| lda | 0.606 [0.572, 0.641] | 0.84 | 0.59 | 0.56 | 0.40 | 0.58 | 0.40 | 0.52 |
| oracle_experts | 0.669 [0.646, 0.692] | 0.79 | 0.83 | 0.56 | 0.51 | 0.59 | 0.45 | 0.67 |
| replay_eqmem | 0.636 [0.576, 0.696] | 0.81 | 0.62 | 0.46 | 0.37 | 0.49 | 0.53 | 0.71 |
| shipped | 0.546 [0.436, 0.656] | 0.66 | 0.52 | 0.32 | 0.29 | 0.55 | 0.55 | 0.67 |

**mix** (3 seeds)

| system | final mean | banking | agnews | newsgroups | emotion | sst2 | svamp | arith |
|---|---|---|---|---|---|---|---|---|
| U-deep | 0.596 [0.550, 0.642] | 0.75 | 0.58 | 0.43 | 0.36 | 0.53 | 0.49 | 0.66 |
| U-real | 0.596 [0.550, 0.642] | 0.75 | 0.58 | 0.43 | 0.36 | 0.53 | 0.49 | 0.66 |
| U-recipe | 0.582 [0.555, 0.610] | 0.74 | 0.53 | 0.43 | 0.30 | 0.52 | 0.52 | 0.66 |
| finetune | 0.552 [0.539, 0.564] | 0.69 | 0.51 | 0.40 | 0.27 | 0.50 | 0.54 | 0.64 |
| joint | 0.668 [0.650, 0.686] | 0.85 | 0.61 | 0.50 | 0.40 | 0.50 | 0.64 | 0.74 |
| knn_eqmem | 0.517 [0.492, 0.542] | 0.63 | 0.49 | 0.33 | 0.27 | 0.41 | 0.46 | 0.65 |
| lda | 0.610 [0.589, 0.632] | 0.83 | 0.59 | 0.56 | 0.40 | 0.57 | 0.41 | 0.54 |
| oracle_experts | 0.663 [0.652, 0.674] | 0.77 | 0.84 | 0.56 | 0.46 | 0.62 | 0.53 | 0.66 |
| replay_eqmem | 0.632 [0.620, 0.645] | 0.79 | 0.64 | 0.47 | 0.39 | 0.55 | 0.55 | 0.68 |
| shipped | 0.596 [0.550, 0.642] | 0.75 | 0.58 | 0.43 | 0.36 | 0.53 | 0.49 | 0.66 |

**drift** (3 seeds)

| system | final mean | banking | agnews | newsgroups | emotion | sst2 | svamp | arith |
|---|---|---|---|---|---|---|---|---|
| U-deep | 0.512 [0.412, 0.612] | 0.64 | 0.52 | 0.28 | 0.32 | 0.34 | 0.49 | 0.64 |
| U-real | 0.512 [0.412, 0.612] | 0.64 | 0.52 | 0.28 | 0.32 | 0.34 | 0.49 | 0.64 |
| U-recipe | 0.549 [0.499, 0.598] | 0.70 | 0.53 | 0.35 | 0.32 | 0.39 | 0.43 | 0.66 |
| finetune | 0.431 [0.349, 0.512] | 0.51 | 0.42 | 0.19 | 0.28 | 0.10 | 0.43 | 0.62 |
| joint | 0.670 [0.662, 0.679] | 0.86 | 0.65 | 0.49 | 0.43 | 0.50 | 0.64 | 0.72 |
| knn_eqmem | 0.534 [0.503, 0.565] | 0.69 | 0.48 | 0.34 | 0.28 | 0.42 | 0.46 | 0.64 |
| lda | 0.605 [0.574, 0.637] | 0.84 | 0.58 | 0.56 | 0.40 | 0.58 | 0.40 | 0.52 |
| oracle_experts | 0.604 [0.544, 0.663] | 0.65 | 0.80 | 0.51 | 0.51 | 0.59 | 0.49 | 0.61 |
| replay_eqmem | 0.600 [0.588, 0.612] | 0.78 | 0.60 | 0.40 | 0.39 | 0.46 | 0.48 | 0.67 |
| shipped | 0.531 [0.484, 0.579] | 0.68 | 0.52 | 0.29 | 0.33 | 0.38 | 0.51 | 0.64 |

#### E2: domain-level routing on the six text domains (sequential schedule, seeds 500-502; generated)

| router | accuracy | banking | agnews | newsgroups | emotion | sst2 | svamp |
|---|---|---|---|---|---|---|---|
| pool (U-real routing) | 0.833 (seeds 0.862, 0.868, 0.769) | 0.98 | 0.75 | 0.67 | 0.84 | 0.78 | 0.95 |
| nearest centroid | 0.781 (seeds 0.786, 0.778, 0.779) | 0.98 | 0.64 | 0.46 | 0.86 | 0.88 | 1.00 |
| logistic-regression domain classifier | 0.893 (seeds 0.891, 0.897, 0.891) | 0.99 | 0.72 | 0.86 | 0.91 | 0.80 | 0.99 |
| asking reader (six cards + none) | 0.647 (seeds 0.682, 0.682, 0.576) | 0.88 | 0.84 | 0.10 | 0.76 | 0.48 | 1.00 |

Asking reader says "none of these" to 0.329 of known text messages (false refusals) and to out-of-distribution text: stories 0.863, wiki 0.847, science_questions 0.957.


#### E1 expert quality (oracle experts told the regime, sequential)

```
oracle_linear_s20          seq   final mean 0.696 (seeds 0.691, 0.685, 0.712) experts 22.0 grow events 0.0 params 360382
oracle_linear_s50          seq   final mean 0.694 (seeds 0.688, 0.683, 0.711) experts 22.0 grow events 0.0 params 360382
oracle_mlp128_s20          seq   final mean 0.669 (seeds 0.663, 0.663, 0.679) experts 22.0 grow events 0.0 params 1092542
oracle_mlp128_s50          seq   final mean 0.658 (seeds 0.657, 0.649, 0.667) experts 22.0 grow events 0.0 params 1092542
oracle_mlp512_s20          seq   final mean 0.671 (seeds 0.664, 0.667, 0.684) experts 22.0 grow events 0.0 params 3263678
oracle_mlp512_s50          seq   final mean 0.661 (seeds 0.651, 0.664, 0.669) experts 22.0 grow events 0.0 params 3263678
```

#### E4 experts that start small (sequential)

```
U-deep-h16                 seq   final mean 0.581 (seeds 0.583, 0.612, 0.548) experts 19.7 grow events 0.0 params 412258
U-real-gap0.35-fast        seq   final mean 0.622 (seeds 0.621, 0.632, 0.613) experts 19.3 grow events 0.0 params 960113
U-real-gap0.50-fast        seq   final mean 0.622 (seeds 0.621, 0.632, 0.613) experts 19.3 grow events 0.0 params 960113
U-real-h16                 seq   final mean 0.581 (seeds 0.583, 0.612, 0.548) experts 19.7 grow events 0.0 params 412258
U-real-h16-fast            seq   final mean 0.581 (seeds 0.583, 0.612, 0.548) experts 19.7 grow events 0.0 params 412258
U-real-h512                seq   final mean 0.631 (seeds 0.631, 0.632, 0.630) experts 19.7 grow events 0.0 params 2917530
U-real-no-growth-h16       seq   final mean 0.581 (seeds 0.583, 0.612, 0.548) experts 19.7 grow events 0.0 params 412258
shipped-h16                seq   final mean 0.501 (seeds 0.469, 0.541, 0.494) experts 17.7 grow events 0.0 params 370203
```

#### E6 growth at one inner step

```
U-real-s1                  mix   final mean 0.596 (seeds 0.584, 0.617, 0.586) experts 1.0 grow events 0.0 params 49661
U-real-s1                  seq   final mean 0.591 (seeds 0.587, 0.614, 0.572) experts 18.3 grow events 0.0 params 912086
U-real-s1-fast             mix   final mean 0.593 (seeds 0.588, 0.610, 0.581) experts 1.0 grow events 1.0 params 132218
U-real-s1-fast             seq   final mean 0.589 (seeds 0.584, 0.614, 0.570) experts 18.3 grow events 0.7 params 967124
U-real-s1-h16              mix   final mean 0.442 (seeds 0.448, 0.447, 0.431) experts 1.0 grow events 0.0 params 20877
U-real-s1-h16              seq   final mean 0.564 (seeds 0.570, 0.581, 0.541) experts 18.7 grow events 0.0 params 391338
U-real-s1-h16-fast         mix   final mean 0.478 (seeds 0.488, 0.477, 0.469) experts 1.0 grow events 1.0 params 45866
U-real-s1-h16-fast         seq   final mean 0.563 (seeds 0.570, 0.580, 0.541) experts 18.7 grow events 1.0 params 416327
```

#### E5b growth threshold with fast maturity (mixed)

```
U-real-fast-gap0.17        mix   final mean 0.510 (seeds 0.502, 0.509, 0.519) experts 1.0 grow events 0.0 params 49661
U-real-fast-gap0.35        mix   final mean 0.510 (seeds 0.502, 0.509, 0.519) experts 1.0 grow events 0.0 params 49661
U-real-fast-gap0.5         mix   final mean 0.510 (seeds 0.502, 0.509, 0.519) experts 1.0 grow events 0.0 params 49661
```

#### E7 linear experts inside the pool

```
U-real-linear              drift final mean 0.523 (seeds 0.485, 0.543, 0.541) experts 1.3 grow events 0.0 params 21841
U-real-linear              mix   final mean 0.576 (seeds 0.568, 0.586, 0.574) experts 1.0 grow events 0.0 params 16381
U-real-linear              seq   final mean 0.612 (seeds 0.607, 0.617, 0.613) experts 18.7 grow events 0.0 params 306596
```

## Interpretation (what we understood, and on what basis)
Full text with the ranked shortfalls: `results/unimoe/ANALYSIS.md`. In short. **Works:** the batch-level set gate is the core (removing it costs 0.079 and is all of U-real's +0.076 over the shipped pool; transfer rule 0.038, batch vote 0.029); forgetting 0.046 against 0.604 for fine-tuning; the representation keeps most of the information; the pool sends 0.833 of text messages to an expert of their own domain (0.893 for a classifier trained to name the domain). **Does not:** the pool does not beat equal-memory replay (0.622 against 0.636) or kNN (0.642), the offline joint network reaches 0.669; width and depth growth never fire (the fit-gap statistic depends on the inner-step count, 0.09 / 0.59 / 0.98 at 1 / 5 / 20 steps, and its maturity age of 300 batches exceeds the 162-221 batches of the streams) and the suite barely needs capacity (16 / 128 / 512 units: 0.581 / 0.622 / 0.631); on mixed and drifting streams the pool is one expert (0.596 and 0.512, below replay 0.632 and 0.600); its unknown flag is silent on real out-of-distribution text; arithmetic exact match is low for every learner. **Why:** a ladder of losses (domain-told offline linear 0.752, online MLP experts told the regime 0.669, pool 0.622) and a task-ambiguity floor (offline joint network 9.0 % wrong-domain, pool 10.8 %, almost all among overlapping text domains).

## Caveats and what went wrong
- First round tuned only the sequential schedule and reused the recipe (F-57): report v1's mixed and drifting numbers are superseded by report v2.
- The tuned recipe disabled growth and I had not checked the coupling (F-58); the joint ceiling was first too small and the auto-launched evaluation had to be killed seconds after it started (F-59); depth-zero experts crash the shipped pool, patched in the E7 driver only (F-61).
- Three seeds, dev only; seeds 503-504 and the reserved 600-604 unused. Not measured: vertical growth by layers (no trigger exists), the refusal layer wired into the pool, exact per-draw ROC, Stage 2.
- The repository's best-of-seeds warning on the drivers' seed loops is a false positive (every seed run and reported).

## Consequence
The combined system is a working continual learner whose headline growth and refusal features are not yet demonstrated. Next, in the order of the analysis: recipe-independent growth tested on a capacity-hungry stream; a supervised domain head; the refusal layer with calibration on verified outcomes; expert fit; spawning on stationary streams; a layer trigger. Stage 2 should reuse only the batch-level set test and the transfer rule. Awaiting the user's choice of what to build first.

## Evidence
- `results/unimoe/ANALYSIS.md`, `REPORT_v1.md` (superseded for mixed and drifting), `REPORT_v2.md`, `E2_domain_routing.md`, `tuning.json`, `tuning_mix.json`, `tuning_drift.json`, cells in `runs/`, `runs_v2/`, `p3_*/`
- `research/unimoe/PROGRESS.md` (protocol, every prediction and result in order), `docs/superpowers/specs/2026-10-30-unimoe-design.md` (with amendment 1), `docs/superpowers/plans/2026-10-30-unimoe-stage1.md`

## Reproduce
```
cd D:/shm_um
PYTHONPATH=D:/shm_um python -m unimoe.experiment tune --schedule seq   # then mix, drift
PYTHONPATH=D:/shm_um python -m unimoe.experiment run --seeds 500 501 502 --runs runs_v2 --per-schedule
PYTHONPATH=D:/shm_um python -m unimoe.experiment report --seeds 500 501 502 --out REPORT_v2.md --runs runs_v2
python -m pytest tests/unimoe
```


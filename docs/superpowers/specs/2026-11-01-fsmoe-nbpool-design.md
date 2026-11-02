# FS-MoE: the notebook-first newborn in the real pool (design, 2026-11-01)

The user chose "Port into real pool" after E-060 (the growth lab: experts born as notebooks and right-sized by a held-out test matched 64-unit fixed experts with 61 % of the parameters). This is open question Q-63. Same rules: dev seeds 700-702 for tuning, report seeds 710-714, predictions before any report-seed run, nothing committed, growmoe/ and unimoe/ imported and never edited.

## What is built (`fsmoe/nbpool.py`)
`NbPool(UniPool)`: the shipped pool of growmoe's `PhyloMoE` with the closed-form head in every expert (ULDA), plus one switch `nb_first`:
* **A newborn has no network.** `NotebookExpert` implements the interface the pool uses on an expert's network (`proba`, `__call__`, `clone`, `inherit`, `shrink_perturb`, `state_dict` / `load_state_dict`, `seen`, `train_step`) over a closed-form head, so the gate, the transfer verdict, the spawn path and the accounting run unchanged. Every module the pool hands to `add_expert` is replaced by a notebook.
* **Learning.** A notebook expert's batch is split 85 / 15: the training part goes once into its head and its replay reservoir, the held-out part into a held-out reservoir (counted in the pool's memory). No optimiser steps, no network.
* **A child is primed exactly once** with its founding samples (the shipped pretraining steps would count them many times); a cluster that joins a notebook parent unchanged enters its head once.
* **Judging** (every 10 observe calls, for each notebook expert that is not an infant, has at least 200 replay samples and 60 held-out ones): candidate networks (8 units x 1 block, 8 x 3, 32 x 1, 64 x 1) are trained 200 steps on the expert's replay samples and each is compared, as it would answer (log-linear blend with the notebook, `lda_blend` = 0.5), with the notebook alone on the held-out samples. The notebook stays unless a candidate wins by 2 points; otherwise the cheapest candidate within 1.5 points of the best becomes the expert's network (`rec.module` replaced, optimiser rebuilt, as a growth does; the notebook becomes `meta["lda"]`; the held-out reservoir is freed). A kept notebook is judged again when its data has doubled (never, when its replay is already full).
* **A parent that does worse than chance cannot vouch for a cluster** (`nb_chance` 1.0): the shipped transfer verdict compares the cluster's loss under the parent with the parent's own reference losses, and a notebook or a small network has high ones, so almost any cluster looked "no worse than usual" and was absorbed instead of founding an expert. Dev seeds showed 2-3 regimes lost per run (also in the shipped pool with 8-unit networks); the rule makes such a cluster found a child, as the shipped pool does with 128-unit networks. Only the notebook-first arms carry this rule; the U arms are the shipped pool.
* **Left out on purpose:** notebook experts sit out the shipped merge routine; the width-growth step skips them; it needs `spawn_rule="transfer"` and `growth_rule="gap"` (the shipped defaults), others raise.
* **Differences from the lab (E-060):** the pool's own routing, spawn rule and probation; candidate networks train on the replay reservoir only (256 samples by default, 4,000 in the FS-MoE recipe), not on all of an expert's data; a notebook expert gives up 15 % of its samples to the held-out test until it is promoted.

## Beds and arms
* `f10`, `f30`: growmoe's function-regime stream (the viability-shootout stream; d = 128, 8 classes; each regime a small teacher network, 800 samples per visit, two visits) with 10 and 30 regimes, the pool recipe of that shootout (5 inner steps, q 0.02, alpha_spawn 0.01, m_min 32), accuracy after the whole stream (`run_fast`, final) and pool memory in bytes (`pool_report`: parameters, notebook heads, replay, signatures).
* `g`: the growth lab's mixed-needs world of E-060 (16 regimes on a sphere, 8 of them needing a wide network by the offline oracle, d = 32, 6 classes, 128-sample batches) run through the real pool, accuracy per regime on held test sets, recipe 2 inner steps.
* `t`: the FS-MoE three-domain teaching bed (real text embeddings, 101 skills; taught domain by domain), the tuned FS recipe (1 inner step, replay 4,000, `lda_blend` 0.5), accuracy on known skills in teaching and in held-out wordings, memory.
* Arms: `U128` (the shipped ULDA pool, networks of 128 units at birth, the default), `U32`, `U8` (the same with smaller networks: the honest equal-memory comparison), `NB` (notebook-first). Tuning variants of NB on dev seeds: `NBa` (100 candidate steps), `NBb` (margin 0.01), `NBc` (sizes 8 and 32 only: cheaper judging), `NBd` (first judgement at 100 samples). `NBc` is reported next to `NB` as the cheaper variant.

## Measures
Accuracy (f: mean over regimes after the stream and the worst regime; t: mean over domains of known skills in teaching wording and in held-out wordings), experts, network parameters, memory bytes, multiply-adds per prediction, multiply-adds spent on judging, the share of experts that ended as notebooks and the sizes of the networks given, wall time.

## Dev findings that fix the predictions below (seeds 700-702; `g` and `t` on seed 700 only)
Accuracy means over the three dev seeds, f10 / f30: U128 0.495 / 0.488, U32 0.489 / 0.478, U8 0.370 / 0.288 (it loses regimes: 4-9 of 10 and 12-16 of 30 experts), NB 0.490 / 0.484, NBc 0.499 / 0.487. NB keeps one expert per regime, ends with 1-5 networks among 10 / 30 experts (NBc: none) and has 1-7 % of U128's network parameters (10k-68k against 346k / 1.04M) and 78 % of its bytes on f30 (14.8 against 18.9 MB); judging costs about 1.4 times the wall time of U128 on f30. Teaching bed `t` (seed 700): taught / held-out wording U128 0.966 / 0.897, NB 0.958 / 0.892, 3 notebooks, 10.8 against 12.8 MB. Growth world `g` (seed 700): U8 0.605, U64 0.595, NB 0.613 (16 of 16 experts against 14, 4 networks of 8-32 units), NB's worst regime 0.342 (U8 0.428). Before the transfer fix NB lost 2-3 regimes on two dev seeds (accuracy 0.42 / 0.45).

## Predictions (written before any report-seed run; scored by `fsmoe/scorecard_n.py` on seeds 710-714, means over seeds)
* P1 (f10, f30) NB's accuracy is at most 0.015 below U128's on both beds.
* P2 (f10, f30) NBc's accuracy is at most 0.01 below U128's on both beds.
* P3 (f30) NB's network parameters are at most 5 % of U128's.
* P4 (f30) NB's memory is at most 0.85 times U128's, and NBc's at most NB's plus 0.1 MB.
* P5 (f30) NB's accuracy is at least U32's minus 0.01 with no more memory than U32.
* P6 (f10, f30) NB keeps at least 0.97 experts per regime while U8 keeps at most 0.8 on f30.
* P7 (t) NB's mean accuracy on taught wordings and on held-out wordings are each at most 0.015 below U128's, with at most 0.95 times its memory.
* P8 (g) NB's mean accuracy is at least U128's minus 0.01 with at most 10 % of its network parameters.
* P9 (all beds) at most 30 % of NB's experts end with a network.
* P10 (f30) NB's wall time is at most 1.6 times U128's.
* P11 (f10, f30) NB beats U8's accuracy by at least 0.03 on f10 and 0.1 on f30.
* P12 (g) U128 is not better than U8 by more than 0.02 (the pool trains a network on a batch and 256 replay samples, so capacity cannot be used).

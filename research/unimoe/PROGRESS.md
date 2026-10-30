# UniMoE Stage 1: progress ledger, protocol and predictions

Spec `docs/superpowers/specs/2026-10-30-unimoe-design.md`; plan `docs/superpowers/plans/2026-10-30-unimoe-stage1.md`. Branch `unimoe-stage1`, worktree `D:/shm_um`. A task is done when its line is here.

## Ledger
- 2026-10-30 T1 frontends done (4 tests). T2 domains done (12 tests; seven domains + three OOD sets from the local cache). T3 suite done (8 tests). T4 systems done (5 tests). T5 runner done (8 tests, 6 of 6 mutants caught after two assertions were strengthened). Commits 9e670de, 634713e, b51f02b.
- Design amendment 1: layers of representation experts (`growmoe.stack.StackMoE`) are NOT a variant in Stage 1. The module has the machinery to add a layer but nothing decides when (its growth policy is not wired into `observe`); Stage 1 evaluates depth at the expert level (`U-deep`, the shadow-trial policy that can deepen) and records the missing trigger as a finding for P4.

## Engineering smoke (seed 499, excluded from every result)
Real streams build in 27 s (cached embeddings): banking 3976 train / 1500 test in 6 regimes, agnews 4000 / 1500 in 2, newsgroups 4000 / 1500 in 4, emotion 3907 / 1199 in 2, sst2 4000 / 872 in 1, svamp 700 / 300 in 1, arithmetic 3996 / 1552 in 6 operators: 22 regimes, 125 global classes, 28 phases in the sequential schedule, 12,772 training samples. The shipped pool (inner_steps 5) took 43 s, ended with 12 experts (22 regimes), mean final regime accuracy 0.55, OOD AUROC 0.81-0.86 and an unknown-flag rate of 0 on the OOD sets. I saw these numbers before writing the predictions below; they are the only evaluation numbers I have seen.

## Protocol (fixed now)
- **Seeds.** 499 = tuning only (per-arm recipe grids, sequential schedule, metric = mean final regime accuracy). 500-502 = the first evaluation (all schedules, all systems); 503-504 are added for the robustness block; 600-604 are NOT used.
- **Arms and recipes.** Every trainable arm is tuned on seed 499: UniMoE variants over inner_steps in {1, 5, 20}; `U-recipe` over inner_steps in {5, 20, 50} x replay_size in {256, 1024}; `replay_eqmem` over width in {256, 1024, 4096} x steps in {1, 5, 20} at the memory of the tuned `U-real` pool; `finetune` and `joint` over steps in {5, 20}. The oracle experts use the tuned `finetune` steps. The tuned recipe is then fixed for all evaluation seeds and printed with every table.
- **Reading.** Final mean accuracy = mean over the 22 regimes of the accuracy after the last phase (arithmetic regimes scored per digit slot; exact match per problem is reported separately). Forgetting, transfer, structure, routing, novelty, cost, failure analysis as in the design, section 6. Intervals: 95 % over seeds; with three seeds they are wide and are read as such.

## Predictions (written before the evaluation, 2026-10-30)
Sequential schedule, final mean regime accuracy:
- joint (offline ceiling) 0.68-0.78; oracle experts 0.62-0.78; replay at equal memory 0.52-0.68; U-real 0.52-0.68; shipped 0.48-0.62; kNN at equal memory 0.52-0.66; streaming LDA 0.42-0.60; finetune 0.08-0.25 (it forgets).
- U-real above shipped on the mean over seeds: 70 %, by +1 to +5 points. U-deep within 2 points of U-real, with more parameters: 70 %. U-recipe above U-real by at least 2 points: 65 % (the training recipe is the large lever, C-52).
- The pool beats the equal-memory replay network: 40 % (C-44 and C-76 were won where regimes have different rules; here half the domains are class-incremental text where lookup and one network do well).
Structure:
- Experts at the end: shipped 9-14, U-real 9-16, against 22 regimes: fewer experts than regimes in every seed (85 %); at least one expert holds regimes of two different domains (60 %).
Routing:
- The oracle-routed accuracy among the same experts exceeds the routed accuracy by at least 3 points on the sequential schedule (70 %): routing at answer time loses accuracy, as C-55.
Novelty:
- AUROC of known against OOD p-values 0.75-0.92 on each OOD set; the pool's unknown flag at q = 0.05 finds under 0.30 of OOD messages (80 %), because the per-sample gate is nearly blind to real text categories (C-39); false-novel rate on known data under 0.05 (85 %).
Continual metrics:
- Mean forgetting of the pool 0.03-0.20; finetune above 0.40 (85 %).
Mixed (stationary) schedule:
- The pool founds 1-4 experts only (80 %: C-48, C-53), and its final mean accuracy is within 3 points of the equal-memory replay network (55 %).
Arithmetic:
- Exact match per problem for the pool at the end of the sequential schedule 0.25-0.50 per operator for add, sub, max and under 0.15 for mul, mod, div (65 %).
Shortfalls I expect the report to rank highest (ordered): (1) expert training recipe, (2) routing at answer time, (3) too few experts (domains merged), (4) novelty flags that stay silent, (5) growth decisions that never fire. I will say which of these the data supports and which it does not.

## Finding during tuning (seed 499, sequential; tuning data, not an evaluation result), 2026-10-30
`U-real` (width growth by the fit-gap rule) and `U-deep` (shadow-trial growth, which can also deepen) gave IDENTICAL predictions and identical event logs at every step setting (final mean 0.606 / 0.622 / 0.629; 18-20 spawns, 2-4 extends, no `grow` event of any kind): on 28 phases of real streams neither growth rule ever fired, because the experts start at 128 units and none needs more. So automatic width and depth growth is inert on this suite, as in C-30, C-45 and C-82, and the evaluation cannot say whether it works. Consequence for P3: a scaling test with experts that START TOO SMALL (16 units, where C-86 found the gap rule rescues an expert and the shadow policy does not): `shipped`, `U-real`, `U-deep` at hidden = 16 against fixed widths 16 / 128 / 512, on the sequential schedule, three seeds. Prediction written now: at hidden = 16 the gap rule grows experts (at least one `grow` event per seed) and recovers at least 0.9 of the 128-unit pool's final mean (65 %); the shadow-trial policy does not (it recovers under 0.9: 70 %); neither deepens an expert (85 %). Also confirmed: `extend` events exist (an existing expert absorbs a cluster) and are not growth.

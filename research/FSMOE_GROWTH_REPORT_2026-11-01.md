# FS-MoE growth: experts and a pool that grow as needed, under one governor (2026-11-01)

You asked for experts that grow as needed and a system that grows horizontally and vertically, governed by a proper algorithm. I built it as a lab and tested it (predictions first: 14, of which 8 held). Figure: [FIGURE_GROWTH.png](../results/fsmoe/FIGURE_GROWTH.png). Every number below is copied from a generated table (links at the end).

## The short answer

1. **The part that works is how a new expert is born.** A new expert starts as a *notebook* (the closed-form memory; it answers from the first sample). When it has enough examples, the governor trains four candidate networks (8 units, 8 units x 3 layers, 32 units, 64 units) and tests each, on examples the expert never trained on, against the notebook alone. The notebook stays unless a network wins; otherwise the cheapest network within 1.5 points of the best is kept. With this, **the system matched the best fixed-size experts (64 units: accuracy 0.730 against 0.726) with 61 % of their parameters (56k against 92k) and 57 % of the compute per answer.** In a world where nothing needs a big network it beat the fixed experts (0.902 against 0.861) with fewer parameters and no growth event at all.
2. **Growing an expert later, by trial, added nothing in these tests.** The arm without later growth scored exactly the same (0.730 against 0.730), and starting from small networks and growing by trial ended at the level of never growing (0.672 against 0.669). The growth trial rejected 14 times per run and committed 0.6 times. So "grow as needed" works here as "give each expert the right size when it is born", not as "keep growing it".
3. **More experts (horizontal): works.** One expert per regime (1.00 experts per regime, purity 0.986), and a group that is just an existing expert's own stragglers is absorbed instead of founding a duplicate (12.8 absorptions per run). Regimes in this lab are kept apart on purpose.
4. **Deeper experts: not shown.** Of 9 regimes whose offline test said "needs depth", the governor gave 6 wider networks and 1 a deeper one; wider did most of the job (accuracy 0.630 against 0.558 for the fixed small expert).
5. **A taller system (a coarse layer that groups experts) hurt, and my safety check let it through.** In the big world (48 regimes) the governor built the tier in every seed. It cut the compute per answer by 12 % but lowered the accuracy from 0.774 to 0.756, the worst regime from 0.564 to 0.471, and purity from 0.954 to 0.924. The tier should not be adopted as it is.
6. **Governing is not free.** The trials and promotion tests cost 4.5 times the training itself (29 against 6.5 billion multiply-adds). Brute force (grow everything at every look, no tests) cost 126 and ended at 0.732 with 10 times the parameters; it was also *worse* than fixed experts in the easy world (0.842).

## What was built

A compact lab in `fsmoe/grow/` (the production pool is untouched):
- **A world that demands growth.** Inputs are directions on a sphere. Each regime has its own labelling rule, from three families (linear, "many pieces", "interactions between features"). What a regime needs is *measured* by an oracle: three fixed networks are trained offline on 12,000 of its examples, and the need is "width" or "depth" if a wider or deeper net beats the small one by 5 points. Per seed the mixed world has 8 width-need regimes, 1 to 3 depth-need regimes and 5 to 7 that need nothing.
- **The system.** Experts = notebook + optional network (answer = a 50/50 mix). Routing by the direction of each expert's samples. Samples nobody accepts wait in a room; a dense group becomes an expert, or is absorbed by the expert it sits next to.
- **The governor.** One rule for every direction: grow only on evidence of need, check on unseen data that it helped, keep the cheapest option that works, undo it if it did not, write down why. Four directions: notebook to network (right-sized at birth), wider or deeper (growmoe's own trial, with a guard that undoes a growth that made things worse and a compute budget), more experts (room and absorb), a taller system (a tier of expert groups).
- **Arms:** fixed experts of 8, 16, 32, 64 units and 8 units x 3 layers; the governed system (G); G without later growth; G with a network at birth; G without the tier; brute force.

| arm (mixed needs, 5 seeds) | accuracy | worst regime | total parameters | k multiply-adds per answer |
|---|---|---|---|---|
| fixed, 8 units | 0.669 | 0.506 | 34k | 1.6 |
| fixed, 16 units | 0.690 | 0.537 | 42k | 2.1 |
| fixed, 32 units | 0.711 | 0.572 | 59k | 3.1 |
| fixed, 64 units | 0.726 | 0.595 | 92k | 5.2 |
| fixed, 8 units x 3 layers | 0.673 | 0.523 | 54k | 2.8 |
| **governed (G)** | **0.730** | **0.580** | **56k** | **3.0** |
| G without later growth | 0.730 | 0.580 | 56k | 3.0 |
| G with a network at birth | 0.672 | 0.506 | 38k | 1.8 |
| brute force | 0.732 | 0.623 | 563k | 34.6 |

What the governor gave each regime (by what the oracle says it needs): 85 % of the 40 width-need regimes got 16+ units; 42 % of the 31 regimes that need nothing stayed notebook-only; 23 % of those got a wide network and 10 % a deeper one (the governor's own bar is 2 points; the oracle's is 5, so part of that is a difference in the bars, not a pure mistake).

## What I take from it

- **Port this one thing to the real pool:** notebook-first newborns with a held-out size test. It answers an open problem from the earlier notes (experts that live too short to be judged, right-sizing at birth, Q-41), and it is cheap in memory and in answer-time compute.
- **Do not port the tier or the late growth trial as they are.** The tier needs a check on the *outcome* (held-out accuracy and purity with and without it), not only agreement of routing on held-out samples. The late growth trial needs a different test (the birth test works; the 240-step shadow trial barely fires).
- **Depth is untested, not refuted:** the lab had only 9 regimes that need it, and width covered them.
- **Compute:** governing multiplied the training cost by about 5; for a system that is trained once and answers many times that can pay (the answer-time cost is nearly halved against the 64-unit experts), but it is a real cost and the budget knob (`trial_ratio`) matters.

## Caveats

- This is a lab: the regimes are separated, labels come with every sample, routing is by centroid cosine, not the production pool's conformal gate and vMF signatures. It isolates growth decisions from the (separate) near-twin problem on purpose.
- Three design repairs came from dev runs before any report seed: the room had to extract dense groups, a group near an existing expert had to be absorbed, and regimes had to be kept apart. The governor's knobs were tuned on dev seeds 700-702 (a cheaper setting that matched the best accuracy was chosen).
- Hit rate: 8 of 14 predictions held. Missed: depth (P5), false growth on regimes that need nothing (P6), "network at birth with the governor is as good" (P9: it is as bad as fixed small experts), the big-world gain (P11, +0.014), "the tier is refused" (P12: it was built in every seed) and purity under the tier (P13).
- I said earlier that this governor would settle the held decision on the "gap rule" for width; it does not: the lab uses growmoe's shadow-trial policy, so that decision is still open.
- About the GPU: these networks are so small that one training step took 1.0-1.8 ms on 2 CPU threads and 1.4-2.3 ms on the GPU (measured), so the runs used CPU cores in parallel.

## What I would do next (your call)

1. **Wire the notebook-first newborn with the size test into the real pool** and test it on the real regimes (the lab says it should help; the real pool has a conformal gate and a probation buffer that already do part of this).
2. **Repair the tier's check** (an outcome test) on dev seeds, then re-test; or drop the tier until the gate is expensive enough to need it.
3. **Build a world where depth is needed** (compositions) so that deeper experts can be tested properly.

## Evidence

[REPORT_G.md](../results/fsmoe/G/REPORT_G.md), [SCORECARD_G.md](../results/fsmoe/SCORECARD_G.md), [FIGURE_GROWTH.png](../results/fsmoe/FIGURE_GROWTH.png), design and predictions [2026-11-01-fsmoe-growth-design.md](../docs/superpowers/specs/2026-11-01-fsmoe-growth-design.md), notebook entry E-060; code `fsmoe/grow/` (`world.py`, `system.py`, `governor.py`, `lab.py`, `report_g.py`, `scorecard_g.py`, `figures_g.py`); tests `tests/fsmoe/test_fs_grow.py` (31 tests, 25 mutants killed).

# What the pool of experts shows: size, growth, forgetting, cost (one page, 2026-11-02)

Figure: [SUMMARY_FIGURE_2026-11-02.png](summary/SUMMARY_FIGURE_2026-11-02.png) (script `summary/make_summary_figure.py`; panel B is recomputed from `results/fsmoe/G`, the rest is copied from the lab notebook, each number tagged with its claim or entry). Everything below is small networks on synthetic or sentence-embedding streams, grade B unless marked A.

## Your three questions

| Question | Answer |
|---|---|
| Is 64 units the minimum for a good expert? | **No.** The best size depends on the regime. In the growth lab 85 % of the regimes that need width got 16+ units, and 42 % of the regimes that need nothing stayed a closed-form notebook with no network at all (C-160). On the text bed 0.2 of 3 experts had a network (E-061). 64 was the best of the fixed sizes, not a minimum. |
| Does growing the architecture as needed beat predefined sizes? | **Growing the number of experts: yes. Growing each expert later: not measured to help. Right-sizing at birth: a small, steady gain and a memory saving.** Details below. |
| Gains in continual learning, forgetting, efficiency? | Forgetting: yes, measured. Answer-time compute: yes. Training time: depends on the baseline (cheaper than a same-memory single network, dearer than fixed-size experts once governing is added). Memory: a modest saving. Details below. |

## Growth, by kind

- **More experts as regimes arrive (works).** The pool beats the best single network of the same memory by +6.2, +9.1 and +13.0 points at 10, 30 and 100 regimes (C-76, dev seeds), and it founds one expert per regime (1.00, C-159). A pool with a fixed number of slots that drops new regimes once full answers them at chance (about 25 %); evicting or sharing instead scores 66, 65 and 59 against 51, 48 and 53 (C-73, exploratory).
- **Right size at birth (small gain).** A newborn starts as a closed-form notebook and gets a network only if it wins a test on unseen examples. Against a fixed 32-unit expert at the same memory: +0.019 (±0.005, 5 of 5 seeds). Against a fixed 64-unit expert: +0.005 (±0.005, no difference) with 61 % of the parameters. In the real pool: same accuracy, 14-22 % less memory than a pool born with 128 units on three of four beds (0.5 % on the growth-lab world; C-170, E-061), but only about 7 % less than one born with 32 (14.8 against 15.9 MB on 30 regimes), and its worst regime is lower in the growth-lab world (0.337 against 0.408, C-169).
- **Growing an expert after birth (idle).** Starting at 8 units and growing later scores 0.672, the fixed 8-unit score is 0.669; the gain of the whole governor comes from the birth test alone (C-158). On real suites capacity is not the limit: 16, 128 and 512 units score 0.581, 0.622, 0.631 (C-115), so these suites cannot show whether automatic scaling pays. The one clear use: an expert that starts too small (16 units) grows to a mean of 124 units and 15 of 15 seeds reach 0.9 of the 128-unit expert, where the older policy managed 6 of 15 (C-86, A). It is insurance against a bad starting size, and it is silent at 128 units. Deeper experts are not demonstrated (C-161).
- **A taller system of expert groups (hurt).** Accuracy 0.756 against 0.774 without it on 48 regimes, and the check that allowed it was too weak (F-81).

## Forgetting (points of accuracy lost on earlier operators; lower is better)

Plain network 55.1; network plus 20,000 stored examples 28.9; pool as shipped 15.2; pool with the adopted transfer rule and router 7.1 (pre-registered, fresh seeds, A); experts told the operator 0.0 (not a learner). On real text the closed-form-head pool shows no forgetting, but equal-memory replay, nearest-neighbour and streaming LDA do not forget either (0.965 against 0.952, 0.934, 0.945; C-131): no advantage there. The pool's advantage is where the answer must be computed, because lookup fails (kNN 21-26 % against 45-78 %, C-16).

## Efficiency

- **Cheaper at answer time:** 0.08-0.20 of the multiply-adds of the best same-memory single network (C-42, dev seeds). This depends on the fingerprint shortlist, which cut gate multiply-adds to 0.09-0.16 of the exact gate on fresh seeds with accuracy within 0.1 point (C-25, A) and by default acts only from 64 experts; below that the gate is 97 % of the work and the pool was no cheaper than one network (C-21).
- **Training time depends on the baseline.** Against a single network of the same memory, the pool's whole run (training plus one final evaluation, same runner) took 7, 22 and 133 s against 20, 99 and 526 s at 10, 30 and 100 regimes, i.e. 0.22-0.35 of the time (E-025, dev seeds, one machine, before the router was adopted). Against fixed-size experts the governed variants cost more: governing is 4.5 times the training multiply-adds (35.5 against 20.8 for fixed 64), the notebook-first pool took 1.79 times the wall time of a pool born with 128 units (1.17 with two candidate sizes), and adopting the router lifted the demo's pool phase from 5-6 s to 30-31 s (C-90); the skip flag saves at most about 7.5 % of a run (E-063). The older shootout time has not been re-measured with today's default stack.

## Corrections to what I told you earlier

1. I said compute savings were "never established" and that the 0.08x figure "predates" the gate finding. That was wrong: the shortlist is what repaired the gate cost, and C-25 and C-42 stand (dev seeds for the 0.08-0.20 ratio, fresh seeds for the gate).
2. Notebook claim C-25 says the shortlist cuts the gate "7-14x"; its own table gives 986.3 to 88.5, 662 to 97 and 493 to 77 (ratios 0.090, 0.147, 0.157, i.e. 6.4 to 11 times). The "7-14x" was the gatekeeper research prediction carried into a measured claim. Corrected in the notebook on 2026-11-02 (C-25, F-107); the same figure remains in the 2026-10-30 UniMoE design note, left as written.

3. My first summary said "training compute: no, it goes up". That compared governed variants with fixed-size experts only; against a same-memory single network the pool's whole run was faster (E-025). Both hold; the baseline decides. Fixed in the table, the efficiency section and figure panel D on 2026-11-02.

## Not shown

Any scale above small networks and 384-dimension embeddings; whether automatic expert sizing pays on a task where capacity truly limits; deeper experts; token-level routing; independent replication (one lab, no outside reviewer).

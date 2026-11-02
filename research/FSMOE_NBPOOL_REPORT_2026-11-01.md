# FS-MoE: the notebook-first newborn in the real pool (2026-11-01)

You said "port into real pool". I ported the notebook-first newborn (E-060) into the real pool as a switch, tested it on four beds (predictions first: 12, of which 9 held), and found one thing that matters more than the port itself. Figure: [FIGURE_NBPOOL.png](../results/fsmoe/FIGURE_NBPOOL.png). Every number below is copied from a generated table (links at the end).

## The short answer

1. **It works in the real pool.** On all four beds the notebook-first pool (NB) scores within noise of the shipped pool (128-unit networks, the default, U128): 10 function regimes 0.497 against 0.493; 30 function regimes 0.482 against 0.485; the text bed (real embeddings) 0.961 / 0.890 against 0.965 / 0.895 on known skills in teaching and in new wordings; the growth-lab world 0.627 against 0.607. It does so with **about 1 % of the network parameters** (11k against 1,037k on 30 regimes) and **14-22 % less memory** (14.8 against 18.9 MB on 30 regimes; 11.0 against 12.9 MB on the text bed). It finds one expert per regime, like the shipped pool.
2. **But the size test itself added nothing I can measure.** Pools with notebook experts only (never judged, no network at all) score the same: 0.493 / 0.481 / 0.620 / 0.926 against NB's 0.497 / 0.482 / 0.627 / 0.925 (mean of the two text-bed scores for the last one). The test gave a network to only 2 % (10 regimes), 4 % (30 regimes) and 7 % (text bed) of the experts; in the growth-lab world, where half the regimes need width, 34 %, and the gain there (+0.007 over notebooks only) is within the noise. In this pool the closed-form notebook does the work; the networks inside experts are almost unnecessary on these beds.
3. **The most useful thing I found is a flaw in the shipped pool, and a one-line fix.** Shrinking the networks breaks the shipped pool: with 8-unit networks it lost more than half of the regimes (0.45 experts per regime on 30 regimes) and fell to 0.293. The cause is the spawn test: it compares how badly a parent predicts a new cluster with the parent's own past losses, and a weak parent has high past losses, so almost any cluster looks "no worse than usual" and is absorbed into it. The rule "a parent that does worse than chance on a cluster cannot vouch for it" fixes it: the 8-unit pool then keeps one expert per regime and scores **0.471 against 0.293**, and the default pool is unchanged within noise on the function-regime streams (0.478 against 0.485). **Correction (E-062, same day):** on the real 7-domain suite the rule is not free for strong pools (drifting schedule 0.6236 against 0.6423; sequential 0.6343 against 0.6454), so it is now a flag, off by default; see the chance-flag report.
4. **Judging is not free.** Training the candidate networks costs 93 billion multiply-adds on 30 regimes and **1.79 times the wall time** (147 s against 82 s). A cheaper variant (candidate sizes 8 and 32 only, 31 billion) costs 1.17 times and scores the same (0.479).
5. **Not good everywhere:** in the growth-lab world the worst regime is lower with NB (0.337 against 0.408 for the default), and a network does help a little there (the default beats the 8-unit pool by 0.039).

## What was built

`NbPool` in `fsmoe/nbpool.py`: the shipped pool with a switch `nb_first` (growmoe and unimoe are not edited).
- **A newborn expert has no network.** It is a *notebook* (the closed-form head the pool already keeps in every expert) wearing the interface of a network, so the gate, the spawn test and the accounting run unchanged.
- **15 % of a notebook expert's samples are held out** (150 kept), never trained on.
- **The size test:** every 10 batches each notebook with 200+ training samples trains candidate networks (8 units, 8 units x 3 layers, 32 units, 64 units) and compares each, as it would answer (blended with the notebook), with the notebook alone on its held-out samples. The notebook stays unless a candidate wins by 2 points; otherwise the cheapest one within 1.5 points of the best becomes the expert's network.
- **The worse-than-chance rule** in the spawn test, and notebook experts sit out the merge routine.

| bed (5 seeds) | arm | accuracy | experts per regime | network parameters | memory MB |
|---|---|---|---|---|---|
| 30 function regimes | shipped, 128 units (default) | 0.485 | 1.00 | 1,037,040 | 18.9 |
| | shipped, 32 units | 0.479 | 1.00 | 296,880 | 15.9 |
| | shipped, 8 units | 0.293 | 0.45 | 50,701 | 9.0 |
| | **notebook-first** | **0.482** | **1.00** | **11,493** | **14.8** |
| | notebook-first, sizes 8 and 32 | 0.479 | 1.00 | 20,077 | 14.8 |
| growth-lab world (16 regimes) | shipped, 128 units | 0.607 | 0.89 | 133,895 | 3.8 |
| | shipped, 8 units | 0.568 | 0.82 | 12,961 | 3.1 |
| | **notebook-first** | **0.627** | **0.93** | **11,983** | **3.8** |
| text bed (3 experts) | shipped, 128 units | 0.930 | 1.00 | 278,703 | 12.9 |
| | **notebook-first** | **0.925** | **1.00** | **6,271** | **11.0** |

Most of the pool's memory is not the networks: replay samples, signatures and the closed-form heads are the bulk, so removing 99 % of the network parameters saves 14-22 % of the bytes.

## What I take from it

- **Safe to offer as a switch for memory-limited use, not as a default.** It matches the shipped pool, but its main effect is to show that the networks are nearly idle here.
- **Adopt the worse-than-chance rule on its own** (it needs a change in growmoe's spawn test, behind a flag): it repairs small-network pools; its cost to the default pool was later measured on real streams and is not nothing (E-062). This is the piece I would put forward first.
- **The size test needs a bed where networks can use their capacity.** The shipped pool trains each network on one batch plus 256 stored samples, so even 128 units can barely beat 8 units here. In the growth lab (E-060) the networks had thousands of samples and right-sizing helped; moving that into the real pool would also need a larger training buffer.

## Caveats

- The function-regime beds are data-limited (800-1,600 samples per regime), the text bed has only 3 experts, and I did not test the deployment stream or the 7-domain streams of the earlier rounds.
- Hit rate: 9 of 12 predictions held. Missed: networks at most 30 % of experts (34 % in the growth world), wall time at most 1.6 times (1.79), and "bigger networks do not beat 8 units by more than 0.02" in the growth world (0.039).
- The ablation (the worse-than-chance rule alone, and notebooks only) was added after the first results; it is exploratory. The notebooks-only arm carries held-out buffers it never uses, so its memory (16.6 MB on 30 regimes) is higher than it needs to be.
- Dev runs changed the design twice: the first version lost 2-3 regimes per run on two seeds (the flaw above), and held-out buffers are now released once a notebook can never be judged again.

## What I would do next (your call)

1. **Put the worse-than-chance rule into growmoe's spawn test behind a flag** and run the standard checks on the regimes of the earlier rounds.
2. **Test notebook-only experts properly** (no held-out buffers) on the deployment stream and the 7-domain streams, since they are the cheapest setting that matched the shipped pool.
3. **Give the size test a bed where networks can learn** (a larger training buffer per expert), so that right-sizing can be judged in the real pool.

## Evidence

[REPORT_N.md](../results/fsmoe/N/REPORT_N.md), [SCORECARD_N.md](../results/fsmoe/SCORECARD_N.md), [FIGURE_NBPOOL.png](../results/fsmoe/FIGURE_NBPOOL.png), design and predictions [2026-11-01-fsmoe-nbpool-design.md](../docs/superpowers/specs/2026-11-01-fsmoe-nbpool-design.md), notebook entry E-061; code `fsmoe/nbpool.py`, `fsmoe/nbpool_lab.py`; tests `tests/fsmoe/test_fs_nbpool.py` (18 tests, 16 mutants killed).

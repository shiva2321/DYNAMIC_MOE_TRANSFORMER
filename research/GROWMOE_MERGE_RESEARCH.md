# GrowMoE: what to do about wrongly created experts (research, evidence, options) - 2026-10-16

You asked on 2026-10-16: the merge routine cannot repair a wrong spawn, so a wrongly created expert is permanent; that could make or break the system and fail silently; do deep research, synthesis and
brainstorming, say what can be done, and you will decide. This note answers that. It is written to be read top to bottom: plain words first, the technical statement after them, every number generated from a file in
`research/growmoe_results/merge_study/` (built by `build_report.py`). Nothing in the product code was changed to produce it; the stage-4 dev runs used the same code the whole time.

**Read first: this note corrects itself twice, and it found a problem that is bigger than the question.**

1. *My first end-to-end study was wrong (retracted).* It injected the false spawns at the end of every regime and reported a 20-point loss. A newborn expert accepts every input for its first five batches, so the pool
   never saw the next regime as new, founded only 1 natural expert of 12, and the "duplicates" became the regime experts (`healing_diag.py`, `healing.json`). That number is withdrawn.
2. *My second version compared against a clean pool with a built-in head start.* The function-regime test bench draws the labels of regime r from a teacher network seeded `stream_seed * 1000 + r`, and the pool founds
   expert k with weights seeded `pool_seed * 1000 + k`. With pool seed = stream seed, the expert founded for regime r starts with the first hidden units of regime r's teacher, worth 4 to 7 points, but only to pools whose
   expert ids line up with the regimes, that is, pools without false spawns. That made false spawns look two to three times as costly as they are. Every number below uses pool seed = stream seed + 7, and a control arm
   (`idshift`: the expert ids advance as if a spawn had happened, nothing is founded) now equals the clean pool within noise.
3. *The same head start sits in the project's own function-regime streams*, and therefore in the stage-4 dev runs and in earlier results. It is measured in section 6 (about 8.0 points on F-15, S3 arm) together with the decision it
   needs from you. It does not touch the arithmetic and real-text results, which carry the main findings of stage 4.

## 0. The answer on one page

The worry is justified. The accuracy damage of one wrong expert is modest, about 4 points; what makes it dangerous is that it is permanent, that it uses a slot for ever, and that the system gives no signal when slots or
detection run out.

1. **A wrong expert is permanent.** The only code that can delete an expert is the merge routine (`growmoe/pools/phylo.py:471`). Across 20 end-to-end runs it merged 8 times against 296 injected false
   spawns; it removed 0 of 4 in the earlier audit and 0 of 42 constructed pairs, true duplicates included. The cause is statistical, not a threshold: it compares experts that have memorised their stored samples with a
   clone that never trained on half of them (section 1.3).
2. **A false spawn costs about 4 points and one slot.** Against the same pool without false spawns: 3.7 points with one false spawn per regime, 4.3 with two per regime, 4.3 when regimes are twice as
   long, 2.7 when the false spawn arrives after the original has seen 11 batches (12 regimes, 5 seeds). The accuracy cost does not grow with the number of duplicates; the expert count does.
3. **A full pool silently stops learning.** With a limit of 16 experts and 12 false spawns, the shipped behaviour (drop the cluster, write nothing) left 4 of 12 regimes without an expert and answered them at
   25 % (chance is 25 %): 50.7 % overall against 67.4 % for the same stream without a limit.
4. **A pool can also miss whole regimes with no false spawn at all.** A young expert accepts every input for five batches (that is what stops a spawn cascade); on regimes of three batches the pool founded an expert of its
   own for 1, 11 and 8 of 12 regimes (three seeds), and with that protection switched off it founds one expert per batch (121, 121 and 69 experts for 12 regimes) (section 2.6).
5. **Repair works when the evidence is honest and plentiful.** Scoring the survivor on live traffic that nobody has trained on yet ("rival tracking", no sample withheld) removed 11.2 of 12 false spawns at
   15 batches per regime and 11.2 at 20, and recovered 4.1 of the 4.3 points lost at 20 batches; at 10 batches it removed only 2.0. Held-out samples (1 in 8 withheld
   from training) do better at 10 batches (10.6 of 12) but cost 4.4 points when three experts share a regime. A label-free agreement test removed 11.8 of 12 at a threshold fitted to this task
   and 0.0 at the default 0.8. Of 968 merges of every kind, 0 joined experts of different regimes, in a bench whose regimes are far apart, so this says little about overlapping regimes.
6. **Merging frees slots; accuracy comes back only if the survivor holds the knowledge.** With two false spawns per regime the label-free test merged 23.4 of 24 and accuracy stayed at 62.8 % (nothing
   done: 63.1 %), whereas averaging the answers of overlapping experts at answer time gave +3.7 points (+1.5 with one false spawn per regime) and changed nothing on a pool without twins (difference
   0.00 points).
7. **When the pool is full, evict the most redundant expert, otherwise force the cluster into the nearest expert.** That policy scored 66.1 % / 59.0 % / 64.6 % in the three cap settings of
   section 2.5, against 50.7 % / 53.4 % / 48.3 % for the silent drop.

**What I recommend** (sections 4-5): (i) make each silent failure loud (small, behaviour-neutral); (ii) replace the silent drop by evict-else-share, which changes nothing until the cap is reached; (iii) build repair in a
later, separately pre-registered stage on live-traffic evidence (free, safe in these tests, needs regimes of about 15 batches or more) and add answer-time averaging of overlapping experts; (iv) do not build a held-out
reservoir, weight merging or an uncalibrated label-free merge as the repair; (v) let the dev runs say how often stage 4b still makes false spawns before deciding how much repair is needed; (vi) decide what to do about the
head start (section 6) before the dev review.

**What I need from you:** six decisions, listed in section 5. Nothing here has been implemented in the product.

## 1. The problem in plain words

### 1.1 How experts are born, and how they can die

The pool answers a question by sending it to one expert (a small neural network) chosen by a gate that knows, for every expert, what its inputs looked like (its **signature**: up to 256 stored example inputs, called
**prototypes**). Input the gate cannot place is called **novel**. Novel samples wait in a buffer; when 32 of them form a cluster, the pool decides between two things: **extend** the best matching expert (train it on the
cluster) or **spawn** a new expert. Until stage 4 that decision was made by a *probe* (train a copy on the cluster and see whether the old data suffer); with the stage-4b flag it is made by a label test. A wrong decision to
spawn is a **false spawn**: the new expert duplicates one that already knows the function.

Experts move through states (infant for 5 batches, active, dormant after 400 idle batches, archived after 800; an archived expert drops its stored samples and its optimiser but stays in the pool and is still scanned by
the gate). None of these states deletes anything. The only call that removes an expert from the pool is in the merge routine (`phylo.py:471`; the other pool type, `keyed.py:221`, is not used here).

### 1.2 What the merge routine does

Every 50 batches it looks at pairs of experts whose signatures cover each other by at least 20 %, trains a *clone* of one of them for 60 steps on the stored samples of both, and merges only if the clone loses at most
2 accuracy points on each expert's own validation half of its stored samples. Experts without stored samples (every archived one) and infants are never candidates.

### 1.3 Why it can never say yes

Both experts have *trained on* their stored samples (0.88-0.94 accuracy on them in the audit); the clone has not trained on the validation half. So the clone looks worse on exactly the samples the experts memorised, by
6-9 points, against a tolerance of 2. On constructed pairs the shipped test accepted 0 of 42 pairs, true duplicates included (column T0 below). Fixing the bias by testing a clone on honest data (T1) then says yes to 83 % of
duplicates, but also to 83 % of pairs where one expert is simply worse, and to 67 % of pairs that differ on 5 % of the input space: it is not safe either. The flaw is the mirror image of the old spawn probe's: both validate on
samples the experts have already seen. The same trap is described for budgeted learners (Wang, Crammer and Vucetic 2012, section 2.4: validating a removal on the stored examples themselves "could lead to misleading
accuracy estimation").

| pair kind | T0 shipped test | T1 clone on honest data | T2 agreement on stored prototypes >= 0.80 | T5 retire B | T5 retire A | T5 both directions |
|---|---|---|---|---|---|---|
| dup_ancestry | 0 % | 83 % | 100 % | 100 % | 100 % | 100 % |
| dup_independent | 0 % | 83 % | 100 % | 100 % | 100 % | 100 % |
| dominated | 0 % | 83 % | 33 % | 100 % | 0 % | 0 % |
| distinct_region | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| distinct_perm | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| partial20 | 0 % | 17 % | 17 % | 0 % | 50 % | 0 % |
| partial5 | 0 % | 67 % | 100 % | 100 % | 100 % | 100 % |

*Reading the table.* Each column is a candidate test; each cell is the share of pairs for which the test says "merge". T0 = the shipped test. T1 = a clone trained on both experts' data, judged on honest fresh data.
T2 = the two experts' answers agree on at least 80 % of the stored prototypes (no labels needed). T5 = **directional transfer**: the survivor is scored on the retiree's stored samples against the retiree's own honest
loss (one-sided test with a margin). Wanted: duplicates yes; "dominated" (one expert is simply worse): retire the worse one only; everything else no. T5 is the only test that gets the table right.

### 1.4 What a wrong expert does

| harm | what happens | measured here |
|---|---|---|
| it uses a slot | the pool has a hard cap (`max_experts`), and an expert never leaves | section 2.5 |
| it splits the data | two experts now learn one regime from parts of its data, and the newer one wins the routing tie and answers about two thirds of the questions | section 2.3 |
| it blinds the gate | a young expert accepts every input for 5 batches, so a regime change in that window is absorbed instead of detected | section 2.6 |
| it costs compute and memory | the gate scans every expert's signature; each expert keeps up to 256 prototypes and a replay reservoir | not measured; linear in the number of experts |
| it confuses the router | stage 4a chooses its rule by cross-validated owner accuracy; two experts with the same inputs are indistinguishable owners | not measured |

### 1.5 The cap is silent

When the pool is full, the buffer of novel samples is not even clustered (`phylo.py:668` and `:693`), and if a cluster does arrive, `_handle_cluster` returns at line 250. No event is written and no counter moves. The
novel samples are assigned to nobody, so nobody trains on them, and at answer time they are still sent to the best-matching expert, which does not know them. The default cap is 256, far above anything the
benchmarks reach (the stage-3 and stage-4 dev streams end with 1 to 30 experts), so this is a long-deployment risk; it is also the cheapest of the failures to make loud.

## 2. What we measured

All end-to-end numbers come from `healing2_study.py` (12 function regimes, 4 classes, 32 inputs, experts of 32 hidden units, batches of 64, two revisits at the end; the stage-3 pool with the stage-4b transfer test on; pool
seed = stream seed + 7). Tables are generated by `report_healing2.py`, `run_study.py report` and `build_report.py`; "+-" is the standard error over 5 seeds. False spawns are injected by handing a batch of a regime the pool
already knows to the spawn routine with the extend option switched off, the way the merge audit forced one; the new expert is born a clone of its parent (lambda = 1.0).

### 2.1 Which merge tests separate duplicates from different experts (constructed pairs)

Seven kinds of pairs, six seeds each, experts of 32 hidden units trained as the pool trains them (`RESULTS_MERGE_STUDY.md`). The directional test T5 needs a margin, and the margin is the dial that trades missed
duplicates against wrong merges:

| pair kind | direction | delta 0.05 | delta 0.10 | delta 0.15 | delta 0.20 | delta 0.30 | delta 0.50 | delta 1.00 | median upper bound |
|---|---|---|---|---|---|---|---|---|---|
| dup_ancestry | retire B | 0 % | 17 % | 50 % | 100 % | 100 % | 100 % | 100 % | 0.15 |
| dup_ancestry | retire A | 17 % | 50 % | 83 % | 83 % | 100 % | 100 % | 100 % | 0.09 |
| dup_independent | retire B | 17 % | 83 % | 83 % | 100 % | 100 % | 100 % | 100 % | 0.07 |
| dup_independent | retire A | 50 % | 50 % | 83 % | 83 % | 100 % | 100 % | 100 % | 0.09 |
| dominated | retire B | 0 % | 17 % | 67 % | 67 % | 100 % | 100 % | 100 % | 0.13 |
| dominated | retire A | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 17 % | 1.91 |
| distinct_region | retire B | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 2.62 |
| distinct_region | retire A | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 3.81 |
| distinct_perm | retire B | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 4.83 |
| distinct_perm | retire A | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 4.97 |
| partial20 | retire B | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 50 % | 0.99 |
| partial20 | retire A | 0 % | 0 % | 0 % | 0 % | 17 % | 67 % | 100 % | 0.44 |
| partial5 | retire B | 0 % | 0 % | 17 % | 33 % | 67 % | 100 % | 100 % | 0.25 |
| partial5 | retire A | 0 % | 17 % | 50 % | 83 % | 100 % | 100 % | 100 % | 0.15 |

Between 0.2 and 0.3 nats (a loss 0.2 nats higher means the survivor gives the right label about 18 % less probability on average) duplicates are accepted and the "partial" pairs mostly are not. The partial pairs
differing on 20 % of the input space are the hard case: retiring the expert that knows more costs 13.6 accuracy points (section 4 of `RESULTS_MERGE_STUDY.md`) and a loose margin accepts it.

### 2.2 The same test on real experts

The saved operator-by-operator arithmetic pool (6 experts of 512 hidden units, one per operator):

| quantity | duplicates | different operators |
|---|---|---|
| upper bound of the excess loss (nats): min / median / max | 0.04 / 0.20 / 0.38 | 4.23 / 7.59 / 141526585 |
| agreement of the two experts' answers: min / median / max | 0.60 / 0.80 / 0.86 (on fresh inputs) | 0.22 / 0.30 / 0.49 (on stored prototypes) |

| margin delta (nats) | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 1.00 |
|---|---|---|---|---|---|---|---|
| duplicates accepted (of 12 directions) | 1 | 3 | 5 | 6 | 10 | 12 | 12 |
| different operators accepted (of 30 pairs) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

On real experts the separation is wide: duplicates stay under 0.4 nats, experts for different operators sit above 4 nats. A label-free agreement threshold is much weaker: duplicates of the hard operator (multiplication)
agree only 0.60, while experts for different operators agree up to 0.49. So agreement without labels needs a per-task calibration before it can be trusted, and it cannot tell "equal" from "one is worse".

### 2.3 What false spawns cost end to end

| stream | experts with false spawns | accuracy without false spawns | with false spawns | accuracy lost (paired, +-) | share of the questions answered by a false spawn |
|---|---|---|---|---|---|
| 10 batches, 1 per regime | 24 | 67.4 +-0.9 | 63.7 +-0.9 | 3.7 +-0.7 | 65 % |
| 15 batches, 1 per regime | 24 | 69.5 +-0.7 | 65.6 +-0.5 | 4.0 +-1.0 | 66 % |
| 20 batches, 1 per regime | 23 | 70.8 +-0.7 | 66.5 +-0.8 | 4.3 +-0.6 | 68 % |
| 20 batches, false spawn at batch 12 | 24 | 70.8 +-0.7 | 68.2 +-0.5 | 2.7 +-0.4 | 40 % |
| 10 batches, 2 per regime | 35 | 67.4 +-0.9 | 63.1 +-0.5 | 4.3 +-0.7 | 61 % |

The cost is about 4 points whatever the number of duplicates or the length of the regime, and smaller (2.7) when the false spawn is a clone of an original that has already seen 11 batches. Mechanism (`twin_diag.py`,
seed 0; "N" is the original, "D" the false spawn; the columns are N alone, D alone, the pool's routed answer, the average of N and D, either of them right, and the share of questions routed to D):

```
n_train 640 seed 0: per regime  N alone | D alone | routed | N+D average | oracle(either right) | share routed to D | samples assigned N / D
mean         56.3 |  66.6 |  64.1 |  64.9 |  77.3 |   64 %
n_train 1280 seed 0: per regime  N alone | D alone | routed | N+D average | oracle(either right) | share routed to D | samples assigned N / D
mean         54.6 |  71.3 |  68.6 |  66.6 |  80.3 |   74 %
n_train 1280 seed 0: per regime  N alone | D alone | routed | N+D average | oracle(either right) | share routed to D | samples assigned N / D
mean         71.6 |  64.2 |  68.4 |  68.8 |  80.5 |   44 %
```

D ends up with most of the regime's traffic, so N is starved at about three batches of data; either expert is right on 77-81 % of the questions, but the pool sends each question to one of them. A
false spawn born late, when N is mature, is the weaker of the two (D alone 64 %, N alone 72 %) and still answers 44 % of the questions.

### 2.4 Which repairs work end to end

Each cell: false spawns removed out of the false spawns injected; accuracy change against doing nothing (paired over seeds, +- standard error). "10 batches" and "15 / 20 batches" are the length of a regime; the false
spawn comes at batch 3 except in the column "batch 12". The number injected varies a little between arms because an injection is skipped when the regime has no expert of its own yet (the held-out arm, which withholds data,
founds a few experts later).

| | 10 batches, 1 per regime | 15 batches, 1 per regime | 20 batches, 1 per regime | 20 batches, false spawn at batch 12 | 10 batches, 2 per regime |
|---|---|---|---|---|---|
| accuracy without false spawns (points) | 67.4 | 69.5 | 70.8 | 70.8 | 67.4 |
| accuracy with false spawns, nothing done | 63.7 | 65.6 | 66.5 | 68.2 | 63.1 |
| experts without / with false spawns | 12 / 24 | 12 / 24 | 12 / 23 | 12 / 24 | 12 / 35 |
| the shipped merge: false spawns removed; accuracy change vs doing nothing | 0.0 of 11.8; +0.0 +-0.0 | - | 0.4 of 11.8; +0.5 +-0.4 | 0.0 of 12.0; +0.0 +-0.0 | 1.2 of 23.6; -0.0 +-0.4 |
| prequential references: false spawns removed; accuracy change vs doing nothing | 0.8 of 11.8; +0.0 +-0.1 | - | 0.8 of 12.0; +0.2 +-0.3 | - | 6.0 of 23.2; -0.3 +-0.3 |
| label-free, agreement >= 0.8: false spawns removed; accuracy change vs doing nothing | 0.0 of 11.8; +0.0 +-0.0 | - | - | - | 0.0 of 23.2; +0.0 +-0.0 |
| label-free, agreement >= 0.5 (fitted to this task): false spawns removed; accuracy change vs doing nothing | 11.8 of 11.8; +3.9 +-0.4 | 12.0 of 12.0; +4.4 +-0.7 | 12.0 of 12.0; +4.4 +-0.6 | - | 23.4 of 23.6; -0.3 +-0.5 |
| rival tracking on live traffic (no sample withheld): false spawns removed; accuracy change vs doing nothing | 2.0 of 11.8; +0.8 +-0.3 | 11.2 of 11.8; +4.4 +-0.5 | 11.2 of 12.0; +4.1 +-0.5 | 2.8 of 11.8; +0.2 +-0.1 | 2.0 of 23.6; -0.4 +-0.4 |
| held-out samples (1 in 8 withheld): false spawns removed; accuracy change vs doing nothing | 10.6 of 12.0; +3.0 +-0.8 | 10.8 of 12.0; +3.4 +-0.7 | 11.8 of 12.0; +3.6 +-0.6 | 2.4 of 12.0; -0.4 +-0.2 | 1.4 of 20.8; -3.7 +-0.7 |
| answer-time averaging (no merge): false spawns removed; accuracy change vs doing nothing | -; +1.5 +-0.4 | -; +0.4 +-0.3 | -; +0.6 +-0.7 | -; -0.1 +-0.3 | -; +3.7 +-0.4 |

*Arms, in plain words.* `shipped`: today's merge. `prequential references`: retire an expert when another predicts its stored samples as well as it did itself, judged against its own past honest losses. `agreement`: retire
the younger of two experts that give the same answer on at least 80 % (or 50 %) of their stored prototypes; no labels. `rival tracking`: every batch routed to an expert is also scored on its nearest rivals before anyone
trains on it, so paired loss differences accumulate for free; a pair is retired when the survivor's loss is not worse than the retiree's by more than 0.1 ln K (one-sided, 99 %, at least 96 samples). `held-out samples`: 1 in
8 samples is withheld from every expert's training, filed under the expert the pool routes it to, and the pair is tested on those never-trained samples (paired, one-sided, same margin). `answer-time averaging`: no merge; the
answers of the chosen expert and of every overlapping expert that also accepts the input are averaged. Extras measured in `RESULTS_HEALING2.md`: a guard that the survivor must beat the chance-level loss by 0.2 nats
(it removed the one wrong merge we saw in an earlier, head-start version of the study), moving the retiree's knowledge into the survivor by weight midpoint (66.5 %) or distillation (66.5 %),
against 66.7 % for the plain redirect: no gain (fine-tuning the survivor on the stored samples was measured only in the head-start version of the study: no gain either).

What the table says, in order of importance:

- Evidence per expert is the bottleneck, and it grows with the length of a regime. Free rival tracking is weak at 10 batches and complete from 15; the false spawn needs five batches to mature before it may be
  tested, so a false spawn that arrives late in a regime waits for the next visit (column "batch 12").
- Withholding data is not free: 1 in 8 withheld costs 0.3 points on a clean pool but 4.4 points when three experts share a regime, so held-out repair is a net loss there (-3.7
  against doing nothing).
- A label-free test can be fast and exact when its threshold fits the task (all 12 removed, accuracy back to the clean level at one false spawn per regime) and blind when it does not (0 at 0.8). Its threshold has to be
  calibrated, and agreement alone cannot tell two equal experts from an expert and a worse one (the pair study's "dominated" and "partial" pairs agree 0.72-0.85).
- Removing twins does not return accuracy when three experts shared a regime's data; averaging their answers does.
- No test merged experts of different regimes (0 of 968), but the regimes of this bench are far apart.

### 2.5 The silent stop, and what a full pool could do instead

The pool is limited to a few experts. "drop" is what the shipped cap does. "extend" forces the cluster into the nearest expert. "evict" removes the most redundant expert (largest mutual signature coverage; the less
used of the pair) and founds the new one; "evict_lru" removes the least recently used. Cell: accuracy over all 12 regimes; accuracy on the regimes first met after the pool filled up; regimes that never got an expert.

| what a full pool does | limit 16, 12 false spawns (needs 24) | limit 20, 24 false spawns (needs 36) | limit 8, no false spawns (12 regimes need 12) |
|---|---|---|---|
| (reference) the same stream without a limit | 67.4 | 67.4 | - |
| drop the cluster (today, silent) | 50.7; regimes met after it filled: 25 %; 4.0 regimes without an expert | 48.3; regimes met after it filled: 26 %; 5.0 regimes without an expert | 53.4; regimes met after it filled: 26 %; 4.0 regimes without an expert |
| force the cluster into the nearest expert | 56.5; regimes met after it filled: 45 %; 4.0 regimes without an expert | 55.6; regimes met after it filled: 45 %; 5.0 regimes without an expert | 59.0; regimes met after it filled: 43 %; 4.0 regimes without an expert |
| evict the most redundant expert (else the least recently used) | 66.1; regimes met after it filled: 67 %; 0.2 regimes without an expert | 64.6; regimes met after it filled: 67 %; 0.4 regimes without an expert | 53.3; regimes met after it filled: 68 %; 0.0 regimes without an expert |
| evict the least recently used expert | 61.8; regimes met after it filled: 67 %; 0.0 regimes without an expert | 62.6; regimes met after it filled: 67 %; 0.2 regimes without an expert | 53.3; regimes met after it filled: 68 %; 0.0 regimes without an expert |
| evict the most redundant expert; if none is redundant, force an extend | 66.1; regimes met after it filled: 67 %; 0.2 regimes without an expert | 64.6; regimes met after it filled: 67 %; 0.4 regimes without an expert | 59.0; regimes met after it filled: 43 %; 4.0 regimes without an expert |

Dropping is the silent death: the regimes met after the pool fills are answered at chance and nothing is logged. When the shortage is caused by redundancy (first two columns), evicting the most redundant expert keeps
most of the accuracy of the unlimited pool (66.1 of 67.4, 64.6 of 67.4), because it removes the twin of a regime that has two experts, and it needs no statistical test, only signature overlap and usage. When the shortage is real (last column, nothing
is redundant) eviction only trades forgetting for not learning (53.3 % against 53.4 %), and forcing the cluster into the nearest expert is best (59.0 %): so the combined rule "evict a
redundant expert if there is one, otherwise share" is best or tied in every setting. Evicting by recency alone forgets real specialists.

### 2.6 The infant window: a different silent failure found on the way

A new expert is an *infant* for 5 batches. In the gate its p-values are multiplied by 4 (`infant_relax`, `phylo.py:1289`); with 64 stored calibration points the smallest possible p-value is 1/65 = 0.015, times 4 = 0.06,
which is above the acceptance level q = 0.02, so an infant accepts every input. That is deliberate: without it the pool founds an expert for every batch of a new regime. But it also means the pool is blind to novelty for
5 batches after every spawn. The pool below has no false spawns at all (3 seeds per row, listed in order):

| infant relaxation | batches per regime | experts at the end (12 regimes) | regimes that got an expert of their own | accuracy (points) |
|---|---|---|---|---|
| x4 (shipped) | 10 | 12, 12, 12 | 12, 12, 12 | 67.7 |
| x4 (shipped) | 5 | 10, 12, 5 | 10, 12, 5 | 55.0 |
| x4 (shipped) | 3 | 1, 11, 8 | 1, 11, 8 | 45.4 |
| x4 (shipped) | 2 | 1, 3, 4 | 1, 3, 4 | 35.1 |
| x4 (shipped) | 1 | 5, 5, 5 | 5, 5, 5 | 35.3 |
| x1 | 10 | 121, 121, 69 | 12, 12, 10 | 50.4 |
| x1 | 5 | 26, 57, 15 | 6, 12, 6 | 44.5 |
| x1 | 3 | 1, 27, 12 | 1, 10, 6 | 38.8 |
| x1 | 2 | 1, 6, 9 | 1, 4, 6 | 35.2 |
| x1 | 1 | 13, 13, 13 | 12, 12, 12 | 45.6 |

A regime shorter than about 5 batches is partly invisible (the next regime is absorbed by the infant, then overwrites it), and without the relaxation the pool cascades. The shipped setting is therefore a trade of one
silent failure for another. It is harmless for the long regimes of the benchmarks (thousands of samples per block) but it is a property of the system you should know, and the reason why a false spawn right before a
regime change is worse than one in the middle of a regime.

### 2.7 What was not measured

- A real-stream end-to-end test with injected false spawns (the arithmetic benchmark): the function-regime stream is synthetic, with little data per regime; the real-expert check (section 2.2) is at the level of pairs.
- The false-spawn rate of stage 4b beyond the finished dev units (the dev review decides how large the problem is in practice).
- Compute and memory costs, and the effect of redundancy on the stage-4a router.
- Provisional spawning (a challenger trained beside the incumbent, option B3), Fisher or RegMean-style closed-form merges, repair of dormant or archived experts (they receive no traffic, so only stored prototypes remain).
- Five seeds per cell; differences below about 1 point are not resolved.

## 3. What the literature says

Every source below was found by search on 2026-10-16. "read" = I read the passage in the paper's text; "abstract" = only the abstract or listing.

| Source | What it says | What we take |
|---|---|---|
| Hughes, Sudderth. *Memoized online variational inference for Dirichlet process mixture models*. NeurIPS 2013. [link](https://proceedings.neurips.cc/paper/2013/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html) (read) | Births always accepted; merges accepted only if an exactly tracked full-data objective improves; noisy single-batch merge scores can accept merges that hurt; merge pairs chosen by a marginal-likelihood ratio from cached sufficient statistics. | Births are optimistic and merges are judged by an honest global objective. Ours had no honest objective; live or held-out loss is the analogue. Evidence must accumulate over batches, not be read from one. |
| Wang, Crammer, Vucetic. *Breaking the curse of kernelization: budgeted SGD*. JMLR 13, 2012. [link](https://www.jmlr.org/papers/v13/wang12b.html) (read) | Budget maintenance (remove, project or merge a support vector) is a source of error; the gap to the optimum is bounded by the average degradation it causes, so choose the action that degrades least; merging is the most attractive; validating on the stored examples themselves can mislead. | When the budget forces a removal, pick the least damaging one, honestly measured. "Evict the most redundant expert" is this idea at the expert level. |
| Miller, Harrison. *A simple example of Dirichlet process mixture inconsistency for the number of components*. NeurIPS 2013. [link](https://arxiv.org/abs/1301.2708) (abstract) | The DP mixture posterior on the number of clusters does not converge to the true number even when the model is right. | A model that only adds components over-counts; the count must be controlled explicitly. |
| Kolter, Maloof. *Dynamic weighted majority*. JMLR 8, 2007. [link](https://jmlr.org/papers/v8/kolter07a.html) (abstract) | Experts are added when the ensemble errs and removed when their weight falls; weights follow performance. | Removal by sustained poor performance or usage is the standard cheap rule in stream learning. |
| Gomes et al. *Adaptive random forests for evolving data stream classification*. Machine Learning 106, 2017. [doi 10.1007/s10994-017-5642-8](https://doi.org/10.1007/s10994-017-5642-8) (abstract) | A drift warning starts a background learner beside the incumbent; it replaces it only when drift is confirmed. | A challenger trained in parallel is judged on the future stream, which is always honest, and takes no data from the incumbent (option B3). |
| Guo et al. *Dynamic mixture of experts (DynMoE)*. ICLR 2025. [arXiv 2405.14297](https://arxiv.org/abs/2405.14297) (read) | Experts are added when tokens choose to activate no expert and removed when no token activates them. | Usage-based removal is the accepted pruning rule in dynamic mixtures of experts. |
| Li et al. *Merge, then compress (MC-SMoE)*. ICLR 2024. [arXiv 2310.01334](https://arxiv.org/abs/2310.01334) (read) | Naive weight merging of experts fails; aligning neuron permutations first (Git Re-Basin weight matching) helps (their ablation: +0.7 to +2.0 points); experts are grouped by routing statistics and averaged weighted by activation frequency; distillation helps. | Weight merging is an offline compression of related, aligned experts. Our false spawns are clones (no alignment needed); unrelated experts would need it. |
| Wortsman et al. *Model soups*. ICML 2022. [arXiv 2203.05482](https://arxiv.org/abs/2203.05482) (abstract); Frankle et al. 2020 [arXiv 1912.05671](https://arxiv.org/abs/1912.05671), Entezari et al. 2022 [arXiv 2110.06296](https://arxiv.org/abs/2110.06296), Ainsworth et al. 2023 [arXiv 2209.04836](https://arxiv.org/abs/2209.04836) (abstracts) | Averaging weights works inside one loss basin (models fine-tuned from the same start, or aligned by permutation); a greedy soup keeps a model only if held-out accuracy does not drop. | Our pair study agrees: averaging clones gains 1.9 points, averaging independently trained experts loses 12; the held-out acceptance rule is the one we used. |
| Matena, Raffel. *Fisher-weighted averaging* [arXiv 2111.09832](https://arxiv.org/abs/2111.09832); Jin et al. *RegMean* [arXiv 2212.09849](https://arxiv.org/abs/2212.09849) (abstracts) | Merge weights by Fisher information or by layer-input statistics, without the original data. | Possible consolidation for archived experts that keep no samples; untested here. |
| Schwarz et al. *Progress & Compress*. ICML 2018. [arXiv 1805.06370](https://arxiv.org/abs/1805.06370); Hinton et al. *Distillation* [arXiv 1503.02531](https://arxiv.org/abs/1503.02531) (abstracts) | A knowledge base absorbs each new column by distillation; no growth, no stored data. | Consolidation by distillation has precedent; in our test it added nothing beyond the redirect. |
| Wang et al. *SEMA*. CVPR 2025. [arXiv 2403.18886](https://arxiv.org/abs/2403.18886); Lee et al. *CN-DPM*. ICLR 2020. [arXiv 2001.00689](https://arxiv.org/abs/2001.00689); Aljundi et al. *Expert Gate*. CVPR 2017. [arXiv 1611.06194](https://arxiv.org/abs/1611.06194) (abstracts) | Expansion-based continual learners; SEMA adds a module only when a descriptor detects a shift, to avoid linear growth. | Growth control is usually put at the novelty detector (prevention); I found no expansion-based method in the abstracts that also repairs a wrong expansion. |
| Lakens. *Equivalence tests: a practical primer*. 2017. [PMC5502906](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5502906/) (abstract) | "Not significantly different" does not show equivalence; test against a smallest effect of interest (two one-sided tests). | Our merge criterion is a one-sided non-inferiority test with a margin, which is the right logic; the shipped test and agreement thresholds are not. |

Not found: a method for expanding mixtures of experts that also removes a wrongly created expert with a statistical guarantee. Where sources are silent, the design here is ours and is justified only by the measurements above.

## 4. The options

Ideas are grouped by what they do about the problem. "Evidence" says what we measured; "not measured" is stated where it applies. Effort: small = a few lines and tests, medium = a module plus experiments, large = a redesign.

| id | option | removes false spawns | gives accuracy back | can it hurt | effort | evidence |
|---|---|---|---|---|---|---|
| A1 | loud cap: event, counter, warning | no | no | no | small | code reading, section 2.5 |
| A2 | redundancy telemetry (overlapping pairs, agreement) | no | no | no | small | overlap and agreement of twins versus other experts, `healing2_diag.py` |
| A3 | count samples that mature experts call novel but an infant accepted | no | no | no | small | section 2.6 |
| B1 | stage-4b transfer test (built, flag off) | prevents | prevents | needs the dev run | done | dev snapshot below |
| B2 | persistence: two consecutive conflicts before a spawn | prevents | prevents | delays real new regimes | small | not measured |
| B3 | provisional spawn: a challenger beside the incumbent, judged on the future stream | prevents | prevents (no data split) | slower service of a new regime | large | literature only |
| B4 | decouple the infant relaxation (attract training samples, but do not clear novelty) | prevents | prevents | cascade risk | medium | section 2.6 shows the trade |
| C1 | redirect merge on held-out samples | yes where evidence suffices | yes (+3.0 of 3.7 at one per regime) | withholding costs 4.4 points at three experts per regime | medium | section 2.4 |
| C2 | redirect merge on rival tracking (live traffic, nothing withheld) | yes from about 15 batches per regime | yes (4.1 of 4.3 at 20) | none seen; useless below 15 batches | medium | section 2.4 |
| C3 | merge on prequential references (exists in stage 4) | weak | no | low | exists | section 2.4 |
| C4 | label-free agreement merge | only with a fitted threshold (11.8 of 12 against 0.0 at 0.8) | at one per regime | wrong merge of unequal pairs (not tested here) | medium | sections 2.2, 2.4 |
| C5 | consolidation after the merge (weight midpoint, distillation, fine-tuning) | no | not in the test | extra compute | small | section 2.4 |
| D1 | when full: evict the most redundant expert, otherwise force an extend | yes (by force) | indirectly | forgets nothing when twins exist | small | section 2.5 |
| D2 | when full: widen the merge margin with the pressure | yes | no | wrong merge | medium | not measured |
| E1 | answer-time averaging of overlapping experts | no | yes (+1.5 to +3.7 with twins) | compute at answer time | small | section 2.4 |
| F1 | shared trunk with small expert heads or adapters | makes a false spawn cheap | yes | redesign | large | literature only (SEMA, progressive networks) |

**A. See it.** *A1 loud cap.* When `max_experts` blocks a cluster, write an event (step, cluster size, experts, cap), count it in the report and warn once. The shipped pool shows nothing today. Cost: about ten lines and a
test; it cannot change a result. *A2 redundancy telemetry.* On demand, report the number of expert pairs whose signatures cover each other by at least 20 % and the agreement of their answers; in our injected pools twins sat at
0.26-0.59 signature overlap and 0.58-0.72 answer agreement (`healing2_diag_output.txt`), and on the real arithmetic experts different operators agree 0.22-0.49 against 0.60-0.86 for duplicates
(section 2.2). *A3 infant suppression counter.* Count samples that every mature expert calls novel but an infant accepted. All three are read-only.

**B. Do not make it.** *B1* is stage 4b and is what the dev runs test: label-based, exact, margin 0.25 ln K. Its false-spawn rate is the number that decides how much repair we need. A snapshot of the finished dev units (157
of 157 cells): with the transfer rule on, the pool ended with about as many experts as the stream has functions (table; one to three seeds per unit; the accuracy differences on the function-regime streams carry the head
start of section 6). The pool without the rule (S3) ends with too few experts on three streams (arith-seq 4 of 6, S1 5 of 7, S2-5 3.3 of 5): it merges distinct functions. That is why the declared selection rule, run on these
cells, left the transfer margin at its default: its guard "experts at most 1.2 times S3's" fails on S1 (1.53) and S2-5 (1.50), where S3 under-spawns and the transfer rule finds all the functions (`rv2d/DEV_SELECTION.md`).

| unit | seeds | true number of functions | experts without the transfer rule (S3) | experts with it (S3+T) | accuracy difference T minus S3 (points, as measured with the head start where the stream has one) |
|---|---|---|---|---|---|
| arith-seq | 2 | 6 | 4.0 | 6.0 | +8.2 [+5.3, +11.1] (n=2) |
| s3:F-15 | 3 | 15 | 15.0 | 15.0 | +1.1 [-0.5, +2.7] (n=3) |
| s3:OV-0.6 | 3 | 10 | 10.0 | 10.0 | +0.4 [-0.2, +1.1] (n=3) |
| s3:OV-0.85 | 3 | 10 | 10.3 | 10.0 | -0.1 [-1.5, +1.3] (n=3) |
| s3:S1 | 3 | 7 | 5.0 | 7.7 | +4.1 [-2.1, +10.2] (n=3) |
| s3:S2-5 | 3 | 5 | 3.3 | 5.0 | +1.7 [-4.3, +7.8] (n=3) |
| banking | 2 | n/a | 5.0 | 6.0 | +1.4 [-12.7, +15.6] (n=2) |

*B2 persistence* (the device the set gate already uses) halves false positives at the price of a delayed spawn; not measured. *B3 provisional spawn* follows adaptive random forests: the new expert is trained beside the nearest
incumbent on a copy of the incumbent's traffic (no data split), is judged on the future stream (paired, honest, no held-out data needed), and either becomes live (the incumbent is bad on the new data) or is discarded
(equivalent). It removes the data split entirely, but it needs a new lifecycle state, a promotion test fast enough for a true new regime, and care that the gate counts the provisional signature as covering its region
(otherwise the pool spawns again). *B4* separates the two jobs of the infant relaxation; section 2.6 is the evidence that they are in conflict today.

**C. Fix it afterwards.** *C2* (rival tracking) is the one I would build first: while an expert receives a batch, its nearest rivals (experts whose signature covers its inputs) are scored on the same samples before anyone
trains on them; the paired loss differences accumulate in a window of 512, and when the upper bound of (rival minus expert) stays below a margin of 0.1 ln K at 99 % with at least 96 samples, the expert is retired in favour
of the rival by moving its signature, stored samples and references into the rival; no weight is touched. It costs one forward pass per rival per batch, withholds nothing, and in the tests removed 11.2 to
11.2 of 12 false spawns at 15-20 batches per regime with no wrong merge. Weak spots: it needs the false spawn to be mature and to receive traffic, so it does nothing in regimes shorter than about 15 batches
and nothing for dormant duplicates, and it was tested only on clones. *C1* (held-out samples) heals earlier (10.6 of 12 at 10 batches) because it can judge past samples, but withholding is paid by every expert and
backfires when experts share a regime (-3.7 points at three per regime). *C3* is the stage-4 mechanism and is too weak alone. *C4*: the agreement of two equal experts depends on their accuracy, so a
threshold needs a per-task calibration and a small labelled check against "one expert is worse". *C5*: weight midpoints, distillation and fine-tuning gained nothing here; the weight midpoint is only safe for clones (our false spawns
are clones: lambda = 1.0), and MC-SMoE shows that unrelated experts need permutation alignment.

**D. Contain it.** *D1*: when the pool is full, evict the expert whose signature is most covered by another's (the less used of the two); if nothing is redundant, force the cluster into the nearest expert instead of
forgetting one; log it as an event. It needs no statistics and changes nothing until the cap is reached. Budgeted learning has the same structure (remove, project or merge the item that degrades the model least; Wang
et al. 2012; DynMoE and dynamic weighted majority remove unused or underperforming experts). *D2*: let the margin of a merge grow with how full the pool is; untested and risky because a loose margin accepts the
"partial" pairs of section 2.1.

**E. Make it harmless.** *E1*: at answer time average the answers of the chosen expert and the overlapping experts that also accept the input. It never touches training, so it cannot damage learning; on a pool without
duplicates the overlap group is empty and the answers are identical (0.00 points difference). It recovered most of the loss when three experts shared a regime (+3.7 of 4.3), little when
regimes were long (+0.6), nothing for a late false spawn (-0.1). It does not free slots and costs compute proportional to the group size. A natural home is a candidate of the stage-4a router.

**F. Rebuild.** *F1*: make the expert a small head or adapter on a shared trunk, so a false spawn costs a few parameters and cannot split the data. It would change the character of the system and is not recommended now.

## 5. Recommendation, sequence, and the decisions I need

**Recommended package**

1. *Now, no code:* the dev runs are finished (157 cells) and the declared selection rule has been run (`rv2d/DEV_SELECTION.md`: grow_gap 0.17, router free-0.01, transfer margin left at the default because the
   expert-count guard misfires, see option B1). dev2 has NOT been started: section 6 needs a decision first, because it changes how the function-regime cells are to be read or re-run. The dev review gets two more lines: the
   expert-count guard should compare with the number of true functions instead of with S3 (T/true is 1.0 on arith-seq, F-15, OV-0.6, OV-0.85, S2-5 and 1.1 on S1), and the one-function iid stream must stay at one expert.
2. *Before the confirmatory run (stage 4.1), behaviour-neutral until the pool is full:* A1, A2, A3 and D1. None of them can change a benchmark result (the dev streams end with at most 30 experts, the cap is 256), so they
   do not disturb the pre-registered comparisons; each gets a test that the pool is bit-identical until the cap is reached.
3. *Stage 5, own pre-registration:* C2 behind a flag, tested on the end-to-end harness and then on the real arithmetic stream with injected false spawns; E1 as a router candidate.
4. *Research track:* B3 and B4, only if the dev review shows false spawns or short-regime losses that matter.
5. *Do not build:* a tweak of the shipped test (its flaw is structural), a held-out reservoir (it costs data exactly where repair is needed), a label-free merge without calibration, weight averaging as a repair.

**Decisions for you**

1. **What should a full pool do?** (a) keep dropping, but loudly (A1 only), (b) evict the most redundant expert, otherwise share the nearest one (D1, my pick), (c) always force the cluster into the nearest expert
   (measured worse than (b) whenever twins exist).
2. **How should repair get its evidence?** (a) free live-traffic scoring only (my pick: needs regimes of about 15 batches or more), (b) also withhold about 1 in 8 labelled samples (heals earlier at 10 batches but loses
   4.4 points when experts share a regime), (c) no repair, rely on prevention and eviction.
3. **Adopt answer-time averaging of overlapping experts (E1)** as a router candidate once validated on a real stream?
4. **Sequence:** (a) A1-A3 + D1 into stage 4 before the freeze, the rest in stage 5 (my pick), (b) everything in stage 5, (c) everything now (delays the single-use confirmatory run).
5. **The head start (section 6):** (a) re-run the dev cells of the function-regime streams with the pool seed offset and use the offset in the confirmatory run (clean, a few hours of compute and a new code stamp), (b) keep dev
   as it is, run only a small non-aliased control of the arm differences, and use the offset in the confirmatory run (my pick), (c) fix the generator of the streams and re-check the earlier headline numbers (the
   largest job, the only one that settles the old claims).
6. **The infant window (section 2.6):** document it as a limit (regimes of at least about 5 batches) and decide after the dev review, or fix it now.

## 6. Validity alert: the function-regime streams give the pool a head start

*What it is.* `function_regimes` builds the labels of regime r with `MLPExpert(seed = stream_seed * 1000 + r)` as teacher (`growmoe/streams.py`, `_balanced_teacher`). The pool founds expert k with weights from
`MLPExpert(seed = pool_seed * 1000 + k)` (`phylo.py`, via `shrink_perturb`). Same generator, same seed: the first 8 hidden units of the expert are bit-identical to the teacher's hidden units
(`alias_check.py`, `torch.equal` is True for the expert with the same seed and False for any other). Every runner I looked at sets the pool's seed to the stream's seed (`run_m0.py`, `run_m2a.py`, `run_rv2a.py`,
`run_rv2b.py`, `run_rv2c.py`, `run_m5*.py`, and through them `rv2d_lib.py`), so a pool that founds exactly one expert per regime in order starts every regime with the teacher's features.

*What it is worth.* In my harness (experts of 32 hidden units):

```
seed 0: expert's first 8 hidden units == teacher's hidden units: True   (an expert with another seed: False)
seed 1: expert's first 8 hidden units == teacher's hidden units: True   (an expert with another seed: False)
seed 5: expert's first 8 hidden units == teacher's hidden units: True   (an expert with another seed: False)
n_train 640 stream seed 0: pool seed = stream seed (aliased) 74.5 (12 experts) | pool seed = stream seed + 7 (not aliased) 68.2 (12 experts)
n_train 640 stream seed 1: pool seed = stream seed (aliased) 73.5 (12 experts) | pool seed = stream seed + 7 (not aliased) 69.5 (12 experts)
n_train 640 stream seed 2: pool seed = stream seed (aliased) 72.2 (12 experts) | pool seed = stream seed + 7 (not aliased) 65.4 (12 experts)
n_train 1280 stream seed 0: pool seed = stream seed (aliased) 76.8 (12 experts) | pool seed = stream seed + 7 (not aliased) 71.4 (12 experts)
n_train 1280 stream seed 1: pool seed = stream seed (aliased) 76.5 (12 experts) | pool seed = stream seed + 7 (not aliased) 72.3 (12 experts)
n_train 1280 stream seed 2: pool seed = stream seed (aliased) 75.5 (12 experts) | pool seed = stream seed + 7 (not aliased) 68.9 (12 experts)
```

and the id-shift control, which moves the expert ids as a false spawn would but founds nothing (3 seeds, regimes of 20 batches; the first block has the head start, the second not):

```
pool seed offset 0 (the head start is present):
n_train 1280 inject at 12 | clean                own expert alone  76.2 (+-0.4)   whole pool routed  76.2 (+-0.4)   experts 12
n_train 1280 inject at 12 | id_shift_only        own expert alone  70.8 (+-1.0)   whole pool routed  70.8 (+-1.0)   experts 12
n_train 1280 inject at 12 | inject_then_remove   own expert alone  70.8 (+-1.0)   whole pool routed  70.8 (+-1.0)   experts 12
n_train 1280 inject at 12 | inject_keep          own expert alone  70.0 (+-0.4)   whole pool routed  68.3 (+-0.2)   experts 24
pool seed offset 7 (no head start):
n_train 1280 inject at 12 | clean                own expert alone  70.9 (+-1.0)   whole pool routed  70.9 (+-1.0)   experts 12
n_train 1280 inject at 12 | id_shift_only        own expert alone  70.9 (+-0.8)   whole pool routed  70.9 (+-0.8)   experts 12
n_train 1280 inject at 12 | inject_then_remove   own expert alone  70.9 (+-0.8)   whole pool routed  70.9 (+-0.8)   experts 12
n_train 1280 inject at 12 | inject_keep          own expert alone  70.5 (+-0.8)   whole pool routed  68.4 (+-0.8)   experts 24
```

On the project's own dev runner (stage-3 pool, arms S3 and S3+T, stream seeds 100 and 101; `alias_check_s3.py`):

| stream | arm | stream seed | accuracy, pool seed = stream seed (as the dev runs) | accuracy, pool seed = stream seed + 7 | head start |
|---|---|---|---|---|---|
| F-15 | S3 | 100 | 51.2 (15 experts) | 43.0 (15 experts) | +8.2 |
| F-15 | S3 | 101 | 49.7 (15 experts) | 41.9 (15 experts) | +7.7 |
| F-15 | S3+T | 100 | 52.3 (15 experts) | 43.2 (15 experts) | +9.0 |
| F-15 | S3+T | 101 | 51.4 (15 experts) | 41.8 (15 experts) | +9.6 |
| OV-0.3 | S3 | 100 | 46.8 (10 experts) | 41.4 (10 experts) | +5.3 |
| OV-0.3 | S3 | 101 | 44.3 (10 experts) | 40.1 (10 experts) | +4.2 |
| OV-0.3 | S3+T | 100 | 48.0 (10 experts) | 41.0 (10 experts) | +7.0 |
| OV-0.3 | S3+T | 101 | 45.8 (10 experts) | 40.0 (10 experts) | +5.8 |
| OV-0.6 | S3 | 100 | 43.1 (10 experts) | 42.5 (10 experts) | +0.5 |
| OV-0.6 | S3 | 101 | 42.2 (10 experts) | 40.7 (10 experts) | +1.5 |
| S1 | S3 | 100 | 79.5 (4 experts) | 78.7 (4 experts) | +0.9 |
| S1 | S3 | 101 | 80.0 (5 experts) | 79.1 (4 experts) | +0.9 |
| S1 | S3+T | 100 | 81.0 (7 experts) | 80.9 (7 experts) | +0.1 |
| S1 | S3+T | 101 | 81.4 (8 experts) | 78.9 (7 experts) | +2.6 |
| S2-5 | S3 | 100 | 93.2 (5 experts) | 91.0 (5 experts) | +2.2 |
| S2-5 | S3 | 101 | 89.8 (2 experts) | 84.0 (1 experts) | +5.8 |
| S2-5 | S3+T | 100 | 92.4 (5 experts) | 90.4 (5 experts) | +2.0 |
| S2-5 | S3+T | 101 | 94.0 (5 experts) | 92.6 (5 experts) | +1.4 |

*What it does to the stage-4 comparison.* Both arms get the head start, the transfer arm somewhat more. On F-15 and OV-0.3 the small gain of the transfer rule (+1.1 to +1.7) disappears without the head start; on S1 and S2-5 the
two seeds disagree with each other (S2-5, seed 101: S3 collapses to one expert without the head start and the gap is +8.6), so those streams need more seeds either way:

| stream | stream seed | transfer rule minus S3, as the dev runs measure it | the same without the head start |
|---|---|---|---|
| F-15 | 100 | +1.1 | +0.3 |
| F-15 | 101 | +1.7 | -0.1 |
| OV-0.3 | 100 | +1.2 | -0.4 |
| OV-0.3 | 101 | +1.5 | -0.2 |
| S1 | 100 | +1.4 | +2.2 |
| S1 | 101 | +1.4 | -0.3 |
| S2-5 | 100 | -0.8 | -0.6 |
| S2-5 | 101 | +4.2 | +8.6 |

The large stage-4 results are on streams without a teacher and are not affected: the router on arith-seq (+13.7 points), the transfer rule on arith-seq (+8.2), growth on the arithmetic mixed streams (+3.2 and +1.1 at the default gap); banking is
real text. The finished dev1 cells of the function-regime streams, and every earlier number on F-* and OV-* streams (notebook entries E-013, E-014, E-015, E-017, E-019, E-021, E-023, E-025, E-032 mention function regimes;
I did not re-run them), carry the head start. In pool-versus-baseline comparisons it favours the pool by up to about 8 points on F-15 and much less on the overlapping streams (OV-0.6: +0.5 and +1.5); the size depends on
the stream and on the width of the experts, so it has to be measured per claim. The notebook records it as F-30 and Q-23.

*Options* are decision 5 above. A fix at the source is small: give the teachers their own seed namespace in `function_regimes` (an argument that defaults to the old value so that old results stay reproducible) or add
an offset to the expert seeds in the pool; both break bit-identity with old results on purpose.

## 7. Limits of this study

- Synthetic stream with little data per regime (clean accuracy 67-71 %); regimes are far apart on the sphere, which makes wrong merges across regimes unlikely; hidden size 32.
- False spawns are injected as clones with the extend option off; real ones may differ in timing and in how well their parent was trained. The pair study and the arithmetic check cover other kinds of pair.
- Five seeds; the throwaway scripts are not part of the product and were not mutation-tested. The two earlier versions of this study are kept as evidence of what went wrong (`healing.json`, `healing2_[a,b,a20,cap*,m*].json`
  are the head-start versions; the corrected ones carry the prefix `n_`).
- The dev review has not been done; the dev snapshot above is a first look.

## 8. How to reproduce

```
cd research/growmoe_results/merge_study
python run_study.py && python run_study.py report            # constructed pairs, RESULTS_MERGE_STUDY.md
python arith_check.py                                        # real arithmetic experts (needs the saved pool)
python healing2_study.py --inject 3 --seeds 0,1,2,3,4 --arms clean,idshift,none,... --tag n_a        # config a; --inject 3,4 --tag n_b; --n-train 960 / 1280 --tag n_a15 / n_a20; --inject 12 --n-train 1280 --tag n_m12
python healing2_study.py --inject 3 --cap 16 --arms clean,cap_drop,cap_extend,cap_evict,cap_evict_lru,cap_smart --tag n_cap16     # also --inject 3,4 --cap 20 --tag n_cap20 and --inject none --cap 8 --tag n_cap8clean
python infant_study.py                                       # RESULTS_INFANT.md
python alias_check.py; python idshift_check.py 1280 12 0,1,2 0; python twin_diag.py 640 0 3
python alias_check_s3.py F-15 --seeds 100 101 --arms S3,S3+T # also S1, S2-5, OV-0.3, OV-0.6
python report_healing2.py && python build_report.py          # RESULTS_HEALING2.md and this note
```

Files: `study_lib.py`, `run_study.py`, `arith_check.py`, `healing_study.py` and `healing_diag.py` (the first, flawed version and its diagnosis), `healing2_study.py`, `healing2_diag.py`, `twin_diag.py`, `infant_study.py`,
`alias_check.py`, `alias_check_s3.py`, `idshift_check.py`, `report_healing2.py`, `build_report.py`, `report_template.md`, the `RESULTS_*.md`, `*.json` and `*.txt` files next to them. Lab notebook: E-034 (this study),
E-035 (the head start), claims C-68 to C-75, failures F-29 and F-30, open questions Q-19 (updated) and Q-20 to Q-23.

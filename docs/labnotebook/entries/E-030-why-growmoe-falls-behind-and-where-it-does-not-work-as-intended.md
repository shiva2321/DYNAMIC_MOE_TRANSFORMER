# E-030 - Why GrowMoE falls behind and where it does not work as intended: a root-cause study

- **ID:** E-030
- **Date:** 2026-10-14
- **Track:** G
- **Type:** experiment (diagnosis)
- **Status:** completed (exploratory; the run was interrupted once by a power cut and finished from checkpoints)
- **Evidence grade:** B (script-generated and reproducible, one or two seeds per arm, NOT pre-registered; the suspects were written down before the runs and several were refuted by them)
- **Code / commit:** `research/growmoe_results/diagnosis/` (uncommitted; product code untouched, all instrumentation is done by wrapping functions inside the scripts)
- **People:** Claude designed, built, ran and wrote it at the user's request

## Question
The user asked (2026-10-14): "find out exactly why grow moe is failing and falling behind and not properly working as intended", and then: "is the growmoe able to dynamically expand vertically and horizontally as needed with variable structure or networks as needed?"

## Why we asked it
The arithmetic benchmark (E-026) showed GrowMoE far behind dense networks (27 to 37 % against 64 %), never growing in a mixed stream, and its lifecycle parts (growth, merge, archive) had never been exercised at benchmark scale (component audit). Without knowing which part is at fault no repair can be chosen.

## Prediction (written BEFORE the runs, after reading the code)
S1 the training recipe of the pool (one pass, bounded replay, constant learning rate), not the expert, explains the arithmetic gap. S2 experts are created only in response to a change, so a stationary mixture never splits. S3 the extend-versus-spawn interference probe is unreliable. S4 the stage-3 pool loses accuracy because the first batches of a new operator contaminate the old expert (a leak). S5 the growth policy decides on too little evidence. S6 its 480-step probe is too short to see the value of capacity.
Outcome: S1, S2, S3 and S5 confirmed; S4 REFUTED (99.8 to 100 % of every block trains exactly one expert; the loss is routing at answer time); S6 REFUTED in the form stated (a longer probe makes the policy's yardstick worse; the yardstick itself is the problem).

## Method (how)
All on the mixed-operator arithmetic stream of E-026 (8,000 problems per operator, about 298,000 examples, same held-out test grid and exact-match scoring), pool = `PhyloMoE` as in the benchmark.
- Study 1, a ladder of 19 (network x recipe) cells, two seeds each (82 runs with the memory sweep): the pool's own expert (`MLPExpert`, hidden 128 shipped / 512 tuned), MLP-S and MLP-L, trained with the dense recipe of the benchmark (shuffled batches of 512, one-cycle schedule, 40 or 10 epochs) or with the pool's recipe (`BasePool.train_expert`: batches of 64 seen once, 5 steps on each batch mixed with an equal share of replay from a reservoir of N, Adam 3e-3 constant), plus reservoir sweep, learning-rate decay, one step per batch, and an offline learner restricted to 20,000 stored examples.
- Study 2, the gate's own density model per operator (vMF signatures of 256 stored inputs) and the routing rules of the product on signatures built like the sequential pool's.
- Study 3, the operator-by-operator stream with the tuned pool and stage 3: the shipped lifecycle (`default`), a 10-times longer interference probe (`probe600`), and a variant in which an extend can never be accepted (`forced_spawn`, `delta_forget = -1`); instrumentation of the interference probe's four losses, the training assignment of every batch, routing at every block boundary, oracle routing (best existing expert per operator), the routing rules of the product, and a learned router (logistic regression on the pool's own assignments) on the saved final pools; two seeds.
- Studies 4 and 5, the growth policy: every decision logged (shipped and tuned configuration), the policy asked again with more evidence, a deliberately under-sized start (16 hidden units) with fixed experts of 16 to 512 units as references, the capacity added mid-stream (128 or 16 units start), probe length against measured gain, and lowered minimum gains.
- Study 6, real text, from the saved m5b / m5c / m5d runs.

## Results (tables copied from `research/growmoe_results/diagnosis/RESULTS_DIAGNOSIS.md`)
### 1. The arithmetic gap is the training recipe (unseen exact match %, mean of 2 seeds)
| network (parameters) | D40 | D10 | O5-r256 | O5-r2000 | O5-r20000 | O5-r100000 | O5-r300000 | O1-r20000 | O5-r20000-anneal | M2000-S3k | M2000-S23k | M20000-S3k | M20000-S23k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E128 (35,084) | 47.2 (2) | 39.0 (2) | 26.2 (2) | 23.2 (2) | 33.6 (2) | - | 36.9 (2) | 31.2 (2) | - | 5.3 (2) | 5.3 (2) | 13.4 (2) | 12.0 (2) |
| E512 (133,772) | 48.6 (2) | 44.6 (2) | 32.7 (2) | 31.6 (2) | 36.7 (2) | 42.9 (2) | 44.7 (2) | 37.1 (2) | 40.7 (2) | 6.5 (2) | 6.3 (2) | 13.9 (2) | 14.5 (2) |
| S256x2 (101,900) | 51.6 (2) | 42.7 (2) | 22.2 (2) | - | 36.1 (2) | - | 45.0 (2) | - | - | 6.6 (2) | 6.3 (2) | 17.4 (2) | 16.4 (2) |
| S256x2n (101,900) | 47.9 (2) | - | - | - | - | - | - | - | - | - | - | - | - |
| L1024x3 (2,243,596) | 63.8 (2) | 62.5 (2) | - | - | 32.8 (2) | - | - | - | - | - | - | - | - |

The pool equals a single expert trained with its recipe (the pool wrapper adds nothing in a stationary stream):

| seed | pool shipped (GrowMoE) | E128 + O5-r256 | pool tuned (h512, replay 20,000) | E512 + O5-r20000 |
|---|---|---|---|---|
| 0 | 26.1 | 26.1 | 37.3 | 37.3 |
| 1 | 28.0 | 26.2 | 36.1 | 36.1 |

From the best dense network down to the shipped pool:

| step | exact match | change |
|---|---|---|
| MLP-L (2.2 M parameters), dense recipe, 40 epochs: the benchmark's best dense network | 63.8 |  |
| the pool's expert architecture (E512, 134 k parameters) with the same dense recipe | 48.6 | -15.2 |
| ... the same expert, streaming recipe with unlimited replay (reservoir 300,000) | 44.7 | -3.9 |
| ... streaming recipe, reservoir 20,000 (= the tuned pool) | 36.7 | -8.0 |
| ... reservoir 256 | 32.7 | -4.0 |
| narrower expert, E128, reservoir 256 (= the shipped pool) | 26.2 | -6.5 |

### 2. The gate can rank operators but cannot call one "new"
- Which operator has the highest density at this input: **87.0 %** right (chance 16.7 %); operator x output slot: **74.9 %** right (chance 2.7 %).
- The nearest stored input has the same operator in 81.8 % of the cases.
- BUT an expert that has never seen operator j still ACCEPTS 88.5 % of j's inputs as familiar (conformal p >= 0.02), against 97.7 % of its own operator's inputs. So the per-sample gate can rank the operators, but it cannot call an operator it has never met "new".
- Geometry: changing only the operator moves the input exactly as far as changing one digit (cosine 0.833 in both cases); two random inputs of the same operator have mean cosine 0.332, of different operators 0.250.

| expert \ input | add | sub | mul | mod | div | max |
|---|---|---|---|---|---|---|
| add | 98 | 89 | 87 | 89 | 89 | 89 |
| sub | 88 | 97 | 88 | 89 | 89 | 89 |
| mul | 88 | 88 | 97 | 89 | 89 | 88 |
| mod | 89 | 89 | 87 | 98 | 90 | 90 |
| div | 87 | 87 | 85 | 87 | 97 | 88 |
| max | 89 | 90 | 87 | 90 | 90 | 99 |

Density routing among experts built like the sequential pool's (share of an operator's test inputs routed to its own expert, %):

| routing rule | max | add | sub | mod | div | mul |
|---|---|---|---|---|---|---|
| pool rule: accepted experts, highest log-density (with each expert's normaliser) | 71.3 | 75.8 | 77.7 | 91.9 | 91.2 | 91.5 |
| highest log-density without the normaliser | 69.9 | 74.5 | 76.7 | 92.8 | 92.1 | 91.5 |
| highest conformal p-value | 91.6 | 92.3 | 93.9 | 71.8 | 76.6 | 88.8 |
| nearest stored input | 79.1 | 81.6 | 83.7 | 75.4 | 62.4 | 55.7 |
| accepted experts, nearest stored input | 79.6 | 82.3 | 84.1 | 74.9 | 62.4 | 57.6 |

### 3. Operators one after the other
| variant | seed | final mean exact % | forgetting (points) | experts at the end | extends | spawns | alarms | oracle-routed mean exact % |
|---|---|---|---|---|---|---|---|---|
| default | 0 | 26.6 | 15.8 | 4 | 3 | 4 | 1 | 40.5 |
| default | 1 | 27.1 | 15.4 | 4 | 3 | 4 | 1 | 40.0 |
| forced_spawn | 0 | 35.0 | 6.5 | 6 | 0 | 6 | 0 | 46.8 |
| forced_spawn | 1 | 34.9 | 7.9 | 6 | 0 | 6 | 0 | 47.9 |
| probe600 | 0 | 31.9 | 12.9 | 1 | 7 | 1 | 1 | - |

default, seed 0, exact match % after every block:

| after block | max | add | sub | mod | div | mul | experts |
|---|---|---|---|---|---|---|---|
| +max | 86.1 | 1.9 | 1.1 | 0.0 | 2.9 | 3.1 | 1 |
| +add | 66.0 | 39.2 | 1.9 | 0.1 | 0.4 | 0.4 | 1 |
| +sub | 66.7 | 34.0 | 29.9 | 2.0 | 0.3 | 0.1 | 1 |
| +mod | 38.3 | 24.8 | 22.7 | 51.7 | 2.7 | 1.5 | 2 |
| +div | 30.2 | 21.4 | 19.4 | 49.6 | 46.2 | 1.6 | 3 |
| +mul | 28.0 | 20.2 | 17.3 | 48.7 | 44.4 | 6.6 | 4 |

Where the test inputs went at the end, what each expert scores when it answers every example of an operator, and who was trained on each block (default, seed 0):

| operator | e0 | e1 | e2 | e3 |
|---|---|---|---|---|
| max | 68 | 15 | 11 | 6 |
| add | 78 | 10 | 7 | 5 |
| sub | 79 | 9 | 8 | 4 |
| mod | 2 | 94 | 3 | 1 |
| div | 4 | 3 | 91 | 2 |
| mul | 4 | 4 | 3 | 89 |

| expert | max | add | sub | mod | div | mul |
|---|---|---|---|---|---|---|
| e0 | 66.1 | 30.8 | 29.8 | 1.5 | 0.3 | 0.1 |
| e1 | 0.5 | 0.6 | 1.3 | 54.4 | 2.9 | 2.8 |
| e2 | 0.8 | 0.2 | 0.0 | 2.2 | 53.9 | 3.0 |
| e3 | 1.0 | 0.0 | 0.0 | 1.9 | 4.7 | 8.2 |
| actual pool (with the gate) | 26.7 | 18.6 | 16.9 | 46.6 | 44.0 | 7.1 |
| oracle routing | 66.1 | 30.8 | 29.8 | 54.4 | 53.9 | 8.2 |

| block | none | e0 | e1 | e2 | e3 |
|---|---|---|---|---|---|
| +max | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| +add | 0.2 | 99.8 | 0.0 | 0.0 | 0.0 |
| +sub | 0.1 | 99.9 | 0.0 | 0.0 | 0.0 |
| +mod | 0.2 | 0.0 | 99.8 | 0.0 | 0.0 |
| +div | 0.2 | 0.0 | 0.0 | 99.8 | 0.0 |
| +mul | 0.1 | 0.0 | 0.0 | 0.0 | 99.9 |

forced_spawn, seed 0, exact match % after every block:

| after block | max | add | sub | mod | div | mul | experts |
|---|---|---|---|---|---|---|---|
| +max | 86.1 | 1.9 | 1.1 | 0.0 | 2.9 | 3.1 | 1 |
| +add | 81.9 | 47.3 | 1.4 | 0.1 | 1.3 | 1.5 | 2 |
| +sub | 77.6 | 45.1 | 31.4 | 1.1 | 1.2 | 1.1 | 3 |
| +mod | 71.7 | 42.5 | 29.1 | 43.3 | 2.6 | 1.3 | 4 |
| +div | 67.3 | 40.2 | 26.6 | 42.0 | 39.5 | 1.3 | 5 |
| +mul | 65.1 | 38.8 | 26.2 | 40.5 | 38.1 | 4.9 | 6 |

What the interference probe saw (default, seed 0):

| step | cluster size | old-regime loss, parent | new-regime loss, dedicated child | old-regime loss rise (d_old) | new-regime gap | outcome |
|---|---|---|---|---|---|---|
| 626 | 128 | 0.030 | 0.0042 | +0.009 | +0.002 | extend |
| 627 | 32 | 0.033 | 0.0024 | +0.012 | -0.001 | extend |
| 1376 | 128 | 0.154 | 0.0114 | +0.028 | +0.016 | extend |
| 2126 | 128 | 0.163 | 0.0088 | +0.061 | +0.019 | spawn |
| 2751 | 128 | 0.157 | 0.0060 | +0.063 | +0.014 | spawn |

A learned router on the saved final pools (mean exact match % over the six operators):

| variant | seed | shipped gate | accept_then_logf | argmax_logf | argmax_p | nearest_proto | zscore_logf | learned router | oracle |
|---|---|---|---|---|---|---|---|---|---|
| default | 0 | 26.6 | 26.6 | 26.7 | 26.5 | 21.3 | 21.2 | 40.5 | 40.5 |
| default | 1 | 27.1 | 27.1 | 27.3 | 27.3 | 21.8 | 21.7 | 40.0 | 40.0 |
| forced_spawn | 0 | 35.0 | 35.0 | 35.1 | 28.7 | 22.6 | 17.7 | 46.8 | 46.8 |
| forced_spawn | 1 | 34.9 | 34.9 | 34.9 | 31.5 | 22.6 | 18.5 | 47.9 | 47.9 |

### 4. The growth policy inside the pool
| config | seed | unseen exact % | start params | end params | plateau trigger fired | would be true on this share of batches (offline replay) | probes run | skipped (backoff) | committed | probe training steps as % of the run's own training steps |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 0 | 26.1 | 35,084 | 35,084 | 51 | 53.6 % | 6 | 45 | 0 | 62 % |
| shipped | 1 | 28.0 | 35,084 | 150,988 | 52 | 54.5 % | 11 | 41 | 3 | 114 % |
| tuned | 0 | 37.3 | 133,772 | 133,772 | 51 | 52.7 % | 6 | 45 | 0 | 50 % |

### 5. Does it expand as needed? Start too small
| expert | parameters | unseen exact % (per seed) | mean |
|---|---|---|---|
| fixed, 16 hidden units | 6,300 | 10.9, 11.0 | 11.0 |
| fixed, 32 hidden units | 10,412 | 18.4, 19.1 | 18.8 |
| fixed, 64 hidden units | 18,636 | 28.4, 28.3 | 28.3 |
| fixed, 128 hidden units | 35,084 | 34.4, 32.9 | 33.6 |
| fixed, 512 hidden units | 133,772 | 37.3, 36.1 | 36.7 |
| **growing pool from 16** (shipped growth policy) | end: 18,636, 6,300 | 29.1, 11.0 | 20.1 |

Seed 0: ended with 64 hidden units and 1 block(s); 1 growth event(s) (step 359: widen +48, gain +2.8 %, p 0.001); 8 probes ran and 7 were rejected. Best gains of the rejected probes (relative held-out loss, minimum required 2 %): -0.2 % (p 0.801), -0.3 % (p 0.905), +0.1 % (p 0.263), -0.2 % (p 0.816), +0.8 % (p 0.001), -0.3 % (p 0.923), +0.1 % (p 0.383).

Seed 1: ended with 16 hidden units and 1 block(s); 0 growth event(s) ; 6 probes ran and 6 were rejected. Best gains of the rejected probes (relative held-out loss, minimum required 2 %): -0.7 % (p 0.970), +1.9 % (p 0.001), +1.5 % (p 0.004), +1.0 % (p 0.017), +1.5 % (p 0.002), +0.3 % (p 0.065).

Capacity added in the middle of the stream (start 128 units, then start 16 units):

| arm | parameters | unseen exact % (per seed) | mean | change against left alone |
|---|---|---|---|---|
| E128 left alone | 35,084 | 34.4, 32.9 | 33.6 | +0.0 |
| E512 from the start | 133,772 | 37.3, 36.1 | 36.7 | +3.1 |
| E128, widen +64 at half | 51,532 | 34.3, 32.5 | 33.4 | -0.2 |
| E128, widen +384 at half | 133,772 | 34.4, 33.5 | 33.9 | +0.3 |
| E128, widen +896 at half | 265,356 | 33.8, 33.0 | 33.4 | -0.3 |
| E128, one more block at half | 68,364 | 34.1, 33.3 | 33.7 | +0.1 |

| arm | parameters | unseen exact % (per seed) | mean | change against left alone |
|---|---|---|---|---|
| E16 left alone | 6,300 | 10.9, 11.0 | 11.0 | +0.0 |
| E16 widened to 128 at half | 35,084 | 28.9, 29.3 | 29.1 | +18.1 |
| E16 widened to 512 at half | 133,772 | 32.3, 31.8 | 32.0 | +21.1 |
| E16, one more block at half | 10,796 | 10.8, 11.9 | 11.3 | +0.4 |
| E128 from the start | 35,084 | 34.4, 32.9 | 33.6 | +22.7 |
| E512 from the start | 133,772 | 37.3, 36.1 | 36.7 | +25.7 |

Probe length against the gain the policy measures (16-wide expert, widened candidates, held-out cross-entropy):

| probe steps | control cross-entropy | widen+8 gain | widen+48 gain | widen+112 gain | widen+496 gain |
|---|---|---|---|---|---|
| 480 | 1.192 | +0.5 % | +1.7 % | +2.1 % | +1.2 % |
| 1000 | 1.199 | +1.3 % | +2.0 % | -0.7 % | -8.6 % |
| 2000 | 1.208 | -0.1 % | -4.1 % | -17.5 % | -66.3 % |
| 5000 | 1.220 | -2.1 % | -21.4 % | -82.5 % | -193.2 % |

Lowered minimum gain (start 16 units):

| min gain | seed | unseen exact % | end hidden units | end blocks | end parameters | growth events (kind +units, gain) |
|---|---|---|---|---|---|---|
| 0.005 | 0 | 19.4 | 36 | 1 | 11,440 | widen +8 (+1.2 %); widen +12 (+0.8 %) |
| 0.005 | 1 | 27.5 | 64 | 1 | 18,636 | widen +48 (+1.9 %) |
| 0.0 | 0 | 19.4 | 36 | 1 | 11,440 | widen +8 (+1.2 %); widen +12 (+0.8 %) |
| 0.0 | 1 | 27.5 | 64 | 1 | 18,636 | widen +48 (+1.9 %) |

### 6. Real text (existing runs)
| arm | accuracy % | what it is |
|---|---|---|
| kNN on all stored data | 87.3 | lookup ceiling |
| one network trained jointly on all classes (128 wide) | 86.2 | parametric ceiling |
| experts told the regime at test time (oracle routing) | 81.8 | what splitting the data into experts costs, with perfect routing |
| replay network (same-size replay buffer) | 83.7 | |
| GrowMoE with the stage-3 gate | 83.7 | |
| GrowMoE today (no stage 3) | 71.8 | routing failure |

## Interpretation (what we understood, and on what basis)
Plain words. **Why it falls behind dense networks on arithmetic.** It is not the experts and not the routing. The pool trains an expert the way a streaming learner must: every batch is seen once, a few steps are taken on it, and only a bounded set of old examples can be replayed, at a constant learning rate. A dense network in the benchmark saw all 298,000 examples forty times with a decaying rate. Give the same large dense network (MLP-L, 2.2 million parameters, 63.8 % offline) the pool's recipe with a 20,000-example reservoir and it drops to 32.8 %, which is what the pool's own 16-times smaller expert gets (36.7 %) and what MLP-S gets (36.1 %). From 63.8 to 36.7 the points split into about 15 for network size (usable only when training offline), 8 for the number of examples that can be kept and replayed (reservoir 20,000 against 300,000: 36.7 against 44.7) and about 4 for one pass with a constant rate (a decaying rate alone adds 4.0). Repeating each batch five times instead of once changes nothing (36.7 against 37.1). An offline learner that may keep only 20,000 examples scores 12 to 17 %, so the pool uses the data it passes without keeping it far better than storage alone would. The arithmetic table of E-026 was therefore not an equal-memory fight: the dense networks could keep and revisit everything.
**Why it never grows or splits in a mixed stream.** New experts are created only when something changes (a batch looks unfamiliar or the loss jumps). A stationary mixture never changes, so one expert absorbs all six operators, although the operators are separable: the gate's own density model names the operator 87 % of the time. What it cannot do is call an operator new: an expert that never saw operator j accepts 88 % of j's inputs as familiar (its own operator's: 98 %), because the operator is one of twelve equally weighted input coordinates (changing the operator moves an input as far as changing one digit). With perfect splitting and routing the same streaming recipe would give about 47 % instead of 37 %.
**Why operator-by-operator learning ends at 26.9 %.** Three separate things, in this order of size. (1) Routing at answer time: with the pool's final experts the best expert per operator would reach 40.3 %, the gate delivers 26.9 %; newer single-operator experts attract 21 to 32 % of the inputs of max, add and sub by the end of the stream (11 to 19 % right after the first newer expert appeared), because the expert that holds three operators spreads its density over three clusters, and the same loss is 12.4 points in the one-expert-per-operator pool. There is NO training leak: 99.8 to 100 % of every block trains exactly one expert, so the damage is done when answering, not when learning. (2) The extend-versus-spawn probe: it lets add and sub share the expert of max, and sharing costs 7.1 points even with perfect routing; the probe decides on losses of samples the networks have been trained on (0.001 to 0.2 nats, the task's real held-out cross-entropy is above 0.7) with differences of 0.01 to 0.06 against a tolerance of 0.05, a coin flip; a ten times longer probe always extends (one expert, 31.9 %), and forbidding extends gives 34.9 % against 26.9 % with forgetting 6.5 / 7.9 against 15.8 / 15.4 points. (3) The streaming recipe itself: even with one expert per operator and perfect routing the pool reaches 47.4 %, against 56.4 % for the experts of the benchmark that are told the operator and trained offline.
**Can it expand vertically and horizontally as needed?** The mechanism does what it says: widening and deepening are exactly function-preserving (maximum output change 0.0), and the added capacity is used (a 16-wide expert widened half-way through the stream goes from 11.0 % to 29.1 % at 128 units and 32.0 % at 512; one more block of the same width changes nothing, 11.3 %). At pool level horizontal growth tracks the number of regimes when they are far apart in input space (10, 30 and 100.7 experts for 10, 30 and 100 regimes in E-025). What does not work is the decision to use it. Started with 16 hidden units on a task where fixed experts of 16 / 32 / 64 / 128 / 512 units score 11.0 / 18.8 / 28.3 / 33.6 / 36.7 %, the policy left one seed at 16 units (11.0 %) and stopped the other at 64 (29.1 %). Its yardstick is the held-out cross-entropy after training on a fixed stored set; that measure turns against capacity as the probe gets longer (widening by 112 units: +2.1 % at 480 steps, -82.5 % at 5,000) although the held-out accuracy still rises, so no probe length shows the benefit; a minimum gain of 0.5 % or of zero changes nothing (it grows by 8 to 48 units and stops: the p-value and better-than-now gates and the doubling back-off). In the shipped configuration the policy decides on 51 held-out examples (gains swing from -58 % to +49 %), commits by chance (0 and 3 times in two seeds) and its probes cost 62 to 114 % extra training steps. Where capacity is NOT the limit it is right to refuse: widening an adequate 128-wide expert half-way gives +0.3 points. Limits: every expert is a residual MLP with LayerNorm (width at most 1024, depth at most 8, at most 256 experts), the front end is fixed, children copy the parent's current width and depth, and nothing is ever shrunk except whole experts (merge, archive). It cannot change the type of network.
Technical statement. Measured: exact match on a held-out grid, held-out cross-entropy and accuracy of the growth probes, routing shares, conformal acceptance rates, oracle routing as the best existing expert per operator. Assumed: that the sequential and mixed streams of E-026 are representative of regimes that overlap in input space; that two seeds suffice for effects of 7 points or more (the differences between variants are 7 to 14 points and replicate in both seeds).

## Caveats and what went wrong
- Exploratory; one or two seeds per arm; probe600 has one seed; arithmetic only (the real-text and synthetic results are from earlier runs). Effects of a point or two (annealing, normalised inputs 3.7 points, mid-stream widening of an adequate expert) are within seed noise and are reported as such.
- The routing-loss figure uses a per-operator oracle (the best existing expert for each operator), which is an upper bound, not a per-example oracle.
- The learned router is trained on the labels the pool itself produces; it is evidence that routing is the bottleneck, not a tested product component.
- My first reading of the growth probe was wrong: I expected a longer probe to reveal the value of capacity, and the measurement showed the opposite (the yardstick penalises capacity); the horizon test was added because of it. My suspicion of a training leak was also wrong.
- A first set of equal-stored-example arms had too few optimiser steps (160 and 1,600) and was superseded by arms with 3,000 and 23,000 steps; a name collision (`Path.with_suffix` on a name with a dot) overwrote two growth-threshold result files and they were rerun.
- A power cut at about 13:26 killed four 25-minute runs that had no checkpoint; they were rerun with block-level checkpoints (resumed and uninterrupted runs give identical results on a tiny stream) and every result file is now written atomically.

## Consequence
Ranked by measured size, with the evidence for the expected gain (nothing below has been changed in the product):
1. Routing among operator-like experts: a learned router on the pool's own assignments (this entry's test: 40.3 % against 26.9 % for the shipped gate, oracle 40.3 %; one-expert-per-operator pool 47.4 % against 34.9 %); keep the density gate for novelty and abstention.
2. The extend-versus-spawn probe: replace its statistic (losses of trained-on samples) by held-out, long-horizon performance, or spawn whenever the batch-level gate calls a batch new and let merge clean up (merge itself is untested): +8.0 points and -8.4 points of forgetting measured.
3. The growth policy: a yardstick that does not penalise capacity (accuracy or prequential loss on fresh stream data, train-versus-held-out gap, level of the training loss), more evidence than 51 examples, selective trigger, no cheapest-first creeping: starving experts gain 18 to 21 points when widened.
4. A split rule for stationary mixtures and a per-expert learning-rate schedule (+4) and a larger reservoir when memory allows (+8 for 300,000 against 20,000).
Decisions for the user (not taken): whether to implement 1 to 3 as a new stage with its own pre-registration; whether the arithmetic benchmark should be re-run at equal memory.

## Evidence
- `research/growmoe_results/diagnosis/RESULTS_DIAGNOSIS.md`, `research/growmoe_results/diagnosis/figures/fig_summary.png`, `research/growmoe_results/diagnosis/figures/fig_ladder.png`
- `research/growmoe_results/diagnosis/ladder/` (82 cells), `seqprobe/`, `seqprobe2/`, `seqprobe3/`, `growth/`, `grow_from_small/`, `widen_midstream/`, `grow_threshold/`, `horizon/`, `separability.json`, `routing_rule.json`, `router.json`
- scripts: `diag_lib.py`, `ladder_cells.py`, `run_ladder.py`, `run_seqprobe.py`, `probe_separability.py`, `probe_routing_rule.py`, `probe_growth.py`, `probe_grow_from_small.py`, `probe_widen_midstream.py`, `probe_horizon.py`, `probe_grow_threshold.py`, `analyse_router.py`, `report_diagnosis.py`, `jobqueue.py`, `RESUME_DIAGNOSIS.cmd`

## Reproduce
```
python research/growmoe_results/diagnosis/jobqueue.py run --wait      # everything, resumable; after a power cut run RESUME_DIAGNOSIS.cmd
python research/growmoe_results/diagnosis/report_diagnosis.py
```
About 2 hours of compute on a 12-thread PC with an RTX 3060 for the whole study (the four sequential runs take 25 to 35 minutes each, in parallel).

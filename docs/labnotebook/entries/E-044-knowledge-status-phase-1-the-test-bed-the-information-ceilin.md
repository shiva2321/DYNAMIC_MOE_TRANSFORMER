# E-044 - Knowledge status Phase 1 - the test bed, the information ceiling of the readers and the learned map

- **ID:** E-044
- **Date:** 2026-10-27
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B (script-generated and multi-seed, dev seeds, predictions written before the numbers but not a frozen pre-registration)
- **Code / commit:** branch `knowledge-status-phase1`; f59168e (E1 table), efc99cb (learned map, tuning, predictions), c16ca50 (R5 results); worktree D:/shm_ks
- **People:** Claude designed, built and reviewed (no independent reviewer yet); the user decided to build the learned map and, after the near miss, to go on (option A, 2026-10-27)

## Question
Before any detector exists: can ANY signal in a reader (a way of turning a sentence into a vector) tell (a) the same skill in a new wording, (b) near-twins that differ only in the operator word, (c) taught skills from skills that were never taught? If no reader can, the detector is not the bottleneck and building one is pointless; if one can, which reader, and does a small learned map help?

## Why we asked it
The expert pool makes two costly errors (E-034: a skill learned twice, about 4 points; E-030: a new skill absorbed by an old one, about 7 points). Both are errors of a novelty alarm, and an alarm is only as good as the vectors it looks at. The design of 2026-10-26 (`docs/superpowers/specs/2026-10-26-knowledge-status-and-routing-design.md`) therefore puts an information-ceiling test (E1) before any detector.

## Prediction (written BEFORE the numbers)
Copied from `research/knowledge_status/PROGRESS.md` by script:

## 2026-10-27, Task 8 (before the numbers): what I expect E1 to show

Written before any E1 result exists, so that we can see afterwards where I was wrong. E1 asks three things of each representation, with no detector involved: B1a, can a trained probe tell which of the six operations (or "none") a message asks for, when it was taught on three wordings and tested on five new ones? B1b, with no training at all, is a request closer to the same request reworded than to its near-twin (same wrapper, same numbers, only the operation changed)? B1c, with no training on untaught skills, are parts of taught skills closer to the taught examples than parts of untaught skills? The draft bars are 0.95, 0.95 and 0.90 (chance 0.14, 0.50, 0.50).

1. **R0, the letter-pattern counter.** B1b should fail badly, well below 0.5 (I guess 0.05 to 0.30): the near-twin shares the wrapper and the numbers, the reworded request shares neither, so to a letter-counter the twin is the closer one. B1a should be middling (0.35 to 0.60): "plus" or "+" is visible inside the three teaching wordings, but stories have no operation word and typos break the letters. This is what makes the test bed valid (B2): a word-matcher must fail.
2. **Sentence models and BERT (R1, R2, R3).** B1a 0.50 to 0.80 (strong on "arithmetic against everyday text", weak on "which operation"); B1b below 0.5 (0.15 to 0.45): same wording with another operator looks more alike than the same operator worded differently; B1c 0.55 to 0.75.
3. **Language-model states after the "operation requested is" line (R4).** The best chance, because the model has just been asked. B1a for Qwen at the middle layers 0.75 to 0.95, GPT-2 0.60 to 0.85. B1b not above 0.6 (a plain distance does not care about the question); B1c 0.55 to 0.75.
4. **Overall.** I expect none of R1 to R4 to pass all three draft bars, with B1b (pure distance, no training) failing for all of them. That would mean the learned map (Task 9) is needed. A pass of Qwen on B1a alone would not surprise me; a pass on B1b by any off-the-shelf model would, and I would then suspect the test bed before believing it.

## 2026-10-27, Task 9 (before the report-seed numbers): what I expect from the learned map

**What we did.** You decided to build the learned map (R5). Before looking at the main table again, three things happened that you should know about, in order. (1) I added a deeper look at the language model: layers 22 and 26 (the plan had 9, 14, 18). With no map at all they are far better than layer 18 (see below), so a bigger part of the earlier "fail" was *where we read the model*, not only the wording problem. (2) I built the map (`learned_map.py`, 7 tests, each of three deliberate breakages of the code is caught by a test). It is one small linear layer that is trained so that the same operation in different words moves closer together and the same wording with another operation moves apart, with extra push on exactly those near-twins. It refuses by construction to see an untaught skill or a test wording (S4 to S8). (3) I tuned it on **other** dev seeds (105 to 107; the E1 table uses 100 to 104) so that tuning cannot flatter the table. I also re-ran the whole E1 baseline under the new code: all 55 earlier numbers came out identical, so adding the map changed nothing about the old results.

**Tuning facts (seeds 105 to 107, no report-seed number was looked at for this).**

| reader | plain: B1a / B1b / B1c | with the map: B1a / B1b / B1c (B1b with a story partner) |
|---|---|---|
| mpnet | 0.80 / 0.10 / 0.74 | 0.82 / 0.94 (0.83) / 0.76 |
| MiniLM 384 | 0.81 / 0.03 / 0.74 | 0.79 / 0.95 (0.84) / 0.78 |
| Qwen layer 26 | 0.94 / 0.62 / 0.84 | 0.92 / 0.999 (0.997) / 0.89 |

The map settings hardly matter: 12 variations (width, strength, number of steps, sharpness, an extra hidden layer) all land within about 0.01 of each other, so I keep one default operating point for every reader (64 numbers out, weight decay 0.01, near-twin push 4x, sharpness 0.1, 300 steps). The layer scan on the same seeds: layer 20 B1b 0.05, layer 22 0.47, layer 24 0.55, **layer 26 0.62**, layer 28 0.37; so the model seems to settle "which operation" between layers 20 and 22, and layer 26 is the best place to read. The map is therefore run on three readers: mpnet, MiniLM and Qwen layer 26.

**Prediction for seeds 100 to 104** (written before those numbers exist; they should look like the tuning seeds above, give or take 0.03):
1. **Qwen layer 26 + map:** B1b 0.99 or more, also for story partners; B1a 0.90 to 0.95; B1c 0.86 to 0.92. I expect B1b to pass its 0.95 bar and B1a and B1c to land within about 0.03 of their bars: whether the mean crosses 0.95 or 0.90 is close to a coin flip, and B1a is more likely to stay just below.
2. **mpnet + map and MiniLM + map:** B1b 0.93 to 0.96 (just at the bar, driven by the stories: 0.83), B1a 0.78 to 0.86 and B1c 0.72 to 0.80, both clearly failing.
3. **A caution I am writing down now so we do not forget it.** "B1b with a taught-wording partner = 1.000" is *in-distribution*: those wrapper phrases were seen during the map's training with other numbers, so it is the expected result, not evidence of generalising. The real evidence of wording invariance is the story column (never taught) and B1a on S4 to S8. And all of this is on the **dev pool**: its templates are the ones I looked at while designing, so dev numbers are optimistic. The confirmatory pool has different templates and is the only honest test.

## Method (how)
- Test bed: six skills (add, sub, mul, div, mod, max) on 1 to 2 digit operands; eight wording families, S1 symbols, S2 number words, S3 verbal phrasing (these three are the TEACHING wordings), S4 short stories, S5 operands in the other order, S6 chatter around the request, S7 typos / case / spacing noise, S8 register (all held out); near-twins (only the operator changed) and look-alike traps as controls; template pool `pool_dev.py` (the confirmatory pool does not exist yet and shares nothing with it).
- Readers: R0 hashed n-grams (the word-matcher control), R1 MiniLM, R2 mpnet, R3 BERT, R4 last-token hidden state of GPT-2 and of Qwen2.5-1.5B at several layers (a fixed closing prompt), R5 a small linear contrastive map (64 dimensions) on top of a frozen reader, fitted on the teaching wordings of taught skills only (it refuses S4-S8 and untaught skills).
- Measures: B1a a linear probe for the skill, trained on S1-S3, tested on S4-S8 (bar 0.95, chance 0.14); B1b near-twin AUROC by distance alone, no training (bar 0.95, chance 0.50); B1c taught against untaught by distance to the nearest taught prototype over 20 teaching assignments (bar 0.90, chance 0.50); B2 validity of the test bed: R0 must fail B1a or B1b by 0.15 or more.
- Seeds: report seeds 100-104; tuning seeds 105-107 (the Qwen layer scan and 12 one-at-a-time variations of the map's settings); the bars are draft values.
- Commands: `python run_e1.py --pool dev --seeds 100-104 --reps all,R4_qwen_L22,R4_qwen_L26,R5_R2_mpnet,R5_R1_minilm384,R5_R4_qwen_L26`, `python layer_scan.py`, `python tune_map.py --base R4_qwen_L26 --seeds 105-107 --stage 1` (run from `research/knowledge_status` with the repository root on PYTHONPATH).

## Results (what came out)
Report seeds 100-104, mean with [min - max] (rows copied by script from `research/knowledge_status/results/E1_TABLE.md`):

# E1, the information-ceiling table (pool `dev`, seeds 100-104)

Mean over seeds, with [min - max]. Draft bars: B1a >= 0.95 (chance 0.14), B1b >= 0.95 (chance 0.50), B1c >= 0.90 (chance 0.50). Centred = the mean of the teaching embeddings is subtracted first (the primary value); raw = cosine on the vectors as they are. The two split columns repeat B1b for the triples whose reworded partner is in a taught wording (S1 to S3) and for those whose partner is a story (S4, never taught). R5_<id> is the learned map on top of <id>.

| representation | seeds | B1a probe (whole message) | B1a probe (gold span) | B1b near-twin AUROC (centred) | B1b raw | B1b, partner in a taught wording | B1b, partner a story | B1c taught-vs-untaught AUROC (centred) | B1c raw |
|---|---|---|---|---|---|---|---|---|---|
| R0_hash | 5 | 0.54 [0.53 - 0.56] fail | 0.72 [0.70 - 0.74] | 0.00 [0.00 - 0.01] fail | 0.01 [0.01 - 0.01] | 0.00 [0.00 - 0.01] | 0.00 [0.00 - 0.00] | 0.68 [0.67 - 0.68] fail | 0.66 [0.65 - 0.67] |
| R1_minilm384 | 5 | 0.81 [0.79 - 0.82] fail | 0.81 [0.80 - 0.83] | 0.03 [0.02 - 0.04] fail | 0.04 [0.03 - 0.05] | 0.05 [0.04 - 0.07] | 0.00 [0.00 - 0.00] | 0.74 [0.73 - 0.74] fail | 0.69 [0.68 - 0.70] |
| R2_mpnet | 5 | 0.80 [0.79 - 0.81] fail | 0.81 [0.79 - 0.84] | 0.10 [0.08 - 0.12] fail | 0.12 [0.10 - 0.13] | 0.15 [0.13 - 0.18] | 0.00 [0.00 - 0.01] | 0.74 [0.73 - 0.75] fail | 0.71 [0.69 - 0.72] |
| R3_bert | 5 | 0.62 [0.60 - 0.64] fail | 0.74 [0.73 - 0.75] | 0.00 [0.00 - 0.00] fail | 0.02 [0.01 - 0.03] | 0.00 [0.00 - 0.00] | 0.00 [0.00 - 0.00] | 0.62 [0.60 - 0.63] fail | 0.56 [0.55 - 0.57] |
| R4_qwen_L18 | 5 | 0.85 [0.82 - 0.87] fail | 0.85 [0.80 - 0.88] | 0.04 [0.03 - 0.04] fail | 0.06 [0.05 - 0.06] | 0.05 [0.04 - 0.06] | 0.01 [0.00 - 0.01] | 0.61 [0.60 - 0.62] fail | 0.57 [0.56 - 0.58] |
| R4_qwen_L26 | 5 | 0.93 [0.92 - 0.93] fail | 0.93 [0.92 - 0.94] | 0.64 [0.59 - 0.70] fail | 0.62 [0.58 - 0.69] | 0.75 [0.71 - 0.81] | 0.42 [0.36 - 0.46] | 0.84 [0.83 - 0.86] fail | 0.87 [0.86 - 0.89] |
| R5_R2_mpnet | 5 | 0.80 [0.78 - 0.84] fail | 0.80 [0.79 - 0.84] | 0.93 [0.92 - 0.95] fail | 0.93 [0.92 - 0.94] | 1.00 [1.00 - 1.00] | 0.79 [0.75 - 0.83] | 0.76 [0.74 - 0.78] fail | 0.75 [0.73 - 0.76] |
| R5_R1_minilm384 | 5 | 0.79 [0.78 - 0.81] fail | 0.80 [0.79 - 0.83] | 0.95 [0.93 - 0.96] fail | 0.94 [0.93 - 0.95] | 1.00 [1.00 - 1.00] | 0.85 [0.82 - 0.87] | 0.78 [0.76 - 0.79] fail | 0.77 [0.75 - 0.78] |
| R5_R4_qwen_L26 | 5 | 0.92 [0.91 - 0.93] fail | 0.93 [0.92 - 0.94] | 1.00 [0.99 - 1.00] PASS | 1.00 [0.99 - 1.00] | 1.00 [1.00 - 1.00] | 0.99 [0.98 - 1.00] | 0.89 [0.88 - 0.89] fail | 0.87 [0.86 - 0.88] |

**B2 validity of the test bed:** R0 is below the bar by +0.41 on B1a and +0.95 on B1b (a gap of 0.15 or more on either is required): test bed VALID

Qwen layer scan on the tuning seeds 105-107 (copied from `research/knowledge_status/results/layer_scan/LAYER_SCAN.md`):

Qwen layer scan, plain representation (no map), tuning seeds 105-107, mean over seeds.

| layer | B1a | B1b | B1b taught | B1b story | B1c |
|---|---|---|---|---|---|
| 20 | 0.786 | 0.050 | 0.061 | 0.019 | 0.592 |
| 22 | 0.942 | 0.469 | 0.581 | 0.241 | 0.816 |
| 24 | 0.936 | 0.545 | 0.662 | 0.301 | 0.830 |
| 26 | 0.935 | 0.619 | 0.722 | 0.402 | 0.837 |
| 28 | 0.919 | 0.373 | 0.483 | 0.157 | 0.811 |

Settings of the map, one at a time (tuning seeds 105-107, copied from `research/knowledge_status/results/map_tuning/TUNING_R4_qwen_L26_stage1.md`):

Map tuning on `R4_qwen_L26`, seeds 105-107, 6 teaching assignments per seed. Mean over seeds; score = mean of B1a, B1b, B1c.

| setting | B1a | B1b | B1b taught | B1b story | B1c | score |
|---|---|---|---|---|---|---|
| tau0.2 | 0.943 | 0.999 | 1.000 | 0.997 | 0.892 | 0.945 |
| ep600 | 0.933 | 0.999 | 1.000 | 0.998 | 0.892 | 0.941 |
| tw1 | 0.927 | 0.999 | 1.000 | 0.996 | 0.891 | 0.939 |
| dim128 | 0.925 | 0.999 | 1.000 | 0.997 | 0.890 | 0.938 |
| wd1e-1 | 0.922 | 0.999 | 1.000 | 0.996 | 0.891 | 0.937 |
| default | 0.921 | 0.999 | 1.000 | 0.996 | 0.891 | 0.937 |
| wd1e-3 | 0.921 | 0.999 | 1.000 | 0.996 | 0.891 | 0.937 |
| dim32 | 0.920 | 0.998 | 1.000 | 0.995 | 0.891 | 0.937 |
| tw16 | 0.918 | 0.999 | 1.000 | 0.997 | 0.889 | 0.935 |
| ep150 | 0.916 | 0.998 | 1.000 | 0.995 | 0.889 | 0.935 |
| tau0.05 | 0.919 | 0.997 | 1.000 | 0.992 | 0.882 | 0.933 |
| hid256 | 0.903 | 0.997 | 1.000 | 0.989 | 0.877 | 0.926 |

Where the remaining misses are (one seed, copied from `research/knowledge_status/results/E1_DIAG_R5_FAILURES.txt`):

R5 on R4_qwen_L26, dev seed 100

B1a by test wording family (probe C 0.1): overall 0.913
   S4      n= 206  accuracy 0.86
   S5      n=  86  accuracy 1.00
   S6      n= 200  accuracy 0.92
   S7      n= 190  accuracy 0.86
   S8      n= 218  accuracy 1.00
   other   n=  50  accuracy 1.00
   trap    n=  50  accuracy 0.64
   unseen  n=  50  accuracy 0.94
   by class: add 0.99, div 0.79, max 0.93, mod 0.92, mul 0.96, none 0.86, sub 0.95
   most common mistakes (true -> said): div->mod x27, none->max x21, mod->sub x9, sub->none x7, max->none x6, mul->none x3

B1c by the untaught skill (AUROC of that skill's parts against the taught skills' parts, mean over the teaching assignments where it is untaught)
   add: AUROC 0.92 (n=11)  nearest taught skill: sub 64%, mul 19%, max 14%
   sub: AUROC 0.94 (n=7)  nearest taught skill: add 53%, max 15%, mul 13%
   mul: AUROC 0.96 (n=11)  nearest taught skill: div 30%, add 29%, sub 17%
   div: AUROC 0.77 (n=11)  nearest taught skill: mod 77%, sub 17%, mul 3%
   mod: AUROC 0.83 (n=8)  nearest taught skill: div 66%, sub 30%, max 2%
   max: AUROC 0.92 (n=12)  nearest taught skill: add 35%, sub 31%, div 21%
   untaught skill whose parent skill is taught (Related in the design): AUROC 0.88 (n=34); no parent taught: 0.90 (n=26)

## Interpretation (what we understood, and on what basis)
In plain words: ordinary readers file sentences by how they are worded, not by what they ask. The same request in a new wording lands FARTHER from its twin than a sentence with the operator changed lands (B1b is 0.00 to 0.10 for every off-the-shelf reader at the plan's layers). Deeper layers of Qwen read the operation much better (the jump between layers 20 and 22 in the scan), and a small map trained to pull rewordings together and push near-twins apart repairs the ordering almost completely for Qwen layer 26 (B1b 1.00, and 0.99 for story partners, a wording the map never saw). B1a (what a trained probe can still read) is not improved by a map, as it must be: a probe measures the information in the vectors and a map only re-sorts distances. Measured: the table. Assumed: that the dev numbers carry over to the confirmatory pool, whose templates differ.

## Caveats and what went wrong
- The dev templates were seen while the test bed was designed, so the dev numbers are optimistic; the confirmatory pool will be disjoint and sealed.
- B1b with a partner in a TAUGHT wording is in-distribution (1.00 for the map); the evidence of real wording invariance is the story partner.
- Qwen already knows multiplication from its pretraining; "Unseen" means never taught to OUR system, not unknown to the reader.
- The MiniLM and mpnet readers with the map do not pass (B1a about 0.80, B1c about 0.77): the pool's current input (MiniLM) would have to be replaced by a larger reader.
- The three bars are draft values. R5 on Qwen layer 26 passes one (B1b) and misses two narrowly (B1a 0.92 against 0.95, B1c 0.89 against 0.90). The plan's rule was to STOP when R5 also fails B1; the user chose to go on (option A, D-40), and no bar was moved: B1 is reported as one pass and two misses.
- The off-the-shelf rows were re-run under the new code stamp after the mapper hook was added: all 55 numbers were identical to the committed ones.

## Consequence
The reader for the status machine is the learned map on Qwen layer 26. The status machine and the standard detectors are built and evaluated in E-045.

## Evidence
- `research/knowledge_status/results/E1_TABLE.md` - the E1 table (generated by `run_e1.py`)
- `research/knowledge_status/results/e1` - one result file per reader and seed
- `research/knowledge_status/results/layer_scan/LAYER_SCAN.md` - the Qwen layer scan
- `research/knowledge_status/results/map_tuning/TUNING_R4_qwen_L26_stage1.md` - the map's settings
- `research/knowledge_status/results/E1_DIAG_R5_FAILURES.txt` - where the misses of R5 are
- `research/knowledge_status/results/E1_DIAG_B1A_BY_FAMILY.txt` - B1a by wording family
- `research/knowledge_status/PROGRESS.md` - the predictions and the plain-English account
- `research/knowledge_status/ceiling.py`, `research/knowledge_status/learned_map.py`, `research/knowledge_status/run_e1.py` - the code

## Reproduce
```
cd research/knowledge_status
PYTHONPATH=<repo root> python run_e1.py --pool dev --seeds 100-104 --reps all,R4_qwen_L22,R4_qwen_L26,R5_R2_mpnet,R5_R1_minilm384,R5_R4_qwen_L26
```

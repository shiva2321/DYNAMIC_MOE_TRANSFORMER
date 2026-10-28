# Progress log (plain words)

One short block per finished task: what we did, what we learned, what comes next. Written for a reader who is learning along, so technical terms are explained the first time they appear.

## 2026-10-26, Task 1: reading the closest papers

**What we did.** Before writing any code we read 16 of the closest papers in full or in main text (14 read in text form, 2 only as abstracts or listings) and wrote what each does, what it does NOT test, and how it differs from our design. The file is `research/knowledge_status/RELATED_WORK.md`.

**What we learned.**
- The field already has the skeleton of our idea: several signals mixed to decide "new", an "unknown pile" of items waiting to be grouped, and "promote only solid groups". The newest paper that does this (2026) still finds only 39 to 61 % of the new intents it should, and says it cannot separate closely related intents.
- None of the intent papers measures our headline question: how often a plain rewording of a known thing is wrongly called new. They measure how well "new" things are caught, not how often "old" things in new clothes are falsely flagged. We will measure it.
- Testing a group of inputs together is an established idea (it needs enough inputs; with 10 inputs it catches about 19 % of shifts in one paper, which matches what we found: 16 or more are needed).
- Learners are known to fail at reusing known parts in new combinations (the SCAN benchmark), and the one system that does build new skills out of old ones from solved examples (DreamCoder) works on tidy symbolic problems, not on messy language. That is the gap our trace idea aims at.
- Two papers (MINAS on novelty in data streams, and Focus on Facts on embeddings that blur near-copies) were only reachable as abstracts. We will read them in full before any paper draft.

**What it changes.** Two extra baselines go into the plan (Relative Mahalanobis, and using a classifier's outputs as the reduced space for group tests), the false-novelty rate on rewordings becomes our headline number, and we treat "the leftover after explaining away" as a claim to test, because one image paper found the leftover carries little signal.

**Next.** Task 2: the small foundation (the six skills, the family tree of skills, and safe file-writing helpers) with tests first.

## 2026-10-26, Task 2: the foundation

**What we did.** Wrote the smallest base everything else stands on: the six skills (add, subtract, multiply, divide, remainder, larger-of), the "family tree" of skills (multiplication is built from addition, division from subtraction, remainder from division and subtraction, larger-of from subtraction), the exact right answer for any pair of numbers, and three safe helpers for a PC that loses power (write a file so it is never half-written, stamp the code that made a result, tell whether a result file is finished and valid). Ten small tests check each of these, and the answers agree with the arithmetic code we already trust. We also reserved the random-number ranges: 100 to 199 for design and tuning, 400 to 429 for the one-time final run.

**What we learned.** Nothing about the science yet; this is plumbing. The tests fixed two things that would have bitten later: dividing or taking a remainder by zero is refused loudly, and a half-written result file is never mistaken for a finished one.

**Next.** Task 3: the test messages (the same arithmetic request written eight different ways) with exact labels.

## 2026-10-26, Task 3: the test messages

**What we did.** Built the machine that writes the test messages. Every message is one arithmetic request (add, subtract, multiply, divide, remainder, larger-of) written in one of eight styles: symbols ("17 + 5"), number words ("seventeen plus five"), verbal ("the product of 7 and 5"), little stories, operands in the other order, chatter around the request, typos and odd casing, and formal / casual / terse tone. Each message carries an exact label: which skill, which two numbers, the true answer, and exactly where in the text the request sits. A message is built from a "frame" (the wrapper: tone, punctuation, chatter) around a "core" (the request itself). Only the core names the operation, so the wrapper can never give the answer away; a test confirms that the wrapper alone predicts the skill about as well as a blind guess (17.6 % on average over ten runs, range 16.4 to 19.3 %, against 16.7 % for a blind guess; the small excess is because one style only exists for three of the six skills). 16 new tests check the labels with a separate, independent reader of the text, zero and negative cases, determinism, and that typos never touch digits.

**What we learned.** The plumbing works and the messages look natural enough: for example "Thanks for helping me out. Here it is: what is ninety-six times five" and "Each of 4 children gets 2 books. How many books are given out? Thanks!". One design fact surfaced: the "other order" style can only exist for operations where order does not matter (add, multiply, larger-of), so that style is deliberately limited to those three.

**Next.** Task 4: the controls a word-matcher should fail (the same sentence with only the operation changed, look-alike traps) and everyday-sentence chatter taken from real text, with checks that the chatter never smuggles in numbers or operation words.

## 2026-10-26, Task 4: the controls a word-matcher should fail

**What we did.** Added the pieces that make the test fair. (1) Near-twins: the same message with only the operation changed ("Q: 13 plus 22" becomes "Q: 13 times 22"; numbers, wrapper and wording stay), each with its own exact label. (2) Traps: sentences that use arithmetic words or numbers for something else ("Add me to the guest list for the 7th of July"), labelled as not-a-request. (3) Everyday sentences ("Other content") taken from real, locally stored text: 10,224 clean sentences from banking questions and 38,803 from news, which is also where the chatter around requests can come from. (4) A kind of text that will never be taught (1,675 science questions), the easy cross-type control. A "clean" sentence has no digits, no operation word (matched as whole words, so "modern" is fine but "times" is not) and no arithmetic symbol, and is 5 to 40 words long. Nine new tests check all this, including that a missing dataset gives a loud error that names it instead of silently using less data.

**What we learned.** The real sentences are messier than our templates (news headlines run into their first line; some science questions end in a blank). That is useful: it means "Other content" looks like real text, not tidy test text. Near-twins only exist for the simple "a plus b" style requests (not for stories or the larger-of skill), so the near-twin test will use those.

**Next.** Task 5: messages with two parts (for example a sum plus an everyday sentence, or two different sums), with exact character positions for each part.

## 2026-10-26, Task 5: messages with two things tangled together

**What we did.** Added the generator for the situation you described: one message carrying two different things. A tangled message is two parts joined by a short linking phrase, in random order (about half start with the request, half with the everyday sentence). Each part is either a request ("54 times 2?") or an everyday sentence taken from real text, and we can ask for "request + everyday sentence" or "request + a different request". Two requests in one message always use two different skills (for example "whichever is greater, 99 or 82? Next: Each of 29 children gets 8 apples. How many apples are given out?" has a larger-of part and a multiplication part). Every part records its exact start and end position in the text, and the linking phrase belongs to neither part, so later we can score "did the system find the right piece and call it the right thing" by position, not by guessing. Six new tests check: the two parts really cut the text exactly (first starts at 0, last ends at the end, nothing but the link between them); each request's numbers and answer are right; the skill label survives the join (read back by an independent reader); both orders occur; two different skills per message (and a loud error for asking for the same skill twice, or for no request at all); and the linking phrases contain no digits, operation words or symbols (so a link can never give the answer away).

**What we learned.** A tangled message is easy to build, but the first draft sounded wrong ("Also, Each of 29 children..." has a capital letter after a comma), so the links now end in a colon or a full stop, where a capital is natural. We also decided that the request part is only the request itself, without the wrapper words ("Q:", "Quick one") that single messages have: the wrapper belongs to neither part, and leaving it out keeps the cut positions exact. Real everyday sentences are often news headlines, so "request + everyday sentence" looks like "Perle Asserts Hollinger's Conrad Black Misled Him Richard N. By the way: add 8 to 52?", which is a fair picture of mixed real text.

**Next.** Task 6: the first "can anything in a representation tell the cases apart?" tool, the information-ceiling test. We build the cheap, deliberately blunt representation (hashed word groups) as the control and a first sentence-embedding model.

## 2026-10-26, Task 6: turning sentences into numbers (the representations) and keeping the results

**What we did.** A "representation" is how a sentence becomes a list of numbers so that two sentences can be compared. We built eleven, from blunt to rich, so the experiments can tell us which kind of numbers can carry "what operation is this?" at all. (R0) A plain letter-pattern counter, the control: it sees spelling only and cannot understand meaning. (R1) A small sentence-meaning model (MiniLM): 384 numbers, plus the 128-number squeezed version that the GrowMoE pool uses today. (R2) A bigger sentence-meaning model (mpnet). (R3) BERT, averaged over the words. (R4) The inside of a language model: GPT-2 (layers 4, 6, 8 of 12) and Qwen2.5-1.5B (layers 9, 14, 18 of 28). For R4 we add the fixed line "The operation requested is" after the message and read the model's state at that last position, the point where the model has "just been asked" which operation it is. Everything loads from the local cache only; a missing model is a loud error that names it, and a test proves no network connection is attempted. Every result is saved to disk in blocks of 1,000 sentences, each file named by a fingerprint of the sentences and the model and written so that a power cut cannot leave half a file; a damaged or half-written file is detected and simply recomputed. 15 new tests (41 + 15 = 56 in the suite).

**What we learned.** (1) The test stand-in for the language model had to change: the small model named in the plan (distilgpt2) has no weights on this PC (only its settings), so the tests use pythia-70m, and the real GPT-2 has its own test. (2) A quick peek on 60 sentences (20 additions, 20 multiplications, 20 everyday; a smoke check, not a result): the sentence-meaning models (R1, R2) put arithmetic far from everyday text (similarity near 0.01 to 0.4) but put additions only slightly closer to additions than to multiplications (0.47 against 0.40). That is our whole problem in miniature: the operation word is a small part of the sentence, so "this is arithmetic" is easy and "this is a different operation" is subtle. (3) The raw language-model states are almost all pointing the same way: any two sentences have cosine similarity 0.94 to 0.99 on GPT-2 and 0.75 to 0.99 on Qwen, and one single direction holds 17 to 28 % of the squared length. This matches what is generally reported for such models. It means plain distance on R4 is nearly meaningless until the average is subtracted and the directions rescaled. Learned probes and Mahalanobis-style detectors do that themselves, so Task 8's test (which uses probes) is fair to R4, but any distance-based detector in Task 10 must be run on R4 after centring on the taught set, or it would be unfairly handicapped.

**Next.** Task 7: who "knows" what in each run (which skills are well known, thinly known or never seen), and a measured table of how well each taught skill can really answer, so that "well known" is a measured fact and not a label.

## 2026-10-27, Task 7: who knows what, and how well it really works

**What we did.** Built the part that decides what the system "has been taught" in each run, and then measures whether it really learned it. (1) Each run draws an assignment of the six skills to Solid, Thin or Unseen at random, so a skill's state is never tied to its identity (over 200 draws every skill is Solid, Thin and Unseen at least once; a multiplication run can force addition to be Solid, a division run can force subtraction). (2) Teaching examples come only from the three teaching wordings (symbols, number words, verbal phrasing); test examples come only from the other five (stories, swapped order, chatter, typos, register), and never reuse a problem that was set aside for testing, so no text and no number pair is shared. (3) A small "solver" network, one per skill, is trained on the taught examples and scored on problems it has never seen; that gives a competence table (examples taught, held-out accuracy, and the "always guess the most common answer" level as chance). (4) The status truth is read off those measured numbers by the draft rule from the design (Solid: at least 500 examples and at least 90 % correct; Thin: 100 examples or fewer, or between chance plus 10 points and 60 % correct; otherwise "grey", reported but kept out of the scores). The thresholds live in one place so they can be changed on dev and recorded at the freeze. 25 new tests (suite 81); I also added a small resumable runner for the sweep below (one file per skill, count and seed, safe to re-run after a power cut).

**What we learned.** The solver needed tuning before the table meant anything: the first version got 89 % on addition with 2,000 examples, just under the bar, and a small weight penalty (so it generalises instead of memorising) lifted it to 97 %. The tuning used one fixed recipe for every skill, chosen on dev seeds (17 settings tried), not skill by skill. The bar stayed at 90 % and the number range stayed at 1 to 2 digits, as planned. Here is the competence table (3 dev seeds; held-out accuracy, with the chance level in brackets):

| skill | 5 examples | 20 | 100 | 500 | 2,000 |
|---|---|---|---|---|---|
| add | 0.02 | 0.01 | 0.01 | 0.49 | **0.93 (solid)** |
| sub | 0.01 | 0.01 | 0.03 | 0.17 | 0.89 (borderline) |
| mul | 0.02 | 0.07 | 0.11 | 0.16 | 0.33 (never solid) |
| div | 0.40 | 0.38 | 0.42 | 0.53 | 0.77 (grey) |
| mod | 0.13 | 0.24 | 0.36 | 0.49 | 0.59 |
| max | 0.03 | 0.17 | 0.45 | 0.77 | **0.95 (solid)** |

Plain reading: even with 2,000 taught examples, only addition and larger-of are reliably Solid; subtraction sits right on the line (0.89, so it flips between Solid and grey by seed); multiplication, division and remainder are never reliably learned by this kind of solver. This is not a flaw to hide, it is how the design is meant to work: "taught 2,000 times" and "really knows it" are different things, and the status truth follows the measurement. It has two consequences for later tasks. First, a skill that is labelled Solid but measures at 33 % is, correctly, scored as Thin ("a known skill with uncertain competence"). Second, the division trace needs subtraction to be Solid, which it is only about half the time at 2,000 examples. In Task 12 I will either teach subtraction more examples for that run or report the runs where it was not Solid separately. One more practical point: the solver trains on the GPU now (you asked why it was on the CPU, and the GPU is faster and keeps the CPU free); each device gives exactly repeatable results for a given seed.

**Next.** Task 8 is the first big result and a stop point for you: for each representation, can ANY trained probe tell the cases apart (which operation, taught against untaught, an operation against its near-twin)? If even the best probe cannot, no detector built on that representation can. I will report that table in plain words and you decide whether the learned-map step (Task 9) is worth building.

## 2026-10-27, Task 8 (before the numbers): what I expect E1 to show

Written before any E1 result exists, so that we can see afterwards where I was wrong. E1 asks three things of each representation, with no detector involved: B1a, can a trained probe tell which of the six operations (or "none") a message asks for, when it was taught on three wordings and tested on five new ones? B1b, with no training at all, is a request closer to the same request reworded than to its near-twin (same wrapper, same numbers, only the operation changed)? B1c, with no training on untaught skills, are parts of taught skills closer to the taught examples than parts of untaught skills? The draft bars are 0.95, 0.95 and 0.90 (chance 0.14, 0.50, 0.50).

1. **R0, the letter-pattern counter.** B1b should fail badly, well below 0.5 (I guess 0.05 to 0.30): the near-twin shares the wrapper and the numbers, the reworded request shares neither, so to a letter-counter the twin is the closer one. B1a should be middling (0.35 to 0.60): "plus" or "+" is visible inside the three teaching wordings, but stories have no operation word and typos break the letters. This is what makes the test bed valid (B2): a word-matcher must fail.
2. **Sentence models and BERT (R1, R2, R3).** B1a 0.50 to 0.80 (strong on "arithmetic against everyday text", weak on "which operation"); B1b below 0.5 (0.15 to 0.45): same wording with another operator looks more alike than the same operator worded differently; B1c 0.55 to 0.75.
3. **Language-model states after the "operation requested is" line (R4).** The best chance, because the model has just been asked. B1a for Qwen at the middle layers 0.75 to 0.95, GPT-2 0.60 to 0.85. B1b not above 0.6 (a plain distance does not care about the question); B1c 0.55 to 0.75.
4. **Overall.** I expect none of R1 to R4 to pass all three draft bars, with B1b (pure distance, no training) failing for all of them. That would mean the learned map (Task 9) is needed. A pass of Qwen on B1a alone would not surprise me; a pass on B1b by any off-the-shelf model would, and I would then suspect the test bed before believing it.

## 2026-10-27, Task 8 (the numbers): can any off-the-shelf representation tell the operations apart?

**What we did.** Built the information-ceiling test and ran it on all eleven representations, five development seeds each (55 small result files, safe to re-run after a power cut). Three questions per representation, none of them involves a detector: B1a, can a probe trained on three wordings say which operation (or "none") a message asks for in five new wordings; B1b, with no training, is a request closer to the same request reworded than to its near-twin (same wrapper, same numbers, other operation); B1c, are parts of taught skills closer to the taught examples than parts of untaught skills. Table: `results/E1_TABLE.md`. Test bed check (B2): the letter-pattern counter R0 misses the B1a bar by 0.41 and the B1b bar by 0.95, far more than the required 0.15, so **the test bed is valid**: a word-counter cannot pass it, and a perfect fake representation does (that is in the tests).

**The result, in plain words.** No representation passes any of the three draft bars.

| representation | B1a (bar 0.95) | B1b (bar 0.95) | B1c (bar 0.90) |
|---|---|---|---|
| R0 letter counter (control) | 0.54 | 0.00 | 0.68 |
| R1 MiniLM (384) | 0.81 | 0.03 | 0.74 |
| R1 MiniLM squeezed to 128 (what the pool uses now) | 0.73 | 0.04 | 0.69 |
| R2 mpnet | 0.80 | 0.10 | 0.74 |
| R3 BERT | 0.62 | 0.00 | 0.62 |
| R4 GPT-2, layers 4 / 6 / 8 | 0.52 / 0.53 / 0.57 | 0.00 / 0.01 / 0.01 | 0.53 / 0.53 / 0.53 |
| R4 Qwen, layers 9 / 14 / 18 | 0.67 / 0.74 / 0.85 | 0.01 / 0.01 / 0.04 | 0.55 / 0.58 / 0.61 |

(Chance is 0.14 for B1a and 0.50 for B1b and B1c.)

1. **B1b is the clearest finding, and it is not "chance", it is the wrong way round.** A score of 0.00 does not mean "cannot tell"; it means that in practically every pair the near-twin ("same sentence, other operation") looks closer than the same request in other words. Every representation, including the language-model states after "The operation requested is", organises sentences by how they are worded, not by which operation they ask for. Subtracting the average first (centring) barely helps (best 0.10). This is exactly the "17 + 5 looks like 17 x 5" problem from the design, now measured.
2. **B1a shows the information is partly there.** A trained probe on the middle-late layer of Qwen reaches 0.85 against 0.14 chance, and it climbs with depth (0.67, 0.74, 0.85 at layers 9, 14, 18), so layers deeper than 18 may be better still (not tested). MiniLM reaches 0.81. GPT-2 is much weaker (0.52 to 0.57), hardly better than the letter counter.
3. **B1c is poor everywhere**: 0.53 to 0.74. The best (MiniLM, 0.74) is only a little better than counting letters (0.68), so for "is this part of a skill we were taught?" the sentence models are mostly seeing shared words.
4. **Where B1a breaks (a side check, dev seed 100 only, `diag_b1a_by_family.py`, output `results/E1_DIAG_B1A_BY_FAMILY.txt`).** By test wording: swapped operand order (S5) and register (S8) are ~1.00 for MiniLM and 0.98 for Qwen, chatter (S6) 0.96 and 0.83, typos (S7) 0.78 and 0.75, and **short stories (S4) only 0.45 for MiniLM and 0.64 for Qwen**. So the encoders carry the operation when it is said in a way close to what was taught, and fail mainly when the operation has to be worked out from a story. Subtraction and division are the weakest classes for Qwen (0.68, 0.71).

**How my prediction did.** Direction right, size wrong in places. Right: R0 fails B1b badly and B1a is middling (0.54, predicted 0.35 to 0.60); R1 to R3 B1c (0.62 to 0.74, predicted 0.55 to 0.75); Qwen layer 18 B1a (0.85, predicted 0.75 to 0.95); none of them passes all three bars. Wrong: I expected sentence models to reach B1b 0.15 to 0.45 and got 0.00 to 0.10, so the wording problem is worse than I guessed; I expected GPT-2 at 0.60 to 0.85 on B1a and got 0.52 to 0.57; Qwen layers 9 and 14 (0.67 and 0.74) are just below the 0.75 to 0.95 I gave; MiniLM (0.81) is just above the 0.50 to 0.80 I gave. Nothing here came out better than predicted on the question that matters (B1b), so no reason to suspect the test bed.

**What it means for the next step.** Plain distance on any of these representations cannot work as a skill detector: it would call the near-twin "known" and the reworded request "new", the opposite of the truth. The probe numbers say a learned map on top has something to work with (0.85 on Qwen). One honest caution for later: the weakest family is the story wording (S4), where the operation must be understood, not matched, and a map trained only on the three teaching wordings (symbols, number words, verbal phrasing) may stay weak there whatever we do. The 0.95 bar is currently an average over all five test wordings; I have not changed it, and whether it should be set per wording family is a decision for the dev review (Task 11) and the freeze (Task 14), not for now.

**Next.** STOP POINT 1: the decision on Task 9 (build the learned map) is yours.

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

## 2026-10-27, Task 9 (the numbers): the learned map nearly closes the gap, and stops short of two bars

**What we did.** Ran the map on the report seeds 100 to 104 for the three readers and added the rows to `results/E1_TABLE.md` (plus the plain Qwen layers 22 and 26). Side check of where the remaining misses are: `diag_r5_failures.py`, output `results/E1_DIAG_R5_FAILURES.txt` (one seed). Suite: 100 tests pass.

**Result (mean over 5 seeds; bars B1a 0.95, B1b 0.95, B1c 0.90).**

| reader | B1a which operation | B1b twin vs reworded (story partner) | B1c taught vs untaught |
|---|---|---|---|
| Qwen layer 26, plain | 0.93 | 0.64 (0.42) | 0.84 |
| **Qwen layer 26 + map** | **0.92 (fail)** | **0.997 (0.99) PASS** | **0.89 (fail, by 0.012)** |
| MiniLM + map | 0.79 | 0.949 (0.85) | 0.78 |
| mpnet + map | 0.80 | 0.93 (0.79) | 0.76 |

**What this means, in plain words.**
1. **The wording problem is solved for the best reader.** In every test where a message is compared with another by distance alone, the reworded request is now closer than its near-twin: B1b goes from 0.04 (Qwen layer 18, the original table) to 0.997, and the story partners, a wording the map never saw, score 0.99. The taught-and-untaught test (B1c) climbs from 0.61 to 0.89.
2. **The map does not add information, and B1a shows it.** B1a is a trained probe, so it measures how much "which operation" the reader holds at all; the map only re-sorts, so it cannot raise it (0.93 plain, 0.92 with the map). The 0.92 against a bar of 0.95 is therefore a limit of the reader, not of the map.
3. **Where the misses are (side check on one seed).** B1a mistakes are mostly division read as remainder (27 of the mistakes on that seed) and the "trap" sentences (arithmetic-looking but not a request) that get read as an operation (0.64 correct). Stories are now fine (0.86, they were 0.64 before the deeper layer). B1c is weakest exactly for division and remainder as the untaught skill (0.77 and 0.83): remainder messages sit nearest to division ones (66 to 77 %). In the design these two are also "Related" (remainder is built from division), so this is partly honest relatedness, not only a flaw; as a test of "taught vs untaught" it still counts as a miss.
4. **MiniLM and mpnet with the map are not enough**: B1a 0.79 to 0.80 and B1c 0.76 to 0.78. The small readers cannot carry the operation well enough, whatever the map does. The pool's current input (MiniLM) would need replacing by the larger reader.

**How my prediction did.** Every number landed inside the range I wrote down before looking (Qwen map: B1b 0.99 or more, B1a 0.90 to 0.95, B1c 0.86 to 0.92; the two small readers: B1b 0.93 to 0.96, B1a 0.78 to 0.86, B1c 0.72 to 0.80). That is expected, since the tuning seeds and the report seeds come from the same dev pool; it says the result is stable, not that it will hold on the confirmatory pool, whose templates differ. The caution I wrote beforehand stands: "B1b with a taught-wording partner = 1.00" is in-distribution; the evidence of real wording invariance is the story partner (0.99) and the story/typo/chatter cases in B1a.

**The rule says stop here.** The plan, step 5 of Task 9: "if R5 also fails B1, STOP and report the negative result". R5 on Qwen layer 26 passes one of the three bars and misses the other two narrowly (B1a by 0.03, B1c by 0.012). I have not changed any bar. What happens next is your decision (options below in the chat message and in the ledger).

## 2026-10-27, your decision after Task 9: option A (go on with the Qwen layer-26 reader plus the learned map)

**What we did.** You chose A. This goes past the plan's own rule ("if R5 also fails B1, STOP"), so it is written down as YOUR decision. No bar was moved: in the final report B1 will be stated as one pass and two narrow misses (B1a 0.92 against 0.95, B1c 0.89 against 0.90), and the sealed confirmatory pool decides whether the reader holds up on wordings it has never seen. I will switch to option B (one improvement round on division/remainder and trap sentences) if the Task 10/11 results show those reader mistakes are the main reason the status machine fails.

## 2026-10-27, Task 10: the status machine and the seven standard detectors

**What we did.** Built the two pieces that turn "a vector for a part of a message" into a status, and the standard methods we must beat or match:
- `status.py`, the status machine. It is handed only "skill groups" (the vectors of the examples a skill was taught, how many examples that was, how well its small solver does on problems it never saw) plus a few held-out examples used for calibration. It is never given the label of the part it judges. For a part it answers one of: **solid** (a known skill covers it, with enough examples and a good solver), **thin** (covered, but few examples or a weak solver), **unseen** (nothing taught covers it), **other** (everyday content that was taught as "not arithmetic"), or **conflict** (covered, but the known answer disagrees; only on request, not scored in Phase 1).
- `baselines.py`, the seven standard detectors from the literature (the expert pool's own gate, nearest-neighbour distance, Mahalanobis, Relative Mahalanobis, classifier confidence, energy, autoencoder), all with the same two calls, plus the group test (a whole batch of 1, 4, 8 or 16 parts judged together) and one shared calibration rule so every detector is held to the same false-alarm level.

**How it decides (in plain words).** For each group the machine measures how unlike the group's examples the part is, and turns that into a **conformal p-value**: "if this part really belonged to the group, how often would a part look at least this unusual?" A p-value of 0.05 or less flags the part as unlike the group. The calibration uses held-out examples from the taught wordings only (the first three families), so the promise "at most 5 in 100 normal parts are wrongly called new" is a real test of wording invariance later. A part is **unseen only if every group flags it**; otherwise it goes to the group whose nearest example is closest. I chose this over "test only the nearest group" after measuring it on a small toy world: with a wide group next to a narrow one, the nearest-group-only rule wrongly called 18 in 100 normal parts new, the every-group rule 5 in 100.

**What we learned.** Nothing about language yet: this task is tested on small made-up clouds of points (a toy world), where we know the truth exactly. In that toy world the machine flags about 5 in 100 normal parts as new (bar: at most 7), flags 90 in 100 or more of far-away parts, and gives "thin" for 20 taught examples and "solid" for 2,000 with everything else equal. Each of the seven baselines scores far points higher than normal ones, and each, after the shared calibration, flags about 5 in 100 normal held-out points. **Test quality check:** I planted 18 deliberate bugs in the code one at a time (flipping a sign, dropping a rule, and so on); the tests caught 15 straight away, and the three they missed pointed at three missing tests, which I added; all 18 are now caught. (An interruption left one planted bug in the file for a moment; I found it, restored the file and checked it is byte-identical to the backup before committing.)

**Declared simplifications.** Competence is one number per skill, not "competence near this part"; a "grey" skill (neither clearly solid nor clearly thin) is reported thin, because solid should mean demonstrated. Local evidence (how many taught examples lie near the part) is reported but does not override the count-and-competence rule; I will see in Task 11 whether it should.

**What comes next.** Task 11: cut messages into parts, run the machine and all seven baselines on the real dev data with the Qwen layer-26 reader plus the learned map, and write the first real status table. Then STOP POINT 2 for your go.

## 2026-10-27, Task 11 (before any number): what is built, the rules I fix now, and my prediction

**What we built (all tested; no real number yet).** `segment.py` cuts a message into parts (the answer key's own parts; sentences and clauses; sliding windows of 8 words). `evaluate.py` scores: the status table (macro F1 over solid / thin / unseen, the confusion counts, false "unseen" on new wordings, Other content wrongly called unseen), the same numbers for plain flag-or-not detectors, which skill a part is nearest to, the two-part messages (B4a, B4b, B6), the wording claim and the labelled cost estimate. `run_e2.py` runs it all: per seed one teaching assignment (two skills taught 2,000 examples, one taught 20, three never taught), a measured competence table, teaching, calibration, tests, near-twins, traps, everyday sentences, never-taught content and 120 two-part messages; for every reader the status machine at q = 0.01, 0.02, 0.05, 0.10 in the STANDARD condition (calibration from held-out S1-S3 parts only) and the ADAPTED one (a few S4-S8 parts in the calibration; reported next to it, never gated), the seven standard detectors at the same calibration, and group tests of 1, 4, 8 and 16 parts. Readers: the learned map on Qwen layer 26 (the one you chose), plain Qwen layer 26 (no map) and the hashed word-matcher (the control that must do badly).

**Rules fixed now, before looking (so they cannot be bent by the results).**
1. **Bars.** The draft bars of the spec stay the reading of PASS / fail. For the freeze, each bar is re-set from the dev seeds by one rule: a bar to stay ABOVE (macro F1, AUROC, the B4 rates) becomes max(draft bar, dev mean - 2 sd over seeds); a bar to stay BELOW (false-unseen rates) becomes min(draft bar, dev mean + 2 sd). So a dev result can tighten a draft bar and can never make it easier. The dev values come from the learned-map reader, status machine, q = 0.05, standard condition.
2. **Solid / Thin / grey thresholds (`THRESHOLDS`) stay at the draft values** (Solid = at least 500 examples and accuracy 0.90; Thin = at most 100 examples, or accuracy between chance + 0.10 and 0.60). They are defined on the measured competence table of Task 7, not on the status results, so there is nothing to tune them against. If the competence table shows that a Solid skill (2,000 examples) lands in the grey zone in more than a quarter of the seeds, I report it at the stop and propose a change; I do not change it silently.
3. **The headline level is q = 0.05.** The curve over 0.01, 0.02, 0.05, 0.10 is shown; no level is picked afterwards.
4. **The standard detectors are tuned on other seeds (150-154), one setting per detector,** by the highest mean AUROC (taught against untaught parts), before the report seeds 100-129 are run; the chosen settings are written next to the results. The status machine has no tuned setting (q is fixed; the kernel width is chosen by its own leave-one-out likelihood).
5. **Calibration is held-out data of the taught groups in S1 to S3 only** (120 parts per group, none of them a teaching text); a Thin skill still gets 120 calibration parts, which is more than its 20 teaching examples: a declared simplification, because a conformal test needs at least 19 (q = 0.05) or 99 (q = 0.01) calibration parts.
6. **Seeds.** The map was tuned on seeds 105-107 (Task 9) and these lie inside the report seeds 100-129; the map's settings were insensitive (12 variants within 0.01), but I note the overlap here.

**Prediction (written before the dev run; learned-map reader, status machine, q = 0.05, standard condition, mean over seeds).**
- False "unseen" on the new wordings S4-S8: **0.08 to 0.25** (bar 0.05, so I expect it to fail); on the taught wordings S1-S3: 0.03 to 0.08. If this is right, the wording-invariance claim becomes "needs adaptation to new wording" or "fails".
- Untaught skills found as unseen: **0.35 to 0.65** (the taught-against-untaught AUROC of the reader was 0.89, and division and remainder look like each other).
- Macro F1 over solid / thin / unseen: **0.60 to 0.80** (bar 0.85, I expect it to fail).
- Thin separated from Solid (AUROC): **0.95 or more** (the number of taught examples differs by a factor of 100; this is almost by construction).
- Other content called unseen: **0.04 to 0.12**.
- Adapted condition: false unseen falls toward **0.05 to 0.10**.
- Plain Qwen layer 26 (no map): clearly worse than the learned map on false unseen (0.20 to 0.45). The hashed word-matcher: worst, false unseen above 0.30.
- Standard detectors: none beats the status machine by more than 0.05 on both false unseen and unseen found at the same calibration; kNN and Mahalanobis are the strongest. The pool's own gate (vMF) is the weakest on the Thin skill.
- Two-part messages (gold parts): B4a leftover is the unseen part **0.45 to 0.75**, known part not flagged **0.80 to 0.93**; B4b nothing flagged **0.75 to 0.90**, Other labelled **0.85 to 0.95**. Automatic cutting: within 0.10 of the gold number for sentence/clause pieces, not for sliding windows.
- Group tests: unseen skills found rises with the batch size (0.85 or more at 16); false unseen for batches of taught skills stays at or below 0.15.

## 2026-10-27, your decision: a from-scratch reader as one more arm (Task 11b, after Stop Point 2)

**What we decided.** After you have seen the Task 11 results, I build a reader trained from scratch (no pretrained language model) and run it through the same tests as the others. The rule, written before there is any result:
- **If it helps or comes close** to the pretrained reader with the learned map (within 0.05 on the four numbers: near-twin test B1b, taught-against-untaught B1c, overall status score, and false "unseen" on new wordings), it is worth keeping on its merits.
- **If it is a bit behind** (within 0.15), it is not thrown away. It is parked, with its code and numbers, until you decide whether the whole system will be built from scratch. If yes, a reader that is a bit behind is worth it, because one custom architecture whose parts are designed to work together is the aim. If no, the pretrained reader stays.
- **If it is much further behind** than 0.15, it is not worth building further for now.

**Two versions, because a fair test needs both.** (a) *Strict*: trained only on the teaching data of this project (three wording families), the hardest case. (b) *With pre-training*: first learns general language from the everyday-text collections already on this PC (banking, news, chat), never from the new wordings S4-S8 or the test templates, then the same contrastive training. A from-scratch reader that has only seen three wordings cannot be expected to understand a story or a typo, so (b) is closer to what a real "own architecture" would do.

**What it does not test.** This is the reader (the "eyes") only. Decoders, which an expert needs to produce its answers, are a separate later piece.

## 2026-10-27, Task 11 (the numbers): the machine finds new skills well; the false alarms on new wordings come almost entirely from two wordings

**What we did.** Ran E2 (single parts) and E3 (two-part messages) on 30 dev seeds with three readers: 90 pieces, all finished. Results: `results/E2_TABLE.md` (full tables), `results/E2_DIAG.txt` (where the misses come from). Suite: 206 tests pass.

**Headline: learned-map reader, status machine, q = 0.05, calibration on S1 to S3 only (mean over 30 seeds, [lowest - highest]).**

| measure | bar | result | what I predicted |
|---|---|---|---|
| Overall status score (macro F1 over solid / thin / unseen) | at least 0.85 | **0.855** [0.52 - 0.92] PASS, by 0.005 | 0.60 - 0.80 (better than predicted) |
| Known skills wrongly called "unseen" on new wordings S4-S8 | at most 0.05 | **0.289** [0.17 - 0.43] **fail** | 0.08 - 0.25 (worse) |
| the same on the taught wordings S1-S3 | - | 0.05 | 0.03 - 0.08 (right) |
| Thin separated from Solid (AUROC) | at least 0.85 | **0.844** fail, by 0.006 | 0.95 or more (wrong) |
| Everyday content called unseen | at most 0.05 | **0.249** fail (but see point 4) | 0.04 - 0.12 |
| Never-taught skills found as unseen | - | 0.89 | 0.35 - 0.65 (better) |
| Adapted condition, known skills called unseen | - | 0.17 | 0.05 - 0.10 (worse) |
| Two-part: leftover is the unseen part (B4a) | at least 0.85 | 0.66 fail | 0.45 - 0.75 (right) |
| Two-part: known part not flagged (B4a) | at least 0.95 | 0.67 fail | 0.80 - 0.93 (worse) |
| Two-part: nothing flagged, request plus everyday text (B4b) | at least 0.95 | 0.64 fail | 0.75 - 0.90 (worse) |
| Two-part: everyday part labelled "other" (B4b) | at least 0.95 | 0.97 PASS | 0.85 - 0.95 (right) |
| Automatic cutting within 0.10 of the gold number (B6) | - | 13 % of seeds (sentences), 7 % (sliding windows) | "sentences yes, windows no" (sentences also no) |

The wording claim from the calibration rule is **"fails"**: even with a few new-wording examples in the calibration (adapted) the rate is 0.17, not below 0.05. Plain Qwen without the learned map: 0.39 (the map helps by 10 points). The word-matcher control: 0.12 but it finds only 0.36 of the new skills, which is the failure it is supposed to show. All dev-fixed bars equal the draft bars: no dev result beat a draft bar by enough to tighten it.

**What this means, in plain words.**
1. **The false alarms are concentrated in two wordings.** Known skills are wrongly called new in **67 % of short stories (S4)** and **47 % of typo / case / spacing noise (S7)**; swapped operand order (S5) 19 %; chatter (S6) 4 % and register (S8) 7 % are about as good as the taught wordings (2 to 9 %). Stories and noisy text sit systematically farther from the taught examples (best cosine 0.95 and 0.89 against 0.99; median p-value 0.025 and 0.07 against about 0.5). The group test shows the same: a batch of 16 known-skill parts in new wordings is called new in 78 % of cases, a whole-group shift, not noise. Caveat: with gold spans a part is the request alone, so S6 and S8 (which differ from the taught wordings mostly in the sentence around the request) are not really tested by this unit.
2. **New skills are found well:** 80 to 95 % per skill (division 80 %, remainder 87 %, the rest 90 to 95 %). The misses are taken by the **remainder or division group**: they absorb each other (97 to 99 %), and also most of the rest.
3. **Against the standard methods the machine is level, not better.** At the same alarm level, kNN, Mahalanobis and the pool's own gate give false alarms 0.31 to 0.33 and find 0.95 to 0.97; the machine gives 0.29 and 0.89: the same trade-off. Classifier confidence and energy raise fewer false alarms but miss half the new skills (found 0.41 to 0.51); Relative Mahalanobis fails (AUROC 0.44). The value of the machine is the statuses (thin, other, calibrated per group), not detection power.
4. **The "everyday content" number is misleading.** Ordinary everyday sentences are called new only 3.3 % of the time (fine). The 25 % is dragged up by the hand-written chatter sentences of the S6 template (52 %), which are not the same kind of text as the taught everyday content, and by traps with test-only templates (47 %); never-taught content (science questions) is found only 62 % of the time. This is a **defect of my test bed** (the spec says the chatter should be drawn from the same corpora; the dev pool uses a hand-written list). I found it after seeing the numbers, so I record it as found afterwards.
5. **The Thin-vs-Solid failure is the wording problem again:** the local evidence of a Solid part in a story or noisy wording is about zero (median 0 and 1, against 33 to 76 in taught wordings).
6. **My own declared rule fired:** 17 of 60 "Solid" skills (28 %, over the quarter I set) are not solid by measured accuracy. With 2,000 examples the small solver learns addition (0.94), subtraction (0.91) and maximum (0.95) but not multiplication (0.33), division (0.72) or remainder (0.58). I did not change any threshold; the choice is yours (see below).
7. **Offline cost estimate** (not gated, an estimate): 269 pool points per seed for the learned-map reader, 305 for plain Qwen, 994 for the word-matcher.

**What I propose, not yet done (your call at Stop Point 2).**
- **A. One bounded round on the wording problem before Task 12**, with its prediction written first: (i) calibrate on *leave-one-wording-out* data (each taught wording is tested against prototypes of the other taught wordings, which uses S1-S3 only and teaches the alarm what a wording shift looks like); (ii) reading the text after generic clean-up (case and spacing). Training with S7-style noise would break the rule that S4-S8 are never used for teaching, so I would not do that without your say.
- **B. Fix the test bed** (teach the everyday content from the same source as the chatter) before the freeze, and say so.
- **C. The small solver:** improve it for multiplication, division and remainder, or accept that only addition, subtraction and maximum can be Solid.
- Then Task 12 (worked traces), and the from-scratch reader (Task 11b) as you decided.

## 2026-10-28, your go: investigate WHY the new wordings set off the alarm, then try fixes (option A first)

**What we decided to do.** (1) Find out why stories (S4) and noisy text (S7) sit farther from the taught examples, before choosing a fix. (2) Then try mitigations, option A first. To keep the numbers honest: all investigations and the choice between mitigations run on the TUNING seeds 150-154; the chosen design is then run ONCE on the report seeds 100-129. Anything that teaches the reader a held-out wording (for example training with S7-style noise) stays out unless you say so.

**Five questions, and my guess for each (written before measuring).**
1. **Is it the learned map or the reader?** Guess: the reader. Plain Qwen shows the same pattern (worst on stories and noise), so the map is not the cause, though it does not remove it.
2. **Which kind of noise hurts?** I test case flips, typos and extra spaces separately on taught-wording parts. Guess: case flips hurt most (they change how the text is cut into tokens), then typos, extra spaces least.
3. **Is the stories shift a few directions or spread everywhere?** Guess: a few. Stories add content (objects, names) that the map never saw in the taught wordings, so it never learned to ignore those directions. Test (a diagnostic, not a fix, because it peeks at the test wording): remove the top 1, 2, 4 shift directions from story parts and see how much of the false alarm disappears. Guess: removing 4 directions removes more than half of it.
4. **Does a different score separate "known skill in a new wording" from "a skill I was never taught"?** The current alarm asks "is this close to what I was taught?". The alternative asks "is one skill clearly nearer than the others?" (the margin). Guess: the margin separates much better on stories and noise (AUROC 0.85 or more) than closeness does (0.6 to 0.75), because a story is far from the examples but still clearly nearest to its own skill (94 % of the time), while a never-taught skill is not clearly nearest to anything.
5. **Is it just text length?** Guess: partly. Stories are long, and false alarms grow with the number of words even within the taught wordings.

**Option A as I will build it (not yet built).** (i) Calibrate on leave-one-wording-out scores: each taught wording (S1, S2, S3) is scored against prototypes of the other two, so the alarm learns what a wording shift looks like. (ii) Read the text after generic clean-up (case and spacing). Prediction for A on the report seeds, written now: false alarms on new wordings fall from 0.29 to **0.20 - 0.27** (a small gain: S1-S3 differ little from each other compared with stories and noise), unseen skills found fall from 0.89 to **0.80 - 0.88**; the clean-up helps the noise wording most (S7 from 0.47 to **0.25 - 0.40**) and stories not at all. I do not expect A alone to reach the 0.05 bar.

## 2026-10-28, why the new wordings set off the alarm: what the measurements say (tuning seeds 150-154)

Files: `results/E2_WHY.txt` (from `diag_shift.py`). Each line: my guess, then what was found.

1. **Map or reader?** Guess: the reader. **Found: the reader.** Plain Qwen vectors already sit 5.6 standard deviations farther from the taught examples on stories and 3.3 on noisy text. The learned map helps on stories (97 % of known stories flagged without it, 67 % with it) but cannot remove the shift; in standard deviations it makes the noise shift look bigger (it packs the taught wordings tighter).
2. **Which noise?** Guess: case, then typos, then spaces. **Found, as guessed:** case flips 62 % false alarm (and the nearest skill is right only 67 % of the time, so even the sorting breaks), typos 31 % (81 % right), extra spaces 10 % (97 % right).
3. **A few directions?** Guess: a few. **Found: no.** Removing the top shift directions of the story wording lowers the false alarm only from 62 % to 54 %, and removing random directions of the same number makes it worse (69 to 80 %). The displacement is scattered (each story moves in its own direction), so a projection fix will not work.
4. **A better score?** Guess: the margin (nearest minus second nearest skill) separates well. **Found: it does not.** Known skill against never-taught skill in the same wording: stories closeness 0.79, density 0.82, margin 0.83; noisy text 0.74, 0.77, 0.69. On the taught wordings the margin is clearly worse (0.78 to 0.85 against 0.97 to 0.99). A first look at one seed had suggested 0.90; five seeds say no.
5. **Length?** Guess: partly. **Found: partly.** In new wordings the false alarm grows with length (12 % / 14 % / 64 % for short / middle / long), but in the taught wordings it does not (5 % / 3 % / 0 %). So length alone is not the cause; the story content is.

**What this tells us.** The reader gives stories and noisy text a systematically different reading; the alarm, calibrated only on three short template wordings, is right to find them unusual. The sorting is still mostly right (stories 89 %, typos 81 %, case flips 67 %).

## 2026-10-28, mitigations lab: plan and predictions, written before running it

**Mitigation arms (new module `wording.py`, so the earlier results stay valid).**
- Reader: (a) the plain reader; (b) *clean-up* (lower case, one space between words); (c) a closing prompt that says to ignore extra words (P1: "Ignoring any story or extra words, the arithmetic operation asked for is"); (d) another prompt (P2: "In one word, the arithmetic operation that this asks for is"); and clean-up combined with each prompt. The prompts do not name any skill (that would give the reader the label set).
- Machine: (a) as it is; (b) *leave-one-wording-out* calibration (option A); (c) a *margin gate* at 0.10, 0.25, 0.50 (a part every group flags is called unseen only if it is also ambiguous between skills).

**Procedure, fixed now.** Stage 1: the six reader variants with the machine as it is. Stage 2: the machine variants on the stage-1 winner. Everything on the tuning seeds 150-154. The rule for choosing a winner: the lowest false alarm on S4-S8 among variants that still find at least 0.80 of the never-taught skills and keep macro F1 at 0.80 or more; within 0.01 the simpler variant wins (plain, clean-up, prompt, both). The winner is then run ONCE on the report seeds 100-129 and compared with the numbers already on disk; no bar changes.

**Predictions (tuning seeds, false alarm on S4-S8 now about 0.28).**
- Clean-up: S7 from about 45 % to 25-38 %, stories unchanged; overall 0.22 to 0.27.
- Prompts: stories from 67 % to somewhere between 45 % and 65 %; I give it even odds that a prompt gives a material gain (more than 0.05 overall). Found stays 0.85 to 0.92.
- Leave-one-wording-out: a small gain only (0.00 to 0.03), because the taught wordings differ far less from each other than stories and noise differ from them.
- Margin gate 0.25: the false alarm drops by 0.05 to 0.10 but the found rate drops by 0.10 to 0.20 (the margin separates weakly), so the 0.50 version probably breaks the 0.80 floor.
- Best design overall: false alarm 0.15 to 0.25. None reaches 0.05.

## 2026-10-28, the mitigations lab (tuning seeds 150-154): no cheap fix works; the winner by the declared rule is "no change"

Files: `results/WORDING_LAB_STAGE1.md`, `results/WORDING_LAB_STAGE2.md`. False alarm = known skills in new wordings S4-S8 called unseen (now 0.256 on these seeds; bar 0.05).

| what was tried | false alarm | never-taught skills found | macro F1 | verdict |
|---|---|---|---|---|
| nothing (the machine as it is) | 0.256 | 0.903 | 0.866 | the reference |
| clean-up of case and spacing before reading | 0.250 | 0.890 | 0.861 | noisy text 45 % to 38 %, stories unchanged; gain too small to count |
| closing prompt P1 ("ignoring any story...") | 0.333 | 0.869 | 0.821 | **worse**; stories 62 % to 93 % |
| closing prompt P2 ("in one word...") | 0.309 | 0.877 | 0.828 | **worse**; stories 62 % to 77 % |
| clean-up with a prompt | 0.251 / 0.261 | 0.863 / 0.872 | 0.843 / 0.838 | noisy text down to 30-32 %, stories worse |
| leave-one-wording-out calibration (option A) | 0.252 | 0.891 | 0.858 | gain 0.004: nothing |
| margin gate 0.10 | 0.063 | **0.250** | 0.511 | reaches the bar only by finding almost no new skill |
| margin gate 0.25 | 0.068 | **0.292** | 0.542 | same |
| margin gate 0.50 | 0.114 | 0.722 | 0.785 | below both floors |

**The declared rule picks "as is"**: no variant that keeps finding at least 0.80 of the new skills and a macro F1 of at least 0.80 lowers the false alarm by more than the 0.01 tie margin. So the planned single run on the report seeds would repeat numbers we already have; it was **not run** (it would have cost about 45 minutes of GPU for nothing).

**How the predictions did.** Right: clean-up (noisy text 45 to 38 %, stories unchanged), leave-one-wording-out (0.00 to 0.03 gain), "none reaches 0.05". Wrong: the prompts (I guessed even odds of a gain; they hurt), and the size of the margin gate's cost (I predicted found would fall by 0.10-0.20; it fell by 0.61).

**What the margin gate teaches.** Most never-taught skills sit CLEARLY nearest to some taught skill (a large margin), just as known stories do. So neither closeness nor margin can separate "known skill in an unfamiliar wording" from "skill never taught" for these two wordings with this reader. The reader is the limit, not the alarm.

## 2026-10-28, reading: Focus On Facts (FOF), what it adds

Read: the poster abstract and the model card (the full paper is not reachable; `RELATED_WORK.md` updated). Their problem is ours: an embedding prefers a near-copy with one changed fact to a faithful rewrite (99.58 % of 8,327 held-out triplets). Their cure is full fine-tuning on 83,263 triplets of LLM-written rewrites with hard negatives; the wrong preference falls only to 29.37 % (mean separation +0.033, tiny), and the model gets worse at other tasks (SNLI 71 % to 53 %). Their listed failure modes match ours: a typo is read as a changed entity ("orthographic brittleness"), cosine values are poorly calibrated (so only the ordering is trusted), and the training text is normalised to stop shortcuts through formatting. What it means for us: (1) the field's best known cure for this problem is wording DIVERSITY in training (many LLM-written rewrites), which our rule "teaching uses S1-S3 only" excludes on purpose; (2) even that cure does not reach a low error, so we should not expect miracles from any encoder trained on a few wordings (this includes the from-scratch encoder); (3) specialising an encoder costs it other abilities, which matters for a one-size-fits-all custom architecture.

## 2026-10-28, your decision: option 2, the function check first (then wording diversity). Plan and predictions, written before running

**The idea in plain words.** How a message reads changes with its wording; the numbers in it and its answer do not. When an example arrives WITH its answer (during teaching, or when a worked example is shown), we can ask: does a skill the system already knows turn these two numbers into this answer? "I had 17 apples and got 5 more" with answer 22: the addition skill says 17 + 5 = 22, so this is addition in a story wording, not a new skill. "17 times 5" with answer 85: no known skill gives 85, so it really is something new. This does not look at how the message reads, so stories and noisy text cannot fool it.

**What was built (all tested, nothing run yet).** `function_check.py`: a number reader that takes the two numbers out of a message and never reads the operator word or symbol (digits when there are digits, number words otherwise; "one" in "in one bag" is not a number), and a verifier that tries the numbers in both orders (some wordings name them in the other order: "5 subtracted from 17"). Task 12 (worked traces) needs the same two parts, so they are shared code. `lab_function.py`: runs it on the tuning seeds 150-154. Planted-bug check: 11 deliberate bugs, all caught except one that matters only on a very rare text collision (and that case now fails loudly).

**Rules of the check, fixed now.** (1) Only a COMPETENT skill may explain: Solid state and a solver that scores at least 0.90 on problems it never saw. A skill taught 20 examples, or with a weak solver (multiplication, division, remainder), cannot confirm anything, so its parts keep the look-based call. (2) The check only ever removes "unseen" calls; it never adds one. (3) It needs the answer: it is a check for teaching and worked examples, not for every message.

**Rule for going on, fixed now.** If on the tuning seeds the false alarm on S4-S8 falls by at least 0.05 (from about 0.256 to 0.206 or lower) while the share of never-taught skills found falls by no more than 0.05 and the explaining skill is the right one in at least 95 % of the overturned known parts, I run it ONCE on the report seeds 100-129 and report it. Otherwise it is recorded as a negative result and we move on.

**Prediction (tuning seeds).**
- The number reader finds exactly two numbers in at least 98 % of the requests of every wording.
- Only addition, subtraction and maximum can be competent, and only when they are in the Solid state: about 35 to 45 % of the known test parts belong to such skills. That is the most the check can rescue.
- For the parts of competent skills the false alarm falls from about 0.30 to 0.05-0.12; for thin and weak skills it does not move (about 0.30).
- Overall false alarm: 0.256 to 0.15-0.20. Never-taught skills found: 0.90 to 0.86-0.90 (a few coincidences, where some taught skill happens to give the same answer). When a known part is overturned, the skill is the right one in at least 97 % of cases.
- It does NOT reach the 0.05 bar overall, because the skills that cannot be checked make up more than half of the parts.

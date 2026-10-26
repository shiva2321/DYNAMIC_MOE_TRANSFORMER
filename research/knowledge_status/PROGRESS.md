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

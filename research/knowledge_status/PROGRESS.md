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

# M2 reading job (R-1): read six papers in full, extract the protocol facts, audit our baselines against them. Task spec for Gemini (one job)

**STOP RULES (also the first line of the kickoff message).** The job ends when `docs/superpowers/handoff/M2_READING_NOTES.md` exists and you have sent the closing message of section 5. Then STOP. This job creates exactly ONE file. No edit of any other file, no `git add/commit/push/stash/reset`, no Python, no tests, no pip, no downloads into the repository, no second job. A job starts only on the user's explicit go; a go covers ONE job.

## 0. Why

Milestone M2 (`docs/roadmap/MILESTONES.md`, section M2) puts the pool on a standard five-dataset text benchmark with frozen encoders and compares it with the strong simple baselines of the field. The road plan (`docs/roadmap/RESEARCH_DIGEST.md`, section 7) says the papers behind that must be read IN FULL before the milestone starts, because today they are tagged L (abstract or search summary level) or R (recalled from memory). The preregistration of M2 (written by Claude, not by you) will fix bars and a protocol from what you extract, so a wrong number here becomes a wrong bar. Second purpose: our baselines A5a, A5b and A6 are named after published methods; if ours are weaker than the published ones, the pool looks better than it is. That must be found now and stated as a limit.

## 1. Papers (find each by title and open the full text; the identifiers are hints from memory and must be confirmed by what you open)

1. d'Autume et al., "Episodic Memory in Lifelong Language Learning" (MbPA++), 2019, arXiv 1906.01076.
2. Sun et al., "LAMOL: Language Modeling for Lifelong Language Learning", 2020, arXiv 1909.03329.
3. Zhou et al., "Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need" (SimpleCIL, ADAM), arXiv 2303.07338.
4. McDonnell et al., "RanPAC: Random Projections and Pre-trained Models for Continual Learning", NeurIPS 2023, arXiv 2307.02251.
5. Goswami et al., "FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning", NeurIPS 2023, arXiv 2309.14062.
6. Hayes and Kanan, "Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis" (SLDA), 2020, arXiv 1909.01520.

## 2. What to extract (one section per paper in the notes file; every fact has a locator: section, table or figure number, page)

For papers 1 and 2 (the benchmark protocol), exactly and in implementable form:
- The five datasets, their original source, how the 115,000 training and 7,600 test examples per dataset are drawn (random, seeded? stratified?), the number of classes of each, whether Yelp and Amazon share one label space (total number of classes of the single head), what "no task id at test time" means in each paper.
- The four task orders, as sequences.
- The metric (average accuracy over what? macro or micro over datasets?), and every reported number for every method in the main table, each with its order, so that the table can be rebuilt.
- What the multitask or joint upper bound is and its number.
- Anything that makes the protocol differ between the two papers (sample sizes, label merging, text length truncation).
For papers 3 to 6 (the methods): the method as equations or steps precise enough to implement (what is stored, what is updated per batch, what is solved when), every hyperparameter and how it was chosen (and whether on held-out data or on the test set), their benchmarks and the numbers that matter for a frozen-feature baseline, and anything they say about text or sentence encoders (say "not covered" if so). Also: has anyone reported a closed-form or prototype head on frozen encoder features for this five-dataset text sequence? If you find such a result, give it with its source; if you do not, write "NOT FOUND after searching" and name what you searched.

## 3. Fidelity audit of our baselines (read-only reading of our code; do NOT run it)
Read `growmoe/memory_baselines.py` (`ClassMean`, `StreamingLDA`, `KnnMemory`) and `m1/baselines_extra.py` (`MahalanobisProto` = A5b, FeCAM style; `RandomFeatureRidge` = A6, RanPAC style) and write a table: one row per difference between our implementation and the published method, columns: our choice, the paper's choice with locator, which way it probably moves the baseline's accuracy (helps, hurts, unknown), and whether it must be named as a limit when M1 is reported. Include at least: normalisation of inputs, covariance shrinkage and its value, any feature transformation (for example Tukey's), how the random projection is built and its width, how the ridge strength is chosen, whether features are centred, whether a class-prototype stage precedes the random projection, how classes seen later are handled. If our code does the same as the paper, say "same" with the locator. Do not say "probably the same": say what you checked.

## 4. Rules that make the notes usable
- Read the full text of each paper (arXiv PDF or the publisher's page). For each paper write one line "READ IN FULL: yes/no, version read (arXiv version or date), pages". If you could not open the full text, write NOT READ IN FULL and give only what the abstract page shows. **Never fill a number from memory or from a blog post.** A number you could not find is "NOT FOUND", never an estimate. A blog or secondary source is marked "secondary".
- Paraphrase. No quotation longer than 15 words, at most one short quotation per paper. Do not copy tables whole: rebuild them with your own layout from the numbers.
- Also check what `docs/roadmap/RESEARCH_DIGEST.md` section 4 and the M2 and M1 passages of `docs/roadmap/MILESTONES.md` claim about these papers (five datasets, 115,000 and 7,600, four orders, merged Yelp and Amazon labels, LAMOL "2 to 3 % below multitask") and mark each claim CONFIRMED, CORRECTED (give the right fact) or NOT FOUND.
- End the notes with "Open questions the papers do not answer" (what M2's preregistration will have to decide itself).

## 5. Closing message and fence
Create only `docs/superpowers/handoff/M2_READING_NOTES.md`. Your closing message to the user lists, per paper: read in full yes/no, how many facts are NOT FOUND, and the three most important findings for the M2 protocol and for the fidelity of our baselines. Claude spot-checks ten numbers against the papers before anything is used. This job needs no CPU on this machine beyond your own tools: the timed pass of the scored run is running and must not be disturbed.

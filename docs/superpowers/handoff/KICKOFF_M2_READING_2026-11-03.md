STOP RULES: this job ends when `docs/superpowers/handoff/M2_READING_NOTES.md` exists and you have sent your closing message. Then STOP. You create exactly ONE file. No edit of any other file, no git commands, no Python, no tests, no installs, no second job.

You are Agent G (Gemini) on the GrowMoE lab in D:\spherical_hyperspace_memory (the user passes this message to you; Claude reviews).

Task: a reading job (R-1) for the next milestone, M2 (a standard five-dataset text benchmark with frozen encoders). Read `docs/superpowers/handoff/M2_READING_SPEC.md` completely first; it lists the six papers, what to extract from each, and a read-only audit of our baseline code against the published methods.

Three things that are easy to get wrong:
1. Never fill a number from memory or from a blog post. A number you could not find in the full text is "NOT FOUND". For every paper write whether you read it IN FULL, and which version. Claude will spot-check ten numbers against the papers; one invented number makes the whole file unusable.
2. Every fact has a locator (section, table or figure, page). Paraphrase; no quotation over 15 words.
3. A long CPU run is in progress on this machine. This job needs no CPU: do not run Python, tests or any script. Read, search and write the one notes file only.

When the notes file is written, tell the user in one message: per paper, read in full yes/no and how many facts are NOT FOUND, plus the three most important findings for the M2 protocol and for the fidelity of our baselines. Do not start anything else.

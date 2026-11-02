# Nuance detection: what the lab already knew, and one new test (2026-11-01)

You asked me to build a nuance detector that works for anything, using our knowledge and the research. Before building I read the nuance lab's own log (`research/knowledge_status/PROGRESS.md`), as our own rule says to do. That changed the plan, so this report starts with what I got wrong.

## What I got wrong in chat

1. **Two of my five ideas had already been tested and had failed.** Teaching the reader many wordings so it learns to ignore rewording: false alarm 0.262 and 0.232 against 0.256 for doing nothing, and 14 to 16 points of new skills lost. Even a contaminated version, where the writer was told the held-out wordings, only halved it (0.144, bar 0.05). Asking once per group: 0.228 to 0.241 with real experts, no gain.
2. **I said stories "bury the request in long text". That is not how this benchmark works.** In it a "story" is the whole word problem ("Each of 4 children gets 2 books. How many books are given out?"), and the operation is implicit in it. There is no filler to cut away, so chunking does not apply to that failure.
3. **I said earlier rounds "could not close it without labels".** More exactly: a reader that asks the language model a multiple-choice question over the taught skills does close it for arithmetic wording shifts (0.025 false-new against a bar of 0.05; 0.974 of new skills found). It fails to beat plain embeddings on topical novelty, and it depends on the model's own vocabulary.

## What the lab found (the short version)

- **A double dissociation.** When the new thing differs in topic (a new banking intent, a new news topic), the sentence embedding already separates it: look-based detectors find 0.86 of never-taught banking intents, the asking reader 0.50. When only the wording or function changes (arithmetic in stories, noisy text), the embedding follows the wording and only a reader that understands holds: 0.10 against 0.38 to 0.50 known messages wrongly flagged.
- **No fixed combination of the two works in every domain**, and a rewrite-based way of deciding which reader to trust failed (correlation 0.003).
- **Projecting out the wording directions, margins, clean-up, prompts, leave-one-wording-out, a function check on single answers** were all tried; none brings the false alarm near 0.05 without losing the new skills.
- **Predictive-coding-style surprise** was not run as such; the nearest thing, an autoencoder's reconstruction error in latent space, ties the other detectors (0.28 false alarm, 0.88 found).

## The one new test: a character-aware front end (E-064)

Same worlds, same detectors, same calibration as the lab; four readers: a sentence-meaning model (MiniLM), a spelling-only counter (hashed word and letter groups), and two mixes of them. Tuning seeds 150 to 154. Predictions were written first; **2 of 5 held**.

| reader | false alarm on new wordings (vMF / 1-NN / Mahalanobis) | stories | noisy text | new skills found |
|---|---|---|---|---|
| MiniLM | 0.342 / 0.329 / 0.223 | 0.98 / 0.99 / 0.80 | 0.52 / 0.56 / 0.24 | 0.56 / 0.26 / 0.13 |
| spelling only | 0.121 / 0.221 / 0.221 | 0.10 / 0.59 / 0.60 | 0.26 / 0.45 / 0.40 | 0.36 / 0.31 / 0.26 |
| MiniLM + 0.5 spelling | 0.297 / 0.327 / 0.269 | 0.83 / 0.99 / 0.89 | 0.44 / 0.55 / 0.35 | 0.56 / 0.28 / 0.15 |
| MiniLM + 1.0 spelling | 0.332 / 0.317 / 0.274 | 0.91 / 0.98 / 0.89 | 0.49 / 0.48 / 0.37 | 0.68 / 0.33 / 0.15 |

- **A character view does not fix it.** No mix lowers the false alarm by the lab's own go-on margin (5 points) while keeping the new skills found; for Mahalanobis, noisy text gets worse (0.24 to 0.35).
- **The dominant failure is stories, not spelling.** Nearly every known story (0.80 to 0.99) is flagged by the meaning model, so the problem is understanding, which a spelling view cannot supply.
- **The spelling-only reader looks better only because it flags little**: it finds 0.26 to 0.36 of the new skills.
- Misses: I guessed the meaning model would find at least 0.75 of the new skills (it finds 0.13 to 0.56 at this calibration), and that the spelling reader's false alarm stayed under 0.20 (two detectors are at 0.22).

Files: `research/knowledge_status/results/char_front/TABLE_150_154.md`, design and predictions `docs/superpowers/specs/2026-11-01-nuance-reconciliation-and-char-front-design.md`.

## What this means for a general nuance algorithm

The evidence points to a layered design rather than one detector: (1) embeddings for topical novelty; (2) a reader that understands, for wording and function shifts; (3) trust between them decided by verified outcomes, not by looks. Chunk vectors, a spelling view, rewrites and surprise scores do not replace the second layer. What is still open is the part the lab flagged itself: the layer is not wired into the pool, it costs two language-model passes per part, and it relies on the model already knowing the skill's name.

## What I would do next (your call)

1. **Wire the two-reader gate into the pool behind a flag** and measure the pool (accuracy, experts, compute) on a stream with wording shifts. This is the missing link between the nuance lab and the pool.
2. **Distil the two questions into one small head** on the model's hidden state, so the reader costs one pass.
3. **Test wave or phase vectors only on a bed where order or roles matter**; nothing in the failures so far needs them.

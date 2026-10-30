# E-052 - General routing layer: the router assembled as one working object and run end to end on real models (demo, not an experiment)

- **ID:** E-052
- **Date:** 2026-10-30
- **Track:** G
- **Type:** process
- **Status:** completed (one demo run per q; not an experiment)
- **Evidence grade:** C (one seed, one run per setting, a demonstration of an assembled object; the evidence for its parts is E-048 to E-051)
- **Code / commit:** branch `knowledge-status-phase1`, commits 6f9df2a and 95cc4a7; `research/knowledge_status/routing_general/router.py`, `demo_router.py`; tests `tests/knowledge_status/test_rg_router.py` (10, 7 of 7 mutants caught)
- **People:** Claude, after the user's complaint of 2026-10-30 that the work was producing experiments and no working system

## Question
After G4c the user said we were spending time and compute on runs that return nothing and had not moved towards a working algorithm. The parts existed (cards, asking reader, density gate, calibration on outcomes, a container of our own) but there was no single object a pool could use, and no run that went from a message to an answer through all of them.

## Why we asked it
The user's goal since 2026-10-29 is a general algorithm AND a working system for the expert pool. G4b and G4c refined a weak spot and ended with 'keep the default'; that was my recommendation and the wrong priority.

## Prediction (written BEFORE the numbers, if there was one)
None. A demonstration of an assembled object, not an experiment; its numbers are not claims about the method.

## Method (how)
- `router.py`: `Router(names, teach, encode, reader_factory, q, tau, window, initial_cal, states, experts, featurize)`; `observe` (verified outcomes calibrate both readers' cuts, at least 19 needed, latest 60 kept), `route` (claimed with expert, state, team and probability, or unclaimed with the readers that said so; before calibration the asking reader's own claim rule, or an initial calibration in the teaching wording if one is given), `respond` (the claimed expert's answer or 'not mine'), `add_expert` (a growing pool tells the router; verified outcomes are kept). Built test first, 10 tests, mutation-checked.
- `demo_router.py`: the whole thing on real models; outcomes arrive for a random 30 % of the messages whatever the router decided (calibrating on what it claimed would be circular, E-051).

## Results (what came out)
Copied by script from `research/knowledge_status/results/routing_general/demo/` (arithmetic in everyday language, seed 150 = a tuning seed, three taught operations: multiplication (thin, 20 teaching examples), remainder, larger of two; real trained solvers, real language model, real encoder; a stream of 476 messages, outcomes for 30 % of them independent of the router).

q = 0.05 (default):
```
By mode (messages the router handled while in that mode):
       initial: 240 messages; taught-thing messages refused 0.260 (false refusals), untaught refused 0.714, everyday refused 1.000; answers right 0.556 of 54 answered
    calibrated: 236 messages; taught-thing messages refused 0.064 (false refusals), untaught refused 0.125, everyday refused 0.992; answers right 0.568 of 44 answered
```
q = 0.15:
```
By mode (messages the router handled while in that mode):
       initial:  90 messages; taught-thing messages refused 0.320 (false refusals), untaught refused 1.000, everyday refused 1.000; answers right 0.647 of 17 answered
    calibrated: 386 messages; taught-thing messages refused 0.232 (false refusals), untaught refused 0.371, everyday refused 1.000; answers right 0.548 of 73 answered
```
q = 0.25:
```
By mode (messages the router handled while in that mode):
       initial:  60 messages; taught-thing messages refused 0.583 (false refusals), untaught refused 1.000, everyday refused 1.000; answers right 0.800 of 5 answered
    calibrated: 416 messages; taught-thing messages refused 0.306 (false refusals), untaught refused 0.693, everyday refused 1.000; answers right 0.560 of 75 answered
```
Messages written by hand, routed at q = 0.05:
```
Your messages, routed by the router as it stands now:
  'Maya had 37 stickers and her friend gave her 58 more. How many stickers does she have now?'
      -> NOT MINE (said so: ask_exemplars)   [calibrated]
  'What is 12 times 7?'
      -> multiplication [thin expert] (p=1.00) answers 32   [calibrated]
  'What is the remainder when 47 is divided by 6?'
      -> the remainder after a division [solid expert] (p=1.00) answers 5   [calibrated]
  'Which is bigger, 23 or 41?'
      -> the larger of two numbers [solid expert] (p=1.00) answers 41   [calibrated]
  'Sam has 90 marbles and loses 35. How many are left?'
      -> NOT MINE (said so: ask_exemplars)   [calibrated]
  'What is the capital of France?'
      -> NOT MINE (said so: ask_exemplars, vmf_conformal)   [calibrated]
```

## Interpretation (what we understood, and on what basis)
Plain words. **It works end to end.** A message goes in, the router names the expert or says 'not mine', the expert answers, outcomes recalibrate it. Everyday sentences are refused (1.000 before, 0.992 after calibration). The hand-written messages behave as hoped: remainder and larger-of-two go to the right solid expert and are answered right (5, 41); addition and subtraction (never taught here) and the capital of France are refused. **What it also shows, honestly:** (1) routing right is not answering right: multiplication is a thin expert (20 examples) and answers 32 for 12 times 7; the router now says 'thin expert', but answers on known messages are right only about 0.56 of the time, so the experts, not the router, are the weak part in this test bed (as in E-049). (2) At the default q = 0.05 only 12.5 % of messages about untaught operations are refused after calibration: the closed list hands them to the nearest card. Moving q buys more refusals at the price of refusing taught things (q = 0.15: 0.37 untaught refused, 0.23 taught refused; q = 0.25: 0.69 and 0.31). That is a price to choose, not a defect to remove. (3) The refusal rate on taught things is above q (0.064, 0.232, 0.306): the sum-of-two-readers effect (F-53), the wording shift, and a calibration that starts from 3 to 6 messages at high q.

## Caveats and what went wrong
- One tuning seed, one run per q, about 30 verified messages at the end: a demonstration, not a measurement. The outcome contamination is the real one (the answer check vouches for 15.8 % of untaught messages), so the numbers include it.
- The first version of the demo did not show the expert's state; the thin multiplication expert answered wrongly without a warning. Fixed (Route.state, test first).
- The router is a container of our own, not the growmoe pool; `add_expert` is the hook but nothing spawns experts from refused messages yet.

## Consequence
There is now one object to build on. What stands between it and a good system, from this run: expert competence (the pool's job), the closed list's habit of handing untaught things to the nearest card (a better asking reader or a different question is research), and the loop that is the point of the architecture: refused messages recur, a teacher supplies examples, `add_expert`, the pool grows. Next, for the user's decision: close that loop in the demo (teach a refused operation, watch it get claimed), or connect the router to the growmoe pool.

## Evidence
- `research/knowledge_status/results/routing_general/demo/` (outputs at q 0.05 / 0.15 / 0.25)
- `research/knowledge_status/routing_general/router.py`, `demo_router.py`, `tests/knowledge_status/test_rg_router.py`

## Reproduce
```
cd D:/shm_ks
PYTHONPATH=D:/shm_ks python research/knowledge_status/routing_general/demo_router.py --domain arithmetic --seed 150 --q 0.05 --ask "What is 12 times 7?"
python -m pytest tests/knowledge_status/test_rg_router.py
```


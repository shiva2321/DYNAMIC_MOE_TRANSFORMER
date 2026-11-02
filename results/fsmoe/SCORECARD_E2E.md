# End-to-end scorecard (seeds [710, 711, 712, 713, 714])

11 declared predictions: **7 held, 4 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| P1 | A3 learns at least 0.80 of the 5 never-taught code skills with no human label on code | HELD | 1.000 |
| P2 | A3 opens at most 8 classes for them | HELD | 5.40 classes |
| P3 | A2 (tool on 10 % of code inputs) learns at least 0.40 of the code skills | HELD | 0.840 |
| P4 | A3 learns at least 0.10 more of all 28 never-taught skills than A1 | MISSED | A3 0.321, A1 0.364 |
| P5 | A3 uses at most 0.65 of A1's human labels | HELD | 228 against 386 (0.59) |
| P6 | A3's known code skills in held-out wordings end at 0.95 or more | HELD | 0.959 (before 0.960) |
| P7 | under A3 language learned <= 0.35 and language known skills in held-out wordings <= 0.85 | HELD | learned 0.170; known ho 0.798 |
| P8 | A4 (human labels doubled) learns at least 0.05 more language skills than A3 | MISSED | A4 0.170, A3 0.170 |
| P9 | no text from outside code is ever executed, and at most 5 % of the far code snippets are executed | MISSED | non-code texts run: 0; far code snippets run in the sandbox: 0.390 (running is harmless there, the sandbox has no capabilities; the behaviours it produced opened 0.4 extra classes) |
| Q1 | A5 learns at least 0.50 of the arithmetic skills and 0.80 of the code skills, with code known skills in held-out wordings at 0.93 or more | MISSED | arithmetic 0.400 (A3r 0.200); code 0.840; code known ho 0.933 |
| Q2 | the arithmetic share of the non-code human labels in A3 is within 0.05 of A1's (the dip is not a label-allocation effect) | HELD | A3 0.454, A1 0.449 |

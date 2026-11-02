# FS-MoE follow-up scorecard (seeds [710, 711, 712, 713, 714])

16 declared predictions: **9 held, 7 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| E1 | at M = 10 result_all separates known-in-held-out-styles from near-twins at AUROC >= 0.95 while the look stays <= 0.85 | HELD | result_all 1.000; look 0.588 |
| E2 | real library idioms of known functions are flagged <= 0.10 at the stream cut (result_all) | HELD | 0.000 (look 0.167) |
| E3 | result_all finds >= 0.90 of the withheld real library functions | HELD | 1.000 |
| E4 | the cascade is within 0.05 AUROC of result_all on near-twins | MISSED | gap 0.062 (cascade 0.938) |
| E5 | at M = 1 result_all still reaches AUROC >= 0.85 on near-twins | HELD | 1.000 |
| E6 | the behaviour fingerprint reaches AUROC 1.00 | HELD | 1.000 |
| L0 | of the known held-out-wording items that turn from right to wrong, at least half go to a newly learned skill (interleaved arms) | HELD | 0.671 |
| L1 | focused lifts the learned fraction under diverse 10 % from 0.28 to >= 0.50 | MISSED | 0.271 (interleaved 0.207) |
| L2 | focused_replay keeps known held-out-wording accuracy >= 0.85 while learning >= 0.40 (diverse 10 %) | MISSED | known ho after 0.788; learned 0.250 |
| L3 | under the oracle at 10 %, focused_replay learns >= 0.70 | HELD | 0.771 |
| L4 | known skills in taught wordings stay >= 0.95 in every arm | MISSED | lowest arm mean 0.930 |
| L5 | under balanced + focused_replay, known held-out-wording accuracy >= 0.85 while the learned fraction >= 0.40 | MISSED | known ho after 0.818; learned 0.093 |
| L6 | under balanced + interleaved, the learned fraction stays <= 0.30 | HELD | 0.029 (known ho after 0.867) |
| L7 | global_fast + balanced: learned fraction >= 0.60 and known held-out-wording accuracy >= 0.85 | MISSED | learned 0.100; known ho after 0.882 |
| L8 | global_fast + diverse: learned fraction >= 0.60 and known held-out accuracy <= 0.82 | MISSED | learned 0.364; known ho after 0.858 |
| L9 | after teaching only, global_fast reads taught skills within 0.01 of the pool alone (taught and held-out wordings) | HELD | difference tw -0.0058, ho +0.0002 |

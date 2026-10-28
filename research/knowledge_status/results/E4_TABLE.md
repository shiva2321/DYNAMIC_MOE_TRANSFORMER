# E4, relation from worked traces (learned-map reader; 10 of 15 candidate seeds had addition and subtraction really Solid)

B5: the parent set of a multiplication trace is exactly {add}, of a division trace exactly {sub}, in at least 0.90 of the traces; a decoy gets a false relation in at most 0.10.

| trace | seeds | traces per seed | parent exactly right | any parent named | chain valid | steps verified | steps the machine calls Solid | bar |
|---|---|---|---|---|---|---|---|---|
| mul (genuine) | 10 | 40 | 0.447 | 0.447 | 1.000 | 0.211 | 0.223 | at least 0.90: fail |
| div (genuine) | 10 | 40 | 0.508 | 0.508 | 1.000 | 0.444 | 0.457 | at least 0.90: fail |
| mul, decoy wrong_steps | 10 | 40 | n/a | 0.000 | 0.000 | 0.194 | 0.217 | at most 0.10: PASS |
| div, decoy wrong_steps | 10 | 40 | n/a | 0.000 | 0.000 | 0.292 | 0.450 | at most 0.10: PASS |
| max, decoy irrelevant_valid_steps | 10 | 40 | n/a | 0.000 | 0.000 | 0.106 | 0.120 | at most 0.10: PASS |
| mod, decoy unrelated_skill | 10 | 40 | n/a | 0.000 | 0.000 | 0.044 | 0.044 | at most 0.10: PASS |

Genuine traces by wording (0 symbols, 1 verbal, 2 number words, 3 story), share with the parent exactly right:
- mul: 0: 0.060, 1: 0.780, 2: 0.890, 3: 0.060
- div: 0: 0.020, 1: 1.000, 2: 0.960, 3: 0.050

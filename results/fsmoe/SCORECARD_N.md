# Notebook-first pool scorecard (seeds [710, 711, 712, 713, 714])

12 declared predictions: **9 held, 3 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| P1 | NB's accuracy is at most 0.015 below U128's on f10 and f30 | HELD | f10: NB 0.497 vs U128 0.493 (gap -0.004), f30: NB 0.482 vs U128 0.485 (gap +0.002) |
| P2 | NBc's accuracy is at most 0.01 below U128's on f10 and f30 | HELD | f10: NBc 0.493 vs U128 0.493 (gap +0.000), f30: NBc 0.479 vs U128 0.485 (gap +0.006) |
| P3 | f30: NB's network parameters are at most 5 % of U128's | HELD | 11493 against 1037040 (1.1 %) |
| P4 | f30: NB's memory is at most 0.85 times U128's and NBc's at most NB's plus 0.1 MB | HELD | NB 14.80 MB (0.78 x U128's 18.90), NBc 14.83 MB |
| P5 | f30: NB's accuracy is at least U32's minus 0.01 with no more memory than U32 | HELD | NB 0.482 (14.80 MB), U32 0.479 (15.94 MB) |
| P6 | NB keeps at least 0.97 experts per regime on f10 and f30 while U8 keeps at most 0.8 on f30 | HELD | NB 1.00 / 1.00, U8 on f30 0.45 |
| P7 | t: NB's taught-wording and new-wording accuracy are each at most 0.015 below U128's, with at most 0.95 times its memory | HELD | gaps +0.004 / +0.005; memory 11.01 against 12.85 MB |
| P8 | g: NB's accuracy is at least U128's minus 0.01 with at most 10 % of its network parameters | HELD | NB 0.627 vs U128 0.607; parameters 8.9 % |
| P9 | on every bed at most 30 % of NB's experts end with a network | MISSED | f10 2 %, f30 4 %, g 34 %, t 7 % |
| P10 | f30: NB's wall time is at most 1.6 times U128's | MISSED | 1.79 times |
| P11 | NB beats U8's accuracy by at least 0.03 on f10 and 0.1 on f30 | HELD | f10 +0.142, f30 +0.189 |
| P12 | g: U128 is not better than U8 by more than 0.02 | MISSED | U128 0.607, U8 0.568 (+0.039) |

# B7, realism check (dev; 13 of 30 seeds have both a Solid and an Unseen skill among add, sub, mul, div)

Balanced accuracy of the call 'Unseen or not' over Solid and Unseen parts; template = held-out wordings S4 to S8; realism = SVAMP one-step problems (never used for teaching or tuning).

| set | balanced accuracy |
|---|---|
| templates (S4 to S8, the same four operations) | 0.800 |
| realism problems | 0.475 |
| gap (realism - templates), mean over seeds | -0.325 |

On the realism problems: Solid skills called Unseen 0.981; Unseen skills found 0.932.

Reported, not gated: on the problems whose operands are both at most 99 (the size the template bed uses), 13 seeds can be scored, balanced accuracy 0.476 against 0.800 on the templates (gap -0.325).

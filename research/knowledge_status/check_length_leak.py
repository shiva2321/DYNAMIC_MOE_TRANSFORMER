"""Anti-goal A1 check that needs no model: how well does the LENGTH of a request (or of the whole message) alone give the skill away, inside one surface family?

    KS_ALLOW_CONFIRM=1 PYTHONPATH=<repo root> python check_length_leak.py

For each family the draws are split in half; on the first half the most common skill per length bin (4 characters wide) is learned, on the second half it is applied. Chance is 1 over the
number of skills in the family. Some leak is natural (a story about a remainder is longer than one about a sum) and equal in the dev and the confirmatory pool; this prints both pools side
by side so that a pool that is much leakier than the other can be seen. It is a disclosure, not a bar. The real test of surface cues is B2: the hashed word-matcher R0 must fail B1a or B1b.
"""

import collections
from typing import Dict, Tuple

import numpy as np

import generator as G
import skills

FAMILIES = tuple(f"S{i}" for i in range(1, 9))


def length_leak(pool: str, n: int = 600, seed: int = 0, bin_width: int = 4) -> Dict[str, Tuple[float, float, float]]:
    """{family: (accuracy from the request's length, accuracy from the whole message's length, chance)}."""
    out: Dict[str, Tuple[float, float, float]] = {}
    rng = np.random.default_rng(seed)
    for fam in FAMILIES:
        rows = []
        for _ in range(n):
            skill = skills.SKILLS[int(rng.integers(0, 6))]
            if fam == "S5" and skill not in skills.COMMUTATIVE:
                continue
            m = G.make_single(rng, pool, skill, fam)
            p = next(q for q in m.parts if q.kind == "skill")
            rows.append((len(m.text[p.start:p.end]), len(m.text), skill))
        half = len(rows) // 2
        accs = []
        for col in (0, 1):
            table = collections.defaultdict(collections.Counter)
            for r in rows[:half]:
                table[r[col] // bin_width][r[2]] += 1
            hit = seen = 0
            for r in rows[half:]:
                counts = table.get(r[col] // bin_width)
                if counts:
                    seen += 1
                    hit += counts.most_common(1)[0][0] == r[2]
            accs.append(hit / max(seen, 1))
        out[fam] = (round(accs[0], 3), round(accs[1], 3), round(1 / len({r[2] for r in rows}), 3))
    return out


if __name__ == "__main__":
    for pool in ("dev", "confirm"):
        print(pool, "(request-length accuracy, whole-message-length accuracy, chance):")
        for fam, row in length_leak(pool).items():
            print(f"  {fam}: {row}")

"""A stream that needs capacity: arithmetic on digit codes only, six operators, thousands of examples each (the earlier arithmetic study found about 10,000 per operator are needed before width
matters: a 16-unit expert reaches 11 % exact match, a 512-unit one 37 %). The Stage 1 suite gave every regime 400 messages, where 16, 128 and 512 units score 0.581, 0.622 and 0.631: it cannot say
whether growth pays. Here it can. No text, no encoder, no out-of-distribution sets."""

from . import domains as DM
from . import suite as SU


def build_capacity_suite(schedule: str, seed: int, fe, n_per_op: int = 10000, n_test: int = 600) -> SU.UStream:
    dom = DM.load_domain("arith", fe, seed, cap_train=6 * n_per_op, cap_test=6 * n_test)
    return SU.build_suite(schedule, {"arith": dom}, seed, {}, n_visit_arith=n_per_op, n_test=n_test, revisit_k=0, n_phases_mix=5)

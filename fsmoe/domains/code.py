"""Code as skill recognition: sixteen function families written in several styles, with the outside anchor being EXECUTION.

A snippet implements one family (``sum_list``, ``is_prime``, ...). Its behaviour is its output on fixed probe inputs (the signature); a snippet is genuine only if that signature equals the family's
reference signature, and no two families share a signature, so "same function, other surface" and "other function, same surface" are decided by running the code, never by reading it. The system
never sees the signature: it is the checker (and the source of verified outcomes for calibration).

Teaching families: loop, builtin, recursion (descriptive names). Held-out families: renamed (neutral names), short (one-letter names), commented (generic comments and docstrings), formatted (type
hints and blank lines), boilerplate (imports, an unrelated helper, a main guard), altstyle (while loops, a style never taught). Near-twins differ by one operator or one word: sum/product,
max/min, count_even/count_odd, upper/lower, reverse/is_palindrome, factorial/fibonacci.
"""

import contextlib
import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Domain, Item, rng_for

SKILLS = ("sum_list", "max_list", "min_list", "count_even", "count_positive", "reverse_str", "is_palindrome", "count_vowels", "upper_str", "factorial", "fibonacci",
          "product_list", "count_odd", "lower_str", "is_prime", "digit_sum")
WITHHELD = ("product_list", "count_odd", "lower_str", "is_prime", "digit_sum")      # the first three are near-twins of taught skills, the last two are far
FAR_WITHHELD = ("is_prime", "digit_sum")
TWINS = {"sum_list": "product_list", "product_list": "sum_list", "max_list": "min_list", "min_list": "max_list", "count_even": "count_odd", "count_odd": "count_even",
         "upper_str": "lower_str", "lower_str": "upper_str", "reverse_str": "is_palindrome", "is_palindrome": "reverse_str", "factorial": "fibonacci", "fibonacci": "factorial",
         "count_positive": "count_even", "count_vowels": "upper_str", "is_prime": "digit_sum", "digit_sum": "is_prime"}
TEACH = ("loop", "builtin", "recursion")
HELDOUT = ("renamed", "short", "commented", "formatted", "boilerplate", "altstyle")
KIND = {s: "list" for s in ("sum_list", "max_list", "min_list", "count_even", "count_positive", "product_list", "count_odd")}
KIND.update({s: "str" for s in ("reverse_str", "is_palindrome", "count_vowels", "upper_str", "lower_str")})
KIND.update({s: "int" for s in ("factorial", "fibonacci", "is_prime", "digit_sum")})
PROBES = {"list": ([3, 1, 4, 1, 5], [2, 2], [-3, 8, 0, -1], [10], [7, 4, 9, 6, 2, 8]),
          "str": ("level", "banana", "Hello World", "a", "abcdef"),
          "int": (0, 1, 5, 7, 10)}


def _ref(skill: str, v):
    if skill == "sum_list":
        return sum(v)
    if skill == "product_list":
        return math.prod(v)
    if skill == "max_list":
        return max(v)
    if skill == "min_list":
        return min(v)
    if skill == "count_even":
        return sum(1 for e in v if e % 2 == 0)
    if skill == "count_odd":
        return sum(1 for e in v if e % 2 == 1)
    if skill == "count_positive":
        return sum(1 for e in v if e > 0)
    if skill == "reverse_str":
        return v[::-1]
    if skill == "is_palindrome":
        return v == v[::-1]
    if skill == "count_vowels":
        return sum(1 for c in v if c in "aeiouAEIOU")
    if skill == "upper_str":
        return v.upper()
    if skill == "lower_str":
        return v.lower()
    if skill == "factorial":
        return math.factorial(v)
    if skill == "fibonacci":
        a, b = 0, 1
        for _ in range(v):
            a, b = b, a + b
        return a
    if skill == "is_prime":
        return v > 1 and all(v % d for d in range(2, v))
    if skill == "digit_sum":
        return sum(int(c) for c in str(abs(v)))
    raise KeyError(skill)


REF_SIGNATURE: Dict[str, tuple] = {s: tuple(repr(_ref(s, p)) for p in PROBES[KIND[s]]) for s in SKILLS}

# ---------------------------------------------------------------------------------------------------------------------------------- templates
# placeholders: {f} function, {x} parameter, {i} loop variable, {acc} accumulator, {p} {q} fixed pair for fibonacci
_LOOP_COUNT = ["def {f}({x}):", "    {acc} = 0", "    for {i} in {x}:", "        if {cond}:", "            {acc} += 1", "    return {acc}"]


def _count(cond: str, rec_cond: str, builtin_cond: str) -> Dict[str, List[str]]:
    return {"loop": [l.replace("{cond}", cond) for l in _LOOP_COUNT],
            "builtin": ["def {f}({x}):", "    return len([{i} for {i} in {x} if " + builtin_cond + "])"],
            "recursion": ["def {f}({x}):", "    if not {x}:", "        return 0", "    return (" + rec_cond + ") + {f}({x}[1:])"],
            "while": ["def {f}({x}):", "    {acc} = 0", "    {i} = 0", "    while {i} < len({x}):", "        if " + cond.replace("{i}", "{x}[{i}]") + ":", "            {acc} += 1",
                      "        {i} += 1", "    return {acc}"]}


BODIES: Dict[str, Dict[str, List[str]]] = {
    "sum_list": {"loop": ["def {f}({x}):", "    {acc} = 0", "    for {i} in {x}:", "        {acc} += {i}", "    return {acc}"],
                 "builtin": ["def {f}({x}):", "    return sum({x})"],
                 "recursion": ["def {f}({x}):", "    if not {x}:", "        return 0", "    return {x}[0] + {f}({x}[1:])"],
                 "while": ["def {f}({x}):", "    {acc} = 0", "    {i} = 0", "    while {i} < len({x}):", "        {acc} += {x}[{i}]", "        {i} += 1", "    return {acc}"]},
    "product_list": {"loop": ["def {f}({x}):", "    {acc} = 1", "    for {i} in {x}:", "        {acc} *= {i}", "    return {acc}"],
                     "builtin": ["def {f}({x}):", "    import math", "    return math.prod({x})"],
                     "recursion": ["def {f}({x}):", "    if not {x}:", "        return 1", "    return {x}[0] * {f}({x}[1:])"],
                     "while": ["def {f}({x}):", "    {acc} = 1", "    {i} = 0", "    while {i} < len({x}):", "        {acc} *= {x}[{i}]", "        {i} += 1", "    return {acc}"]},
    "max_list": {"loop": ["def {f}({x}):", "    {acc} = {x}[0]", "    for {i} in {x}:", "        if {i} > {acc}:", "            {acc} = {i}", "    return {acc}"],
                 "builtin": ["def {f}({x}):", "    return max({x})"],
                 "recursion": ["def {f}({x}):", "    if len({x}) == 1:", "        return {x}[0]", "    rest = {f}({x}[1:])", "    return {x}[0] if {x}[0] > rest else rest"],
                 "while": ["def {f}({x}):", "    {acc} = {x}[0]", "    {i} = 1", "    while {i} < len({x}):", "        if {x}[{i}] > {acc}:", "            {acc} = {x}[{i}]", "        {i} += 1", "    return {acc}"]},
    "min_list": {"loop": ["def {f}({x}):", "    {acc} = {x}[0]", "    for {i} in {x}:", "        if {i} < {acc}:", "            {acc} = {i}", "    return {acc}"],
                 "builtin": ["def {f}({x}):", "    return min({x})"],
                 "recursion": ["def {f}({x}):", "    if len({x}) == 1:", "        return {x}[0]", "    rest = {f}({x}[1:])", "    return {x}[0] if {x}[0] < rest else rest"],
                 "while": ["def {f}({x}):", "    {acc} = {x}[0]", "    {i} = 1", "    while {i} < len({x}):", "        if {x}[{i}] < {acc}:", "            {acc} = {x}[{i}]", "        {i} += 1", "    return {acc}"]},
    "count_even": _count("{i} % 2 == 0", "{x}[0] % 2 == 0", "{i} % 2 == 0"),
    "count_odd": _count("{i} % 2 == 1", "{x}[0] % 2 == 1", "{i} % 2 == 1"),
    "count_positive": _count("{i} > 0", "{x}[0] > 0", "{i} > 0"),
    "reverse_str": {"loop": ["def {f}({x}):", "    {acc} = ''", "    for {i} in {x}:", "        {acc} = {i} + {acc}", "    return {acc}"],
                    "builtin": ["def {f}({x}):", "    return {x}[::-1]"],
                    "recursion": ["def {f}({x}):", "    if len({x}) <= 1:", "        return {x}", "    return {f}({x}[1:]) + {x}[0]"],
                    "while": ["def {f}({x}):", "    {acc} = ''", "    {i} = len({x}) - 1", "    while {i} >= 0:", "        {acc} += {x}[{i}]", "        {i} -= 1", "    return {acc}"]},
    "is_palindrome": {"loop": ["def {f}({x}):", "    {acc} = ''", "    for {i} in {x}:", "        {acc} = {i} + {acc}", "    return {acc} == {x}"],
                      "builtin": ["def {f}({x}):", "    return {x} == {x}[::-1]"],
                      "recursion": ["def {f}({x}):", "    if len({x}) < 2:", "        return True", "    return {x}[0] == {x}[-1] and {f}({x}[1:-1])"],
                      "while": ["def {f}({x}):", "    {i} = 0", "    while {i} < len({x}) // 2:", "        if {x}[{i}] != {x}[-1 - {i}]:", "            return False", "        {i} += 1", "    return True"]},
    "count_vowels": {"loop": ["def {f}({x}):", "    {acc} = 0", "    for {i} in {x}:", "        if {i} in 'aeiouAEIOU':", "            {acc} += 1", "    return {acc}"],
                     "builtin": ["def {f}({x}):", "    return sum(1 for {i} in {x} if {i} in 'aeiouAEIOU')"],
                     "recursion": ["def {f}({x}):", "    if not {x}:", "        return 0", "    return ({x}[0] in 'aeiouAEIOU') + {f}({x}[1:])"],
                     "while": ["def {f}({x}):", "    {acc} = 0", "    {i} = 0", "    while {i} < len({x}):", "        if {x}[{i}] in 'aeiouAEIOU':", "            {acc} += 1", "        {i} += 1", "    return {acc}"]},
    "upper_str": {"loop": ["def {f}({x}):", "    {acc} = ''", "    for {i} in {x}:", "        {acc} += {i}.upper()", "    return {acc}"],
                  "builtin": ["def {f}({x}):", "    return {x}.upper()"],
                  "recursion": ["def {f}({x}):", "    if not {x}:", "        return ''", "    return {x}[0].upper() + {f}({x}[1:])"],
                  "while": ["def {f}({x}):", "    {acc} = ''", "    {i} = 0", "    while {i} < len({x}):", "        {acc} += {x}[{i}].upper()", "        {i} += 1", "    return {acc}"]},
    "lower_str": {"loop": ["def {f}({x}):", "    {acc} = ''", "    for {i} in {x}:", "        {acc} += {i}.lower()", "    return {acc}"],
                  "builtin": ["def {f}({x}):", "    return {x}.lower()"],
                  "recursion": ["def {f}({x}):", "    if not {x}:", "        return ''", "    return {x}[0].lower() + {f}({x}[1:])"],
                  "while": ["def {f}({x}):", "    {acc} = ''", "    {i} = 0", "    while {i} < len({x}):", "        {acc} += {x}[{i}].lower()", "        {i} += 1", "    return {acc}"]},
    "factorial": {"loop": ["def {f}({x}):", "    {acc} = 1", "    for {i} in range(2, {x} + 1):", "        {acc} *= {i}", "    return {acc}"],
                  "builtin": ["def {f}({x}):", "    import math", "    return math.factorial({x})"],
                  "recursion": ["def {f}({x}):", "    if {x} <= 1:", "        return 1", "    return {x} * {f}({x} - 1)"],
                  "while": ["def {f}({x}):", "    {acc} = 1", "    {i} = 2", "    while {i} <= {x}:", "        {acc} *= {i}", "        {i} += 1", "    return {acc}"]},
    "fibonacci": {"loop": ["def {f}({x}):", "    {p}, {q} = 0, 1", "    for {i} in range({x}):", "        {p}, {q} = {q}, {p} + {q}", "    return {p}"],
                  "builtin": ["def {f}({x}):", "    {p}, {q} = 0, 1", "    {i} = 0", "    while {i} < {x}:", "        {p}, {q} = {q}, {p} + {q}", "        {i} += 1", "    return {p}"],
                  "recursion": ["def {f}({x}):", "    if {x} < 2:", "        return {x}", "    return {f}({x} - 1) + {f}({x} - 2)"],
                  "while": ["def {f}({x}):", "    {p}, {q} = 0, 1", "    {i} = 0", "    while {i} < {x}:", "        {p}, {q} = {q}, {p} + {q}", "        {i} += 1", "    return {p}"]},
    "is_prime": {"loop": ["def {f}({x}):", "    if {x} < 2:", "        return False", "    for {i} in range(2, {x}):", "        if {x} % {i} == 0:", "            return False", "    return True"],
                 "builtin": ["def {f}({x}):", "    return {x} > 1 and all({x} % {i} for {i} in range(2, {x}))"],
                 "recursion": ["def {f}({x}, d=2):", "    if {x} < 2:", "        return False", "    if d >= {x}:", "        return True", "    if {x} % d == 0:", "        return False", "    return {f}({x}, d + 1)"],
                 "while": ["def {f}({x}):", "    if {x} < 2:", "        return False", "    {i} = 2", "    while {i} < {x}:", "        if {x} % {i} == 0:", "            return False", "        {i} += 1", "    return True"]},
    "digit_sum": {"loop": ["def {f}({x}):", "    {acc} = 0", "    for {i} in str(abs({x})):", "        {acc} += int({i})", "    return {acc}"],
                  "builtin": ["def {f}({x}):", "    return sum(int({i}) for {i} in str(abs({x})))"],
                  "recursion": ["def {f}({x}):", "    {x} = abs({x})", "    if {x} < 10:", "        return {x}", "    return {x} % 10 + {f}({x} // 10)"],
                  "while": ["def {f}({x}):", "    {x} = abs({x})", "    {acc} = 0", "    while {x} > 0:", "        {acc} += {x} % 10", "        {x} //= 10", "    return {acc}"]},
}

FUNC_DESCRIPTIVE = {"sum_list": ["sum_list", "total", "add_all", "list_sum"], "product_list": ["product_list", "multiply_all", "product", "list_product"],
                    "max_list": ["max_list", "largest", "find_max", "biggest"], "min_list": ["min_list", "smallest", "find_min", "lowest"],
                    "count_even": ["count_even", "evens", "num_even", "count_evens"], "count_odd": ["count_odd", "odds", "num_odd", "count_odds"],
                    "count_positive": ["count_positive", "positives", "num_positive", "count_pos"],
                    "reverse_str": ["reverse_str", "reverse", "flip_text", "backwards"], "is_palindrome": ["is_palindrome", "palindrome", "reads_same", "check_palindrome"],
                    "count_vowels": ["count_vowels", "vowels", "num_vowels", "vowel_count"], "upper_str": ["upper_str", "to_upper", "shout", "uppercase"],
                    "lower_str": ["lower_str", "to_lower", "whisper", "lowercase"], "factorial": ["factorial", "fact", "compute_factorial", "n_factorial"],
                    "fibonacci": ["fibonacci", "fib", "nth_fib", "fib_number"], "is_prime": ["is_prime", "prime_check", "check_prime", "primality"],
                    "digit_sum": ["digit_sum", "sum_digits", "digits_total", "add_digits"]}
X_DESC = {"list": ["numbers", "nums", "values", "items", "data"], "str": ["text", "word", "string", "s"], "int": ["n", "num", "number", "value"]}
I_DESC = {"list": ["i", "v", "item", "e"], "str": ["ch", "c", "letter", "char"], "int": ["i", "k", "step"]}
ACC_DESC = ["acc", "result", "total", "count", "out", "res"]
NEUTRAL = {"f": ["process", "compute", "run", "handle", "calc", "apply_all", "transform", "evaluate", "helper_fn", "step_one"],
           "x": ["data", "arg", "inp", "payload", "thing"], "i": ["e", "elem", "k", "tok"], "acc": ["r", "out", "tmp", "val", "buf"]}
SHORT = {"f": ["f", "g", "h", "fn"], "x": ["a", "s", "l", "xs", "w"], "i": ["i", "j", "e"], "acc": ["r", "t", "c", "m"]}
COMMENTS = ("# helper function", "# TODO: refactor later", "# NOTE: keep this simple", "# utility", "# used in several places")
DOCS = ('"""Utility function."""', '"""Helper."""', '"""See module notes."""')
WORDS = ("level", "banana", "racecar", "python", "abcba", "noon", "Hello World", "robot")
HINT = {"list": "list", "str": "str", "int": "int"}


def _pick(rng: np.random.Generator, pool: List[str], taken: set) -> str:
    options = [p for p in pool if p not in taken]
    name = str(rng.choice(options if options else pool))
    taken.add(name)
    return name


def _example(skill: str, rng: np.random.Generator) -> str:
    kind = KIND[skill]
    if kind == "list":
        return repr([int(v) for v in rng.integers(-5, 21, int(rng.integers(3, 7)))])
    if kind == "str":
        return repr(str(rng.choice(WORDS)))
    if skill == "factorial":
        return str(int(rng.integers(3, 9)))
    if skill == "fibonacci":
        return str(int(rng.integers(3, 10)))
    if skill == "is_prime":
        return str(int(rng.integers(2, 41)))
    return str(int(rng.integers(10, 9999)))


def render(skill: str, family: str, rng: np.random.Generator) -> Tuple[str, str]:
    """One snippet of ``skill`` in ``family``; returns (source text, function name)."""
    kind = KIND[skill]
    style = {"loop": "loop", "builtin": "builtin", "recursion": "recursion", "altstyle": "while"}.get(family) or str(rng.choice(("loop", "builtin", "recursion")))
    taken: set = {"p", "q", "d", "rest"}
    if family in ("renamed",):
        f, x, i, acc = (_pick(rng, NEUTRAL[k], taken) for k in ("f", "x", "i", "acc"))
    elif family == "short":
        f, x, i, acc = (_pick(rng, SHORT[k], taken) for k in ("f", "x", "i", "acc"))
    else:
        f = _pick(rng, FUNC_DESCRIPTIVE[skill], taken)
        x, i, acc = _pick(rng, X_DESC[kind], taken), _pick(rng, I_DESC[kind], taken), _pick(rng, ACC_DESC, taken)
    lines = [l.replace("{f}", f).replace("{x}", x).replace("{i}", i).replace("{acc}", acc).replace("{p}", "p").replace("{q}", "q") for l in BODIES[skill][style]]
    ex = _example(skill, rng)
    if family == "commented":
        lines.insert(1, "    " + str(rng.choice(DOCS)))
        lines.insert(0, str(rng.choice(COMMENTS)))
        lines[-1] += "  # done"
        lines.append(f"print({f}({ex}))")
    elif family == "formatted":
        head = lines[0]
        head = head.replace(f"({x}, d=2)", f"({x}: int, d: int = 2)") if ", d=2)" in head else head.replace(f"({x})", f"({x}: {HINT[kind]})")
        lines[0] = head.replace("):", ") -> object:")
        lines = [""] + lines + ["", f"print({f}({ex}))", ""]
    elif family == "boilerplate":
        lines = ["import sys", "", "", "def _log(msg):", "    print(msg, file=sys.stderr)", "", ""] + lines + ["", "", "if __name__ == '__main__':", "    _log('running')", f"    print({f}({ex}))"]
    else:
        lines.append(f"print({f}({ex}))")
    return "\n".join(lines) + "\n", f


def signature(code: str, fn: str, kind: str) -> Optional[tuple]:
    """Run ``code`` and call ``fn`` on the probe inputs of ``kind``; None when it fails. The module name is not "__main__", so a main guard does not run."""
    ns = {"__name__": "fsmoe_check"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, "<snippet>", "exec"), ns)
            f = ns[fn]
            return tuple(repr(f(list(p) if kind == "list" else p)) for p in PROBES[kind])
    except Exception:
        return None


class CodeDomain(Domain):
    name = "code"
    skills = SKILLS
    twins = TWINS
    teach_families = TEACH
    heldout_families = HELDOUT
    withheld = WITHHELD
    far_withheld = FAR_WITHHELD

    def make(self, skill: str, family: str, n: int, seed: int, split: str = "teach") -> List[Item]:
        if skill not in SKILLS or family not in self.families:
            raise ValueError(f"unknown skill {skill!r} or family {family!r}")
        rng = rng_for(seed, "code", skill, family, split)
        out = []
        for _ in range(n):
            text, fn = render(skill, family, rng)
            out.append(Item(text, self.name, skill, family, (fn,)))
        return out

    def check(self, item: Item) -> bool:
        return signature(item.text, item.meta[0], KIND[item.skill]) == REF_SIGNATURE[item.skill]

"""Arithmetic as skill recognition: eight operations, eight wording families, near-twin operations, three operations never taught.

The task of a system here is to say WHICH operation a message asks for (the generator knows the true answer and uses it as the checker; the answer is never a feature). Teaching families: symbols,
number words, verbal phrasing. Held-out families: short stories, operands in the other order, chatter around the request, typos and casing, register (formal, casual, terse). Operands and numbers are
the same in every family, so only the wording moves.
"""

from typing import Dict, List, Tuple

import numpy as np

from .base import Domain, Item, noisy, rng_for

SKILLS = ("add", "sub", "mul", "floordiv", "max", "mod", "min", "absdiff")
TWINS = {"add": "sub", "sub": "add", "mul": "floordiv", "floordiv": "mod", "mod": "floordiv", "max": "min", "min": "max", "absdiff": "sub"}
WITHHELD = ("mod", "min", "absdiff")
TEACH = ("symbols", "words", "verbal")
HELDOUT = ("stories", "reorder", "chatter", "typos", "register")

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def words(n: int) -> str:
    if not 0 <= n <= 99:
        raise ValueError(f"{n} is outside 0..99")
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")


def answer(skill: str, a: int, b: int) -> int:
    return {"add": a + b, "sub": a - b, "mul": a * b, "floordiv": a // b, "mod": a % b, "max": max(a, b), "min": min(a, b), "absdiff": abs(a - b)}[skill]


T: Dict[str, Dict[str, List[str]]] = {
    "symbols": {
        "add": ["{a} + {b}", "{a}+{b}=", "what is {a} + {b}?", "compute {a} + {b}"],
        "sub": ["{a} - {b}", "{a}-{b}=", "what is {a} - {b}?", "compute {a} - {b}"],
        "mul": ["{a} * {b}", "{a}*{b}=", "what is {a} * {b}?", "compute {a} x {b}"],
        "floordiv": ["{a} // {b}", "{a}//{b}=", "what is {a} // {b}?", "compute {a} // {b}"],
        "mod": ["{a} % {b}", "{a}%{b}=", "what is {a} % {b}?", "compute {a} % {b}"],
        "max": ["max({a}, {b})", "max({a},{b})=", "what is max({a}, {b})?", "compute max({a}, {b})"],
        "min": ["min({a}, {b})", "min({a},{b})=", "what is min({a}, {b})?", "compute min({a}, {b})"],
        "absdiff": ["|{a} - {b}|", "abs({a} - {b})", "what is |{a} - {b}|?", "compute abs({a}-{b})"],
    },
    "words": {
        "add": ["{A} plus {B}", "what is {A} plus {B}?", "{A} and {B} added together"],
        "sub": ["{A} minus {B}", "what is {A} minus {B}?", "{B} subtracted from {A}"],
        "mul": ["{A} times {B}", "what is {A} times {B}?", "{A} multiplied by {B}"],
        "floordiv": ["{A} divided by {B}, rounded down", "what is {A} divided by {B} without the remainder?", "{A} over {B}, whole part only"],
        "mod": ["{A} modulo {B}", "what is {A} mod {B}?", "the remainder of {A} divided by {B}"],
        "max": ["the larger of {A} and {B}", "which is bigger, {A} or {B}?", "the maximum of {A} and {B}"],
        "min": ["the smaller of {A} and {B}", "which is smaller, {A} or {B}?", "the minimum of {A} and {B}"],
        "absdiff": ["the absolute difference between {A} and {B}", "how far apart are {A} and {B}?", "the gap between {A} and {B}"],
    },
    "verbal": {
        "add": ["Add {a} and {b}.", "Find the sum of {a} and {b}.", "Please add {a} to {b}."],
        "sub": ["Subtract {b} from {a}.", "Take {b} away from {a}.", "Find the difference when {b} is taken from {a}."],
        "mul": ["Multiply {a} by {b}.", "Find the product of {a} and {b}.", "Work out {a} times {b}."],
        "floordiv": ["Divide {a} by {b} and drop the fraction.", "How many whole times does {b} go into {a}?", "Find the whole-number quotient of {a} divided by {b}."],
        "mod": ["Find the remainder when {a} is divided by {b}.", "What is left over after dividing {a} by {b}?", "Take {a} modulo {b}."],
        "max": ["Give the larger of {a} and {b}.", "Which number is bigger, {a} or {b}?", "Find the maximum of {a} and {b}."],
        "min": ["Give the smaller of {a} and {b}.", "Which number is smaller, {a} or {b}?", "Find the minimum of {a} and {b}."],
        "absdiff": ["Find the absolute difference of {a} and {b}.", "How far apart are {a} and {b}?", "Find the gap between {a} and {b}."],
    },
    "stories": {
        "add": ["I had {a} apples and then got {b} more. How many do I have now?", "There were {a} birds on a branch and {b} more landed. How many birds are there now?",
                "Sam saved {a} dollars last month and {b} dollars this month. How much has he saved in total?"],
        "sub": ["Maria had {a} stickers and gave away {b}. How many are left?", "A shop had {a} shirts and sold {b}. How many shirts remain?",
                "There were {a} cookies in the jar and kids ate {b}. How many are still in the jar?"],
        "mul": ["There are {a} boxes with {b} pencils in each. How many pencils are there altogether?", "A bus makes {a} trips and carries {b} people each trip. How many people were carried in all?",
                "Each of {a} tables has {b} chairs. How many chairs are there?"],
        "floordiv": ["{a} cookies are shared equally among {b} kids. How many whole cookies does each kid get?", "A rope {a} metres long is cut into pieces of {b} metres. How many full pieces can be cut?",
                     "{a} students are split into teams of {b}. How many complete teams are there?"],
        "mod": ["{a} cookies are shared equally among {b} kids. How many cookies are left over?", "A rope {a} metres long is cut into pieces of {b} metres. How many metres remain after the last full piece?",
                "{a} students are split into teams of {b}. How many students cannot join a full team?"],
        "max": ["Tom scored {a} points and Ana scored {b}. What is the highest score?", "One bag weighs {a} kilos and another weighs {b} kilos. How heavy is the heavier bag?",
                "Two trains carry {a} and {b} passengers. How many passengers are on the fuller train?"],
        "min": ["Tom scored {a} points and Ana scored {b}. What is the lowest score?", "One bag weighs {a} kilos and another weighs {b} kilos. How heavy is the lighter bag?",
                "Two trains carry {a} and {b} passengers. How many passengers are on the emptier train?"],
        "absdiff": ["Tom is {a} cm tall and Ana is {b} cm tall. By how many cm do their heights differ?", "One road is {a} km long and another is {b} km long. How much longer is the longer road than the shorter one?",
                    "Two shops sold {a} and {b} items. What is the difference between their sales?"],
    },
    "reorder": {
        "add": ["{b} + {a}", "{b} plus {a}", "Add {b} to {a}."],
        "sub": ["{a} take away {b}", "{b} subtracted from {a}", "From {a}, remove {b}."],
        "mul": ["{b} * {a}", "{b} times {a}", "Multiply {b} by {a}."],
        "floordiv": ["{b} goes into {a} how many whole times?", "{a} split into groups of {b}: how many whole groups?", "{a} over {b}, whole number only."],
        "mod": ["{b} goes into {a}, what is left?", "{a} split into groups of {b}: how many left over?", "{a} over {b}, remainder only."],
        "max": ["max({b}, {a})", "the larger of {b} and {a}", "Which is bigger, {b} or {a}?"],
        "min": ["min({b}, {a})", "the smaller of {b} and {a}", "Which is smaller, {b} or {a}?"],
        "absdiff": ["|{b} - {a}|", "the gap between {b} and {a}", "How far apart are {b} and {a}?"],
    },
}
PRE = ("hey, quick question: ", "ok so, ", "sorry to bother you, ", "my homework says: ", "random thought, ", "hi! ")
POST = (" thanks a lot!", " (no calculator please)", " just curious.", " need it before lunch.", " cheers", "")
REGISTER = ("Kindly compute the following: {t}.", "I would be grateful if you could work out: {t}.", "yo, {t}", "can u tell me {t} lol", "{t}?", "{t} -> ?")


def _draw(skill: str, rng: np.random.Generator) -> Tuple[int, int]:
    if skill in ("floordiv", "mod"):
        return int(rng.integers(13, 100)), int(rng.integers(2, 13))
    if skill == "sub":
        a = int(rng.integers(5, 100))
        return a, int(rng.integers(2, a))
    return int(rng.integers(2, 100)), int(rng.integers(2, 100))


def _fill(template: str, a: int, b: int) -> str:
    return template.format(a=a, b=b, A=words(a), B=words(b))


class ArithDomain(Domain):
    name = "arith"
    skills = SKILLS
    twins = TWINS
    teach_families = TEACH
    heldout_families = HELDOUT
    withheld = WITHHELD
    far_withheld: Tuple[str, ...] = ()          # every operation has a taught near neighbour; far novelty here is everyday text (fsmoe.domains.far)

    def make(self, skill: str, family: str, n: int, seed: int, split: str = "teach") -> List[Item]:
        if skill not in SKILLS or family not in self.families:
            raise ValueError(f"unknown skill {skill!r} or family {family!r}")
        rng = rng_for(seed, "arith", skill, family, split)
        out = []
        for _ in range(n):
            a, b = _draw(skill, rng)
            if family in TEACH or family in ("stories", "reorder"):
                base = family
            else:
                base = str(rng.choice(("words", "verbal") if family == "typos" else TEACH))
            text = _fill(str(rng.choice(T[base][skill])), a, b)
            if family == "chatter":
                text = str(rng.choice(PRE)) + text + str(rng.choice(POST))
            elif family == "typos":
                text = noisy(text, rng)
            elif family == "register":
                text = str(rng.choice(REGISTER)).format(t=text.rstrip("?.:= "))
            out.append(Item(text, self.name, skill, family, (a, b, answer(skill, a, b))))
        return out

    def check(self, item: Item) -> bool:
        a, b, ans = item.meta
        return item.skill in SKILLS and answer(item.skill, a, b) == ans

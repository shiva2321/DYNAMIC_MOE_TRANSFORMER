"""The DEV template pool (design, tuning, bars). Hand-written. The confirmatory pool (pool_confirm.py, written at the freeze) shares no frame or core with it."""

from pools import Core, Frame, Pool


def _cores(family, skill, texts, swap=False):
    return tuple(Core(f"{family}.{skill}.c{i}", skill, t, swap) for i, t in enumerate(texts))


def _frames(family, texts, noises=None):
    noises = noises or [None] * len(texts)
    return tuple(Frame(f"{family}.f{i}", t, n) for i, (t, n) in enumerate(zip(texts, noises)))


S1 = {   # symbols and infix words, digits
    "add": ["{a} + {b}", "{a} plus {b}", "{a}+{b}", "{a} plus {b} equals"],
    "sub": ["{a} - {b}", "{a} minus {b}", "{a}-{b}", "{a} minus {b} equals"],
    "mul": ["{a} * {b}", "{a} times {b}", "{a}*{b}", "{a} times {b} equals"],
    "div": ["{a} / {b}", "{a} divided by {b}", "{a}/{b}", "{a} divided by {b} equals"],
    "mod": ["{a} % {b}", "{a} mod {b}", "{a}%{b}", "{a} mod {b} equals"],
    "max": ["max({a}, {b})", "the larger of {a} and {b}", "max {a} {b}", "larger of {a} and {b}"],
}
S2 = {   # the same requests in number words
    "add": ["{aw} plus {bw}", "{aw} plus {bw} equals", "what is {aw} plus {bw}", "tell me {aw} plus {bw}"],
    "sub": ["{aw} minus {bw}", "{aw} minus {bw} equals", "what is {aw} minus {bw}", "tell me {aw} minus {bw}"],
    "mul": ["{aw} times {bw}", "{aw} times {bw} equals", "what is {aw} times {bw}", "tell me {aw} times {bw}"],
    "div": ["{aw} divided by {bw}", "{aw} divided by {bw} equals", "what is {aw} divided by {bw}", "tell me {aw} divided by {bw}"],
    "mod": ["{aw} mod {bw}", "{aw} mod {bw} equals", "what is {aw} mod {bw}", "tell me {aw} mod {bw}"],
    "max": ["the larger of {aw} and {bw}", "max of {aw} and {bw}", "which is larger, {aw} or {bw}", "larger of {aw} and {bw}"],
}
S3 = {   # verbal phrasing, digits (some phrasings name the operands in the other order)
    "add": ["the sum of {a} and {b}", "{a} and {b} added together", "what you get when you add {b} to {a}", "add {b} to {a}"],
    "sub": ["{b} subtracted from {a}", "{a} take away {b}", "what you get when you subtract {b} from {a}", "subtract {b} from {a}"],
    "mul": ["the product of {a} and {b}", "{a} multiplied by {b}", "what you get when you multiply {a} and {b}", "multiply {a} by {b}"],
    "div": ["{a} divided by {b}, rounded down", "how many times {b} goes into {a}", "the quotient of {a} and {b}", "divide {a} by {b}"],
    "mod": ["the remainder when {a} is divided by {b}", "what is left over after dividing {a} by {b}", "{a} modulo {b}", "the remainder of {a} divided by {b}"],
    "max": ["the larger of {a} or {b}", "the bigger number out of {a} and {b}", "whichever is greater, {a} or {b}", "the maximum of {a} and {b}"],
}
S4 = {   # short stories
    "add": ["I had {a} {things} and got {b} more. How many do I have now?", "There are {a} {things} in one bag and {b} in another. How many {things} altogether?",
            "Sam collected {a} {things} on Monday and {b} on Tuesday. How many in total?"],
    "sub": ["I had {a} {things} and gave away {b}. How many are left?", "There were {a} {things} and {b} got lost. How many remain?",
            "Sam had {a} {things} and used {b} of them. How many does he have now?"],
    "mul": ["There are {a} boxes with {b} {things} in each. How many {things} in all?", "Each of {a} children gets {b} {things}. How many {things} are given out?",
            "A shelf holds {b} {things} and there are {a} shelves. How many {things} fit in all?"],
    "div": ["I share {a} {things} equally among {b} friends. How many does each friend get?", "{a} {things} are packed {b} to a box. How many full boxes are there?",
            "Split {a} {things} into {b} equal groups. How many are in each group?"],
    "mod": ["I share {a} {things} equally among {b} friends and give out as many as possible. How many are left over?",
            "{a} {things} are packed {b} to a box. How many are left that do not fill a box?", "Split {a} {things} into groups of {b}. How many are left over?"],
    "max": ["Ann has {a} {things} and Ben has {b}. What is the largest number either of them has?", "One jar holds {a} {things} and another holds {b}. How many are in the fuller jar?",
            "Team A scored {a} and team B scored {b}. What is the higher score?"],
}
S5 = {   # commutative skills with the operands shown in the other order
    "add": ["{b} and {a} added together", "the sum of {b} and {a}", "{b} plus {a}"],
    "mul": ["{b} and {a} multiplied together", "the product of {b} and {a}", "{b} times {a}"],
    "max": ["the bigger of {b} and {a}", "of {b} and {a}, the greater one", "max of {b} and {a}"],
}

CORES = {fam: tuple(c for skill, texts in table.items() for c in _cores(fam, skill, texts, swap=(fam == "S5")))
         for fam, table in (("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4), ("S5", S5))}

FRAMES = {
    "S1": _frames("S1", ["{core}?", "Calculate: {core}", "Q: {core}", "Quick one: {core}", "First: {core}."]),
    "S2": _frames("S2", ["{core}?", "Read this out: {core}", "Q: {core} ?", "Say it in words: {core}", "Spoken form: {core}."]),
    "S3": _frames("S3", ["Question: {core}", "{core}?", "Hi! {core} please", "Help me with this: {core}", "Quick one, {core}"]),
    "S4": _frames("S4", ["{core} Answer with a number.", "Puzzle: {core}", "Story problem. {core}", "{core} Thanks!", "Here is one for you. {core}"]),
    "S5": _frames("S5", ["{core}?", "Work out {core}.", "Q: {core}", "Hmm, {core}?", "Tell me {core}."]),
    "S6": _frames("S6", ["{pre} {core}?", "{core}? {post}", "{pre} {core}? {post}", "{pre} {core}. {post}", "{pre} Here it is: {core}", "{core} {post}"]),
    "S7": _frames("S7", ["{core}?", "Q: {core}", "{core} ??", "{core} please", "Quick: {core}"], noises=["typo", "typo", "case", "space", "mix"]),
    "S8": _frames("S8", ["Could you kindly work out the following: {core}", "I would be grateful if you could answer: {core}", "hey, quick q: {core}", "yo, {core}?", "{core} ?"]),
}

POOL = Pool(
    name="dev",
    frames=FRAMES,
    cores=CORES,
    core_sources={"S6": ("S1", "S2", "S3"), "S7": ("S1", "S3", "S4"), "S8": ("S1", "S3")},
    chatter=("Hope your day is going well.", "Thanks for helping me out.", "I was just thinking about the weekend.", "The weather has been strange lately.",
             "Sorry to bother you again.", "My friend asked me about this earlier.", "This is for a little project of mine.", "Let me know if anything is unclear.",
             "I have been meaning to ask this for a while.", "Anyway, here is my question.", "It has been a long week.", "I will explain why I need it later."),
    things=("apples", "marbles", "stickers", "coins", "books"),
    traps=("Add me to the guest list for the {n}th of July, please.", "My plus one is arriving at about {n} in the evening.", "He has been a mod on that forum for {n} years now.",
           "The max volume on this speaker is too loud for the {n} am class.", "The remainder of the {n} pizzas went to the neighbours.",
           "We watched that film three times and it earned a {n} out of ten.", "Please double check the {n}th page of the report before sending.",
           "She won the product design prize in {n} different cities.", "The total silence in room {n} was almost spooky.",
           "A multiplied sense of dread followed me on bus {n}."),
)

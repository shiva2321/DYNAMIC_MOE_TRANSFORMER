"""The CONFIRMATORY template pool (single use). Hand-written, with no model run on it, and sharing no frame, core or trap with ``pool_dev``.

The module refuses to be imported unless KS_ALLOW_CONFIRM is "1" (``pools.load_pool("confirm")`` checks the same flag), so development code cannot touch it by accident. The freeze
records the sha256 of this file; after the freeze a change to it is a change to the pre-registration.

One deliberate difference in KIND from the dev pool: the S6 chatter sentences are drawn from the same corpora as the "Other content" the machine is taught (banking77 and AG News),
as spec 5.2 asks. The dev pool used a short hand-written list, which is a defect of the dev test bed found after the dev numbers were seen (PROGRESS.md, 2026-10-28); the fix is made
here, before the freeze, and the dev numbers on chatter are therefore not comparable with the confirmatory ones.
"""

import os
import re
import sys

if os.environ.get("KS_ALLOW_CONFIRM") != "1":
    raise PermissionError("the confirmatory pool is single-use: set KS_ALLOW_CONFIRM=1 only for the pre-registered confirmatory run")

_ENTRY = os.path.basename(getattr(sys.modules.get("__main__"), "__file__", "") or "")
if re.match(r"(run_|lab_|diag_|tune_|layer_scan)", _ENTRY) and _ENTRY != "run_confirm.py":
    raise PermissionError(f"{_ENTRY} is a development entry point and writes into the dev result folders; the confirmatory pool is run only through run_confirm.py")

from pools import Core, Frame, Pool


def _cores(family, skill, texts, swap=False):
    return tuple(Core(f"{family}.{skill}.c{i}", skill, t, swap) for i, t in enumerate(texts))


def _frames(family, texts, noises=None):
    noises = noises or [None] * len(texts)
    return tuple(Frame(f"{family}.f{i}", t, n) for i, (t, n) in enumerate(zip(texts, noises)))


S1 = {   # symbols and infix words, digits
    "add": ["( {a} + {b} )", "{a} + {b} = ?", "{a} plus {b} gives", "compute {a}+{b}"],
    "sub": ["( {a} - {b} )", "{a} - {b} = ?", "{a} minus {b} gives", "compute {a}-{b}"],
    "mul": ["( {a} * {b} )", "{a} * {b} = ?", "{a} times {b} gives", "compute {a}*{b}"],
    "div": ["( {a} / {b} )", "{a} / {b} = ?", "{a} divided by {b} gives", "compute {a}/{b}"],
    "mod": ["( {a} % {b} )", "{a} % {b} = ?", "{a} mod {b} gives", "compute {a}%{b}"],
    "max": ["max of ( {a} , {b} )", "max [ {a} , {b} ] = ?", "maximum value ( {a} , {b} )", "pick the max: {a} {b}"],
}
S2 = {   # the same requests in number words
    "add": ["how much is {aw} plus {bw}", "( {aw} plus {bw} )", "what do I get from {aw} plus {bw}", "{aw} plus {bw}, written out"],
    "sub": ["how much is {aw} minus {bw}", "( {aw} minus {bw} )", "what do I get from {aw} minus {bw}", "{aw} minus {bw}, written out"],
    "mul": ["how much is {aw} times {bw}", "( {aw} times {bw} )", "what do I get from {aw} times {bw}", "{aw} times {bw}, written out"],
    "div": ["how much is {aw} divided by {bw}", "( {aw} divided by {bw} )", "what do I get from {aw} divided by {bw}", "{aw} divided by {bw}, written out"],
    "mod": ["how much is {aw} mod {bw}", "( {aw} mod {bw} )", "what do I get from {aw} mod {bw}", "{aw} mod {bw}, written out"],
    "max": ["{aw} or {bw}: which one is bigger", "maximum among {aw} and {bw}", "( the bigger one, {aw} or {bw} )", "pick the max of {aw} and {bw}"],
}
S3 = {   # verbal phrasing, digits (some phrasings name the operands in the other order)
    "add": ["{a} increased by {b}", "total of {a} and {b}", "how far do you get if you put {b} on top of {a}", "start at {a} and go up by {b}"],
    "sub": ["{a} decreased by {b}", "from {a} remove {b}", "how much is left if you take {b} out of {a}", "start at {a} and go down by {b}"],
    "mul": ["{a} copies of {b}, all counted", "{a} lots of {b}", "{b} taken {a} times over", "{a} scaled up by a factor of {b}"],
    "div": ["how many whole {b}s fit inside {a}", "{a} split into {b} even portions, rounded down", "the whole-number result of {a} over {b}", "size of one share when {a} is cut {b} ways, whole units only"],
    "mod": ["what remains of {a} after taking out as many whole {b}s as possible", "{a} leftover after cutting it into pieces of {b}", "the part of {a} that cannot fill another {b}",
            "rest of {a} once every whole {b} is removed"],
    "max": ["the top value among {a} and {b}", "{a} versus {b}: take the bigger", "the greater one, {a} or {b}", "pick the biggest of {a} and {b}"],
}
S4 = {   # short stories
    "add": ["A baker made {a} {things} before noon and {b} more after lunch. What is the day's count?", "Tom's left pocket has {a} {things} and his right pocket has {b}. How many is he carrying?",
            "Mia found {a} {things} at the park and then {b} more at home. How many does she own now?"],
    "sub": ["A baker made {a} {things} and sold {b} of them by noon. How many are still on the tray?", "Tom started with {a} {things} but dropped {b} along the way. What does he have left?",
            "The box held {a} {things} until {b} were taken out. How many stay in the box?"],
    "mul": ["A baker fills {a} trays with {b} {things} on each tray. What is the count of {things} baked?", "Each of {a} pockets holds {b} {things}. How many {things} can all the pockets carry?",
            "There are {a} rows in the garden with {b} {things} per row. What is the number of {things} in the garden?"],
    "div": ["A baker packs {a} {things} into bags of {b}. How many full bags does the baker get?", "Tom hands out {a} {things} evenly to {b} neighbours. How many does each neighbour take home?",
            "The box of {a} {things} is shared fairly by {b} teams. What does each team's share come to?"],
    "mod": ["A baker packs {a} {things} into bags of {b}. How many {things} cannot fill a bag and stay on the counter?",
            "Tom hands out {a} {things} evenly to {b} neighbours and each takes as many as possible. How many does Tom still hold?",
            "{a} {things} go into crates holding {b} each. How many {things} sit outside the crates?"],
    "max": ["Mia counted {a} {things} and Raj counted {b}. Give the bigger of the two counts.", "The first basket holds {a} {things} and the second holds {b}. How many does the fuller basket hold?",
            "A club had {a} visitors in winter and {b} in summer. What is its busiest season's count?"],
}
S5 = {   # commutative skills with the operands shown in the other order
    "add": ["{b} with {a} included, in total", "start from {b} and add {a}", "{b} + {a} , totalled"],
    "mul": ["{b} copies of {a}, counted up", "{b} * {a} , multiplied out", "{b} multiplied up by {a}"],
    "max": ["the top one of {b} and {a}", "{b} against {a}, bigger wins", "the top of {b} and {a}"],
}

CORES = {fam: tuple(c for skill, texts in table.items() for c in _cores(fam, skill, texts, swap=(fam == "S5")))
         for fam, table in (("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4), ("S5", S5))}

FRAMES = {
    "S1": _frames("S1", ["Evaluate {core}", "Compute for me {core}", "Task: {core}", "Please solve {core}", "Result of {core}"]),
    "S2": _frames("S2", ["Say it aloud: {core}", "Spoken, as words: {core}", "Out loud: {core}.", "Words only: {core}", "Dictated: {core}"]),
    "S3": _frames("S3", ["Problem: {core}", "Hello, would you tell me {core}", "Could you answer: {core}", "Need a hand with: {core}", "One thing I wonder: {core}"]),
    "S4": _frames("S4", ["Little riddle for you. {core}", "{core} A number is all I need.", "From a workbook: {core}", "Imagine this. {core}", "{core} Show only the final count."]),
    "S5": _frames("S5", ["Find {core}", "How about {core}", "Look at {core}", "Reckon {core}", "I wonder about {core}"]),
    "S6": _frames("S6", ["{pre}\n{core}", "{core}\n{post}", "{pre}\n{core}\n{post}", "{pre}\n\n{core}", "{core}\n\n{post}", "{pre}\n\n{core}\n\n{post}"]),   # only whitespace outside the slots: the parts scored as Other are exactly the corpus sentences
    "S7": _frames("S7", ["{core} when you can", "Okay: {core}", "{core} thank you", "Hmm {core}", "Small request: {core}"], noises=["typo", "typo", "case", "space", "mix"]),
    "S8": _frames("S8", ["Dear assistant, may I request the result of: {core}", "Respectfully, I ask you to evaluate: {core}", "sup, can u do {core}", "lol {core}", "{core} thx"]),
}

# 48 sentences drawn with corpora.load_other_sentences(48, 0xC0F1) (banking77 and AG News, the source of the taught "Other content"), written out here so that the sha256 of this
# file fixes them; tests/knowledge_status/test_ks_freeze.py checks that this list is exactly that draw.
CHATTER = (
    'chain store sales rose in the latest week, but business continued to fluctuate from the effects of Hurricane Charley and high gasoline prices, a report said on Tuesday.',
    'Microsoft to Launch New Small Business Software Microsoft Corp.',
    'The search engines like Google or Yahoo make money by indexing the web sites all over',
    "Six Killed in Afghan Jail Break Two Pakistani prisoners were holed up in Kabul's main prison Friday after a bloody breakout attempt that left at least six dead and eight wounded, officials said.",
    'said deals with Visto Corp.',
    'Terror Informant Ignites Self Near White House A Falls Church man who worked as a federal informant set himself on fire hours after announcing his suicide attempt and citing his growing despondency over how the FBI managed his case.',
    'I think my card payment has been reverted',
    'But could global warming thaw and dry the bogs, turning them into greenhouse gas faucets?',
    "Im trying to top up my card but it's still pending and it's been an hour.",
    'Hurricane Ivan Damages Parts of Barbados ST.',
    'Over Before It Started No.',
    'a national team spokesman said...',
    'Do you offer SWIFT transfers?',
    'Britain to cut troop levels in Iraq The British Army is to start pulling troops out of Iraq next month despite the deteriorating security situation in much of the country, The Observer has learnt.',
    'I have a transfer that was declined.',
    "Bush Aides Laugh Off 'Wired Debate' Rumor CHANHASSEN, Minn.",
    'US Geological Survey scientists said they did not believe a major eruption was',
    'Unlikely duo advances to Pilot Pen championship No.',
    'Dennis Hastert last week enunciated a policy in which Congress will pass bills only if most House Republicans back them, regardless of how many Democrats favor them.',
    'Dorfmeister leads Austrian charge LAKE LOUISE, Alta.',
    'Peter Harris, a former campaign manager ...',
    'New chip designs help AMD boost market share Design engineer Alan Lear has rolled up his sleeves.',
    'United was throwing inexperienced players into the starting lineup and calling up minor leaguers to fill the roster.',
    'Johnson backs Wilkinson to succeed MARTIN JOHNSON believes Jonny Wilkinson will get the support he needs to succeed as England rugby captain.',
    'Spitzer Probing Record Biz Promoters pay radio stations annual fees for advance copies of playlists and bill record companies for each new song played, running up a tab that costs the industry tens of millions of dollars annually.',
    'Knew of Venezuela Coup The U.S.',
    'is betting its survival on low prices, strong inventory and good customer service.',
    'Now she is coming back for more.',
    'Blunkett concern on loyalty cards More checks on the use of information collected by supermarket loyalty cards needed says David Blunkett.',
    'in North American financing operations as part of its business rehabilitation efforts.',
    'Canceled checks, float soon to disappear Every night, millions of cashed checks fly around the country, headed for their home bank.',
    'Is there a tracking number for the card?',
    'how to get virtual card',
    'Barghouthi Drops Palestinian Election Bid .',
    "I've transferred money before and it worked without problem",
    'Top Colombia militant gunned down Colombian paramilitary leader Miguel Arroyave is killed, casting further doubts on ongoing peace talks.',
    'I must quickly transfer something from China, and I need to know how long it will take.',
    'Nvidia Strikes Processor Deal with Intel Analysts say the agreement was a boost to both companies since it will pave the way for Nvidia get access to millions more desktop computers built on Intel processors and because Intel will',
    'This recipe is neither particularly original, nor particularly authentic.',
    'This could be a concern for astronauts taking long trips beyond Earth orbit.',
    'Chinese Riot Police Heading for Haiti China has sent a contingent of riot police to Haiti to assist United Nations peacekeepers in the troubled Caribbean nation.',
    'His comments sent the currency plunging to a record low against the euro.',
    'Can you please guide me on how to go about it?',
    'NASCAR Wants TV Ratings It is a new format, but it still encompasses all the old problems.',
    'The Finns had pulled their goaltender.',
    "Williams backs the idea of considering private investors to reduce the city's cost of building a ballpark for the Expos.",
    'Tiger Woods comes in as the most popular player in golf, and Adam Scott is the third man in the field because he won The Players Championship in March.',
    'When can I expect it?',
)

POOL = Pool(
    name="confirm",
    frames=FRAMES,
    cores=CORES,
    core_sources={"S6": ("S1", "S2", "S3"), "S7": ("S1", "S3", "S4"), "S8": ("S1", "S3")},
    chatter=CHATTER,
    things=("pencils", "cookies", "tickets", "shells", "buttons"),
    traps=("Please add a note to page {n} of the manual.", "The minus sign on key {n} of my keyboard is stuck.", "Our team set its best times in lap {n} of the relay.",
           "I need the remainder of the {n} sandwiches wrapped up for tomorrow.", "That mod for the game has been downloaded {n} thousand times.",
           "He divided his attention between the {n} screens on his desk.", "She reached the max level in round {n} of the quiz show.",
           "A product manager from office {n} sent a summary of the feedback.", "What a difference {n} days made to the garden.",
           "The total weight of suitcase {n} surprised the airline staff.", "Multiply my thanks by {n} for helping with the move.",
           "The larger of the two dogs weighs about {n} kilos."),
)

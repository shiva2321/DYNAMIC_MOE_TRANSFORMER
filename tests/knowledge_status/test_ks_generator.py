import re
from collections import Counter

import numpy as np
import pytest

import generator as G
import pools
import skills

FAMILIES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")

# ----------------------------------------------------------------------------------------------------------- an independent reader of S1 and S2 text
_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_CUES = (("divided by", "div"), ("larger", "max"), ("max", "max"), ("mod", "mod"), ("plus", "add"), ("minus", "sub"), ("times", "mul"),
         ("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div"), ("%", "mod"))


def read_numbers_from_words(text: str):
    toks = re.sub(r"[^a-z\- ]", " ", text.lower()).replace("-", " ").split()
    out, cur, total, active = [], 0, 0, False
    for t in toks:
        if t in _ONES or t in _TENS:
            cur += _ONES.get(t, _TENS.get(t, 0))
            active = True
        elif t == "hundred" and active:
            cur *= 100
        elif t == "thousand" and active:
            total += cur * 1000
            cur = 0
        else:
            if active:
                out.append(total + cur)
            cur, total, active = 0, 0, False
    if active:
        out.append(total + cur)
    return out


def read_skill(text: str) -> str:
    low = text.lower()
    for cue, skill in _CUES:
        if cue in low:
            return skill
    raise AssertionError(f"no cue in {text!r}")


def request_text(msg: G.Message) -> str:
    part = next(p for p in msg.parts if p.kind == "skill")
    return msg.text[part.start:part.end]


# ---------------------------------------------------------------------------------------------------------------------------------- tests
def test_number_to_words_known_values():
    assert G.number_to_words(0) == "zero"
    assert G.number_to_words(7) == "seven"
    assert G.number_to_words(13) == "thirteen"
    assert G.number_to_words(21) == "twenty-one"
    assert G.number_to_words(100) == "one hundred"
    assert G.number_to_words(1234) == "one thousand two hundred thirty-four"
    assert G.number_to_words(-8) == "minus eight"


def test_every_family_renders_every_skill():
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        for skill in skills.SKILLS:
            if fam == "S5" and skill not in skills.COMMUTATIVE:
                with pytest.raises(ValueError):
                    G.make_single(rng, "dev", skill, fam)
                continue
            msg = G.make_single(rng, "dev", skill, fam)
            assert msg.family == fam and msg.pool == "dev"
            skill_parts = [p for p in msg.parts if p.kind == "skill"]
            assert len(skill_parts) == 1 and skill_parts[0].skill == skill
            p = skill_parts[0]
            assert 0 <= p.start < p.end <= len(msg.text)
            assert p.answer == skills.truth(skill, p.a, p.b)


@pytest.mark.parametrize("digits", [(1, 2), (1, 4)])
def test_labels_are_exact(digits):
    rng = np.random.default_rng(1)
    for fam in ("S1", "S2"):
        for skill in skills.SKILLS:
            for _ in range(60):
                msg = G.make_single(rng, "dev", skill, fam, digits=digits)
                p = next(q for q in msg.parts if q.kind == "skill")
                text = request_text(msg)
                nums = [int(x) for x in re.findall(r"\d+", text)] if fam == "S1" else read_numbers_from_words(text)
                assert nums == [p.a, p.b], (fam, skill, text, nums, p)
                assert read_skill(text) == skill, (fam, skill, text)
                assert p.answer == skills.truth(skill, p.a, p.b)


def test_commutative_reorder_keeps_the_label():
    rng = np.random.default_rng(2)
    shown_b_first = 0
    for skill in sorted(skills.COMMUTATIVE):
        for _ in range(80):
            msg = G.make_single(rng, "dev", skill, "S5", digits=(2, 2))
            p = next(q for q in msg.parts if q.kind == "skill")
            nums = [int(x) for x in re.findall(r"\d+", request_text(msg))]
            assert sorted(nums) == sorted([p.a, p.b])
            assert p.answer == skills.truth(skill, p.a, p.b) == skills.truth(skill, p.b, p.a)
            shown_b_first += int(p.a != p.b and nums == [p.b, p.a])
    assert shown_b_first > 100                                                  # the reordering really happens


def test_no_zero_divisor_for_div_and_mod():
    rng = np.random.default_rng(3)
    for skill in ("div", "mod"):
        for fam in FAMILIES:
            if fam == "S5":
                continue                                                          # S5 exists only for commutative skills (tested above)
            for _ in range(70):
                msg = G.make_single(rng, "dev", skill, fam)
                assert next(q for q in msg.parts if q.kind == "skill").b != 0


def test_zero_and_one_are_drawn_and_rendered():
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(1500):
        msg = G.make_single(rng, "dev", "add", "S2", digits=(1, 1))
        p = next(q for q in msg.parts if q.kind == "skill")
        seen |= {p.a, p.b}
        assert read_numbers_from_words(request_text(msg)) == [p.a, p.b]
    assert 0 in seen and 1 in seen


def test_negative_subtraction_renders_and_labels():
    rng = np.random.default_rng(5)
    found = False
    for _ in range(400):
        msg = G.make_single(rng, "dev", "sub", "S2", digits=(1, 2))
        p = next(q for q in msg.parts if q.kind == "skill")
        if p.a < p.b:
            assert p.answer == p.a - p.b < 0
            assert read_numbers_from_words(request_text(msg)) == [p.a, p.b]
            found = True
    assert found
    assert G.number_to_words(17 - 25) == "minus eight"


def test_same_seed_same_messages():
    a = [G.make_single(np.random.default_rng(9), "dev", s, f).text for s in skills.SKILLS for f in ("S1", "S4", "S6", "S7")]
    b = [G.make_single(np.random.default_rng(9), "dev", s, f).text for s in skills.SKILLS for f in ("S1", "S4", "S6", "S7")]
    assert a == b


def test_template_id_alone_does_not_predict_the_skill():
    rng = np.random.default_rng(6)
    rows = []
    for _ in range(3000):
        skill = skills.SKILLS[int(rng.integers(0, 6))]
        fam = FAMILIES[int(rng.integers(0, 8))]
        if fam == "S5" and skill not in skills.COMMUTATIVE:
            continue
        msg = G.make_single(rng, "dev", skill, fam)
        rows.append((msg.template_ids[0], skill))                                 # the wrapper frame id, which is opaque and shared across skills
    half = len(rows) // 2
    by_frame = {}
    for f, s in rows[:half]:
        by_frame.setdefault(f, Counter())[s] += 1
    right = sum(1 for f, s in rows[half:] if f in by_frame and by_frame[f].most_common(1)[0][0] == s)
    n = sum(1 for f, _ in rows[half:] if f in by_frame)
    assert right / n <= 1 / 6 + 0.06, right / n
    skills_per_frame = {}
    for f, s in rows:
        skills_per_frame.setdefault(f, set()).add(s)
    assert all(len(v) >= 3 for v in skills_per_frame.values())                    # every frame is used with many skills


def test_dev_pool_templates_are_unique_and_nonempty_per_family():
    pool = pools.load_pool("dev")
    ids, texts = set(), set()
    for fam in FAMILIES:
        assert len(pool.frames[fam]) >= 4
        own = set()
        for fr in pool.frames[fam]:
            assert fr.frame_id not in ids and fr.text not in own                  # frame ids are unique everywhere, frame texts within their family
            assert "{core}" in fr.text and fr.text.strip() != "{core}", "a bare {core} frame would be shared by every pool"
            ids.add(fr.frame_id)
            own.add(fr.text)
            assert not re.search(r"\d", fr.text), "a frame must not contain digits"
    for fam in ("S1", "S2", "S3", "S4"):
        for skill in skills.SKILLS:
            assert len([c for c in pool.cores[fam] if c.skill == skill]) >= 3, (fam, skill)
    for skill in skills.COMMUTATIVE:
        assert len([c for c in pool.cores["S5"] if c.skill == skill]) >= 3
    assert {c.skill for c in pool.cores["S5"]} == set(skills.COMMUTATIVE)
    for fam, cs in pool.cores.items():
        for c in cs:
            assert c.core_id not in ids and c.text not in texts
            ids.add(c.core_id)
            texts.add(c.text)


def test_request_span_cuts_the_text_exactly_and_chatter_is_other():
    rng = np.random.default_rng(7)
    for fam in FAMILIES:
        for _ in range(40):
            msg = G.make_single(rng, "dev", "mul", fam)
            parts = sorted(msg.parts, key=lambda q: q.start)
            for x, y in zip(parts, parts[1:]):
                assert x.end <= y.start                                          # spans never overlap
            req = next(q for q in parts if q.kind == "skill")
            assert msg.text[req.start:req.end].strip() == msg.text[req.start:req.end]
            for q in parts:
                if q.kind != "skill":
                    assert q.kind == "other" and q.skill is None
            if fam == "S6":
                assert any(q.kind == "other" for q in parts)


def test_noise_functions_change_the_text_but_keep_the_digits():
    sample = "what is the remainder when 17 is divided by 5, please answer quickly"
    for kind in ("typo", "case", "space", "mix"):
        rng = np.random.default_rng(10)
        out = G.apply_noise(sample, kind, rng)
        assert out != sample, kind
        assert re.findall(r"\d+", out) == ["17", "5"], (kind, out)
    assert "  " in G.apply_noise(sample, "space", np.random.default_rng(10))
    assert G.apply_noise(sample, "case", np.random.default_rng(10)).lower() == sample
    assert sorted(G.apply_noise(sample, "typo", np.random.default_rng(10))) == sorted(sample)      # a typo only reorders letters
    assert G.apply_noise("a b", "typo", np.random.default_rng(10)) == "a b"                         # short words are left alone


def test_s7_messages_keep_their_numbers_and_use_every_noise_kind():
    pool = pools.load_pool("dev")
    kinds = {fr.noise for fr in pool.frames["S7"]}
    assert kinds == {"typo", "case", "space", "mix"}
    rng = np.random.default_rng(8)
    seen = set()
    for _ in range(300):
        skill = skills.SKILLS[int(rng.integers(0, 6))]
        msg = G.make_single(rng, "dev", skill, "S7")
        p = next(q for q in msg.parts if q.kind == "skill")
        assert sorted(re.findall(r"\d+", request_text(msg))) == sorted([str(p.a), str(p.b)])
        seen.add(next(fr.noise for fr in pool.frames["S7"] if fr.frame_id == msg.template_ids[0]))
    assert seen == kinds


def test_dev_chatter_has_no_digits_or_operator_cues():
    pool = pools.load_pool("dev")
    assert len(pool.chatter) >= 10
    for line in pool.chatter:
        assert not re.search(r"\d", line)
        words = set(re.findall(r"[a-z]+", line.lower()))
        assert not (words & pools.ARITH_CUES), (line, words & pools.ARITH_CUES)


def test_confirm_pool_is_blocked_without_the_flag(monkeypatch):
    monkeypatch.delenv("KS_ALLOW_CONFIRM", raising=False)
    with pytest.raises(PermissionError):
        pools.load_pool("confirm")
    with pytest.raises(KeyError):
        pools.load_pool("nonsense")

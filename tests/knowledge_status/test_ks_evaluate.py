"""Task 11, scoring: status scores, the novelty-only scores used for the baselines, the skill confusion, the tangled-message logic (B4a, B4b, B6), the wording claim and the labelled
consequence estimate. Everything here is hand-made data with numbers worked out on paper."""

import math

import numpy as np
import pytest

import evaluate as EV
import generator as G
import segment as SG
import skills
import status as ST

NAN = float("nan")


def A(status, skill=None, evidence=0, competence=NAN, nearest=""):
    return ST.Assessment(status, skill, 0.5, evidence, competence, 1, nearest)


def C(status, family="S4", skill="add", kind="arith"):
    return EV.Case(status, family, skill, kind)


def hand_made():
    """Arithmetic: 8 solid truths (S4 x4, S7 x4), 4 thin (S1), 8 unseen (S5), 3 grey. Plus other, trap and never-taught content."""
    cases, preds = [], []

    def add(case, pred, n=1):
        cases.extend([case] * n)
        preds.extend([pred] * n)

    add(C("solid", "S4"), A("solid", "add", 100, 0.95), 3); add(C("solid", "S4"), A("thin", "add", 5, 0.5)); add(C("solid", "S7"), A("solid", "add", 100, 0.95), 3)
    add(C("solid", "S7"), A("unseen", None, 0, NAN))
    add(C("thin", "S1", "mod"), A("thin", "mod", 5, 0.5), 3); add(C("thin", "S1", "mod"), A("solid", "mod", 100, 0.95))
    add(C("unseen", "S5", "mul"), A("unseen"), 6); add(C("unseen", "S5", "mul"), A("solid", "add", 100, 0.95)); add(C("unseen", "S5", "mul"), A("other"))
    add(C("grey", "S4"), A("unseen"), 3)
    add(C("other", "other", None, "other"), A("other"), 9); add(C("other", "other", None, "other"), A("unseen"))
    add(C("other", "trap", None, "trap"), A("other"), 5)
    add(C("unseen", "unseen", None, "unseen_content"), A("unseen"), 3); add(C("unseen", "unseen", None, "unseen_content"), A("other"))
    return preds, cases


def test_status_scores_on_a_hand_made_confusion():
    preds, cases = hand_made()
    r = EV.status_scores(preds, cases)
    # solid: tp 6, fn 2, fp 2 (a thin and an unseen truth called solid) -> 12/16; thin: tp 3, fn 1, fp 1 -> 6/8; unseen: tp 6, fn 2 (a solid and an Other call), fp 1 -> 12/15
    assert r["macro_f1"]["per_class"] == {"solid": pytest.approx(0.75), "thin": pytest.approx(0.75), "unseen": pytest.approx(0.8)}
    assert r["macro_f1"]["macro_f1"] == pytest.approx((0.75 + 0.75 + 0.8) / 3) and r["macro_f1"]["absent"] == []
    assert r["confusion"]["solid"] == {"solid": 6, "thin": 1, "unseen": 1, "other": 0}
    assert r["confusion"]["unseen"] == {"solid": 1, "thin": 0, "unseen": 6, "other": 1}
    assert r["n_arith"] == 20 and r["n_grey"] == 3
    # one of the 8 solid truths on a held-out wording was called unseen; none of the 4 thin truths (taught wording) was
    assert r["false_unseen_heldout"] == pytest.approx(1 / 8) and r["false_unseen_taught_wording"] == pytest.approx(0.0)
    assert r["false_unseen_by_family"] == {"S4": pytest.approx(0.0), "S7": pytest.approx(0.25)}
    assert r["unseen_recall"] == pytest.approx(6 / 8)
    assert r["other_as_unseen"] == pytest.approx(0.1) and r["trap_as_unseen"] == pytest.approx(0.0)
    assert r["other_labelled"] == pytest.approx(0.9) and r["trap_labelled_other"] == pytest.approx(1.0)
    assert r["unseen_content_flagged"] == pytest.approx(0.75)


def test_grey_cases_are_reported_and_never_scored():
    preds, cases = hand_made()
    with_grey = EV.status_scores(preds, cases)
    kept = [(p, c) for p, c in zip(preds, cases) if c.status != "grey"]
    without = EV.status_scores([p for p, _ in kept], [c for _, c in kept])
    assert with_grey["macro_f1"] == without["macro_f1"] and with_grey["false_unseen_heldout"] == without["false_unseen_heldout"]
    assert with_grey["n_grey"] == 3 and without["n_grey"] == 0


def test_a_missing_status_is_not_scored_as_perfect():
    r = EV.status_scores([A("solid", "add", 100, 0.95)] * 4, [C("solid")] * 4)
    assert r["macro_f1"]["absent"] == ["thin", "unseen"] and r["macro_f1"]["macro_f1"] == pytest.approx(1.0)
    assert math.isnan(r["unseen_recall"]) and math.isnan(r["other_as_unseen"]) and math.isnan(r["unseen_content_flagged"])


def test_thin_is_separated_from_solid_by_evidence_and_competence():
    preds, cases = hand_made()
    r = EV.status_scores(preds, cases)
    # among covered parts of solid or thin truth: the scores log1p(evidence) * competence of the 6 solid truths called solid and 1 called thin, against the 3+1 thin truths
    assert 0.5 < r["thin_vs_solid_auroc"] < 1.0
    perfect = EV.status_scores([A("solid", "add", 100, 0.95), A("thin", "mod", 5, 0.5)] * 3, [C("solid"), C("thin", "S1", "mod")] * 3)
    assert perfect["thin_vs_solid_auroc"] == pytest.approx(1.0)


def test_both_evidence_and_competence_move_the_thin_vs_solid_score():
    by_competence = EV.status_scores([A("solid", "add", 100, 0.95), A("thin", "mod", 100, 0.5)] * 3, [C("solid"), C("thin", "S1", "mod")] * 3)
    by_evidence = EV.status_scores([A("solid", "add", 100, 0.95), A("thin", "mod", 5, 0.95)] * 3, [C("solid"), C("thin", "S1", "mod")] * 3)
    neither = EV.status_scores([A("solid", "add", 100, 0.95), A("thin", "mod", 100, 0.95)] * 3, [C("solid"), C("thin", "S1", "mod")] * 3)
    assert by_competence["thin_vs_solid_auroc"] == pytest.approx(1.0) and by_evidence["thin_vs_solid_auroc"] == pytest.approx(1.0)
    assert neither["thin_vs_solid_auroc"] == pytest.approx(0.5)


def test_flag_scores_work_without_a_status_machine():
    preds, cases = hand_made()
    flags = [p.status == "unseen" for p in preds]
    r = EV.flag_scores(flags, cases)
    full = EV.status_scores(preds, cases)
    for key in ("false_unseen_heldout", "false_unseen_taught_wording", "unseen_recall", "other_as_unseen", "trap_as_unseen", "unseen_content_flagged"):
        assert r[key] == pytest.approx(full[key], nan_ok=True)
    # with raw novelty scores: parts of taught skills against parts of untaught skills (arithmetic only, grey left out)
    scores = np.array([2.0 if c.status == "unseen" else 0.0 for c in cases])
    assert EV.flag_scores(flags, cases, scores)["auroc_taught_vs_untaught"] == pytest.approx(1.0)
    assert EV.flag_scores(flags, cases, -scores)["auroc_taught_vs_untaught"] == pytest.approx(0.0)


def test_scores_refuse_what_they_cannot_pair():
    with pytest.raises(ValueError, match="same length"):
        EV.status_scores([A("solid")], [C("solid"), C("thin")])
    with pytest.raises(ValueError, match="truth status"):
        EV.status_scores([A("solid")], [C("related")])
    with pytest.raises(ValueError, match="same length"):
        EV.flag_scores([True], [C("solid"), C("thin")])


def test_skill_confusion_counts_the_nearest_taught_group_per_true_skill():
    preds = [A("solid", "div", nearest="div"), A("solid", "mod", nearest="div"), A("unseen", None, nearest="sub"), A("unseen", None, nearest="other"), A("other", None, nearest="other")]
    cases = [C("solid", skill="div"), C("solid", skill="mod"), C("unseen", skill="mod"), C("unseen", skill="mod"), C("other", "trap", None, "trap")]
    m, rows, cols = EV.skill_confusion(preds, cases)
    assert rows == list(skills.SKILLS) and cols == list(skills.SKILLS) + ["other"]
    at = lambda r, c: int(m[rows.index(r), cols.index(c)])
    assert at("div", "div") == 1 and at("mod", "div") == 1 and at("mod", "sub") == 1 and at("mod", "other") == 1
    assert m.sum() == 4                                                       # the trap is not an arithmetic part


# ------------------------------------------------------------------------------------------------------------------------------ tangled messages
def hand_message(first_skill, second_skill_or_other):
    """"Compute 17 plus 5 please." (23 chars of request) then a long second part, so the two parts differ a lot in length."""
    a = "Compute 17 plus 5 please."
    b = "Here is a much longer second part of this little message."
    kind2 = "other" if second_skill_or_other == "other" else "skill"
    parts = (G.Part(0, len(a), "skill", first_skill, 17, 5, 22), G.Part(len(a) + 1, len(a) + 1 + len(b), kind2, None if kind2 == "other" else second_skill_or_other, 3, 4, 7 if kind2 == "skill" else None))
    return G.Message(a + " " + b, parts, "tangled", ("x", "y"), "dev")


STATES = {"add": "solid", "sub": "solid", "mul": "unseen", "div": "unseen", "mod": "thin", "max": "unseen"}


def by_part(msg, *statuses):
    return [(span, A(s)) for span, s in zip(SG.gold_parts(msg), statuses)]


def test_other_content_in_a_tangled_message_is_not_flagged_as_novel_arithmetic():
    msgs = [hand_message("add", "other") for _ in range(10)]
    good = EV.tangled_scores(msgs, [by_part(m, "solid", "other") for m in msgs], "gold", STATES)
    assert good["b4b"] == {"n": 10, "nothing_flagged": 1.0, "other_labelled": 1.0}
    bad_msgs = [hand_message("add", "other") for _ in range(10)]
    flagged = [by_part(m, "solid", "unseen") if k < 5 else by_part(m, "solid", "other") for k, m in enumerate(bad_msgs)]
    bad = EV.tangled_scores(bad_msgs, flagged, "gold", STATES)
    assert bad["b4b"]["nothing_flagged"] == 0.5 and bad["b4b"]["other_labelled"] == 0.5
    assert good["mode"] == "gold"


def test_a_known_part_and_an_unseen_part_are_told_apart():
    msgs = [hand_message("add", "mul") for _ in range(8)]
    right = EV.tangled_scores(msgs, [by_part(m, "solid", "unseen") for m in msgs], "gold", STATES)
    assert right["b4a"] == {"n": 8, "residual_hit": 1.0, "known_not_flagged": 1.0}
    swapped = EV.tangled_scores(msgs, [by_part(m, "unseen", "solid") for m in msgs], "gold", STATES)
    assert swapped["b4a"] == {"n": 8, "residual_hit": 0.0, "known_not_flagged": 0.0}
    both = EV.tangled_scores(msgs, [by_part(m, "unseen", "unseen") for m in msgs], "gold", STATES)
    assert both["b4a"]["known_not_flagged"] == 0.0 and both["b4a"]["residual_hit"] == 1.0   # the long unseen part holds 57 of the 82 flagged characters: more than half
    none = EV.tangled_scores(msgs, [by_part(m, "solid", "solid") for m in msgs], "gold", STATES)
    assert none["b4a"] == {"n": 8, "residual_hit": 0.0, "known_not_flagged": 1.0}


def test_exactly_half_of_the_unseen_part_flagged_counts_as_found():
    even = G.Message("aaaaaaaaaa bbbbbbbbbb", (G.Part(0, 10, "skill", "add", 1, 2, 3), G.Part(11, 21, "skill", "mul", 1, 2, 2)), "tangled", ("x", "y"), "dev")
    r = EV.tangled_scores([even], [[((0, 10), A("solid")), ((11, 16), A("unseen")), ((16, 21), A("solid"))]], "auto_window", STATES)
    assert r["b4a"]["residual_hit"] == 1.0                              # 5 of 10 characters of the unseen part, and all flagged characters lie in it
    r2 = EV.tangled_scores([even], [[((0, 10), A("solid")), ((11, 15), A("unseen")), ((15, 21), A("solid"))]], "auto_window", STATES)
    assert r2["b4a"]["residual_hit"] == 0.0                             # 4 of 10 is under half


def test_other_part_is_labelled_only_when_every_piece_of_it_is():
    m = hand_message("add", "other")
    gold = SG.gold_parts(m)
    o = gold[1]
    mid = o[0] + 28
    split = [(gold[0], A("solid")), ((o[0], mid), A("other")), ((mid, o[1]), A("unseen"))]
    r = EV.tangled_scores([m], [split], "auto_sentence", STATES)
    assert r["b4b"] == {"n": 1, "nothing_flagged": 0.0, "other_labelled": 0.0}
    clean = [(gold[0], A("solid")), ((o[0], mid), A("other")), ((mid, o[1]), A("other"))]
    assert EV.tangled_scores([m], [clean], "auto_sentence", STATES)["b4b"] == {"n": 1, "nothing_flagged": 1.0, "other_labelled": 1.0}
    assert EV.tangled_scores([m], [[(gold[0], A("solid"))]], "auto_sentence", STATES)["b4b"]["other_labelled"] == 0.0           # no piece belongs to the Other part at all


def test_flagging_the_known_part_alone_fails_nothing_flagged_but_not_other_labelled():
    m = hand_message("add", "other")
    gold = SG.gold_parts(m)
    r = EV.tangled_scores([m], [[(gold[0], A("unseen")), (gold[1], A("other"))]], "gold", STATES)
    assert r["b4b"] == {"n": 1, "nothing_flagged": 0.0, "other_labelled": 1.0}


def test_flagging_both_of_two_equal_parts_is_not_finding_the_residual():
    text = "aaaaaaaaaa bbbbbbbbbb"
    parts = (G.Part(0, 10, "skill", "add", 1, 2, 3), G.Part(11, 21, "skill", "mul", 1, 2, 2))
    m = G.Message(text, parts, "tangled", ("x", "y"), "dev")
    r = EV.tangled_scores([m], [by_part(m, "unseen", "unseen")], "gold", STATES)
    assert r["b4a"] == {"n": 1, "residual_hit": 0.0, "known_not_flagged": 0.0}           # exactly half of the flagged characters are in the unseen part: not more than half


def test_automatic_segments_are_matched_to_gold_parts_by_where_most_of_their_characters_lie():
    m = hand_message("add", "mul")
    gold = SG.gold_parts(m)
    first = gold[0]
    # a window that straddles the separator evenly belongs to neither part; a flagged straddling window is not a flag on the known part
    straddle = (gold[0][1] - 10, gold[0][1] + 10)
    segs = [(first, A("solid")), (straddle, A("unseen")), ((gold[1][0] + 5, gold[1][1]), A("unseen"))]
    r = EV.tangled_scores([m], [segs], "auto_window", STATES)
    assert r["b4a"]["residual_hit"] == 1.0 and r["b4a"]["known_not_flagged"] == 1.0 and r["mode"] == "auto_window"
    only_known_flagged = EV.tangled_scores([m], [[(first, A("unseen")), ((gold[1][0], gold[1][1]), A("solid"))]], "auto_sentence", STATES)
    assert only_known_flagged["b4a"] == {"n": 1, "residual_hit": 0.0, "known_not_flagged": 0.0}


def test_message_kinds_are_told_from_the_gold_states_and_others_are_ignored():
    mixed = [hand_message("add", "mul"), hand_message("add", "other"), hand_message("mul", "other"), hand_message("add", "sub")]
    r = EV.tangled_scores(mixed, [by_part(m, "solid", "unseen") for m in mixed[:1]] + [by_part(m, "solid", "other") for m in mixed[1:2]]
                          + [by_part(m, "unseen", "other") for m in mixed[2:3]] + [by_part(m, "solid", "solid") for m in mixed[3:]], "gold", STATES)
    assert r["b4a"]["n"] == 1 and r["b4b"]["n"] == 1 and r["b4c"] == {"n": 1, "unseen_found": 1.0, "other_labelled": 1.0} and r["n_ignored"] == 1


def test_b6_compares_the_automatic_number_with_the_gold_number():
    gold = {"b4a": {"n": 60, "residual_hit": 0.90, "known_not_flagged": 0.97}}
    near = {"b4a": {"n": 60, "residual_hit": 0.83, "known_not_flagged": 0.95}}
    far = {"b4a": {"n": 60, "residual_hit": 0.60, "known_not_flagged": 0.95}}
    assert EV.b6_gap(gold, near) == {"residual_hit": pytest.approx(0.07), "known_not_flagged": pytest.approx(0.02), "within": True}
    assert EV.b6_gap(gold, far)["within"] is False and EV.b6_gap(gold, far, tol=0.5)["within"] is True


def test_tangled_scores_need_one_list_of_segments_per_message():
    with pytest.raises(ValueError, match="same length"):
        EV.tangled_scores([hand_message("add", "mul")], [], "gold", STATES)


# ------------------------------------------------------------------------------------------------------------------------------ claims and consequence
def test_adapted_condition_is_reported_separately_and_never_gates():
    assert EV.wording_claim(0.04, 0.03) == "holds"
    assert EV.wording_claim(0.12, 0.04) == "needs adaptation to new wording"              # only the adapted condition passes: the claim is downgraded, not passed
    assert EV.wording_claim(0.12, 0.09) == "fails"
    assert EV.wording_claim(0.12, 0.04, bar=0.15) == "holds"
    assert EV.wording_claim(0.05, 0.9) == "holds" and EV.wording_claim(0.0501, 0.05) == "needs adaptation to new wording"     # the bar itself passes
    assert EV.wording_claim(NAN, 0.01) == "undetermined"
    g = EV.gate(0.12, 0.04, bar=0.05)
    assert g["pass"] is False and g["adapted_pass"] is True and g["gated_on"] == "standard" and g["claim"] == "needs adaptation to new wording"
    assert EV.gate(0.04, 0.5, bar=0.05)["pass"] is True                                   # a bad adapted number cannot fail a passing standard one


def test_a_bar_is_fixed_from_dev_values_and_never_made_easier_than_the_draft():
    """The rule declared before the dev run: a bar to stay ABOVE becomes max(draft, mean - 2 sd of the dev seeds); a bar to stay BELOW becomes min(draft, mean + 2 sd)."""
    sd = float(np.std([0.95, 0.97, 0.99], ddof=1))
    assert EV.fix_bar([0.95, 0.97, 0.99], 0.85, "min") == pytest.approx(0.97 - 2 * sd)               # a good dev result tightens the bar
    assert EV.fix_bar([0.60, 0.62, 0.64], 0.85, "min") == pytest.approx(0.85)                       # a poor one cannot lower it
    sd2 = float(np.std([0.01, 0.02, 0.03], ddof=1))
    assert EV.fix_bar([0.01, 0.02, 0.03], 0.05, "max") == pytest.approx(0.02 + 2 * sd2)
    assert EV.fix_bar([0.20, 0.22, 0.24], 0.05, "max") == pytest.approx(0.05)
    assert EV.fix_bar([0.5], 0.85, "min") == pytest.approx(0.85)                                    # one seed has no spread: the draft stands
    with pytest.raises(ValueError, match="direction"):
        EV.fix_bar([0.5, 0.6], 0.85, "sideways")
    with pytest.raises(ValueError, match="no values"):
        EV.fix_bar([float("nan")], 0.85, "min")


def test_offline_consequence_is_labelled_estimate():
    errors = {"false_unseen": 10, "missed_unseen": 3}
    assert EV.offline_consequence(errors) == pytest.approx(10 * 4.0 + 3 * 7.0)
    assert EV.offline_consequence(errors, {"false_unseen": 1.0, "missed_unseen": 1.0}) == pytest.approx(13.0)
    text = EV.consequence_text(errors)
    assert "estimate" in text.lower() and "not measured" in text.lower() and "61" in text
    with pytest.raises(ValueError, match="unknown"):
        EV.offline_consequence({"false_unseen": 1, "typo": 2})
    with pytest.raises(ValueError, match="negative"):
        EV.offline_consequence({"false_unseen": -1})

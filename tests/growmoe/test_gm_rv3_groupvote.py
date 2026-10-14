"""RV3: the group voting rule (spec section 5), without a pool."""

import pytest
import torch

from growmoe.groupvote import vote_groups


def _t(*v):
    return torch.tensor(v, dtype=torch.long)


def test_a_group_goes_to_the_expert_most_of_its_inputs_chose_and_the_inputs_are_untouched():
    choice = _t(0, 0, 0, 0, 0, 0, 0, 1, 1, 2)
    groups = torch.zeros(10, dtype=torch.long)
    before = choice.clone()
    out, info = vote_groups(choice, groups, group_min=8)
    assert out.tolist() == [0] * 10 and torch.equal(choice, before)
    assert info == {"groups": 1, "moved": 3, "vetoed": 0}


def test_ties_go_to_the_lowest_expert_id():
    out, _ = vote_groups(_t(5, 5, 5, 5, 3, 3, 3, 3), torch.zeros(8, dtype=torch.long), group_min=8)
    assert out.tolist() == [3] * 8


def test_groups_below_the_minimum_size_and_ungrouped_inputs_keep_their_own_choice():
    choice = _t(0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2)
    groups = _t(0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1)                                     # group 0 has 3 members (< 8); the last input is ungrouped
    out, info = vote_groups(choice, groups, group_min=8)
    assert out.tolist() == choice.tolist() and info["groups"] == 1 and info["moved"] == 0


def test_inputs_with_a_negative_group_id_are_never_voted_however_many_there_are():
    choice = _t(0, 1, 0, 1, 0, 1, 0, 1, 2, 2)
    out, info = vote_groups(choice, _t(*([-1] * 10)), group_min=8)
    assert torch.equal(out, choice) and info["groups"] == 0


def test_two_groups_are_answered_independently():
    choice = _t(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1)
    groups = _t(*([0] * 9 + [1] * 9))
    out, info = vote_groups(choice, groups, group_min=8)
    assert out.tolist() == [0] * 9 + [1] * 9 and info == {"groups": 2, "moved": 2, "vetoed": 0}


def test_the_veto_keeps_an_input_with_its_own_choice_when_the_group_expert_does_not_accept_it():
    choice = _t(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)
    seen = []

    def veto(expert, idx):
        seen.append((expert, idx.tolist()))
        return torch.tensor([i < 8 for i in idx.tolist()])                                # the last two inputs are rejected by expert 0

    out, info = vote_groups(choice, torch.zeros(10, dtype=torch.long), group_min=8, veto=veto)
    assert seen == [(0, list(range(10)))]
    assert out.tolist() == [0] * 8 + [1, 1] and info == {"groups": 1, "moved": 2, "vetoed": 2}


def test_negative_choices_are_ignored_by_the_vote_and_an_all_negative_group_is_left_alone():
    out, _ = vote_groups(_t(-1, -1, -1, -1, -1, 2, 2, 2, 2), torch.zeros(9, dtype=torch.long), group_min=8)    # "none" outnumbers every expert but is not a candidate
    assert out.tolist() == [2] * 9
    none, info = vote_groups(_t(*([-1] * 8)), torch.zeros(8, dtype=torch.long), group_min=8)
    assert none.tolist() == [-1] * 8 and info["groups"] == 0


def test_empty_input_and_bad_arguments():
    out, info = vote_groups(_t(), _t(), group_min=8)
    assert out.numel() == 0 and info["groups"] == 0
    with pytest.raises(ValueError):
        vote_groups(_t(0, 1), _t(0), group_min=1)
    with pytest.raises(ValueError):
        vote_groups(_t(0, 1), torch.tensor([0.0, 1.0]), group_min=1)
    with pytest.raises(ValueError):
        vote_groups(_t(0, 1), _t(0, 0), group_min=0)

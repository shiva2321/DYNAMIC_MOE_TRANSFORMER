import pytest
import torch

from fsmoe.ledger import TrustLedger
from fsmoe.team import choose_team, mix, team_size


def test_a_sure_input_gets_a_team_of_one_and_a_shared_input_gets_two():
    w = torch.tensor([[0.995, 0.003, 0.002], [0.55, 0.44, 0.01], [0.4, 0.35, 0.25]])
    members, ww = choose_team(w, mass=0.99, cap=3)
    assert team_size(members).tolist() == [1, 2, 3]
    assert torch.allclose(ww.sum(1), torch.ones(3)) and (ww[members < 0] == 0).all()
    assert members[0, 0] == 0 and members[1, 0] == 0 and members[1, 1] == 1


def test_the_cap_and_padding_are_respected_with_few_experts():
    w = torch.full((2, 5), 0.2)
    members, ww = choose_team(w, mass=0.99, cap=2)
    assert members.shape == (2, 2) and team_size(members).tolist() == [2, 2]
    one = choose_team(torch.tensor([[1.0]]), cap=3)
    assert one[0].tolist() == [[0, -1, -1]] and one[1].tolist() == [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        choose_team(torch.zeros(3))


def test_mix_blends_the_members_distributions():
    probs = torch.zeros(3, 2, 4)
    probs[0, :, 0] = 1.0
    probs[1, :, 1] = 1.0
    probs[2, :, 2] = 1.0
    members = torch.tensor([[0, 1], [2, -1]])
    ww = torch.tensor([[0.75, 0.25], [1.0, 0.0]])
    out = mix(probs, members, ww)
    assert torch.allclose(out[0], torch.tensor([0.75, 0.25, 0.0, 0.0])) and torch.allclose(out[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_trust_starts_high_drops_on_failure_and_recovers():
    led = TrustLedger()
    led.add(0)
    assert led.get(0) == led.start and not led.thin(0)
    t1 = led.verified(0, False)
    t2 = led.verified(0, False)
    assert t1 < led.start and t2 < t1                                  # a run of failures cuts deeper
    for _ in range(30):
        led.verified(0, True)
    assert led.get(0) > t2 and led.get(0) > 0.85
    assert led.get(99) == led.start                                    # an unknown expert is trusted until caught


def test_one_bad_batch_is_forgiven_but_a_streak_is_not():
    a, b = TrustLedger(), TrustLedger()
    a.verified(0, False)                                               # a single failure: generosity softens it
    for _ in range(3):
        b.verified(0, False)
    assert a.get(0) > 0.6 and b.get(0) < 0.3
    for _ in range(200):
        a.tick()
        b.tick()
    assert abs(a.get(0) - a.start) < 0.02 and b.get(0) > 0.7           # forgiveness drifts back toward the start


def test_trust_stays_inside_its_bounds():
    led = TrustLedger()
    for _ in range(100):
        led.verified(1, False)
    assert led.get(1) >= led.floor
    for _ in range(100):
        led.verified(1, True)
    assert led.get(1) <= 1.0

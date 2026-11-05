import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dmoe_v2_easy_eval as easy  # noqa: E402
import run_dmoe_v2_hard_control_eval as hc  # noqa: E402


def _oracle_clean_byte(domain_id: int, x_row, t: int) -> int:
    """Independent pure-python re-implementation of the clean target rule."""
    slot = t % 4
    b = int(x_row[t])
    if slot == 0:
        mult, add = {0: (3, 0), 1: (5, 1), 2: (7, 2)}[domain_id]
        return 32 + (b * mult + add) % 16
    v = b - 32
    if domain_id == 0:
        r = (v + slot) % 16
    elif domain_id == 1:
        r = (v ^ (slot + 5)) % 16
    else:
        r = (3 * v + 7 - slot) % 16
    return 32 + r


def test_easy_generator_matches_independent_oracle_and_noise_rate():
    total, match = 0, 0
    for d in range(3):
        x, y = easy.generate_easy_domain_batch(d, batch_size=256, seq_len=16, seed=123 + d)
        xl, yl = x.tolist(), y.tolist()
        for bi in range(len(xl)):
            for t in range(16):
                total += 1
                match += int(_oracle_clean_byte(d, xl[bi], t) == yl[bi][t])
    rate = match / total
    # expected agreement = 0.95 + 0.05/16 = 0.9531
    assert abs(rate - 0.9531) < 0.01, rate


def test_easy_generator_is_deterministic_overlapping_and_prefix_disjoint():
    x1, y1 = easy.generate_easy_domain_batch(1, 16, 16, 5)
    x2, y2 = easy.generate_easy_domain_batch(1, 16, 16, 5)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    payload = []
    prefixes = []
    for d in range(3):
        x, _ = easy.generate_easy_domain_batch(d, 64, 16, 9)
        slot = torch.arange(16) % 4
        payload.append(set(x[:, slot != 0].unique().tolist()))
        prefixes.append(set(x[:, slot == 0].unique().tolist()))
    assert payload[0] == payload[1] == payload[2]
    assert prefixes[0].isdisjoint(prefixes[1]) and prefixes[1].isdisjoint(prefixes[2])


def test_dense_baseline_freeze_backbone_really_freezes_and_head_still_trains():
    m = hc._make_dense_baseline(0)
    m.freeze_backbone()
    before_enc = [p.detach().clone() for p in m.encoder.parameters()]
    before_head = m.decoder.byte_chunk_head.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=4e-3)
    for st in range(3):
        x, y = easy.generate_easy_domain_batch(0, 16, 16, st)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x)["output"].reshape(-1, 256), y.reshape(-1))
        loss.backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before_enc, m.encoder.parameters()))
    assert not torch.equal(before_head, m.decoder.byte_chunk_head.weight)


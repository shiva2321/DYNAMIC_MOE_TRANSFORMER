import torch
import torch.nn.functional as F

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.jepa import JEPAModule, MASK_TOKEN_ID
from dmoe.model import SelfStabilizingDynamicMorphicMoE


def _small_encoder():
    return UniversalMultimodalEncoder(d_model=16)


def test_mask_token_id_is_reserved_top_of_vocab():
    assert MASK_TOKEN_ID == 31999


def test_compute_loss_returns_scalar_and_is_finite():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    token_ids = torch.randint(0, 1000, (4, 10), dtype=torch.long)
    loss = jepa.compute_loss(token_ids)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_loss_masks_a_contiguous_span_not_the_whole_sequence():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3)
    seq_len = 10
    start, end = jepa._masked_span(seq_len)
    assert 0 <= start < end <= seq_len
    assert end - start == max(1, int(seq_len * 0.3))


def test_gradients_flow_to_predictor_and_context_encoder_only():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    token_ids = torch.randint(0, 1000, (4, 10), dtype=torch.long)

    loss = jepa.compute_loss(token_ids)
    loss.backward()

    assert jepa.predictor.linear1.weight_real.grad is not None
    assert not torch.isnan(jepa.predictor.linear1.weight_real.grad).any()
    assert encoder.text_embed.weight.grad is not None
    for p in jepa.target_encoder.parameters():
        assert p.grad is None


def test_target_encoder_starts_as_exact_copy_of_context_encoder():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    assert torch.equal(jepa.target_encoder.text_embed.weight, jepa.context_encoder.text_embed.weight)
    assert jepa.target_encoder.text_embed.weight is not jepa.context_encoder.text_embed.weight


def test_update_target_encoder_moves_target_toward_context():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, ema_decay=0.5)

    with torch.no_grad():
        encoder.text_embed.weight.add_(1.0)

    before = jepa.target_encoder.text_embed.weight.clone()
    jepa.update_target_encoder()
    after = jepa.target_encoder.text_embed.weight

    assert not torch.equal(before, after)
    expected = before * 0.5 + (before + 1.0) * 0.5
    assert torch.allclose(after, expected, atol=1e-5)


def test_update_target_encoder_does_not_require_grad():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    jepa.update_target_encoder()
    for p in jepa.target_encoder.parameters():
        assert p.requires_grad is False


def test_trainable_parameters_returns_only_predictor_params():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    trainable_ids = {id(p) for p in jepa.trainable_parameters()}
    predictor_ids = {id(p) for p in jepa.predictor.parameters()}
    assert trainable_ids == predictor_ids


def test_jepa_loss_decreases_with_training():
    torch.manual_seed(0)
    encoder = UniversalMultimodalEncoder(d_model=16)
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3, ema_decay=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(jepa.trainable_parameters()), lr=5e-3
    )

    losses = []
    for _ in range(60):
        token_ids = torch.randint(0, 50, (8, 10), dtype=torch.long)
        optimizer.zero_grad()
        loss = jepa.compute_loss(token_ids)
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()
        losses.append(loss.item())

    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    print(f"\nJEPA training loss: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}")
    assert late_avg < early_avg


def test_target_encoder_representations_do_not_collapse_after_training():
    torch.manual_seed(0)
    encoder = UniversalMultimodalEncoder(d_model=16)
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3, ema_decay=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(jepa.trainable_parameters()), lr=5e-3
    )

    for _ in range(60):
        token_ids = torch.randint(0, 50, (8, 10), dtype=torch.long)
        optimizer.zero_grad()
        loss = jepa.compute_loss(token_ids)
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()

    with torch.no_grad():
        inputs = torch.randint(0, 50, (5, 10), dtype=torch.long)
        target_r, _ = jepa.target_encoder(inputs)
        pooled = target_r.mean(dim=1)  # [5, d_model]

    pairwise_dists = torch.cdist(pooled, pooled)
    off_diagonal = pairwise_dists[~torch.eye(5, dtype=torch.bool)]
    print(f"\nTarget encoder pairwise off-diagonal mean distance: {off_diagonal.mean().item():.4f}")
    assert off_diagonal.mean().item() > 1e-3


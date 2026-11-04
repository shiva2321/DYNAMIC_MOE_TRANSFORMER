import pytest
import torch
import torch.nn.functional as F

from dmoe.chunker import BYTE_MASK_ID, BYTE_PAD_ID, MultiScaleByteChunker
from dmoe.continual import ContinualLearner
from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.jepa import JEPAModule, MASK_TOKEN_ID
from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.replay_buffer import ReplayBuffer


def test_text_to_bytes_and_bytes_to_text_utf8_round_trip_and_special_tokens():
    texts = ["Hello, dMoE!", "Ω_corr ≈ 0.42", "def fn(x): return x + 1"]
    byte_tensor = MultiScaleByteChunker.text_to_bytes(texts)
    assert byte_tensor.dim() == 2
    assert byte_tensor.size(0) == 3

    # Shorter strings are padded with BYTE_PAD_ID (257), never confused with NUL (0)
    assert (byte_tensor == BYTE_PAD_ID).any()
    decoded = MultiScaleByteChunker.bytes_to_text(byte_tensor)
    for orig, dec in zip(texts, decoded):
        assert dec == orig

    # Real NUL byte (0x00) and 0xFF (255) are preserved distinct from BYTE_PAD_ID and BYTE_MASK_ID
    raw_with_nul = MultiScaleByteChunker.text_to_bytes([b"a\x00b", b"x"])
    assert raw_with_nul[0, 1].item() == 0
    assert raw_with_nul[1, 1].item() == BYTE_PAD_ID
    assert MultiScaleByteChunker.bytes_to_text(raw_with_nul) == ["a\x00b", "x"]

    # BYTE_MASK_ID (256) uses replacement placeholder rather than clamping to 255
    masked_seq = torch.tensor([[65, BYTE_MASK_ID, 66, BYTE_PAD_ID]], dtype=torch.long)
    dec_masked = MultiScaleByteChunker.bytes_to_text(masked_seq, mask_placeholder=b"?")
    assert dec_masked == ["A?B"]


def test_chunker_handles_divisible_and_non_divisible_byte_lengths():
    chunker = MultiScaleByteChunker(d_model=16, chunk_size=4)
    x_even = torch.randint(0, 256, (2, 12), dtype=torch.long)
    out_even = chunker(x_even)
    assert out_even.shape == (2, 3, 16)

    x_odd = torch.randint(0, 256, (2, 10), dtype=torch.long)
    out_odd = chunker(x_odd)
    assert out_odd.shape == (2, 3, 16)  # ceil(10 / 4) == 3


def test_tokenizer_free_encoder_breaks_mean_pool_permutation_invariance():
    enc = UniversalMultimodalEncoder(d_model=16, tokenizer_free=True, chunk_size=4)
    x1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
    x2 = torch.tensor([[80, 70, 60, 50, 40, 30, 20, 10]], dtype=torch.long)

    r1, i1 = enc(x1)
    r2, i2 = enc(x2)
    diff_r = (r1.mean(dim=1) - r2.mean(dim=1)).abs().max().item()
    diff_i = (i1.mean(dim=1) - i2.mean(dim=1)).abs().max().item()
    assert diff_r > 1e-3 or diff_i > 1e-3


def test_tokenizer_free_model_dramatically_reduces_parameter_count():
    tokenized_model = SelfStabilizingDynamicMorphicMoE(
        d_model=32, num_heads=2, d_ff=64, initial_layers=1, initial_experts=2, tokenizer_free=False
    )
    byte_model = SelfStabilizingDynamicMorphicMoE(
        d_model=32, num_heads=2, d_ff=64, initial_layers=1, initial_experts=2, tokenizer_free=True, chunk_size=4
    )

    n_tokenized = sum(p.numel() for p in tokenized_model.parameters())
    n_byte = sum(p.numel() for p in byte_model.parameters())
    print(f"\nParameter count (d_model=32): tokenized={n_tokenized:,}, tokenizer_free={n_byte:,}")
    assert n_byte * 10 < n_tokenized


def test_tokenizer_free_forward_and_backward_on_byte_stream():
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        initial_experts=2,
        tokenizer_free=True,
        chunk_size=4,
    )
    raw_bytes = torch.randint(0, 256, (3, 16), dtype=torch.long)
    target_bytes = (raw_bytes + 1) % 256

    out = model(raw_bytes)
    logits = out["output"]
    assert logits.shape == (3, 4, 4, 256)

    loss = F.cross_entropy(logits.reshape(-1, 256), target_bytes.reshape(-1))
    loss.backward()

    assert model.encoder.byte_chunker.byte_embed.weight.grad is not None
    assert model.decoder.byte_chunk_head.weight.grad is not None
    assert torch.isfinite(loss)


def test_tokenizer_free_multimodal_checkpoint_and_continual_learner(tmp_path):
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        initial_experts=2,
        enable_simulation_loop=True,
        max_simulation_steps=2,
        skip_threshold=-100.0,
        tokenizer_free=True,
        chunk_size=4,
    )
    img = torch.randn(2, 3, 16, 16)
    num = torch.randn(2, 5, 6)
    assert model(img, output_head="continuous")["output"].shape == (2, 16, 8)
    assert model(num, output_head="continuous")["output"].shape == (2, 5, 8)

    # Checkpoint round-trip including last_novelty_score buffer persistence
    model.eval()
    raw_bytes = torch.randint(0, 256, (2, 12), dtype=torch.long)
    with torch.no_grad():
        before = model(raw_bytes)["output"]
        novelty_before = model.manifold_corr.last_novelty_score.clone()

    ckpt_path = tmp_path / "tokenizer_free.pt"
    model.save_checkpoint(str(ckpt_path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(ckpt_path))
    loaded.eval()
    with torch.no_grad():
        after = loaded(raw_bytes)["output"]

    assert loaded.tokenizer_free is True
    assert loaded.chunk_size == 4
    assert torch.allclose(loaded.manifold_corr.last_novelty_score, novelty_before, atol=1e-6)
    assert torch.allclose(before, after, atol=1e-6)

    # ContinualLearner step on BOTH divisible (12) and non-divisible (10) byte lengths
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, opt, replay_buffer_capacity=16)
    target_bytes = (raw_bytes + 1) % 256
    metrics = learner.step(raw_bytes, target_bytes, output_head="byte_chunk")
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))

    odd_bytes = torch.randint(0, 256, (2, 10), dtype=torch.long)
    odd_targets = (odd_bytes + 1) % 256
    odd_metrics = learner.step(odd_bytes, odd_targets, output_head="byte_chunk")
    assert torch.isfinite(torch.tensor(odd_metrics["total_loss"]))


def test_review_fixes_simulation_loop_stats_replay_novelty_and_jepa_guards():
    # 1. Simulation loop in training mode must not pollute expert_age or running_error_magnitude
    torch.manual_seed(0)
    m_no_sim = SelfStabilizingDynamicMorphicMoE(
        d_model=8, num_heads=2, d_ff=16, initial_layers=1, initial_experts=2, enable_simulation_loop=False
    )
    torch.manual_seed(0)
    m_sim = SelfStabilizingDynamicMorphicMoE(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=1,
        initial_experts=2,
        enable_simulation_loop=True,
        max_simulation_steps=4,
        skip_threshold=-100.0,
    )
    m_no_sim.train()
    m_sim.train()
    x = torch.randint(0, 100, (2, 4), dtype=torch.long)
    out_no_sim = m_no_sim(x)
    out_sim = m_sim(x)
    assert out_sim["simulation_steps_used"] >= 1
    assert torch.equal(m_sim.layers[0].moe.expert_age, torch.ones(2))
    assert torch.allclose(
        m_sim.manifold_corr.running_error_magnitude,
        m_no_sim.manifold_corr.running_error_magnitude,
        atol=1e-6,
    )
    assert torch.allclose(out_sim["novelty_score"], out_no_sim["novelty_score"], atol=1e-6)

    # 2. ReplayBuffer negative/zero k guards
    buf = ReplayBuffer(capacity=4)
    buf.add("item1")
    assert buf.sample(0) == []
    with pytest.raises(ValueError, match="non-negative"):
        buf.sample(-1)

    # 3. JEPAModule explicit mask_token_id override & structural mismatch guard
    enc_tf = UniversalMultimodalEncoder(d_model=8, tokenizer_free=True, chunk_size=4)
    jepa_explicit = JEPAModule(enc_tf, predictor_hidden_dim=16, mask_token_id=MASK_TOKEN_ID)
    assert jepa_explicit.mask_token_id == MASK_TOKEN_ID

    # Trigger asymmetric numeric projector creation on context_encoder only
    _ = enc_tf(torch.randn(2, 3, 5))
    with pytest.raises(RuntimeError, match="mismatched parameter counts"):
        jepa_explicit.update_target_encoder()


def _structured_byte_batch(batch_size: int = 8, num_chunks: int = 4, chunk_size: int = 4):
    topic = torch.randint(0, 16, (batch_size, 1), dtype=torch.long)
    offsets = torch.arange(num_chunks * chunk_size, dtype=torch.long).unsqueeze(0)
    raw_bytes = (topic * 16 + (offsets % 4)) % 256
    coarse_parity_targets = raw_bytes % 2
    return raw_bytes, coarse_parity_targets, topic.squeeze(1)


def _train_byte_model(use_jepa: bool, steps: int = 80, seed: int = 0) -> SelfStabilizingDynamicMorphicMoE:
    torch.manual_seed(seed)
    m = SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        tokenizer_free=True,
        chunk_size=4,
    )
    jepa_mod = JEPAModule(m.encoder, predictor_hidden_dim=32, mask_ratio=0.25, ema_decay=0.95)
    params = list(m.parameters()) + (list(jepa_mod.trainable_parameters()) if use_jepa else [])
    opt = torch.optim.AdamW(params, lr=5e-3)

    for step in range(steps):
        torch.manual_seed(seed * 10000 + 1000 + step)
        b_in, b_coarse_tgt, _ = _structured_byte_batch(batch_size=16)
        opt.zero_grad()
        out = m(b_in)
        t_loss = F.cross_entropy(out["output"].reshape(-1, 256), b_coarse_tgt.reshape(-1))
        if use_jepa:
            torch.manual_seed(seed * 10000 + 5000 + step)
            t_loss = t_loss + jepa_mod.compute_loss(b_in)
        t_loss.backward()
        opt.step()
        if use_jepa:
            jepa_mod.update_target_encoder()
    return m


def _eval_topic_probe(m: SelfStabilizingDynamicMorphicMoE, seed: int = 0) -> float:
    torch.manual_seed(seed + 100)
    for p in m.encoder.parameters():
        p.requires_grad_(False)
    probe = torch.nn.Linear(m.d_model * 2, 16)
    p_opt = torch.optim.AdamW(probe.parameters(), lr=2e-2)
    for step in range(80):
        torch.manual_seed(seed * 10000 + 9000 + step)
        b_in, _, topics = _structured_byte_batch(batch_size=16)
        with torch.no_grad():
            r, i = m.encoder(b_in)
            pooled = torch.cat([r.mean(dim=1), i.mean(dim=1)], dim=-1)
        p_opt.zero_grad()
        p_loss = F.cross_entropy(probe(pooled), topics)
        p_loss.backward()
        p_opt.step()

    torch.manual_seed(seed * 10000 + 9999)
    eval_in, _, eval_topics = _structured_byte_batch(batch_size=64)
    with torch.no_grad():
        r, i = m.encoder(eval_in)
        pooled = torch.cat([r.mean(dim=1), i.mean(dim=1)], dim=-1)
        return F.cross_entropy(probe(pooled), eval_topics).item()


def test_jepa_on_tokenizer_free_byte_chunks_flows_context_gradients_and_improves_probe():
    torch.manual_seed(0)
    enc = UniversalMultimodalEncoder(d_model=16, tokenizer_free=True, chunk_size=4)
    jepa = JEPAModule(enc, predictor_hidden_dim=32)
    assert jepa.mask_token_id == BYTE_MASK_ID

    raw_bytes, _, _ = _structured_byte_batch(batch_size=4)
    loss = jepa.compute_loss(raw_bytes)
    loss.backward()

    byte_grad = enc.byte_chunker.byte_embed.weight.grad
    assert byte_grad is not None
    assert byte_grad[:256].abs().max().item() > 0.0

    m_jepa = _train_byte_model(use_jepa=True, seed=0)
    m_base = _train_byte_model(use_jepa=False, seed=0)
    loss_jepa = _eval_topic_probe(m_jepa, seed=0)
    loss_base = _eval_topic_probe(m_base, seed=0)
    print(f"\nStructured byte-chunk topic probe loss: jepa={loss_jepa:.4f}, baseline={loss_base:.4f}")
    assert loss_jepa < loss_base

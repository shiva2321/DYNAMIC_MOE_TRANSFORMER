import torch
import torch.nn.functional as F

from dmoe.chunker import BYTE_MASK_ID, MultiScaleByteChunker
from dmoe.continual import ContinualLearner
from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.jepa import JEPAModule
from dmoe.model import SelfStabilizingDynamicMorphicMoE


def test_text_to_bytes_and_bytes_to_text_utf8_round_trip():
    texts = ["Hello, dMoE!", "Ω_corr ≈ 0.42", "def fn(x): return x + 1"]
    byte_tensor = MultiScaleByteChunker.text_to_bytes(texts)
    assert byte_tensor.dim() == 2
    assert byte_tensor.size(0) == 3
    assert byte_tensor.min().item() >= 0
    assert byte_tensor.max().item() <= 255

    decoded = MultiScaleByteChunker.bytes_to_text(byte_tensor)
    for orig, dec in zip(texts, decoded):
        assert dec.startswith(orig)


def test_chunker_handles_divisible_and_non_divisible_byte_lengths():
    chunker = MultiScaleByteChunker(d_model=16, chunk_size=4)
    x_even = torch.randint(0, 256, (2, 12), dtype=torch.long)
    out_even = chunker(x_even)
    assert out_even.shape == (2, 3, 16)

    x_odd = torch.randint(0, 256, (2, 10), dtype=torch.long)
    out_odd = chunker(x_odd)
    assert out_odd.shape == (2, 3, 16)  # ceil(10 / 4) == 3


def test_tokenizer_free_encoder_breaks_mean_pool_permutation_invariance():
    # Unlike the affine 32k token lookup path, the multi-scale conv + chunk
    # projection encodes byte order so reversing a sequence changes the
    # sequence-pooled representation.
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
    # Verify 4D image and 3D numeric streams still work alongside 2D byte streams
    img = torch.randn(2, 3, 16, 16)
    num = torch.randn(2, 5, 6)
    assert model(img, output_head="continuous")["output"].shape == (2, 16, 8)
    assert model(num, output_head="continuous")["output"].shape == (2, 5, 8)

    # Checkpoint round-trip
    model.eval()
    raw_bytes = torch.randint(0, 256, (2, 12), dtype=torch.long)
    with torch.no_grad():
        before = model(raw_bytes)["output"]

    ckpt_path = tmp_path / "tokenizer_free.pt"
    model.save_checkpoint(str(ckpt_path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(ckpt_path))
    loaded.eval()
    with torch.no_grad():
        after = loaded(raw_bytes)["output"]

    assert loaded.tokenizer_free is True
    assert loaded.chunk_size == 4
    assert torch.allclose(before, after, atol=1e-6)

    # ContinualLearner step on raw byte chunks
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, opt, replay_buffer_capacity=16)
    target_bytes = (raw_bytes + 1) % 256
    metrics = learner.step(raw_bytes, target_bytes, output_head="byte_chunk")
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def _structured_byte_batch(batch_size: int = 8, num_chunks: int = 4, chunk_size: int = 4):
    # Each sequence has a latent topic ID c in 0..15 repeated across chunks,
    # while the coarse training task only asks for parity (c % 2), causing
    # task-only training to suffer supervision collapse on fine-grained topic c.
    topic = torch.randint(0, 16, (batch_size, 1), dtype=torch.long)
    offsets = torch.arange(num_chunks * chunk_size, dtype=torch.long).unsqueeze(0)
    raw_bytes = (topic * 16 + (offsets % 4)) % 256
    coarse_parity_targets = raw_bytes % 2
    return raw_bytes, coarse_parity_targets, topic.squeeze(1)


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

    # Train two tokenizer-free models (coarse task + JEPA vs coarse task only) with isolated data RNG
    def train_byte_model(use_jepa: bool, steps: int = 80, seed: int = 0):
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
            torch.manual_seed(1000 + step)
            b_in, b_coarse_tgt, _ = _structured_byte_batch(batch_size=16)
            opt.zero_grad()
            out = m(b_in)
            t_loss = F.cross_entropy(out["output"].reshape(-1, 256), b_coarse_tgt.reshape(-1))
            if use_jepa:
                torch.manual_seed(5000 + step)
                t_loss = t_loss + jepa_mod.compute_loss(b_in)
            t_loss.backward()
            opt.step()
            if use_jepa:
                jepa_mod.update_target_encoder()
        return m

    def eval_topic_probe(m: SelfStabilizingDynamicMorphicMoE, seed: int = 42) -> float:
        torch.manual_seed(seed)
        for p in m.encoder.parameters():
            p.requires_grad_(False)
        probe = torch.nn.Linear(m.d_model * 2, 16)
        p_opt = torch.optim.AdamW(probe.parameters(), lr=2e-2)
        for step in range(80):
            torch.manual_seed(9000 + step)
            b_in, _, topics = _structured_byte_batch(batch_size=16)
            with torch.no_grad():
                r, i = m.encoder(b_in)
                pooled = torch.cat([r.mean(dim=1), i.mean(dim=1)], dim=-1)
            p_opt.zero_grad()
            p_loss = F.cross_entropy(probe(pooled), topics)
            p_loss.backward()
            p_opt.step()

        torch.manual_seed(9999)
        eval_in, _, eval_topics = _structured_byte_batch(batch_size=64)
        with torch.no_grad():
            r, i = m.encoder(eval_in)
            pooled = torch.cat([r.mean(dim=1), i.mean(dim=1)], dim=-1)
            return F.cross_entropy(probe(pooled), eval_topics).item()

    m_jepa = train_byte_model(use_jepa=True)
    m_base = train_byte_model(use_jepa=False)
    loss_jepa = eval_topic_probe(m_jepa)
    loss_base = eval_topic_probe(m_base)
    print(f"\nStructured byte-chunk topic probe loss: jepa={loss_jepa:.4f}, baseline={loss_base:.4f}")
    assert loss_jepa < loss_base

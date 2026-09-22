import pytest
import torch
import torch.nn.functional as F

from ishmn import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES
from ishmn.deliberative_logits import DeliberativeConceptLogitsProcessor


@pytest.fixture
def dual_cortex_system():
    device = torch.device("cpu")
    cfg = MODEL_PROFILES["micro"]
    tokenizer = SovereignTokenizer("gpt2")

    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        hyperspace_dim=cfg["hyperspace_dim"],
        d_ff=cfg["d_ff"],
        max_seq_len=64,
        mtp_future_tokens=cfg["mtp_future_tokens"],
        vsa_capacity=cfg["vsa_capacity"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        max_experts=cfg["num_experts"],
        enable_shared_expert=cfg["enable_shared_expert"],
        enable_aux_loss_free_balancing=cfg["enable_aux_loss_free_balancing"],
        enable_vsa_memory=True,
        enable_world_model=False,
        tie_weights=True,
        device=device
    )

    uce = UniversalCognitiveEngine(
        d_model=cfg["d_model"],
        d_hidden=cfg["d_ff"],
        hyperspace_dim=cfg["hyperspace_dim"],
        num_memory_slots=16,
        vocab_size=tokenizer.vocab_size,
        initial_experts=cfg["num_experts"],
        device=device
    )
    uce.attach_native_transformer(model, tokenizer=tokenizer)
    return uce, model, tokenizer, device


def test_seed_scientific_invariants(dual_cortex_system):
    uce, model, tokenizer, device = dual_cortex_system
    num_seeded = uce.seed_scientific_invariants()
    assert num_seeded > 0
    assert len(uce.memory.semantic_concepts) > 0
    assert len(uce.hopfield) > 0


def test_closed_loop_train_unified_step(dual_cortex_system):
    uce, model, tokenizer, device = dual_cortex_system
    uce.seed_scientific_invariants()

    B, T = 2, 16
    inp = torch.randint(100, 1000, (B, T), dtype=torch.long, device=device)
    tgt = torch.randint(100, 1000, (B, T), dtype=torch.long, device=device)

    uce.train()
    model.train()
    step_res = uce.train_unified_step(inp, tgt)

    assert "loss" in step_res
    assert torch.isfinite(step_res["loss"])
    assert step_res["loss"].item() > 0

    # Test backward pass
    step_res["loss"].backward()

    # Verify gradients flow into both transformer and UCE
    has_trans_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    has_perceptor_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in uce.perceptor.parameters())

    assert has_trans_grad, "Language cortex did not receive gradients"
    assert has_perceptor_grad, "Universal perceptor did not receive gradients"


def test_think_and_speak_with_cognitive_context(dual_cortex_system):
    uce, model, tokenizer, device = dual_cortex_system
    uce.seed_scientific_invariants()
    uce.eval()
    model.eval()

    prompt = "The conservation of energy states that"
    res = uce.think_and_speak(
        prompt,
        max_new_tokens=10,
        temperature=0.0,
        enable_scratchpad=True,
        enable_deliberation=True,
        write_to_vsa=False
    )

    assert "response" in res
    assert len(res["response"]) >= len(prompt)
    assert res["tokens_generated"] > 0


def test_deliberative_concept_logits_processor(dual_cortex_system):
    uce, model, tokenizer, device = dual_cortex_system
    uce.seed_scientific_invariants()

    processor = DeliberativeConceptLogitsProcessor(uce=uce, tokenizer=tokenizer, boost_weight=3.0, enabled=True)

    # Prompt with Wien's displacement law keywords
    input_ids = torch.tensor([tokenizer.encode("The wien displacement law describes")], dtype=torch.long)
    scores = torch.zeros((1, tokenizer.vocab_size))

    boosted = processor(input_ids, scores.clone())

    # Verify that at least one token received a boost
    assert (boosted > 0).any(), "Expected deliberative concept processor to boost matching tokens"


def test_hopfield_checkpoint_loading_compatibility(dual_cortex_system):
    """Verify that ModernHopfieldNetwork resizes or preserves buffers cleanly across checkpoints."""
    uce, model, tokenizer, device = dual_cortex_system

    # 1. Seed invariants into UCE
    uce.seed_scientific_invariants()
    seeded_count = len(uce.hopfield)
    assert seeded_count > 0

    # 2. Simulate loading an older checkpoint where memory_matrix was empty [0, d_model]
    old_state_dict = {"memory_matrix": torch.empty(0, uce.hopfield.d_model, device=device)}
    uce.hopfield.load_state_dict(old_state_dict, strict=False)
    # The seeded count should be preserved because checkpoint had 0 attractors
    assert len(uce.hopfield) == seeded_count

    # 3. Simulate loading a checkpoint with new attractors [5, d_model]
    new_matrix = torch.randn(5, uce.hopfield.d_model, device=device)
    new_state_dict = {"memory_matrix": new_matrix}
    uce.hopfield.load_state_dict(new_state_dict, strict=False)
    assert len(uce.hopfield) == 5
    assert uce.hopfield.memory_matrix.shape == (5, uce.hopfield.d_model)


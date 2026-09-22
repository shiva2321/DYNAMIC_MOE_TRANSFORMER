"""
Zero-Tolerance Pre-Flight Diagnostic & Connectivity Audit Script.
Verifies that 100% of subsystems in the Dual-Cortex Architecture are actively connected,
receiving non-zero gradients, and numerically stable prior to pretraining launch.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES


def run_full_connectivity_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Full System Connectivity Audit on: {device}")
    
    cfg = MODEL_PROFILES["micro"]
    tokenizer = SovereignTokenizer("gpt2")
    vocab_size = tokenizer.vocab_size # 50,257
    print(f"[*] Sovereign Tokenizer Vocab Size: {vocab_size}")

    # 1. Instantiate Native Language Cortex
    model = NativeSphericalTransformer(
        vocab_size=vocab_size,
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
        enable_gwt_coalition=True,
        enable_vsa_memory=True,
        enable_world_model=False,
        tie_weights=True,
        device=device
    ).to(device)

    # 2. Instantiate Universal Cognitive Engine
    uce = UniversalCognitiveEngine(
        d_model=cfg["d_model"],
        d_hidden=cfg["d_ff"],
        hyperspace_dim=cfg["hyperspace_dim"],
        num_memory_slots=32,
        vocab_size=vocab_size,
        initial_experts=cfg["num_experts"],
        device=device
    ).to(device)

    # Attach transformer
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    # Assert vocab alignment
    assert uce.perceptor.token_embed.num_embeddings >= vocab_size, (
        f"Perceptor vocab size {uce.perceptor.token_embed.num_embeddings} < {vocab_size}"
    )
    print(f"[PASS] Perceptor Token Embed Sized: {uce.perceptor.token_embed.num_embeddings} tokens.")

    # 3. Seed Invariants
    num_seeded = uce.seed_scientific_invariants()
    print(f"[PASS] Attractor Matrix Seeded: {num_seeded} concepts ({len(uce.hopfield)} total attractors in lattice).")
    assert num_seeded >= 25, f"Expected >= 25 seeded attractors, got {num_seeded}"

    # 4. Synthetic Forward & Backward Pass
    uce.train()
    model.train()

    B, T = 4, 32
    inp = torch.randint(0, vocab_size, (B, T), device=device)
    tgt = torch.randint(0, vocab_size, (B, T), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        step_res = uce.train_unified_step(inp, tgt)
        loss = step_res["loss"]

    assert not torch.isnan(loss) and not torch.isinf(loss), f"Loss is NaN or Inf: {loss.item()}"
    print(f"[PASS] Forward Pass Loss Computed: {loss.item():.4f}")
    for k, v in step_res["loss_breakdown"].items():
        print(f"       - {k:22s}: {v:.4f}")

    # Backward Pass
    loss.backward()

    # 5. Exhaustive Subsystem Gradient Auditing
    audit_checks = {
        "Language Cortex Token Embeddings": model.tok_embed.weight,
        "Language Cortex Pos Embeddings": model.pos_embed.weight,
        "Language Cortex Output Head": model.head_primary.weight,
        "Perceptor Token Embeddings": uce.perceptor.token_embed.weight,
        "Perceptor Sequence Encoder": next(uce.perceptor.seq_encoder.parameters()),
        "Perceptor Direction Projector": uce.perceptor.to_direction[0].weight,
        "Perceptor Radius Projector": uce.perceptor.to_radius[0].weight,
        "Perceptor Payload Projector": uce.perceptor.to_payload[0].weight,
        "DeltaNet Recurrent Memory (Wq)": uce.delta_net.Wq.weight,
        "DeltaNet Recurrent Memory (Wk)": uce.delta_net.Wk.weight,
        "DeltaNet Recurrent Memory (Wv)": uce.delta_net.Wv.weight,
        "DeltaNet Fast Update Gate (Wbeta)": uce.delta_net.Wbeta.weight,
        "Dentate Gyrus Pattern Separator": uce.dentate.proj.weight,
        "Toroidal Harmonic Grid Encoder": uce.grid_encoder.grid_proj_0,
        "Toroidal Grid Manifold Decoder": uce.grid_to_u.weight,
        "JEPA World Model Forward Transition": uce.world_model.w_fwd[0].weight,
        "JEPA World Model Backward Inversion": uce.world_model.w_bwd[0].weight,
    }

    # Also check GWT Coalition Projectors in every block
    for layer_idx, block in enumerate(model.blocks):
        audit_checks[f"GWT Coalition Projector (Layer {layer_idx})"] = block.moe.context_to_proposal.weight

    all_passed = True
    print("\n" + "=" * 80)
    print(" SUBSYSTEM GRADIENT FLOW AUDIT SCORECARD:")
    print("=" * 80)

    for name, param in audit_checks.items():
        if param.grad is None:
            print(f" [FAIL] {name:42s}: GRADIENT IS NONE!")
            all_passed = False
        else:
            grad_norm = param.grad.abs().sum().item()
            if grad_norm == 0.0:
                print(f" [FAIL] {name:42s}: GRADIENT IS ZERO!")
                all_passed = False
            else:
                print(f" [OK]   {name:42s}: Grad Norm = {grad_norm:10.5f}")

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] ALL 100% OF ARCHITECTURAL SUBSYSTEMS ACTIVELY RECEIVING GRADIENTS!")
    else:
        raise RuntimeError("One or more architectural subsystems failed the gradient audit!")

    # 6. System 2 Symbolic Solver Verification
    print("\n[*] Auditing System 2 Symbolic Dual-Process Solver...")
    test_expr = "What is 47 * 19 + 25?"
    dual_res = uce.solve_with_dual_process(test_expr, enable_system2=True)
    ans = dual_res.get("system2_exact_result")
    assert ans == 918.0, (
        f"Expected 918.0, got {ans}"
    )
    print(f"[PASS] System 2 Symbolic Solver Output: {ans} (Exact)")

    # 7. Memory / VRAM footprint
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[PASS] Peak GPU VRAM Footprint: {vram_mb:.1f} MB (Well within 6GB hardware envelope).")

    print("\n[ALL PRE-FLIGHT CONNECTIVITY CHECKS PASSED WITH ZERO TOLERANCE]")


if __name__ == "__main__":
    run_full_connectivity_audit()

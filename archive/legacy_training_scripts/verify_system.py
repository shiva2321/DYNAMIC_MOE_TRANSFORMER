"""
Verification Suite for Dynamic Hyperspace MoE and Resonator Factorization.
Executes mathematical unit tests, resonator convergence tests, dynamic spawning benchmarks,
and autograd stability checks.
"""

import sys
import math
import torch
import torch.nn.functional as F

from hyperspace.vsa import ComplexPhasorVSA
from hyperspace.resonator import ResonatorFactorizer
from hyperspace.memory import SemanticHyperspaceMemory
from model.expert import MicroExpert
from model.hyper_moe import DynamicHyperMoE
from model.nanogpt import HyperTransformerLM

def test_vsa_algebra(device: torch.device):
    print("==================================================")
    print("TEST 1: Complex Phasor VSA Algebraic Properties")
    print("==================================================")
    d_hyper = 4096
    
    # 1. Random Orthogonality Test
    v1 = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
    v2 = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
    
    sim_identical = ComplexPhasorVSA.hermitian_similarity(v1, v1).item()
    sim_orthogonal = ComplexPhasorVSA.hermitian_similarity(v1, v2).item()
    
    print(f"-> Similarity(v1, v1) [Expect ~1.000]: {sim_identical:.4f}")
    print(f"-> Similarity(v1, v2) [Expect ~0.000 (quasi-orthogonal)]: {sim_orthogonal:.4f}")
    assert abs(sim_identical - 1.0) < 1e-4, "Self-similarity must be exactly 1.0"
    assert abs(sim_orthogonal) < 0.08, f"Quasi-orthogonality violated: {sim_orthogonal}"

    # 2. Inversion / Unbinding Test: (A (x) B) (x) B^(-1) == A
    compound = ComplexPhasorVSA.bind(v1, v2)
    recovered_v1 = ComplexPhasorVSA.unbind(compound, v2)
    recovery_sim = ComplexPhasorVSA.hermitian_similarity(v1, recovered_v1).item()
    
    print(f"-> Recovery Similarity(v1, (v1 x v2) x v2^-1) [Expect ~1.000]: {recovery_sim:.4f}")
    assert recovery_sim > 0.999, f"Unbinding inversion failed: {recovery_sim}"
    print("[PASS] Complex Phasor VSA algebraic properties verified successfully!\n")

def test_resonator_factorization(device: torch.device):
    print("==================================================")
    print("TEST 2: Resonator Network Algebraic Factorization")
    print("==================================================")
    d_hyper = 4096
    num_domains = 8
    num_tasks = 8
    num_syntaxes = 8
    
    # Create 3 distinct codebooks in hyperspace
    cb_domain = ComplexPhasorVSA.random_hyperspace_vector((num_domains, d_hyper), device=device)
    cb_task = ComplexPhasorVSA.random_hyperspace_vector((num_tasks, d_hyper), device=device)
    cb_syntax = ComplexPhasorVSA.random_hyperspace_vector((num_syntaxes, d_hyper), device=device)
    
    resonator = ResonatorFactorizer([cb_domain, cb_task, cb_syntax], max_iterations=20).to(device)
    
    # Synthesize composite test queries: Q = Domain_i (x) Task_j (x) Syntax_k
    batch_size = 10
    target_domains = torch.randint(0, num_domains, (batch_size,), device=device)
    target_tasks = torch.randint(0, num_tasks, (batch_size,), device=device)
    target_syntaxes = torch.randint(0, num_syntaxes, (batch_size,), device=device)
    
    vec_d = cb_domain[target_domains]
    vec_t = cb_task[target_tasks]
    vec_s = cb_syntax[target_syntaxes]
    
    # Bound 3-aspect query
    query = ComplexPhasorVSA.bind(ComplexPhasorVSA.bind(vec_d, vec_t), vec_s)
    
    # Add random noise to test robustness
    noise = ComplexPhasorVSA.random_hyperspace_vector((batch_size, d_hyper), device=device)
    noisy_query = ComplexPhasorVSA.bundle(torch.stack([query, query, noise], dim=1), dim=1)
    
    estimates, factor_indices, iters = resonator(noisy_query)
    
    pred_d, pred_t, pred_s = factor_indices
    acc_d = (pred_d == target_domains).float().mean().item() * 100.0
    acc_t = (pred_t == target_tasks).float().mean().item() * 100.0
    acc_s = (pred_s == target_syntaxes).float().mean().item() * 100.0
    
    print(f"-> Resonator Convergence Steps: {iters}")
    print(f"-> Domain Factorization Accuracy: {acc_d:.1f}%")
    print(f"-> Task Factorization Accuracy: {acc_t:.1f}%")
    print(f"-> Syntax Factorization Accuracy: {acc_s:.1f}%")
    
    assert acc_d == 100.0 and acc_t == 100.0 and acc_s == 100.0, "Resonator factorization failed"
    print("[PASS] Resonator factorized composite queries with 100% accuracy!\n")

def test_dynamic_spawning(device: torch.device):
    print("==================================================")
    print("TEST 3: Dynamic Novelty Detection & Expert Spawning")
    print("==================================================")
    d_model = 128
    d_hyper = 2048
    d_ff = 256
    
    moe = DynamicHyperMoE(
        d_model=d_model,
        d_ff=d_ff,
        d_hyper=d_hyper,
        spawn_threshold=0.30,
        initial_experts=2
    ).to(device)
    
    print(f"-> Initial Expert Count: {moe.num_experts}")
    assert moe.num_experts == 2
    
    # 1. In-distribution query (should not spawn)
    x_indist = torch.randn(2, 8, d_model, device=device) * 0.1
    moe.train()
    out1, telem1 = moe(x_indist, allow_spawning=True)
    print(f"-> Pass 1 (In-Distribution) Expert Count: {moe.num_experts} (Spawned: {moe.num_experts > 2})")
    
    # 2. Out-of-distribution shifted query (triggers novelty detection)
    x_ood = torch.randn(2, 8, d_model, device=device) * 8.0 + 5.0
    out2, telem2 = moe(x_ood, allow_spawning=True)
    print(f"-> Pass 2 (Out-of-Distribution Shift) Expert Count: {moe.num_experts}")
    assert moe.num_experts > 2, "Novelty detector failed to spawn new expert"
    print("[PASS] Dynamic expert spawning verified!\n")

def test_nanogpt_autograd(device: torch.device):
    print("==================================================")
    print("TEST 4: NanoGPT End-to-End Autograd & Gradient Stability")
    print("==================================================")
    model = HyperTransformerLM(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        d_hyper=1024,
        top_k=2,
        max_seq_len=64
    ).to(device)
    
    # Sample input and target tokens
    dummy_input = torch.randint(0, 256, (4, 32), device=device)
    dummy_targets = torch.randint(0, 256, (4, 32), device=device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    logits, loss, telemetries = model(dummy_input, targets=dummy_targets)
    
    print(f"-> Forward Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    
    loss.backward()
    
    # Check for valid gradients across parameters
    has_grads = all(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters() if p.requires_grad)
    print(f"-> All Parameters Have Non-NaN Gradients: {has_grads}")
    assert has_grads, "Gradients failed during backpropagation"
    
    optimizer.step()
    
    # Generation test
    prompt = torch.tensor([[ord('d'), ord('e'), ord('f'), ord(' ')]], device=device)
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"-> Sample Autoregressive Generation Shape: {generated.shape}")
    print("[PASS] End-to-End NanoGPT with Dynamic Hyper-MoE is fully functional!\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running System Verification on Device: {device}\n")
    
    try:
        test_vsa_algebra(device)
        test_resonator_factorization(device)
        test_dynamic_spawning(device)
        test_nanogpt_autograd(device)
        print("**************************************************")
        print("ALL HYPERSPACE & DYNAMIC MoE TESTS PASSED (4/4)!")
        print("**************************************************")
    except Exception as e:
        print(f"\n[FAIL] Test encountered error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
